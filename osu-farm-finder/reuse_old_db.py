"""Measure and optionally reuse current data from an older farm database.

This utility never copies the old player sample into the new database. The new
balanced player selection remains authoritative. It only reuses score snapshots
for user IDs that the NEW sample already selected, plus optional beatmap metadata.

Typical usage after the new player stage finishes:

    python reuse_old_db.py --old farm.db --new farm_v2.db check
    python reuse_old_db.py --old farm.db --new farm_v2.db copy-scores

After copy-scores, normal collection resumes with:

    python run.py --db farm_v2.db collect --stage scores

The collector will skip reused users because this script records them in
score_fetches using the source DB's player fetched_at timestamp as conservative
same-session provenance. This is appropriate when the old DB is a recent
snapshot from the current collection cycle; do not use it to bless a months-old
DB as current.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import time
from pathlib import Path

BANDS = [
    (1000, 2000, "1000-2000"),
    (2000, 3000, "2000-3000"),
    (3000, 4000, "3000-4000"),
    (4000, 5000, "4000-5000"),
    (5000, 6000, "5000-6000"),
    (6000, 7000, "6000-7000"),
    (7000, 8000, "7000-8000"),
    (8000, 9000, "8000-9000"),
    (9000, 10000, "9000-10000"),
    (10000, 12000, "10000-12000"),
    (12000, 99999, "12000+"),
]


def connect_new(path: str) -> sqlite3.Connection:
    """Open new DB and ensure the score_fetches/meta tables exist."""
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS score_fetches (
            user_id INTEGER PRIMARY KEY,
            fetched_at REAL NOT NULL,
            score_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )
    return conn


def require_tables(conn: sqlite3.Connection, alias: str, tables: list[str]) -> None:
    present = {
        r[0]
        for r in conn.execute(
            f"SELECT name FROM {alias}.sqlite_master WHERE type='table'"
        )
    }
    missing = [t for t in tables if t not in present]
    if missing:
        raise SystemExit(
            f"{alias} database is missing required table(s): {', '.join(missing)}"
        )


def attach_old(conn: sqlite3.Connection, old_path: str) -> None:
    conn.execute("ATTACH DATABASE ? AS old", (str(Path(old_path).resolve()),))
    require_tables(conn, "main", ["players", "scores"])
    require_tables(conn, "old", ["players", "scores"])


def scalar(conn: sqlite3.Connection, sql: str, params=()) -> int:
    return int(conn.execute(sql, params).fetchone()[0] or 0)


def overlap_stats(conn: sqlite3.Connection) -> dict[str, int]:
    out = {
        "new_players": scalar(conn, "SELECT COUNT(*) FROM main.players"),
        "old_score_users": scalar(conn, "SELECT COUNT(DISTINCT user_id) FROM old.scores"),
        "overlap_users": scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM main.players n
            WHERE EXISTS (SELECT 1 FROM old.scores s WHERE s.user_id=n.user_id)
            """,
        ),
        "overlap_scores": scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM old.scores s
            JOIN main.players n ON n.user_id=s.user_id
            """,
        ),
        "already_seeded": scalar(
            conn,
            "SELECT COUNT(*) FROM main.score_fetches",
        ),
    }
    return out


def print_check(conn: sqlite3.Connection) -> None:
    st = overlap_stats(conn)
    print("[reuse check]")
    print(f"  new selected players : {st['new_players']:,}")
    print(f"  old score users      : {st['old_score_users']:,}")
    print(f"  overlapping users    : {st['overlap_users']:,}")
    pct = 100.0 * st["overlap_users"] / max(st["new_players"], 1)
    print(f"  overlap of new sample: {pct:.1f}%")
    print(f"  reusable score rows  : {st['overlap_scores']:,}")
    print(f"  score_fetches already: {st['already_seeded']:,}")
    print()
    print("  overlap by pp band:")
    total_overlap = 0
    for lo, hi, label in BANDS:
        row = conn.execute(
            """
            SELECT COUNT(*) AS selected,
                   SUM(CASE WHEN EXISTS (
                       SELECT 1 FROM old.scores s WHERE s.user_id=n.user_id
                   ) THEN 1 ELSE 0 END) AS overlap
            FROM main.players n
            WHERE n.pp >= ? AND n.pp < ?
            """,
            (lo, hi),
        ).fetchone()
        selected = int(row["selected"] or 0)
        overlap = int(row["overlap"] or 0)
        total_overlap += overlap
        pct = 100.0 * overlap / max(selected, 1)
        print(f"    {label:>11}: {overlap:>4}/{selected:<4} ({pct:5.1f}%)")

    # The old schema did not have an independent score-fetch timestamp. Show
    # its player-ranking timestamp range as provenance, not as a claim that
    # scores were fetched at those exact seconds.
    row = conn.execute(
        """
        SELECT MIN(p.fetched_at) AS mn, MAX(p.fetched_at) AS mx
        FROM old.players p
        WHERE EXISTS (SELECT 1 FROM old.scores s WHERE s.user_id=p.user_id)
        """
    ).fetchone()
    if row and row["mn"] and row["mx"]:
        lo = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["mn"]))
        hi = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["mx"]))
        print()
        print("  source ranking timestamps for score-bearing users:")
        print(f"    {lo}  ->  {hi}")
        print("  (provenance only; the old schema did not store an independent score-fetch time)")


def copy_scores(conn: sqlite3.Connection) -> None:
    st = overlap_stats(conn)
    if st["overlap_users"] == 0:
        raise SystemExit("No overlapping score-bearing users found; nothing to copy.")

    print("[copy scores]")
    print(f"  seeding {st['overlap_users']:,} overlapping users")
    print(f"  copying up to {st['overlap_scores']:,} score rows")

    # Do not mix an old partial seed with new score rows for the same selected
    # user. Replace per-user state in one transaction, mirroring collect.py.
    user_rows = conn.execute(
        """
        SELECT n.user_id,
               COALESCE(op.fetched_at, ?) AS source_time,
               COUNT(os.score_id) AS score_count
        FROM main.players n
        JOIN old.scores os ON os.user_id=n.user_id
        LEFT JOIN old.players op ON op.user_id=n.user_id
        GROUP BY n.user_id
        """,
        (time.time(),),
    ).fetchall()

    with conn:
        # Only delete rows for users we're about to seed.
        conn.execute(
            """
            DELETE FROM main.scores
            WHERE user_id IN (
                SELECT n.user_id
                FROM main.players n
                WHERE EXISTS (SELECT 1 FROM old.scores os WHERE os.user_id=n.user_id)
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO main.scores
            SELECT os.*
            FROM old.scores os
            JOIN main.players n ON n.user_id=os.user_id
            """
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO main.score_fetches(user_id, fetched_at, score_count)
            VALUES (?,?,?)
            """,
            [
                (int(r["user_id"]), float(r["source_time"]), int(r["score_count"]))
                for r in user_rows
            ],
        )
        conn.execute(
            "INSERT OR REPLACE INTO main.meta(key,value) VALUES (?,?)",
            ("reused_scores_from", str(Path(args.old).resolve())),
        )
        conn.execute(
            "INSERT OR REPLACE INTO main.meta(key,value) VALUES (?,?)",
            ("reused_scores_seeded_at", str(time.time())),
        )

    current_users = scalar(conn, "SELECT COUNT(*) FROM main.score_fetches")
    current_scores = scalar(conn, "SELECT COUNT(*) FROM main.scores")
    print(f"  score_fetches now     : {current_users:,}")
    print(f"  scores in new DB      : {current_scores:,}")
    print("  done")


def copy_beatmaps(conn: sqlite3.Connection) -> None:
    require_tables(conn, "main", ["beatmaps", "scores"])
    require_tables(conn, "old", ["beatmaps"])

    reusable = scalar(
        conn,
        """
        SELECT COUNT(DISTINCT s.beatmap_id)
        FROM main.scores s
        JOIN old.beatmaps b ON b.beatmap_id=s.beatmap_id
        """,
    )
    needed = scalar(conn, "SELECT COUNT(DISTINCT beatmap_id) FROM main.scores")
    print("[copy beatmaps]")
    print(f"  score-referenced maps : {needed:,}")
    print(f"  found in old metadata : {reusable:,}")

    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO main.beatmaps
            SELECT b.*
            FROM old.beatmaps b
            WHERE EXISTS (
                SELECT 1 FROM main.scores s WHERE s.beatmap_id=b.beatmap_id
            )
            """
        )
    print(f"  beatmaps in new DB    : {scalar(conn, 'SELECT COUNT(*) FROM main.beatmaps'):,}")
    print("  done")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--old", default="farm.db", help="existing/source DB")
    p.add_argument("--new", default="farm_v2.db", help="new balanced DB")
    p.add_argument("command", choices=["check", "copy-scores", "copy-beatmaps"])
    return p.parse_args()


args = parse_args()
if Path(args.old).resolve() == Path(args.new).resolve():
    raise SystemExit("--old and --new must be different database files")
if not os.path.exists(args.old):
    raise SystemExit(f"Old DB not found: {args.old}")
if not os.path.exists(args.new):
    raise SystemExit(f"New DB not found: {args.new}")

conn = connect_new(args.new)
try:
    attach_old(conn, args.old)
    if args.command == "check":
        print_check(conn)
    elif args.command == "copy-scores":
        copy_scores(conn)
    elif args.command == "copy-beatmaps":
        copy_beatmaps(conn)
finally:
    conn.close()