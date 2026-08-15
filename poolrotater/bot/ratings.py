"""Per-map feedback and silent performance evidence collected from the lobby.

This database is intentionally separate from farm.db. Human .good/.bad votes
and observed "this map just produced unusually strong top plays" evidence are
expensive live signals that should survive rebuilding the farm-finder data.

Human votes stay in ``map_ratings``.

Automatically detected performance evidence goes into two separate tables:
``performance_evidence`` is one aggregate row per notable match, while
``performance_records`` stores the individual top-play results that caused the
match to qualify. Automatic evidence never pretends to be a human .good vote.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Iterable

log = logging.getLogger("ratings")

SCHEMA = """
CREATE TABLE IF NOT EXISTS map_ratings (
    beatmap_id INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    rating     INTEGER NOT NULL,   -- +1 good, -1 bad
    created_at REAL,
    PRIMARY KEY (beatmap_id, user_id)
);
CREATE INDEX IF NOT EXISTS ix_ratings_map ON map_ratings(beatmap_id);

CREATE TABLE IF NOT EXISTS performance_evidence (
    evidence_key      TEXT PRIMARY KEY,
    beatmap_id        INTEGER NOT NULL,
    room_id           INTEGER,
    playlist_item_id  INTEGER,
    created_at        REAL NOT NULL,

    participant_count INTEGER NOT NULL,
    scored_count      INTEGER NOT NULL,
    top100_count      INTEGER NOT NULL,
    top30_count       INTEGER NOT NULL,
    top20_count       INTEGER NOT NULL,
    top10_count       INTEGER NOT NULL,
    top1_count        INTEGER NOT NULL,
    best_rank         INTEGER,

    notable_ratio     REAL NOT NULL,
    evidence_score    REAL NOT NULL,
    trigger_reason    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_perf_evidence_map
    ON performance_evidence(beatmap_id);
CREATE INDEX IF NOT EXISTS ix_perf_evidence_created
    ON performance_evidence(created_at);

CREATE TABLE IF NOT EXISTS performance_records (
    evidence_key      TEXT NOT NULL,
    beatmap_id        INTEGER NOT NULL,
    room_id           INTEGER,
    playlist_item_id  INTEGER,
    user_id           INTEGER NOT NULL,
    top_rank          INTEGER NOT NULL,
    score_pp          REAL,
    profile_gain_pp   REAL,
    accuracy          REAL,
    created_at        REAL NOT NULL,
    PRIMARY KEY (evidence_key, user_id),
    FOREIGN KEY (evidence_key)
        REFERENCES performance_evidence(evidence_key)
        ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_perf_records_map
    ON performance_records(beatmap_id);
CREATE INDEX IF NOT EXISTS ix_perf_records_rank
    ON performance_records(top_rank);
"""


class RatingStore:
    def __init__(self, path: str = "ratings.db"):
        self.path = path
        conn = self._conn()
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=20)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        return c

    # ---------------------------------------------------------- human ratings

    def rate(self, beatmap_id: int, user_id: int, rating: int) -> bool:
        """Record a human vote.

        Returns True if this is a change worth acknowledging in chat. One vote
        per person per map; repeating the same vote is silently ignored.
        """
        conn = self._conn()
        prev = conn.execute(
            "SELECT rating FROM map_ratings WHERE beatmap_id=? AND user_id=?",
            (int(beatmap_id), int(user_id)),
        ).fetchone()

        if prev and prev["rating"] == rating:
            conn.close()
            return False

        conn.execute(
            "INSERT OR REPLACE INTO map_ratings VALUES (?,?,?,?)",
            (int(beatmap_id), int(user_id), int(rating), time.time()),
        )
        conn.commit()
        conn.close()
        return True

    def tally(self, beatmap_id: int) -> tuple[int, int]:
        conn = self._conn()
        r = conn.execute(
            "SELECT SUM(rating>0) up, SUM(rating<0) down "
            "FROM map_ratings WHERE beatmap_id=?",
            (int(beatmap_id),),
        ).fetchone()
        conn.close()
        return int(r["up"] or 0), int(r["down"] or 0)

    def worst(self, min_votes: int = 4, ratio: float = 0.6) -> list[int]:
        """Maps the lobby dislikes, for pruning from the pool later."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT beatmap_id, SUM(rating>0) up, SUM(rating<0) down,
                   COUNT(*) n
            FROM map_ratings
            GROUP BY beatmap_id
            HAVING n >= ?
            """,
            (min_votes,),
        ).fetchall()
        conn.close()
        return [
            r["beatmap_id"]
            for r in rows
            if (r["down"] or 0) / max(r["n"], 1) >= ratio
        ]

    # ------------------------------------------------ performance observations

    @staticmethod
    def _record_weight(rank: int) -> float:
        """Simple evidence weight retained in the DB for later analysis."""
        if rank <= 1:
            return 4.0
        if rank <= 10:
            return 3.0
        if rank <= 20:
            return 2.0
        if rank <= 30:
            return 1.0
        return 0.0

    def record_performance_evidence(
        self,
        *,
        beatmap_id: int,
        room_id: int | None,
        playlist_item_id: int | None,
        participant_count: int,
        scored_count: int,
        records: Iterable[dict],
        notable_ratio: float = 0.40,
        min_top20: int = 2,
        min_top30: int = 3,
    ) -> bool:
        """Silently persist a match when it looks like strong farm evidence.

        Default trigger rules intentionally match the lobby use-case:

        * any player gets a new #1;
        * at least two players get new top-20s;
        * at least three players get new top-30s; or
        * at least 40% of the match participants (minimum two people) get
          new top-30s.

        The aggregate match is stored once, plus the individual <= #30 records
        that contributed to the signal. Nothing is posted to lobby chat.

        Returns True if the match qualified and was written.
        """
        beatmap_id = int(beatmap_id)
        participant_count = max(int(participant_count or 0), 0)
        scored_count = max(int(scored_count or 0), 0)

        cleaned: list[dict] = []
        for rec in records:
            try:
                uid = int(rec["user_id"])
                rank = int(rec["rank"])
            except (KeyError, TypeError, ValueError):
                continue
            if rank < 1 or rank > 100:
                continue

            cleaned.append({
                "user_id": uid,
                "rank": rank,
                "pp": float(rec.get("pp") or 0.0),
                "gain": float(rec.get("gain") or 0.0),
                "accuracy": float(rec.get("accuracy") or 0.0),
            })

        if not cleaned:
            return False

        ranks = [r["rank"] for r in cleaned]
        top100 = len(ranks)
        top30 = sum(r <= 30 for r in ranks)
        top20 = sum(r <= 20 for r in ranks)
        top10 = sum(r <= 10 for r in ranks)
        top1 = sum(r <= 1 for r in ranks)
        best_rank = min(ranks)

        denom = max(participant_count, 1)
        ratio = top30 / denom
        evidence_score = sum(self._record_weight(r) for r in ranks)

        reasons: list[str] = []
        if top1 >= 1:
            reasons.append("top1")
        if top20 >= int(min_top20):
            reasons.append(f"{top20}_top20")
        if top30 >= int(min_top30):
            reasons.append(f"{top30}_top30")
        if top30 >= 2 and ratio >= float(notable_ratio):
            reasons.append(f"{ratio:.0%}_top30")

        if not reasons:
            return False

        now = time.time()
        room_part = "none" if room_id is None else str(int(room_id))
        item_part = "none" if playlist_item_id is None else str(int(playlist_item_id))
        evidence_key = f"{room_part}:{item_part}:{beatmap_id}"

        conn = self._conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO performance_evidence (
                evidence_key, beatmap_id, room_id, playlist_item_id, created_at,
                participant_count, scored_count,
                top100_count, top30_count, top20_count, top10_count, top1_count,
                best_rank, notable_ratio, evidence_score, trigger_reason
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                evidence_key,
                beatmap_id,
                int(room_id) if room_id is not None else None,
                int(playlist_item_id) if playlist_item_id is not None else None,
                now,
                participant_count,
                scored_count,
                top100,
                top30,
                top20,
                top10,
                top1,
                best_rank,
                ratio,
                evidence_score,
                ",".join(reasons),
            ),
        )

        # Only <= #30 records are detailed; weaker top-100s still contribute to
        # top100_count above but are not individually stored as "notable".
        for rec in cleaned:
            if rec["rank"] > 30:
                continue
            conn.execute(
                """
                INSERT OR REPLACE INTO performance_records (
                    evidence_key, beatmap_id, room_id, playlist_item_id,
                    user_id, top_rank, score_pp, profile_gain_pp,
                    accuracy, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    evidence_key,
                    beatmap_id,
                    int(room_id) if room_id is not None else None,
                    int(playlist_item_id) if playlist_item_id is not None else None,
                    rec["user_id"],
                    rec["rank"],
                    rec["pp"],
                    rec["gain"],
                    rec["accuracy"],
                    now,
                ),
            )

        conn.commit()
        conn.close()

        # Debug only: no lobby chat and normally no INFO console line.
        log.debug(
            "saved silent performance evidence map=%s participants=%s "
            "top30=%s best=#%s reasons=%s",
            beatmap_id, participant_count, top30, best_rank, ",".join(reasons),
        )
        return True