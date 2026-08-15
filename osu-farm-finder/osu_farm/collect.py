"""Data collection: player pool -> top-100 scores -> beatmap metadata.

Every stage is resumable. Re-running skips work already in the database.

On sampling depth: the GLOBAL rankings endpoint caps at page 200 (10,000
players), all of whom are very high level. That pool cannot populate the lower
pp brackets at all. COUNTRY rankings get their own 200-page allowance each, so
iterating countries reaches far deeper into the tail. Use collect_players_deep
unless you specifically only care about the top of the game.
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter, deque

from .core import Config, OsuClient, connect, normalize_mods

PAGE_SIZE = 50
MAX_RANKING_PAGE = 200  # per ranking list, global or per-country

# Default bracket quotas. Keys are the lower bound of each pp bracket.
DEFAULT_QUOTAS = {0: 4500, 2000: 4000, 3500: 3500, 5000: 1500, 7000: 500}


def _bracket_of(pp: float) -> int:
    # MUST match the keys in DEFAULT_QUOTAS (or whatever quotas dict is
    # passed to collect_players_deep). A mismatch here means most players
    # get bucketed into a value that isn't in quotas at all, and
    # collect_players_deep silently skips them -- collection looks like it's
    # working but quietly stops populating most brackets.
    for lo in (7000, 5000, 3500, 2000):
        if pp >= lo:
            return lo
    return 0


async def collect_players(cfg: Config, pages: int = 200, country: str | None = None) -> int:
    """Walk one ranking list (global, or a single country) and upsert players."""
    conn = connect(cfg.db_path)
    n = 0
    async with OsuClient(cfg) as client:
        for page in range(1, min(pages, MAX_RANKING_PAGE) + 1):
            params = {"page": page}
            if country:
                params["country"] = country
            data = await client.get(f"/rankings/{cfg.mode}/performance", params)
            if not data or not data.get("ranking"):
                break
            rows = []
            for entry in data["ranking"]:
                u = entry.get("user", {})
                rows.append((
                    u.get("id"), u.get("username"), u.get("country_code"),
                    entry.get("global_rank"), entry.get("pp"),
                    entry.get("play_count"), u.get("last_visit"), time.time(),
                ))
            conn.executemany(
                "INSERT OR REPLACE INTO players VALUES (?,?,?,?,?,?,?,?)", rows)
            conn.commit()
            n += len(rows)
            if page % 20 == 0:
                print(f"  page {page}: {n} players")
    conn.close()
    return n


async def get_country_codes(cfg: Config, limit: int = 60) -> list[str]:
    """Country codes ordered by total playerbase, largest first."""
    codes = []
    async with OsuClient(cfg) as client:
        for page in (1, 2):
            data = await client.get(f"/rankings/{cfg.mode}/country", {"page": page})
            if not data:
                break
            for entry in data.get("ranking", []):
                code = entry.get("code") or (entry.get("country") or {}).get("code")
                if code:
                    codes.append(code)
    return codes[:limit]


async def collect_players_deep(
    cfg: Config,
    quotas: dict[int, int] | None = None,
    countries: list[str] | None = None,
    pp_floor: float = 1000.0,
    pages_per_country: int = 200,
) -> dict[int, int]:
    """Stratified sampling across country rankings until bracket quotas fill.

    Walks each country's ranking from the top down. Stops that country early
    once pp drops below pp_floor, and skips storing players whose bracket is
    already full -- so the expensive top-100 pull later isn't wasted on an
    already-overrepresented tier.
    """
    quotas = quotas or dict(DEFAULT_QUOTAS)
    conn = connect(cfg.db_path)

    filled = Counter()
    for row in conn.execute("SELECT pp FROM players WHERE pp IS NOT NULL"):
        filled[_bracket_of(row["pp"])] += 1
    print("  existing coverage:", dict(sorted(filled.items())) or "empty")

    if countries is None:
        print("  fetching country list")
        countries = await get_country_codes(cfg)
    print(f"  walking {len(countries)} countries, floor {pp_floor:.0f}pp")

    async with OsuClient(cfg) as client:
        for ci, code in enumerate(countries, 1):
            if all(filled[b] >= q for b, q in quotas.items()):
                print("  all quotas filled")
                break

            added_here = 0
            for page in range(1, min(pages_per_country, MAX_RANKING_PAGE) + 1):
                data = await client.get(
                    f"/rankings/{cfg.mode}/performance",
                    {"page": page, "country": code})
                if not data or not data.get("ranking"):
                    break

                rows, page_min_pp = [], 1e9
                for entry in data["ranking"]:
                    pp = entry.get("pp") or 0.0
                    page_min_pp = min(page_min_pp, pp)
                    b = _bracket_of(pp)
                    if b not in quotas or filled[b] >= quotas[b]:
                        continue
                    u = entry.get("user", {})
                    if not u.get("id"):
                        continue
                    rows.append((
                        u.get("id"), u.get("username"), u.get("country_code"),
                        entry.get("global_rank"), pp, entry.get("play_count"),
                        u.get("last_visit"), time.time(),
                    ))
                    filled[b] += 1

                if rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO players VALUES (?,?,?,?,?,?,?,?)", rows)
                    conn.commit()
                    added_here += len(rows)

                # This country has run past the floor -- move on.
                if page_min_pp < pp_floor:
                    break
                if all(filled[b] >= q for b, q in quotas.items()):
                    break

            print(f"  [{ci}/{len(countries)}] {code}: +{added_here} "
                  f"| totals {dict(sorted(filled.items()))}")

    conn.close()
    return dict(filled)


TIMEOUT = object()  # sentinel: request never completed, distinct from "no scores"


async def _fetch_top(client, cfg, user_id, hard_timeout=45.0):
    """One player's top-100, with an outer ceiling independent of internal retries.

    Returns TIMEOUT (not None) when the request never completes, so the caller
    can count it as a failure. Returning None here would be indistinguishable
    from a genuine empty score list and would hide real breakage.
    """
    try:
        result = await asyncio.wait_for(
            client.get(f"/users/{user_id}/scores/best",
                      {"mode": cfg.mode, "limit": 100, "legacy_only": 0}),
            timeout=hard_timeout,
        )
        return user_id, (TIMEOUT if result is None else result)
    except asyncio.TimeoutError:
        return user_id, TIMEOUT


async def collect_scores(cfg: Config, limit: int | None = None, refresh_days: int = 14,
                         balanced: bool = True, progress_every: float = 2.0,
                         commit_every: int = 3, stall_warn_after: float = 45.0) -> int:
    """Pull top-100 lists.

    With balanced=True, players are interleaved across pp brackets so that a
    partial run still covers every skill tier rather than only the strongest.

    Each player's scores are committed to the database as soon as that
    player's request finishes -- not once per 40-player batch. A stall now
    costs at most a few players' worth of re-fetching, not a whole batch, and
    everything before the stall point is already safely on disk.
    """
    conn = connect(cfg.db_path)
    cutoff = time.time() - refresh_days * 86400
    done = {r["user_id"] for r in conn.execute(
        "SELECT DISTINCT s.user_id FROM scores s JOIN players p USING(user_id) "
        "WHERE p.fetched_at > ?", (cutoff,))}

    rows = [(r["user_id"], r["pp"]) for r in conn.execute(
        "SELECT user_id, pp FROM players WHERE pp IS NOT NULL ORDER BY pp DESC")
        if r["user_id"] not in done]

    if balanced:
        buckets: dict[int, list[int]] = {}
        for uid, pp in rows:
            buckets.setdefault(_bracket_of(pp), []).append(uid)
        todo, order, i = [], sorted(buckets), 0
        while any(buckets[b] for b in order):
            b = order[i % len(order)]
            if buckets[b]:
                todo.append(buckets[b].pop(0))
            i += 1
    else:
        todo = [uid for uid, _ in rows]

    if limit:
        todo = todo[:limit]
    print(f"  {len(todo)} players to fetch")
    if not todo:
        conn.close()
        return 0

    total, errors, empty = 0, 0, 0
    done_n, t0 = 0, time.time()
    last_print = last_progress_at = time.time()
    pending_rows: list[tuple] = []
    stall_warned = False
    recent: deque[int] = deque(maxlen=20)   # 1 = failure, 0 = success
    cur_rpm = float(cfg.requests_per_minute)
    floor_rpm, ceil_rpm = 10.0, float(cfg.requests_per_minute)
    cooldowns = 0

    def tick(force=False):
        nonlocal last_print
        now = time.time()
        if not force and now - last_print < progress_every:
            return
        last_print = now
        elapsed = max(now - t0, 1e-6)
        rate = done_n / elapsed
        eta = (len(todo) - done_n) / rate if rate > 0 else 0
        pct = 100.0 * done_n / max(len(todo), 1)
        print(f"\r  {done_n}/{len(todo)} players ({pct:5.1f}%) | "
              f"{total:,} scores | {rate * 60:.0f} players/min | "
              f"ETA {eta / 60:.0f}m | {errors} err   ", end="", flush=True)

    def flush():
        nonlocal pending_rows
        if pending_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO scores VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", pending_rows)
            conn.commit()
            pending_rows = []

    async def watchdog():
        nonlocal stall_warned
        while True:
            await asyncio.sleep(5)
            idle = time.time() - last_progress_at
            if idle > stall_warn_after:
                if not stall_warned:
                    print(f"\n  no player has finished in {idle:.0f}s -- "
                          f"one request may be stuck. Safe to Ctrl+C: "
                          f"{done_n} players are already committed, a "
                          f"re-run will resume from there.")
                    stall_warned = True
            else:
                stall_warned = False

    # ONE client for the whole run. A client per batch means a fresh OAuth
    # token per batch, and the token endpoint 429s long before the API does.
    async with OsuClient(cfg) as client:
        watchdog_task = asyncio.create_task(watchdog())
        try:
            for chunk_start in range(0, len(todo), cfg.concurrency * 4):
                chunk = todo[chunk_start:chunk_start + cfg.concurrency * 4]
                for fut in asyncio.as_completed(
                        [_fetch_top(client, cfg, uid) for uid in chunk]):
                    try:
                        res = await fut
                    except BaseException as e:  # noqa: BLE001
                        res = e
                    done_n += 1
                    last_progress_at = time.time()

                    if isinstance(res, BaseException):
                        errors += 1
                        recent.append(1)
                        if errors <= 3:
                            print(f"\n    error: {type(res).__name__}: {res}")
                    elif not res:
                        errors += 1
                        recent.append(1)
                    else:
                        user_id, scores = res
                        if scores is TIMEOUT:
                            # Request never completed. NOT the same as a player
                            # with no scores -- counting it as empty is what
                            # made earlier failures invisible.
                            errors += 1
                            recent.append(1)
                        elif not scores:
                            empty += 1
                            recent.append(0)
                        else:
                            recent.append(0)
                            for pos, sc in enumerate(scores, start=1):
                                bm = sc.get("beatmap") or {}
                                st = sc.get("statistics") or {}
                                bid = bm.get("id")
                                sid = sc.get("id") or sc.get("best_id")
                                if not (bid and sid):
                                    continue
                                pending_rows.append((
                                    sid, user_id, bid, pos,
                                    sc.get("pp"), sc.get("accuracy"),
                                    sc.get("max_combo"),
                                    st.get("count_miss", st.get("miss", 0)),
                                    st.get("count_300", st.get("great", 0)),
                                    st.get("count_100", st.get("ok", 0)),
                                    st.get("count_50", st.get("meh", 0)),
                                    normalize_mods(sc.get("mods")),
                                    sc.get("rank"),
                                    sc.get("created_at") or sc.get("ended_at"),
                                ))
                    if done_n % commit_every == 0:
                        flush()
                        total = conn.execute(
                            "SELECT COUNT(*) c FROM scores").fetchone()["c"]
                    tick()

                    # Adaptive throttle. Sustained failures almost always mean
                    # server-side throttling rather than broken code, so slow
                    # down and pause instead of grinding through the list.
                    if len(recent) == recent.maxlen:
                        fail_rate = sum(recent) / len(recent)
                        if fail_rate >= 0.5:
                            cooldowns += 1
                            pause = min(60 * cooldowns, 600)
                            cur_rpm = max(cur_rpm / 2, floor_rpm)
                            client.limiter.set_rate(cur_rpm)
                            print(f"\n  {fail_rate:.0%} of the last "
                                  f"{len(recent)} requests failed -- pausing "
                                  f"{pause}s and dropping to {cur_rpm:.0f} "
                                  f"req/min ({done_n} players done, all "
                                  f"committed)")
                            recent.clear()
                            await asyncio.sleep(pause)
                        elif fail_rate == 0 and cur_rpm < ceil_rpm:
                            cur_rpm = min(cur_rpm * 1.25, ceil_rpm)
                            client.limiter.set_rate(cur_rpm)
                            recent.clear()

                    if done_n == cfg.concurrency and errors >= cfg.concurrency:
                        print("\n    every one of the first requests failed "
                              "-- stopping early. Check credentials/network.")
                        flush()
                        watchdog_task.cancel()
                        conn.close()
                        return total
        finally:
            watchdog_task.cancel()

    flush()
    total = conn.execute("SELECT COUNT(*) c FROM scores").fetchone()["c"]
    tick(force=True)
    print()
    conn.close()
    if errors or empty:
        print(f"  finished with {errors} failed and {empty} empty responses")
    return total


async def collect_beatmaps(cfg: Config, refresh_days: int = 30) -> int:
    """Backfill metadata for every beatmap referenced by a stored score.

    passcount is the field that matters most -- it is the exposure denominator
    for the overrepresentation model.
    """
    conn = connect(cfg.db_path)
    cutoff = time.time() - refresh_days * 86400
    todo = [r["beatmap_id"] for r in conn.execute(
        "SELECT DISTINCT s.beatmap_id FROM scores s "
        "LEFT JOIN beatmaps b USING(beatmap_id) "
        "WHERE b.beatmap_id IS NULL OR b.fetched_at < ?", (cutoff,))]
    print(f"  {len(todo)} beatmaps to fetch")

    total = 0
    async with OsuClient(cfg) as client:
        for i in range(0, len(todo), 50):
            chunk = todo[i:i + 50]
            data = await client.get("/beatmaps", [("ids[]", b) for b in chunk])
            if not data:
                continue
            rows = []
            for b in data.get("beatmaps", []):
                st = b.get("beatmapset") or {}
                rows.append((
                    b.get("id"), b.get("beatmapset_id"),
                    st.get("title"), st.get("artist"), b.get("version"),
                    b.get("status"), b.get("difficulty_rating"),
                    b.get("cs"), b.get("ar"), b.get("accuracy"), b.get("drain"),
                    b.get("bpm"), b.get("total_length"), b.get("hit_length"),
                    b.get("max_combo"), b.get("count_circles"),
                    b.get("count_sliders"), b.get("count_spinners"),
                    b.get("playcount"), b.get("passcount"),
                    b.get("ranked_date") or st.get("ranked_date"), time.time(),
                ))
            conn.executemany(
                "INSERT OR REPLACE INTO beatmaps VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            conn.commit()
            total += len(rows)
            print(f"\r  {min(i + len(chunk), len(todo))}/{len(todo)} beatmaps",
                  end="", flush=True)
    print()
    conn.close()
    return total


def coverage(db_path: str) -> None:
    """Print how many players and scores exist per pp bracket."""
    conn = connect(db_path)
    print(f"{'bracket':>14} {'players':>9} {'scores':>10} {'pp range':>18}")
    for lo, hi in [(0, 2000), (2000, 3500), (3500, 5000), (5000, 7000), (7000, 99999)]:
        r = conn.execute(
            "SELECT COUNT(DISTINCT p.user_id) np, COUNT(s.score_id) ns, "
            "MIN(p.pp) lo, MAX(p.pp) hi FROM players p "
            "LEFT JOIN scores s USING(user_id) WHERE p.pp >= ? AND p.pp < ?",
            (lo, hi)).fetchone()
        rng = f"{r['lo']:.0f}-{r['hi']:.0f}" if r["lo"] else "-"
        print(f"{lo:>7}-{hi:<6} {r['np']:>9} {r['ns']:>10} {rng:>18}")
    conn.close()


async def run_all(cfg: Config, pages: int = 200, player_limit: int | None = None,
                  deep: bool = True, pp_floor: float = 1000.0):
    print("[1/3] players")
    if deep:
        await collect_players_deep(cfg, pp_floor=pp_floor)
    else:
        await collect_players(cfg, pages=pages)
    print("[2/3] top-100 scores")
    await collect_scores(cfg, limit=player_limit)
    print("[3/3] beatmap metadata")
    await collect_beatmaps(cfg)
    print("\ncoverage:")
    coverage(cfg.db_path)