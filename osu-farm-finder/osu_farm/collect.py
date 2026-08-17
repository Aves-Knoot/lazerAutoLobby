"""Data collection: balanced player pool -> current top-100 -> beatmap metadata.

The collector deliberately separates three ideas:

1. Player sampling is stratified by relatively narrow pp bands.
2. No one country is allowed to dominate a band while the sample is built.
3. A successful score refresh atomically REPLACES that player's stored top-100.

That last point matters on repeated collections: a score that falls out of a
player's top 100 must disappear from the database rather than lingering forever.

Every completed player-score refresh and beatmap metadata refresh is resumable.
For a new clean study, point --db at a new filename rather than overwriting the
old baseline database.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import Counter, defaultdict, deque

from .core import Config, OsuClient, connect, normalize_mods

PAGE_SIZE = 50
MAX_RANKING_PAGE = 200  # per ranking list, global or per-country

# Collection bands are intentionally narrower than the analysis brackets.
# That prevents a broad 5k-7k bucket, for example, from being filled entirely
# at one edge while leaving a large hole at the other edge.
#
# Total target = 15,000 players. At 100 top scores/player this gives a ceiling
# near 1.5M current top-play observations before API failures/empty profiles.
DEFAULT_QUOTAS = {
    1000: 1500,
    2000: 1500,
    3000: 1500,
    4000: 1500,
    5000: 1500,
    6000: 1500,
    7000: 1500,
    8000: 1500,
    9000: 1500,
    10000: 1000,
    12000: 500,
}

# A country may supply at most this fraction of a pp band's target while the
# sample is being built. 15% means a filled band necessarily draws from at
# least seven countries when the ranking data permits it.
DEFAULT_MAX_COUNTRY_SHARE = 0.15
DEFAULT_COUNTRY_LIMIT = 60
MIN_COUNTRY_CAP = 25


def _bracket_of(pp: float, quotas: dict[int, int] | None = None) -> int | None:
    """Return the lower bound of the collection band containing ``pp``."""
    qs = quotas or DEFAULT_QUOTAS
    for lo in sorted(qs, reverse=True):
        if pp >= lo:
            return lo
    return None


def _band_hi(lo: int, quotas: dict[int, int] | None = None) -> int:
    qs = quotas or DEFAULT_QUOTAS
    keys = sorted(qs)
    i = keys.index(lo)
    return keys[i + 1] if i + 1 < len(keys) else 99999


def _spread_country_order(codes: list[str]) -> list[str]:
    """Interleave large and smaller countries from the ranking country list.

    get_country_codes() returns the largest playerbases first. Taking the list
    as-is still makes early quota filling favor only the largest countries, so
    use high/low/high/low order. The per-band country cap is the hard guard;
    this ordering simply improves diversity before a quota fills.
    """
    out: list[str] = []
    left, right = 0, len(codes) - 1
    while left <= right:
        out.append(codes[left])
        left += 1
        if left <= right:
            out.append(codes[right])
            right -= 1
    return out


def _country_interleave(rows: list[tuple[int, float, str]]) -> list[int]:
    """Round-robin user IDs across countries within one pp band."""
    buckets: dict[str, deque[int]] = defaultdict(deque)
    for uid, _pp, country in rows:
        buckets[country or "??"].append(uid)

    order = sorted(buckets)
    out: list[int] = []
    while order:
        next_order: list[str] = []
        for country in order:
            bucket = buckets[country]
            if bucket:
                out.append(bucket.popleft())
            if bucket:
                next_order.append(country)
        order = next_order
    return out


async def collect_players(cfg: Config, pages: int = 200,
                          country: str | None = None) -> int:
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
                if not u.get("id"):
                    continue
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


async def get_country_codes(cfg: Config, limit: int = DEFAULT_COUNTRY_LIMIT) -> list[str]:
    """Country codes ordered by total playerbase, largest first."""
    codes = []
    async with OsuClient(cfg) as client:
        for page in (1, 2):
            data = await client.get(f"/rankings/{cfg.mode}/country", {"page": page})
            if not data:
                break
            for entry in data.get("ranking", []):
                code = entry.get("code") or (entry.get("country") or {}).get("code")
                if code and code not in codes:
                    codes.append(code)
    return codes[:limit]


async def collect_players_deep(
    cfg: Config,
    quotas: dict[int, int] | None = None,
    countries: list[str] | None = None,
    pp_floor: float = 1000.0,
    pages_per_country: int = 200,
    max_country_share: float = DEFAULT_MAX_COUNTRY_SHARE,
    country_limit: int = DEFAULT_COUNTRY_LIMIT,
) -> dict[int, int]:
    """Build a pp-stratified, country-capped player sample.

    The old collector could walk one country deeply enough to fill an entire
    broad pp quota before another country was considered. Here the pp bands are
    narrower and every band has a per-country cap.

    This remains resumable: players already stored in the database count toward
    both the global band quota and that country's cap. Re-running therefore
    continues filling missing bands rather than duplicating selected players.
    """
    quotas = quotas or dict(DEFAULT_QUOTAS)
    if not 0 < max_country_share <= 1:
        raise ValueError("max_country_share must be > 0 and <= 1")

    conn = connect(cfg.db_path)

    filled: Counter[int] = Counter()
    by_country: Counter[tuple[str, int]] = Counter()
    existing_ids = set()
    for row in conn.execute(
            "SELECT user_id, country, pp FROM players WHERE pp IS NOT NULL"):
        b = _bracket_of(float(row["pp"]), quotas)
        if b not in quotas:
            continue
        filled[b] += 1
        by_country[(row["country"] or "??", b)] += 1
        existing_ids.add(row["user_id"])

    def all_full() -> bool:
        return all(filled[b] >= q for b, q in quotas.items())

    print("  existing collection coverage:")
    for b in sorted(quotas):
        print(f"    {b:>5}-{_band_hi(b, quotas):<5}: "
              f"{filled[b]:>4}/{quotas[b]:<4}")

    if all_full():
        print("  all collection quotas already filled; skipping ranking scan")
        conn.close()
        return dict(filled)

    if countries is None:
        print("  fetching country list")
        countries = await get_country_codes(cfg, limit=country_limit)
    countries = _spread_country_order(list(dict.fromkeys(countries)))

    caps = {
        b: max(MIN_COUNTRY_CAP, math.ceil(q * max_country_share))
        for b, q in quotas.items()
    }
    print(f"  walking up to {len(countries)} countries, floor {pp_floor:.0f}pp")
    print(f"  max country share per band: {max_country_share:.0%}")

    async with OsuClient(cfg) as client:
        for ci, code in enumerate(countries, 1):
            if all_full():
                print("  all quotas filled")
                break

            added_here = 0
            added_by_band: Counter[int] = Counter()

            for page in range(1, min(pages_per_country, MAX_RANKING_PAGE) + 1):
                data = await client.get(
                    f"/rankings/{cfg.mode}/performance",
                    {"page": page, "country": code},
                )
                if not data or not data.get("ranking"):
                    break

                rows = []
                page_min_pp = float("inf")
                for entry in data["ranking"]:
                    pp = float(entry.get("pp") or 0.0)
                    page_min_pp = min(page_min_pp, pp)
                    if pp < pp_floor:
                        continue

                    b = _bracket_of(pp, quotas)
                    if b not in quotas or filled[b] >= quotas[b]:
                        continue
                    if by_country[(code, b)] >= caps[b]:
                        continue

                    u = entry.get("user", {})
                    uid = u.get("id")
                    if not uid or uid in existing_ids:
                        continue

                    rows.append((
                        uid, u.get("username"), u.get("country_code") or code,
                        entry.get("global_rank"), pp, entry.get("play_count"),
                        u.get("last_visit"), time.time(),
                    ))
                    existing_ids.add(uid)
                    filled[b] += 1
                    by_country[(code, b)] += 1
                    added_here += 1
                    added_by_band[b] += 1

                if rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO players VALUES (?,?,?,?,?,?,?,?)", rows)
                    conn.commit()

                # Rankings are descending pp. Once the bottom of the page is
                # under our floor there is nothing useful on later pages.
                if page_min_pp < pp_floor:
                    break
                if all_full():
                    break

            changed = ", ".join(
                f"{b//1000}k:+{n}" for b, n in sorted(added_by_band.items())) or "+0"
            remaining = sum(max(quotas[b] - filled[b], 0) for b in quotas)
            print(f"  [{ci}/{len(countries)}] {code}: {changed} "
                  f"({added_here} players) | {remaining} slots remain")

    print("  final collection coverage:")
    for b in sorted(quotas):
        status = "OK" if filled[b] >= quotas[b] else "UNDERFILLED"
        print(f"    {b:>5}-{_band_hi(b, quotas):<5}: "
              f"{filled[b]:>4}/{quotas[b]:<4} {status}")
    if not all_full():
        print("  NOTE: one or more bands could not reach target while respecting "
              "the country cap. That is safer than silently letting one country "
              "dominate the band; we can relax the cap after inspecting coverage.")

    conn.close()
    return dict(filled)


TIMEOUT = object()  # request never completed, distinct from a real empty top-100


async def _fetch_top(client, cfg, user_id, hard_timeout=45.0):
    """Fetch one player's current top-100 with an outer timeout ceiling."""
    try:
        result = await asyncio.wait_for(
            client.get(
                f"/users/{user_id}/scores/best",
                {"mode": cfg.mode, "limit": 100, "legacy_only": 0},
            ),
            timeout=hard_timeout,
        )
        return user_id, (TIMEOUT if result is None else result)
    except asyncio.TimeoutError:
        return user_id, TIMEOUT


def _score_rows(user_id: int, scores: list[dict]) -> list[tuple]:
    rows: list[tuple] = []
    for pos, sc in enumerate(scores, start=1):
        bm = sc.get("beatmap") or {}
        st = sc.get("statistics") or {}
        bid = bm.get("id")
        sid = sc.get("id") or sc.get("best_id")
        if not (bid and sid):
            continue
        rows.append((
            sid, user_id, bid, pos,
            sc.get("pp"), sc.get("accuracy"), sc.get("max_combo"),
            st.get("count_miss", st.get("miss", 0)),
            st.get("count_300", st.get("great", 0)),
            st.get("count_100", st.get("ok", 0)),
            st.get("count_50", st.get("meh", 0)),
            normalize_mods(sc.get("mods")), sc.get("rank"),
            sc.get("created_at") or sc.get("ended_at"),
        ))
    return rows


def _replace_user_scores(conn, user_id: int, rows: list[tuple]) -> tuple[int, int]:
    """Atomically replace one player's stored top-100 and mark refresh time.

    Returns (old_count, new_count). On a failed API request this function is
    never called, so a good previous snapshot is preserved for a later retry.
    """
    old_count = conn.execute(
        "SELECT COUNT(*) c FROM scores WHERE user_id=?", (user_id,)
    ).fetchone()["c"]

    with conn:
        conn.execute("DELETE FROM scores WHERE user_id=?", (user_id,))
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO scores VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        conn.execute(
            "INSERT OR REPLACE INTO score_fetches(user_id, fetched_at, score_count) "
            "VALUES (?,?,?)",
            (user_id, time.time(), len(rows)),
        )
    return old_count, len(rows)


async def collect_scores(cfg: Config, limit: int | None = None,
                         refresh_days: int = 14, balanced: bool = True,
                         progress_every: float = 2.0,
                         stall_warn_after: float = 45.0) -> int:
    """Pull current top-100 lists and atomically refresh each stored player.

    Freshness is read from score_fetches, not players.fetched_at. With
    balanced=True, the todo order is round-robin across both pp bands and
    countries, so even an interrupted/limited run is representative.
    """
    conn = connect(cfg.db_path)
    cutoff = time.time() - refresh_days * 86400
    done = {
        r["user_id"] for r in conn.execute(
            "SELECT user_id FROM score_fetches WHERE fetched_at > ?", (cutoff,)
        )
    }

    player_rows = [
        (r["user_id"], float(r["pp"]), r["country"] or "??")
        for r in conn.execute(
            "SELECT user_id, pp, country FROM players "
            "WHERE pp IS NOT NULL ORDER BY pp DESC"
        )
        if r["user_id"] not in done
    ]

    if balanced:
        band_rows: dict[int, list[tuple[int, float, str]]] = defaultdict(list)
        for row in player_rows:
            b = _bracket_of(row[1])
            if b is not None:
                band_rows[b].append(row)

        by_band = {
            b: deque(_country_interleave(rows))
            for b, rows in band_rows.items()
        }
        order = sorted(by_band)
        todo: list[int] = []
        while order:
            next_order: list[int] = []
            for b in order:
                bucket = by_band[b]
                if bucket:
                    todo.append(bucket.popleft())
                if bucket:
                    next_order.append(b)
            order = next_order
    else:
        todo = [uid for uid, _pp, _country in player_rows]

    if limit:
        todo = todo[:limit]
    print(f"  {len(todo)} players to fetch ({len(done)} already fresh)")
    if not todo:
        total = conn.execute("SELECT COUNT(*) c FROM scores").fetchone()["c"]
        conn.close()
        return total

    stored_total = conn.execute("SELECT COUNT(*) c FROM scores").fetchone()["c"]
    errors = empty = refreshed = 0
    done_n, t0 = 0, time.time()
    last_print = last_progress_at = time.time()
    stall_warned = False
    recent: deque[int] = deque(maxlen=20)  # 1 = failure, 0 = success
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
              f"{stored_total:,} current scores | {rate * 60:.0f} players/min | "
              f"ETA {eta / 60:.0f}m | {errors} err   ", end="", flush=True)

    async def watchdog():
        nonlocal stall_warned
        while True:
            await asyncio.sleep(5)
            idle = time.time() - last_progress_at
            if idle > stall_warn_after:
                if not stall_warned:
                    print(f"\n  no player has finished in {idle:.0f}s -- "
                          f"one request may be stuck. Safe to Ctrl+C: "
                          f"{refreshed} successful player snapshots are already "
                          f"committed; a re-run will resume from score_fetches.")
                    stall_warned = True
            else:
                stall_warned = False

    # One client for the whole run; creating one per batch would repeatedly hit
    # the OAuth token endpoint and invite throttling.
    async with OsuClient(cfg) as client:
        watchdog_task = asyncio.create_task(watchdog())
        try:
            for chunk_start in range(0, len(todo), cfg.concurrency * 4):
                chunk = todo[chunk_start:chunk_start + cfg.concurrency * 4]
                for fut in asyncio.as_completed(
                        [_fetch_top(client, cfg, uid) for uid in chunk]):
                    try:
                        res = await fut
                    except BaseException as exc:  # noqa: BLE001
                        res = exc

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
                            errors += 1
                            recent.append(1)
                        else:
                            recent.append(0)
                            rows = _score_rows(user_id, scores or [])
                            old_n, new_n = _replace_user_scores(conn, user_id, rows)
                            stored_total += new_n - old_n
                            refreshed += 1
                            if not scores:
                                empty += 1

                    tick()

                    # Adaptive throttle. Sustained failures are much more useful
                    # as a signal to slow down than as a reason to burn through
                    # thousands of profiles with bad responses.
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
                                  f"req/min ({refreshed} snapshots committed)")
                            recent.clear()
                            await asyncio.sleep(pause)
                        elif fail_rate == 0 and cur_rpm < ceil_rpm:
                            cur_rpm = min(cur_rpm * 1.25, ceil_rpm)
                            client.limiter.set_rate(cur_rpm)
                            recent.clear()

                    if done_n == cfg.concurrency and errors >= cfg.concurrency:
                        print("\n    every one of the first requests failed "
                              "-- stopping early. Check credentials/network.")
                        watchdog_task.cancel()
                        conn.close()
                        return stored_total
        finally:
            watchdog_task.cancel()

    tick(force=True)
    print()
    conn.close()
    if errors or empty:
        print(f"  finished with {errors} failed and {empty} genuine empty responses")
    return stored_total


async def collect_beatmaps(cfg: Config, refresh_days: int = 30) -> int:
    """Backfill metadata for every beatmap referenced by a current stored score.

    passcount is especially important because it is the exposure denominator
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
    """Show sample balance in the narrower collection pp bands."""
    conn = connect(db_path)
    print(f"{'pp band':>13} {'target':>7} {'players':>8} {'scored':>8} "
          f"{'scores':>10} {'countries':>10} {'top share':>10}")
    for lo in sorted(DEFAULT_QUOTAS):
        hi = _band_hi(lo)
        target = DEFAULT_QUOTAS[lo]
        row = conn.execute(
            "SELECT COUNT(*) players, COUNT(DISTINCT country) countries "
            "FROM players WHERE pp >= ? AND pp < ?", (lo, hi)
        ).fetchone()
        scored = conn.execute(
            "SELECT COUNT(DISTINCT p.user_id) n, COUNT(s.score_id) ns "
            "FROM players p JOIN scores s USING(user_id) "
            "WHERE p.pp >= ? AND p.pp < ?", (lo, hi)
        ).fetchone()
        top = conn.execute(
            "SELECT country, COUNT(*) n FROM players "
            "WHERE pp >= ? AND pp < ? GROUP BY country ORDER BY n DESC LIMIT 1",
            (lo, hi),
        ).fetchone()
        share = (top["n"] / row["players"] if top and row["players"] else 0.0)
        print(f"{lo:>6}-{hi:<6} {target:>7} {row['players']:>8} "
              f"{scored['n']:>8} {scored['ns']:>10} {row['countries']:>10} "
              f"{share:>9.1%}")
    conn.close()


async def run_all(cfg: Config, pages: int = 200,
                  player_limit: int | None = None, deep: bool = True,
                  pp_floor: float = 1000.0):
    print("[1/3] players")
    if deep:
        await collect_players_deep(cfg, pp_floor=pp_floor)
    else:
        await collect_players(cfg, pages=pages)
    print("[2/3] current top-100 scores")
    await collect_scores(cfg, limit=player_limit)
    print("[3/3] beatmap metadata")
    await collect_beatmaps(cfg)
    print("\ncoverage:")
    coverage(cfg.db_path)