"""Config, database schema, and a polite rate-limited osu! API v2 client."""
from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

API = "https://osu.ppy.sh/api/v2"
TOKEN_URL = "https://osu.ppy.sh/oauth/token"

# The rework newspost is 2026-07-03; reindexing completed ~2026-07-26.
# Scores created after this are set under the current algorithm.
REWORK_DATE = "2026-07-26"


@dataclass
class Config:
    client_id: int = field(default_factory=lambda: int(os.environ.get("OSU_CLIENT_ID", 0)))
    client_secret: str = field(default_factory=lambda: os.environ.get("OSU_CLIENT_SECRET", ""))
    db_path: str = "farm.db"
    mode: str = "osu"
    # Documented cap is ~1200/min. We sit WAY below it on purpose -- slow and
    # steady is far less likely to trigger quiet server-side throttling that
    # shows up as hangs rather than a clean, retryable 429.
    requests_per_minute: int = 60
    concurrency: int = 2
    api_version: str = "20220705"
    user_agent: str = "osu-farm-finder/0.1 (personal research)"

    def validate(self) -> None:
        if not self.client_id or not self.client_secret:
            raise SystemExit(
                "Set OSU_CLIENT_ID and OSU_CLIENT_SECRET. Create an OAuth app at "
                "https://osu.ppy.sh/home/account/edit -> OAuth."
            )


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS players (
    user_id      INTEGER PRIMARY KEY,
    username     TEXT,
    country      TEXT,
    global_rank  INTEGER,
    pp           REAL,
    playcount    INTEGER,
    last_visit   TEXT,
    fetched_at   REAL
);

CREATE TABLE IF NOT EXISTS scores (
    score_id     INTEGER PRIMARY KEY,
    user_id      INTEGER NOT NULL,
    beatmap_id   INTEGER NOT NULL,
    position     INTEGER,
    pp           REAL,
    accuracy     REAL,
    max_combo    INTEGER,
    misses       INTEGER,
    n300 INTEGER, n100 INTEGER, n50 INTEGER,
    mods         TEXT,
    rank         TEXT,
    created_at   TEXT
);
CREATE INDEX IF NOT EXISTS ix_scores_map  ON scores(beatmap_id);
CREATE INDEX IF NOT EXISTS ix_scores_user ON scores(user_id);

-- Tracks when a player's COMPLETE current top-100 was last refreshed.
-- Do not infer score freshness from players.fetched_at: ranking collection
-- updates that timestamp independently of score collection.
CREATE TABLE IF NOT EXISTS score_fetches (
    user_id      INTEGER PRIMARY KEY,
    fetched_at   REAL NOT NULL,
    score_count  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS beatmaps (
    beatmap_id     INTEGER PRIMARY KEY,
    beatmapset_id  INTEGER,
    title          TEXT,
    artist         TEXT,
    version        TEXT,
    status         TEXT,
    sr             REAL,
    cs REAL, ar REAL, od REAL, hp REAL,
    bpm            REAL,
    total_length   INTEGER,
    hit_length     INTEGER,
    max_combo      INTEGER,
    count_circles  INTEGER,
    count_sliders  INTEGER,
    count_spinners INTEGER,
    playcount      INTEGER,
    passcount      INTEGER,
    ranked_date    TEXT,
    fetched_at     REAL
);

-- Derived offline from the .osu file; see features.py
CREATE TABLE IF NOT EXISTS map_features (
    beatmap_id                INTEGER PRIMARY KEY,
    aim REAL, speed REAL, flashlight REAL, stars REAL,
    speed_note_count          REAL,
    slider_factor             REAL,
    aim_strain_gini           REAL,
    speed_strain_gini         REAL,
    aim_spike_ratio           REAL,
    hardest_section_pos       REAL,   -- 0 = start of map, 1 = end
    late_difficulty_mass      REAL,   -- share of total strain in final third
    angle_entropy             REAL,
    slider_ratio              REAL,
    acc_gradient              REAL,   -- (pp@99 - pp@97) / 2
    forgiveness_ratio         REAL,   -- pp(97%, 1 miss) / pp(99% FC)
    pp_99fc                   REAL,
    computed_at               REAL
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _register_adapters() -> None:
    """sqlite3 stores numpy scalars as BLOBs, which silently breaks every JOIN.

    Register adapters once so numpy values coming out of pandas/numpy are
    always coerced to native types on the way in.
    """
    try:
        import numpy as np
    except ImportError:
        return
    for t in (np.int8, np.int16, np.int32, np.int64,
              np.uint8, np.uint16, np.uint32, np.uint64):
        sqlite3.register_adapter(t, int)
    for t in (np.float16, np.float32, np.float64):
        sqlite3.register_adapter(t, float)
    sqlite3.register_adapter(np.bool_, int)


_register_adapters()


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def set_meta(conn, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, str(value)))
    conn.commit()


def get_meta(conn, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


class RateLimiter:
    """Token bucket. Keeps us well under the documented ceiling.

    The rate is adjustable at runtime so the collector can back off on its own
    when the server starts throttling, and recover once it stops.
    """

    def __init__(self, per_minute: int):
        self.per_minute = max(per_minute, 1)
        self.interval = 60.0 / self.per_minute
        self._lock = asyncio.Lock()
        self._next = 0.0

    def set_rate(self, per_minute: float) -> None:
        self.per_minute = max(per_minute, 1)
        self.interval = 60.0 / self.per_minute

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
                now = time.monotonic()
            self._next = max(now, self._next) + self.interval


class OsuClient:
    """Async osu! API v2 client with client_credentials auth and retry."""

    def __init__(self, cfg: Config):
        cfg.validate()
        self.cfg = cfg
        self._token = None
        self._expires = 0.0
        self.limiter = RateLimiter(cfg.requests_per_minute)
        self._limiter = self.limiter
        self._sem = asyncio.Semaphore(cfg.concurrency)
        self._auth_lock = asyncio.Lock()
        # Split connect/read/write timeouts. A single combined timeout can
        # keep resetting on a slowly-trickling response and never fire; these
        # bound each phase independently so one bad connection can't hang a
        # whole batch indefinitely.
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0))

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def _auth(self, retries: int = 6) -> str:
        """Fetch/refresh the OAuth token.

        The token endpoint is rate-limited far more aggressively than the API
        itself, so this retries with backoff rather than raising. Reuse ONE
        OsuClient for a whole run -- constructing a new one per batch means a
        new token request per batch, which gets you 429'd fast.
        """
        async with self._auth_lock:
            if self._token and time.time() < self._expires - 120:
                return self._token
            for attempt in range(retries):
                try:
                    r = await self._http.post(
                        TOKEN_URL,
                        json={
                            "client_id": self.cfg.client_id,
                            "client_secret": self.cfg.client_secret,
                            "grant_type": "client_credentials",
                            "scope": "public",
                        },
                    )
                except (httpx.TransportError, httpx.TimeoutException):
                    await asyncio.sleep(2 ** attempt)
                    continue

                if r.status_code == 200:
                    data = r.json()
                    self._token = data["access_token"]
                    self._expires = time.time() + data.get("expires_in", 86400)
                    return self._token
                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After", 15 * (attempt + 1)))
                    print(f"    token endpoint rate-limited, waiting {wait:.0f}s")
                    await asyncio.sleep(min(wait, 180))
                    continue
                if r.status_code in (401, 403):
                    raise SystemExit(
                        "osu! rejected your credentials (HTTP "
                        f"{r.status_code}). Check OSU_CLIENT_ID / "
                        "OSU_CLIENT_SECRET."
                    )
                await asyncio.sleep(2 ** attempt)
            raise RuntimeError("Could not obtain an OAuth token after retries.")

    async def get(self, path: str, params=None, retries: int = 3,
                  request_timeout: float = 12.0):
        """GET /api/v2{path}. Returns parsed JSON, or None on a hard 404.

        Wrapped in asyncio.wait_for as a hard ceiling: this is what actually
        prevents one stuck request from stalling a whole batch, independent
        of whatever httpx's own timeout does. retries/timeout are kept low
        deliberately -- a fast failure that the caller can retry later beats
        a slow one that blocks a concurrency slot for minutes.
        """
        for attempt in range(retries):
            token = await self._auth()
            await self._limiter.wait()
            async with self._sem:
                try:
                    r = await asyncio.wait_for(
                        self._http.get(
                            API + path,
                            params=params,
                            headers={
                                "Authorization": f"Bearer {token}",
                                "x-api-version": self.cfg.api_version,
                                "User-Agent": self.cfg.user_agent,
                                "Accept": "application/json",
                            },
                        ),
                        timeout=request_timeout,
                    )
                except (httpx.TransportError, httpx.TimeoutException,
                        asyncio.TimeoutError):
                    if attempt < retries - 1:
                        print(f"\n    retry {attempt + 1}/{retries - 1} "
                              f"({path.split('/')[-2] if '/' in path else path})",
                              end="", flush=True)
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue

            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 401:
                self._token = None
                continue
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 2 ** attempt * 5))
                await asyncio.sleep(min(wait, 120))
                continue
            if 500 <= r.status_code < 600:
                await asyncio.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        return None


# ---------------------------------------------------------------- mods

# Mods that do not affect pp at all -- drop them so score buckets collapse.
COSMETIC = {"SD", "PF", "MR", "AT", "CN", "RX", "AP", "TP", "DA", "SV2", "CL"}
# Alias newer/lazer acronyms onto their pp-equivalent classic mod.
ALIAS = {"NC": "DT", "DC": "HT", "TC": "", "BL": "", "SG": ""}
PP_RELEVANT = {"EZ", "NF", "HT", "HR", "DT", "HD", "FL", "SO", "TD"}


def normalize_mods(raw) -> str:
    """Accepts old-style ['HD','DT'] or lazer-style [{'acronym':'HD'}].

    Returns a canonical sorted string like 'HDDT', or 'NM' for no mods.
    """
    if not raw:
        return "NM"
    acronyms = []
    for m in raw:
        a = m.get("acronym") if isinstance(m, dict) else str(m)
        if not a:
            continue
        a = ALIAS.get(a.upper(), a.upper())
        if a and a not in COSMETIC and a in PP_RELEVANT:
            acronyms.append(a)
    return "".join(sorted(set(acronyms))) or "NM"


MOD_BITS = {"NF": 1, "EZ": 2, "TD": 4, "HD": 8, "HR": 16, "DT": 64,
            "HT": 256, "FL": 1024, "SO": 4096}


def mods_to_bits(mods: str) -> int:
    """Canonical mod string -> legacy bitflag, for rosu-pp."""
    if mods == "NM":
        return 0
    bits = 0
    for i in range(0, len(mods), 2):
        bits |= MOD_BITS.get(mods[i:i + 2], 0)
    return bits