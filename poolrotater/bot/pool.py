"""Map pool, sourced from the farm-finder database.

The pool consumes the analyzed ``farm_report`` table produced by
osu-farm-finder. Lobby-specific filters (SR, length, passcount, DT exclusion)
are applied here, after farm scoring.

Auto difficulty uses a continuous pp -> star-rating target rather than fixed
skill tiers. The default model is intentionally simple and easy to tune:

    2000pp -> 4.0*
    3000pp -> 4.4*
    4000pp -> 4.8*
    5000pp -> 5.2*
    6000pp -> 5.6*
    7000pp -> 6.0*

That is 0.4* per 1000pp, with the final target clamped so the default
+/-0.40* window stays inside 3.0-6.8*.
"""
from __future__ import annotations

import logging
import random
import sqlite3
from dataclasses import dataclass

log = logging.getLogger("pool")


# ----------------------------------------------------- auto difficulty model

AUTO_SR_FLOOR = 3.0
AUTO_SR_CEILING = 6.8
AUTO_SR_HALF_WIDTH = 0.40
AUTO_SR_STEP = 0.05


def target_sr_for_pp(pp: float, half_width: float = AUTO_SR_HALF_WIDTH,
                     step: float = AUTO_SR_STEP) -> float:
    """Return the continuous target star rating for a player's pp.

    The raw relationship is:
        target = 4.0 + (pp - 2000) * 0.0004

    Targets are rounded to a small 0.05* step so tiny pp changes do not cause
    constant pool reloads, then clamped so the requested range remains inside
    the supported 3.0-6.8* band.
    """
    pp = max(float(pp or 0.0), 0.0)
    half_width = max(float(half_width), 0.0)
    step = max(float(step), 0.0)

    raw = 4.0 + (pp - 2000.0) * 0.0004

    if step > 0:
        raw = round(raw / step) * step

    lo_target = AUTO_SR_FLOOR + half_width
    hi_target = AUTO_SR_CEILING - half_width
    if hi_target < lo_target:
        # Defensive fallback for a nonsensically huge configured half-width.
        return (AUTO_SR_FLOOR + AUTO_SR_CEILING) / 2.0

    return min(max(raw, lo_target), hi_target)


def auto_range_for_pp(pp: float, half_width: float = AUTO_SR_HALF_WIDTH,
                      step: float = AUTO_SR_STEP) -> tuple[float, float]:
    """Return the automatic SR range centered on the pp-derived target."""
    target = target_sr_for_pp(pp, half_width=half_width, step=step)
    lo = max(AUTO_SR_FLOOR, target - half_width)
    hi = min(AUTO_SR_CEILING, target + half_width)
    return round(lo, 2), round(hi, 2)


# Backwards-compatible name used by older bot.py versions.
def tier_for_pp(pp: float, half_width: float = AUTO_SR_HALF_WIDTH) -> tuple[float, float]:
    return auto_range_for_pp(pp, half_width=half_width)


# --------------------------------------------------------------- map records

@dataclass
class PoolMap:
    beatmap_id: int
    artist: str
    title: str
    version: str
    sr: float
    hit_length: int
    farm_score: float
    passcount: int
    quadrant: str = ""
    forgiveness: float = 0.0
    max_pp: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.artist} - {self.title} [{self.version}] ({self.sr:.2f}*)"

    @property
    def url(self) -> str:
        return f"https://osu.ppy.sh/b/{self.beatmap_id}"


class MapPool:
    def __init__(self, db_path: str, sr_min=4.0, sr_max=5.5,
                 max_length=150, min_passcount=7_500, limit=250):
        self.db_path = db_path
        self.sr_min, self.sr_max = sr_min, sr_max
        self.max_length = max_length
        self.min_passcount = min_passcount
        self.limit = limit
        self.maps: list[PoolMap] = []
        self._recent: list[int] = []
        self.reload()

    def _has_table(self, conn, name: str) -> bool:
        return bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone())

    def _has_report(self, conn) -> bool:
        return self._has_table(conn, "farm_report")

    def reload(self) -> int:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            if self._has_report(conn):
                # If farm_report exists, it is authoritative. An empty result
                # means this particular SR/filter combination has no eligible
                # farm-ranked maps; do NOT silently switch to raw popularity.
                rows = conn.execute("""
                    SELECT f.beatmap_id, f.artist, f.title, f.version, f.sr,
                           f.hit_length, f.passcount, f.farm_score, f.quadrant,
                           f.forgiveness, f.max_pp
                    FROM farm_report f
                    WHERE f.sr BETWEEN ? AND ?
                      AND f.hit_length <= ?
                      AND f.passcount >= ?
                      AND COALESCE(f.dominant_mods, '') NOT LIKE '%DT%'
                    ORDER BY f.farm_score DESC
                    LIMIT ?
                """, (self.sr_min, self.sr_max, self.max_length,
                      self.min_passcount, self.limit)).fetchall()
                source = "farm_report (DT excluded)"

            elif self._has_table(conn, "beatmaps"):
                # Compatibility fallback only for databases that genuinely do
                # not have farm_report yet.
                rows = conn.execute("""
                    SELECT b.beatmap_id, b.artist, b.title, b.version, b.sr,
                           b.hit_length, b.passcount,
                           0.0 AS farm_score, '' AS quadrant,
                           0.0 AS forgiveness, 0.0 AS max_pp
                    FROM beatmaps b
                    WHERE b.sr BETWEEN ? AND ?
                      AND b.hit_length <= ?
                      AND b.passcount >= ?
                      AND b.status = 'ranked'
                    ORDER BY b.passcount DESC
                    LIMIT ?
                """, (self.sr_min, self.sr_max, self.max_length,
                      self.min_passcount, self.limit)).fetchall()
                source = "beatmaps"

            else:
                # A report-only DB is valid for the lobby. If neither supported
                # table exists, return an empty pool instead of throwing a raw
                # SQLite exception from the middle of a live room.
                rows = []
                source = "database (no farm_report/beatmaps table)"
                log.error(
                    "farm database %s has neither farm_report nor beatmaps",
                    self.db_path,
                )
        finally:
            conn.close()

        self.maps = [
            PoolMap(
                beatmap_id=r["beatmap_id"],
                artist=r["artist"] or "?",
                title=r["title"] or "?",
                version=r["version"] or "?",
                sr=r["sr"] or 0.0,
                hit_length=r["hit_length"] or 0,
                farm_score=r["farm_score"] or 0.0,
                passcount=r["passcount"] or 0,
                quadrant=r["quadrant"] or "",
                forgiveness=r["forgiveness"] or 0.0,
                max_pp=r["max_pp"] or 0.0,
            )
            for r in rows
        ]

        log.info(
            "pool loaded from %s: %d maps, %.2f-%.2f*, <=%ds, >=%d passes",
            source, len(self.maps), self.sr_min, self.sr_max,
            self.max_length, self.min_passcount,
        )
        if self.maps and self.maps[0].farm_score:
            log.info(
                "  top farm scores: %s",
                ", ".join(f"{m.farm_score:.2f}" for m in self.maps[:5]),
            )
        return len(self.maps)

    def set_range(self, sr_min: float, sr_max: float) -> int:
        """Try a new SR range without destroying a known-good live pool.

        Auto difficulty can legitimately ask for a range that contains no
        farm_report maps. A live lobby must never crash or empty itself because
        of that. If the candidate range cannot load at least one map, restore
        the previous range and map list and return 0.
        """
        sr_min, sr_max = float(sr_min), float(sr_max)
        if sr_min >= sr_max:
            log.warning("rejected invalid SR range %.2f-%.2f", sr_min, sr_max)
            return 0

        old_min, old_max = self.sr_min, self.sr_max
        old_maps = self.maps

        self.sr_min, self.sr_max = sr_min, sr_max

        try:
            n = self.reload()
        except sqlite3.Error as e:
            self.sr_min, self.sr_max = old_min, old_max
            self.maps = old_maps
            log.error(
                "could not load candidate range %.2f-%.2f*: %s; "
                "keeping %.2f-%.2f*",
                sr_min, sr_max, e, old_min, old_max,
            )
            return 0

        if n <= 0:
            self.sr_min, self.sr_max = old_min, old_max
            self.maps = old_maps
            log.warning(
                "no eligible maps in candidate range %.2f-%.2f*; "
                "keeping previous %.2f-%.2f* pool (%d maps)",
                sr_min, sr_max, old_min, old_max, len(old_maps),
            )
            return 0

        return n

    def next_map(self) -> PoolMap | None:
        """Pick a map while spacing out recently played beatmaps."""
        if not self.maps:
            return None

        avoid = set(self._recent)
        candidates = [m for m in self.maps if m.beatmap_id not in avoid]
        if not candidates:
            self._recent.clear()
            candidates = self.maps

        pick = random.choice(candidates)
        self._recent.append(pick.beatmap_id)

        if len(self._recent) > max(len(self.maps) // 2, 5):
            self._recent.pop(0)

        return pick
