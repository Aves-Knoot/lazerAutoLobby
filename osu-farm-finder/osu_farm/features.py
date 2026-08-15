"""Map features and the pp-efficiency signal.

Two independent sources of "how much pp does this map pay for its difficulty":

1. strain_features()  -- shape of the difficulty curve, from the .osu file.
   Strain SHAPE (spikiness, where the hard part sits) is mostly a property of
   the map, not of the pp algorithm, so these survive reworks well.

2. fit_pp_surrogate() -- learns the CURRENT pp function directly from live
   post-rework scores. This is the important one. No released calculator
   implements the July 2026 rework yet (rosu-pp-py 4.0.2 returns None for
   `reading` on osu!standard), so simulating pp would use a stale algorithm.
   Fitting against scores the API already reprocessed sidesteps that entirely.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .core import connect, mods_to_bits

OSU_FILE_URL = "https://osu.ppy.sh/osu/{}"


# ------------------------------------------------------------------ 1. strains

def _gini(x: np.ndarray) -> float:
    """Concentration of the strain series. High = difficulty is spiky."""
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def strain_features(osu_path: str, mods: str = "NM") -> dict | None:
    """Strain-shape features for one (map, mods) pair.

    hardest_section_pos is the underrated one: a filter at 0.1 means cheap
    retries, at 0.9 it means every failed attempt costs the whole song.
    """
    import rosu_pp_py as rosu

    try:
        bm = rosu.Beatmap(path=osu_path)
    except Exception:
        return None

    diff = rosu.Difficulty(mods=mods_to_bits(mods))
    try:
        attrs = diff.calculate(bm)
        strains = diff.strains(bm)
    except Exception:
        return None

    aim = np.asarray(strains.aim or [], dtype=float)
    speed = np.asarray(strains.speed or [], dtype=float)
    if aim.size == 0:
        return None

    combined = aim + speed if speed.size == aim.size else aim
    peak_idx = int(np.argmax(combined))
    n = combined.size
    third = max(n // 3, 1)
    late_mass = float(combined[-third:].sum() / combined.sum()) if combined.sum() else 0.0

    n_obj = (bm.n_circles or 0) + (bm.n_sliders or 0)
    slider_ratio = (bm.n_sliders or 0) / n_obj if n_obj else 0.0

    return {
        "stars": attrs.stars,
        "aim": attrs.aim,
        "speed": attrs.speed,
        "flashlight": attrs.flashlight,
        "speed_note_count": attrs.speed_note_count,
        "slider_factor": attrs.slider_factor,
        "aim_strain_gini": _gini(aim),
        "speed_strain_gini": _gini(speed) if speed.size else 0.0,
        "aim_spike_ratio": float(aim.max() / aim.mean()) if aim.mean() else 0.0,
        "hardest_section_pos": peak_idx / max(n - 1, 1),
        "late_difficulty_mass": late_mass,
        "slider_ratio": slider_ratio,
    }


def pp_surface(osu_path: str, mods: str = "NM") -> dict | None:
    """Accuracy gradient and forgiveness ratio.

    WARNING: unlike strain_features, these depend on the pp algorithm, so they
    reflect the PRE-rework formula until rosu-pp ships the 2026 changes. Treat
    as a rough prior; the surrogate model below is authoritative.
    """
    import rosu_pp_py as rosu

    try:
        bm = rosu.Beatmap(path=osu_path)
        attrs = rosu.Difficulty(mods=mods_to_bits(mods)).calculate(bm)
    except Exception:
        return None

    def pp_at(acc: float, misses: int = 0) -> float:
        perf = rosu.Performance(accuracy=acc, misses=misses, lazer=False)
        if misses:
            perf.set_combo(max(int(attrs.max_combo * 0.6), 1))
        return perf.calculate(attrs).pp

    pp99, pp97 = pp_at(99.0), pp_at(97.0)
    return {
        "pp_99fc": pp99,
        "acc_gradient": (pp99 - pp97) / 2.0,
        "forgiveness_ratio": pp_at(97.0, misses=1) / pp99 if pp99 else 0.0,
    }


async def download_osu_files(beatmap_ids, out_dir="maps", concurrency=4):
    """Fetch raw .osu files. Cached on disk; safe to re-run."""
    import httpx

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    todo = [b for b in beatmap_ids if not (out / f"{b}.osu").exists()]
    sem = __import__("asyncio").Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        async def one(bid):
            async with sem:
                try:
                    r = await client.get(OSU_FILE_URL.format(bid))
                    if r.status_code == 200 and r.text.strip():
                        (out / f"{bid}.osu").write_text(r.text, encoding="utf-8")
                except Exception:
                    pass
                await __import__("asyncio").sleep(0.25)

        await __import__("asyncio").gather(*(one(b) for b in todo))
    return len(todo)


def build_map_features(db_path: str, maps_dir="maps", limit=None) -> int:
    """Compute and store strain features for every downloaded map."""
    conn = connect(db_path)
    ids = [r["beatmap_id"] for r in conn.execute(
        "SELECT DISTINCT s.beatmap_id FROM scores s "
        "LEFT JOIN map_features f USING(beatmap_id) WHERE f.beatmap_id IS NULL")]
    if limit:
        ids = ids[:limit]

    rows, n = [], 0
    for bid in ids:
        path = Path(maps_dir) / f"{bid}.osu"
        if not path.exists():
            continue
        f = strain_features(str(path))
        if not f:
            continue
        s = pp_surface(str(path)) or {}
        rows.append((
            bid, f["aim"], f["speed"], f["flashlight"], f["stars"],
            f["speed_note_count"], f["slider_factor"],
            f["aim_strain_gini"], f["speed_strain_gini"], f["aim_spike_ratio"],
            f["hardest_section_pos"], f["late_difficulty_mass"],
            None,  # angle_entropy: see note in README
            f["slider_ratio"],
            s.get("acc_gradient"), s.get("forgiveness_ratio"), s.get("pp_99fc"),
            time.time(),
        ))
        n += 1
        if len(rows) >= 200:
            conn.executemany("INSERT OR REPLACE INTO map_features VALUES "
                             "(" + ",".join("?" * 18) + ")", rows)
            conn.commit()
            rows = []
    if rows:
        conn.executemany("INSERT OR REPLACE INTO map_features VALUES "
                         "(" + ",".join("?" * 18) + ")", rows)
        conn.commit()
    conn.close()
    return n


# ------------------------------------------------- 2. surrogate pp model

# DELIBERATELY UNDERSPECIFIED.
#
# The point of this model is its RESIDUAL, not its fit. Feed it ar/od/cs/
# length/object-count and it reconstructs the pp formula almost exactly
# (R^2 ~ 0.99), leaving a residual with no efficiency signal left in it --
# every map looks equally "fair" and the farm ranking collapses onto the
# overrepresentation axis alone.
#
# Restricting the inputs to star rating + how the player performed + mods
# makes the residual mean something useful: "this map pays more than its star
# rating justifies at that level of performance". Expect R^2 around 0.90-0.95.
# A near-perfect fit here is a FAILURE, not a success.
SURROGATE_FEATURES = [
    "sr", "accuracy", "misses", "combo_ratio",
    "m_HD", "m_HR", "m_DT", "m_FL", "m_EZ", "m_HT", "m_NF",
]

# Kept for reference / experimentation -- this is the version that overfits.
SURROGATE_FEATURES_FULL = [
    "sr", "accuracy", "misses", "combo_ratio", "log_length",
    "ar", "od", "cs", "log_objects",
    "m_HD", "m_HR", "m_DT", "m_FL", "m_EZ", "m_HT", "m_NF",
]


def _score_frame(conn, post_rework_only=True, rework_date="2026-07-26") -> pd.DataFrame:
    q = """
        SELECT s.beatmap_id, s.pp, s.accuracy, s.misses, s.max_combo, s.mods,
               s.created_at, b.sr, b.ar, b.od, b.cs, b.hit_length, b.max_combo AS map_combo,
               b.count_circles, b.count_sliders, b.passcount, b.playcount
        FROM scores s JOIN beatmaps b USING(beatmap_id)
        WHERE s.pp > 0 AND b.sr > 0 AND b.max_combo > 0
    """
    if post_rework_only:
        q += f" AND s.created_at >= '{rework_date}'"
    df = pd.read_sql_query(q, conn)
    if df.empty:
        return df

    df["accuracy"] = np.where(df.accuracy <= 1.0, df.accuracy * 100, df.accuracy)
    df["combo_ratio"] = (df.max_combo / df.map_combo).clip(0, 1)
    df["log_length"] = np.log1p(df.hit_length.fillna(0))
    df["log_objects"] = np.log1p(df.count_circles.fillna(0) + df.count_sliders.fillna(0))
    df["misses"] = df.misses.fillna(0)
    for m in ["HD", "HR", "DT", "FL", "EZ", "HT", "NF"]:
        df[f"m_{m}"] = df.mods.fillna("NM").str.contains(m).astype(int)
    df["mod_bucket"] = df.mods.fillna("NM")
    return df.dropna(subset=["sr", "accuracy", "combo_ratio"])


def fit_pp_surrogate(db_path: str, post_rework_only=True, min_scores=2000):
    """Learn pp = f(difficulty stats, performance, mods) from real scores.

    Because the API already reprocessed every score under the July 2026
    algorithm, this model IS the current pp function -- no calculator needed.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    conn = connect(db_path)
    df = _score_frame(conn, post_rework_only)
    conn.close()

    if len(df) < min_scores:
        if post_rework_only:
            warnings.warn(
                f"Only {len(df)} post-rework scores. Falling back to all scores; "
                "pp values are recalculated so this is still current-algorithm, "
                "but map SELECTION reflects the old meta."
            )
            return fit_pp_surrogate(db_path, post_rework_only=False, min_scores=min_scores)
        raise SystemExit(f"Not enough scores to fit ({len(df)}). Collect more.")

    X, y = df[SURROGATE_FEATURES].values, np.log(df.pp.values)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=7, random_state=0)
    model.fit(Xtr, ytr)
    r2 = model.score(Xte, yte)
    resid_sd = float(np.std(yte - model.predict(Xte)))
    print(f"  surrogate fit on {len(df):,} scores | test R2={r2:.4f} "
          f"| residual sd={resid_sd:.4f} (log pp)")
    # NOTE: do NOT diagnose overfitting from R^2. R^2 = 1 - resid_var /
    # total_var, and total_var here is dominated by the pp gap between
    # brackets (roughly 90pp to 800pp), so a perfectly healthy residual
    # still yields R^2 ~ 0.98 on real data. Residual sd is the meaningful
    # number, and the real check happens in map_pp_efficiency where the
    # map-level spread can actually be measured.
    if resid_sd < 0.03:
        warnings.warn(
            f"Residual sd is only {resid_sd:.4f} (log pp). The model is "
            "reconstructing the pp formula rather than leaving efficiency "
            "in the residual. Drop features from SURROGATE_FEATURES."
        )
    return model, df


def map_pp_efficiency(model, df: pd.DataFrame, min_scores=8,
                      features=None) -> pd.DataFrame:
    """Per-map pp efficiency: how much more pp a map pays than its stats predict.

    Positive = overpays for its difficulty, accuracy and mods. This is the
    mechanistic axis, and it is current-algorithm by construction.
    """
    feats = features or SURROGATE_FEATURES
    pred = model.predict(df[feats].values)
    df = df.assign(resid=np.log(df.pp.values) - pred)

    g = df.groupby("beatmap_id").agg(
        pp_efficiency=("resid", "mean"),
        efficiency_sd=("resid", "std"),
        n_scores=("resid", "size"),
        mean_pp=("pp", "mean"),
        mean_acc=("accuracy", "mean"),
        sr=("sr", "first"),
        passcount=("passcount", "first"),
        playcount=("playcount", "first"),
    ).reset_index()

    g = g[g.n_scores >= min_scores].copy()
    # Shrink toward zero for maps with few observations.
    prior_var = g.pp_efficiency.var()
    obs_var = (g.efficiency_sd.fillna(g.efficiency_sd.median()) ** 2) / g.n_scores
    g["pp_efficiency_shrunk"] = g.pp_efficiency * (prior_var / (prior_var + obs_var))

    # Center residuals WITHIN mod bucket.
    #
    # beatmaps.sr from the API is the NOMOD star rating, so the surrogate has
    # to infer a mod's difficulty boost from a binary flag. DT's effect is
    # strongly BPM-dependent, so whatever the flag can't capture lands in the
    # residual and makes every DT score look efficient. Centering per mod
    # removes that offset, leaving genuine map-level signal: "efficient
    # RELATIVE TO other maps played with the same mods".
    if "mod_bucket" in df.columns:
        mod_of = df.groupby("beatmap_id").mod_bucket.agg(
            lambda x: x.mode().iat[0] if len(x.mode()) else "NM")
        g["mod_bucket"] = g.beatmap_id.map(mod_of).fillna("NM")
        offsets = g.groupby("mod_bucket").pp_efficiency_shrunk.transform("mean")
        g["mod_offset"] = offsets
        g["pp_efficiency_shrunk"] = g.pp_efficiency_shrunk - offsets

    spread = float(g.pp_efficiency_shrunk.std())
    print(f"  pp_efficiency spread across maps: sd={spread:.4f} "
          f"(range {g.pp_efficiency_shrunk.min():+.3f} to "
          f"{g.pp_efficiency_shrunk.max():+.3f})")
    if spread < 0.02:
        warnings.warn(
            f"pp_efficiency spread is only {spread:.4f} -- the efficiency "
            "axis is degenerate and the ranking will collapse onto "
            "overrepresentation. Drop features from SURROGATE_FEATURES."
        )
    return g.sort_values("pp_efficiency_shrunk", ascending=False)


# --------------------------------------------------- 3. achievability

def map_achievability(df: pd.DataFrame, min_scores=8) -> pd.DataFrame:
    """How demanding is a map, conditional on it appearing in a top 100?

    pp_efficiency asks "does this map overpay for a given performance?".
    That is only half the question. A map that overpays but requires 99.3%
    and a full combo is not farmable; one that pays nearly as much at 96%
    with a couple of misses is. These features separate the two.

    Everything here is derived from scores already in the database -- the
    accuracy, miss count and combo actually achieved by players who got a
    top-100 play out of the map.

    NOTE ON SELECTION BIAS: these are top-100 scores, so accuracy is biased
    high on every map. That is fine for RELATIVE comparison -- a map whose
    top-100 entries average 96.5% is genuinely more forgiving than one
    averaging 99.2% -- but the absolute numbers are not population accuracy.
    """
    d = df.copy()
    d["miss_free"] = (d.misses == 0).astype(float)
    d["near_fc"] = (d.combo_ratio > 0.95).astype(float)

    g = d.groupby("beatmap_id").agg(
        median_acc=("accuracy", "median"),
        acc_p25=("accuracy", lambda x: float(np.percentile(x, 25))),
        acc_sd=("accuracy", "std"),
        pct_miss_free=("miss_free", "mean"),
        mean_misses=("misses", "mean"),
        median_combo_ratio=("combo_ratio", "median"),
        pct_near_fc=("near_fc", "mean"),
        n=("accuracy", "size"),
    ).reset_index()
    g = g[g.n >= min_scores].copy()
    if g.empty:
        return g

    # Empirical accuracy gradient: within each map, how steeply does pp rise
    # with accuracy? A steep slope means the map only pays at high acc --
    # punishing. A flat slope means it pays across a range -- forgiving.
    slopes = {}
    for bid, sub in d.groupby("beatmap_id"):
        if len(sub) < min_scores or sub.accuracy.std() < 0.15:
            continue
        try:
            slopes[bid] = float(np.polyfit(sub.accuracy, np.log(sub.pp), 1)[0])
        except Exception:
            continue
    g["acc_slope"] = g.beatmap_id.map(slopes)
    g["acc_slope"] = g.acc_slope.fillna(g.acc_slope.median())

    def z(s, invert=False):
        sd = s.std()
        out = (s - s.mean()) / (sd if sd else 1.0)
        return -out if invert else out

    # Forgiving = tolerates lower accuracy, tolerates misses, doesn't demand
    # a full combo, and pp doesn't fall off a cliff as accuracy drops.
    g["forgiveness"] = (
        0.35 * z(g.median_acc, invert=True)
        + 0.30 * z(g.pct_miss_free, invert=True)
        + 0.20 * z(g.median_combo_ratio, invert=True)
        + 0.15 * z(g.acc_slope, invert=True)
    )
    return g.sort_values("forgiveness", ascending=False)