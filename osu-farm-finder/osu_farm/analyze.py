"""Farm-map analysis and Farm Score v2.

Farm Score v2 is intentionally popularity-neutral.  The ranking is driven by
PP efficiency, while popularity/overrepresentation and profile position are
kept as diagnostics/validation signals only.

Two independent efficiency axes are combined:

1. Formula efficiency
   How much more pp a score pays than expected from star rating, the actual
   achieved accuracy/misses/combo, and mods.

2. Realized efficiency
   How much pp players of comparable overall skill actually extract from a map
   of similar star rating, mod bucket and drain time.

Farm Score v2.1 also penalizes a specific failure mode found during player
testing: a map that looks extremely efficient to the formula model while the
realized player-skill model is negative.  Agreement is not rewarded with
popularity; disagreement simply reduces confidence in the formula-only claim.

Farm Score v2.2 adds SR-local calibration.  After the efficiency score and
disagreement penalty are computed, a robust smooth estimate of the typical
score around each star rating is subtracted.  This removes systematic SR drift
without turning sparse star-rating bands into automatic winners.  The local
baseline is estimated from per-bin medians and shrunk toward zero when local
support is weak.

Low sample size affects empirical-Bayes shrinkage and a separate confidence
field; raw popularity is never a positive farm-score term.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .core import REWORK_DATE, connect

# Match the balanced v2 collection bands.  A 0-1k bucket remains for backwards
# compatibility with older databases, even though the v2 collector starts at
# 1,000 pp by default.
BRACKETS = [
    (0, 1000),
    (1000, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 5000),
    (5000, 6000),
    (6000, 7000),
    (7000, 8000),
    (8000, 9000),
    (9000, 10000),
    (10000, 12000),
    (12000, 99999),
]


# ------------------------------------------------------- activity weighting

def player_weights(conn, half_life_days=90.0, ref_date=None) -> pd.DataFrame:
    """Weight players by how recently they have added scores to their top 100."""
    df = pd.read_sql_query(
        "SELECT user_id, created_at, pp FROM scores WHERE created_at IS NOT NULL",
        conn,
    )
    if df.empty:
        return pd.DataFrame(columns=["user_id", "activity_weight"])

    df["created_at"] = pd.to_datetime(df.created_at, errors="coerce", utc=True)
    ref = pd.Timestamp(ref_date, tz="UTC") if ref_date else df.created_at.max()
    age_days = (ref - df.created_at).dt.total_seconds() / 86400.0
    df["recency"] = 0.5 ** (age_days / half_life_days)

    g = df.groupby("user_id").agg(
        recent_mass=("recency", "sum"),
        n_scores=("recency", "size"),
    ).reset_index()
    g["activity_weight"] = 1.0 - np.exp(-g.recent_mass / 10.0)
    return g[["user_id", "activity_weight", "recent_mass", "n_scores"]]


def score_table(conn, post_rework_only=False, weighted=True) -> pd.DataFrame:
    """Scores joined to player skill, profile position and beatmap metadata."""
    q = """
        SELECT s.user_id, s.beatmap_id, s.position, s.pp, s.mods, s.created_at,
               p.pp AS player_pp, p.global_rank,
               b.sr, b.cs, b.ar, b.od,
               b.passcount, b.playcount, b.hit_length, b.status, b.ranked_date,
               b.title, b.artist, b.version
        FROM scores s
        JOIN players  p USING(user_id)
        JOIN beatmaps b USING(beatmap_id)
        WHERE b.passcount > 0 AND b.sr > 0 AND p.pp > 0 AND s.pp > 0
    """
    if post_rework_only:
        q += f" AND s.created_at >= '{REWORK_DATE}'"
    df = pd.read_sql_query(q, conn)
    if df.empty:
        return df

    edges = [b[0] for b in BRACKETS] + [BRACKETS[-1][1]]
    labels = [f"{a}-{b}" for a, b in BRACKETS]
    df["bracket"] = pd.cut(
        df.player_pp,
        bins=edges,
        labels=labels,
        right=False,
    )

    if weighted:
        w = player_weights(conn)
        df = df.merge(w[["user_id", "activity_weight"]], on="user_id", how="left")
        df["activity_weight"] = df.activity_weight.fillna(0.5)
    else:
        df["activity_weight"] = 1.0
    return df


# --------------------------------------------- popularity-controlled adoption

def overrepresentation(df: pd.DataFrame, min_passcount=500, shrink=3.0) -> pd.DataFrame:
    """Popularity-controlled top-100 adoption, retained as a diagnostic only.

    ``overrep`` is NOT part of Farm Score v2.  It answers whether a map appears
    in more sampled top-100 lists than its public passcount/exposure predicts.
    """
    out = []
    for bracket, sub in df.groupby("bracket", observed=True):
        if len(sub) < 200:
            continue
        n_players = sub.user_id.nunique()

        agg = sub.groupby("beatmap_id").agg(
            appearances=("activity_weight", "sum"),
            overrep_bracket_count=("user_id", "size"),
            sr=("sr", "first"),
            passcount=("passcount", "first"),
            playcount=("playcount", "first"),
            hit_length=("hit_length", "first"),
            dominant_mods=(
                "mods",
                lambda s: s.mode().iat[0] if len(s.mode()) else "NM",
            ),
        ).reset_index()

        agg = agg[agg.passcount >= min_passcount].copy()
        if len(agg) < 50:
            continue

        agg["log_sr"] = np.log(agg.sr.clip(lower=0.1))
        agg["log_len"] = np.log1p(agg.hit_length.fillna(0))
        X = pd.DataFrame({
            "log_sr": agg.log_sr,
            "log_sr2": agg.log_sr ** 2,
            "log_sr3": agg.log_sr ** 3,
            "log_len": agg.log_len,
            "dt": agg.dominant_mods.str.contains("DT", na=False).astype(float),
            "hd": agg.dominant_mods.str.contains("HD", na=False).astype(float),
            "hr": agg.dominant_mods.str.contains("HR", na=False).astype(float),
        })
        X = sm.add_constant(X)
        offset = np.log(agg.passcount.values) + np.log(max(n_players, 1))
        y = agg.appearances.values

        try:
            pois = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
            mu = pois.mu
            aux_y = ((y - mu) ** 2 - y) / np.maximum(mu, 1e-9)
            alpha = max(float(sm.OLS(aux_y, mu).fit().params[0]), 1e-3)
            fit = sm.GLM(
                y,
                X,
                family=sm.families.NegativeBinomial(alpha=alpha),
                offset=offset,
            ).fit()
            expected = fit.mu
        except Exception:
            expected = np.maximum(y.mean(), 1e-6) * np.ones_like(y, dtype=float)

        agg["expected_appearances"] = expected
        agg["overrep"] = np.log((y + shrink) / (expected + shrink))
        agg["bracket"] = str(bracket)
        out.append(agg)

    if not out:
        return pd.DataFrame(columns=["beatmap_id", "overrep", "bracket"])
    return pd.concat(out, ignore_index=True).sort_values("overrep", ascending=False)


# ----------------------------------------------------- profile-impact diagnostics

def profile_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize where a map lands inside sampled players' current top 100.

    This is validation, not a ranking term.  It is useful for answering whether
    an efficiency signal is showing up as meaningful profile plays without
    giving established/popular maps an automatic advantage.
    """
    d = df.dropna(subset=["position"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["beatmap_id", "profile_impact"])

    d["position"] = d.position.clip(lower=1, upper=100)
    # Smoothly decreases from 1.0 at #1 to 0.1 at #100.
    d["position_weight"] = 1.0 / np.sqrt(d.position.astype(float))
    d["top5"] = (d.position <= 5).astype(int)
    d["top10"] = (d.position <= 10).astype(int)
    d["top20"] = (d.position <= 20).astype(int)
    d["top30"] = (d.position <= 30).astype(int)
    d["top50"] = (d.position <= 50).astype(int)

    return d.groupby("beatmap_id").agg(
        profile_impact=("position_weight", "mean"),
        median_position=("position", "median"),
        top5_count=("top5", "sum"),
        top10_count=("top10", "sum"),
        top20_count=("top20", "sum"),
        top30_count=("top30", "sum"),
        top50_count=("top50", "sum"),
    ).reset_index()


def _map_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """One metadata/summary row per beatmap from the joined score table."""
    return df.groupby("beatmap_id").agg(
        artist=("artist", "first"),
        title=("title", "first"),
        version=("version", "first"),
        dominant_mods=(
            "mods", lambda s: s.mode().iat[0] if len(s.mode()) else "NM"
        ),
        sr=("sr", "first"),
        hit_length=("hit_length", "first"),
        cs=("cs", "first"),
        od=("od", "first"),
        passcount=("passcount", "first"),
        playcount=("playcount", "first"),
        mean_pp=("pp", "mean"),
        p90_pp=("pp", lambda x: float(np.percentile(x, 90))),
        max_pp=("pp", "max"),
        raw_count=("user_id", "size"),
    ).reset_index()


def _robust_z(s: pd.Series, clip=6.0) -> pd.Series:
    """Median/MAD standardization so a few extreme maps do not set the scale."""
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    med = float(x.median())
    mad = float((x - med).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(x.std())
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return ((x - med) / scale).clip(-clip, clip)


def _robust_scale(s: pd.Series) -> float:
    """Return a robust spread estimate while preserving zero as a real origin."""
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    med = float(x.median())
    mad = float((x - med).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(x.std())
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return scale



def _sr_local_calibration(
    sr: pd.Series,
    score: pd.Series,
    *,
    bin_width: float = 0.25,
    bandwidth: float = 1.25,
    prior_maps: float = 300.0,
) -> pd.DataFrame:
    """Estimate a robust SR-local score baseline and how strongly to apply it.

    The calibration deliberately subtracts only a location shift; it does not
    rescale the score distribution.  That preserves the interpretation of a
    one-point Farm Score difference while removing broad SR-dependent drift.

    Steps:
      1. take the median pre-calibration Farm Score in 0.25-star bins;
      2. Gaussian-smooth those medians across nearby SR bins;
      3. estimate local support with the same kernel;
      4. shrink the baseline toward zero in sparse tails.

    ``prior_maps`` controls that final shrinkage.  With 300 effective local
    maps, half of the estimated SR baseline is applied; dense regions approach
    full calibration.  The deliberately broad 1.25-star default bandwidth
    corrects slow SR drift without materially reordering maps inside a lobby-
    sized SR window.
    """
    if bin_width <= 0:
        raise ValueError("SR calibration bin_width must be > 0")
    if bandwidth <= 0:
        raise ValueError("SR calibration bandwidth must be > 0")
    if prior_maps < 0:
        raise ValueError("SR calibration prior_maps must be >= 0")

    x = pd.to_numeric(sr, errors="coerce").astype(float)
    y = pd.to_numeric(score, errors="coerce").astype(float)
    valid = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    result = pd.DataFrame(index=sr.index)
    result["sr_local_baseline_raw"] = 0.0
    result["sr_calibration_support"] = 0.0
    result["sr_calibration_strength"] = 0.0
    result["sr_calibration_adjustment"] = 0.0
    if not valid.any():
        return result

    xv = x[valid].to_numpy()
    yv = y[valid].to_numpy()
    keys = np.floor(xv / bin_width).astype(int)
    tmp = pd.DataFrame({"key": keys, "score": yv})
    bins = tmp.groupby("key").score.agg(["median", "count"]).reset_index()
    centers = (bins["key"].to_numpy(dtype=float) + 0.5) * bin_width
    medians = bins["median"].to_numpy(dtype=float)
    counts = bins["count"].to_numpy(dtype=float)

    # Work on the compact set of SR bins rather than all map pairs.  A square-
    # root reliability weight trusts dense bins more without allowing the very
    # dense 4-6 star region to completely dominate neighboring bins.
    dist = centers[:, None] - centers[None, :]
    kernel = np.exp(-0.5 * (dist / bandwidth) ** 2)
    reliability = np.sqrt(np.maximum(counts, 1.0))[None, :]
    weights = kernel * reliability
    denom = weights.sum(axis=1)
    smooth = np.divide(
        (weights * medians[None, :]).sum(axis=1),
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )

    # Effective nearby map count.  This uses full map counts, not sqrt counts,
    # because it represents evidence volume rather than smoothing influence.
    support = (kernel * counts[None, :]).sum(axis=1)

    # Interpolate the smoothed bin values to each map's exact SR.  np.interp
    # holds the edge value outside the observed center range; sparse-tail
    # shrinkage below prevents those extrapolated edge estimates from being
    # applied aggressively.
    baseline_raw = np.interp(xv, centers, smooth)
    support_map = np.interp(xv, centers, support)
    if prior_maps == 0:
        strength = np.ones_like(support_map)
    else:
        strength = support_map / (support_map + prior_maps)
    adjustment = baseline_raw * strength

    idx = result.index[valid]
    result.loc[idx, "sr_local_baseline_raw"] = baseline_raw
    result.loc[idx, "sr_calibration_support"] = support_map
    result.loc[idx, "sr_calibration_strength"] = strength
    result.loc[idx, "sr_calibration_adjustment"] = adjustment
    return result

# ------------------------------------------------------------- final scoring

def farm_report(
    db_path: str,
    top_n=200,
    post_rework_only=False,
    min_observations=5,
    sr_min=None,
    sr_max=None,
    max_length=None,
    min_passcount=None,
    mods=None,
    exclude_dt=False,
    formula_weight=0.65,
    disagreement_weight=0.30,
    sr_calibration=True,
    sr_calibration_bandwidth=1.25,
    sr_calibration_prior_maps=300.0,
) -> pd.DataFrame:
    """Build Farm Score v2.2.

    Ranking formula:

        formula_weight * formula efficiency
      + (1 - formula_weight) * realized player-skill/time efficiency
      - disagreement_weight * formula-positive/realized-negative penalty
      - SR-local baseline adjustment

    The disagreement term only activates when the formula model says a map
    overpays while realized efficiency is actually below zero.  It was added
    after player testing found that this pattern can describe legitimately
    difficult/high-strain maps rather than easy PP.

    Neither overrepresentation, HITS/farmer clustering, map age, nor raw score
    count contributes positively to the score.  Observation count only affects
    empirical-Bayes shrinkage and the separate confidence field.  SR calibration
    subtracts a local location bias only; it never adds a novelty/popularity term.
    """
    from .features import (
        fit_pp_surrogate,
        fit_realized_pp_surrogate,
        map_achievability,
        map_pp_efficiency,
        map_realized_efficiency,
    )

    conn = connect(db_path)
    df = score_table(conn, post_rework_only=post_rework_only)
    if df.empty:
        conn.close()
        raise SystemExit("No joined data. Run score + beatmap collection first.")

    print("[1/5] popularity/adoption diagnostic")
    over = overrepresentation(df)

    print("[2/5] formula pp-efficiency model")
    model, sdf = fit_pp_surrogate(
        db_path, post_rework_only=post_rework_only)
    eff = map_pp_efficiency(model, sdf, min_scores=min_observations)

    print("[3/5] realized player-skill/time efficiency model")
    realized_model, rdf = fit_realized_pp_surrogate(
        db_path, post_rework_only=post_rework_only)
    realized = map_realized_efficiency(
        realized_model, rdf, min_scores=min_observations)

    print("[4/5] diagnostics")
    ach = map_achievability(sdf, min_scores=min_observations)
    prof = profile_impact(df)
    meta = _map_metadata(df)
    print(f"  forgiveness computed for {len(ach):,} maps")

    print("[5/5] combining Farm Score v2.2 + SR calibration")
    # Efficiency is the authoritative base.  A map no longer needs a strong
    # overrepresentation row merely to enter the candidate set.
    out = eff.copy()
    out = out[out.n_scores >= min_observations].copy()
    out = out.merge(realized, on="beatmap_id", how="left")
    out = out.merge(meta, on="beatmap_id", how="left", suffixes=("", "_meta"))
    out = out.merge(prof, on="beatmap_id", how="left")

    if not over.empty:
        over_best = (
            over.sort_values("overrep", ascending=False)
                .drop_duplicates("beatmap_id")
        )
        over_cols = [
            "beatmap_id", "overrep", "bracket", "appearances",
            "expected_appearances", "overrep_bracket_count",
        ]
        out = out.merge(
            over_best[[c for c in over_cols if c in over_best.columns]],
            on="beatmap_id",
            how="left",
        )
    else:
        out["overrep"] = 0.0
        out["bracket"] = ""

    ach_cols = [
        "beatmap_id", "forgiveness", "median_acc", "pct_miss_free",
        "median_combo_ratio", "acc_slope",
    ]
    out = out.merge(
        ach[[c for c in ach_cols if c in ach.columns]],
        on="beatmap_id",
        how="left",
    )

    for col in [
        "realized_efficiency_shrunk", "overrep", "forgiveness",
        "profile_impact",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if not 0.0 <= formula_weight <= 1.0:
        raise ValueError("formula_weight must be between 0 and 1")
    if disagreement_weight < 0.0:
        raise ValueError("disagreement_weight must be >= 0")

    # Main efficiency terms.  Popularity still does not contribute to rank.
    out["z_formula_efficiency"] = _robust_z(out.pp_efficiency_shrunk)
    out["z_realized_efficiency"] = _robust_z(out.realized_efficiency_shrunk)
    realized_weight = 1.0 - formula_weight
    out["base_farm_score"] = (
        formula_weight * out.z_formula_efficiency
        + realized_weight * out.z_realized_efficiency
    )

    # Player-test correction: formula-only false positives.
    #
    # Zero has semantic meaning here, so measure strength relative to a robust
    # scale but do NOT median-center before checking the sign.  The penalty is
    # therefore exactly zero whenever realized efficiency is non-negative.
    formula_scale = _robust_scale(out.pp_efficiency_shrunk)
    realized_scale = _robust_scale(out.realized_efficiency_shrunk)
    out["positive_formula_strength"] = (
        out.pp_efficiency_shrunk / formula_scale
    ).clip(lower=0.0, upper=6.0)
    out["negative_realized_strength"] = (
        -out.realized_efficiency_shrunk / realized_scale
    ).clip(lower=0.0, upper=6.0)
    out["efficiency_disagreement"] = (
        out.positive_formula_strength * out.negative_realized_strength
    )
    out["disagreement_penalty"] = (
        disagreement_weight * out.efficiency_disagreement
    )
    out["precalibration_farm_score"] = (
        out.base_farm_score - out.disagreement_penalty
    )

    if sr_calibration:
        cal = _sr_local_calibration(
            out.sr,
            out.precalibration_farm_score,
            bandwidth=sr_calibration_bandwidth,
            prior_maps=sr_calibration_prior_maps,
        )
        out = out.join(cal)
        out["farm_score"] = (
            out.precalibration_farm_score - out.sr_calibration_adjustment
        )
        print(
            "  SR calibration: "
            f"bandwidth={sr_calibration_bandwidth:.2f}*, "
            f"prior={sr_calibration_prior_maps:g} maps | "
            f"median adjustment={out.sr_calibration_adjustment.median():+.3f} | "
            f"range {out.sr_calibration_adjustment.min():+.3f} "
            f"to {out.sr_calibration_adjustment.max():+.3f}"
        )
    else:
        out["sr_local_baseline_raw"] = 0.0
        out["sr_calibration_support"] = 0.0
        out["sr_calibration_strength"] = 0.0
        out["sr_calibration_adjustment"] = 0.0
        out["farm_score"] = out.precalibration_farm_score

    out["efficiency_agreement"] = np.select(
        [
            (out.pp_efficiency_shrunk > 0) & (out.realized_efficiency_shrunk > 0),
            (out.pp_efficiency_shrunk > 0) & (out.realized_efficiency_shrunk <= 0),
            (out.pp_efficiency_shrunk <= 0) & (out.realized_efficiency_shrunk > 0),
        ],
        ["corroborated", "formula_only", "realized_only"],
        default="neither",
    )

    # Confidence is intentionally separate from farm_score.  The empirical-
    # Bayes estimates above already shrink low-N maps; confidence tells us how
    # much evidence supports the estimate without turning popularity into a
    # ranking bonus.
    out["confidence"] = out.n_scores / (out.n_scores + 20.0)
    out["confidence_label"] = np.select(
        [out.confidence >= 0.80, out.confidence >= 0.50],
        ["high_conf", "supported"],
        default="explore",
    )
    # Keep the legacy column because the lobby schema currently reads it, but
    # leave it blank: stale/known/undiscovered labels no longer affect v2 and
    # do not need to be announced in the lobby.
    out["quadrant"] = ""

    out["formula_efficiency_se"] = (
        out.efficiency_sd.fillna(0.0) / np.sqrt(out.n_scores.clip(lower=1))
    )
    if "realized_efficiency_sd" in out.columns:
        out["realized_efficiency_se"] = (
            out.realized_efficiency_sd.fillna(0.0)
            / np.sqrt(out.realized_n_scores.clip(lower=1))
        )

    # Apply requested presentation/pool filters BEFORE taking top_n.  This fixes
    # the old CLI behavior where DT maps consumed the global top-N and were only
    # filtered out afterward.
    if sr_min is not None:
        out = out[out.sr >= sr_min]
    if sr_max is not None:
        out = out[out.sr <= sr_max]
    if max_length is not None:
        out = out[out.hit_length <= max_length]
    if min_passcount is not None:
        out = out[out.passcount >= min_passcount]
    if mods:
        out = out[out.dominant_mods == mods]
    if exclude_dt:
        out = out[~out.dominant_mods.fillna("").str.contains("DT", na=False)]

    out["url"] = "https://osu.ppy.sh/b/" + out.beatmap_id.astype(str)
    conn.close()

    out = out.sort_values("farm_score", ascending=False).reset_index(drop=True)
    if top_n is not None:
        out = out.head(top_n).reset_index(drop=True)
    return out


def save_report(db_path: str, df: pd.DataFrame) -> int:
    """Persist Farm Score v2.2 in the lobby-compatible ``farm_report`` table."""
    keep = [
        "beatmap_id", "artist", "title", "version", "dominant_mods", "sr",
        "bracket", "farm_score", "precalibration_farm_score",
        "base_farm_score", "sr_local_baseline_raw",
        "sr_calibration_support", "sr_calibration_strength",
        "sr_calibration_adjustment", "quadrant", "confidence",
        "confidence_label", "efficiency_agreement", "efficiency_disagreement",
        "disagreement_penalty", "pp_efficiency", "pp_efficiency_shrunk",
        "realized_efficiency", "realized_efficiency_shrunk", "overrep",
        "forgiveness", "profile_impact", "median_position", "top5_count",
        "top10_count", "top20_count", "top30_count", "top50_count",
        "median_acc", "pct_miss_free", "median_combo_ratio", "hit_length",
        "cs", "od", "mean_pp", "p90_pp", "max_pp", "raw_count",
        "n_scores", "passcount", "playcount",
    ]
    cols = [c for c in keep if c in df.columns]
    best = (
        df[cols].sort_values("farm_score", ascending=False)
        .drop_duplicates("beatmap_id")
    )

    conn = connect(db_path)
    conn.execute("DROP TABLE IF EXISTS farm_report")
    best.to_sql("farm_report", conn, index=False)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_farm_report_score "
        "ON farm_report(farm_score DESC)"
    )
    conn.commit()
    conn.close()
    return len(best)