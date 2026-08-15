"""Overrepresentation modelling and farm scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .core import REWORK_DATE, connect

DEFAULT_QUOTAS = {0: 4500, 2000: 4000, 3500: 3500, 5000: 1500, 7000: 500}

BRACKETS = [
    (0, 2000),
    (2000, 3500),
    (3500, 5000),
    (5000, 7000),
    (7000, 99999),
]


# ------------------------------------------------------- activity weighting

def player_weights(conn, half_life_days=90.0, ref_date=None) -> pd.DataFrame:
    """Weight each player by how recently they've been ADDING to their top 100.

    Score dates come free with the top-100 pull, so this needs no extra API
    calls, and it measures pp-motivated activity rather than raw playtime --
    which is closer to what we actually want to weight by.
    """
    df = pd.read_sql_query(
        "SELECT user_id, created_at, pp FROM scores WHERE created_at IS NOT NULL", conn)
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
    # Saturating weight: a player with ~10 recent top-100 entries is fully active.
    g["activity_weight"] = 1.0 - np.exp(-g.recent_mass / 10.0)
    return g[["user_id", "activity_weight", "recent_mass", "n_scores"]]


def score_table(conn, post_rework_only=False, weighted=True) -> pd.DataFrame:
    """Scores joined to player bracket, map stats, and activity weight."""
    q = """
        SELECT s.user_id, s.beatmap_id, s.pp, s.mods, s.created_at,
               p.pp AS player_pp, p.global_rank,
               b.sr, b.cs, b.ar, b.od,
               b.passcount, b.playcount, b.hit_length, b.status, b.ranked_date,
               b.title, b.artist, b.version
        FROM scores s
        JOIN players  p USING(user_id)
        JOIN beatmaps b USING(beatmap_id)
        WHERE b.passcount > 0 AND b.sr > 0 AND p.pp > 0
    """
    if post_rework_only:
        q += f" AND s.created_at >= '{REWORK_DATE}'"
    df = pd.read_sql_query(q, conn)
    if df.empty:
        return df

    df["bracket"] = pd.cut(
        df.player_pp,
        bins=[b[0] for b in BRACKETS] + [BRACKETS[-1][1]],
        labels=[f"{a}-{b}" for a, b in BRACKETS],
        right=False,
    )
    if weighted:
        w = player_weights(conn)
        df = df.merge(w[["user_id", "activity_weight"]], on="user_id", how="left")
        df["activity_weight"] = df.activity_weight.fillna(0.5)
    else:
        df["activity_weight"] = 1.0
    return df


# --------------------------------------------- negative binomial rate model

def overrepresentation(df: pd.DataFrame, min_passcount=500, shrink=3.0) -> pd.DataFrame:
    """Fit  log E[appearances] = log(passcount) + f(sr, length, mods) + eps.

    The offset term is the popularity control: it asks whether a map appears in
    more top-100 lists than its EXPOSURE justifies, not whether it appears a lot.

    Returns the per-map, per-bracket log rate ratio (eps), empirical-Bayes
    shrunk so that low-exposure maps don't dominate the top of the table.
    """
    out = []
    for bracket, sub in df.groupby("bracket", observed=True):
        if len(sub) < 200:
            continue
        n_players = sub.user_id.nunique()

        agg = sub.groupby("beatmap_id").agg(
            appearances=("activity_weight", "sum"),
            raw_count=("user_id", "size"),
            sr=("sr", "first"),
            passcount=("passcount", "first"),
            playcount=("playcount", "first"),
            hit_length=("hit_length", "first"),
            cs=("cs", "first"),
            od=("od", "first"),
            mean_pp=("pp", "mean"),
            # mean_pp averages what the SAMPLED BRACKET actually earned, which
            # badly understates the map for anyone who plays better than that
            # bracket. p90/max show what the map is really worth on a good run.
            p90_pp=("pp", lambda x: float(np.percentile(x, 90))),
            max_pp=("pp", "max"),
            title=("title", "first"),
            artist=("artist", "first"),
            version=("version", "first"),
            dominant_mods=("mods", lambda s: s.mode().iat[0] if len(s.mode()) else "NM"),
        ).reset_index()

        agg = agg[agg.passcount >= min_passcount].copy()
        if len(agg) < 50:
            continue

        agg["log_sr"] = np.log(agg.sr.clip(lower=0.1))
        agg["log_len"] = np.log1p(agg.hit_length.fillna(0))
        # Spline-ish basis on SR so difficulty is controlled flexibly.
        X = pd.DataFrame({
            "log_sr": agg.log_sr,
            "log_sr2": agg.log_sr ** 2,
            "log_sr3": agg.log_sr ** 3,
            "log_len": agg.log_len,
            "dt": agg.dominant_mods.str.contains("DT").astype(float),
            "hd": agg.dominant_mods.str.contains("HD").astype(float),
            "hr": agg.dominant_mods.str.contains("HR").astype(float),
        })
        X = sm.add_constant(X)
        offset = np.log(agg.passcount.values) + np.log(n_players)
        y = agg.appearances.values

        try:
            # Estimate dispersion via an auxiliary Poisson fit, then refit as NB.
            pois = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
            mu = pois.mu
            aux_y = ((y - mu) ** 2 - y) / mu
            alpha = max(float(sm.OLS(aux_y, mu).fit().params[0]), 1e-3)
            fit = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha),
                         offset=offset).fit()
            expected = fit.mu
        except Exception:
            expected = np.maximum(y.mean(), 1e-6) * np.ones_like(y, dtype=float)

        agg["expected"] = expected
        # Shrunk log rate ratio: pseudo-counts damp tiny-sample maps.
        agg["overrep"] = np.log((y + shrink) / (expected + shrink))
        agg["bracket"] = bracket
        out.append(agg)

    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True).sort_values("overrep", ascending=False)


# --------------------------------------------------------- HITS farmer loop

def hits_farm_scores(df: pd.DataFrame, seed: pd.Series | None = None,
                     iters=30, min_appearances=5):
    """Co-determine 'which players farm' and 'which maps are farm'.

    Bipartite HITS on the player-map graph. Seeding with the mechanistic
    pp-efficiency scores breaks the circularity, so the fixed point separates
    the farmer cluster from tournament and skill-focused players instead of
    just re-finding popular maps.
    """
    sub = df.groupby("beatmap_id").filter(lambda g: len(g) >= min_appearances)
    if sub.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    users = sub.user_id.unique()
    maps = sub.beatmap_id.unique()
    ui = {u: i for i, u in enumerate(users)}
    mi = {m: i for i, m in enumerate(maps)}

    rows = sub.user_id.map(ui).values
    cols = sub.beatmap_id.map(mi).values
    vals = sub.activity_weight.values

    from scipy.sparse import csr_matrix
    A = csr_matrix((vals, (rows, cols)), shape=(len(users), len(maps)))

    if seed is not None:
        m_auth = np.array([max(seed.get(m, 0.0), 0.0) for m in maps], dtype=float)
        if m_auth.sum() == 0:
            m_auth = np.ones(len(maps))
    else:
        m_auth = np.ones(len(maps))
    m_auth /= np.linalg.norm(m_auth) or 1.0

    for _ in range(iters):
        u_hub = A @ m_auth
        u_hub /= np.linalg.norm(u_hub) or 1.0
        m_auth = A.T @ u_hub
        m_auth /= np.linalg.norm(m_auth) or 1.0

    return (pd.Series(m_auth, index=maps, name="farm_authority"),
            pd.Series(u_hub, index=users, name="farmer_score"))


# ------------------------------------------------------------- final scoring

def farm_report(db_path: str, top_n=200, post_rework_only=False,
                min_observations=10) -> pd.DataFrame:
    """End-to-end: combine mechanistic efficiency, overrepresentation, HITS."""
    from .features import (fit_pp_surrogate, map_achievability,
                           map_pp_efficiency)

    conn = connect(db_path)
    df = score_table(conn, post_rework_only=post_rework_only)
    if df.empty:
        conn.close()
        raise SystemExit("No joined data. Run collection first.")

    print("[1/4] overrepresentation model")
    over = overrepresentation(df)

    print("[2/4] surrogate pp model")
    model, sdf = fit_pp_surrogate(db_path, post_rework_only=post_rework_only)
    eff = map_pp_efficiency(model, sdf)

    print("[3/4] achievability")
    ach = map_achievability(sdf)
    print(f"  forgiveness computed for {len(ach):,} maps")

    print("[3/4] HITS farmer weighting")
    seed = eff.set_index("beatmap_id").pp_efficiency_shrunk
    auth, farmers = hits_farm_scores(df, seed=seed)

    print("[4/4] combining")
    # A map ranked on 1-4 observations is noise, not a finding. The
    # shrinkage prior damps these but does not remove them, so enforce a
    # hard floor before anything reaches the output.
    before = len(over)
    over = over[over.raw_count >= min_observations]
    print(f"  dropped {before - len(over):,} maps with < {min_observations} "
          f"observations ({len(over):,} remain)")

    best = (over.sort_values("overrep", ascending=False)
                .drop_duplicates("beatmap_id"))
    out = best.merge(
        eff[["beatmap_id", "pp_efficiency_shrunk", "n_scores"]],
        on="beatmap_id", how="left")
    out = out.merge(auth.rename("farm_authority"),
                    left_on="beatmap_id", right_index=True, how="left")
    ach_cols = ["beatmap_id", "forgiveness", "median_acc", "pct_miss_free",
                "median_combo_ratio", "acc_slope"]
    out = out.merge(ach[[c for c in ach_cols if c in ach.columns]],
                    on="beatmap_id", how="left")
    out["forgiveness"] = out.forgiveness.fillna(0.0)

    for col in ["overrep", "pp_efficiency_shrunk", "farm_authority",
                "forgiveness"]:
        out[col] = out[col].fillna(0.0)
        s = out[col].std()
        out[f"z_{col}"] = (out[col] - out[col].mean()) / (s if s else 1.0)

    # Mechanistic axis leads: it is the half that survives a meta shift.
    # Forgiveness is the achievability check -- a map that overpays but
    # demands near-perfect accuracy and a full combo is not actually
    # farmable, and without this term it would rank identically to one
    # that pays out at 96% with a couple of misses.
    out["farm_score"] = (0.40 * out.z_pp_efficiency_shrunk
                         + 0.25 * out.z_overrep
                         + 0.20 * out.z_forgiveness
                         + 0.15 * out.z_farm_authority)

    hi_eff = out.z_pp_efficiency_shrunk > 0.5
    hi_adopt = out.z_overrep > 0.5
    out["quadrant"] = np.select(
        [hi_eff & ~hi_adopt, hi_eff & hi_adopt, ~hi_eff & hi_adopt],
        ["undiscovered", "known_farm", "stale_farm"],
        default="ignore")

    out["url"] = "https://osu.ppy.sh/b/" + out.beatmap_id.astype(str)
    conn.close()
    return out.sort_values("farm_score", ascending=False).head(top_n).reset_index(drop=True)


def save_report(db_path: str, df: pd.DataFrame) -> int:
    """Persist the ranking into farm.db as a `farm_report` table.

    The CSV is a snapshot for humans; the lobby bot needs the rankings in the
    database so it can select a pool by farm_score instead of falling back to
    raw popularity. Best row per beatmap only -- the report has one row per
    (map, bracket) and a lobby just wants the map's best case.
    """
    keep = ["beatmap_id", "artist", "title", "version", "dominant_mods", "sr",
            "bracket", "farm_score", "quadrant", "pp_efficiency_shrunk",
            "overrep", "forgiveness", "median_acc", "pct_miss_free",
            "median_combo_ratio", "hit_length", "cs", "od", "mean_pp",
            "p90_pp", "max_pp", "raw_count", "passcount"]
    cols = [c for c in keep if c in df.columns]
    best = (df[cols].sort_values("farm_score", ascending=False)
                    .drop_duplicates("beatmap_id"))

    conn = connect(db_path)
    conn.execute("DROP TABLE IF EXISTS farm_report")
    best.to_sql("farm_report", conn, index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_farm_report_score "
                 "ON farm_report(farm_score DESC)")
    conn.commit()
    conn.close()
    return len(best)