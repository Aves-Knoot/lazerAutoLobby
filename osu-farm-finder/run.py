#!/usr/bin/env python3
"""osu! farm map finder -- CLI.

  export OSU_CLIENT_ID=12345
  export OSU_CLIENT_SECRET=...

  python run.py collect --players 4000      # deep, all skill levels
  python run.py coverage                    # players/scores per pp bracket
  python run.py maps                 # download .osu files + strain features
  python run.py report --top 100 --out farm_maps.csv
  python run.py report --post-rework # current-meta-only view
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from osu_farm.core import Config


def main() -> int:
    ap = argparse.ArgumentParser(prog="osu-farm")
    ap.add_argument("--db", default="farm.db")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="pull players, top-100 scores, beatmaps")
    c.add_argument("--pages", type=int, default=100,
                   help="ranking pages, only used with --global-only")
    c.add_argument("--players", type=int, default=None, help="cap players fetched")
    c.add_argument("--rpm", type=int, default=None,
                   help="requests per minute (default: the value in core.py Config)")
    c.add_argument("--global-only", action="store_true",
                   help="use global rankings only (caps at top 10,000 players)")
    c.add_argument("--pp-floor", type=float, default=1000.0,
                   help="stop sampling a country below this pp")
    c.add_argument("--stage", choices=["all", "players", "scores", "beatmaps"],
                   default="all",
                   help="run one collection stage or the full pipeline")
    c.add_argument("--score-refresh-days", type=int, default=14,
                   help="skip top-100 snapshots refreshed within this many days")

    sub.add_parser("coverage", help="show players and scores per pp bracket")

    m = sub.add_parser("maps", help="download .osu files and build strain features")
    m.add_argument("--limit", type=int, default=None)
    m.add_argument("--dir", default="maps")

    r = sub.add_parser("report", help="run models and write the farm table")
    r.add_argument("--top", type=int, default=200,
                   help="number of highest-ranked maps to save (ignored with --all)")
    r.add_argument("--all", action="store_true",
                   help="save every qualifying map to farm_report instead of truncating to --top")
    r.add_argument("--out", default="farm_maps.csv")
    r.add_argument("--post-rework", action="store_true",
                   help="only use scores set after the July 2026 rework")
    r.add_argument("--min-obs", type=int, default=10,
                   help="drop maps seen in fewer than N sampled top-100s")
    r.add_argument("--sr-min", type=float, default=None, help="minimum star rating")
    r.add_argument("--sr-max", type=float, default=None, help="maximum star rating")
    r.add_argument("--max-length", type=int, default=None,
                   help="maximum drain time in seconds")
    r.add_argument("--min-passcount", type=int, default=None,
                   help="minimum public passcount")
    r.add_argument("--mods", default=None,
                   help="only show this exact dominant mod bucket, e.g. NM")
    r.add_argument("--exclude-dt", action="store_true",
                   help="exclude any dominant mod bucket containing DT")
    r.add_argument("--formula-weight", type=float, default=0.65,
                   help="Farm Score v2.1 formula-efficiency weight (default 0.65)")
    r.add_argument("--disagreement-penalty", type=float, default=0.30,
                   help="penalty strength for formula-positive/realized-negative disagreement (default 0.30)")
    r.add_argument("--no-sr-calibration", action="store_true",
                   help="disable SR-local Farm Score calibration")
    r.add_argument("--sr-calibration-bandwidth", type=float, default=1.25,
                   help="SR smoothing bandwidth in stars (default 1.25)")
    r.add_argument("--sr-calibration-prior-maps", type=float, default=300.0,
                   help="maps of prior support used to shrink sparse-tail calibration (default 300)")

    args = ap.parse_args()
    cfg = Config(db_path=args.db)

    if args.cmd == "collect":
        # Only override the Config default when the flag is actually passed.
        # A default value here would silently stomp core.py's setting.
        if args.rpm is not None:
            cfg.requests_per_minute = args.rpm
        print(f"  rate: {cfg.requests_per_minute} req/min, "
              f"concurrency {cfg.concurrency}")
        from osu_farm.collect import (
            collect_beatmaps, collect_players, collect_players_deep,
            collect_scores, coverage,
        )

        async def _collect_selected_stages():
            if args.stage in ("all", "players"):
                print("[players]")
                if args.global_only:
                    await collect_players(cfg, pages=args.pages)
                else:
                    await collect_players_deep(cfg, pp_floor=args.pp_floor)

            if args.stage in ("all", "scores"):
                print("[current top-100 scores]")
                await collect_scores(
                    cfg, limit=args.players, refresh_days=args.score_refresh_days
                )

            if args.stage in ("all", "beatmaps"):
                print("[beatmap metadata]")
                await collect_beatmaps(cfg)

        asyncio.run(_collect_selected_stages())
        print("\ncoverage:")
        coverage(cfg.db_path)

    elif args.cmd == "coverage":
        from osu_farm.collect import coverage
        coverage(cfg.db_path)

    elif args.cmd == "maps":
        from osu_farm.core import connect
        from osu_farm.features import build_map_features, download_osu_files
        conn = connect(cfg.db_path)
        ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT beatmap_id FROM scores")]
        conn.close()
        if args.limit:
            ids = ids[:args.limit]
        print(f"downloading up to {len(ids)} .osu files")
        n = asyncio.run(download_osu_files(ids, out_dir=args.dir))
        print(f"  {n} new files")
        print("building strain features")
        print(f"  {build_map_features(cfg.db_path, maps_dir=args.dir)} maps processed")

    elif args.cmd == "report":
        from osu_farm.analyze import farm_report
        df = farm_report(
            cfg.db_path,
            top_n=None if args.all else args.top,
            post_rework_only=args.post_rework,
            min_observations=args.min_obs,
            sr_min=args.sr_min,
            sr_max=args.sr_max,
            max_length=args.max_length,
            min_passcount=args.min_passcount,
            mods=args.mods,
            exclude_dt=args.exclude_dt,
            formula_weight=args.formula_weight,
            disagreement_weight=args.disagreement_penalty,
            sr_calibration=not args.no_sr_calibration,
            sr_calibration_bandwidth=args.sr_calibration_bandwidth,
            sr_calibration_prior_maps=args.sr_calibration_prior_maps,
        )
        cols = [
            "beatmap_id", "artist", "title", "version", "dominant_mods",
            "sr", "farm_score", "precalibration_farm_score",
            "base_farm_score", "sr_local_baseline_raw",
            "sr_calibration_strength", "sr_calibration_adjustment",
            "confidence",
            "confidence_label", "efficiency_agreement",
            "efficiency_disagreement", "disagreement_penalty",
            "pp_efficiency_shrunk", "realized_efficiency_shrunk",
            "overrep", "profile_impact", "median_position",
            "top10_count", "top20_count", "top30_count",
            "forgiveness", "median_acc", "pct_miss_free",
            "median_combo_ratio", "hit_length", "cs", "od", "mean_pp",
            "p90_pp", "max_pp", "raw_count", "passcount", "url",
        ]
        cols = [c for c in cols if c in df.columns]
        df[cols].to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
        from osu_farm.analyze import save_report
        n = save_report(cfg.db_path, df)
        print(f"wrote {n} maps to the farm_report table in {cfg.db_path}")
        with __import__("pandas").option_context(
            "display.width", 220, "display.max_columns", 35
        ):
            print(df[cols].head(25).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())