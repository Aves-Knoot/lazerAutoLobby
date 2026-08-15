#!/usr/bin/env python3
"""Run the lobby bot.

Local first:

    python run_bot.py --check        # config + token + db, no connection
    python run_bot.py --dry-run      # full loop against a fake hub, no osu!
    python run_bot.py                # live

--check and --dry-run are the point of this script existing: the live path
touches your real account and creates a real room, so there should be a way
to exercise everything else without doing that.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bot.api import OsuApi
from bot.auth import TokenManager
from bot.bot import LobbyBot
from bot.pool import MapPool

DEFAULT_CONFIG = {
    "spectator_url": "https://spectator.ppy.sh",
    "token_file": "tokens.json",
    "farm_db": "../osu-farm-finder/farm.db",
    "room_name": "FARM | auto-rotate | .help",
    "command_prefix": ".",
    "bot_user_id": 0,
    "admins": [],
    "sr_min": 4.5,
    "sr_max": 5.5,
    "max_length": 150,
    "min_passcount": 100000,
    "pool_limit": 120,
    "allowed_mods": ["HD", "HR", "NF"],
    "required_mods": [],
    "freestyle": False,
    "countdown_seconds": 15,
    "auto_start_seconds": 45,
    "min_players": 1,
    "auto_difficulty": True,
    "chat_poll_seconds": 3.0,
    "score_delay_seconds": 4,
    "requests_per_minute": 55,
    "close_room_on_exit": True,
    "ratings_db": "ratings.db",
    "rating_window_seconds": 90,
    "info_interval_seconds": 900,
    "queue_ahead": 0,
}


def load_config(path: str) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    p = Path(path)
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    else:
        p.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        print(f"Wrote a default {path} -- edit it, then re-run.")
        raise SystemExit(0)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--check", action="store_true",
                    help="validate config, token and pool, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the full loop against a fake hub")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="DEBUG level on the console (file log is always "
                         "DEBUG for our own modules)")
    ap.add_argument("--debug-libs", action="store_true",
                    help="also log httpx/httpcore/signalrcore internals -- "
                         "very noisy, only for debugging the libraries "
                         "themselves")
    ap.add_argument("--errors", action="store_true",
                    help="print WARNING/ERROR lines + tracebacks from the "
                         "log file and exit -- run this to get something "
                         "worth pasting instead of the whole terminal")
    args = ap.parse_args()

    from bot.logsetup import configure, extract_problems

    if args.errors:
        print(extract_problems())
        return 0

    log_path = configure(verbose=args.verbose, debug_libs=args.debug_libs)
    log = logging.getLogger("run")
    log.info("full session log: %s", log_path.resolve())

    cfg = load_config(args.config)

    db = Path(cfg["farm_db"])
    if not db.exists():
        log.error("farm db not found at %s -- set farm_db in %s",
                  db.resolve(), args.config)
        return 1

    # Validate before anything touches osu!: a bad mod list would otherwise
    # create a room and then immediately fail, burning a room slot.
    from bot.hub import validate_mods
    validate_mods(cfg.get("allowed_mods", []), cfg.get("required_mods", []))
    if cfg.get("required_mods"):
        log.warning("required_mods=%s -- pool SR values are NOMOD; the "
                    "effective difficulty in-lobby will be higher",
                    cfg["required_mods"])

    pool = MapPool(cfg["farm_db"], sr_min=cfg["sr_min"], sr_max=cfg["sr_max"],
                   max_length=cfg["max_length"],
                   min_passcount=cfg["min_passcount"], limit=cfg["pool_limit"])
    if not pool.maps:
        log.error("pool is empty -- widen sr range or lower min_passcount")
        return 1
    log.info("pool sample: %s", pool.maps[0].label)

    if args.dry_run:
        from fake_hub import FakeHub, FakeApi
        events: queue.Queue = queue.Queue()
        hub = FakeHub(events)
        api = FakeApi()
        bot = LobbyBot(hub, api, pool, cfg, events)
        log.info("DRY RUN -- no osu! connection, simulated events")
        bot.run()
        return 0

    secret = os.environ.get("OSU_BOT_CLIENT_SECRET")
    if not secret:
        log.error("OSU_BOT_CLIENT_SECRET is not set")
        return 1

    tokens = TokenManager(cfg["token_file"],
                          int(os.environ.get("OSU_BOT_CLIENT_ID", 0)) or
                          json.loads(Path(cfg["token_file"]).read_text())["client_id"],
                          secret)

    if args.check:
        tok = tokens.access_token()
        log.info("token OK (%d chars)", len(tok))
        api = OsuApi(tokens, cfg["requests_per_minute"])
        me = api._req("GET", "/me")
        if me:
            log.info("authenticated as %s (id %s)", me.get("username"), me.get("id"))
            if not cfg.get("bot_user_id"):
                log.warning("set bot_user_id to %s in %s so the bot ignores "
                            "its own messages", me.get("id"), args.config)
        api.close()
        log.info("all checks passed")
        return 0

    from bot.hub import RefereeHub
    events = queue.Queue()
    api = OsuApi(tokens, cfg["requests_per_minute"])
    hub = RefereeHub(cfg["spectator_url"], tokens.access_token, events)
    hub.connect()

    bot = LobbyBot(hub, api, pool, cfg, events)
    try:
        bot.run()
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        hub.stop()
        api.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
