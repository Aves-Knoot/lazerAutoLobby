#!/usr/bin/env python3
"""Find and close leaked multiplayer rooms.

    python rooms.py                 # list rooms the API will tell us about
    python rooms.py --close-all     # try to close every open one
    python rooms.py --close 123 456 # try specific ids

IMPORTANT CAVEAT: CloseRoom only works while you are joined to the room as a
referee. If the bot process died, you are no longer joined, and JoinRoom may
refuse because leaving a room revokes referee privileges. In that case the
room is not recoverable through this API -- it will close on its own once it
has been empty for a while. This script tells you which is which instead of
failing silently.
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
from bot.hub import RefereeHub

log = logging.getLogger("rooms")


def load_ctx():
    cfg_path = Path("config.json")
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    token_file = cfg.get("token_file", "tokens.json")
    secret = os.environ.get("OSU_BOT_CLIENT_SECRET")
    if not secret:
        raise SystemExit("OSU_BOT_CLIENT_SECRET is not set")
    client_id = (int(os.environ.get("OSU_BOT_CLIENT_ID", 0))
                 or json.loads(Path(token_file).read_text())["client_id"])
    tokens = TokenManager(token_file, client_id, secret)
    return cfg, tokens


def list_rooms(api: OsuApi) -> list[dict]:
    """Ask osu!web which rooms we own.

    /rooms is documented as a lazer route, which may not be available to
    authorization-code tokens. If it 404s or errors we fall back to telling
    the user to supply ids manually rather than pretending we know.
    """
    found = []
    for mode in ("owned", "participated"):
        data = api._req("GET", "/rooms", params={"mode": mode})
        if data is None:
            log.warning("/rooms?mode=%s not available to this token", mode)
            continue
        rooms = data if isinstance(data, list) else data.get("rooms", [])
        for r in rooms:
            if r.get("ends_at") or r.get("status") == "ended":
                continue                      # already closed
            found.append({
                "id": r.get("id"),
                "name": r.get("name"),
                "participants": r.get("participant_count"),
                "mode": mode,
            })
    # dedupe
    seen, out = set(), []
    for r in found:
        if r["id"] and r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def try_close(hub: RefereeHub, room_id: int) -> bool:
    try:
        hub.invoke("JoinRoom", [room_id])
        log.info("  rejoined %s", room_id)
    except Exception as e:
        log.info("  JoinRoom(%s) refused: %s", room_id, str(e)[:120])
    try:
        hub.close_room(room_id)
        log.info("  CLOSED %s", room_id)
        return True
    except Exception as e:
        msg = str(e)
        if "not in the room" in msg:
            log.warning("  %s: cannot close -- no longer joined as referee. "
                        "This room will expire on its own.", room_id)
        else:
            log.warning("  %s: %s", room_id, msg[:160])
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--close", nargs="*", type=int, help="room ids to close")
    ap.add_argument("--close-all", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg, tokens = load_ctx()
    api = OsuApi(tokens, cfg.get("requests_per_minute", 55))

    targets = list(args.close or [])
    if not targets:
        log.info("looking up your rooms...")
        rooms = list_rooms(api)
        if rooms:
            log.info("\nopen rooms:")
            for r in rooms:
                log.info("  %-10s %-45s participants=%s (%s)",
                         r["id"], (r["name"] or "")[:45], r["participants"],
                         r["mode"])
            targets = [r["id"] for r in rooms] if args.close_all else []
        else:
            log.info("\nNo open rooms reported. Either they have all expired, "
                     "or the /rooms route is not available to this token.")
            log.info("You can still close specific ones: "
                     "python rooms.py --close 3954653 3954781")
            api.close()
            return 0

    if not targets:
        log.info("\n(no --close-all given, nothing closed)")
        api.close()
        return 0

    hub = RefereeHub(cfg.get("spectator_url", "https://spectator.ppy.sh"),
                     tokens.access_token, queue.Queue())
    hub.connect()
    ok = 0
    for rid in targets:
        log.info("room %s:", rid)
        if try_close(hub, rid):
            ok += 1
    hub.stop()
    api.close()

    log.info("\nclosed %d/%d", ok, len(targets))
    if ok < len(targets):
        log.info("Rooms that could not be closed are orphaned: the referee "
                 "connection is gone and cannot be re-established. They "
                 "auto-close once empty, so this resolves itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
