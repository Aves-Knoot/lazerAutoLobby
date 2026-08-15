#!/usr/bin/env python3
"""Close a room that got left open, e.g. after a crash.

    python close_room.py 3954653

Non-bot accounts can only have 4 open rooms at once, so leaked rooms
eventually block you from making new ones. Rooms do expire on their own, but
this is faster than waiting.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bot.auth import TokenManager
from bot.hub import RefereeHub


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)-5s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("close")

    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    room_ids = [int(a) for a in sys.argv[1:]]

    cfg_path = Path("config.json")
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    token_file = cfg.get("token_file", "tokens.json")
    spectator = cfg.get("spectator_url", "https://spectator.ppy.sh")

    secret = os.environ.get("OSU_BOT_CLIENT_SECRET")
    if not secret:
        log.error("OSU_BOT_CLIENT_SECRET is not set")
        return 1
    client_id = (int(os.environ.get("OSU_BOT_CLIENT_ID", 0))
                 or json.loads(Path(token_file).read_text())["client_id"])

    tokens = TokenManager(token_file, client_id, secret)
    hub = RefereeHub(spectator, tokens.access_token, queue.Queue())
    hub.connect()

    rc = 0
    for rid in room_ids:
        try:
            # A crashed process is no longer joined as referee, so rejoin
            # before trying to close. Referee privileges persist on the room.
            try:
                hub.invoke("JoinRoom", [rid])
                log.info("rejoined room %s", rid)
            except Exception as e:
                log.info("JoinRoom(%s): %s (continuing anyway)", rid, e)
            hub.close_room(rid)
            log.info("closed room %s", rid)
        except Exception as e:
            log.error("could not close %s: %s", rid, e)
            rc = 1

    hub.stop()
    return rc


if __name__ == "__main__":
    sys.exit(main())
