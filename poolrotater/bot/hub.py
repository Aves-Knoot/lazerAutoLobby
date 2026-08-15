"""Referee hub client (SignalR).

signalrcore is thread-based, not asyncio. Rather than bridging the two, hub
callbacks simply push onto a thread-safe queue that the bot's main loop
drains. Keeps the concurrency story boring, which matters for something meant
to run unattended.

Method and event names come from IRefereeHubServer / IRefereeHubClient.
Payload field names come from the published JSON Schemas, so they should be
exact -- but this API is explicitly marked unstable in the docs, so treat a
sudden flood of errors as "the contract moved" rather than "my code broke".
"""
from __future__ import annotations

import logging
import queue
import threading
import time

from signalrcore.hub_connection_builder import HubConnectionBuilder

log = logging.getLogger("hub")

# Server -> client events we care about. The rest exist but are noise for a
# rotating farm lobby (teams, rolls, bans, countdown chatter).
# Rate-changing mods alter song length, so every player in a room must share
# them. osu! therefore rejects them as FREE mods -- they can only be required
# for the whole lobby. Sending one in allowed_mods returns
# "Error 12: Invalid mods. Details: Invalid free mods were selected: DT".
RATE_CHANGING = {"DT", "NC", "HT", "DC"}

# Mods that make no sense in a multiplayer farm context or are outright
# rejected: autoplay/cinema/relax variants.
NEVER_FREE = RATE_CHANGING | {"AT", "CN", "RX", "AP", "TP"}


def validate_mods(allowed: list[str], required: list[str]) -> None:
    """Fail fast, before a room exists.

    Getting this wrong costs a room slot: MakeRoom succeeds, then the first
    EditCurrentPlaylistItem is rejected and the room has to be torn down.
    """
    bad = [m for m in (allowed or []) if m.upper() in NEVER_FREE]
    if bad:
        raise SystemExit(
            f"Invalid allowed_mods: {', '.join(bad)}.\n"
            f"Rate-changing mods ({', '.join(sorted(RATE_CHANGING))}) cannot "
            f"be free mods -- every player must share the same song speed.\n"
            f"Move them to required_mods to apply them to the whole lobby, "
            f"or drop them.\n"
            f"Note: with a rate-changing mod required, your pool's star "
            f"ratings are still NOMOD values -- lower sr_min/sr_max to suit."
        )
    overlap = set(m.upper() for m in (allowed or [])) & \
              set(m.upper() for m in (required or []))
    if overlap:
        raise SystemExit(
            f"Mods in both allowed_mods and required_mods: "
            f"{', '.join(sorted(overlap))}. Pick one.")


EVENTS = [
    "MatchStarted", "MatchCompleted", "MatchAborted",
    "UserJoined", "UserLeft", "UserKicked",
    "UserStatusChanged",          # idle / ready / playing / finished_play
    "PlaylistItemAdded", "PlaylistItemChanged", "PlaylistItemRemoved",
    "RoomSettingsChanged",
    "CountdownStarted", "CountdownStopped",
]


class RefereeHub:
    def __init__(self, spectator_url: str, token_provider, events: queue.Queue):
        self.url = spectator_url.rstrip("/") + "/referee"
        self.token_provider = token_provider
        self.events = events
        self.conn = None
        self._connected = threading.Event()
        self._results: dict[str, queue.Queue] = {}

    def connect(self, timeout: float = 30.0) -> None:
        self.conn = (
            HubConnectionBuilder()
            .with_url(self.url, options={
                # Called on every (re)connect, so a rotated token is picked up
                # automatically rather than the connection dying after an hour.
                "access_token_factory": self.token_provider,
                "verify_ssl": True,
            })
            .with_automatic_reconnect({
                "type": "raw",
                "keep_alive_interval": 15,
                "reconnect_interval": 5,
                "max_attempts": 60,
            })
            .build()
        )

        self.conn.on_open(self._on_open)
        self.conn.on_close(lambda: log.warning("hub connection closed"))
        self.conn.on_error(self._on_error)

        for name in EVENTS:
            self.conn.on(name, self._make_handler(name))

        log.info("connecting to %s", self.url)
        self.conn.start()
        if not self._connected.wait(timeout):
            raise SystemExit(
                f"Hub did not connect within {timeout}s. Check that the token "
                "has multiplayer.write_manage scope and the spectator URL is "
                "correct."
            )

    def _on_open(self) -> None:
        log.info("hub connected")
        self._connected.set()

    @staticmethod
    def describe(msg) -> str:
        """Pull readable text out of a CompletionMessage.

        signalrcore's error callback hands over the raw object, whose repr is
        just a memory address -- which hides the one piece of information that
        actually matters. The hub returns business-logic errors as strings
        here (see ThrowHelper in the docs), so this is where the real reason
        for a failed invocation lives.
        """
        for attr in ("error", "result", "invocation_id"):
            val = getattr(msg, attr, None)
            if val:
                return f"{attr}={val!r}"
        return repr(msg)

    def _on_error(self, msg) -> None:
        log.error("hub error: %s", self.describe(msg))
        # Unblock any invoke() waiting on a result -- otherwise a rejected
        # call just sits until its timeout with no explanation.
        for q in list(self._results.values()):
            try:
                q.put_nowait(msg)
            except Exception:
                pass

    def _make_handler(self, name: str):
        def handler(args):
            payload = args[0] if isinstance(args, list) and args else args
            log.debug("event %s: %s", name, payload)
            self.events.put((name, payload))
        return handler

    def stop(self) -> None:
        if self.conn:
            try:
                self.conn.stop()
            except Exception:
                pass

    # ------------------------------------------------------- invocation

    def invoke(self, method: str, args: list, timeout: float = 15.0):
        """Invoke a hub method and wait for its result.

        Hub failures surface as opaque strings (a documented SignalR
        limitation). Known business errors carry codes; see ThrowHelper.
        """
        key = f"{method}-{time.time()}"
        result_q: queue.Queue = queue.Queue(maxsize=1)
        self._results[key] = result_q
        log.debug("-> %s %s", method, args)
        try:
            self.conn.send(method, args, lambda m: result_q.put(m))
            try:
                msg = result_q.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError(
                    f"{method} did not respond within {timeout}s. If an error "
                    f"was logged just above, that is the reason.")
        finally:
            self._results.pop(key, None)

        err = getattr(msg, "error", None)
        if err:
            raise RuntimeError(f"{method} failed: {err}")
        result = getattr(msg, "result", None)
        log.debug("<- %s %s", method, result)
        return result

    # ------------------------------------------------ typed conveniences

    def make_room(self, name: str, beatmap_id: int, ruleset_id: int = 0):
        return self.invoke("MakeRoom", [{
            "name": name,
            "beatmap_id": int(beatmap_id),
            "ruleset_id": int(ruleset_id),
        }])

    def change_room_settings(self, room_id: int, name=None, password=None,
                             match_type=None):
        """NOTE the password semantics: null keeps the existing password,
        empty string clears it. Passing None to 'remove' the password would
        silently leave the room locked and unjoinable."""
        body = {}
        if name is not None:
            body["name"] = name
        if password is not None:
            body["password"] = password
        if match_type is not None:
            body["type"] = match_type
        return self.invoke("ChangeRoomSettings", [int(room_id), body])

    def edit_current_item(self, room_id: int, beatmap_id: int,
                          allowed_mods=None, required_mods=None,
                          freestyle=None, ruleset_id: int = 0):
        body = {"beatmap_id": int(beatmap_id), "ruleset_id": int(ruleset_id)}
        if required_mods is not None:
            body["required_mods"] = [{"acronym": m} for m in required_mods]
        if allowed_mods is not None:
            body["allowed_mods"] = [{"acronym": m} for m in allowed_mods]
        if freestyle is not None:
            body["freestyle"] = bool(freestyle)
        return self.invoke("EditCurrentPlaylistItem", [int(room_id), body])

    def add_playlist_item(self, room_id: int, beatmap_id: int,
                          allowed_mods=None, required_mods=None,
                          freestyle: bool = False, ruleset_id: int = 0):
        """Queue an extra map so players can pre-download it.

        Confirmed against AddPlaylistItemRequest's published schema: all five
        fields are REQUIRED (unlike EditCurrentPlaylistItemRequest, where
        everything is optional). No position/order field -- items append and
        the server sequences them, so queue order follows call order.
        """
        body = {
            "beatmap_id": int(beatmap_id),
            "ruleset_id": int(ruleset_id),
            "required_mods": [{"acronym": m} for m in (required_mods or [])],
            "allowed_mods": [{"acronym": m} for m in (allowed_mods or [])],
            "freestyle": bool(freestyle),
        }
        return self.invoke("AddPlaylistItem", [int(room_id), body])

    def remove_playlist_item(self, room_id: int, item_id: int):
        return self.invoke("RemovePlaylistItem",
                           [int(room_id), {"playlist_item_id": int(item_id)}])

    def start_match(self, room_id: int, countdown: int | None = None):
        return self.invoke("StartMatch", [int(room_id), {"countdown": countdown}])

    def abort_match(self, room_id: int):
        return self.invoke("AbortMatch", [int(room_id)])

    def close_room(self, room_id: int):
        return self.invoke("CloseRoom", [int(room_id)])

    def invite(self, room_id: int, user_id: int):
        return self.invoke("InvitePlayer", [int(room_id), int(user_id)])

    def kick(self, room_id: int, user_id: int):
        return self.invoke("KickPlayer", [int(room_id), int(user_id)])
