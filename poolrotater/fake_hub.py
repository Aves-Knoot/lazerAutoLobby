"""Fake hub + API for --dry-run.

Simulates a lobby locally: players join, matches start and finish, scores come
back, and -- importantly -- the lazer playlist advances itself after a match.
That lets dry-run catch bugs where the bot accidentally replaces an item that
was already queued and promoted by the server.
"""
from __future__ import annotations

import logging
import queue
import random
import threading
import time

log = logging.getLogger("fake")


class FakeHub:
    def __init__(self, events: queue.Queue, speed: float = 1.0):
        self.events = events
        self.speed = speed
        self.room_id = 999001
        self._next_item_id = 1
        self._playlist: list[dict] = []
        self._current_item_id: int | None = None
        self._players = [
            (101, "AlphaPlayer"), (102, "BetaPlayer"), (103, "GammaPlayer")]

    def _new_item(self, beatmap_id: int, ruleset_id: int = 0,
                  allowed_mods=None, required_mods=None,
                  freestyle: bool = False) -> dict:
        item = {
            "id": self._next_item_id,
            "beatmap_id": int(beatmap_id),
            "ruleset_id": int(ruleset_id),
            "was_played": False,
            "order": len(self._playlist),
            "required_mods": [{"acronym": m} for m in (required_mods or [])],
            "allowed_mods": [{"acronym": m} for m in (allowed_mods or [])],
            "freestyle": bool(freestyle),
        }
        self._next_item_id += 1
        return item

    def _current_item(self) -> dict | None:
        for item in self._playlist:
            if item["id"] == self._current_item_id:
                return item
        return None

    def make_room(self, name, beatmap_id, ruleset_id=0):
        first = self._new_item(beatmap_id, ruleset_id)
        self._playlist = [first]
        self._current_item_id = first["id"]
        log.info("[fake] MakeRoom name=%r beatmap=%s item=%s",
                 name, beatmap_id, first["id"])
        threading.Timer(2.0 / self.speed, self._join_players).start()
        return {
            "room_id": self.room_id,
            "chat_channel_id": 555001,
            "name": name,
            "password": "r4nd0m",          # mirrors the real random default
            "playlist": [dict(first)],
            "players": [],
            "referees": [{"user_id": 1}],
        }

    def _join_players(self):
        for uid, _name in self._players:
            self.events.put(("UserJoined", {"room_id": self.room_id,
                                            "user_id": uid}))
            time.sleep(0.3 / self.speed)

    def change_room_settings(self, room_id, name=None, password=None,
                             match_type=None):
        log.info("[fake] ChangeRoomSettings password=%r", password)
        if password == "":
            log.info("[fake] room is now public")
        return {}

    def add_playlist_item(self, room_id, beatmap_id, allowed_mods=None,
                          required_mods=None, freestyle=False, ruleset_id=0):
        item = self._new_item(
            beatmap_id, ruleset_id,
            allowed_mods=allowed_mods,
            required_mods=required_mods,
            freestyle=freestyle,
        )
        self._playlist.append(item)
        log.info("[fake] AddPlaylistItem beatmap=%s item=%s",
                 beatmap_id, item["id"])
        self.events.put(("PlaylistItemAdded", {"item": dict(item)}))
        return dict(item)

    def edit_current_item(self, room_id, beatmap_id, allowed_mods=None,
                          required_mods=None, freestyle=None, ruleset_id=0):
        item = self._current_item()
        if item is None:
            raise RuntimeError("no current playlist item")
        item["beatmap_id"] = int(beatmap_id)
        item["ruleset_id"] = int(ruleset_id)
        if allowed_mods is not None:
            item["allowed_mods"] = [{"acronym": m} for m in allowed_mods]
        if required_mods is not None:
            item["required_mods"] = [{"acronym": m} for m in required_mods]
        if freestyle is not None:
            item["freestyle"] = bool(freestyle)
        log.info("[fake] EditCurrentPlaylistItem beatmap=%s allowed=%s item=%s",
                 beatmap_id, allowed_mods, item["id"])
        self.events.put(("PlaylistItemChanged", {"item": dict(item)}))
        return dict(item)

    def remove_playlist_item(self, room_id, item_id):
        item_id = int(item_id)
        before = len(self._playlist)
        self._playlist = [i for i in self._playlist if i["id"] != item_id]
        if len(self._playlist) == before:
            raise RuntimeError(f"playlist item {item_id} not found")
        log.info("[fake] RemovePlaylistItem item=%s", item_id)
        self.events.put(("PlaylistItemRemoved", {"playlist_item_id": item_id}))
        return {}

    def start_match(self, room_id, countdown=None):
        item = self._current_item()
        if item is None:
            raise RuntimeError("no current playlist item")
        log.info("[fake] StartMatch countdown=%s item=%s beatmap=%s",
                 countdown, item["id"], item["beatmap_id"])
        delay = (countdown or 0) / max(self.speed, 1) * 0.2
        threading.Timer(max(delay, 0.5), self._match_started).start()
        return {}

    def _match_started(self):
        item = self._current_item()
        if item is None:
            return
        item_id = item["id"]
        self.events.put(("MatchStarted", {
            "room_id": self.room_id,
            "playlist_item_id": item_id,
        }))
        threading.Timer(4.0 / self.speed,
                        lambda: self._match_completed(item_id)).start()

    def _match_completed(self, completed_item_id: int):
        completed = None
        for item in self._playlist:
            if item["id"] == completed_item_id:
                item["was_played"] = True
                completed = item
                break

        self.events.put(("MatchCompleted", {
            "room_id": self.room_id,
            "playlist_item_id": completed_item_id,
        }))

        # Simulate lazer's built-in playlist behavior: after a match the next
        # unplayed item becomes active automatically. The bot should observe
        # this event, not overwrite the item with EditCurrentPlaylistItem.
        next_item = next((i for i in self._playlist if not i["was_played"]), None)
        if next_item is not None:
            self._current_item_id = next_item["id"]
            self.events.put(("PlaylistItemChanged", {"item": dict(next_item)}))
            log.info("[fake] playlist auto-advanced to item=%s beatmap=%s",
                     next_item["id"], next_item["beatmap_id"])
        elif completed is not None:
            self._current_item_id = completed["id"]

    def abort_match(self, room_id):
        log.info("[fake] AbortMatch")
        self.events.put(("MatchAborted", {"room_id": self.room_id}))
        return {}

    def close_room(self, room_id):
        log.info("[fake] CloseRoom")
        return {}

    def invite(self, room_id, user_id):
        return {}

    def kick(self, room_id, user_id):
        return {}


class FakeApi:
    """Stands in for OsuApi. Emits plausible players and scores, including
    some above the player's #100 threshold so the celebration path fires."""

    def __init__(self):
        self.rng = random.Random(7)
        self.sent: list[str] = []
        self._profiles = {
            101: ("AlphaPlayer", 4800.0, 180.0),
            102: ("BetaPlayer", 5200.0, 205.0),
            103: ("GammaPlayer", 3100.0, 120.0),
        }

    def chat_ack(self):
        return {}

    def chat_messages(self, channel_id, since=None):
        return []

    def chat_send(self, channel_id, message):
        self.sent.append(message)
        print(f"    CHAT> {message}")
        return {}

    def user(self, user_id, mode="osu"):
        name, pp, _thr = self._profiles.get(user_id, (str(user_id), 1000.0, 50.0))
        return {"id": user_id, "username": name, "statistics": {"pp": pp}}

    def user_top_scores(self, user_id, mode="osu", limit=100):
        _n, _pp, thr = self._profiles.get(user_id, (None, None, 50.0))
        # descending, with beatmap ids, like the real endpoint
        return [{"pp": thr + (limit - i) * 1.5,
                 "beatmap": {"id": 500000 + i}} for i in range(limit)]

    def room_item_scores(self, room_id, playlist_item_id):
        scores = []
        for uid, (_n, _pp, thr) in self._profiles.items():
            # ~40% of the time the score beats their #100
            pp = thr * (self.rng.uniform(1.02, 1.25)
                        if self.rng.random() < 0.4
                        else self.rng.uniform(0.5, 0.95))
            scores.append({"user_id": uid, "pp": round(pp, 2),
                           "accuracy": self.rng.uniform(0.94, 0.995)})
        return {"scores": scores}

    def beatmap(self, beatmap_id):
        return {}

    def close(self):
        pass

    def _req(self, *a, **k):
        return {}