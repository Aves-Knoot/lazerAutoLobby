"""osu!web API v2 client (chat, users, room scores).

Rate limiting is deliberately conservative and SHARED across every caller.
osu!'s terms ask for no more than 60 requests/minute, roughly 1/second. Chat
polling is the greedy consumer here -- polling once a second would eat the
entire budget for a single lobby -- so the default poll interval is 3s and
everything draws from the same bucket.
"""
from __future__ import annotations

import logging
import threading
import time

import httpx

log = logging.getLogger("api")

API = "https://osu.ppy.sh/api/v2"


class RateLimiter:
    def __init__(self, per_minute: int = 55):
        self.interval = 60.0 / max(per_minute, 1)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._next > now:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = max(now, self._next) + self.interval


class OsuApi:
    def __init__(self, tokens, per_minute: int = 55):
        self.tokens = tokens
        self.limiter = RateLimiter(per_minute)
        self.http = httpx.Client(
            timeout=httpx.Timeout(connect=10, read=20, write=10, pool=10))

    def close(self) -> None:
        self.http.close()

    def _req(self, method: str, path: str, retries: int = 3, **kw):
        for attempt in range(retries):
            self.limiter.wait()
            try:
                r = self.http.request(
                    method, API + path,
                    headers={
                        "Authorization": f"Bearer {self.tokens.access_token()}",
                        "x-api-version": "20220705",
                        "Accept": "application/json",
                        "User-Agent": "poolrotater/0.1",
                    }, **kw)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                log.warning("%s %s network error: %s", method, path, e)
                time.sleep(1.5 * (attempt + 1))
                continue

            if r.status_code in (200, 201, 204):
                if r.status_code == 204 or not r.text:
                    return {}
                try:
                    return r.json()
                except Exception:
                    return {}
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 10 * (attempt + 1)))
                log.warning("rate limited, sleeping %.0fs", wait)
                time.sleep(min(wait, 120))
                continue
            if 500 <= r.status_code < 600:
                time.sleep(2 ** attempt)
                continue
            log.error("%s %s -> %s %s", method, path, r.status_code, r.text[:200])
            return None
        return None

    # ---------------------------------------------------------- chat

    def chat_ack(self):
        """Keeps us 'present' in chat. Without periodic acks osu! silently
        stops treating us as an active participant in the channel."""
        return self._req("POST", "/chat/ack")

    def chat_messages(self, channel_id: int, since: int | None = None):
        params = {"limit": 50}
        if since:
            params["since"] = since
        return self._req("GET", f"/chat/channels/{channel_id}/messages",
                         params=params) or []

    def chat_send(self, channel_id: int, message: str):
        """Returns the created message, whose id the caller MUST record.

        The bot posts as the operator's own account, so sender_id cannot
        distinguish bot output from the operator typing. Tracking the ids of
        messages we sent is the only reliable way to avoid reading our own
        output back as commands.
        """
        return self._req("POST", f"/chat/channels/{channel_id}/messages",
                         json={"message": message, "is_action": False})

    # ---------------------------------------------------------- users

    def user(self, user_id: int, mode: str = "osu"):
        return self._req("GET", f"/users/{user_id}/{mode}", params={"key": "id"})

    def user_top_scores(self, user_id: int, mode: str = "osu", limit: int = 100):
        return self._req("GET", f"/users/{user_id}/scores/best",
                         params={"mode": mode, "limit": limit}) or []

    # ---------------------------------------------------------- rooms

    def room_item_scores(self, room_id: int, playlist_item_id: int):
        """Scores for one playlist item. MatchCompletedEvent gives us both IDs,
        so no extra state tracking is needed to build this call."""
        return self._req(
            "GET", f"/rooms/{room_id}/playlist/{playlist_item_id}/scores")

    def beatmap(self, beatmap_id: int):
        return self._req("GET", f"/beatmaps/{beatmap_id}")
