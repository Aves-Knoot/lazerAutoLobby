"""OAuth token management.

The single most important behaviour here: osu! returns a NEW refresh token on
every refresh and invalidates the old one. If we fail to persist the new value
we are locked out permanently and the browser consent flow has to be redone.
So the write happens BEFORE the new access token is returned to callers, and
it is atomic (temp file + replace) so a crash mid-write cannot corrupt it.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import httpx

log = logging.getLogger("auth")

TOKEN_URL = "https://osu.ppy.sh/oauth/token"


class TokenManager:
    def __init__(self, token_file: str, client_id: int, client_secret: str):
        self.path = Path(token_file)
        self.client_id = int(client_id)
        self.client_secret = client_secret
        self._lock = threading.Lock()
        self._access: str | None = None
        self._expires_at = 0.0

        if not self.path.exists():
            raise SystemExit(
                f"{self.path} not found. Run get_token.py first (on a machine "
                "with a browser) to produce it."
            )
        data = json.loads(self.path.read_text())
        self._refresh_token = data.get("refresh_token")
        if not self._refresh_token:
            raise SystemExit(f"No refresh_token in {self.path}.")
        # An access token may be present from the bootstrap run; treat it as
        # already expired so the first call proves refresh works immediately
        # rather than failing an hour later.
        log.info("loaded refresh token from %s", self.path)

    def _persist(self, refresh_token: str, access_token: str, expires_in: int) -> None:
        payload = {
            "client_id": self.client_id,
            "refresh_token": refresh_token,
            "access_token": access_token,
            "expires_in": expires_in,
            "updated_at": time.time(),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self.path)  # atomic on POSIX and Windows

    def _refresh(self) -> None:
        log.info("refreshing access token")
        r = httpx.post(TOKEN_URL, json={
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
        }, timeout=30)

        if r.status_code != 200:
            raise SystemExit(
                f"Token refresh failed ({r.status_code}): {r.text}\n"
                "If this says the refresh token is invalid, re-run "
                "get_token.py to get a fresh one."
            )
        data = r.json()
        new_refresh = data.get("refresh_token", self._refresh_token)

        # Persist FIRST. If the process dies right after this line we still
        # hold a usable token; if we returned the access token first and then
        # died, the rotated refresh token would be lost forever.
        self._persist(new_refresh, data["access_token"], data.get("expires_in", 86400))

        self._refresh_token = new_refresh
        self._access = data["access_token"]
        self._expires_at = time.time() + data.get("expires_in", 86400)
        log.info("token refreshed, valid for %.1f hours",
                 data.get("expires_in", 86400) / 3600)

    def access_token(self) -> str:
        with self._lock:
            if not self._access or time.time() > self._expires_at - 300:
                self._refresh()
            return self._access
