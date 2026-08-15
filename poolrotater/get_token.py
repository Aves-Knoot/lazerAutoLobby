#!/usr/bin/env python3
"""One-time OAuth bootstrapper -- run this ONCE on a machine with a browser.

Produces a refresh token that a headless server can use indefinitely without
ever opening a browser again.

Why this exists: the referee hub needs multiplayer.write_manage, and the
client_credentials grant would additionally need the `delegate` scope, which
osu! only grants to bot accounts. The authorization_code grant works for a
non-bot account, but ONLY for the owner of the OAuth client -- which is you.
The catch is that authorization_code needs an interactive browser consent
step, which a VPS can't do. So: consent once here, keep the refresh token.

    python get_token.py

Then copy tokens.json to the server, or just the refresh_token from it.

IMPORTANT: refresh responses return a NEW refresh token each time and
invalidate the old one. The runtime client must persist every new token it
receives, or you'll be locked out and have to re-run this script.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import secrets
import socket
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

import httpx

AUTHORIZE_URL = "https://osu.ppy.sh/oauth/authorize"
TOKEN_URL = "https://osu.ppy.sh/oauth/token"

# public              -- read beatmaps, users, room scores
# chat.read/write     -- read and post in the lobby channel
# multiplayer.write_manage -- the referee hub itself
#
# `delegate` is deliberately NOT here: it is implied by authorization_code and
# is only grantable to bot accounts anyway.
SCOPES = ["public", "chat.read", "chat.write", "multiplayer.write_manage"]


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Catches the single redirect osu! sends back after you click Authorize."""

    result: dict = {}

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith("/callback"):
            self.send_response(404)
            self.end_headers()
            return

        params = urllib.parse.parse_qs(parsed.query)
        _CallbackHandler.result = {k: v[0] for k, v in params.items()}

        ok = "code" in _CallbackHandler.result
        body = (
            "<html><body style='font-family:sans-serif;padding:3em'>"
            + ("<h2>Authorized.</h2><p>You can close this tab and return "
               "to the terminal.</p>"
               if ok else
               f"<h2>Authorization failed.</h2><pre>{_CallbackHandler.result}</pre>")
            + "</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())

        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, *args):
        pass  # keep the terminal clean


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Get an osu! refresh token.")
    ap.add_argument("--client-id", default=os.environ.get("OSU_CLIENT_ID"))
    ap.add_argument("--client-secret",
                    default=os.environ.get("OSU_BOT_CLIENT_SECRET"),
                    help="prefer setting OSU_BOT_CLIENT_SECRET in the "
                         "environment so it stays out of shell history")
    ap.add_argument("--port", type=int, default=3000)
    ap.add_argument("--out", default="tokens.json")
    ap.add_argument("--no-browser", action="store_true",
                    help="print the URL instead of opening a browser")
    args = ap.parse_args()

    if not args.client_id or not args.client_secret:
        print("Need --client-id and --client-secret, or OSU_BOT_CLIENT_ID / "
              "OSU_BOT_CLIENT_SECRET in the environment.\n"
              "These are NOT the OSU_CLIENT_ID / OSU_CLIENT_SECRET used by "
              "the map collector -- the lobby bot is a separate OAuth app.",
              file=sys.stderr)
        return 1

    redirect_uri = f"http://localhost:{args.port}/callback"

    if not _port_free(args.port):
        print(f"Port {args.port} is already in use. Close whatever is on it, "
              f"or pass --port and set that callback URL on your OAuth app.",
              file=sys.stderr)
        return 1

    # CSRF protection: osu! echoes this back, we verify it matches.
    state = secrets.token_urlsafe(24)
    auth_url = AUTHORIZE_URL + "?" + urllib.parse.urlencode({
        "client_id": args.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "state": state,
    })

    print("Scopes requested:")
    for s in SCOPES:
        print(f"  - {s}")
    print(f"\nCallback URL: {redirect_uri}")
    print("This MUST exactly match a callback URL registered on your OAuth "
          "app, or osu! will reject the request.\n")

    server = http.server.HTTPServer(("127.0.0.1", args.port), _CallbackHandler)

    if args.no_browser:
        print("Open this in a browser:\n")
        print(auth_url + "\n")
    else:
        print("Opening your browser. Click Authorize.\n")
        webbrowser.open(auth_url)
        print("If nothing opened, paste this manually:\n")
        print(auth_url + "\n")

    print("Waiting for the redirect...")
    server.serve_forever()      # _CallbackHandler shuts this down
    server.server_close()

    result = _CallbackHandler.result
    if "code" not in result:
        print(f"\nNo authorization code received. osu! said: {result}",
              file=sys.stderr)
        return 1
    if result.get("state") != state:
        print("\nState mismatch -- the response didn't come from the request "
              "we started. Aborting rather than trusting it.", file=sys.stderr)
        return 1

    print("Got the code. Exchanging it for tokens...")
    r = httpx.post(TOKEN_URL, json={
        "client_id": int(args.client_id),
        "client_secret": args.client_secret,
        "code": result["code"],
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }, timeout=30)

    if r.status_code != 200:
        print(f"\nToken exchange failed ({r.status_code}): {r.text}",
              file=sys.stderr)
        return 1

    data = r.json()
    if "refresh_token" not in data:
        print(f"\nNo refresh_token in the response: {data}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.write_text(json.dumps({
        "client_id": int(args.client_id),
        "refresh_token": data["refresh_token"],
        "access_token": data["access_token"],
        "expires_in": data.get("expires_in"),
        "scopes": SCOPES,
    }, indent=2))
    try:
        os.chmod(out, 0o600)
    except OSError:
        pass

    tok = data["refresh_token"]
    # Deliberately NOT printed. Terminal output gets pasted into chats, issue
    # trackers and screenshots; a refresh token plus the client secret is full
    # account access for the granted scopes. It lives in the file only.
    print(f"\nWrote {out.resolve()} (permissions 600)")
    print(f"Refresh token obtained: {tok[:6]}...{tok[-4:]} "
          f"({len(tok)} chars) -- full value is in the file, not printed here.")
    print("\nCopy that file to the server. Do NOT commit it, and do not "
          "paste its contents anywhere.")
    print("Note: each refresh returns a NEW refresh token and invalidates "
          "the previous one -- the runtime client must persist every new "
          "value it receives.")
    print("\nIf this token is ever exposed, reset the client secret on your "
          "OAuth app. That invalidates the token too, since refreshing "
          "requires both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
