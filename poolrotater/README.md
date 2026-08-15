# poolrotater

Auto-rotating osu!lazer multiplayer farm lobby, driven by the referee hub API
and a map pool from the farm-finder database.

## Local setup

```
pip install httpx signalrcore
python get_token.py                 # once, on a machine with a browser
python run_bot.py                   # writes a default config.json, then exit
# edit config.json
python run_bot.py --check           # verify token + pool, no room created
python run_bot.py --dry-run         # full loop, fake hub, no osu! contact
python run_bot.py                   # live
```

`--check` and `--dry-run` exist so you can exercise everything without
creating a real room under your account. Use them first.

## config.json

| key | meaning |
|---|---|
| `farm_db` | path to the farm-finder sqlite db |
| `bot_user_id` | your osu! user id, so the bot ignores its own chat |
| `admins` | user ids allowed to run `!start !abort !range !close` |
| `sr_min` / `sr_max` | pool difficulty band |
| `max_length` | max drain seconds; short maps keep a lobby moving |
| `min_passcount` | download-friction gate, see below |
| `allowed_mods` | freemod options offered to players |
| `auto_difficulty` | retune the band to the room's median pp |

`--check` prints your user id; put it in `bot_user_id` or the bot will read
and react to its own messages.

## Why min_passcount matters

The farm finder's most valuable maps for personal play are the "undiscovered"
ones -- efficient and unfarmed. Those are the *worst* lobby maps, because
nobody has them installed and the room stalls on downloads. The pool therefore
gates on passcount and favours maps people already own. Lobby pools and
personal farm lists want opposite things.

## Protocol notes

- Rooms are created with a **random password**. `MakeRoomRequest` has no
  password field, so the bot must call `ChangeRoomSettings` with
  `password: ""` immediately after. Empty string clears; `null` would keep it.
- `MatchCompletedEvent` carries both `room_id` and `playlist_item_id`, which
  is exactly what `/rooms/{room}/playlist/{item}/scores` needs.
- Scores are not queryable the instant a match ends. The bot waits
  `score_delay_seconds` then retries for ~20s.
- Never edit the playlist while a match is running; the hub rejects it.
- The referee API is documented as unstable. A sudden flood of hub errors
  most likely means the contract moved.

## Rate limits

osu! asks for <=60 requests/minute. Chat polling is the greedy consumer, so
it defaults to one poll per 3s and everything shares a single bucket.
Outgoing chat is separately throttled to ~1 message per 1.5s -- this runs as
your own account, so a spam loop is your reputation.

## Moving to a VPS

Copy the folder, `pip install httpx signalrcore`, set `OSU_BOT_CLIENT_SECRET`,
and bring `tokens.json` with you. Nothing is machine-specific. Note the token
file rotates on every refresh, so don't overwrite it with a stale copy.
