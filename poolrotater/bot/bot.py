"""Lobby orchestration.

State machine, deliberately small:

    IDLE     -> map set, waiting for players / countdown to start
    STARTING -> StartMatch sent, waiting for MatchStarted
    PLAYING  -> match in progress, do not touch the playlist
    SCORING  -> match ended, fetching scores before rotating

The important invariant: never call EditCurrentPlaylistItem while PLAYING.
The hub rejects playlist edits during a match, and doing it anyway is how you
end up in a state the bot can't reason about.
"""
from __future__ import annotations

import logging
import queue
import re
import time
from statistics import median

from .pool import (AUTO_SR_HALF_WIDTH, MapPool, auto_range_for_pp,
                   target_sr_for_pp)
from .ppmath import gain_from
from .ratings import RatingStore

log = logging.getLogger("bot")

IDLE, STARTING, PLAYING, SCORING = "IDLE", "STARTING", "PLAYING", "SCORING"


class LobbyBot:
    def __init__(self, hub, api, pool: MapPool, cfg: dict, events: queue.Queue):
        self.hub, self.api, self.pool, self.cfg = hub, api, pool, cfg
        self.events = events

        self.room_id: int | None = None
        self.channel_id: int | None = None
        # Keep the configured room name as the template. Whenever the active
        # SR range changes, only the "(x.xx-y.yy*)" portion is rewritten.
        self._room_name_template = str(cfg.get("room_name", "Farm lobby"))
        self._current_room_name = self._room_name_template
        self.state = IDLE
        self.current_map = None
        self.current_item_id: int | None = None
        # Durable item -> map lookup. The server can advance the visible
        # playlist before MatchCompleted reaches us, so .good/.bad must be tied
        # to the playlist item that actually finished rather than whichever map
        # happens to be current when the completion event is processed.
        self._playlist_maps: dict[int, object] = {}
        # Keep the map/item that actually entered gameplay separate from the
        # playlist's newly-active item. lazer can advance the playlist as soon
        # as a match finishes, before score reporting is done.
        # Freeze the intended map/item as soon as StartMatch is sent. Playlist
        # events can arrive out of order around countdown completion, so this
        # is the authoritative "what did we ask lazer to play?" snapshot.
        self.starting_map = None
        self.starting_item_id: int | None = None

        self.playing_map = None
        self.playing_item_id: int | None = None
        self.playing_player_ids: set[int] = set()
        self.scoring_item_id: int | None = None

        # Protect a just-started match from one stale countdown-abort event
        # arriving out of order. Explicit .abort bypasses this protection.
        self._match_started_at = 0.0
        self._explicit_abort_pending = False

        self.players: dict[int, dict] = {}      # non-ref users only
        # Referees are visible in lazer rooms, but they are not gameplay
        # participants for auto-start / ready-count purposes.
        self.referee_ids: set[int] = set()
        self._start_immediate_retry = False

        # Joining/leaving changes the desired auto-difficulty, but we do NOT
        # retune immediately. Pool range + room name are updated only after the
        # current map has concluded, before the queue is refilled.
        self._auto_retune_pending = False

        # Room settings cannot be changed while a match/countdown is active.
        # A manual .range issued during STARTING/PLAYING is still accepted for
        # future map selection, but its title change is deferred until the
        # next safe between-map window (SCORING/IDLE), before another
        # countdown can begin.
        self._pending_room_name_range: tuple[float, float] | None = None

        self.last_message_id = 0
        # Ids of messages WE posted. The bot runs as the operator's own
        # account, so sender_id cannot tell the two apart -- without this the
        # bot reads its own output back and executes it as commands.
        self._own_messages: set[int] = set()
        # Set by the range command. Stops auto_difficulty from silently
        # clobbering a manual choice on the next join.
        self.manual_range = False
        # osu! chat swallows messages beginning with "!" -- that is the
        # BanchoBot command prefix, so such messages reach the API but never
        # render in the channel. Players cannot see each other's commands,
        # which makes the lobby look broken. Any other prefix displays fine.
        self.prefix = str(cfg.get("command_prefix", ".")) or "."
        self.ready: set[int] = set()          # user ids currently readied
        self.ratings = RatingStore(cfg.get("ratings_db", "ratings.db"))
        self.last_played_map = None           # what .good/.bad refers to
        self._rating_window_until = 0.0
        self._last_info = time.time()
        self._info_idx = 0
        self.queued: list[tuple[int, object]] = []   # [(playlist_item_id, PoolMap)]
        self._last_chat_poll = 0.0
        self._last_ack = 0.0
        self._last_sent = 0.0
        self._state_since = time.time()
        self._idle_since = self._state_since
        self._match_end_at = 0.0
        self.running = True

    # ------------------------------------------------------------ chat

    def say(self, msg: str) -> None:
        """Outgoing chat is rate limited hard and separately from the API
        budget. This runs as your own account -- a loop that spams the channel
        is your reputation, not an anonymous bot's."""
        if not self.channel_id:
            return
        gap = time.time() - self._last_sent
        if gap < 1.5:
            time.sleep(1.5 - gap)
        sent = self.api.chat_send(self.channel_id, msg[:400])
        mid = (sent or {}).get("message_id") or (sent or {}).get("id")
        if mid:
            self._own_messages.add(int(mid))
            if len(self._own_messages) > 200:
                self._own_messages = set(sorted(self._own_messages)[-100:])
        self._last_sent = time.time()

    def poll_chat(self) -> None:
        now = time.time()
        if now - self._last_chat_poll < self.cfg.get("chat_poll_seconds", 3.0):
            return
        self._last_chat_poll = now

        if now - self._last_ack > 30:
            self.api.chat_ack()
            self._last_ack = now

        msgs = self.api.chat_messages(self.channel_id, since=self.last_message_id or None)
        for m in msgs or []:
            mid = m.get("message_id", 0)
            if mid > self.last_message_id:
                self.last_message_id = mid
            if mid in self._own_messages:
                continue
            # bot_user_id is a fallback for a DEDICATED bot account. When the
            # bot runs as the operator (the normal case) it must stay unset,
            # or the operator's own commands get ignored too.
            if self.cfg.get("bot_user_id") and \
                    m.get("sender_id") == self.cfg["bot_user_id"]:
                continue
            content = (m.get("content") or "").strip()
            sender_id = m.get("sender_id")

            # Keep a searchable transcript of all incoming human chat in the
            # normal session log. Bot-authored messages were already filtered
            # above via _own_messages, so this captures player conversation,
            # complaints, feedback, and commands without echoing our own chat.
            if content:
                sender = self.players.get(sender_id, {}) if sender_id else {}
                sender_name = sender.get("name") or str(sender_id or "?")
                # Keep one physical log line per chat message even if the API
                # returns embedded newlines.
                log_content = content.replace("\r", " ").replace("\n", " ")
                log.info("CHAT <%s:%s> %s",
                         sender_name, sender_id or "?", log_content)

            if content.startswith(self.prefix):
                self.handle_command(sender_id, content)

    def handle_command(self, user_id: int, text: str) -> None:
        parts = text.split()
        cmd = parts[0].lower()
        if cmd.startswith(self.prefix):
            cmd = cmd[len(self.prefix):]
        log.info("command from %s: %s", user_id, text)
        p = self.prefix

        if cmd in ("help", "commands"):
            # NOTE: leading text is deliberate. If this line began with a
            # command token and the self-filter ever failed, the bot would
            # parse its own help output as that command.
            self.say(f"Commands: {p}queue {p}pool {p}stats | "
                     f"admin: {p}skip {p}start {p}abort "
                     f"{p}range <min> <max> {p}auto {p}close")

        elif cmd == "queue":
            if self.current_map:
                self.say(f"Now: {self.current_map.label} — {self.current_map.url}")
            if self.queued:
                up_next = " | ".join(m.title[:30] for _i, m in self.queued)
                self.say(f"Up next: {up_next}")

        elif cmd == "pool":
            scored = sum(1 for m in self.pool.maps if m.farm_score)
            src = "farm-ranked" if scored else "popularity-ranked"

            active = self.gameplay_player_ids()
            pps = [
                self.players[uid]["pp"]
                for uid in active
                if self.players.get(uid, {}).get("pp")
            ]
            range_text = f"{self.pool.sr_min:.2f}-{self.pool.sr_max:.2f}*"

            if self.manual_range:
                difficulty_text = f"manual {range_text}"

            elif self.cfg.get("auto_difficulty", True):
                if pps:
                    med = float(median(pps))
                    half = float(self.cfg.get(
                        "auto_sr_half_width", AUTO_SR_HALF_WIDTH))
                    target = target_sr_for_pp(med, half_width=half)
                    next_lo, next_hi = auto_range_for_pp(
                        med, half_width=half)

                    pending_change = (
                        self._auto_retune_pending
                        and (
                            abs(next_lo - self.pool.sr_min) >= 0.001
                            or abs(next_hi - self.pool.sr_max) >= 0.001
                        )
                    )

                    if pending_change:
                        difficulty_text = (
                            f"auto current {range_text}; next "
                            f"{next_lo:.2f}-{next_hi:.2f}* after map "
                            f"(target {target:.2f}*, median {med:.0f}pp/"
                            f"{len(pps)} player"
                            f"{'s' if len(pps) != 1 else ''})"
                        )
                    else:
                        difficulty_text = (
                            f"auto {range_text}, target {target:.2f}* "
                            f"from median {med:.0f}pp/{len(pps)} player"
                            f"{'s' if len(pps) != 1 else ''}"
                        )
                else:
                    difficulty_text = (
                        f"auto on, current/default {range_text}, "
                        f"waiting for a player"
                    )

            else:
                difficulty_text = f"auto off, {range_text}"

            self.say(
                f"Pool: {len(self.pool.maps)} {src} maps, "
                f"{difficulty_text}, under {self.pool.max_length}s"
            )

        elif cmd in ("good", "bad"):
            if not self.last_played_map:
                return
            if time.time() > self._rating_window_until:
                self.say(f"Rating window closed. Rate right after a map ends.")
                return
            first = self.ratings.rate(self.last_played_map.beatmap_id,
                                      user_id, 1 if cmd == "good" else -1)
            if first:
                up, down = self.ratings.tally(self.last_played_map.beatmap_id)
                self.say(f"Noted ({up} good / {down} bad on "
                         f"{self.last_played_map.title[:30]})")

        elif cmd == "stats":
            active = self.gameplay_player_ids()
            if active:
                pps = [self.players[uid]["pp"] for uid in active
                       if self.players.get(uid, {}).get("pp")]
                if pps:
                    med = sorted(pps)[len(pps) // 2]
                    self.say(f"{len(active)} players, median {med:.0f}pp")

        # ---- admin only
        elif user_id in self.cfg.get("admins", []):
            if cmd == "skip":
                if self.state == PLAYING:
                    self.say("Can't skip mid-match.")
                else:
                    self.say("Skipping.")
                    self.skip_map()
            elif cmd == "start":
                self.start_match()
            elif cmd == "abort":
                self._explicit_abort_pending = True
                self.hub.abort_match(self.room_id)
            elif cmd == "close":
                self.say("Closing lobby.")
                self.running = False
            elif cmd == "range" and len(parts) == 3:
                try:
                    lo, hi = float(parts[1]), float(parts[2])
                    n = self.pool.set_range(lo, hi)
                    if n == 0:
                        # MapPool restored the previous known-good range.
                        self.say(
                            f"No eligible maps in {lo:.1f}-{hi:.1f}*; "
                            f"keeping {self.pool.sr_min:.1f}-"
                            f"{self.pool.sr_max:.1f}*."
                        )
                    else:
                        self._auto_retune_pending = False
                        # Lock it in until .auto is used again.
                        self.manual_range = True

                        renamed_now = self.update_room_name_for_range(
                            self.pool.sr_min, self.pool.sr_max)

                        if renamed_now:
                            self.say(
                                f"Pool set to {self.pool.sr_min:.1f}-"
                                f"{self.pool.sr_max:.1f}* ({n} maps). "
                                f"Auto-difficulty off; {p}auto to re-enable."
                            )
                        else:
                            self.say(
                                f"Pool set to {self.pool.sr_min:.1f}-"
                                f"{self.pool.sr_max:.1f}* ({n} maps). "
                                f"Room title will update between maps; "
                                f"auto-difficulty off; {p}auto to re-enable."
                            )
                except ValueError:
                    self.say(f"Usage: {p}range 4.5 5.5")
            elif cmd == "auto":
                # Enable auto in-memory even if config started with it disabled.
                self.cfg["auto_difficulty"] = True
                self.manual_range = False
                self._auto_retune_pending = True

                # If .range was issued during an active match and then .auto
                # is requested before that match ends, the pending manual title
                # is stale. The deferred auto calculation below becomes the
                # authoritative desired range/title instead.
                self._pending_room_name_range = None

                # Explicit admin command while safely IDLE can apply now.
                # During countdown/gameplay/scoring, defer until the map ends.
                if self.state == IDLE:
                    self.apply_pending_auto_retune(force=True)
                    self.say("Auto-difficulty enabled.")
                else:
                    self.say("Auto-difficulty enabled; range will update "
                             "after this map.")

    # ------------------------------------------------------------ room

    def _room_name_for_range(self, lo: float, hi: float) -> str:
        """Render the configured room title with the active SR range.

        If the configured title already contains something like
        "(4.8-5.4*)" that part is replaced. Otherwise the range is inserted
        before the first bracketed suffix (for example "[ALPHA]"), or appended
        to the end if there is no suffix.
        """
        label = f"({lo:.2f}-{hi:.2f}*)"
        name = self._room_name_template

        range_pattern = r"\(\s*\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*\*\s*\)"
        if re.search(range_pattern, name):
            return re.sub(range_pattern, label, name, count=1)

        bracket = name.find("[")
        if bracket >= 0:
            left = name[:bracket].rstrip()
            right = name[bracket:].lstrip()
            return f"{left} {label} {right}".strip()

        return f"{name.rstrip()} {label}".strip()

    def update_room_name_for_range(
            self, lo: float | None = None,
            hi: float | None = None) -> bool:
        """Synchronize the room title, deferring when room state is unsafe.

        osu!'s referee server rejects ChangeRoomSettings while a match or
        match-start countdown is active. Range changes are therefore allowed
        immediately for future map selection, but title changes are queued
        until SCORING/IDLE.

        Returns True when the live title is already synchronized or was
        changed now. Returns False when the update is deferred/retry-pending.
        """
        if not self.room_id:
            return False

        lo = self.pool.sr_min if lo is None else float(lo)
        hi = self.pool.sr_max if hi is None else float(hi)
        new_name = self._room_name_for_range(lo, hi)

        if new_name == self._current_room_name:
            self._pending_room_name_range = None
            return True

        # Do not even ask the server in these states. Error 9 is expected if
        # ChangeRoomSettings is attempted during gameplay/countdown.
        if self.state in (STARTING, PLAYING):
            self._pending_room_name_range = (lo, hi)
            log.info(
                "deferring room rename to %.2f-%.2f* until between maps "
                "(state=%s)",
                lo, hi, self.state,
            )
            return False

        try:
            self.hub.change_room_settings(self.room_id, name=new_name)
            self._current_room_name = new_name
            self._pending_room_name_range = None
            log.info("room renamed for active range: %s", new_name)
            return True
        except Exception as e:
            # Keep the desired title queued. A transient hub race should not
            # lose the requested name, and it must never crash the lobby.
            self._pending_room_name_range = (lo, hi)
            log.warning(
                "could not update room name for %.2f-%.2f*: %s; "
                "will retry at the next safe between-map boundary",
                lo, hi, e,
            )
            return False

    def apply_pending_room_name_update(self) -> bool:
        """Apply a queued room title only during a safe between-map state."""
        if self._pending_room_name_range is None:
            return True

        if self.state not in (SCORING, IDLE):
            return False

        lo, hi = self._pending_room_name_range
        return self.update_room_name_for_range(lo, hi)

    def create_room(self) -> None:
        first = self.pool.next_map()
        if not first:
            raise SystemExit("Pool is empty -- widen the SR range or lower "
                             "min_passcount in config.")

        log.info("creating room")
        initial_name = self._room_name_for_range(
            self.pool.sr_min, self.pool.sr_max)
        resp = self.hub.make_room(initial_name, first.beatmap_id)
        self._current_room_name = initial_name
        self.room_id = resp["room_id"]
        self.channel_id = resp["chat_channel_id"]
        self.current_map = first

        # Referees and players are separate concepts in the referee API. Keep
        # referee accounts out of player counts even if they are visible in the
        # lazer room or emit UserJoined/UserStatusChanged events.
        self.referee_ids = {
            int(r["user_id"]) for r in (resp.get("referees") or [])
            if r.get("user_id") is not None
        }
        if self.referee_ids:
            log.info("referee ids (excluded from player counts): %s",
                     ", ".join(map(str, sorted(self.referee_ids))))

        self._sync_playlist(resp)
        if self.current_item_id is not None:
            self._playlist_maps[self.current_item_id] = first
        log.info("room %s created (channel %s)", self.room_id, self.channel_id)

        # Rooms are created with a RANDOM PASSWORD. Empty string clears it;
        # null would keep it. Until this lands nobody can join.
        self.hub.change_room_settings(self.room_id, password="")
        log.info("password cleared, room is public")

        self.say(f"{self._current_room_name} — auto-rotating farm pool. "
                 f"Type {self.prefix}help for commands.")

        # From here on the room EXISTS on osu!'s side. Any exception must not
        # escape without closing it -- non-bot accounts cap at 4 open rooms,
        # so a few crashed starts will lock you out of making more.
        try:
            self.apply_map(first, announce=True)
            self.fill_queue()
        except Exception:
            log.exception("failed to set the first map; closing room %s",
                          self.room_id)
            try:
                self.hub.close_room(self.room_id)
                log.info("room %s closed", self.room_id)
            except Exception:
                log.error("could not close room %s -- close it manually, or "
                          "it will occupy one of your 4 room slots until it "
                          "expires", self.room_id)
            raise

    def _sync_playlist(self, resp: dict) -> None:
        items = resp.get("playlist") or []
        active = [i for i in items if not i.get("was_played")]
        if active:
            self.current_item_id = active[0].get("id")

    def apply_map(self, m, announce: bool = False) -> None:
        resp = self.hub.edit_current_item(
            self.room_id, m.beatmap_id,
            allowed_mods=self.cfg.get("allowed_mods", ["HD", "HR", "NF"]),
            required_mods=self.cfg.get("required_mods", []),
            freestyle=self.cfg.get("freestyle", False),
        )
        if isinstance(resp, dict) and resp.get("id") is not None:
            self.current_item_id = int(resp["id"])
        if self.current_item_id is not None:
            self._playlist_maps[self.current_item_id] = m
        self.current_map = m
        if announce:
            self._announce_map(m)

    def _announce_map(self, m) -> None:
        extra = ""
        if m.max_pp:
            extra += f" ~{m.max_pp:.0f}pp"
        if m.quadrant:
            extra += f" [{m.quadrant}]"
        self.say(f"Next: {m.label} [{m.hit_length}s]{extra} — {m.url}")

    def _adopt_playlist_item(self, item: dict) -> bool:
        """Mirror lazer's active playlist item into local bot state.

        The server owns playlist advancement. When it promotes one of our
        queued items, remove that item from ``self.queued`` and make it the
        bot's current map. No EditCurrentPlaylistItem call is needed.
        """
        item_id = item.get("id")
        beatmap_id = item.get("beatmap_id")

        match_idx = None
        for i, (queued_id, queued_map) in enumerate(self.queued):
            if item_id is not None and queued_id == item_id:
                match_idx = i
                break
            if beatmap_id and queued_map.beatmap_id == beatmap_id:
                match_idx = i
                break

        if match_idx is None:
            if item_id is not None:
                self.current_item_id = item_id
                known = self._playlist_maps.get(int(item_id))
                if known is not None:
                    self.current_map = known
                    return True
            return False

        # Normally match_idx is zero. If the hub skipped/removed an item, drop
        # everything up through the item it says is active so local order stays
        # aligned with lazer's playlist.
        item_id, m = self.queued[match_idx]
        del self.queued[:match_idx + 1]
        self.current_item_id = item.get("id") or item_id
        if self.current_item_id is not None:
            self._playlist_maps[int(self.current_item_id)] = m
        self.current_map = m
        log.info("playlist advanced to queued map: %s", m.label)
        return True

    def rotate(self) -> None:
        """Finish a completed match without replacing lazer's next item.

        lazer automatically advances to the next unplayed playlist entry. The
        old implementation popped that same queued map and then called
        EditCurrentPlaylistItem, effectively replacing an item the server had
        already promoted. Here we trust the playlist and only append a new map
        to the END afterward to keep the queue full.
        """
        # Usually PlaylistItemChanged has already promoted the next queued map
        # while we were waiting for score ingestion. If it has not arrived,
        # adopt the first queued item optimistically; it is the next item lazer
        # will choose because AddPlaylistItem preserves append order.
        if self.current_item_id == self.scoring_item_id or self.current_map is self.last_played_map:
            if self.queued:
                item_id, nxt = self.queued.pop(0)
                self.current_item_id = item_id
                self.current_map = nxt
                log.info("playlist advance event not seen; assuming queued item %s",
                         item_id)
            else:
                # Queueing may be disabled. Add one fresh item instead of
                # editing the already-played slot. Once it is the only unplayed
                # entry, lazer can make it current naturally.
                nxt = self.pool.next_map()
                if not nxt:
                    self.say("Pool exhausted.")
                    self.set_state(IDLE)
                    return
                resp = self.hub.add_playlist_item(
                    self.room_id, nxt.beatmap_id,
                    allowed_mods=self.cfg.get("allowed_mods", ["HD", "HR", "NF"]),
                    required_mods=self.cfg.get("required_mods", []),
                    freestyle=self.cfg.get("freestyle", False),
                )
                item_id = (resp or {}).get("id") if isinstance(resp, dict) else None
                self.current_item_id = item_id
                if item_id is not None:
                    self._playlist_maps[int(item_id)] = nxt
                self.current_map = nxt

        if self.current_map:
            self._announce_map(self.current_map)

        # Roster changes only affect difficulty BETWEEN maps. Existing lazer
        # queue entries stay untouched; after retuning, only newly appended
        # maps come from the new range.
        self.apply_pending_auto_retune()

        # A .range command may have arrived while the just-finished map (or its
        # countdown) was active. The title change is only legal now, after
        # gameplay has concluded and before the next countdown begins.
        self.apply_pending_room_name_update()

        # Refill AFTER adopting the promoted item and applying any deferred
        # auto range/title. One map leaves the front, one new map is appended.
        self.fill_queue()
        self.starting_map = None
        self.starting_item_id = None
        self.playing_map = None
        self.playing_item_id = None
        self.playing_player_ids.clear()
        self.scoring_item_id = None
        self._match_started_at = 0.0
        self._explicit_abort_pending = False
        self.set_state(IDLE)

    def skip_map(self) -> None:
        """Manual skip while idle.

        Skipping is different from normal post-match advancement: the current
        playlist item has not been played, so replacing it is intentional. If
        the replacement already existed in the pre-download queue, remove that
        duplicate queued entry first.
        """
        if self.queued:
            item_id, nxt = self.queued.pop(0)
            if item_id is not None:
                try:
                    self.hub.remove_playlist_item(self.room_id, item_id)
                    self._playlist_maps.pop(int(item_id), None)
                except Exception as e:
                    log.warning("could not remove queued item %s before skip: %s",
                                item_id, e)
        else:
            nxt = self.pool.next_map()

        if not nxt:
            self.say("Pool exhausted.")
            return

        self.apply_map(nxt, announce=True)
        self.fill_queue()
        self.set_state(IDLE)

    def fill_queue(self) -> None:
        """Keep queue_ahead extra items sitting in the playlist so people can
        pre-download them instead of waiting when their turn comes."""
        n = int(self.cfg.get("queue_ahead", 0))
        if n <= 0:
            return
        while len(self.queued) < n:
            m = self.pool.next_map()
            if not m:
                break
            try:
                resp = self.hub.add_playlist_item(
                    self.room_id, m.beatmap_id,
                    allowed_mods=self.cfg.get("allowed_mods", ["HD", "HR", "NF"]),
                    required_mods=self.cfg.get("required_mods", []),
                    freestyle=self.cfg.get("freestyle", False),
                )
            except Exception as e:
                # Queuing ahead is a nice-to-have. If the hub rejects it for
                # any reason, drop back to single-item rotation rather than
                # taking the whole bot down over it.
                log.warning("could not queue ahead: %s -- disabling for "
                           "this session", e)
                self.cfg["queue_ahead"] = 0
                return
            item_id = (resp or {}).get("id") if isinstance(resp, dict) else None
            if item_id is not None:
                self._playlist_maps[int(item_id)] = m
            self.queued.append((item_id, m))
            log.debug("queued ahead silently: %s", m.label)

    def gameplay_player_ids(self) -> set[int]:
        """Users who should count toward starting a match.

        Referees are excluded entirely. A normal user explicitly spectating
        also does not count until they return to a playable lobby status.
        """
        return {
            uid for uid, info in self.players.items()
            if uid not in self.referee_ids
            and info.get("status", "idle") != "spectating"
        }

    def gameplay_player_count(self) -> int:
        return len(self.gameplay_player_ids())

    def start_match(self) -> None:
        if self.state != IDLE:
            return

        minimum = int(self.cfg.get("min_players", 1))
        actual = self.gameplay_player_count()
        if actual < minimum:
            log.info("not starting: %d/%d gameplay players", actual, minimum)
            return

        # Final room-settings gate before countdown. If an earlier active
        # match forced a title update to be deferred, IDLE is the last safe
        # point to apply it before StartMatch makes room settings illegal.
        self.apply_pending_room_name_update()

        cd = self.cfg.get("countdown_seconds", 15)
        self._start_immediate_retry = False
        self._explicit_abort_pending = False

        # This snapshot is what later score/rating logic follows. Do it before
        # StartMatch so PlaylistItemChanged cannot move current_map underneath
        # us during the countdown.
        self.starting_map = self.current_map
        self.starting_item_id = self.current_item_id

        self.hub.start_match(self.room_id, countdown=cd)
        self.set_state(STARTING)
        if self.last_played_map and time.time() <= self._rating_window_until:
            title = self.last_played_map.title[:45]
            self.say(
                f"Starting in {cd}s — rate the last map ({title}) with "
                f"{self.prefix}good or {self.prefix}bad."
            )
        else:
            self.say(f"Starting in {cd}s.")

    def _stop_match_countdown(self) -> None:
        """Stop a pending countdown using the dedicated referee operation."""
        try:
            stop = getattr(self.hub, "stop_match_countdown", None)
            if callable(stop):
                stop(self.room_id)
                return

            # RefereeHub exposes generic invoke(). This keeps this file usable
            # without requiring a simultaneous hub.py replacement.
            invoke = getattr(self.hub, "invoke", None)
            if callable(invoke):
                invoke("StopMatchCountdown", [int(self.room_id)])
                return

            # Local fake/older test doubles may only expose abort_match.
            self.hub.abort_match(self.room_id)
        except Exception as e:
            log.warning("could not stop pending match countdown: %s", e)

    def cancel_start_if_too_few_players(self) -> bool:
        """Cancel a pending countdown if the room drops below min_players."""
        if self.state != STARTING:
            return False

        minimum = int(self.cfg.get("min_players", 1))
        actual = self.gameplay_player_count()
        if actual >= minimum:
            return False

        log.info("cancelling pending start: %d/%d gameplay players remain",
                 actual, minimum)
        self._stop_match_countdown()
        self.ready.clear()
        self.starting_map = None
        self.starting_item_id = None
        self.set_state(IDLE)
        self.say("Start cancelled — waiting for players.")
        return True

    def ready_threshold_met(self) -> bool:
        """Start once enough actual gameplay players have readied up."""
        active = self.gameplay_player_ids()
        n = len(active)
        if n == 0:
            return False
        r = len(self.ready & active)
        need = 0.75 if n >= 4 else 0.66
        return (r / n) >= need

    def maybe_post_info(self) -> None:
        """Occasional nudges. Deliberately infrequent -- this posts as the
        operator's own account and a chatty bot is an annoying one."""
        every = self.cfg.get("info_interval_seconds", 900)
        if time.time() - self._last_info < every:
            return
        self._last_info = time.time()
        p = self.prefix
        self.say(
            f"Commands: {p}queue {p}pool {p}stats {p}good {p}bad"
        )
        self._info_idx += 1

    def set_state(self, s: str) -> None:
        if s != self.state:
            log.info("state %s -> %s", self.state, s)
        now = time.time()
        self.state = s
        self._state_since = now
        if s != STARTING:
            self._start_immediate_retry = False
        if s == IDLE:
            self._idle_since = now

    # ---------------------------------------------------------- players

    def on_user_joined(self, user_id: int) -> None:
        if user_id in self.referee_ids:
            log.info("referee joined room; not counting as player: %s", user_id)
            return
        if user_id in self.players:
            return
        u = self.api.user(user_id)
        if not u:
            self.players[user_id] = {
                "pp": None, "name": str(user_id), "status": "idle"
            }
            return
        stats = u.get("statistics") or {}
        pp = stats.get("pp") or 0.0
        name = u.get("username", str(user_id))

        # Cache the WHOLE top 100, not just the cut-off. We need the full
        # list to work out which slot a new score takes and how much profile
        # pp it actually adds. Fetched once on join -- re-polling after every
        # map would blow the request budget instantly.
        raw = self.api.user_top_scores(user_id, limit=100)
        top = sorted(
            ((s.get("pp") or 0.0, (s.get("beatmap") or {}).get("id") or 0)
             for s in raw if s.get("pp")),
            reverse=True)
        threshold = top[-1][0] if len(top) >= 100 else 0.0

        self.players[user_id] = {"pp": pp, "name": name,
                                 "threshold": threshold, "top": top,
                                 "status": "idle"}
        log.info("joined: %s (%.0fpp, #100 = %.0f)", name, pp, threshold)
        self.mark_auto_retune_pending()

    def mark_auto_retune_pending(self) -> None:
        """Remember that roster changes should affect the NEXT queue refill."""
        if self.cfg.get("auto_difficulty", True) and not self.manual_range:
            self._auto_retune_pending = True

    def apply_pending_auto_retune(self, force: bool = False) -> None:
        """Apply a deferred auto range at a safe between-map boundary."""
        if not force and not self._auto_retune_pending:
            return

        # Consume the pending flag now. A later roster event will set it again.
        self._auto_retune_pending = False

        if not self.cfg.get("auto_difficulty", True):
            return
        if self.manual_range and not force:
            return

        active = self.gameplay_player_ids()
        pps = [
            self.players[uid]["pp"]
            for uid in active
            if self.players.get(uid, {}).get("pp")
        ]
        if not pps:
            return

        med = float(median(pps))
        half = float(self.cfg.get("auto_sr_half_width", AUTO_SR_HALF_WIDTH))
        target = target_sr_for_pp(med, half_width=half)
        lo, hi = auto_range_for_pp(med, half_width=half)

        same_range = (
            abs(lo - self.pool.sr_min) < 0.001
            and abs(hi - self.pool.sr_max) < 0.001
        )
        if same_range:
            log.debug(
                "deferred auto range unchanged at %.2f-%.2f* "
                "(median %.0fpp)", lo, hi, med)
            return

        n = self.pool.set_range(lo, hi)
        if n <= 0:
            # The requested range may simply have no farm_report maps. The
            # pool object has already restored its previous known-good state.
            # Keep this silent in lobby chat; a range shortage is an internal
            # data condition, not something players need spammed with.
            log.warning(
                "auto retune %.2f-%.2f* (target %.2f*, median %.0fpp) "
                "had no eligible maps; keeping %.2f-%.2f*",
                lo, hi, target, med,
                self.pool.sr_min, self.pool.sr_max,
            )
            return

        self.update_room_name_for_range(
            self.pool.sr_min, self.pool.sr_max)
        log.info(
            "auto retuned between maps to %.2f-%.2f* (target %.2f*) "
            "for median %.0fpp across %d player(s) (%d maps)",
            self.pool.sr_min, self.pool.sr_max,
            target, med, len(pps), n,
        )
        self.say(
            f"Pool retuned to {self.pool.sr_min:.2f}-"
            f"{self.pool.sr_max:.2f}* "
            f"(target {target:.2f}*, median {med:.0f}pp, "
            f"{len(pps)} player{'s' if len(pps) != 1 else ''}, {n} maps)"
        )

    # ---------------------------------------------------------- scoring

    def handle_match_completed(self, payload: dict) -> None:
        # The map that actually entered gameplay was frozen at MatchStarted.
        # Treat that as authoritative for both score lookup and .good/.bad.
        #
        # Do NOT prefer MatchCompleted's playlist_item_id here. In live lazer
        # rooms that event can arrive after playlist advancement, and the id in
        # the payload has proven unreliable enough to make the rating prompt
        # lag by a map.
        self.scoring_item_id = (self.playing_item_id
                                or payload.get("playlist_item_id")
                                or self.current_item_id)

        completed_map = self.playing_map

        # Fallback only if we somehow missed MatchStarted / playing_map.
        if completed_map is None and self.scoring_item_id is not None:
            completed_map = self._playlist_maps.get(int(self.scoring_item_id))
        if completed_map is None:
            completed_map = self.current_map

        self.last_played_map = completed_map
        if self.last_played_map:
            log.info("rating target set to map that entered gameplay: %s "
                     "(playing_item=%s completed_payload_item=%s)",
                     self.last_played_map.label,
                     self.playing_item_id,
                     payload.get("playlist_item_id"))

        self._rating_window_until = time.time() + self.cfg.get(
            "rating_window_seconds", 90)
        self.set_state(SCORING)
        # Scores are not queryable the instant gameplay ends; the server still
        # has to ingest submissions. Delay then retry rather than firing
        # immediately and getting an empty list.
        self._match_end_at = time.time()

    def try_score_report(self) -> bool:
        elapsed = time.time() - self._match_end_at
        if elapsed < self.cfg.get("score_delay_seconds", 4):
            return False

        data = self.api.room_item_scores(self.room_id, self.scoring_item_id)
        scores = (data or {}).get("scores") or []
        if not scores and elapsed < 20:
            return False       # keep retrying until the timeout

        bid = self.last_played_map.beatmap_id if self.last_played_map else 0

        # Silent farm-evidence collection. These records are NOT human ratings:
        # they are stored separately in ratings.db so farm-finder can use them
        # as empirical "this map produced notable top plays in a real lobby"
        # signals later.
        evidence_records: list[dict] = []
        scored_users: set[int] = set()

        for s in scores:
            uid = s.get("user_id")
            pp = s.get("pp")
            if not uid or not pp:
                continue

            uid = int(uid)

            # Count score submitters from the actual match. If we have a
            # MatchStarted participant snapshot, ignore unrelated/spectating
            # users for the evidence denominator.
            if not self.playing_player_ids or uid in self.playing_player_ids:
                scored_users.add(uid)

            info = self.players.get(uid)
            if not info or not info.get("top"):
                continue

            gain, rank, new_top = gain_from(info["top"], pp, bid)
            if rank is None:
                continue          # didn't make their top 100

            info["top"] = new_top
            info["threshold"] = new_top[-1][0] if len(new_top) >= 100 else 0.0

            acc = (s.get("accuracy") or 0) * 100

            evidence_records.append({
                "user_id": uid,
                "rank": int(rank),
                "pp": float(pp),
                "gain": float(gain),
                "accuracy": float(acc),
            })

            self.say(f"{info['name']} — {pp:.0f}pp ({acc:.2f}%) — "
                     f"new #{rank} top play, +{gain:.1f}pp profile!")

        if bid and evidence_records:
            participant_count = len(self.playing_player_ids)
            if participant_count <= 0:
                participant_count = max(len(scored_users),
                                        self.gameplay_player_count())

            try:
                self.ratings.record_performance_evidence(
                    beatmap_id=bid,
                    room_id=self.room_id,
                    playlist_item_id=self.scoring_item_id,
                    participant_count=participant_count,
                    scored_count=len(scored_users),
                    records=evidence_records,
                    notable_ratio=float(self.cfg.get(
                        "performance_evidence_ratio", 0.40)),
                    min_top20=int(self.cfg.get(
                        "performance_evidence_min_top20", 2)),
                    min_top30=int(self.cfg.get(
                        "performance_evidence_min_top30", 3)),
                )
            except Exception:
                # Evidence collection is useful but must never interrupt room
                # rotation or score reporting.
                log.exception("could not save silent performance evidence")

        return True

    # ------------------------------------------------------------- loop

    def handle_event(self, name: str, payload: dict) -> None:
        if name == "MatchStarted":
            minimum = int(self.cfg.get("min_players", 1))
            actual = self.gameplay_player_count()

            if actual < minimum:
                # A MatchStarted can race with somebody leaving during the
                # countdown. Never enter PLAYING with an undersized room.
                log.warning("MatchStarted with only %d/%d gameplay players; "
                            "aborting", actual, minimum)
                try:
                    self.hub.abort_match(self.room_id)
                except Exception as e:
                    log.warning("could not abort undersized match start: %s", e)
                self.ready.clear()
                self.set_state(IDLE)
                return

            if self.state == PLAYING:
                # Duplicate MatchStarted notifications are harmless.
                log.debug("duplicate MatchStarted ignored")
                return

            if self.state == IDLE:
                # The server is authoritative. This can happen if somebody
                # joined during a countdown and our local recovery reached
                # IDLE just before the server delivered MatchStarted.
                log.info("MatchStarted arrived while locally IDLE; accepting "
                         "server state instead of aborting the match")
            elif self.state != STARTING:
                log.warning("unexpected MatchStarted while state=%s; ignoring",
                            self.state)
                return

            self.ready.clear()
            self.playing_player_ids = set(self.gameplay_player_ids())

            payload_item_id = payload.get("playlist_item_id")
            self.playing_item_id = (
                self.starting_item_id
                or payload_item_id
                or self.current_item_id
            )
            self.playing_map = self.starting_map

            if self.playing_map is None and self.playing_item_id is not None:
                self.playing_map = self._playlist_maps.get(
                    int(self.playing_item_id))
            if self.playing_map is None:
                self.playing_map = self.current_map

            if (self.starting_item_id is not None
                    and payload_item_id is not None
                    and int(self.starting_item_id) != int(payload_item_id)):
                log.warning(
                    "MatchStarted item differs from StartMatch snapshot: "
                    "started=%s event=%s; keeping snapshotted map %s",
                    self.starting_item_id,
                    payload_item_id,
                    self.playing_map.label if self.playing_map else "?",
                )

            log.info(
                "match map frozen from StartMatch: %s (item=%s)",
                self.playing_map.label if self.playing_map else "?",
                self.playing_item_id,
            )

            self.starting_map = None
            self.starting_item_id = None
            self._match_started_at = time.time()
            self._explicit_abort_pending = False
            self.set_state(PLAYING)

        elif name == "MatchCompleted":
            self.handle_match_completed(payload)

        elif name == "MatchAborted":
            now = time.time()
            minimum = int(self.cfg.get("min_players", 1))
            actual = self.gameplay_player_count()

            stale_after_start = (
                self.state == PLAYING
                and not self._explicit_abort_pending
                and self._match_started_at > 0
                and (now - self._match_started_at) <= 3.0
                and actual >= minimum
            )

            if stale_after_start:
                log.warning(
                    "ignoring MatchAborted %.2fs after MatchStarted with "
                    "%d/%d players; treating as stale countdown event",
                    now - self._match_started_at,
                    actual,
                    minimum,
                )
                return

            was_active = self.state in (STARTING, PLAYING)
            self.ready.clear()
            self.starting_map = None
            self.starting_item_id = None
            self.playing_player_ids.clear()
            self._explicit_abort_pending = False
            self._match_started_at = 0.0
            self.set_state(IDLE)
            if was_active:
                self.say("Match aborted.")

        elif name == "UserJoined":
            uid = payload.get("user_id")
            if uid:
                # Update roster/pp data only. Do not touch an active countdown
                # or retune the pool because someone joined.
                self.on_user_joined(uid)

        elif name in ("UserLeft", "UserKicked"):
            uid = payload.get("user_id")
            if uid not in self.referee_ids:
                self.players.pop(uid, None)
                self.ready.discard(uid)
                self.mark_auto_retune_pending()

            # Safety only: if the countdown no longer has the configured
            # minimum number of players, cancel it so the lobby can recover.
            # This is unrelated to difficulty retuning.
            self.cancel_start_if_too_few_players()

        elif name == "UserStatusChanged":
            uid, status = payload.get("user_id"), payload.get("status")
            if uid and uid not in self.referee_ids:
                if uid in self.players:
                    self.players[uid]["status"] = status
                if status == "ready":
                    self.ready.add(uid)
                else:
                    self.ready.discard(uid)

                # Status changes affect the NEXT auto-difficulty calculation
                # only. Do not stop/restart a countdown here: lazer can emit
                # transient spectator-like states around joins/loading.
                self.mark_auto_retune_pending()

        elif name == "PlaylistItemChanged":
            item = payload.get("item") or payload
            self._adopt_playlist_item(item)

    def run(self) -> None:
        try:
            self.create_room()
        except Exception:
            # create_room already closed the room if it got that far.
            raise

        self._idle_since = time.time()
        try:
            self._loop()
        finally:
            self.shutdown()

    def _loop(self) -> None:
        while self.running:
            try:
                while True:
                    name, payload = self.events.get_nowait()
                    self.handle_event(name, payload)
            except queue.Empty:
                pass

            if self.channel_id:
                self.poll_chat()
                self.maybe_post_info()

            if self.state == SCORING:
                if self.try_score_report() or (time.time() - self._match_end_at) > 25:
                    self.rotate()

            elif self.state == IDLE:
                wait = self.cfg.get("auto_start_seconds", 45)
                minimum = int(self.cfg.get("min_players", 1))
                active = self.gameplay_player_ids()
                enough = len(active) >= minimum
                now = time.time()

                waited = now - self._idle_since > wait
                if enough and (waited or self.ready_threshold_met()):
                    if not waited:
                        log.info("ready threshold met (%d/%d), starting early",
                                 len(self.ready & active), len(active))
                    self.start_match()

            elif self.state == STARTING:
                minimum = int(self.cfg.get("min_players", 1))
                actual = self.gameplay_player_count()

                if actual < minimum:
                    self.cancel_start_if_too_few_players()
                else:
                    # Normally countdown completion emits MatchStarted. If the
                    # visible countdown ends but that event never arrives, try
                    # one immediate start before returning to IDLE.
                    cd = float(self.cfg.get("countdown_seconds", 15))
                    elapsed = time.time() - self._state_since

                    if not self._start_immediate_retry and elapsed > cd + 5.0:
                        log.warning("countdown finished but MatchStarted was not "
                                    "seen; retrying immediate start with %d "
                                    "gameplay player(s)", actual)
                        try:
                            self.hub.start_match(self.room_id, countdown=None)
                            self._start_immediate_retry = True
                            self._state_since = time.time()
                        except Exception as e:
                            log.warning("immediate start retry failed: %s", e)
                            self.ready.clear()
                            self.starting_map = None
                            self.starting_item_id = None
                            self.set_state(IDLE)

                    elif self._start_immediate_retry and elapsed > 10.0:
                        log.warning("immediate start produced no MatchStarted; "
                                    "reverting to IDLE")
                        self.ready.clear()
                        self.starting_map = None
                        self.starting_item_id = None
                        self.set_state(IDLE)

            elif self.state == PLAYING:
                # Safety valve: no map in the pool is longer than max_length,
                # so a match running far past that means we missed completion.
                if time.time() - self._state_since > self.pool.max_length + 180:
                    log.warning("PLAYING timed out, assuming match ended")
                    self.set_state(IDLE)

            time.sleep(0.4)

    def shutdown(self) -> None:
        """Close the room. MUST run even on Ctrl+C.

        CloseRoom only works while still joined as referee -- once the
        connection drops you are no longer in the room and the call fails
        with InvalidStateException, leaving it open and occupying one of the
        four room slots a non-bot account gets.
        """
        if not self.room_id or not self.cfg.get("close_room_on_exit", True):
            return
        try:
            self.hub.close_room(self.room_id)
            log.info("room %s closed", self.room_id)
        except Exception as e:
            log.error("could not close room %s: %s", self.room_id, e)
            log.error("close it manually with: python close_room.py %s",
                      self.room_id)
        finally:
            self.room_id = None
