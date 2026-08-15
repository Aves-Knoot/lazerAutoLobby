"""Logging setup.

Two independent outputs:
  - console: whatever -v controls, same as before
  - file (logs/bot.log): always captures our own modules at DEBUG, so a
    problem is fully logged even if you didn't think to pass -v beforehand

Both suppress third-party debug noise (httpx/httpcore/signalrcore) unless
--debug-libs is explicitly passed. That noise is what made the last few
terminal pastes mostly unreadable -- hundreds of lines of TCP/TLS/ping
handshake detail surrounding the one line that actually explained the error.

Rotates at 2MB x 3 files so old sessions don't accumulate forever, but a
single problematic run is very unlikely to lose its own history mid-session.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

OUR_LOGGERS = ("run", "auth", "api", "hub", "bot", "pool", "ratings", "fake", "close", "rooms")
NOISY_LIBS = ("httpx", "httpcore", "SignalRCoreClient", "urllib3")


def configure(verbose: bool = False, debug_libs: bool = False,
             log_dir: str = "logs") -> Path:
    """Set up console + file logging. Returns the log file path."""
    Path(log_dir).mkdir(exist_ok=True)
    log_path = Path(log_dir) / "bot.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(name)-8s %(levelname)-7s %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotates so a long-running bot doesn't grow an unbounded log file, but
    # 2MB is generous enough that a single problem session stays intact.
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if not debug_libs:
        for name in NOISY_LIBS:
            logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger("logsetup").info(
        "logging to %s (console=%s, libs=%s)", log_path.resolve(),
        "DEBUG" if verbose else "INFO", "shown" if debug_libs else "suppressed")
    return log_path


# ------------------------------------------------------- error extraction

def extract_problems(log_path: str = "logs/bot.log", context: int = 3,
                     max_chars: int = 6000) -> str:
    """Pull WARNING/ERROR lines and any traceback blocks out of the log.

    This is the thing to run before pasting into a chat -- it turns a
    multi-thousand-line session log into the handful of lines that actually
    explain what went wrong, with a little surrounding context.
    """
    p = Path(log_path)
    if not p.exists():
        return f"No log file at {p.resolve()} yet -- run the bot first."

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    flagged: set[int] = set()

    for i, line in enumerate(lines):
        if " ERROR " in line or " WARNING " in line or line.startswith("Traceback"):
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                flagged.add(j)
        # Tracebacks are multi-line and unindented lines end them; grab the
        # whole block once one starts.
        if line.startswith("Traceback (most recent call last):"):
            j = i
            while j < len(lines) and (lines[j].startswith((" ", "Traceback"))
                                      or lines[j].strip().endswith("Error")
                                      or ":" in lines[j][:40]):
                flagged.add(j)
                j += 1
                if j - i > 40:   # safety cap on runaway tracebacks
                    break

    if not flagged:
        return (f"No WARNING/ERROR lines found in {p.resolve()} "
                f"({len(lines)} lines total). If something went wrong, "
                f"paste the last ~30 lines of that file directly instead.")

    ordered = sorted(flagged)
    out, last = [], -2
    for i in ordered:
        if i != last + 1:
            out.append("...")
        out.append(lines[i])
        last = i

    text = "\n".join(out)
    if len(text) > max_chars:
        text = text[-max_chars:]
        text = "...(truncated)...\n" + text[text.index("\n") + 1:]
    return text
