"""Proof that an auto-switch engine is alive, and how to read it back.

Nothing in the tree recorded that a tick happened. ``autoswitch_state.json``
is written only when a switch commits, so silence there is ambiguous by
construction: a healthy engine on a quiet fleet and a process that died
hours ago leave byte-identical files. That ambiguity cost a 36-hour outage
during which every surface still showed fresh usage — because any surface
refreshes the shared usage store — while nothing was deciding anything.

The rules that make this readable:

* **Every beat carries its own deadline.** A beat is fresh until
  ``next_tick_by``, which the writer computes from the delay it is about to
  sleep. A reader never guesses an interval, and never treats "the process
  is alive" as evidence that the loop is turning — a live PID may only
  DEMOTE a verdict, never promote a stale beat to a healthy one.
* **One beat per loop iteration.** ``cswap auto --once`` writes nothing: a
  cron wrapper firing every few minutes would otherwise keep overwriting a
  dead long-running engine's file and make it look alive forever.
* **Silence is armed by intent, not by the file.** An engine that never
  started writes no beat at all, so a reader keyed on the file's existence
  would stay quiet in exactly the case worth shouting about. The caller
  decides whether a beat was expected (``autoswitch.background``); this
  module only reports what it finds.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path

from claude_swap.settings import atomic_write_json

HEARTBEAT_FILENAME = "autoswitch_heartbeat.json"

# Added to the writer's own delay before a beat is called stale. Covers a tick
# that runs long (parallel usage fetches have a ~10s timeout each) and clock
# jitter, without letting a dead engine pass for a whole extra interval.
GRACE_S = 90.0

SCHEMA_VERSION = 1


def heartbeat_path(backup_dir: Path) -> Path:
    return backup_dir / HEARTBEAT_FILENAME


@dataclass(frozen=True)
class Beat:
    """One recorded loop iteration, as read back from disk."""

    tick_at: float
    next_tick_by: float
    outcome: str | None
    host: str
    pid: int
    schema_version: int = SCHEMA_VERSION

    def age_s(self, now: float) -> float:
        # Clamped: a beat written by a clock ahead of ours is odd, not
        # negative-aged, and "-4s ago" in a status line reads as a bug.
        return max(0.0, now - self.tick_at)

    def is_fresh(self, now: float) -> bool:
        return now <= self.next_tick_by

    def is_local(self) -> bool:
        return self.host == socket.gethostname()

    def pid_is_live(self) -> bool:
        """Whether our own machine still has that process.

        Only ever used to DEMOTE: a fresh beat whose process is gone is dead
        now, whatever its deadline says. The converse is not true — PIDs are
        reused, so a live PID cannot make a stale beat healthy.
        """
        if not self.is_local():
            return False
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True  # exists but not ours to signal
        return True


def write_beat(
    backup_dir: Path,
    *,
    now: float,
    next_delay: float,
    outcome: str | None,
) -> None:
    """Record one loop iteration. Never raises.

    Deliberately outside ``.autoswitch_state.lock``: ``_perform`` holds that
    lock across a whole ``switch_to()``, and FileLock gives up after 10s, so a
    per-tick write inside it would be a new way for the engine to fail. Two
    hosts on one machine share this file, last writer wins, and ``os.replace``
    is atomic — the cost of that is a slightly wrong PID, not corruption.
    """
    record = {
        "schemaVersion": SCHEMA_VERSION,
        "tickAt": now,
        "nextTickBy": now + max(0.0, next_delay) + GRACE_S,
        "outcome": outcome,
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    try:
        atomic_write_json(heartbeat_path(backup_dir), record)
    except Exception:
        # A status file that cannot be written must not take the engine with
        # it. The reader treats a missing beat as "no engine", which is the
        # safe reading of a machine that cannot write to its own backup dir.
        pass


def read_beat(backup_dir: Path) -> Beat | None:
    """The last recorded beat, or None if there is none to read."""
    try:
        raw = json.loads(heartbeat_path(backup_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        tick_at = float(raw["tickAt"])
        next_tick_by = float(raw["nextTickBy"])
        pid = int(raw["pid"])
    except (KeyError, TypeError, ValueError):
        return None
    outcome = raw.get("outcome")
    return Beat(
        tick_at=tick_at,
        next_tick_by=next_tick_by,
        outcome=str(outcome) if outcome is not None else None,
        host=str(raw.get("host") or ""),
        pid=pid,
        schema_version=int(raw.get("schemaVersion") or 0),
    )


def describe(backup_dir: Path, *, expected: bool, now: float) -> str | None:
    """One line about engine liveness, or None when there is nothing to say.

    ``expected`` is the operator's intent (``autoswitch.background``). With it
    false this returns None for every input: someone who deliberately runs no
    engine must never be nagged, and that is the shipped default.
    """
    if not expected:
        return None

    beat = read_beat(backup_dir)
    if beat is None:
        return "auto-switch is enabled but no engine has ever reported in"
    if not beat.is_fresh(now):
        return (
            "auto-switch is enabled but its engine last ticked "
            f"{_ago(beat.age_s(now))} ago"
        )
    if beat.is_local() and not beat.pid_is_live():
        return (
            "auto-switch is enabled but the engine that reported "
            f"{_ago(beat.age_s(now))} ago is gone"
        )
    return None


def _ago(seconds: float) -> str:
    """Compact duration: "45s", "12m", "2h 13m", "1d 12h".

    Deliberately the same shape as ``tui.data.format_duration`` — a liveness
    line and the countdown next to it must not describe the same span two
    ways. Not imported from there because that is the UI layer and this is
    read by the engine; ``test_heartbeat`` asserts the two agree instead.
    """
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        h, m = divmod(s // 60, 60)
        return f"{h}h {m}m" if m else f"{h}h"
    d, h = divmod(s // 3600, 24)
    return f"{d}d {h}h" if h else f"{d}d"
