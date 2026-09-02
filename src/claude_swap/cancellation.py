"""Accounts that are going away, and the deadline that creates.

A weekly window resets forever while a subscription is live, so
``consume-first``'s "soonest weekly reset" is a *recurring* deadline: nothing
is really lost by draining it a day late. A cancelled subscription is the
other kind. Its quota expires once, on a fixed date, and whatever is left at
that moment is gone — so it outranks every weekly reset that falls after it.

That distinction is invisible to the tool: the usage API reports windows, not
subscriptions, and ``/api/oauth/profile`` keeps reporting ``active`` for a
subscription cancelled to run out its current period. Only the operator knows.
So the record is a manual mark, and everything else is derived: the end date
defaults to the account's own monthly anniversary (``subscription.py``), which
is the day the current paid period closes.

Marks live in the autoswitch state file, alongside quarantine and cooldown,
for two reasons. It is already lock-protected against a second engine, and it
is never exported — a cancellation is a fact about *this* operator's billing,
not about the account, and copying it to another machine via
``cswap export`` would be wrong. Keyed by slot with the email stored beside
it, matching the quarantine record, so a slot reassigned by ``cswap move``
drops its stale mark instead of silently applying it to a different account.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Callable

from claude_swap.locking import FileLock
from claude_swap.settings import atomic_write_json

STATE_FILENAME = "autoswitch_state.json"
LOCK_FILENAME = ".autoswitch_state.lock"
CANCELLED_KEY = "cancelled"


def state_path(backup_dir: Path) -> Path:
    return backup_dir / STATE_FILENAME


def _lock(backup_dir: Path) -> FileLock:
    return FileLock(backup_dir / LOCK_FILENAME)


def read_state(backup_dir: Path) -> dict:
    """The autoswitch state file, or an empty dict when absent or unreadable.

    Mirrors ``AutoSwitchEngine._read_state``: a corrupt file reads as empty
    rather than raising, because every caller here is on a path where the
    right answer to "unreadable" is "no marks", never a traceback.
    """
    try:
        raw = json.loads(state_path(backup_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def mutate_state(backup_dir: Path, mutator: Callable[[dict], None]) -> dict:
    """Read-modify-write the state file under the engine's own lock.

    Uses the same lock file as ``AutoSwitchEngine._mutate_state`` so a CLI
    mark and a running engine's quarantine write cannot lose each other.
    Deliberately does NOT set ``schemaVersion``: the engine owns that key and
    stamping it from here would let a CLI write claim a schema the engine did
    not produce.
    """
    with _lock(backup_dir):
        state = read_state(backup_dir)
        mutator(state)
        atomic_write_json(state_path(backup_dir), state)
        return state


def cancellations(state: dict) -> dict:
    """The cancellation records in a state dict; always a dict."""
    marks = state.get(CANCELLED_KEY)
    return marks if isinstance(marks, dict) else {}


def record_for(state: dict, number: str, email: str) -> dict | None:
    """This slot's cancellation record, if it still names this account.

    A mark whose stored email no longer matches the slot is ignored: the slot
    was reassigned (``cswap move``/``swap``/``add --slot``) and the mark
    belongs to an account that is no longer there. Returning None rather than
    deleting keeps this function pure — the stale record is cleaned up by the
    next write, or harmlessly ignored forever.
    """
    record = cancellations(state).get(str(number))
    if not isinstance(record, dict):
        return None
    stored_email = record.get("email")
    if isinstance(stored_email, str) and stored_email and stored_email != email:
        return None
    return record


def ends_at_ts(state: dict, number: str, email: str) -> float | None:
    """POSIX timestamp at which this account's quota expires for good.

    None when the slot carries no usable mark. The date is stored as a plain
    ``YYYY-MM-DD`` and resolved to the END of that day in UTC: the period runs
    through its last day, so treating the date as midnight would give up the
    final day's quota — the exact thing the mark exists to avoid.
    """
    record = record_for(state, number, email)
    if record is None:
        return None
    raw = record.get("endsOn")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = date.fromisoformat(raw)
    except ValueError:
        return None
    return datetime.combine(parsed, time.max, tzinfo=timezone.utc).timestamp()


def merge_deadline(
    weekly_reset_ts: float | None, ends_ts: float | None
) -> float | None:
    """``min`` of a weekly reset and a cancellation end, either of them absent.

    Split out so the ranking can call it with an end timestamp it resolved
    once per tick, keeping ``_rank_candidates`` pure and free of state reads —
    it is run twice per tick by the consume-first two-phase commit.
    """
    if ends_ts is None:
        return weekly_reset_ts
    if weekly_reset_ts is None:
        return ends_ts
    return min(ends_ts, weekly_reset_ts)


def drain_deadline_ts(
    state: dict, number: str, email: str, weekly_reset_ts: float | None
) -> float | None:
    """The deadline ``consume-first`` should rank this account by.

    ``min`` of the weekly reset and the cancellation end, because whichever
    comes first is what makes the quota perishable. A cancelled account whose
    subscription ends before its weekly window rolls therefore outranks every
    account whose only deadline is a reset — which is the whole point.

    A cancellation end already in the past is still returned. It sorts first,
    which is correct: either the account still works and its remaining quota
    is maximally urgent, or it does not and the ordinary quarantine and
    at-limit paths deal with it. Suppressing it here would strand quota on a
    guess about a date the operator entered.
    """
    return merge_deadline(weekly_reset_ts, ends_at_ts(state, number, email))


def set_cancelled(
    backup_dir: Path, number: str, email: str, ends_on: date, note: str = ""
) -> dict:
    """Mark a slot as cancelled, ending at the close of ``ends_on``."""

    def add(state: dict) -> None:
        record = {"email": email, "endsOn": ends_on.isoformat()}
        if note:
            record["note"] = note
        state.setdefault(CANCELLED_KEY, {})[str(number)] = record

    return mutate_state(backup_dir, add)


def clear_cancelled(backup_dir: Path, number: str) -> bool:
    """Drop a slot's mark. True when one was actually removed."""
    removed = False

    def drop(state: dict) -> None:
        nonlocal removed
        marks = state.get(CANCELLED_KEY)
        if isinstance(marks, dict) and str(number) in marks:
            del marks[str(number)]
            removed = True
            if not marks:
                del state[CANCELLED_KEY]

    mutate_state(backup_dir, drop)
    return removed
