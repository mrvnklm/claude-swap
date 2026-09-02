"""Accounts that are going away, and the deadline that creates.

A weekly window resets forever while a subscription is live, so
``consume-first``'s "soonest weekly reset" is a *recurring* deadline: nothing
is really lost by draining it a day late. A cancelled subscription is the
other kind. Its quota expires once, on a fixed date, and whatever is left at
that moment is gone — so it outranks every weekly reset that falls after it.

That distinction is invisible to the tool: the usage API reports windows, not
subscriptions, and ``/api/oauth/profile`` keeps reporting ``active`` for a
subscription cancelled to run out its current period. Only the operator knows,
so the mark and its date are both given by hand.

TWO RULES CARRY THIS MODULE, and both exist because of a measured failure.

*Past means unknown.* A mark whose day is over reads exactly as no mark. This
mirrors ``autoswitch._seven_day_reset_ts``, whose docstring records why:
treated as a real instant, an elapsed deadline sorts the least perishable
account as the soonest and pins the fleet there. A lapsed mark did precisely
that — every candidate rejected by the strictly-sooner gate, tick after tick
of ``already-consuming-soonest`` while healthy peers idled.

*Marks are keyed by identity, never by slot.* On a fleet where two slots hold
the same account under different organizations, and where two machines number
the same account differently, a slot number names a position and not an
account. The key is ``<uuid>:<organizationUuid>`` — the pair the rest of the
tree already treats as identity — so a mark cannot be applied to whatever
moved into a slot later, and means the same account on both machines.

The record lives in the autoswitch state file, beside quarantine: already
locked against a running engine, and never exported, because a cancellation is
a fact about this operator's billing rather than about the account.

Local time, deliberately, and the only place in this package that is. The date
comes off a billing page the operator read in their own zone; resolving it in
UTC would end the mark on the wrong day for anyone west of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, tzinfo
from pathlib import Path
from typing import Callable, Iterable, Mapping

from claude_swap.locking import FileLock
from claude_swap.settings import atomic_write_json

STATE_FILENAME = "autoswitch_state.json"
LOCK_FILENAME = ".autoswitch_state.lock"
CANCELLED_KEY = "cancelled"
KEY_SEP = ":"


# -- identity ----------------------------------------------------------------


def identity_key(identity: Mapping) -> str | None:
    """``<uuid>:<organizationUuid>`` for an account, or None if either half is missing.

    Both halves are required. The uuid alone is not an identity — one account
    can hold seats in two organizations and appear as two managed slots — and
    the organization alone is not either. A slot missing either half cannot be
    marked, which is the safe refusal: a mark that cannot name its account
    would be applied by position, which is the bug this key exists to remove.
    """
    if not isinstance(identity, Mapping):
        return None
    uuid = str(identity.get("uuid") or "").strip()
    org = str(identity.get("organizationUuid") or "").strip()
    if not uuid or not org:
        return None
    return f"{uuid}{KEY_SEP}{org}"


@dataclass(frozen=True)
class Mark:
    """One cancellation record, resolved against the current roster."""

    key: str
    ends_on: date | None  # None when the stored date is unusable
    email: str  # as recorded, for display only — never for matching
    note: str
    slot: str | None  # None when no managed account carries this identity
    lapsed: bool  # the stated day is over

    @property
    def orphaned(self) -> bool:
        return self.slot is None


# -- the state file ----------------------------------------------------------


def state_path(backup_dir: Path) -> Path:
    return backup_dir / STATE_FILENAME


def _lock(backup_dir: Path) -> FileLock:
    return FileLock(backup_dir / LOCK_FILENAME)


def read_state(backup_dir: Path) -> dict:
    """The autoswitch state file, or an empty dict when absent or unreadable.

    Mirrors ``AutoSwitchEngine._read_state``: a corrupt file reads as empty
    rather than raising, because every caller here is on a path where the right
    answer to "unreadable" is "no marks", never a traceback.
    """
    try:
        raw = json.loads(state_path(backup_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def mutate_state(backup_dir: Path, mutator: Callable[[dict], None]) -> dict:
    """Read-modify-write the state file under the engine's own lock.

    Uses the same lock as ``AutoSwitchEngine._mutate_state`` so a CLI mark and
    a running engine's quarantine write cannot lose each other. Deliberately
    does NOT set ``schemaVersion``: the engine owns that key, and stamping it
    from here would claim a schema this code did not produce.
    """
    with _lock(backup_dir):
        state = read_state(backup_dir)
        mutator(state)
        atomic_write_json(state_path(backup_dir), state)
        return state


def cancellations(state: dict) -> dict:
    """The raw records in a state dict; always a dict."""
    marks = state.get(CANCELLED_KEY)
    return marks if isinstance(marks, dict) else {}


# -- the one deadline rule ---------------------------------------------------


def _day_bounds(day: date, tz: tzinfo | None) -> tuple[float, float]:
    """First and last instant of ``day``, as POSIX timestamps.

    ``tz=None`` leaves the datetimes naive, so ``.timestamp()`` resolves them
    in the system local zone — the zone the date was read in. ``fold=0`` is
    explicit because a naive timestamp is ambiguous across a DST fold and the
    default is not stated in the call.
    """
    first = datetime.combine(day, time.min, tzinfo=tz).replace(fold=0)
    last = datetime.combine(day, time.max, tzinfo=tz).replace(fold=0)
    return first.timestamp(), last.timestamp()


def _stored_date(record: object) -> date | None:
    if not isinstance(record, dict):
        return None
    raw = record.get("endsOn")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def ends_ts(record: object, now: float, *, tz: tzinfo | None = None) -> float | None:
    """When this mark's quota expires, or None once it no longer says anything.

    Returns the FIRST instant of the stated local day while the mark is live,
    and None once ``now`` is past that day's last instant.

    Two deliberate choices, each in the direction that cannot lose quota:

    *Past is None*, mirroring ``_seven_day_reset_ts``. An elapsed deadline is
    not evidence about perishability, and ranked as one it pins the fleet.

    *The floor, not the ceiling.* Ranking from the day's first instant can only
    make the account look more urgent than it is; ranking from its last could
    place the deadline hours after the real period boundary, which is the
    direction that strands quota. The cost of the floor is a sub-day ordering
    difference between two already-perishable accounts.
    """
    day = _stored_date(record)
    if day is None:
        return None
    first, last = _day_bounds(day, tz)
    return None if now > last else first


def merge_deadline(
    weekly_reset_ts: float | None, ends_ts_value: float | None
) -> float | None:
    """``min`` of a weekly reset and a cancellation end, either of them absent.

    Kept separate so the ranking can call it with a value resolved once per
    tick: ``_rank_candidates`` is pure and is run twice by the consume-first
    two-phase commit, so it must not read state.
    """
    if ends_ts_value is None:
        return weekly_reset_ts
    if weekly_reset_ts is None:
        return ends_ts_value
    return min(ends_ts_value, weekly_reset_ts)


# -- the one reader ----------------------------------------------------------


def resolve_marks(
    state: dict,
    identities: Mapping[str, Mapping],
    now: float,
    *,
    tz: tzinfo | None = None,
) -> list[Mark]:
    """Every stored mark, resolved against the current roster.

    ``identities`` maps slot number to that slot's identity dict. A record
    whose key matches no current identity comes back with ``slot=None`` rather
    than being hidden: an orphan is exactly what the operator needs to see and
    remove, and hiding it is how the previous shape made one unremovable.

    This is the ONLY place a stored record becomes a decision. The engine, the
    listing and the delete path all read through it, so they cannot disagree
    about what a mark means.
    """
    by_key: dict[str, str] = {}
    for slot, identity in identities.items():
        key = identity_key(identity)
        if key is not None:
            by_key[key] = str(slot)

    marks: list[Mark] = []
    for key, record in cancellations(state).items():
        if not isinstance(record, dict):
            record = {}
        marks.append(
            Mark(
                key=str(key),
                ends_on=_stored_date(record),
                email=str(record.get("email") or ""),
                note=str(record.get("note") or ""),
                slot=by_key.get(str(key)),
                lapsed=ends_ts(record, now, tz=tz) is None
                and _stored_date(record) is not None,
            )
        )
    marks.sort(key=lambda m: (m.slot is None, int(m.slot) if m.slot else 0, m.key))
    return marks


def deadlines_by_slot(
    state: dict,
    identities: Mapping[str, Mapping],
    now: float,
    *,
    tz: tzinfo | None = None,
) -> dict[str, float]:
    """``{slot: expiry timestamp}`` for every live, resolvable mark.

    Lapsed marks and orphans are absent, so a caller that merges this into a
    deadline can never rank on one.
    """
    out: dict[str, float] = {}
    for key, record in cancellations(state).items():
        ts = ends_ts(record, now, tz=tz)
        if ts is None:
            continue
        for slot, identity in identities.items():
            if identity_key(identity) == str(key):
                out[str(slot)] = ts
                break
    return out


# -- writers -----------------------------------------------------------------


def set_cancelled(
    backup_dir: Path, key: str, ends_on: date, *, email: str = "", note: str = ""
) -> dict:
    """Record that this identity's quota ends at the close of ``ends_on``."""

    def add(state: dict) -> None:
        record: dict = {"endsOn": ends_on.isoformat()}
        if email:
            record["email"] = email
        if note:
            record["note"] = note
        # The same non-dict guard the readers carry: a hand-edited or corrupt
        # container must be replaced, not indexed into.
        if not isinstance(state.get(CANCELLED_KEY), dict):
            state[CANCELLED_KEY] = {}
        state[CANCELLED_KEY][key] = record

    return mutate_state(backup_dir, add)


def clear_cancelled(backup_dir: Path, keys: Iterable[str]) -> int:
    """Drop the named marks. Returns how many were actually removed."""
    wanted = {str(k) for k in keys}
    removed = 0

    def drop(state: dict) -> None:
        nonlocal removed
        marks = state.get(CANCELLED_KEY)
        if not isinstance(marks, dict):
            return
        for key in list(marks):
            if str(key) in wanted:
                del marks[key]
                removed += 1
        if not marks:
            state.pop(CANCELLED_KEY, None)

    if wanted:
        mutate_state(backup_dir, drop)
    return removed
