"""Which machine is using which account, so two engines stop colliding.

Every lock and state file this tool owns lives under a per-machine backup root,
and the usage store's cross-host fence fields are null in practice — so two
machines running the auto-switch engine against one pool of accounts have no
way to see each other. They rank the same accounts on the same signals and
therefore pick the same winner. Measured on one such pair: both machines sat on
the same account 34.8% of the time, and the account's five-hour window burned
at roughly double the rate either engine believed it was regulating against.

A claim is the smallest thing that fixes it. Each machine publishes, after
every switch, one small file saying which account it is now using. The peer
reads it and drops that account from its own candidate list. Nothing is
negotiated, nothing is locked across machines, and a missing or stale claim
degrades to today's behaviour rather than to a deadlock.

WHAT IS AND IS NOT IN THE FILE. Identity (uuid + organization), a timestamp,
the publisher's own clock, and a busy-session count. No token, no credential,
no email is required for the decision — the email rides along for display only.
The file is meant to be readable by a peer over ssh, so it must stay boring.

TIME IS THE HARD PART. Nothing in this tool keeps a monotonic or logical clock,
so a peer's timestamp cannot be compared against the reader's own — a minute of
NTP drift would silently change who owns what. Freshness is therefore measured
entirely inside the READER's clock domain: ``write_peer_claim`` stamps
``pulledAt`` when the file lands here, and a claim is live while
``now - pulledAt`` is under the TTL. Both ends of that subtraction are this
machine's own clock, so drift cannot enter.

The publisher's own timestamps ride along and are used for exactly one thing:
``publisherNow - since`` says how long that machine had been sitting on the
account, which breaks a tie when two peers name the same one. It is NOT a
freshness measure, and it was one — with ``since`` assigned from the same
``clock()`` call as ``publisherNow`` a line earlier, every published record had
an age of ~0 and ``is_live`` was unconditionally true. A claim on the live
fleet stood for 38 hours naming an account its machine had left. A TTL that
cannot fire is worse than no TTL, because it reads like a safeguard.

A claim with no ``pulledAt`` is not live. That is the fail-open direction: it
re-admits the account rather than excluding it forever on a file nobody can
date.

TWO FAILURE RULES, and they point in opposite directions on purpose:

*Targeting fails OPEN.* If the peer is unreachable or its claim is stale, the
account is a candidate again. A claim that keeps excluding accounts after the
peer goes away would park a machine on a shrinking pool — worse than the
collision it prevents.

*Pressure fails SAFE.* The same missing knowledge makes the engine less sure
how fast a shared window is burning, so a caller reading the session count
should treat "unknown" as "more", never as zero. Under-counting means switching
later, which is the outcome that costs a session.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from claude_swap.identity import identity_key
from claude_swap.settings import atomic_write_json

CLAIM_FILENAME = "host_claim.json"
PEER_DIRNAME = "peer_claims"
SCHEMA_VERSION = 1

# How long a claim speaks for its machine. An engine republishes on every
# switch, and a machine that has not switched in this long is either idle on
# the account it holds — in which case the claim is still true — or gone. The
# ambiguity is why this is generous: excluding an account the peer still uses
# costs a little quota, while re-admitting one it is actively burning costs the
# collision this module exists to prevent. Six hours is longer than a 5h window,
# so a live peer always republishes within it.
CLAIM_TTL_S = 6 * 3600.0


def claim_path(backup_dir: Path) -> Path:
    return backup_dir / CLAIM_FILENAME


def peer_dir(backup_dir: Path) -> Path:
    return backup_dir / PEER_DIRNAME


def host_name() -> str:
    """This machine's name, as a peer would refer to it."""
    return socket.gethostname().split(".")[0] or "unknown"


@dataclass(frozen=True)
class Claim:
    """One machine's statement about the account it is using."""

    host: str
    key: str  # identity key, never a slot number
    email: str  # display only
    since: float  # publisher clock when this account became active
    published_at: float  # publisher clock at write time
    busy_sessions: int | None  # None = the publisher could not tell
    unreadable_sessions: int
    # Reader's clock when this file landed here. None for a claim that was
    # never pulled through write_peer_claim (a hand-copied file, or one from
    # before this field existed) — such a claim cannot be dated and is
    # therefore not live.
    pulled_at: float | None = None
    # The name WE pulled this file under (its stem in peer_claims/), which is
    # not the same as the name the peer reports for itself. When those two
    # disagree and the self-reported one happens to match ours, the claim is
    # silently dropped by `exclude_host` — a hostname collision that would
    # otherwise make the whole mechanism inert with no signal at all. Two
    # machines called "Mac" is not far-fetched; one of this fleet's is.
    source: str | None = None

    @property
    def standing_s(self) -> float:
        """Seconds this machine had been on the account when it published.

        The publisher's own clock on both sides of the subtraction. This is a
        tie-break between two peers naming one account, NOT a freshness
        measure — see the module docstring for what happened when it was one.
        """
        return max(0.0, self.published_at - self.since)

    def stale_s(self, now: float) -> float | None:
        """Seconds since this claim was pulled, or None if it cannot be dated."""
        if self.pulled_at is None:
            return None
        return max(0.0, now - self.pulled_at)

    def is_live(self, now: float, ttl_s: float = CLAIM_TTL_S) -> bool:
        """Whether this claim still speaks for its machine.

        ``now`` is the READER's clock, and so is ``pulled_at`` — the publisher's
        clock is deliberately absent from this comparison.
        """
        stale = self.stale_s(now)
        return stale is not None and stale < ttl_s

    @property
    def pressure(self) -> int:
        """Busy sessions to assume this peer is contributing.

        ``None`` means the publisher could not read its own sessions, and
        unreadable session files mean it could only read some. Both resolve
        upward: a caller sizing a safety margin must never be told "zero" by
        an absence of information.
        """
        # One floor, in one place: a peer that is present contributes at
        # least one session whether it reported 0, reported nothing, or could
        # not read its own. Two guards for the same floor would let a mutation
        # to either survive.
        return max(self.busy_sessions or 0, 1) + max(0, self.unreadable_sessions)


def build_claim(
    identity: Mapping,
    *,
    since: float,
    now: float,
    busy_sessions: int | None,
    unreadable_sessions: int = 0,
    host: str | None = None,
) -> dict | None:
    """The record to publish, or None when the account cannot be named."""
    key = identity_key(identity)
    if key is None:
        return None
    return {
        "schemaVersion": SCHEMA_VERSION,
        "host": host or host_name(),
        "identityKey": key,
        "email": str(identity.get("email") or ""),
        "since": float(since),
        "publisherNow": float(now),
        "busySessions": busy_sessions,
        "unreadableSessions": int(unreadable_sessions),
    }


def publish(backup_dir: Path, record: dict | None) -> bool:
    """Write this machine's claim. False when there is nothing to say.

    Never raises: publishing is a courtesy to the other machine, and a failure
    to perform it must not take down the switch that prompted it.
    """
    if not record:
        return False
    try:
        atomic_write_json(claim_path(backup_dir), record)
    except OSError:
        return False
    return True


def _parse(raw: object) -> Claim | None:
    if not isinstance(raw, dict):
        return None
    key = raw.get("identityKey")
    if not isinstance(key, str) or not key:
        return None
    try:
        since = float(raw.get("since"))
        published_at = float(raw.get("publisherNow"))
    except (TypeError, ValueError):
        return None
    busy = raw.get("busySessions")
    busy = int(busy) if isinstance(busy, (int, float)) else None
    unreadable = raw.get("unreadableSessions")
    pulled = raw.get("pulledAt")
    return Claim(
        host=str(raw.get("host") or "unknown"),
        key=key,
        email=str(raw.get("email") or ""),
        since=since,
        published_at=published_at,
        busy_sessions=busy,
        unreadable_sessions=int(unreadable) if isinstance(unreadable, (int, float)) else 0,
        # bool excluded: `True` is an int in Python and would date a claim to
        # the epoch, i.e. permanently stale rather than undatable.
        pulled_at=(
            float(pulled)
            if isinstance(pulled, (int, float)) and not isinstance(pulled, bool)
            else None
        ),
    )


def read_peers(backup_dir: Path) -> list[Claim]:
    """Every peer claim pulled onto this machine, newest write order aside.

    Reads ``<backup_dir>/peer_claims/*.json``, which something outside this
    module puts there — the engine must not shell out to ssh inside a tick, and
    this package holds locks across decisions but never across network calls.
    A malformed or partly written file is skipped rather than raising: a peer
    that publishes garbage must not stop this machine from deciding.
    """
    directory = peer_dir(backup_dir)
    claims: list[Claim] = []
    try:
        entries = sorted(directory.glob("*.json"))
    except OSError:
        return claims
    for path in entries:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        claim = _parse(raw)
        if claim is not None:
            claims.append(replace(claim, source=path.stem))
    return claims


def claimed_keys(
    claims: list[Claim],
    *,
    now: float,
    exclude_host: str | None = None,
    ttl_s: float = CLAIM_TTL_S,
) -> dict[str, Claim]:
    """``{identity key: claim}`` for every live peer claim.

    ``now`` is the reader's clock; freshness never leaves that domain.

    ``exclude_host`` drops this machine's own claim, which matters when a claim
    directory has been synced rather than pulled — a machine must never exclude
    the account it is itself using, or it would refuse to stay put.
    """
    out: dict[str, Claim] = {}
    for claim in claims:
        if exclude_host is not None and claim.host == exclude_host:
            continue
        if not claim.is_live(now, ttl_s):
            continue
        # Two peers naming one account should not happen; if it does, the one
        # that has held it longest is the more established fact.
        current = out.get(claim.key)
        if current is None or claim.standing_s > current.standing_s:
            out[claim.key] = claim
    return out


def peer_pressure(claims: Mapping[str, Claim], key: str | None) -> int:
    """Busy sessions peers are contributing to THIS account, or 0 for none.

    Scoped to one account on purpose: burn is per account, so a peer working
    hard on a different one adds nothing to the window being measured here.
    """
    if key is None:
        return 0
    claim = claims.get(key)
    return claim.pressure if claim is not None else 0


def write_peer_claim(
    backup_dir: Path, host: str, raw_text: str, *, now: float | None = None
) -> bool:
    """Store text pulled from a peer. False when it is not a usable claim.

    Validated before it lands, so a truncated scp or an error message captured
    into a file can never become a claim that excludes an account.
    """
    try:
        parsed = json.loads(raw_text)
    except (TypeError, ValueError):
        return False
    if _parse(parsed) is None:
        return False
    # Stamped here, with OUR clock, because this is the one moment a reader can
    # honestly date: the file exists on this machine now. The publisher's own
    # timestamps are not comparable against ours and are never used for
    # freshness. Overwrites any pulledAt the peer sent — a publisher does not
    # get to declare how fresh it is on our side.
    parsed["pulledAt"] = float(now if now is not None else time.time())
    safe = "".join(c for c in host if c.isalnum() or c in "-_.") or "peer"
    target = peer_dir(backup_dir) / f"{safe}.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, parsed)
        os.chmod(target, 0o600)
    except OSError:
        return False
    return True
