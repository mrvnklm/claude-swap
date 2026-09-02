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
NTP drift would silently change who owns what. Every claim therefore carries
``publisherNow``, the publisher's clock at the moment it wrote, and freshness is
judged as ``publisherNow - since`` inside one clock domain. The reader's clock
is used for exactly one thing: how long ago it *pulled* the file, which is its
own measurement and not the peer's.

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
from dataclasses import dataclass
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

    @property
    def age_s(self) -> float:
        """Seconds the claim had been standing when it was written.

        Measured inside the publisher's own clock domain — the reader's clock
        never enters, because there is no shared time base to compare against.
        """
        return max(0.0, self.published_at - self.since)

    def is_live(self, ttl_s: float = CLAIM_TTL_S) -> bool:
        return self.age_s < ttl_s

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
    return Claim(
        host=str(raw.get("host") or "unknown"),
        key=key,
        email=str(raw.get("email") or ""),
        since=since,
        published_at=published_at,
        busy_sessions=busy,
        unreadable_sessions=int(unreadable) if isinstance(unreadable, (int, float)) else 0,
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
            claims.append(claim)
    return claims


def claimed_keys(
    claims: list[Claim], *, exclude_host: str | None = None, ttl_s: float = CLAIM_TTL_S
) -> dict[str, Claim]:
    """``{identity key: claim}`` for every live peer claim.

    ``exclude_host`` drops this machine's own claim, which matters when a claim
    directory has been synced rather than pulled — a machine must never exclude
    the account it is itself using, or it would refuse to stay put.
    """
    out: dict[str, Claim] = {}
    for claim in claims:
        if exclude_host is not None and claim.host == exclude_host:
            continue
        if not claim.is_live(ttl_s):
            continue
        # Two peers naming one account should not happen; if it does, the one
        # that has held it longest is the more established fact.
        current = out.get(claim.key)
        if current is None or claim.age_s > current.age_s:
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


def write_peer_claim(backup_dir: Path, host: str, raw_text: str) -> bool:
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
    safe = "".join(c for c in host if c.isalnum() or c in "-_.") or "peer"
    target = peer_dir(backup_dir) / f"{safe}.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, parsed)
        os.chmod(target, 0o600)
    except OSError:
        return False
    return True
