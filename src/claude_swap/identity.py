"""One way to name an account, shared by everything that has to.

A slot number names a *position*, not an account: two machines number the same
account differently, and ``cswap move`` renumbers it under you. An email is not
an account either — one login can hold seats in two organizations and show up
as two managed slots with one email and one account uuid.

So the name is the pair, and it lives here rather than in whichever module
happened to need it first: a second definition is how two readers come to
disagree about which account a record refers to.
"""

from __future__ import annotations

from typing import Mapping

KEY_SEP = ":"


def identity_key(identity: Mapping | None) -> str | None:
    """``<uuid>:<organizationUuid>``, or None when either half is missing.

    Both halves are required. The uuid alone is not an identity — one account
    with seats in two organizations yields two managed slots sharing it — and
    the organization alone is not either. A slot missing either half cannot be
    named, which is the safe refusal: a record that cannot name its account
    would end up applied by position.
    """
    if not isinstance(identity, Mapping):
        return None
    uuid = str(identity.get("uuid") or "").strip()
    org = str(identity.get("organizationUuid") or "").strip()
    if not uuid or not org:
        return None
    return f"{uuid}{KEY_SEP}{org}"


def slots_by_key(identities: Mapping[str, Mapping]) -> dict[str, str]:
    """``{identity key: slot}``, lowest slot wins a duplicate.

    Two slots can end up carrying the same identity after a botched add. Either
    answer is arbitrary, so the tie is broken deterministically rather than by
    dict order — which is how two independent resolvers once came to disagree.
    """
    out: dict[str, str] = {}
    for slot in sorted(
        identities, key=lambda n: (not str(n).isdigit(), str(n).zfill(9))
    ):
        key = identity_key(identities[slot])
        if key is not None and key not in out:
            out[key] = str(slot)
    return out
