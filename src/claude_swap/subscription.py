"""Subscription facts an account already carries, read offline.

Every stored account config holds an ``oauthAccount`` object, and Claude Code
puts more in it than cswap has ever read: the billing model, the rate-limit
tier, and — the useful one — when the subscription was created. For a
``stripe_subscription`` that creation day is the monthly anniversary, which is
the day the current period ends and a new one begins.

That matters for anyone running several accounts. A weekly window resets
forever on a live subscription, so "resets soonest" is a *recurring* deadline;
the period boundary is the one date after which an account may stop existing.
Someone who has cancelled a subscription has a fixed amount of quota that
expires for good on that day, and nothing in the tool can see it today.

Everything here is a pure function over the stored config text — no network,
no request budget, no new settings. The live ``subscription_status`` is a
separate question that only ``/api/oauth/profile`` can answer; this module
deliberately reports only what is already on disk, so it costs nothing to call
for every account on every list.

``next_period_start`` is a *derivation*, not a fact the API sent: it assumes
the monthly cadence that ``billingType == "stripe_subscription"`` implies, and
is omitted for every other billing type rather than guessed.
"""

from __future__ import annotations

import calendar
import json
from datetime import datetime, timezone

# Billing types whose period is a calendar month anchored on the creation day.
# Anything else (annual plans, invoiced orgs, API-key accounts) gets no
# derived date at all — a wrong deadline is worse than no deadline, because
# the whole point of the field is to be trusted as one.
_MONTHLY_BILLING_TYPES = frozenset({"stripe_subscription"})


def _parse_iso(value: object) -> datetime | None:
    """A UTC datetime from an ISO string, or None if absent/unparseable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def next_period_start(created: datetime, now: datetime) -> datetime:
    """The next monthly anniversary of ``created`` at or after ``now``.

    Clamps to the last day of the target month, so a subscription created on
    the 31st renews on the 30th in November and the 28th in February rather
    than raising or skipping a month. Time of day is carried over from
    ``created``; the exact hour is Stripe's business, and callers that only
    need the date should take ``.date()``.
    """
    year, month = now.year, now.month
    for _ in range(2):
        day = min(created.day, calendar.monthrange(year, month)[1])
        candidate = created.replace(year=year, month=month, day=day)
        if candidate >= now:
            return candidate
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    # Unreachable: two iterations always straddle ``now``. Kept as a total
    # function rather than an assert so a display path can never raise.
    return created.replace(
        year=year, month=month, day=min(created.day, calendar.monthrange(year, month)[1])
    )


def subscription_fields(config_text: str, now: datetime | None = None) -> dict | None:
    """Subscription facts from a stored account config, or None.

    Returns only the keys the config actually carried, so a caller can tell
    "absent" from "empty" without a sentinel. None means the config held no
    ``oauthAccount`` (or is not readable as JSON) — never an error: this runs
    on a display path and a malformed backup must not break a listing.
    """
    try:
        config = json.loads(config_text)
    except (TypeError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    oauth_account = config.get("oauthAccount")
    if not isinstance(oauth_account, dict):
        return None

    fields: dict = {}
    for source, key in (
        ("billingType", "billingType"),
        ("organizationRateLimitTier", "rateLimitTier"),
        ("organizationType", "organizationType"),
    ):
        value = oauth_account.get(source)
        if isinstance(value, str) and value:
            fields[key] = value

    created_raw = oauth_account.get("subscriptionCreatedAt")
    created = _parse_iso(created_raw)
    if created is not None:
        fields["createdAt"] = created_raw
        if fields.get("billingType") in _MONTHLY_BILLING_TYPES:
            moment = now or datetime.now(timezone.utc)
            fields["nextPeriodStart"] = (
                next_period_start(created, moment).date().isoformat()
            )

    trial_ends = oauth_account.get("claudeCodeTrialEndsAt")
    if isinstance(trial_ends, str) and trial_ends:
        fields["trialEndsAt"] = trial_ends

    return fields or None
