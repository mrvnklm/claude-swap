"""Unit tests for the offline subscription helper."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

from claude_swap import subscription


def _utc(year: int, month: int, day: int, hour: int = 12) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _config(**oauth_account) -> str:
    return json.dumps({"oauthAccount": oauth_account, "unrelated": {"a": 1}})


STRIPE = {
    "billingType": "stripe_subscription",
    "organizationRateLimitTier": "default_claude_max_20x",
    "organizationType": "claude_max",
}


class TestNextPeriodStart:
    def test_anniversary_later_this_month(self):
        created = _utc(2025, 12, 22)
        assert subscription.next_period_start(created, _utc(2026, 9, 2)) == _utc(
            2026, 9, 22
        )

    def test_anniversary_already_passed_rolls_to_next_month(self):
        created = _utc(2026, 2, 7)
        assert subscription.next_period_start(created, _utc(2026, 9, 20)) == _utc(
            2026, 9, 7
        ).replace(month=10)

    def test_anniversary_today_counts_as_next(self):
        # The boundary is >=, not >: on the anniversary itself the period that
        # is starting is the answer, not the one a month later.
        created = _utc(2026, 3, 11, hour=8)
        now = _utc(2026, 9, 11, hour=8)
        assert subscription.next_period_start(created, now) == now

    def test_a_moment_after_the_anniversary_rolls_forward(self):
        created = _utc(2026, 3, 11, hour=8)
        now = _utc(2026, 9, 11, hour=9)
        assert subscription.next_period_start(created, now) == _utc(2026, 10, 11, hour=8)

    def test_december_rolls_the_year(self):
        created = _utc(2025, 6, 15)
        assert subscription.next_period_start(created, _utc(2026, 12, 20)) == _utc(
            2027, 1, 15
        )

    @pytest.mark.parametrize(
        "now, expected_day",
        [(_utc(2026, 2, 1), 28), (_utc(2026, 4, 1), 30), (_utc(2026, 5, 1), 31)],
    )
    def test_day_31_clamps_to_the_month_length(self, now, expected_day):
        # Without the clamp this raises ValueError on any short month, which on
        # a display path would take out the whole listing.
        created = _utc(2025, 1, 31)
        assert subscription.next_period_start(created, now).day == expected_day

    def test_february_29_in_a_leap_year(self):
        created = _utc(2025, 1, 31)
        assert subscription.next_period_start(created, _utc(2028, 2, 1)).day == 29


class TestSubscriptionFields:
    def test_reads_the_stored_facts(self):
        fields = subscription.subscription_fields(
            _config(**STRIPE, subscriptionCreatedAt="2025-12-22T07:01:44.781805Z"),
            now=_utc(2026, 9, 2),
        )
        assert fields == {
            "billingType": "stripe_subscription",
            "rateLimitTier": "default_claude_max_20x",
            "organizationType": "claude_max",
            "createdAt": "2025-12-22T07:01:44.781805Z",
            "nextPeriodStart": "2026-09-22",
        }

    def test_zulu_and_offset_timestamps_agree(self):
        zulu = subscription.subscription_fields(
            _config(**STRIPE, subscriptionCreatedAt="2026-02-11T09:00:00Z"),
            now=_utc(2026, 9, 2),
        )
        offset = subscription.subscription_fields(
            _config(**STRIPE, subscriptionCreatedAt="2026-02-11T10:00:00+01:00"),
            now=_utc(2026, 9, 2),
        )
        assert zulu["nextPeriodStart"] == offset["nextPeriodStart"] == "2026-09-11"

    @pytest.mark.skipif(
        not hasattr(time, "tzset"), reason="TZ cannot be changed at runtime on Windows"
    )
    def test_a_naive_timestamp_is_read_as_utc_not_as_local_time(self, monkeypatch):
        # The tzinfo guard only bites on a machine that is not on UTC, so the
        # test has to supply one: read as America/New_York (-5), a naive
        # 23:30 lands on the NEXT day in UTC and the derived date moves with
        # it. Pinning TZ is what makes this fail when the guard is deleted,
        # rather than passing everywhere by accident.
        monkeypatch.setenv("TZ", "America/New_York")
        time.tzset()
        try:
            fields = subscription.subscription_fields(
                _config(**STRIPE, subscriptionCreatedAt="2026-02-11T23:30:00"),
                now=_utc(2026, 9, 2),
            )
        finally:
            monkeypatch.undo()
            time.tzset()
        assert fields["nextPeriodStart"] == "2026-09-11"

    def test_no_derived_date_for_a_non_monthly_billing_type(self):
        # The derivation assumes a monthly cadence. Anything else must report
        # the raw facts and stay silent about the period, never guess.
        fields = subscription.subscription_fields(
            _config(
                billingType="invoiced",
                subscriptionCreatedAt="2025-12-22T07:01:44Z",
            ),
            now=_utc(2026, 9, 2),
        )
        assert fields["billingType"] == "invoiced"
        assert fields["createdAt"] == "2025-12-22T07:01:44Z"
        assert "nextPeriodStart" not in fields

    def test_no_derived_date_when_billing_type_is_absent(self):
        fields = subscription.subscription_fields(
            _config(subscriptionCreatedAt="2025-12-22T07:01:44Z"), now=_utc(2026, 9, 2)
        )
        assert "nextPeriodStart" not in fields

    def test_unparseable_creation_date_yields_neither_key(self):
        fields = subscription.subscription_fields(
            _config(**STRIPE, subscriptionCreatedAt="not-a-date"), now=_utc(2026, 9, 2)
        )
        assert "createdAt" not in fields
        assert "nextPeriodStart" not in fields
        assert fields["billingType"] == "stripe_subscription"

    def test_trial_end_is_carried_when_present(self):
        fields = subscription.subscription_fields(
            _config(**STRIPE, claudeCodeTrialEndsAt="2026-09-30T00:00:00Z"),
            now=_utc(2026, 9, 2),
        )
        assert fields["trialEndsAt"] == "2026-09-30T00:00:00Z"

    def test_null_trial_end_is_omitted_rather_than_carried_as_none(self):
        fields = subscription.subscription_fields(
            _config(**STRIPE, claudeCodeTrialEndsAt=None), now=_utc(2026, 9, 2)
        )
        assert "trialEndsAt" not in fields

    @pytest.mark.parametrize(
        "text",
        ["", "not json", "[]", '"a string"', "null", json.dumps({"oauthAccount": None})],
    )
    def test_unusable_config_is_none_not_an_exception(self, text):
        # This runs on a display path: a malformed or empty backup must degrade
        # to "no information", never take out the listing.
        assert subscription.subscription_fields(text, now=_utc(2026, 9, 2)) is None

    def test_config_without_any_known_field_is_none(self):
        assert (
            subscription.subscription_fields(
                _config(emailAddress="a@b.c"), now=_utc(2026, 9, 2)
            )
            is None
        )

    def test_defaults_to_the_wall_clock_when_now_is_omitted(self):
        fields = subscription.subscription_fields(
            _config(**STRIPE, subscriptionCreatedAt="2026-02-11T09:00:00Z")
        )
        assert fields["nextPeriodStart"] >= datetime.now(timezone.utc).date().isoformat()
