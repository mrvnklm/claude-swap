"""Unit tests for cross-machine account claims."""

from __future__ import annotations

import json

import pytest

from claude_swap import host_claim
from claude_swap.host_claim import CLAIM_TTL_S, Claim

UUID_A = "6a64d5cc-9af1-4bca-aaa4-7409aad57394"
ORG_1 = "af232ed4-8c8a-4a46-b942-f5ff6528c80c"
ORG_6 = "bed07a55-be7d-469a-b348-38ea4b909c4f"
KEY_1 = f"{UUID_A}:{ORG_1}"
KEY_6 = f"{UUID_A}:{ORG_6}"

NOW = 1_800_000_000.0


def _ident(org: str = ORG_1, uuid: str = UUID_A, email: str = "a@b.c") -> dict:
    return {"uuid": uuid, "organizationUuid": org, "email": email}


def _claim(key: str = KEY_1, *, host: str = "studio", standing: float = 60.0,
           pulled_ago: float | None = 30.0,
           busy: int | None = 2, unreadable: int = 0) -> Claim:
    """A claim as it exists AFTER being pulled onto this machine.

    ``standing`` is how long the publisher had been on the account (its own
    clock, both ends) and ``pulled_ago`` is how long ago we fetched the file
    (our clock, both ends). Only the second decides freshness — see the
    module docstring for what happened when the first did.
    """
    return Claim(
        host=host, key=key, email="a@b.c",
        since=NOW - standing, published_at=NOW,
        busy_sessions=busy, unreadable_sessions=unreadable,
        pulled_at=None if pulled_ago is None else NOW - pulled_ago,
    )


def _keys(claims, **kw):
    return host_claim.claimed_keys(claims, now=NOW, **kw)


class TestBuildClaim:
    def test_names_the_account_by_identity_not_by_slot(self):
        record = host_claim.build_claim(
            _ident(), since=NOW - 30, now=NOW, busy_sessions=3, host="air"
        )
        assert record["identityKey"] == KEY_1
        assert "slot" not in json.dumps(record).lower()

    def test_carries_the_publishers_own_clock(self):
        # Freshness is judged inside one clock domain; a reader comparing
        # against its own now would be at the mercy of NTP drift.
        record = host_claim.build_claim(
            _ident(), since=NOW - 30, now=NOW, busy_sessions=0
        )
        assert record["publisherNow"] == NOW and record["since"] == NOW - 30

    def test_carries_no_credential_material(self):
        blob = json.dumps(host_claim.build_claim(
            _ident(), since=NOW, now=NOW, busy_sessions=1
        )).lower()
        for forbidden in ("token", "secret", "key\":", "refresh", "bearer"):
            assert forbidden not in blob.replace("identitykey", "")

    def test_an_unnameable_account_publishes_nothing(self):
        assert host_claim.build_claim(
            {"uuid": "", "organizationUuid": ORG_1}, since=NOW, now=NOW, busy_sessions=1
        ) is None


class TestPublishAndRead:
    def test_publish_then_read_round_trips(self, tmp_path):
        record = host_claim.build_claim(
            _ident(), since=NOW - 10, now=NOW, busy_sessions=2, host="studio"
        )
        assert host_claim.publish(tmp_path, record)
        assert host_claim.write_peer_claim(
            tmp_path, "studio", host_claim.claim_path(tmp_path).read_text()
        )
        (claim,) = host_claim.read_peers(tmp_path)
        assert claim.key == KEY_1 and claim.host == "studio"

    def test_publishing_nothing_is_not_an_error(self, tmp_path):
        assert host_claim.publish(tmp_path, None) is False
        assert not host_claim.claim_path(tmp_path).exists()

    @pytest.mark.parametrize(
        "text", ["", "not json", "[]", "null", '{"host": "x"}',
                 '{"identityKey": "k", "since": "x", "publisherNow": 1}'],
    )
    def test_unusable_peer_text_is_refused_before_it_lands(self, tmp_path, text):
        # A truncated scp or a captured error message must never become a
        # claim that excludes an account.
        assert host_claim.write_peer_claim(tmp_path, "studio", text) is False
        assert host_claim.read_peers(tmp_path) == []

    def test_a_malformed_file_does_not_hide_a_good_one(self, tmp_path):
        good = host_claim.build_claim(_ident(), since=NOW, now=NOW, busy_sessions=1)
        host_claim.write_peer_claim(tmp_path, "studio", json.dumps(good))
        (host_claim.peer_dir(tmp_path) / "broken.json").write_text("{ truncated")
        keys = [c.key for c in host_claim.read_peers(tmp_path)]
        assert keys == [KEY_1]

    def test_a_peer_file_is_not_world_readable(self, tmp_path):
        good = host_claim.build_claim(_ident(), since=NOW, now=NOW, busy_sessions=1)
        host_claim.write_peer_claim(tmp_path, "studio", json.dumps(good))
        mode = (host_claim.peer_dir(tmp_path) / "studio.json").stat().st_mode & 0o777
        assert mode == 0o600

    def test_a_hostile_peer_name_cannot_escape_the_directory(self, tmp_path):
        good = host_claim.build_claim(_ident(), since=NOW, now=NOW, busy_sessions=1)
        host_claim.write_peer_claim(tmp_path, "../../etc/passwd", json.dumps(good))
        written = list(host_claim.peer_dir(tmp_path).glob("*.json"))
        assert len(written) == 1
        assert written[0].parent == host_claim.peer_dir(tmp_path)

    def test_no_peer_directory_reads_as_no_claims(self, tmp_path):
        assert host_claim.read_peers(tmp_path) == []


class TestLiveness:
    def test_a_recently_pulled_claim_is_live(self):
        assert _keys([_claim(pulled_ago=60.0)]) != {}

    def test_a_claim_pulled_longer_ago_than_the_ttl_is_ignored(self):
        # Targeting fails OPEN: a peer that has gone quiet must not keep an
        # account excluded forever, or this machine parks on a shrinking pool.
        assert _keys([_claim(pulled_ago=CLAIM_TTL_S + 1)]) == {}

    def test_a_long_standing_claim_is_still_live_while_freshly_pulled(self):
        """The defect this replaces, in one assertion.

        Freshness used to be `publisherNow - since`, so a claim was called
        stale precisely when the peer had been sitting on the account longest —
        the case where it is MOST true. And because the engine assigned both
        timestamps from one clock() call, every real record had a standing of
        ~0 and could never expire at all.
        """
        claim = _claim(standing=CLAIM_TTL_S * 10, pulled_ago=5.0)
        assert _keys([claim]) != {}

    def test_a_claim_that_cannot_be_dated_is_not_live(self):
        """No pulledAt: hand-copied, or written before the field existed. It
        cannot be aged, so it fails OPEN rather than excluding forever."""
        assert _keys([_claim(pulled_ago=None)]) == {}

    def test_freshness_never_consults_the_publishers_clock(self):
        # A peer whose wall clock is a day ahead of ours. Its own timestamps
        # must not make it look fresher or staler than when we pulled it.
        skewed = Claim(**{**_claim(pulled_ago=10.0).__dict__,
                          "since": NOW + 86400 - 10, "published_at": NOW + 86400})
        assert _keys([skewed]) != {}
        expired = Claim(**{**skewed.__dict__,
                           "pulled_at": NOW - CLAIM_TTL_S - 1})
        assert _keys([expired]) == {}

    def test_our_own_claim_is_excluded(self):
        # A synced (rather than pulled) directory would otherwise make a machine
        # exclude the very account it is using.
        claims = [_claim(host="air")]
        assert _keys(claims, exclude_host="air") == {}
        assert _keys(claims, exclude_host="studio") != {}

    def test_two_peers_on_one_account_resolve_to_the_longest_holder(self):
        newer = _claim(host="a", standing=10.0)
        older = _claim(host="b", standing=500.0)
        assert _keys([newer, older])[KEY_1].host == "b"


class TestPressure:
    def test_a_peer_working_on_another_account_contributes_nothing(self):
        # Burn is per account: a busy peer elsewhere does not touch this window.
        claims = _keys([_claim(key=KEY_6, busy=9)])
        assert host_claim.peer_pressure(claims, KEY_1) == 0

    def test_a_peer_on_this_account_contributes_its_sessions(self):
        claims = _keys([_claim(busy=3)])
        assert host_claim.peer_pressure(claims, KEY_1) == 3

    def test_unknown_session_count_reads_as_at_least_one(self):
        # Pressure fails SAFE. Under-counting means switching later, which is
        # the outcome that costs a session.
        claims = _keys([_claim(busy=None)])
        assert host_claim.peer_pressure(claims, KEY_1) >= 1

    def test_zero_busy_still_counts_as_a_peer_being_there(self):
        claims = _keys([_claim(busy=0)])
        assert host_claim.peer_pressure(claims, KEY_1) >= 1

    def test_unreadable_sessions_raise_the_estimate(self):
        claims = _keys([_claim(busy=2, unreadable=3)])
        assert host_claim.peer_pressure(claims, KEY_1) == 5

    def test_no_key_is_no_pressure(self):
        claims = _keys([_claim(busy=4)])
        assert host_claim.peer_pressure(claims, None) == 0


class TestEndToEndAgainstARealRecord:
    """Every earlier liveness test hand-built a Claim, and hand-built one with
    a `since` the engine never emits. That is why a TTL that could not fire
    survived two review rounds. These start from build_claim's own output."""

    def test_a_record_the_engine_would_publish_can_expire(self, tmp_path):
        # _publish_claim passes state["lastSwitchAt"], assigned from the same
        # clock() call as `now` a few lines earlier — so standing is ~0 in
        # every real record, and anything keyed on it is unconditionally live.
        record = host_claim.build_claim(
            _ident(), since=NOW, now=NOW, busy_sessions=1, host="studio"
        )
        assert record["publisherNow"] - record["since"] == 0.0

        host_claim.write_peer_claim(
            tmp_path, "studio", json.dumps(record), now=NOW
        )
        pulled = host_claim.read_peers(tmp_path)
        assert len(pulled) == 1

        assert host_claim.claimed_keys(pulled, now=NOW + 60) != {}
        assert host_claim.claimed_keys(pulled, now=NOW + CLAIM_TTL_S + 1) == {}

    def test_the_pull_stamp_is_ours_not_the_peers(self, tmp_path):
        """A publisher does not get to declare how fresh it is on our side."""
        record = host_claim.build_claim(
            _ident(), since=NOW, now=NOW, busy_sessions=1, host="studio"
        )
        record["pulledAt"] = NOW + 999_999  # a peer claiming eternal freshness

        host_claim.write_peer_claim(
            tmp_path, "studio", json.dumps(record), now=NOW
        )
        pulled = host_claim.read_peers(tmp_path)

        assert pulled[0].pulled_at == NOW
        assert host_claim.claimed_keys(pulled, now=NOW + CLAIM_TTL_S + 1) == {}

    def test_a_pull_refreshes_an_existing_claim(self, tmp_path):
        """The whole mechanism: a live peer re-pulled stays excluded."""
        record = host_claim.build_claim(
            _ident(), since=NOW, now=NOW, busy_sessions=1, host="studio"
        )
        host_claim.write_peer_claim(tmp_path, "studio", json.dumps(record), now=NOW)
        late = NOW + CLAIM_TTL_S + 1
        assert host_claim.claimed_keys(host_claim.read_peers(tmp_path), now=late) == {}

        host_claim.write_peer_claim(tmp_path, "studio", json.dumps(record), now=late)
        assert host_claim.claimed_keys(
            host_claim.read_peers(tmp_path), now=late
        ) != {}


class TestHostnameCollision:
    """Two machines reporting the same hostname make every claim drop out via
    exclude_host, and the mechanism goes inert with nothing said. One of this
    fleet's machines calls itself "Mac", so this is not hypothetical."""

    def test_a_pulled_claim_remembers_what_we_pulled_it_as(self, tmp_path):
        record = host_claim.build_claim(
            _ident(), since=NOW, now=NOW, busy_sessions=1, host="Mac"
        )
        host_claim.write_peer_claim(
            tmp_path, "mac-studio", json.dumps(record), now=NOW
        )
        claim = host_claim.read_peers(tmp_path)[0]

        assert claim.source == "mac-studio", "the name we fetched it under"
        assert claim.host == "Mac", "the name it reports for itself"

    def test_a_colliding_name_still_drops_the_claim(self, tmp_path):
        """The drop itself is correct — a machine must never exclude the
        account it is using. What was missing is any way to see it happened."""
        record = host_claim.build_claim(
            _ident(), since=NOW, now=NOW, busy_sessions=1, host="Mac"
        )
        host_claim.write_peer_claim(
            tmp_path, "mac-studio", json.dumps(record), now=NOW
        )
        pulled = host_claim.read_peers(tmp_path)

        assert host_claim.claimed_keys(pulled, now=NOW, exclude_host="Mac") == {}
        assert host_claim.claimed_keys(pulled, now=NOW, exclude_host="other") != {}
