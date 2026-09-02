"""Unit tests for cancellation marks and the drain deadline they create."""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta, timezone

import pytest

from claude_swap import cancellation

UTC = timezone.utc
UUID_A = "6a64d5cc-9af1-4bca-aaa4-7409aad57394"
UUID_B = "18cdb663-5e67-4271-8407-d240e05673f5"
ORG_1 = "af232ed4-8c8a-4a46-b942-f5ff6528c80c"
ORG_6 = "bed07a55-be7d-469a-b348-38ea4b909c4f"
KEY_1 = f"{UUID_A}:{ORG_1}"
KEY_6 = f"{UUID_A}:{ORG_6}"


def _ident(uuid: str, org: str, email: str = "a@b.c") -> dict:
    return {"uuid": uuid, "organizationUuid": org, "email": email}


def _ts(y: int, m: int, d: int, hour: int = 12) -> float:
    return datetime(y, m, d, hour, tzinfo=UTC).timestamp()


def _state(**marks) -> dict:
    return {"schemaVersion": 1, cancellation.CANCELLED_KEY: marks}


class TestIdentityKey:
    def test_pairs_the_account_uuid_with_the_organization(self):
        assert cancellation.identity_key(_ident(UUID_A, ORG_1)) == KEY_1

    def test_the_same_account_in_two_organizations_gets_two_keys(self):
        # The measured case this key exists for: one account holding seats in
        # two organizations appears as two managed slots with one uuid and one
        # email. Neither alone can tell them apart.
        assert cancellation.identity_key(_ident(UUID_A, ORG_1)) != cancellation.identity_key(
            _ident(UUID_A, ORG_6)
        )

    @pytest.mark.parametrize(
        "identity",
        [
            {"uuid": "", "organizationUuid": ORG_1},
            {"uuid": UUID_A, "organizationUuid": ""},
            {"uuid": "  ", "organizationUuid": ORG_1},
            {"organizationUuid": ORG_1},
            {"uuid": UUID_A},
            {},
        ],
    )
    def test_a_missing_half_refuses_to_key(self, identity):
        # The safe refusal: a mark that cannot name its account would end up
        # applied by position, which is the bug this key removes.
        assert cancellation.identity_key(identity) is None

    def test_a_non_mapping_is_none(self):
        assert cancellation.identity_key(None) is None


class TestEndsTs:
    LIVE = {"endsOn": "2026-09-08"}

    def test_ranks_from_the_first_instant_of_the_stated_day(self):
        # The floor, not the ceiling: the ceiling can place the deadline hours
        # after the real period boundary, which is the direction that strands
        # quota.
        ts = cancellation.ends_ts(self.LIVE, _ts(2026, 9, 1), tz=UTC)
        assert ts == datetime(2026, 9, 8, 0, 0, tzinfo=UTC).timestamp()

    def test_still_live_during_the_last_hour_of_the_day(self):
        assert cancellation.ends_ts(self.LIVE, _ts(2026, 9, 8, 23), tz=UTC) is not None

    def test_none_once_the_day_is_over(self):
        # Past == unknown, mirroring _seven_day_reset_ts. Ranked as a real
        # instant, a lapsed mark pins the fleet to a dead account.
        assert cancellation.ends_ts(self.LIVE, _ts(2026, 9, 9, 0), tz=UTC) is None

    def test_there_is_no_grace_period(self):
        one_second_after = (
            datetime(2026, 9, 9, tzinfo=UTC) + timedelta(seconds=1)
        ).timestamp()
        assert cancellation.ends_ts(self.LIVE, one_second_after, tz=UTC) is None

    @pytest.mark.parametrize("raw", ["", "not-a-date", "2026-13-01", None, 20260908])
    def test_an_unusable_date_is_none(self, raw):
        assert cancellation.ends_ts({"endsOn": raw}, _ts(2026, 9, 1), tz=UTC) is None

    def test_a_non_dict_record_is_none(self):
        assert cancellation.ends_ts("2026-09-08", _ts(2026, 9, 1), tz=UTC) is None

    def test_the_zone_decides_the_boundary(self):
        # A date read off a billing page is a local date. Resolving it in UTC
        # ends the mark on the wrong day for anyone far enough from it.
        far_east = timezone(timedelta(hours=14))
        moment = datetime(2026, 9, 8, 12, tzinfo=far_east).timestamp()
        assert cancellation.ends_ts(self.LIVE, moment, tz=far_east) is not None
        assert cancellation.ends_ts(self.LIVE, moment, tz=timezone(timedelta(hours=-11))) is not None
        after_east = datetime(2026, 9, 9, 1, tzinfo=far_east).timestamp()
        assert cancellation.ends_ts(self.LIVE, after_east, tz=far_east) is None


class TestMergeDeadline:
    WEEKLY = _ts(2026, 9, 10)

    def test_no_mark_leaves_the_weekly_reset_alone(self):
        assert cancellation.merge_deadline(self.WEEKLY, None) == self.WEEKLY

    def test_no_mark_and_no_reset_is_unknown(self):
        assert cancellation.merge_deadline(None, None) is None

    def test_an_earlier_end_wins(self):
        earlier = self.WEEKLY - 86400
        assert cancellation.merge_deadline(self.WEEKLY, earlier) == earlier

    def test_a_later_end_does_not_delay_the_reset(self):
        # min(), not "the mark always wins".
        later = self.WEEKLY + 86400
        assert cancellation.merge_deadline(self.WEEKLY, later) == self.WEEKLY

    def test_a_mark_supplies_a_deadline_where_the_reset_is_unknown(self):
        assert cancellation.merge_deadline(None, self.WEEKLY) == self.WEEKLY


class TestResolveMarks:
    IDENTITIES = {"1": _ident(UUID_A, ORG_1), "6": _ident(UUID_A, ORG_6)}

    def test_resolves_a_mark_to_the_slot_that_holds_the_identity(self):
        state = _state(**{KEY_6: {"endsOn": "2026-09-08", "email": "a@b.c"}})
        (mark,) = cancellation.resolve_marks(state, self.IDENTITIES, _ts(2026, 9, 1))
        assert mark.slot == "6" and not mark.orphaned and not mark.lapsed

    def test_two_slots_sharing_one_account_are_told_apart(self):
        # The measured fleet shape. Under the old email key both marks matched
        # both slots.
        state = _state(**{
            KEY_1: {"endsOn": "2026-09-08"},
            KEY_6: {"endsOn": "2026-09-20"},
        })
        marks = cancellation.resolve_marks(state, self.IDENTITIES, _ts(2026, 9, 1))
        assert {m.slot: m.ends_on.isoformat() for m in marks} == {
            "1": "2026-09-08",
            "6": "2026-09-20",
        }

    def test_a_mark_for_an_account_that_is_gone_is_reported_not_hidden(self):
        # Hiding it is how the previous shape made an orphan unremovable.
        state = _state(**{"someone:else": {"endsOn": "2026-09-08", "email": "x@y.z"}})
        (mark,) = cancellation.resolve_marks(state, self.IDENTITIES, _ts(2026, 9, 1))
        assert mark.orphaned and mark.slot is None and mark.email == "x@y.z"

    def test_a_lapsed_mark_is_flagged_but_still_listed(self):
        state = _state(**{KEY_1: {"endsOn": "2020-01-01"}})
        (mark,) = cancellation.resolve_marks(state, self.IDENTITIES, _ts(2026, 9, 1))
        assert mark.lapsed and mark.slot == "1"

    def test_an_unparseable_date_is_not_called_lapsed(self):
        # Inert, but not provably dead — and --prune must not delete it.
        state = _state(**{KEY_1: {"endsOn": "nonsense"}})
        (mark,) = cancellation.resolve_marks(state, self.IDENTITIES, _ts(2026, 9, 1))
        assert mark.ends_on is None and not mark.lapsed

    def test_slots_sort_before_orphans(self):
        state = _state(**{"gone:gone": {"endsOn": "2026-09-08"}, KEY_1: {"endsOn": "2026-09-09"}})
        marks = cancellation.resolve_marks(state, self.IDENTITIES, _ts(2026, 9, 1))
        assert [m.slot for m in marks] == ["1", None]

    def test_an_identity_missing_a_half_matches_nothing(self):
        state = _state(**{":": {"endsOn": "2026-09-08"}})
        identities = {"3": {"uuid": "", "organizationUuid": ""}}
        (mark,) = cancellation.resolve_marks(state, identities, _ts(2026, 9, 1))
        assert mark.orphaned


class TestDuplicateIdentities:
    """Two slots can carry the same identity after a botched add. Either answer
    is arbitrary — but the engine and the listing must give the SAME one, which
    is how the two resolvers this replaced came to disagree."""

    DUPES = {"6": _ident(UUID_A, ORG_1), "1": _ident(UUID_A, ORG_1)}
    STATE = _state(**{KEY_1: {"endsOn": "2026-09-08"}})

    def test_the_lowest_slot_wins_deterministically(self):
        (mark,) = cancellation.resolve_marks(self.STATE, self.DUPES, _ts(2026, 9, 1))
        assert mark.slot == "1"

    def test_the_engine_and_the_listing_agree(self):
        (mark,) = cancellation.resolve_marks(self.STATE, self.DUPES, _ts(2026, 9, 1))
        deadlines = cancellation.deadlines_by_slot(
            self.STATE, self.DUPES, _ts(2026, 9, 1)
        )
        assert list(deadlines) == [mark.slot]

    def test_ordering_of_the_identity_map_does_not_decide(self):
        reversed_map = dict(reversed(list(self.DUPES.items())))
        a = cancellation.resolve_marks(self.STATE, self.DUPES, _ts(2026, 9, 1))[0].slot
        b = cancellation.resolve_marks(self.STATE, reversed_map, _ts(2026, 9, 1))[0].slot
        assert a == b == "1"


class TestDeadlinesBySlot:
    IDENTITIES = {"1": _ident(UUID_A, ORG_1), "2": _ident(UUID_B, ORG_6)}

    def test_maps_a_live_mark_onto_its_slot(self):
        state = _state(**{KEY_1: {"endsOn": "2026-09-08"}})
        out = cancellation.deadlines_by_slot(state, self.IDENTITIES, _ts(2026, 9, 1), tz=UTC)
        assert list(out) == ["1"]

    def test_a_lapsed_mark_is_absent(self):
        # The blocker, at the source: the engine can never see a past deadline.
        state = _state(**{KEY_1: {"endsOn": "2020-01-01"}})
        assert cancellation.deadlines_by_slot(
            state, self.IDENTITIES, _ts(2026, 9, 1), tz=UTC
        ) == {}

    def test_an_orphaned_mark_is_absent(self):
        state = _state(**{"gone:gone": {"endsOn": "2026-09-08"}})
        assert cancellation.deadlines_by_slot(
            state, self.IDENTITIES, _ts(2026, 9, 1), tz=UTC
        ) == {}

    def test_no_marks_is_empty(self):
        assert cancellation.deadlines_by_slot({}, self.IDENTITIES, _ts(2026, 9, 1)) == {}


class TestPersistence:
    def test_set_then_resolve_round_trips(self, tmp_path):
        cancellation.set_cancelled(tmp_path, KEY_1, date(2026, 9, 8), email="a@b.c")
        state = cancellation.read_state(tmp_path)
        (mark,) = cancellation.resolve_marks(
            state, {"1": _ident(UUID_A, ORG_1)}, _ts(2026, 9, 1)
        )
        assert mark.key == KEY_1 and mark.ends_on == date(2026, 9, 8)

    def test_marking_preserves_unrelated_state(self, tmp_path):
        # The file is shared with the engine's quarantine and cooldown.
        cancellation.state_path(tmp_path).write_text(
            json.dumps({
                "schemaVersion": 1,
                "lastSwitchAt": 1234.5,
                "quarantine": {"7": {"email": "q@b.c", "reason": "invalid_grant"}},
            }),
            encoding="utf-8",
        )
        cancellation.set_cancelled(tmp_path, KEY_1, date(2026, 9, 8))
        state = cancellation.read_state(tmp_path)
        assert state["lastSwitchAt"] == 1234.5
        assert state["quarantine"]["7"]["reason"] == "invalid_grant"
        assert state["schemaVersion"] == 1

    def test_marking_does_not_stamp_a_schema_version_of_its_own(self, tmp_path):
        cancellation.set_cancelled(tmp_path, KEY_1, date(2026, 9, 8))
        assert "schemaVersion" not in cancellation.read_state(tmp_path)

    def test_a_non_dict_container_is_replaced_not_indexed(self, tmp_path):
        cancellation.state_path(tmp_path).write_text(
            json.dumps({"cancelled": []}), encoding="utf-8"
        )
        cancellation.set_cancelled(tmp_path, KEY_1, date(2026, 9, 8))
        assert KEY_1 in cancellation.cancellations(cancellation.read_state(tmp_path))

    def test_clear_removes_only_the_named_keys(self, tmp_path):
        cancellation.set_cancelled(tmp_path, KEY_1, date(2026, 9, 8))
        cancellation.set_cancelled(tmp_path, KEY_6, date(2026, 9, 9))
        assert cancellation.clear_cancelled(tmp_path, [KEY_1]) == 1
        assert list(cancellation.cancellations(cancellation.read_state(tmp_path))) == [KEY_6]

    def test_clearing_an_unmarked_key_reports_zero(self, tmp_path):
        assert cancellation.clear_cancelled(tmp_path, [KEY_1]) == 0

    def test_clearing_nothing_does_not_touch_the_file(self, tmp_path):
        cancellation.set_cancelled(tmp_path, KEY_1, date(2026, 9, 8))
        before = cancellation.state_path(tmp_path).read_text()
        assert cancellation.clear_cancelled(tmp_path, []) == 0
        assert cancellation.state_path(tmp_path).read_text() == before

    def test_an_orphan_can_be_cleared_by_key(self, tmp_path):
        # The case the old slot-keyed delete could not reach at all.
        cancellation.set_cancelled(tmp_path, "gone:gone", date(2026, 9, 8))
        marks = cancellation.resolve_marks(
            cancellation.read_state(tmp_path), {}, _ts(2026, 9, 1)
        )
        assert cancellation.clear_cancelled(tmp_path, [m.key for m in marks if m.orphaned]) == 1
        assert cancellation.cancellations(cancellation.read_state(tmp_path)) == {}

    def test_an_unreadable_state_file_reads_as_empty(self, tmp_path):
        cancellation.state_path(tmp_path).write_text("{ not json", encoding="utf-8")
        assert cancellation.read_state(tmp_path) == {}


class TestLockIsActuallyTaken:
    """The path-identity assertion in test_autoswitch proves the two sides
    agree on WHICH lock. This proves the writer actually takes it."""

    def test_set_cancelled_waits_for_a_held_state_lock(self, tmp_path):
        from claude_swap.locking import FileLock

        started = threading.Event()
        finished = threading.Event()

        def mark():
            started.set()
            cancellation.set_cancelled(tmp_path, KEY_1, date(2026, 9, 8))
            finished.set()

        held = FileLock(tmp_path / cancellation.LOCK_FILENAME)
        with held:
            worker = threading.Thread(target=mark, daemon=True)
            worker.start()
            assert started.wait(5.0), "worker never ran"
            assert not finished.wait(0.75), (
                "set_cancelled wrote while the state lock was held by someone else"
            )
        assert finished.wait(10.0), "set_cancelled never completed after release"
        assert KEY_1 in cancellation.cancellations(cancellation.read_state(tmp_path))
