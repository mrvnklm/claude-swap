"""Unit tests for cancellation marks and the drain deadline they create."""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, timezone

import pytest

from claude_swap import cancellation


def _ts(year: int, month: int, day: int, hour: int = 0) -> float:
    return datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp()


def _state(**marks) -> dict:
    return {"schemaVersion": 1, cancellation.CANCELLED_KEY: marks}


class TestRecordFor:
    def test_returns_the_record_for_a_matching_slot(self):
        state = _state(**{"4": {"email": "a@b.c", "endsOn": "2026-09-08"}})
        assert cancellation.record_for(state, "4", "a@b.c")["endsOn"] == "2026-09-08"

    def test_a_slot_reassigned_to_another_account_ignores_the_mark(self):
        # cswap move/swap/add --slot can put a different account in the slot.
        # Applying the old mark to it would drain the wrong account first.
        state = _state(**{"4": {"email": "old@b.c", "endsOn": "2026-09-08"}})
        assert cancellation.record_for(state, "4", "new@b.c") is None

    def test_a_record_without_an_email_still_applies(self):
        # Hand-edited or older records carry no cross-check; honour them
        # rather than silently dropping a mark the operator set on purpose.
        state = _state(**{"4": {"endsOn": "2026-09-08"}})
        assert cancellation.record_for(state, "4", "a@b.c") is not None

    @pytest.mark.parametrize("state", [{}, {"cancelled": None}, {"cancelled": []}])
    def test_missing_or_malformed_container_is_none(self, state):
        assert cancellation.record_for(state, "4", "a@b.c") is None

    def test_a_non_dict_record_is_none(self):
        assert cancellation.record_for(_state(**{"4": "2026-09-08"}), "4", "a@b.c") is None


class TestEndsAtTs:
    def test_resolves_to_the_end_of_the_stated_day(self):
        # The period runs THROUGH its last day. Midnight would give up the
        # final day's quota, which is exactly what the mark exists to save.
        state = _state(**{"4": {"email": "a@b.c", "endsOn": "2026-09-08"}})
        ts = cancellation.ends_at_ts(state, "4", "a@b.c")
        assert ts > _ts(2026, 9, 8, 23)
        assert ts < _ts(2026, 9, 9)

    @pytest.mark.parametrize("raw", ["", "not-a-date", "2026-13-01", None, 20260908])
    def test_unusable_end_date_is_none(self, raw):
        state = _state(**{"4": {"email": "a@b.c", "endsOn": raw}})
        assert cancellation.ends_at_ts(state, "4", "a@b.c") is None

    def test_no_mark_is_none(self):
        assert cancellation.ends_at_ts({}, "4", "a@b.c") is None


class TestDrainDeadline:
    WEEKLY = _ts(2026, 9, 10)

    def test_without_a_mark_the_weekly_reset_is_unchanged(self):
        assert cancellation.drain_deadline_ts({}, "4", "a@b.c", self.WEEKLY) == self.WEEKLY

    def test_without_a_mark_an_unknown_reset_stays_unknown(self):
        assert cancellation.drain_deadline_ts({}, "4", "a@b.c", None) is None

    def test_an_earlier_cancellation_wins_over_the_weekly_reset(self):
        # The whole point: quota that expires for good on the 8th outranks
        # quota that merely resets on the 10th.
        state = _state(**{"4": {"email": "a@b.c", "endsOn": "2026-09-08"}})
        deadline = cancellation.drain_deadline_ts(state, "4", "a@b.c", self.WEEKLY)
        assert deadline is not None and deadline < self.WEEKLY

    def test_a_later_cancellation_does_not_delay_the_weekly_reset(self):
        # min(), not "the mark always wins" — a reset before the period end is
        # still the nearer deadline and must keep its ordering.
        state = _state(**{"4": {"email": "a@b.c", "endsOn": "2026-09-30"}})
        assert (
            cancellation.drain_deadline_ts(state, "4", "a@b.c", self.WEEKLY) == self.WEEKLY
        )

    def test_a_mark_supplies_a_deadline_where_the_reset_is_unknown(self):
        # consume-first rejects a candidate whose reset is unknown; a
        # cancellation gives it one, which is how a freshly-reset account with
        # no resets_at still gets drained before it expires.
        state = _state(**{"4": {"email": "a@b.c", "endsOn": "2026-09-08"}})
        assert cancellation.drain_deadline_ts(state, "4", "a@b.c", None) is not None

    def test_a_past_end_date_still_ranks_first(self):
        state = _state(**{"4": {"email": "a@b.c", "endsOn": "2020-01-01"}})
        deadline = cancellation.drain_deadline_ts(state, "4", "a@b.c", self.WEEKLY)
        assert deadline is not None and deadline < self.WEEKLY

    def test_a_mark_for_a_reassigned_slot_does_not_move_the_deadline(self):
        state = _state(**{"4": {"email": "old@b.c", "endsOn": "2026-09-08"}})
        assert (
            cancellation.drain_deadline_ts(state, "4", "new@b.c", self.WEEKLY) == self.WEEKLY
        )


class TestPersistence:
    def test_set_then_read_round_trips(self, tmp_path):
        cancellation.set_cancelled(tmp_path, "4", "a@b.c", date(2026, 9, 8))
        state = cancellation.read_state(tmp_path)
        assert cancellation.record_for(state, "4", "a@b.c") == {
            "email": "a@b.c",
            "endsOn": "2026-09-08",
        }

    def test_a_note_is_carried_when_given_and_omitted_when_not(self, tmp_path):
        cancellation.set_cancelled(tmp_path, "4", "a@b.c", date(2026, 9, 8), note="hi")
        assert cancellation.record_for(
            cancellation.read_state(tmp_path), "4", "a@b.c"
        )["note"] == "hi"
        cancellation.set_cancelled(tmp_path, "5", "d@e.f", date(2026, 9, 9))
        assert "note" not in cancellation.record_for(
            cancellation.read_state(tmp_path), "5", "d@e.f"
        )

    def test_marking_preserves_unrelated_state(self, tmp_path):
        # The file is shared with the engine's quarantine and cooldown; a CLI
        # write that dropped those would silently un-quarantine an account.
        path = cancellation.state_path(tmp_path)
        path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "lastSwitchAt": 1234.5,
                    "quarantine": {"7": {"email": "q@b.c", "reason": "invalid_grant"}},
                }
            ),
            encoding="utf-8",
        )
        cancellation.set_cancelled(tmp_path, "4", "a@b.c", date(2026, 9, 8))
        state = cancellation.read_state(tmp_path)
        assert state["lastSwitchAt"] == 1234.5
        assert state["quarantine"]["7"]["reason"] == "invalid_grant"
        assert state["schemaVersion"] == 1

    def test_marking_does_not_stamp_a_schema_version_of_its_own(self, tmp_path):
        # The engine owns schemaVersion. A CLI write claiming one would assert
        # a schema this code did not produce.
        cancellation.set_cancelled(tmp_path, "4", "a@b.c", date(2026, 9, 8))
        assert "schemaVersion" not in cancellation.read_state(tmp_path)

    def test_clear_removes_the_mark_and_reports_it(self, tmp_path):
        cancellation.set_cancelled(tmp_path, "4", "a@b.c", date(2026, 9, 8))
        assert cancellation.clear_cancelled(tmp_path, "4") is True
        assert cancellation.cancellations(cancellation.read_state(tmp_path)) == {}

    def test_clearing_an_unmarked_slot_reports_false(self, tmp_path):
        assert cancellation.clear_cancelled(tmp_path, "4") is False

    def test_clearing_one_of_two_leaves_the_other(self, tmp_path):
        cancellation.set_cancelled(tmp_path, "4", "a@b.c", date(2026, 9, 8))
        cancellation.set_cancelled(tmp_path, "5", "d@e.f", date(2026, 9, 9))
        cancellation.clear_cancelled(tmp_path, "4")
        marks = cancellation.cancellations(cancellation.read_state(tmp_path))
        assert list(marks) == ["5"]

    def test_an_unreadable_state_file_reads_as_empty(self, tmp_path):
        cancellation.state_path(tmp_path).write_text("{ not json", encoding="utf-8")
        assert cancellation.read_state(tmp_path) == {}

    def test_a_corrupt_state_file_does_not_block_a_new_mark(self, tmp_path):
        cancellation.state_path(tmp_path).write_text("[]", encoding="utf-8")
        cancellation.set_cancelled(tmp_path, "4", "a@b.c", date(2026, 9, 8))
        assert cancellation.record_for(
            cancellation.read_state(tmp_path), "4", "a@b.c"
        ) is not None


class TestLockIsActuallyTaken:
    """The path-identity assertion in test_autoswitch proves the two sides
    agree on WHICH lock. This proves the CLI writer actually takes it."""

    def test_set_cancelled_waits_for_a_held_state_lock(self, tmp_path):
        from claude_swap.locking import FileLock

        started = threading.Event()
        finished = threading.Event()

        def mark():
            started.set()
            cancellation.set_cancelled(tmp_path, "4", "a@b.c", date(2026, 9, 8))
            finished.set()

        held = FileLock(tmp_path / cancellation.LOCK_FILENAME)
        with held:
            worker = threading.Thread(target=mark, daemon=True)
            worker.start()
            assert started.wait(5.0), "worker never ran"
            # Generous enough not to flake, short enough to stay fast. With the
            # `with _lock(...)` removed this write lands immediately and the
            # assertion fails — which is the whole point of the test.
            assert not finished.wait(0.75), (
                "set_cancelled wrote while the state lock was held by someone else"
            )
        assert finished.wait(10.0), "set_cancelled never completed after release"
        assert cancellation.record_for(
            cancellation.read_state(tmp_path), "4", "a@b.c"
        ) is not None

