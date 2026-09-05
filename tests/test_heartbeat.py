"""Tests for engine liveness (claude_swap.heartbeat).

The two rules worth pinning are the ones a first design got wrong: a beat is
fresh only until its own recorded deadline (a live PID may not promote a
stale beat), and silence is armed by the operator's intent rather than by
whether the file happens to exist.
"""

from __future__ import annotations

import json
import os
import socket

import pytest

from claude_swap import heartbeat
from claude_swap.heartbeat import GRACE_S, Beat, describe, read_beat, write_beat

NOW = 1_700_000_000.0


def _beat(tmp_path, **kw):
    defaults = dict(
        tick_at=NOW,
        next_tick_by=NOW + 60 + GRACE_S,
        outcome="no-action",
        host=socket.gethostname(),
        pid=os.getpid(),
    )
    defaults.update(kw)
    record = {
        "schemaVersion": 1,
        "tickAt": defaults["tick_at"],
        "nextTickBy": defaults["next_tick_by"],
        "outcome": defaults["outcome"],
        "host": defaults["host"],
        "pid": defaults["pid"],
    }
    heartbeat.heartbeat_path(tmp_path).write_text(json.dumps(record))
    return Beat(**defaults)


class TestWriteAndRead:
    def test_a_beat_round_trips(self, tmp_path):
        write_beat(tmp_path, now=NOW, next_delay=60.0, outcome="switched")
        beat = read_beat(tmp_path)
        assert beat is not None
        assert beat.tick_at == NOW
        assert beat.outcome == "switched"
        assert beat.pid == os.getpid()
        assert beat.host == socket.gethostname()

    def test_the_deadline_comes_from_the_delay_the_writer_will_sleep(self, tmp_path):
        """A reader must never guess the interval. A long backoff has to widen
        its own window, or every sleep longer than one interval reads as dead."""
        write_beat(tmp_path, now=NOW, next_delay=1800.0, outcome="blocked")
        beat = read_beat(tmp_path)
        assert beat.next_tick_by == NOW + 1800.0 + GRACE_S
        assert beat.is_fresh(NOW + 1800.0)

    def test_no_file_reads_as_no_beat(self, tmp_path):
        assert read_beat(tmp_path) is None

    @pytest.mark.parametrize(
        "content", ["not json", "[]", '{"tickAt": "soon"}', '{"nextTickBy": 1}']
    )
    def test_an_unusable_file_reads_as_no_beat(self, tmp_path, content):
        heartbeat.heartbeat_path(tmp_path).write_text(content)
        assert read_beat(tmp_path) is None

    def test_an_unwritable_directory_does_not_raise(self, tmp_path):
        """A status file must never be able to take the engine down with it."""
        write_beat(tmp_path / "nope", now=NOW, next_delay=60.0, outcome=None)


class TestFreshness:
    def test_fresh_up_to_the_deadline_and_stale_after(self, tmp_path):
        beat = _beat(tmp_path)
        assert beat.is_fresh(beat.next_tick_by)
        assert not beat.is_fresh(beat.next_tick_by + 0.001)

    def test_a_clock_ahead_of_ours_does_not_produce_a_negative_age(self, tmp_path):
        beat = _beat(tmp_path, tick_at=NOW + 500)
        assert beat.age_s(NOW) == 0.0


class TestDescribe:
    def test_it_says_nothing_when_no_engine_was_asked_for(self, tmp_path):
        """The shipped default. Someone who runs no engine must never be
        nagged, whatever is or is not on disk."""
        assert describe(tmp_path, expected=False, now=NOW) is None
        _beat(tmp_path, tick_at=NOW - 999_999, next_tick_by=NOW - 999_000)
        assert describe(tmp_path, expected=False, now=NOW) is None

    def test_an_engine_that_never_started_is_reported(self, tmp_path):
        """Armed on intent, not on the file: _start_engine can fail, and a
        reader keyed on the file's existence would then stay silent forever —
        the one case most worth shouting about."""
        assert read_beat(tmp_path) is None
        line = describe(tmp_path, expected=True, now=NOW)
        assert line is not None and "no engine has ever reported in" in line

    def test_a_stale_beat_is_reported_with_its_age(self, tmp_path):
        _beat(tmp_path, tick_at=NOW - 36 * 3600, next_tick_by=NOW - 36 * 3600 + 150)
        line = describe(tmp_path, expected=True, now=NOW)
        assert line is not None and "1d 12h" in line

    def test_a_live_pid_cannot_rescue_a_stale_beat(self, tmp_path):
        """The refuted rule, pinned: liveness hung on the PID alone, so a
        36-hour-old beat whose PID had been reused read as 'running'."""
        _beat(
            tmp_path,
            tick_at=NOW - 36 * 3600,
            next_tick_by=NOW - 36 * 3600 + 150,
            pid=os.getpid(),  # unmistakably alive
        )
        assert describe(tmp_path, expected=True, now=NOW) is not None

    def test_a_fresh_beat_from_a_living_process_says_nothing(self, tmp_path):
        _beat(tmp_path)
        assert describe(tmp_path, expected=True, now=NOW) is None

    def test_a_fresh_beat_from_a_dead_local_process_is_reported(self, tmp_path):
        """The other direction: the deadline has not passed yet, but the
        process is already gone. A PID may demote, and only demote."""
        _beat(tmp_path, pid=_a_dead_pid())
        assert describe(tmp_path, expected=True, now=NOW) is not None

    def test_a_fresh_beat_from_another_host_is_trusted(self, tmp_path):
        """A shared backup dir can carry a peer's beat. We cannot check its
        PID, and refusing to trust it would report a healthy engine as dead."""
        _beat(tmp_path, host="some-other-machine", pid=999_999)
        assert describe(tmp_path, expected=True, now=NOW) is None


def _a_dead_pid() -> int:
    """A PID that is certainly not running: fork a child and reap it."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def test_the_age_wording_matches_the_rest_of_the_tool():
    """One span, one spelling. heartbeat keeps its own copy because it is read
    by the engine and format_duration lives in the UI layer; that is only safe
    while they agree."""
    from claude_swap.tui.data import format_duration

    for seconds in (0, 45, 59, 60, 599, 3599, 3600, 7980, 86399, 86400, 129600):
        assert heartbeat._ago(seconds) == format_duration(seconds), seconds
