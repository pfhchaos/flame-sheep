"""Tests for the control pipe command parsing.

Tests parsing logic and queue behavior directly — no actual FIFO
needed since we can test _parse_line and the queue independently.
"""

import os
import stat
import tempfile
import time
import threading
from pathlib import Path

import pytest

from flame_sheep.control import ControlPipe, ControlEvent


class TestParseLine:
    """Verify command parsing without any pipe I/O."""

    def _parse(self, line: str):
        pipe = ControlPipe.__new__(ControlPipe)
        return pipe._parse_line(line)

    def test_simple_command(self):
        ev = self._parse('swap')
        assert ev.command == 'swap'
        assert ev.args == []

    def test_command_with_args(self):
        ev = self._parse('tempo 140')
        assert ev.command == 'tempo'
        assert ev.args == ['140']

    def test_case_insensitive(self):
        ev = self._parse('SWAP')
        assert ev.command == 'swap'

    def test_all_valid_commands(self):
        for cmd in ('swap', 'song', 'tempo', 'like', 'dislike',
                    'next', 'pause', 'resume', 'seek', 'quit'):
            ev = self._parse(cmd)
            assert ev is not None
            assert ev.command == cmd

    def test_unknown_command_returns_none(self):
        assert self._parse('foobar') is None

    def test_empty_line_returns_none(self):
        assert self._parse('') is None

    def test_extra_whitespace(self):
        ev = self._parse('  tempo   120  ')
        assert ev.command == 'tempo'
        assert ev.args == ['120']

    def test_multiple_args(self):
        ev = self._parse('tempo 120 hint')
        assert ev.command == 'tempo'
        assert ev.args == ['120', 'hint']


class TestQueueBehavior:
    """Test poll/poll_all without starting the read thread."""

    def test_poll_empty_returns_none(self):
        pipe = ControlPipe.__new__(ControlPipe)
        pipe._queue = __import__('queue').Queue()
        assert pipe.poll() is None

    def test_poll_returns_queued_event(self):
        pipe = ControlPipe.__new__(ControlPipe)
        pipe._queue = __import__('queue').Queue()
        ev = ControlEvent(command='swap', args=[])
        pipe._queue.put(ev)
        assert pipe.poll() == ev
        assert pipe.poll() is None

    def test_poll_all_drains_queue(self):
        pipe = ControlPipe.__new__(ControlPipe)
        pipe._queue = __import__('queue').Queue()
        for cmd in ('swap', 'like', 'dislike'):
            pipe._queue.put(ControlEvent(command=cmd, args=[]))
        events = pipe.poll_all()
        assert len(events) == 3
        assert [e.command for e in events] == ['swap', 'like', 'dislike']
        assert pipe.poll() is None


class TestPipeLifecycle:
    """Test pipe creation and thread start/stop with a real FIFO."""

    def test_creates_fifo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.ctl'
            pipe = ControlPipe(pipe_path=path)
            pipe._ensure_pipe_exists()
            assert path.exists()
            assert stat.S_ISFIFO(os.stat(path).st_mode)

    def test_replaces_regular_file_with_fifo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.ctl'
            path.write_text('not a fifo')
            pipe = ControlPipe(pipe_path=path)
            pipe._ensure_pipe_exists()
            assert stat.S_ISFIFO(os.stat(path).st_mode)

    def test_start_stop_no_crash(self):
        """Start and stop without sending any commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.ctl'
            pipe = ControlPipe(pipe_path=path)
            pipe.start()
            time.sleep(0.05)
            assert pipe._running
            pipe.stop()
            assert not pipe._running

    def test_write_and_read_command(self):
        """Write to the FIFO and verify the command is queued."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.ctl'
            pipe = ControlPipe(pipe_path=path)
            pipe.start()
            try:
                # Write a command — must open in a separate thread since
                # opening a FIFO for writing blocks until a reader is ready
                # (our reader thread is already running)
                with open(path, 'w') as f:
                    f.write('swap\nlike\n')

                # Give the reader thread time to process
                for _ in range(50):
                    events = pipe.poll_all()
                    if events:
                        break
                    time.sleep(0.02)

                cmds = [e.command for e in events]
                assert 'swap' in cmds
                assert 'like' in cmds
            finally:
                pipe.stop()
