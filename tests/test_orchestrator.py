"""
Tests for the Orchestrator — event distribution, command dispatch, lifecycle.
"""

import pytest

from flame_sheep.orchestrator import Orchestrator, TimestampedEvent
from flame_sheep_audio._types import BeatEvent
from .conftest import FakeClock


@pytest.fixture
def clock():
    return FakeClock(start=1000.0)


@pytest.fixture
def orch(clock):
    o = Orchestrator(test_audio=True, clock=clock)
    o.start()
    yield o
    o.stop()


class TestRegistration:

    def test_register_returns_id(self, orch):
        cid = orch.register('wallpaper')
        assert cid == 'wallpaper'

    def test_drain_empty_on_fresh_consumer(self, orch):
        cid = orch.register('test')
        events = orch.drain_events(cid)
        assert events == []


class TestEventDistribution:

    def test_events_distributed_to_all_consumers(self, orch, clock):
        c1 = orch.register('consumer1')
        c2 = orch.register('consumer2')
        # Tick past the first beat boundary so events fire
        clock.advance(0.6)
        orch.tick()
        e1 = orch.drain_events(c1)
        e2 = orch.drain_events(c2)
        # Both consumers should get the same events
        assert len(e1) == len(e2)
        assert len(e1) > 0
        for te in e1:
            assert isinstance(te, TimestampedEvent)
            assert isinstance(te.event, BeatEvent)

    def test_drain_clears_queue(self, orch, clock):
        cid = orch.register('test')
        clock.advance(0.6)
        orch.tick()
        first = orch.drain_events(cid)
        assert len(first) > 0
        second = orch.drain_events(cid)
        assert second == []

    def test_slow_consumer_doesnt_block_fast(self, orch, clock):
        slow = orch.register('slow')
        fast = orch.register('fast')
        # Generate some events
        clock.advance(0.6)
        orch.tick()
        # Fast consumer drains
        orch.drain_events(fast)
        # Tick again
        clock.advance(0.5)
        orch.tick()
        # Fast consumer gets only new events
        fast_events = orch.drain_events(fast)
        # Slow consumer gets all accumulated events
        slow_events = orch.drain_events(slow)
        assert len(slow_events) >= len(fast_events)


class TestAudioState:

    def test_audio_state_updated_on_tick(self, orch, clock):
        clock.advance(0.6)
        orch.tick()
        snap = orch.audio_state
        # Should have mode from audio engine
        assert snap.mode in ('idle', 'energy', 'beat')

    def test_audio_state_has_spectrum(self, orch, clock):
        clock.advance(0.6)
        orch.tick()
        assert orch.audio_state.spectrum is not None
        assert len(orch.audio_state.spectrum) > 0


class TestCommandDispatch:

    def test_on_command_callback(self, orch):
        received = []
        orch.on_command('swap', lambda cmd: received.append(cmd))
        # Simulate a control pipe event by pushing directly
        from flame_sheep.control import ControlEvent
        orch.control._queue.put(ControlEvent(command='swap', args=[]))
        orch.tick()
        assert len(received) == 1
        assert received[0].command == 'swap'

    def test_multiple_handlers_for_same_command(self, orch):
        calls = []
        orch.on_command('config', lambda cmd: calls.append('a'))
        orch.on_command('config', lambda cmd: calls.append('b'))
        from flame_sheep.control import ControlEvent
        orch.control._queue.put(ControlEvent(command='config', args=['reload']))
        orch.tick()
        assert calls == ['a', 'b']

    def test_unhandled_command_no_error(self, orch):
        from flame_sheep.control import ControlEvent
        orch.control._queue.put(ControlEvent(command='unknown_cmd', args=[]))
        orch.tick()  # should not raise


class TestLifecycle:

    def test_start_stop(self, clock):
        o = Orchestrator(test_audio=True, clock=clock)
        o.start()
        o.stop()  # should not raise

    def test_tick_before_register(self, orch, clock):
        """Tick with no consumers should not error."""
        clock.advance(0.6)
        orch.tick()
