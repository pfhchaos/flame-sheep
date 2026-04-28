"""Tests for the structured audio feature logger."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from flame_sheep.orchestrator import Orchestrator, TimestampedEvent
from flame_sheep.logger import AudioFeatureLogger
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


@pytest.fixture
def log_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def logger(orch, log_file):
    lg = AudioFeatureLogger(orch, path=log_file)
    yield lg
    lg.close()


class TestAudioFeatureLogger:

    def test_writes_valid_json_lines(self, logger, log_file, orch, clock):
        clock.advance(0.6)
        orch.tick()
        logger.tick()
        logger.flush()
        lines = log_file.read_text().strip().split('\n')
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert 't' in record
        assert 'mode' in record

    def test_multiple_ticks(self, logger, log_file, orch, clock):
        for _ in range(5):
            clock.advance(0.1)
            orch.tick()
            logger.tick()
        logger.flush()
        lines = log_file.read_text().strip().split('\n')
        assert len(lines) == 5
        for line in lines:
            json.loads(line)  # should not raise

    def test_record_has_all_fields(self, logger, log_file, orch, clock):
        clock.advance(0.6)
        orch.tick()
        logger.tick()
        logger.flush()
        record = json.loads(log_file.read_text().strip())
        assert 'mode' in record
        assert 'bpm' in record
        assert 'effective_bpm' in record
        assert 'tempo_confidence' in record
        assert 'percussiveness' in record
        assert 'section_change' in record
        assert 'break_intensity' in record
        assert 'centroid' in record
        assert 'bands' in record

    def test_bands_have_rms(self, logger, log_file, orch, clock):
        clock.advance(0.6)
        orch.tick()
        logger.tick()
        logger.flush()
        record = json.loads(log_file.read_text().strip())
        for band_name, band_data in record['bands'].items():
            assert 'rms' in band_data

    def test_events_included_when_present(self, logger, log_file, orch, clock):
        clock.advance(0.6)  # past first beat boundary
        orch.tick()
        logger.tick()
        logger.flush()
        record = json.loads(log_file.read_text().strip())
        if 'events' in record:
            for event in record['events']:
                assert 'kind' in event
                assert 'energy' in event

    def test_field_filtering(self, orch, log_file):
        lg = AudioFeatureLogger(orch, path=log_file, fields=['bpm', 'mode'])
        orch.tick()
        lg.tick()
        lg.flush()
        lg.close()
        record = json.loads(log_file.read_text().strip())
        assert 'bpm' in record
        assert 'mode' in record
        assert 'bands' not in record
        assert 'percussiveness' not in record

    def test_appends_to_existing_file(self, orch, log_file):
        log_file.write_text('{"existing": true}\n')
        lg = AudioFeatureLogger(orch, path=log_file)
        orch.tick()
        lg.tick()
        lg.flush()
        lg.close()
        lines = log_file.read_text().strip().split('\n')
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"existing": True}

    def test_close_is_idempotent(self, logger):
        logger.close()
        # Second close should not raise
        logger.close()
