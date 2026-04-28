"""Structured audio feature logger — orchestrator consumer.

Writes per-frame AudioSnapshot data as JSON lines for offline
analysis and tuning. No spectrum/waveform arrays — just scalar
features and discrete events.

Usage:
    logger = AudioFeatureLogger(orchestrator, path='features.jsonl')
    # In main loop:
    logger.tick()
    # On shutdown:
    logger.close()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from flame_sheep_audio._types import AudioSnapshot, BandState
from flame_sheep.orchestrator import Orchestrator, TimestampedEvent

log = logging.getLogger(__name__)

# Default output path
DEFAULT_FEATURE_FILE = Path('~/.local/share/flame-sheep/features.jsonl').expanduser()


class AudioFeatureLogger:
    """Orchestrator consumer that writes audio features as JSON lines."""

    def __init__(self, orchestrator: Orchestrator,
                 path: str | Path = DEFAULT_FEATURE_FILE,
                 fields: list[str] | None = None) -> None:
        self._orch = orchestrator
        self._consumer_id = orchestrator.register('feature_logger')
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, 'a')
        self._fields = set(fields) if fields else None
        log.info(f'feature logger writing to {self._path}')

    def tick(self) -> None:
        """Serialize current audio state + events to one JSON line."""
        snap = self._orch.audio_state
        events = self._orch.drain_events(self._consumer_id)
        record = self._build_record(snap, events)
        self._file.write(json.dumps(record) + '\n')

    def flush(self) -> None:
        """Flush buffered output to disk."""
        self._file.flush()

    def close(self) -> None:
        """Close the output file."""
        self._file.close()
        log.info('feature logger closed')

    def _build_record(self, snap: AudioSnapshot,
                      events: list[TimestampedEvent]) -> dict[str, Any]:
        """Build a JSON-serializable record from the current state."""
        record: dict[str, Any] = {}
        record['t'] = time.time()

        fields = self._fields

        if fields is None or 'mode' in fields:
            record['mode'] = snap.mode
        if fields is None or 'bpm' in fields:
            record['bpm'] = round(snap.bpm, 1)
        if fields is None or 'effective_bpm' in fields:
            record['effective_bpm'] = round(snap.effective_bpm, 1)
        if fields is None or 'tempo_confidence' in fields:
            record['tempo_confidence'] = round(snap.tempo_confidence, 3)
        if fields is None or 'percussiveness' in fields:
            record['percussiveness'] = round(snap.percussiveness, 3)
        if fields is None or 'section_change' in fields:
            record['section_change'] = round(snap.section_change, 4)
        if fields is None or 'break_intensity' in fields:
            record['break_intensity'] = round(snap.break_intensity, 3)
        if fields is None or 'centroid' in fields:
            record['centroid'] = round(snap.centroid, 1)
        if fields is None or 'centroid_delta' in fields:
            record['centroid_delta'] = round(snap.centroid_delta, 1)

        if fields is None or 'bands' in fields:
            bands: dict[str, dict[str, float]] = {}
            for name, bs in snap.bands.items():
                bd: dict[str, float] = {
                    'rms': round(bs.rms, 4),
                }
                if bs.onset_density > 0 or bs.density_delta != 0:
                    bd['onset_density'] = round(bs.onset_density, 2)
                    bd['density_delta'] = round(bs.density_delta, 3)
                bands[name] = bd
            record['bands'] = bands

        if fields is None or 'events' in fields:
            if events:
                record['events'] = [
                    {'kind': te.event.kind, 'energy': round(te.event.energy, 2)}
                    for te in events
                ]

        return record
