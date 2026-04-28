"""Scrolling timeline data structure for onset visualization."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from flame_sheep.orchestrator import TimestampedEvent


@dataclass(slots=True)
class OnsetMark:
    """One onset dot on the timeline."""
    time: float       # perf_counter timestamp
    energy: float     # 0..1 onset strength
    band: str         # detection band name


class TimelineBuffer:
    """Ring buffer of recent onsets for the scrolling piano-roll display.

    Stores per-band deques of OnsetMark. Old marks are auto-evicted
    by deque maxlen. Use marks_in_window() to get marks within
    a time window for rendering.
    """

    def __init__(self, band_names: tuple[str, ...] | list[str],
                 max_seconds: float = 8.0) -> None:
        self._band_names: tuple[str, ...] = tuple(band_names)
        self._max_seconds: float = max_seconds
        self._marks: dict[str, deque[OnsetMark]] = {
            name: deque(maxlen=2000) for name in self._band_names
        }

    @property
    def band_names(self) -> tuple[str, ...]:
        return self._band_names

    @property
    def max_seconds(self) -> float:
        return self._max_seconds

    def push(self, te: TimestampedEvent) -> None:
        """Add an onset from a drained TimestampedEvent."""
        if te.event.kind in self._marks:
            self._marks[te.event.kind].append(
                OnsetMark(time=te.timestamp, energy=te.event.energy,
                          band=te.event.kind)
            )

    def marks_in_window(self, now: float | None = None) -> dict[str, list[OnsetMark]]:
        """Return marks within [now - max_seconds, now] per band."""
        if now is None:
            now = time.perf_counter()
        cutoff = now - self._max_seconds
        return {
            name: [m for m in marks if m.time >= cutoff]
            for name, marks in self._marks.items()
        }

    def clear(self) -> None:
        """Clear all marks."""
        for marks in self._marks.values():
            marks.clear()
