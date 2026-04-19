"""Shared types for audio analysis."""

from dataclasses import dataclass


@dataclass
class BeatEvent:
    kind: str       # 'kick' | 'snare' | 'hihat'
    energy: float   # normalized 0..1, how strong the onset was
