"""Visualization configuration — loads tuning constants from TOML.

Visual axis tuning constants live here. Audio engine has its own config
in the flame_sheep_audio package (loaded from audio.toml).

Config is loaded from (in order):
  1. Built-in defaults (DEFAULTS dict below)
  2. ~/.config/flame-sheep/config.toml (user overrides)

Call `cfg.reload()` to re-read from disk (e.g., on control pipe signal).
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from collections.abc import Callable

log = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / '.config' / 'flame-sheep' / 'config.toml'

DEFAULTS = {
    'audio_device': None,  # auto-detect, or string like "Companion Speaker"
    'max_fps': 60,
    'genome': {
        'drift_morph_speed': 0.001,
        'density_morph_scale': 0.003,
        'low_morph_pulse': 0.008,
        'strong_beat_threshold': 2.0,
        'break_decay': 0.97,
        'density_damping': 0.3,
        'centroid_swap_threshold': 50.0,
        'centroid_swap_density_gate': 0.5,   # low-band density below this enables centroid swap
        'rotation_speed': 0.02,
        'rotation_beat_boost': 0.04,
        'dwell_beats': 16,
        'morph_beats': 4,
    },
    'zoom': {
        'boost_max': 0.3,
        'decay': 0.95,
        'density_damping': 0.05,
        'density_decay_scale': 0.15,
    },
    'palette': {
        'drift_morph_speed': 0.001,
        'density_damping': 0.2,
        'centroid_swap_density_gate': 0.5,  # backbeat density below this enables centroid swap
    },
    'drift': {
        'rms_threshold': 0.0001,
        'enter_frames': 480,
        'morph_speed': 0.001,
        'min_genome_distance': 0.15,
        'cycles_per_loop': 3,
    },
    'roles': {
        'downbeat': 'low',
        'backbeat': 'mid',
        'subdivision': 'high',
        'energy': 'subbass',
    },
    'scoring': {
        'store_histograms': False,  # dev-only: store raw histogram blobs in DB
        'render_sleep': 2.0,        # seconds between genomes in GPU render worker
        'render_size': 512,         # render resolution for scoring
        'store_cluster_detail': False,
    },
    'debug': {
        'panels': ['header', 'timeline', 'spectrum', 'band_metrics', 'break'],
        'timeline_seconds': 8.0,
        'window_width': 640,
        'window_height': 400,
    },
    'logging': {
        'feature_file': '~/.local/share/flame-sheep/features.jsonl',
        'feature_fields': [],
        'levels': {},
    },
    'monitors': {
        'default_diagonal': 27.0,  # fallback diagonal in inches
        'overscan': 0.02,          # fraction to overscan for skew edge coverage
        # Per-monitor overrides: [monitors.NAME]
        # diagonal = 43.0        # inches
        # skew = 11.0            # degrees, positive = angled right
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge override into base, recursively for nested dicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _to_namespace(d: dict[str, Any]) -> SimpleNamespace:
    """Convert nested dict to nested SimpleNamespace for dot access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


class Config:
    """Global configuration with dot-access and hot reload."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = dict(DEFAULTS)
        self._ns: SimpleNamespace = _to_namespace(self._data)
        self._reload_callbacks: list[Callable[[], None]] = []
        self._load_user_config()

    def _load_user_config(self) -> None:
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, 'rb') as f:
                    user = tomllib.load(f)
                self._data = _deep_merge(DEFAULTS, user)
                self._ns = _to_namespace(self._data)
                log.info(f'Loaded config from {CONFIG_PATH}')
            except Exception:
                log.exception(f'Failed to load {CONFIG_PATH}, using defaults')

    def on_reload(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called after config is reloaded."""
        self._reload_callbacks.append(callback)

    def reload(self) -> None:
        """Re-read config from disk and notify listeners."""
        self._data = dict(DEFAULTS)
        self._ns = _to_namespace(self._data)
        self._load_user_config()
        log.info('Config reloaded')
        for cb in self._reload_callbacks:
            cb()

    def __getattr__(self, name: str) -> Any:
        # Delegate to namespace for dot access: cfg.detection.low_threshold
        return getattr(self._ns, name)


# Global config instance
cfg = Config()
