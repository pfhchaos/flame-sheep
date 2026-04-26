"""Visualization configuration — loads tuning constants from TOML.

Visual axis tuning constants live here. Audio engine has its own config
in the flame_sheep_audio package (loaded from audio.toml).

Config is loaded from (in order):
  1. Built-in defaults (DEFAULTS dict below)
  2. ~/.config/flame-sheep/config.toml (user overrides)

Call `cfg.reload()` to re-read from disk (e.g., on control pipe signal).
"""

import logging
import tomllib
from pathlib import Path
from types import SimpleNamespace

log = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / '.config' / 'flame-sheep' / 'config.toml'

DEFAULTS = {
    'genome': {
        'drift_morph_speed': 0.001,
        'density_morph_scale': 0.003,
        'kick_morph_pulse': 0.008,
        'strong_beat_threshold': 2.0,
        'break_decay': 0.97,
        'density_damping': 0.3,
        'centroid_swap_threshold': 500.0,
        'centroid_swap_perc_gate': 0.15,
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
    },
    'drift': {
        'rms_threshold': 0.0001,
        'enter_frames': 480,
        'morph_speed': 0.001,
        'min_genome_distance': 0.15,
        'cycles_per_loop': 3,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursively for nested dicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _to_namespace(d: dict) -> SimpleNamespace:
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

    def __init__(self):
        self._data = dict(DEFAULTS)
        self._ns = _to_namespace(self._data)
        self._load_user_config()

    def _load_user_config(self):
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, 'rb') as f:
                    user = tomllib.load(f)
                self._data = _deep_merge(DEFAULTS, user)
                self._ns = _to_namespace(self._data)
                log.info(f'Loaded config from {CONFIG_PATH}')
            except Exception:
                log.exception(f'Failed to load {CONFIG_PATH}, using defaults')

    def reload(self):
        """Re-read config from disk."""
        self._data = dict(DEFAULTS)
        self._ns = _to_namespace(self._data)
        self._load_user_config()
        log.info('Config reloaded')

    def __getattr__(self, name):
        # Delegate to namespace for dot access: cfg.detection.kick_threshold
        return getattr(self._ns, name)


# Global config instance
cfg = Config()
