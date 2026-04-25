"""Configuration system — loads tuning constants from TOML.

All audio and visual tuning constants live here. Components read from
the global `cfg` object instead of defining their own class-level constants.

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
    'detection': {
        'base_threshold': 1.5,
        'kick_threshold': 3.5,
        'cooldown_frames': 12,
        'kick_cooldown_frames': 8,
        'kick_cooldown_beat_fraction': 0.4,
        'stability_scaling': 1.0,
        'sharpness': 3.0,
        'min_flux': 1e-7,
    },
    'stability': {
        'fast_alpha': 0.95,
        'slow_alpha': 0.995,
    },
    'energy': {
        'rms_alpha': 0.9,
        'centroid_alpha': 0.85,
        'percussiveness_alpha': 0.92,
    },
    'density': {
        'window': 1.0,
        'delta_window': 2.0,
        'alpha': 0.9,
    },
    'tempo': {
        'min_bpm': 60,
        'max_bpm': 400,
        'default_bpm': 120,
    },
    'breaks': {
        'enabled': True,
        'quiet_threshold_frames': 60,
        'bass_quiet_threshold_frames': 45,
        'drop_energy_ratio': 0.15,
        'subbass_drop_ratio': 0.10,
        'min_kicks_before_break': 8,
        'cooldown_seconds': 15.0,
    },
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
    'adaptive': {
        'enabled': False,
        'update_interval': 9,
        'anchor_strength': 0.3,
        'repulsion_strength': 0.1,
        'flux_pull_strength': 0.5,
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
