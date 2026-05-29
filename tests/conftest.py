"""
Shared test fixtures for flame-sheep tests.
"""

import sys
from pathlib import Path

_tests_dir = str(Path(__file__).resolve().parent)
_audio_tests = str(Path(__file__).resolve().parent.parent / 'flame_sheep_audio' / 'tests')

# Ensure both test directories are importable
for _p in [_tests_dir, _audio_tests]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-export visualizer helpers
from viz_helpers import trivial_genome, FakeClock  # noqa: F401, E402

# Re-export audio helpers for integration tests
from audio_helpers import make_processor, make_sine, make_silence, make_impulse, feed_audio  # noqa: F401, E402

import pytest  # noqa: E402
from flame_sheep_audio.config import cfg as _audio_cfg  # noqa: E402


@pytest.fixture(autouse=True)
def _hermetic_audio_config():
    """Reset the global audio cfg to DEFAULTS before each test, undoing
    any user overrides loaded at import time from
    ~/.config/flame-sheep/audio.toml. Mirrors the fixture in
    flame_sheep_audio/tests/conftest.py — both test trees need it
    independently since pytest autouse fixtures only apply in the
    conftest's own directory tree."""
    _audio_cfg.reset_to_defaults()
    yield
    _audio_cfg.reset_to_defaults()
