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
