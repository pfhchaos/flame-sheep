"""
Shared test fixtures for flame-sheep tests.
"""

import sys
from pathlib import Path

_tests_dir = str(Path(__file__).resolve().parent)
_audio_tests = str(Path(__file__).resolve().parent.parent / 'flame_sheep_audio' / 'tests')

# Ensure both test directories are importable. NOTE: _audio_tests is NOT
# dead despite living in flame_sheep_audio/ — `audio_helpers.py` (imported
# below for "Re-export audio helpers for integration tests") lives only at
# flame_sheep_audio/tests/audio_helpers.py, not in this directory. Removing
# this breaks that import. (Tried removing it during the 0.9-prep pass;
# `from audio_helpers import ...` below failed immediately. Left in place.)
for _p in [_tests_dir, _audio_tests]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Package-import sanity check
#
# A test directory named after an installed package (e.g.
# `tests/wallpaper_ml/`) can shadow the real package on sys.path. When a
# test file then catches its own ImportError silently (a common
# `try: import VkCompute; except: HAS_GPU = False` pattern), every test
# in the module gets marked SKIPPED with no signal that the regression
# coverage is gone. Hit on 2026-05-31: tests/wallpaper_ml/__init__.py
# had been quietly skipping all GRU/Linear numerical-correctness tests
# for an unknown duration.
#
# This check resolves each critical package once at collection time and
# fails loudly if it didn't come from where we expect. Belt to the
# `--import-mode=importlib` suspenders in pyproject.toml — that
# structurally prevents the auto-prepend that allows shadowing in the
# first place; this assertion catches drift (e.g. a stray sys.path edit
# in a future conftest) before it silently rots a test suite.
# ---------------------------------------------------------------------------

def _verify_package_imports():
    import importlib
    _repo = Path(__file__).resolve().parent.parent
    expected = {
        'wallpaper_ml':      _repo / 'wallpaper_ml' / 'src' / 'wallpaper_ml',
        'flame_sheep':       _repo / 'flame_sheep',
        'flame_sheep_audio': _repo / 'flame_sheep_audio' / 'src' / 'flame_sheep_audio',
        'viz_authoring':     _repo / 'viz_authoring' / 'src' / 'viz_authoring',
    }
    for name, expected_path in expected.items():
        try:
            mod = importlib.import_module(name)
        except ImportError:
            # Package legitimately unavailable (e.g., extras not installed)
            # is fine — we only catch the shadowing case.
            continue
        actual_path = Path(mod.__file__).resolve().parent
        if actual_path != expected_path.resolve():
            raise RuntimeError(
                f"Package {name!r} resolves to {actual_path} but expected "
                f"{expected_path.resolve()}. A test directory may be "
                f"shadowing the installed package — check for a stray "
                f"__init__.py at tests/{name}/ or similar."
            )

_verify_package_imports()


# ---------------------------------------------------------------------------
# Minimum-test-count guard
#
# Third layer of the silent-regression net (alongside --import-mode=importlib
# and the package-shadowing check above). Catches the class regardless of
# mechanism: import errors, accidental skip markers, broken plugins,
# fixture scope errors, anything that quietly shrinks the suite.
#
# Only enforced when the full default suite is being collected — running a
# subset (e.g. `pytest wallpaper_ml/tests/test_rnn_shaders.py` during dev)
# is exempt. Bump _MIN_TEST_COUNT when intentionally adding tests; failing
# means something silently regressed, NOT that the floor needs to move.
# ---------------------------------------------------------------------------

# Bump when intentionally adding tests. As of 2026-05-31 (after restoring
# the silently-skipped wallpaper_ml suite + adding the seq-backward
# regression test + shader-health tests): 1715 non-slow tests collect.
# 1700 gives ~15 tests of headroom for legitimate removal while still
# catching any larger silent loss.
_MIN_TEST_COUNT = 1700


def pytest_collection_modifyitems(config, items):
    import pytest as _pytest
    # User filtered with -k → subset run, don't enforce.
    if config.option.keyword:
        return
    # User filtered with -m to something other than the default → subset run.
    if config.option.markexpr and config.option.markexpr != "not slow":
        return
    # If user passed paths different from the testpaths defaults, they're
    # selecting a subset explicitly.
    testpaths = list(config.getini('testpaths'))
    if list(config.args) != testpaths:
        return
    if len(items) < _MIN_TEST_COUNT:
        raise _pytest.UsageError(
            f"Collected {len(items)} tests but expected at least "
            f"{_MIN_TEST_COUNT}. Probable silent regression: import "
            f"error, accidental skip marker, broken plugin, shadowed "
            f"package, etc. Investigate the missing tests before "
            f"adjusting the floor in tests/conftest.py."
        )


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


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    """Point the XDG data/config dirs at a fresh per-test temp location so no
    test reads or writes the real ~/.local/share/flame-sheep. Mirrors the
    fixture in flame_sheep_audio/tests/conftest.py (both trees need it — autouse
    only applies in the conftest's own directory tree).

    This isolates the AGC's persisted gain state: every AudioProcessor's
    AudioLevelAgc loads slow_rms from data_dir()/agc_state.json on construction
    and saves it back, so that on-disk file leaks gain calibration across tests
    AND across pytest runs — the cross-run non-determinism behind the flaky
    beat-detection tests and the tests/eval band_routing regression. A fresh
    empty dir per test => the AGC always bootstraps at target => deterministic.
    Production persistence is intentional and untouched; this only sandboxes
    the tests."""
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg-data'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg-config'))
    yield
