"""
Shared test fixtures for flame-sheep tests.
"""

import sys
from pathlib import Path

_tests_dir = str(Path(__file__).resolve().parent)

# audio_helpers.py and viz_helpers.py are vendored in this directory —
# audio_helpers was copied in when flame_sheep_audio split out into its own
# repo, so this tree no longer reaches into a sibling package's tests. Only
# this directory needs to be importable.
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

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
    # Post-split, only flame_sheep lives in this repo; the three siblings are
    # external installed deps (their own repos now), so there are no sibling
    # dirs at the repo root to shadow — only flame_sheep is meaningful to
    # check here.
    expected = {
        'flame_sheep':       _repo / 'flame_sheep',
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

# Bump when intentionally adding tests. As of 2026-10-08 (after the 0.9 repo
# split moved flame_sheep_audio/wallpaper_ml/viz_authoring to their own repos
# with their own suites): 1012 non-slow tests collect in this repo.
# 1000 gives ~12 tests of headroom for legitimate removal while still
# catching any larger silent loss.
_MIN_TEST_COUNT = 1000


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
