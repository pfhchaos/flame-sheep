"""Tests for the configuration system.

Covers deep merge, namespace conversion, dot access, defaults,
reload behavior, and user config override.
"""

import tempfile
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from flame_sheep.config import _deep_merge, _to_namespace, Config, DEFAULTS


class TestDeepMerge:
    """Tests for recursive dict merging."""

    def test_flat_override(self):
        base = {'a': 1, 'b': 2}
        over = {'b': 99}
        assert _deep_merge(base, over) == {'a': 1, 'b': 99}

    def test_nested_override(self):
        base = {'x': {'a': 1, 'b': 2}}
        over = {'x': {'b': 99}}
        assert _deep_merge(base, over) == {'x': {'a': 1, 'b': 99}}

    def test_add_new_key(self):
        base = {'a': 1}
        over = {'b': 2}
        assert _deep_merge(base, over) == {'a': 1, 'b': 2}

    def test_add_nested_key(self):
        base = {'x': {'a': 1}}
        over = {'x': {'b': 2}}
        assert _deep_merge(base, over) == {'x': {'a': 1, 'b': 2}}

    def test_empty_override(self):
        base = {'a': 1}
        assert _deep_merge(base, {}) == {'a': 1}

    def test_empty_base(self):
        over = {'a': 1}
        assert _deep_merge({}, over) == {'a': 1}

    def test_override_dict_with_scalar(self):
        """Scalar override replaces an entire sub-dict."""
        base = {'x': {'a': 1}}
        over = {'x': 42}
        assert _deep_merge(base, over) == {'x': 42}

    def test_override_scalar_with_dict(self):
        base = {'x': 42}
        over = {'x': {'a': 1}}
        assert _deep_merge(base, over) == {'x': {'a': 1}}

    def test_does_not_mutate_base(self):
        base = {'x': {'a': 1}}
        _deep_merge(base, {'x': {'a': 99}})
        assert base == {'x': {'a': 1}}


class TestToNamespace:
    """Tests for dict-to-namespace conversion."""

    def test_flat(self):
        ns = _to_namespace({'a': 1, 'b': 'hello'})
        assert ns.a == 1
        assert ns.b == 'hello'

    def test_nested(self):
        ns = _to_namespace({'x': {'y': 42}})
        assert ns.x.y == 42

    def test_missing_attr_raises(self):
        ns = _to_namespace({'a': 1})
        with pytest.raises(AttributeError):
            _ = ns.nonexistent


class TestConfigDefaults:
    """Verify the global config loads with sane defaults."""

    def test_defaults_load(self):
        """Config with no user file should have all default sections."""
        with mock.patch('flame_sheep.config.CONFIG_PATH',
                        Path('/nonexistent/path/config.toml')):
            c = Config()
        assert c.genome.drift_morph_speed == DEFAULTS['genome']['drift_morph_speed']
        assert c.zoom.boost_max == DEFAULTS['zoom']['boost_max']
        assert c.drift.rms_threshold == DEFAULTS['drift']['rms_threshold']

    def test_all_sections_accessible(self):
        with mock.patch('flame_sheep.config.CONFIG_PATH',
                        Path('/nonexistent/path/config.toml')):
            c = Config()
        for section in DEFAULTS:
            assert hasattr(c, section), f'Missing section: {section}'

    def test_all_keys_accessible(self):
        with mock.patch('flame_sheep.config.CONFIG_PATH',
                        Path('/nonexistent/path/config.toml')):
            c = Config()
        for section, values in DEFAULTS.items():
            if not isinstance(values, dict):
                # Top-level scalar keys (e.g. audio_device)
                assert getattr(c, section) == values, \
                    f'cfg.{section} != {values}'
                continue
            ns = getattr(c, section)
            for key, default_val in values.items():
                actual = getattr(ns, key)
                if isinstance(default_val, dict):
                    # Nested dicts become SimpleNamespace; compare as dicts
                    assert vars(actual) == default_val, \
                        f'cfg.{section}.{key} != {default_val}'
                else:
                    assert actual == default_val, \
                        f'cfg.{section}.{key} != {default_val}'


class TestConfigUserOverride:
    """Verify user TOML overrides merge correctly."""

    def test_partial_override(self):
        toml_content = textwrap.dedent("""\
            [genome]
            drift_morph_speed = 99.0
        """)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            with mock.patch('flame_sheep.config.CONFIG_PATH', path):
                c = Config()
            # Overridden value
            assert c.genome.drift_morph_speed == 99.0
            # Non-overridden value preserved
            assert c.genome.kick_morph_pulse == \
                DEFAULTS['genome']['kick_morph_pulse']
            # Other section untouched
            assert c.zoom.boost_max == DEFAULTS['zoom']['boost_max']
        finally:
            path.unlink()

    def test_invalid_toml_falls_back_to_defaults(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write('this is not valid [[[ toml')
            f.flush()
            path = Path(f.name)

        try:
            with mock.patch('flame_sheep.config.CONFIG_PATH', path):
                c = Config()
            # Should fall back to defaults
            assert c.genome.drift_morph_speed == \
                DEFAULTS['genome']['drift_morph_speed']
        finally:
            path.unlink()


class TestConfigReload:
    """Test hot-reload behavior."""

    def test_reload_picks_up_changes(self):
        toml_v1 = '[genome]\ndrift_morph_speed = 10.0\n'
        toml_v2 = '[genome]\ndrift_morph_speed = 20.0\n'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_v1)
            f.flush()
            path = Path(f.name)

        try:
            with mock.patch('flame_sheep.config.CONFIG_PATH', path):
                c = Config()
                assert c.genome.drift_morph_speed == 10.0

                # Update the file
                path.write_text(toml_v2)
                c.reload()
                assert c.genome.drift_morph_speed == 20.0
        finally:
            path.unlink()

    def test_reload_fires_callbacks(self):
        with mock.patch('flame_sheep.config.CONFIG_PATH',
                        Path('/nonexistent/path/config.toml')):
            c = Config()
        calls = []
        c.on_reload(lambda: calls.append('a'))
        c.on_reload(lambda: calls.append('b'))
        c.reload()
        assert calls == ['a', 'b']

    def test_reload_callbacks_fire_after_values_updated(self):
        toml_v1 = '[genome]\ndrift_morph_speed = 10.0\n'
        toml_v2 = '[genome]\ndrift_morph_speed = 20.0\n'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_v1)
            f.flush()
            path = Path(f.name)
        try:
            with mock.patch('flame_sheep.config.CONFIG_PATH', path):
                c = Config()
                observed = []
                c.on_reload(lambda: observed.append(c.genome.drift_morph_speed))
                path.write_text(toml_v2)
                c.reload()
                assert observed == [20.0]
        finally:
            path.unlink(missing_ok=True)

    def test_reload_resets_to_defaults_if_file_removed(self):
        toml = '[genome]\ndrift_morph_speed = 10.0\n'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml)
            f.flush()
            path = Path(f.name)

        try:
            with mock.patch('flame_sheep.config.CONFIG_PATH', path):
                c = Config()
                assert c.genome.drift_morph_speed == 10.0

                path.unlink()
                c.reload()
                assert c.genome.drift_morph_speed == \
                    DEFAULTS['genome']['drift_morph_speed']
        finally:
            path.unlink(missing_ok=True)


class TestAudioConfigReloadCallbacks:
    """Test reload callbacks on audio engine config."""

    def test_audio_reload_fires_callbacks(self):
        from flame_sheep_audio.config import Config as AudioConfig
        with mock.patch('flame_sheep_audio.config.CONFIG_PATH',
                        Path('/nonexistent/path/audio.toml')):
            c = AudioConfig()
        calls = []
        c.on_reload(lambda: calls.append('fired'))
        c.reload()
        assert calls == ['fired']
