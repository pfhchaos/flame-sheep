"""
Tests for RoleMapper: band-to-role routing, event translation.
"""

import pytest

from flame_sheep.role_mapper import (
    RoleMapper, DEFAULT_MAPPING,
    DOWNBEAT, BACKBEAT, SUBDIVISION, ENERGY, ALL_ROLES,
)
from flame_sheep_audio._types import BeatEvent, BandState


class TestDefaultMapping:
    def test_all_roles_mapped(self):
        rm = RoleMapper()
        for role in ALL_ROLES:
            assert rm.has_role(role)

    def test_default_routing(self):
        rm = RoleMapper()
        assert rm.band_for_role(SUBDIVISION) == 'low'
        assert rm.band_for_role(DOWNBEAT) == 'mid'
        assert rm.band_for_role(BACKBEAT) == 'high'
        assert rm.band_for_role(ENERGY) == 'subbass'

    def test_reverse_lookup(self):
        rm = RoleMapper()
        assert rm.role_for_band('low') == SUBDIVISION
        assert rm.role_for_band('mid') == DOWNBEAT
        assert rm.role_for_band('high') == BACKBEAT

    def test_unknown_band_returns_none(self):
        rm = RoleMapper()
        assert rm.role_for_band('nonexistent') is None


class TestCustomMapping:
    def test_custom_overrides_default(self):
        custom = {DOWNBEAT: 'low', BACKBEAT: 'mid', SUBDIVISION: 'high', ENERGY: 'low'}
        rm = RoleMapper(custom)
        assert rm.band_for_role(DOWNBEAT) == 'low'
        assert rm.band_for_role(BACKBEAT) == 'mid'

    def test_missing_role_raises(self):
        rm = RoleMapper({DOWNBEAT: 'low'})
        with pytest.raises(KeyError):
            rm.band_for_role(BACKBEAT)

    def test_has_role_false_for_unmapped(self):
        rm = RoleMapper({DOWNBEAT: 'low'})
        assert rm.has_role(DOWNBEAT)
        assert not rm.has_role(BACKBEAT)


class TestEventRouting:
    def test_role_for_beat_event(self):
        rm = RoleMapper()
        event = BeatEvent(kind='low', energy=0.8)
        assert rm.role_for_event(event) == SUBDIVISION

    def test_role_for_unknown_event(self):
        rm = RoleMapper()
        event = BeatEvent(kind='unknown_band', energy=0.5)
        assert rm.role_for_event(event) is None


class TestBandState:
    def test_returns_band_state(self):
        rm = RoleMapper()

        class FakeAudio:
            bands = {'low': BandState(rms=0.9), 'mid': BandState(rms=0.1)}

        audio = FakeAudio()
        state = rm.band_state(audio, SUBDIVISION)
        assert state.rms == 0.9

    def test_missing_band_returns_default(self):
        rm = RoleMapper()

        class FakeAudio:
            bands = {}

        state = rm.band_state(FakeAudio(), DOWNBEAT)
        assert state.rms == 0.0

    def test_unmapped_role_returns_default(self):
        rm = RoleMapper({DOWNBEAT: 'low'})

        class FakeAudio:
            bands = {'low': BandState(rms=0.5)}

        state = rm.band_state(FakeAudio(), BACKBEAT)
        assert state.rms == 0.0


class TestMappingProperty:
    def test_returns_copy(self):
        rm = RoleMapper()
        m = rm.mapping
        m['garbage'] = 'test'
        # Original should be unaffected
        assert 'garbage' not in rm.mapping
