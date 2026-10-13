"""ZoomAxis — downbeat-driven zoom pulse (UNIT spec).

Covers ``axes/zoom_axis.py`` in isolation: a ZoomAxis + a RoleMapper, no
FlameSheepCore, no orchestrator, no synthetic audio. Previously ZoomAxis only
had the thin ``TestZoomAxisUnit`` smoke tests in test_component_isolation.py
(boost/no-boost/decay/bounded) and the end-to-end TestZoomPulse in
test_core.py; neither pins the contribute() scaling contract, and both route
through heavier fixtures. This file is the pure-axis behavior spec.

Behavior spec (the contract):
  * tick() with a DOWNBEAT-role event (default mapping: the 'low' band) RAISES
    zoom_boost by energy * 0.15, CLAMPED at ZOOM_BOOST_MAX.
  * tick() with a NON-downbeat event (e.g. 'high'/'mid') does NOT raise it.
  * tick() with no event DECAYS zoom_boost toward 0 (× ZOOM_DECAY per tick,
    snapping to 0 below 0.001).
  * contribute() multiplies frame.genome.zoom by (1 + zoom_boost * scale),
    where scale = 1/max(zoom, 0.1) — so an equal boost produces a LARGER
    relative zoom change at a smaller base zoom (inverse-zoom scaling, keeps
    the visual pulse perceptually constant across zoom levels).

MUTATION GUARDS (each reddens a named test):
  * neuter the boost line (zoom_axis.py, the `self.zoom_boost = min(...)`)
      -> test_downbeat_raises_zoom_boost
  * drop the `event.kind == band_for_role(DOWNBEAT)` gate (always boost)
      -> test_non_downbeat_does_not_boost
  * drop the `self.zoom_boost *= ZOOM_DECAY` decay line
      -> test_tick_without_event_decays
  * neuter contribute's `frame.genome.zoom *= (1 + ...)` multiply
      -> test_contribute_scales_genome_zoom
  * flatten contribute's inverse scale (`scale = 1.0`)
      -> test_contribute_inverse_zoom_scaling
"""
import numpy as np
import pytest

from flame_sheep.axes.zoom_axis import ZoomAxis
from flame_sheep.runtime import RoleMapper, FlameSheepCore
from flame_sheep_audio import BeatEvent, AudioState

from viz_helpers import trivial_genome


def _axis():
    return ZoomAxis(role=RoleMapper())  # default mapping: downbeat -> 'low'


def _audio(events=None):
    """Minimal AudioState — ZoomAxis.tick only reads .events."""
    return AudioState(events=events or [])


def _frame(genome):
    return FlameSheepCore.FrameState(
        genome=genome, palette=None,
        spectrum=np.zeros(108, dtype=np.float32),
        brightness=1.0, iterations=100)


# --- (a) downbeat raises the boost, bounded by the max ---------------------

def test_downbeat_raises_zoom_boost():
    axis = _axis()
    assert axis.zoom_boost == 0.0
    axis.tick(_audio(events=[BeatEvent('low', 1.0)]), 1 / 60, 0.0)
    # energy 1.0 * 0.15 = 0.15 (below the 0.3 cap), minus nothing (started at 0)
    assert axis.zoom_boost == pytest.approx(1.0 * 0.15)


def test_downbeat_boost_is_bounded():
    axis = _axis()
    # Many max-energy downbeats must not push boost past ZOOM_BOOST_MAX.
    for _ in range(100):
        axis.tick(_audio(events=[BeatEvent('low', 1.0)]), 1 / 60, 0.0)
    assert axis.zoom_boost <= axis.ZOOM_BOOST_MAX + 1e-9
    assert axis.zoom_boost == pytest.approx(axis.ZOOM_BOOST_MAX)


# --- (b) a non-downbeat event does NOT boost -------------------------------

@pytest.mark.parametrize("band", ["mid", "high", "subbass"])
def test_non_downbeat_does_not_boost(band):
    axis = _axis()
    axis.tick(_audio(events=[BeatEvent(band, 1.0)]), 1 / 60, 0.0)
    assert axis.zoom_boost == 0.0


# --- (c) a tick with no event decays toward 0 ------------------------------

def test_tick_without_event_decays():
    axis = _axis()
    axis.tick(_audio(events=[BeatEvent('low', 1.0)]), 1 / 60, 0.0)
    peak = axis.zoom_boost
    assert peak > 0.0
    axis.tick(_audio(), 1 / 60, 0.0)                 # no event -> pure decay
    assert axis.zoom_boost < peak
    assert axis.zoom_boost == pytest.approx(peak * axis.ZOOM_DECAY)


def test_tick_decays_to_zero_eventually():
    axis = _axis()
    axis.tick(_audio(events=[BeatEvent('low', 1.0)]), 1 / 60, 0.0)
    for _ in range(200):
        axis.tick(_audio(), 1 / 60, 0.0)
    assert axis.zoom_boost == 0.0                    # snaps to 0 below 0.001


# --- (d) contribute() scales genome.zoom, inverse-weighted -----------------

def test_contribute_scales_genome_zoom():
    axis = _axis()
    axis.zoom_boost = 0.2
    g = trivial_genome(0)
    g.zoom = 1.0
    axis.contribute(_frame(g))
    # scale at zoom=1.0 is 1.0, so zoom *= (1 + 0.2 * 1.0) = 1.2
    assert g.zoom == pytest.approx(1.2)


def test_contribute_inverse_zoom_scaling():
    """Equal boost -> LARGER relative zoom change at a smaller base zoom."""
    boost = 0.2

    axis_big = _axis()
    axis_big.zoom_boost = boost
    g_big = trivial_genome(0)
    g_big.zoom = 1.0
    axis_big.contribute(_frame(g_big))
    ratio_big = g_big.zoom / 1.0

    axis_small = _axis()
    axis_small.zoom_boost = boost
    g_small = trivial_genome(1)
    g_small.zoom = 0.25
    axis_small.contribute(_frame(g_small))
    ratio_small = g_small.zoom / 0.25

    # 1/max(0.25,0.1)=4 vs 1/1=1 -> the small-zoom pulse is relatively stronger.
    assert ratio_small > ratio_big
