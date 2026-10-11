"""Tests for GpuLoadSampler's clock + power telemetry (gpu_load.py).

Busy-fraction alone is a misleading GPU-load signal (100% busy at the
frequency floor looks identical to 100% busy at boost). These tests
cover the honest signals added on top of it: actual-vs-max GT clock
and board power draw, read from sysfs/hwmon, plus the two derived
fractions (throughput, power/TDP).

All tests point the module's sysfs discovery at a fake tree under
tmp_path (monkeypatching _DRM_BASE) rather than touching the real
/sys/class/drm — hermetic, and deterministic regardless of what the
real GPU happens to be doing. The Xe PMU is independently forced
unavailable (_discover_xe_pmu -> None) so busy_fraction tests don't
depend on real hardware/perf_event_paranoid either.
"""
from __future__ import annotations

import time

import pytest

from flame_sheep.scheduler import gpu_load
from flame_sheep.scheduler.gpu_load import GpuLoadSampler, GpuTelemetrySample


class _FakeClock:
    """Deterministic stand-in for time.perf_counter() — tests control
    elapsed time exactly instead of racing a real sleep()."""

    def __init__(self, start: float = 100.0):
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(gpu_load.time, 'perf_counter', clock)
    return clock


@pytest.fixture
def no_pmu(monkeypatch):
    """Force the Xe PMU branch unavailable so tests are deterministic
    regardless of the real box's perf_event_paranoid / driver state."""
    monkeypatch.setattr(gpu_load, '_discover_xe_pmu', lambda: None)


@pytest.fixture
def fake_sysfs(tmp_path, monkeypatch):
    """Build a fake /sys/class/drm tree with:
      - card0: a DECOY card whose driver is NOT xe (must be skipped)
      - card0-DP-1: a connector pseudo-entry (must be skipped by name)
      - card1: the xe-driven card, with a real freq0 dir and two
        hwmon nodes — one named 'xe' (the real one) and one decoy
        named 'coretemp' — to prove hwmon selection is by name, not
        "first entry found".
    Ordering (card0 before card1, hwmon0 decoy before hwmon2 xe) is
    deliberately adversarial to the sorted()-iteration in the
    discovery helpers, so these tests fail if that code ever
    regresses to "take the first thing on disk".
    """
    drm = tmp_path / 'drm'

    decoy_device = drm / 'card0' / 'device'
    decoy_device.mkdir(parents=True)
    decoy_driver_target = tmp_path / 'drivers' / 'i915'
    decoy_driver_target.mkdir(parents=True)
    (decoy_device / 'driver').symlink_to(decoy_driver_target)

    (drm / 'card0-DP-1' / 'device').mkdir(parents=True)

    xe_device = drm / 'card1' / 'device'
    freq_dir = xe_device / 'tile0' / 'gt0' / 'freq0'
    freq_dir.mkdir(parents=True)
    (freq_dir / 'act_freq').write_text('1200\n')
    (freq_dir / 'max_freq').write_text('2400\n')

    xe_driver_target = tmp_path / 'drivers' / 'xe'
    xe_driver_target.mkdir(parents=True)
    (xe_device / 'driver').symlink_to(xe_driver_target)

    decoy_hwmon = xe_device / 'hwmon' / 'hwmon0'
    decoy_hwmon.mkdir(parents=True)
    (decoy_hwmon / 'name').write_text('coretemp\n')

    xe_hwmon = xe_device / 'hwmon' / 'hwmon2'
    xe_hwmon.mkdir(parents=True)
    (xe_hwmon / 'name').write_text('xe\n')
    (xe_hwmon / 'energy2_input').write_text('1000000\n')  # 1,000,000 uJ
    (xe_hwmon / 'power2_max').write_text('190000000\n')   # 190 W cap

    monkeypatch.setattr(gpu_load, '_DRM_BASE', drm)
    return {
        'xe_device': xe_device,
        'freq_dir': freq_dir,
        'xe_hwmon': xe_hwmon,
    }


def test_discover_xe_drm_device_skips_non_xe_card_and_connector(fake_sysfs):
    device = gpu_load._discover_xe_drm_device()
    assert device == fake_sysfs['xe_device']


def test_discover_freq_dir_globs_tile_gt(fake_sysfs):
    freq_dir = gpu_load._discover_freq_dir(fake_sysfs['xe_device'])
    assert freq_dir == fake_sysfs['freq_dir']


def test_discover_xe_hwmon_dir_matches_by_name_field(fake_sysfs):
    hwmon_dir = gpu_load._discover_xe_hwmon_dir(fake_sysfs['xe_device'])
    assert hwmon_dir == fake_sysfs['xe_hwmon']


def test_sampler_wires_up_freq_and_power_sources(fake_sysfs, no_pmu):
    sampler = GpuLoadSampler()
    assert sampler._freq_dir == fake_sysfs['freq_dir']
    assert sampler._energy_input_path == fake_sysfs['xe_hwmon'] / 'energy2_input'
    assert sampler._power_input_path is None  # this kernel has no power*_input
    assert sampler._power_max_uw == 190_000_000
    sampler.close()


def test_read_freq_mhz(fake_sysfs, no_pmu):
    sampler = GpuLoadSampler()
    act, mx = sampler._read_freq_mhz()
    assert act == 1200.0
    assert mx == 2400.0
    sampler.close()


def test_read_power_w_integrates_energy_delta_over_time(
        fake_sysfs, no_pmu, fake_clock):
    sampler = GpuLoadSampler()  # primes prev_energy_uj=1_000_000 @ t=100.0

    # Advance 2.0s and bump the energy counter by 200,000,000 uJ ->
    # 100,000,000 uJ/s == 100 W.
    fake_clock.advance(2.0)
    energy_file = fake_sysfs['xe_hwmon'] / 'energy2_input'
    energy_file.write_text(str(1_000_000 + 200_000_000) + '\n')

    power_w = sampler._read_power_w()
    assert power_w == pytest.approx(100.0)
    sampler.close()


def test_read_power_w_none_on_energy_counter_rollover(
        fake_sysfs, no_pmu, fake_clock):
    sampler = GpuLoadSampler()

    fake_clock.advance(1.0)
    energy_file = fake_sysfs['xe_hwmon'] / 'energy2_input'
    # Value went DOWN -> counter wrapped. Must not report negative power.
    energy_file.write_text('500\n')

    assert sampler._read_power_w() is None
    sampler.close()


def test_sample_telemetry_derives_throughput_and_power_fractions(
        fake_sysfs, no_pmu, fake_clock, monkeypatch):
    sampler = GpuLoadSampler()
    # Drive busy_fraction deterministically — the PMU plumbing itself
    # is covered by the existing sample() contract; here we're
    # verifying sample_telemetry()'s NEW derivation math.
    monkeypatch.setattr(sampler, 'sample', lambda: (0.8, 0.25))

    fake_clock.advance(2.0)
    energy_file = fake_sysfs['xe_hwmon'] / 'energy2_input'
    energy_file.write_text(str(1_000_000 + 200_000_000) + '\n')  # -> 100W

    t = sampler.sample_telemetry()
    assert isinstance(t, GpuTelemetrySample)
    assert t.busy_fraction == 0.8
    assert t.window_s == 0.25
    assert t.act_freq_mhz == 1200.0
    assert t.max_freq_mhz == 2400.0
    # throughput = busy * act/max = 0.8 * (1200/2400) = 0.4
    assert t.throughput_fraction == pytest.approx(0.4)
    assert t.power_w == pytest.approx(100.0)
    assert t.tdp_w == pytest.approx(190.0)
    # power_fraction = power_w / tdp_w = 100/190
    assert t.power_fraction == pytest.approx(100.0 / 190.0)
    sampler.close()


def test_sample_telemetry_degrades_to_none_when_sysfs_missing(
        tmp_path, monkeypatch, no_pmu):
    """No fake sysfs tree at all: every new field degrades to None
    rather than raising — same graceful-degradation contract the PMU
    branch already has on perf_event_paranoid denial."""
    monkeypatch.setattr(gpu_load, '_DRM_BASE', tmp_path / 'nonexistent')
    sampler = GpuLoadSampler()

    t = sampler.sample_telemetry()
    assert t.busy_fraction is None
    assert t.window_s == 0.0
    assert t.act_freq_mhz is None
    assert t.max_freq_mhz is None
    assert t.throughput_fraction is None
    assert t.power_w is None
    # tdp_w still gets the hardcoded A770 fallback even with no sysfs cap.
    assert t.tdp_w == gpu_load._A770_TDP_W
    assert t.power_fraction is None
    sampler.close()
