"""Debug overlay panels — vertical stack of visualization strips."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from flame_sheep_audio._types import AudioSnapshot, BandState
from flame_sheep_audio._constants import FREQS
from ._timeline import TimelineBuffer

if TYPE_CHECKING:
    from flame_sheep_audio._band_config import BandConfig
    from ._draw import SolidRenderer
    from ._text import TextRenderer


# One Dark palette
_BG        = (0.15, 0.17, 0.20, 0.9)
_BG_ALT    = (0.18, 0.20, 0.24, 0.9)
_FG        = (0.67, 0.69, 0.73, 1.0)
_GREEN     = (0.60, 0.82, 0.46, 1.0)
_YELLOW    = (0.90, 0.79, 0.42, 1.0)
_RED       = (0.88, 0.40, 0.37, 1.0)
_BLUE      = (0.38, 0.68, 0.93, 1.0)
_PURPLE    = (0.78, 0.51, 0.85, 1.0)
_CYAN      = (0.34, 0.76, 0.76, 1.0)
_ORANGE    = (0.85, 0.55, 0.33, 1.0)
_GRAY      = (0.40, 0.42, 0.46, 1.0)
_DIM       = (0.30, 0.32, 0.36, 1.0)

# Per-band colors (cycles for arbitrary band counts)
_BAND_COLORS = [_BLUE, _PURPLE, _CYAN, _ORANGE, _GREEN, _YELLOW, _RED]


def _band_color(idx: int) -> tuple[float, float, float, float]:
    return _BAND_COLORS[idx % len(_BAND_COLORS)]


# Stable name->color mapping so timeline and spectrum brackets match
_band_color_cache: dict[str, tuple[float, float, float, float]] = {}


def _color_for_band(name: str, all_names: tuple[str, ...] | list[str]) -> tuple[float, float, float, float]:
    """Get a stable color for a band name (consistent across panels)."""
    if name not in _band_color_cache:
        idx = list(all_names).index(name) if name in all_names else len(_band_color_cache)
        _band_color_cache[name] = _band_color(idx)
    return _band_color_cache[name]


class Panel:
    """Base class for a panel strip."""

    def __init__(self, height: int = 40, visible: bool = True) -> None:
        self.height = height
        self.visible = visible

    def update(self, snap: AudioSnapshot, timeline: TimelineBuffer, dt: float) -> None:
        pass

    def render(self, draw: SolidRenderer, text: TextRenderer,
               x: int, y: int, w: int, h: int) -> None:
        pass


class HeaderPanel(Panel):
    """Mode, BPM, confidence, percussiveness."""

    def __init__(self) -> None:
        super().__init__(height=28)
        self._mode = 'idle'
        self._bpm = 0.0
        self._confidence = 0.0
        self._percussiveness = 0.0

    def update(self, snap: AudioSnapshot, timeline: TimelineBuffer, dt: float) -> None:
        self._mode = snap.mode
        self._bpm = snap.effective_bpm
        self._confidence = snap.tempo_confidence
        self._percussiveness = snap.percussiveness

    def render(self, draw: SolidRenderer, text: TextRenderer,
               x: int, y: int, w: int, h: int) -> None:
        draw.rect(x, y, w, h, _BG)

        # Mode badge
        mode_colors = {'beat': _GREEN, 'energy': _YELLOW, 'idle': _GRAY}
        mode_color = mode_colors.get(self._mode, _GRAY)
        text.draw(self._mode.upper(), x + 6, y + 6, mode_color, scale=2)

        # BPM
        bpm_x = x + 90
        bpm_str = f'{self._bpm:.1f} bpm'
        text.draw(bpm_str, bpm_x, y + 9, _FG)

        # Confidence bar
        conf_x = bpm_x + 90
        bar_w = 60
        text.draw('conf', conf_x, y + 9, _DIM)
        conf_x += 30
        draw.rect(conf_x, y + 9, bar_w, 10, _DIM)
        draw.rect(conf_x, y + 9, bar_w * self._confidence, 10, _BLUE)
        text.draw(f'{self._confidence:.2f}', conf_x + bar_w + 4, y + 9, _FG)

        # Percussiveness bar
        perc_x = conf_x + bar_w + 40
        text.draw('perc', perc_x, y + 9, _DIM)
        perc_x += 30
        draw.rect(perc_x, y + 9, bar_w, 10, _DIM)
        draw.rect(perc_x, y + 9, bar_w * self._percussiveness, 10, _ORANGE)


class TimelinePanel(Panel):
    """Scrolling piano-roll of detected onsets."""

    def __init__(self, band_names: tuple[str, ...],
                 all_band_names: tuple[str, ...] | None = None) -> None:
        n_bands = max(len(band_names), 1)
        super().__init__(height=20 + n_bands * 20)
        self._band_names = band_names
        self._all_band_names = all_band_names or band_names
        self._now = time.perf_counter()

    def update(self, snap: AudioSnapshot, timeline: TimelineBuffer, dt: float) -> None:
        self._now = time.perf_counter()
        self._timeline = timeline

    def render(self, draw: SolidRenderer, text: TextRenderer,
               x: int, y: int, w: int, h: int) -> None:
        draw.rect(x, y, w, h, _BG_ALT)

        marks = self._timeline.marks_in_window(self._now)
        window = self._timeline.max_seconds
        cutoff = self._now - window
        lane_h = 16
        label_w = 50
        plot_x = x + label_w
        plot_w = w - label_w - 4

        for i, name in enumerate(self._band_names):
            lane_y = y + 4 + i * (lane_h + 4)
            color = _color_for_band(name, self._all_band_names)

            # Band label
            text.draw(name[:6], x + 4, lane_y + 3, _DIM)

            # Lane background line
            draw.rect(plot_x, lane_y + lane_h // 2, plot_w, 1, _DIM)

            # Onset dots
            for mark in marks.get(name, []):
                t = (mark.time - cutoff) / window
                mx = plot_x + t * plot_w
                size = 3 + mark.energy * 4
                alpha = 0.4 + mark.energy * 0.6
                c = (color[0], color[1], color[2], alpha)
                draw.rect(mx - size / 2, lane_y + (lane_h - size) / 2,
                          size, size, c)


class SpectrumPanel(Panel):
    """FFT spectrum with band frequency brackets underneath."""

    def __init__(self, band_config: BandConfig) -> None:
        super().__init__(height=76)
        self._band_config = band_config
        self._spectrum: np.ndarray = np.zeros(0)
        self._stability: np.ndarray = np.zeros(0)

    def update(self, snap: AudioSnapshot, timeline: TimelineBuffer, dt: float) -> None:
        self._spectrum = snap.spectrum
        self._stability = snap.stability

    def _freq_to_x(self, freq: float, plot_x: float, plot_w: float) -> float:
        """Map frequency to x coordinate on log scale."""
        min_log = math.log2(max(20.0, 1.0))
        max_log = math.log2(24000.0)
        if freq <= 0:
            freq = 20.0
        log_f = math.log2(max(freq, 20.0))
        t = (log_f - min_log) / (max_log - min_log)
        return plot_x + t * plot_w

    def render(self, draw: SolidRenderer, text: TextRenderer,
               x: int, y: int, w: int, h: int) -> None:
        draw.rect(x, y, w, h, _BG)

        plot_x = x + 4.0
        plot_w = w - 8.0
        spec_h = h - 16  # leave room for band brackets (one line)

        if len(self._spectrum) > 1:
            n_bins = len(self._spectrum)
            max_val = max(self._spectrum.max(), 1e-10)
            has_stability = len(self._stability) == n_bins

            for i in range(1, n_bins):
                freq = FREQS[i] if i < len(FREQS) else 24000.0
                if freq < 20 or freq > 24000:
                    continue
                bx = self._freq_to_x(freq, plot_x, plot_w)
                next_freq = FREQS[min(i + 1, len(FREQS) - 1)]
                bx2 = self._freq_to_x(next_freq, plot_x, plot_w)
                bw = max(bx2 - bx, 1.0)
                val = self._spectrum[i] / max_val
                bar_h = val * spec_h

                if has_stability:
                    s = self._stability[i]
                    harm_h = bar_h * s
                    perc_h = bar_h * (1.0 - s)
                    # Harmonic (teal) on bottom, percussive (orange) on top
                    draw.rect(bx, y + spec_h - harm_h, bw, harm_h,
                              (0.34, 0.76, 0.76, 0.7))
                    draw.rect(bx, y + spec_h - bar_h, bw, perc_h,
                              (0.85, 0.55, 0.33, 0.7))
                else:
                    draw.rect(bx, y + spec_h - bar_h, bw, bar_h, _FG)

        # Band brackets below spectrum
        # Detection bands: one shared line (springs keep them apart)
        # Energy bands: separate line (can overlap detection bands)
        bracket_y = y + spec_h + 2
        ranges = self._band_config.all_band_ranges
        all_names = self._band_config.all_band_names
        det_names = set(self._band_config.detection_band_names)
        for name, (lo, hi) in ranges.items():
            color = _color_for_band(name, all_names)
            bx1 = self._freq_to_x(lo, plot_x, plot_w)
            bx2 = self._freq_to_x(hi, plot_x, plot_w)
            if name in det_names:
                row_y = bracket_y
            else:
                color = (color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, color[3])
                row_y = bracket_y + 12
            draw.rect(bx1, row_y, bx2 - bx1, 2, color)
            text.draw(name[:4], bx1, row_y + 3, color)


class BandMetricsPanel(Panel):
    """Per-band metrics: density, density_delta, RMS."""

    def __init__(self, band_config: BandConfig) -> None:
        all_names = band_config.all_band_names
        super().__init__(height=4 + len(all_names) * 20)
        self._band_config = band_config
        self._bands: dict[str, BandState] = {}
        self._det_names: set[str] = set(band_config.detection_band_names)

    def update(self, snap: AudioSnapshot, timeline: TimelineBuffer, dt: float) -> None:
        self._bands = snap.bands

    def render(self, draw: SolidRenderer, text: TextRenderer,
               x: int, y: int, w: int, h: int) -> None:
        draw.rect(x, y, w, h, _BG_ALT)

        all_names = self._band_config.all_band_names
        row_h = 16
        bar_w = 80
        label_w = 50

        for i, name in enumerate(all_names):
            ry = y + 4 + i * (row_h + 4)
            bs = self._bands.get(name, BandState())
            color = _color_for_band(name, all_names)

            # Band name
            text.draw(name[:6], x + 4, ry + 3, color)

            col_x = x + label_w

            # Density (detection bands only)
            if name in self._det_names:
                text.draw(f'{bs.onset_density:.1f}/s', col_x, ry + 3, _FG)
                col_x += 50

                # Density delta arrow
                if bs.density_delta > 0.1:
                    arrow = '^'
                    ac = _GREEN
                elif bs.density_delta < -0.1:
                    arrow = 'v'
                    ac = _RED
                else:
                    arrow = '-'
                    ac = _DIM
                text.draw(arrow, col_x, ry + 3, ac)
                col_x += 12
            else:
                col_x += 62

            # RMS bar
            text.draw('rms', col_x, ry + 3, _DIM)
            col_x += 24
            draw.rect(col_x, ry + 3, bar_w, 10, _DIM)
            # Slow RMS ghost
            slow_w = min(bs.slow_rms * 200, bar_w)
            draw.rect(col_x, ry + 3, slow_w, 10, (0.3, 0.3, 0.4, 0.5))
            # Current RMS
            rms_w = min(bs.rms * 200, bar_w)
            draw.rect(col_x, ry + 3, rms_w, 10, color)


class BreakPanel(Panel):
    """Break intensity + section change placeholder."""

    def __init__(self) -> None:
        super().__init__(height=24)
        self._break_intensity = 0.0
        self._section_change = 0.0

    def update(self, snap: AudioSnapshot, timeline: TimelineBuffer, dt: float) -> None:
        self._break_intensity = snap.break_intensity
        self._section_change = snap.section_change

    def render(self, draw: SolidRenderer, text: TextRenderer,
               x: int, y: int, w: int, h: int) -> None:
        draw.rect(x, y, w, h, _BG)

        # Break intensity bar
        text.draw('break', x + 4, y + 7, _DIM)
        bar_x = x + 44
        bar_w = 80
        draw.rect(bar_x, y + 7, bar_w, 10, _DIM)
        if self._break_intensity > 0:
            draw.rect(bar_x, y + 7, bar_w * self._break_intensity, 10, _RED)
        text.draw(f'{self._break_intensity:.2f}', bar_x + bar_w + 4, y + 7, _FG)

        # Section change (deferred visualization, show value for now)
        sec_x = bar_x + bar_w + 50
        text.draw(f'section: {self._section_change:.3f}', sec_x, y + 7, _DIM)


def build_panels(band_config: BandConfig,
                 enabled: list[str] | None = None) -> list[Panel]:
    """Build panels from band config and enabled list."""
    all_panels: dict[str, Panel] = {
        'header': HeaderPanel(),
        'timeline': TimelinePanel(band_config.detection_band_names,
                                  all_band_names=band_config.all_band_names),
        'spectrum': SpectrumPanel(band_config),
        'band_metrics': BandMetricsPanel(band_config),
        'break': BreakPanel(),
    }
    if enabled is None:
        enabled = list(all_panels.keys())
    return [all_panels[name] for name in enabled if name in all_panels]
