"""Build 4-channel scoring representation from renderer histograms.

Two layers:
  Raw:        stored on disk as compressed .npz — full histogram data
  Normalized: built at training time from raw data

Raw channel layout (uint32):
  static_hits:   (H, W) hit counts from static render
  static_colors: (H, W) color accumulators (palette index * COLOR_SCALE)
  swept_hits:    (H, W) hit counts from swept rotation render
  low_iter_hits: (H, W) hit counts from partial render (iteration response)

Normalized channel layout (float32, [0,1]):
  H — average palette index (color_acc / hit_count / COLOR_SCALE)
  S — swept rotation density (log-normalized)
  L — log-transformed static hit count (structure)
  A — iteration response (log(full) - log(partial), normalized)

Normalization (log transform, scaling) happens at training time, not storage.
Raw data preserves maximum information for experimentation.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

COLOR_SCALE = 1_000_000  # must match #define in flame.comp


# ---------------------------------------------------------------------------
# DB blob pack/unpack helpers
# ---------------------------------------------------------------------------

def pack_histogram(arr: np.ndarray) -> bytes:
    """Pack a numpy array into a zlib blob with dimension header.

    Header: struct.pack('II', height, width)
    Body: zlib(arr.tobytes())
    """
    h, w = arr.shape[:2]
    header = struct.pack('II', h, w)
    return zlib.compress(header + arr.tobytes())


def pack_static_histogram(hits: np.ndarray, colors: np.ndarray) -> bytes:
    """Pack static histogram (hits + colors) into a single blob.

    Header: struct.pack('II', height, width)
    Body: zlib(hits.tobytes() + colors.tobytes())
    """
    h, w = hits.shape
    header = struct.pack('II', h, w)
    return zlib.compress(header + hits.astype(np.uint32).tobytes()
                         + colors.astype(np.uint32).tobytes())


def unpack_histogram(blob: bytes, dtype=np.uint32) -> np.ndarray:
    """Unpack a single-array histogram blob with dimension header."""
    raw = zlib.decompress(blob)
    h, w = struct.unpack('II', raw[:8])
    return np.frombuffer(raw[8:], dtype=dtype).reshape(h, w)


def unpack_static_histogram(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Unpack static histogram blob into (hits, colors) arrays."""
    raw = zlib.decompress(blob)
    h, w = struct.unpack('II', raw[:8])
    n = h * w
    data = np.frombuffer(raw[8:], dtype=np.uint32)
    return data[:n].reshape(h, w), data[n:2*n].reshape(h, w)


def save_raw_histograms(
    path: str,
    static_hits: np.ndarray,
    static_colors: np.ndarray,
    swept_hits: np.ndarray,
    first_hit: np.ndarray | None = None,
) -> None:
    """Save raw histogram data as compressed .npz.

    first_hit: (H, W) uint8 — iteration step (0-255) when each pixel was
    first hit. 255 = never hit. Lower = earlier emergence = stable structure.
    Higher = late emergence = detail that appears with more iterations.
    """
    arrays = {
        'static_hits': static_hits,
        'static_colors': static_colors,
        'swept_hits': swept_hits,
    }
    if first_hit is not None:
        arrays['first_hit'] = first_hit
    np.savez_compressed(path, **arrays)


def load_raw_histograms(path: str) -> dict[str, np.ndarray]:
    """Load raw histogram data from .npz."""
    return dict(np.load(path))


def standardize_channels(channels: np.ndarray,
                         mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-channel standardization: (x - mean) / std.

    channels: (4, H, W) float32.
    mean, std: (4,) arrays of per-channel statistics.

    Guards against std=0 (degenerate channel) by passing through.
    Result: zero-mean / unit-variance per channel, which matches what
    Kaiming weight initialization expects. Without this the first conv
    layer's outputs can land below zero everywhere and ReLU kills the
    gradient flow permanently.
    """
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std_safe = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return ((channels - mean[:, None, None]) / std_safe[:, None, None]).astype(np.float32)


def normalize_channels(
    static_hits: np.ndarray,
    static_colors: np.ndarray,
    swept_hits: np.ndarray,
    first_hit: np.ndarray | None = None,
    output_size: int = 256,
) -> np.ndarray:
    """Build (4, output_size, output_size) float32 normalized channels.

    Applies log transforms and normalization suitable for CNN input.
    """
    from PIL import Image

    # --- Channel L: log hit count (structure) ---
    # Matches the renderer's tonemap: log(1+hits)/log(1+max_hits), then gamma.
    # Gamma boosts faint structure so it's visible — same as what the user sees.
    GAMMA = 6.0  # matches default u_gamma in tonemap.frag
    L = np.log1p(static_hits.astype(np.float64))
    L_max = L.max()
    if L_max > 0:
        L /= L_max
    L = np.power(L, 1.0 / GAMMA)
    L = L.astype(np.float32)

    # --- Channel H: average palette index ---
    # Compress active range to [0, 240/255] so 1.0 (255/255) can serve as
    # a sentinel for never-hit pixels. Otherwise never-hit pixels (color=0,
    # hits=0 → safe_hits=1 → H=0) collide with "hit with low palette index".
    safe_hits = np.maximum(static_hits, 1).astype(np.float64)
    H = static_colors.astype(np.float64) / (safe_hits * COLOR_SCALE)
    H = np.clip(H, 0.0, 1.0)
    H = H * (240.0 / 255.0)
    H[static_hits == 0] = 1.0
    H = H.astype(np.float32)

    # --- Channel S: swept rotation density ---
    # log compresses the heavy tail, then per-image percentile contrast
    # stretch boosts edge contrast. Without the stretch, the corrected
    # swept render produces a mid-gray-to-white range with almost no pure
    # black — hard for conv filters to detect shape boundaries.
    # Gamma compression (as used for L) is not applied: the swept
    # distribution is broad-and-bright (every pixel hits at some rotation),
    # so gamma would just push everything to ~1.0.
    S = np.log1p(swept_hits.astype(np.float64))
    S_max = S.max()
    if S_max > 0:
        S /= S_max
    # Per-image contrast stretch: clip bottom 5% to black, top 5% to white,
    # linearly stretch the middle. Guard against degenerate genomes (all-zero
    # S, flying dots that snuck through).
    lo = np.percentile(S, 5)
    hi = np.percentile(S, 95)
    if hi > lo:
        S = np.clip((S - lo) / (hi - lo), 0.0, 1.0)
    S = S.astype(np.float32)

    # --- Channel A: first-hit iteration (emergence order) ---
    # 0 = appeared first (stable structure), 255 = appeared last or never.
    # Invert so bright = late emergence = detail that appears with intensity.
    if first_hit is not None:
        A = first_hit.astype(np.float32) / 255.0
    else:
        A = np.zeros_like(L)

    # --- Stack and resize ---
    combined = np.stack([H, S, L, A], axis=-1)  # (H, W, 4)

    h, w = combined.shape[:2]
    if h != output_size or w != output_size:
        result = np.zeros((output_size, output_size, 4), dtype=np.float32)
        for c in range(4):
            ch_img = Image.fromarray((combined[:, :, c] * 255).astype(np.uint8))
            ch_img = ch_img.resize((output_size, output_size), Image.LANCZOS)
            result[:, :, c] = np.array(ch_img, dtype=np.float32) / 255.0
        combined = result

    return combined.transpose(2, 0, 1)  # (4, H, W)

