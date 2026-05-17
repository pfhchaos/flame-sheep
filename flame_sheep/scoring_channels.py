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
    safe_hits = np.maximum(static_hits, 1).astype(np.float64)
    H = static_colors.astype(np.float64) / (safe_hits * COLOR_SCALE)
    H = np.clip(H, 0.0, 1.0).astype(np.float32)

    # --- Channel S: swept rotation density ---
    # log + linear (no gamma compression). The corrected swept render
    # (with real angular structure, not a repeated spirograph) produces a
    # broad-and-bright distribution where every pixel in the attractor
    # at any rotation accumulates hits. Gamma=1/6 (as used for L) would
    # compress this to ~1.0 everywhere, killing the input signal.
    S = np.log1p(swept_hits.astype(np.float64))
    S_max = S.max()
    if S_max > 0:
        S /= S_max
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

