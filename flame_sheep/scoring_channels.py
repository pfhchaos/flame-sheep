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

import json
import logging
import struct
import zlib
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

COLOR_SCALE = 1_000_000  # must match #define in flame.comp

# Sentinel value for H/A pre-standardization: never-hit pixels have H=1.0,
# A=1.0 (first_hit==255). Used as the categorical out-of-band marker.
SENTINEL_VALUE = 1.0

# Sentinel value for H/A POST-standardization: clamped to this fixed value
# so the model sees the same sentinel marker regardless of corpus stats
# or normalization version.
#
# Originally +5.0 (outside hit-pixel range on the positive side). That
# produced systematic anti-correlation between random Kaiming init and
# liked-genome scoring: a no-training model scored thumbs val pairs at
# 2% accuracy (97% wrong direction). Mechanism: large positive sentinels
# create asymmetric ReLU activation magnitude that correlates with
# sentinel count; liked genomes have fewer sentinels, so they consistently
# scored lower than disliked regardless of weight sign.
#
# -5.0 puts sentinels on the negative side. ReLU(negative) = 0, so
# sentinel-rich pixels contribute zero to downstream activations through
# positive-weighted filters and zero through negative-weighted ones (the
# negative input × negative weight = positive product, which passes ReLU,
# but only filters with net-negative weights survive — symmetric across
# random init signs). Restores random-init behavior near 50%.
SENTINEL_STANDARDIZED = 5.0


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


def normalization_sidecar_path(data_dir: str | Path, version: str) -> Path:
    """Conventional location for a dataset's normalization sidecar.

    A sidecar is for datasets that live outside the Library DB (e.g., a
    manifest.csv + image_dir tree like ~/datasets/esheep-cnn/). Library-
    managed datasets use Library.get_normalization() instead.
    """
    return Path(data_dir) / f'normalization_{version}.json'


def load_normalization_sidecar(data_dir: str | Path, version: str):
    """Read a dataset's normalization sidecar.

    Returns (mean, std) as (4,) float32 arrays, or None if the file is
    absent or malformed. Caller is expected to handle None by either
    falling back to a different source or refusing to train.
    """
    path = normalization_sidecar_path(data_dir, version)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get('version') != version:
            log.error('sidecar %s has version=%r, expected %r',
                      path, data.get('version'), version)
            return None
        return (np.asarray(data['mean'], dtype=np.float32),
                np.asarray(data['std'], dtype=np.float32))
    except (json.JSONDecodeError, KeyError, OSError) as e:
        log.error('failed to read normalization sidecar %s: %s', path, e)
        return None


def save_normalization_sidecar(data_dir: str | Path, version: str,
                               mean: np.ndarray, std: np.ndarray,
                               n_samples: int) -> Path:
    """Write a dataset's normalization sidecar.

    Datasets that don't live in the Library (e.g., ES pretrain manifests)
    store their per-channel mean/std here. The model's .npz only stamps
    the formula version, not these constants — so library and ES models
    can both claim normalization=v1 while standardizing against their own
    dataset's stats.
    """
    path = normalization_sidecar_path(data_dir, version)
    path.write_text(json.dumps({
        'mean': [float(x) for x in mean],
        'std': [float(x) for x in std],
        'n_samples': int(n_samples),
        'version': version,
    }, indent=2))
    return path


def channel_stats_hit_only(
    static_hits: np.ndarray,
    static_colors: np.ndarray,
    swept_hits: np.ndarray,
    first_hit: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (mean, std) over HIT pixels only — sentinels excluded.

    Sentinels (H=1.0 on never-hit pixels, A=1.0 on first_hit==255) dominate
    the raw mean/std of sparse-coverage renders and make the standardized
    value of "this is a sentinel" land at different z-scores across corpora
    with different sentinel fractions. Computing stats over only the
    actually-hit pixels gives a per-image distribution that's comparable
    across corpora (library vs ES come out within ~5% of each other on H
    and L). The trainer's standardize_channels then centers the hit-region
    around zero, and the sentinel value (1.0) lands at a consistent
    high-positive z across both sources.

    Returns (means, stds) as (4,) float32 arrays. Degenerate channels (no
    hit pixels — typically A on legacy ES renders without first_hit) get
    (0.0, 0.0).
    """
    H, S, L, A, masks = _build_native_channels_and_masks(
        static_hits, static_colors, swept_hits, first_hit)

    def stats(arr, mask):
        if mask is None or mask.sum() == 0:
            return 0.0, 0.0
        v = arr[mask]
        return float(v.mean()), float(v.std())

    h_m, h_s = stats(H, masks['H'])
    s_m, s_s = stats(S, masks['S'])
    l_m, l_s = stats(L, masks['L'])
    a_m, a_s = stats(A, masks['A'])
    means = np.array([h_m, s_m, l_m, a_m], dtype=np.float32)
    stds  = np.array([h_s, s_s, l_s, a_s], dtype=np.float32)
    return means, stds


def _build_native_channels_and_masks(static_hits, static_colors, swept_hits, first_hit):
    """Build H/S/L/A channels at native resolution + sentinel masks.

    Same math as normalize_channels but without the final resize step,
    so the sentinel masks (derived from raw histogram inputs) line up
    pixel-exactly with the channels. Used by channel_stats_hit_only.
    """
    L = np.log1p(static_hits.astype(np.float64))
    L_max = L.max()
    if L_max > 0:
        L /= L_max
    L = np.power(L, 1.0 / 6.0).astype(np.float32)

    safe_hits = np.maximum(static_hits, 1).astype(np.float64)
    H = static_colors.astype(np.float64) / (safe_hits * COLOR_SCALE)
    H = np.clip(H, 0.0, 1.0) * (240.0 / 255.0)
    H[static_hits == 0] = 1.0
    H = H.astype(np.float32)

    S = np.log1p(swept_hits.astype(np.float64))
    S_max = S.max()
    if S_max > 0:
        S /= S_max
    lo, hi = np.percentile(S, 5), np.percentile(S, 95)
    if hi > lo:
        S = np.clip((S - lo) / (hi - lo), 0.0, 1.0)
    S = S.astype(np.float32)

    if first_hit is not None:
        A = (first_hit.astype(np.float32) / 255.0)
        A_mask = first_hit < 255
    else:
        # Legacy renders without first_hit data (e.g., the ES corpus
        # before iteration-snapshot rendering existed). Fill A with the
        # sentinel value so the model sees "this entire image has no A
        # signal" as a coherent categorical state, the same encoding it
        # uses for never-hit regions in library renders. Without this
        # ES A would be all zeros and conv weights for A would never
        # see meaningful inputs during pretrain.
        A = np.full_like(L, SENTINEL_VALUE)
        A_mask = np.zeros_like(static_hits, dtype=bool)

    masks = {
        'H': static_hits > 0,
        'S': swept_hits  > 0,
        'L': static_hits > 0,
        'A': A_mask,
    }
    return H, S, L, A, masks


def standardize_channels(channels: np.ndarray,
                         mean: np.ndarray, std: np.ndarray,
                         masks: np.ndarray | None = None) -> np.ndarray:
    """Per-channel standardization: (x - mean) / std.

    channels: (4, H, W) float32.
    mean, std: (4,) arrays of per-channel statistics.
    masks: optional (4, H, W) bool array. True = hit pixel (standardize
        normally). False = sentinel pixel (snap to SENTINEL_STANDARDIZED).
        If None, every pixel is standardized (legacy behavior).

    Sentinel-aware path: the raw sentinel value (1.0) inside a corpus
    with hit-only mean ~0.4 and std ~0.10 lands near +5–+6 under naive
    standardization — exactly where it should be to stay outside the
    hit-pixel range. But that location drifts with corpus stats and
    normalization versions. Snapping sentinels to a constant
    SENTINEL_STANDARDIZED guarantees the model sees the categorical
    marker at the same place every time, decoupled from the centering
    of the hit-pixel distribution.

    Guards against std=0 (degenerate channel) by passing through.
    """
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std_safe = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    out = ((channels - mean[:, None, None]) / std_safe[:, None, None]).astype(np.float32)
    if masks is not None:
        # Sentinel pixels (mask False): snap to fixed canonical value.
        out = np.where(masks, out, np.float32(SENTINEL_STANDARDIZED))
    return out


def build_cnn_input(static_hits: np.ndarray,
                    static_colors: np.ndarray,
                    swept_hits: np.ndarray,
                    first_hit: np.ndarray | None = None,
                    normalization: tuple | None = None,
                    output_size: int = 256) -> np.ndarray:
    """Build the (4, output_size, output_size) standardized CNN input.

    Combines channel construction + resize + sentinel-aware standardization
    so masks derived from raw histograms stay aligned with the channels
    they describe through the whole pipeline. Resize uses bilinear for
    channels (smooth interp) and nearest for masks (preserves boolean
    sentinel/hit distinction without bleeding).

    normalization=None returns the un-standardized raw channels at
    output_size (useful for inspection / visualization). With
    normalization=(mean, std), hit pixels are z-scored and sentinel
    pixels are snapped to SENTINEL_STANDARDIZED.
    """
    from PIL import Image

    H, S, L, A, masks = _build_native_channels_and_masks(
        static_hits, static_colors, swept_hits, first_hit)

    h, w = H.shape
    if h != output_size or w != output_size:
        def resize_bilinear(a):
            return np.array(Image.fromarray(
                (a * 255).astype(np.uint8)).resize(
                (output_size, output_size), Image.BILINEAR),
                dtype=np.float32) / 255.0

        def resize_nearest_bool(m):
            return np.array(Image.fromarray(
                m.astype(np.uint8) * 255).resize(
                (output_size, output_size), Image.NEAREST),
                dtype=np.uint8) > 127

        H = resize_bilinear(H)
        S = resize_bilinear(S)
        L = resize_bilinear(L)
        A = resize_bilinear(A)
        masks = {
            'H': resize_nearest_bool(masks['H']),
            'S': resize_nearest_bool(masks['S']),
            'L': resize_nearest_bool(masks['L']),
            'A': resize_nearest_bool(masks['A']),
        }

    channels = np.stack([H, S, L, A], axis=0).astype(np.float32)

    if normalization is None:
        return channels

    # Per-channel hit mask. L and S don't carry a discrete value-1.0
    # sentinel encoding (their zeros are natural log-min, not categorical
    # markers), but the underlying condition is the same — pixels with
    # zero hits in the relevant histogram. Snapping those to
    # SENTINEL_STANDARDIZED prevents L from standardizing to extreme
    # negative outliers like -9 in sparse-coverage corpora, and gives
    # the model a consistent "dead pixel" signal across all four
    # channels at the same spatial locations.
    mask_stack = np.stack([
        masks['H'],
        masks['S'],
        masks['L'],
        masks['A'],
    ], axis=0)
    return standardize_channels(channels, *normalization, masks=mask_stack)


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
    # If first_hit is missing (legacy ES corpus), fill with sentinel so the
    # model sees a coherent "no A signal here" marker instead of all zeros.
    if first_hit is not None:
        A = first_hit.astype(np.float32) / 255.0
    else:
        A = np.full_like(L, SENTINEL_VALUE)

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

