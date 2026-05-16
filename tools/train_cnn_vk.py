#!/usr/bin/env python3
"""Train CNN aesthetic scorer on GPU via Vulkan compute shaders.

No PyTorch dependency at training time — forward and backward passes
run as Vulkan compute shader dispatches on the Arc A770.

Usage:
    python tools/train_cnn_vk.py --data ~/datasets/esheep-cnn/
    python tools/train_cnn_vk.py --data ~/datasets/esheep-cnn/ --epochs 30 --lr 0.001
"""
from __future__ import annotations

import argparse
import csv
import logging
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flame_sheep.vk_compute import VkCompute

log = logging.getLogger(__name__)

SHADER_DIR = Path(__file__).resolve().parent.parent / 'flame_sheep' / 'shaders' / 'cnn'

# Layer definitions: (in_channels, out_channels, kernel, stride, padding)
MODEL_CONFIGS = {
    '25k': [
        (4,  8,  3, 2, 1),
        (8,  16, 3, 2, 1),
        (16, 32, 3, 2, 1),
        (32, 64, 3, 2, 1),
    ],
    '55k': [
        (4,  12, 3, 2, 1),
        (12, 24, 3, 2, 1),
        (24, 48, 3, 2, 1),
        (48, 96, 3, 2, 1),
    ],
    '100k': [
        (4,  16, 3, 2, 1),
        (16, 32, 3, 2, 1),
        (32, 64, 3, 2, 1),
        (64, 128, 3, 2, 1),
    ],
}
LAYERS = MODEL_CONFIGS['25k']  # default, overridden by --model-size
MLP_HEAD = False  # overridden by --mlp-head
MLP_HIDDEN = 16


def spatial_size(h: int, layers: list) -> list[tuple[int, int]]:
    """Compute output spatial dims for each layer."""
    sizes = [(h, h)]
    for _, _, k, s, p in layers:
        h = (h + 2 * p - k) // s + 1
        sizes.append((h, h))
    return sizes


def weight_counts(layers: list, mlp_head: bool = False) -> list[int]:
    """Compute weight count per layer (weights + bias)."""
    counts = []
    for c_in, c_out, k, _, _ in layers:
        counts.append(c_out * c_in * k * k + c_out)  # kernel + bias
    C = layers[-1][1]
    if mlp_head:
        # MLP: W1(C×H) + b1(H) + W2(H) + b2(1)
        counts.append(C * MLP_HIDDEN + MLP_HIDDEN + MLP_HIDDEN + 1)
    else:
        counts.append(C + 1)
    return counts


class ImageStore:
    """Lazy-loading image store. Loads from disk on demand with LRU cache.

    At 256×256×4×4 = 1MB per image, a 4GB cache holds ~4000 images.
    For 18K images, most batches hit cache after the first epoch.
    """

    def __init__(self, manifest_path: Path, image_dir: Path,
                 image_size: int, cache_gb: float = 4.0,
                 channels: str = 'rgb'):
        self.image_dir = image_dir
        self.image_size = image_size
        self._channels = channels

        self.entries = []
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                self.entries.append((
                    row['static_path'], row['swept_path'],
                    int(row['rating']), int(row['generation']),
                ))

        bytes_per_image = 4 * image_size * image_size * 4
        self.max_cache = int(cache_gb * 1e9 / bytes_per_image)
        self._cache: dict[int, np.ndarray] = {}
        self._access_order: list[int] = []

    def __len__(self):
        return len(self.entries)

    def _load_one(self, idx: int) -> np.ndarray:
        static, swept, _, _ = self.entries[idx]
        sz = self.image_size

        if self._channels == 'domain':
            from flame_sheep.scoring_channels import normalize_channels, load_raw_histograms
            hist_name = static.replace('_static.png', '_hist.npz')
            hist_path = self.image_dir / hist_name
            if hist_path.exists():
                raw = load_raw_histograms(str(hist_path))
                return normalize_channels(
                    raw['static_hits'], raw['static_colors'],
                    raw['swept_hits'], raw.get('first_hit'),
                    output_size=sz,
                )
            # Fall through to RGB if no .npz

        s_img = Image.open(self.image_dir / static).convert('RGB').resize(
            (sz, sz), Image.LANCZOS)
        w_img = Image.open(self.image_dir / swept).convert('L').resize(
            (sz, sz), Image.LANCZOS)
        img = np.zeros((4, sz, sz), dtype=np.float32)
        img[:3] = np.array(s_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        img[3] = np.array(w_img, dtype=np.float32) / 255.0
        return img

    def get_batch(self, indices: list[int] | np.ndarray) -> np.ndarray:
        """Load a batch of images by index. Uses LRU cache."""
        batch = np.zeros((len(indices), 4, self.image_size, self.image_size),
                         dtype=np.float32)
        for i, idx in enumerate(indices):
            if idx not in self._cache:
                # Evict oldest if cache full
                while len(self._cache) >= self.max_cache:
                    old = self._access_order.pop(0)
                    self._cache.pop(old, None)
                self._cache[idx] = self._load_one(idx)
            else:
                # Move to end of access order
                try:
                    self._access_order.remove(idx)
                except ValueError:
                    pass
            self._access_order.append(idx)
            batch[i] = self._cache[idx]
        return batch


def load_images(manifest_path: Path, image_dir: Path,
                image_size: int) -> tuple[np.ndarray, list]:
    """Load all images as (N, 4, H, W) float32 + metadata.

    For backward compatibility. Use ImageStore for streaming.
    """
    store = ImageStore(manifest_path, image_dir, image_size, cache_gb=100)
    entries = store.entries

    images = np.zeros((len(entries), 4, image_size, image_size), dtype=np.float32)
    for i in range(len(entries)):
        images[i] = store._load_one(i)
        if (i + 1) % 2000 == 0:
            log.info('  loaded %d/%d images', i + 1, len(entries))

    return images, entries


def sample_pairs(entries: list, n_pairs: int, min_gap: float = 0.5,
                 exclude_gen: int | None = None) -> list[tuple[int, int]]:
    """Sample pairwise comparisons using composite scoring.

    Composite score = generation_rank + normalized_within_gen_rating.
    Later generations are globally better (that's what selection pressure means).
    Cross-generation pairs give strong signal; within-generation pairs add local detail.
    """
    rng = np.random.default_rng()

    # Build generation rank (sorted by generation number)
    gen_map: dict[int, list[int]] = {}
    for i, (_, _, _, gen) in enumerate(entries):
        if gen != exclude_gen:
            gen_map.setdefault(gen, []).append(i)

    sorted_gens = sorted(gen_map.keys())
    gen_rank = {g: rank for rank, g in enumerate(sorted_gens)}

    # Compute composite score for each entry:
    #   composite = generation_rank + normalized_within_gen_rating
    # Within a generation, ratings are normalized to [0, 1]
    composites = np.zeros(len(entries), dtype=np.float64)
    for gen, indices in gen_map.items():
        ratings = np.array([entries[i][2] for i in indices], dtype=np.float64)
        rmin, rmax = ratings.min(), ratings.max()
        if rmax > rmin:
            norm = (ratings - rmin) / (rmax - rmin)
        else:
            norm = np.full(len(ratings), 0.5)
        rank = gen_rank[gen]
        for j, idx in enumerate(indices):
            composites[idx] = rank + norm[j]

    # Sample pairs: winner has higher composite score
    all_indices = []
    for indices in gen_map.values():
        all_indices.extend(indices)
    all_indices = np.array(all_indices)
    n_total = len(all_indices)

    pairs = []
    for _ in range(n_pairs * 2):  # oversample, filter by gap
        a, b = rng.choice(n_total, 2, replace=False)
        ia, ib = all_indices[a], all_indices[b]
        ca, cb = composites[ia], composites[ib]
        gap = abs(ca - cb)
        if gap < min_gap:
            continue
        if ca > cb:
            pairs.append((ia, ib))
        else:
            pairs.append((ib, ia))
        if len(pairs) >= n_pairs:
            break

    rng.shuffle(pairs)
    return pairs


class VkTrainer:
    """Manages GPU buffers and pipelines for CNN training."""

    def __init__(self, gpu: VkCompute, image_size: int, batch_size: int):
        self.gpu = gpu
        self.image_size = image_size
        self.batch_size = batch_size

        self.sizes = spatial_size(image_size, LAYERS)
        self.w_counts = weight_counts(LAYERS, MLP_HEAD)
        self.total_weights = sum(self.w_counts)
        self.mlp_head = MLP_HEAD

        log.info('Model: %d params across %d layers', self.total_weights,
                 len(LAYERS) + 1)
        log.info('Spatial sizes: %s', self.sizes)

        self._create_buffers()
        self._create_pipelines()

    def _create_buffers(self):
        B = self.batch_size
        H = self.image_size

        # Activation buffers for forward pass
        # Two sets: one for winner, one for loser (reuse by re-uploading)
        self.input_buf = self.gpu.create_buffer(
            B * 4 * H * H * 4)  # (B, 4, H, H) float32

        self.act_bufs = []
        for i, (_, c_out, _, _, _) in enumerate(LAYERS):
            h_out, w_out = self.sizes[i + 1]
            size = B * c_out * h_out * w_out * 4
            self.act_bufs.append(self.gpu.create_buffer(size))

        # GAP pooled + output + optional hidden (MLP)
        final_c = LAYERS[-1][1]
        self.pooled_buf = self.gpu.create_buffer(B * final_c * 4)
        self.output_buf = self.gpu.create_buffer(B * 4)
        if MLP_HEAD:
            self.hidden_buf = self.gpu.create_buffer(B * MLP_HIDDEN * 4)
        else:
            self.hidden_buf = None

        # Weights and gradients (flat, all layers concatenated)
        self.weights_buf = self.gpu.create_buffer(self.total_weights * 4)
        self.grads_buf = self.gpu.create_buffer(self.total_weights * 4)
        self.gpu.zero_buffer(self.grads_buf)

        # Initialize weights (Kaiming uniform)
        rng = np.random.default_rng(42)
        weights = np.zeros(self.total_weights, dtype=np.float32)
        offset = 0
        for c_in, c_out, k, _, _ in LAYERS:
            n_w = c_out * c_in * k * k
            fan_in = c_in * k * k
            std = np.sqrt(2.0 / fan_in)
            weights[offset:offset + n_w] = rng.standard_normal(n_w).astype(np.float32) * std
            offset += n_w
            weights[offset:offset + c_out] = 0.0  # bias
            offset += c_out
        # Head (linear or MLP)
        fan_in = LAYERS[-1][1]
        if MLP_HEAD:
            # W1: (C × hidden)
            std = np.sqrt(2.0 / fan_in)
            n_w1 = fan_in * MLP_HIDDEN
            weights[offset:offset + n_w1] = rng.standard_normal(n_w1).astype(np.float32) * std
            offset += n_w1
            weights[offset:offset + MLP_HIDDEN] = 0.0  # b1
            offset += MLP_HIDDEN
            # W2: (hidden)
            std = np.sqrt(2.0 / MLP_HIDDEN)
            weights[offset:offset + MLP_HIDDEN] = rng.standard_normal(MLP_HIDDEN).astype(np.float32) * std
            offset += MLP_HIDDEN
            weights[offset] = 0.0  # b2
            offset += 1
        else:
            std = np.sqrt(2.0 / fan_in)
            weights[offset:offset + fan_in] = rng.standard_normal(fan_in).astype(np.float32) * std
            offset += fan_in
            weights[offset] = 0.0  # bias
            offset += 1
        self.gpu.upload(self.weights_buf, weights)

        # Gradient activation buffer (for backward pass, reused per layer)
        max_act_size = max(
            B * 4 * H * H,
            max(B * c * h * w for (_, c, _, _, _), (h, w) in zip(LAYERS, self.sizes[1:]))
        )
        self.grad_act_buf = self.gpu.create_buffer(max_act_size * 4)

        # Upstream gradient for GAP+linear backward
        self.d_output_buf = self.gpu.create_buffer(B * 4)

    def _weight_slice(self, layer_idx: int) -> tuple[int, int]:
        """Get byte offset and size for a layer's weights in the flat buffer."""
        offset = sum(self.w_counts[:layer_idx])
        return offset * 4, self.w_counts[layer_idx] * 4

    def _create_pipelines(self):
        # Forward pipelines — one per conv layer
        self.conv_fwd_pipelines = []
        for i, (c_in, c_out, k, s, p) in enumerate(LAYERS):
            inp = self.input_buf if i == 0 else self.act_bufs[i - 1]
            # We need separate weight/bias views — but our flat buffer means
            # we pass the full weights buffer and use push constants for offset
            # Actually, the shader reads from binding 1 (weights) starting at 0.
            # We need per-layer weight buffers OR modify the shader to take an offset.
            # For simplicity, create per-layer weight buffer views by creating
            # separate pipelines with the right buffer slices.
            # BUT Vulkan doesn't support buffer views in SSBOs easily.
            # Alternative: pass weight_offset as push constant.
            pass  # Will use a different approach — see forward() method

        # We'll dispatch dynamically with per-layer buffer rebinding.
        # Actually, let's just create pipelines on the fly since the cffi
        # overhead is minimal compared to GPU work.

    def forward(self, images: np.ndarray) -> np.ndarray:
        """Run forward pass, return (B,) scores."""
        B = len(images)
        self.gpu.upload(self.input_buf, images)

        # Conv layers
        for i, (c_in, c_out, k, s, p) in enumerate(LAYERS):
            h_in, w_in = self.sizes[i]
            h_out, w_out = self.sizes[i + 1]

            inp_buf = self.input_buf if i == 0 else self.act_bufs[i - 1]

            # Extract layer weights from flat buffer to a temp buffer
            w_offset, w_size = self._weight_slice(i)
            w_data = self.gpu.download(self.weights_buf, np.uint8, self.total_weights * 4)
            layer_w = w_data[w_offset:w_offset + w_size]

            n_kernel = c_out * c_in * k * k
            kernel_data = np.frombuffer(layer_w[:n_kernel * 4], dtype=np.float32)
            bias_data = np.frombuffer(layer_w[n_kernel * 4:], dtype=np.float32)

            # Create temp buffers for this layer's weights
            kernel_buf = self.gpu.create_buffer(n_kernel * 4)
            bias_buf = self.gpu.create_buffer(c_out * 4)
            self.gpu.upload(kernel_buf, kernel_data)
            self.gpu.upload(bias_buf, bias_data)

            pipeline = self.gpu.create_pipeline(
                str(SHADER_DIR / 'conv2d_forward.comp'),
                buffers=[inp_buf, kernel_buf, bias_buf, self.act_bufs[i]],
                push_constant_size=44,
            )
            push = struct.pack('11i', B, c_in, c_out, h_in, w_in,
                               h_out, w_out, k, s, p, 1)
            gx = (w_out + 7) // 8
            gy = (h_out + 7) // 8
            gz = B * c_out
            self.gpu.dispatch(pipeline, gx, gy, gz, push)

            pipeline.destroy()
            kernel_buf.destroy()
            bias_buf.destroy()

        # GAP + head (linear or MLP)
        final_c = LAYERS[-1][1]
        h_final, w_final = self.sizes[-1]

        # Extract head weights
        head_offset = sum(self.w_counts[:-1]) * 4
        all_w = self.gpu.download(self.weights_buf, np.uint8, self.total_weights * 4)
        head_w = np.frombuffer(all_w[head_offset:], dtype=np.float32)

        head_buf = self.gpu.create_buffer(len(head_w) * 4)
        self.gpu.upload(head_buf, head_w)

        if self.mlp_head:
            gap_pipeline = self.gpu.create_pipeline(
                str(SHADER_DIR / 'gap_mlp_forward.comp'),
                buffers=[self.act_bufs[-1], head_buf, self.pooled_buf,
                         self.output_buf, None, self.hidden_buf],
                push_constant_size=20,
            )
            push = struct.pack('5i', B, final_c, h_final, w_final, MLP_HIDDEN)
        else:
            gap_pipeline = self.gpu.create_pipeline(
                str(SHADER_DIR / 'gap_linear_forward.comp'),
                buffers=[self.act_bufs[-1], head_buf, self.pooled_buf, self.output_buf],
                push_constant_size=16,
            )
            push = struct.pack('4i', B, final_c, h_final, w_final)
        self.gpu.dispatch(gap_pipeline, (B + 31) // 32, push_constants=push)

        gap_pipeline.destroy()
        head_buf.destroy()

        return self.gpu.download(self.output_buf, np.float32, B)

    def train_step(self, winner_images: np.ndarray, loser_images: np.ndarray,
                   lr: float) -> float:
        """One training step: forward both, compute loss, backward, update."""
        # Forward pass for winners
        w_scores = self.forward(winner_images)
        w_acts = [self.gpu.download(buf, np.float32) for buf in self.act_bufs]
        w_pooled = self.gpu.download(self.pooled_buf, np.float32)
        w_hidden = self.gpu.download(self.hidden_buf, np.float32) if self.mlp_head else None

        # Forward pass for losers
        l_scores = self.forward(loser_images)
        l_acts = [self.gpu.download(buf, np.float32) for buf in self.act_bufs]
        l_pooled = self.gpu.download(self.pooled_buf, np.float32)
        l_hidden = self.gpu.download(self.hidden_buf, np.float32) if self.mlp_head else None

        # Margin ranking loss: max(0, margin - (w_score - l_score))
        B = len(winner_images)
        margin = 1.0
        diff = w_scores - l_scores
        losses = np.maximum(0, margin - diff)
        loss = losses.mean()

        # Gradient of margin loss: d_loss/d_w_score = -1/B if loss > 0, else 0
        #                          d_loss/d_l_score = +1/B if loss > 0, else 0
        active = (losses > 0).astype(np.float32)
        d_w_scores = -active / B
        d_l_scores = active / B

        # Zero gradients
        self.gpu.zero_buffer(self.grads_buf)

        # Backward for winners (accumulates into shared grads_buf)
        self._backward(winner_images, w_acts, w_pooled, d_w_scores, w_hidden)

        # Backward for losers (accumulates into same grads_buf)
        self._backward(loser_images, l_acts, l_pooled, d_l_scores, l_hidden)

        # SGD update
        sgd_pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'sgd_update.comp'),
            buffers=[self.weights_buf, self.grads_buf],
            push_constant_size=8,
        )
        push = struct.pack('fI', lr, self.total_weights)
        self.gpu.dispatch(sgd_pipeline, (self.total_weights + 255) // 256,
                          push_constants=push)
        sgd_pipeline.destroy()

        return float(loss)

    def _backward(self, images: np.ndarray, fwd_acts: list[np.ndarray],
                  fwd_pooled: np.ndarray, d_scores: np.ndarray,
                  fwd_hidden: np.ndarray | None = None):
        """Backward pass through all layers, accumulating weight gradients."""
        B = len(images)
        final_c = LAYERS[-1][1]
        h_final, w_final = self.sizes[-1]

        # --- GAP + head backward ---
        d_out_buf = self.gpu.create_buffer(B * 4)
        self.gpu.upload(d_out_buf, d_scores.astype(np.float32))

        pooled_buf = self.gpu.create_buffer(fwd_pooled.nbytes)
        self.gpu.upload(pooled_buf, fwd_pooled)

        # Head weights
        head_offset = sum(self.w_counts[:-1]) * 4
        all_w = self.gpu.download(self.weights_buf, np.uint8, self.total_weights * 4)
        head_w = np.frombuffer(all_w[head_offset:], dtype=np.float32)
        head_buf = self.gpu.create_buffer(len(head_w) * 4)
        self.gpu.upload(head_buf, head_w)

        head_grad_offset = sum(self.w_counts[:-1])
        head_grad_size = self.w_counts[-1]
        head_grad_buf = self.gpu.create_buffer(head_grad_size * 4)
        self.gpu.zero_buffer(head_grad_buf)

        d_act_buf = self.gpu.create_buffer(B * final_c * h_final * w_final * 4)

        if self.mlp_head:
            hidden_buf = self.gpu.create_buffer(fwd_hidden.nbytes)
            self.gpu.upload(hidden_buf, fwd_hidden)
            gap_bw_pipeline = self.gpu.create_pipeline(
                str(SHADER_DIR / 'gap_mlp_backward.comp'),
                buffers=[d_out_buf, pooled_buf, head_buf, d_act_buf,
                         head_grad_buf, hidden_buf],
                push_constant_size=20,
            )
            push = struct.pack('5i', B, final_c, h_final, w_final, MLP_HIDDEN)
        else:
            gap_bw_pipeline = self.gpu.create_pipeline(
                str(SHADER_DIR / 'gap_linear_backward.comp'),
                buffers=[d_out_buf, pooled_buf, head_buf, d_act_buf, head_grad_buf],
                push_constant_size=16,
            )
            push = struct.pack('4i', B, final_c, h_final, w_final)
        gx = (w_final + 7) // 8
        gy = (h_final + 7) // 8
        gz = B * final_c
        self.gpu.dispatch(gap_bw_pipeline, gx, gy, gz, push)

        # Accumulate head grads into flat grad buffer
        head_grads_f = np.frombuffer(
            self.gpu.download(head_grad_buf, np.uint8, head_grad_size * 4).tobytes(),
            dtype=np.float32)
        flat_grads_f = np.frombuffer(
            self.gpu.download(self.grads_buf, np.uint8, self.total_weights * 4).tobytes(),
            dtype=np.float32).copy()
        flat_grads_f[head_grad_offset:head_grad_offset + head_grad_size] += head_grads_f
        self.gpu.upload(self.grads_buf, np.frombuffer(flat_grads_f.tobytes(), dtype=np.uint32))

        gap_bw_pipeline.destroy()
        if self.mlp_head:
            hidden_buf.destroy()
        d_out_buf.destroy()
        pooled_buf.destroy()
        lin_buf.destroy()
        lin_grad_buf.destroy()

        # --- Conv layers backward (reverse order) ---
        d_upstream = self.gpu.download(d_act_buf, np.float32)
        d_act_buf.destroy()

        for i in range(len(LAYERS) - 1, -1, -1):
            c_in, c_out, k, s, p = LAYERS[i]
            h_in, w_in = self.sizes[i]
            h_out, w_out = self.sizes[i + 1]

            # Get forward input and output for this layer
            if i == 0:
                fwd_input = images.reshape(-1).astype(np.float32)
            else:
                fwd_input = fwd_acts[i - 1]
            fwd_output = fwd_acts[i]

            # Upload upstream gradient and forward data
            d_up_buf = self.gpu.create_buffer(len(d_upstream) * 4)
            self.gpu.upload(d_up_buf, d_upstream.astype(np.float32))

            fwd_in_buf = self.gpu.create_buffer(len(fwd_input) * 4)
            self.gpu.upload(fwd_in_buf, fwd_input.astype(np.float32))

            fwd_out_buf = self.gpu.create_buffer(len(fwd_output) * 4)
            self.gpu.upload(fwd_out_buf, fwd_output.astype(np.float32))

            # Extract layer weights
            w_offset, w_size = self._weight_slice(i)
            layer_w = all_w[w_offset:w_offset + w_size]
            n_kernel = c_out * c_in * k * k
            kernel_data = np.frombuffer(layer_w[:n_kernel * 4], dtype=np.float32)

            kernel_buf = self.gpu.create_buffer(n_kernel * 4)
            self.gpu.upload(kernel_buf, kernel_data)

            # --- Backward weights ---
            layer_grad_size = self.w_counts[i]
            layer_grad_buf = self.gpu.create_buffer(layer_grad_size * 4)
            self.gpu.zero_buffer(layer_grad_buf)

            bw_pipeline = self.gpu.create_pipeline(
                str(SHADER_DIR / 'conv2d_backward_weights.comp'),
                buffers=[d_up_buf, fwd_in_buf, layer_grad_buf, fwd_out_buf],
                push_constant_size=44,
            )
            push = struct.pack('11i', B, c_in, c_out, h_in, w_in,
                               h_out, w_out, k, s, p, 1)
            n_threads = layer_grad_size
            self.gpu.dispatch(bw_pipeline, (n_threads + 63) // 64, push_constants=push)
            bw_pipeline.destroy()

            # Accumulate layer grads into flat grad buffer (add, not replace)
            layer_offset = sum(self.w_counts[:i])
            lg_f = np.frombuffer(
                self.gpu.download(layer_grad_buf, np.uint8, layer_grad_size * 4).tobytes(),
                dtype=np.float32)
            flat_grads_f = np.frombuffer(
                self.gpu.download(self.grads_buf, np.uint8, self.total_weights * 4).tobytes(),
                dtype=np.float32).copy()
            flat_grads_f[layer_offset:layer_offset + layer_grad_size] += lg_f
            self.gpu.upload(self.grads_buf, np.frombuffer(flat_grads_f.tobytes(), dtype=np.uint32))
            layer_grad_buf.destroy()

            # --- Backward input (only if not first layer) ---
            if i > 0:
                d_input_size = B * c_in * h_in * w_in
                d_input_buf = self.gpu.create_buffer(d_input_size * 4)

                # Need to apply ReLU mask from previous layer's output
                prev_act_buf = self.gpu.create_buffer(len(fwd_input) * 4)
                self.gpu.upload(prev_act_buf, fwd_input.astype(np.float32))

                bi_pipeline = self.gpu.create_pipeline(
                    str(SHADER_DIR / 'conv2d_backward_input.comp'),
                    buffers=[d_up_buf, kernel_buf, d_input_buf, prev_act_buf],
                    push_constant_size=48,
                )
                # relu_mask_mode=1: apply ReLU mask based on fwd_input > 0
                push = struct.pack('12i', B, c_in, c_out, h_in, w_in,
                                   h_out, w_out, k, s, p, 0, 1)
                gx = (w_in + 7) // 8
                gy = (h_in + 7) // 8
                gz = B * c_in
                self.gpu.dispatch(bi_pipeline, gx, gy, gz, push)

                d_upstream = self.gpu.download(d_input_buf, np.float32)
                bi_pipeline.destroy()
                d_input_buf.destroy()
                prev_act_buf.destroy()

            d_up_buf.destroy()
            fwd_in_buf.destroy()
            fwd_out_buf.destroy()
            kernel_buf.destroy()


def main():
    parser = argparse.ArgumentParser(description='Train CNN scorer on GPU (Vulkan)')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--pairs-per-epoch', type=int, default=5000)
    parser.add_argument('--val-gen', type=int, default=244)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--cache-gb', type=float, default=4.0,
                        help='Image cache size in GB (default: 4.0)')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model-size', type=str, default='25k',
                        choices=['25k', '55k', '100k'],
                        help='Model size: 25k / 55k / 100k')
    parser.add_argument('--mlp-head', action='store_true',
                        help='Use MLP head (Linear→ReLU→Linear) instead of single Linear')
    parser.add_argument('--channels', type=str, default='rgb',
                        choices=['rgb', 'domain'],
                        help='Input channels: rgb (RGB+swept) or domain (histogram-based H/S/L/A)')
    parser.add_argument('--init-weights', type=str, default=None,
                        help='Load initial weights from .npy file (for resuming or fine-tuning)')
    args = parser.parse_args()

    global LAYERS, MLP_HEAD
    LAYERS = MODEL_CONFIGS[args.model_size]
    MLP_HEAD = args.mlp_head

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')

    data_dir = Path(args.data)
    output_path = Path(args.output) if args.output else data_dir / 'cnn_scorer_vk.npy'

    # Streaming image store (loads from disk on demand)
    store = ImageStore(data_dir / 'manifest.csv', data_dir, args.image_size,
                       cache_gb=args.cache_gb, channels=args.channels)
    entries = store.entries
    log.info('Dataset: %d genomes, image cache: %.1f GB (%d images)',
             len(entries), args.cache_gb, store.max_cache)

    # Init Vulkan
    gpu = VkCompute()
    log.info('GPU: %s', gpu.device_name)

    trainer = VkTrainer(gpu, args.image_size, args.batch_size)

    # Load initial weights if provided (resume or fine-tune)
    if args.init_weights:
        init_w = np.load(args.init_weights).astype(np.float32)
        gpu.upload(trainer.weights_buf, init_w)
        log.info('Loaded initial weights: %d params from %s', len(init_w), args.init_weights)

    # Validation pairs — cross-gen composite, same method as training.
    # Hold out val_gen from training; val uses cross-gen pairs involving val_gen.
    val_pairs = sample_pairs(entries, 2000, min_gap=0.5, exclude_gen=None)
    # Keep only pairs where at least one side is from val_gen
    val_pairs = [(w, l) for w, l in val_pairs
                 if entries[w][3] == args.val_gen or entries[l][3] == args.val_gen]
    log.info('Validation pairs: %d (gen %d)', len(val_pairs), args.val_gen)

    best_val_acc = 0.0
    patience = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Sample training pairs from ALL generations except val
        train_pairs = sample_pairs(entries, args.pairs_per_epoch, min_gap=0,
                                   exclude_gen=args.val_gen)

        # Train
        epoch_loss = 0.0
        n_batches = 0
        rng = np.random.default_rng()
        pair_arr = np.array(train_pairs)
        rng.shuffle(pair_arr)

        for i in range(0, len(pair_arr), args.batch_size):
            batch = pair_arr[i:i + args.batch_size]
            if len(batch) < 2:
                continue
            w_imgs = store.get_batch(batch[:, 0])
            l_imgs = store.get_batch(batch[:, 1])
            loss = trainer.train_step(w_imgs, l_imgs, args.lr)
            if np.isnan(loss):
                log.warning('NaN loss at batch %d, skipping', i)
                continue
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate
        val_indices = sorted(set(w for w, _ in val_pairs) | set(l for _, l in val_pairs))
        val_scores = {}
        for i in range(0, len(val_indices), args.batch_size):
            batch_idx = val_indices[i:i + args.batch_size]
            batch_imgs = store.get_batch(batch_idx)
            if len(batch_imgs) < args.batch_size:
                pad = np.zeros((args.batch_size - len(batch_imgs), *batch_imgs.shape[1:]),
                               dtype=np.float32)
                batch_imgs = np.concatenate([batch_imgs, pad])
            scores = trainer.forward(batch_imgs)
            for j, idx in enumerate(batch_idx):
                val_scores[idx] = scores[j]

        correct = sum(1 for w, l in val_pairs if val_scores.get(w, 0) > val_scores.get(l, 0))
        val_acc = correct / max(len(val_pairs), 1)

        elapsed = time.time() - t0
        log.info('Epoch %2d/%d  loss=%.4f  val_acc=%.3f  cache=%d  (%.0fs)',
                 epoch, args.epochs, avg_loss, val_acc, len(store._cache), elapsed)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            weights = gpu.download(trainer.weights_buf, np.float32, trainer.total_weights)
            np.save(str(output_path), weights)
            log.info('  -> saved (best val_acc=%.3f)', best_val_acc)
        else:
            patience += 1
            if patience >= 15:
                log.info('Early stopping (15 epochs without improvement)')
                break

    log.info('Training complete. Best val accuracy: %.3f', best_val_acc)
    gpu.destroy()


if __name__ == '__main__':
    main()
