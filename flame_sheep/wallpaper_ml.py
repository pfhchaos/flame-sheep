"""Vulkan compute ML framework — PyTorch-style layer abstraction.

Wraps the raw VkCompute shader dispatch into composable layers with
automatic buffer management, weight serialization, and gradient tracking.

Usage:
    gpu = VkCompute()
    model = build_cnn_scorer(gpu, layers_config, batch_size=8)
    model.load_weights(np.load('weights.npy'))
    scores = model.forward(input_buf)
    model.backward(grad_buf)
    model.sgd_step(lr=0.003)
    np.save('weights.npy', model.save_weights())
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .vk_compute import VkCompute, VkBuffer

SHADER_DIR = Path(__file__).parent / 'shaders' / 'cnn'


# ---------------------------------------------------------------------------
# Base layer
# ---------------------------------------------------------------------------

class VkLayer(ABC):
    """Base class for Vulkan compute layers."""

    def __init__(self, gpu: VkCompute):
        self.gpu = gpu
        self._param_bufs: list[VkBuffer] = []    # float32 weight buffers
        self._grad_bufs: list[VkBuffer] = []      # uint32 gradient buffers (CAS)
        self._param_sizes: list[int] = []          # element counts per param

    @abstractmethod
    def forward(self, input_buf: VkBuffer, batch_size: int,
                shape: tuple[int, ...]) -> tuple[VkBuffer, tuple[int, ...]]:
        """Forward pass. Returns (output_buf, output_shape)."""
        ...

    @abstractmethod
    def backward(self, grad_output: VkBuffer, batch_size: int) -> VkBuffer:
        """Backward pass. Returns gradient w.r.t. input."""
        ...

    def zero_grad(self) -> None:
        for g in self._grad_bufs:
            self.gpu.zero_buffer(g)

    def parameters(self) -> list[tuple[VkBuffer, VkBuffer, int]]:
        """Returns [(param_buf, grad_buf, n_elements), ...]"""
        return list(zip(self._param_bufs, self._grad_bufs, self._param_sizes))

    def param_count(self) -> int:
        return sum(self._param_sizes)

    def load_weights(self, flat: np.ndarray, offset: int) -> int:
        """Load weights from flat array at offset. Returns new offset."""
        for buf, n in zip(self._param_bufs, self._param_sizes):
            self.gpu.upload(buf, flat[offset:offset + n].astype(np.float32))
            offset += n
        return offset

    def save_weights(self) -> np.ndarray:
        """Save weights to flat float32 array."""
        parts = []
        for buf, n in zip(self._param_bufs, self._param_sizes):
            parts.append(self.gpu.download(buf, np.float32, n))
        return np.concatenate(parts) if parts else np.array([], dtype=np.float32)

    def init_weights(self, rng: np.random.Generator) -> None:
        """Initialize weights with Kaiming uniform. Override per layer."""
        pass


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

class VkConv2d(VkLayer):
    """2D convolution + bias + optional ReLU."""

    def __init__(self, gpu: VkCompute, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int,
                 batch_size: int, in_h: int, in_w: int,
                 relu: bool = True):
        super().__init__(gpu)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.relu = relu

        self.in_h = in_h
        self.in_w = in_w
        self.out_h = (in_h + 2*padding - kernel_size) // stride + 1
        self.out_w = (in_w + 2*padding - kernel_size) // stride + 1
        self.batch_size = batch_size

        # Weight buffers
        n_kernel = out_channels * in_channels * kernel_size * kernel_size
        n_bias = out_channels
        self.kernel_buf = gpu.create_buffer(n_kernel * 4)
        self.bias_buf = gpu.create_buffer(n_bias * 4)
        self._param_bufs = [self.kernel_buf, self.bias_buf]
        self._param_sizes = [n_kernel, n_bias]

        # Gradient buffers (uint32 for CAS atomics)
        self.grad_buf = gpu.create_buffer((n_kernel + n_bias) * 4)
        self._grad_bufs = [self.grad_buf]

        # Output buffer
        self.output_buf = gpu.create_buffer(
            batch_size * out_channels * self.out_h * self.out_w * 4)

        # Saved for backward
        self._saved_input: VkBuffer | None = None
        self._saved_output: VkBuffer | None = None

    def forward(self, input_buf, batch_size, shape):
        B = batch_size
        self._saved_input = input_buf

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'conv2d_forward.comp'),
            buffers=[input_buf, self.kernel_buf, self.bias_buf, self.output_buf],
            push_constant_size=44,
        )
        push = struct.pack('11i', B, self.in_channels, self.out_channels,
                           self.in_h, self.in_w, self.out_h, self.out_w,
                           self.kernel_size, self.stride, self.padding,
                           1 if self.relu else 0)
        gx = (self.out_w + 7) // 8
        gy = (self.out_h + 7) // 8
        gz = B * self.out_channels
        self.gpu.dispatch(pipeline, gx, gy, gz, push)
        pipeline.destroy()

        # Save output for ReLU mask in backward
        if self.relu:
            self._saved_output = self.output_buf

        return self.output_buf, (self.out_channels, self.out_h, self.out_w)

    def backward(self, grad_output, batch_size):
        B = batch_size

        # Apply ReLU mask to upstream gradient if needed
        if self.relu and self._saved_output is not None:
            d_relu_buf = grad_output  # shader handles mask via fwd_output
        else:
            d_relu_buf = grad_output

        # Backward input gradient
        d_input_buf = self.gpu.create_buffer(
            B * self.in_channels * self.in_h * self.in_w * 4)

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'conv2d_backward_input.comp'),
            buffers=[d_relu_buf, self.kernel_buf, d_input_buf],
            push_constant_size=44,
        )
        push = struct.pack('11i', B, self.in_channels, self.out_channels,
                           self.in_h, self.in_w, self.out_h, self.out_w,
                           self.kernel_size, self.stride, self.padding, 0)
        gx = (self.in_w + 7) // 8
        gy = (self.in_h + 7) // 8
        gz = B * self.in_channels
        self.gpu.dispatch(pipeline, gx, gy, gz, push)
        pipeline.destroy()

        # Backward weight gradient
        n_total = self._param_sizes[0] + self._param_sizes[1]
        fwd_buf = self._saved_output if self.relu else grad_output

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'conv2d_backward_weights.comp'),
            buffers=[grad_output, self._saved_input, self.grad_buf,
                     fwd_buf if self.relu else self.output_buf],
            push_constant_size=44,
        )
        push = struct.pack('11i', B, self.in_channels, self.out_channels,
                           self.in_h, self.in_w, self.out_h, self.out_w,
                           self.kernel_size, self.stride, self.padding,
                           1 if self.relu else 0)
        self.gpu.dispatch(pipeline, (n_total + 63) // 64, push_constants=push)
        pipeline.destroy()

        return d_input_buf

    def zero_grad(self):
        self.gpu.zero_buffer(self.grad_buf)

    def parameters(self):
        # Single combined grad buffer for kernel+bias
        n_k = self._param_sizes[0]
        n_b = self._param_sizes[1]
        return [(self.kernel_buf, self.grad_buf, n_k),
                (self.bias_buf, self.grad_buf, n_b)]

    def init_weights(self, rng):
        n_k = self._param_sizes[0]
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        std = float(np.sqrt(2.0 / fan_in))
        kernel = (rng.standard_normal(n_k) * std).astype(np.float32)
        bias = np.zeros(self.out_channels, dtype=np.float32)
        self.gpu.upload(self.kernel_buf, kernel)
        self.gpu.upload(self.bias_buf, bias)

    def save_weights(self):
        k = self.gpu.download(self.kernel_buf, np.float32, self._param_sizes[0])
        b = self.gpu.download(self.bias_buf, np.float32, self._param_sizes[1])
        return np.concatenate([k, b])

    def load_weights(self, flat, offset):
        n_k = self._param_sizes[0]
        n_b = self._param_sizes[1]
        self.gpu.upload(self.kernel_buf, flat[offset:offset+n_k].astype(np.float32))
        offset += n_k
        self.gpu.upload(self.bias_buf, flat[offset:offset+n_b].astype(np.float32))
        offset += n_b
        return offset


# ---------------------------------------------------------------------------
# GAP + Linear (fused)
# ---------------------------------------------------------------------------

class VkGAPLinear(VkLayer):
    """Global Average Pool + Linear(C, 1) → scalar score."""

    def __init__(self, gpu: VkCompute, channels: int, batch_size: int):
        super().__init__(gpu)
        self.channels = channels
        self.batch_size = batch_size

        n_w = channels + 1  # weights + bias
        self.weight_buf = gpu.create_buffer(n_w * 4)
        self._param_bufs = [self.weight_buf]
        self._param_sizes = [n_w]

        self.grad_buf = gpu.create_buffer(n_w * 4)
        self._grad_bufs = [self.grad_buf]

        self.pooled_buf = gpu.create_buffer(batch_size * channels * 4)
        self.output_buf = gpu.create_buffer(batch_size * 4)

        self._spatial_h = 0
        self._spatial_w = 0

    def forward(self, input_buf, batch_size, shape):
        C, H, W = shape
        self._spatial_h = H
        self._spatial_w = W

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'gap_linear_forward.comp'),
            buffers=[input_buf, self.weight_buf, self.pooled_buf, self.output_buf],
            push_constant_size=16,
        )
        push = struct.pack('4i', batch_size, C, H, W)
        self.gpu.dispatch(pipeline, (batch_size + 31) // 32, push_constants=push)
        pipeline.destroy()

        return self.output_buf, (1,)

    def backward(self, grad_output, batch_size):
        C = self.channels
        H, W = self._spatial_h, self._spatial_w
        d_input_buf = self.gpu.create_buffer(batch_size * C * H * W * 4)

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'gap_linear_backward.comp'),
            buffers=[grad_output, self.pooled_buf, self.weight_buf,
                     d_input_buf, self.grad_buf],
            push_constant_size=16,
        )
        push = struct.pack('4i', batch_size, C, H, W)
        gx = (W + 7) // 8
        gy = (H + 7) // 8
        gz = batch_size * C
        self.gpu.dispatch(pipeline, gx, gy, gz, push)
        pipeline.destroy()

        return d_input_buf

    def zero_grad(self):
        self.gpu.zero_buffer(self.grad_buf)

    def init_weights(self, rng):
        C = self.channels
        std = float(np.sqrt(2.0 / C))
        w = (rng.standard_normal(C) * std).astype(np.float32)
        b = np.zeros(1, dtype=np.float32)
        self.gpu.upload(self.weight_buf, np.concatenate([w, b]))


# ---------------------------------------------------------------------------
# GAP + MLP (fused)
# ---------------------------------------------------------------------------

class VkGAPMLP(VkLayer):
    """Global Average Pool + Linear(C,H) + ReLU + Linear(H,1) → scalar."""

    def __init__(self, gpu: VkCompute, channels: int, batch_size: int,
                 hidden: int = 16):
        super().__init__(gpu)
        self.channels = channels
        self.hidden = hidden
        self.batch_size = batch_size

        n_w = channels * hidden + hidden + hidden + 1
        self.weight_buf = gpu.create_buffer(n_w * 4)
        self._param_bufs = [self.weight_buf]
        self._param_sizes = [n_w]

        self.grad_buf = gpu.create_buffer(n_w * 4)
        self._grad_bufs = [self.grad_buf]

        self.pooled_buf = gpu.create_buffer(batch_size * channels * 4)
        self.hidden_buf = gpu.create_buffer(batch_size * hidden * 4)
        self.output_buf = gpu.create_buffer(batch_size * 4)

        self._spatial_h = 0
        self._spatial_w = 0

    def forward(self, input_buf, batch_size, shape):
        C, H, W = shape
        self._spatial_h = H
        self._spatial_w = W

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'gap_mlp_forward.comp'),
            buffers=[input_buf, self.weight_buf, self.pooled_buf,
                     self.output_buf, self.hidden_buf],
            push_constant_size=20,
        )
        push = struct.pack('5i', batch_size, C, H, W, self.hidden)
        self.gpu.dispatch(pipeline, (batch_size + 31) // 32, push_constants=push)
        pipeline.destroy()

        return self.output_buf, (1,)

    def backward(self, grad_output, batch_size):
        C = self.channels
        H, W = self._spatial_h, self._spatial_w
        d_input_buf = self.gpu.create_buffer(batch_size * C * H * W * 4)

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'gap_mlp_backward.comp'),
            buffers=[grad_output, self.pooled_buf, self.weight_buf,
                     d_input_buf, self.grad_buf, self.hidden_buf],
            push_constant_size=20,
        )
        push = struct.pack('5i', batch_size, C, H, W, self.hidden)
        gx = (W + 7) // 8
        gy = (H + 7) // 8
        gz = batch_size * C
        self.gpu.dispatch(pipeline, gx, gy, gz, push)
        pipeline.destroy()

        return d_input_buf

    def zero_grad(self):
        self.gpu.zero_buffer(self.grad_buf)

    def init_weights(self, rng):
        C, H = self.channels, self.hidden
        std1 = float(np.sqrt(2.0 / C))
        w1 = (rng.standard_normal(C * H) * std1).astype(np.float32)
        b1 = np.zeros(H, dtype=np.float32)
        std2 = float(np.sqrt(2.0 / H))
        w2 = (rng.standard_normal(H) * std2).astype(np.float32)
        b2 = np.zeros(1, dtype=np.float32)
        self.gpu.upload(self.weight_buf, np.concatenate([w1, b1, w2, b2]))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VkModel:
    """Sequential model of VkLayers."""

    def __init__(self, gpu: VkCompute, layers: list[VkLayer]):
        self.gpu = gpu
        self.layers = layers
        self._sgd_pipeline = None

    def forward(self, input_buf: VkBuffer, batch_size: int,
                input_shape: tuple[int, ...]) -> VkBuffer:
        """Run forward pass through all layers. Returns output buffer."""
        x = input_buf
        shape = input_shape
        for layer in self.layers:
            x, shape = layer.forward(x, batch_size, shape)
        return x

    def backward(self, grad_output: VkBuffer, batch_size: int) -> None:
        """Run backward pass, accumulating weight gradients."""
        g = grad_output
        for layer in reversed(self.layers):
            g = layer.backward(g, batch_size)

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def sgd_step(self, lr: float) -> None:
        """SGD weight update: w -= lr * grad, then zero grads."""
        for layer in self.layers:
            for param_buf, grad_buf, n in layer.parameters():
                pipeline = self.gpu.create_pipeline(
                    str(SHADER_DIR / 'sgd_update.comp'),
                    buffers=[param_buf, grad_buf],
                    push_constant_size=8,
                )
                push = struct.pack('fI', lr, n)
                self.gpu.dispatch(pipeline, (n + 255) // 256, push_constants=push)
                pipeline.destroy()

    def param_count(self) -> int:
        return sum(l.param_count() for l in self.layers)

    def save_weights(self) -> np.ndarray:
        """Save all weights as a flat float32 array (compatible with .npy)."""
        return np.concatenate([l.save_weights() for l in self.layers])

    def load_weights(self, flat: np.ndarray) -> None:
        """Load weights from flat float32 array."""
        offset = 0
        for layer in self.layers:
            offset = layer.load_weights(flat, offset)
        if offset != len(flat):
            raise ValueError(f'Weight count mismatch: loaded {offset}, '
                             f'file has {len(flat)}')

    def init_weights(self, seed: int = 42) -> None:
        """Initialize all weights with Kaiming uniform."""
        rng = np.random.default_rng(seed)
        for layer in self.layers:
            layer.init_weights(rng)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_cnn_scorer(gpu: VkCompute, layers_config: list[tuple],
                     batch_size: int, image_size: int = 256,
                     mlp_head: bool = False, hidden: int = 16) -> VkModel:
    """Build CNN aesthetic scorer model.

    Args:
        gpu: VkCompute instance
        layers_config: [(in_c, out_c, kernel, stride, padding), ...]
        batch_size: fixed batch size for buffer allocation
        image_size: input image H=W
        mlp_head: use MLP head instead of linear
        hidden: MLP hidden dimension
    """
    layers: list[VkLayer] = []
    h, w = image_size, image_size
    for c_in, c_out, k, s, p in layers_config:
        layers.append(VkConv2d(gpu, c_in, c_out, k, s, p,
                               batch_size=batch_size, in_h=h, in_w=w, relu=True))
        h = (h + 2*p - k) // s + 1
        w = (w + 2*p - k) // s + 1

    final_c = layers_config[-1][1]
    if mlp_head:
        layers.append(VkGAPMLP(gpu, final_c, batch_size, hidden=hidden))
    else:
        layers.append(VkGAPLinear(gpu, final_c, batch_size))

    return VkModel(gpu, layers)
