"""Numerical tests for Vulkan compute shaders against PyTorch reference.

Each test creates small tensors, runs the PyTorch equivalent, dispatches
the Vulkan shader, and verifies outputs match within tolerance.

Requires: Vulkan GPU, PyTorch, numpy.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
import torch.nn.functional as F

# Add tools to path for SHADER_DIR
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'tools'))

SHADER_DIR = Path(__file__).resolve().parent.parent.parent / 'flame_sheep' / 'shaders' / 'cnn'

try:
    from flame_sheep.vk_compute import VkCompute
    gpu = VkCompute()
    HAS_VK = True
except Exception:
    HAS_VK = False
    gpu = None

pytestmark = pytest.mark.skipif(not HAS_VK, reason='No Vulkan GPU available')

ATOL = 1e-4  # float32 tolerance


@pytest.fixture(scope='module')
def vk():
    return gpu


# ----------------------------------------------------------------
# Conv2d forward
# ----------------------------------------------------------------

class TestConv2dForward:
    def test_basic(self, vk):
        B, Ci, Co, H, W, K, S, P = 2, 4, 8, 16, 16, 3, 2, 1
        Ho = (H + 2*P - K) // S + 1
        Wo = (W + 2*P - K) // S + 1

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((B, Ci, H, W)).astype(np.float32)
        weight = rng.standard_normal((Co, Ci, K, K)).astype(np.float32)
        bias = rng.standard_normal(Co).astype(np.float32)

        # PyTorch reference
        t_inp = torch.from_numpy(inp)
        t_w = torch.from_numpy(weight)
        t_b = torch.from_numpy(bias)
        ref = F.conv2d(t_inp, t_w, t_b, stride=S, padding=P)
        ref = F.relu(ref)  # shader does ReLU by default

        # Vulkan
        inp_buf = vk.create_buffer(inp.nbytes)
        w_buf = vk.create_buffer(weight.nbytes)
        b_buf = vk.create_buffer(bias.nbytes)
        out_buf = vk.create_buffer(B * Co * Ho * Wo * 4)
        vk.upload(inp_buf, inp)
        vk.upload(w_buf, weight)
        vk.upload(b_buf, bias)

        pipeline = vk.create_pipeline(
            str(SHADER_DIR / 'conv2d_forward.comp'),
            buffers=[inp_buf, w_buf, b_buf, out_buf],
            push_constant_size=44,
        )
        push = struct.pack('11i', B, Ci, Co, H, W, Ho, Wo, K, S, P, 1)
        vk.dispatch(pipeline, (Wo+7)//8, (Ho+7)//8, B*Co, push)
        result = vk.download(out_buf, np.float32, B * Co * Ho * Wo)
        result = result.reshape(B, Co, Ho, Wo)

        np.testing.assert_allclose(result, ref.numpy(), atol=ATOL)

        pipeline.destroy()
        for b in [inp_buf, w_buf, b_buf, out_buf]:
            b.destroy()

    def test_no_relu(self, vk):
        B, Ci, Co, H, W, K, S, P = 1, 2, 4, 8, 8, 3, 1, 1
        Ho = (H + 2*P - K) // S + 1
        Wo = (W + 2*P - K) // S + 1

        rng = np.random.default_rng(123)
        inp = rng.standard_normal((B, Ci, H, W)).astype(np.float32)
        weight = rng.standard_normal((Co, Ci, K, K)).astype(np.float32)
        bias = rng.standard_normal(Co).astype(np.float32)

        ref = F.conv2d(torch.from_numpy(inp), torch.from_numpy(weight),
                       torch.from_numpy(bias), stride=S, padding=P)

        inp_buf = vk.create_buffer(inp.nbytes)
        w_buf = vk.create_buffer(weight.nbytes)
        b_buf = vk.create_buffer(bias.nbytes)
        out_buf = vk.create_buffer(B * Co * Ho * Wo * 4)
        vk.upload(inp_buf, inp)
        vk.upload(w_buf, weight)
        vk.upload(b_buf, bias)

        pipeline = vk.create_pipeline(
            str(SHADER_DIR / 'conv2d_forward.comp'),
            buffers=[inp_buf, w_buf, b_buf, out_buf],
            push_constant_size=44,
        )
        push = struct.pack('11i', B, Ci, Co, H, W, Ho, Wo, K, S, P, 0)  # relu=0
        vk.dispatch(pipeline, (Wo+7)//8, (Ho+7)//8, B*Co, push)
        result = vk.download(out_buf, np.float32, B*Co*Ho*Wo).reshape(B, Co, Ho, Wo)

        np.testing.assert_allclose(result, ref.numpy(), atol=ATOL)

        pipeline.destroy()
        for b in [inp_buf, w_buf, b_buf, out_buf]:
            b.destroy()


# ----------------------------------------------------------------
# Conv2d backward (input gradient)
# ----------------------------------------------------------------

class TestConv2dBackwardInput:
    def test_basic(self, vk):
        B, Ci, Co, H, W, K, S, P = 2, 4, 8, 16, 16, 3, 2, 1
        Ho = (H + 2*P - K) // S + 1
        Wo = (W + 2*P - K) // S + 1

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((B, Ci, H, W)).astype(np.float32)
        weight = rng.standard_normal((Co, Ci, K, K)).astype(np.float32)
        bias = rng.standard_normal(Co).astype(np.float32)

        # PyTorch reference: forward + backward
        t_inp = torch.from_numpy(inp).requires_grad_(True)
        t_w = torch.from_numpy(weight)
        t_b = torch.from_numpy(bias)
        out = F.conv2d(t_inp, t_w, t_b, stride=S, padding=P)
        out = F.relu(out)
        d_out = rng.standard_normal(out.shape).astype(np.float32)
        out.backward(torch.from_numpy(d_out))
        ref_d_input = t_inp.grad.numpy()

        # Vulkan: need forward output for ReLU mask
        fwd_out = out.detach().numpy()
        # ReLU mask: d_out * (fwd_out > 0)
        d_relu = d_out * (fwd_out > 0).astype(np.float32)

        d_relu_buf = vk.create_buffer(d_relu.nbytes)
        w_buf = vk.create_buffer(weight.nbytes)
        d_input_buf = vk.create_buffer(inp.nbytes)
        vk.upload(d_relu_buf, d_relu)
        vk.upload(w_buf, weight)
        vk.zero_buffer(d_input_buf)

        pipeline = vk.create_pipeline(
            str(SHADER_DIR / 'conv2d_backward_input.comp'),
            buffers=[d_relu_buf, w_buf, d_input_buf],
            push_constant_size=44,
        )
        push = struct.pack('11i', B, Ci, Co, H, W, Ho, Wo, K, S, P, 0)
        gx = (W + 7) // 8
        gy = (H + 7) // 8
        gz = B * Ci
        vk.dispatch(pipeline, gx, gy, gz, push)
        result = vk.download(d_input_buf, np.float32, B*Ci*H*W).reshape(B, Ci, H, W)

        np.testing.assert_allclose(result, ref_d_input, atol=ATOL)

        pipeline.destroy()
        for b in [d_relu_buf, w_buf, d_input_buf]:
            b.destroy()


# ----------------------------------------------------------------
# GAP + Linear forward
# ----------------------------------------------------------------

class TestGapLinearForward:
    def test_basic(self, vk):
        B, C, H, W = 4, 16, 8, 8

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((B, C, H, W)).astype(np.float32)
        lin_w = rng.standard_normal(C).astype(np.float32)
        lin_b = np.array([rng.standard_normal()], dtype=np.float32)
        weights = np.concatenate([lin_w, lin_b])

        # PyTorch reference
        t_inp = torch.from_numpy(inp)
        pooled_ref = t_inp.mean(dim=(2, 3))  # (B, C)
        scores_ref = (pooled_ref * torch.from_numpy(lin_w)).sum(dim=1) + lin_b[0]

        # Vulkan
        inp_buf = vk.create_buffer(inp.nbytes)
        w_buf = vk.create_buffer(weights.nbytes)
        pooled_buf = vk.create_buffer(B * C * 4)
        out_buf = vk.create_buffer(B * 4)
        vk.upload(inp_buf, inp)
        vk.upload(w_buf, weights)

        pipeline = vk.create_pipeline(
            str(SHADER_DIR / 'gap_linear_forward.comp'),
            buffers=[inp_buf, w_buf, pooled_buf, out_buf],
            push_constant_size=16,
        )
        push = struct.pack('4i', B, C, H, W)
        vk.dispatch(pipeline, (B+31)//32, push_constants=push)
        scores = vk.download(out_buf, np.float32, B)
        pooled = vk.download(pooled_buf, np.float32, B * C).reshape(B, C)

        np.testing.assert_allclose(pooled, pooled_ref.numpy(), atol=ATOL)
        np.testing.assert_allclose(scores, scores_ref.numpy(), atol=ATOL)

        pipeline.destroy()
        for b in [inp_buf, w_buf, pooled_buf, out_buf]:
            b.destroy()


# ----------------------------------------------------------------
# GAP + MLP forward
# ----------------------------------------------------------------

class TestGapMlpForward:
    def test_basic(self, vk):
        B, C, H, W = 4, 16, 8, 8
        HIDDEN = 16

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((B, C, H, W)).astype(np.float32)
        w1 = rng.standard_normal((C, HIDDEN)).astype(np.float32)
        b1 = rng.standard_normal(HIDDEN).astype(np.float32)
        w2 = rng.standard_normal(HIDDEN).astype(np.float32)
        b2 = np.array([rng.standard_normal()], dtype=np.float32)
        weights = np.concatenate([w1.flatten(), b1, w2, b2])

        # PyTorch reference
        t_inp = torch.from_numpy(inp)
        pooled_ref = t_inp.mean(dim=(2, 3))  # (B, C)
        hidden_ref = pooled_ref @ torch.from_numpy(w1) + torch.from_numpy(b1)
        relu_ref = F.relu(hidden_ref)
        scores_ref = (relu_ref * torch.from_numpy(w2)).sum(dim=1) + b2[0]

        # Vulkan
        inp_buf = vk.create_buffer(inp.nbytes)
        w_buf = vk.create_buffer(weights.nbytes)
        pooled_buf = vk.create_buffer(B * C * 4)
        out_buf = vk.create_buffer(B * 4)
        hidden_buf = vk.create_buffer(B * HIDDEN * 4)
        vk.upload(inp_buf, inp)
        vk.upload(w_buf, weights)

        pipeline = vk.create_pipeline(
            str(SHADER_DIR / 'gap_mlp_forward.comp'),
            buffers=[inp_buf, w_buf, pooled_buf, out_buf, hidden_buf],
            push_constant_size=20,
        )
        push = struct.pack('5i', B, C, H, W, HIDDEN)
        vk.dispatch(pipeline, (B+31)//32, push_constants=push)
        scores = vk.download(out_buf, np.float32, B)
        pooled = vk.download(pooled_buf, np.float32, B * C).reshape(B, C)
        hidden = vk.download(hidden_buf, np.float32, B * HIDDEN).reshape(B, HIDDEN)

        np.testing.assert_allclose(pooled, pooled_ref.numpy(), atol=ATOL)
        np.testing.assert_allclose(hidden, hidden_ref.numpy(), atol=ATOL)
        np.testing.assert_allclose(scores, scores_ref.numpy(), atol=ATOL)

        pipeline.destroy()
        for b in [inp_buf, w_buf, pooled_buf, out_buf, hidden_buf]:
            b.destroy()


# ----------------------------------------------------------------
# GAP + Linear backward
# ----------------------------------------------------------------

class TestGapLinearBackward:
    def test_basic(self, vk):
        B, C, H, W = 4, 16, 8, 8

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((B, C, H, W)).astype(np.float32)
        lin_w = rng.standard_normal(C).astype(np.float32)
        lin_b = np.array([rng.standard_normal()], dtype=np.float32)
        weights = np.concatenate([lin_w, lin_b])

        # PyTorch reference
        t_inp = torch.from_numpy(inp).requires_grad_(True)
        pooled = t_inp.mean(dim=(2, 3))
        t_w = torch.from_numpy(lin_w)
        scores = (pooled * t_w).sum(dim=1) + lin_b[0]
        d_scores = rng.standard_normal(B).astype(np.float32)
        scores.backward(torch.from_numpy(d_scores))
        ref_d_input = t_inp.grad.numpy()

        # Vulkan
        d_out_buf = vk.create_buffer(B * 4)
        pooled_buf = vk.create_buffer(B * C * 4)
        w_buf = vk.create_buffer(weights.nbytes)
        d_input_buf = vk.create_buffer(inp.nbytes)
        d_w_buf = vk.create_buffer(weights.nbytes)

        vk.upload(d_out_buf, np.array(d_scores, dtype=np.float32))
        vk.upload(pooled_buf, pooled.detach().numpy())
        vk.upload(w_buf, weights)
        vk.zero_buffer(d_w_buf)

        pipeline = vk.create_pipeline(
            str(SHADER_DIR / 'gap_linear_backward.comp'),
            buffers=[d_out_buf, pooled_buf, w_buf, d_input_buf, d_w_buf],
            push_constant_size=16,
        )
        push = struct.pack('4i', B, C, H, W)
        gx = (W + 7) // 8
        gy = (H + 7) // 8
        gz = B * C
        vk.dispatch(pipeline, gx, gy, gz, push)

        result = vk.download(d_input_buf, np.float32, B*C*H*W).reshape(B, C, H, W)
        np.testing.assert_allclose(result, ref_d_input, atol=ATOL)

        # Also check weight gradients
        d_weights = vk.download(d_w_buf, np.float32, len(weights))
        # d_w[c] = sum_b(d_scores[b] * pooled[b, c])
        ref_d_w = (torch.from_numpy(d_scores).unsqueeze(1) * pooled.detach()).sum(0).numpy()
        ref_d_b = d_scores.sum()
        np.testing.assert_allclose(d_weights[:C], ref_d_w, atol=ATOL)
        np.testing.assert_allclose(d_weights[C], ref_d_b, atol=ATOL)

        pipeline.destroy()
        for b in [d_out_buf, pooled_buf, w_buf, d_input_buf, d_w_buf]:
            b.destroy()


# ----------------------------------------------------------------
# SGD update
# ----------------------------------------------------------------

class TestSgdUpdate:
    def test_basic(self, vk):
        N = 256
        lr = 0.01

        rng = np.random.default_rng(42)
        weights = rng.standard_normal(N).astype(np.float32)
        grads = rng.standard_normal(N).astype(np.float32)
        expected = weights - lr * grads

        # Shader expects grads as uint (CAS atomic float-as-uint pattern)
        grads_u = grads.view(np.uint32)

        w_buf = vk.create_buffer(weights.nbytes)
        g_buf = vk.create_buffer(grads_u.nbytes)
        vk.upload(w_buf, weights)
        vk.upload(g_buf, grads_u)

        pipeline = vk.create_pipeline(
            str(SHADER_DIR / 'sgd_update.comp'),
            buffers=[w_buf, g_buf],
            push_constant_size=8,
        )
        # Push constant order: (lr: float, count: uint)
        push = struct.pack('fI', lr, N)
        vk.dispatch(pipeline, (N + 255) // 256, push_constants=push)
        result = vk.download(w_buf, np.float32, N)

        np.testing.assert_allclose(result, expected, atol=ATOL)

        # Verify grads zeroed
        grads_after = vk.download(g_buf, np.uint32, N)
        assert np.all(grads_after == 0)

        pipeline.destroy()
        w_buf.destroy()
        g_buf.destroy()
