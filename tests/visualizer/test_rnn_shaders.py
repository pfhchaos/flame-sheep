"""Numerical tests for RNN shader layers — verify against PyTorch.

Each test:
1. Creates random inputs + weights
2. Runs the Vulkan shader
3. Runs the PyTorch equivalent
4. Asserts outputs match within tolerance

Requires: GPU context (Vulkan compute), PyTorch.
"""

import numpy as np
import pytest

try:
    from flame_sheep.vk_compute import VkCompute
    gpu = VkCompute()
    HAS_GPU = True
except Exception:
    HAS_GPU = False
    gpu = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = [
    pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context'),
    pytest.mark.skipif(not HAS_TORCH, reason='PyTorch not available'),
]


class TestVkLinearForward:
    """VkLinear forward must match torch.nn.Linear."""

    def test_linear_no_relu(self):
        from flame_sheep.wallpaper_ml import VkLinear

        B, I, O = 4, 16, 8
        rng = np.random.default_rng(42)

        # Random input + weights
        x = rng.standard_normal((B, I)).astype(np.float32)
        W = rng.standard_normal((I, O)).astype(np.float32)
        b = rng.standard_normal(O).astype(np.float32)

        # Vulkan
        layer = VkLinear(gpu, I, O, batch_size=B, relu=False)
        gpu.upload(layer.weight_buf, W.ravel())
        gpu.upload(layer.bias_buf, b)

        in_buf = gpu.create_buffer(B * I * 4)
        gpu.upload(in_buf, x.ravel())

        out_buf, shape = layer.forward(in_buf, B, (I,))
        vk_out = gpu.download(out_buf, np.float32, B * O).reshape(B, O)

        # PyTorch reference
        # W layout in shader: [I, O], torch.nn.Linear weight is [O, I]
        lin = torch.nn.Linear(I, O, bias=True)
        lin.weight.data = torch.from_numpy(W.T.copy())  # [O, I]
        lin.bias.data = torch.from_numpy(b)
        with torch.no_grad():
            pt_out = lin(torch.from_numpy(x)).numpy()

        np.testing.assert_allclose(vk_out, pt_out, atol=1e-4, rtol=1e-4)

    def test_linear_with_relu(self):
        from flame_sheep.wallpaper_ml import VkLinear

        B, I, O = 4, 32, 16
        rng = np.random.default_rng(123)

        x = rng.standard_normal((B, I)).astype(np.float32)
        W = rng.standard_normal((I, O)).astype(np.float32)
        b = rng.standard_normal(O).astype(np.float32)

        layer = VkLinear(gpu, I, O, batch_size=B, relu=True)
        gpu.upload(layer.weight_buf, W.ravel())
        gpu.upload(layer.bias_buf, b)

        in_buf = gpu.create_buffer(B * I * 4)
        gpu.upload(in_buf, x.ravel())

        out_buf, shape = layer.forward(in_buf, B, (I,))
        vk_out = gpu.download(out_buf, np.float32, B * O).reshape(B, O)

        # PyTorch reference: Linear + ReLU
        lin = torch.nn.Linear(I, O, bias=True)
        lin.weight.data = torch.from_numpy(W.T.copy())
        lin.bias.data = torch.from_numpy(b)
        with torch.no_grad():
            pt_out = torch.relu(lin(torch.from_numpy(x))).numpy()

        np.testing.assert_allclose(vk_out, pt_out, atol=1e-4, rtol=1e-4)


class TestVkGRUForward:
    """VkGRU forward must match torch.nn.GRU."""

    def test_single_frame(self):
        from flame_sheep.wallpaper_ml import VkGRU

        B, I, H = 2, 8, 4
        rng = np.random.default_rng(99)

        # Random input, hidden, weights
        x = rng.standard_normal((B, I)).astype(np.float32)
        h0 = rng.standard_normal((B, H)).astype(np.float32)

        # PyTorch GRU for reference
        gru = torch.nn.GRU(I, H, num_layers=1, batch_first=True, bias=True)

        # Extract PyTorch weights → our layout
        # PyTorch weight_ih: [3*H, I] (gates: r, z, n for PyTorch ordering)
        # PyTorch weight_hh: [3*H, H]
        # PyTorch bias_ih: [3*H], bias_hh: [3*H]
        # PyTorch gate order: reset(r), update(z), new(n)
        # Our gate order: update(z), reset(r), candidate(h)
        with torch.no_grad():
            wih = gru.weight_ih_l0.numpy()  # [3*H, I]
            whh = gru.weight_hh_l0.numpy()  # [3*H, H]
            bih = gru.bias_ih_l0.numpy()    # [3*H]
            bhh = gru.bias_hh_l0.numpy()    # [3*H]

        # PyTorch order: r=0, z=1, n=2
        # Our order:     z=0, r=1, h=2
        # Rearrange: our[0]=pt[1], our[1]=pt[0], our[2]=pt[2]
        perm = [1, 0, 2]  # z, r, n → our z, r, h

        # W: [3, I, H] — our layout. PyTorch has [3*H, I], need transpose per gate
        W = np.zeros((3, I, H), dtype=np.float32)
        for g in range(3):
            pg = perm[g]
            W[g] = wih[pg*H:(pg+1)*H, :].T  # [H, I].T = [I, H]

        # U: [3, H, H]
        U = np.zeros((3, H, H), dtype=np.float32)
        for g in range(3):
            pg = perm[g]
            U[g] = whh[pg*H:(pg+1)*H, :].T  # [H, H].T = [H, H]

        # Bias: [6, H] = [bW_z, bW_r, bW_h, bU_z, bU_r, bU_h]
        bias = np.zeros(6 * H, dtype=np.float32)
        for g in range(3):
            pg = perm[g]
            bias[g * H:(g+1) * H] = bih[pg*H:(pg+1)*H]
            bias[(g+3) * H:(g+4) * H] = bhh[pg*H:(pg+1)*H]

        # Run Vulkan GRU
        layer = VkGRU(gpu, I, H, batch_size=B, max_seq_len=4)
        gpu.upload(layer.W_buf, W.ravel())
        gpu.upload(layer.U_buf, U.ravel())
        gpu.upload(layer.bias_buf, bias)
        gpu.upload(layer.hidden_buf, h0.ravel())

        in_buf = gpu.create_buffer(B * I * 4)
        gpu.upload(in_buf, x.ravel())

        out_buf, shape = layer.forward(in_buf, B, (I,))
        vk_h = gpu.download(out_buf, np.float32, B * H).reshape(B, H)

        # Run PyTorch GRU
        x_t = torch.from_numpy(x).unsqueeze(1)    # [B, 1, I]
        h0_t = torch.from_numpy(h0).unsqueeze(0)  # [1, B, H]
        with torch.no_grad():
            _, hn = gru(x_t, h0_t)
        pt_h = hn.squeeze(0).numpy()  # [B, H]

        np.testing.assert_allclose(vk_h, pt_h, atol=1e-4, rtol=1e-4)

    def test_multi_frame_sequence(self):
        """Multiple frames should match PyTorch GRU on a sequence."""
        from flame_sheep.wallpaper_ml import VkGRU

        B, I, H, T = 2, 8, 4, 5
        rng = np.random.default_rng(77)

        x_seq = rng.standard_normal((B, T, I)).astype(np.float32)
        h0 = np.zeros((B, H), dtype=np.float32)

        # PyTorch GRU
        gru = torch.nn.GRU(I, H, num_layers=1, batch_first=True, bias=True)
        with torch.no_grad():
            wih = gru.weight_ih_l0.numpy()
            whh = gru.weight_hh_l0.numpy()
            bih = gru.bias_ih_l0.numpy()
            bhh = gru.bias_hh_l0.numpy()

        perm = [1, 0, 2]
        W = np.zeros((3, I, H), dtype=np.float32)
        U = np.zeros((3, H, H), dtype=np.float32)
        bias = np.zeros(6 * H, dtype=np.float32)
        for g in range(3):
            pg = perm[g]
            W[g] = wih[pg*H:(pg+1)*H, :].T
            U[g] = whh[pg*H:(pg+1)*H, :].T
            bias[g * H:(g+1) * H] = bih[pg*H:(pg+1)*H]
            bias[(g+3) * H:(g+4) * H] = bhh[pg*H:(pg+1)*H]

        # Run Vulkan GRU frame-by-frame
        layer = VkGRU(gpu, I, H, batch_size=B, max_seq_len=T)
        gpu.upload(layer.W_buf, W.ravel())
        gpu.upload(layer.U_buf, U.ravel())
        gpu.upload(layer.bias_buf, bias)
        gpu.upload(layer.hidden_buf, h0.ravel())

        vk_hiddens = []
        for t in range(T):
            frame = x_seq[:, t, :].copy()
            in_buf = gpu.create_buffer(B * I * 4)
            gpu.upload(in_buf, frame.ravel())
            out_buf, _ = layer.forward(in_buf, B, (I,))
            vk_h = gpu.download(out_buf, np.float32, B * H).reshape(B, H)
            vk_hiddens.append(vk_h.copy())

        # PyTorch: full sequence
        x_t = torch.from_numpy(x_seq)
        h0_t = torch.from_numpy(h0).unsqueeze(0)
        with torch.no_grad():
            pt_all, _ = gru(x_t, h0_t)
        pt_hiddens = pt_all.numpy()  # [B, T, H]

        # Compare final hidden state
        np.testing.assert_allclose(
            vk_hiddens[-1], pt_hiddens[:, -1, :], atol=1e-4, rtol=1e-4)

        # Compare all timesteps
        for t in range(T):
            np.testing.assert_allclose(
                vk_hiddens[t], pt_hiddens[:, t, :], atol=1e-4, rtol=1e-4)


class TestBeatCRNNForward:
    """End-to-end BeatCRNN forward produces correct shape."""

    def test_build_and_forward(self):
        from flame_sheep.wallpaper_ml import build_beat_crnn

        B = 4
        input_size = 216  # 108 CQT + 108 diff
        model = build_beat_crnn(gpu, input_size=input_size, hidden_size=48,
                                n_classes=3, batch_size=B)
        model.init_weights(seed=42)

        # Random input frame
        x = np.random.default_rng(0).standard_normal((B, input_size)).astype(np.float32)
        in_buf = gpu.create_buffer(B * input_size * 4)
        gpu.upload(in_buf, x.ravel())

        out_buf = model.forward(in_buf, B, (input_size,))
        output = gpu.download(out_buf, np.float32, B * 3).reshape(B, 3)

        # Should produce finite outputs
        assert output.shape == (B, 3)
        assert np.all(np.isfinite(output))

    def test_param_count(self):
        from flame_sheep.wallpaper_ml import build_beat_crnn

        model = build_beat_crnn(gpu, input_size=216, hidden_size=48,
                                n_classes=3, batch_size=1)
        n = model.param_count()
        # Linear(216→32): 216*32 + 32 = 6944
        # GRU(32→48): 3*32*48 + 3*48*48 + 6*48 = 4608+6912+288 = 11808
        # Linear(48→3): 48*3 + 3 = 147
        # Total: 18899
        assert n == 6944 + 11808 + 147
