# CNN Scorer Model Benchmarks

Comparison of model architectures and data representations.
All accuracy measured on cross-gen composite validation involving gen 244.

## Results

| Model | Params | Channels | LR | CPU ms | RSS MB | ES Acc | Personal Acc | Score Std | Notes |
|-------|--------|----------|-----|--------|--------|--------|-------------|-----------|-------|
| 25K RGB (ES base) | 24,665 | RGB+swept | 0.003 | 48 | — | 65.9% | 58.2% | — | Original baseline (71.7% on unrestricted val) |
| 25K HSL | 24,665 | HSL+swept | 0.001 | — | — | 51% | — | — | Failed — circular hue, wrong LR |
| 25K domain (no gamma) | 24,665 | H/S/L/A | 0.001 | — | — | 55.9% | — | — | Missing density contrast |
| 25K domain (gamma) | 24,665 | H/S/L/A | 0.001 | — | — | 58.0% | — | — | Better, capacity limited |
| 25K domain (gamma) | 24,665 | H/S/L/A | 0.003 | — | — | 58.3%+ | 57.0% | — | ES base, not fine-tuned. Measured 2026-05-15 on 60 liked / 146 disliked |
| 55K RGB (wrong LR) | 55,141 | RGB+swept | 0.001 | — | — | 62.1% | — | — | LR too low |
| 55K RGB (ES base) | 55,141 | RGB+swept | 0.003 | 72 | 222 | 69.3% | **42.7%** | 0.734 | Anti-correlated with personal taste! Never deployed. |
| 25K RGB (personal) | 24,665 | RGB+swept | — | 44 | 220 | — | 65.7% | — | Fine-tuned on personal votes from 25K ES base. Measured 2026-05-15. |
| 25K domain (personal) | 24,665 | H/S/L/A | 0.001 | — | — | — | 72.6% | — | Fine-tuned from ES base. Measured 2026-05-16. |
| **25K domain (scratch)** | 24,665 | H/S/L/A | 0.003 | — | — | — | **73.3%** | — | **New best**. Random init, no ES pretraining. 3768 pairs. 2026-05-16. |
| 25K domain MLP (ES) | 25,657 | H/S/L/A | 0.003 | — | — | 59.7% | — | — | MLP head (64→16→1). Worse than linear on ES — harder to optimize? |
| 25K domain MLP (personal) | 25,657 | H/S/L/A | 0.003 | — | — | — | 79.1% / **50.4%** | — | Previously deployed. 79.1% on old smaller pairwise set; 50.4% on current 1725-pair set after pruning + corrected inputs. The 79.1% was a measurement artifact. 2026-05-16/18. |

### v10 era (corrected swept render + standardization + sentinel-aware normalization)

The 79.1% above was achieved with three latent bugs that we found during v10 work:
1. Swept render rotated viewport not affines (spirograph copies, S channel useless)
2. No input standardization (Kaiming init mismatched the input distribution)
3. Sentinel pixels dominated channel variance (cross-corpus normalization broken)

Numbers from v10 era are not directly comparable — different data semantics.

| Model | Seed | Variant | Personal Acc | Notes |
|-------|------|---------|-------------|-------|
| 25K MLP v10 fine-tune | 42 | from broken-S deployed | 40.5% | Old weights can't unlearn. Best=40.5% over 10 epochs, early stop. 2026-05-17. |
| 25K MLP v10 scratch | 42 | LR=0.003, sigmoid+std | 45.8% | Bias init + standardization unlocked learning. Slow climb. 2026-05-17. |
| 25K MLP v10 scratch | 42 | LR=0.005 | 49.4% | Same as above, higher LR. Steady climb to plateau. 2026-05-18. |
| 25K MLP v10 scratch | 42 | LR=0.01 | diverged | Loss exploded to 1e25 at epoch 11. Too high. 2026-05-18. |
| 25K MLP v10 scratch | 1 | LR=0.005 | 45.6% | Seed variance experiment. 2026-05-18. |
| 25K MLP v10 scratch | 2 | LR=0.005 | TBD | Running 2026-05-18. |
| 25K MLP v10 scratch | 3 | LR=0.005 | TBD | Running 2026-05-18. |
| 25K MLP ES pretrain v3 | 42 | sentinel-aware std, LR=0.001 | 55.3% (ES val) | Plateaued by epoch 8. Stopped. 2026-05-18. |
| 25K MLP ES pretrain v3 | 42 | sentinel-aware std, LR=0.003 | 57.7% (ES val) | 30 epochs, still climbing at end. Used as fine-tune base. 2026-05-18. |
| 25K MLP v3 personal fine-tune | 42 | from ES v3 base, LR=0.001 | TBD | Running 2026-05-18. Tests whether ES pretrain transfers to library. |

## Key Findings

### v10 era (post-bug-hunt, 2026-05-17/18)

- **The deployed 79.1% model is actually 50.4% on current data.** That number was measured on a smaller, easier pairwise set before the pruner removed flying dots and pulsars, and before the swept render bug was fixed. Re-benchmarked on the current 1725-pair val set against corrected inputs: 50.4%. Every "we used to be at 79.1%" comparison should be calibrated against 50.4% as the real baseline.
- **Sentinel-aware normalization is required for cross-corpus transfer.** With sentinels included, library and ES had different channel distributions; with sentinels excluded, they're statistically indistinguishable. The previous "ES anti-correlation is structural" conclusion was wrong — it was an artifact of sentinel coverage in the unnormalized input.
- **Input standardization is required for training to escape Kaiming bias.** Without it, ReLU dead-channel collapse limits all training to ~40%.
- **Bias init at 0.1 is a band-aid that helps gradient flow.** Real fix is per-channel standardization.
- **LR sweet spot at 0.005.** 0.003 too slow, 0.01 diverges.
- **Seed variance ±3-4 points.** Single-seed results are anecdotes; best-of-3 is the minimum honest comparison.
- **MLP head (64→16→1) over single Linear** gives the model room to carve multiple "good" regions for split-solution-space data. ~1K extra params, real accuracy gain.
- **Swept render bug** (rotating viewport not affines) destroyed S channel for entire pre-v10 history. All prior swept-related findings invalid.

### Pre-v10 (historical, retain for context)

- RGB representation is fundamentally flawed: green channel encodes palette index (34% saliency), not real color
- Three distinct "good" archetypes (dense movers, sparse structural, chaotic compelling) — MLP head addresses this
- Pairwise accuracy << liked/disliked because active learning concentrates pairwise in the hard low-score range

## Pending Tests

- [ ] ES pretrain v3 → library fine-tune (in progress, ES at 54% epoch 2)
- [ ] Best-of-3 seed comparison at LR=0.005 (seeds 1 done at 45.6%, seeds 2/3 running)
- [ ] Lower-capacity 5K-10K model (does smaller help generalization?)
- [ ] Recompute library normalization with sentinel exclusion, retrain
- [ ] 55K and 100K domain MLP — does more capacity help?

## Methodology

- **ES validation**: 2000 cross-gen composite pairs involving gen 244, composite = gen_rank + normalized_within_gen_rating
- **Personal validation**: pairwise_ratings table (direct A/B comparisons) or genome ratings (like/dislike aggregated)
- **Inference timing**: 1000 runs on random 256×256×4 tensor, CPU time from time.process_time()
- **Score distribution**: std dev and range across ~3500 scored genomes
- **Training**: Vulkan compute shaders on Arc A770, SGD, margin ranking loss

# Beat Detection RNN

Distilled from BeatNet (~400K params LSTM+particle filter) into a tiny streaming CRNN.

## Architecture

- Input: 216 features per frame (108-bin CQT spectrum + 108-bin first derivative)
- Linear projection (216 → 32)
- GRU(32 → 48 hidden, 1 layer)
- Linear (48 → 3) → softmax (non-beat / beat / downbeat)
- ~25K params, target inference cost ~10μs per frame on CPU

## Training data

- 7519 .npz files from `generate_beat_labels.py` (BeatNet as teacher)
- 138GB on disk, ~170GB uncompressed (must stream — won't fit in RAM)
- LazyFileCache with 64-file LRU window (~640MB working set)

## Training pipeline (2026-05-19)

Sequence-mode dispatches: forward + backward each run as ONE Vulkan
dispatch per layer per batch, not T dispatches per layer.

- `linear_in.forward(input_seq, B*T)` — flat-batch one dispatch
- `gru.forward_sequence(...)` — single dispatch via `gru_seq_forward.comp`
  with internal T-step loop and shared-memory hidden-state handoff
- `linear_out.forward(gru_seq_out, B*T)` — flat-batch one dispatch
- `cross_entropy_dispatch(..., batch=B*T, t_inv=1.0)` — single dispatch
  computes softmax + CE loss + grad with per-(t, b) accumulation
- `linear_out.backward(grad_acc, B*T)` — one dispatch
- `gru.backward_sequence(linear_out._grad_input, B, T)` — single
  dispatch via `gru_seq_backward.comp` with internal BPTT loop and
  true per-timestep upstream gradient handling
- `linear_in.backward(gru._grad_input, B*T)` — one dispatch

Per-batch total: ~9 dispatches (vs ~4T = ~1000 in the previous
per-timestep design at T=256).

## Throughput (Arc A770, smoke at B=8 T=64 hidden=48)

- Previous per-timestep design: ~236ms/batch, backward dominated
  at 87% (T per-step BPTT dispatches)
- Current seq-mode: ~56ms/batch, backward 47%, forward 15%, SGD 28%
- ~4.2x speedup on small smoke; bigger at production batch sizes
  because per-batch overhead amortizes harder

## Optimizations applied this session (2026-05-19)

1. **Lazy file loading via LRU cache** — 7519 .npz files at 64-file
   LRU window = ~640MB working set instead of 170GB if eagerly
   loaded. Required because the eager loader was OOM-crashing the
   system.
2. **GPU cross-entropy** — softmax + CE loss + gradient on the GPU,
   no per-timestep logits download. Per-batch downloads: T → 1.
   Also fixed a label-layout bug — labels were uploaded as (B, T, 3)
   but indexed as (T, B, 3); now transposed at upload.
3. **Sequence-mode forward** — single GRU dispatch over T timesteps
   with shared-memory hidden state. Linear layers run on B*T flat
   batch. 4 forward dispatches per batch instead of 3*T.
4. **Sequence-mode backward** — single BPTT dispatch via
   `gru_seq_backward.comp` with true per-timestep upstream
   contribution from `linear_out._grad_input_buf` (shape (T, B, H)).
   Replaces T per-step dispatches AND fixes the
   last-step-only-upstream BPTT approximation that previously limited
   gradient flow. Smoke val_acc improved 0.844 → 0.916 from this
   correctness fix.
5. **CPU-side pipelining** — `BatchPrepPipeline` materializes +
   transposes batch N+1 on a background thread while GPU processes
   N. LazyFileCache made thread-safe with a lock.
6. **Profile flag** — `--profile` prints per-phase wall-time
   breakdown every 10 batches: wait_prep | reset | upload | forward
   | loss | backward | sgd.
7. **Resumable training** — `.npz` checkpoint with `weights`,
   `next_epoch`, `best_val_loss`. Mid-epoch checkpoint saves at
   `--val-interval` wall-time intervals (default 7200s = 2h), with
   val sampling (`--val-sample-size`, default 200 random batches)
   to keep mid-epoch validation cheap.
