# Session handoff for the freshly-compacted alter ego — 2026-05-19

Hi. You got compacted. The "today's session" context you carry is from
before today's actual work. Here's what's on disk now that wasn't there
when your previous transcript ended.

## TL;DR

1. **Beat-RNN training is fast now.** ~4-10x throughput improvement from
   sequence-mode forward + backward dispatches plus CPU-side pipelining.
   Plus a correctness fix that bumped smoke val_acc 0.844 → 0.916.
2. **Generational data architecture is partially shipped.** Schema +
   weighting + retrain heuristic + pruner-thread integration are all on
   disk. Breeding loop still deferred.
3. **The CNN scorer (gen 1) is live** at
   `flame_sheep/data/cnn_scorer_personal_vk.npz` — the v3 pairwise-only
   model from yesterday is now the deployed model for compare-mode
   scoring and wallpaper use.

## What's running

User has the wallpaper running. Pruner_worker now does a periodic
retrain-recommendation log every 5 minutes — that's expected output, not
a bug. Look for `[retrain]` log lines.

User has been training the beat-RNN in foreground at various points.
Check `~/datasets/beat-labels/beat_rnn_v1.npz` mtime + the live log if
they reference it. Loss was dropping cleanly from ~0.93 → ~0.69 last
time I looked.

## Today's commits (most recent first)

```
a7267ef  pruner_worker: periodic retrain recommendation log
d6f7530  generational: retrain heuristic + pair-weighting for mixed-mode training
d159bfc  generational schema: tag ratings with generation, intra-gen synthesis
32a2f0f  gru_seq_backward: BPTT in one dispatch, true per-timestep upstream
7dd872f  train_beat_rnn: pipeline batch prep on background thread
3699b28  train_beat_rnn: seq-mode training (1 dispatch per layer per batch)
d4f46c0  gru_seq_forward: single-dispatch GRU over T timesteps
6dca1b9  train_beat_rnn: GPU loss + input-offset linear layer
4ee664d  train_beat_rnn: sample val for mid-epoch checks
2a3c815  train_beat_rnn: mid-epoch validation + resumable training
749ca11  train_beat_rnn: lazy file loading via LRU cache
e62220d  eval_cnn_on_library_val: reuse model across checkpoints
3e5a9b1  cnn_scorer: route .npz weights to vk loader
```

You probably remember up through `c742c1f` (yesterday's v3 CNN era
commit). Everything from `3e5a9b1` forward is today.

## Beat-RNN throughput pipeline (read `docs/model_benchmarks.md`)

Each batch is now ~9 dispatches total instead of ~4*T (~1000 at T=256):

- `linear_in.forward(input_seq, B*T)` — flat-batch one dispatch
- `gru.forward_sequence(...)` — single dispatch via `gru_seq_forward.comp`
- `linear_out.forward(...)` — flat-batch one dispatch
- `cross_entropy_dispatch(...)` — softmax + CE + grad in one dispatch
- `linear_out.backward(grad_acc, B*T)` — one dispatch
- `gru.backward_sequence(linear_out._grad_input, B, T)` — single
  BPTT dispatch via `gru_seq_backward.comp`, true per-timestep upstream
- `linear_in.backward(gru._grad_input, B*T)` — one dispatch

Sequence-mode shaders live at
`flame_sheep/shaders/rnn/gru_seq_{forward,backward}.comp`.

Per-timestep BPTT upstream addition is a correctness fix — the previous
backward only seeded `dh` from the last timestep's upstream, treating
intermediate timesteps' gradients as zero. New backward reads
`upstream_dh[t]` at every step and adds to running recurrent grad.

Other beat-RNN features:
- `BatchPrepPipeline` (background thread) — materializes + transposes
  batches ahead of the GPU. LazyFileCache is thread-safe (lock around
  cache writes; np.load decompression outside the lock).
- `--profile` flag prints per-phase wall-time breakdown.
- Resumable training via `.npz` checkpoint with `next_epoch` +
  `best_val_loss`. `--val-interval` (default 7200s) controls mid-epoch
  checkpoint cadence. `--val-sample-size` (default 200) bounds the
  mid-epoch val pass — the full val set is 21K batches which would
  take 15h.

## Generational architecture (read `docs/generational_architecture.md`)

The doc has a status section that's currently accurate. The migration
has run on the live DB. Key state:

- `current_generation` in metadata = 1
- All 732 existing ratings + 1954 pairwise rows tagged with generation 0
- New rating-insertions stamp the current generation
- `load_training_pairs` filters thumbs to current-gen-only for synthesis
- Pairwise pairs survive across generations (durable)
- Pair weights: real pairwise = `K*M/(K+M)`, synthesized = 1.0
  (effective sample size correction; see doc for the math)
- `pruner_worker._pruner_main` calls `Library.retrain_recommendation()`
  every 5 min and logs the result

### Deferred (still need to ship)

- **Breeding loop**: `advance_generation()` is implemented but never
  called yet. Need a runbook (or script) that: trains gen N+1 model →
  breeds new population using that model → calls `advance_generation()`
  → resets per-gen state. User suggested they'll bootstrap gen 1 by
  running `python -m flame_sheep --evolve --loop-length 6` in a loop
  for ~20 iterations to seed new genomes; that produces gen-1-era
  genomes (created_at after current_generation = 1 was set), which is
  what compare mode will pick from.

- **"Evolve ignores downvoted genomes"**: user raised this as worth
  thinking about — pro: don't waste compute breeding from things they
  dislike. Con: gen-0 downvotes may not reflect current taste; hard
  exclusion shrinks parent pool and risks premature convergence.
  Likely better implementations: down-weighted selection probability,
  or filter only on current-gen downvotes, or leave preference
  filtering to the CNN scorer and only filter on `archived = 1`.
  No code change yet — design discussion item.

- Decay weighting for older pairwise (low priority — pairwise pairs
  are durable, this just lets the model down-weight ancient ones).

- Sub-generation "mini-generations" (retrain-without-breed cycles).

## State the user might surface

- CNN scorer for the wallpaper: gen 1, v3 with sentinel-aware
  standardization, deployed yesterday. Path:
  `flame_sheep/data/cnn_scorer_personal_vk.npz`. Backup of previous
  model still in flame_sheep/data/ as
  `cnn_scorer_personal_vk.npy.backup_20260518_210941`.

- Beat-RNN: training script is `tools/train_beat_rnn.py`. Output
  default is `flame_sheep/data/beat_rnn.npy` but the user typically
  uses `-o ~/datasets/beat-labels/beat_rnn_v1.npz`.

- DB at `~/.local/share/flame-sheep/library.db` has the live data.
  Generational migration has applied (verified via direct query).

## If user asks "where were we"

Answer: the generational data architecture's "schema + heuristic +
weighting" chunk shipped. Remaining big chunk is the breeding pipeline
that actually advances the generation counter and produces the next
population. User wants to bootstrap gen 1 candidates by running
`evolve` in a loop ~20 times; no code change required for that.

The other thing on user's mind: the "evolve ignores downvoted"
question — they explicitly said they want to think through implications
before deciding.
