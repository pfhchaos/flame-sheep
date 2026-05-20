# Self-handoff for the freshly-compacted me — 2026-05-20

You got compacted. Here's what's on disk that wasn't there when your
previous transcript ended.

## TL;DR

Today's session was three things:
1. **Generational schema finished**: retag/breed recovery, thumbs-up
   breeder live, loop-vote propagation removed (it was poisoning gen-0
   thumbs), pruner periodic retrain logging.
2. **Beat-RNN 3-class training diagnosed as broken** — class weights +
   focal loss couldn't fix it. Downbeat recall climbed 2.9% → 5.3%,
   beat stuck at 0.5%. Asymmetric improvement = representation
   problem, not loss-weighting.
3. **Continuous-activation reformulation shipped to disk** — single
   output channel (beat_probability), sigmoid + BCE, peak-pick +
   F1@70ms eval. Trainer ready, waiting on label augmentation to
   finish.

## What's running RIGHT NOW (or was, when this was written)

- **PID 114748**: `tools/add_beat_score_to_labels.py` — adds a
  `beat_score` array to each of 7519 .npz label files
  (`= 1 - labels[:, 2]`, the BeatNet "anything but non-beat" prob).
  ~25 min total. Log at `/tmp/add_beat_score.log`.
- **No training is running.** The 3-class training was killed at
  12:07. New training (continuous) shouldn't start until label
  augmentation completes.

## After augmentation completes

Smoke test first:
```bash
python tools/train_beat_rnn_continuous.py /home/chaos/datasets/beat-labels/ \
    -o /tmp/beat_continuous_smoke.npz \
    --epochs 1 --hidden 48 --chunk-len 64 --batch-size 8 --lr 0.003 \
    --max-files 8 --val-interval 0 --no-resume
```

Expected: F1 > 0 (vs the 3-class run's beat recall stuck at 0.005).
If F1 lands ~0.3-0.5 on the smoke that's success — full corpus
will go higher.

Then full run:
```bash
nohup python -u tools/train_beat_rnn_continuous.py /home/chaos/datasets/beat-labels/ \
    -o /home/chaos/datasets/beat-labels/beat_rnn_continuous.npz \
    --epochs 30 --batch-size 64 --chunk-len 256 --hidden 48 --lr 0.003 \
    --no-resume \
    > /home/chaos/datasets/beat-labels/train_continuous.log 2>&1 &
```

(Use `> log` not `>>` was an issue earlier today — user was tail-f'ing
the old log when I clobbered it. Mention this.)

## Today's commits, most recent first

```
95d650b  beat_rnn: continuous-activation reformulation (single channel + BCE)
8002638  breed_existing_gen1_thumbs: one-off retag + retroactive breed
b8e7e8e  beat_rnn: class-weighted + focal cross-entropy (failed approach but committed)
88b30a4  evolve_loops: coverage-driven offspring scaling + fresh-blood default 0.5
de1009b  Library.rate: stop propagating loop votes to constituent genomes
6c65974  train_beat_rnn: per-class metrics, confusion matrix, majority baseline
5d8d224  docs/generational: mark thumbs-up breeder as shipped
34a0a8c  thumbs-up breeder: 7 jittered children per direct genome upvote
51eff79  docs: update for today's beat-RNN + generational work
a7267ef  pruner_worker: periodic retrain recommendation log
d6f7530  generational: retrain heuristic + pair-weighting for mixed-mode training
d159bfc  generational schema: tag ratings with generation, intra-gen synthesis
```

## Beat-RNN architecture (continuous, NEW)

- 1 output channel (logit) instead of 3 (softmax classes)
- `flame_sheep/shaders/rnn/bce_loss.comp` instead of cross_entropy.comp
- `flame_sheep.wallpaper_ml.bce_loss_dispatch()` instead of
  cross_entropy_dispatch
- `tools/train_beat_rnn_continuous.py` is the trainer (separate file,
  imports load_dataset/make_chunks/LazyFileCache/save+load_checkpoint
  from train_beat_rnn.py)
- Peak-pick + F1@70ms validation (mir_eval convention)
- Target: `beat_score = 1 - labels[:, 2]` — continuous beat probability
- `LazyFileCache.get()` returns a dict now (refactored to allow
  beat_score to ride alongside the existing keys without breaking the
  3-class code path)

## Continuous vs 3-class: why we pivoted

The 3-class softmax model collapsed to "always predict majority
(non-beat)" and class weights + focal loss only got downbeat moving,
not beat. Asymmetric pattern in three mid-epoch checkpoints:

| | downbeat recall | beat recall | non-beat recall |
|---|---|---|---|
| t=574 | 2.9% | 0.5% | 99.1% |
| t=1157 | 4.7% | 0.5% | 99.1% |
| t=1737 | 5.3% | 0.4% | 99.2% |

The asymmetry IS the diagnosis: beats are intermediate-strength
percussive events that aren't acoustically separable from many
non-beat percussive hits (ghost notes, hi-hat patterns, fills).
Multi-class softmax forced a fine line the input features can't
draw cleanly. Continuous beat-probability collapses this:
"how beat-like is this moment" is a smoother target.

## Generational architecture state

All shipped except the actual breed+train+advance_generation loop:
- `current_generation = 1` in metadata
- 732 ratings + 1954 pairwise at gen 0 (legacy, pre-generational)
- Today's 14 ratings retagged from gen 0 → 1 via
  `tools/breed_existing_gen1_thumbs.py`
- 42 new genomes bred from 6 thumbs-up parents
- Pruner_worker logs retrain recommendation every 5 min
- Library.rate triggers immediate 7-child breeding on direct genome
  upvotes (loop-prop removed)
- Pair weighting via K*M/(K+M) effective-sample-size formula

Deferred: breed+train cycle that actually calls advance_generation.

## Lessons explicitly worth flagging

1. **Aggregate accuracy on imbalanced classes is lying to you.**
   We added per-class metrics, then immediately caught the
   "predicting majority on everything" pathology. Will be the first
   metric for any future multi-class training.
2. **Asymmetric per-class improvement diagnoses the failure mode.**
   When downbeat moves but beat doesn't despite proportional class
   weights, the input representation is the bottleneck, not the loss.
3. **`> log` truncates the old log.** Append (`>>`) or timestamp
   filenames for long-running training that the user is tail-f'ing.
4. **The user's alter-ego workflow is real.** "His argument is right"
   / "thoughts?" exchanges between me, alter-ego, and user surface
   correct diagnoses through triangulation. Trust the alter-ego
   when its reasoning is independent and points the same way.

## Critical files reference

- `docs/generational_architecture.md` — generational design + status
- `docs/model_benchmarks.md` — beat-RNN throughput + the 7
  optimizations applied yesterday + (today) the per-class metrics
  diagnostic that broke the 3-class run
- `docs/session_handoff_2026-05-19.md` — yesterday's handoff
- `tools/train_beat_rnn_continuous.py` — the new trainer
- `tools/train_beat_rnn.py` — the old 3-class trainer (still works,
  just produces majority-collapse models on imbalanced data)
- `tools/breed_existing_gen1_thumbs.py` — one-off recovery, already
  ran today
- `tools/eval_beat_rnn_cpu.py` — CPU eval of saved checkpoints
  (used to confirm the 3-class collapse)
- `tools/check_retrain.py` — CLI for the retrain heuristic
- `flame_sheep/shaders/rnn/{cross_entropy,bce_loss,gru_seq_forward,gru_seq_backward}.comp`

## If user says "where were we"

Beat-RNN reformulation is on disk and committed. Label augmentation
running in background (~25 min, started ~12:35 PDT). Once it
finishes, smoke-test the continuous trainer on a small sample, then
kick off the full 30-epoch run. The smoke is critical because the
1-channel + BCE pipeline hasn't been end-to-end tested against real
data yet — only the shader compiled and the unit-test math.

Generational architecture is mostly shipped. Breed+train cycle that
advances generations is the remaining real chunk of work but it's
not blocking anything right now.
