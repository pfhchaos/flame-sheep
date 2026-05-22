# Beat-RNN Iteration Strategy

Forward-looking notes on how future retrains should be sourced, sequenced,
and evaluated. Companion to `beat_rnn_deploy_plan.md` (which is about
shipping v1) — this is about everything *after* v1 ships.

## The data-scale advantage

For most ML problems, the bottleneck is labels — humans annotating images,
transcribing audio, rating preferences. For beat detection, **BeatNet is
the labeler**: any audio file run through `tools/generate_beat_labels.py`
produces training-ready soft labels with no human in the loop.

This is rare in ML. It means scaling the training corpus is "find more
audio + let it run overnight," not "design an annotation protocol + hire
annotators + build a labeling UI + wait six months." The existing fast
pipeline (`pack_beat_labels.py`, mmap corpus, multi-thread prep) takes
the rest from there.

The implication: when planning the next training round, **more data
usually beats more parameters at this model size**. Classic ML
observation, particularly pronounced when the marginal cost of new data
is "remember to run a script."

## Iteration priorities (in order)

1. **Broaden the corpus first.** Current training corpus is classical +
   Michael Jackson. The v1 eval showed transfer to anime/game/electronic/
   math rock (F1 = 0.553 vs 0.394 for current_system on disjoint domains).
   That suggests the model has signal that generalizes; growing the corpus
   should push that further. Lower priority on parameter count until the
   data side has been pushed.

2. **Sweep hyperparams once data is broader.** Hidden size, projection
   size, second GRU layer, learning rate. At ~80 min/epoch (post-mmap fix),
   a 4-6-variant sweep of ~10 epochs each is 1-2 days. Look for variants
   where the F1 curve is still climbing at epoch 10 — those have headroom
   the current architecture didn't. Current model hit asymptote at ~0.56;
   bigger probably still gains.

3. **Architecture deviations only after (1) and (2).** Attention layers,
   transformer variants, deeper RNNs. These have more knobs and harder
   debugging. Worth doing only when the simpler-model + more-data
   trajectory has stopped paying.

## Sourcing new corpora

### Pre-filter: check BeatNet's confidence

BeatNet is the upper bound. If BeatNet's labels are wrong on a new
corpus, training on those labels actively *hurts* the student model.

Cheap quality check before dumping a corpus into the training pool:
- Per corpus, compute the mean of `max(soft_label_per_frame)` across all
  frames in a sample of files
- High mean (>0.7) → BeatNet is confident about most frames → corpus is
  trustable
- Low mean (<0.5) → BeatNet is uncertain → likely a genre BeatNet handles
  poorly → skip or downsample

Genres BeatNet is known to handle poorly: extreme tempo rubato (some
classical, jazz ballad), very sparse percussion, heavy syncopation,
non-Western rhythmic conventions. Worth eyeballing a few labels on
samples from these categories before committing the corpus.

### Public corpora worth considering

- **FMA-large** (~106k tracks, 161 genres, CC-licensed, ~93 GB).
  Best general-purpose option. Strong genre coverage.
- **MTG-Jamendo** (~55k tracks, mood/genre tagged, CC-licensed, ~46 GB).
  Strong on mood diversity, useful for "does the model generalize across
  emotional registers."
- **MUSDB18** (~150 tracks, stems-separated, ~30 GB). Smaller but has
  separated drum stems — useful for training a percussion-focused variant
  or for sanity-checking beats against pure percussion.
- **GTZAN** (1k tracks, 10 genres, well-known benchmark). Too small to
  meaningfully expand training but useful as a *fixed* eval slice that
  doesn't drift.

### Storage considerations

Per-file packed `.npy` is ~19 MB. Scaling implications:
- 7,500 files (current) = ~140 GB — fits NVMe
- 50,000 files = ~1 TB — beyond NVMe; goes on `/mnt/btrfs-hdd`
- 100,000+ files = ~2 TB — definitely HDD

mmap-from-HDD works (kernel handles the paging) but adds cold-cache
latency on the first epoch. Steady-state still fast because the corpus
mostly lives in page cache after warmup. For large-corpus training, the
bcache option discussed yesterday (SSD-cached HDD) gets practically
relevant — first-epoch warmup time is paid once, then training is
roughly NVMe-speed.

## The self-distillation loop

If beat-RNN ever beats BeatNet on osu (or any independent ground truth),
**it becomes the labeler for the next generation**:
- v1: trained against BeatNet labels (osu F1 = 0.553)
- v2: trained against BeatNet on a broader corpus (osu F1 = ?)
- If v2 > BeatNet on osu → v3 trains against v2's labels, not BeatNet's
- v4 onward continues the chain

Self-distillation works particularly well in this case because:
- Each generation only needs to beat its teacher, not beat the absolute
  ground truth (osu)
- The eval against osu is independent of the teacher → no circularity in
  measuring whether the student improved
- Each round can broaden the corpus AND switch teachers — two
  improvement axes at once

Most projects can't run this loop because regenerating labels at scale
costs more than human annotation in the first place. Here it costs
"run the trained model on the audio."

## Eval discipline across retrains

When multiple training rounds are coming, **the eval corpus must be
fixed** to make iteration comparable. Without a stable benchmark, "v2
got 0.6" becomes ambiguous: better model or easier eval?

Pre-commit to a stable eval corpus *now*:
- 30 osu tracks (whatever was used for v1) is the obvious anchor.
  Document the exact track list in the eval-results JSON so it's
  reproducible.
- Add 5-10 tracks from a different genre cluster than the training
  data, also fixed. Stress-tests cross-corpus generalization.
- The canonical examples (`docs/canonical_examples.md` for genome
  scoring) have a beat-detection analog: pick 3-5 individual tracks
  where the failure modes are particularly informative (low-tempo,
  syncopated, gradual tempo change, etc.). Per-track F1 on these
  becomes the qualitative diagnostic alongside aggregate F1.

Each retrain reports against the same anchor. Growing the eval corpus
alongside the training corpus is tempting but defeats the purpose —
the benchmark needs to stay still.

## When to retrain

Triggers for the next training round (any one is sufficient):
- **Daemon's CQT differs meaningfully from training CQT** —
  *empirically confirmed for v1*: prtcqt produces magnitudes ~77×
  smaller than librosa.cqt for the same PCM. v1 ships with a scale-
  correction band-aid in `BeatRNNDetector` (`_CQT_SCALE_TO_TRAINING
  = 770.0`); next retrain MUST use daemon CQT directly to eliminate
  the train/serve skew (see "Required: switch to daemon CQT" below).
- New genre cluster identified where v1 underperforms (gap fill)
- More than 2-3x corpus expansion since last train (diminishing returns
  on existing model size; time to scale)
- A model improvement (architecture or hyperparams) wants validation
  against the existing data — even without new audio, worth retraining
  to confirm

### Required: switch to daemon CQT for the next retrain

The v1 training pipeline (`tools/generate_beat_labels.py`) computes CQT
via `librosa.cqt`. The live daemon uses `prtcqt` (rt-cqt SlidingCqt).
Per `tools/diagnose_cqt_skew.py`, these produce ~77× different magnitude
scales for identical PCM input; per-bin correlation median is 0.59,
not 1.0, so the mismatch isn't a clean scale factor either.

For the next retrain:
1. **Generate labels through the daemon's CQT, not librosa**. Modify
   `tools/generate_beat_labels.py` to use `CqtEngine` directly
   (instantiate it and feed audio hop-by-hop, mirroring the live
   path).
2. **Train on those daemon-CQT labels**. Same model architecture; the
   features change.
3. **Delete `_CQT_SCALE_TO_TRAINING` from `BeatRNNDetector`** —
   training and deployment now use the same CQT, no scale correction
   needed.

This is the structural "train/serve skew" fix vs the v1 band-aid.
Pattern recognition: the broader lesson (data representation is
always where the bugs are) is also documented in
`memory/feedback_*.md` if/when filed; this is the concrete
instance.

Avoid retraining triggers:
- "It's been a while" — calendar time isn't a quality signal
- "Someone has new ideas" — ideas without expected-improvement
  justification cost compute without payoff

## Reporting discipline

For every retrained model, log to `docs/model_benchmarks.md` (existing
table format):
- Final model F1 on the stable eval corpus
- F1 on each per-track canonical example
- Train corpus size + composition
- Notable hyperparam choices
- Optimistic vs realistic (causal-peak-picker) numbers, both reported

Don't report just the headline ceiling number — report the realistic
deployed equivalent alongside. The lesson from v1: the optimistic eval
result was misleading by ~5-10pp without the realistic correction.
Calling out the gap explicitly prevents the "deployed model performs
worse than benchmark" failure mode each new iteration.

## Effort estimates

For planning purposes:
- New corpus integration (sourcing + BeatNet labeling + packing):
  1-3 days depending on corpus size
- Single retraining run (30 epochs, current size): ~40 hours wall time
- Full hyperparameter sweep (4-6 variants × 10 epochs): 1-2 days
- Eval against stable corpus: ~30 min per model

So a typical "broaden + retrain + evaluate" cycle is a long weekend of
mostly-background compute, with a few hours of active work. Multiple
cycles per month is feasible without it dominating dev time.
