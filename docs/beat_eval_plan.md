# Beat Detection Eval Harness

A unified evaluation harness comparing multiple beat detection systems against osu-derived ground truth. Produces F1 numbers that let us answer "is the new beat-RNN actually better than the current system, and how much of BeatNet's quality did distillation capture?"

## Purpose

Three load-bearing questions this eval needs to answer:

1. **What's the current system's actual quality?** The May 1 baseline of F1=0.544 is stale (code has changed). Need a fresh number against independent ground truth.
2. **What's BeatNet's quality against independent ground truth?** Distilling from a teacher is only useful if the teacher itself is correct. BeatNet's F1 against osu is the ceiling for any distilled student.
3. **What's the beat-RNN's quality?** And specifically: how much of the BeatNet → current gap does it close?

The answer to (3) determines whether the RNN ships. The answers to (1) and (2) let us interpret (3) — closing 75% of the gap is great; closing 5% means the distillation lost most of the signal.

## Ground truth

**Source**: osu beatmap files at `~/.cache/flame-sheep/corpus/osu/`. 226 `.osu` files; many are different difficulties of the same audio, so deduplicate by `AudioFilename`.

**Beat derivation**: parse `[TimingPoints]` section. Each timing point has format `offset_ms, beatLength, meter, sample_set, sample_idx, volume, uninherited, kiai`. Use **uninherited points only** (column 7 = `1`, equivalently `beatLength > 0`). Inherited points (`beatLength < 0`) are slider velocity multipliers, not tempo changes.

For each uninherited timing point at `offset` with `beatLength`:
- Beat positions = `offset, offset + beatLength, offset + 2*beatLength, ...`
- Continue until the next uninherited timing point or end-of-audio

This gives the music's underlying beat grid as a list of beat timestamps in seconds. That's the ground truth.

**Downbeat derivation** (optional, for downbeat F1): every Nth beat where N is the meter (column 3). So in 4/4 time, every 4th beat is a downbeat. This is the analytically-derived downbeat position; mappers may interpret it differently in placement, but the math says where the measure boundary is.

**Edge cases to handle**:
- Multiple timing points per song (tempo changes mid-song) — common in osu maps
- Non-4/4 meters (5/4, 7/8) — possible but rare; eval can either include or exclude with a flag
- Audio offset (the `AudioLeadIn` field in `[General]`) — usually 0, but worth honoring if present

## Systems to evaluate

Detector interface (the minimum contract):

```python
class BeatDetector:
    name: str

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Returns beat positions in seconds.

        audio: mono float32 in [-1, 1]
        sr: sample rate
        Returns: 1D array of beat times in seconds, sorted ascending.
        """
```

Initial three implementations:

1. **`BeatNetDetector`** — wraps the existing BeatNet inference path. The teacher; tells us the distillation ceiling.
2. **`CurrentSystemDetector`** — wraps `flame_sheep_audio.beat_detector` (the phase-coherence system shipped in the live wallpaper). This is the baseline to beat.
3. **`BeatRNNDetector`** — loads `beat_rnn_v1.npz`, runs forward pass, applies peak picking on the continuous activation output. New system under test.

The interface is intentionally minimal — beats out, in seconds. Anything else (downbeats, tempo, meter) is optional and can be added when needed via additional methods.

## Metrics

F1 alone is insufficient for diagnosis — different failure modes (tempo-tracking drift, spurious firing, octave errors) can produce identical F1 numbers. Report multiple metrics side-by-side.

**Beat F1** (MIREX-style):
- A predicted beat matches a ground truth beat if it falls within ±`tolerance_ms` of any ground truth beat
- Each ground truth beat can match at most one prediction (greedy one-to-one matching)
- Precision = matched / total predicted; Recall = matched / total ground truth; F1 = harmonic mean
- Report at three tolerances: ±25ms (strict), ±50ms (standard), ±70ms (loose)

**Cemgil score** (Gaussian-weighted accuracy):
- Each predicted beat contributes a Gaussian-weighted score based on its distance to the nearest ground truth
- Captures "how close" matches are, not just "matched or not." A detector consistently 30ms early scores higher than one with random scatter
- Available as `madmom.evaluation.beats.cemgil()`

**Continuity metrics** (CMLc, CMLt, AMLc, AMLt):
- CMLc: longest correct contiguous segment / total length (does the detector find the grid and stay on it?)
- CMLt: total correct contiguous segments / total length
- AML variants: same but allow octave/half-tempo errors as correct
- Distinguishes "finds beats sometimes but doesn't lock" from "locks to wrong tempo"
- Available via `madmom.evaluation.beats.BeatEvaluation`

**Why each one**:
- F1 says "matched or not"
- Cemgil says "how close were the matches"  
- Continuity says "did the detector stay locked once it found the grid"

Two detectors at identical F1 can fail very differently on the other axes. Continuity in particular separates "tempo-tracking failure" from "spurious-firing failure," which is the most common diagnostic ambiguity.

**Per-track and aggregate reporting**:
- Per-track: all metrics for each track at each tolerance
- Aggregate: mean across the corpus (equal weight per track), plus also report on a `single_tempo=True` subset (tracks with exactly one uninherited timing point). This subset isolates "does the detector work at all" from "does it handle tempo changes" — the cleanest apples-to-apples comparison between systems.

The `madmom.evaluation.beats` module has all four families. Use it rather than reimplementing.

## Eval corpus

Start with **30 tracks** spanning genre and tempo. Diminishing returns past that for stable F1.

**Selection strategy** (pseudo):
1. Find unique audio files (dedupe across difficulties)
2. For each unique track, pick one .osu file (any difficulty — they all reference the same beat grid)
3. Stratify by: tempo bucket (slow / medium / fast), single-tempo vs multi-tempo (tempo changes mid-song), 4/4 vs other meters
4. Random sample within strata

**First-pass filter**: exclude tracks where:
- `[TimingPoints]` has only inherited points (no uninherited — broken map)
- Audio file is missing or unreadable
- Track is shorter than 30 seconds (not enough beats for stable F1)

Save the selected track list to a manifest file so eval is reproducible.

## Implementation order

Minimum viable first, expand once it works. Each step produces something runnable.

### Step 0: corpus sanity check

10-line script to scan all `.osu` files in the corpus and report:
- Total file count, unique audio file count (dedupe by `AudioFilename`)
- Distribution of `AudioLeadIn` values — if 95%+ are 0, the parser can ignore lead-in; otherwise honor it
- Histogram of meters (probably mostly 4/4, but worth verifying)
- Count of tracks with multiple uninherited timing points (tempo changes)
- Any unreadable files

Cheap to write, prevents designing around assumptions that don't hold.

### Step 1: osu parser + beat derivation

`flame_sheep/eval/osu_parser.py`:
- Parse `.osu` file → extract `[General]`, `[TimingPoints]`
- Derive beat timestamps from uninherited timing points
- Return `(audio_filename, beat_times_sec, downbeat_times_sec, meter, lead_in_ms)`

Verify by hand: pick a known map (e.g., MEGALOVANIA Easy at 240 BPM), confirm beats are 0.25s apart starting from `(offset_ms - lead_in_ms) / 1000`.

### Step 2: detector interface + first implementation

`flame_sheep/eval/detectors.py`:
- Abstract `BeatDetector` base
- `CurrentSystemDetector` (easiest — already in codebase, just wrap)
- Test by running on a single track, eyeballing the output

### Step 3: F1 computation + per-track eval script

`tools/eval_beat_detection.py`:
- CLI: `--corpus DIR --detector NAME --tolerance-ms 50 --tracks N`
- Load corpus, run detector on each, compute F1, print per-track + aggregate
- Use `madmom.evaluation.beats.BeatEvaluation` for the F1 math

At this point you can answer: "what's the current system's F1 against osu?"

### Step 4: add BeatNetDetector

Wrap whatever BeatNet inference path already exists in `tools/generate_beat_labels.py`. Run the same eval. Now you have **two** F1 numbers and the gap between them.

### Step 5: add BeatRNNDetector

Load the latest beat-RNN checkpoint, run forward pass per track, peak-pick the activation, return beat times. The forward pass logic exists in `tools/eval_beat_rnn_cpu.py`; the peak-picking logic exists in `flame_sheep_audio.beat_detector` (probably — verify).

At this point all three numbers exist. Compare.

### Step 6 (optional, deferred): downbeat F1, tempo accuracy

Extend the detector interface with `detect_downbeats` and `estimate_tempo`. Eval against the analytically-derived downbeats and the BPM from the timing points. Useful follow-on data but not needed for the initial "did the RNN work" question.

### Caching

Each detector's output for each track should be cached so re-running eval doesn't redo inference. Cache key: `(track_id, detector_name, detector_version)`.

**`detector_version` must be concrete and content-derived**:
- `CurrentSystemDetector`: git commit hash of the relevant module (`flame_sheep_audio/src/flame_sheep_audio/beat_detector.py`), or a hash of the source file contents
- `BeatNetDetector`: BeatNet's package version + model checkpoint hash if non-default
- `BeatRNNDetector`: first 16 chars of `sha256(weights.npz)`. **Critical**: this means a fresh training checkpoint automatically invalidates its cache entries without manual intervention. Without this, you'll re-eval the *same* old model after retraining and not realize it.

Cache location: `~/datasets/beat-eval-cache/<detector_name>/<detector_version>/<track_id>.npy`. The version segment in the path makes it human-inspectable which detector run produced what.

## Open questions

- **Lead-in handling**: `AudioLeadIn` in osu is typically 0 but can be non-zero. Should be subtracted from beat timestamps if non-zero. Verify by inspecting a few files first; might not need handling for the dedupe'd corpus.
- **Audio loading**: each detector wants its own preprocessing (sample rate, mono, etc.). The harness should provide raw audio + sample rate; each detector handles its own resampling. Less efficient (multiple resamples per track), but cleaner separation.
- **Peak picking for BeatRNN**: if the trained model outputs continuous activation, peak-picking needs threshold + minimum-distance parameters. These are hyperparameters that should be tuned on a held-out subset of the eval corpus, NOT on the same tracks the F1 is computed on. Pick first; tune later if F1 is suspiciously low.
- **Stratification**: the "spans genres / tempos" goal is hard to verify without knowing what's in the corpus. Worth a quick `tempo histogram` pass before selecting the 30 tracks to make sure the spread is real.
- **Tempo-change tracks**: include or exclude? Tempo changes are where weaker detectors fail hardest, so including them is informative but also makes F1 numbers harder to interpret. Probably include in main eval, also report F1 on the single-tempo subset for cleaner comparison.

## Success criteria

The eval is doing its job if, after Step 5, we can fill in this table:

| System | F1@50ms (mean) | F1@25ms | F1@70ms | Cemgil | CMLt | Notes |
|--------|---------------|---------|---------|--------|------|-------|
| Current (phase coherence) | ? | ? | ? | ? | ? | baseline |
| BeatNet | ? | ? | ? | ? | ? | teacher / ceiling |
| Beat-RNN v1 | ? | ? | ? | ? | ? | distilled student |

Plus the same table on the `single_tempo=True` subset (cleaner apples-to-apples).

**Important interpretive caveat**: BeatNet's F1 against osu will not be 1.0. BeatNet was trained on professional MIREX-style annotations; osu maps were placed by community mappers listening to the same audio with their own subjective interpretations of where the beat is (backbeat emphasis, syncopation, ghost-note handling all differ between annotators). So expect BeatNet at maybe 0.75-0.85 against osu, not 0.95+. The framing that matters is **the gap between BeatNet and current-system, not the absolute distance of BeatNet from 1.0**.

The decision input is: "the RNN closed X% of the BeatNet → current gap" — for example, if current is 0.55 and BeatNet is 0.78, a 0.70 RNN closes 65% of the gap. That's the ship/no-ship signal. Closing >50% of the gap is probably ship; <30% probably iterate; in between is a judgment call.

## Two-corpora cross-check

The RNN training already produces an F1 number on its own validation split (data from the same source distribution it trained on, but held-out songs). The osu eval gives an independent number on a different corpus type entirely.

These two numbers together diagnose what the RNN learned:

- **High on both**: RNN learned real beat detection. The osu corpus and the training corpus both agree it works.
- **High on val, low on osu**: RNN learned to mimic BeatNet's idiosyncrasies — including ones the osu corpus disagrees with. Distillation success, generalization failure.
- **Low on val, low on osu**: RNN didn't learn beats. Probably the focal+weights still wasn't enough, or the architecture/formulation needs more work.
- **Low on val, high on osu**: would be very surprising. Indicates the val set is somehow worse than osu, or there's a bug in val computation.

The osu eval is therefore not just "another F1 number" — it's the *generalization probe* that the within-training-distribution val number can't be.

## Non-goals (deferred)

- Real-time / streaming evaluation — this is offline batch eval
- Latency profiling — separate concern from accuracy
- Per-genre stratified analysis — useful but secondary to "does the system work at all"
- Tempo tracking eval — possible Step 6 extension, not required for initial decision
- Auto-running on every CI commit — manual invocation is fine for now; turn into regression test once stable
