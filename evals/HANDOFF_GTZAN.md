# Adding GTZAN-rhythm as second eval corpus — handoff for implementation

Decision context: osu downbeat annotations are mapper-derived and not a
reliable musicological ground truth. Beat annotations are mostly right
(period from TimingPoint math) but absolute phase is mapper-dependent.
We need a second, hand-labeled corpus to:

1. Get trustworthy downbeat F1 numbers (osu cannot give us this).
2. Calibrate how much our existing beat F1 numbers are limited by osu
   annotation quality vs detector quality.
3. Establish a cross-corpus consistency check so single-corpus overfit
   doesn't quietly sneak in.

## What's on disk (already done as of 2026-06-03)

```
~/datasets/GTZAN/
├── genres/                          # 1000 audio files, 22050 Hz mono 16-bit WAV
│   ├── blues/    blues.00000.wav ... blues.00099.wav      (100)
│   ├── classical/                                          (100)
│   ├── country/                                            (100)
│   ├── disco/                                              (100)
│   ├── hiphop/                                             (100)
│   ├── jazz/     jazz.00000.wav ... jazz.00099.wav         (100; .00054 is the corrupt one)
│   ├── metal/                                              (100)
│   ├── pop/                                                (100)
│   ├── reggae/                                             (100)
│   └── rock/                                               (100)
├── annotations/                     # cloned from upstream, .git retained
│   ├── beats/    blues.00000.beats ... rock.00099.beats    (999; no jazz.00054)
│   ├── tempo/    blues.00000.bpm   ... rock.00099.bpm      (999; no jazz.00054)
│   └── README.md
└── genres.tar.gz                    # 1.23 GB; can delete, kept for re-extract
```

- **Annotation source**: https://github.com/TempoBeatDownbeat/gtzan_tempo_beat
  cloned with .git intact so upstream commit is traceable
- **Audio source**: https://huggingface.co/datasets/marsyas/gtzan
  (`data/genres.tar.gz`, SHA256 b28cc067...)
- **Citation**: Marchand, Fresnel, Peeters 2015 ISMIR
- **Beats format**: `<time_sec>\t<beat_position_in_bar>` — tab-separated;
  `beat_position == 1` rows are the downbeats
- **Tempo format**: single float on one line, BPM

**Naming was already normalized**: upstream annotation files were named
`gtzan_<genre>_<id>.{beats,bpm}` (underscores). Renamed at extract time
to `<genre>.<id>.{beats,bpm}` to match audio canonical naming
`<genre>.<id>.wav`. Loader joins directly on `<genre>.<id>` stem; no
translation table needed.

**Join: 999 tracks have all three of (wav, beats, bpm).** jazz.00054
has audio but no annotations — loader should silently skip it (same
behavior as mirdata + most published GTZAN evals).

## Implementation tasks

### 1. Corpus loader

Mirror the osu pattern from `flame_sheep/eval/osu_parser.py`. New module
`flame_sheep/eval/gtzan_parser.py` with:

```python
class GtzanTrack:
    audio_path: Path
    beat_times: np.ndarray          # all beats in seconds
    downbeat_times: np.ndarray      # subset where bar_pos == 1
    tempo_bpm: float
```

A single parse function reads both annotation files and returns the
struct. No more cleverness than that.

### 2. Eval runner — extend, don't fork

`tools/eval_beat_detection.py` currently hardcodes osu via
`--corpus` defaulting to the osu cache and `_collect_tracks` walking
osu beatmap dirs. Refactor:

- Add `--corpus-kind {osu, gtzan}` argument (default osu).
- Dispatch to per-corpus track collector that returns a uniform
  `(track_id, audio_path, ground_truth_struct)` triple.
- Emit predictions for downbeats too when ground truth has them
  (gtzan does, osu does but with the quality caveat above).
  Detectors that don't produce downbeats (percentile) emit `None`
  rather than fabricating a downbeat stream; the eval skips downbeat
  scoring for that detector.
- Report F1 separately for beats and downbeats.

The output format should make it grep-able for the same
`_parse_beat_report` regex in `evals/builtin.py`, just with an
additional `downbeat.f1@70ms` row.

### 3. Register the new evals

In `evals/builtin.py`, register per-corpus per-detector evals:

```
beat.gtzan.percentile        ← deployed detector; gate-relevant
beat.gtzan.madmom            ← oracle, already wired for osu
beat.gtzan.beatnet           ← oracle, already wired for osu
downbeat.gtzan.madmom        ← oracle; needs downbeat output extraction
downbeat.gtzan.beatnet       ← oracle; needs downbeat output extraction
```

Note: percentile does not emit downbeats, so `downbeat.gtzan.percentile`
is intentionally omitted (the runner returns `None` from §2).

The existing `beat.osu.*` registrations stay. No `downbeat.osu.*`
registrations currently exist; if the downbeat side of this work
extends to osu, those need to be added too — but per the decision
context above, osu downbeat ground truth is not trustworthy so
that's lower priority.

### 4. Cross-corpus consistency meta-eval

This is the part that's NEW infra (not in current registry).

Design: a meta-eval reads the latest scorecard rows for the SAME
detector across multiple corpora and computes:

```python
{
    'beat.consistency.percentile.mean':   mean(per_corpus_f1s),
    'beat.consistency.percentile.min':    min(per_corpus_f1s),
    'beat.consistency.percentile.spread': max - min,
    'beat.consistency.percentile.n_corpora': len(per_corpus_f1s),
}
```

Implementation: in `evals/registry.py`, add a `meta_eval` decorator
that doesn't run a subprocess but reads from `evals/results.jsonl`
via `storage.read_runs()` (already returns typed `RunEntry` list).
This lets `tools/eval_all.py` produce consistency metrics without
re-running the underlying corpus evals.

Read semantics: "latest row containing the metric key" — runs are
append-only and the same eval may appear in multiple rows across
commits; the meta-eval consumes the most recent value for each
`<beat|downbeat>.<corpus>.<detector>.mean.f1@70ms` key.

Why this matters: a detector that wins on osu but tanks on gtzan is
overfitting to osu annotation conventions (mapper-derived
quirks). The mean hides this. The spread surfaces it.

### 5. Deploy gate update

`tools/check_beat_gate.py` currently gates on a single beat F1 number.
Update it to:

- Gate the **deployed detector only** (currently `percentile`). Oracle
  baselines (`madmom`, `beatnet`) are not deployed and not gate-relevant
  — they exist as ceilings to interpret percentile's score against.
- Require BOTH per-corpus beat F1 >= baseline per-corpus beat F1 for
  the deployed detector (not mean). A detector change that improves
  osu by 0.05 but regresses gtzan by 0.10 should fail the gate.
- The gate activates **after** §6 lands the gtzan baseline rows for
  percentile. Until those rows exist, the gate falls back to current
  single-corpus osu behavior. This avoids a chicken-and-egg failure
  on the first commit that introduces gtzan.
- For downbeat detectors when shipping a downbeat consumer, gate on
  downbeat F1 the same way (not relevant today; percentile is beats-only).

### 6. Re-run the existing baselines on gtzan

After the loader + runner work, append scorecard rows for:
- `beat.gtzan.percentile`  ← deployed; becomes the gate baseline
- `beat.gtzan.madmom`      ← oracle ceiling
- `beat.gtzan.beatnet`     ← oracle ceiling
- `downbeat.gtzan.madmom`  ← oracle; needs downbeat extraction (§3)
- `downbeat.gtzan.beatnet` ← oracle; needs downbeat extraction (§3)

These become the published baselines we judge new detectors against.
The percentile row is what §5's gate reads; the oracle rows exist for
interpretation (how much headroom is left) but are not gate-relevant.

### 7. Every accuracy eval also dumps latency

Same run, same source of truth: every detector eval produces both an
accuracy table AND a latency table. Eliminates the "tested accuracy
but didn't think about speed" failure mode and gives Q3 (why is madmom
slow?) a real measurement substrate instead of one-track guesses.

**Per-track measurements** (already partially captured — `detect_sec`
exists, just needs aggregation and per-component breakdown):

- `detect_sec` — total wall-clock detection time
- `realtime_factor` = `detect_sec / duration_sec` (sub-1.0 = streamable)
- `timing_breakdown` — per-component (e.g. madmom `{'rnn': ..., 'dbn': ...}`,
  percentile `{'csd_transform': ..., 'peak_pick': ...}`); summing
  equals `detect_sec` modulo overhead

Detectors expose `last_timing` (or similar) after `detect()`; the eval
runner reads it. Detectors that can't decompose (BeatNet wraps a black
box) emit `{'total': ...}` only — explicit, not faked.

**Per-eval aggregations** added to scorecard row alongside F1 metrics:

```
latency.total.mean_sec
latency.total.p50_sec
latency.total.p95_sec
latency.total.p99_sec
latency.realtime_factor.mean
latency.realtime_factor.p95     ← worst-case realtime headroom
latency.<component>.mean_sec    ← per component, where exposed
latency.<component>.p95_sec
```

**Cross-corpus latency consistency** (parallels §4): a detector that's
fast on gtzan (30 s tracks) and slow on osu (3 min tracks) — or vice
versa — exposes either a per-track overhead constant (favors short
tracks) or O(N²) scaling (favors short tracks differently). The meta-
eval reports `latency.realtime_factor.spread` per detector.

**Implementation note**: the existing `_cached_detect` in
`tools/eval_beat_detection.py` returns cached `beats` directly. Cache
needs to also store `last_timing` — bump cache schema and re-run, or
re-store on cache miss. Don't reuse cached predictions without their
timing, or the latency table is silently from whichever uncached run
happened to populate the cache (instrument bias).

## Critical: gtzan track length

GTZAN tracks are 30 seconds. The detectors run very fast on them
compared to 3-min osu tracks. Total runtime for 999 tracks * 3
detectors should be well under an hour for percentile, longer for
madmom (the slow one). Cache layer in `eval_beat_detection.py`
already handles re-runs; this should "just work."

## Known gotchas

- jazz.00054 is corrupt audio and missing from annotations. Skip it.
- A few GTZAN files have known duplicate-tracks-across-genres issues
  (mismatches between filename label and actual song). Sturm 2013 did
  the audit. For beat tracking this is mostly fine — the beat times
  are correct, only the genre label is suspect — but if any
  consistency check assumes 100 tracks per genre, account for the
  ~50 known dupes.
- Annotations are in tab-separated format; pandas read_csv or numpy
  loadtxt with sep='\t' handles it.

## Scope discipline

All seven numbered sections ship together in one handoff. Don't split
the meta-eval (§4) into a follow-up just because it's the only piece
introducing new registry infra — the consistency-spread metric is
the point of adding gtzan, and shipping the corpus loader without
the cross-corpus check loses the motivation in §0.

§7 (latency table) belongs in the same ship because it bumps the
detector-prediction cache schema. Retrofitting later means re-running
every detector across both corpora a second time to backfill timing.
One re-run is acceptable; two is wasteful and invites the bias the
implementation note in §7 calls out.

## Out of scope here

- Adding Ballroom or Hainsworth as third corpus (revisit if osu/gtzan
  disagree dramatically; pick whichever is the missing genre coverage).
- Migrating osu away as primary corpus (still useful, just no longer
  alone).
- Hand-validating osu downbeat subset (defer until we know whether
  gtzan downbeat numbers are good enough to act on alone).
