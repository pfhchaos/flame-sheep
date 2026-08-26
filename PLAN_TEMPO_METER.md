# Tempo + meter redesign — handoff for implementation

**Date:** 2026-06-05
**Scope:** Complete replacement of tempo tracking. Greenfield meter classifier. Eval framework redesign. Cleanup of dead/experimental code.
**Not scope:** Beat / downbeat NN. That's the next phase after tempo + meter ship.

## Why now

Three converging reasons:

1. **Tempo tracking is fragmented**: three trackers exist (BTrack, ACF, Percival), one failed reconciliation module, an IOI tracker in a parallel package whose actual use is unverified. The accumulated cruft makes any change to tempo behavior require touching multiple places. Replacement is also cleanup.

2. **Production BTrack tracker silently ignores hints**: `BTrackTempoTracker` does not override `TempoTrackerBase.hint_tempo()`, which defaults to a no-op. Any tag-BPM data we have is currently inert. This is a real bug, not just an architectural smell.

3. **The label-channel-order bug we discovered (2026-06-04)** opened the question of whether the beat-RNN should be retrained. Before committing to that, we should fix the inputs the beat-RNN would consume — tempo and meter — so a future retrained beat model gets clean inputs and the eval can isolate "is the model good" from "are the inputs good." Fix the upstream signals first; that work generalizes whether or not we eventually re-train the beat model.

## Design constraints

**Perceptual** (this drives priority order for tempo accuracy):
- Tempo's main role: set cooldowns (in beat-lengths) and EMA window sizes (in beat-lengths). It does NOT directly drive event timing.
- Tempo is also used as a refractory in beat detection: `Beats(0.25)` minimum spacing between beat events, scaled to frames via `TempoScaler`.
- **Octave errors are catastrophic**: 2× tempo error → EMA windows mis-sized by 2×, cooldowns mis-sized by 2×, beat detection refractory mis-sized by 2×. Multiple downstream consumers visibly break.
- **Small numerical errors are barely perceptible**: 5-15% tempo error produces sub-perception cooldown deltas. Viewer doesn't notice.
- **Within-song stability matters more than per-frame accuracy**: a stably-wrong tracker is better than a jittery-correct one because EMA windows yo-yo visibly when tempo jumps.

**Deployment**:
- Real-time on production load (user's system is never idle: wallpaper render + llama.cpp + audio daemon all coexist on the same machine).
- "Fast on idle system" benchmarks are useless. Profile under representative load.
- CPU-bound (audio daemon thread). GPU is contested by the wallpaper render.

**Engineering scope**:
- flame-sheep is a visualization optimizer, not a music transcription engine. Optimization target is "produce smooth visually-coherent event stream from audio." Music-theoretical correctness is incidental.
- Inconsistency is the real failure mode, not interpretation difference. Treating beat 2 of 6/8 as a downbeat is fine; flickering between interpretations is not.

## Priority order for tempo accuracy

This shapes both algorithm choice and eval metric design:

1. **Avoid octave errors** (catastrophic) — highest priority
2. **Get within tolerable tempo (~20%)** — most of what's left after #1
3. **Within-song stability** — protects EMA window sizing from jitter
4. **Strict precision (4% F1)** — nice-to-have, academic comparison point only

Allocate optimization effort proportional to this list.

## Architecture: tempo as state machine with conditioning inputs

The intended pattern: tempo is an internal state with inertia. Raw estimates feed in continuously and move the state proportional to confidence. Hints nudge state strongly via a high-confidence injection. **No hard lock anywhere.**

**Correction from earlier framing:** the plan initially preserved a "locked" property and described hints as "reset + lock." That was based on a misread of the existing code — the existing trackers do have a lock/unlock hysteresis, but inertia (not lock) is what we actually want. Drop the lock concept entirely. State stability emerges from the inertia constant, not from a separate locked mode.

**State machine properties to preserve:**
- `bpm`, `effective_bpm`, `confidence`, `phase` (existing `TempoTrackerBase` interface)
- `effective_bpm` blends with 120 BPM default at low confidence
- `song_started()` → `reset()` (verified: `flame_sheep/audio/tempo.py` `reset()` correctly clears state)

**State machine properties to add:**
- Hints actually accepted by production tracker (fix the BTrack no-op). Hint sets `bpm` to hint value, sets `confidence` to a high (but not maximum) value, then continues normal inertia-based update. Subsequent evidence can override if accumulated counter-evidence is strong enough — this is what catches mis-tagged tracks (half-time / full-time confusion) without needing a separate "is this hint trustworthy" check.
- Stability metric exposed as a property (for eval and for downstream "is tempo trustworthy right now" decisions)
- Tempo + meter both exposed as outputs (meter classification result available alongside BPM)

**State machine properties to remove:**
- `locked` property and all lock/unlock hysteresis. Replaced by confidence + inertia. If anything currently consumes `locked`, replace the consumer's check with `confidence > threshold`.

## Tempo algorithm choice (decision deferred to implementation)

Three viable directions. Pick one based on prototyping and eval results:

### Option A: classical ACF (autocorrelation) with octave-correction postprocessing

- Implement clean ACF on continuous onset strength (per existing `AutocorrelationTempoTracker` reference)
- Rayleigh prior toward ~120 BPM
- Octave correction in postprocessing:
  - BPM-range prior: if predicted outside [60, 180], try ×2 / ×0.5 first
  - Beat-density check: count beat events/second, compare to predicted tempo period
  - Tag hint anchor: if tag-BPM available, use as octave reference even if exact value disagrees
  - Hysteresis: sticky octave commitment within a song
- Sub-millisecond per analysis window expected. Should be the cheapest option.

### Option B: comb filter bank

- Bank of comb filters at candidate BPMs
- Each filter integrates onset strength at its lag
- Highest-energy comb = tempo estimate
- Octave correction same as Option A
- Tunable resolution (number of filters) vs compute trade-off
- Similar latency to ACF; potentially more robust to noise

### Option C: hybrid (ACF + comb filter ensemble)

- Both methods run in parallel
- Combine via weighted vote or confidence-based switching
- More robust on hard cases (osu electronic content)
- 2× compute of either alone, but both are cheap

**Recommended starting point:** Option A. Cheapest, simplest, most likely to be adequate. Upgrade to B or C only if A doesn't reach acceptable accuracy on osu (the harder corpus).

**Do not pursue NN-based tempo unless A/B/C all fail.** Tempo is well-served by classical algorithms. NN would add training infrastructure, deployment compute, and train/serve consistency risk for marginal accuracy gain. The cost-benefit doesn't justify it for tempo specifically.

## Octave error mitigation

Octave errors are the catastrophic failure mode. Handle them explicitly at multiple layers:

**In the estimator** (algorithm-level):
- Rayleigh prior on tempo (Percival-style): weight peaks by perceptual tempo preference (peaks at ~120 BPM)
- Range constraint: hard-limit BPM to [60, 220] before postprocessing
- Existing ACF code (line 239) already has "prefer lower octave: if BPM > 200, check for half-BPM peak" — keep this pattern

**In postprocessing**:
- If tag-BPM hint available, use it as octave anchor (snap to nearest octave of tag)
- Beat-density sanity: if beat events fire at 2× or 0.5× predicted period rate, octave-correct
- Bar-length sanity: if predicted bar length is implausibly short (<1s) or long (>4s), octave error suspected

**In state machine** (jitter mitigation):
- Sticky octave commitment: once committed to an octave in a song, prefer to stay unless strong evidence dictates change
- Hysteresis: switching from low-octave to high-octave requires more evidence than maintaining current
- Prevents the "yo-yo between octaves" failure mode that's the worst stability failure

## Meter classifier (greenfield)

**Simplification chain we landed on:**

1. Full meter detection (4/4, 3/4, 6/8, 5/4, 7/8, 11/8, ...) is overkill for visualization
2. Even/odd beat-count classification is the natural simplification
3. Within odd: treat all as 3/4 (since 3/4 is by far the most common odd-meter case, the rest are <1% of music)
4. So: binary classifier between 4/4-strategy and 3/4-strategy

**Two strategies for downbeat prediction:**

- **Regular (even/4/4ish)**: predict downbeats every 4 beats. Reset phase on particularly strong beats.
- **Irregular (odd/3/4 strategy)**: predict downbeats every 3 beats. Reset phase on particularly strong beats.

Both strategies include strong-beat resync, so misclassification cost is low. Wrong meter → wrong default cadence, but resync catches structural anchors.

**Implementation approach** (decision deferred):

- **Classical heuristic first**: 3-beat vs 4-beat periodicity in autocorrelation of detected beat intervals. Compute peaks in beat-interval ACF at lag-3 and lag-4; whichever is stronger wins.
- **Tiny NN fallback** if heuristic doesn't reach acceptable accuracy: small CNN or MLP on log-mel spectrogram features, outputs single bit per long window
- **Data acquisition** (for either approach):
  - Default everything to 4/4
  - Crawl music nerd sources for non-4/4 tags: Reddit (/r/musictheory, /r/progrockmusic, /r/jazz), Wikipedia song articles, "songs in odd time signatures" listicles, Hooktheory database, Genius.com user annotations, TheoryTab database, Songsterr/Ultimate Guitar tabs
  - For odd meters (5/4, 7/8, etc.) → label as 3/4 (matches the "all odd → 3/4" simplification)
  - Cross-reference sources for disagreements (especially 6/8 vs 3/4 confusion) as a quality signal

**Confidence handling**: meter classifier outputs binary, but classification confidence should be exposed. Low-confidence classification → bias toward 4/4 (more common, faster cadence, lower cost of being wrong).

**Cold start**: default to 4/4 before classifier converges. Stasis during initial song seconds is worse than committed-but-wrong 4/4 prediction.

## Hint handling

**Bug to fix**: `BTrackTempoTracker` currently silently ignores hints because it doesn't override `TempoTrackerBase.hint_tempo()`. The new tracker must implement hint acceptance.

**Hint policy** (no lock — inertia + confidence does the work):
- Tag-BPM hints are accepted in-range only (clamp to [60, 220]).
- Accepting a hint sets `bpm` to the hint value and `confidence` to a high (but not maximum) value. Normal inertia-based update continues from there.
- Subsequent estimator evidence can move the state if it accumulates against the hint. This naturally handles mis-tagged tracks (the common half-time / full-time confusion: a song tagged 160 BPM that's actually 80 BPM gets a strong initial push to 160, but ACF evidence consistently at 80 will pull it down over several windows).
- Meter hints (if available from tags): same pattern for meter classifier — strong initial bias, not absolute commitment.
- Bootstrap: hints during cold-start phase prevent the 120-BPM default from dominating early convergence.

**Out-of-band hint sources to integrate** (later phase, not initial implementation):
- File metadata (ID3 BPM tag, FLAC tags)
- MusicBrainz / Spotify Web API lookups (need network, async)
- Last.fm metadata
- User-provided override

Initial implementation only needs: accept a hint when provided via API; don't worry about acquiring hints from external sources yet.

## Eval registrations for tempo + meter

**Not a new framework.** The existing `evals/` system (registry + scorecard
+ DISCIPLINE.md) already covers the discipline this plan needs; the
work here is registering new tempo and meter metrics on top of it,
using the same `@register` mechanism that `beat.osu.*` and
`beat.gtzan.*` use. Multiplicative-ratio histograms become per-eval
artifacts dumped alongside the scorecard row; the cross-corpus
consistency check uses the meta-eval mechanism already added per
`HANDOFF_GTZAN.md` §4. Latency is captured via the §7 mechanism in
the same handoff (detector `last_timing` → scorecard aggregates) —
no separate timing pipeline needed.

**Two corpora, dual-evaluated:**

- **GTZAN** (already on disk, see `HANDOFF_GTZAN.md`): 999 tracks with tempo + beat annotations. Mainstream genre distribution. Academic-grade ground truth. "Is this implementation in the right ballpark" check.
- **osu** (already in eval infrastructure per `tools/eval_tempo_reconcile.py`): electronic / anime / J-pop / game music. Crowd-sourced timing-critical annotations. Genre distribution closer to deployment audience. "Does this matter to users" check.

**Tempo metrics** (multi-metric, not just F1):

1. **Tolerable accuracy** (within ~20%): captures viewer-perceived correctness. Primary deployment metric.
2. **Octave error rate**: % of songs where prediction is integer multiple/divisor of truth (×2, ×3, ×0.5, ×1/3). Catastrophic-failure-mode metric.
3. **Within-song stability**: variance of tempo estimate within a song. EMA-jitter prevention metric.
4. **Real-tempo-change adaptation**: on a small held-out set of songs with known tempo changes, how long after a real change does state catch up.
5. **Cold-start convergence**: how long until state reaches within tolerance of truth from 120 init.
6. **Strict F1 at 4% tolerance**: academic comparison point. Reports for sanity check, not for optimization target.
7. **OE1 (F1 octave-tolerant)**: predicted × 1, ×2, or ×0.5 all counted correct. Isolates "periodicity right, octave wrong" cases.

**Multiplicative-ratio histogram**: track distribution of (predicted / truth) ratios across the corpus. Clusters at 1.0 = good, clusters at 2.0/0.5 = octave error, clusters at 1.5/3.0 = triplet error, smear = some other pathology. Diagnostic granularity that point metrics hide.

**Meter metrics:**
1. **Classification accuracy** (4/4 vs 3/4 binary): straightforward
2. **Mid-song stability**: variance of meter classification within a song. Wrong-but-stable is fine; oscillating is broken.

**Per-corpus reporting**: every metric reported separately for GTZAN and osu. Improvement on GTZAN alone is academic; improvement on osu translates to user-visible improvement. Compare GTZAN-delta vs osu-delta to detect overfit-to-corpus changes.

**Performance profiling alongside accuracy** — two-tier:

- **In-band (default, every eval run)**: `HANDOFF_GTZAN.md` §7 mechanism. Detectors expose `last_timing: dict[str, float]`; eval runner aggregates into scorecard rows. Tempo trackers and meter classifiers must implement the same contract. Gives p50/p95/p99 latency per component per corpus, plus cross-corpus realtime-factor spread, without extra invocation.
- **Out-of-band (investigating regressions)**: `cProfile` (offline statistical), `py-spy` (live, no code changes), `perf` (system-level). Reach for these when an in-band scorecard row regresses and you need to know which line moved.

Don't fork the in-band mechanism — use the same `last_timing` contract; the existing test coverage and cache schema already handle it.

**Profile under production load**: run a representative wallpaper-rendering process while eval is running. Idle-system timing is not representative.

**Beat-detection F1 leverage measurement**: separately measure beat-detection F1 with (a) ground-truth tempo fed in vs (b) estimated tempo fed in. The delta is an upper bound on how much beat detection can improve from tempo improvements alone. Tells us whether tempo is high-leverage or low-leverage for downstream beat work.

## Code to delete

After replacement, these modules should be removed (verify each is actually unused before deletion):

- `flame_sheep_audio/src/flame_sheep_audio/tempo.py` (Percival implementation; mark experimental, replace)
- `flame_sheep_audio/src/flame_sheep_audio/tempo_acf.py` (ACF tracker; subsumed by new tracker)
- `flame_sheep_audio/src/flame_sheep_audio/tempo_btrack.py` (BTrack wrapper; replaced)
- `flame_sheep_audio/src/flame_sheep_audio/tempo_reconcile.py` (failed reconciliation per `feedback_reconciliation_failed.md`; not deployed)
- `flame_sheep/audio/tempo.py` (separate IOI tracker; verify it's actually dead before deletion — may be used in different code path)

**Keep:**
- `flame_sheep_audio/src/flame_sheep_audio/tempo_scaler.py` (utility for Beats → frames conversion; unchanged)
- `TempoTrackerBase` interface (in `tempo.py`) — replace the Percival class but keep the abstract base. Add meter classifier output to the interface.

**Net code reduction**: probably ~1000 lines removed, ~300-500 added. Project complexity goes down.

## Implementation phases

**Phase 0: prep**
- Verify dead-code question for `flame_sheep/audio/tempo.py` — is it actually used anywhere? If not, delete.
- Get the multi-metric tempo eval framework in place before changing anything. Establish baselines for current BTrack tracker on GTZAN and osu across all metrics. Without baselines, "improvement" is not measurable.

**Phase 1: tempo tracker replacement**
- Implement Option A (ACF with octave-correction postprocessing) as new `TempoTracker` class
- Wire hint acceptance from day 1 (fix the BTrack bug as part of new implementation)
- Pass eval framework. Compare to BTrack baseline on all metrics.
- If acceptable: ship it, delete dead modules
- If not: prototype Option B, repeat

**Phase 2: meter classifier**
- Acquire labeled corpus via pedant-trail crawl + tag scraping
- Implement classical heuristic (beat-interval ACF peak comparison)
- Evaluate accuracy on GTZAN (mostly 4/4, validates 4/4 detection) and osu (more meter variety, validates 3/4 detection)
- If classical heuristic insufficient: small NN (rules table converts pedant-label time-signature → odd/even per the simplification chain)
- Expose meter classification result alongside tempo through `TempoTrackerBase`-like interface

**Phase 3: integration with existing beat detection**
- Feed tempo and meter into existing percentile beat detector via `TempoScaler`
- Tempo prior on percentile threshold (looser when tempo confident, tighter when uncertain)
- Meter conditioning on downbeat selection (4-beat or 3-beat cadence)
- Measure beat-detection F1 delta. Should improve via reduced refractory-error and downbeat-cadence-error.
- This integration validates the "tempo + meter as inputs" architecture before committing to NN beat model that takes them as inputs.

**Phase 4 (out of scope for this handoff)**:
- NN beat / downbeat model that takes tempo + meter as conditioning inputs
- Causal CRNN architecture (reference: BeatNet, with the numpy `in1d` → `isin` patch — see earlier inventory work)
- Multi-band activation function inputs
- Activation-function-as-interface for swap-in/swap-out testing

## Pluggable component choices

The goal is **plug-and-play, not pick-one-up-front**. Each of the choices below should land as a swappable implementation behind a common interface, so the eval framework can run all combinations and the scorecard picks winners. Same approach as the existing detector swap in `eval_beat_detection.py --detector {current_system,beat_rnn,...}`.

1. **Tempo algorithm** — implement A (ACF + octave correction) and B (comb filter bank) behind the `TempoTrackerBase` interface. C (hybrid) becomes a thin wrapper that holds both and combines confidences. Eval reports each across both corpora.
2. **State update style** — inertia-only (per architecture section above) is the default; if anyone wants to also try a "snap-to-locally-confident-estimate" style for comparison, it goes in as a third `TempoTrackerBase` implementation. The lock/unlock hysteresis from the existing code is NOT one of the swappable options — drop it everywhere.
3. **Meter classifier** — classical (beat-interval ACF peak comparison) ships first. Tiny NN goes in as second implementation behind the same `MeterClassifierBase` interface only if classical numbers are inadequate. Eval framework already supports both.
4. **Fallback tracker** — defer the decision; ship without a fallback. If the new tracker turns out to have a load-failure failure mode in production (missing native deps, model files), add fallback then. Current BTrack-with-ACF-fallback pattern was added reactively, not designed in advance; replicate that pattern only if a real failure mode appears.

## Verification tasks (not punted)

- **`flame_sheep/audio/tempo.py` (the IOI tracker)**: verify whether it's actually dead code or used in some forgotten code path. Delete or document. Do this in Phase 0 before any new code lands; deleting it later is harder if something silently consumes it.

## Connects to

- `evals/HANDOFF_GTZAN.md`: corpus this plan builds on for eval
- `evals/DISCIPLINE.md`: eval discipline this plan should match
- Memory: `project_beat_rnn_third_failure_pattern.md` — explains why beat-RNN is out of scope for now
- Memory: `feedback_reconciliation_failed.md` — explains why `tempo_reconcile.py` is dead
- Memory: `feedback_data_representation_first.md` — explains why fixing inputs (tempo, meter) before model is the right order

## Calibration notes for future-Claude reading this

- Flame-sheep optimizes for visualization, not transcription. Don't apply academic-tempo-detection metrics as the primary success criterion; use the perception-weighted multi-metric stack above.
- The user's recall about existing tempo behavior had several errors during planning (e.g., thought initial state was 120 — actually 0 internally with 120 effective via blending). Verify against code before changing behavior.
- "This project has too fucking much code" — accurate. Replacement work is also cleanup work. Don't let new modules accumulate alongside the old ones; delete the predecessor as you ship the replacement.
