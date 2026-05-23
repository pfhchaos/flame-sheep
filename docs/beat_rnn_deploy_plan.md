# Beat-RNN Deployment Plan

Wrap the trained beat-RNN in the proper detector interface and deploy with a rollbackable feature flag. Eval results justify deployment; this is the operationalization step.

## Eval-driven justification

- F1@50ms: **0.553** (Beat-RNN) vs **0.394** (current_system) — +40% relative improvement against osu ground truth
- Matches BeatNet teacher family across model variants (tied with Ballroom at 0.567, ahead of Rock at 0.547, GTZAN at 0.517)
- Inference cost: **~165x real-time** (~0.6% of audio duration), vs current_system's ~16x real-time → **~10x faster**
- Memory: 19K params ≈ 75KB for model weights
- Per-frame live inference: GRU step on CPU, expected sub-millisecond

Data leakage check: training corpus is classical + Michael Jackson, eval is anime/game/electronic/math rock — disjoint domains. The model generalizes across genre clusters, which is the strongest possible case for production deployment.

## Goal: rollbackable deployment

Ship the RNN behind a config flag with the current detector as fallback. Both implementations coexist in the codebase. Default starts as current_system; flip to RNN after live validation. Easy to revert with a config toggle.

## What needs to happen

### 1. Streaming inference wrapper

The eval pipeline used **batched** inference (whole track at once). Live inference is per-frame:
- GRU hidden state must persist across frames
- Linear input projection (216 → 32) per frame
- GRU step (32 → 48) per frame, carrying hidden state
- Linear output (48 → 1) + sigmoid per frame
- Output continuous beat probability per frame

Expected compute: <1ms per frame on CPU. Plenty of budget at 100Hz frame rate (10ms budget).

### 2. Detector interface implementation

`flame_sheep_audio` already has a clean abstraction:

```python
class BeatDetectorBase(ABC):
    @abstractmethod
    def detect(self, frame: SpectrumFrame) -> list[BeatEvent]:
        ...
    
    @abstractmethod
    def reset_bands(self):
        ...
```

Existing implementations: `FluxBeatDetector` (current), `PercentileBeatDetector`.

New implementation: `BeatRNNDetector(BeatDetectorBase)` — same interface, RNN-driven internally.

**Input mismatch caveat**: `SpectrumFrame` is FFT-based. The Beat-RNN was trained on CQT features (108-bin spectrum + 108-bin first derivative). Two options:
- (a) `BeatRNNDetector` does FFT → CQT conversion internally per frame
- (b) Add CQT computation to the spectrum pipeline upstream, expose via `SpectrumFrame` or a sibling type

Recommend (a) for the initial implementation — keeps the RNN's input dependency contained to its detector class. If CQT becomes useful for other detectors later, refactor to (b).

### 3. Causal peak picker

The offline eval's peak picker has full lookahead. Live has none. Need a causal version:
- Use a small rolling buffer (~10-20 frames = 100-200ms latency)
- Emit a beat when activation has clearly passed a local maximum within the buffer
- Tunable: minimum peak height (threshold), minimum gap between peaks (min_peak_distance), buffer size (lookahead)

Acceptable latency for live: probably 100-200ms — barely perceptible against visual response. The current detector likely has similar latency.

### 4. Configuration

Add to `flame_sheep_audio/config.py` (or similar):

```python
beat_detector: Literal['flux', 'percentile', 'rnn'] = 'flux'  # default to existing
beat_rnn_weights_path: str = 'flame_sheep/data/beat_rnn_continuous.npz'
beat_rnn_threshold: float = 0.5
beat_rnn_min_peak_distance_ms: float = 100
beat_rnn_lookahead_ms: float = 150
```

Default remains `flux` (the current `FluxBeatDetector`) for safety. Override via config file or CLI flag (`--beat-detector rnn`).

### 5. Daemon integration

`flame_sheep_audio` runs as a daemon (per yesterday's design discussion). RNN inference goes inside the daemon, beat events get published to shmem like the existing detectors.

No client-side changes needed. Visualizations consuming beats from shmem don't care which detector produced them.

### 6. Weights file location and bundling

The trained weights at `~/datasets/beat-labels-mmap/beat_rnn_continuous.npz` are not in the repo. For deployment:
- Copy to `flame_sheep/data/beat_rnn_continuous.npz` (alongside `cnn_scorer_personal_vk.npz` etc.)
- Add to packaged data so it ships with the package install
- Document the model version + training corpus in a sidecar JSON (so future updates can be tracked)

## Implementation stages

Each stage is independently shippable. Tests pass at end of each stage.

### Stage 1: Streaming wrapper, offline validated

- Implement `BeatRNNDetector` with the per-frame streaming logic
- Validate by streaming a **multi-minute audio file (4+ min minimum)** frame-by-frame and comparing beat positions to the batched eval output
- Acceptance criteria:
  - Stream output beats ≈ batch output beats (modulo causal peak picker's small latency offset)
  - **F1 in first 30 seconds ≈ F1 in last 30 seconds** (within ~5pp) — this validates hidden state isn't degenerating over long sessions
  - If degenerating: implement periodic hidden state reset every 256 frames, re-validate
- ~half day (assuming no hidden state issues; +half day if degeneration requires the reset mitigation)

### Stage 2: Interface compliance + daemon plug

- Make `BeatRNNDetector` conform to `BeatDetectorBase`
- Plug into the daemon's detector selection path
- Smoke test: daemon starts with RNN selected, audible audio produces beat events in shmem
- ~half day

### Stage 3: Config flag and CLI override

- Add config field + CLI flag for detector selection
- Document the toggle in README + docs
- Default stays `flux`; manual override required to enable RNN
- ~few hours

### Stage 4: Live wallpaper test (foreground use)

- Run `python -m flame_sheep --audio-detector rnn` (or however the override works)
- Observe wallpaper behavior with RNN-driven beats
- Check: does the wallpaper feel responsive? Are beats syncing visually? Any audible→visual lag?
- Compare subjective experience to default flux detector
- Document findings; identify any edge cases (silence handling, polyrhythmic music, abrupt tempo changes)

### Stage 5: Side-by-side telemetry

- Run both detectors concurrently (one drives visuals, both log events)
- Build a comparison view: timeline of beat events per detector
- Identify systematic differences
- This is the validation that justifies a default flip

### Stage 6: Default flip (deferred until Stage 5 results are confidence-high)

- Change config default from `flux` to `rnn`
- `flux` remains available via override
- Keep `FluxBeatDetector` code intact for rollback

### Stage 7 (deferred, optional): Remove FluxBeatDetector

- Only after the RNN has run as default for an extended period without issues
- Confidence threshold: "I haven't manually overridden to flux for at least a month"
- This is cleanup, not blocking

## Beat classification: heuristic-based, unchanged from current

The Beat-RNN outputs a single continuous beat probability — it doesn't distinguish downbeat / backbeat / other. The training labels had this distinction; the continuous formulation collapsed it.

**This isn't a regression vs current_system**. The current system also uses a frequency-band heuristic for kind classification (low band = downbeat, mid = backbeat, high = other). The Beat-RNN replaces ONLY the temporal beat detection layer; the kind classification logic reuses the existing heuristic, applied at each Beat-RNN-detected frame.

Integration shape:
```
1. Beat-RNN detects beat at frame N (improved accuracy vs current_system)
2. Existing frequency-band heuristic classifies at frame N (same accuracy as current_system)
3. Emit BeatEvent with kind set
```

Net result: improved "when" detection, parity "what kind" classification. Same shortcomings as the existing heuristic (struggles with electronic music, acoustic-without-drums, polyrhythms) — you already have those failure modes in production.

**Future v2 model** could output two channels (beat + downbeat) and replace the heuristic. The case for v2 is "the heuristic fails in cases you actually hear" — empirical question, observe after v1 deployment.

## Critical empirical question: hidden state degeneration

**Training regime**: BPTT on 256-frame chunks (2.73s), hidden state reset at each chunk boundary. The model has *never* seen its own hidden state past frame 256.

**Live regime as-built**: hidden state accumulates indefinitely across the entire song.

**Possible failure modes**:
1. **Fine actually**: GRUs often generalize OK to longer contexts. Most likely outcome.
2. **Slow degeneration**: hidden state drifts into unexplored activation regions, accuracy declines over song duration.
3. **Catastrophic divergence**: hidden state saturates/explodes mid-song.

**This must be empirically validated, not assumed**. Stage 1's validation track MUST be multi-minute (4+ minutes), and the comparison must measure F1 in the first 30 seconds vs the last 30 seconds. If F1 drops more than ~5pp from early to late, hidden state is degenerating.

**Mitigation if degeneration is observed**: reset hidden state to zero every 256 frames, matching the training regime exactly. Cost: ~2.7s of post-reset warm-up per chunk. On a 4-minute track that's ~3% of frames in warm-up mode. Probably minor F1 impact. Safe by construction since the model was trained for this exact pattern.

**More sophisticated mitigation if simple reset hurts**: overlap two parallel hidden states with staggered reset cycles, so at any moment at least one has accumulated useful context. Don't build this unless simple reset is clearly insufficient.

## Eval fairness caveat — peak picker lookahead

The headline F1@50ms = 0.553 was measured with **offline peak picking** (full lookahead over entire activation curve). Live inference can't see the future. Expected degradation: **5-10pp** when constrained to a causal peak picker with 100-200ms lookahead window. Realistic deployed F1 estimate: **0.45-0.53** depending on how aggressive the causal peak picker can be.

Even at the worst case (0.45), Beat-RNN is still +6pp over current_system (0.394). The ship decision survives the lookahead correction. But the headline number should be restated as an upper bound, with a separate "realistic deployed" range.

Recommendation: add a `--causal` flag to the eval harness that uses a streaming peak picker matching the live design. Re-run the comparison with all three detectors in causal mode. That gives the apples-to-apples comparison number that matches what the user will actually experience.

## Open design questions for the alter ego

- **FFT → CQT conversion location**: in `BeatRNNDetector` (encapsulated but recomputes if other detectors want CQT later) or in `SpectrumFrame`-adjacent pipeline (shared but invasive)? Lean toward encapsulated for stage 1; refactor only if needed.
- **Causal peak picker lookahead size**: 100ms? 150ms? 200ms? Tradeoff is audio→visual latency vs F1 accuracy. Probably tune empirically once Stage 1 is running.
- **CPU vs Vulkan inference**: RNN is small enough for CPU (sub-ms per frame). Vulkan path adds complexity but would be uniform with the rest of the ML infra. Recommend CPU for v1; Vulkan only if profiling shows CPU is a bottleneck (it won't be).
- **Threshold tuning**: live deployment threshold may differ from offline-eval optimal. Worth tuning once Stage 4 is running; default 0.5 sigmoid threshold is a starting point, not necessarily best.

## Rollback path

The rollback story is "change one config line":

```yaml
# config.yaml
beat_detector: flux  # was: rnn
```

No code changes. Both implementations coexist. Trivially A/B'd.

Removal of `FluxBeatDetector` is **not** part of this plan — that's Stage 7, optional, only after extended confidence. Until then, the cost of keeping both detectors is small (a few hundred lines) and the value of the rollback option is high.

## Effort estimate

- Stage 1 + 2 (streaming wrapper + interface): ~1 day
- Stage 3 (config flag): few hours
- Stage 4 (live test): half day of operator time, some calendar time for observation
- Stage 5 (telemetry): half day to build, some calendar time for data
- Stage 6 (flip default): trivial code change, all the cost is in confidence-building from Stages 4-5

Total active development: ~2 days. Plus an indeterminate amount of "let it run in foreground and see how it feels" time before flipping the default.

## Success criteria

After this plan is complete:

- ✅ `BeatRNNDetector` exists in `flame_sheep_audio` and is selectable via config
- ✅ Live wallpaper runs with the RNN detector for an extended session without crashing
- ✅ Beat events fire at audibly-correct moments (subjective)
- ✅ Latency budget is met (visual response is in sync with audio beats)
- ✅ The flip from `flux` to `rnn` (and back) takes one config line change
- ✅ Documentation reflects both detectors and how to choose between them

Once these are met, the eval result is shipped to users — actual UX gain, not just a benchmark number.
