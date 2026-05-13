# Centroid Survey Refactor: Two-Stage Rendering Pipeline

## Context

Genomes render off-center because the fractal attractor's centroid doesn't align with the viewport. Currently, centroid is computed during GPU render and a magic 85% correction is applied at `load_genome()` time. This is fragile: the CNN scorer sees off-center images, users downvote fractals that are good but misplaced, and the correction factor is arbitrary.

The fix: compute centroid cheaply before the GPU render, permanently correct the genome's center and zoom, so everything downstream sees a properly framed fractal.

## Key Insight

`_aesthetic_score_cpu()` already runs a 5000-iteration chaos game on a 64×64 grid and computes centroid offsets. This is a natural survey step (~10ms) that already runs during `save_genome()`. We just need to extract the centroid/bbox, correct the genome, and then let the rest of the pipeline work on a properly framed genome.

## Changes

### 1. `genome.py` — Add survey + correction methods

**`survey_attractor(n_test=5000, survey_bound=8.0) -> dict`**
- Runs chaos game on 2x-wide viewport (bound=8.0 vs display bound=4.0)
- Collects `sum_x`, `sum_y`, `n_hits` directly from point positions (not grid cells) for accurate centroid
- Computes bounding box from 5th/95th percentile of hit positions
- Returns: `centroid_x`, `centroid_y`, `bbox_min/max`, `coverage`, `in_viewport` flag
- ~10ms, CPU only

**`correct_framing(survey, margin=1.2) -> None`**
- Sets `self.center` to rotated centroid (must account for `self.rotation`)
- Sets `self.zoom` from bounding box extent with margin, clamped to sane range (0.3-5.0)
- Rotation transform: `center = R(rotation) @ centroid` because the shader applies rotation before center offset

**`survey_and_correct(reject_outside=True) -> bool`**
- Convenience: runs survey + correct, returns False if attractor misses viewport

### 2. `genome.py` — Integrate into `Genome.random()`

```python
for _ in range(50):
    g = cls()
    # ... generate params ...
    if g.is_viable():
        if g.survey_and_correct():
            return g
```

Every genome exits `random()` already framed. Callers don't need to remember.

### 3. `storage.py` — Schema + load changes

**Add column:**
```sql
ALTER TABLE genomes ADD COLUMN framing_version INTEGER DEFAULT 0
```

**`save_genome()`** — set `framing_version=1` for corrected genomes.

**`load_genome()`** — only apply legacy 85% correction for `framing_version=0`:
```python
if auto_center and framing_version == 0 and centroid is not None:
    # legacy correction for pre-survey genomes
    genome.center -= 0.85 * np.array([cx * 4.0, cy * 4.0])
```

New genomes skip correction entirely — their params already have the right center/zoom.

### 4. `loops.py` — Evolution offspring

After mutation/crossover produces a new genome, call `survey_and_correct()` before `save_genome()`. The genome is already viable (parent was), but framing may have shifted.

### 5. `gpu_render_worker.py` — No structural changes

The renderer loads genome params from DB. Since corrected center/zoom are baked into the stored JSON, it automatically renders well-centered images. The centroid it computes from the swept histogram becomes a **validation metric** (should be near zero for corrected genomes).

### 6. CNN scorer — No changes

Gets properly centered PNGs now. Scores improve because it's judging content, not placement.

## Operation Sequence (per genome)

```
1. Generate random params                       ~0.1ms
2. is_viable() — 2000-iter bounds check          ~5ms
3. survey_attractor() — 5000-iter, 2x viewport   ~10ms
4. Check in_viewport — reject if outside
5. correct_framing() — set center + zoom          ~0ms
6. aesthetic_score() — runs on corrected genome   ~10ms
7. save_genome() — stores corrected params        ~1ms
   ... background ...
8. GPU render — sees centered genome              ~seconds
9. CNN score — sees centered image                ~1ms
10. load_genome() — no correction needed
```

Total overhead added: ~10ms per genome (the survey). The aesthetic_score call that follows is already happening.

## Files to Modify

| File | Changes |
|------|---------|
| `flame_sheep/genome.py` | Add `survey_attractor()`, `correct_framing()`, `survey_and_correct()`. Modify `random()` to call `survey_and_correct()`. |
| `flame_sheep/storage.py` | Add `framing_version` column migration. Update `save_genome()` to set it. Gate legacy correction in `load_genome()` on `framing_version == 0`. |
| `flame_sheep/loops.py` | Call `survey_and_correct()` on offspring after mutation/crossover, before `save_genome()`. |

## Migration

- New genomes: automatically framed at creation time
- Existing genomes: continue to use 85% correction until backfilled
- Backfill script: load each genome, run `survey_and_correct()`, re-save params + set `framing_version=1`
- After backfill: remove `auto_center` code path

## Verification

1. Generate 100 random genomes — verify `survey_and_correct()` pass rate > 50%
2. Check centroid offsets in DB — new genomes should have near-zero centroid_x/y
3. Render a few and visually confirm centering
4. Run evolution — verify offspring are properly framed
5. Load a legacy genome — verify 85% correction still applies
6. Load a new genome — verify no correction applied
