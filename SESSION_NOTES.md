# flame-sheep — Session Notes

## Session 13 (2026-04-20 / 2026-04-21)

### Major features added

#### 8 new variations (genome.py, flame.comp)
- splits (30), cloverleaf (31), julian (32), juliascope (33), tangent (34), cross (35), butterfly (36), curl (37)
- New var_params SSBO (binding 6) for per-transform variation parameters: julian power/dist, splits x/y, curl c1/c2
- Julian/juliascope param ranges tuned from JWildfire's randomize() methods: power 2-12 (50% negative), dist multi-modal
- Backwards compatible: old genomes with 30-element variation arrays pad to 38 on load
- `--benchmark-variations` CLI flag for GPU timing per variation

#### Audio thread decoupling (processor.py, source.py, _spectrum.py)
- Audio processing now runs in its own daemon thread at 512-sample hops (10.7ms, was 2048/42.7ms)
- 75% overlap FFT via SpectrumEngine.push_hop() sliding window
- PipeWireSource.read_hop() with threading.Condition for blocking reads + stop_event for clean shutdown
- AudioProcessor auto-detects mode: threaded for PipeWireSource, synchronous for FeedSource (tests)
- drain() → AudioSnapshot (events + spectrum + rms + waveform) replaces process() for render thread
- Removed Wayland frame-wait audio.process() keepalive hack — no longer needed
- COOLDOWN scaled from 6 to 12 frames to maintain ~128ms at faster hop rate

#### Test suite optimization
- Class-scoped fixtures for compose_loops — 1 call instead of 7 (saved ~240s)
- @pytest.mark.slow for loop composition/breeding tests
- Default pytest excludes slow tests: 63s (was 462s)
- Full suite: 187s (was 462s), 326 tests pass

#### Other changes
- Startup loop selection uses weighted random from top 20 (was always top 1)
- Mako notification styling: transparent One Dark, monospace, sharp corners, gray borders

### MUSDB18 evaluation results (150 tracks)

| Config | Precision | TP/sec | FP/sec | Vocal FP/sec | Median P |
|--------|-----------|--------|--------|--------------|----------|
| 2048-hop + sharpness | 0.790 | 3.1 | 0.8 | 0.3 | 0.829 |
| 2048-hop + no sharpness | 0.795 | 3.5 | 0.9 | 0.4 | 0.847 |
| **512-hop + sharpness** | **0.919** | 7.5 | 0.7 | 0.2 | **0.975** |
| 512-hop + no sharpness | 0.917 | 9.7 | 0.9 | 0.3 | 0.982 |

Key findings:
- 512-hop improved precision from 0.790 → 0.919 (+16pp)
- Vocal false positives dropped from 0.3 → 0.2/sec with finer hop
- Sharpness filter effect is marginal at 512-hop (0.919 vs 0.917) — hop size dominates
- Median per-track precision: 0.829 → 0.975

### Synthetic pattern benchmark (12 genres, old vs new hop)

| Type | Old recall | New recall | Old timing | New timing |
|------|-----------|-----------|------------|------------|
| Kick | 82.0% | 100.0% | 13.9ms | 8.4ms |
| Snare | 62.0% | 100.0% | 16.9ms | 3.2ms |
| Hihat | 37.5% | 90.7% | 29.6ms | 11.8ms |

### GPU variation benchmark (all 38 variations)
- All new variations well within budget (0.3-0.6x relative to linear baseline)
- Julian 0.33x, juliascope 0.44x, curl 0.36x — no performance concern
- Fastest: splits 0.29x, handkerchief 0.32x (scatter reduces atomic contention)
- Slowest: exponential 1.51x (exp() cost), cross 1.03x

### JWildfire variation catalog
- Analyzed 412 compatible 2D point transforms from JWildfire source
- Clustered by type (tiling, reflective, rotational, polar, conformal, fractal, etc.) + compute cost
- Saved to docs/jwildfire_variation_catalog.txt
- Identified crystallographic symmetry groups (sym_bg1-7, sym_ng1-17) — 24 cheap variations producing exact band/wallpaper symmetry

### Design plans (documented in memory, not yet implemented)
- Fitness redesign: symmetry detection (rotational, reflective, radial, periodic), fractal dimension, self-similarity, vote clamping (-1/0/+1), genome vote propagation from loops, palette graph diffusion, idle scoring
- Continuous audio features: spectral centroid, centroid-weighted RMS, percussiveness ratio, dynamic range, loudness normalization
- Tiling variations: arctruchet, hex, rectangles, kaleidoscope, voronoi, checkers
- Loop topology: palindrome and rondo modes
- Generalize wave-family variations on import (one parameterized version instead of 12 copies)

### Known issues
- OOM when running multiple MUSDB18 evals concurrently — eval_musdb.py fixed to process one track at a time instead of pre-loading all stems
- Variation class __firstlineno__ attribute (Python 3.13) collides with BLOB index 23 in benchmark name lookup — fixed with dunder filter

## Session 11-12 (2026-04-17 / 2026-04-18)

### Major features added

#### Aesthetic scoring (`genome.py`)
- `_score_from_histogram()` — shared scoring for CPU and GPU paths
- Five metrics: coverage, entropy, color_entropy, balance, complexity
- CPU scorer: 64x64 coarse grid, fast pre-filter for candidate genomes
- GPU scorer: full-res histogram readback via `renderer.histogram_data()`

#### Persistent storage (`storage.py`)
- SQLite database at `~/.local/share/flame-sheep/library.db`
- Tables: genomes, loops, loop_items, palettes, ratings
- Genome serialization (JSON roundtrip of all params)
- Loop storage with fitness components: mean/min coherence, diversity, palette_flow, smoothness
- Palette graph: palettes stored as nodes with fitness scores (contrast, saturation, harmony, smoothness)
- `score_loop()` — automated loop-level fitness with smoothness metric
- `score_palette()` — palette quality metrics
- User ratings feed into `update_loop_fitness()` (user likes add 0.5 per vote)
- Breeding lineage tracking (parent_a, parent_b on loops)

#### Motion fields (`storage.py`)
- 3x3 vector field characterizing visual motion between genome pairs
- `compute_motion_field()` — density centroid displacement per grid cell
- `motion_field_coherence()` — cosine similarity for comparing transitions

#### Loop composition (`loops.py`)
- `compose_loops()` — greedy construction from genome pool with min/max distance
- `crossover()` — splice subsequences between loops
- `mutate_loop()` — swap one genome for a nearby variant
- `evolve_loops()` — multi-generation evolution with fresh blood (30%) and diversity floor (>50% overlap rejected)
- `loop_overlap()` / `_too_similar()` — set intersection check for diversity

#### Loop playback (`main.py`)
- `FlameSheepCore` takes optional `Library`, loads top loop on startup
- `_swap_next_genome()` advances through loop cyclically
- Random start position in loop on load
- Downbeat detection: strong kicks reset the beat counter
- Swap every 4th kick (configurable `KICK_SWAP_EVERY`)
- Non-swap kicks pulse morph speed (`KICK_MORPH_PULSE = 0.03`)
- `active_loop_id` tracks current loop for voting
- Dislike switches to next best loop via `next_loop()`

#### Orthogonal musical axes
Three independent visual dimensions driven by different drum types:

1. **Kick → genome** (structure)
   - Every 4th kick swaps to next genome in loop
   - Intermediate kicks pulse morph speed
   - Downbeat detection resets counter on strong kicks

2. **Snare → palette** (color)
   - Palette graph traversal on snare hits
   - Energy controls jump distance (quiet=subtle shift, loud=dramatic)
   - Independent morph state (palette_current, palette_target, palette_t)
   - Minimum palette fitness threshold (0.5) filters low-quality palettes

3. **Hihat → zoom pulse** (texture)
   - Additive zoom boost on hits, decays at 0.95/frame
   - Subtle in heavy sections, noticeable in quiet passages

4. **RMS → brightness** (dynamics)
   - Tonemap brightness driven by audio RMS
   - Range: 3.0 (quiet/dim) to 9.0 (loud/vivid)
   - Centered around default of 6.0

#### Tempo tracker fixes (`tempo.py`)
- Per-beat-type IOI measurement (was global, causing tiny IOIs from mixed drum hits)
- `MAX_BPM` raised to 300 (was 200, too low for DnB)
- `IOI_HISTORY` tuned to 48 (was 64, too slow to unlock; 32 was too fast)
- `MIN_PASS_CONFIDENCE` lowered to 0.2 (was 0.5, blocked all beats during learning)
- Histogram range widened to 0.15-1.1s (was 0.25-1.1s)

#### Walker management
- `renderer.reset_walkers()` — re-randomize walker positions
- `needs_walker_reset` flag on FlameSheepCore, set on force_swap and loop switch
- Shader respawn: walkers that escape to infinity (isinf/isnan/abs>1e6) are respawned in-place

#### Sway reload handling
- EGL error checking on `make_current()` and `swap()` (return bool)
- Wayland connection health check via fd polling
- Watchdog thread: force `os._exit(1)` if render loop stalls >3 seconds
- Use `exec_always` in sway config for automatic restart

#### Evolution
- Vote-triggered: 5 votes for first few cycles, drops to 3 after cycle 3
- Runs in subprocess (not thread) to avoid GIL contention
- Throttled: only one evolution at a time, votes during evolution count toward next
- Uses separate Library instance (separate DB connection for subprocess)

#### CLI (`main.py`)
- `--generate-genomes N` — generate random genomes
- `--generate-palettes N` — generate random palettes
- `--compose-loops N` — compose loops from genome pool
- `--evolve` — run one evolution cycle
- `--loop-length N` — genomes per loop (default 6)
- `--stats` — print library statistics
- `--test-audio` — synthetic metronome for testing

#### Packaging
- Editable install via venv: `pip install -e .` in `~/projects/flame-sheep/.venv`
- `exec_always ~/.venv/bin/flame-sheep --wallpaper` in sway config
- Fixed pyproject.toml build backend (`setuptools.build_meta`)

### Known issues

1. **Sway reload watchdog** — uses `os._exit(1)` after 3s stall. Works but ugly. The real fix is detecting EGL surface invalidation, but `eglMakeCurrent` doesn't return errors on dead Wayland surfaces.

2. **Beat detection sensitivity** — tuned for drums, false positives on vocals (distorted vocals trigger snare, falsetto triggers hihat, Mario sound effects trigger everything).

3. **4K performance** — histogram atomic contention not addressed. Half-res render plan exists in session 10 notes.

4. **DB schema migration** — no migration system. Schema changes require deleting and regenerating the DB. Will matter once user data is worth keeping.

5. **Palette distance** — uses mean RGB for graph traversal, which is a rough approximation. Two very different palettes can have similar mean RGB.

### System info
- **Monitors:** DP-4 (1440×2560 portrait), DP-3 (1440×2560 portrait), DP-2 (3840×2160 landscape, main)
- **GPU:** Intel Arc A770 (Mesa, DG2), EGL 1.5, OpenGL 4.6, Vulkan
- **Stack:** Python + moderngl (GL 4.3 compute), sounddevice → PipeWire, scipy FFT, SQLite
- **Audio device:** Bose USB Audio (PipeWire monitor source, aliased as "Companion Speaker Analog Surround 5.1")

### Future work
- Genre-specific behavior profiles (swap timing, beat sensitivity, palette jump distance)
- Genre detection from audio spectral profile or music player metadata
- Palette evolution (breeding, mutation — analogous to genome loops)
- Source separation (Demucs) for clean drum stems
- DB migration system
- Half-res render + upscale for 4K
- Monitor auto-detection via EDID
