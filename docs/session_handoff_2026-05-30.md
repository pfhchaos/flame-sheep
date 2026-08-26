# Session Handoff 2026-05-30

## Subject: Vulkan perf investigation + per-genome shader specialization plan

This brief covers (a) the perf-gap investigation that landed on a Mesa-Xe Vulkan compiler issue, (b) population measurements that bound the per-genome vs. superset architecture decision, and (c) implementation considerations + failure modes for whatever path is chosen.

Audience: alter ego picking this up with full codebase context to actually build.

---

## 1. Vulkan perf gap: findings

### Symptom

- GL: vsync-stable 60 fps (~16.7 ms/frame), GPU work async during vsync
- Vulkan: 30 fps (32 ms/frame), chaos game GPU work alone takes 27 ms
- Earlier baseline was 12.7 fps; jumped to 27.2 fps after rendering at physical pixel density (4.7MP virtual canvas) instead of native pixel density (17MP). That's a 3.6x architectural win; remaining ~2x gap is what this investigation drilled into.

### Hypothesis 1 (atomic-add contention): ruled out

Both backends emit identical hardware atomic ISA on the trimmed shader:
```
Vulkan: send(16) ... ugm MsgDesc: ( atomic_inc, a32, d32, V1, L1UC_L3WB ... ) opcode 0x64040508
OpenGL: send(16) ... ugm MsgDesc: ( atomic_inc, a32, d32, V1, L1UC_L3WB ... ) opcode 0x64040508
```

Trimmed-shader stats:
| | Vulkan | OpenGL | Δ |
|---|---|---|---|
| Cycles | 7066 | 6760 | +4.5% |
| Instructions | 76 | 73 | +4% |
| SIMD width | 16 | 16 | = |
| Spills | 0 | 0 | = |
| Sends | 8 | 6 | +2 |

4.5% gap is not 2x. Atomic-add pattern isn't the problem.

### Hypothesis 2 (compiler optimization differs on full shader): CONFIRMED

Full flame_chaos.comp comparison:
| Shader | Backend | Instructions | Cycles | Spills:Fills |
|---|---|---|---|---|
| trimmed | Vulkan SIMD16 | 76 | 7,066 | 0:0 |
| trimmed | OpenGL SIMD16 | 73 | 6,760 | 0:0 |
| FULL flame_chaos | Vulkan SIMD8 | **43,377** | **61,609,696** | **1929:3326** |
| FULL flame_chaos | OpenGL SIMD8 | 15,903 | 15,684,896 | 0:0 |

- Vulkan compiles to **2.7× more instructions** for the same shader source
- Cycles estimated **3.9× more** (matches measured ~2× real-world gap; static estimate is upper-bound)
- **5,255 register spill/fill operations on Vulkan vs zero on GL**

Root cause: Mesa's Vulkan compiler doesn't optimize the universal 100+ variation switch(var_idx) dispatcher as aggressively as the GL compiler. The big switch produces massive register pressure that GL can fit in registers via optimization but Vulkan can't, causing spills that dominate runtime.

This is a real Mesa driver/compiler quality issue — same shader source, same target hardware, dramatically different codegen quality.

---

## 2. Proposed fix: per-genome shader specialization (with superset variant)

### Core idea

Instead of one universal shader containing all 100+ variations, compile a custom shader per genome (or per group of genomes) containing **only the variations actually used**. Cache compiled pipelines, prefetch opportunistically.

Two implementations possible:
- **Per-genome**: one shader per unique variation set. Maximum specialization, max cache entries.
- **Supersets**: shaders containing slightly-larger variation sets that cover multiple genomes' usage. Fewer shaders total, slightly more register pressure per shader.

The architecture choice depends on Mesa-Xe Vulkan's register-spill threshold (the empirical unknown).

### Population measurements (2026-05-30)

From `library.db`, querying non-archived genomes:

- **3796 non-archived genomes**
- **2206 distinct unique-variation-sets** (cache keys for per-genome)
- **1.72 genomes per pipeline** average reuse
- **76.6% of sets used by exactly 1 genome**, 23.4% shared by 2+
- **Mean 15 unique variations per genome, median 12, max 54**

Long tail: ~150 genomes use 30+ variations, ~25 use 40+. One outlier (**genome 3063**) uses 54 unique variations. The outliers cluster on specific dates (2026-05-23 and 2026-05-25) suggesting they're emergent from extended mutation sessions without weight-pruning. See deferred items.

### Set-cover analysis (greedy, with size limit)

Trade-off between max superset size and number of supersets needed:

| Max shader size | Supersets | Oversized (need own shader) | Total shaders | vs per-genome |
|---|---|---|---|---|
| 20 | 446 | 655 | ~1101 | 50% reduction |
| 25 | 415 | 445 | ~860 | 61% reduction |
| 30 | 367 | 284 | ~651 | 70% reduction |
| 40 | 276 | 67 | ~343 | 84% reduction |
| **50** | **149** | **1** | **150** | **93% reduction** |
| 75 | 14 | 0 | 14 | 99.4% reduction |

The 50→75 jump is dramatic — strong clustering at the higher size limits. The Top 10 supersets at max=20 only cover 20.6% of genomes (long tail, not power-law dominated).

### Compile-budget implications

Assuming 1-2 seconds per pipeline compile (Mesa Vulkan, complex shader):

| Architecture | Pipelines | Cold-start compile time |
|---|---|---|
| Per-genome | 2206 | 36-73 min |
| Supersets max=50 | 150 | 2.5-5 min |
| Supersets max=30 | 651 | 10-22 min |
| Supersets max=20 | 1101 | 18-37 min |

---

## 3. Empirical next step: measure register-spill threshold

The architecture decision hinges on one measurement we don't have:

**At what shader complexity (variations active in the switch) does Mesa-Xe Vulkan start spilling registers?**

What we know:
- 100+ variations: spills heavily (5255 spill/fill ops, confirmed)
- ~15 variations (typical specialized): probably fine (untested)
- Threshold somewhere in between

Suggested test:
- Build flame_chaos.comp variants with switch dispatchers containing 15, 20, 30, 40, 50, 75 active variations
- Compile each, measure instructions / cycles / spill count
- Find the knee where spill count starts climbing
- That defines the safe `max_size` for supersets

Once threshold known:
- If threshold ≥ 50: supersets at max=50 → **150 total shaders, 5 min cold compile**
- If 30-50: supersets at threshold → 200-650 total shaders
- If 20-30: supersets less helpful; per-genome closer to right answer
- If <20: per-genome is the only viable approach

---

## 4. Implementation considerations

### Cache design

- **Cache key**: variation-set hash (sorted unique indices), NOT genome ID. Two genomes with the same variation set share a pipeline.
- **Be conservative with cache key**: include anything that affects codegen. If parameter values affect compile output (e.g., parametric variations), include them in key. Conservative under-sharing > silent wrong-output bugs.
- **Cache storage**: in-memory dict + disk-backed via `VkPipelineCache` serialization. Persist on shutdown, load on startup.
- **LRU eviction** beyond memory budget (suggest 500MB ceiling; 100-300KB per pipeline).

### Prefetch strategy

- **Normal mode**: opportunistic flood from current location in genome graph. N-hop neighborhood (suggest N=2 starting, tune empirically). Compile during idle.
- **Compare mode**: explicit pre-select next pair when current pair displays. Compile both genomes' pipelines during user review window (3-10s typical). Selector picks most informative pair regardless of cache state; pre-compile handles the cache.
- **Queue priority**: current (cached) > genome_next (immediate need) > 1-hop neighbors > 2+ hop > rest.

### Compile coordination

- Background compile thread (or worker process) consumes prefetch queue.
- Yields to render and other realtime work (until proper GPU scheduler exists, manual heuristic: don't compile during active render).
- Thread-safe cache structure with in-progress tracking (no double-compiling same key).

### Hot reload

- If variations.glsl source changes, invalidate cache. Include source hash in cache key as cheap solution.
- For dev workflow: source-change detection triggers cache flush.

### Compare-mode specifics

- Pre-selected pair must be committed before pre-compile starts (no speculative compile of pair candidates).
- Could pre-compile 2-3 pairs deep if user pace warrants.
- The cache layer should NOT influence pair selection — selector picks for information value, cache mechanism handles availability.

---

## 5. Failure modes to plan for

### High impact

1. **Compile time per pipeline much higher than estimated.** If actual compile is 5s+ instead of 1-2s, prefetch can't keep ahead. Need to measure first specialized compile before committing to architecture. Could need worker-pool parallelism.

2. **Cache key wrong → silent wrong output.** Most insidious failure. If cache hits give wrong rendering, looks like genome variation, hard to detect via inspection. Golden-master tests at variation level might not catch cross-genome cache-sharing bugs. Conservative cache key (over-include params, source hash, etc.) preferred.

3. **Mesa Vulkan compiler instability under load.** Mesa-Xe Vulkan is experimental. Compiling 150-2000 pipelines could expose driver bugs unseen so far. Need recovery path: compile failure → fall back to universal shader for that genome + log issue.

### Medium impact

4. **Async race conditions.** Cache lookups during compile, genome changes during in-progress compile. Standard concurrency hazards, addressable via in-progress tracking + thread-safe cache.

5. **First-render-after-compile still slow.** Pipeline compile isn't the only first-use cost; drivers sometimes have additional warmup (instruction cache, descriptor population). Worth measuring after pre-compile lands to confirm stutter is actually gone.

6. **Variation set explosion during evolution.** New mutations create new variation sets, low cache hit rate during active breeding. Mitigated by population stabilization over time.

7. **No GPU scheduler yet → compile competes with render.** Without proper scheduler, background compile during render time can cause frame drops. Need manual priority heuristic: only compile during idle/transition windows.

### Low impact

8. Pipeline cache serialization bugs in Mesa-Xe Vulkan (cache version-stamping, fall back to recompile on load failure)
9. Cache memory unbounded growth (LRU eviction)
10. Hardware-specific cache invalidation (include driver version + GPU PCI ID in validity stamp)

---

## 6. Deferred / follow-up items

### Variation-weight pruning (post-Vulkan, real work)

The 54-variation outlier and the long tail in general are downstream of mutation accumulating low-weight variations without pruning. A simple flat threshold (e.g., prune weights < 0.01) WOULDN'T work — variations have different sensitivities to weight perturbations; some are visually delicate at low weights, others fade smoothly. Also, the identity variation isn't necessarily at index 0, so naive pruning could strip the identity.

A correct sensitivity-aware approach would need:
- Per-variation sensitivity calibration (probably empirical — perturb weight, measure histogram-distance change, derive sensitivity)
- Threshold on `weight × sensitivity` (actual contribution magnitude)
- Identity handled explicitly per-genome

This is a real piece of work, not a one-line tweak. Deferred until after Vulkan port.

Effect when done: collapses long-tail variation-set diversity, reduces the 2206 distinct unique-sets to something smaller, improves caching, reduces shader complexity uniformly across the population.

### Mesa bug report (parallel low-effort)

The disassembly comparison data (trimmed vs full shader, 5255 spills vs 0) is exactly the artifact a Mesa bug report should include. Low effort given data is already gathered. Long-term win: if Mesa fixes the optimization, per-genome specialization becomes less load-bearing. Doesn't block current work; file in parallel.

### GPU scheduler integration

The 3-tier (realtime/preemptable/batch) GPU scheduler design lives in memory as a future project. Background pipeline compile is a natural "batch" tier consumer. Until that scheduler exists, use manual heuristic (compile during idle windows, yield to render).

### Validation strategy for the port

After specialization is implemented:
- Per-variation golden-master tests already cover correctness (the existing 1670 tests + multi-param-set goldens). New shaders compiled per-genome must pass same goldens.
- Cross-genome cache-sharing risk: ensure cache key truly captures all codegen-affecting state. Possibly add a test that renders the same genome via per-genome shader vs universal shader and asserts byte-equivalence.

---

## TL;DR for the implementation

1. **Build per-genome shader specialization with variation-set-hash cache key.** Implementation skeleton: cache, prefetch queue, compile coordination.
2. **Measure the Mesa-Xe Vulkan register-spill threshold** (shaders at 15/20/30/40/50/75 vars). This determines whether to use per-genome (one shader per unique set) or supersets (consolidated shaders covering multiple genomes).
3. **If threshold ≥ 40**: implement supersets via greedy set-cover. Massive consolidation, fast warmup.
4. **If threshold < 30**: stay with per-genome. 36-73 min cold compile is real but tolerable as one-time tax (disk cache amortizes).
5. **Wire prefetch** (opportunistic flood for normal mode, explicit pre-select for compare mode).
6. **Plan recovery path** for compile failures (fall back to universal shader, log).
7. **Defer** variation-weight pruning (sensitivity-aware work, post-Vulkan).
8. **File Mesa bug report** in parallel with implementation (low effort).

---

## Data sources

- Mesa-Xe Vulkan vs GL disassembly: `tools/vk_perf_diag/dump_vk.txt`, `tools/vk_perf_diag/dump_gl.txt`
- Population measurement queries: see this brief's section 2. Run against `~/.local/share/flame-sheep/library.db`, `genomes` table, `variation_signature` column. Dedupe comma-separated indices to unique sets per genome.
- Set-cover analysis: greedy algorithm sorted by genome usage desc, then size desc. Process each set: direct cover if subset of existing superset, extend existing superset if would fit within max_size, else new superset.
