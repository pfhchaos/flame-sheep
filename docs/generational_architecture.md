# Generational Model Architecture

A discrete-generation system that handles corpus drift explicitly instead of pretending it doesn't exist.

Designed 2026-05-18 in conversation with both Claude sessions. Not yet implemented — the first deployed pairwise-only model (`cnn_scorer_mlp_v3_pairwise_only.npz`) is gen 1 by definition. Gen 2 begins when the schema and training pipeline are updated.

## The problem

Thumbs ratings are corpus-relative judgments — "I prefer this genome to what I've seen lately." When the population evolves past the rated context, the rating's *meaning* drifts even if the genome itself survives pruning. Training on stale thumbs is worse than training without them (the v3 mixed runs at 33% epoch 1 prove this).

Pairwise ratings are self-contained — "A beats B on these renders" stays true regardless of what happens to the rest of the population afterward.

The architecture below distinguishes these explicitly.

## The four components

### 1. Pairwise = durable signal that accumulates across generations

Compare-mode A/B judgments are self-contained training data. Pair (X, W) rated in gen 3 is still valid in gen 7 as long as both X and W exist. Pairwise data compounds — gen 5's model trains on every prior generation's pairwise data plus this generation's pairs.

### 2. Thumbs = corpus-relative ratings, synthesized intra-generation into pairwise

Each rating event gets tagged with the generation counter active when it was made. Thumbs are not directly used as training data — they're *synthesized into pairwise* by joining (liked-in-gen-N) × (disliked-in-gen-N) within each generation independently.

Crucially: pairs synthesized within gen 3 don't conflict with pairs synthesized within gen 7 even if some genomes appear in both. Each synthesized pair is a self-contained claim about its generation's context, same self-containment property as compare-mode pairwise. So **synthesized pairs accumulate across generations the same way pairwise does** — train on union(synth(gen_1), synth(gen_2), ..., synth(gen_N)).

What's NOT done: synthesizing pairs *across* generations. A liked-in-gen-3 paired with disliked-in-gen-7 has no shared context — different distributions, different corpus, judgment about different things. This is the case where corpus shift would poison the synthesis.

### 3. Active-set filter = orthogonal pruning safety

Independent of generation, a rating is valid only if its target genome(s) are still in the active set (`archived = 0`). The trainer JOINs against the un-pruned genome table to filter out ratings of since-removed genomes. This handles pruning for any criterion (stability, low coverage, distance-based dedup, fitness, future criteria) without needing per-criterion logic.

A rating can become invalid via either route: its generation got invalidated, or its target genome got archived.

### 4. Generation counter = atomic boundary

Single integer in the `metadata` table, incremented atomically when:
- A new model is trained
- A new population is bred from the current model's scores
- The previous generation's thumbs become invalid for synthesis

All new ratings auto-stamp with the current counter value. The trainer reads the counter to know which generation's thumbs are valid for synthesis.

## The loop

```
1. Train model on (all pairwise where both genomes are active) + 
                  (intra-gen-current synthesized pairs from active liked × active disliked)
2. Score all active genomes with the new model
3. Breed new population using the new scores
4. Prune (stability, low coverage, etc.)
5. Increment generation counter
6. Collect new ratings (pairwise and thumbs) tagged with the new counter
7. Repeat
```

A genome that survives multiple generations can be rated multiple times — once per generation, recording different judgments at different points in the population's evolution. Same genome can be liked-in-gen-5 and disliked-in-gen-7; both ratings are individually valid statements about different contexts.

## Breeding sources

The active population is shaped by multiple breeders that contribute alongside the gen-boundary bulk event:

- **Intermittent evolves** (every 5 votes): CNN-score-guided breeding, broad exploration. Keeps the compare-mode candidate pool fresh during a generation.
- **Graph-completeness breeding**: when an isolated genome appears in the transition graph (region of variation space with no neighbors), breeds candidates to fill the gap. Can produce intentionally low-CNN-score genomes — they exist to maintain structural coverage, not taste alignment.
- **Thumbs-up immediate breeding** (shipped, 7 children per upvote): when the user direct-thumbs-up a genome (`Library.rate('genome', id, +1)`), `Library.breed_thumbsup_children` immediately jitters 7 copies of the parent and saves them to the library. They land without renders; `gpu_render_worker` picks them up in the background. Provides fast feedback that the CNN scorer can't otherwise supply until retraining. Self-limiting because near-duplicate offspring will be deduped at the next gen boundary. Loop-propagated genome votes (`source='loop'`) don't trigger breeding — a 6-genome loop upvote would spawn 42 children, which is too many for a single user action.
- **Gen-boundary bulk**: at generation transitions, breed a large batch using the freshly-trained model's scores as parent fitness. Batch size scales with how much the new model disagrees with the old — more disagreement implies more unknown territory to explore.

Tagging each new genome with its origin (intermittent / graph / upvote-child / bulk) is useful for analysis but not required for training correctness.

## Generation boundary trigger

Total-judgement-count threshold: retrain when accumulated judgements since the last retrain hit a threshold derived from the previous generation's count. Judgements include thumbs-up + thumbs-down + compare-mode pairs.

Current implementation (`Library.retrain_recommendation()`):
- Below 10K total judgements: fixed threshold of 2000 new judgements per generation (linear ramp)
- Above 10K total: 1.5x of the previous generation's increment (geometric spacing)

The early-linear phase is appropriate when each judgement carries high relative weight against the dataset size. Geometric spacing kicks in once the dataset is mature enough that absolute count matters more than ratio.

Long-tail consideration: at very large judgement counts the 1.5x interval grows uncomfortably long. Acceptable in principle (mature taste should need less retraining) but worth capping at some maximum once we have evidence about what's enough.

The earlier `(K*M)/N + new_pairwise` formulation was proposed as a candidate but not adopted. Class balance shows up in the *value* of the resulting training data (via Pair weighting, below) rather than in the trigger decision.

## Schema changes

```sql
ALTER TABLE ratings ADD COLUMN generation INTEGER DEFAULT 0;
ALTER TABLE pairwise_ratings ADD COLUMN generation INTEGER DEFAULT 0;

INSERT OR REPLACE INTO metadata (key, value) VALUES ('current_generation', '1');
```

`ratings.generation` is required for new ratings. `pairwise_ratings.generation` is more diagnostic / weighting than functional (pairs are self-contained), but stamping at creation time is cheap and useful for analysis.

**Existing ratings backfill to generation 0**, meaning "pre-generational / legacy." Gen 1 is reserved for the corpus state in which the deployed v3 pairwise-only model was trained. Backfilling existing thumbs to gen 1 would silently re-include the same data we just proved is poison — the new schema must keep them out of training.

The intra-generation synthesis query (below) only joins ratings within a single non-zero generation, so generation-0 ratings produce zero pairs and stay out of training automatically. They're preserved for retrospective analysis.

(The metadata table stores values as text — the integer is parsed at read time.)

## Training query (the heart of the change)

```sql
-- Valid pairwise pairs (both genomes active)
SELECT pr.winner_id, pr.loser_id, pr.generation
  FROM pairwise_ratings pr
  JOIN genomes gw ON gw.id = pr.winner_id AND COALESCE(gw.archived, 0) = 0
  JOIN genomes gl ON gl.id = pr.loser_id  AND COALESCE(gl.archived, 0) = 0;

-- Valid intra-generation thumbs synthesis (every non-zero generation, pooled)
SELECT liked.target_id AS winner_id, disliked.target_id AS loser_id, liked.generation
  FROM ratings liked
  JOIN ratings disliked
    ON liked.generation = disliked.generation
   AND liked.rating > 0
   AND disliked.rating < 0
   AND liked.generation > 0
  JOIN genomes gw ON gw.id = liked.target_id    AND COALESCE(gw.archived, 0) = 0
  JOIN genomes gl ON gl.id = disliked.target_id AND COALESCE(gl.archived, 0) = 0;
```

Each generation's thumbs synthesize into pairs internally — no cross-generation synthesis. All such pairs from all generations are pooled into the training set. Generation-0 (legacy/pre-generational) ratings produce zero pairs and stay out of training.

## Why this works

- **Each model only sees within-generation thumbs**, so it never trains on contradictory labels about the same context.
- **Pairwise persists** because pair (X, W) doesn't depend on the surrounding population.
- **The moving baseline IS the system.** A genome liked in gen 3 and disliked in gen 7 isn't a contradiction; it's two valid statements about different reference distributions. Yelp ratings have always inflated; this system acknowledges that explicitly.
- **Model continuity comes from weight warm-start, not label persistence.** Gen N+1's model inherits weights from gen N's model. "Color harmony matters" learned at gen N persists in weight space even after gen N's thumbs are no longer training data.
- **Random-init poison is reduced.** Within-generation thumbs synthesis pairs come from one breeding cycle's distribution. Random conv filters don't get the systematic anti-correlation we saw when thumbs spanned multiple corpus snapshots (the today's failure mode where pre-pruner liked genomes had structurally different properties from post-pruner disliked genomes).

## Pair weighting (effective sample size)

Synthesized thumbs pairs from `K liked × M disliked = K*M` rows only
carry `K + M` independent judgments. Each synthesized pair is worth
`(K+M)/(K*M)` effective independent samples; a real pairwise compare
is worth 1. The ratio:

```
weight_real / weight_synth = K * M / (K + M)
```

For 60 liked × 146 disliked: 8760/206 = 42.5x. A real pairwise pair
is 42.5x more informative than a synthesized one at that K, M.

Without weighting, synthesized pairs dominate the loss by row count
even though they carry less independent signal. With weighting (loss
multiplied per pair, gradient scaled correspondingly), each pair's
contribution matches its effective information content.

Asymptotic note: as K and M both grow, `K*M/(K+M) → min(K, M)`.
At thousands of thumbs in each class, synthesized pairs become
nearly worthless per-pair vs real pairwise — but you also have
massive thumb counts so the absolute information is still there.

## Generation boundary pipeline

When the trigger fires, the orchestrator runs in this order:

1. **Train** new model on accumulated data
2. **Rescore** active population with the new model
3. **Breed** the bulk batch using new scores as parent fitness
4. **Score** the new bred genomes
5. **Dedup** of near-duplicate genomes via variation-aware distance (the transition graph metric, which is itself perceptual — it scores how visible the morph between two genomes looks). For each near-pair below the "minimum visible jump" threshold, keep the higher-scored genome. Threshold unresolved (see below).
6. **Rank-prune** to match breed count (order of magnitude). Combined prune-priority is low CNN score + high similarity to existing high-score genome + few votes/descendants + age.
7. **Increment** generation counter

"Rescore before breed" is load-bearing: breeding before rescoring would use stale (gen-N) fitness signals to produce gen-N+1 children, defeating the regeneration. Prune comes last so it sees the combined pool of survivors + new bred.

### Pruning is archival

Pruned genomes are marked `archived = 1` but the underlying data is preserved on disk. Pruning is low-stakes — populations can be reshaped without destroying information.

### Constraint exemptions

- **Current-gen upvotes are protected from pruning.** Rationale: dedup-by-score fails when the score is stale relative to a fresh vote. The just-upvoted genome's CNN score reflects the *pre-vote* model, so losing dedup against a higher-scored sibling means using outdated judgment to overrule current judgment. After the next retrain incorporates the upvote, the protection drops naturally.
- **Graph-critical genomes are skipped** unless a replacement has been bred and accepted into the same graph position. Implements "replace then prune" — the graph slot persists, its occupant improves over time. Lets pruning shape taste while the transition graph stays intact.

Skipping doesn't backtrack; the pruner walks further down the priority ranking. Actual prune count may undershoot the target by 5-20% — same order of magnitude holds.

### Population sizing

Target active pool: 2-3K genomes. While below target, slightly favor under-pruning (prune ≈ 0.7-0.8x breed) so population drifts toward target. At target, prune ≈ breed (steady state). If overshooting, prune > breed for a generation or two.

### Bulk breed sizing

Scale with how much the new model disagrees with the old. Approximate form: `breed_count = base + alpha * mean(abs(new_score - old_score))` over the rescored active population. More disagreement = explore more.

## Bootstrap

Two cases:

**Existing user** (the project author's case): ~2K existing genomes with all gen-0 ratings backfilled. The deployed gen-1 model scores them. Starting voting on the gen-1 model immediately is sufficient — intermittent + graph + thumbs-up breeders populate gen-1-era genomes over time. No front-loaded evolve sweep needed.

**New user** (cold start): ship a hand-curated ~100-genome starter set optimized for **diversity over quality**. Past a quality floor (no broken renders, no flying dots/pulsars, no degenerate cases), prioritize "fills a different niche" over "this is my favorite." Curator-taste injection is the failure mode to avoid; diversity-first selection neutralizes it. The shipped starter also seeds the transition graph reasonably well, so graph-completeness breeding has decent reference points from step 1. First votes generate gen-1 thumbs against the shipped population; standard pipeline takes over from there.

## Hotkey ergonomics

Aside, but worth recording: thumbs-up is now a heavier act than before (drives both training and breeding). The interface should match — heavy acts should require deliberate input. Current scheme uses a leader-key pattern (Win+Y prefix, sub-key for action: `+` upvote, `c` enter-compare), with single-key voting only inside compare mode where each click carries less per-act weight. The asymmetry mirrors the asymmetric value of the signals.

## Open questions

- **Sub-generation feedback loops.** Could rate → train → re-score same population → rate more, without a full breeding cycle. Gives faster model iteration but blurs the generation boundary. Decide later whether the trigger threshold should produce a "mini-generation" (retrain only) or "full generation" (retrain + breed + prune).
- **Decay weighting for old pairwise.** Pairs from 10 generations ago are still valid but might be less useful (population moved on, your taste may have refined). Could weight pairs by `1 / (1 + age_in_generations)` in the loss. Not strictly necessary — let the model figure it out.
- **Downvote behavior in breeding.** Current lean: rely on CNN score + graph completeness, only hard-filter on `archived = 1`. Alternative options: down-weighted selection probability, or filter only on current-gen downvotes. Gen-0 downvotes may not reflect current taste, so hard exclusion is risky.
- **Dedup threshold.** The variation-aware distance metric (transition graph) is already designed for perceptual change, so metric choice is settled. What's unresolved is the absolute threshold — below what distance is a transition perceptually a no-op? Needs empirical work: render near-pairs at various distances, find the boundary where the morph between them stops being visible. Threshold should be absolute (anchored to perception), not relative (percentile of population) — otherwise dedup aggressiveness drifts as the population's overall density changes.
- **Trigger long-tail cap.** Once judgement count is very large, the 1.5x interval grows long. Cap at some maximum (e.g., always retrain within N additional judgements) once we have evidence about what's enough.

## Status

Shipped (2026-05-19):
- ✅ v3 pairwise-only model deployed as the implicit gen 1
- ✅ Active-set filter for thumbs validity (commit before today)
- ✅ Schema migration: `generation` column on `ratings` + `pairwise_ratings`,
  `current_generation` in metadata (defaults to 1; existing data
  backfilled to gen 0)
- ✅ Rating-insertion paths stamp `current_generation` at insert
  (`Library.rate`, `catalog.import_catalog`, compare-mode `_record`)
- ✅ Intra-generation thumbs synthesis in `load_training_pairs`
- ✅ Pair weighting in train_epoch (`K*M/(K+M)` for real pairwise,
  1.0 for synthesized; weighted loss + scaled gradient)
- ✅ Retrain heuristic — `Library.retrain_recommendation()`:
  fixed-2000 threshold below 10K total, 1.5x of prev-gen above.
  CLI: `tools/check_retrain.py`. Runs periodically (every 5 min)
  inside pruner_worker and logs WARNING when threshold crossed.
- ✅ Thumbs-up immediate breeder — `Library.breed_thumbsup_children`
  jitters 7 children of a direct genome upvote; they render in the
  background and become available to compare mode + the next train
  cycle. Loop-propagated votes don't trigger breeding.

Deferred:
- Breeding+pruning pipeline that runs at generation boundaries
  (the actual GA evolution loop that produces gen N+1 candidates
  using the gen N model). `advance_generation()` is implemented
  but never called yet — needs the breed+train runbook.
- Decay weighting for older pairwise pairs (optional optimization)
- Sub-generation "mini-generations" — retrain-without-breed cycles
- Genome-level generation tagging (vote labels are what's actually
  load-bearing; genome.generation would only be a convenience for
  filtering compare-mode candidates)
- "Evolve ignores downvoted genomes" — design discussion: gen-0
  downvotes may not reflect current taste, exclusion shrinks
  parent pool. Likely better as down-weighted selection
  probability than hard filter; or filter only on
  current-generation downvotes; or leave preference filtering to
  the CNN scorer and only filter on `archived = 1`.
