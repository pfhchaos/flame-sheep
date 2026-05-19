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

## Generation boundary trigger

Generations end when there's enough new training data to justify retraining. Candidate formula:

```
trigger_score = (thumbs_up_this_gen × thumbs_down_this_gen) / N
              + (pairwise_count_now - pairwise_count_at_last_generation_start)
```

The thumbs term requires balanced ratings (need both classes for synthesis to produce pairs). The pairwise term captures new comparison information directly. Threshold tuned empirically — start with "enough to noticeably move val accuracy."

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

## Implementation notes (deferred)

- Migration adds the `generation` column to both ratings tables, backfills existing data to gen 1.
- New rating handlers (`_handle_like`, `_handle_dislike`, compare-mode `_record`) read current_generation from metadata at rating time.
- Trainer reads current_generation at training start, uses it in the synthesis query.
- Generation boundary logic (trigger formula, atomic counter increment, breeding pipeline) wires into the evolution scheduler.
- UI affordance: maybe a soft "new generation" indicator in the wallpaper, so the user knows the population has evolved. Optional.

## Open questions

- **Sub-generation feedback loops.** Could rate → train → re-score same population → rate more, without a full breeding cycle. Gives faster model iteration but blurs the generation boundary. Decide later whether the trigger threshold should produce a "mini-generation" (retrain only) or "full generation" (retrain + breed + prune).
- **Decay weighting for old pairwise.** Pairs from 10 generations ago are still valid but might be less useful (population moved on, your taste may have refined). Could weight pairs by `1 / (1 + age_in_generations)` in the loss. Not strictly necessary — let the model figure it out.
- **Generation 0 / bootstrap.** New users have zero generations of data. Default behavior: ship with the gen-1 model + a small starter population. The user's first ratings are stamped gen 1 (same generation as the shipped model's training corpus). When they accumulate enough new data to trigger retraining, gen 2 begins — the first model they train themselves.

## Status

- v3 pairwise-only model deployed as the implicit gen 1.
- Active-set filter for thumbs validity shipped (alter ego, 2026-05-18).
- Generation column + intra-gen synthesis not yet implemented.
- Generation trigger formula not yet implemented.
