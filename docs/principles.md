# Architectural Principles

Long-lived design wisdom from building the CNN aesthetic scorer. These outlive any specific model — they're how to think about the problem.

## Preference data must be pairwise between distributionally similar candidates

### The principle

Any preference dataset where the two compared items have different distributional properties (palette, density, coverage, sentinel fraction, etc.) will train poorly. Random conv response amplifies whatever distributional difference exists between the two sides, and that amplification is uncorrelated with user taste. The result: gradient is dominated by "the model latched onto the distributional shortcut" rather than "the model is learning preference."

In a properly-structured pairwise dataset, both items in the pair come from a similar distribution, so random conv response cancels out and only genuine preference signal remains.

### Empirical proof (2026-05-18)

Random Kaiming-init model, no training, evaluated on validation pairs:

| Pair type | Val accuracy | Description |
|-----------|-------------|-------------|
| Pairwise (compare mode A/B) | 50.4% | Both genomes selected by same scorer in same session — distributionally similar |
| Thumbs (liked-vs-disliked synthesis) | 2.4% | Liked and disliked sampled from globally different distributions |

A random model with no information should score 50% on any pair type. The 2.4% on thumbs is "97.6% wrong" — random conv response anti-correlates with user taste because random filters detect properties (sentinel count, density, etc.) that systematically differ between liked and disliked genomes.

Training has to first overcome this 47-point head start before it can learn anything. Models trained on mixed data spend their gradient budget undoing init bias instead of learning features.

### Implications for data collection

1. **All preference elicitation should be pairwise.** Even bootstrap. New users should see two genomes side-by-side and pick, not rate single genomes thumbs up/down.
2. **Pairs should be from similar distributions.** Active learning selecting pairs by score proximity is good (close-in-score pairs are close-in-distribution). Random pairs are not.
3. **Hard-negative mining matters.** Pairs where both sides "look like winners" but one is preferred — that's where real preference signal lives.

### Cross-references

- ML metric learning literature: hard-negative mining in face recognition, triplet loss variants
- Bradley-Terry pairwise ranking (vs. absolute rating scales)
- Tinder's UI design — pair-selection rather than rating, accidental product genius

## Preference data must respect the genome lifecycle

### The principle

A preference rating is only valid as long as the rated genome (or both genomes in a pair) is still in the active set. When a genome is archived — pruned for instability, low coverage, deduplication, fitness, or any other reason — its ratings become stale because they no longer compare against the current corpus.

This is broader than "corpus shift over time." Any change to the active set invalidates ratings that referenced removed genomes:
- Stability check archives a flying-dot genome → its old thumbs-up no longer means "I prefer this fractal" because it's not available anymore
- Distance-based pruning removes near-duplicates → ratings that compared against the removed twin lose context
- Score-based culling removes low-CNN-scored genomes → ratings of those genomes were against a population that no longer exists

### The implementation invariant

A thumb/rating is valid iff its target genome(s) are still in the current `WHERE archived = 0` set. Trainer JOINs against the un-pruned table to select valid training data. No version arithmetic, no "WHERE corpus >= N" guesswork. The active-set query encodes "still comparable" without needing to enumerate what "comparable" means.

### Implications

1. Ratings table doesn't need a `corpus_version` column. The implicit version is "exists in the active genome table."
2. Pruning automatically invalidates downstream ratings without manual cleanup.
3. Old ratings aren't deleted — just filtered out. If a genome is later un-archived (rare), its ratings become valid again.
4. This works for any pruning criterion present or future: stability, low coverage, distance-based, score-based, etc.

## Random initialization variance lies about model quality

### The principle

Single-seed training runs report anecdotal accuracy. Real model quality has a distribution across seeds, and the variance can be substantial — ±3-5% is normal for small models on small datasets, ±10% is possible with strong distributional structure in the data.

When the data has structural patterns that random filters amplify (see distributional-similarity principle above), the variance shifts: it's not "where does this seed land relative to a fair 50% baseline," it's "where does this seed land relative to a biased 30% baseline." The baseline being wrong makes the numbers look like training quality when they're really initialization luck.

### The diagnostic

Run the same architecture across multiple init seeds, single epoch each. If results cluster tightly (stdev < 2%) far from 50%, the data has a structural bias that random filters detect. If results spread widely, single-seed results are anecdotes.

### Implications

1. Always run 3-5 seed minimum for any accuracy claim worth comparing.
2. If random-init accuracy is far from 50%, fix the data or initialization before interpreting trained accuracy.
3. Best-of-N model selection is honest but inflates the reported number relative to expectation. Disclose N.

## Aesthetic scoring is downstream of representation

### The principle

The CNN doesn't learn "what looks good." It learns "what features in the input correlate with user preference." Whether those features are *useful* depends entirely on whether the input encoding exposes the right properties.

If your input encoding has artifacts that correlate with labels by accident (RGB green channel encoding palette index, broken swept render producing spirographs, sentinel pixels dominating channel variance), the model will latch onto the artifact. The accuracy number will look good. The model will fail to generalize when the artifact changes.

### Bug history (2026-05-11 → 2026-05-18)

Six layered bugs that each individually inflated apparent accuracy:

1. **Green channel artifact** — RGB encoding treated palette index as a real color. Model learned green-edge detection that happened to correlate with ES crowd taste. Anti-correlated with personal taste.
2. **Broken swept render** — Rotated the viewport instead of affines for 200+ days. Swept channel was a rotated copy of static, carried zero rotation information.
3. **Dead-ReLU gradient collapse** — Unstandardized inputs hit zero-mean assumption of Kaiming init, ReLU killed half the conv channels permanently.
4. **Sentinel value sign × ReLU asymmetry** — "No-hit" pixels encoded at +5 created a magnitude signal random filters amplified consistently.
5. **Corpus-fragile thumbs** — Old liked/disliked ratings became invalid as the genome population evolved past them, but they still got used as training data.
6. **Distributional-shortcut training** — Thumbs pairs had distributional differences between liked and disliked that the model learned as shortcut features instead of genuine preference signal.

Every "we're at 79.1% accuracy" finding was on inputs with at least one of these bugs active. Each bug fix revealed lower honest accuracy on cleaner data. Real quality was always lower than reported.

### Implications

1. Be suspicious of any accuracy number that "just works." Try to break it.
2. Random-init validation is a cheap diagnostic for representation problems. If random scores 30%, the model can't honestly score 80%.
3. Invest in representation before architecture. A 25K model on clean data beats a 100K model on dirty data.
4. The "deployed model is the best we've made" assumption is wrong. Re-benchmark deployed models on current data when fundamental things change.
