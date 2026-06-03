# Working discipline for evaluated changes

Three months of iteration on beat detection produced this pattern:

1. Make a change that feels right.
2. Tune it by ear on a few tracks.
3. Deploy it.
4. Find out weeks later it's worse than the baseline.
5. Goto 1.

The unified eval framework, deploy gate, and scorecard exist to break
step 4. Discipline below is the social/process layer that keeps them
load-bearing.

## The four rules

**1. No measurable change ships without a scorecard row.**

If you're changing anything that affects `beat.osu.*`, `tempo.*`, or
any other registered metric — run the scorecard before and after.
Append a row with `tools/eval_all.py --notes "..."`. The before-row
might already exist; if so, point to it in the notes. The after-row
records the effect.

Counter to discipline: "I tweaked thresholds and it feels better."
Always run the scorecard. The "feels better" might be confirmation
bias on one track.

**2. Negative results get the same recording treatment as wins.**

When an experiment doesn't pan out:
- Record the scorecard row (with `--notes "tried X, didn't work"`).
- Write a memory note that captures the *shape* of why, not just the
  numbers. Future-you needs to know whether to try the same thing again.
- Keep the code if it's reusable infrastructure (tempo_reconcile.py is
  the example). Discard if it's a dead-end one-off.

Counter to discipline: deleting failed experiments without trace.
The 2026-06-03 reconciliation work is retained precisely because
"why we tried it and why it didn't work" is more valuable than the
code itself.

**3. One independent change per measurement.**

If you change three things at once and the metric moves +0.02, you
don't know which change contributed. Sometimes the right move is to
land them as a single commit (when they're a coherent unit), but the
*measurements* between them should isolate each.

Counter to discipline: bundling unrelated improvements into one
scorecard row because each one is small. Run them separately. If
that's expensive, automate so it stops being expensive.

**4. Hypothesis written down before measuring.**

For any new detector variant, write what you expect to happen
*before* running the eval. Three sentences max:
- What you changed.
- What you predict the effect will be on which metric.
- What would falsify the prediction.

Then run the eval. The honesty-with-yourself test is whether your
prediction was right. If it's consistently wrong, your model of the
system needs work — not your eval.

Counter to discipline: looking at the result first and then post-
rationalizing why it makes sense.

## What the framework enforces

- `tools/eval_all.py` writes one JSONL row per run with commit hash
  + dirty flag + notes. The scorecard is append-only.
- `tools/check_beat_gate.py` exits non-zero if a new beat detector
  underperforms the recorded percentile baseline by more than the
  tolerance. Run before swapping anything into `audio.toml`.
- `evals/results.jsonl` lives in git, so the metric trajectory is
  part of the project history alongside the code that produced it.

## What the framework doesn't enforce (yet)

- Pre-commit hook that runs the gate on weight-file changes.
- Automatic memory-note creation for negative results.
- "You shipped a thing without a scorecard row" pre-push check.

These are friction-level improvements; the rules above are the
substance. Add the hooks when the manual discipline breaks down.

## When the discipline pays off

Most iterations don't need it. The discipline is for the cases where
"this feels right" diverges from "the metric agrees" — which is
exactly where bias and confirmation hit hardest. Three beat-RNN
failures this month, all of which felt right when shipped, are the
existence proof.
