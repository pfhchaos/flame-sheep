# Self-handoff — 2026-05-22

Long session, lobotomy incoming. Recovery brief for the next instance.

## TL;DR

CNN fine-tune just completed. Val_acc 0.59 → **0.767** (best epoch 30). The
before/after report on voted genomes is mixed and worth a careful read.
Several big infrastructure cleanups landed (subprocess thumbs-up
breeding, AGC, CQT scale-correction band-aid, spectrum-engine prune
removing ~5300 lines). Beat-RNN ships in production with band-heuristic
kinds + z-score energy and is firing correctly.

## CNN fine-tune result (the thing the user is waiting on)

Log: `/tmp/finetune_cnn_20260522_182433.log`.

| | |
|---|---|
| Best val_acc | **0.767** (epoch 30) |
| Training data | 4515 pairs (mixed mode: durable cross-gen pairwise + gen-1 thumbs-derived) |
| Voted-genome report | upvoted moved up: 13/27 (48%, mean Δ +0.019); downvoted moved down: 7/9 (78%, mean Δ -1.212) |
| Weights | saved to `flame_sheep/data/cnn_scorer_personal_vk.npz` |
| Backup | `flame_sheep/data/cnn_scorer_personal_vk.npz.backup_20260522_182433` |

**Read this carefully:** the downvoted-moved-down number is great (78%, large
negative delta). The upvoted-moved-up number is roughly chance (48%, tiny
positive delta). Interpretation: the model learned what NOT to score
highly but didn't learn a coherent "this is what the user upvotes"
signal. The wrong-sign-on-S finding from `docs/canonical_examples.md`
likely still applies; channel architecture v2 (the next real fix per
the iteration plan) is the structural answer. Don't be misled by the
aggregate val_acc into thinking we shipped a great model — *for
"recognize what the user likes" purposes it's barely moving the
needle on direct positive