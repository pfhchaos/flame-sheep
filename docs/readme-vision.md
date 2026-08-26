# README — Aspirational Version

For when the feature list isn't a lie anymore.

---

```markdown
# Flame-sheep

A wallpaper.

## Features

- Audio-reactive
- Multi-channel CNN aesthetic scoring with personal preference learning
- Curriculum-learned multi-model committee for active learning
- Genetic algorithm with axis-weighted evolution
- Real-time genome graph navigation
- General-purpose user-space GPU work scheduler
- Multi-process architecture with hot-reloadable models
- Octave-resistant beat-tracking via hierarchical CNN
- Distributed taste-sharing protocol via model weight exchange
- Renders fractals.
```

---

## Feature status (2026-05-28)

Status of each line, since most are aspirational right now:

| Feature | Status |
|---|---|
| Audio-reactive | ✅ shipped |
| Multi-channel CNN aesthetic scoring | ✅ shipped (deployed model) |
| Personal preference learning | ✅ shipped (compare-mode + pairwise training) |
| Curriculum-learned multi-model committee | ⏳ partial — curriculum model exists, committee architecture designed not built |
| Genetic algorithm | ✅ shipped |
| Axis-weighted evolution | ❌ designed, not built |
| Real-time genome graph navigation | ✅ shipped |
| General-purpose user-space GPU scheduler | ❌ designed, not built (flame-sheep is first-user; framework is a separate project) |
| Multi-process architecture | ✅ shipped |
| Hot-reloadable models | ✅ shipped |
| Octave-resistant beat-tracking via hierarchical CNN | ⏳ partial — flat 3-head model in training, hierarchical multi-depth is a later experiment |
| Distributed taste-sharing protocol | ❌ designed, not built |
| Renders fractals | ✅ shipped (always was) |

The README opener stays gated on getting the feature list to "honest" before publishing.

## Why this format

- "A wallpaper" + escalating feature list + deadpan close ("Renders fractals") is funnier than direct description
- Linux cultural convention — `cat`, `ls`, `vim` etc. all use trivial descriptions for sophisticated tools
- The comedic tension makes the actual scope more memorable than direct description
- Self-selecting filter for the right contributors (people who get the joke fit the project culture)
- Preserves the original motivation ("I wanted a pretty wallpaper that responds to music") as the defining story; the rest is consequences of doing that honestly

## When to deploy

When the feature list is no longer a lie. Probably:
- After Vulkan port unblocks 8b/9b stages
- After GPU scheduler framework lands as standalone project
- After multi-model committee active learning is wired up
- After beat-RNN BPM extraction replaces the heavy tempo detector

At that point the README can be the opener above and the project's external presentation matches its actual capability. Until then it would be misrepresenting the state.
