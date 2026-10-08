# flame-sheep packaging plan

_Goal: install flame-sheep **natively on the Gentoo workstation via portage** — live
ebuilds (`-9999`) in the personal overlay, tracking per-package git repos. **Not** PyPI
distribution. Secondary first-class goal: **make running the test suite trivial.**_

Decisions below are settled (2026-10-07) unless marked **OPEN**. Status: `[ ]` todo ·
`[~]` in progress · `[x]` done.

---

## Target shape — multi-repo, one ebuild each

The monorepo gets **split into separate repos** (each independently-packaged thing →
its own repo → its own live ebuild tracking it via `git-r3`). This turns
`wallpaper_ml`/`viz_authoring` from path-deps into real external deps.

| Package | Repo (split out) | Ebuild | Backend |
|---|---|---|---|
| `flame_sheep` (main daemon) | stays the `flame-sheep` repo | `gui-apps/flame-sheep-9999` | **hatchling** |
| `flame_sheep_audio` (+ native `prtcqt`) | extract → own repo | `dev-python/flame-sheep-audio-9999` | **scikit-build-core** |
| `wallpaper_ml` | extract → own repo | `dev-python/wallpaper-ml-9999` | hatchling |
| `viz_authoring` | extract → own repo | `dev-python/viz-authoring-9999` | hatchling |

- All `-9999` live ebuilds: `inherit distutils-r1 git-r3` (+ `systemd` where a unit
  ships), `KEYWORDS=""`, accepted with `**` in `package.accept_keywords`.
- **Category `gui-apps`** for the daemon (documented: "applications across various WMs/gui
  backends"; precedent `gui-apps/swaybg`, and your own `gui-apps/glpaper`). The libs are
  `dev-python/*`.
- **The audio systemd user unit ships in `flame_sheep_audio`** (it runs that daemon).

## Acceptance criteria

1. `emerge flame-sheep` installs the daemon + deps + the systemd user unit.
2. The daemon runs from the **installed** package, not the repo `.venv`.
3. **Tests run with one obvious command** — no src-layout shadow, no fiddly invocation.
4. A **fresh checkout builds the native extension** (no pre-built `.so` needed).
5. Config/state honor `$XDG_*` (see 0.7).

---

## THE BLOCKER (before any ebuild): native extensions build from a clean checkout

- **`prtcqt`** (vendored rt-cqt + bundled pffft + pybind-style binding) — **mandatory**.
  Upstream `jmerkt/rt-cqt` has no releases + WIP Python bindings, so vendoring stays.
  **Build it via `scikit-build-core`**, NOT the old `python-r1 + cmake` hand-compile glue
  (that btrack idiom is obsolete for our own source). Give `flame_sheep_audio` a real
  `pyproject.toml` with `build-backend = "scikit_build_core.build"`, `pybind11` in
  `build-system.requires`; vendored rt-cqt + pffft compile inside the CMake target.
  (Template: pybind/scikit_build_example; in-tree: `dev-python/mapbox-earcut`.)
- **`_btrack`** — **externalize.** Upstream now ships `btrack-beat-tracker` on PyPI
  (1.0.6+) and your overlay has `dev-python/btrack-beat-tracker-1.0.7`. RDEPEND on it,
  drop the in-repo fork (`tempo_btrack.py` already prefers the system one).

---

## Phase 0 — repo prep (do in the monorepo first, split last)

- [x] **0.1 Native build** — `flame_sheep_audio/pyproject.toml` → `scikit-build-core`
      backend + CMakeLists that builds `prtcqt` (rt-cqt + pffft + binding). A clean
      `pip install .` must compile it. Wire `_btrack` to the overlay package.
- [x] **0.2 Declare real deps** — pyproject declares `moderngl, moderngl-window, numpy,
      scipy, sounddevice` but code also imports **`pywayland`, `Pillow`, `dbus-python`,
      `PyGObject`(gi)/GLib, `cffi`, `glcontext`** — add them.
- [x] **0.3 Package-data** — declare `flame_sheep/rendering/shaders/*` + `default_config.toml`
      (would be dropped from a built dist today). `wallpaper_ml/shaders/*` already declared.
- [x] **0.4 Canonical model weights** — `.npz/.npy` in `flame_sheep/data/` are gitignored
      with `.backup_*` copies; pick the canonical set, `git add -f`, declare as package-data.
- [x] **0.5 Constrain flat layout** — `where=["."]` sweeps `docs/scripts/tools/datasets`
      into the package; add an explicit allowlist (or move `flame_sheep` under `src/`).
      **Also kills the pytest namespace-shadow → half of criterion #3.**
- [x] **0.6 Vendor + de-hardcode the systemd unit** (into `flame_sheep_audio`) — add a
      `flame-sheep-audio` console script (`flame_sheep_audio.daemon:main`), rewrite
      `ExecStart` to it, ship the unit (`data/systemd/…`), install via `systemd_douserunit`.
- [x] **0.7 XDG paths** — runtime/IPC already respect `$XDG_RUNTIME_DIR`; **config + data
      are hardcoded to `~/.config` / `~/.local/share`** (ignore `$XDG_CONFIG_HOME` /
      `$XDG_DATA_HOME`) across ~10 files. Centralize in one module via **`platformdirs`**.
- [ ] **0.8 Test-ease** — add `pytest` to a `[test]` extra so the `pytest` entry point
      works (avoids `python -m pytest` CWD-prepend shadow) + a `make test`. With 0.5 this
      satisfies criterion #3.
- [ ] **0.9 Repo split** — extract `flame_sheep_audio` / `wallpaper_ml` / `viz_authoring`
      into their own repos with history preserved (`git filter-repo`). Do this AFTER the
      above so each repo carries the cleaned state. (The ebuilds need the repos to exist.)

## Phase 1 — the ebuilds

- [ ] **1.1** Finalize each `pyproject.toml` (backend, deps, `[project.scripts]`, data).
- [ ] **1.2** `dev-python/flame-sheep-audio-9999` (the CMake-extension one) — skeleton:
      ```bash
      EAPI=8
      DISTUTILS_EXT=1
      DISTUTILS_USE_PEP517=scikit-build-core
      PYTHON_COMPAT=( python3_{12..13} )
      inherit distutils-r1 git-r3 systemd      # NOT cmake — scikit-build-core pulls it
      EGIT_REPO_URI="<flame-sheep-audio repo>"
      KEYWORDS=""
      BDEPEND="dev-python/pybind11[${PYTHON_USEDEP}]"
      RDEPEND="dev-python/numpy dev-python/scipy dev-python/sounddevice
               dev-python/btrack-beat-tracker media-libs/portaudio ..."
      distutils_enable_tests pytest
      # src_install(): systemd_douserunit the audio unit
      ```
- [ ] **1.3** `gui-apps/flame-sheep-9999` (pure-Python daemon) — skeleton:
      ```bash
      EAPI=8
      DISTUTILS_USE_PEP517=hatchling
      PYTHON_COMPAT=( python3_{12..13} )
      inherit distutils-r1 git-r3
      EGIT_REPO_URI="<flame-sheep repo>"
      KEYWORDS=""
      RDEPEND="~dev-python/flame-sheep-audio-${PV}[${PYTHON_USEDEP}]
               dev-python/wallpaper-ml dev-python/viz-authoring
               dev-python/moderngl dev-python/pillow dev-python/dbus-python
               dev-python/pygobject dev-libs/wayland media-libs/libglvnd ..."
      distutils_enable_tests pytest
      # pkg_postinst(): ewarn — needs a wlr-layer-shell compositor (sway/hyprland/river/
      #   wayfire); will NOT work on GNOME/KDE or X11.
      ```
- [ ] **1.4** `dev-python/wallpaper-ml-9999`, `dev-python/viz-authoring-9999` (hatchling).
- [ ] **1.5** Off-tree dep ebuilds — **GURU-first** (copy/enable, don't author). Already in
      overlay: `sounddevice`, `pywayland`, `btrack-beat-tracker`. Verify tree: `moderngl`,
      `moderngl-window`, `glcontext`.
- [ ] **1.6** `package.accept_keywords` (`** ` for each `-9999`) + `emerge` end-to-end.

## RDEPEND vs warning

- **Hard RDEPEND:** the libs — wayland, libglvnd/mesa (EGL), pipewire, portaudio, + the
  python deps.
- **Warning, not dep:** the **compositor**. No virtual for wlr-layer-shell; won't run on
  GNOME/KDE/X. `pkg_postinst` `ewarn`: needs sway/hyprland/river/wayfire/etc.

## OPEN decisions

- **Eval harness in the ebuild?** Recommend **no** — musdb/MUSDB datasets are large +
  license-encumbered; keep eval a dev-only `make eval`/repo target, not an installed
  feature. If shipped anyway: a local `eval` (or `benchmark`) IUSE, default-off,
  documented in `metadata.xml`, deps guarded `eval? ( ... )`.
- **Where the split repos live** (self-hosted / forge / local bare repos for `git-r3`).
- **prtcqt pedantry:** leave pffft bundled in the CMake target (fine for a `-9999`
  personal ebuild) vs unbundle to a system lib later.

## Notes

- Python `>=3.12`; `.so` are cpython-3.13-specific → set `PYTHON_COMPAT` to the supported
  interpreters.
- Daemon `os.execv`-restarts on SIGSEGV; assumes sway/wlr-layer-shell + EGL + pipewire.
- Install an **example** config (schema = `DEFAULTS` in `config.py` + `default_config.toml`).
- btrack overlay ebuild: once upstream's official module is confirmed, simplify it off the
  manual pybind-compile loop.
