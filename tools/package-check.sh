#!/usr/bin/env bash
# package-check — prove the flame-sheep cluster still PACKAGES, installing
# nothing onto the system. The one resident copy stays the editable dev venv;
# this is a throwaway check for packaging bit-rot, not a deploy.
#
# Why this exists: the four repos (flame-sheep + the three it depends on) each
# ship a pyproject.toml and a git-r3 ebuild. Day-to-day dev runs out of the
# editable venv, so nothing exercises the *packaged* path until an emerge
# months later fails on rot nobody noticed. This gives that path a fast local
# dry-run and a real-ebuild release gate.
#
# Two tiers, matching the run-vs-release split:
#
#   (A) default     per-package wheel build from the CURRENT local tree.
#                   `pip wheel --no-deps` each repo: exercises the build
#                   backend, pyproject metadata, and package discovery — the
#                   whole "does this still produce a distribution" surface —
#                   without resolving or downloading any dependency. Fast, no
#                   root, run it on commit / pre-push.
#                   NB: --no-deps means this does NOT validate that the private
#                   inter-cluster deps (flame-sheep -> flame-sheep-audio etc.)
#                   resolve. Builds, not resolution. Resolution is (B)'s job.
#
#   (B) --ebuild    `emerge --buildpkgonly` each Gentoo atom: builds the real
#                   binpkg through the actual ebuild (RDEPEND resolution and
#                   all) but MERGES NOTHING onto the system. The release gate.
#                   The ebuilds are git-r3 (they fetch from GitHub), so this
#                   validates the PUSHED state — run it AFTER pushing, not
#                   against uncommitted local work.
#
# Usage:
#   tools/package-check.sh            # (A) local wheel build-check
#   tools/package-check.sh --ebuild   # (B) ebuild build-check (post-push)
#
# Exit status = number of packages that failed (0 = all clean).
set -uo pipefail

# Leaf deps first, flame-sheep last — only matters for readable output ordering
# (each package is built independently; there's no install step to order).
#   repo-subdir-under-$PROJECTS | gentoo atom
PACKAGES=(
  "flame-sheep-audio|dev-python/flame-sheep-audio"
  "viz-authoring|dev-python/viz-authoring"
  "wallpaper-ml|dev-python/wallpaper-ml"
  "flame-sheep|gui-apps/flame-sheep"
)

PROJECTS="${PROJECTS_DIR:-$HOME/projects}"
PY="${PYTHON:-python3}"

mode="local"
case "${1:-}" in
  "")        mode="local"  ;;
  --ebuild)  mode="ebuild" ;;
  -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *)         echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
esac

fails=0

for entry in "${PACKAGES[@]}"; do
  IFS='|' read -r sub atom <<<"$entry"
  repo="$PROJECTS/$sub"
  name="${atom##*/}"

  if [[ "$mode" == "local" ]]; then
    printf '== %-20s wheel build == ' "$name"
    if [[ ! -f "$repo/pyproject.toml" && ! -f "$repo/setup.py" ]]; then
      echo "SKIP (no build config at $repo)"
      fails=$((fails + 1))
      continue
    fi
    out="$(mktemp -d)"
    if "$PY" -m pip wheel --quiet --no-deps --wheel-dir "$out" "$repo" >/dev/null 2>&1; then
      whl="$(ls "$out"/*.whl 2>/dev/null | head -n1)"
      echo "OK  ($(basename "${whl:-?}"))"
    else
      echo "FAIL"
      echo "       repro: $PY -m pip wheel --no-deps -w /tmp/wh $repo"
      fails=$((fails + 1))
    fi
    rm -rf "$out"

  else # ebuild
    printf '== %-20s ebuild build == ' "$name"
    if sudo emerge --buildpkgonly --oneshot --quiet-build=y "$atom" >/dev/null 2>&1; then
      echo "OK"
    else
      echo "FAIL"
      echo "       repro: sudo emerge --buildpkgonly $atom"
      fails=$((fails + 1))
    fi
  fi
done

echo
total="${#PACKAGES[@]}"
if [[ "$fails" -eq 0 ]]; then
  echo "OK: all $total packages build ($mode)"
else
  echo "FAIL: $fails/$total failed ($mode)"
fi
exit "$fails"
