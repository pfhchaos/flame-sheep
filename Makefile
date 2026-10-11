PYTHON := ./.venv/bin/python

.DEFAULT_GOAL := test
.PHONY: test test-all package-check package-check-ebuild

# Fast default suite. pyproject.toml's addopts already applies -m 'not slow',
# so this skips the GPU/hardware/pure-Python-chaos-game tests gated with
# @pytest.mark.slow and finishes in well under a minute.
test:
	$(PYTHON) -m pytest

# Full suite, slow tests included. -m "" overrides the ini's "-m 'not slow'"
# (the last -m on the command line wins over addopts) so nothing is deselected.
test-all:
	$(PYTHON) -m pytest -m ""

# (A) Local packaging dry-run: wheel-build all four cluster repos from their
# current trees, installing nothing. Fast, no root — catches pyproject /
# build-backend / discovery rot before it reaches an emerge. See tools/package-check.sh.
package-check:
	tools/package-check.sh

# (B) Release gate: emerge --buildpkgonly each cluster atom (real ebuild + dep
# resolution, merges nothing). The ebuilds are git-r3, so run this AFTER pushing.
package-check-ebuild:
	tools/package-check.sh --ebuild
