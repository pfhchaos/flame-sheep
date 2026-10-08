PYTHON := ./.venv/bin/python

.DEFAULT_GOAL := test
.PHONY: test test-all

# Fast default suite. pyproject.toml's addopts already applies -m 'not slow',
# so this skips the GPU/hardware/pure-Python-chaos-game tests gated with
# @pytest.mark.slow and finishes in well under a minute.
test:
	$(PYTHON) -m pytest

# Full suite, slow tests included. -m "" overrides the ini's "-m 'not slow'"
# (the last -m on the command line wins over addopts) so nothing is deselected.
test-all:
	$(PYTHON) -m pytest -m ""
