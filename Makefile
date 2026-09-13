# actoris_harena -- task runner for the shared rig pipeline.
#
# The simulated arenas are NOT covered here: their tests are imperative
# __main__ scripts under test/, run individually, and they need the [sim]
# extra's dependency stack. These targets cover the rig pipeline, which is what
# the robot repos import.
#
# THIS PACKAGE HAS NO VENV OF ITS OWN, and that is the whole design: it is
# installed into each robot repo's environment, because a rig's venv is what
# pins its LeRobot checkout and its hardware libraries. So these targets need an
# interpreter that has `actoris_harena[rig]` installed, and the sensible one is
# a rig's:
#
#     make test-unit PY=../so101_garment/venv/bin/python
#
# With no PY given they look for a sibling rig's venv and use the first that can
# import the package. If none can, they say so rather than failing with
# thirty-odd ModuleNotFoundErrors about numpy.

# Candidate interpreters, in order: whatever PY says, then any sibling repo's
# venv. `wildcard` expands only to paths that exist, so a machine with one rig
# checked out tries exactly one.
CANDIDATES := $(PY) $(wildcard ../*/venv/bin/python)
# The first candidate that can import the package AND its dependencies.
#
# Testing `import actoris_harena` alone proves nothing, which cost a confusing
# ten minutes: the package __init__ is lazy by design and imports only os and
# pathlib, so it succeeds under any interpreter run from this directory -- the
# directory itself is importable. The deps are what actually distinguish an
# environment that can run the tests from one that cannot.
_PROBE := import numpy, aiohttp, cv2, yaml, actoris_harena
PYTHON := $(firstword $(foreach p,$(CANDIDATES),\
    $(shell $(p) -c "$(_PROBE)" >/dev/null 2>&1 && echo $(p))))

.PHONY: test test-unit test-integration lint check-python

check-python:
	@if [ -z "$(PYTHON)" ]; then \
	  echo "❌ no interpreter found with actoris_harena installed."; \
	  echo ""; \
	  echo "   This package installs into a ROBOT REPO's venv, not its own."; \
	  echo "   Install it into one, then point these targets at it:"; \
	  echo ""; \
	  echo "       <rig>/venv/bin/pip install -e '$(CURDIR)[rig]'"; \
	  echo "       make test-unit PY=<rig>/venv/bin/python"; \
	  echo ""; \
	  exit 1; \
	fi

test: test-unit test-integration

test-unit: check-python
	@echo "using $(PYTHON)"
	@$(PYTHON) -m unittest discover -s test/rig -t .

# Slower: builds real models and loads real checkpoints. The one that matters
# most is test_policy_ports_checkpoints, which compares a port against its
# upstream twin bit for bit -- that is what makes "the module tree MOVED, it was
# not rewritten" a checkable claim rather than an intention.
test-integration: check-python
	@$(PYTHON) -m unittest discover -s test/integration -t .

# black/isort/flake8/mypy, scoped to the rig pipeline -- see the comment at the
# head of .pre-commit-config.yaml for why the sim tree is excluded. Run through
# the same interpreter, since pre-commit is installed in a rig's venv too.
lint: check-python
	@$(PYTHON) -m pre_commit run --all-files
