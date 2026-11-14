# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

# Use the project's virtualenv when it exists; override with `make PYTHON=...`.
PYTHON ?= $(if $(wildcard venv/bin/python),venv/bin/python,$(if $(wildcard .venv/bin/python),.venv/bin/python,python3))

# Files/directories checked for the license header.
LICENSE_PATHS ?= src tests scripts

.PHONY: help license-check license-add test

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

license-check:  ## Check that every file has the license header
	$(PYTHON) scripts/check_license.py $(LICENSE_PATHS)

license-add:  ## Add the license header to the files missing it
	$(PYTHON) scripts/check_license.py --fix $(LICENSE_PATHS)

test:  ## Run the test suite
	$(PYTHON) -m pytest tests
