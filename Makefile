# Checks, from fastest to slowest. `make check` must pass before every commit; `make check-docker`
# before calling a change done. See AGENTS.md.

export PYTHONPATH := src
IMAGE = $(shell uv run --quiet python scripts/versions.py image)
FIXTURE ?= tests/fixtures/input_item.json
# `make run` runs one product without Docker, on a staged item and an AOI.
WORKFLOW ?= eodh-ndvi
STAC_ITEM_DIR ?= tests/fixtures/golden/l2a/staged
AOI_FILE ?= tests/fixtures/golden/aoi.geojson

.PHONY: install lint format test validate-cwl check fixture run image check-docker bump versions helpers-checksum clean

install:  ## Install dependencies and the pre-commit hooks
	uv sync
	uv run pre-commit install

lint:  ## Level 1: ruff lint and format check
	uv run ruff check .
	uv run ruff format --check .

format:  ## Fix lint and formatting where ruff can
	uv run ruff check --fix .
	uv run ruff format .

test:  ## Levels 2 and 3: unit tests and the offline end-to-end run
	uv run pytest

validate-cwl:  ## Level 4: cwltool --validate on every CWL file
	for cwl in cwl/*.cwl; do uv run cwltool --validate "$$cwl" || exit 1; done

check: lint test validate-cwl  ## Levels 1-4
	uv run python scripts/versions.py check

fixture:  ## Write a synthetic stage-in directory for FIXTURE to build/staged
	rm -rf build/staged
	uv run python -m eodh_workflow stage $(FIXTURE) build/staged

run:  ## Run WORKFLOW on STAC_ITEM_DIR and AOI_FILE, without Docker, into build/out
	rm -rf build/out
	uv run python src/main.py --stac_item_dir=$(STAC_ITEM_DIR) --output_dir=build/out \
		--workflow=$(WORKFLOW) --aoi="$$(cat $(AOI_FILE))"
	uv run python -m eodh_workflow validate build/out

image:  ## Build the linux/amd64 image with the tag in the CWL dockerPull
	docker buildx build --platform linux/amd64 --load -t $(IMAGE) .

check-docker: image  ## Level 5: run each CWL in the image on its cwl/<name>.job.json fixture, and validate
	rm -rf build/cwl-out
	for cwl in cwl/*.cwl; do \
		job="$${cwl%.cwl}.job.json"; \
		test -f "$$job" || { echo "$$cwl has no fixture $$job" >&2; exit 1; }; \
		id="$$(uv run --quiet python scripts/versions.py workflow-id "$$cwl")" || exit 1; \
		out="build/cwl-out/$$id"; \
		uv run cwltool --outdir "$$out" "$$cwl#$$id" "$$job" || exit 1; \
		uv run python -m eodh_workflow validate "$$(find "$$out" -mindepth 1 -maxdepth 1 -type d | head -n 1)" || exit 1; \
	done

bump:  ## Set the workflow version everywhere: make bump VERSION=x.y.z
	@test -n "$(VERSION)" || (echo "usage: make bump VERSION=x.y.z" && exit 1)
	uv run python scripts/versions.py bump $(VERSION)
	uv lock

versions:  ## Show the workflow version in each place it is set
	uv run python scripts/versions.py check

helpers-checksum:  ## Template maintainers only: record the checksum of src/eodh_workflow/
	uv run python -m eodh_workflow checksum --write

clean:
	rm -rf build .pytest_cache .ruff_cache

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'
