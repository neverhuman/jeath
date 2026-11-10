.PHONY: required test experiment paper
export CARGO_INCREMENTAL ?= 0
export CARGO_PROFILE_DEV_DEBUG ?= line-tables-only
export CARGO_BUILD_JOBS ?= 2

required:
	bash scripts/required.sh

test:
	cargo test --locked --all-targets

experiment:
	python3 scripts/run.py target/runs/$$(date -u +%Y%m%dT%H%M%SZ)

paper:
	python3 scripts/paper.py --check
	python3 scripts/build_paper.py

