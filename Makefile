.PHONY: required test experiment paper paper-safe-ai paper-brain performance-before-outcomes minds-we-choose
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

paper-safe-ai: paper

paper-brain: minds-we-choose

performance-before-outcomes:
	python3 scripts/validate.py evidence/reference --study performance-before-outcomes
	python3 scripts/paper.py --study performance-before-outcomes
	python3 scripts/build_paper.py --study performance-before-outcomes

minds-we-choose:
	python3 scripts/validate.py evidence/brain/reference --study minds-we-choose
	python3 scripts/paper.py --study minds-we-choose
	python3 scripts/build_paper.py --study minds-we-choose
