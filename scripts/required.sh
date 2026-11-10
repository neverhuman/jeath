#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
export CARGO_PROFILE_DEV_DEBUG="${CARGO_PROFILE_DEV_DEBUG:-line-tables-only}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
run_path="target/gate/$(date -u +%Y%m%dT%H%M%SZ)-$$"
python3 scripts/run.py "$run_path" --quick
python3 scripts/validate.py "$run_path"
python3 scripts/validate.py evidence/reference --source
python3 scripts/paper.py --check
python3 scripts/build_paper.py

