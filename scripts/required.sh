#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
export CARGO_PROFILE_DEV_DEBUG="${CARGO_PROFILE_DEV_DEBUG:-line-tables-only}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace --all-targets
python3 -m unittest discover -s tests -p 'test_*.py'
run_path="target/gate/$(date -u +%Y%m%dT%H%M%SZ)-$$"
python3 scripts/run.py "$run_path" --quick
python3 scripts/validate.py "$run_path"
python3 scripts/validate.py evidence/reference --source-ref 12b5579
if [[ "$(uname -s)" == Linux ]]; then
    python3 scripts/run_v2.py "$run_path-v2"
    python3 scripts/analyze_v2.py "$run_path-v2" --source
fi
python3 scripts/analyze_v2.py evidence/v2/reference --source
python3 scripts/run_v3.py "$run_path-v3" --quick
python3 scripts/validate_v3.py "$run_path-v3" --source
python3 scripts/validate_v3.py evidence/v3/reference --source
python3 scripts/validate_v6.py
python3 scripts/paper.py --check
python3 scripts/build_paper.py
python3 scripts/order_references.py target/paper-build/intelligence-is-not-authority.aux --check
