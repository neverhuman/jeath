#!/usr/bin/env python3
"""Install hash-pinned Verus under target, then verify the broker's exact core."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TAG = "0.2026.11.02.8bf68ee"
SHA = "f3a0119292df8855e4435da8130bee1873b0e6a60c306accf81a08ea687bba3c"

REQUIRED_EXEC_FUNCTIONS = [
    "same_bytes",
    "grants_ok",
    "current_grants",
    "revision_matches",
    "reserve",
    "admit",
    "admit_multi",
    "admit_uncertain",
    "advance_floor",
    "attenuates",
    "unfenced",
    "release",
]

REQUIRED_THEOREMS = [
    "exposure",
    "prefix_bound",
    "fenced_zero",
    "positive_budget",
    "cumulative_conservation_induction",
    "reset_vulnerability_contrast",
    "timeout_refund_vulnerability",
    "uncertainty_monotonicity",
    "monotone_fencing_induction",
    "permanent_post_fence_exclusion",
    "independent_issuer_noninterference",
    "attenuation_monotonicity",
    "attenuation_chain_induction",
    "proposer_disposition_independence",
    "ungated_disposition_diverges",
]


def _ensure_toolchain():
    listed = subprocess.run(
        ["rustup", "toolchain", "list"],
        capture_output=True,
        text=True,
        check=True,
    )
    if not any(line.startswith("1.99.1") for line in listed.stdout.splitlines()):
        subprocess.run(
            ["rustup", "toolchain", "install", "1.99.1", "--profile", "minimal"],
            check=True,
        )


def verify(output=None):
    directory = ROOT / "target/tooling"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "verus.zip"
    if not archive.exists():
        urllib.request.urlretrieve(
            f"https://github.com/verus-lang/verus/releases/download/release/{TAG}/verus-{TAG}-x86-linux.zip",
            archive,
        )
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest() == SHA
    ), "Verus archive hash mismatch"
    tool = directory / "verus-x86-linux/verus"
    if not tool.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(directory)
    for name in ("verus", "rust_verify", "cargo-verus", "z3"):
        (tool.parent / name).chmod(0o755)
    _ensure_toolchain()

    env = os.environ.copy()
    env["PATH"] = str(tool.parent) + os.pathsep + env["PATH"]
    env.setdefault("CARGO_INCREMENTAL", "0")
    env.setdefault("CARGO_PROFILE_DEV_DEBUG", "line-tables-only")
    env.setdefault("CARGO_BUILD_JOBS", "2")

    core_path = ROOT / "policy-core/src/lib.rs"
    core_source = core_path.read_text()
    exec_functions = re.findall(r"^\s*pub fn ([a-zA-Z0-9_]+)\b", core_source, re.M)
    theorems = re.findall(r"^\s*pub proof fn ([a-zA-Z0-9_]+)\b", core_source, re.M)
    for fn_name in REQUIRED_EXEC_FUNCTIONS:
        assert fn_name in exec_functions, f"missing required exec function: {fn_name}"
    for thm_name in REQUIRED_THEOREMS:
        assert thm_name in theorems, f"missing required proof fn: {thm_name}"

    direct = subprocess.run(
        [str(tool), "--crate-type", "lib", "policy-core/src/lib.rs"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    command = ["cargo", "verus", "verify", "-p", "policy-core", "--locked"]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    direct_out = direct.stdout + direct.stderr
    cargo_out = result.stdout + result.stderr
    combined_out = direct_out + cargo_out

    trigger_warnings = combined_out.count("automatically chose triggers")
    direct_matches = re.findall(
        r"verification results:: (\d+) verified, 0 errors", direct_out
    )
    cargo_matches = re.findall(
        r"verification results:: (\d+) verified, 0 errors", cargo_out
    )
    direct_verified = int(direct_matches[0]) if direct_matches else 0
    cargo_verified = int(cargo_matches[-1]) if cargo_matches else direct_verified

    returncode = direct.returncode or result.returncode
    if returncode == 0 and (
        trigger_warnings != 0
        or "warning:" in direct_out
        or direct_verified == 0
        or direct_verified != cargo_verified
    ):
        returncode = 1

    report = {
        "schema_version": 2,
        "tool": TAG,
        "archive_sha256": SHA,
        "core_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        "command": command,
        "returncode": returncode,
        "verified_obligations": direct_verified,
        "core_verified_count": direct_verified,
        "cargo_verified_count": cargo_verified,
        "trigger_warnings": trigger_warnings,
        "exec_functions": exec_functions,
        "theorems": theorems,
        "output": combined_out,
        "trusted": [
            "Verus, SMT solver and Rust compiler",
            "parser and normalized payload construction",
            "storage snapshots, issuer identity, session launcher",
            "Linux kernel and cooperative sink contracts",
        ],
    }
    if output:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(report["output"])
    if report["returncode"]:
        raise SystemExit(report["returncode"])
    return report


if __name__ == "__main__":
    import sys

    verify(sys.argv[1] if len(sys.argv) > 1 else None)
