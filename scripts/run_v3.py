#!/usr/bin/env python3
"""Run deterministic pure-Rust V3 mechanism, graph, coverage, monitor and selection suites."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/release/authority-lab"

SOURCE_PATTERNS = (
    "src/**/*.rs",
    "policy-core/**/*.rs",
    "scripts/*.py",
    "scripts/*.sh",
    "tests/*.rs",
    "tests/*.py",
    "fixtures/v2/*.json",
    ".github/workflows/*.yml",
)
EXPLICIT_SOURCES = (
    "Cargo.toml",
    "Cargo.lock",
    "policy-core/Cargo.toml",
    "Makefile",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_source_hashes():
    hashes = {}
    for pattern in SOURCE_PATTERNS:
        for f in sorted(ROOT.glob(pattern)):
            rel = str(f.relative_to(ROOT))
            assert not rel.startswith(("docs/", "evidence/")) and rel != "README.md", rel
            hashes[rel] = digest(f)
    for name in EXPLICIT_SOURCES:
        hashes[name] = digest(ROOT / name)
    return dict(sorted(hashes.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--quick", action="store_true", help="run smoke/quick sample counts")
    p.add_argument("--profile", choices=["smoke", "publication"], default=None)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--no-build", action="store_true")
    args = p.parse_args()

    if args.quick and args.profile == "publication":
        raise SystemExit("cannot combine --quick with --profile publication")
    profile = "smoke" if (args.quick or args.profile in (None, "smoke")) else "publication"

    os.chdir(ROOT)
    env = os.environ.copy()
    env.setdefault("CARGO_INCREMENTAL", "0")
    env.setdefault("CARGO_PROFILE_DEV_DEBUG", "line-tables-only")
    env.setdefault("CARGO_BUILD_JOBS", "2")

    if not args.no_build:
        subprocess.run(["cargo", "build", "--locked", "--release"], check=True, env=env)

    cmd = [
        str(BINARY),
        "run-v3",
        str(args.output),
        "--profile",
        profile,
        "--seed",
        str(args.seed),
    ]
    subprocess.run(cmd, check=True, env=env)

    manifest = {
        "schema_version": 3,
        "profile": profile,
        "quick": profile == "smoke",
        "seed": args.seed,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True).strip(),
        "cargo": subprocess.check_output(["cargo", "-V"], text=True).strip(),
        "binary_sha256": digest(BINARY),
        "gpu_used": False,
        "gpu_seconds": 0,
        "command": ["authority-lab", "run-v3", str(args.output), "--profile", profile, "--seed", str(args.seed)],
        "source_sha256": collect_source_hashes(),
    }
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("v3 evidence:", args.output)


if __name__ == "__main__":
    main()
