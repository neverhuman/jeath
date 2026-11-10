#!/usr/bin/env python3
"""Install hash-pinned Verus under target, then verify the broker's exact core."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TAG = "0.2026.11.02.8bf68ee"
SHA = "f3a0119292df8855e4435da8130bee1873b0e6a60c306accf81a08ea687bba3c"


def verify(output=None):
    directory = ROOT / "target/tooling"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "verus.zip"
    if not archive.exists():
        urllib.request.urlretrieve(f"https://github.com/verus-lang/verus/releases/download/release/{TAG}/verus-{TAG}-x86-linux.zip", archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == SHA, "Verus archive hash mismatch"
    tool = directory / "verus-x86-linux/verus"
    if not tool.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(directory)
    for name in ("verus", "rust_verify", "cargo-verus", "z3"):
        (tool.parent / name).chmod(0o755)
    subprocess.run(["rustup", "toolchain", "install", "1.99.1", "--profile", "minimal"], check=True)
    env = os.environ.copy()
    env["PATH"] = str(tool.parent) + os.pathsep + env["PATH"]
    direct = subprocess.run([str(tool),"--crate-type","lib","policy-core/src/lib.rs","--triggers-mode","silent"],
                            cwd=ROOT,capture_output=True,text=True)
    command = ["cargo", "verus", "verify", "-p", "policy-core", "--locked"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    report = {"schema_version": 2, "tool": TAG, "archive_sha256": SHA,
              "core_sha256": hashlib.sha256((ROOT / "policy-core/src/lib.rs").read_bytes()).hexdigest(),
              "command": command, "returncode": direct.returncode or result.returncode,
              "output": direct.stdout + direct.stderr + result.stdout + result.stderr,
              "trusted": ["Verus, SMT solver and Rust compiler", "parser and normalized payload construction", "storage snapshots, issuer identity, session launcher", "Linux kernel and cooperative sink contracts"]}
    if output:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(report["output"])
    if report["returncode"]:
        raise SystemExit(report["returncode"])
    return report


if __name__ == "__main__":
    import sys
    verify(sys.argv[1] if len(sys.argv) > 1 else None)
