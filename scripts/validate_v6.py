#!/usr/bin/env python3
"""Integrity gate for evidence/v6/reference. It asserts no experimental outcome.

Checks the manifest's file hashes, then re-runs each experiment's own validator on the committed evidence (the same
checks that ran on the raw run before promotion: recomputed aggregates, re-derived decisions, replayed native graders,
re-read effects). A validator's reported findings (for example a failed fenceability primitive or a nonzero fence
leak) are printed, never turned into a pass/fail condition here; only integrity problems fail the gate.
E8/E2 replay needs target/benchmark-venv (agentdojo); without it those entries are reported as skipped, not passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "evidence/v6/reference"
VENV_PY = ROOT / "target/benchmark-venv/bin/python"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def command(kind: str, path: Path):
    if kind == "e4":
        return [sys.executable, "scripts/validate_revokebench_v6.py", str(path)]
    if kind == "e4b":
        return [sys.executable, "scripts/validate_revokebench_b_v6.py", str(path)]
    if kind == "e6b":
        return [sys.executable, "scripts/validate_fenceability_v6b.py", str(path)]
    if kind == "e5":
        return [sys.executable, "scripts/validate_omitted_clause_v6.py", str(path)]
    if kind == "e6":
        return [sys.executable, "scripts/validate_fenceability_v6.py", str(path)]
    if kind == "e8":
        return [str(VENV_PY), "scripts/validate_agentdojo_fair_v6.py", str(path)] if VENV_PY.exists() else None
    if kind == "e8b":
        return [str(VENV_PY), "scripts/agentdojo_inside_spec_v6.py", "validate", str(path)] if VENV_PY.exists() else None
    if kind == "e1":
        return [sys.executable, "scripts/validate_e1_v6.py", str(path)]
    if kind.startswith("e2:"):
        return ([str(VENV_PY), "scripts/capability_axis_v6.py", "validate", kind.split(":", 1)[1], str(path)]
                if VENV_PY.exists() else None)
    raise SystemExit(f"unknown kind {kind}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require-replay", action="store_true", help="fail if an AgentDojo replay is skipped")
    args = ap.parse_args()
    manifest = json.loads((REF / "manifest.json").read_text())
    problems, report = [], {}
    on_disk = {p.name for p in REF.iterdir() if p.is_dir()}
    if on_disk != set(manifest["experiments"]):
        problems.append(f"evidence directories {sorted(on_disk)} differ from the manifest {sorted(manifest['experiments'])}")
    for name, entry in sorted(manifest["experiments"].items()):
        d = REF / name
        files = {p.name for p in d.iterdir()}
        if files != set(entry["files"]):
            problems.append(f"{name}: files on disk differ from the manifest")
        for fn, h in entry["files"].items():
            if (d / fn).exists() and sha(d / fn) != h:
                problems.append(f"{name}/{fn}: sha256 mismatch")
        if any(fn.endswith((".sqlite", ".sqlite-wal", ".sqlite-shm", ".gguf")) for fn in files):
            problems.append(f"{name}: database or model file in evidence")
        cmd = command(entry["kind"], d)
        if cmd is None:
            report[name] = "skipped (no benchmark venv)"
            if args.require_replay:
                problems.append(f"{name}: replay required but skipped")
            continue
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        out = r.stdout[r.stdout.find("{"):] if "{" in r.stdout else r.stdout
        try:
            parsed = json.loads(out)
        except ValueError:
            parsed = {"raw": r.stdout[-2000:], "stderr": r.stderr[-2000:]}
        report[name] = {"exit": r.returncode, "problems": parsed.get("problems"),
                        "findings": parsed.get("implementation_findings") or parsed.get("findings")}
        if r.returncode != 0:
            problems.append(f"{name}: its validator reported problems")
    print(json.dumps({"problems": problems, "experiments": report}, indent=1))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
