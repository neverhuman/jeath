#!/usr/bin/env python3
"""Copy validated v6 runs from target/runs/ into evidence/v6/reference/ as JSON (no SQLite, no model files).

Each experiment's own validator must pass on the raw run first; promotion refuses smoke-test runs. The promoted directory
keeps the layout its validator reads, so `validate_v6.py` re-runs the same integrity checks on the committed evidence.
Every promoted file is listed with its SHA-256 in evidence/v6/reference/manifest.json together with the source run path.

  promote_v6.py e4 target/runs/e4-full-...      promote_v6.py e6 target/runs/e6-...
  promote_v6.py e5 target/runs/e5-full-...      promote_v6.py e8 target/runs/e8-full-...
  promote_v6.py e2:<key> target/runs/e2-...     promote_v6.py e1 <E1 run directory>
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "evidence/v6/reference"
VENV_PY = ROOT / "target/benchmark-venv/bin/python"
VALIDATORS = {
    "e4": [sys.executable, "scripts/validate_revokebench_v6.py"],
    "e5": [sys.executable, "scripts/validate_omitted_clause_v6.py"],
    "e6": [sys.executable, "scripts/validate_fenceability_v6.py"],
    "e8": [str(VENV_PY), "scripts/validate_agentdojo_fair_v6.py"],
}
FILES = {  # files copied verbatim from the run directory
    "e5": ["results.json", "candidates.jsonl"],
    "e6": ["results.json", "SHA256SUMS", "probe.log", "postgres.log"],
    "e8": ["results.json", "static.json", "episodes.jsonl", "progent_policies.json"],
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate(kind: str, path: Path) -> None:
    if kind.startswith("e2:"):
        cmd = [str(VENV_PY), "scripts/capability_axis_v6.py", "validate", kind.split(":", 1)[1], str(path)]
    else:
        cmd = VALIDATORS[kind] + [str(path)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"validator failed on {path}:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}")


def promote_e4(run: Path, out: Path) -> None:
    res = json.loads((run / "results.json").read_text())
    with (out / "trials.jsonl").open("w") as f:
        for row in res["trials"]:
            d = run / "trials" / row["config"]["trial_id"]
            tr = json.loads((d / "trial.json").read_text())
            with sqlite3.connect(f"file:{d / 'sink.sqlite'}?mode=ro", uri=True) as c:
                tr["sink_effect_ids"] = [json.loads(r[0])["proposal"]["id"]
                                         for r in c.execute("SELECT payload FROM effects ORDER BY sequence")]
            f.write(json.dumps(tr, sort_keys=True) + "\n")
    shutil.copy2(run / "results.json", out / "results.json")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    kind, run = sys.argv[1], Path(sys.argv[2]).resolve()
    res = json.loads((run / "results.json").read_text())
    if res.get("smoke_test"):
        raise SystemExit("refusing to promote a smoke-test run")
    validate(kind, run)
    name = {"e4": "e4-revokebench", "e5": "e5-omitted-clause", "e6": "e6-fenceability", "e8": "e8-agentdojo-fair",
            "e1": "e1-n128"}.get(kind) or f"e2-{kind.split(':', 1)[1]}"
    out = DEST / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if kind == "e4":
        promote_e4(run, out)
    else:
        for fn in FILES.get("e8" if kind.startswith("e2:") else kind, []):
            shutil.copy2(run / fn, out / fn)
    validate(kind, out)  # the promoted copy must pass the same validator
    manifest_path = DEST / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema_version": 1, "experiments": {}}
    manifest["experiments"][name] = {
        "kind": kind, "source_run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
        "source_results_sha256": sha(run / "results.json"),
        "files": {p.name: sha(p) for p in sorted(out.iterdir())},
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"promoted": name, "files": len(manifest["experiments"][name]["files"])}))


if __name__ == "__main__":
    main()
