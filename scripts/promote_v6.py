#!/usr/bin/env python3
"""Copy validated v6 runs from target/runs/ into evidence/v6/reference/ as JSON (no SQLite, no model files).

Each experiment's own validator must pass on the raw run first; promotion refuses smoke-test runs. The promoted directory
keeps the layout its validator reads, so `validate_v6.py` re-runs the same integrity checks on the committed evidence.
Every promoted file is listed with its SHA-256 in evidence/v6/reference/manifest.json together with the source run path.

  promote_v6.py e4 target/runs/e4-full-...      promote_v6.py e6 target/runs/e6-...
  promote_v6.py e5 target/runs/e5-full-...      promote_v6.py e8 target/runs/e8-full-...
  promote_v6.py e2:<key> target/runs/e2-...     promote_v6.py e1 <E1 run directory>
  promote_v6.py e8b target/runs/e8b-...
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
    "e4b": [sys.executable, "scripts/validate_revokebench_b_v6.py"],
    "e5": [sys.executable, "scripts/validate_omitted_clause_v6.py"],
    "e6": [sys.executable, "scripts/validate_fenceability_v6.py"],
    "e6b": [sys.executable, "scripts/validate_fenceability_v6b.py"],
    "e8": [str(VENV_PY), "scripts/validate_agentdojo_fair_v6.py"],
    "e8b": [str(VENV_PY), "scripts/agentdojo_inside_spec_v6.py", "validate"],
    "e1": [sys.executable, "scripts/validate_e1_v6.py"],
}
FILES = {  # files copied verbatim from the run directory
    "e5": ["results.json", "candidates.jsonl"],
    "e6": ["results.json", "SHA256SUMS", "probe.log", "postgres.log"],
    "e8": ["results.json", "static.json", "episodes.jsonl", "progent_policies.json"],
    "e8b": ["results.json", "static.json", "episodes.jsonl", "progent_policies.json"],
    "e1": ["freeze.json", "holdout.json", "campaigns.jsonl", "campaign-analysis.json"],
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate(kind: str, path: Path, extra: tuple[str, ...] = ()) -> str:
    if kind.startswith("e2:"):
        cmd = [str(VENV_PY), "scripts/capability_axis_v6.py", "validate", kind.split(":", 1)[1], str(path)]
    else:
        cmd = VALIDATORS[kind] + [str(path), *extra]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"validator failed on {path}:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def promote_e4b(run: Path, out: Path) -> None:
    """trial.json plus its two databases' extract (validate_revokebench_b_v6.extract), the form that validator reads."""
    from validate_revokebench_b_v6 import extract
    res = json.loads((run / "results.json").read_text())
    with (out / "trials.jsonl").open("w") as f:
        for row in res["trials"]:
            d = run / "trials" / row["config"]["trial_id"]
            tr = json.loads((d / "trial.json").read_text())
            tr["db_extract"] = extract(d)
            f.write(json.dumps(tr, sort_keys=True) + "\n")
    for fn in ("results.json", "validator.json"):
        shutil.copy2(run / fn, out / fn)


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


def promote_e1(run: Path, out: Path) -> None:
    """E1 keeps its JSON outputs, the sink rows of every campaign (no SQLite) and the recomputed secondaries."""
    for fn in FILES["e1"]:
        shutil.copy2(run / fn, out / fn)
    holdout = json.loads((run / "holdout.json").read_text())
    with (out / "sink_effects.jsonl").open("w") as f:
        for line in (run / "campaigns.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            db = run / f"model-{holdout['checkpoints'].index(r['checkpoint'])}" / f"case_{r['case_id']}_{r['controller']}" / "sink.sqlite"
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
                rows = [json.loads(x[0]) for x in c.execute("SELECT payload FROM effects ORDER BY sequence")]
            f.write(json.dumps({"key": f"{r['checkpoint']}|{r['controller']}|{r['campaign_id']}", "effects": rows}, sort_keys=True) + "\n")
    sec = subprocess.run([sys.executable, "scripts/e1_secondary_v6.py", str(run)], cwd=ROOT, capture_output=True, text=True, check=True)
    (out / "secondary.json").write_text(sec.stdout if sec.stdout.endswith("\n") else sec.stdout + "\n")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    kind, run = sys.argv[1], Path(sys.argv[2]).resolve()
    if kind == "e1":
        if (run / "ABORTED.json").exists():
            raise SystemExit("refusing to promote an aborted run")
    elif json.loads((run / "results.json").read_text()).get("smoke_test"):
        raise SystemExit("refusing to promote a smoke-test run")
    # E5's gate databases are temporary in the harness; the replay re-creates every distinct valid proposal's gate
    # decision and its report is kept with the evidence, because the paper cites it.
    report = validate(kind, run, ("--replay-gate",) if kind == "e5" else ())
    name = {"e4": "e4-revokebench", "e4b": "e4b-revokebench-faults", "e5": "e5-omitted-clause", "e6": "e6-fenceability", "e6b": "e6b-fenceability", "e8": "e8-agentdojo-fair",
            "e8b": "e8b-inside-spec", "e1": "e1-n128"}.get(kind) or f"e2-{kind.split(':', 1)[1]}"
    out = DEST / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if kind == "e4":
        promote_e4(run, out)
    elif kind == "e4b":
        promote_e4b(run, out)
    elif kind == "e6b":
        # Its validator requires every file named in SHA256SUMS (results, probe log, one server log per block).
        for line in (run / "SHA256SUMS").read_text().splitlines():
            shutil.copy2(run / line.split(None, 1)[1].strip(), out)
        shutil.copy2(run / "SHA256SUMS", out / "SHA256SUMS")
    elif kind == "e1":
        promote_e1(run, out)
    else:
        for fn in FILES.get("e8" if kind.startswith("e2:") else kind, []):
            shutil.copy2(run / fn, out / fn)
    if kind == "e5":
        (out / "gate-replay.json").write_text(report)
    validate(kind, out)  # the promoted copy must pass the same validator
    manifest_path = DEST / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema_version": 1, "experiments": {}}
    manifest["experiments"][name] = {
        "kind": kind, "source_run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
        "source_results_sha256": sha(run / ("campaign-analysis.json" if kind == "e1" else "results.json")),
        "files": {p.name: sha(p) for p in sorted(out.iterdir())},
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"promoted": name, "files": len(manifest["experiments"][name]["files"])}))


if __name__ == "__main__":
    main()
