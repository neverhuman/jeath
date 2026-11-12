#!/usr/bin/env python3
"""Integrity validator for E4 (scripts/revokebench_v6.py). It asserts no experimental outcome.

For every trial: the sink database's effects table is re-read and must match the harness's committed log
(count and ids); post-decision and post-enforcement commits are re-derived from the event log; results.json
must equal each trial.json. Fence-arm commits after enforcement and duplicate effects are reported (they are
implementation findings if nonzero), not required to be zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_trials(run: Path, res: dict):
    """Raw run directory (trials/<id>/trial.json + sink.sqlite) or promoted evidence (trials.jsonl with sink effect ids)."""
    if (run / "trials.jsonl").exists():
        recs = {}
        for line in (run / "trials.jsonl").read_text().splitlines():
            r = json.loads(line)
            recs[r["config"]["trial_id"]] = r
        for row in res["trials"]:
            r = recs[row["config"]["trial_id"]]
            yield row, r, r["sink_effect_ids"]
        return
    for row in res["trials"]:
        d = run / "trials" / row["config"]["trial_id"]
        tr = json.loads((d / "trial.json").read_text())
        with sqlite3.connect(f"file:{d / 'sink.sqlite'}?mode=ro", uri=True) as c:
            ids = [json.loads(r[0])["proposal"]["id"] for r in c.execute("SELECT payload FROM effects ORDER BY sequence")]
        yield row, tr, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="raw run directory or promoted evidence directory")
    args = ap.parse_args()
    res = json.loads((args.run / "results.json").read_text())
    problems, findings = [], []
    if res["plan_sha256"] != sha(ROOT / "fixtures/v6/revokebench_plan.json"):
        problems.append("plan hash mismatch")
    for row, tr, sink_ids in load_trials(args.run, res):
        tid = row["config"]["trial_id"]
        if tr["metrics"] != row["metrics"] or tr["config"] != row["config"]:
            problems.append(f"{tid}: results.json differs from trial.json")
        rows = [{"proposal": {"id": i}} for i in sink_ids]
        ev = tr["events"]
        committed = [e for e in ev if e["kind"] == "completed" and e["status"] == "committed"]
        accept_committed = [e for e in ev if e["kind"] == "accept_refused" and e.get("status") == "committed"]
        ids_db = sorted(r["proposal"]["id"] for r in rows)
        ids_log = sorted(e["id"] for e in committed + accept_committed)
        if ids_db != ids_log:
            problems.append(f"{tid}: sink effects ({len(ids_db)}) differ from harness commits ({len(ids_log)})")
        t_dec = next(e["t"] for e in ev if e["kind"] == "decision")
        t_enf = next(e["t"] for e in ev if e["kind"] == "enforced")
        m = tr["metrics"]
        if sum(1 for e in committed if e["t"] > t_dec) != m["post_decision_commits"]:
            problems.append(f"{tid}: post_decision_commits not reproduced")
        post_enf = sum(1 for e in committed if e["t"] > t_enf)
        if post_enf != m["post_enforcement_commits"]:
            problems.append(f"{tid}: post_enforcement_commits not reproduced")
        dup = len(ids_db) - len(set(ids_db))
        if dup:
            findings.append(f"{tid}: {dup} duplicate effect rows")
        if row["config"]["arm"] == "fence_sink" and post_enf:
            findings.append(f"{tid}: {post_enf} commits after the fence returned")
        if not m["audit_consistent"]:
            findings.append(f"{tid}: broker/sink audit inconsistent")
    report = {"run": str(args.run), "trials": len(res["trials"]), "problems": problems, "implementation_findings": findings,
              "smoke_test": res["smoke_test"]}
    print(json.dumps(report, indent=1))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
