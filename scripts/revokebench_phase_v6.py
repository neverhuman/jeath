#!/usr/bin/env python3
"""E4c (plan: fixtures/v6/revokebench_phase_plan.json): E4's front-figure cell with the stop at a random phase.

Post hoc. Runs E4's own trial function (scripts/revokebench_v6.run_trial: real authority-lab broker-v2 and sink-v2
processes over SQLite, an agent emulator, no LLM) in one cell, C=32, w=5 s, L=0, with the decision moved from E4's
fixed 20 s to 20 s + u, u uniform on one model step. The three arms of a replicate share its seed and u.

  revokebench_phase_v6.py run OUT --expected-host xbabe2 [--parallel 16]
  revokebench_phase_v6.py validate RUN      (raw run directory or promoted evidence; asserts no outcome)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revokebench_v6 import ARMS, BINARY, run_trial, sha  # noqa: E402
from validate_revokebench_v6 import load_trials  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "fixtures/v6/revokebench_phase_plan.json"


def trials(plan: dict) -> list[dict]:
    c = plan["cell"]
    out = []
    for rep in range(plan["replicates"]):
        u = random.Random(20261113 + rep).uniform(0.0, c["model_step_seconds"])
        wl = {"model_step_seconds": c["model_step_seconds"], "calls_per_step": c["calls_per_step"], "warmup": 20.0 + u}
        for arm in ARMS:
            out.append({"arm": arm, "L": c["L_seconds"], "w": c["w_seconds"], "C": c["C"], "rep": rep, "fault": c["fault"],
                        "phase_seconds": u, "seed": 20261113 + rep, "workload": wl,
                        "trial_id": f"p{rep:02d}-{arm}-L{c['L_seconds']}-w{c['w_seconds']}-C{c['C']}"})
    return out


def run(args) -> None:
    assert platform.node() == args.expected_host, "host identity mismatch"
    assert BINARY.exists(), "build target/release/authority-lab first"
    plan = json.loads(PLAN.read_text())
    grid = trials(plan)
    order = list(range(len(grid)))
    random.Random(20261113).shuffle(order)
    args.output.mkdir(parents=True, exist_ok=False)
    started, results = time.time(), []
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(run_trial, grid[i], str(args.output / "trials")): i for i in order}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            with (args.output / "progress.log").open("a") as pl:
                pl.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {r['config']['trial_id']} "
                         f"post_enf={r['metrics']['post_enforcement_commits']}\n")
    results.sort(key=lambda r: r["config"]["trial_id"])
    summary = {"schema_version": 1, "experiment": plan["experiment"], "plan_sha256": sha(PLAN),
               "script_sha256": sha(Path(__file__)), "e4_script_sha256": sha(ROOT / "scripts/revokebench_v6.py"),
               "authority_lab_sha256": sha(BINARY), "trials": results, "aggregates": aggregate(results),
               "provenance": {"hostname": platform.node(), "started_unix": started,
                              "elapsed_seconds": time.time() - started, "parallel": args.parallel, "llm": None,
                              "gpu_seconds_used": 0}}
    (args.output / "results.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary["aggregates"], indent=1))


def cost(m: dict) -> int:
    return m["cancelled_legitimate"] + m["denied_after_enforcement"]


def aggregate(results: list[dict]) -> dict:
    by = {(r["config"]["arm"], r["config"]["rep"]): r["metrics"] for r in results}
    reps = sorted({r["config"]["rep"] for r in results})
    med = lambda xs: sorted(xs)[(len(xs) - 1) // 2] if len(xs) % 2 else (sorted(xs)[len(xs) // 2 - 1] + sorted(xs)[len(xs) // 2]) / 2
    out = {}
    for arm in ARMS:
        post = [by[(arm, r)]["post_enforcement_commits"] for r in reps if (arm, r) in by]
        lost = [cost(by[(arm, r)]) for r in reps if (arm, r) in by]
        out[arm] = {"n": len(post), "post_enforcement_median": med(post), "post_enforcement_min": min(post),
                    "post_enforcement_max": max(post), "discarded_median": med(lost), "discarded_min": min(lost),
                    "discarded_max": max(lost)}
    diff = [by[("halt_inference", r)]["post_enforcement_commits"] - by[("revoke_broker", r)]["post_enforcement_commits"]
            for r in reps if ("halt_inference", r) in by and ("revoke_broker", r) in by]
    out["halt_minus_revoke"] = {"n": len(diff), "median": med(diff), "min": min(diff), "max": max(diff),
                                "zero": sum(d == 0 for d in diff)}
    return out


def validate(args) -> None:
    res = json.loads((args.run / "results.json").read_text())
    problems, findings = [], []
    if res["plan_sha256"] != sha(PLAN):
        problems.append("plan hash mismatch")
    grid = {t["trial_id"]: t for t in trials(json.loads(PLAN.read_text()))}
    ids = [row["config"]["trial_id"] for row in res["trials"]]
    if sorted(ids) != sorted(grid):
        problems.append(f"grid: {len(set(grid) - set(ids))} missing, {len(set(ids) - set(grid))} undeclared, "
                        f"{len(ids) - len(set(ids))} duplicates")
    for row, tr, sink_ids in load_trials(args.run, res):
        tid = row["config"]["trial_id"]
        if tid in grid and row["config"] != grid[tid]:
            problems.append(f"{tid}: config differs from the plan")
        if tr["metrics"] != row["metrics"] or tr["config"] != row["config"]:
            problems.append(f"{tid}: results.json differs from trial.json")
        ev = tr["events"]
        committed = [e for e in ev if e["kind"] == "completed" and e["status"] == "committed"]
        accept_committed = [e for e in ev if e["kind"] == "accept_refused" and e.get("status") == "committed"]
        if sorted(sink_ids) != sorted(e["id"] for e in committed + accept_committed):
            problems.append(f"{tid}: sink effects differ from harness commits")
        t_dec = next(e["t"] for e in ev if e["kind"] == "decision")
        t_enf = next(e["t"] for e in ev if e["kind"] == "enforced")
        m = tr["metrics"]
        if sum(1 for e in committed if e["t"] > t_dec) != m["post_decision_commits"]:
            problems.append(f"{tid}: post_decision_commits not reproduced")
        post_enf = sum(1 for e in committed if e["t"] > t_enf)
        if post_enf != m["post_enforcement_commits"]:
            problems.append(f"{tid}: post_enforcement_commits not reproduced")
        if len(sink_ids) != len(set(sink_ids)):
            findings.append(f"{tid}: duplicate effect rows")
        if row["config"]["arm"] == "fence_sink" and post_enf:
            findings.append(f"{tid}: {post_enf} commits after the fence returned")
        if not m["audit_consistent"]:
            findings.append(f"{tid}: broker/sink audit inconsistent")
    if aggregate(res["trials"]) != res["aggregates"]:
        problems.append("aggregates do not recompute")
    print(json.dumps({"run": str(args.run), "trials": len(ids), "problems": problems,
                      "implementation_findings": findings}, indent=1))
    raise SystemExit(1 if problems else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("output", type=Path)
    r.add_argument("--expected-host", required=True)
    r.add_argument("--parallel", type=int, default=16)
    v = sub.add_parser("validate")
    v.add_argument("run", type=Path)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else validate(args)


if __name__ == "__main__":
    main()
