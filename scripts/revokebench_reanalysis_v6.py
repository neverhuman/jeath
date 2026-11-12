#!/usr/bin/env python3
"""E4 RevokeBench reanalysis (analysis only; no new data). It asserts no outcome.

Reads the promoted E4 trials (evidence/v6/reference/e4-revokebench/trials.jsonl) and the E4 plan, and reports per
cell (fault x arm x L x w x C):

  * offered load rho = lambda * w / C, with lambda = calls_per_step / model_step_seconds (the emitter's design rate,
    2 calls/s; the emitter is open loop, so lambda does not depend on w or C), and the split into stable (rho < 1)
    and overloaded (rho >= 1) cells;
  * the plan's Little's-law prediction (lambda_measured x (L + w), halt_inference only) beside post_decision_commits,
    reported only where rho < 1: where rho >= 1 the executor queue has no steady state and the prediction is not
    defined;
  * legitimate work lost = cancelled + denied. This is the plan's own definition of cancelled_legitimate ("operations
    that were authorized when emitted and were cancelled or denied by the stop"); the promoted metric
    cancelled_legitimate counted only cancellations.

Every other derived quantity is labelled post hoc in the output's `definitions`. Nothing here was predeclared except
the plan's own definitions it quotes; the reanalysis was written after the E4 data were seen, in response to a design
review.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "evidence/v6/reference/e4-revokebench/trials.jsonl"
PLAN = ROOT / "fixtures/v6/revokebench_plan.json"

DEFINITIONS = {
    "rho": "post hoc: lambda_offered * w / C with lambda_offered = calls_per_step / model_step_seconds (2/s); w is the "
           "sampled latency mean before the 5w truncation (truncated mean = w * (1 - e^-5) = 0.9933 w)",
    "regime": "post hoc: 'stable' if rho < 1 else 'overloaded'",
    "legit_lost": "plan's own definition (fixtures/v6/revokebench_plan.json metrics.cancelled_legitimate: 'cancelled or "
                  "denied by the stop') = cancelled + denied_all",
    "cancelled": "completions or accepts whose sink status was 'cancelled' (equals the promoted cancelled_legitimate)",
    "denied_all": "post hoc split of legit_lost: broker submits refused with status 'denied' at any time (every proposal "
                  "is pre-approved and the budget never binds, so every denial in E4 follows the withdraw)",
    "denied_after_enforcement": "promoted metric: denials logged after the enforcement call returned",
    "denied_during_enforcement": "post hoc: denied_all - denied_after_enforcement (between the withdraw returning and "
                                 "the enforcement call returning, e.g. while the fence RPC ran)",
    "little_law_plan": "plan's own prediction (metrics.little_law_prediction), lambda measured from completions in "
                       "[warmup/2, decision) = a 10 s window; reported only where rho < 1",
    "little_law_offered": "post hoc: lambda_offered * (L + w); reported only where rho < 1",
    "in_flight_little_offered": "post hoc: lambda_offered * w, compared with in_flight_at_enforcement; rho < 1 only",
    "backlog_at_t": "post hoc: calls emitted by time t but not yet accepted or refused by the sink/broker (harness log)",
    "holding_s_mean": "post hoc: mean accept->complete-response time of calls accepted before the decision; it includes "
                      "the complete RPC, so it exceeds w by the sink's round trip",
    "rho_holding": "post hoc: lambda_offered * holding_s_mean / C (a lower bound on executor utilization, since submit, "
                   "accept and settle RPCs also occupy the executor)",
    "kill_offsets_s": "post hoc: SIGKILL event times minus the enforcement-return time; enforcement start = return - "
                      "total_ms",
    "logged_retries": "promoted retried_calls: retries logged on completed/accept_refused events only; retries of a "
                      "successful accept and of the fence are not logged by the E4 harness",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summary(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def backlog(events, t):
    emitted = {e["id"] for e in events if e["kind"] == "emitted" and e["t"] <= t}
    resolved = {e["id"] for e in events if e["kind"] in ("accepted", "submit_refused", "accept_refused") and e["t"] <= t}
    return len(emitted - resolved)


def per_trial(tr: dict) -> dict:
    cfg, m, ev = tr["config"], tr["metrics"], tr["events"]
    wl = cfg["workload"]
    lam = wl["calls_per_step"] / wl["model_step_seconds"]
    rho = lam * cfg["w"] / cfg["C"]
    stable = rho < 1
    t_dec = next(e["t"] for e in ev if e["kind"] == "decision")
    enf = next(e for e in ev if e["kind"] == "enforced")
    t_enf = enf["t"]
    cancelled = sum(1 for e in ev if e["kind"] in ("completed", "accept_refused") and e.get("status") == "cancelled")
    denied_all = sum(1 for e in ev if e["kind"] == "submit_refused" and e.get("status") == "denied")
    acc = {e["id"]: e["t"] for e in ev if e["kind"] == "accepted"}
    hold = [e["t"] - acc[e["id"]] for e in ev if e["kind"] == "completed" and e["id"] in acc and acc[e["id"]] < t_dec]
    hold_mean = statistics.fmean(hold) if hold else None
    out = {
        "trial_id": cfg["trial_id"], "fault": cfg["fault"], "arm": cfg["arm"], "L": cfg["L"], "w": cfg["w"],
        "C": cfg["C"], "rep": cfg["rep"],
        "rho": rho, "regime": "stable" if stable else "overloaded",
        "post_decision_commits": m["post_decision_commits"],
        "post_enforcement_commits": m["post_enforcement_commits"],
        "in_flight_at_enforcement": m["in_flight_at_enforcement"],
        "emitted_not_accepted_at_enforcement": m["emitted_not_accepted_at_enforcement"],
        "cancelled": cancelled,
        "cancelled_legitimate_promoted": m["cancelled_legitimate"],
        "denied_all": denied_all,
        "denied_after_enforcement": m["denied_after_enforcement"],
        "denied_during_enforcement": denied_all - m["denied_after_enforcement"],
        "legit_lost": cancelled + denied_all,
        "backlog_at_10s": backlog(ev, wl["warmup"] / 2),
        "backlog_at_decision": backlog(ev, t_dec),
        "holding_s_mean": hold_mean,
        "rho_holding": lam * hold_mean / cfg["C"] if hold_mean is not None else None,
        "lambda_measured_plan_window": m["pre_decision_completion_rate_per_s"],
        "little_law_plan": None, "little_law_offered": None, "in_flight_little_offered": None,
        "little_law_status": "reported" if stable else "not_applicable_rho_ge_1",
        "logged_retries": m["retried_calls"],
    }
    if stable:
        out["in_flight_little_offered"] = lam * cfg["w"]
        if cfg["arm"] == "halt_inference":
            out["little_law_plan"] = m["little_law_post_decision_prediction"]
            out["little_law_offered"] = lam * (cfg["L"] + cfg["w"])
    if cfg["fault"] == "sigkill_sink":
        t_start = t_enf - enf["total_ms"] / 1000.0
        kills = [e["t"] for e in ev if e["kind"] == "sink_sigkill"]
        out["kill_offsets_s"] = [k - t_enf for k in kills]
        out["kills_during_enforcement_call"] = sum(1 for k in kills if t_start <= k <= t_enf)
    if cfg["fault"] == "late_completion":
        held = {e["id"] for e in ev if e["kind"] == "held_delivery"}
        out["held_completion_statuses"] = dict(collections.Counter(
            e["status"] for e in ev if e["kind"] == "completed" and e["id"] in held))
    return out


CELL_FIELDS = ("post_decision_commits", "post_enforcement_commits", "in_flight_at_enforcement",
               "emitted_not_accepted_at_enforcement", "cancelled", "denied_all", "denied_during_enforcement",
               "legit_lost", "backlog_at_10s", "backlog_at_decision", "holding_s_mean", "rho_holding",
               "lambda_measured_plan_window", "little_law_plan", "little_law_offered", "in_flight_little_offered",
               "logged_retries")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=Path, default=TRIALS)
    ap.add_argument("--out", type=Path, help="also write the JSON here")
    args = ap.parse_args()
    rows = [per_trial(json.loads(line)) for line in args.trials.read_text().splitlines() if line.strip()]
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["fault"], r["arm"], r["L"], r["w"], r["C"])].append(r)
    cell_out = []
    for k in sorted(cells):
        rs = cells[k]
        c = {"fault": k[0], "arm": k[1], "L": k[2], "w": k[3], "C": k[4], "rho": rs[0]["rho"], "regime": rs[0]["regime"],
             "replicates": len(rs), "little_law_status": rs[0]["little_law_status"]}
        for f in CELL_FIELDS:
            c[f] = summary([r[f] for r in rs])
            c[f + "_by_rep"] = [r[f] for r in sorted(rs, key=lambda r: r["rep"])]
        if k[0] == "sigkill_sink":
            c["kill_offsets_s_by_rep"] = [r["kill_offsets_s"] for r in sorted(rs, key=lambda r: r["rep"])]
            c["kills_during_enforcement_call"] = sum(r["kills_during_enforcement_call"] for r in rs)
        if k[0] == "late_completion":
            c["held_completion_statuses_by_rep"] = [r["held_completion_statuses"] for r in sorted(rs, key=lambda r: r["rep"])]
        cell_out.append(c)
    wc = sorted({(r["w"], r["C"], r["rho"], r["regime"]) for r in rows})
    by_regime = collections.defaultdict(list)
    for r in rows:
        if r["fault"] == "none":
            by_regime[(r["arm"], r["regime"])].append(r)
    regime_out = [{"arm": a, "regime": g, "trials": len(rs),
                   "post_enforcement_commits": summary([r["post_enforcement_commits"] for r in rs]),
                   "in_flight_at_enforcement": summary([r["in_flight_at_enforcement"] for r in rs]),
                   "emitted_not_accepted_at_enforcement": summary([r["emitted_not_accepted_at_enforcement"] for r in rs]),
                   "cancelled": summary([r["cancelled"] for r in rs]),
                   "denied_all": summary([r["denied_all"] for r in rs]),
                   "legit_lost": summary([r["legit_lost"] for r in rs]),
                   "legit_lost_total": sum(r["legit_lost"] for r in rs),
                   "cancelled_total": sum(r["cancelled"] for r in rs),
                   "denied_total": sum(r["denied_all"] for r in rs)}
                  for (a, g), rs in sorted(by_regime.items())]
    little = [r for r in rows if r["little_law_plan"] is not None]
    kills = [r for r in rows if r["fault"] == "sigkill_sink"]
    report = {
        "schema_version": 1,
        "analysis": "E4 reanalysis (post hoc, after E4 data; no new data)",
        "inputs": {"trials": str(args.trials.relative_to(ROOT) if args.trials.is_relative_to(ROOT) else args.trials),
                   "trials_sha256": sha(args.trials), "plan_sha256": sha(PLAN), "script_sha256": sha(Path(__file__))},
        "definitions": DEFINITIONS,
        "offered_load_cells": [{"w": w, "C": C, "capacity_C_over_w_per_s": C / w, "rho": rho, "regime": g}
                               for w, C, rho, g in wc],
        "by_arm_and_regime_fault_none": regime_out,
        "little_law_stable_halt_trials": {
            "trials": len(little),
            "ratio_measured_over_plan_prediction": summary([r["post_decision_commits"] / r["little_law_plan"]
                                                            for r in little if r["little_law_plan"]]),
            "ratio_measured_over_offered_prediction": summary([r["post_decision_commits"] / r["little_law_offered"]
                                                               for r in little if r["little_law_offered"]]),
        },
        "sigkill_coverage": {
            "trials": len(kills),
            "trials_with_zero_logged_retries": sum(1 for r in kills if r["logged_retries"] == 0),
            "kills_during_enforcement_call": sum(r["kills_during_enforcement_call"] for r in kills),
            "kills": 2 * len(kills),
        },
        "cells": cell_out,
        "trials": rows,
        "notes": [
            "Disclosure, not a result: fence_sink post_enforcement_commits = 0 follows from the design (each sink RPC is a "
            "serialized SQLite transaction, fence() cancels every pending record below the floor, complete() re-checks the "
            "floor); E4 is an implementation test of that property.",
            "Disclosure, not a result: duplicate_effects = 0 is guaranteed by the sink schema (effects.id UNIQUE) as well "
            "as by complete() committing only Pending records.",
            "Where rho >= 1 the open-loop emitter outpaces executor capacity, so the executor queue grows for the whole "
            "warmup and leak counts depend on how long the trial ran before the stop.",
        ],
    }
    text = json.dumps(report, indent=1, sort_keys=True, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
