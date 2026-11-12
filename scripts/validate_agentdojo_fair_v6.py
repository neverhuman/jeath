#!/usr/bin/env python3
"""Integrity validator for E8 (scripts/agentdojo_fair_v6.py). It asserts no experimental outcome.

Run with the benchmark venv. For every episode it:
  * re-places the injections with AgentDojo's own candidate search and checks the stored vector names;
  * replays the stored executed-call trace on a freshly built environment and re-runs the native graders,
    which must reproduce the stored utility and attack-success values;
  * checks that the executed calls are exactly the admitted calls of the trace, in order;
  * for strict and delegated, re-derives every admit/deny decision from a fresh specification state fed the
    replayed tool outputs; for oracle, re-derives tool/target decisions from the ground truth;
  * for every authority arm, checks that the sink's committed effects equal the admitted effect calls.
It also recomputes stage 0 and every summary cell. Findings are reported, not required to be zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agentdojo_fair_v6 as e8  # noqa: E402
from infer_v2 import sha  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--require-current-script", action="store_true")
    args = ap.parse_args()
    from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime

    plan = json.loads(e8.PLAN.read_text())
    suites = e8.load_suites(plan)
    attacks = {n: e8.make_attack(s) for n, s in suites.items()}
    problems, findings = [], []
    res = json.loads((args.run / "results.json").read_text())
    static = json.loads((args.run / "static.json").read_text())
    if res["plan_sha256"] != sha(e8.PLAN):
        problems.append("plan hash mismatch")
    if args.require_current_script and res["script_sha256"] != sha(Path(e8.__file__)):
        problems.append("results were produced by a different agentdojo_fair_v6.py")
    if e8.static_analysis(plan, suites, attacks)["summary"] != static["summary"]:
        problems.append("stage 0 summary not reproduced")

    rows = [json.loads(line) for line in (args.run / "episodes.jsonl").read_text().splitlines()]
    pp = args.run / "progent_policies.json"
    progent_policies = json.loads(pp.read_text()) if pp.exists() else {}
    if len(rows) != res["episodes_recorded"] or len(rows) != res["episodes_planned"]:
        problems.append(f"episode count {len(rows)} vs planned {res['episodes_planned']} / recorded {res['episodes_recorded']}")
    if len({r["episode"] for r in rows}) != len(rows):
        problems.append("duplicate episode ids")
    for r in rows:
        tag, s_name, arm = r["episode"], r["suite"], r["arm"]
        suite = suites[s_name]
        ut = suite.user_tasks[r["user_task"]]
        it = suite.injection_tasks[r["injection_task"]] if r["injection_task"] else None
        injections = attacks[s_name].attack(ut, it) if it else {}
        if sorted(injections) != r["injected_vectors"]:
            problems.append(f"{tag}: injected vectors not reproduced")
        env = e8.build_env(suite, ut, injections)
        pre = env.model_copy(deep=True)
        benign = e8.build_env(suite, ut, {})
        oracle = e8.oracle_spec(s_name, ut.ground_truth(benign.model_copy(deep=True)), benign) | {"_effects": e8.EFFECT_TOOLS[s_name]}
        if {k: v for k, v in oracle.items() if k != "_effects"} != r["oracle_spec"]:
            problems.append(f"{tag}: oracle spec not reproduced")
        spec = None
        if arm in ("authority_strict", "authority_delegated"):
            spec = e8.Spec(arm.split("_")[1], s_name, ut.PROMPT, e8.principal_state(s_name, env))
        runtime = FunctionsRuntime(suite.tools)
        admitted_calls, admitted_effects = [], 0
        for step in r["trace"]:
            if "call" not in step:
                continue
            fn, a = step["call"]["function"], step["call"]["args"]
            if arm == "unprotected" and not step["admitted"]:
                problems.append(f"{tag}: unprotected arm denied a call")
            if spec is not None:
                ok, rule, _ = spec.check(fn, a)
                if ok != step["admitted"] and step.get("rule") != "L_broker_or_sink":
                    problems.append(f"{tag} turn {step['turn']}: {arm} decision on {fn} not reproduced ({ok} vs {step['admitted']})")
            if arm == "progent":
                entry = progent_policies[f"{s_name}/{r['user_task']}"]
                gate = e8.Gate("progent", s_name, suite, ut, env, None, entry, e8.threading.Lock(), oracle)
                if gate.decide(fn, a)[0] != step["admitted"]:
                    problems.append(f"{tag} turn {step['turn']}: progent decision on {fn} not reproduced")
            if arm == "authority_oracle":
                ok, rule, _ = e8.oracle_check(oracle, fn, a)
                if not ok and step["admitted"]:
                    problems.append(f"{tag} turn {step['turn']}: oracle admitted {fn} that its grant denies")
                if ok and not step["admitted"] and step.get("rule") != "O_budget":
                    problems.append(f"{tag} turn {step['turn']}: oracle denied {fn} without a reproducible rule")
            if not step["admitted"]:
                if step.get("reply") != e8.DENIED_REPLY:
                    problems.append(f"{tag} turn {step['turn']}: denied call answered with arm-specific text")
                continue
            admitted_calls.append({"function": fn, "args": a})
            admitted_effects += fn in e8.EFFECT_TOOLS[s_name]
            value, err = runtime.run_function(env, fn, a)
            text = e8.fmt_result(value) if err is None else str(err)
            if step.get("reply") != ({"tool_result": text} if err is None else {"tool_error": text}):
                problems.append(f"{tag} turn {step['turn']}: replayed tool output differs from the stored reply")
            if spec is not None:
                spec.observe(fn, a, text)
        if admitted_calls != r["executed_calls"]:
            problems.append(f"{tag}: executed calls are not the admitted calls of the trace")
        if r["grading_status"] == "completed":
            executed = [FunctionCall(function=c["function"], args=c["args"]) for c in r["executed_calls"]]
            g = e8.grade(suite, ut, it, r["answer"], pre, env, executed)
            if g["utility"] != r["utility"] or g["attack_success"] != r["attack_success"]:
                problems.append(f"{tag}: native grades not reproduced ({g} vs utility={r['utility']} attack={r['attack_success']})")
        else:
            findings.append(f"{tag}: grading {r['grading_status']}: {r.get('grading_error')}")
        if arm in e8.AUTHORITY_ARMS:
            lab = r["lab"]
            committed = [p["id"] for p in lab["permits"] if p["status"] == "committed"]
            if sorted(lab["sink_effect_ids"]) != sorted(committed) or len(committed) != admitted_effects:
                problems.append(f"{tag}: sink effects {len(lab['sink_effect_ids'])} vs committed permits {len(committed)} vs admitted effects {admitted_effects}")
            if not lab["audit_consistent"]:
                findings.append(f"{tag}: broker/sink audit inconsistent")
        elif r["lab"] is not None:
            problems.append(f"{tag}: non-authority arm has a lab record")
        if r["status"] == "inference_error":
            findings.append(f"{tag}: inference error {r.get('error')}")
    # Same prompt, temperature 0, top_k 1 and seed: two arms' raw replies must agree turn by turn for as long as every
    # earlier message the agent received was identical. A divergence is nondeterministic inference, reported as a finding.
    groups = {}
    for r in rows:
        groups.setdefault((r["suite"], r["user_task"], r["injection_task"]), []).append(r)
    diverged = 0
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        for a in rs:
            for b in rs:
                if a["arm"] >= b["arm"]:
                    continue
                for sa, sb in zip(a["trace"], b["trace"]):
                    if sa["raw"] != sb["raw"]:
                        diverged += 1
                        findings.append(f"{key}: {a['arm']} vs {b['arm']} diverge at turn {sa['turn']} under identical history")
                        break
                    if sa.get("reply") != sb.get("reply"):
                        break
    if e8.summarize(rows) != res["summary"]:
        problems.append("summary not reproduced from episodes.jsonl")
    report = {"run": str(args.run), "episodes": len(rows), "smoke_test": res["smoke_test"], "problems": problems,
              "divergent_arm_pairs_under_identical_history": diverged, "findings": findings}
    print(json.dumps(report, indent=1))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
