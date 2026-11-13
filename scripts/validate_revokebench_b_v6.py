#!/usr/bin/env python3
"""Integrity validator for E4b (scripts/revokebench_b_v6.py). It asserts no experimental outcome.

Checks, without trusting the harness's own metrics:
  * grid coverage: the run holds every trial the plan (fixtures/v6/revokebench_b_plan.json) predeclares, exactly
    once, with the predeclared config and seed, and nothing else (the enumeration is re-implemented here);
  * counts from the databases: each trial's sink effects table and state row, and the broker's state row, are re-read;
    post-enforcement commits, commits during enforcement, cancelled and denied work are re-derived from them and
    must equal the recorded metrics; the set of committed ids must equal the harness's committed responses;
  * sink-side ordering: each enforcement watermark must be a prefix of the final effects table, and every effects
    row after the fence watermark whose permit epoch is below the fence floor is listed (an implementation finding);
  * kill records: classifications are re-derived from the recorded outstanding request, EOF and post-kill database
    snapshot, and each snapshot must be consistent with the final database (an id absent at the kill can only appear
    later with a higher sequence; an id present at the kill must keep its sequence);
  * accounting: every emitted call has exactly one terminal outcome; results.json equals each trial.json.
Disagreements between the sink-DB and harness-log post-enforcement counts are listed, not failed. Duplicate effects
and a nonzero fence count are reported as findings (the first is excluded by the schema, effects.id UNIQUE).

  validate_revokebench_b_v6.py <raw run dir>          (trials/<id>/{trial.json,sink.sqlite,broker.sqlite})
  validate_revokebench_b_v6.py <promoted dir>         (trials.jsonl: trial.json + "db_extract" per line)
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "fixtures/v6/revokebench_b_plan.json"
TERMINAL = ("completed", "submit_refused", "accept_refused")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def expected_grid(plan: dict) -> dict:
    out, i = {}, 0
    for c in plan["conditions"]:
        for arm, L, (w, C), rep in itertools.product(c["arms"], c["L"], c["cells_w_C"], range(c["replicates"])):
            tid = f"b{i:03d}-{c['name']}-{arm}-{c['emitter']}-L{L}-w{w}-C{C}-r{rep}"
            out[tid] = {"condition": c["name"], "emitter": c["emitter"], "arm": arm, "L": L, "w": w, "C": C,
                        "rep": rep, "seed": 20261212 + i}
            i += 1
    return out


def extract(d: Path) -> dict:
    """Everything the checks need from a trial's two databases (also the promoted form of them)."""
    with closing(sqlite3.connect(f"file:{d / 'sink.sqlite'}?mode=ro", uri=True)) as c:
        rows = [{"sequence": s, "id": i, "epoch0": json.loads(p)["proposal"]["epochs"][0],
                 "payload_id": json.loads(p)["proposal"]["id"]}
                for s, i, p in c.execute("SELECT sequence, id, payload FROM effects ORDER BY sequence")]
        sink = json.loads(c.execute("SELECT body FROM state WHERE singleton=1").fetchone()[0])
    with closing(sqlite3.connect(f"file:{d / 'broker.sqlite'}?mode=ro", uri=True)) as c:
        broker = json.loads(c.execute("SELECT body FROM state WHERE singleton=1").fetchone()[0])
    return {"effects": rows, "sink_floors": sink["floors"],
            "sink_records": [[r["permit"]["proposal"]["id"], r["status"]] for r in sink["records"]],
            "broker_records": [[r["permit"]["proposal"]["id"], r["status"]] for r in broker["records"]],
            "broker_epochs": broker["epochs"]}


def load(run: Path, res: dict):
    if (run / "trials.jsonl").exists():
        recs = {}
        for line in (run / "trials.jsonl").read_text().splitlines():
            r = json.loads(line)
            recs[r["config"]["trial_id"]] = r
        for row in res["trials"]:
            r = recs.get(row["config"]["trial_id"])
            yield row, r, (r or {}).get("db_extract")
        return
    for row in res["trials"]:
        d = run / "trials" / row["config"]["trial_id"]
        if not (d / "trial.json").exists():
            yield row, None, None
            continue
        yield row, json.loads((d / "trial.json").read_text()), extract(d)


def classify(cond, rec, trap_id, floor_before):
    out, snap, eof = rec.get("outstanding"), rec.get("db_after_kill") or {}, rec.get("eof_seen_for")
    if cond == "kill_in_complete":
        ok = (rec.get("how") == "SIGKILL" and out and out["op"] == "complete" and out["id"] == trap_id and eof
              and eof["op"] == "complete" and eof["id"] == trap_id and snap.get("target_has_effect") is False
              and snap.get("target_record_status") == "pending")
        return "inside_complete_uncommitted" if ok else "other"
    if cond == "kill_in_fence":
        ok = (rec.get("how") == "SIGKILL" and out and out["op"] == "fence" and eof and eof["op"] == "fence"
              and snap.get("floor0") == floor_before)
        return "inside_fence_uncommitted" if ok else "other"
    if cond == "exit_after_commit_in_complete":
        ok = (rec.get("how") == "self_exit" and rec.get("exit_status") == 87 and out and out["op"] == "complete"
              and out["id"] == trap_id and eof and eof["id"] == trap_id and snap.get("target_has_effect") is True)
        return "after_commit_before_reply" if ok else "other"
    return "other"


def check_trial(row, tr, db, problems, findings, disagreements, kills_other):
    tid = row["config"]["trial_id"]
    cfg, m, ev = tr["config"], tr["metrics"], tr["events"]
    p = lambda msg: problems.append(f"{tid}: {msg}")
    if tr["metrics"] != row["metrics"] or tr["config"] != row["config"]:
        p("results.json differs from trial.json")
    rows = db["effects"]
    if any(r["id"] != r["payload_id"] for r in rows):
        p("effects.id differs from its payload's proposal id")
    seqs = [r["sequence"] for r in rows]
    if seqs != sorted(seqs):
        p("effects sequence not increasing")
    # accounting: one terminal outcome per emitted call
    emitted = [e["id"] for e in ev if e["kind"] == "emitted"]
    terminal = [e["id"] for e in ev if e["kind"] in TERMINAL]
    if len(set(emitted)) != len(emitted) or sorted(emitted) != sorted(terminal):
        p(f"terminal accounting: {len(emitted)} emitted, {len(terminal)} terminal outcomes")
    # harness committed responses vs sink rows
    ids_log = sorted(e["id"] for e in ev if e["kind"] in ("completed", "accept_refused") and e.get("status") == "committed")
    ids_db = sorted(r["id"] for r in rows)
    if ids_log != ids_db:
        p(f"sink effects ({len(ids_db)}) differ from harness committed responses ({len(ids_log)})")
    # watermarks
    ws, wr = m["watermark_start"], m["watermark_return"]
    for name, wm in (("start", ws), ("return", wr)):
        if sum(1 for s in seqs if s <= wm["max_sequence"]) != wm["effects_rows"]:
            p(f"watermark_{name} is not a prefix of the final effects table")
    if ws["max_sequence"] > wr["max_sequence"]:
        p("watermark_start after watermark_return")
    post = sum(1 for s in seqs if s > wr["max_sequence"])
    during = sum(1 for s in seqs if ws["max_sequence"] < s <= wr["max_sequence"])
    if post != m["post_enforcement_commits_sinkdb"]:
        p("post_enforcement_commits_sinkdb not reproduced from the sink database")
    if during != m["commits_during_enforcement_sinkdb"]:
        p("commits_during_enforcement_sinkdb not reproduced from the sink database")
    enf = [e for e in ev if e["kind"] == "enforced"]
    if len(enf) != 1:
        p("expected one enforced event")
        return None
    t_enf = enf[0]["t"]
    committed_ev = [e for e in ev if e["kind"] == "completed" and e.get("status") == "committed"]
    post_h = sum(1 for e in committed_ev if e["t"] > t_enf)
    if post_h != m["post_enforcement_commits_harness"]:
        p("post_enforcement_commits_harness not reproduced from the event log")
    if post_h != post:
        late_log = {e["id"] for e in committed_ev if e["t"] > t_enf}
        late_db = {r["id"] for r in rows if r["sequence"] > wr["max_sequence"]}
        disagreements.append({"trial": tid, "sinkdb": post, "harness": post_h,
                              "harness_only": sorted(late_log - late_db), "sinkdb_only": sorted(late_db - late_log)})
    # sink-side ordering against the fence
    floor = m["fence_floor"]
    if cfg["arm"] == "fence_sink":
        if floor is None or enf[0].get("floor") != floor:
            p("fence floor not recorded")
        viol = [r["id"] for r in rows if r["sequence"] > wr["max_sequence"] and r["epoch0"] < floor]
        if viol != m["fence_order_violations"]:
            p("fence_order_violations not reproduced from the sink database")
        if viol:
            findings.append(f"{tid}: {len(viol)} effects committed after the fence watermark below the floor: {viol}")
        if wr["floor0"] != floor:
            findings.append(f"{tid}: fence not durable in the watermark snapshot (floor0={wr['floor0']}, floor={floor})")
        if db["sink_floors"][0] != floor:
            findings.append(f"{tid}: final sink floor {db['sink_floors'][0]} differs from the fence floor {floor}")
    elif m["fence_order_violations"]:
        p("fence_order_violations outside fence_sink")
    # legitimate work lost from the databases
    sink_status = dict(db["sink_records"])
    cancelled_db = sorted(i for i, s in db["sink_records"] if s == "cancelled")
    cancelled_ev = sorted(e["id"] for e in ev if e["kind"] in ("completed", "accept_refused") and e.get("status") == "cancelled")
    if cancelled_db != cancelled_ev:
        p(f"cancelled records in the sink ({len(cancelled_db)}) differ from cancelled responses ({len(cancelled_ev)})")
    in_broker = {i for i, _ in db["broker_records"]}
    denied_db = sorted(i for i in set(emitted) if i not in in_broker)
    denied_ev = sorted(e["id"] for e in ev if e["kind"] == "submit_refused" and e.get("status") == "denied")
    if denied_db != denied_ev:
        p(f"emitted calls absent from the broker ({len(denied_db)}) differ from denied submits ({len(denied_ev)})")
    if (m["cancelled"], m["denied"], m["legit_work_lost"]) != (len(cancelled_db), len(denied_db),
                                                              len(cancelled_db) + len(denied_db)):
        p("legit_work_lost not reproduced from the databases")
    committed_records = sorted(i for i, s in db["sink_records"] if s == "committed")
    if committed_records != sorted(set(ids_db)):
        findings.append(f"{tid}: sink committed records differ from its effects rows")
    if len(ids_db) != len(set(ids_db)):
        findings.append(f"{tid}: {len(ids_db) - len(set(ids_db))} duplicate effect rows (excluded by effects.id UNIQUE)")
    if not m["audit_consistent"]:
        findings.append(f"{tid}: broker/sink audit inconsistent")
    # kills
    pos = {r["id"]: r["sequence"] for r in rows}
    classes = [classify(cfg["condition"], k, m["trap_id"], ws["floor0"]) for k in tr["kills"]]
    if classes != m["kill_classes"]:
        p("kill classes not reproduced")
    for k, c in zip(tr["kills"], classes):
        snap = k.get("db_after_kill") or {}
        tgt = snap.get("target_id")
        if tgt is not None:
            if snap.get("target_has_effect") is False and tgt in pos and pos[tgt] <= snap["max_sequence"]:
                p("an id absent at the kill has a sequence inside the post-kill snapshot")
            if snap.get("target_has_effect") is True and (tgt not in pos or pos[tgt] > snap["max_sequence"]):
                p("an id present at the kill is missing or moved in the final database")
        if sum(1 for s in seqs if s <= snap.get("max_sequence", 0)) != snap.get("effects_rows"):
            p("post-kill snapshot is not a prefix of the final effects table")
        if c == "other":
            kills_other.append({"trial": tid, "kill": k})
    expected_kills = {"kill_in_complete": 1, "kill_in_fence": 1, "exit_after_commit_in_complete": 1}.get(cfg["condition"], 0)
    if len(tr["kills"]) != expected_kills:
        kills_other.append({"trial": tid, "kills": len(tr["kills"]), "expected": expected_kills})
    # held permits
    held = [e["id"] for e in ev if e["kind"] == "permit_held"]
    acc = {e["id"]: e.get("status") for e in ev if e["kind"] == "accept_refused"}
    acc.update({e["id"]: "pending" for e in ev if e["kind"] == "accepted"})
    want = [{"id": i, "accept_status": acc.get(i), "sink_record_status": sink_status.get(i),
             "effect_sequence": pos.get(i), "effect_after_watermark": (pos[i] > wr["max_sequence"]) if i in pos else None}
            for i in held]
    if want != m["held_permits"]:
        p("held_permits not reproduced from the event log and databases")
    trapped = next((e["status"] for e in ev if e["kind"] == "completed" and e.get("trapped")), None)
    if trapped != m["trapped_final_status"]:
        p("trapped_final_status not reproduced")
    return {"post_sinkdb": post, "post_harness": post_h, "during": during, "cancelled": len(cancelled_db),
            "denied": len(denied_db), "legit_work_lost": len(cancelled_db) + len(denied_db), "kill_classes": classes,
            "fence_order_violations": len(m["fence_order_violations"]), "trapped_final_status": trapped,
            "held_accept": [h["accept_status"] for h in want], "held_with_effect": sum(1 for h in want if h["effect_sequence"] is not None),
            "in_flight": m["in_flight_at_enforcement"], "rho_design": m["rho_design"], "rho_measured": m["rho_measured"],
            "retried_calls": m["retried_calls"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="raw run directory or promoted evidence directory")
    args = ap.parse_args()
    res = json.loads((args.run / "results.json").read_text())
    plan = json.loads(PLAN.read_text())
    problems, findings, disagreements, kills_other = [], [], [], []
    if res["plan_sha256"] != sha(PLAN):
        problems.append("plan hash mismatch")
    grid = expected_grid(plan)
    ids = [r["config"]["trial_id"] for r in res["trials"]]
    missing, extra = sorted(set(grid) - set(ids)), sorted(set(ids) - set(grid))
    dup = len(ids) - len(set(ids))
    coverage = {"predeclared": len(grid), "present": len(ids), "missing": len(missing), "extra": extra,
                "duplicates": dup, "complete": not missing and not extra and not dup}
    if extra or dup:
        problems.append(f"grid: {len(extra)} undeclared trials, {dup} duplicates")
    if missing and not res["smoke_test"]:
        problems.append(f"grid: {len(missing)} predeclared trials missing")
    for r in res["trials"]:
        c, want = r["config"], grid.get(r["config"]["trial_id"])
        if want and any(c[k] != v for k, v in want.items()):
            problems.append(f"{c['trial_id']}: config differs from the plan")
    groups = {}
    for row, tr, db in load(args.run, res):
        tid = row["config"]["trial_id"]
        if tr is None or db is None:
            problems.append(f"{tid}: trial record or database extract missing")
            continue
        out = check_trial(row, tr, db, problems, findings, disagreements, kills_other)
        if out is not None:
            c = row["config"]
            groups.setdefault((c["condition"], c["arm"], c["emitter"], c["L"], c["w"], c["C"]), []).append(out)
    sys.path.insert(0, str(ROOT / "scripts"))
    from revokebench_b_v6 import summarize  # results.json's summary must equal one recomputed from its trials
    if summarize(res["trials"]) != res["summary"]:
        problems.append("results.json summary differs from its trials")
    table = []
    for k, outs in sorted(groups.items()):
        g = {"condition": k[0], "arm": k[1], "emitter": k[2], "L": k[3], "w": k[4], "C": k[5], "trials": len(outs),
             "rho_design": outs[0]["rho_design"],
             "rho_measured_median": statistics.median(o["rho_measured"] for o in outs)}
        for f in ("post_sinkdb", "post_harness", "during", "cancelled", "denied", "legit_work_lost", "in_flight",
                  "retried_calls"):
            g[f] = [o[f] for o in outs]
        g["kill_classes"] = sorted(c for o in outs for c in o["kill_classes"])
        g["fence_order_violations"] = sum(o["fence_order_violations"] for o in outs)
        g["trapped_final_status"] = sorted(str(o["trapped_final_status"]) for o in outs)
        g["held_accept"] = sorted(str(s) for o in outs for s in o["held_accept"])
        g["held_with_effect"] = sum(o["held_with_effect"] for o in outs)
        table.append(g)
    report = {"run": str(args.run), "trials": len(res["trials"]), "smoke_test": res["smoke_test"],
              "plan_sha256_ok": res["plan_sha256"] == sha(PLAN), "grid_coverage": coverage,
              "problems": problems, "implementation_findings": findings,
              "sinkdb_vs_harness_post_enforcement_disagreements": disagreements,
              "kills_not_meeting_criterion": kills_other, "table": table}
    print(json.dumps(report, indent=1))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
