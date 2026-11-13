#!/usr/bin/env python3
"""Validator for E6 (scripts/fenceability_probes_v6.py). It asserts no probe outcome.

Checks SHA256SUMS, the plan, probe and validator hashes recorded in results.json, the schema and outcome
vocabulary, and recomputes every check, outcome and derived value (including the P2 enforcement class)
from the recorded evidence. The probe labels its results with judge() from this file, so a later change
to these rules, or an edit to results.json, shows up as a mismatch. Failing probes are reported, not
treated as problems.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "fixtures/v6/fenceability_plan.json"
PROBE = ROOT / "scripts/fenceability_probes_v6.py"
OUTCOMES = {"pass", "fail", "not_applicable", "documented_not_executed"}
EXECUTED = ("postgresql16", "etcd_v3_5_17", "posix_fs")
DOCUMENTED = ("aws_s3", "kubernetes_api")
RECORD_KEYS = {"outcome", "mechanism", "checks", "derived", "evidence"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def visible(xid: int, snap: str) -> bool:
    """PostgreSQL pg_snapshot visibility: committed before the snapshot was taken."""
    xmin, xmax, xip = snap.split(":")
    return xid < int(xmin) or (xid < int(xmax) and str(xid) not in xip.split(","))


def violations(ev: dict) -> list[str]:
    """Stale race effects that the surface's own ordering evidence places after the fence."""
    o = ev["order"]

    def after(e):
        if o["kind"] == "pg_snapshot":
            return not visible(e[2], o["snap"])
        if o["kind"] == "etcd_revision":
            return e[2] > o["fence_revision"]
        if o["kind"] == "listing_under_lock":
            return e[0] not in o["listing_under_lock"]
        raise ValueError(o["kind"])
    return [e[0] for e in ev["effects"] if e[0].startswith("race-") and e[1] < ev["floor_after"] and after(e)]


def judge(prim: str, ev: dict) -> tuple[dict, dict]:
    """(checks, derived) for one executed primitive, per the plan's common_pass_conditions."""
    if prim == "P1":
        eff, n, dl = ev["effects"], ev["ops"], ev["delivery_log"]
        first = {x[0]: x for x in dl if x[1]}
        c = {"one_effect_per_op": len(eff) == len(set(eff)) == n, "one_success_per_op": len(first) == sum(x[1] for x in dl) == n,
             "duplicates_delivered": len(dl) == ev["deliveries"] > n}
        if "versions" in ev:
            c["each_key_written_once"] = sorted(ev["versions"]) == sorted(eff) and set(ev["versions"].values()) == {1}
        overlap = sum(not x[1] and x[0] in first and x[2] < first[x[0]][3] and first[x[0]][2] < x[3] for x in dl)
        return c, {"duplicate_deliveries": len(dl) - n, "duplicates_in_flight_with_the_effecting_delivery": overlap}
    if prim == "P2":
        e, att = ev["writer_epoch"], ev["attempts"]
        count = lambda s: sum(a[1] == s for a in att)
        refused, viol = count("refused_sink"), violations(ev)
        c = {"no_violations": not viol, "sink_refused_stale": refused > 0, "fresh_epoch_committed": ev["fresh"] == "committed",
             "observations_consistent": all((a[2] is not None and a[2] <= e) == (a[1] == "committed") for a in att)}
        enf = "sink_enforced" if ev["bare"] == "refused_sink" else "writer_supplied_condition" if refused else "advisory"
        f = ev["fence"]  # client-side monotonic seconds: latency, not ordering evidence
        d = {"enforcement": enf, "violations": viol, "attempts": len(att), "committed": count("committed"),
             "refused_by_sink": refused, "refused_self": count("refused_self"), "errors": sum(a[1].startswith("error") for a in att),
             "attempts_started_after_fence_ack": sum(a[3] > f["t1"] for a in att), "fence_wait_seconds": round(f["t1"] - f["t0"], 4),
             "fence_lock_wait_seconds": round(f["t_locked"] - f["t0"], 4) if "t_locked" in f else None,
             "stale_commits_acked_while_fence_waited": sum(a[1] == "committed" and f["t0"] < a[4] <= f["t1"] for a in att)}
        if "fair_fence_variant" in ev:  # plan amendment 1
            c2, d2 = judge("P2", ev["fair_fence_variant"])
            d["fair_fence_variant"] = {"outcome": "pass" if all(c2.values()) else "fail", "checks": c2, "derived": d2}
        return c, d
    if prim == "P3":
        pre, post, fin, cr, holds = ev["pre"], ev["post"], ev["final"], ev["crash"], set(ev["holds"])
        unclean = cr.get("controldata_state") == "in production" or cr.get("exit_code") == 137
        return {"effect_absent_while_held": not pre["effects"] and not post["effects"],
                "hold_survived_crash": holds <= set(pre["held"]) and holds <= set(post["held"])
                and pre["boot"] != post["boot"] and unclean,
                "revoked_no_effect": "revoke" not in fin["held"] and "revoke" not in fin["effects"],
                "control_one_effect": fin["effects"].count("control") == 1}, {"unclean_stop": unclean}
    raise ValueError(prim)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    run = ap.parse_args().run
    res = json.loads((run / "results.json").read_text())
    plan = json.loads(PLAN.read_text())
    problems, findings, table = [], [], {}
    for line in (run / "SHA256SUMS").read_text().splitlines():
        h, name = line.split(None, 1)
        if sha(run / name.strip()) != h:
            problems.append(f"{name.strip()}: sha256 mismatch")
    for key, p in (("plan_sha256", PLAN), ("script_sha256", PROBE), ("validator_sha256", Path(__file__).resolve())):
        if res.get(key) != sha(p):
            problems.append(f"{key} does not match {p.relative_to(ROOT)}")
    if set(res.get("surfaces", {})) != set(EXECUTED + DOCUMENTED):
        problems.append("surface set differs from the plan")
    for s, sv in res.get("surfaces", {}).items():
        if (s in EXECUTED) != bool(sv.get("executed")):
            problems.append(f"{s}: executed flag disagrees with the plan")
        for p in ("P1", "P2", "P3"):
            r = sv.get("primitives", {}).get(p)
            if not isinstance(r, dict) or set(r) != RECORD_KEYS or r["outcome"] not in OUTCOMES:
                problems.append(f"{s}.{p}: missing or malformed record")
                continue
            dv = r["derived"]
            table[f"{s}.{p}"] = r["outcome"] + (f" [{dv['enforcement']}]" if "enforcement" in dv else "") + \
                (f"; fair_fence_variant {dv['fair_fence_variant']['outcome']}" if "fair_fence_variant" in dv else "")
            if s in DOCUMENTED or r["outcome"] == "not_applicable":
                want = "documented_not_executed" if s in DOCUMENTED else plan["surfaces"][s][p].get("decision")
                if r["outcome"] != want or r["checks"]:
                    problems.append(f"{s}.{p}: {r['outcome']} is not what the plan records ({want})")
                continue
            checks, derived = judge(p, r["evidence"])
            outcome = "pass" if all(checks.values()) else "fail"
            if (checks, derived, outcome) != (r["checks"], r["derived"], r["outcome"]):
                problems.append(f"{s}.{p}: recorded checks/derived/outcome differ from recomputation")
            if outcome == "fail":
                findings.append(f"{s}.{p} fails: " + ", ".join(k for k, v in checks.items() if not v))
            ev = r["evidence"]
            if p == "P2":
                for tag, e, d in ((f"{s}.P2", ev, derived), (f"{s}.P2.fair_fence_variant", ev.get("fair_fence_variant"),
                                                              derived.get("fair_fence_variant", {}).get("derived"))):
                    if e is None:
                        continue
                    log_ops = sorted(a[0] for a in e["attempts"] if a[1] == "committed")
                    sink_ops = sorted(x[0] for x in e["effects"] if x[0].startswith("race-"))
                    if log_ops != sink_ops:
                        findings.append(f"{tag}: {len(log_ops)} committed attempts vs {len(sink_ops)} race effects in the sink")
                    if "server_crosscheck_violations" in e and e["server_crosscheck_violations"] != len(d["violations"]):
                        problems.append(f"{tag}: pg_visible_in_snapshot cross-check disagrees with recomputation")
                    findings.append(f"{tag}: fence raise took {d['fence_wait_seconds']} s; {d['stale_commits_acked_while_fence_waited']} "
                                    f"stale commits acked meanwhile; {d['refused_by_sink']} refused by sink; {len(d['violations'])} violations")
                if "negative_control" in ev:
                    findings.append(f"{s}.P2 negative control (unlocked guard): checker flags "
                                    f"{len(violations(ev['negative_control']))} violation(s); >0 means the check has power")
    v = res.get("surfaces", {})
    if not str(v.get("postgresql16", {}).get("versions", {}).get("server_version", "")).startswith("PostgreSQL 16"):
        problems.append("postgresql16: server version string missing or not 16")
    ev_e = v.get("etcd_v3_5_17", {}).get("versions", {})
    if ev_e.get("server", {}).get("etcdserver") != "3.5.17" or ev_e.get("container_image_id") != ev_e.get("image_id") \
            or not ev_e.get("repo_digests"):
        problems.append("etcd: server version, image id or digest evidence inconsistent")
    print(json.dumps({"run": str(run), "outcomes": table, "problems": problems, "findings": findings}, indent=1))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
