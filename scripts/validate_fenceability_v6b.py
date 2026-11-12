#!/usr/bin/env python3
"""Validator for E6b (scripts/fenceability_probes_v6b.py; plan fixtures/v6/fenceability_b_plan.json).

Integrity only. It recomputes every status, class, outcome, event, count, rate, Clopper-Pearson interval and
median from the raw attempt rows, and requires the recorded summary to equal that recomputation. It also checks
hashes, that the plan was committed before the run started and that its design is unchanged since its first
commit, the host, completeness and predeclared order, deterministic-probe orderings, race sizes, load records
and versions. It never asserts an outcome, class or rate. Disagreements between a client log and the surface's
read-back are reported as findings. The probe labels its results with analyze() from this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_REL = "fixtures/v6/fenceability_b_plan.json"
PLAN, PROBE, E6_PROBE = ROOT / PLAN_REL, ROOT / "scripts/fenceability_probes_v6b.py", ROOT / "scripts/fenceability_probes_v6.py"
VARIANTS = ("cond", "omit_epoch", "omit_all", "forge", "fresh")
STATUSES = {"committed", "refused_by_surface", "refused_self", "error"}
EVENTS = ("starved", "refusal_observed", "violation_free", "stale_commit_started_after_ack")
CONTINUOUS = ("fence_wait_s", "fence_lock_wait_s", "attempts_started_after_fence_ack", "stale_commits_acked_while_fence_waited",
              "committed", "refused_by_surface", "refused_self", "errors")
# E6's recorded P2 enforcement labels (evidence/v6/reference/e6-fenceability/results.json, derived.enforcement); descriptive only
E6_LABELS = {"postgresql16": "sink_enforced", "etcd_v3_5_17": "writer_supplied_condition", "posix_fs": "advisory"}
AUTH_OLD = "revision of auth store is old"
ALPHA = 0.05


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_plan(path: Path = PLAN) -> dict:
    return json.loads(Path(path).read_text())


def pg_origin(plan: dict) -> dict:
    """SQLSTATE -> class, read from the plan's refusal_origin table (first word of each entry)."""
    return {k: v.split()[0] for k, v in plan["classification_rule"]["refusal_origin"]["postgresql_sqlstate"].items()}


# ------------------------------------------------------------------ statuses and the classification rule
def status(raw, plan: dict) -> str:
    """Status of one write, from the raw facts the probe recorded (plan: classification_rule.refusal_origin)."""
    if not isinstance(raw, dict):
        return "error"
    k = raw.get("kind")
    if k == "pg":
        if raw.get("ok"):
            return "committed"
        return "refused_by_surface" if raw.get("sqlstate") in pg_origin(plan) else "error"
    if k == "etcd":
        if raw.get("http") == 200:
            return "refused_by_surface" if raw.get("succeeded") is False else "committed"
        if raw.get("grpc_code") == 7 or AUTH_OLD in str(raw.get("message") or ""):
            return "refused_by_surface"
        return "error"
    if k == "fs":
        if raw.get("ok"):
            return "committed"
        if raw.get("self_refused_floor") is not None:
            return "refused_self"
        return "refused_by_surface" if raw.get("errno") in ("EACCES", "EPERM") else "error"
    return "error"


def origin(raw, plan: dict):
    if status(raw, plan) != "refused_by_surface":
        return None
    if raw["kind"] == "pg":
        return pg_origin(plan)[raw["sqlstate"]]
    return "surface_native"  # etcd server compare / RBAC check, kernel permission check (plan refusal_origin)


def classify(rows: dict, plan: dict) -> tuple[str, str]:
    """Rules R1-R4 of the plan. rows: variant -> attempt row [variant, op, status, raw, t_start, t_end]."""
    st = {v: status(rows[v][3], plan) for v in VARIANTS}
    if st["cond"] == "error":
        return "indeterminate", "R4: cond error"
    if st["cond"] != "refused_by_surface":
        return "advisory", f"R1: cond {st['cond']}"
    om = (st["omit_epoch"], st["omit_all"])
    if "committed" in om:
        return "writer_supplied_condition", f"R2: omit_epoch {om[0]}, omit_all {om[1]}"
    if om == ("refused_by_surface", "refused_by_surface"):
        o = origin(rows["cond"][3], plan)
        return (o, f"R3: cond refused ({rows['cond'][3].get('sqlstate') or 'native'})") if o else ("indeterminate", "R4: unmapped origin")
    return "indeterminate", f"R4: omit_epoch {om[0]}, omit_all {om[1]}"


def det_derive(det: dict, plan: dict) -> dict:
    rows = {r[0]: r for r in det["attempts"]}
    st = {v: status(r[3], plan) for v, r in rows.items()}
    cls, why = classify(rows, plan)
    if "error" in det["fence"]:  # amendment 1: no acknowledged fence, so R1-R3 have no input (R4)
        cls, why = "indeterminate", f"R4: fence raise failed ({det['fence']['error'].get('sqlstate') or det['fence']['error'].get('exc')})"
    committed = sorted(r[1] for r in det["attempts"] if st[r[0]] == "committed")
    readback_ok = sorted(det["readback"]) == committed
    outcome = "pass" if ("error" not in det["fence"] and st["pre"] == "committed" and st["cond"] == "refused_by_surface"
                         and st["fresh"] == "committed" and readback_ok) else "fail"
    return {"class": cls, "class_reason": why, "outcome": outcome, "statuses": st, "readback_matches_log": readback_ok,
            "forge_refused": st["forge"] == "refused_by_surface", "cond_origin": origin(rows["cond"][3], plan)}


# ------------------------------------------------------------------ race
def visible(xid: int, snap: str) -> bool:
    """PostgreSQL pg_snapshot visibility (as validate_fenceability_v6.visible)."""
    xmin, xmax, xip = snap.split(":")
    return xid < int(xmin) or (xid < int(xmax) and str(xid) not in xip.split(","))


def violations(race: dict) -> list[str]:
    """Committed stale race effects that the mechanism's own ordering evidence places after the fence."""
    o = race["order"]

    def after(e):
        if o["kind"] == "pg_snapshot":
            return not visible(e[2], o["snap"])
        if o["kind"] == "etcd_revision":
            return e[2] > o["fence_revision"]
        if o["kind"] in ("listing_under_lock", "listing_after_ack"):
            return e[0] not in o["listing"]
        raise ValueError(o["kind"])
    return [e[0] for e in race["effects"] if e[0].startswith("race-") and after(e)]


def race_derive(race: dict, plan: dict) -> dict:
    att, f = race["attempts"], race["fence"]
    st = [status(a[2], plan) for a in att]
    t0, t1 = f["t0"], f["t1"]
    after_ack = sum(a[3] > t1 for a in att)
    late_commits = sum(s == "committed" and a[3] > t1 for s, a in zip(st, att))
    failed = "error" in f  # amendment 1: a failed fence has no ordering evidence and contributes no events
    viol = [] if failed else violations(race)
    d = {"attempts": len(att), "committed": st.count("committed"), "refused_by_surface": st.count("refused_by_surface"),
         "refused_self": st.count("refused_self"), "errors": st.count("error"),
         "attempts_started_after_fence_ack": after_ack, "stale_commits_started_after_ack": late_commits,
         "stale_commits_acked_while_fence_waited": sum(s == "committed" and t0 < a[4] <= t1 for s, a in zip(st, att)),
         "fence_wait_s": round(t1 - t0, 4), "fence_lock_wait_s": round(f["t_locked"] - t0, 4) if "t_locked" in f else None,
         "fence_started_after_writers_done": bool(att) and t0 > max(a[4] for a in att), "violations": viol, "fence_failed": failed}
    d["events"] = None if failed else {"starved": after_ack == 0, "refusal_observed": d["refused_by_surface"] > 0,
                                       "violation_free": not viol, "stale_commit_started_after_ack": late_commits > 0}
    return d


# ------------------------------------------------------------------ statistics
def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> list[float]:
    """Exact two-sided interval by bisection on the binomial CDF; each tail alpha/2."""
    def solve(fn, target):  # fn is decreasing in p; returns p with fn(p) = target
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fn(mid) > target else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: binom_cdf(k - 1, n, p), 1 - alpha / 2)  # P(X >= k | p) = alpha/2
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p), alpha / 2)  # P(X <= k | p) = alpha/2
    return [round(lower, 4), round(upper, 4)]


def rate(flags: list[bool]) -> dict:
    k, n = sum(flags), len(flags)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": clopper_pearson(k, n) if n else None}


def spread(xs: list) -> dict:
    xs = [x for x in xs if x is not None]
    return {"median": round(statistics.median(xs), 4), "min": min(xs), "max": max(xs)} if xs else None


# ------------------------------------------------------------------ design, order and the summary
def mechanisms(plan: dict, surface: str | None = None) -> list[str]:
    return [m for m, v in plan["mechanisms"].items() if surface is None or v["surface"] == surface]


def surfaces(plan: dict) -> list[str]:
    return list(dict.fromkeys(v["surface"] for v in plan["mechanisms"].values()))


def expected_order(plan: dict, repeats: int) -> list[tuple]:
    out = []
    for r in range(1, repeats + 1):
        for cond in (("idle", "loaded") if r % 2 else ("loaded", "idle")):
            for s in surfaces(plan):
                ms = mechanisms(plan, s)
                random.Random(f"e6b-{cond}-{r}-{s}").shuffle(ms)
                out += [(cond, r, s, m) for m in ms]
    return out


def e6_mechanism(plan: dict, surface: str) -> str:
    return next(m for m in mechanisms(plan, surface) if plan["mechanisms"][m]["source"].startswith("E6's planned P2 probe"))


def analyze(plan: dict, res: dict) -> dict:
    runs, ranks = res["runs"], plan["classification_rule"]["classes_strongest_first"]
    det = {}
    for m in mechanisms(plan):
        ds = [det_derive(r["det"], plan) for r in runs if r["mechanism"] == m]
        if not ds:
            continue
        cls = sorted({d["class"] for d in ds})
        out = sorted({d["outcome"] for d in ds})
        det[m] = {"class": cls[0] if len(cls) == 1 else "varies", "outcome": out[0] if len(out) == 1 else "varies",
                  "deterministic": len(cls) == 1 and len(out) == 1, "runs": len(ds),
                  "class_counts": {c: sum(d["class"] == c for d in ds) for c in cls},
                  "outcome_counts": {o: sum(d["outcome"] == o for d in ds) for o in out},
                  "class_reasons": sorted({d["class_reason"] for d in ds}),
                  "forge_refused": sum(d["forge_refused"] for d in ds),
                  "readback_mismatches": sum(not d["readback_matches_log"] for d in ds),
                  "statuses": {v: dict(sorted(Counter(d["statuses"][v] for d in ds).items())) for v in ("pre",) + VARIANTS},
                  "surface": plan["mechanisms"][m]["surface"], "selected_as": plan["mechanisms"][m]["selected_as"],
                  "epoch_binding": plan["mechanisms"][m]["epoch_binding"]}
    race = {}
    for cond in plan["conditions"]:
        if cond == "per_block_record":
            continue
        for m in mechanisms(plan):
            ds = [race_derive(r["race"], plan) for r in runs if r["mechanism"] == m and r["condition"] == cond]
            ok = [d for d in ds if not d["fence_failed"]]
            if ds:
                race.setdefault(cond, {})[m] = {"n": len(ok), "runs": len(ds), "fence_failed": len(ds) - len(ok),
                                                "events": {e: rate([d["events"][e] for d in ok]) for e in EVENTS},
                                                "continuous": {c: spread([d[c] for d in ok]) for c in CONTINUOUS},
                                                "violations_total": sum(len(d["violations"]) for d in ok),
                                                "fence_started_after_writers_done": sum(d["fence_started_after_writers_done"] for d in ok)}
    lead = {}
    for s in surfaces(plan):
        ms = [m for m in mechanisms(plan, s) if m in det]
        if not ms:
            continue
        ok = [m for m in ms if det[m]["deterministic"] and det[m]["class"] in ranks]
        best = min((det[m]["class"] for m in ok), key=ranks.index, default=None)
        e6m = e6_mechanism(plan, s)
        lead[s] = {"strongest_class": best, "via": [m for m in ok if det[m]["class"] == best],
                   "mechanism_classes": {m: det[m]["class"] for m in ms},
                   "mechanism_outcomes": {m: det[m]["outcome"] for m in ms},
                   "e6_mechanism": e6m, "e6_mechanism_class": det.get(e6m, {}).get("class"), "e6_label": E6_LABELS.get(s)}
    for s, d in plan["documented_surfaces"].items():
        lead[s] = {"strongest_class": d["class"].split(":")[0], "executed": False, "note": d["class"]}
    primary = {m: {e: race.get("idle", {}).get(m, {}).get("events", {}).get(e) for e in ("starved", "refusal_observed")}
               for m in (e6_mechanism(plan, s) for s in surfaces(plan))}
    return {"surface_lead": lead, "deterministic": det, "primary_idle_e6_mechanisms": primary, "race": race}


# ------------------------------------------------------------------ integrity
def git(*a) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a], check=True, capture_output=True).stdout.decode()


def git_bytes(*a) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *a], check=True, capture_output=True).stdout


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def check_plan_commit(res: dict, problems: list, facts: dict) -> None:
    c = res.get("plan_commit")
    try:
        blob = git_bytes("show", f"{c}:{PLAN_REL}")
        ctime = git("log", "-1", "--format=%cI", c).strip()
        first = git("log", "--diff-filter=A", "--format=%H", "--", PLAN_REL).split()[-1]
        first_plan = json.loads(git_bytes("show", f"{first}:{PLAN_REL}"))
        anc = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", c, "HEAD"]).returncode == 0
    except (subprocess.CalledProcessError, IndexError, ValueError) as e:
        problems.append(f"plan commit {c!r} cannot be read from git: {e}")
        return
    facts.update(plan_commit=c, plan_commit_time=ctime, plan_first_commit=first)
    if hashlib.sha256(blob).hexdigest() != res.get("plan_sha256"):
        problems.append("plan at plan_commit does not hash to plan_sha256")
    if not ts(ctime) < ts(res["started_utc"]):
        problems.append(f"plan commit time {ctime} is not before started_utc {res['started_utc']}")
    if not anc:
        problems.append("plan_commit is not an ancestor of HEAD")
    run_plan = json.loads(blob)
    if {k: v for k, v in run_plan.items() if k != "amendments"} != {k: v for k, v in first_plan.items() if k != "amendments"}:
        problems.append("plan used by the run differs from its first commit outside 'amendments' (design not frozen)")


def check_run(plan: dict, res: dict, problems: list, findings: list) -> None:
    smoke, n = bool(res.get("smoke")), res.get("repeats")
    if not smoke and n != plan["repeats"]:
        problems.append(f"repeats {n} != plan {plan['repeats']}")
    want = expected_order(plan, n)
    got = [(r["condition"], r["repeat"], r["surface"], r["mechanism"]) for r in sorted(res["runs"], key=lambda r: r["seq"])]
    if got != want:
        problems.append(f"runs differ from the predeclared order/completeness ({len(got)} recorded, {len(want)} expected)")
    blocks = {(b["condition"], b["repeat"]): b for b in res["blocks"]}
    for cond, r in {(c, r) for c, r, _, _ in want}:
        b = blocks.get((cond, r))
        need = 96 if cond == "loaded" else 0
        if b is None or b["burners"] != need or b["burners_alive_at_end"] != need:
            problems.append(f"block {cond}-{r}: load record missing or burners != {need}")
    cp = plan["contention_probe"]["parameters"]
    for run in res["runs"]:
        tag = f"{run['condition']}-{run['repeat']}-{run['mechanism']}"
        d, rows = run["det"], {a[0]: a for a in run["det"]["attempts"]}
        if [a[0] for a in d["attempts"]] != ["pre", *VARIANTS]:
            problems.append(f"{tag}: deterministic attempts are not pre, {', '.join(VARIANTS)} in order")
            continue
        for op, rec, raw in [(a[1], a[2], a[3]) for a in d["attempts"]] + [(a[0], a[1], a[2]) for a in run["race"]["attempts"]]:
            if rec not in STATUSES or rec != status(raw, plan):
                problems.append(f"{tag}: recorded status {rec!r} for {op} != recomputed {status(raw, plan)!r}")
        f = d["fence"]
        if not rows["pre"][5] <= f["t0"] <= f["t1"]:
            problems.append(f"{tag}: deterministic pre write did not end before the fence request")
        prev = f["t1"]
        for v in VARIANTS:  # no timing race: each write starts no earlier than the ack and the previous write's return
            if not (prev <= rows[v][4] <= rows[v][5]):
                problems.append(f"{tag}: deterministic {v} did not start after the ack and the previous write")
            prev = rows[v][5]
        ra = run["race"]["attempts"]
        if len(ra) != cp["writers"] * cp["attempts_per_writer"] or len({a[0] for a in ra}) != len(ra):
            problems.append(f"{tag}: race has {len(ra)} attempts (or duplicate ops), expected {cp['writers'] * cp['attempts_per_writer']}")
        rf = run["race"]["fence"]
        if not rf["t0"] <= rf.get("t_locked", rf["t0"]) <= rf["t1"]:
            problems.append(f"{tag}: race fence times out of order")
        if "server_crosscheck_violations" in run["race"] and "error" not in rf \
                and run["race"]["server_crosscheck_violations"] != len(violations(run["race"])):
            problems.append(f"{tag}: pg_visible_in_snapshot cross-check disagrees with recomputation")
        if run["surface"] == "etcd_v3_5_17" and not str(run.get("versions", {}).get("container", "")).startswith("e6b-"):
            problems.append(f"{tag}: etcd container name does not start with e6b-")
        log_ops = sorted(a[0] for a in ra if status(a[2], plan) == "committed")
        sink_ops = sorted(e[0] for e in run["race"]["effects"] if e[0].startswith("race-"))
        if log_ops != sink_ops:
            findings.append(f"{tag}: race log has {len(log_ops)} committed vs {len(sink_ops)} effects in the surface")
        if not det_derive(d, plan)["readback_matches_log"]:
            findings.append(f"{tag}: deterministic read-back differs from committed log")


def check_versions(plan: dict, res: dict, problems: list) -> None:
    sw, host = res.get("software", {}), res.get("host", {})
    if host.get("hostname") != plan["host"]["hostname"] or host.get("uname_sm") != plan["host"]["uname_sm"]:
        problems.append(f"host {host.get('hostname')}/{host.get('uname_sm')} is not the plan's")
    if sw.get("postgres_binary_sha256") != plan["software"]["postgresql"].split("sha256 must be ")[1].split(",")[0]:
        problems.append("postgres binary sha256 differs from the plan")
    if any(not str(r.get("versions", {}).get("server_version", "")).startswith("PostgreSQL 16.15")
           for r in res["runs"] if r["surface"] == "postgresql16"):
        problems.append("a PostgreSQL run is not on server 16.15")
    digest = plan["software"]["etcd"].split("repo digest ")[1].split()[0]
    for r in res["runs"]:
        if r["surface"] == "etcd_v3_5_17":
            v = r.get("versions", {})
            if v.get("etcdserver") != "3.5.17" or not any(digest in d for d in v.get("repo_digests", [])):
                problems.append(f"{r['condition']}-{r['repeat']}-{r['mechanism']}: etcd version or image digest differs")
                break
    for k, pkg in plan["software"]["postgresql_package_sha256"].items():
        if sw.get("postgresql_packages", {}).get(k) != pkg:
            problems.append(f"package {k}: sha256 differs from the plan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    run = ap.parse_args().run
    res = json.loads((run / "results.json").read_text())
    plan = load_plan()
    problems, findings, facts = [], [], {}
    listed = {}
    for line in (run / "SHA256SUMS").read_text().splitlines():
        h, name = line.split(None, 1)
        listed[name.strip()] = h
        if not (run / name.strip()).is_file() or sha(run / name.strip()) != h:
            problems.append(f"{name.strip()}: sha256 mismatch or missing")
    present = {p.name for p in run.iterdir() if p.is_file() and p.name != "SHA256SUMS"}
    if present != set(listed):
        problems.append(f"files not covered by SHA256SUMS: {sorted(present ^ set(listed))}")
    for key, p in (("plan_sha256", PLAN), ("script_sha256", PROBE), ("validator_sha256", Path(__file__).resolve()),
                   ("e6_probe_sha256", E6_PROBE)):
        if res.get(key) != sha(p):
            problems.append(f"{key} does not match {p.relative_to(ROOT)}")
    if res.get("e6_probe_sha256") != plan["base_experiment"]["probe_sha256"]:
        problems.append("E6 probe that ran is not the promoted E6 version")
    if res.get("aborted"):
        problems.append(f"run aborted: {res['aborted'][:300]}")
    check_plan_commit(res, problems, facts)
    check_versions(plan, res, problems)
    check_run(plan, res, problems, findings)
    summary = analyze(plan, res)
    if summary != res.get("summary"):
        problems.append("recorded summary differs from recomputation from raw rows")
    lead = {s: (f"{v['strongest_class']}" + (f" via {', '.join(v['via'])}" if v.get("via") else "") +
                (f"; E6 mechanism {v['e6_mechanism']} -> {v['e6_mechanism_class']} (E6 label {v['e6_label']})" if "e6_mechanism" in v else "") +
                ("" if v.get("executed", True) else " [not executed]"))
            for s, v in summary["surface_lead"].items()}
    det = {m: f"{d['class']} | {d['outcome']} | deterministic={d['deterministic']} ({d['runs']} runs) | forge refused {d['forge_refused']}/{d['runs']}"
           for m, d in summary["deterministic"].items()}
    rates = {f"{c}.{m}": ({e: f"{v['k']}/{v['n']} [{v['ci95'][0]}, {v['ci95'][1]}]" for e, v in d["events"].items()} if d["n"]
                          else f"no events: fence raise failed in {d['fence_failed']}/{d['runs']} runs")
             for c, ms in summary["race"].items() for m, d in ms.items()}
    print(json.dumps({"run": str(run), "smoke": bool(res.get("smoke")), **facts, "started_utc": res.get("started_utc"),
                      "surface_lead_class": lead, "deterministic_class_outcome": det,
                      "race_rates_k_n_ci95": rates, "problems": problems, "findings": findings}, indent=1))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
