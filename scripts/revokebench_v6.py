#!/usr/bin/env python3
"""E4 RevokeBench: halting inference vs revoking broker admission vs fencing the sink.

Predeclared design: fixtures/v6/revokebench_plan.json. Each trial runs real authority-lab broker-v2 and sink-v2
release processes over SQLite, driven by an agent emulator (a model thread that emits tool calls and C executor
threads that dispatch them; no LLM). At the decision time the operator decides to stop; after an enforcement
latency L the stop is enforced at one of three cumulative layers. Every trial writes trial.json (event log and
metrics) beside its own broker and sink databases, so the validator can re-derive the commits from the sink.

The sink mediates only its own SQLite effects; nothing here is external exactly-once execution.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import queue
import random
import signal
import sqlite3
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "fixtures/v6/revokebench_plan.json"
BINARY = ROOT / "target/release/authority-lab"
ARMS = ("halt_inference", "revoke_broker", "fence_sink")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Crashed(Exception):
    pass


class Proc:
    """One authority-lab service process speaking JSON lines; can be SIGKILLed and restarted on the same database."""

    def __init__(self, role: str, db: Path):
        self.role, self.db = role, db
        self.lock = threading.Lock()
        self.alive = threading.Event()
        self.restarts = 0
        self._start()

    def _start(self):
        self.p = subprocess.Popen([str(BINARY), self.role + "-v2", str(self.db)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.alive.set()

    def call(self, **q):
        while True:
            self.alive.wait()
            with self.lock:
                if not self.alive.is_set():
                    continue
                try:
                    self.p.stdin.write(json.dumps(q) + "\n")
                    self.p.stdin.flush()
                    line = self.p.stdout.readline()
                except (BrokenPipeError, OSError):
                    line = ""
                if line:
                    return json.loads(line)
            raise Crashed(self.role)

    def kill_and_restart(self):
        # SIGKILL first, without the call lock, so the kill can land while a request is being processed;
        # the in-progress call then sees EOF, raises Crashed and retries after the restart.
        self.alive.clear()
        self.p.send_signal(signal.SIGKILL)
        with self.lock:
            self.p.wait()
            self.restarts += 1
            self._start()

    def close(self):
        with self.lock:
            self.p.stdin.close()
            self.p.wait(timeout=10)


def call_retrying(proc: Proc, **q):
    """Retry a call across a crash. The protocol's retry semantics decide what a repeated call does."""
    attempts = 0
    while True:
        try:
            return proc.call(**q), attempts
        except Crashed:
            attempts += 1
            time.sleep(0.01)


def sink_effects(db: Path) -> list[dict]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        return [json.loads(r[0]) for r in c.execute("SELECT payload FROM effects ORDER BY sequence")]


def run_trial(cfg: dict, out_root: str) -> dict:
    wl = cfg["workload"]
    d = Path(out_root) / cfg["trial_id"]
    d.mkdir(parents=True)
    rng = random.Random(cfg["seed"])
    broker, sink = Proc("broker", d / "broker.sqlite"), Proc("sink", d / "sink.sqlite")
    assert broker.call(op="init", budget=10**6)["status"] == "initialized"
    assert sink.call(op="init")["status"] == "initialized"
    expires = int(time.time()) + 86400
    pool_size = int((wl["warmup"] + cfg["L"] + 5) / wl["model_step_seconds"] + 2) * wl["calls_per_step"]
    proposals = [{"id": f"{cfg['trial_id']}-{n}", "epochs": [0, 0], "expires": expires,
                  "effect": {"kind": "transfer", "recipient": "approved", "units": 1}} for n in range(pool_size)]
    for p in proposals:
        assert broker.call(op="approve", permit={"actor": "alice", "proposal": p}, issuer=0)["status"] == "approved"
    latencies = [min(rng.expovariate(1.0 / cfg["w"]), 5 * cfg["w"]) for _ in proposals]
    held = [rng.random() < 0.25 for _ in proposals] if cfg["fault"] == "late_completion" else [False] * len(proposals)

    events, ev_lock = [], threading.Lock()
    t0 = time.monotonic()

    def log(kind, pid=None, **kw):
        with ev_lock:
            events.append({"t": time.monotonic() - t0, "kind": kind, "id": pid, **kw})

    work: queue.Queue = queue.Queue()
    stop_emitting, enforced, done_emitting = threading.Event(), threading.Event(), threading.Event()
    state = {"in_flight": set(), "queued": 0, "dispatching": set()}
    st_lock = threading.Lock()

    def model():
        n = 0
        next_step = time.monotonic()
        while not stop_emitting.is_set() and n + wl["calls_per_step"] <= len(proposals):
            for _ in range(wl["calls_per_step"]):
                with st_lock:
                    state["queued"] += 1
                work.put(n)
                log("emitted", proposals[n]["id"])
                n += 1
            next_step += wl["model_step_seconds"]
            stop_emitting.wait(max(0.0, next_step - time.monotonic()))
        done_emitting.set()

    def executor():
        while True:
            try:
                n = work.get(timeout=0.1)
            except queue.Empty:
                if done_emitting.is_set():
                    return
                continue
            p = proposals[n]
            with st_lock:
                state["queued"] -= 1
                state["dispatching"].add(p["id"])
            r, _ = call_retrying(broker, op="submit", proposal=p, session_actor="alice")
            if r.get("status") != "pending":
                with st_lock:
                    state["dispatching"].discard(p["id"])
                log("submit_refused", p["id"], status=r.get("status"))
                continue
            a, retries = call_retrying(sink, op="accept", permit=r["permit"])
            if a.get("status") != "pending":
                with st_lock:
                    state["dispatching"].discard(p["id"])
                log("accept_refused", p["id"], status=a.get("status"), retries=retries)
                if a.get("status") in ("committed", "cancelled"):
                    call_retrying(broker, op="settle", id=p["id"], sink_status=a["status"])
                continue
            with st_lock:
                state["dispatching"].discard(p["id"])
                state["in_flight"].add(p["id"])
            log("accepted", p["id"])
            time.sleep(latencies[n])
            if held[n] and not enforced.is_set():
                enforced.wait()
                time.sleep(10.0)
                log("held_delivery", p["id"])
            c, retries = call_retrying(sink, op="complete", id=p["id"])
            with st_lock:
                state["in_flight"].discard(p["id"])
            log("completed", p["id"], status=c.get("status"), retries=retries)
            if c.get("status") in ("committed", "cancelled"):
                call_retrying(broker, op="settle", id=p["id"], sink_status=c["status"])

    threads = [threading.Thread(target=model)] + [threading.Thread(target=executor) for _ in range(cfg["C"])]
    for t in threads:
        t.start()
    time.sleep(wl["warmup"])
    log("decision")
    snap = {}
    if cfg["fault"] == "sigkill_sink":
        time.sleep(cfg["L"] / 2 + 0.25)
        sink.kill_and_restart()
        log("sink_sigkill", phase="before_enforcement")
        time.sleep(max(0.0, cfg["L"] / 2 - 0.25))
    else:
        time.sleep(cfg["L"])
    e_start = time.monotonic()
    stop_emitting.set()
    ack = {"halt_ms": (time.monotonic() - e_start) * 1000}
    if cfg["arm"] in ("revoke_broker", "fence_sink"):
        t = time.monotonic()
        rev, _ = call_retrying(broker, op="withdraw", issuer=0)
        ack["withdraw_ms"] = (time.monotonic() - t) * 1000
        if cfg["arm"] == "fence_sink":
            t = time.monotonic()
            f, _ = call_retrying(sink, op="fence", issuer=0, floor=rev["floor"])
            assert f["status"] == "fenced", f
            ack["fence_ms"] = (time.monotonic() - t) * 1000
    ack["total_ms"] = (time.monotonic() - e_start) * 1000
    with st_lock:
        snap = {"in_flight": len(state["in_flight"]), "not_accepted": state["queued"] + len(state["dispatching"])}
    log("enforced", **ack)
    enforced.set()
    if cfg["fault"] == "sigkill_sink":
        time.sleep(0.5)
        sink.kill_and_restart()
        log("sink_sigkill", phase="after_enforcement")
    for t in threads:
        t.join()
    audit_b, audit_s = broker.call(op="audit"), sink.call(op="audit")
    broker.close()
    sink.close()

    by = lambda k: [e for e in events if e["kind"] == k]
    t_dec = next(e["t"] for e in events if e["kind"] == "decision")
    t_enf = next(e["t"] for e in events if e["kind"] == "enforced")
    commits = [e for e in by("completed") if e["status"] == "committed"]
    rows = sink_effects(d / "sink.sqlite")
    pre = [e for e in commits if e["t"] < t_dec]
    window = max(1e-9, t_dec - wl["warmup"] / 2)
    lam = sum(1 for e in pre if e["t"] >= wl["warmup"] / 2) / window
    metrics = {
        "post_decision_commits": sum(1 for e in commits if e["t"] > t_dec),
        "post_enforcement_commits": sum(1 for e in commits if e["t"] > t_enf),
        "in_flight_at_enforcement": snap["in_flight"],
        "emitted_not_accepted_at_enforcement": snap["not_accepted"],
        "cancelled_legitimate": sum(1 for e in by("completed") if e["status"] == "cancelled")
        + sum(1 for e in by("accept_refused") if e["status"] == "cancelled"),
        "denied_after_enforcement": sum(1 for e in by("submit_refused") if e["t"] > t_enf),
        "enforcement_ack_ms": ack,
        "harness_commits": len(commits),
        "sink_effect_rows": len(rows),
        "duplicate_effects": len(rows) - len({r["proposal"]["id"] for r in rows}),
        "sink_restarts": sink.restarts,
        "retried_calls": sum(e.get("retries", 0) for e in events),
        "pre_decision_completion_rate_per_s": lam,
        "little_law_post_decision_prediction": lam * (cfg["L"] + cfg["w"]),
        "audit_consistent": bool(audit_b.get("consistent") and audit_s.get("consistent")),
    }
    result = {"config": cfg, "metrics": metrics, "events": events}
    (d / "trial.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return {"config": cfg, "metrics": metrics}


def trials(plan: dict) -> list[dict]:
    wl = plan["workload"]
    base = {"model_step_seconds": wl["model_step_seconds"], "calls_per_step": wl["calls_per_step"],
            "warmup": wl["warmup_seconds_before_decision"]}
    out = []
    for arm, L, w, C, rep in itertools.product(ARMS, plan["enforcement_latency_L_seconds"], wl["tool_latency_mean_seconds_w"],
                                               wl["subagents_C"], range(plan["replicates"])):
        out.append({"arm": arm, "L": L, "w": w, "C": C, "rep": rep, "fault": "none"})
    for fault, arm, L, rep in itertools.product(("sigkill_sink", "late_completion"), ARMS, (1, 35), range(plan["replicates"])):
        out.append({"arm": arm, "L": L, "w": 5.0, "C": 32, "rep": rep, "fault": fault})
    for i, t in enumerate(out):
        t["trial_id"] = f"t{i:03d}-{t['arm']}-L{t['L']}-w{t['w']}-C{t['C']}-{t['fault']}-r{t['rep']}"
        t["seed"] = 20261114 + i
        t["workload"] = base
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    ap.add_argument("--expected-host", required=True)
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--limit", type=int, help="smoke test: run only the first N trials of a seeded shuffle")
    args = ap.parse_args()
    assert platform.node() == args.expected_host, "host identity mismatch"
    assert BINARY.exists(), "build target/release/authority-lab first"
    plan = json.loads(PLAN.read_text())
    grid = trials(plan)
    order = list(range(len(grid)))
    random.Random(20261114).shuffle(order)
    if args.limit:
        order = order[: args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(run_trial, grid[i], str(args.output / "trials")): i for i in order}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            with (args.output / "progress.log").open("a") as pl:
                pl.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {r['config']['trial_id']} "
                         f"post_enf={r['metrics']['post_enforcement_commits']}\n")
    results.sort(key=lambda r: r["config"]["trial_id"])
    summary = {
        "schema_version": 1, "experiment": plan["experiment"], "plan_sha256": sha(PLAN),
        "script_sha256": sha(Path(__file__)), "authority_lab_sha256": sha(BINARY),
        "smoke_test": bool(args.limit), "trials": results,
        "provenance": {"hostname": platform.node(), "started_unix": started, "elapsed_seconds": time.time() - started,
                       "parallel": args.parallel, "llm": None, "gpu_seconds_used": 0},
        "disclosures": plan["disclosures"],
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    fence = [r for r in results if r["config"]["arm"] == "fence_sink"]
    print(json.dumps({"trials": len(results), "fence_post_enforcement_max": max((r["metrics"]["post_enforcement_commits"] for r in fence), default=None),
                      "duplicates_max": max(r["metrics"]["duplicate_effects"] for r in results),
                      "audit_all_consistent": all(r["metrics"]["audit_consistent"] for r in results)}))


if __name__ == "__main__":
    main()
