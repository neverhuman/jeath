#!/usr/bin/env python3
"""E4b RevokeBench: kills inside sink requests, late accepts across the fence, and a closed-loop emitter.

Predeclared design: fixtures/v6/revokebench_b_plan.json (committed before any E4b data). Each trial runs real
authority-lab broker-v2 and sink-v2 release processes over SQLite, driven by an agent emulator (no LLM), and writes
trial.json beside its own broker and sink databases. Unlike E4 (scripts/revokebench_v6.py, unchanged), this harness

  * records the request outstanding on each service channel, so every kill logs what it interrupted;
  * uses the sink's existing serve-loop hooks (delay_before_commit_ms, crash = after_commit) to end the sink inside
    a `complete` or `fence` transaction; src/ and policy-core/ are not modified;
  * reads the sink database itself: a snapshot after each kill (before restart) and enforcement watermarks taken while
    holding the sink's only RPC channel, so post-enforcement commits and the fence ordering come from the sink's own
    effects table rather than from the harness's event times.

The sink mediates only its own SQLite effects; nothing here is external exactly-once execution.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import queue
import random
import signal
import sqlite3
import statistics
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "fixtures/v6/revokebench_b_plan.json"
BINARY = ROOT / "target/release/authority-lab"
TRAP_DELAY_MS = 600       # delay_before_commit_ms on the trapped request (the hook caps it at 1000)
KILL_AFTER_WRITE_S = 0.2  # SIGKILL this long after the trapped request is written
TRAP_WAIT_S = 15.0        # give up arming the complete trap after this long
TRAPPED_RETRY_DELAY_S = 0.5
HOLD_COUNT, HOLD_LEAD_S, HOLD_RELEASE_S = 8, 4.0, 2.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Crashed(Exception):
    pass


def request_id(q: dict):
    if "id" in q:
        return q["id"]
    for k in ("permit", "proposal"):
        if isinstance(q.get(k), dict):
            p = q[k].get("proposal", q[k])
            return p.get("id")
    return None


class Proc:
    """One authority-lab service speaking JSON lines; tracks its outstanding request; can be killed and restarted."""

    def __init__(self, role: str, db: Path):
        self.role, self.db = role, db
        self.lock = threading.Lock()
        self.alive = threading.Event()
        self.restarts = 0
        self.outstanding = None  # {"op", "id", "t_written"} while a written request has no reply
        self.eofs = []           # requests that saw EOF instead of a reply
        self._start()

    def _start(self):
        self.p = subprocess.Popen([str(BINARY), self.role + "-v2", str(self.db)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.alive.set()

    def call(self, after=None, on_written=None, **q):
        """Send one request. `after` runs while the channel is still held (no other request can reach the sink)."""
        while True:
            if not self.alive.wait(timeout=120):
                raise RuntimeError(f"{self.role} was not restarted")
            with self.lock:
                if not self.alive.is_set():
                    continue
                try:
                    self.p.stdin.write(json.dumps(q) + "\n")
                    self.p.stdin.flush()
                    self.outstanding = {"op": q.get("op"), "id": request_id(q), "t_written": time.monotonic()}
                    if on_written:
                        on_written()
                    line = self.p.stdout.readline()
                except (BrokenPipeError, OSError):
                    line = ""
                if line:
                    self.outstanding = None
                    resp = json.loads(line)
                    return (resp, after()) if after else resp
                if self.outstanding is not None:
                    self.eofs.append({**self.outstanding, "t_eof": time.monotonic()})
                self.alive.clear()
            raise Crashed(self.role)

    def locked(self, fn):
        """Run fn while holding the channel of a live process."""
        while True:
            self.alive.wait()
            with self.lock:
                if self.alive.is_set():
                    return fn()

    def _restart_locked(self, rec: dict, snapshot):
        rec["exit_status"] = self.p.wait(timeout=30)
        rec["eof_seen_for"] = self.eofs[-1] if self.eofs else None
        rec["db_after_kill"] = snapshot()
        self.restarts += 1
        self.outstanding = None
        self._start()

    def kill_and_restart(self, snapshot, t0: float) -> dict:
        out, now = self.outstanding, time.monotonic()
        rec = {"how": "SIGKILL", "t": now - t0,
               "outstanding": None if out is None else {"op": out["op"], "id": out["id"],
                                                        "ms_since_written": (now - out["t_written"]) * 1000}}
        self.alive.clear()
        self.p.send_signal(signal.SIGKILL)
        with self.lock:
            self._restart_locked(rec, snapshot)
        rec["t_restarted"] = time.monotonic() - t0
        return rec

    def await_exit_and_restart(self, snapshot, t0: float, timeout: float = 10.0) -> dict:
        """For crash = after_commit: the process exits by itself; restart it after logging what was outstanding."""
        rec = {"how": "self_exit"}
        try:
            self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rec["how"] = "self_exit_timeout_then_SIGKILL"
            self.alive.clear()
            self.p.send_signal(signal.SIGKILL)
        now = time.monotonic()
        out = self.outstanding
        rec["t"] = now - t0
        rec["outstanding"] = None if out is None else {"op": out["op"], "id": out["id"],
                                                       "ms_since_written": (now - out["t_written"]) * 1000}
        with self.lock:
            self.alive.clear()
            self._restart_locked(rec, snapshot)
        rec["t_restarted"] = time.monotonic() - t0
        return rec

    def close(self):
        with self.lock:
            self.p.stdin.close()
            self.p.wait(timeout=10)


def call_retrying(proc: Proc, **q):
    attempts = 0
    while True:
        try:
            return proc.call(**q), attempts
        except Crashed:
            attempts += 1
            time.sleep(0.01)


def db_snapshot(db: Path, target_id: str | None = None) -> dict:
    """Read-only view of the sink database: effects watermark, issuer-0 floor, and one record's state."""
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)) as c:
        n, mx = c.execute("SELECT count(*), max(sequence) FROM effects").fetchone()
        row = c.execute("SELECT body FROM state WHERE singleton=1").fetchone()
        st = json.loads(row[0]) if row else {"floors": [None], "records": []}
        snap = {"effects_rows": n, "max_sequence": mx or 0, "floor0": st["floors"][0]}
        if target_id is not None:
            snap["target_id"] = target_id
            snap["target_has_effect"] = c.execute("SELECT 1 FROM effects WHERE id=?", (target_id,)).fetchone() is not None
            snap["target_record_status"] = next((r["status"] for r in st["records"]
                                                 if r["permit"]["proposal"]["id"] == target_id), None)
    return snap


def sink_rows(db: Path) -> list[dict]:
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as c:
        return [{"sequence": s, "id": i, "epoch0": json.loads(p)["proposal"]["epochs"][0]}
                for s, i, p in c.execute("SELECT sequence, id, payload FROM effects ORDER BY sequence")]


def state_body(db: Path) -> dict:
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as c:
        return json.loads(c.execute("SELECT body FROM state WHERE singleton=1").fetchone()[0])


def classify_kill(cond: str, rec: dict, trap_id, floor_before) -> str:
    out, snap, eof = rec.get("outstanding"), rec.get("db_after_kill", {}), rec.get("eof_seen_for")
    if cond == "kill_in_complete":
        ok = (rec["how"] == "SIGKILL" and out and out["op"] == "complete" and out["id"] == trap_id and eof
              and eof["op"] == "complete" and eof["id"] == trap_id and snap.get("target_has_effect") is False
              and snap.get("target_record_status") == "pending")
        return "inside_complete_uncommitted" if ok else "other"
    if cond == "kill_in_fence":
        ok = (rec["how"] == "SIGKILL" and out and out["op"] == "fence" and eof and eof["op"] == "fence"
              and snap.get("floor0") == floor_before)
        return "inside_fence_uncommitted" if ok else "other"
    if cond == "exit_after_commit_in_complete":
        ok = (rec["how"] == "self_exit" and rec.get("exit_status") == 87 and out and out["op"] == "complete"
              and out["id"] == trap_id and eof and eof["id"] == trap_id and snap.get("target_has_effect") is True)
        return "after_commit_before_reply" if ok else "other"
    return "other"


def run_trial(cfg: dict, out_root: str) -> dict:
    wl, cond, arm, L, w, C = cfg["workload"], cfg["condition"], cfg["arm"], cfg["L"], cfg["w"], cfg["C"]
    closed = cfg["emitter"] == "closed"
    d = Path(out_root) / cfg["trial_id"]
    d.mkdir(parents=True)
    sink_db = d / "sink.sqlite"
    rng = random.Random(cfg["seed"])
    broker, sink = Proc("broker", d / "broker.sqlite"), Proc("sink", sink_db)
    assert broker.call(op="init", budget=10**6)["status"] == "initialized"
    assert sink.call(op="init")["status"] == "initialized"
    warmup = wl["warmup"]
    Z = wl["think_per_agent_per_C"] * C if closed else None
    if closed:
        pool_size = C * (math.ceil((warmup + L + 5) / Z) + 2)
    else:  # margin of 20 s beyond L: a trap may delay enforcement past decision + L
        pool_size = int((warmup + L + 20) / wl["model_step_seconds"] + 2) * wl["calls_per_step"]
    expires = int(time.time()) + 86400
    proposals = [{"id": f"{cfg['trial_id']}-{n}", "epochs": [0, 0], "expires": expires,
                  "effect": {"kind": "transfer", "recipient": "approved", "units": 1}} for n in range(pool_size)]
    for p in proposals:
        assert broker.call(op="approve", permit={"actor": "alice", "proposal": p}, issuer=0)["status"] == "approved"
    latencies = [min(rng.expovariate(1.0 / w), 5 * w) for _ in proposals]
    offsets = [rng.uniform(0, Z) for _ in range(C)] if closed else []

    events, ev_lock = [], threading.Lock()
    t0 = time.monotonic()

    def log(kind, pid=None, t=None, **kw):
        with ev_lock:
            events.append({"t": (time.monotonic() if t is None else t) - t0, "kind": kind, "id": pid, **kw})

    work: queue.Queue = queue.Queue()
    stop_emitting, enforced, done_emitting = threading.Event(), threading.Event(), threading.Event()
    state = {"in_flight": set(), "queued": 0, "dispatching": set(), "next": 0, "held": 0}
    st_lock = threading.Lock()
    trap = {"armed": False, "fired": False, "id": None, "extra": {}, "written": threading.Event()}
    trap_lock = threading.Lock()

    def settle(pid, status):
        if status in ("committed", "cancelled"):
            call_retrying(broker, op="settle", id=pid, sink_status=status)

    def complete(pid):
        mine = False
        with trap_lock:
            if trap["armed"] and not trap["fired"]:
                trap["fired"], trap["id"], mine = True, pid, True
        if not mine:
            c, retries = call_retrying(sink, op="complete", id=pid)
            return c, retries, False
        try:
            c = sink.call(op="complete", id=pid, on_written=trap["written"].set, **trap["extra"])
            log("trap_answered", pid, status=c.get("status"))  # the kill missed the request
            return c, 0, True
        except Crashed:
            log("trap_eof", pid)
        enforced.wait()
        time.sleep(TRAPPED_RETRY_DELAY_S)
        log("trap_retry", pid)
        c, retries = call_retrying(sink, op="complete", id=pid)
        return c, retries + 1, True

    def dispatch(n):
        p = proposals[n]
        pid = p["id"]
        r, sr = call_retrying(broker, op="submit", proposal=p, session_actor="alice")
        if r.get("status") != "pending":
            with st_lock:
                state["dispatching"].discard(pid)
            log("submit_refused", pid, status=r.get("status"), reason=r.get("reason"), retries=sr)
            return
        held = False
        if cond == "late_accept":
            with st_lock:
                held = (time.monotonic() - t0 >= warmup - HOLD_LEAD_S and state["held"] < HOLD_COUNT
                        and not enforced.is_set())
                state["held"] += held
            if held:
                log("permit_held", pid)
                enforced.wait()
                time.sleep(HOLD_RELEASE_S)
                log("held_accept_delivered", pid)
        a, ar = call_retrying(sink, op="accept", permit=r["permit"])
        if a.get("status") != "pending":
            with st_lock:
                state["dispatching"].discard(pid)
            log("accept_refused", pid, status=a.get("status"), reason=a.get("reason"), retries=ar, held=held)
            settle(pid, a.get("status"))
            return
        with st_lock:
            state["dispatching"].discard(pid)
            state["in_flight"].add(pid)
        log("accepted", pid, retries=ar + sr, held=held)
        time.sleep(latencies[n])
        c, cr, trapped = complete(pid)
        with st_lock:
            state["in_flight"].discard(pid)
        log("completed", pid, status=c.get("status"), retries=cr, trapped=trapped, held=held)
        settle(pid, c.get("status"))

    def model():  # open loop, as E4
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
            with st_lock:
                state["queued"] -= 1
                state["dispatching"].add(proposals[n]["id"])
            dispatch(n)

    def agent(i):  # closed loop: think, emit one call, wait for its terminal result
        if stop_emitting.wait(offsets[i]):
            return
        while True:
            with st_lock:
                if stop_emitting.is_set() or state["next"] >= len(proposals):
                    return
                n = state["next"]
                state["next"] += 1
                state["dispatching"].add(proposals[n]["id"])
            log("emitted", proposals[n]["id"], agent=i)
            dispatch(n)
            if stop_emitting.wait(Z):
                return

    if closed:
        threads = [threading.Thread(target=agent, args=(i,)) for i in range(C)]
    else:
        threads = [threading.Thread(target=model)] + [threading.Thread(target=executor) for _ in range(C)]
    for t in threads:
        t.start()
    time.sleep(warmup)
    log("decision")
    time.sleep(L)
    kills = []

    if cond in ("kill_in_complete", "exit_after_commit_in_complete"):
        extra = {"delay_before_commit_ms": TRAP_DELAY_MS} if cond == "kill_in_complete" else {"crash": "after_commit"}
        with trap_lock:
            trap["armed"], trap["extra"] = True, extra
        log("trap_armed", **extra)
        deadline = time.monotonic() + TRAP_WAIT_S
        fired = False
        while not trap["written"].wait(0.05):
            if time.monotonic() > deadline:
                with trap_lock:
                    if not trap["fired"]:
                        trap["armed"] = False
                        break
                deadline = time.monotonic() + TRAP_WAIT_S  # fired but still waiting for the channel
        else:
            fired = True
        if fired:
            snap = lambda: db_snapshot(sink_db, trap["id"])
            if cond == "kill_in_complete":
                time.sleep(KILL_AFTER_WRITE_S)
                kills.append(sink.kill_and_restart(snap, t0))
            else:
                kills.append(sink.await_exit_and_restart(snap, t0))
            log("sink_restarted")
        else:
            log("trap_not_fired")

    w_start = sink.locked(lambda: db_snapshot(sink_db))
    e_start = time.monotonic()
    log("enforcement_start", t=e_start)
    stop_emitting.set()
    ack = {"halt_ms": (time.monotonic() - e_start) * 1000}
    fence_floor = None
    take = lambda: (db_snapshot(sink_db), time.monotonic())
    if arm in ("revoke_broker", "fence_sink"):
        t = time.monotonic()
        rev, _ = call_retrying(broker, op="withdraw", issuer=0)
        ack["withdraw_ms"] = (time.monotonic() - t) * 1000
        fence_floor = rev["floor"]
    if arm == "fence_sink":
        t = time.monotonic()
        if cond == "kill_in_fence":
            written = threading.Event()

            def killer():
                if written.wait(timeout=TRAP_WAIT_S):
                    time.sleep(KILL_AFTER_WRITE_S)
                    kills.append(sink.kill_and_restart(lambda: db_snapshot(sink_db), t0))
            kt = threading.Thread(target=killer)
            kt.start()
            try:
                f, (w_return, t_ret) = sink.call(op="fence", issuer=0, floor=fence_floor, after=take,
                                                 on_written=written.set, delay_before_commit_ms=TRAP_DELAY_MS)
                log("fence_trap_answered")
            except Crashed:
                log("fence_eof")
                (f, (w_return, t_ret)), _ = call_retrying(sink, op="fence", issuer=0, floor=fence_floor, after=take)
            kt.join()
        else:
            (f, (w_return, t_ret)), _ = call_retrying(sink, op="fence", issuer=0, floor=fence_floor, after=take)
        assert f["status"] == "fenced", f
        ack["fence_ms"] = (time.monotonic() - t) * 1000
    else:
        w_return, t_ret = sink.locked(take)
    ack["total_ms"] = (t_ret - e_start) * 1000
    with st_lock:
        snap = {"in_flight": len(state["in_flight"]), "not_accepted": state["queued"] + len(state["dispatching"])}
    log("enforced", t=t_ret, floor=fence_floor, **ack)
    enforced.set()
    for t in threads:
        t.join()
    audit_b, audit_s = broker.call(op="audit"), sink.call(op="audit")
    broker.close()
    sink.close()

    by = lambda k: [e for e in events if e["kind"] == k]
    t_dec = next(e["t"] for e in events if e["kind"] == "decision")
    t_enf = t_ret - t0
    commits = [e for e in by("completed") if e["status"] == "committed"]
    rows = sink_rows(sink_db)
    final = state_body(sink_db)
    wm_s, wm_r = w_start["max_sequence"], w_return["max_sequence"]
    emitted = [e for e in by("emitted")]
    half = warmup / 2
    lam = sum(1 for e in emitted if half <= e["t"] < t_dec) / max(1e-9, t_dec - half)
    cancelled = sum(1 for e in by("completed") + by("accept_refused") if e["status"] == "cancelled")
    denied = sum(1 for e in by("submit_refused") if e["status"] == "denied")
    floor_before = w_start["floor0"]
    kill_classes = [classify_kill(cond, k, trap["id"], floor_before) for k in kills]
    pos = {r["id"]: r["sequence"] for r in rows}
    rec_status = {r["permit"]["proposal"]["id"]: r["status"] for r in final["records"]}
    held_ids = [e["id"] for e in by("permit_held")]
    accept_of = {e["id"]: e.get("status") for e in by("accept_refused")}
    accept_of.update({e["id"]: "pending" for e in by("accepted")})
    metrics = {
        "post_enforcement_commits_sinkdb": sum(1 for r in rows if r["sequence"] > wm_r),
        "commits_during_enforcement_sinkdb": sum(1 for r in rows if wm_s < r["sequence"] <= wm_r),
        "post_enforcement_commits_harness": sum(1 for e in commits if e["t"] > t_enf),
        "post_decision_commits_harness": sum(1 for e in commits if e["t"] > t_dec),
        "fence_order_violations": [r["id"] for r in rows if arm == "fence_sink" and r["sequence"] > wm_r
                                   and r["epoch0"] < fence_floor],
        "watermark_start": w_start, "watermark_return": w_return,
        "final_floor0": final["floors"][0], "fence_floor": fence_floor,
        "legit_work_lost": cancelled + denied, "cancelled": cancelled, "denied": denied,
        "in_flight_at_enforcement": snap["in_flight"],
        "emitted_not_accepted_at_enforcement": snap["not_accepted"],
        "emitted": len(emitted),
        "rho_design": (wl["offered_rate_per_s"] * w / C) if not closed else w / (Z + w),
        "rho_measured": lam * w / C, "lambda_measured_per_s": lam,
        "enforcement_ack_ms": ack, "enforcement_delay_after_decision_s": (e_start - t0) - t_dec,
        "kills": len(kills), "kill_classes": kill_classes, "trap_id": trap["id"],
        "trap_fired": bool(trap["fired"]),
        "trapped_final_status": next((e["status"] for e in by("completed") if e.get("trapped")), None),
        "held_permits": [{"id": i, "accept_status": accept_of.get(i), "sink_record_status": rec_status.get(i),
                          "effect_sequence": pos.get(i),
                          "effect_after_watermark": (pos[i] > wm_r) if i in pos else None} for i in held_ids],
        "harness_commits": len(commits), "sink_effect_rows": len(rows),
        "duplicate_effects": len(rows) - len({r["id"] for r in rows}),
        "sink_restarts": sink.restarts,
        "retried_calls": sum(e.get("retries", 0) for e in events),
        "audit_consistent": bool(audit_b.get("consistent") and audit_s.get("consistent")),
    }
    result = {"config": cfg, "metrics": metrics, "events": events, "kills": kills}
    (d / "trial.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return {"config": cfg, "metrics": metrics}


def trials(plan: dict) -> list[dict]:
    ol, cl = plan["workload_open_loop"], plan["workload_closed_loop"]
    open_wl = {"model_step_seconds": ol["model_step_seconds"], "calls_per_step": ol["calls_per_step"],
               "offered_rate_per_s": ol["offered_rate_per_s"], "warmup": ol["warmup_seconds_before_decision"]}
    closed_wl = {"think_per_agent_per_C": 0.5, "warmup": cl["warmup_seconds_before_decision"]}
    out = []
    for c in plan["conditions"]:
        for arm, L, (w, C), rep in itertools.product(c["arms"], c["L"], c["cells_w_C"], range(c["replicates"])):
            out.append({"condition": c["name"], "emitter": c["emitter"], "arm": arm, "L": L, "w": w, "C": C,
                        "rep": rep, "workload": open_wl if c["emitter"] == "open" else closed_wl})
    for i, t in enumerate(out):
        t["trial_id"] = (f"b{i:03d}-{t['condition']}-{t['arm']}-{t['emitter']}-L{t['L']}-w{t['w']}-C{t['C']}"
                         f"-r{t['rep']}")
        t["seed"] = 20261212 + i
    return out


def summarize(results: list[dict]) -> list[dict]:
    keys = ("post_enforcement_commits_sinkdb", "commits_during_enforcement_sinkdb", "post_enforcement_commits_harness",
            "legit_work_lost", "cancelled", "denied", "in_flight_at_enforcement",
            "emitted_not_accepted_at_enforcement", "rho_measured", "retried_calls")
    groups = {}
    for r in results:
        c = r["config"]
        groups.setdefault((c["condition"], c["arm"], c["emitter"], c["L"], c["w"], c["C"]), []).append(r["metrics"])
    out = []
    for k, ms in sorted(groups.items()):
        g = {"condition": k[0], "arm": k[1], "emitter": k[2], "L": k[3], "w": k[4], "C": k[5], "trials": len(ms),
             "rho_design": ms[0]["rho_design"]}
        for f in keys:
            xs = [m[f] for m in ms]
            g[f] = {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}
        g["kill_classes"] = sorted(c for m in ms for c in m["kill_classes"])
        g["fence_order_violations"] = sum(len(m["fence_order_violations"]) for m in ms)
        g["trapped_final_status"] = sorted(str(m["trapped_final_status"]) for m in ms)
        g["held_accept_status"] = sorted(str(h["accept_status"]) for m in ms for h in m["held_permits"])
        out.append(g)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", type=Path)
    ap.add_argument("--expected-host", required=True)
    ap.add_argument("--source-commit", required=True, help="git commit of the lane the scripts were copied from")
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--trial-ids", help="smoke test: comma-separated trial ids to run")
    args = ap.parse_args()
    uname = f"{platform.system()} {platform.machine()}"
    assert platform.node() == args.expected_host, "host identity mismatch"
    assert uname == "Linux x86_64", f"unexpected platform {uname}"
    assert BINARY.exists(), "copy target/release/authority-lab first"
    plan = json.loads(PLAN.read_text())
    grid = trials(plan)
    order = list(range(len(grid)))
    random.Random(20261212).shuffle(order)
    if args.trial_ids:
        want = set(args.trial_ids.split(","))
        order = [i for i in order if grid[i]["trial_id"] in want]
        assert len(order) == len(want), "unknown trial id"
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(run_trial, grid[i], str(args.output / "trials")): i for i in order}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            m = r["metrics"]
            with (args.output / "progress.log").open("a") as pl:
                pl.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {r['config']['trial_id']} "
                         f"post_enf_sinkdb={m['post_enforcement_commits_sinkdb']} kills={m['kill_classes']}\n")
    results.sort(key=lambda r: r["config"]["trial_id"])
    summary = {
        "schema_version": 1, "experiment": plan["experiment"], "plan_sha256": sha(PLAN),
        "script_sha256": sha(Path(__file__)), "authority_lab_sha256": sha(BINARY),
        "source_commit": args.source_commit, "smoke_test": bool(args.trial_ids), "amendments": [],
        "summary": summarize(results), "trials": results,
        "provenance": {"hostname": platform.node(), "uname_sm": uname, "started_unix": started,
                       "elapsed_seconds": time.time() - started, "parallel": args.parallel,
                       "python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                       "llm": None, "gpu_seconds_used": 0},
        "disclosures": plan["disclosures"],
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    fence = [r for r in results if r["config"]["arm"] == "fence_sink"]
    print(json.dumps({"trials": len(results),
                      "fence_post_enforcement_sinkdb_max": max((r["metrics"]["post_enforcement_commits_sinkdb"]
                                                                for r in fence), default=None),
                      "fence_order_violations": sum(len(r["metrics"]["fence_order_violations"]) for r in fence),
                      "kill_classes": sorted({c for r in results for c in r["metrics"]["kill_classes"]}),
                      "audit_all_consistent": all(r["metrics"]["audit_consistent"] for r in results)}))


if __name__ == "__main__":
    main()
