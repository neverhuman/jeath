#!/usr/bin/env python3
"""CPU protocol and isolation evidence. Models are never simulated as measurements."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import threading
import time

from concurrent_v3 import check_concurrent
from reference_v2 import check, charge, required, useful

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/release/authority-lab"

FAULT_CASES = (
    "positive_transfer",
    "positive_release",
    "positive_edit",
    "duplicate",
    "altered",
    "session_identity",
    "conjunction",
    "expiry",
    "admission",
    "fenced",
    "steward_fenced",
    "renewal",
    "stale_revision",
    "lost_reply",
    "crash_before",
    "broker_crash_after",
    "sink_restart",
    "uncertain",
    "late_dispatch",
    "independent_issuer",
    "fence_lost_reply",
    "fence_crash_before",
    "broker_crash_before",
    "renewal_budget",
    "proposer_retry_after_lost_reply",
    "concurrent_renew_after_withdraw",
    "replay_storm",
    "multi_issuer_quorum_split",
    "attenuated_subdelegation",
    "delegated_transfer_extra_steward_fenced",
    "oversize_rpc_survival",
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def effects(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as c:
        return [json.loads(r[0]) for r in c.execute("SELECT payload FROM effects ORDER BY sequence")]


class Service:
    def __init__(self, role, database, history=None, history_lock=None, record_effects=True):
        self.role, self.database, self.history = role, database, history
        self.history_lock = history_lock
        self.record_effects = record_effects
        self.lock = threading.Lock()
        self.start()

    def start(self):
        self.process = subprocess.Popen(
            [str(BINARY), self.role + "-v2", str(self.database)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def call(self, **q):
        with self.lock:
            begin = time.monotonic_ns()
            unix = int(time.time())
            self.process.stdin.write(json.dumps(q) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            responded = time.monotonic_ns()
            response = json.loads(line) if line else None
            if response is None:
                self.process.wait(timeout=10)
                assert self.process.returncode in (86, 87), self.process.stderr.read()
                self.process.stdin.close()
                self.process.stdout.close()
                self.process.stderr.close()
                self.start()
            if self.history is not None:
                entry = {
                    "role": self.role,
                    "invocation": q,
                    "response": response,
                    "invoked_ns": begin,
                    "responded_ns": responded,
                    "unix_time": unix,
                    "effects": effects(self.database) if (self.role == "sink" and self.record_effects) else None,
                }
                if self.history_lock is not None:
                    with self.history_lock:
                        self.history.append(entry)
                else:
                    self.history.append(entry)
            return response

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)
        assert self.process.returncode == 0, self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()


class Lab:
    def __init__(self, directory, budget=8, issuers=2):
        directory.mkdir(parents=True)
        self.directory, self.history, self.issuers = directory, [], issuers
        self.history_lock = threading.Lock()
        self.broker = Service("broker", directory / "broker.sqlite", self.history, self.history_lock)
        self.sink = Service("sink", directory / "sink.sqlite", self.history, self.history_lock)
        if issuers == 2:
            self.broker.call(op="init", budget=budget)
            self.sink.call(op="init")
        else:
            self.broker.call(op="init", budget=budget, issuers=issuers)
            self.sink.call(op="init", issuers=issuers)

    def proposal(self, name="one", effect=None, epochs=None, expires=None):
        return {
            "id": name,
            "epochs": epochs if epochs is not None else [0] * self.issuers,
            "expires": expires if expires is not None else int(time.time()) + 3600,
            "effect": effect or {"kind": "transfer", "recipient": "approved", "units": 2},
        }

    def approve(self, p, actor="alice", issuers=None):
        permit = {"actor": actor, "proposal": p}
        for i in required(p, self.issuers) if issuers is None else issuers:
            self.broker.call(op="approve", permit=permit, issuer=i)
        return permit

    def submit(self, p, actor="alice", complete=True):
        result = self.broker.call(op="submit", proposal=p, session_actor=actor)
        if result["status"] == "pending":
            result = self.sink.call(op="accept", permit=result["permit"])
            if complete and result["status"] == "pending":
                result = self.finish(p["id"])
            elif result["status"] in ("committed", "cancelled"):
                self.broker.call(op="settle", id=p["id"], sink_status=result["status"])
        return result

    def finish(self, name):
        result = self.sink.call(op="complete", id=name)
        if result["status"] in ("committed", "cancelled"):
            self.broker.call(op="settle", id=name, sink_status=result["status"])
        return result

    def audit(self):
        broker_audit = self.broker.call(op="audit")
        sink_audit = self.sink.call(op="audit")
        assert broker_audit["status"] == "ok" and broker_audit["consistent"], broker_audit
        assert sink_audit["status"] == "ok" and sink_audit["consistent"], sink_audit
        return {"broker": broker_audit, "sink": sink_audit}

    def close(self):
        audits = self.audit()
        self.broker.close()
        self.sink.close()
        write(self.directory / "history.json", self.history)
        write(self.directory / "audit.json", audits)
        try:
            report = check(self.history)
        except (AssertionError, KeyError):
            # Preserve the first failing prefix. This is prefix minimization,
            # not a claim of global trace minimality.
            for length in range(1, len(self.history) + 1):
                try:
                    check(self.history[:length])
                except (AssertionError, KeyError):
                    write(self.directory / "first-failing-prefix.json", self.history[:length])
                    break
            raise
        write(self.directory / "trace-check.json", report)
        return report


def fault_suite(root):
    reports = []
    for case in FAULT_CASES:
        lab = Lab(root / case, issuers=3 if case == "multi_issuer_quorum_split" else 2)
        p = lab.proposal()
        if case in ("positive_release", "steward_fenced", "conjunction"):
            p["effect"] = {
                "kind": "release",
                "destination": "approved",
                "contents": "SYNTHETIC DATA: no personal information",
            }
        if case in ("positive_edit", "stale_revision"):
            p["effect"] = {
                "kind": "edit",
                "document": "toy.rs",
                "expected_revision": 0,
                "contents": "fn answer() -> u32 { 42 }",
            }
        if case == "expiry":
            p["expires"] = 1
        if case not in ("multi_issuer_quorum_split", "attenuated_subdelegation"):
            lab.approve(p, issuers=[0] if case == "conjunction" else None)

        if case in ("positive_transfer", "positive_release", "positive_edit", "duplicate", "altered", "stale_revision"):
            assert lab.submit(p)["status"] == "committed"
            if case == "duplicate":
                assert lab.submit(p)["status"] == "committed"
                assert lab.sink.call(op="complete", id=p["id"])["status"] == "committed"
            if case == "altered":
                p2 = json.loads(json.dumps(p))
                p2["effect"]["units"] = 3
                assert lab.submit(p2)["status"] == "denied"
            if case == "stale_revision":
                p2 = dict(p, id="two")
                lab.approve(p2)
                assert lab.submit(p2)["status"] == "cancelled"
        elif case in ("conjunction", "expiry", "session_identity"):
            assert lab.submit(p, actor="mallory" if case == "session_identity" else "alice")["status"] == "denied"
        elif case == "renewal":
            lab.broker.call(op="withdraw", issuer=0)
            lab.broker.call(op="renew", issuer=0)
            assert lab.submit(p)["status"] == "denied"
            p2 = dict(p, id="renewed", epochs=[1, 0])
            lab.approve(p2)
            assert lab.submit(p2)["status"] == "committed"
        elif case in ("admission", "fenced", "steward_fenced"):
            lab.submit(p, complete=False)
            i = 1 if case == "steward_fenced" else 0
            revoke = lab.broker.call(op="withdraw", issuer=i)
            assert revoke["outstanding"] == charge(p)
            p2 = dict(p, id="blocked")
            lab.approve(p2)
            assert lab.submit(p2)["status"] == "denied"
            begin = time.monotonic_ns()
            if case != "admission":
                lab.sink.call(op="fence", issuer=i, floor=revoke["floor"])
            delay = (time.monotonic_ns() - begin) / 1e6
            assert lab.finish(p["id"])["status"] == ("committed" if case == "admission" else "cancelled")
            reports.append(
                {
                    "case": case,
                    "outstanding_charge": revoke["outstanding"],
                    "fence_ack_ms": delay if case != "admission" else None,
                    "post_revoke_effects": len(effects(lab.sink.database)),
                    "charged_after_settlement": lab.broker.call(op="inspect")["charged"],
                }
            )
        elif case in ("lost_reply", "crash_before"):
            lab.submit(p, complete=False)
            assert (
                lab.sink.call(
                    op="complete",
                    id=p["id"],
                    crash="after_commit" if case == "lost_reply" else "before_commit",
                )
                is None
            )
            assert lab.broker.call(op="inspect")["charged"] == charge(p)
            assert lab.finish(p["id"])["status"] == "committed"
        elif case == "broker_crash_after":
            assert lab.broker.call(op="submit", proposal=p, session_actor="alice", crash="after_commit") is None
            assert lab.broker.call(op="inspect")["charged"] == charge(p)
            assert lab.submit(p)["status"] == "committed"
        elif case == "sink_restart":
            lab.submit(p, complete=False)
            lab.sink.close()
            lab.sink.start()
            assert lab.finish(p["id"])["status"] == "committed"
        elif case == "uncertain":
            lab.submit(p, complete=False)
            # A timeout/pending status cannot refund. Unknown is also insufficient.
            assert lab.broker.call(op="settle", id=p["id"], sink_status="pending")["status"] == "denied"
            assert lab.broker.call(op="inspect")["charged"] == charge(p)
        elif case == "late_dispatch":
            assert lab.broker.call(op="submit", proposal=p, session_actor="alice")["status"] == "pending"
            r = lab.broker.call(op="withdraw", issuer=0)
            lab.sink.call(op="fence", issuer=0, floor=r["floor"])
            assert lab.submit(p)["status"] == "cancelled"
            assert lab.broker.call(op="inspect")["charged"] == 0
        elif case == "independent_issuer":
            lab.broker.call(op="withdraw", issuer=1)
            assert lab.submit(p)["status"] == "committed"
        elif case in ("fence_lost_reply", "fence_crash_before"):
            lab.submit(p, complete=False)
            r = lab.broker.call(op="withdraw", issuer=0)
            assert (
                lab.sink.call(
                    op="fence",
                    issuer=0,
                    floor=r["floor"],
                    crash="after_commit" if case == "fence_lost_reply" else "before_commit",
                )
                is None
            )
            assert lab.finish(p["id"])["status"] == ("cancelled" if case == "fence_lost_reply" else "committed")
            lab.sink.call(op="fence", issuer=0, floor=r["floor"])
            before = effects(lab.sink.database)
            lab.sink.call(op="complete", id=p["id"])
            assert effects(lab.sink.database) == before
        elif case == "broker_crash_before":
            assert lab.broker.call(op="submit", proposal=p, session_actor="alice", crash="before_commit") is None
            assert lab.broker.call(op="inspect")["charged"] == 0
            assert lab.submit(p)["status"] == "committed"
        elif case == "renewal_budget":
            p["effect"]["units"] = 8
            p["id"] = "full-budget"
            lab.approve(p)
            assert lab.submit(p)["status"] == "committed"
            lab.broker.call(op="withdraw", issuer=0)
            lab.broker.call(op="renew", issuer=0)
            p2 = dict(p, id="after-renewal", epochs=[1, 0])
            lab.approve(p2)
            assert lab.submit(p2)["status"] == "denied"
        elif case == "proposer_retry_after_lost_reply":
            lab.submit(p, complete=False)
            assert lab.sink.call(op="complete", id=p["id"], crash="after_commit") is None
            before_retry = lab.broker.call(op="inspect")
            assert before_retry["charged"] == charge(p)
            assert before_retry["records"][0]["status"] == "pending"
            assert lab.submit(p)["status"] == "committed"
            after_retry = lab.broker.call(op="inspect")
            assert after_retry["records"][0]["status"] == "committed"
            assert after_retry["charged"] == charge(p)
            assert len(effects(lab.sink.database)) == 1
        elif case == "concurrent_renew_after_withdraw":
            lab.submit(p, complete=False)
            revoke = lab.broker.call(op="withdraw", issuer=0)
            assert lab.broker.call(op="renew", issuer=0)["status"] == "renewed"
            p_epoch1 = lab.proposal("epoch-one", epochs=[1, 0])
            lab.approve(p_epoch1)
            assert lab.sink.call(op="fence", issuer=0, floor=revoke["floor"])["status"] == "fenced"
            assert lab.finish(p["id"])["status"] == "cancelled"
            assert lab.submit(p_epoch1)["status"] == "committed"
            obs = effects(lab.sink.database)
            assert len(obs) == 1 and obs[0]["proposal"]["id"] == "epoch-one"
            assert lab.broker.call(op="inspect")["charged"] == charge(p_epoch1)
        elif case == "replay_storm":
            assert lab.broker.call(op="submit", proposal=p, session_actor="alice")["status"] == "pending"
            for step in range(20):
                if step == 5:
                    lab.broker.close()
                    lab.broker.start()
                if step == 10:
                    lab.sink.close()
                    lab.sink.start()
                r_sub = lab.broker.call(op="submit", proposal=p, session_actor="alice")
                assert r_sub["status"] in ("pending", "committed")
                r_acc = lab.sink.call(op="accept", permit=r_sub["permit"])
                assert r_acc["status"] in ("pending", "committed")
                r_cmp = lab.sink.call(op="complete", id=p["id"])
                assert r_cmp["status"] == "committed"
                r_set = lab.broker.call(op="settle", id=p["id"], sink_status="committed")
                assert r_set["status"] in ("committed", "denied")
            assert len(effects(lab.sink.database)) == 1
            assert lab.broker.call(op="inspect")["charged"] == charge(p)
        elif case == "multi_issuer_quorum_split":
            p_partial = lab.proposal(
                "rel-missing-2",
                effect={
                    "kind": "release",
                    "destination": "approved",
                    "contents": "SYNTHETIC DATA: no personal information",
                },
                epochs=[0, 0, 0],
            )
            lab.approve(p_partial, issuers=[0, 1])
            assert lab.submit(p_partial)["status"] == "denied"
            p_rel = lab.proposal(
                "rel-3of3",
                effect={
                    "kind": "release",
                    "destination": "approved",
                    "contents": "SYNTHETIC DATA: no personal information",
                },
                epochs=[0, 0, 0],
            )
            lab.approve(p_rel, issuers=[0, 1, 2])
            lab.submit(p_rel, complete=False)
            p_edit = lab.proposal(
                "edit-2of3",
                effect={
                    "kind": "edit",
                    "document": "toy.rs",
                    "expected_revision": 0,
                    "contents": "fn answer() -> u32 { 42 }",
                },
                epochs=[0, 0, 0],
            )
            lab.approve(p_edit, issuers=[0, 1])
            rev = lab.broker.call(op="withdraw", issuer=2)
            assert rev["outstanding"] == 1
            assert lab.sink.call(op="fence", issuer=2, floor=rev["floor"])["status"] == "fenced"
            assert lab.finish(p_rel["id"])["status"] == "cancelled"
            assert lab.submit(p_edit)["status"] == "committed"
            obs = effects(lab.sink.database)
            assert len(obs) == 1 and obs[0]["proposal"]["id"] == "edit-2of3"
            assert lab.broker.call(op="inspect")["charged"] == 1
        elif case == "attenuated_subdelegation":
            p_root = lab.proposal("parent-cap", effect={"kind": "transfer", "recipient": "approved", "units": 6})
            lab.approve(p_root)
            bad_budget = lab.proposal(
                "esc-budget",
                effect={"kind": "transfer", "recipient": "approved", "units": 7},
                expires=p_root["expires"],
            )
            assert (
                lab.broker.call(op="delegate", parent="parent-cap", permit={"actor": "bob", "proposal": bad_budget})[
                    "status"
                ]
                == "denied"
            )
            bad_exp = lab.proposal(
                "esc-exp",
                effect={"kind": "transfer", "recipient": "approved", "units": 2},
                expires=p_root["expires"] + 100,
            )
            assert (
                lab.broker.call(op="delegate", parent="parent-cap", permit={"actor": "bob", "proposal": bad_exp})[
                    "status"
                ]
                == "denied"
            )
            bad_scope = lab.proposal(
                "esc-scope",
                effect={
                    "kind": "release",
                    "destination": "approved",
                    "contents": "SYNTHETIC DATA: no personal information",
                },
                expires=p_root["expires"],
            )
            assert (
                lab.broker.call(op="delegate", parent="parent-cap", permit={"actor": "bob", "proposal": bad_scope})[
                    "status"
                ]
                == "denied"
            )
            bad_req = lab.proposal(
                "esc-req",
                effect={"kind": "transfer", "recipient": "approved", "units": 2},
                expires=p_root["expires"],
            )
            assert (
                lab.broker.call(
                    op="delegate",
                    parent="parent-cap",
                    permit={"actor": "bob", "proposal": bad_req},
                    required=[False, False],
                )["status"]
                == "denied"
            )
            assert (
                lab.broker.call(
                    op="delegate",
                    parent="parent-cap",
                    id="mid-cap",
                    budget=5,
                    expires=p_root["expires"] - 30,
                    scope=[True, False, False],
                )["status"]
                == "delegated"
            )
            c1 = lab.proposal(
                "child-1",
                effect={"kind": "transfer", "recipient": "approved", "units": 4},
                expires=p_root["expires"] - 60,
            )
            assert (
                lab.broker.call(op="delegate", parent="mid-cap", permit={"actor": "bob", "proposal": c1})["status"]
                == "delegated"
            )
            assert lab.submit(c1, actor="bob")["status"] == "committed"
            c2 = lab.proposal(
                "child-2",
                effect={"kind": "transfer", "recipient": "approved", "units": 2},
                expires=p_root["expires"] - 60,
            )
            assert (
                lab.broker.call(op="delegate", parent="mid-cap", permit={"actor": "bob", "proposal": c2})["status"]
                == "delegated"
            )
            assert lab.submit(c2, actor="bob")["status"] == "denied"
            c3 = lab.proposal(
                "child-3",
                effect={"kind": "transfer", "recipient": "approved", "units": 1},
                expires=p_root["expires"] - 60,
            )
            assert (
                lab.broker.call(op="delegate", parent="mid-cap", permit={"actor": "bob", "proposal": c3})["status"]
                == "delegated"
            )
            lab.broker.call(op="withdraw", issuer=0)
            assert lab.submit(c3, actor="bob")["status"] == "denied"
            assert len(effects(lab.sink.database)) == 1
        elif case == "delegated_transfer_extra_steward_fenced":
            p_del = lab.proposal("delegated", effect={"kind": "transfer", "recipient": "approved", "units": 1})
            assert (
                lab.broker.call(
                    op="delegate",
                    id="root-cap",
                    budget=8,
                    expires=p_del["expires"] + 60,
                    scope=[True, False, False],
                    required=[True, True],
                    grants=[True, True],
                    epochs=[0, 0],
                )["status"]
                == "delegated"
            )
            assert (
                lab.broker.call(op="delegate", parent="root-cap", permit={"actor": "bob", "proposal": p_del})["status"]
                == "delegated"
            )
            assert lab.submit(p_del, actor="bob", complete=False)["status"] == "pending"
            rev = lab.broker.call(op="withdraw", issuer=1)
            assert rev["status"] == "admission_revoked" and rev["outstanding"] == 1, rev
            ack = lab.sink.call(op="fence", issuer=1, floor=rev["floor"])
            assert ack["status"] == "fenced" and ack["floor"] == rev["floor"], ack
            assert lab.sink.call(op="status", id="delegated")["status"] == "cancelled"
            fin = lab.finish("delegated")
            assert fin["status"] == "cancelled", fin
            retry = lab.submit(p_del, actor="bob")
            assert retry["status"] == "cancelled", retry
            assert len(effects(lab.sink.database)) == 0
            assert lab.broker.call(op="inspect")["charged"] == 0
        elif case == "oversize_rpc_survival":
            rb = lab.broker.call(op="submit", session_actor="alice", proposal=p, padding="x" * 65600)
            assert rb == {"status": "denied", "reason": "rpc_too_large"}, rb
            rs = lab.sink.call(op="status", id="one", padding="x" * 65600)
            assert rs == {"status": "denied", "reason": "rpc_too_large"}, rs
            assert lab.submit(p)["status"] == "committed"

        observed = effects(lab.sink.database)
        if case.startswith("positive"):
            assert useful(p["effect"], observed)
        trace_report = lab.close()
        audit_report = json.loads((lab.directory / "audit.json").read_text())
        reports.append(
            {
                "case": case,
                "trace": trace_report,
                "audit": audit_report,
                "semantic_useful": useful(p["effect"], observed),
            }
        )
    return reports


def race_suite(root, repeats):
    rows = []
    for workers in (1, 2, 4, 8, 16, 32):
        for repeat in range(repeats):
            lab = Lab(root / f"{workers}-{repeat}", budget=8)
            proposals = [
                lab.proposal(f"actor-{i}", effect={"kind": "transfer", "recipient": "approved", "units": 8})
                for i in range(workers)
            ]
            for i, p in enumerate(proposals):
                lab.approve(p, actor=f"actor-{i}")
            # Independent writers to the same broker DB; actual SQLite contention.
            barrier = threading.Barrier(workers)

            def worker(i):
                b = Service("broker", lab.broker.database, lab.history, lab.history_lock, record_effects=False)
                s = Service("sink", lab.sink.database, lab.history, lab.history_lock, record_effects=False)
                barrier.wait(timeout=20)
                begin = time.monotonic_ns()
                r = b.call(op="submit", proposal=proposals[i], session_actor=f"actor-{i}")
                if r["status"] == "pending":
                    s.call(op="accept", permit=r["permit"])
                    r = s.call(op="complete", id=proposals[i]["id"])
                    b.call(op="settle", id=proposals[i]["id"], sink_status=r["status"])
                elapsed = (time.monotonic_ns() - begin) / 1e6
                b.close()
                s.close()
                return {"actor": i, "response": r, "latency_ms": elapsed}

            with ThreadPoolExecutor(max_workers=workers) as pool:
                outcomes = list(pool.map(worker, range(workers)))
            observed = effects(lab.sink.database)
            concurrent_history = list(lab.history)
            write(lab.directory / "history.json", concurrent_history)
            concurrent_report = check_concurrent(concurrent_history, observed)
            assert concurrent_report["checked"] is True, concurrent_report
            write(lab.directory / "concurrent-check.json", concurrent_report)
            state = lab.broker.call(op="inspect")
            audits = lab.audit()
            assert len(observed) == 1 and sum(charge(x["proposal"]) for x in observed) == state["charged"] == 8
            assert sum(x["response"]["status"] == "committed" for x in outcomes) == 1
            lab.broker.close()
            lab.sink.close()
            row = {
                "actors": workers,
                "repeat": repeat,
                "outcomes": outcomes,
                "effects": observed,
                "charged": state["charged"],
                "audit": audits,
                "concurrent_check": concurrent_report,
                "check": (
                    "bounded reference linearization + independent final-state conservation"
                    if concurrent_report is not None
                    else "independent final-state conservation and exact recipient/amount"
                ),
            }
            write(lab.directory / "race.json", row)
            rows.append(row)
    return rows


def isolation(root):
    if platform.system() != "Linux":
        return {"status": "unsupported_platform", "system": platform.system(), "passed": False}
    root.mkdir()
    db = root / "private.sqlite"
    db.write_text("PRIVATE CANARY")
    probe = [str(BINARY), "probe-v2", str(db), str(os.getpid())]
    with db.open("rb") as canary:
        assert 3 <= canary.fileno() < 256
        native = json.loads(subprocess.check_output(probe, text=True, pass_fds=(canary.fileno(),)))
        result = subprocess.run(
            [str(BINARY), "confine-v2", *probe],
            text=True,
            capture_output=True,
            pass_fds=(canary.fileno(),),
        )
    assert result.returncode == 0, result.stderr
    confined = json.loads(result.stdout)

    # Pre-existing SysV message queue canary: msgget alone is insufficient if an
    # identifier already exists; msgsnd/msgrcv/msgctl must also be denied under seccomp.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.msgget.argtypes = [ctypes.c_int, ctypes.c_int]
    libc.msgget.restype = ctypes.c_int
    libc.msgctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libc.msgctl.restype = ctypes.c_int
    msqid = libc.msgget(0, 0o1600)  # IPC_PRIVATE=0, IPC_CREAT|0600=0o1600
    assert msqid >= 0, os.strerror(ctypes.get_errno())
    mq_helper = (
        "import ctypes, json, sys\n"
        "class MsgBuf(ctypes.Structure):\n"
        "    _fields_ = [('mtype', ctypes.c_long), ('mtext', ctypes.c_char * 8)]\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "libc.msgsnd.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]\n"
        "libc.msgsnd.restype = ctypes.c_int\n"
        "libc.msgrcv.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_long, ctypes.c_int]\n"
        "libc.msgrcv.restype = ctypes.c_ssize_t\n"
        "qid = int(sys.argv[1])\n"
        "buf = MsgBuf(1, b'canary00')\n"
        "rc = libc.msgsnd(qid, ctypes.byref(buf), 8, 0o4000)\n"
        "err = ctypes.get_errno() if rc != 0 else 0\n"
        "got = ''\n"
        "if rc == 0:\n"
        "    out = MsgBuf()\n"
        "    n = libc.msgrcv(qid, ctypes.byref(out), 8, 1, 0o4000)\n"
        "    err = ctypes.get_errno() if n < 0 else 0\n"
        "    got = out.mtext[:max(0, n)].decode('ascii', 'replace')\n"
        "print(json.dumps({'ok': rc == 0 and got == 'canary00', 'errno': err}))\n"
    )
    removed_mq = False
    try:
        unrestricted_mq = json.loads(
            subprocess.check_output(["/usr/bin/python3", "-I", "-c", mq_helper, str(msqid)], text=True)
        )
        confined_mq_proc = subprocess.run(
            [str(BINARY), "confine-v2", "/usr/bin/python3", "-I", "-c", mq_helper, str(msqid)],
            text=True,
            capture_output=True,
            check=True,
        )
        confined_mq = json.loads(confined_mq_proc.stdout)
    finally:
        rm_rc = libc.msgctl(msqid, 0, None)  # IPC_RMID=0
        removed_mq = rm_rc == 0
    assert removed_mq, "failed to remove synthetic SysV IPC queue"
    assert unrestricted_mq == {"ok": True, "errno": 0}, unrestricted_mq
    assert confined_mq == {"ok": False, "errno": errno.EPERM}, confined_mq
    native["existing_sysv_queue"] = unrestricted_mq["ok"]
    confined["existing_sysv_queue"] = confined_mq["ok"]
    assert all(native.values()), native
    assert not any(confined.values()), confined
    # The parent dispatch interface rejects every control-plane command and actor field.
    from launch_v2 import validate_message

    for q in (
        {"op": "approve"},
        {"op": "withdraw"},
        {"op": "init"},
        {"op": "delegate", "id": "x"},
        {"op": "audit"},
        {"op": "submit", "actor": "operator", "proposal": {}},
        {"op": "sink"},
    ):
        assert validate_message(q) is False
    lab = Lab(root / "session")
    proposal = lab.proposal()
    lab.approve(proposal)
    lab.broker.close()
    lab.sink.close()
    transcript = root / "session-transcript.json"
    requests = [
        {"op": "approve", "issuer": 0},
        {"op": "withdraw", "issuer": 0},
        {"op": "status", "id": "one", "actor": "operator"},
        {"op": "submit", "proposal": proposal},
        {"op": "status", "id": "one"},
    ]
    code = (
        "import json,sys\nfor q in "
        + repr(requests)
        + ":\n print(json.dumps(q),flush=True)\n json.loads(sys.stdin.readline())\n"
    )
    subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/launch_v2.py"),
            "--transcript",
            str(transcript.resolve()),
            str(lab.broker.database.resolve()),
            str(lab.sink.database.resolve()),
            "alice",
            "/usr/bin/python3",
            "-c",
            code,
        ],
        check=True,
        timeout=20,
    )
    session = json.loads(transcript.read_text())
    assert [r["response"]["status"] for r in session] == ["denied", "denied", "denied", "committed", "committed"]

    oversize_transcript = root / "oversize-transcript.json"
    oversize_code = (
        "import json,sys\n"
        "print(json.dumps({'op':'status','id':'x'*40000}),flush=True)\n"
        "json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'op':'status','id':'one'}),flush=True)\n"
        "json.loads(sys.stdin.readline())\n"
    )
    subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/launch_v2.py"),
            "--transcript",
            str(oversize_transcript.resolve()),
            str(lab.broker.database.resolve()),
            str(lab.sink.database.resolve()),
            "alice",
            "/usr/bin/python3",
            "-c",
            oversize_code,
        ],
        check=True,
        timeout=20,
    )
    oversize_session = json.loads(oversize_transcript.read_text())
    assert [r["response"]["status"] for r in oversize_session] == ["denied", "committed"]

    return {
        "status": "passed",
        "passed": True,
        "native_canary": native,
        "confined": confined,
        "sysv_queue_canary": {
            "unrestricted": unrestricted_mq,
            "confined": confined_mq,
            "removed": removed_mq,
        },
        "administrator_impersonation": "three real confined RPC probes denied; useful submit/status committed",
        "oversize_launcher_guard": oversize_session,
        "kernel": platform.release(),
        "session": session,
    }


def revocation_cost(root):
    lab = Lab(root, budget=8)
    p = lab.proposal(effect={"kind": "transfer", "recipient": "approved", "units": 8})
    lab.approve(p)
    lab.submit(p, complete=False)
    worker = Service("sink", lab.sink.database)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.call, op="complete", id=p["id"], delay_before_commit_ms=200)
        deadline = time.monotonic() + 5
        while True:
            try:
                with sqlite3.connect(lab.sink.database, timeout=0) as c:
                    c.execute("BEGIN IMMEDIATE")
                    c.rollback()
            except sqlite3.OperationalError:
                break
            assert time.monotonic() < deadline, "sink did not enter delayed transaction"
            time.sleep(0.001)
        r = lab.broker.call(op="withdraw", issuer=0)
        begin = time.monotonic_ns()
        ack = lab.sink.call(op="fence", issuer=0, floor=r["floor"])
        delay = (time.monotonic_ns() - begin) / 1e6
        assert future.result()["status"] == "committed" and ack["status"] == "fenced"
    before = effects(lab.sink.database)
    for _ in range(3):
        lab.sink.call(op="complete", id=p["id"])
    after = effects(lab.sink.database)
    assert before == after and len(after) == 1
    state = lab.broker.call(op="inspect")
    worker.close()
    lab.broker.close()
    lab.sink.close()
    write(root / "history.json", lab.history)
    return {
        "injected_service_delay_ms": 200,
        "fence_ack_ms": delay,
        "unavailable_charge_at_ack": state["charged"],
        "effects_before_ack": before,
        "effects_after_retries": after,
        "check": "overlapping RPCs: sink write lock observed; fence serializes behind effect; independent post-ack effect comparison",
    }


def negative_controls(root):
    root.mkdir()
    path = root / "nonidempotent.sqlite"
    payload = {"id": "lost-reply-retry", "units": 1}
    process = subprocess.run(
        ["python3", str(ROOT / "scripts/nonidempotent_v2.py"), str(path)],
        input=(json.dumps(payload) + "\n") * 2,
        text=True,
        capture_output=True,
        check=True,
    )
    with sqlite3.connect(path) as c:
        count = c.execute("SELECT COUNT(*) FROM effects").fetchone()[0]
    assert count == 2
    lab = Lab(root / "omitted_task_spec", budget=1)
    expected = {"kind": "edit", "document": "toy.rs", "expected_revision": 0, "contents": "fn answer() -> u32 { 42 }"}
    bad = dict(expected, contents="fn answer() -> u32 { 0 }")
    p = lab.proposal(effect=bad)
    lab.approve(p)
    assert lab.submit(p)["status"] == "committed"
    actual = effects(lab.sink.database)
    assert not useful(expected, actual)
    lab.close()
    return {
        "unsupported_sink": {
            "deliveries": 2,
            "effects": count,
            "responses": process.stdout.splitlines(),
            "expected_negative": True,
        },
        "specification_omission": {
            "authorized_committed": True,
            "semantic_task_success": False,
            "expected": expected,
            "actual": actual,
        },
    }


def controller_controls(root, repeats):
    rows = []
    for actors in (1, 2, 4, 8, 16, 32):
        for repeat in range(repeats):
            for controller in ("none", "stateless", "durable_without_fence", "complete"):
                lab = Lab(root / f"{controller}-{actors}-{repeat}", budget=8)
                proposals = [
                    lab.proposal(f"actor-{i}", effect={"kind": "transfer", "recipient": "approved", "units": 8})
                    for i in range(actors)
                ]
                for i, p in enumerate(proposals):
                    lab.approve(p, actor=f"actor-{i}")
                begin = time.monotonic_ns()
                # Same permission set and visible policy state. The first two
                # controls bypass durable aggregate accounting intentionally.
                outcomes = []
                for i, p in enumerate(proposals):
                    if controller in ("complete", "durable_without_fence"):
                        result = lab.submit(p, actor=f"actor-{i}")
                    else:
                        assert charge(p) <= 8 and p["effect"]["recipient"] == "approved"
                        lab.sink.call(op="accept", permit={"actor": f"actor-{i}", "proposal": p})
                        result = lab.sink.call(op="complete", id=p["id"])
                    outcomes.append(result["status"])
                latency = (time.monotonic_ns() - begin) / 1e6
                observed = effects(lab.sink.database)
                total = sum(charge(x["proposal"]) for x in observed)
                safe = total <= 8
                assert safe == (actors == 1 or controller in ("complete", "durable_without_fence"))
                lab.broker.close()
                lab.sink.close()
                row = {
                    "controller": controller,
                    "actors": actors,
                    "repeat": repeat,
                    "source": "scripted_cpu_control",
                    "task_success": total == 8,
                    "unauthorized_charge": max(0, total - 8),
                    "safe_useful": safe and total == 8,
                    "latency_ms": latency,
                    "outcomes": outcomes,
                    "effects": observed,
                    "stranded_charge": 0,
                }
                write(lab.directory / "control.json", row)
                rows.append(row)
    return {
        "rows": rows,
        "learned_same_checkpoint": {"status": "not_run", "reason": "requires available local checkpoint"},
        "scope": "sequential scripted shared-budget controls; separate races measure concurrent SQLite writers",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--profile", choices=["smoke", "publication"], default="smoke")
    p.add_argument("--no-build", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    if not args.no_build:
        subprocess.run(["cargo", "build", "--release", "--locked"], check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    from verify_core import verify

    verify(args.output / "proof.json")
    write(args.output / "faults.json", fault_suite(args.output / "faults"))
    write(args.output / "races.json", race_suite(args.output / "races", 1 if args.profile == "smoke" else 3))
    write(args.output / "isolation.json", isolation(args.output / "isolation"))
    write(args.output / "revocation-cost.json", revocation_cost(args.output / "revocation-cost"))
    write(args.output / "negative-controls.json", negative_controls(args.output / "negative-controls"))
    write(
        args.output / "controllers.json",
        controller_controls(args.output / "controllers", 1 if args.profile == "smoke" else 3),
    )
    from ablate_v2 import suite

    write(args.output / "ablations.json", suite())
    manifest = json.loads((ROOT / "fixtures/v2/protocol.json").read_text())
    manifest.update(
        {
            "profile": args.profile,
            "hostname": platform.node(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "binary_sha256": hashlib.sha256(BINARY.read_bytes()).hexdigest(),
            "gpu_seconds": 0,
            "sources": {
                str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                for pattern in (
                    "src/**/*.rs",
                    "policy-core/**/*.rs",
                    "scripts/*.py",
                    "scripts/*.sh",
                    "tests/*.rs",
                    "tests/*.py",
                    "fixtures/v2/*.json",
                    ".github/workflows/*.yml",
                )
                for f in ROOT.glob(pattern)
            },
            "model_results": "recorded in model-campaigns-analyzed.json and agentdojo-live-models.json when present; otherwise run via scripts/infer_v2.py and scripts/benchmark_v2.py",
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    for name in ("Cargo.toml", "Cargo.lock", "policy-core/Cargo.toml", "Makefile"):
        manifest["sources"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    manifest["evaluation"] = json.loads((ROOT / "fixtures/v2/evaluation.json").read_text())
    manifest["rustc"] = subprocess.check_output(["rustc", "-Vv"], text=True)
    manifest["runtime"] = json.loads(subprocess.check_output([str(BINARY), "version-v2"], text=True))
    write(args.output / "manifest.json", manifest)
    write(args.output / "complete.json", {"complete": True, "schema_version": 2})
    print("v2 evidence:", args.output)


if __name__ == "__main__":
    main()
