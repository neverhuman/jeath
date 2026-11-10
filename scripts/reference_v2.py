"""Independent sequential protocol oracle, operating on recorded RPC histories.

The fault harness has one outstanding call per service in these histories.
Concurrent benchmarks use independent final-state and effect checks separately.
"""
from copy import deepcopy


def required(p):
    return [0] if p["effect"]["kind"] == "transfer" else [0, 1]


def charge(p):
    return p["effect"].get("units", 1)


def scoped(p):
    e = p["effect"]
    if e["kind"] == "transfer":
        return e["recipient"] == "approved" and e["units"] > 0
    if e["kind"] == "release":
        return e["destination"] == "approved" and e["contents"] == "SYNTHETIC DATA: no personal information"
    return e["document"] == "toy.rs" and len(e["contents"].encode()) <= 4096


class Reference:
    def __init__(self):
        self.budget = self.charged = 0
        self.epochs = [0, 0]
        self.live = [True, True]
        self.approvals = {}
        self.reserved = {}
        self.floors = [0, 0]
        self.accepted = {}
        self.effects = []
        self.revision = 0

    def step(self, role, q):
        op = q["op"]
        if op == "init":
            if role == "broker":
                self.budget = q["budget"]
            return "initialized"
        if role == "broker":
            if op == "approve":
                permit, i = q["permit"], q["issuer"]
                p = permit["proposal"]
                if not self.live[i] or p["epochs"] != self.epochs:
                    return "denied"
                if p["id"] in self.approvals and self.approvals[p["id"]][0] != permit:
                    return "denied"
                a = self.approvals.setdefault(p["id"], [deepcopy(permit), set()])
                a[1].add(i)
                return "approved"
            if op == "submit":
                p = q["proposal"]
                permit = {"actor": q["session_actor"], "proposal": p}
                key = p["id"]
                if key in self.reserved:
                    prev, status = self.reserved[key]
                    return status if prev == permit else "denied"
                if not key or len(key) > 128 or not scoped(p):
                    return "denied"
                a = self.approvals.get(key)
                if not a or a[0] != permit or charge(p) > self.budget - self.charged or charge(p) <= 0:
                    return "denied"
                if q["oracle_time"] >= p["expires"]:
                    return "denied"
                if any(i not in a[1] or not self.live[i] or len(p["epochs"]) <= i or p["epochs"][i] != self.epochs[i] for i in required(p)):
                    return "denied"
                self.charged += charge(p)
                self.reserved[key] = [deepcopy(permit), "pending"]
                return "pending"
            if op == "withdraw":
                i = q["issuer"]
                self.live[i] = False
                self.epochs[i] += 1
                for a in self.approvals.values():
                    a[1].discard(i)
                return "admission_revoked"
            if op == "renew":
                self.live[q["issuer"]] = True
                return "renewed"
            if op == "settle":
                r = self.reserved[q["id"]]
                status = q["sink_status"]
                if r[1] != "pending" or status not in ("committed", "cancelled"):
                    return "denied"
                assert q["id"] in self.accepted and self.accepted[q["id"]][1] == status, "non-authoritative settlement"
                if status == "cancelled":
                    self.charged -= charge(r[0]["proposal"])
                r[1] = status
                return status
        else:
            if op == "accept":
                permit = q["permit"]
                p = permit["proposal"]
                if p["id"] in self.accepted:
                    prev, status = self.accepted[p["id"]]
                    return status if prev == permit else "denied"
                assert p["id"] in self.reserved and self.reserved[p["id"]][0] == permit, "unreserved dispatch"
                if any(p["epochs"][i] < self.floors[i] for i in required(p)):
                    self.accepted[p["id"]] = [deepcopy(permit), "cancelled"]
                    return "cancelled"
                self.accepted[p["id"]] = [deepcopy(permit), "pending"]
                return "pending"
            if op == "complete":
                r = self.accepted[q["id"]]
                if r[1] == "pending":
                    p = r[0]["proposal"]
                    e = p["effect"]
                    if any(p["epochs"][i] < self.floors[i] for i in required(p)) or (e["kind"] == "edit" and e["expected_revision"] != self.revision):
                        r[1] = "cancelled"
                    else:
                        r[1] = "committed"
                        self.effects.append(deepcopy(r[0]))
                        self.revision += e["kind"] == "edit"
                return r[1]
            if op == "fence":
                i = q["issuer"]
                self.floors[i] = max(self.floors[i], q["floor"])
                for permit, _ in list(self.accepted.values()):
                    p = permit["proposal"]
                    if self.accepted[p["id"]][1] == "pending" and any(p["epochs"][j] < self.floors[j] for j in required(p)):
                        self.accepted[p["id"]][1] = "cancelled"
                return "fenced"
        if op in ("status", "inspect"):
            return None
        return "denied"


def check(history):
    oracle = Reference()
    for n, event in enumerate(history):
        q = event["invocation"].copy()
        if q.get("crash") == "before_commit":
            continue
        q["oracle_time"] = event["unix_time"]
        expected = oracle.step(event["role"], q)
        if event["response"] is not None and expected is not None:
            assert event["response"]["status"] == expected, (n, expected, event)
        if event["role"] == "sink":
            assert event["effects"] == oracle.effects, (n, "independent sink effects differ")
        assert 0 <= oracle.charged <= oracle.budget, (n, "budget")
    return {"events": len(history), "effects": len(oracle.effects), "charged": oracle.charged,
            "checked": True, "scope": "sequential RPC fault histories; independent semantic oracle"}


def useful(expected, effects):
    """Exact recipient/amount, release bytes/destination or document content."""
    return any(p["proposal"]["effect"] == expected for p in effects)
