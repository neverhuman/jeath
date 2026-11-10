"""Independent sequential protocol oracle, operating on recorded RPC histories.

The fault harness has one outstanding call per service in these histories.
Concurrent benchmarks use independent final-state and effect checks separately.
"""
from copy import deepcopy
import json

MAX_RPC_BYTES = 65536


def required(p, issuers=None):
    n = len(p.get("epochs", [0, 0])) if issuers is None else issuers
    if n <= 0:
        return []
    kind = p["effect"]["kind"]
    if kind == "transfer":
        return [0]
    if kind == "edit":
        return [0, 1] if n > 1 else [0]
    return list(range(n))


def base_required_mask(p, n):
    req_idx = set(required(p, n))
    return [j in req_idx for j in range(n)]


def normalize_dispatched_permit(permit, n):
    if not isinstance(permit, dict) or not ({"actor", "proposal"} <= set(permit) <= {"actor", "proposal", "required"}):
        return None, None
    p = permit["proposal"]
    base = base_required_mask(p, n)
    raw_req = permit.get("required")
    if raw_req is None or raw_req == []:
        eff = base
    else:
        if (
            not isinstance(raw_req, list)
            or len(raw_req) != n
            or any(not isinstance(b, bool) for b in raw_req)
            or any(base[j] and not raw_req[j] for j in range(n))
        ):
            return None, None
        eff = list(raw_req)
    norm = {"actor": permit["actor"], "proposal": deepcopy(p)}
    if eff != base:
        norm["required"] = eff
    return norm, eff


def kind_index(effect):
    return {"transfer": 0, "release": 1, "edit": 2}[effect["kind"]]


def scope_vec(effect):
    s = [False, False, False]
    s[kind_index(effect)] = True
    return s


def min_required_for_scope(scope, n):
    if n <= 0:
        return []
    req = [False] * n
    if len(scope) > 1 and scope[1]:
        return [True] * n
    if len(scope) > 2 and scope[2]:
        req[0] = True
        if n > 1:
            req[1] = True
        return req
    if scope and scope[0]:
        req[0] = True
    return req


def charge(p):
    return p["effect"].get("units", 1)


def scoped(p):
    e = p["effect"]
    if e["kind"] == "transfer":
        return e["recipient"] == "approved" and e["units"] > 0
    if e["kind"] == "release":
        return e["destination"] == "approved" and e["contents"] == "SYNTHETIC DATA: no personal information"
    return e["document"] == "toy.rs" and len(e["contents"].encode("utf-8")) <= 4096


def valid_utf8_ident(s):
    return isinstance(s, str) and 0 < len(s.encode("utf-8")) <= 128


def current_grants_ok(req_mask, grant_set, live, epochs, requested):
    n = len(req_mask)
    if len(live) != n or len(epochs) != n or len(requested) != n:
        return False
    return all(
        (i in grant_set and live[i] and epochs[i] == requested[i])
        for i, needed in enumerate(req_mask)
        if needed
    )


def attenuates_ok(p_budget, c_budget, p_expires, c_expires, p_scope, c_scope, p_req, c_req):
    if len(p_scope) != len(c_scope) or len(p_req) != len(c_req):
        return False
    if c_budget > p_budget or c_expires > p_expires:
        return False
    if any(cs and not ps for ps, cs in zip(p_scope, c_scope)):
        return False
    if any(pr and not cr for pr, cr in zip(p_req, c_req)):
        return False
    return True


def unfenced_ok(req_mask, total_issuers, requested_epochs, floors):
    if len(req_mask) != total_issuers or len(requested_epochs) != total_issuers or len(floors) != total_issuers:
        return False
    return all(requested_epochs[i] >= floors[i] for i, needed in enumerate(req_mask) if needed)


class Reference:
    def __init__(self):
        self.budget = self.charged = 0
        self.broker_init = False
        self.sink_init = False
        self.epochs = [0, 0]
        self.live = [True, True]
        self.approvals = {}
        self.delegations = {}
        self.reserved = {}
        self.floors = [0, 0]
        self.accepted = {}
        self.effects = []
        self.revision = 0

    def _delegation_chain(self, did):
        chain = [did]
        cur = did
        for _ in range(len(self.delegations) + 1):
            parent = self.delegations[cur]["parent"]
            if parent is None:
                return chain, None
            if parent in self.delegations:
                assert parent not in chain, "delegation cycle"
                chain.append(parent)
                cur = parent
            elif parent in self.approvals:
                return chain, parent
            else:
                raise KeyError("unknown delegation ancestor")
        raise RuntimeError("delegation depth exceeded")

    def _approval_delegated_charged(self, approval_id):
        return sum(d["charged"] for d in self.delegations.values() if d["parent"] == approval_id)

    def step(self, role, q):
        op = q["op"]
        if op == "init":
            issuers = q.get("issuers", 2)
            if not isinstance(issuers, int) or isinstance(issuers, bool) or not (1 <= issuers <= 64):
                return "denied"
            if role == "broker":
                if self.broker_init:
                    return "denied"
                self.budget = q["budget"]
                self.epochs = [0] * issuers
                self.live = [True] * issuers
                self.broker_init = True
            else:
                if self.sink_init:
                    return "denied"
                self.floors = [0] * issuers
                self.sink_init = True
            return "initialized"
        if role == "broker":
            n = len(self.epochs)
            if op == "approve":
                permit, i = q["permit"], q["issuer"]
                if set(permit) != {"actor", "proposal"}:
                    return "denied"
                if not isinstance(i, int) or isinstance(i, bool) or not (0 <= i < n):
                    return "denied"
                p = permit["proposal"]
                if not self.live[i] or p["epochs"] != self.epochs:
                    return "denied"
                if p["id"] in self.delegations:
                    return "denied"
                if p["id"] in self.approvals and self.approvals[p["id"]][0] != permit:
                    return "denied"
                a = self.approvals.setdefault(p["id"], [deepcopy(permit), set()])
                a[1].add(i)
                return "approved"
            if op == "delegate":
                permit = deepcopy(q.get("permit")) if q.get("permit") is not None else None
                if permit is not None and set(permit) != {"actor", "proposal"}:
                    return "denied"
                if permit is not None:
                    if "id" in q and q["id"] is not None and q["id"] != permit["proposal"]["id"]:
                        return "denied"
                    did = permit["proposal"]["id"]
                else:
                    did = q.get("id")
                if not valid_utf8_ident(did):
                    return "denied"
                if permit is not None:
                    if not valid_utf8_ident(permit["actor"]) or not scoped(permit["proposal"]):
                        return "denied"
                if did in self.approvals or did in self.delegations:
                    return "denied"

                if q.get("budget") is not None and permit is not None:
                    if charge(permit["proposal"]) > q["budget"]:
                        return "denied"
                    c_budget = q["budget"]
                elif q.get("budget") is not None:
                    c_budget = q["budget"]
                elif permit is not None:
                    c_budget = charge(permit["proposal"])
                else:
                    return "denied"
                if c_budget <= 0:
                    return "denied"

                if q.get("expires") is not None and permit is not None:
                    if permit["proposal"]["expires"] > q["expires"]:
                        return "denied"
                    c_expires = permit["proposal"]["expires"]
                elif q.get("expires") is not None:
                    c_expires = q["expires"]
                elif permit is not None:
                    c_expires = permit["proposal"]["expires"]
                else:
                    return "denied"

                if q.get("scope") is not None:
                    c_scope = list(q["scope"])
                elif permit is not None:
                    c_scope = scope_vec(permit["proposal"]["effect"])
                else:
                    c_scope = [True, True, True]
                if len(c_scope) != 3 or not any(c_scope):
                    return "denied"
                if permit is not None and not c_scope[kind_index(permit["proposal"]["effect"])]:
                    return "denied"

                parent_id = q.get("parent")
                if parent_id is None:
                    p_budget = self.budget
                    p_expires = (1 << 64) - 1
                    p_scope = [True, True, True]
                    p_req = min_required_for_scope(c_scope, n)
                    p_grants_list = list(q["grants"]) if q.get("grants") is not None else list(self.live)
                    p_epochs = list(q["epochs"]) if q.get("epochs") is not None else list(self.epochs)
                    if len(p_grants_list) != n or len(p_epochs) != n:
                        return "denied"
                    for j in range(n):
                        if p_grants_list[j] and (not self.live[j] or p_epochs[j] != self.epochs[j]):
                            return "denied"
                    p_grants = {j for j, g in enumerate(p_grants_list) if g}
                elif parent_id in self.delegations:
                    chain, root_app = self._delegation_chain(parent_id)
                    if root_app is not None:
                        pa = self.approvals[root_app]
                        a_req = base_required_mask(pa[0]["proposal"], n)
                        if not current_grants_ok(a_req, pa[1], self.live, self.epochs, pa[0]["proposal"]["epochs"]):
                            return "denied"
                        if root_app in self.reserved and self.reserved[root_app][1] != "cancelled":
                            return "denied"
                    for anc_id in chain:
                        anc = self.delegations[anc_id]
                        if not current_grants_ok(anc["required"], anc["grants"], self.live, self.epochs, anc["epochs"]):
                            return "denied"
                    if parent_id in self.reserved and self.reserved[parent_id][1] != "cancelled":
                        return "denied"
                    pd = self.delegations[parent_id]
                    p_budget = pd["budget"]
                    p_expires = pd["expires"]
                    p_scope = list(pd["scope"])
                    p_req = list(pd["required"])
                    p_grants = set(pd["grants"])
                    p_epochs = list(pd["epochs"])
                elif parent_id in self.approvals:
                    pa = self.approvals[parent_id]
                    a_req = base_required_mask(pa[0]["proposal"], n)
                    if not current_grants_ok(a_req, pa[1], self.live, self.epochs, pa[0]["proposal"]["epochs"]):
                        return "denied"
                    if parent_id in self.reserved and self.reserved[parent_id][1] != "cancelled":
                        return "denied"
                    p_budget = charge(pa[0]["proposal"])
                    p_expires = pa[0]["proposal"]["expires"]
                    p_scope = scope_vec(pa[0]["proposal"]["effect"])
                    p_req = a_req
                    p_grants = set(pa[1])
                    p_epochs = list(pa[0]["proposal"]["epochs"])
                else:
                    return "denied"

                min_c_req = min_required_for_scope(c_scope, n)
                if q.get("required") is not None:
                    c_req = list(q["required"])
                else:
                    c_req = [p_req[j] or min_c_req[j] for j in range(n)]
                if len(c_req) != n or any(min_c_req[j] and not c_req[j] for j in range(n)):
                    return "denied"

                if not attenuates_ok(p_budget, c_budget, p_expires, c_expires, p_scope, c_scope, p_req, c_req):
                    return "denied"

                if permit is not None:
                    c_epochs = list(permit["proposal"]["epochs"])
                elif q.get("epochs") is not None:
                    c_epochs = list(q["epochs"])
                else:
                    c_epochs = list(p_epochs)

                if q.get("grants") is not None:
                    g_list = list(q["grants"])
                    if len(g_list) != n:
                        return "denied"
                    for j in range(n):
                        if g_list[j] and j not in p_grants and (
                            not self.live[j] or len(c_epochs) <= j or c_epochs[j] != self.epochs[j]
                        ):
                            return "denied"
                    c_grants = {j for j, g in enumerate(g_list) if g}
                else:
                    c_grants = set(p_grants)

                if not current_grants_ok(c_req, c_grants, self.live, self.epochs, c_epochs):
                    return "denied"

                self.delegations[did] = {
                    "id": did,
                    "parent": parent_id,
                    "permit": permit,
                    "budget": c_budget,
                    "charged": 0,
                    "expires": c_expires,
                    "scope": c_scope,
                    "required": c_req,
                    "grants": c_grants,
                    "epochs": c_epochs,
                }
                return "delegated"
            if op == "submit":
                p = q["proposal"]
                actor = q.get("session_actor")
                if not isinstance(actor, str):
                    return "denied"
                permit = {"actor": actor, "proposal": p}
                key = p["id"]
                if key in self.reserved:
                    prev, status = self.reserved[key]
                    prev_base = {"actor": prev["actor"], "proposal": prev["proposal"]}
                    return status if prev_base == permit else "denied"
                if not valid_utf8_ident(key) or not valid_utf8_ident(actor) or not scoped(p):
                    return "denied"
                amount = charge(p)
                if key in self.delegations and self.delegations[key]["permit"] is not None:
                    chain, root_app = self._delegation_chain(key)
                    if root_app is not None:
                        pa = self.approvals[root_app]
                        a_req = base_required_mask(pa[0]["proposal"], n)
                        if not current_grants_ok(a_req, pa[1], self.live, self.epochs, pa[0]["proposal"]["epochs"]):
                            return "denied"
                        if root_app in self.reserved and self.reserved[root_app][1] != "cancelled":
                            return "denied"
                        used = self._approval_delegated_charged(root_app)
                        if amount <= 0 or amount > charge(pa[0]["proposal"]) - used:
                            return "denied"
                    leaf = self.delegations[key]
                    if leaf["permit"] != permit:
                        return "denied"
                    for anc_id in chain:
                        anc = self.delegations[anc_id]
                        if not anc["scope"][kind_index(p["effect"])]:
                            return "denied"
                        if not current_grants_ok(anc["required"], anc["grants"], self.live, self.epochs, anc["epochs"]):
                            return "denied"
                        if amount <= 0 or amount > anc["budget"] - anc["charged"] or q["oracle_time"] >= anc["expires"]:
                            return "denied"
                    if (
                        amount <= 0
                        or amount > self.budget - self.charged
                        or q["oracle_time"] >= p["expires"]
                        or not current_grants_ok(leaf["required"], leaf["grants"], self.live, self.epochs, p["epochs"])
                    ):
                        return "denied"
                    dispatched = deepcopy(permit)
                    if list(leaf["required"]) != base_required_mask(p, n):
                        dispatched["required"] = list(leaf["required"])
                    self.charged += amount
                    for anc_id in chain:
                        self.delegations[anc_id]["charged"] += amount
                    self.reserved[key] = [dispatched, "pending"]
                    return "pending"

                a = self.approvals.get(key)
                if not a or self._approval_delegated_charged(key) > 0:
                    return "denied"
                if a[0] != permit or amount > self.budget - self.charged or amount <= 0:
                    return "denied"
                if q["oracle_time"] >= p["expires"]:
                    return "denied"
                req_mask = base_required_mask(p, n)
                if not current_grants_ok(req_mask, a[1], self.live, self.epochs, p["epochs"]):
                    return "denied"
                self.charged += amount
                self.reserved[key] = [deepcopy(permit), "pending"]
                return "pending"
            if op == "withdraw":
                i = q["issuer"]
                if not isinstance(i, int) or isinstance(i, bool) or not (0 <= i < n):
                    return "denied"
                self.live[i] = False
                self.epochs[i] += 1
                for a in self.approvals.values():
                    a[1].discard(i)
                for d in self.delegations.values():
                    d["grants"].discard(i)
                return "admission_revoked"
            if op == "renew":
                i = q["issuer"]
                if not isinstance(i, int) or isinstance(i, bool) or not (0 <= i < n):
                    return "denied"
                self.live[i] = True
                return "renewed"
            if op == "settle":
                if q["id"] not in self.reserved:
                    return "denied"
                r = self.reserved[q["id"]]
                status = q["sink_status"]
                if r[1] != "pending" or status not in ("committed", "cancelled"):
                    return "denied"
                assert q["id"] in self.accepted and self.accepted[q["id"]][1] == status, "non-authoritative settlement"
                if status == "cancelled":
                    amt = charge(r[0]["proposal"])
                    self.charged -= amt
                    if q["id"] in self.delegations and self.delegations[q["id"]]["permit"] is not None:
                        chain, _ = self._delegation_chain(q["id"])
                        for anc_id in chain:
                            self.delegations[anc_id]["charged"] -= amt
                r[1] = status
                return status
            if op == "audit":
                active = sum(
                    charge(perm["proposal"])
                    for perm, st in self.reserved.values()
                    if st in ("pending", "committed")
                )
                assert active == self.charged <= self.budget
                return "ok"
        else:
            n = len(self.floors)
            if op == "accept":
                norm_permit, eff_req = normalize_dispatched_permit(q["permit"], n)
                if norm_permit is None:
                    return "denied"
                p = norm_permit["proposal"]
                if p["id"] in self.accepted:
                    prev, status = self.accepted[p["id"]]
                    return status if prev == norm_permit else "denied"
                assert p["id"] in self.reserved and self.reserved[p["id"]][0] == norm_permit, "unreserved dispatch"
                if not unfenced_ok(eff_req, n, p["epochs"], self.floors):
                    self.accepted[p["id"]] = [deepcopy(norm_permit), "cancelled"]
                    return "cancelled"
                self.accepted[p["id"]] = [deepcopy(norm_permit), "pending"]
                return "pending"
            if op == "complete":
                if q["id"] not in self.accepted:
                    return "denied"
                r = self.accepted[q["id"]]
                if r[1] == "pending":
                    _, eff_req = normalize_dispatched_permit(r[0], n)
                    p = r[0]["proposal"]
                    e = p["effect"]
                    if not unfenced_ok(eff_req, n, p["epochs"], self.floors) or (
                        e["kind"] == "edit" and e["expected_revision"] != self.revision
                    ):
                        r[1] = "cancelled"
                    else:
                        r[1] = "committed"
                        self.effects.append(deepcopy(r[0]))
                        self.revision += e["kind"] == "edit"
                return r[1]
            if op == "fence":
                i = q["issuer"]
                if not isinstance(i, int) or isinstance(i, bool) or not (0 <= i < n):
                    return "denied"
                self.floors[i] = max(self.floors[i], q["floor"])
                for permit, _ in list(self.accepted.values()):
                    _, eff_req = normalize_dispatched_permit(permit, n)
                    p = permit["proposal"]
                    if self.accepted[p["id"]][1] == "pending" and not unfenced_ok(
                        eff_req, n, p["epochs"], self.floors
                    ):
                        self.accepted[p["id"]][1] = "cancelled"
                return "fenced"
            if op == "audit":
                return "ok"
        if op in ("status", "inspect"):
            return None
        return "denied"


def check(history):
    oracle = Reference()
    for n, event in enumerate(history):
        raw_invocation = event["invocation"]
        if raw_invocation.get("crash") == "before_commit":
            continue
        if len(json.dumps(raw_invocation).encode("utf-8")) > MAX_RPC_BYTES:
            expected = "denied"
        else:
            q = raw_invocation.copy()
            q["oracle_time"] = event["unix_time"]
            expected = oracle.step(event["role"], q)
        if event["response"] is not None and expected is not None:
            assert event["response"]["status"] == expected, (n, expected, event)
            if raw_invocation.get("op") == "audit" and expected == "ok":
                assert event["response"].get("consistent") is True, (n, "audit inconsistent", event)
        if event["role"] == "sink":
            assert event["effects"] == oracle.effects, (n, "independent sink effects differ")
        assert 0 <= oracle.charged <= oracle.budget, (n, "budget")
    return {
        "events": len(history),
        "effects": len(oracle.effects),
        "charged": oracle.charged,
        "checked": True,
        "scope": "sequential RPC fault histories; independent semantic oracle",
    }


def useful(expected, effects):
    """Exact recipient/amount, release bytes/destination or document content."""
    return any(p["proposal"]["effect"] == expected for p in effects)
