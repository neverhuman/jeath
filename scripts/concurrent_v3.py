"""Bounded linearization search for recorded broker/sink RPC histories.

Uses the separately implemented Python reference, not the Rust transition
methods. Administrative inspect/audit snapshots are not atomic observations and
are excluded from response checking. Public status queries, decision responses,
real-time order and the independent final sink effect log are checked. Requests
near expiry need a clock-aware specification; these fixtures use long leases.
"""
from copy import deepcopy
import json

from reference_v2 import Reference


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),default=lambda v:sorted(v))


def check_concurrent(history, final_effects, max_calls=40, max_nodes=100000):
    if len(history) > max_calls:
        return {"checked":False,"status":"outside_call_bound","calls":len(history),"max_calls":max_calls}
    for event in history:
        assert event["invoked_ns"] <= event["responded_ns"], "invalid invocation interval"
        proposal=event["invocation"].get("proposal", event["invocation"].get("permit",{}).get("proposal",{}))
        if "expires" in proposal:
            # The reference's existing clock observation is at invocation. Reject
            # a history whose call could cross expiry instead of silently treating
            # a processing-time check as an invocation-time check.
            duration=(event["responded_ns"]-event["invoked_ns"])/1e9
            if event["unix_time"] <= proposal["expires"] <= event["unix_time"]+duration+1:
                return {"checked":False,"status":"expiry_window_unsupported","calls":len(history)}
    before=[]
    for i,event in enumerate(history):
        before.append(sum(1<<j for j,other in enumerate(history)
                          if i!=j and other["responded_ns"] < event["invoked_ns"]))
    visited=set()
    nodes=0
    exhausted=False
    wanted=_canonical(final_effects)

    def apply(oracle,event,commit):
        if not commit:
            assert event["response"] is None
            return oracle
        q=deepcopy(event["invocation"])
        q["oracle_time"]=event["unix_time"]
        expected=oracle.step(event["role"],q)
        if q["op"]=="status":
            records=oracle.reserved if event["role"]=="broker" else oracle.accepted
            record=records.get(q["id"])
            if event["role"]=="broker":
                expected=record[1] if record and record[0]["actor"]==q["session_actor"] else "denied"
            else:
                expected=record[1] if record else "unknown"
        response=event["response"]
        if response is not None and expected is not None:
            assert response["status"]==expected
        assert 0 <= oracle.charged <= oracle.budget
        return oracle

    def search(oracle,remaining):
        nonlocal nodes,exhausted
        if not remaining:
            return [] if _canonical(oracle.effects)==wanted else None
        key=(remaining,_canonical(oracle.__dict__))
        if key in visited:
            return None
        if nodes>=max_nodes:
            exhausted=True
            return None
        visited.add(key);nodes+=1
        for i,event in enumerate(history):
            bit=1<<i
            if not remaining&bit or before[i]&remaining:
                continue
            crash=event["invocation"].get("crash")
            choices=(False,) if crash=="before_commit" else (True,)
            if event["response"] is None and crash not in ("before_commit","after_commit"):
                choices=(False,True)
            for commit in choices:
                try:
                    next_state=apply(deepcopy(oracle),event,commit)
                except (AssertionError,KeyError,ValueError,TypeError,IndexError):
                    continue
                suffix=search(next_state,remaining^bit)
                if suffix is not None:
                    return [{"event":i,"committed":commit}]+suffix
        return None

    order=search(Reference(),(1<<len(history))-1)
    if order is None:
        if exhausted:
            return {"checked":False,"status":"search_bound_exhausted","calls":len(history),"nodes":nodes}
        raise AssertionError("recorded RPC history has no legal reference linearization")
    return {"checked":True,"status":"linearizable_in_declared_scope","calls":len(history),"nodes":nodes,
            "linearization":order,"effects":len(final_effects),
            "scope":"recorded decisions/public status + real-time order + final effects; inspect/audit snapshots excluded"}
