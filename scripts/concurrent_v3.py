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


def _event_target_id(inv):
    op=inv.get("op")
    if op=="submit":
        return inv.get("proposal",{}).get("id")
    if op in ("approve","accept"):
        return inv.get("permit",{}).get("proposal",{}).get("id")
    if op in ("complete","settle","status"):
        return inv.get("id")
    if op=="delegate":
        permit=inv.get("permit")
        if isinstance(permit,dict):
            pid=permit.get("proposal",{}).get("id")
            if pid is not None:
                return pid
        return inv.get("id")
    return None


def _is_permanently_non_mutating(oracle, role, inv, resp_status, target_id):
    op=inv.get("op")
    if op in ("status","audit","inspect"):
        return True
    if resp_status in ("denied","unknown"):
        return True
    if role=="broker" and op=="submit" and target_id in oracle.reserved:
        return True
    if role=="sink" and op=="accept" and target_id in oracle.accepted:
        return True
    if role=="sink" and op=="complete" and target_id in oracle.accepted:
        return oracle.accepted[target_id][1]!="pending"
    return False


def _likely_legal_non_mutating(oracle, role, inv, resp_status, target_id):
    op=inv.get("op")
    if role=="broker" and op=="submit" and resp_status=="denied" and target_id not in oracle.reserved:
        amt=inv.get("proposal",{}).get("effect",{}).get("units",1)
        if (oracle.broker_init and target_id in oracle.approvals
                and all(oracle.live) and amt<=oracle.budget-oracle.charged):
            # Likely still admissible right now unless actor/epochs/expiry mismatch; check cheaply:
            app_permit,grants=oracle.approvals[target_id]
            if (app_permit.get("actor")==inv.get("session_actor")
                    and app_permit.get("proposal")==inv.get("proposal")
                    and 0 in grants):
                return False
    return True


def _verify_witness(history, wanted, before, apply_fn, order):
    assert len(order)==len(history), "witness length mismatch"
    seen=set()
    done_mask=0
    all_mask=(1<<len(history))-1
    oracle=Reference()
    for step in order:
        i=step["event"]
        commit=step["committed"]
        assert 0<=i<len(history) and i not in seen, "duplicate or invalid witness index"
        remaining=all_mask^done_mask
        assert not (before[i]&remaining), "witness violates real-time interval precedence"
        event=history[i]
        crash=event["invocation"].get("crash")
        if crash=="before_commit":
            assert commit is False
        elif event["response"] is not None or crash=="after_commit":
            assert commit is True
        apply_fn(oracle,event,commit)
        seen.add(i)
        done_mask|=1<<i
    assert _canonical(oracle.effects)==wanted, "witness final effects mismatch"


def check_concurrent(history, final_effects, max_calls=512, max_nodes=100000):
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

    def search_classic(oracle,remaining,node_cap):
        nonlocal nodes,exhausted
        if not remaining:
            return [] if _canonical(oracle.effects)==wanted else None
        key=(remaining,_canonical(oracle.__dict__))
        if key in visited:
            return None
        if nodes>=node_cap:
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
                suffix=search_classic(next_state,remaining^bit,node_cap)
                if suffix is not None:
                    return [{"event":i,"committed":commit}]+suffix
        return None

    def search_por(history_meta, wanted_effects_canon):
        nonlocal nodes,exhausted
        por_visited=set()
        pipeline_rank={
            ("broker","approve","approved"):0,
            ("broker","delegate","delegated"):1,
            ("broker","submit","pending"):2,
            ("sink","accept","pending"):3,
            ("sink","complete","committed"):4,
            ("broker","settle","committed"):5,
        }

        def step_por(oracle,remaining):
            nonlocal nodes,exhausted
            prefix=[]

            # Iteratively discharge exact diamond-commuting / non-mutating steps
            # without deep recursion or O(N^2) full-state JSON canonicalization.
            while remaining:
                has_lifecycle_change=False
                id_counts={}
                enabled=[]
                rem_scan=remaining
                while rem_scan:
                    lsb=rem_scan&-rem_scan
                    i=lsb.bit_length()-1
                    rem_scan^=lsb
                    if not (before[i]&remaining):
                        enabled.append(i)
                    op_i=history_meta[i]["op"]
                    if op_i in ("withdraw","renew"):
                        has_lifecycle_change=True
                    tid=history_meta[i]["target_id"]
                    if tid is not None and op_i in ("approve","delegate"):
                        id_counts[tid]=id_counts.get(tid,0)+1
                if not enabled:
                    return None

                progressed=False
                for i in enabled:
                    m=history_meta[i]
                    event=history[i]
                    bit=1<<i
                    if m["choices"]==(False,):
                        if event["response"] is None:
                            if nodes>=max_nodes:
                                exhausted=True
                                return None
                            nodes+=1
                            remaining^=bit
                            prefix.append({"event":i,"committed":False})
                            progressed=True
                            break
                        continue
                    if m["choices"]!=(True,):
                        continue
                    if m["op"]=="init" and m["status"]=="initialized":
                        try:
                            next_state=apply(deepcopy(oracle),event,True)
                        except (AssertionError,KeyError,ValueError,TypeError,IndexError):
                            continue
                        if nodes>=max_nodes:
                            exhausted=True
                            return None
                        nodes+=1
                        oracle=next_state
                        remaining^=bit
                        prefix.append({"event":i,"committed":True})
                        progressed=True
                        break
                    if (_is_permanently_non_mutating(oracle,m["role"],m["inv"],m["status"],m["target_id"])
                            and _likely_legal_non_mutating(oracle,m["role"],m["inv"],m["status"],m["target_id"])):
                        if m["op"] in ("status","audit","inspect"):
                            try:
                                apply(oracle,event,True)
                            except (AssertionError,KeyError,ValueError,TypeError,IndexError):
                                continue
                            if nodes>=max_nodes:
                                exhausted=True
                                return None
                            nodes+=1
                            remaining^=bit
                            prefix.append({"event":i,"committed":True})
                            progressed=True
                            break
                        before_snap=_canonical(oracle.__dict__)
                        try:
                            next_state=apply(deepcopy(oracle),event,True)
                        except (AssertionError,KeyError,ValueError,TypeError,IndexError):
                            continue
                        if _canonical(next_state.__dict__)==before_snap:
                            if nodes>=max_nodes:
                                exhausted=True
                                return None
                            nodes+=1
                            remaining^=bit
                            prefix.append({"event":i,"committed":True})
                            progressed=True
                            break
                    if (not has_lifecycle_change and m["role"]=="broker" and m["op"]=="approve"
                            and m["status"]=="approved" and id_counts.get(m["target_id"],0)==1):
                        try:
                            next_state=apply(deepcopy(oracle),event,True)
                        except (AssertionError,KeyError,ValueError,TypeError,IndexError):
                            continue
                        if nodes>=max_nodes:
                            exhausted=True
                            return None
                        nodes+=1
                        oracle=next_state
                        remaining^=bit
                        prefix.append({"event":i,"committed":True})
                        progressed=True
                        break
                if not progressed:
                    break

            if not remaining:
                return prefix if _canonical(oracle.effects)==wanted else None

            state_canon=_canonical(oracle.__dict__)
            key=(remaining,state_canon)
            if key in por_visited:
                return None
            if nodes>=max_nodes:
                exhausted=True
                return None
            por_visited.add(key);nodes+=1

            enabled=[]
            rem_scan=remaining
            while rem_scan:
                lsb=rem_scan&-rem_scan
                i=lsb.bit_length()-1
                rem_scan^=lsb
                if not (before[i]&remaining):
                    enabled.append(i)

            eff_idx=len(oracle.effects)
            next_eff_id=(final_effects[eff_idx]["proposal"]["id"]
                         if eff_idx<len(final_effects) else None)

            def candidate_key(i):
                m=history_meta[i]
                triple=(m["role"],m["op"],m["status"])
                if m["op"]=="init" and m["status"]=="initialized":
                    bucket=0
                elif next_eff_id is not None and m["target_id"]==next_eff_id and triple in pipeline_rank:
                    bucket=1+pipeline_rank[triple]
                elif m["status"] not in ("denied","unknown") and m["op"]!="status":
                    bucket=10
                else:
                    bucket=20
                return (bucket,m["responded_ns"],m["invoked_ns"],i)

            enabled.sort(key=candidate_key)
            for i in enabled:
                m=history_meta[i]
                event=history[i]
                bit=1<<i
                for commit in m["choices"]:
                    prev_eff_len=len(oracle.effects)
                    try:
                        next_state=apply(deepcopy(oracle),event,commit)
                    except (AssertionError,KeyError,ValueError,TypeError,IndexError):
                        continue
                    if len(next_state.effects)>prev_eff_len:
                        if (len(next_state.effects)>len(wanted_effects_canon)
                                or _canonical(next_state.effects[-1])!=wanted_effects_canon[len(next_state.effects)-1]):
                            continue
                    suffix=step_por(next_state,remaining^bit)
                    if suffix is not None:
                        return prefix+[{"event":i,"committed":commit}]+suffix
            return None

        return step_por(Reference(),(1<<len(history))-1)

    full_mask=(1<<len(history))-1
    order=None
    if len(history)<=40:
        classic_cap=min(max_nodes,2048)
        order=search_classic(Reference(),full_mask,classic_cap)
        if order is None and exhausted and max_nodes>classic_cap:
            exhausted=False
            meta=[]
            for ev in history:
                inv=ev["invocation"]
                crash=inv.get("crash")
                choices=(False,) if crash=="before_commit" else (True,)
                if ev["response"] is None and crash not in ("before_commit","after_commit"):
                    choices=(False,True)
                resp=ev["response"]
                meta.append({
                    "role":ev["role"],
                    "inv":inv,
                    "op":inv.get("op"),
                    "status":resp.get("status") if isinstance(resp,dict) else None,
                    "target_id":_event_target_id(inv),
                    "invoked_ns":ev["invoked_ns"],
                    "responded_ns":ev["responded_ns"],
                    "choices":choices,
                })
            order=search_por(meta,[_canonical(e) for e in final_effects])
    else:
        meta=[]
        for ev in history:
            inv=ev["invocation"]
            crash=inv.get("crash")
            choices=(False,) if crash=="before_commit" else (True,)
            if ev["response"] is None and crash not in ("before_commit","after_commit"):
                choices=(False,True)
            resp=ev["response"]
            meta.append({
                "role":ev["role"],
                "inv":inv,
                "op":inv.get("op"),
                "status":resp.get("status") if isinstance(resp,dict) else None,
                "target_id":_event_target_id(inv),
                "invoked_ns":ev["invoked_ns"],
                "responded_ns":ev["responded_ns"],
                "choices":choices,
            })
        order=search_por(meta,[_canonical(e) for e in final_effects])

    if order is None:
        if exhausted:
            return {"checked":False,"status":"search_bound_exhausted","calls":len(history),"nodes":nodes}
        raise AssertionError("recorded RPC history has no legal reference linearization")
    _verify_witness(history,wanted,before,apply,order)
    return {"checked":True,"status":"linearizable_in_declared_scope","calls":len(history),"nodes":nodes,
            "linearization":order,"effects":len(final_effects),
            "scope":"recorded decisions/public status + real-time order + final effects; inspect/audit snapshots excluded"}
