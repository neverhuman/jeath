"""Ten abstract protocol mutations, including all 45 pairs.

Search all subsequences of ten declared regression templates, increasing length.
Witnesses are shortest only in this explicit search space, not globally. These
mutations are a protocol model, not alternative verified Rust implementations.
"""
from copy import deepcopy
from itertools import combinations

TEMPLATES = {
    "budget": ["authorize0", "submit0", "complete0", "authorize1", "submit1", "complete1"],
    "binding": ["authorize0", "alter0", "submit0", "complete0"],
    "conjunction": ["operator0", "submit0", "complete0"],
    "epoch": ["authorize0", "withdraw", "renew", "submit0", "complete0"],
    "expiry": ["authorize0", "expire", "submit0", "complete0"],
    "revision": ["authorize0", "external_edit", "submit0", "complete0"],
    "dedup": ["authorize0", "submit0", "complete0", "complete0"],
    "durability": ["authorize0", "submit0", "complete0", "crash", "authorize1", "submit1", "complete1"],
    "quarantine": ["authorize0", "submit0", "timeout", "authorize1", "submit1", "complete1", "complete0"],
    "fencing": ["authorize0", "submit0", "withdraw", "fence", "complete0"],
}


def execute(trace, omitted):
    approved, pending, completed = {}, {}, []
    charged = epoch = 0
    floor = 0
    live, expired, altered, stale = True, False, False, False
    harms = []
    for action in trace:
        key = int(action[-1]) if action[-1].isdigit() else 0
        if action.startswith(("authorize", "operator")):
            approved[key] = {"both":action.startswith("authorize"), "epoch":epoch}
        elif action == "alter0":
            altered = True
        elif action == "expire":
            expired = True
        elif action == "external_edit":
            stale = True
        elif action == "withdraw":
            live = False; epoch += 1
        elif action == "renew":
            live = True
        elif action == "fence":
            floor = epoch
        elif action == "crash" and "durability" in omitted:
            charged = 0
        elif action == "timeout" and "quarantine" in omitted and pending:
            charged = max(0, charged - 1)
        elif action.startswith("submit") and key not in pending:
            a = approved.get(key)
            if not a or not live:
                continue
            failures = {"budget":charged >= 1, "binding":altered, "conjunction":not a["both"],
                        "epoch":a["epoch"] != epoch, "expiry":expired}
            if any(bad and mechanism not in omitted for mechanism,bad in failures.items()):
                continue
            charged += 1
            pending[key] = {"epoch":a["epoch"], "violations":[m for m,bad in failures.items() if bad]}
        elif action.startswith("complete") and key in pending:
            p = pending[key]
            if key in completed and "dedup" not in omitted:
                continue
            if p["epoch"] < floor and "fencing" not in omitted:
                continue
            if stale and "revision" not in omitted:
                continue
            harms += [x for x in p["violations"] if x != "budget"]
            if stale:
                harms.append("revision")
            if key in completed:
                harms.append("duplicate")
            if p["epoch"] < floor:
                harms.append("post_fenced_ack")
            completed.append(key)
            if len(completed) > 1:
                harms.append("budget")
    return sorted(set(harms)), completed


def candidates():
    # Deduplicate equivalent subsequences, then sort by length for minimality in
    # this declared finite language. No transition-state-count claim is made.
    all_traces = {tuple(t[i] for i in indices) for t in TEMPLATES.values() for n in range(len(t)+1)
                  for indices in combinations(range(len(t)), n)}
    return sorted(all_traces, key=lambda t:(len(t),t))


def suite():
    traces = candidates()
    assert not any(execute(t, set())[0] for t in traces)
    assert execute(["authorize0", "submit0", "complete0"], set())[1] == [0]
    rows = []
    mechanisms = list(TEMPLATES)
    for n in (1,2):
        for omitted in combinations(mechanisms,n):
            witnesses = []
            for t in traces:
                harms,_ = execute(t,set(omitted))
                if harms:
                    witnesses.append({"trace":t,"harms":harms})
                    break
            rows.append({"omitted":omitted,"shortest_searched_witness":witnesses[0] if witnesses else None})
    return {"model":"abstract protocol mutation model; not Rust code mutants", "search":"all subsequences of declared templates",
            "templates":TEMPLATES,"searched_traces":len(traces),"rows":rows,"positive_control":True,
            "specification_omission":{"verified_core_admits":"exactly approved edit returning 0",
                                      "harm_oracle":"task expects answer 42; exact contents differ",
                                      "interpretation":"permission correctness does not imply task correctness"}}
