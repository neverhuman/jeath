#!/usr/bin/env python3
"""Check actual results, reference oracles and source preservation; no favorable ML outcome required."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(directory, name):
    return json.loads((directory / f"{name}.json").read_text())


def check(directory):
    assert load(directory, "complete")["complete"] is True
    rows = load(directory, "model")
    full = next(r for r in rows if r["enforcer"] == "complete")
    assert (full["states"], full["checks"], full["positive_tasks_passed"]) == (10180, 223960, 20)
    for r in rows:
        assert r["checks"] == 22 * r["states"]
        assert r["positive_tasks_total"] == 20
        if r["enforcer"] in ("complete", "reject_all") or not r["budget_in_specification"]:
            assert not r["declared_violations"]
        else:
            assert r["declared_violations"]
    omitted = next(r for r in rows if not r["budget_in_specification"])
    assert (omitted["states"], omitted["checks"], omitted["max_accepted_world_spend"]) == (41732, 918104, 8)
    assert "budget" in omitted["reference_violations"]
    runtime = load(directory, "runtime")
    assert len(runtime["scenarios"]) == 8
    assert tuple(map(int, runtime["sqlite_version"].split("."))) >= (3, 51, 3)
    assert runtime["race"]["executed"] == 1 and runtime["race"]["denials"] == {"Budget": 7}
    for benchmark in [runtime["race"], *runtime["throughput"]]:
        assert len(benchmark["operation_latency_us"]) == benchmark["attempts"]
        assert all(math.isfinite(x) and x >= 0 for x in benchmark["operation_latency_us"])
    for audit in ([r["audit"] for r in runtime["scenarios"]] +
                  [runtime["race"]["audit"]] + [r["audit"] for r in runtime["throughput"]]):
        assert audit["integrity"] == "ok"
        assert 0 <= audit["spent"] == audit["observed_units"] == audit["receipt_units"] <= audit["budget"]
    effects = load(directory, "observed-effects")
    for name, tables in effects.items():
        policy = tables["policy"][0]
        observed = sum(a["units"] for a in tables["accounts"]) + len(tables["deliveries"]) + sum(o["revision"] for o in tables["objects"])
        assert observed == policy["spent"] == sum(r["units"] for r in tables["receipts"]), name
        ids = [r["id"] for r in tables["receipts"]]
        assert len(ids) == len(set(ids)), name
    campaigns = load(directory, "campaigns")
    assert len(campaigns) == 6
    assert sum(c["unauthorized_proposals"] for c in campaigns) == 4
    assert all(c["unauthorized_effects"] == 0 for c in campaigns)
    assert all(c["useful_task_completed"] for c in campaigns[:5]) and not campaigns[5]["useful_task_completed"]
    for r in load(directory, "graphs"):
        assert r["paths"] == 65536 and r["remaining_paths"] == 0 and r["remaining_with_unreported_bypass"] == 1
    monitor = load(directory, "monitors")
    for r in monitor["tasks"]:
        assert all(0 <= x <= 1 for x in r["single_miss"] + r["joint_miss"])
        assert all(a >= b for a, b in zip(r["joint_miss"], r["joint_miss"][1:]))
        assert abs(math.prod(r["single_miss"]) - r["product_prediction"]) < 1e-12
        assert abs(r["all_miss_count"] / r["episodes"] - r["joint_miss"][-1]) < 1e-12
        chain = 1.0
        for rate in r["conditional_miss"]:
            if rate is None:
                assert chain == 0
                break
            chain *= rate
        assert abs(chain - r["joint_miss"][-1]) < 1e-12
    assert abs(load(directory, "statistics")["zero_of_3000_upper95"] - (1 - 0.05 ** (1 / 3000))) < 1e-14
    print(f"validated {directory}: model, effects, lifecycle, graph and statistical accounting")


def originals():
    for row in json.loads((ROOT / "tips/manifest.json").read_text()):
        data = (ROOT / "tips" / row["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"] and len(data) == row["bytes"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--source", action="store_true", help="also require the recorded source hashes to match this checkout")
    p.add_argument("--source-ref", help="validate preserved source hashes against an explicit git revision")
    a = p.parse_args()
    originals()
    check(a.directory)
    assert not (a.source and a.source_ref), "choose current source or a preserved revision"
    if a.source or a.source_ref:
        provenance = load(a.directory, "provenance")
        for filename, expected in provenance["source_sha256"].items():
            data = subprocess.check_output(["git","show",f"{a.source_ref}:{filename}"],cwd=ROOT) if a.source_ref else (ROOT/filename).read_bytes()
            assert hashlib.sha256(data).hexdigest() == expected, f"Reference evidence uses different source: {filename}"
        print("recorded source hashes match " + (a.source_ref or "this checkout"))
