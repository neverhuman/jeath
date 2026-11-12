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


def originals(study=None):
    all_rows = json.loads((ROOT / "tips/manifest.json").read_text())
    rows = [row for row in all_rows if row.get("study") == study]
    if study is None:
        tips_dir = ROOT / "tips"
        manifest_files = [row["file"] for row in rows]
        actual_files = sorted(
            p.name for p in tips_dir.iterdir() if p.is_file() and p.name != "manifest.json"
        )
        assert len(manifest_files) == len(set(manifest_files)), "duplicate file in tips/manifest.json"
        assert sorted(manifest_files) == actual_files, (
            f"tips/manifest.json files {sorted(manifest_files)} != actual tips/ files {actual_files}"
        )
    for row in rows:
        path = ROOT / row["path"] if "path" in row else ROOT / "tips" / row["file"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"] and len(data) == row["bytes"], (
            f"Archival file mismatch: {row['file']}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--source", action="store_true", help="also require the recorded source hashes to match this checkout")
    p.add_argument("--source-ref", help="validate preserved source hashes against an explicit git revision")
    p.add_argument("--study", choices=["performance-before-outcomes", "minds-we-choose"])
    p.add_argument("--compare", type=Path, help="compare a study rerun with a second run or reference directory")
    a = p.parse_args()
    if a.study == "minds-we-choose":
        import shutil
        from brain.validate_minds import check as check_minds

        if a.source_ref:
            p.error("--source-ref applies to the default study")
        originals("performance-before-outcomes")
        originals(a.study)
        data = load(a.directory, a.study)
        check_minds(data)
        if a.source:
            for filename, expected in data["analysis_sha256"].items():
                assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
            if "fossil_environment" in data:
                from brain.minds_environment import analyze as environment
                env_dir = ROOT / "target/minds-we-choose/inputs/environment"
                cached_lr04 = env_dir / "LR04stack.txt"
                local_lr04 = ROOT / "evidence/paleoclimate/LR04stack.txt"
                if not cached_lr04.exists() and local_lr04.exists():
                    env_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(local_lr04, cached_lr04)
                assert data["fossil_environment"] == environment(ROOT / "target/minds-we-choose/inputs"), (
                    "Environmental source reconstruction differs"
                )
        if a.compare:
            other = load(a.compare, a.study)
            # Snapshot timestamps are evidence, rather than numerical results.
            def numerical(value):
                if isinstance(value, dict):
                    return {k: numerical(v) for k, v in value.items()
                            if k not in {"recorded_utc", "snapshot_utc", "generated_utc", "created_utc", "analysis_run_utc", "execution"}}
                if isinstance(value, list):
                    return [numerical(v) for v in value]
                return value
            assert numerical(data) == numerical(other), "Study rerun differs beyond timestamps"
        print("minds-we-choose evidence and archived sources validated")
        raise SystemExit(0)
    if a.study == "performance-before-outcomes":
        from validate_performance import check as check_performance

        if a.source_ref:
            p.error("--source-ref applies to the default study")
        originals(a.study)
        data = load(a.directory, a.study)
        check_performance(data)
        if a.source:
            for filename, expected in data["analysis_sha256"].items():
                assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
        if a.compare:
            other = load(a.compare, a.study)
            assert data["input_sha256"] == other["input_sha256"], "Saved platform inputs changed"
            assert data["speed_dating"]["input_sha256"] == other["speed_dating"]["input_sha256"], "Speed-dating input changed"
            assert data == other, "Aggregate outputs differ"
        print(f"validated {a.directory}: source hashes, archive preservation, counts, denominators and sensitivities")
        if a.compare:
            print(f"same inputs and aggregate output as {a.compare}")
        raise SystemExit(0)
    if a.compare:
        p.error("--compare requires --study")
    originals()
    check(a.directory)
    assert not (a.source and a.source_ref), "choose current source or a preserved revision"
    if a.source or a.source_ref:
        provenance = load(a.directory, "provenance")
        for filename, expected in provenance["source_sha256"].items():
            data = subprocess.check_output(["git","show",f"{a.source_ref}:{filename}"],cwd=ROOT) if a.source_ref else (ROOT/filename).read_bytes()
            assert hashlib.sha256(data).hexdigest() == expected, f"Reference evidence uses different source: {filename}"
        print("recorded source hashes match " + (a.source_ref or "this checkout"))
