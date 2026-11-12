#!/usr/bin/env python3
"""Run experiments with a source manifest and export observable SQLite effects."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return sorted([*ROOT.glob("src/**/*.rs"), *ROOT.glob("tests/*.rs"),
                   *ROOT.glob("scripts/*.py"), *ROOT.glob("fixtures/*.jsonl"),
                   *ROOT.glob("examples/*.rs"), ROOT / "Cargo.toml", ROOT / "Cargo.lock"])


def run_performance_before_outcomes(output, speed_data):
    from performance_choice import analyze
    from performance_face_attribution import analyze_face_attribution
    from performance_holoface_benchmark import load_or_analyze_holoface_benchmark
    from performance_open import analyze_open, build_macro_selection_kpis
    from performance_paleoclimate_plateau import analyze_paleoclimate_plateau

    scanner = ROOT / "scripts/performance_before_outcomes.py"
    command = ["ssh", "-o", "BatchMode=yes", "xbabe3",
               "cd /home/ubuntu/HON_INDEX && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .review-venv/bin/python -"]
    program = scanner.read_bytes()
    result = json.loads(subprocess.run(command, input=program, stdout=subprocess.PIPE, check=True).stdout)
    result["speed_dating"] = analyze(speed_data)
    result["open_synthesis"] = analyze_open(
        speed_data.parent,
        ROOT,
        result["hon"]["cross_national"]["countries"],
    )
    result["facial_attribution"] = analyze_face_attribution()
    result["paleoclimate_plateau"] = analyze_paleoclimate_plateau()
    result["holoface_benchmark"] = load_or_analyze_holoface_benchmark()
    result["macro_selection_kpis"] = build_macro_selection_kpis(result)
    result["analysis_sha256"] = {
        "scripts/performance_before_outcomes.py": hashlib.sha256(program).hexdigest(),
        "scripts/performance_choice.py": digest(ROOT / "scripts/performance_choice.py"),
        "scripts/performance_face_attribution.py": digest(ROOT / "scripts/performance_face_attribution.py"),
        "scripts/performance_holoface_benchmark.py": digest(ROOT / "scripts/performance_holoface_benchmark.py"),
        "scripts/performance_open.py": digest(ROOT / "scripts/performance_open.py"),
        "scripts/performance_paleoclimate_plateau.py": digest(ROOT / "scripts/performance_paleoclimate_plateau.py"),
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "performance-before-outcomes.json"
    pending = destination.with_suffix(".json.new")
    pending.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(destination)
    print(f"wrote {destination}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--no-build", action="store_true")
    p.add_argument("--study", choices=["performance-before-outcomes", "minds-we-choose"])
    p.add_argument("--remote-snapshot", type=Path,
                   help="Human study only: replay validated aggregate evidence")
    p.add_argument("--demographic-frozen", type=Path,
                   help="Human study only: replay the frozen demographic baseline")
    p.add_argument("--speed-data", type=Path,
                   default=ROOT / "target/performance-before-outcomes/inputs/speed_dating.csv")
    args = p.parse_args()
    if (args.remote_snapshot or args.demographic_frozen) and args.study != "minds-we-choose":
        p.error("--remote-snapshot and --demographic-frozen require --study minds-we-choose")
    os.chdir(ROOT)
    if args.study == "minds-we-choose":
        from brain.minds_study import run
        run(args.output, args.speed_data, args.remote_snapshot, args.demographic_frozen)
        return
    if args.study == "performance-before-outcomes":
        run_performance_before_outcomes(args.output, args.speed_data)
        return
    env = os.environ.copy()
    env.setdefault("CARGO_INCREMENTAL", "0")
    env.setdefault("CARGO_PROFILE_DEV_DEBUG", "line-tables-only")
    env.setdefault("CARGO_BUILD_JOBS", "2")
    if not args.no_build:
        subprocess.run(["cargo", "build", "--locked", "--release"], check=True, env=env)
    binary = ROOT / "target/release/authority-lab"
    manifest = {
        "schema_version": 1, "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "python": platform.python_version(),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True).strip(),
        "cargo": subprocess.check_output(["cargo", "-V"], text=True).strip(),
        "binary_sha256": digest(binary),
        "source_sha256": {str(f.relative_to(ROOT)): digest(f) for f in sources()},
        "command": ["authority-lab", "run", str(args.output), "--seed", str(args.seed)] + (["--quick"] if args.quick else []),
        "gpu_used": False,
    }
    cmd = [str(binary), "run", str(args.output), "--seed", str(args.seed)]
    if args.quick:
        cmd.append("--quick")
    subprocess.run(cmd, check=True, env=env)
    # External-proposal controls are scored through the same public adapter.
    subprocess.run([str(binary), "campaigns", "fixtures/proposals.jsonl", str(args.output / "campaigns")], check=True, env=env)
    (args.output / "campaigns.json").write_bytes((args.output / "campaigns/campaigns.json").read_bytes())
    observed = {}
    for db in sorted([*args.output.glob("runtime/**/*.sqlite"), *args.output.glob("campaigns/*.sqlite")]):
        with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            tables = {}
            for table in ("policy", "receipts", "accounts", "objects", "deliveries"):
                tables[table] = [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
            observed[str(db.relative_to(args.output))] = tables
    (args.output / "observed-effects.json").write_text(json.dumps(observed, indent=2) + "\n")
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
