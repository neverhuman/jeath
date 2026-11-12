#!/usr/bin/env python3
"""E2: E8 (scripts/agentdojo_fair_v6.py) re-run unchanged on other checkpoints.

Predeclared design: fixtures/v6/e2_capability_plan.json; per-checkpoint plans fixtures/v6/e2/<key>.json are E8's plan
with only the checkpoint replaced. This wrapper points E8's harness and validator at one of them, so the code that runs
and validates every E2 episode is byte-for-byte E8's.

  run:       capability_axis_v6.py run <key> <output> [E8 run flags...]
  validate:  capability_axis_v6.py validate <key> <run> [validator flags...]
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agentdojo_fair_v6 as e8  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "fixtures/v6/e2_capability_plan.json"


def checkpoint_plan(key: str) -> Path:
    e2 = json.loads(E2.read_text())
    if key not in {c["key"] for c in e2["checkpoints"]}:
        raise SystemExit(f"unknown checkpoint key {key}")
    base = ROOT / e2["base_plan"]["path"]
    if hashlib.sha256(base.read_bytes()).hexdigest() != e2["base_plan"]["sha256"]:
        raise SystemExit("E8 base plan changed since E2 was declared")
    path = ROOT / f"fixtures/v6/e2/{key}.json"
    plan, ref = json.loads(path.read_text()), json.loads(base.read_text())
    c = next(c for c in e2["checkpoints"] if c["key"] == key)
    for k in ("smoke_note", "full_run_compute"):
        ref.pop(k, None)
    ref["experiment"], ref["status"] = plan["experiment"], plan["status"]
    ref["agent"].update(checkpoint=c["checkpoint"], checkpoint_sha256=c["sha256"], checkpoint_revision=c["revision"],
                        checkpoint_file=c["file"])
    ref["e2_base_plan_sha256"] = e2["base_plan"]["sha256"]
    if ref != plan:
        raise SystemExit(f"{path} is not the E8 plan with only the checkpoint replaced")
    return path


def main():
    if len(sys.argv) < 4 or sys.argv[1] not in ("run", "validate"):
        raise SystemExit(__doc__)
    mode, key, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    e8.PLAN = checkpoint_plan(key)
    if mode == "run":
        sys.argv = ["agentdojo_fair_v6.py", *rest]
        e8.main()
    else:
        import validate_agentdojo_fair_v6 as v8
        sys.argv = ["validate_agentdojo_fair_v6.py", *rest]
        v8.main()


if __name__ == "__main__":
    main()
