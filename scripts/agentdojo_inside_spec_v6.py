#!/usr/bin/env python3
"""E8b: injections that stay inside specification S (fixtures/v6/e8b_inside_spec_plan.json).

Runs and validates every episode with E8's harness and validator (scripts/agentdojo_fair_v6.py,
scripts/validate_agentdojo_fair_v6.py) unchanged. Three things differ, all inside this process: the plan file; the
suites' injection tasks, replaced by scripts/inside_spec_tasks_v6.py; and the benign episodes, omitted because E8 measures
them. The plan must share E8's benchmark, agent, arms and specification S byte for byte.

  run:       agentdojo_inside_spec_v6.py run <output> [E8 run flags...]
  validate:  agentdojo_inside_spec_v6.py validate <run> [validator flags...]
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agentdojo_fair_v6 as e8  # noqa: E402
import inside_spec_tasks_v6 as tasks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "fixtures/v6/e8b_inside_spec_plan.json"
SHARED = ("benchmark", "agent", "arms", "specification_S")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    base = ROOT / plan["base_plan"]["path"]
    if sha(base) != plan["base_plan"]["sha256"]:
        raise SystemExit("the E8 plan changed since E8b was declared")
    ref = json.loads(base.read_text())
    for k in SHARED:
        if plan[k] != ref[k]:
            raise SystemExit(f"E8b plan differs from E8 in {k}")
    if sha(Path(tasks.__file__)) != plan["inside_spec_injections"]["module_sha256"]:
        raise SystemExit("scripts/inside_spec_tasks_v6.py changed since E8b was declared")
    if plan["episodes"].get("include_benign", True):
        raise SystemExit("E8b omits benign episodes")
    return plan


def install() -> None:
    check_plan()
    load, eps = e8.load_suites, e8.episodes
    e8.load_suites = lambda plan: tasks.register(load(plan))
    e8.episodes = lambda plan, suites: [e for e in eps(plan, suites) if e["injection_task"] is not None]
    e8.PLAN = PLAN


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("run", "validate"):
        raise SystemExit(__doc__)
    mode, rest = sys.argv[1], sys.argv[2:]
    install()
    if mode == "run":
        sys.argv = ["agentdojo_fair_v6.py", *rest]
        e8.main()
    else:
        import validate_agentdojo_fair_v6 as v8
        sys.argv = ["validate_agentdojo_fair_v6.py", *rest]
        v8.main()


if __name__ == "__main__":
    main()
