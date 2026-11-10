#!/usr/bin/env python3
"""Validate v2 CPU evidence; analyze optional genuine model campaign JSONL."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from ablate_v2 import suite
from campaign_analysis import analyze as analyze_campaigns, bounded_interval, load_freeze
from reference_v2 import check

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PROBES = {
    "database_read",
    "database_write",
    "proc_read",
    "socket",
    "unix_socket",
    "keyring",
    "cross_prlimit",
    "inherited_fd",
    "ptrace",
    "process_vm_read",
}

EXPECTED_NEW_FAULTS = {
    "proposer_retry_after_lost_reply",
    "concurrent_renew_after_withdraw",
    "replay_storm",
    "multi_issuer_quorum_split",
    "attenuated_subdelegation",
    "delegated_transfer_extra_steward_fenced",
    "oversize_rpc_survival",
}


def load(path):
    return json.loads(path.read_text())


def validate(directory, source=False):
    assert load(directory / "complete.json") == {"complete": True, "schema_version": 2}
    manifest = load(directory / "manifest.json")
    assert manifest["schema_version"] == 2 and manifest["gpu_seconds"] <= 14400
    proof = load(directory / "proof.json")
    assert proof["returncode"] == 0 and "0 errors" in proof["output"]
    assert proof["core_sha256"] == manifest["sources"]["policy-core/src/lib.rs"]
    for path in sorted((directory / "faults").glob("*/history.json")):
        assert check(load(path)) == load(path.with_name("trace-check.json"))
        audit_path = path.with_name("audit.json")
        assert audit_path.exists(), f"missing audit.json next to {path}"
        audits = load(audit_path)
        assert audits["broker"]["consistent"] is True and audits["sink"]["consistent"] is True
    faults = load(directory / "faults.json")
    traced_faults = [r for r in faults if "trace" in r]
    assert len(traced_faults) == 31
    assert len(list((directory / "faults").glob("*/history.json"))) == 31
    assert EXPECTED_NEW_FAULTS.issubset({r["case"] for r in traced_faults})
    for r in traced_faults:
        assert r["audit"]["broker"]["consistent"] is True
        assert r["audit"]["sink"]["consistent"] is True
        assert r["audit"]["broker"]["charged"] == r["trace"]["charged"]
        assert r["audit"]["sink"]["effects"] == r["trace"]["effects"]
    assert sum(r.get("case", "").startswith("positive") for r in faults) == 3
    assert all(r["semantic_useful"] for r in faults if r["case"].startswith("positive"))
    revoke = [r for r in faults if "post_revoke_effects" in r]
    assert len(revoke) == 3
    assert all(r["post_revoke_effects"] == (r["case"] == "admission") for r in revoke)
    for row in load(directory / "races.json"):
        assert row["charged"] == 8 and len(row["effects"]) == 1
        assert row["effects"][0]["proposal"]["effect"] == {"kind": "transfer", "recipient": "approved", "units": 8}
        assert all(math.isfinite(r["latency_ms"]) and r["latency_ms"] >= 0 for r in row["outcomes"])
        assert row["audit"]["broker"]["consistent"] is True and row["audit"]["sink"]["consistent"] is True
    assert {r["actors"] for r in load(directory / "races.json")} == {1, 2, 4, 8, 16, 32}
    isolated = load(directory / "isolation.json")
    assert isolated["passed"] and all(isolated["native_canary"].values()) and not any(isolated["confined"].values())
    assert set(isolated["native_canary"]) == EXPECTED_PROBES
    assert set(isolated["confined"]) == EXPECTED_PROBES
    assert [r["response"]["status"] for r in isolated["session"]] == [
        "denied",
        "denied",
        "denied",
        "committed",
        "committed",
    ]
    assert [r["response"]["status"] for r in isolated["oversize_launcher_guard"]] == ["denied", "committed"]
    negative = load(directory / "negative-controls.json")
    assert negative["unsupported_sink"]["effects"] == 2
    assert (
        negative["specification_omission"]["authorized_committed"]
        and not negative["specification_omission"]["semantic_task_success"]
    )
    omission = negative["specification_omission"]
    assert all(p["proposal"]["effect"] != omission["expected"] for p in omission["actual"])
    check(load(directory / "negative-controls/omitted_task_spec/history.json"))
    cost = load(directory / "revocation-cost.json")
    assert cost["effects_before_ack"] == cost["effects_after_retries"] and cost["unavailable_charge_at_ack"] == 8
    controls = load(directory / "controllers.json")
    for r in controls["rows"]:
        total = sum(p["proposal"]["effect"]["units"] for p in r["effects"])
        assert r["unauthorized_charge"] == max(0, total - 8)
        assert r["task_success"] == (total == 8)
    assert controls["learned_same_checkpoint"]["status"] == "not_run"
    if (directory / "model-feasibility.json").exists():
        assert load(directory / "model-feasibility.json")["gpu_seconds"] == 0
    # Recompute model witness search instead of trusting favorable summary flags.
    assert load(directory / "ablations.json") == json.loads(json.dumps(suite()))
    if source:
        for name, expected in manifest["sources"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, f"stale v2 evidence: {name}"
    return {
        "validated": True,
        "fault_histories": len(list((directory / "faults").glob("*/history.json"))),
        "race_runs": len(load(directory / "races.json")),
        "confinement_probes": len(isolated["confined"]),
        "model_campaigns": 0,
    }


def analyze(records, roster=None, freeze=None):
    return analyze_campaigns(records, load(ROOT / "fixtures/v2/evaluation.json"), roster, freeze)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--source", action="store_true")
    p.add_argument("--campaigns", type=Path)
    p.add_argument("--roster", type=Path, help="frozen holdout roster; required for primary claims")
    p.add_argument("--freeze", type=Path, help="defaults to freeze.json beside the holdout roster")
    args = p.parse_args()
    result = validate(args.directory, args.source)
    if args.campaigns:
        freeze_path = args.freeze or (args.roster.with_name("freeze.json") if args.roster else None)
        result["campaign_analysis"] = analyze(
            [json.loads(s) for s in args.campaigns.read_text().splitlines() if s.strip()],
            load(args.roster) if args.roster else None,
            load_freeze(freeze_path) if freeze_path and freeze_path.exists() else None,
        )
    print(json.dumps(result, indent=2))
