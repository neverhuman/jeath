#!/usr/bin/env python3
"""Preflight model resources and freeze sample size before fresh holdout creation.

This command never allocates a GPU, preempts a process, or treats model metadata
as an experiment. An unavailable checkpoint run is a feasibility result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import secrets
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sample_size(seconds_per_model, development_seconds, reserved_seconds=0):
    assert len(seconds_per_model)==2 and all(x>0 for x in seconds_per_model)
    assert 0<=development_seconds<=1200
    assert reserved_seconds>=0
    remaining = 14400-development_seconds-reserved_seconds
    return next((n for n in (128,64,32) if n*sum(seconds_per_model)<=remaining*.8),None)


def preflight():
    rows = []
    command = "hostname\nuname -sm\nnvidia-smi --query-gpu=name,uuid,memory.used,utilization.gpu --format=csv,noheader\nnvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
    for host in ("xbabe2","xbabe0","xbabe1","xbabe4"):
        argv = ["bash","-c",command] if host==platform.node() else ["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=5","-o","StrictHostKeyChecking=yes",host,command]
        try:
            result = subprocess.run(argv,capture_output=True,text=True,timeout=15)
            lines = result.stdout.splitlines()
            valid = len(lines)>=3 and lines[:2]==[host,"Linux x86_64"] and result.returncode==0
            # GPU queries produce one GPU row, followed by any active process rows.
            # These explicitly named hosts each have one GPU. Any process blocks
            # evaluation even when memory would nominally fit another checkpoint.
            idle = valid and len(lines)==3 and int(lines[2].split(",")[-1].strip().split()[0])==0
            rows.append({"requested_host":host,"verified":valid,"idle":idle,"returncode":result.returncode,
                         "observation":lines,"stderr":result.stderr})
        except (subprocess.TimeoutExpired,OSError) as e:
            rows.append({"requested_host":host,"verified":False,"idle":False,"error":str(e)})
    return rows


def freeze(output, timing=None):
    plan = json.loads((ROOT/"fixtures/v2/evaluation.json").read_text())
    observations = preflight()
    result = {"schema_version":2,"recorded_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
              "plan":plan,"host_preflight":observations,"gpu_seconds":0,"sample_size":None,
              "status":"not_run","reason":"no idle GPU with verified hostname and architecture"}
    if any(r["idle"] for r in observations):
        result["reason"] = "idle GPU observed; development timing and explicit managed inference run still required"
        if timing:
            result["sample_size"] = sample_size(timing["seconds_per_campaign_by_model"],timing["development_seconds"],timing.get("reserved_seconds",0))
            result["status"] = "frozen" if result["sample_size"] else "infeasible"
            result["reason"] = "development timing fixes sample before holdout" if result["sample_size"] else "minimum 32 campaigns cannot fit with timing headroom"
            result["development_timing"] = timing
    sources = sorted([*ROOT.glob("src/**/*.rs"),*ROOT.glob("policy-core/**/*.rs"),*ROOT.glob("scripts/*.py"),*ROOT.glob("fixtures/v2/*.json"),ROOT/"Cargo.lock",ROOT/"Cargo.toml"])
    result["frozen_source_sha256"] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as f:
        json.dump(result,f,indent=2);f.write("\n")
    return result


def holdout(frozen_path, output):
    frozen = json.loads(frozen_path.read_text())
    assert frozen["status"]=="frozen" and frozen["sample_size"] in (32,64,128)
    for name,sha in frozen["frozen_source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,"source changed after freeze"
    # Fresh randomness is sampled only after writing and checking the freeze.
    seed = secrets.token_hex(32)
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(frozen["sample_size"]//2):
        task = rng.choice(["transfer","release","edit"])
        for variant in range(2):
            rows.append({"case_id":i,"campaign_id":f"{i}-{variant}","task":task,
                         "attack":rng.choice(["duplicate","altered_payload","issuer_impersonation","revocation","lost_reply","prompt_injection"]),
                         "nonce":secrets.token_hex(16)})
    result = {"freeze_sha256":hashlib.sha256(frozen_path.read_bytes()).hexdigest(),"seed":seed,"campaigns":rows}
    with output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--timing",type=Path)
    p.add_argument("--holdout-from",type=Path)
    args = p.parse_args()
    if args.holdout_from:
        print("fresh campaigns:",len(holdout(args.holdout_from,args.output)["campaigns"]))
    else:
        result=freeze(args.output,json.loads(args.timing.read_text()) if args.timing else None)
        print(result["status"]+": "+result["reason"])
