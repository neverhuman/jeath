#!/usr/bin/env python3
"""E5c-R2-retry (plan: fixtures/v6/e5c_r2_retry_plan.json): re-run the R2 reviews that timed out in E5c.

Only the per-request socket limit changes. The request body, parsing and counting are E5's own (omitted_clause_v6.review);
the limit is raised by wrapping the urlopen that omitted_clause_v6.chat calls, so no E5 code is edited.

  e5c_r2_retry_v6.py run OUT --expected-host xbabe2 --llama-server BIN --models Q4B.gguf Q8B.gguf [--workers 4 --threads 12]
  e5c_r2_retry_v6.py validate OUT
"""
from __future__ import annotations

import argparse
import functools
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omitted_clause_v6  # noqa: E402
from e5_controls_v6 import allowed  # noqa: E402
from infer_v2 import _run_parallel_tasks, cpu_model_name, cpu_server_pool, gpu_compute_snapshot, sha  # noqa: E402
from omitted_clause_v6 import RULES_PROMPT, review  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "fixtures/v6/e5c_r2_retry_plan.json"
E5C = ROOT / "evidence/v6/reference/e5c-controls"
CHECKPOINTS = ["Qwen/Qwen3-4B-GGUF", "Qwen/Qwen3-8B-GGUF"]


def items() -> list[dict]:
    recs = [json.loads(l) for l in (E5C / "records.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in recs if r["set"] == "originals" and "R2" in r["reviews"] and r["reviews"]["R2"].get("error")]


def aggregate(records: list[dict]) -> dict:
    base = [json.loads(l) for l in (E5C / "records.jsonl").read_text().splitlines() if l.strip()]
    out = {}
    for ck in CHECKPOINTS:
        rs = [r for r in records if r["checkpoint"] == ck]
        done = [r for r in rs if not r["retry"].get("error")]
        prior = [r for r in base if r["checkpoint"] == ck and r["set"] == "originals" and "R2" in r["reviews"]
                 and not r["reviews"]["R2"].get("error")]
        both = [(allowed("R2", r["reviews"]["R2"]), r["e5"]["R2"]) for r in prior] + \
               [(allowed("R2", r["retry"]), r["e5"]["R2"]) for r in done]
        out[ck] = {"rerun": len(rs), "completed": len(done), "accepted": sum(allowed("R2", r["retry"]) for r in done),
                   "truncated": sum(r["retry"].get("finish_reason") == "length" for r in done),
                   "failed_again": len(rs) - len(done),
                   "disagreements_with_e5": sum(allowed("R2", r["retry"]) != r["e5"]["R2"] for r in done),
                   "combined": {"completed": len(both), "accepted": sum(a for a, _ in both),
                                "disagreements_with_e5": sum(a != b for a, b in both)}}
    return out


def run(args) -> None:
    plan = json.loads(PLAN.read_text())
    assert platform.node() == args.expected_host and platform.system() == "Linux" and platform.machine() == "x86_64"
    base = json.loads((E5C / "results.json").read_text())
    if sha(args.llama_server) != base["provenance"]["llama_server_sha256"]:
        raise SystemExit("llama-server differs from E5c's")
    todo = items()
    limit = float(args.request_limit_seconds)
    real = omitted_clause_v6.urllib.request.urlopen

    @functools.wraps(real)
    def urlopen(req, *a, timeout=None, **kw):  # only chat completions get the longer limit; health checks keep theirs
        if getattr(req, "full_url", "").endswith("/v1/chat/completions"):
            timeout = limit
        return real(req, *a, timeout=timeout, **kw) if timeout is not None else real(req, *a, **kw)

    omitted_clause_v6.urllib.request.urlopen = urlopen
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.wall_limit_seconds
    gpu_before, started = gpu_compute_snapshot(), time.time()
    lock, timing = threading.Lock(), {}
    rec_file = (args.output / "records.jsonl").open("w")
    for model, ckpt in zip(args.models, CHECKPOINTS, strict=True):
        mine = [it for it in todo if it["checkpoint"] == ckpt]
        if not mine:
            continue
        t0 = time.monotonic()
        with cpu_server_pool(args.llama_server, model, deadline, args.expected_host,
                             args.output / f"servers-{ckpt.split('/')[1]}", workers=args.workers, threads=args.threads,
                             ctx_size=8192) as pv:
            engines = pv[0] if isinstance(pv, tuple) else pv

            def make_task(it):
                def task(engine):
                    rv = review(engine.port, RULES_PROMPT, it["policy"], it["visible"], it["rpc"], deadline, True)
                    rec = {k: it[k] for k in ("checkpoint", "case_id", "family", "kind", "rpc", "violated", "e5")}
                    rec |= {"e5c_error": it["reviews"]["R2"]["error"], "retry": rv}
                    with lock:
                        rec_file.write(json.dumps(rec, sort_keys=True) + "\n")
                        rec_file.flush()
                        with (args.output / "progress.log").open("a") as pl:
                            pl.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {ckpt} case {it['case_id']}\n")
                    return [((ckpt, it["case_id"], json.dumps(it["rpc"], sort_keys=True)), rec)]  # _run_parallel_tasks' shape
                return task

            _run_parallel_tasks(engines, [make_task(it) for it in mine])
        timing[ckpt] = time.monotonic() - t0
    rec_file.close()
    records = [json.loads(l) for l in (args.output / "records.jsonl").read_text().splitlines() if l.strip()]
    ver = subprocess.run([str(args.llama_server), "--version"], capture_output=True, text=True, timeout=60)
    result = {"schema_version": 1, "experiment": plan["experiment"], "plan_sha256": sha(PLAN),
              "script_sha256": sha(Path(__file__)), "e5c_records_sha256": sha(E5C / "records.jsonl"),
              "request_limit_seconds": limit, "aggregates": aggregate(records),
              "provenance": {"hostname": platform.node(), "cpu_model": cpu_model_name(),
                             "llama_server_version": (ver.stdout + ver.stderr).splitlines(),
                             "llama_server_sha256": sha(args.llama_server), "workers": args.workers,
                             "threads": args.threads, "started_unix": started, "elapsed_seconds_per_checkpoint": timing,
                             "gpu_seconds_used": 0, "gpu_snapshot_before": gpu_before,
                             "gpu_snapshot_after": gpu_compute_snapshot(),
                             "records_jsonl_sha256": sha(args.output / "records.jsonl")}}
    (args.output / "results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records)}))


def validate(args) -> None:
    """Integrity only: plan and E5c hashes, the re-run set is exactly E5c's timed-out R2 reviews, aggregates recompute."""
    res = json.loads((args.output / "results.json").read_text())
    records = [json.loads(l) for l in (args.output / "records.jsonl").read_text().splitlines() if l.strip()]
    problems = []
    if res["plan_sha256"] != sha(PLAN):
        problems.append("plan hash differs")
    if res["e5c_records_sha256"] != sha(E5C / "records.jsonl"):
        problems.append("E5c records hash differs")
    if res["provenance"]["records_jsonl_sha256"] != sha(args.output / "records.jsonl"):
        problems.append("records hash differs")
    key = lambda r: (r["checkpoint"], r["case_id"], json.dumps(r["rpc"], sort_keys=True))
    if sorted(map(key, records)) != sorted(map(key, items())):
        problems.append("re-run set differs from E5c's timed-out R2 reviews")
    if aggregate(records) != res["aggregates"]:
        problems.append("aggregates do not recompute from records.jsonl")
    print(json.dumps({"records": len(records), "problems": problems}, indent=1))
    sys.exit(1 if problems else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("output", type=Path)
    r.add_argument("--expected-host", required=True)
    r.add_argument("--llama-server", type=Path, required=True)
    r.add_argument("--models", nargs=2, type=Path, required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--threads", type=int, default=12)
    r.add_argument("--request-limit-seconds", type=int, default=3600)
    r.add_argument("--wall-limit-seconds", type=int, default=4 * 3600)
    v = sub.add_parser("validate")
    v.add_argument("output", type=Path)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else validate(args)


if __name__ == "__main__":
    main()
