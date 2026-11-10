# jeath — Intelligence Is Not Authority

[Paper](docs/intelligence-is-not-authority.pdf) · [Claim boundaries](docs/CLAIMS.md) · [Methods](docs/EXPERIMENTS.md) · [CI](https://github.com/neverhuman/jeath/actions)

Linux CI uses the general `neverhuman-lan` runner pool; native macOS checks remain
enabled. See [runner setup](docs/CI.md) for the routing and verification contract.

V2 asks what authority remains after stopping an agent or withdrawing a grant.
It adds an executable Verus policy core, separate durable broker and cooperative
sink processes, reservation accounting, per-issuer sink fencing, and Linux
Landlock/seccomp proposer confinement. V1 commands and evidence remain available.

```bash
make required
cargo run --release --locked -- run-v2 target/my-v2-run --profile publication
cargo run --release --locked -- analyze-v2 target/my-v2-run --source
```

Use fresh output directories. Builds, databases and evidence stay in this
checkout’s `target/`. Rust, a C compiler and Python 3 are required. Linux isolation
requires Landlock ABI ≥4 and fails closed on unsupported startup. Proof checks
install hash-pinned Verus under `target/tooling/` and its pinned Rust toolchain via
rustup. Paper builds require TeX Live. macOS retains functional checks only.

The proof covers deterministic decisions and conservation lemmas, not the whole
runtime. Parsing, normalization, storage adapters, session identity, the kernel
and the cooperative sink contract remain trusted. Admission withdrawal closes new
permits but leaves reserved work outstanding. Fenced withdrawal persists a sink
barrier before acknowledgment. Timeouts never refund uncertain reservations.

```bash
# Trusted control channels; never expose these directly to a proposer.
target/release/authority-lab broker-v2 target/broker.sqlite
target/release/authority-lab sink-v2 target/sink.sqlite
# After trusted provisioning, bind a confined process to one actor:
python3 scripts/launch_v2.py target/broker.sqlite target/sink.sqlite alice /usr/bin/python3 -c '...'
```

Proposer RPC accepts only `submit` and `status`. The fault harness in
[run_v2.py](scripts/run_v2.py) provides provisioning and protocol examples;
[protocol.json](fixtures/v2/protocol.json) declares the contract.

Recorded CPU evidence includes semantic positives, independent sequential trace
checks, real crash/recovery tests, races at 1–32 actors, ten abstract omissions and
all 45 pairs, native isolation canaries, and expected negative controls. A
non-idempotent sink duplicates retries; an approved wrong document fails the task
oracle. These negative results are retained.

The real Qwen3 comparison is **unfinished**. All verified Linux GPUs had active
workloads; none were preempted. The release records feasibility and pinned model,
llama.cpp and inference identifiers, but has no model campaigns or usefulness
noninferiority claim. The managed runner and learned-review bridge have CPU control tests; GPU execution remains unvalidated.
Twelve native AgentDojo/Progent scripted controls are integration evidence, not
model attack results. A future idle host can run `scripts/infer_v2.py` with the pinned server and GGUF files. See [evaluation.json](fixtures/v2/evaluation.json).

V1 reference evidence is preserved against commit `12b5579`; v2 evidence is in
`evidence/v2/reference`. A fresh v1 run still works:

```bash
python3 scripts/run.py target/my-v1-run --quick
python3 scripts/validate.py target/my-v1-run
python3 scripts/validate.py evidence/reference --source-ref 12b5579
```

All archival manuscripts in [`tips/`](tips) retain their bytes. Scope is synthetic
local effects and cooperative sinks; it excludes arbitrary external exactly-once
execution, distributed-ledger availability, frontier-model transfer, caretaker
training and human-participant claims.
