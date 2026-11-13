# jeath — Intelligence Is Not Authority

[Paper PDF](docs/intelligence-is-not-authority.pdf) · [Claims & Tiers (`H1`–`H18`)](docs/CLAIMS.md) · [Experiments & Reproduction](docs/EXPERIMENTS.md) · [Revision Audit](docs/REVISION_NOTES.md) · [CI Contract](docs/CI.md) · [Actions](https://github.com/neverhuman/jeath/actions)

**Thesis:** *Intelligence lives in proposals; authority lives in state.* Halting a model stops new proposals, not effects already in flight; a fence at the state-bearing sink also stops those, and only for what the specification names. We conjecture that mediation moves the safety problem from a model's judgment into a written specification; the measured costs and gaps are in the paper and in [`docs/CLAIMS.md`](docs/CLAIMS.md), including the negative results (a reasoning same-model reviewer matched the gate on E5's small cases; the written AgentDojo specification cost more benign utility than Progent).

**v6 (this branch):** predeclared experiments E1, E4, E4b, E5, E6, E6b, E8 and E8b on real broker/sink processes, local open-weight checkpoints and AgentDojo; post hoc controls for E5, each designed after E5's results and committed before its own data: E5c (twins and replays), its R2 timeout retry, and E5d (a structured per-rule verdict); E2 (AgentDojo across checkpoints on a second host), descriptive.

- Plans: `fixtures/v6/`. Evidence: `evidence/v6/reference/` (gate: `python3 scripts/validate_v6.py`).
- Every stopped launch, interim view and by-construction result: [`docs/papers/safe-ai/DISCLOSURES_V6.md`](docs/papers/safe-ai/DISCLOSURES_V6.md).
- Every cited preprint checked against its arXiv record: [`docs/papers/safe-ai/REFERENCE_CHECK_V6.md`](docs/papers/safe-ai/REFERENCE_CHECK_V6.md).
- Claims: the v6 table in [`docs/CLAIMS.md`](docs/CLAIMS.md).

---

## Five-Tier Evidence Architecture

Every claim in this repository is scoped to an explicit verification tier ([`docs/CLAIMS.md`](docs/CLAIMS.md)):

1. **Tier 1 — Mechanized `policy-core` Verus Proofs (`policy-core/src/lib.rs`):** verified obligations, `exec` functions and theorems as counted in `evidence/v2/reference/proof.json` (`0` trigger warnings, `10` soundness mutations rejected; produced by `scripts/verify_core.py`, which neither `make required` nor CI reruns). Covers the verified kernel functions and canonical wire injectivity—**not** the unverified decision logic around them in `src/protocol_v2.rs`, SQLite, outer JSON framing, or OS kernel refinement.
2. **Tier 2 — Exhaustive Finite-State & Exact Math Checks:** V1 reachable-state closure (`10,180` states / `223,960` checks; `41,732`-state omitted-budget control), V2 abstract protocol ablation (`55/55` single and pairwise omission witnesses across `296` traces), exact 3-step concurrent snapshot enumeration (`1,030,301` triples at $B=100$; `49.26%` snapshot overspend vs `0` serialized), `2,048` brute-force weighted-cut DAG checks (`0` discrepancies), and exact Fisher audit PMFs.
3. **Tier 3 — Separate-Process Local SQLite, Simulated Partition Schedules, Linux Confinement & CPU LLMs (`evidence/v2/reference/`, `evidence/v3/reference/`):** Separate `broker-v2` (`broker.sqlite`) and `sink-v2` (`sink.sqlite`) processes under WAL `synchronous=FULL` (SQLite `3.53.2`; the V5 cluster test uses `synchronous=NORMAL`), `31` sequential RPC fault/`SIGKILL`/slowloris-deadline histories, `18/18` real-time diamond-POR concurrent linearizations of one transfer workload across `1,2,4,8,16,32`-actor populations (`scripts/concurrent_v3.py`, `0` overspends), unaudited from-scratch Ed25519 quorum/fence + HMAC-SHA256 Macaroons (`0/10,000` red-team false accepts), a single-process logical-time partition simulation over `AF_UNIX` socket pairs (not Jepsen; schedules with a post-ack leak: `0/1,000` cooperative vs `791` broker-only and `276` uncooperative), `11/11` Linux Landlock ABI v4 + seccomp + IPC confinement probes, operational AgentDojo `v1.2.2` + Linux IPC weighted Dinic cutsets (`real_tool_cutsets.json`), `480` executed 5-step AgentDojo trajectory-monitor evaluations (`trajectory_monitors.json`), and `320` frozen CPU `llama.cpp` `Qwen3-4B`/`Qwen3-8B` holdout campaigns (`0 GPU-s`; `0/64` unauthorized effects under `complete`, which holds by construction, safe-useful completion `28.1%` (`4B`) and `56.2%` (`8B`), while pilot $N=32 < 113$ Bonferroni bounds remain honestly inconclusive) plus live AgentDojo `v1.2.2` runs (`2/12` upstream `banking` tasks `5` and `6` accept unchanged states in empty controls).
4. **Tier 4 — Pure-Rust Deterministic Simulation & Neural RL Suite (`evidence/v3/reference/`):** `19` standalone JSON artifacts (`seed 20261110`, zero external crates) covering calibrated uncertainty lag (`H2`), `6`-state `DiscoveryLedger` & endogenous human-grant harm reversal (`H4, H5, H17`), horizon accumulation & heavy-tail reactive halt leaks (`H9`), weighted cuts & `200`-DAG interdiction (`H10, H11`), Good–Turing/Lincoln–Petersen report-only non-identifiability & planted-audit power (`H12, H13`), `200`-node Golub–Welsch copula + `6`-family classifier correlation + crossed variance ceilings (`H14, H15`), 2-trait Price selection geometry (`H16`), and a `412`-parameter 2-layer neural policy-gradient RL suite (`neural_post_training.json`, `48` seeds) demonstrating emergent `+34.5 pp` evaluation-aware covert shortcutting under audited GRPO tournaments vs `0.00%` gated survival-floor harm.
5. **Tier 5 — Full-Scale Power & Governance Protocols (`fixtures/v2/evaluation.json`):** Full-power $N \ge 113$ paired 5-controller checkpoint scaling (`16` Bonferroni one-sided claims at $\alpha = 0.05/16$ for `5` pp usefulness noninferiority), frontier-model transfer, and the $3\times 3\times 2$ LLM caretaker post-training & human scope-conferral factorial study (**unrun on GPUs or human participants; `0 GPU-s`**).

---

## Quick Start & Reproducibility Commands

Keep all builds, databases, virtual environments, and runs inside this checkout's `target/` directory (never `/tmp`). Linux isolation requires Landlock ABI $\ge 4$ (`x86_64`); macOS CI runs portable functional checks (`docs/CI.md`; Linux CI uses the self-hosted `neverhuman-lan` runner group).

```bash
# Full local publication gate (Rust/Python tests, V1+V2+V3/V4+v6 validation, tips/ hashes, generated tables, PDF build):
make required

# v6 evidence integrity (recomputes every promoted result from raw artefacts; asserts no outcome):
python3 scripts/validate_v6.py
python3 scripts/paper.py --check

# Fresh V2 protocol & confinement run + source-hash validation + interval linearization:
python3 scripts/run_v2.py target/my-v2-run --profile publication
python3 scripts/analyze_v2.py target/my-v2-run --source
python3 scripts/concurrent_v3.py target/my-v2-run

# Fresh V3 pure-Rust simulation suite (16 deterministic JSON artifacts) + validation:
python3 scripts/run_v3.py target/my-v3-run --seed 20261110
python3 scripts/validate_v3.py target/my-v3-run --source

# Preserved V1 suite and historical commit 12b5579 verification:
python3 scripts/run.py target/my-v1-run --quick
python3 scripts/validate.py target/my-v1-run
python3 scripts/validate.py evidence/reference --source-ref 12b5579
```

### Launching Confined Local Broker/Sink Sessions (Linux)
```bash
# Start trusted broker and cooperative sink processes (never expose control sockets to proposers):
target/release/authority-lab broker-v2 target/broker.sqlite
target/release/authority-lab sink-v2 target/sink.sqlite

# Bind a Landlock + seccomp confined proposer process to a single actor session:
python3 scripts/launch_v2.py target/broker.sqlite target/sink.sqlite alice /usr/bin/python3 -c '...'
```

---

## Scope Discipline & Archival Preservation

- **Six Archival Originals Preserved (`tips/`):** `Intelligence_Is_Not_Authority_Final(2).tex`, `intelligence-is-not-authority (1).tex`, `intelligence-is-not-authority.tex`, `Superintelligence_Is_Not_Sovereignty.tex`, `intelligence_is_not_authority.tex`, and `experiments.py` remain byte-identical and SHA-256 verified against [`tips/manifest.json`](tips/manifest.json).
- **Explicit Boundaries:** The SQLite executor and `AF_UNIX` sink cluster mediate only cooperative idempotent effects. Nothing in this repository claims OS-kernel sandbox verification, uncooperative external non-idempotent exactly-once execution, whole-runtime SQLite/OS formal refinement, frontier-model empirical rates, frontier-LLM RLHF post-training, human-participant delegation evidence, or consultation/endorsement by Indigenous communities (`0 GPU-s`; CPU `Qwen3` pilot $N=32 < 113$ leaves formal Bonferroni bounds inconclusive).
