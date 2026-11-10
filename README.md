# jeath — Intelligence Is Not Authority

[Paper PDF](docs/intelligence-is-not-authority.pdf) · [Claims & Tiers (`H1`–`H18`)](docs/CLAIMS.md) · [Experiments & Reproduction](docs/EXPERIMENTS.md) · [Revision Audit](docs/REVISION_NOTES.md) · [CI Contract](docs/CI.md) · [Actions](https://github.com/neverhuman/jeath/actions)

**Thesis:** *Intelligence proposes; institutions confer, bound, and revoke reach.* Capability measures how effectively an optimizer searches an action space; authority governs what real-world state transitions an institution lets commit and what residual exposure survives when permission is withdrawn.

---

## Five-Tier Evidence Architecture

Every claim in this repository is scoped to an explicit verification tier ([`docs/CLAIMS.md`](docs/CLAIMS.md)):

1. **Tier 1 — Mechanized `policy-core` Verus Proofs (`policy-core/src/lib.rs`):** `43` verified SMT obligations (`0` errors, `0` trigger warnings) across `12` total `exec` predicates and `19` `proof fn` trace theorems compiled directly into the broker. Covers core permission, conservative uncertainty admission, attenuation, epoch fencing, and budget conservation—**not** whole-runtime, JSON, SQLite, or OS refinement.
2. **Tier 2 — Exhaustive Finite-State & Exact Math Checks:** V1 reachable-state closure (`10,180` states / `223,960` checks; `41,732`-state omitted-budget control), V2 abstract protocol ablation (`55/55` single and pairwise omission witnesses across `296` traces), exact 3-step concurrent snapshot enumeration (`1,030,301` triples at $B=100$; `49.26%` snapshot overspend vs `0` serialized), `2,048` brute-force weighted-cut DAG checks (`0` discrepancies), and exact Fisher audit PMFs.
3. **Tier 3 — Separate-Process Local SQLite & Linux Confinement (`evidence/v2/reference/`):** Separate `broker-v2` (`broker.sqlite`) and `sink-v2` (`sink.sqlite`) processes under WAL `synchronous=FULL` (SQLite `3.53.2`), `31` sequential RPC fault/`SIGKILL` histories (including immutable strengthened `required` permit fencing and `3` retained negative controls), `9/9` real-time interval-linearized `1,2,4`-actor races (`scripts/concurrent_v3.py`) plus `18` shared-budget races across `1..32` actors (`0` overspends), `11/11` Linux Landlock ABI v4 + seccomp (including SysV/POSIX IPC) + framing-cap probes, `72` scripted controller runs, and `12/12` native pinned AgentDojo (`v1.2.2`) / Progent scripted controls (`2/12` upstream `banking` tasks `5` and `6` accept unchanged states in empty controls).
4. **Tier 4 — Pure-Rust Deterministic Simulation Suite (`evidence/v3/reference/`):** `16` standalone `SplitMix64` artifacts (`seed 20261110`, zero external crates or Python ML libraries) covering calibrated uncertainty lag (`H2`), `6`-state `DiscoveryLedger` & endogenous human-grant harm reversal (`H4, H5, H17`), horizon accumulation & heavy-tail reactive halt leaks (`H9`), weighted cuts & `200`-DAG interdiction (`H10, H11`), Good–Turing/Lincoln–Petersen report-only non-identifiability & planted-audit power (`H12, H13`), `200`-node Golub–Welsch copula + `6`-family classifier correlation + crossed variance ceilings (`H14, H15`), and 2-trait Price selection geometry (`H16`).
5. **Tier 5 — Predeclared Unrun Model & Governance Protocols (`fixtures/v2/evaluation.json`):** Repaired paired 5-controller evaluation on pinned `Qwen3-4B-GGUF` / `Qwen3-8B-GGUF` (`16` Bonferroni one-sided claims at $\alpha = 0.05/16$, exact Clopper–Pearson discordance safety bounds, `>= 113/128` loss-free cases for `5` pp usefulness noninferiority, raw `freeze.json` SHA-256 binding, `4` GPU-h cap) and $3\times 3\times 2$ caretaker/conferral factorial design. **Status: Unrun (`0` campaigns, `0 GPU-s`)**—all checked Linux GPUs (`xbabe2`, `xbabe0`, `xbabe1`) had active workloads and `xbabe4` resolved to Darwin arm64 (`Mac-Studio.local`).

---

## Quick Start & Reproducibility Commands

Keep all builds, databases, virtual environments, and runs inside this checkout's `target/` directory (never `/tmp`). Linux isolation requires Landlock ABI $\ge 4$ (`x86_64`); macOS CI runs portable functional checks (`docs/CI.md`; Linux CI uses the self-hosted `neverhuman-lan` runner group).

```bash
# Full local publication gate (Verus 43 proofs, Rust/Python tests, V1+V2+V3 validation, tips/ hashes, PDF build):
make required

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
- **Explicit Boundaries:** The SQLite executor mediates only its own local cooperative database effects. Nothing in this repository claims OS-kernel sandbox verification, arbitrary external non-idempotent exactly-once execution, whole-runtime formal refinement, live local or frontier LLM empirical campaign results (`0 GPU-s`), neural post-training validation, human-participant delegation evidence, or consultation/endorsement by Indigenous communities.
