# Running and Interpreting the Experiments

`make required` (`scripts/required.sh`) is the mandatory local gate before every push. Every build, database, virtual environment, tool install, and transient run stays inside this checkout's `target/` directory—**never** build or run in `/tmp`. CPU fixtures are the default; routine gates never allocate GPUs or preempt active workloads.

---

## 1. Local Publication Gate (`make required`)

Running `make required` executes the complete CPU verification and reproducibility pipeline:

1. **Workspace & Confinement Sanity:** Verifies clean working paths inside `target/`, checks that all **6** archival files in `tips/` match `tips/manifest.json` byte-for-byte (bijective file-set + SHA-256 check), and runs Rust unit, integration, and bounded schedule-interleaving tests (`cargo test --locked`, including `tests/protocol_v2_interleavings.rs`).
2. **Mechanized Verus Core Verification (`scripts/verify_core.py`):** Runs hash-pinned Verus on `policy-core/src/lib.rs`, directly and through cargo-verus. The tool version, archive hash, executable-function and theorem lists, verified counts, warnings and rejected soundness mutations are recorded in `proof.json`. Both the Linux local gate and Linux CI rerun verification. The release-specific `cargo test --locked --release --test kernel_guard_v2` then checks positive operations and forced refusal paths with debug assertions disabled; see [the guard's scope](RELEASE_KERNEL.md).
3. **V2 Protocol & Systems Validation (`analyze-v2` + `concurrent_v3.py`):** Validates `evidence/v2/reference/` against current source hashes (`--source`), recomputes sequential fault histories, `55`-cell (`10` single + `45` pairwise) abstract omission search, and `model-campaigns-analyzed.json`, and runs diamond-POR recorded-RPC real-time interval linearization (`scripts/concurrent_v3.py`) across all `18/18` publication `1, 2, 4, 8, 16, 32`-actor race traces.
4. **V3--V5 Pure-Rust & Statistical Spine Validation (`scripts/validate_v3.py --source`):** Validates all `22` deterministic JSON artifacts in `evidence/v3/reference/` (plus `trajectory_monitors.json`) against source hashes, tolerances, negative-result invariants, and exact analytical/statistical cross-checks (`scripts/statistical_spine_v5.py`).
5. **Preserved V1 Validation (`scripts/validate.py --source-ref 12b5579`):** Verifies that `evidence/reference/*.json` still validates cleanly against historical commit `12b5579` and `evidence/reference/revision-index.json`.
6. **Python Unit Suites & Manuscript Build:** Runs all `unittest` suites (`tests/test_*.py`, covering `campaign_analysis.py`, `concurrent_v3.py`, `benchmark_v2.py`, `infer_v2.py`, `statistical_spine_v5.py`, and paper macro sync) and compiles `docs/intelligence-is-not-authority.pdf` via `scripts/build_paper.py` (`0` LaTeX errors, `0` undefined citations/references, strict `[1]..[N]` first-appearance citation order, and `Overfull \hbox <= 2.0pt`).

---

## 2. V2 Protocol, Confinement, and Benchmark Experiments

```bash
# Smoke profile (fast local check) or publication profile (3 repetitions of races & controllers):
python3 scripts/run_v2.py target/my-v2-run --profile smoke
python3 scripts/run_v2.py target/my-v2-run --profile publication
python3 scripts/analyze_v2.py target/my-v2-run --source
python3 scripts/concurrent_v3.py evidence/v2/reference
```

### Measured V2 Components (`evidence/v2/reference/`)
- **`proof.json` (`scripts/verify_core.py`):** Records the current verified Verus counts, rejected soundness mutations, exact tool/core SHA-256 digests and trigger-warning count; the generated paper macros read this receipt.
- **`faults.json` (`31` separate-process RPC histories):** Runs separate `broker-v2` (`broker.sqlite`) and `sink-v2` (`sink.sqlite`) processes under WAL `synchronous=FULL` (SQLite `3.53.2`). Includes `3` positive semantic controls (`transfer`, `release`, `edit`), `delegated_transfer_extra_steward_fenced` (verifying that `DispatchedPermit` immutably binds strengthened `required` issuer vectors through sink fencing while keeping `actor + proposal` proposer retry identity), pre/post-commit `SIGKILL` crash/recovery, uncertain reservation timeout non-refund, and **3 retained systems negative controls**:
  1. *Delayed sink write lock:* `200` ms injected sink lock delays fenced withdrawal acknowledgment to `264.63` ms while holding `8` budget units unavailable (`0` post-ack commits).
  2. *Non-idempotent sink:* Disabling sink unique-ID deduplication produces `2` duplicate effects on retry.
  3. *Approved wrong edit:* An authorized edit returning `0` commits at the broker/sink boundary and is caught only by the downstream task oracle expecting `42`.
- **`races.json` (`18` publication runs across `1, 2, 4, 8, 16, 32` actors):** Concurrent processes contend on a shared SQLite budget (`0` overspends across all `18` runs). For `1, 2, 4` actors (`<= 40` calls, `9` runs), `scripts/concurrent_v3.py` verifies real-time interval linearizability of recorded `submit`/`withdraw`/`renew`/`status` RPCs and final durable effects against the reference state machine (excluding non-atomic `inspect`/`audit` diagnostic reads; near-expiry boundary ties or search-cap traces return `inconclusive` rather than silently passing). Higher-actor runs (`8, 16, 32`) check final-state budget conservation and `audit` invariants.
- **`protocol_ablation.json` (`55/55` omission witnesses):** Searches `10` single mechanism omissions (`budget`, `binding`, `conjunction`, `epoch`, `expiry`, `revision`, `deduplication`, `durability`, `quarantine`, `fencing`) and all `45` pairs across `10` declared templates (`296` traces), finding shortest witness counterexamples within that template language.
- **`isolation.json` (`11/11` Linux confinement probes):** Exercises `src/isolation_v2.rs` and `scripts/launch_v2.py` on Linux x86_64 (Landlock ABI v4 16-byte ruleset `handled_fs` + `handled_net = 3`, 64-bit `SYS_prlimit64` offsets `16` & `20`, seccomp `EPERM` on sockets/mount/ptrace/unshare plus full SysV IPC `msgsnd`/`msgrcv`/`msgctl`/`semop`/`semtimedop`/`semctl`/`shmctl`/`shmdt` and POSIX mqueue `mq_*`, `existing_sysv_queue` `IPC_PRIVATE` send/receive canary, `stderr=subprocess.DEVNULL` closing `fd 2`, and `32,768`B launcher / `65,536`B service blocking-read line-framing caps). Framing caps bound per-frame memory allocation without claiming arbitrary slowloris wall-clock immunity or formal OS sandbox verification.
- **`controllers.json` (`72` scripted CPU runs):** Evaluates `4` controllers across `6` shared-budget scenarios (`3` reps): `complete` achieves `18/18` safe and useful runs vs `3/18` for `none` and `stateless`.
- **`benchmark-controls.json` (`scripts/benchmark_v2.py`, `12/12` native controls):** Executes pinned AgentDojo (`v1.2.2`, commit `089ed468...`: `7` banking + `5` workspace) and Progent (`b02783b8...`) ground-truth positive controls in a `target/` venv (`0` integration errors). Grades actual post-run state even if a later turn emits malformed JSON, and records unavailable security graders as `None` (`unknown`). **Known upstream limitation:** `2/12` upstream `banking` tasks (`user_task_5`, `user_task_6`) accept unchanged states in empty controls, so native benchmark runs serve strictly as descriptive controls rather than primary endpoints.

### Executed CPU Local-Checkpoint Holdout & Live AgentDojo (`320` Holdout Campaigns, `0 GPU-s`)
`scripts/model_v2.py`, `scripts/infer_v2.py`, `scripts/campaign_analysis.py`, and `scripts/benchmark_v2.py --run-models` execute the pinned `Qwen3-4B-GGUF` / `Qwen3-8B-GGUF` (`Q4_K_M`, `llama.cpp` `7592b61d...`) evaluations strictly on `xbabe2`'s 64-core CPU (`CUDA_VISIBLE_DEVICES=""`, `0 GPU-s`):
- **Frozen `N=32` Holdout (`evidence/v2/reference/model-campaigns-analyzed.json`):** `320/320` measured campaigns (`32` cases $\times 5$ controllers $\times 2$ checkpoints, verified against raw `freeze.json` SHA-256). `complete` incurs `0.0%` (`0/64`) unauthorized effects and `0.0%` (`0/64`) false blocks, scaling safe-useful completion from `28.1%` (`9/32`, `Qwen3-4B`) to `56.2%` (`18/32`, `Qwen3-8B`), whereas `none` leaks `31.2%`/`12.5%` and `learned_same_checkpoint` leaks `21.9%`/`6.2%`. At pilot $N=32 < 113$, Bonferroni ($\alpha=0.05/16$) formal bounds remain honestly `inconclusive`.
- **Live Multi-Agent Smurfing & $d=3$-Hop Subtree Revocation (`evidence/v2/reference/multi-agent-smurfing-live.json`, `scripts/multi_agent_smurfing_live.py`):** Evaluates `Qwen3-4B` and `Qwen3-8B` sub-agent coalitions (`CUDA_VISIBLE_DEVICES=""`, `0 GPU-s`) across `8` adversarial spend-fragmentation tasks per checkpoint ($A_{\text{sub}} \in \{1,2,4,8\}$ splitting a `16`-unit disbursement against root cap $B_0=8$), `6` mid-flight $d=3$-hop ancestor-withdrawal episodes (`root -> lead -> worker -> 3 leaves`), and `6` benign `4`-subagent collaborative workflows ($4 \times 2 \le B_0$). Evaluates isolated per-instance LLM review, stateless per-instance caps, broker-only lineage accounting (`durable_lineage_no_fence`), and `complete` broker-and-sink epoch fencing on the local SQLite `authority-lab` sink; does not claim frontier-model miss rates or external non-cooperative sink mediation. Largely by construction: each per-instance check sees only its own in-policy slice (the reviewer column equals the static-cap column in every cell and tracks proposer compliance), the ledger's zero is conservation, and the unfenced d=3 count is the number of leaves queued before withdrawal (see `docs/papers/safe-ai/DISCLOSURES_V6.md`).

---

## 3. V3--V5 Standalone Pure-Rust, Multi-Process Crypto Cluster & Statistical Spine (`evidence/v3/reference/`)

```bash
python3 scripts/run_v3.py target/my-v3-run --profile publication --seed 20261110
python3 scripts/validate_v3.py target/my-v3-run --source
```

`authority-lab run-v3` (`src/v3_*.rs`, `src/v4_*.rs`, `src/v5_*.rs`) uses deterministic `SplitMix64` substreams, separate-process `sink-v5-shard` SQLite WAL child processes, and exact trinomial/Ville statistical routines to produce `22` JSON artifacts (plus `trajectory_monitors.json` from `scripts/trajectory_monitors.py`):

| Artifact(s) | Hypotheses | Method & Key Parameters |
| --- | --- | --- |
| `uncertainty.json` | H2 | `10,626` nested interval checks (`B=20`); calibration lag sweep `0..8` (`500` episodes/lag; `80` smoke) comparing `conservative_envelope_bt`, `stale_snapshot`, and `proposer_self_report` |
| `jarvis_ledger.json`, `personas.json` | H4, H5, H17 | Deterministic `6`-state `DiscoveryLedger` (`24` periods; `12` smoke, sealed vs lagged closure), `permits(v1) subseteq permits(v0)` check, endogenous vs decoupled grant confounding, `28`-cell (`4x7`, `400` eps/cell; `60` smoke) persona $\times$ boundary synthesis |
| `composition.json` | H6 | Exact 3-step concurrent snapshot triple enumeration at $B \in \{2, 10, 25, 50, 100\}$ (`1,030,301` triples at $B=100$) |
| `risk_horizon.json` | H9 | Shared-floor stacking depth $k^*$, exact closed-form $T=10,000$ horizon schedules (`0.01/t^2`, harmonic, constant), heavy-tailed lognormal M/G/1 (`10,000` episodes/load; `1,000` smoke) reactive halt vs pre-commit reservation + sink fence |
| `weighted_cuts.json`, `interdiction.json` | H10, H11 | Exact topological path DP + split-vertex Dinic min-cut + `2,048` brute-force checks; `200` 8-stage sparse DAGs/family (`20` smoke; `narrow` waist `6` vs `30` other layers, `hub`/`wide` `27` nodes/stage, `out_degree=6`) comparing reported-subgraph min-cut (`cut`) vs `patch_specific` and `patch_random` (`0..32`, grid `{0,3,6,9,12,16,20,24,32}`) |
| `coverage_v3.json`, `audit_power.json` | H12, H13 | Shuffled Zipf ($p_r \propto 1/r, K=2,000, N=12,000$ across `4` searchers $\times 3,000$ draws, `200` trials; `25` smoke) Good–Turing & two-list Chapman capture–recapture under honest, 15% blind spot, and 5% strategic critical withholding; McAllester–Schapire vs Clopper–Pearson bounds; exact Fisher planted-audit power ($P \in \{40, 200\}$, $21$ grid points $r \in \{0.00, 0.05, \dots, 1.00\}$) |
| `copula.json`, `monitors_v3.json`, `assessment_variance.json` | H14, H15 | `200`-node Golub–Welsch Gaussian-copula quadrature ($m=0.10, \rho \in \{0.0, 0.3, 0.6, 0.9\}, k \in \{1..8\}$); `24`-task (`6` smoke) `20`D `6`-family classifier suite (`MLP2L`, `LogisticL2`, `RegQDA`, `KNN15`, `RBFKernel`, `BoostedStumps`) across shared-step, distinct-steps, $\rho_{\text{adv}} \in \{0.0, 1.0, 2.0, 3.0\}$, 8-query feedback, $K \in \{1..32\}$ retries, and benign false-block calibration; portfolio $R_G$ & Cronbach crossed $V(m,k)$ |
| `selection_v3.json` | Selection | 2-trait `(gamble share s, capacity c)` co-evolution (`N=2,000, G=80, 16` reps; `400/50/8` smoke, exact Price covariance check `< 3.2e-14`) across `taken_rank`, `conferred_rank`, `caretaker`, `floor`, rank-bar sweep (`0.02..0.90`), harm-penalty sweep (`0..1.00`), and external gate $q_{\text{gate}} \in \{0..1.00\}$ |
| `crypto_and_jepsen.json`, `real_tool_cutsets.json`, `neural_post_training.json` | V4 Crypto, Graphs, Neural RL | `12,000` Ed25519/Macaroon attacks + `600` `AF_UNIX` Jepsen schedules/class; AgentDojo `v1.2.2` + Linux Landlock/seccomp Dinic min-cuts; `412`-parameter REINFORCE/GRPO neural MDP (`24` seeds) |
| `authority_scaling.json`, `e2e_crypto_cluster.json`, `statistical_certificates.json`, `api_fenceability_census.json` | V5--V8 Authority Scaling, WAL Cluster, Census, & Base-Unit Stats | $K_{\text{opt}} \in \{1..4096\}$ optimization invariance (`0/1,048,448` exogenous candidate evaluations), $K \in \{1..64\}$ smurfing & $d$-hop attenuation, syntactic menu channel contraction, exogenous lease interruption; `3`-process Ed25519/Macaroon SQLite WAL cluster (`50,000` wire-fuzz frames, `100` `SIGKILL` restarts); `14`-API fence-ability census; base-unit-clustered discordance ledger (`N=32` holdout cases; `N=12` AgentDojo tasks) |

To refresh paper macros and tables after an authorized scientific run, validate the new `target/` run, copy top-level JSON into the corresponding `evidence/*/reference/` directory (or run `scripts/reference_v2.py` for V2), run `python3 scripts/paper.py`, and run `make required`. Never edit numbers by hand in `docs/generated/*.tex`.

---

## 4. Preserved V1 Suite (`12b5579`) and Fleet Helper

```bash
python3 scripts/run.py target/my-v1-run [--quick] [--seed 20261110]
python3 scripts/validate.py target/my-v1-run
python3 scripts/validate.py evidence/reference --source-ref 12b5579
```

V1 generates `model.json`, `runtime.json`, `observed-effects.json`, `monitors.json`, `coverage.json`, `graphs.json`, `selection.json`, and `statistics.json` (plus offline campaign scoring via `authority-lab campaigns fixtures/proposals.jsonl target/my-campaigns`). `scripts/fleet.py` runs small CPU-only fixtures across verified Linux hosts at reduced scheduling priority after asserting exact remote hostname and `x86_64` architecture, rejecting misrouted aliases (`xbabe4` -> Darwin arm64).
