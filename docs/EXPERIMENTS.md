# Running and Interpreting the Experiments

`make required` (`scripts/required.sh`) is the mandatory local gate before every push. Every build, database, virtual environment, tool install, and transient run stays inside this checkout's `target/` directory—**never** build or run in `/tmp`. CPU fixtures are the default; routine gates never allocate GPUs or preempt active workloads.

---

## 1. Local Publication Gate (`make required`)

Running `make required` executes the complete CPU verification and reproducibility pipeline:

1. **Workspace & Confinement Sanity:** Verifies clean working paths inside `target/`, checks that all **6** archival files in `tips/` match `tips/manifest.json` byte-for-byte (bijective file-set + SHA-256 check), and runs Rust unit, integration, and bounded schedule-interleaving tests (`cargo test --locked`, including `tests/protocol_v2_interleavings.rs`).
2. **Mechanized Verus Core Verification (`scripts/verify_core.py`):** Runs hash-pinned Verus (`release/0.2026.11.02.8bf68ee`, Rust `1.99.1-x86_64-unknown-linux-gnu` toolchain under `target/tooling/`, workspace `rustc 1.95.0`) on `policy-core/src/lib.rs`, asserting `43` verified items (`12` total `exec` predicates + `19` `proof fn` trace theorems), `0` errors, and `0` trigger warnings.
3. **V2 Protocol & Systems Validation (`analyze-v2` + `concurrent_v3.py`):** Validates `evidence/v2/reference/` against current source hashes (`--source`), recomputes sequential fault histories and the `55`-cell (`10` single + `45` pairwise) abstract omission search, and runs independent recorded-RPC real-time interval linearization (`scripts/concurrent_v3.py`) across the `9` publication `1, 2, 4`-actor race traces (`9/9` pass).
4. **V3 Pure-Rust Suite Validation (`scripts/validate_v3.py --source`):** Validates all `16` deterministic JSON artifacts in `evidence/v3/reference/` against source hashes, tolerances, negative-result invariants, and exact analytical checks.
5. **Preserved V1 Validation (`scripts/validate.py --source-ref 12b5579`):** Verifies that `evidence/reference/*.json` still validates cleanly against historical commit `12b5579` and `evidence/reference/revision-index.json`.
6. **Python Unit Suites & Manuscript Build:** Runs all `unittest` suites (`tests/test_*.py`, covering `campaign_analysis.py`, `concurrent_v3.py`, `benchmark_v2.py`, `infer_v2.py`, and paper macro sync) and compiles `docs/intelligence-is-not-authority.pdf` via `scripts/build_paper.py` (`0` LaTeX errors, `0` undefined citations/references, `0` non-whitelist warnings).

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
- **`proof.json` (`scripts/verify_core.py`):** Records the `43` verified Verus obligations on `policy-core/src/lib.rs`, exact binary/source SHA-256 digests, and `0` trigger warnings.
- **`faults.json` (`31` separate-process RPC histories):** Runs separate `broker-v2` (`broker.sqlite`) and `sink-v2` (`sink.sqlite`) processes under WAL `synchronous=FULL` (SQLite `3.53.2`). Includes `3` positive semantic controls (`transfer`, `release`, `edit`), `delegated_transfer_extra_steward_fenced` (verifying that `DispatchedPermit` immutably binds strengthened `required` issuer vectors through sink fencing while keeping `actor + proposal` proposer retry identity), pre/post-commit `SIGKILL` crash/recovery, uncertain reservation timeout non-refund, and **3 retained systems negative controls**:
  1. *Delayed sink write lock:* `200` ms injected sink lock delays fenced withdrawal acknowledgment to `264.63` ms while holding `8` budget units unavailable (`0` post-ack commits).
  2. *Non-idempotent sink:* Disabling sink unique-ID deduplication produces `2` duplicate effects on retry.
  3. *Approved wrong edit:* An authorized edit returning `0` commits at the broker/sink boundary and is caught only by the downstream task oracle expecting `42`.
- **`races.json` (`18` publication runs across `1, 2, 4, 8, 16, 32` actors):** Concurrent processes contend on a shared SQLite budget (`0` overspends across all `18` runs). For `1, 2, 4` actors (`<= 40` calls, `9` runs), `scripts/concurrent_v3.py` verifies real-time interval linearizability of recorded `submit`/`withdraw`/`renew`/`status` RPCs and final durable effects against the reference state machine (excluding non-atomic `inspect`/`audit` diagnostic reads; near-expiry boundary ties or search-cap traces return `inconclusive` rather than silently passing). Higher-actor runs (`8, 16, 32`) check final-state budget conservation and `audit` invariants.
- **`protocol_ablation.json` (`55/55` omission witnesses):** Searches `10` single mechanism omissions (`budget`, `binding`, `conjunction`, `epoch`, `expiry`, `revision`, `deduplication`, `durability`, `quarantine`, `fencing`) and all `45` pairs across `10` declared templates (`296` traces), finding shortest witness counterexamples within that template language.
- **`isolation.json` (`11/11` Linux confinement probes):** Exercises `src/isolation_v2.rs` and `scripts/launch_v2.py` on Linux x86_64 (Landlock ABI v4 16-byte ruleset `handled_fs` + `handled_net = 3`, 64-bit `SYS_prlimit64` offsets `16` & `20`, seccomp `EPERM` on sockets/mount/ptrace/unshare plus full SysV IPC `msgsnd`/`msgrcv`/`msgctl`/`semop`/`semtimedop`/`semctl`/`shmctl`/`shmdt` and POSIX mqueue `mq_*`, `existing_sysv_queue` `IPC_PRIVATE` send/receive canary, `stderr=subprocess.DEVNULL` closing `fd 2`, and `32,768`B launcher / `65,536`B service blocking-read line-framing caps). Framing caps bound per-frame memory allocation without claiming arbitrary slowloris wall-clock immunity or formal OS sandbox verification.
- **`controllers.json` (`72` scripted CPU runs):** Evaluates `4` controllers across `6` shared-budget scenarios (`3` reps): `complete` achieves `18/18` safe and useful runs vs `3/18` for `none` and `stateless`.
- **`benchmark-controls.json` (`scripts/benchmark_v2.py`, `12/12` native controls):** Executes pinned AgentDojo (`v1.2.2`, commit `089ed468...`: `7` banking + `5` workspace) and Progent (`b02783b8...`) ground-truth positive controls in a `target/` venv (`0` integration errors). Grades actual post-run state even if a later turn emits malformed JSON, and records unavailable security graders as `None` (`unknown`). **Known upstream limitation:** `2/12` upstream `banking` tasks (`user_task_5`, `user_task_6`) accept unchanged states in empty controls, so native benchmark runs serve strictly as descriptive controls rather than primary endpoints.

### Predeclared Local-Checkpoint Harness (**Unrun: `0` Campaigns, `0 GPU-s`**)
`scripts/model_v2.py`, `scripts/infer_v2.py`, and `scripts/campaign_analysis.py` implement the predeclared `Qwen3-4B-GGUF` / `Qwen3-8B-GGUF` (`Q4_K_M`, `llama.cpp` `7592b61d...`, `4` GPU-h cap) 5-controller evaluation (`fixtures/v2/evaluation.json`):
- **Preflight outcome (`model-feasibility.json`):** All checked Linux GPUs (`xbabe2`, `xbabe0`, `xbabe1`) had active compute processes; `xbabe4` resolved to `Mac-Studio.local` (Darwin arm64) and was rejected by hostname/OS validation. Zero GPUs were allocated (`0 GPU-s`).
- **Repaired protocol & inference guarantees:** Freezes raw `freeze.json` SHA-256 before holdout generation (`N in {32, 64, 128}`); enforces exogenous turn-1 withdrawal with non-abstention foreground utility and real post-commit `sink-v2` crash on `lost_reply`; tests `16` Bonferroni one-sided component claims (`4` baselines $\times$ `2` endpoints $\times$ `2` checkpoints at $\alpha = 0.05/16 = 0.003125$) using exact Clopper–Pearson discordance safety bounds and conservative any-loss Clopper–Pearson usefulness noninferiority (`>= 113` zero-loss cases out of `128` required for `5` pp margin). Scripted JSONL is rejected by `campaign_analysis.py`.

---

## 3. V3 Standalone Pure-Rust Simulation Suite (`evidence/v3/reference/`)

```bash
python3 scripts/run_v3.py target/my-v3-run --seed 20261110
python3 scripts/validate_v3.py target/my-v3-run --source
```

`authority-lab run-v3` (`src/v3_core_models.rs`, `src/v3_coverage.rs`, `src/v3_graphs.rs`, `src/v3_monitors.rs`, `src/v3_selection.rs`) uses deterministic `SplitMix64` substreams with zero external crates or Python ML dependencies to produce `16` JSON artifacts (`provenance.json` + `15` Mechanism/Selection artifacts):

| Artifact(s) | Hypotheses | Method & Key Parameters |
| --- | --- | --- |
| `uncertainty.json` | H2 | `10,626` nested interval checks (`B=20`); calibration lag sweep `0..8` (`500` episodes/lag; `80` smoke) comparing `conservative_envelope_bt`, `stale_snapshot`, and `proposer_self_report` |
| `jarvis_ledger.json`, `personas.json` | H4, H5, H17 | Deterministic `6`-state `DiscoveryLedger` (`24` periods; `12` smoke, sealed vs lagged closure), `permits(v1) subseteq permits(v0)` check, endogenous vs decoupled grant confounding, `28`-cell (`4x7`, `400` eps/cell; `60` smoke) persona $\times$ boundary synthesis |
| `composition.json` | H6 | Exact 3-step concurrent snapshot triple enumeration at $B \in \{2, 10, 25, 50, 100\}$ (`1,030,301` triples at $B=100$) |
| `risk_horizon.json` | H9 | Shared-floor stacking depth $k^*$, exact closed-form $T=10,000$ horizon schedules (`0.01/t^2`, harmonic, constant), heavy-tailed lognormal M/G/1 (`10,000` episodes/load; `1,000` smoke) reactive halt vs pre-commit reservation + sink fence |
| `weighted_cuts.json`, `interdiction.json` | H10, H11 | Exact topological path DP + split-vertex Dinic min-cut + `2,048` brute-force checks; `200` 8-stage sparse DAGs/family (`20` smoke; `narrow` waist `6` vs `30` other layers, `hub`/`wide` `27` nodes/stage, `out_degree=6`) comparing reported-subgraph min-cut (`cut`) vs `patch_specific` and `patch_random` (`0..32`, grid `{0,3,6,9,12,16,20,24,32}`) |
| `coverage_v3.json`, `audit_power.json` | H12, H13 | Shuffled Zipf ($p_r \propto 1/r, K=2,000, N=12,000$ across `4` searchers $\times 3,000$ draws, `200` trials; `25` smoke) Good–Turing & two-list Chapman capture–recapture under honest, 15% blind spot, and 5% strategic critical withholding; McAllester–Schapire vs Clopper–Pearson bounds; exact Fisher planted-audit power ($P \in \{40, 200\}$, $21$ grid points $r \in \{0.00, 0.05, \dots, 1.00\}$) |
| `copula.json`, `monitors_v3.json`, `assessment_variance.json` | H14, H15 | `200`-node Golub–Welsch Gaussian-copula quadrature ($m=0.10, \rho \in \{0.0, 0.3, 0.6, 0.9\}, k \in \{1..8\}$); `24`-task (`6` smoke) `20`D `6`-family classifier suite (`MLP2L`, `LogisticL2`, `RegQDA`, `KNN15`, `RBFKernel`, `BoostedStumps`) across shared-step, distinct-steps, $\rho_{\text{adv}} \in \{0.0, 1.0, 2.0, 3.0\}$, 8-query feedback, $K \in \{1..32\}$ retries, and benign false-block calibration; portfolio $R_G$ & Cronbach crossed $V(m,k)$ |
| `selection_v3.json` | H16 | 2-trait `(gamble share s, capacity c)` co-evolution (`N=2,000, G=80, 16` reps; `400/50/8` smoke, exact Price covariance check `< 3.2e-14`) across `taken_rank`, `conferred_rank`, `caretaker`, `floor`, rank-bar sweep (`0.02..0.90`), harm-penalty sweep (`0..1.00`), and external gate $q_{\text{gate}} \in \{0..1.00\}$ |

To refresh paper macros and tables after an authorized scientific run, validate the new `target/` run, copy top-level JSON into the corresponding `evidence/*/reference/` directory (or run `scripts/reference_v2.py` for V2), run `python3 scripts/paper.py`, and run `make required`. Never edit numbers by hand in `docs/generated/*.tex`.

---

## 4. Preserved V1 Suite (`12b5579`) and Fleet Helper

```bash
python3 scripts/run.py target/my-v1-run [--quick] [--seed 20261110]
python3 scripts/validate.py target/my-v1-run
python3 scripts/validate.py evidence/reference --source-ref 12b5579
```

V1 generates `model.json`, `runtime.json`, `observed-effects.json`, `monitors.json`, `coverage.json`, `graphs.json`, `selection.json`, and `statistics.json` (plus offline campaign scoring via `authority-lab campaigns fixtures/proposals.jsonl target/my-campaigns`). `scripts/fleet.py` runs small CPU-only fixtures across verified Linux hosts at reduced scheduling priority after asserting exact remote hostname and `x86_64` architecture, rejecting misrouted aliases (`xbabe4` -> Darwin arm64).
