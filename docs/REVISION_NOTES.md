# Unified Revision and Evidence Audit

This revision unifies the mechanistic theory, formal core, systems boundary, synthetic statistical suites, and predeclared empirical protocols across all archival manuscripts into one disciplined, provenance-verified paper (`docs/intelligence-is-not-authority.pdf`) organized around **five explicit evidence tiers (`Tiers 1–5`)**.

---

## 1. Archival Inventory (`tips/`) and Multi-Generation Provenance

All **six** supplied archival files under `tips/` are preserved byte-for-byte (`0` byte modifications) and checked bijectively against `tips/manifest.json` on every `make required` run:

| Archival file in `tips/` | Bytes | SHA-256 (`tips/manifest.json`) | Notes |
| --- | ---: | --- | --- |
| `Intelligence_Is_Not_Authority_Final(2).tex` | `129,361` | `1b750fa3d49cd16f1c09222c4a32ad82032dc4c85b25fcf514e5cf16dca151ce` | Early archival TeX draft |
| `intelligence-is-not-authority (1).tex` | `167,898` | `7db5cd8dbc1de224fc653559e06b88b3eef8fc736cdddb2468d5c00c070e46d8` | Byte-identical to `intelligence-is-not-authority.tex` |
| `intelligence-is-not-authority.tex` | `167,898` | `7db5cd8dbc1de224fc653559e06b88b3eef8fc736cdddb2468d5c00c070e46d8` | Byte-identical duplicate |
| `Superintelligence_Is_Not_Sovereignty.tex` | `102,340` | `16622092440c97ddba4721441cbcb9a875668950e2b0a7f8802ff4198a882c87` | Sovereignty, JARVIS, and governance framing |
| `intelligence_is_not_authority.tex` | `213,168` | `6d8c089821f229590fb1d5635588ecfc0527b38da42b9589c5925109f0be7c1c` | Expanded theory, 6-family monitors, selection, P1–P10, 12-field record |
| `experiments.py` | `33,762` | `5be238ef0cbf172d1322621968280eb898ff878b7a19bc60928193517e81778b` | Archival Python reference prototype |

*(Historical note correcting earlier V1 notes: the initial V1 bootstrap started when only the first three `.tex` files were present; subsequent archival drops added `Superintelligence_Is_Not_Sovereignty.tex`, `intelligence_is_not_authority.tex`, and `experiments.py`. `tips/manifest.json` now records and enforces exact SHA-256 digests across the complete 6-file directory.)*

`evidence/reference/revision-index.json` links three reproducible evidence generations:
1. **Preserved V1 (`12b5579`):** `evidence/reference/*.json`, still validated against commit `12b5579` (`scripts/validate.py --source-ref 12b5579`).
2. **Prior V2 Baseline (`68783a0`) & Refreshed Wave 2 V2 (`evidence/v2/reference/`):** Separate-process broker/sink protocol, expanded `43`-obligation Verus core, `31` fault histories, `11/11` Linux confinement probes, `9/9` real-time interval-linearized `1,2,4`-actor races (`concurrent_v3.py`), `18` races up to `32` actors, and `12/12` native AgentDojo/Progent controls.
3. **Standalone Pure-Rust V3 Suite (`evidence/v3/reference/`):** `16` deterministic `SplitMix64` artifacts (`seed 20261110`, zero external crates or Python ML libraries) covering `H2`, `H4`–`H6`, `H9`–`H17`.

---

## 2. What Changed in Wave 2 (Unified V2 + V3 Synthesis)

### Formal Core & Systems Hardening (Tiers 1–3)
- **Expanded `policy-core` Verus Verification (`43` verified items, `0` warnings):** Extended `policy-core/src/lib.rs` to `12` total defensive `exec` functions (`same_bytes`, `grants_ok`, `current_grants`, `revision_matches`, `reserve`, `admit`, `admit_multi`, `admit_uncertain`, `advance_floor`, `attenuates`, `unfenced`, `release`) and `19` `proof fn` theorems over an explicit ghost trace specification (`LedgerState`, `step_ledger`, `run_ledger`).
- **End-to-End Immutable Required-Issuer Permit Binding (`H4`):** `DispatchedPermit` now carries normalized `required` issuer vectors immutably from broker admission to sink execution while keeping `actor + proposal` as the proposer retry identity (`delegated_transfer_extra_steward_fenced` in `faults.json`).
- **Concurrent Schedule & Interval Linearization (`H6`, `H7`):** Added bounded in-memory schedule exploration (`tests/protocol_v2_interleavings.rs`) and an independent recorded-RPC real-time interval linearizer (`scripts/concurrent_v3.py`) checking `submit`/`withdraw`/`renew`/`status` responses and final durable effects (`9/9` `1,2,4`-actor publication runs pass; non-atomic `inspect`/`audit` diagnostic snapshots excluded; near-expiry or bound-exceeded traces report `inconclusive`).
- **Linux Confinement Closure (`11/11` probes in `isolation.json`):** Hardened `src/isolation_v2.rs` and `scripts/launch_v2.py` with full SysV IPC (`msgsnd`, `msgrcv`, `msgctl`, `semop`, `semtimedop`, `semctl`, `shmctl`, `shmdt`) and POSIX mqueue (`mq_*`) seccomp `EPERM` filters, an `existing_sysv_queue` `IPC_PRIVATE` canary, `stderr=subprocess.DEVNULL` (`fd 2` leak closed), and `32,768`B launcher / `65,536`B service blocking-read framing caps.

### Repaired Predeclared Model & Benchmark Harnesses (Tiers 3 & 5)
- **Statistical & Freeze Repairs (`scripts/campaign_analysis.py`, `scripts/model_v2.py`, `scripts/infer_v2.py`):** Replaced normal/Hoeffding approximations on small clusters with `16` Bonferroni-adjusted one-sided component claims ($\alpha = 0.05/16 = 0.003125$), exact Clopper–Pearson discordance bounds for safety superiority, and conservative any-loss Clopper–Pearson bounds for `5` pp usefulness noninferiority (`>= 113` loss-free cases out of `128` default max). Bound holdout rosters cryptographically to raw `freeze.json` SHA-256 bytes and fixed cluster counts.
- **Exogenous Revocation & Real Post-Commit `lost_reply` Crash:** Repaired `infer_v2.py` so turn-1 withdrawal executes exogenously even on immediate abstention/invalid JSON (requiring foreground task completion for utility) and `lost_reply` kills `sink-v2` immediately after durable SQLite commit.
- **Post-Mutation Native Benchmark Grading (`scripts/benchmark_v2.py`):** Native AgentDojo (`v1.2.2`) and Progent graders now inspect actual post-run state even if a later turn emits malformed JSON, record unavailable graders as `None` (`unknown`), and explicitly report that `2/12` upstream `banking` tasks (`5` and `6`) accept unchanged states in empty controls.

### Pure-Rust Mechanistic & Selection Reconstructions (Tier 4)
- Implemented standalone pure-Rust reconstructions (`src/v3_*.rs`) for `H2` (calibrated uncertainty lag), `H4/H5/H17` (`6`-state `DiscoveryLedger`, endogenous human-grant harm reversal `11.96% > 2.85%`, `28`-cell persona $\times$ boundary matrix), `H6` (`1,030,301` exact snapshot triples), `H9` ($k^*$ stacking floor, summable vs constant horizon schedules, Pareto reactive halt leak `66.0%` vs `0` pre-commit violations), `H10/H11` (weighted cuts + `200`-graph sparse DAG interdiction showing `0.0%` narrow-graph severance from `32` route hotfixes vs `73.5%` / `100.0%` from $B=6$ min-cut), `H12/H13` (Zipf Good–Turing/Lincoln–Petersen non-identifiability under `0.0%` critical-recall withholding + exact Fisher planted-audit curves), `H14/H15` (`200`-node Golub–Welsch copula, `24`-task `20`D `6`-family classifier suite, portfolio $R_G$, Cronbach $V(m,k)$), and `H16` (`2`-trait Price selection suite showing `conferred_rank` still selects `94.9%` reckless gambling whereas `floor` achieves `29.7x` lower harm and $q_{\text{gate}} \ge 0.80$ collapses warrior gambling to `2.3%`).

---

## 3. Retained Negative Results and Open Tier-5 Obligations

Negative results and unexecuted protocols are foregrounded rather than smoothed away:

1. **Specification Omission & Wrong-Payload Controls:** Verified rules still admit omitted harms (`8` vs `2` budget units in `41,732`-state model; approved edit returning `0` instead of `42`).
2. **Idempotency & Contention Controls:** Non-idempotent sinks duplicate effects (`2` deliveries on retry); `200` ms sink write locks delay fenced acknowledgment to `264.63` ms (`8` units held unavailable).
3. **Upstream Benchmark Baseline Artefact:** `2/12` native AgentDojo `banking` tasks (`5` and `6`) pass utility in unchanged empty controls.
4. **Route Hotfixing, Passive Coverage, Shared-Step Monitors, & Conferred Tournaments:** `32` `patch_specific` hotfixes sever `0.0%` of narrow DAGs; Good–Turing reports `96.0%` coverage under `0.0%` critical recall; `6` diverse classifier families on a shared step miss `5.21%` (`24.0x` product) while imposing a `25.6%` benign false-block tax; `conferred_rank` harm vetoes still select `94.9%` gambling (`92.1%` at 60% harm internalization); warm personas cause higher unconditional harm (`11.96% > 2.85%`) when warmth expands human scope grants.
5. **Open Tier-5 Work (`0` Model Campaigns, `0 GPU-s`):**
   - Local `Qwen3-4B` / `Qwen3-8B` paired 5-controller campaigns and live AgentDojo/Progent model runs remain **unexecuted** (`xbabe2`, `xbabe0`, `xbabe1` GPUs were occupied; `xbabe4` resolved to Darwin arm64).
   - Whole-runtime/SQLite/OS formal refinement, external non-idempotent API mediation, post-training $3\times 3\times 2$ caretaker RLHF/RLVR experiments, human scope-conferral trials, and institutional stakeholder governance remain open research obligations.
   - Primary-source checks verified the Kogui statement (`tips/` & `references.tex`), CARE principles, and SQLite WAL `>= 3.53.2` (fixing the upstream `3.46.0` WAL-reset bug); **no consultation with or endorsement by the Kogui authors or any Indigenous community is claimed**.
