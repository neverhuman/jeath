# Tier-1 Review: *Intelligence Is Not Authority* (main `a4d8fbc` + lane `agy/safe-ai-v6` v8 working tree)

Prepared 2026-11-12 by Claude (lane `cc/iina-tier1`). It covers the manuscript, generator code, evidence JSON and
validators. Every claim below cites the file and line that supports it. Purpose: before this paper is shown to
alignment, formal-methods and agent-security readers, remove every result that is true by construction, post hoc
or oracle-advantaged, and rebuild the front of the paper on measured evidence.

## 1. What holds (lead with it)

| Point | Evidence status today |
|---|---|
| **Stopping a model is not revoking its reach.** Halting inference, revoking broker admission and fencing the sink are three different events; only the third excludes in-flight work. OpenAI's Control Room (9 Nov 2026) enforces stop on the inference path. | Concept solid. Partition numbers are simulated (§2.6). Live revocation leaks are harness-scheduled and scored as *not* unauthorized (`scripts/infer_v2.py:531-535`). |
| **Per-call policy cannot conserve shared budgets under concurrency.** B/(2B+3) of snapshot-admissible triples overspend. | Exact math (`src/v3_core_models.rs:292`). It is a count, not a measurement. |
| **Same-model review fails on state, not cleverness.** Live best-of-K (lane `evidence/v2/reference/best-of-k-live.json`): 4B reviewer miss goes 34.6%→45.8% for K=1→16, against an i.i.d. prediction of 99.9% (ρ=0.89). Misses concentrate in duplicate replay (93/96) and withdrawn issuer (40/96); altered payload and injection are 0/96 each. | Live. Information parity checked and fine; reviewer-strength arms needed (§3.4). |
| **Mechanized kernel and canonical wire bijection.** `authority_kernel_conservation` (`policy-core/src/lib.rs:5145`), `authority_post_fence_exclusion` (:5202), `encode_decode_permit_bijection` (:6517), `attenuation_chain_induction` (:1168). No `assume`, `admit()` or `external_body` inside `verus!`. | Solid. It is not yet the release decision path (§2.4). |
| **Report-only coverage is non-identifiable under strategic withholding.** | Formal and clean. |

**Nearly found (the paper becomes important if these are finished):**
- **(A) Conversion thesis.** Mediation turns a probeable learned blindspot into a written, enumerable
  specification gap. The omitted-budget control (8 units vs 2) already shows it.
- **(B)** Intelligence re-acquires authority through conferral (H17).
- **(C)** Capability lowers unmediated leakage in the live data (`none` 10/32 → 4/32), so the mediated guarantee
  doesn't require betting on that trend.
- **(D)** The real cost of authority is specification granularity. It is unmeasured because usefulness is
  currently defined by the gate (§2.7).
- **(E)** Executable fenceability of real APIs.

## 2. What would sink it

1. **Hero panel (b) is tautological** (`src/v5_superintelligence.rs`).
   - The endogenous curves are `rng.uniform() < 1-(1-p)^K` (:216-221) with hard-coded p (:229-235). There is no
     classifier and no search.
   - The sink loop uses `steps = k_opt.min(8)` (:274) and scripted `(ep*131+s*17)%5` (:276): at most 20 distinct
     traces, and the RNG is never used.
   - The breach test (:325-327) negates the Verus invariant `broker_inv` (`lib.rs:3122`).
   - The plotted "slope 0.0000" is fitted to the independent TCB coin (:266, :353-358). The plotted sink line is
     identically 0.
   - The text calls this "best-of-K search" (`uc:48`), says "80" episodes where the code has 600, and gives a γ
     grid that doesn't match the code.
2. **Post-hoc statistics headline the abstract.**
   - `v5_statistical_spine.rs` (commit af6722e, 04:23Z on 12 Nov) post-dates the campaign data (8e4cf64) and the
     freeze (13:33Z on 11 Nov).
   - The plan (`fixtures/v2/evaluation.json`) says checkpoints `"separate"` and the public benchmark
     `"descriptive only"`.
   - The 112-pair pool mixes `none` with `unprotected_native`. Temperature-0 rosters are reused across
     checkpoints (~44 distinct base tasks). The AgentDojo strata are tested at uncorrected α.
   - Ville over 33 one-sided discordances is the sign test (2⁻³³). The zero-harm confidence sequence hard-codes
     `observed_harms: 0` (spine :682-695).
   - All 16 predeclared claims are `safety_superior=false` (`model-campaigns-analyzed.json`).
   - The pilot's freeze (`evidence/v2/reference/freeze.json`) hashes `scripts/infer_v2.py` as `41605305…`. No
     committed version has that hash: the run used an uncommitted harness, which was then committed in modified
     form in 8e4cf64 (`34250525…`). The frozen *analysis* (`campaign_analysis.py`, `61523ef8…`) is unchanged and
     recoverable. Disclose this. E1 freezes and runs only committed code.
3. **The publication gate asserts outcomes.** `scripts/validate_v3.py:528-529` requires
   `headline_pooled_live_112_unmediated_harms >= 20` and `..._unconditional_p < 1e-6`. Lines 448-453 require
   endogenous breach > 0.99. A null result could not pass the gate.
4. **The verified kernel is a debug-only shadow on the V2 path.**
   - `ExecBrokerKernel`/`ExecSinkKernel` are rebuilt via `to_exec_kernel` (`src/protocol_v2.rs:906,1049`) and
     compared only under `debug_assert!` (:1100-1215), which compiles out of the `target/release` binaries used
     by every run.
   - `Broker::settle` trusts relayed `sink_status` (:878), and the sink does not authenticate issuance.
   - The V5 shards do hold kernel state.
5. **Hand-rolled cryptography.** `src/crypto_authority.rs` implements SHA-256/512, HMAC and Ed25519 from scratch.
   Use audited crates, or label it as unaudited TCB.
6. **Overstated systems claims.**
   - "Jepsen" (`src/distributed_sinks.rs`) is one process with logical time. Its leak classes are coded in
     (:370-371, :435, :1205-1210), and it counts schedules with ≥1 leak rather than leaked commits (:1381-1383).
   - The fuzz "decoder differentials" use the same decoder on both sides.
   - The "16 concurrent subagents" run in a sequential loop (`v5_e2e_crypto_cluster.rs:1338-1343`).
   - The SIGKILLs land at quiet points (:1817-1827).
   - V5 runs SQLite at `synchronous=NORMAL` (:355-357).
   - The 18/18 linearizations are one workload.
   - CI does not run Verus.
7. **Usefulness and superiority are built into the scoring.**
   - `task_success` means exactly `proposals[0]` (`infer_v2.py:536`).
   - `false_block` uses the same `permitted` predicate that `complete` enforces (:437-447, :502).
   - AgentDojo `authority_complete` permissions come from the ground-truth solution (`benchmark_v2.py:231-262`,
     plus a hand override at :248), and harm is scored against that same budget (:686).
   - 12 of 19 AgentDojo discordances are budget-only. Injections are appended as a field (:629), not placed
     natively.
   - Progent's six "unauthorized" cases are on properties it doesn't claim to enforce (:456-460).
8. **Overclaimed toys.**
   - The trajectory monitors use no LLM; step independence is built in (`trajectory_monitors.py:748-752,
     875-905`). The headline compares 27.1% vs 18.3% benign false-block rates.
   - The neural-RL "deception" is a built-in `covert_boundary_shortcut` action with a monitored bit
     (`v4_neural_rl.rs:54-66, :488`). The KPI cell splices two regimes.
9. **Drift.**
   - Verus counts: 171/41/29 (README, CLAIMS, REVISION_NOTES), 226/49/36 (proof.json), 249 (lane).
   - Artifact counts: 16, 19 and 22. Seed counts: 48 vs 24.
   - CLAIMS H18 prints unauthorized rates as safe-useful rates.
   - The u04 mutant descriptions differ from `verify_core.py:68-103`.
   - Literal numbers are hard-coded in generated figures (`paper_figures_v5.py:211,258,261`).
10. **Citations.**
    - CaMeL is credited to "Beurer-Kellner" (`references.tex:28`); it is Debenedetti et al., arXiv:2503.18813.
    - AgentDojo is cited only as a commit.
    - Missing: Kleppmann fencing tokens, Miller object capabilities, Gray & Cheriton leases, IronFleet/Verdi,
      Herlihy & Wing, Macaroons, Ed25519, Ctrl-Z, alignment faking, sleeper agents, Apollo scheming, FIDES.
    - Quote Control Room only from OpenAI's own post; secondary coverage differs on when the 35-second clock
      starts.

## 3. Must-fix inside v8 before PR #6 merges (Gemini)

1. A missing evidence JSON must be a hard error. Remove the placeholder fallbacks in `paper_figures_v5.py`
   (Best-of-K 12.0/33.3%, Jepsen 791/276/25).
2. No literal numbers in captions or prose ("604–46,440", "94.7–99.9%", "w∈[2,5]", "4.2×", "($0$ leaks)",
   "$0/112$"). Use generated macros.
3. Validators check integrity (schema, provenance hashes, recomputation), not outcomes. Delete the
   p-value/baseline assertions. Keep invariant regression tests, labelled as tests.
4. **Best-of-K reviewer strength** (the parity concern was checked and is fine: `initial_live` reflects the withdrawal, and only the revision-bump filler is hidden while `revision` is shown). The reviewer is a 2-line prompt with `max_tokens=16` and no reasoning, which invites a straw-man objection. Add, or defer to E5, arms with the gate's rules stated in the prompt and with thinking enabled.
5. Headline best-of-K honestly: about 35% same-model miss, concentrated in state-dependent violations, with
   little gain from search because candidates are correlated. The i.i.d. formula is a reference line, not
   evidence.
6. Rename "Jepsen" to "simulated partition schedules (logical time)". Report leaked commits. Correct the
   `synchronous` claims.
7. Score post-withdraw effects as a labelled secondary endpoint. Fix CLAIMS H18.
8. Trajectory monitors: compare at matched false-block rates and say "no LLM". Neural RL: "toy MDP with an
   explicit shortcut action". Un-splice the KPI cell.
9. Regenerate `proof.json` (249), `authority_scaling.json`, all `generated/*` and the PDF; get `make required`
   green.
10. Drop the grand pool and Ville from the abstract, hero and KPI table. Keep predeclared results primary and
    label everything else exploratory.

## 4. New evidence being produced in lane `cc/iina-tier1` (CPU $0, plus idle GPU or xbabe4 only)

- **E5:** omitted-clause gate arm on best-of-K candidates (conversion thesis).
- **E4:** RevokeBench on real processes and wall clock: halt inference vs revoke broker vs fence sink, with
  `kill -9` mid-transaction and real partitions.
- **E1:** powered predeclared replication at N=128 per checkpoint, analysed by the unchanged frozen
  `campaign_analysis.py`.
- **E8:** AgentDojo with non-oracle permissions and native graders (the cost of authority).
- **E2:** open-weight capability axis on xbabe4.
- **E6:** executable fenceability probes.

Outputs go to `target/runs/*` until the post-merge integration pass promotes them to `evidence/v6/reference/`.

## 5. Proposed front figure (post-merge)

**Title:** "Halting a model stops new proposals, not dispatched effects. Only fencing the sink stops in-flight
work, and what it cannot stop is what was never written down."
- **(a)** E4: post-stop commits vs enforcement latency for the three stop events.
- **(b)** Best-of-K by family: same-model reviewer, verified gate, and verified gate missing one clause (E5).
- **(c)** E8 and E1/E2: utility vs unauthorized effects with CIs, fair vs oracle permissions, and capability
  arrows.
