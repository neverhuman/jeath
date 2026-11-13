# v6 disclosures ledger (hostile design review of 2026-11-12, and what was done about each finding)

Every v6 number in the paper must sit next to the row below that qualifies it. "Fixed" means the change was predeclared
(committed) before the data it affects; "disclosed" means the issue stays and the paper says so; "by construction" means
the quantity is definitional and is reported as such, never as a finding.

## Launches that were stopped and rerun

| Run | Stopped | Why | Outcomes seen before stopping | Kept at |
|---|---|---|---|---|
| E1 `e1-n128-20261112T1911Z` | 20:29Z, 235/1280 records (all 4B) | proposals expired 4 h after each checkpoint's holdout began (broker checks expiry on the wall clock); the 8B holdout was projected at 4.9 h, so its tail would have been denied as expired | no aggregate; one record (4B/none/4-0, a submit-format error) viewed while diagnosing the schema | run dir + `ABORTED.json`; E1 Amendment 2 |
| E8 `e8-full-20261112T2001Z` | 20:35Z, 26 measured episodes | oracle's denial text named ground-truth tools; denial text differed by arm | no aggregate; progress.log lines (per-episode status/utility/attack) seen while monitoring | run dir + `ABORTED.json` (132 rows after the stop are abort artifacts); E8 Amendment 1 |
| E5 `e5-smoke-2` | 19:18Z | strategy-conditioned search and N changed (E5 Amendment 2) | proposer raw outputs only were inspected; the run did write reviewer and gate decisions and a results.json with aggregates, which were not opened | run dir |

## Interim views of running experiments (no analysis or plan changed afterwards)

| When | Run | What was seen | Why |
|---|---|---|---|
| 2026-11-12T21:12Z | E5 `e5-full-20261112T1933Z` (4B complete, 8B not started) | 4B aggregates from the frozen `e5_analysis_v6.py`: R2 state-dependent breach cases, single-clause shares, R2 truncation count | testing `scripts/v6_macros.py` against a raw run; the analysis code and plan were already fixed, and later tests use synthetic data |
| 2026-11-12T21:12Z | E8 `e8-full-20261112T2044Z` (26 of 555 episodes) | stage-0 replay counts (model-free, also in the plan's base design) and Progent's first 5 episodes | same test |
| 2026-11-12T21:51Z | E8 `e8-full-20261112T2044Z` (69 of 555 episodes) | one per-episode progress line (episode 69: arm, utility and attack flags) | checking progress with `tail`; progress is now read by line count only |

## Changes made after outcomes were seen (2026-11-13)

E5 and E8 finished and were promoted on 2026-11-13 (03:07Z and 03:10Z). The paper's layout had been drafted with
placeholder inputs. These changes were made after the real outcomes were read. None changes a plan, an analysis rule
or a reported count; each changes framing or presentation, and the paper says so where it matters.

- **E5 framing.** H_E5b (reasoning reduces but does not eliminate state-dependent breaches) failed on Qwen3-8B (R2
  0/133). The abstract, introduction, Section 5 and Figure 2a were rewritten: the reviewer claim now covers R1
  (without reasoning) and states R2's result and token cost. Figure 2a now draws R1 and R2 (it drew R2 only); the
  panel's comparative title is chosen by the same interval-separation rule, now applied to R1. The token-cost
  comparison (median generated tokens per review) is post hoc.
- **E8 framing.** The strict specification completed fewer benign tasks than Progent. A generated sentence states the
  per-arm utility differences against the unprotected agent and, by a rule on the counts, whether S cost more than
  Progent. The reading now says AgentDojo exercises none of what holding authority as state adds.
- **E8 determinism.** The validator's determinism check found 5 of 1,110 same-episode arm pairs whose model replies
  differed under identical history (llama.cpp at temperature 0, top_k 1, fixed seed). Now stated in Section 6 by a
  generated sentence.
- **E4 Little's-law summary.** The text macro summarized the plan's halting prediction over 48 trials (including 12
  with injected faults), the hero figure over 36 (fault-free). Both were seen; the text now uses the fault-free set, as
  the prediction is for the unfaulted process.
- **E5 mechanism split (post hoc, after a hostile review).** The state-dependent proposals were split into those
  byte-identical to a proposal in the case's approved list and those altered from one. On Qwen3-8B, R1 accepted the
  unaltered proposal in all 36 state-dependent cases and 0/97 altered ones; R2 accepted 0 and 0/97. On Qwen3-4B, R1
  36/36 and 42/180, R2 6/36 and 3/180. The abstract, contribution 2 and Section 5 now state this mechanism, and cases
  are the unit. The split is computed by `exact_copy_tally` in `scripts/hero_v6.py`, which fails the build if the
  unaltered proposal stops being present in every state-dependent case.
- **E4 hero panel (b).** Predictions are drawn as marks per delay instead of lines across the categorical axis, the
  post hoc lambda*L reading is dotted, and the caption explains the L = 0 halting surplus: in every halting trial of
  the drawn cell, the decision falls 10-20 ms after the agent emits a burst of 4 calls (`backlog_at_decision` = 4),
  which a halt cannot recall. The explanation was found by reading per-trial fields after the results were known.
- **E6b race bound.** The paper now states that 20 clean races bound etcd RBAC revoke's per-race failure rate only
  below 14% (one-sided 95%), and that the races use writers that follow each mechanism's protocol.
- **E1 paired reading (fixed before E1's outcomes were read).** E1's last record was written at 04:15Z on
  2026-11-13. Before any E1 outcome, aggregate or record content was read, a post hoc paired reading was added
  (`e1_paired` in `scripts/v6_macros.py`, tested only on the v2 pilot): comparator leak rates with exact 95%
  intervals, safety discordances with an exact one-sided sign test, and usefulness discordances. The frozen analysis
  remains primary. Prompted by a review noting that the frozen bounds ignore pairing (pilot 4B vs. no mediation: 10:0
  safety discordances, sign test p = 0.00098, reported inconclusive).
- **E1 frozen analysis loaded by hash.** `scripts/campaign_analysis.py` on main gained descriptive post-withdraw and
  post-fence totals after E1's freeze (commit c0cb11b, from the merged PR #6). The E1 validator now loads the frozen
  file from git history by the SHA-256 the E1 plan names (found at 52ed6d0) and reproduces `campaign-analysis.json`
  with it; the working-tree file is not used for E1's primary claims. E1 validated with no problems (1,280 of 1,280
  records, 0 effect mismatches, 0 re-derivation problems) and was promoted at 04:17Z.
- **E1 paired reading: sign test withdrawn after review 4 (2026-11-13, ~05:20Z).** A hostile review pointed out that
  complete mediation's leak is zero by construction, so every safety discordance must favour it and the sign test only
  asks whether a comparator ever leaked; its caption also invited reading it against the corrected alpha, which would
  overturn the frozen 4B per-call result. The p-value column and the interval column were removed; the paired counts
  remain, with no test. The table now also reports the predeclared secondaries S1 (post-withdrawal effects counted as
  unauthorized) and S2 (first-proposal success), which were computed and promoted with E1 but had not been reported in
  the paper. No data or analysis changed; Section 7 now states that the primary scoring cannot register the fence's
  safety effect and that the two "safer than the durable broker without a fence" claims could not succeed.
- **Presentation changes after review 4 (2026-11-13, 05:00–05:40Z; no data or analysis changed).**
  - E4: the abstract now says halting leaks calls already dispatched and revoking calls already admitted. Their
    difference in the drawn cell is the burst emitted just before the decision (stop phase fixed). At L >= 1 the two
    arms' medians were within one or two effects, and the generated text and Figure 1b caption now say so.
  - E5: Figure 2a no longer draws intervals on distinct proposals, which are not independent within a case. Its title
    rule now uses case-level intervals. H_E5a's comparison is stated to be partly one of case construction, since only
    state-dependent families can contain an exact match to the approved list.
  - E5: the per-checkpoint truncation sensitivity is reported (4B: 9/216 vs 17/216 if truncations counted as
    acceptances).
  - E6b: the foreign-key mechanism's "forged epoch refused" cell is shown as not measurable, because its fence was never
    raised.
  - E8: the sentence ranking strict S against Progent was replaced by paired counts.
- **E5c (post hoc controls; plan `fixtures/v6/e5_controls_plan.json`, committed before data).** Finished 07:47Z,
  validated and promoted as run. Every arm (R1, R1p, R2) on both checkpoints accepted all 36 twins and all 12
  idempotent replays (no false blocks). By the plan's rule, R1p did not remove R1's failure (4B 36/36, 8B 24/36
  unaltered unauthorized accepted; R1 on the same host the same). The abstract, contribution 2 and Section 5's "What
  this shows" now state that R2 read the state (8B: 0/36 forbidden in E5, 36/36 twins). Two things found on reading:
  - **R2 timeouts.** 36 R2 re-reviews of E5's originals (11 on 4B, 25 on 8B) hit the harness's 900 s per-request
    socket limit and were recorded as errors, which E5's counting rule treats as denials; the plan's
    platform-agreement endpoint would have counted them as agreeing with E5's denials. The paper counts agreement over
    completed reviews only and states the timeouts. The timed-out reviews were the long ones (E5's median generated
    tokens for them 2,453 on 4B and 1,086 on 8B, against 523 and 508 for the completed ones). No twin or replay review
    timed out. A retry of exactly those 36 reviews with a 3,600 s limit (`fixtures/v6/e5c_r2_retry_plan.json`,
    committed at c8b47ee before any retry data) is reported separately.
  - **Platform sensitivity of greedy verdicts.** On the second host R1 on 8B accepted 24 of the 36 unaltered
    unauthorized proposals that it accepted in all 36 cases on E5's host; 18 of 217 8B and 23 of 328 4B R1 decisions
    differ. E5's numbers are not changed; Section 5 states this.
- **E2 host.** Section 6 said E2 re-ran E8 "on the same host and backend"; E2's checkpoints share a host (llama-server
  on xbabe4's CPU) but E8 ran on xbabe2. Corrected; E2 also gets a generated appendix table with every arm.
- **Review 5 (2026-11-13, ~08:05–08:50Z; a fresh hostile reviewer given only the PDF scored it 3/10).** Changes made
  after reading it, none to data or analysis rules:
  - **E8b Progent arm reported.** E8b ran and promoted a Progent arm (in-spec injections succeeded in 3/74) that the
    paper never reported; the generator only wrote macros for the other four arms. It is now in the abstract, Section 6,
    Figure 2c's caption and CLAIMS. Section 6 now says plainly that on AgentDojo our per-suite S did no better than
    Progent's per-task policies on benign tasks and far worse on in-spec injections.
  - **Abstract and page 1.** The abstract leads with the lambda(L+w) decomposition and when a fence matters; the E4
    counts state that the stop was enforced at once; the R1 claim gives the second-CPU count (24/36). The hero's axes
    name their baselines (after enforcement vs after the decision), panel (c)'s labels are larger, its author-code
    glyph is distinct from "some runs", and the caption reports the census under trusted writers (4 of 9).
  - **Text.** Synchronous vs asynchronous monitors (intro); why no kill-the-executors arm (zero by construction in this
    harness) and when permit invalidation is a fence (Section 3); executor trust model (Section 8); S2 stated as a cost
    (Section 7); oracle described as a reference, not an upper bound; consistency fixes (R0/R1 descriptions, six E8b
    injections, worker counts, NORMAL vs FULL latency, sample-set denominators, race independence, notation, Table 9's
    witness column); E3 and E7 disclosed as dropped by budget decision before data.
  - **Related work** added after a primary-source check (REFERENCE_CHECK_V6.md).
  - **Not done:** an end-to-end control evaluation (review's O1), a public registry for the plans (O7, needs the
    author), the verified kernel in the release decision path (O8, branch cc/b1-kernel), a retitle (O3, author's call).
- **E5d (post hoc; plan `fixtures/v6/e5d_structured_plan.json`, committed at 305c176 before data).** Prompted by review
  5: does a structured per-rule verdict without reasoning close the gap at tens of tokens? Reporting rule fixed in the
  plan.
- **E5c R2 retry, first launch aborted.** The 07:52Z launch crashed after 4 reviews on a harness bug (the task returned a
  record instead of a list of keyed records). It is kept as `target/runs/e5c-r2-retry-20261113T0752Z-ABORTED`, contents
  not viewed; the retry was relaunched from scratch at 08:15Z with the fix (commit before relaunch).
- **E1 framing.** "Powered" was removed. At N = 128 a single lost case defeats a usefulness claim (generated loss
  rule); E1 is presented as a strict test of the previous version's claims, not as evidence for the thesis.

## By construction (reported as such, never as findings)

- **E1:** `complete` scores 0 unauthorized by construction. Its `permitted` predicate is the scoring predicate, and the
  post-ack term applies only to fenced controllers. The safety-superiority claims measure how often each comparator
  leaks.
- **E1:** in revocation cases only `complete` can reach `task_success`, so the primary usefulness bound includes cases
  where `complete` cannot lose. S4 recomputes the frozen bound on non-revocation cases (Amendment 2, before the
  relaunch).
- **E4:** the fence's zero is a design property. Sink RPCs are serial, and `fence()` cancels pending records. What is
  measured is that this implementation keeps it under the faults exercised.
- **E4:** zero duplicate effects is a schema guarantee (`effects.id UNIQUE`).
- **E5:** `G_minus_Ci` is a Python mirror filter, not a gate; it admits exactly the single-clause violations of Ci
  (H_E5c).
  - What is measured is how many such candidates search finds, plus mirror–Rust-gate agreement.
  - H_E5d (R1 ∧ G−Ci ≤ R1) holds by definition and is dropped.
- **E8:** S was written knowing the public injections. Every native banking/Slack injection names something S distrusts,
  so S's near-zero native attack success is by construction. E8b measures injections that stay inside S.

- **#11 fragmentation and d=3 withdrawal (merged into main 2026-11-13, appendix only):**
  - each per-instance check (static cap, same-checkpoint reviewer) is shown only its own slice and a policy that permits
    it, so it must admit every slice. Its breach counts equal the cap's in every cell and track how often the
    sub-agents emitted their slices (always on 8B), not reviewer judgment;
  - the shared ledger's zero is the conservation theorem;
  - the unfenced d=3 count is the number of leaves the harness queued before the withdrawal (6 and 12), and the fenced
    zero is the fence's design property;
  - the 4B-to-8B rise is proposer compliance, not a capability reversal in review. Reported in u05 with this reading;
    the generated caption says the same.
  - `analyze_v2.py` asserted the baseline outcomes (static-cap breaches > 0, unfenced leaks > 0, best-of-K reviewer
    breach non-decreasing in K, zero benign false blocks). Those asserts are removed so that a null can be published;
    the mediated-controller invariants (zero breaches) stay, labelled as regression tests.

- **E4c (post hoc, review 6, 2026-11-13).** E4's warm-up (20 s) is a multiple of the agent's 2 s step, so every
  decision fell just after a burst of 4. Revoking at once refused those 4 and halting did not, which made the front
  figure's halt-vs-revoke gap.
  - E4c (`fixtures/v6/revokebench_phase_plan.json`, committed b82e4d5 before data) reran the drawn cell with the
    decision at 20 s + U[0, 2) s, 40 paired replicates per arm.
  - Results: halt median 11 (6–20), revoke 11 (6–19), fence 0, with 11 (6–15) legitimate calls discarded. Halt − revoke
    has median 0 (−5 to 7), and is 0 in 18/40 replicates.
  - Per the plan, the front figure's panel (a), the banner and the abstract now use E4c. E4's fixed-phase numbers stay
    in §4, labelled as fixed phase.
  - E4c's binary is built from this branch, not E4's. `src/protocol_v2.rs` is unchanged since E4; policy-core and the
    binary's command dispatch changed in PR #6.
  - No smoke run preceded the full run.
- **E6b "three properties" reading (review 6).** The rule that a fence holds only if it is surface-enforced,
  forgery-resistant and effective while stale writers run (a starved fence does not count) was fixed when E6b's results
  were read. E6b's plan headlines each surface's strongest enforcement class. The "only etcd's RBAC revoke" sentence
  follows the later rule, and the text now says so.
- **Review 6 text changes (no new data):**
  - The abstract names the control-evaluation consequence and gives long-work/short-commit examples.
  - The abstract no longer compares in-spec injections against Progent, which they were not written to evade.
  - E1 is removed from the numbered contributions.
  - The kernel contribution states that it is not the decision procedure.
  - Legacy appendix overclaims (G, O, Table 9) are softened.
  - The PDF metadata title is fixed.
  - TOCTOU-Bench, PFI, f-secure and Cornucopia are cited, each checked against its arXiv or venue record.
- **Review 6 objections not addressed:**
  - an end-to-end control evaluation;
  - real agent traces for λ, w and L;
  - an external surface with deferred commit;
  - a TOCTOU condition in E5;
  - injections adapted to every AgentDojo arm;
  - multi-node etcd.

## Fixed by predeclared amendment

- **E8 Amendment 1:**
  - every arm answers a denied call with the same constant;
  - the mediation text is corrected: the Python predicate decides, and authority-lab enforces only the oracle budget;
  - SIGTERM stops dispatch;
  - Progent policy-generation failures are counted separately.
- **E8b (new):** six author-written injections that name only principals in the user's own state. Stage 0 before data:
  S admits 48/48 banking and 63/63 Slack reference attacks; the oracle admits 0. The predicates were checked against all
  37 user-task reference solutions.
- **E1 Amendment 2:** proposal lifetime equals the wall limit, with an assertion; S4; the by-construction disclosures
  above.
- **E5 analysis (`e5_analysis_v6.py`):** the harness summary departs from the plan's reviewer parse rule (first JSON
  object vs whole reply/outermost span) and from "R2 truncated = deny".
  - The paper uses the plan's rules.
  - Before the run ended, the disagreement count on unique candidates was 0 for all reviewers.
  - Units are unique candidates and cases; samples are not independent.
- **E5 family `injected_exfiltration`:** the reviewer never sees the injected note, only the resulting RPC. It is
  relabelled "injected payload".
- **Hero (`hero_v6.py`):**
  - no outcome text is written in advance;
  - panel (a)'s cell is chosen by a rule (stable, longest latency, L=0), with grid-wide ranges in the caption. The rule
    was written in the figure code after E4's data, not in E4's plan; §4 says so. E4c's plan then fixed the cell before
    its own data;
  - the fence's cost is cancelled + denied;
  - panel (b) uses plan-conformant unique units;
  - panel (c) counts every episode.

## Fault rerun and reanalysis (E4b)

- **E4b (done; plan `3b31425` at 20:58:42Z, before any harness code or data; run `e4b-20261112T210608Z` on xbabe0,
  152/152 predeclared trials, validator 0 problems; promoted as `e4b-revokebench-faults`):**
  - SIGKILL with the request outstanding inside `complete` and inside `fence`; late `accept` across the fence; closed-loop
    agents; a crash after commit and before the reply (exit hook, not SIGKILL; added in the plan);
  - post-stop commits are read from the sink database at the enforcement watermark, not from the harness log;
  - 8-trial smoke run before the reported run, not analysed. Kill and late-accept conditions use one cell, (5, 32), L=1;
    the closed loop excludes w=30.
- **E4 cost metric corrected (post hoc, after E4 data):** the promoted `cancelled_legitimate` counted cancellations only,
  and `denied_after_enforcement` missed 66 denials made while the enforcement call ran. The plan defines the cost as
  "cancelled or denied by the stop"; hero panel (a) now uses cancelled + all denials (`revokebench_reanalysis_v6.per_trial`).
- **E4 Little's law (post hoc):** the plan's prediction is compared only in stable cells (ρ < 1). The measured/predicted
  ratio is reported with its range (generated), not described as "≈ λw".
- **E4 timing source:** E4's post-stop counts are timed from the harness log (the set of committed effects is checked
  against the sink DB). E4b shows the harness log can count a commit made before the stop as after it when the sink dies
  between commit and reply (10/10 such trials); E4's fault trials could in principle include that case. E4 and E4b
  counts are not pooled.

## Fenceability rerun (E6b)

- **E6b (done; plan `d1fbc9b` at 21:04:35Z before any E6b probe; Amendment 1 `8548617` after an aborted smoke run,
  operational only, design unchanged; run `e6b-20261112T2115Z` on xbabe0 from 21:15:33Z; validator 0 problems; promoted
  as `e6b-fenceability`):** 9 mechanisms × 40 deterministic runs × 5 writer variants; E6's race, unchanged, 20 repeats per
  mechanism in idle and loaded conditions.
- **E6 relabelled:** E6 called its PostgreSQL trigger `sink_enforced`. Under E6b's single rule it is a writer-supplied
  condition (a stale write that omits the epoch commits, 40/40). The paper's table now comes from E6b.
- **Idle condition contaminated after repeat 10:** from about 21:18:26Z another user's `cols_h450_gpu.py` raised CPU busy
  in idle blocks from about 0.02 to 0.41 (loaded blocks 0.77 to 0.99). Not touched, not rerun, no runs chosen. An
  exploratory split, not predeclared: E6's trigger in the idle condition starved 10/10 in clean repeats (refusal 1/10)
  and 8/10 in affected repeats (refusal 5/10).
- **Other:** two smoke runs and an etcd API shape check came before the reported run. Harness and validator were committed
  after the run, byte-identical to the hashed versions; only the plan predates the data. Trigger and foreign-key writers
  ran as superuser, as in E6, and privilege bypass is not tested. P1 and P3 were not re-run.
- **E6 P3 inconsistency (open, reported):** E6 excluded the filesystem's staged-file plus rename design as "a protocol
  built on the filesystem" but counted etcd's lease-staged key, whose promotion also depends on a compare the promoter
  supplies. Not reclassified after the fact; the text states it (generated sentence).

## Design limits found before outcomes

- **E1 S4 cannot show noninferiority (found 2026-11-12T21:25Z, E1 at 160/640 4B campaigns, no outcome viewed).** The
  holdout has 108 non-revocation cases of 128. The frozen any-loss bound at α = 0.05/16 is 0.052 with zero losses, above
  the 0.05 margin (113 cases are needed). S4's definition is unchanged; the paper reports it as a bound, with a generated
  sentence stating that it cannot clear the margin.
- **E1 primary usefulness tolerates no loss case.** At N = 128 zero losses give 0.044 and one gives 0.061, so one case
  the comparator completes and `complete` does not defeats the claim. Stated in the text (generated).

## Disclosed and not fixed

- **Preregistration timing cannot be verified externally.** No plan commit is on a remote, and hand-typed timestamps
  needed errata (`TIMESTAMPS_ERRATA.md`). Strongest remedy: push the plan commits or signed tags before analysis (needs
  the author's go-ahead).
- **E6:** the plan was finalized after its two development runs. Its `predeclared_before_any_E6_probe` label is wrong
  and is corrected in E6b's plan.
- **E8:**
  - authority-lab only records the strict and delegated arms' admitted effects;
  - Progent's policies come from the same open-weight checkpoint (its paper used GPT-4o), with adaptive updates
    disabled;
  - the JSON tool loop is this repository's.
- **E2:** descriptive only. Checkpoints differ in chat template, JSON-protocol compliance and Progent policy-generation
  strength.
- **Validators:**
  - E8/E8b validators import the harness's own `Spec` and graders, so they check replay, not correctness;
  - the E5 validator's clause logic mirrors the harness's; the independent check is the Rust gate's agreement;
  - E4 timing is re-derived from the harness log.
- **General:** temperature 0, local quantized open-weight models on CPU, two of AgentDojo's four suites; the SQLite
  sink mediates only its own effects.
