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
  - panel (a)'s cell is chosen by a rule (stable, longest latency, L=0), with grid-wide ranges in the caption;
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
