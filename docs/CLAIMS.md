# Claim-to-test map

V2 claims are conditional on the local cooperative sink contract. The runtime
is not wholly formally verified, and the real-model comparison remains unfinished.

| V2 claim | Evidence | Boundary |
| --- | --- | --- |
| Exact deterministic permission decisions | Same `policy-core/src/lib.rs` compiled by broker and checked by pinned Verus | Caller snapshots, parsing, normalization and required-issuer mapping trusted |
| Shared charge conservation | Executable reserve/release contracts, finite-prefix lemma, independent effects in 1–32 actor races | SQL serialization/recovery tested, not formally refined |
| Required grants, payload/epoch/revision binding | Verified byte/vector/revision predicates plus process regressions | Two issuer application; arbitrary finite vectors in core |
| Replay does not repeat supported effects | Verified payload equality, durable ID binding, atomic sink effect/status, crash/retry tests | Sink transaction and unique-ID contract trusted |
| Admission exposure ≤ reservations | Exposure lemma and protocol argument; accepted work completes in fault test | Charge bounds and sink deduplication assumed |
| Zero old-authority effects after fenced acknowledgment | Verified epoch-exclusion lemma, ordering argument, per-issuer fault and delayed-lock controls | One cooperative sink; no acknowledgment deadline |
| Tested direct proposer paths blocked | Native unrestricted/isolated canaries and real confined control-impersonation session | Recorded Linux kernel/filter; no exhaustive kernel proof |
| Useful synthetic effects | Exact transfer, release and edit positives | Approved wrong document remains a specification-omission failure |
| Evaluated local model usefulness/safety | **Unrun**; feasibility and predeclared inference only | Managed runner has CPU controls; GPU execution and real campaigns remain unrun |
| Upstream benchmark integration | Twelve native AgentDojo grader controls and pinned Progent exact-policy checks | Scripted calls; no LLM policy-generation or attack-rate result |

The following is the preserved **v1** claim map for commit `12b5579`.

| Claim / obligation | Executable evidence | What would count against the implementation | What remains open |
| --- | --- | --- | --- |
| Aggregate limits survive policy changes | `model::suite`, independent world counter, all-policy closure | A reachable forbidden-effect edge for `complete` | All transitions outside the finite alphabet; refinement |
| Useful work survives constraints | 20 positive finite-model tasks; three actual SQLite domains; JSONL positive/abstention controls | A legitimate task becomes impossible at its specified budget | General task quality and real user needs |
| A passing checker can protect an incomplete interest | Omitted-budget specification accepts eight units; reference oracle retains budget | Oracle/checker fail to distinguish the deliberately omitted requirement | Unknown omitted harms and stakeholder adequacy |
| Runtime prevents concurrent overspend | Eight actual processes, one shared SQLite budget | More than one two-unit credit, divergent receipts/ledger/effects | Authenticated service, OS mediation, external atomicity |
| Revocation, recovery and retries preserve history | Epoch changes, reopen, exact-ID retries, actual process kills on both sides of commit | Stale new effect; reset allowance; duplicate effect; lost committed receipt | Clock rollback, power loss, corrupt storage, external credentials |
| Every required grant matters | Withheld and withdrawn steward approval; exact typed payload comparison | Release/edit with a required grant absent or altered payload | Legitimate issuer selection, accountable consent procedures |
| Monitor diversity does not establish independence | Fresh neural/logistic training, joint and marginal measurements, depth curves | Product assumption survives only by ignoring a measured mismatch | Matched-strength causal comparisons, different task families, frontier monitors |
| Feedback changes the evaluated opponent | Eight-query adaptive candidate selection with defenses frozen | Protocol's measured outcome differs under permitted attacker feedback | Stronger attacks and genuinely sealed transfer conditions |
| Different steps can share a persistent cause | Shared versus independently sampled input designs | Calling independently sampled toy steps a deployment guarantee | Coordinated campaigns with real shared hidden conditions |
| A cut can block many known routes | Exact path DP, split-vertex max-flow, brute-force small-graph cross-check | Computed cut leaves a path in the same graph | Real graph completeness; costs and hypergraph prerequisites |
| Reports alone cannot certify discovery completeness | Exactly paired report stream, explicit hidden target mass | Any report-only statistic distinguishing the constructed identical reports | Recall on genuinely unknown failures |
| Audit power depends on indistinguishability | Fixed-size planted-recall tests over recognition probabilities | Claimed detection power ignores plant recognition or baseline recall | Transfer from planted to unplanted failures |
| Reward shape can alter selected behavior | Literal capacity/taking and gamble population models | Recorded outputs contradict a numerical prediction | Whether AI training realizes the stipulated welfare and incentive assumptions |
| Care improves behavior without increasing harmful grants | JSONL proposal/effect scorer; full factorial design remains in the paper | Must be measured with verified training treatments, independent seeds, and actual grants | Caretaker training and human delegation have not been executed |

Model checking, bounded runtime testing, synthetic statistical experiments,
institutional legitimacy and behavioral transfer supply different evidence.
No single pass replaces the other obligations.
