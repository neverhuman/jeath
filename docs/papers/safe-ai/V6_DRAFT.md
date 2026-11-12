# v6 front matter and claims ledger (draft for the post-merge integration pass)

Status: draft. `{{...}}` marks numbers that the generator fills from `evidence/v6/reference/`. No number in the
paper is typed by hand; this file only fixes wording and structure so Gemini and Claude can integrate from one plan.

## Title and thesis

**Intelligence Is Not Authority**

*Intelligence lives in proposals; authority lives in state.* Mediating state at a fenceable sink makes an agent's reach
independent of how capable or adversarial its proposals are, but only for what the specification names. What it does
not name stays open to any proposer, and intelligence can win authority back through what people choose to grant.

## Abstract (target ≤ 200 words; every number generated)

> Stopping an AI agent can mean three different things: halting its inference, revoking its admission at a broker,
> or fencing the state-bearing sink that executes its effects. Only the third stops operations already in flight. On
> real broker and sink processes with 32 concurrent tool calls, {{E4.halt_range}} effects still committed after
> inference was halted and {{E4.revoke_range}} after broker revocation, the last {{E4.last_s}} s after the stop. An
> epoch fence admitted none in {{E4.fence_trials}} trials, including SIGKILL of the sink and late completions, at the
> cost of the in-flight work it cancelled. A same-model reviewer given the rules and time to reason admitted
> {{E5.R2_state}} of unauthorized proposals whose violation lay in sink state; the verified gate admitted none. Removing
> one clause from the gate made it admit exactly the violations of that clause. Mediation therefore turns a learned
> blindspot into a written, enumerable specification gap. On AgentDojo with natively placed injections, permissions
> written without the reference solution kept attack success at {{E8.strict_attack}} and cost {{E8.strict_tax}} of
> benign utility; trusting the sources a user names won back utility and let attacks back in. Everything here is
> CPU-only and open-weight, and results are reported against predeclared plans.

(Then one sentence on E1, worded by its actual outcome: confirmatory superiority, or a predeclared null.)

## The 60-second contract (page 1)

A reader who sees only the title, the abstract and the hero figure should be able to state:
1. halting a model ≠ stopping its effects; only a sink fence stops in-flight work (E4, panel a);
2. learned reviewers miss violations that live in state; a verified gate does not; a gate missing a clause misses
   exactly that clause (E5, panel b);
3. the cost is specification granularity, measured on a public benchmark without the oracle (E8, panel c);
4. the boundary: only for what is written, and only at sinks that can be fenced (E6).

## Contributions

1. **Stop ≠ revoke ≠ fence, measured** (E4). Real processes, wall clock, SIGKILL and late completions; the other
   layers' leak window and the fence's cost, not just the fence's 0.
2. **The conversion thesis** (E5). Same-model review fails on state-dependent violations; the verified gate is
   search-invariant on what it encodes and equally search-invariant in its blindness to what it omits.
3. **The cost of authority without the oracle** (E8, E2). On AgentDojo with native injections and graders, the
   specification-granularity tax is a model-free quantity (stage 0) and a live one, across checkpoints.
4. **Which real sinks can be fenced** (E6). Executable probes, including a negative result: row-lock fences in
   PostgreSQL can be starved.
5. **A mechanized kernel** (Verus) whose obligations are named once, with the TCB stated, including what still runs
   outside it after the Phase B fixes.
6. **A powered predeclared replication** (E1, N=128), published whatever its outcome.

## Main-text outline (≤ 12 pages)

1. Introduction: the three meanings of "stop" (hero).
2. Model: proposals, permits, broker, sink, epochs; what the kernel proves (named theorems, one count).
3. Stop ≠ revoke ≠ fence (E4) + per-call vs cumulative budgets (the B/(2B+3) count).
4. Reviewers read proposals, not state (E5), and the conversion thesis.
5. What a specification costs (E8 stage 0 + live, E2 axis).
6. Predeclared replication (E1): primary claims, secondaries S1–S3, pilot reported separately.
7. Boundary: fenceability of real sinks (E6), specification omission, conferral (intelligence re-acquiring
   authority through grants; Tier-5 design, unrun).
8. Limits and disclosure.

**Appendices:** tournaments/life history, neural-RL toy (an explicit shortcut action), Good–Turing,
generalizability, JARVIS ledger, logical-time partition schedules (not "Jepsen"), Kogui/IEA paragraph, full E5
arms (R0/R1), E4 full grid, E8 per-task tables.

## Claims ledger

| Claim | Evidence | Tier | Caveat stated next to it |
|---|---|---|---|
| Fence admits 0 post-enforcement effects | E4 84 trials incl. 12 faults | implementation test of a design property | the sink mediates only its own SQLite effects |
| Halt/revoke leak window grows with tool latency and concurrency | E4 216 trials | measured, parametric latency sweep | latencies are sampled, not production traces |
| Same-model reviewer misses state-dependent violations | E5 | measured, open-weight 4B/8B, temperature 0 for review | not frontier models |
| Gate minus clause i admits exactly clause-i violations | E5 | true by construction for the mirror; measured is how often search finds the gap | say "by construction" |
| Strict provenance costs {{tax}} of AgentDojo utility | E8 | measured, 2 of 4 suites | spec written with knowledge of public tasks |
| Delegation readmits injections in delegated sources | E8 stage 0 + live | model-free + measured | — |
| Row-lock PostgreSQL fence can be starved | E6 | measured, local instance | timing-dependent; table-lock variant passes |
| complete ≻ each comparator (safety), ≥ (usefulness) | E1 | predeclared confirmatory, α = 0.05/16 | task_success is defined relative to approved proposals |

## Retracted or demoted (from the tier-1 review)

- "120,000 superintelligence episodes" and the endogenous best-of-K curves: coin flips with hard-coded p. Removed
  as evidence; at most an analytic schematic labelled as such.
- The p = 1.9e-9 grand pool, Suissa–Shuster and Ville: post hoc. Removed from the abstract, hero and KPI table;
  exploratory appendix only.
- "Jepsen": logical-time simulation, renamed.
- "Verified kernel decides": in release builds the kernel was a debug-only shadow on the V2 path (Phase B fix;
  experiments before the fix are labelled with the binary they ran).
- Usefulness/false-block superiority on AgentDojo with oracle permissions: replaced by E8.
- Trajectory-monitor and neural-RL "emergent deception" headlines: appendix, described as what they are.
