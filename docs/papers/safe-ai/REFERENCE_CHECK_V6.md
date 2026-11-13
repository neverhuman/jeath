# Reference check (v6), 2026-11-13

Each arXiv-cited preprint was compared against its arXiv abstract page (title, author list, first-submission date,
and, where a version is cited, its version date), and each sentence that cites it was read against its abstract.
Web sources were checked against the publisher's own page where it could be fetched.

## arXiv preprints (all 23 resolve; titles and authors match)

| key | arXiv | first submitted | title |
|---|---|---|---|
| `terekhov` | 2510.09462 | 2025-10-10 | Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols |
| `makins` | 2607.07368 | 2026-07-08 | Multi-Agent AI Control: Distributed Attacks Hamper Per-Instance Monitors |
| `bhatt2025ctrlz` | 2504.10374 | 2025-04-14 | Ctrl-Z: Controlling AI Agents via Resampling |
| `safetycase` | 2411.03336 | 2024-10-29 | Towards evaluations-based safety cases for AI scheming |
| `needham` | 2505.23836 | 2025-05-28 | Large Language Models Often Know When They Are Being Evaluated |
| `schoen` | 2509.15541 | 2025-09-19 | Stress Testing Deliberative Alignment for Anti-Scheming Training |
| `korbak` | 2507.11473 | 2025-07-15 | Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety |
| `greenblatt2024faking` | 2412.14093 | 2024-12-18 | Alignment faking in large language models |
| `hubinger2024sleeper` | 2401.05566 | 2024-01-10 | Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training |
| `meinke2024scheming` | 2412.04984 | 2024-12-06 | Frontier Models are Capable of In-context Scheming |
| `chan2025infra` | 2501.10114 | 2025-01-17 | Infrastructure for AI Agents |
| `camel` | 2503.18813 | 2025-03-24 | Defeating Prompt Injections by Design |
| `costa2025fides` | 2505.23643 | 2025-05-29 | Securing AI Agents with Information-Flow Control |
| `progent` | 2504.11703 | 2025-04-16 | Progent: Securing AI Agents with Privilege Control |
| `agentspec` | 2503.18666 | 2025-03-24 | AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents |
| `conseca` | 2501.17070 | 2025-01-28 | Contextual Agent Security: A Policy for Every Purpose |
| `isolategpt` | 2403.04960 | 2024-03-08 | IsolateGPT: An Execution Isolation Architecture for LLM-Based Agentic Systems |
| `beurerkellner2025patterns` | 2506.08837 | 2025-06-10 | Design Patterns for Securing LLM Agents against Prompt Injections |
| `gsai` | 2405.06624 | 2024-05-10 | Towards Guaranteed Safe AI: A Framework for Ensuring Robust and Reliable AI Systems |
| `ftbudget` | 2610.00349 | 2026-09-29 | Fault-Tolerant Budget Conservation in Distributed Multi-Agent Delegation |
| `limbo` | 2609.29095 | 2026-09-24 | Where Does Exactly-Once Live? Model, Harness, and Tool-Contract Effects on Duplicate Side Effects in LLM Agents |
| `moon` | 2605.09045 | 2026-05-09 | Containment Verification: AI Safety Guarantees Independent of Alignment |
| `gradual` | 2501.16946 | 2025-01-28 | Gradual Disempowerment: Systemic Existential Risks from Incremental AI Development |

Version dates checked: Terekhov et al. v2, 2026-03-02; Progent v3, 2026-05-14 (both cited as 2026).
Venue years: AgentSpec (ICSE 2026), IsolateGPT (NDSS 2025), Chan et al. (TMLR 2025).

Added after review 4 and checked the same way: `goex` 2404.06921 (Patil et al., 2024), `taubench` 2406.12045 (Yao
et al., 2024), `melon` 2502.05174 (Zhu et al., 2025), `taskshield` 2412.16682 (Jia et al., 2024), `shadearena`
2506.15740 (Kutasov et al., 2025). Classic references added from their journal records: Lamport 1998 (TOCS 16(2)),
Ongaro and Ousterhout 2014 (USENIX ATC), Kung and Robinson 1981 (TODS 6(2)), Bishop and Dilger 1996 (Computing
Systems 9(2)).

## Corrections made after reading the abstracts

- Zhu and Wang (`ftbudget`) prove at-most-once *settlement*, not dispatch.
- Li (`limbo`) finds that the tool contract decides duplication *when a write's outcome cannot be read back*
  (the model decides when it can), and the bound in its impossibility result is on in-flight *time*.
- `safetycase`, `needham`, `schoen` and `korbak` are about whether scheming can be found by evaluations and
  chain-of-thought monitors; they were previously described as "anti-scheming evaluations that ask when it chooses
  to allow [being stopped]".

## Web sources

- `metraudit`: metr.org page read; contributors David Rein and Adam Kaufman, dated 22 October 2026, update log
  29 October 2026. Matches.
- `controlroom`: openai.com returns HTTP 403 to automated fetches (direct, via a fetch agent, and the Internet
  Archive copy). OpenAI's RSS feed confirms the title "Control Room: A central system to track and stop model
  workloads across our research infrastructure", the date (9 November 2026) and the description "tracks model
  workloads across our research infrastructure and can rapidly stop them when safety or security concerns arise".
  The body text was not read, so the paper claims only that such a system stops model workloads (the producer of
  proposals). A sentence saying the first two stops are "what most deployed stop controls do" was withdrawn, and
  the 35-second halting figure reported by secondary coverage is not used.
- `iasr`: publication page resolves with the cited title.

