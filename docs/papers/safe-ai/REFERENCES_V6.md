# Citation corrections and additions for the v6 integration (checked 2026-11-12)

`docs/references.tex` is Gemini-owned until PR #6 merges; apply these in the integration pass. Each addition says
what it supports. Add only works whose claims we actually use.

## Corrections

| Key | Now | Fix | Checked against |
|---|---|---|---|
| `camel` (`references.tex:28`) | "L. Beurer-Kellner et al." | **E. Debenedetti, I. Shumailov, T. Fan, J. Hayes, N. Carlini, D. Fabian, C. Kern, C. Shi, A. Terzis, F. Tramèr.** Defeating Prompt Injections by Design. arXiv:2503.18813, 2025. | arXiv abstract page |
| `progent` (`:29`) | "Progent: Securing AI Agents with Privilege Control", v3 2026 | correct as written (v3, 14 May 2026; authors T. Shi, J. He, Z. Wang, H. Li, L. Wu, W. Guo, D. Song). Note that E8 used the released `secagent` code at commit `b02783b` | arXiv abstract page |
| `moon` (`:30`) | Moon & Varshney, arXiv:2605.09045 v2 | correct (v2, 29 June 2026) | arXiv abstract page |
| AgentDojo | cited only as a GitHub commit | add the paper: E. Debenedetti, J. Zhang, M. Balunović, L. Beurer-Kellner, M. Fischer, F. Tramèr. AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. NeurIPS 2024 Datasets and Benchmarks; arXiv:2406.13352. Keep the commit for the code version | — |

## Additions (what each supports)

**Fencing, leases, capabilities, distributed systems** (§2–§3, E4, E6)
- M. Kleppmann. How to do distributed locking. 2016 (blog). *The fencing-token argument this paper's sink epoch fence
  implements; cite at the first use of "fence".*
- C. G. Gray and D. R. Cheriton. Leases: An efficient fault-tolerant mechanism for distributed file cache consistency.
  SOSP 1989. *Time-bounded authority; renew/withdraw.*
- M. S. Miller. Robust Composition: Towards a Unified Approach to Access Control and Concurrency Control. PhD thesis,
  Johns Hopkins University, 2006. *Object capabilities; authority by conferral, not identity.*
- A. Birgisson, J. G. Politz, Ú. Erlingsson, A. Taly, M. Vrable, M. Lentczner. Macaroons: Cookies with Contextual
  Caveats for Decentralized Authorization in the Cloud. NDSS 2014. *Attenuated delegation (already used, uncited).*
- M. P. Herlihy and J. M. Wing. Linearizability: A Correctness Condition for Concurrent Objects. ACM TOPLAS 12(3),
  1990. *The concurrency checker's correctness condition (already used, uncited).*
- K. Kingsbury. Jepsen (jepsen.io). *Cite only to say the partition schedules here are a logical-time simulation, not
  Jepsen.*

**Verified systems** (§2 kernel)
- C. Hawblitzel et al. IronFleet: Proving Practical Distributed Systems Correct. SOSP 2015.
- J. R. Wilcox et al. Verdi: A Framework for Implementing and Formally Verifying Distributed Systems. PLDI 2015.
- A. Lattuada et al. Verus: Verifying Rust Programs using Linear Ghost Types. OOPSLA 2023. *(the verifier used)*

**Cryptography** (§2, TCB statement)
- D. J. Bernstein, N. Duif, T. Lange, P. Schwabe, B.-Y. Yang. High-speed high-security signatures. Journal of
  Cryptographic Engineering 2(2), 2012. *Ed25519 (already used, uncited).*

**AI control, agent security, deceptive models** (§1, §4, §7)
- R. Greenblatt, B. Shlegeris, K. Sachan, F. Roger. AI Control: Improving Safety Despite Intentional Subversion.
  ICML 2024; arXiv:2312.06942. *Trusted monitoring baseline; E5's same-model reviewer is a weaker cousin.*
- A. Bhatt, C. Rushing, A. Kaufman, T. Tracy, V. Georgiev, D. Matolcsi, A. Khan, B. Shlegeris. Ctrl-Z: Controlling AI
  Agents via Resampling. arXiv:2504.10374, 2025. *Resampling protocols; contrast learned control with state
  mediation.*
- M. Costa, B. Köpf, A. Kolluri, A. Paverd, M. Russinovich, A. Salem, S. Tople, L. Wutschitz, S. Zanella-Béguelin.
  Securing AI Agents with Information-Flow Control. arXiv:2505.23643, 2025 (FIDES). *E8's specification S enforces
  identifier provenance but not information flow; FIDES is the IFC alternative and names the gap.*
- R. Greenblatt et al. Alignment Faking in Large Language Models. arXiv:2412.14093, 2024.
- E. Hubinger et al. Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. arXiv:2401.05566,
  2024.
- A. Meinke, B. Schoen, J. Scheurer, M. Balesni, R. Shah, M. Hobbhahn. Frontier Models are Capable of In-context
  Scheming. arXiv:2412.04984, 2024. *Motivation: why evidence about a proposer's disposition cannot carry the safety
  case.*

## Still to verify before citing (2026 web sources already in the draft)

Quote OpenAI's Control Room only from OpenAI's own post. Check the METR/Redwood audit's author line ("Rein &
Kaufman") and every date against the primary pages.
