# V2 revision status

Implemented: shared executable Verus core; separate broker/sink processes and
stores; uncertain reservations; immutable normalized permits; per-issuer
withdrawal, renewal and sink fencing; Landlock/seccomp sessions; semantic controls;
independent sequential history checking; CPU races/controller controls; ten
abstract omissions and 45 pairs; versioned manifests; native pinned
AgentDojo/Progent controls; predeclared inference and model feasibility.

The main manuscript centers revocation exposure, the proof boundary and local
results. Historical synthetic and human-analogy material is in appendices. V1
evidence and `tips/` are unchanged; a separate v2 status pointer is added.

Added managed inference with hash checks, fixed deadlines, one-server scheduling, learned review and CPU bridge controls. Open: GPU execution and five-controller Qwen3 campaigns; model-driven
AgentDojo/Progent comparison; full concurrent history checking; mechanized SQL
adapter/whole-protocol refinement. All verified Linux GPUs were occupied. These
gaps are disclosed, not filled with scripted checkpoint labels. The PR is a
reviewable implementation revision with narrowed claims, not acceptance of all
three target claims.

Earlier v1 revision notes follow.

# Revision and evidence audit

The supplied folder had three TeX sources and no Git repository, scripts, numbers
files or figures. The two lowercase manuscript files are identical. Original
bytes and hashes are preserved under `tips/`. The revised source has one file per
section, front matter, preamble, references and generated experimental components.

| Original claim | Disposition in this revision |
| --- | --- |
| A Python checker and full supplement accompany the paper | Withdrawn as an availability claim; absent from the supplied folder |
| 10,180 states / 223,960 checks / 20 positive tasks | Reproduced by a fresh Rust reconstruction of the described model |
| 41,732 states / 918,104 checks / eight units under an omitted budget | Reproduced, with a separate reference oracle and retained counterexample |
| Eight listed enforcement mutations | Extended to nine, including the omitted destination check |
| Six-family monitor percentages were reproduced | Removed; new two-family training experiment, fresh raw data and explicit design |
| Runtime boundary, multi-issuer grants and recovery are proposed | Limited local SQLite versions implemented and exercised; deployment study remains open |
| Useful tasks prove safety | Explicitly rejected; retain both non-vacuity and specification-omission controls |
| Independence follows from diverse monitors / different stages | Explicitly rejected; conditional products and imposed independent sampling kept separate |
| Caregiving/evolution establishes a caring AI objective | Kept as motivation for an untested hypothesis; no behavioral or human result inferred |
| A model invariant is implementation verification | Explicitly rejected; no formal refinement proof or OS sandbox claimed |

The conditional exposure proof, persistent-ledger induction, fixed-sample zero-failure
limit, transfer assumptions, rank-threshold result and crossed-variance calculation
were retained with their assumptions. The statistics implementation uses stable
`expm1`/log-CDF calculations; exact small cases and the 2,995,731-trial threshold
are checked. Exhaustive closure covers the finite transition system, including
cycles; boundary tests alone do not establish adequate specification or legitimacy.

Primary-source checks covered the newer incident/evaluation claims, the Kogui
statement, CARE principles, AI control, Guaranteed Safe AI, adaptive/distributed
monitor attacks, and SQLite transaction/WAL semantics. The Kogui quotation and
attribution were checked against the linked statement; no endorsement or consultation
is claimed. Bibliographic resolution is checked during the PDF build; it is not
a guarantee that every literature interpretation is correct. An unavailable
author-preprint citation was removed from the argument. Original human-history
illustrations were not promoted into AI measurements.

Reviewing current SQLite documentation exposed the concurrent WAL-reset bug in the
initially cached 3.46.0 bundle. The final dependency was upgraded and a minimum
fixed-version check added; final retained measurements use the patched build.
Preliminary runs remain local in `target/` for inspection and are not the paper's
reference dataset. No claim is made that ordinary passing tests detected that rare
bug. See [SQLite's upstream report](https://www.sqlite.org/wal.html#the_wal_reset_bug).

Open work remains substantial: independently audit specifications and oracles;
establish authenticated role separation and host isolation; prove or test refinement
for external tools; design sealed stronger-attacker transfer studies; train and
evaluate caretaker objectives with independent seeds; and obtain actual participant
data before making claims about human delegation. The supplied offline adapter
supports mechanical proposal/effect endpoints, not semantic truthfulness or a
human consent study. These are scientific limits, not missing evidence relabeled
as completed work.
