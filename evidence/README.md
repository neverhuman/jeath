# V2 evidence

`v2/reference/` records the publication CPU profile: pinned core proof output,
24 sequential fault histories, repeated 1–32 actor races, scripted controller
controls, native Linux confinement probes, negative controls, and the abstract
single/pairwise omission search. It includes native AgentDojo/Progent scripted
integration controls and a separate model feasibility report. No model campaign
or learned-review result is present. Raw SQLite databases remain in the worktree’s
`target/v2-release/`; committed JSON preserves invocations, responses
and independently read sink effects. Kernel, source, protocol, tool and checkpoint
identifiers are recorded in the manifests.

`reference/` below remains the v1 record against commit `12b5579`, with one added
`v2-status.json` pointer. Validate preserved source with `--source-ref 12b5579`.
Original v1 evidence documentation follows.

# Recorded evidence

`reference/` is the full CPU configuration used for the paper's generated tables.
Its `provenance.json` names the actual host, toolchain, measured source hashes and
binary hash. `metadata.json` records seed and configuration; `complete.json` is
written only after the Rust suite finishes. Python additionally exports actual
SQLite effects into `observed-effects.json`.

`fleet/` contains separate small runs on verified hosts. They use the same binary
but a smaller configuration; they are not replications of the full monitor sample
size, independent hardware trials of a probability, or GPU experiments. Each
deployment record identifies the host and binary. Remote runtime databases remain
at the recorded location; no credentials or non-synthetic data were copied.

Raw per-seed outputs, approximate uncertainty, negative controls and shortest
finite-model counterexamples are retained. Timings include host contention and
must not be presented as hardware capability rankings. Published evidence is
never overwritten automatically by CI. CI uploads its own fresh runs separately.
