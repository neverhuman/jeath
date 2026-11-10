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
