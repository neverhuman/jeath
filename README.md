# jeath — Intelligence Is Not Authority

[Revised paper (PDF)](docs/intelligence-is-not-authority.pdf) · [Sectioned LaTeX](docs/intelligence-is-not-authority.tex) · [Recorded evidence](evidence/reference) · [CI](https://github.com/neverhuman/jeath/actions)

Small Rust experiments that test the paper's assumptions, with real SQLite
transactions, competing processes, crash recovery, and classifiers trained from
scratch. No model API, GPU, real funds, or personal data is needed.

```bash
cargo test --locked --all-targets
python3 scripts/run.py target/my-first-run --quick
python3 scripts/validate.py target/my-first-run
```

Use a new output directory for each run. Omit `--quick` for the paper configuration;
the first full CPU run took about 45 seconds, though durable-storage tests can take
longer under host contention. Run `make required` for formatting, Clippy, tests, a
fresh small experiment, evidence consistency and the paper build.

Rust stable, a C compiler and Python 3 are required. The crate bundles a patched
SQLite and requires SQLite ≥3.51.3 at runtime. The Python scripts use only the
standard library. Building the paper also requires `pdflatex` with TeX Live's
LaTeX-extra, fonts-recommended, science and pictures packages. CI checks Linux
and macOS and uploads fresh experiment JSON and the PDF.

What is implemented:

- Exhaustive 22-action model checking, nine enforcement mutations, shortest
  counterexamples, all 20 useful tasks, and a deliberately inadequate specification.
- A persistent executor with exact-request approval, operator/steward conjunction,
  shared quota, epoch revocation, immutable IDs, retries, expiry and revision checks.
- Actual process death on both sides of commit, eight-process budget contention,
  concurrent checkpoint stress, and one/four-process latency measurements.
- Small neural/logistic monitor training, shared versus independent inputs,
  surrogate edits, feedback-guided selection, conditional misses and false blocks.
- Exact graph cuts, an omitted bypass, paired coverage-report worlds, planted-audit
  power, reward-selection simulations and stable binomial confidence limits.
- An offline JSONL adapter for externally generated proposals, with separate
  unauthorized-proposal, executed-effect and useful-task scores.

```bash
cargo run --release --locked -- campaigns fixtures/proposals.jsonl target/my-campaigns
cargo run --locked --example transaction -- target/my-example.sqlite
make paper
```

The supplied JSONL is explicitly scripted control data. Replacing it with a model's
outputs runs an actual model evaluation; metadata labels alone do not establish
that a model was trained on an objective or that persona was randomized.

Protected effects in the prototype are **inside one database**: synthetic account
credits, synthetic mailbox releases, and a versioned toy source document. The
proposer must not have direct database or administrator access. Authentication,
OS isolation, external tools, human persuasion, formal implementation refinement,
frontier-model transfer and caretaker training are still outside these results.

All three original manuscripts remain byte-identical in [`tips/`](tips), including
the duplicate. Their claimed Python supplements were absent. The revised paper
reconstructs the finite model and replaces the unavailable statistical studies
with new, identified experiments. See [experimental methods](docs/EXPERIMENTS.md)
and [revision audit](docs/REVISION_NOTES.md) for the evidence boundary and open work.
