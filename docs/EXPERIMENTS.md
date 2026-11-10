# Running and interpreting the experiments

`make required` is the local publication gate. It keeps build and test artifacts
under `target/`, never under `/tmp`. `Cargo.lock` fixes dependencies; the manifest
of each measured run records Rust, SQLite, platform, seed, source and binary hashes.

`python3 scripts/run.py target/new-run [--quick] [--seed 20261110]` builds once and
runs the following CPU fixtures. All output directories must be new.

| Output | Experimental unit / method | Interpretation |
| --- | --- | --- |
| `model.json` | Full reachable-state closure, 22 actions per accepted state | All policies within the finite model; no probability over actions |
| `runtime.json` | Eight scenario groups, actual children and independent SQLite connections | Bounded implementation checks; no universal deployment failure estimate |
| `observed-effects.json` | Read-only snapshots of balances, deliveries, revisions, receipts and policy | Effects can be checked without trusting worker success strings |
| `monitors.json` | Four/eight independent task/training seeds; 300/1,000 hazardous episodes per arm | Joint misses conditional on synthetic task, fixed trained monitors and the specified attacker |
| `coverage.json` | Exactly paired visible reports; 200/2,000 planted-audit repetitions | Identifiability and power under stipulated reporting mechanisms |
| `graphs.json` | Full DAG path count and independently implemented vertex cut | A true cut blocks the given graph; an unreported route defeats a partial-graph certificate |
| `selection.json` | Four/eight populations; 100/400 agents; 25/60 generations | Consequences of specified rewards, not evidence of AI motivations |
| `statistics.json` | Exact arithmetic and one-sided fixed-sample binomial limits | Assumptions must be checked before applying to empirical episodes |

The quick/full classifier designs use 300/600 training examples, 300/800 benign
calibration examples, 15/25 SGD epochs, eight features, and six gates. Neural
models have eight tanh hidden units. Mixed arms alternate three networks with
three logistic regressions. Models receive separate training samples, with
disjoint calibration and evaluation data. Thresholds use the benign 95th sample
quantile. False blocks are reported for the complete protocol.

The shared-input arms use identical edited examples. The independent-input arm
draws each gate's input independently and edits it separately using a fixed
surrogate. The feedback arm chooses among eight candidates using the defended
protocol's scores. Its attacker has more feedback; this is a changed attack
condition, not a matched-compute experiment. Edits keep the true label hazardous,
remain in the original cube and have L2 radius ≤1.5. No real harmful content is
generated. Inspect both raw per-seed counts and descriptive 1.96-SE summaries;
eight seeds do not justify treating the interval as exact or uniform.

Runtime latency excludes initial schema creation, preparation and grant writes.
It includes transaction locking, validation, effects, receipt persistence and
commit. Throughput includes worker startup/scheduling after the start barrier.
All writes use SQLite WAL and synchronous FULL. This is a local-filesystem
workload, not a network database or external transaction benchmark. The SQLite
dependency includes the WAL-reset fix; the floor rejects unpatched older builds.
Checkpoint stress is a bounded regression, not an organic reproducer of that bug.

The process-death worker prints and flushes a phase marker after effect/accounting
SQL and immediately before commit, or immediately after commit but before returning
a receipt. The controller kills that worker, reopens the database, audits effects,
and retries. This tests actual abrupt process exit and lost receipts. It does not
test power loss or every possible crash instruction.

To update paper evidence, retain the run under `target/`, validate it, copy only
its top-level JSON into `evidence/reference`, run `python3 scripts/paper.py`, and
run `make required`. The `--source` evidence check deliberately fails if the
measured sources differ from the current checkout. CI never silently replaces
the published data with its own timings.

For external proposers, use `authority-lab campaigns INPUT.jsonl NEW_DIRECTORY`.
The schema and six scripted controls are in `fixtures/proposals.jsonl`. Fixtures
cover legitimate spending, shared budget, withheld steward permission, purpose
change and revision conflicts. The executor grants only the fixture's task domain;
the event scorer separately compares observed effects with fixture authorization.
The reject-all control must fail useful-task completion. These are public development
fixtures; make genuinely sealed tasks before claiming held-out behavioral transfer.
Treat model, training-seed, objective and persona fields as provenance to verify,
not facts established by the scorer. Repeated campaigns from one trained model or
human are clustered units, not automatically independent binomial trials.

`python3 scripts/fleet.py target/new-fleet --hosts xbabe0 xbabe1` copies only the
measured Linux binary and runs small CPU fixtures at reduced scheduling priority.
It checks the real hostname and architecture before writing anything and preserves
remote evidence under `~/jeath-evidence`. It rejects misleading aliases. The local
workspace identifies as xbabe2; the current xbabe2/xbabe4 SSH aliases point at the
same Mac Studio, and the xbabe4 VM alias timed out. Neither is claimed as a Linux
xbabe4 measurement. No GPUs were allocated.
