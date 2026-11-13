# Verified kernel checks in release builds

The broker and sink now require agreement with the executable Verus kernel before adopting a guarded transition. This replaces a sink comparison that was previously only a debug assertion. It does not establish formal refinement of the whole broker or tool service.

Each guarded operation constructs the kernel image of its pre-state, computes a candidate Rust transition on a clone, and compares both the outcome and projected post-state with the kernel. Disagreement or a state without a valid kernel image returns a denial. The original state is retained, and the service rolls back its SQLite transaction before responding. A newly committed effect or fence acknowledgment is returned only after agreement and durable commit.

| Guarded operation | Kernel decision |
| --- | --- |
| Submission, including retries and altered identities | Reservation or existing-record decision |
| Settlement | Committed or authoritative-cancellation settlement |
| Renewal | Issuer renewal without replenishing charge |
| Sink acceptance | Immutable identity binding and epoch-floor check |
| Completion | Idempotency, revision, epoch and charge accounting |
| Fencing | Monotone floor and cancellation of older pending work |

Broker refusals that change no state may return without a kernel comparison; the guard does not establish availability equivalence for every denial. Scope rules, approvals, effect-to-issuer mapping, delegation lineage and ancestor budgets remain checked by unverified application code. Withdrawal is not subject to this guard because refusing a withdrawal could leave authority live; malformed or corrupted storage can still prevent withdrawal. Settlement still trusts the sink status relayed by the trusted launcher. State projection, outer JSON parsing, document contents, SQLite serialization and durability, the operating-system kernel, and cryptographic primitives remain in the trusted base.

Run the production-profile regression directly:

```sh
cargo test --locked --release --test kernel_guard_v2
```

It exercises permitted transfers and edits, retries, stale revisions, withdrawal, cancellation, settlement and renewal. Negative controls cover a divergent completion, malformed issuer vectors, states without a kernel image, unchanged state after refusal, and the actual service's effect rows after rollback. A release-only test confirms debug assertions are disabled. The Linux local gate and general-runner CI execute this command; native macOS functional tests remain.

The fresh protocol/fault/race evidence and source hashes are in `evidence/v2/reference`. Verus receipts identify the exact policy core and rejected soundness mutations. The v6 model and stop datasets retain their original binary provenance: they were collected before this guard. Passing the new regressions does not retroactively verify those binaries or establish new model safety or usefulness results.
