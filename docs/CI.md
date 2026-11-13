# CI runners

Linux Rust/proof/isolation checks and the paper build use the existing
`neverhuman-lan` organization runner group with
`[self-hosted, Linux, X64, lan-ci, ubuntu24]`. Jobs select an available disposable
Ubuntu VM from the general pool; they do not pin a physical host or use GPUs.
The native macOS functional check remains on `macos-latest`.

The group already permits all repositories, including public repositories.
No additional repository runner, service restart, group-access expansion, or
controller credential is required for this repo. The live group and selected
job runner names must be checked when verifying an infrastructure change.

General guests are smaller base images than GitHub-hosted runners. The workflow
installs its CPU build or TeX dependencies inside the disposable guest and pins
the Rust toolchain. All jobs check out into `ci-source`, disable checkout
credential persistence, and write builds/evidence under that checkout's
`target/`. Compiler concurrency stays at two jobs. Linux jobs have a bounded
30-minute allowance for cold provisioning, builds, proof tooling, and checks.

The workflow runs on pushes to `main`, pull requests, and manual dispatch.
Same-repository pull requests may use the LAN pool; fork PRs do not schedule
those jobs. The macOS functional check can still run for forks. Branch pushes
do not duplicate an open PR's jobs. A merged revision must also pass its main
push run on the general pool.

Before a source push run `make required`; use `actionlint` for workflow changes.
Verify CI against the exact PR head and read the jobs' `runner_name` and
`runner_group_name` to establish actual pool use. If jobs queue, inspect pool
access, matching labels and fleet health; do not silently substitute a hosted
Linux runner or a project-specific runner. Existing fleet credentials and
services remain controlled by the shared runner administrator.

The Linux job also runs `cargo test --locked --release --test kernel_guard_v2`
after building the separate-service protocol. This checks the authorization
guard with debug assertions disabled, including useful operations, denial
without effects, and compatibility of recovered permit encodings.

V2 reference provenance includes the workflow hash. A workflow update therefore
requires a fresh v2 run, validation, and regeneration of the paper's tables.
This preserves the evidence check without rewriting old source hashes by hand.
