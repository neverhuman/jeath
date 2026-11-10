# jeath

The host's worktree, lane ownership, build-output and preservation rules apply.

- Required local gate before every push: `make required`.
- `tips/*.tex` are archival originals. Never edit their bytes; `tips/manifest.json`
  records the supplied hashes. Maintain the revised manuscript in `docs/sections/`.
- Scientific changes require a new `scripts/run.py` run, validation, and an explicit
  update of `evidence/reference/*.json`; regenerate tables with `scripts/paper.py`.
  Do not edit numbers directly in generated tables. Preserve negative results.
- Put builds and transient runs in this checkout's `target/`. Test evidence is
  intentionally retained there; inspect it before cleanup. Never build in `/tmp`.
- CPU fixtures are the default. Do not allocate GPUs or launch large fleet jobs
  as a routine gate. Verify remote hostname and architecture before fleet work.
- The SQLite executor mediates only its own database effects. Do not describe
  its tests as sandbox verification, external exactly-once execution, formal
  refinement, frontier-model results, or human-participant evidence.
- `scripts/github.py` selects the existing neverhuman credential without changing
  global authentication. Never print credentials or include them in Git URLs.
