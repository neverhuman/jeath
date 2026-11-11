HUMAN BRAIN AND BEHAVIOR — The Minds We Choose

This is one of two separate papers. Read /PAPERS.txt before editing. Existing
Rust authority experiments and docs/intelligence-is-not-authority.tex belong to
the SAFE AI paper. No SAFE AI result is used as human empirical evidence.
Performance Before Outcomes is historical material for this human paper.

CURRENT SOURCES AND OUTPUTS
  Manuscript: docs/sections/brain/minds-*.tex
  Analysis and plotting: scripts/brain/
  Frozen validated reference: evidence/brain/reference/minds-we-choose.json
  Specification, coverage and crosswalk: evidence/brain/
  Immutable originals: tips/ and project/brain/tips/, pinned in tips/manifest.json
  Transient runs, public source cache, plots, PDFs and inspection: target/minds-we-choose/

  make paper-brain
  # Alias: make minds-we-choose

  Main: target/minds-we-choose/minds-we-choose.pdf
  Supplement: target/minds-we-choose/minds-we-choose-supplement.pdf
  Combined: target/minds-we-choose/minds-we-choose-combined.pdf
  Vector figures and aggregate tables: target/minds-we-choose/generated/
  Page counts: target/minds-we-choose/build-report.json

RECOMPUTE AND VALIDATE
  python3 scripts/run.py target/minds-we-choose/run-one --study minds-we-choose
  python3 scripts/validate.py target/minds-we-choose/run-one --study minds-we-choose --source
  python3 scripts/run.py target/minds-we-choose/run-two --study minds-we-choose
  python3 scripts/validate.py target/minds-we-choose/run-two --study minds-we-choose --source --compare target/minds-we-choose/run-one

The runner first reproduces the earlier platform reference, then streams bounded
read-only aggregate scans on xbabe3. It checks hostname/architecture, input sizes,
hashes and unchanged input stats; source identifiers and coordinates remain there.
Do not run concurrent remote scans or modify image-experiment caches. The runner
retrieves hash-pinned public survey/boundary files into this lane's target cache.
An aggregate-only recorded snapshot can be reused with the module CLI's
--remote-snapshot option; label this as a replay rather than a fresh raw-data scan.

After validating a deliberate scientific update, explicitly copy its JSON to
evidence/brain/reference/minds-we-choose.json and regenerate with:
  python3 scripts/paper.py --study minds-we-choose
  python3 scripts/paper.py --study minds-we-choose --check
  python3 scripts/build_paper.py --study minds-we-choose

Never edit numerical text, tables or figures directly in generated/. The committed
human reference is independent of evidence/reference/*.json for SAFE AI, apart
from the explicitly named legacy human performance-before-outcomes reference.

DEPENDENCIES AND CHECKS
Python 3, NumPy, SciPy, pandas, pyarrow, matplotlib, openpyxl, pyshp; pdfLaTeX
with the packages named in minds-preamble.tex; Poppler pdfinfo/pdftoppm.
The analysis-coverage document records the implementation environment. pyshp can
be installed locally in target/minds-we-choose/deps; no remote installation is
needed. CPU aggregate analysis only; no GPU allocation or new image training.

  python3 -m unittest discover -s tests -p 'test_minds*.py'
  python3 -m unittest discover -s tests -p 'test_brain_paper.py'
  make required

The independent B-spline test recalculates every principal HON contrast without
using the production truncated-power basis. Other tests exercise geographic
conflicts, ambiguous aliases, territories, holes, coordinate order, percentile
ties, country weighting, observer/context validity, and weighted decomposition.
make required remains the existing repository gate and also runs the new Python
tests; its default experiment/paper still belongs to SAFE AI. Use paper-brain for
the separate human-paper PDF checks.

INTERPRETATION AND REMAINING SOURCE LIMITS
Expanded work is exploratory; prior visibility and subsequent factual corrections
are documented. Countries are supported convenience samples, not population-
representative national estimates. HON residence is recorded; POV residence is
derived, with ambiguous/conflicting records unresolved. Metadata, inventoried
images, older decoded checkpoints, completed processing and modeling samples are
different populations. GSS's independent-year clustered sensitivity retains its
explicit design and coverage assumptions and does not establish causality.

Walter's literal source gender key remains unavailable, but exact final published
counts support a documented conditional label mapping. Whyte's 265-record article/
deposit discrepancy, POV's original relationshipage_id wording, full matching frozen
image-model prediction vectors, and final all-image duplicate components remain
unresolved. The paper
quarantines or narrows the affected claims. Missing values are not reconstructed
to match a desired story. Read analysis-coverage.json and the claim-change CSV.

COORDINATION
Shared implementation log: project/brain/tips/COMMS.md (append only).
Only one integration writer edits shared run/paper/build entrypoints. Other
writers use their own permitted worktrees. Never alter archival TeX/ZIP/PDF bytes.

AGGREGATE REPLAY
  python3 scripts/brain/minds_study.py target/minds-we-choose/replay --remote-snapshot evidence/brain/reference/minds-we-choose.json --demographic-frozen evidence/brain/reference/age-gender-baseline.json

Both switches together replay the validated platform snapshot and separate
benchmark baseline locally. Public source hashes are still checked and local
survey/GSS fits recomputed. The execution record identifies replay versus fresh
source scans. The age/gender diagnostic uses benchmark ages18-99, distinct from
the primary HON18-79 cohort, with no image reads or new image-model training.
