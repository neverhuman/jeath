# PAPER MISSION: The Charter for World-Class Scientific & Technical White Papers

A forgettable paper reports everything its authors ran in the order they ran it. A world-class white paper—the kind read ten years later across disciplines—feels **inevitable**: every paragraph pays rent, every equation cashes out in data, every figure could stand alone on a seminar slide, every color means one thing from Page 1 to the final Appendix, and every claim stops at the exact boundary of what its evidence proves.

This document specifies the **Eight Governing Laws** for authoring, refactoring, and auditing any flagship scientific or engineering white paper. Every rule is domain-agnostic and enforceable as an explicit pass/fail gate.

---

## The Prime Directive

Maximize **verified insight per page**:

$$\text{Editorial Quality} = \frac{\text{Causal Clarity} \times \text{Quantitative Rigor} \times \text{Visual Self-Sufficiency}}{\text{Pages} \times \text{Epistemic Overreach}}$$

Length is a cost, never a credential. If a `60`-page draft and a `30`-page monograph carry the exact same theorems, datasets, sensitivity audits, and proofs, the `30`-page monograph is twice as strong.

---

## Law I — Zero-Filler Prose & The Single-Home Principle

### 1.1 The Four Legal Moves of a Paragraph
Every paragraph in the main text must execute at least one of four moves—and stop as soon as that move completes:
1. **Frame a non-trivial anomaly or timing/scale paradox** that existing accounts fail to resolve.
2. **State or interpret a formal mechanism** (an identity, bound, equilibrium, or impossibility ceiling) in plain physical or behavioral terms.
3. **Test a sharp prediction** with an explicit sample size ($N$), estimand, uncertainty interval, and sign/magnitude comparison.
4. **Bound scope, failure modes, or decisive falsifiers** showing what would overturn the claim.

### 1.2 Forbidden Prose Patterns
- **No throat-clearing meta-roadmaps**: Delete *"The remainder of this paper is organized as follows: in Section II we discuss..., in Section III we turn to..."*. Let section headings and causal transitions do the routing.
- **No twin summary lists**: Never preview a 10-point bullet/clause list in the Introduction and repeat the same 10 points in the Discussion. State the puzzle and architectural test in the Introduction; synthesize cross-domain implications and limits at the close.
- **No multi-home narration**: Every dataset and derivation has **one canonical home**. Main text carries the causal spine, primary estimands, and flagship graphics; Technical Appendices carry complete proofs, data-hygiene pipelines, and sensitivity grids. Never narrate the same empirical study twice in Main Text and Supplement.
- **No leaked scaffolding**: Internal peer-review Q&A memos, version changelogs, todo notes, and unconfirmed biography placeholders must never appear in a compiled manuscript.

---

## Law II — Five-Act Narrative Momentum & The Page-1 Contract

### 2.1 The Page-1 60-Second Contract
A reader scanning **only Page 1** (Title, Abstract, Opening Columns, and the Page-1 Synthesis Graphic) must recover within 60 seconds:
1. **The Paradox**: Why the phenomenon is hard to explain under standard assumptions.
2. **The Formal Law**: What mathematical mechanism resolves the paradox—and what ceiling or trade-off it imposes.
3. **The Empirical Span**: The independent scales, eras, or modalities tested (with exact $N$s).
4. **The Boundary / Null**: Where the mechanism turns off, saturates, or reverses.

Abstracts must stay tight (`180–225` words): four crisp movements (*puzzle $\to$ formal decomposition $\to$ decisive multi-dataset measurements $\to$ boundary/consequence*), zero undefined jargon, zero citation clutter.

### 2.2 Five-Act Macro-Architecture (Never a Flat Laundry List)
Never present `12–15` flat subsections at equal hierarchy (`II-A` through `II-N`). Group evidence into **4–5 tightly coupled macro-acts** (each with `2–4` subsections), ordered so the closing tension of Act $k$ forces the opening question of Act $k+1$:

```
Act I   : The Anomaly & Timing/Scale Paradox       (Why naïve selection/optimization fails)
Act II  : Formal Mechanism & Information Ceilings  (Exact equations + controlled experimental test)
Act III : Historical, Physical & Cultural Baselines (Deep-time / cross-population invariance & gates)
Act IV  : High-Resolution Modern Crucible          (Large-scale behavioral, geometric, or ML ceilings)
Act V   : Macro-Consequences, Limits & Falsifiers  (Institutional/system drift + how to disprove us)
```

### 2.3 Declarative Thesis Headings
Headings must state **what is established**, not a passive topic bucket:
- *Weak*: `"2.4 Height and dating"` or `"3.2 Marriage and child counts"`
- *World-Class*: `"Male Stature Enforces a Lower-Tail Floor, Triggers a 6'0\" Staging Cliff, and Yields Null Fertility"`

---

## Law III — Four-Tier Epistemic Calibration & Honor for Nulls

### 3.1 Explicit Claim Typing
Every quantitative or historical assertion belongs to one of four strict epistemic tiers—and prose must never silently promote a lower tier into a higher one:

| Tier | Definition | Required Linguistic Precision |
|---|---|---|
| **1. Derived** | Exact mathematical consequence of stated model assumptions | *"Under Assumptions A1–A3, validity saturates at $R^2_{\infty,1}$ (Proposition 20)."* |
| **2. Measured** | Direct empirical estimand with $N$, uncertainty, and sensitivity audit | *"Across $N=6{,}683{,}895$ profiles, the top male decile receives $55.8\%$ of positive votes."* |
| **3. Accounting** | Exact algebraic decomposition of observed rates or variances | *"Rising non-marriage accounts for $94\%$ of the cohort birth decline."* (Not a solo root-cause claim.) |
| **4. Conjectured / Hypothesis-Generating** | Mechanistic alignment across sparse or asynchronous historical records | *"Cooking near $0.78\text{ Ma}$ aligns with Phase IV release, though wide breakpoint intervals make timing suggestive rather than probative."* |

### 3.2 Disciplined Treatment of Contested & Asynchronous Evidence
When engaging active scientific debates (e.g., ancient demographic bottlenecks, paleoclimatic forcing, impact craters, or benchmark saturation):
1. **Check chronological ordering first**: An event at $t_2$ cannot initiate a regime shift that began at $t_1 < t_2$, though it may perturb the late phase or trigger exit. Show lead/lag timing explicitly on the timeline.
2. **Separate physical occurrence from biological/causal impact**: Establishing a dated volcanic eruption, asteroid tektite field, or geomagnetic reversal is physical stratigraphy—not automatic proof of mass mortality or selection.
3. **Pair contested inferences with their published critiques**: Cite both the primary inference and independent methodological reanalyses, distinguishing effective/latent model parameters (such as coalescent $N_e$) from literal census counts.

### 3.3 Elevate Negative, Null, and Ceiling Results
Advocacy hides nulls; world-class science **builds load-bearing walls out of them**. When `50+` distinct model architectures hit the same out-of-sample ceiling, when a strongly preferred cue yields zero net reproductive differential, or when a celebrated biomarker meta-analyzes to $z \approx 0$, feature that boundary prominently in both graphics and text: showing *where an instrument stops* proves the theory has teeth. Always close with **pre-committed decisive falsifiers**.

---

## Law IV — Visual Self-Sufficiency & Float Choreography

### 4.1 The Slide-Deck & Caption Contract
A domain expert reading **only the figures, direct canvas annotations, and captions** must recover $\ge 90\%$ of the paper's empirical and theoretical story. Every caption must follow a four-part template:
1. **Bold declarative headline** stating the takeaway of the whole figure.
2. **Panel-by-panel `(A)–(D)` mapping** naming variables, units, sample sizes ($N$), and error-bar definitions (`95%` bootstrap CI, Wilson interval, SE).
3. **Key quantitative anchor** so the reader sees exact magnitudes alongside visual geometry.
4. **Epistemic tier tag** (*Derived + simulation check*, *Measured microdata*, *Synchronized open paleoclimate + fossil reanalysis*, etc.).

### 4.2 Single Programmatic Engine at Exact 1:1 Print Scale
- **One codebase, one visual voice**: Render **100% of figures** programmatically from a single styling module. Never paste together heterogeneous figures from earlier drafts with mismatched fonts, spine weights, or palettes.
- **1:1 Target Dimensions**: Build every canvas at exact journal column width (`3.49 in` single-column `figure`, `7.16 in` double-column `figure*` in IEEE Transactions) and embed at `100%` scale. That guarantees invariant `8.0 pt` bold panel titles, `7.0 pt` axis labels, `6.4 pt` tick labels, and `5.8–6.3 pt` callouts across every page.
- **Despined Minimalist Geometry**: Omit top and right spines (`0.65 pt` muted axes), use subtle horizontal reference grids (`0.45 pt`, `#E5E4E0`) only where magnitude reading benefits, and maximize data-ink ratio.
- **Direct Canvas Callouts & Synchronized Tracks**: Place inflection markers, regime boundaries, timing offsets, and curve labels *directly beside the data* with crisp leader lines. When comparing coupled processes across time or percentiles (e.g., endocranial volume vs. global benthic $\delta^{18}\mathrm{O}$ cooling, or height density vs. attraction percentile), lock them onto shared horizontal axes.

### 4.3 Zero-Drift Float Choreography
- Balance single-column (`figure`, `table`) and double-column (`figure*`, `table*`) floats so LaTeX never queues five full-width floats in a row.
- Every figure and table must appear **on the same page or immediately facing/following page** (`0–1` page lag) of its first prose callout.
- Terminate main back-matter and each major Appendix block cleanly so **zero floats ever strand at `\end{document}`**.

---

## Law V — Invariant Semantic Color Grammar & Accessibility

Color in a scientific paper is a **typed data channel**, never decoration.

### 5.1 One Hue = One Concept Across the Entire Manuscript
Define a compact, high-contrast, color-vision-deficiency-safe (Okabe–Ito-derived editorial) palette once, and bind each color to a fixed semantic meaning across every panel of every figure:

| Semantic Channel | Hex | Visual Encoding & Redundant Cue |
|---|---|---|
| **Cohort A / Female Series** | `#B84A27` (Deep Vermilion) | Filled diamond (`D`), solid stroke |
| **Cohort B / Male Series** | `#1B5E8A` (Deep Cerulean) | Filled circle (`o`), solid stroke |
| **Multi-Context / Verified Signal / Measured Proxy** | `#0F6E66` (Deep Viridian) | Solid curve / filled bar |
| **Single-Context / Staged Cue / Contested Inference** | `#C87D16` (Burnished Ochre) | Dashed curve `(4, 2)` / open or hatched marker |
| **Exact Formal Law / Step Model / Oracle Ceiling** | `#18222F` (Midnight Ink) | Crisp `1.6 pt` primary trajectory |
| **Null / Unlinked Event / Residual Gap / Baseline** | `#7A7872` (Warm Slate) | Dotted rule `(2, 2)` / muted band `#F2F1ED` |
| **Zero-Centered Spatial / Signed Attribution** | `#08306B ↔ #F5F5F2 ↔ #9E2A2B` | Strictly symmetric around `0.0` (`Sapphire −Δ ↔ Ivory 0 ↔ Crimson-Gold +Δ`) |

### 5.2 Monochrome & Perceptual Safeguards
- **Never reuse a semantic hue** for an unrelated category in another plot (e.g., never use the single-context ochre for female curves in one plot and geographic regions in the next).
- **Redundant non-color encoding**: Pair every color contrast with distinct marker geometry (`D`, `o`, `s`, `^`), line dash cadence, or bar hatching (`///`) plus direct text labels so the paper remains 100% interpretable when printed in black-and-white.
- **Neutral zero in heatmaps**: Diverging attribution maps must map `0.0` to neutral paper-ivory so uninformative regions never create false visual hotspots.

---

## Law VI — Numerical `booktabs` Tables vs. Plots

### 6.1 Strict Division of Labor
- **Use a Figure** when the message is trajectory, curvature, threshold discontinuity, spatial topology, multi-regime synchronization, or cross-group invariance.
- **Use a Table** when the reader needs exact multi-column estimands, sample sizes, uncertainty bounds, model-comparison criteria, or side-by-side episode decompositions.

### 6.2 Ban on Prose-Wall Tables
- Never place `\scriptsize` multi-paragraph literature summaries inside main-text `table*` boxes; prose disguised as a table fragments reading flow and clogs double-column float placement.
- Format all tables with 3-rule `booktabs` (`\toprule`, `\midrule`, `\bottomrule`), **zero vertical lines**, explicit units and denominators in column headers, true minus signs (`−0.042`, never hyphen `-0.042`), and aligned decimals. Keep `3–5` decisive numerical tables in the Main Paper; house exhaustive wave ledgers and sensitivity matrices in the Technical Appendix.

---

## Law VII — Micro-Typography, Math Craft, & Bibliographic Hygiene

1. **Display Equations That Fit Narrow Columns**: Every display equation must carry a clear conceptual role, end with proper grammatical punctuation, and fit cleanly inside a `3.49 in` column using `multline` or `aligned`—zero `Overfull \hbox` bleed into column gutters.
2. **Macro-Bound Quantitative Integrity**: Bind every empirical statistic in prose and tables to generated LaTeX macros from validated reference JSONs. Never hand-type or hand-edit numbers in `.tex` tables.
3. **Appearance-Ordered `[1]–[N]` Citations**: Number references strictly in order of first appearance (`[1], [2], [3], ...` starting in paragraph 1 of Section I). Include **only** works actually cited in the manuscript or appendix, with verified authors, venues, years, and DOIs.
4. **Canonical Back-Matter Sequence**:
   `Main Sections I–VI` $\to$ `Acknowledgments & Disclosure` $\to$ `Data & Code Availability` $\to$ `Sequential References [1]–[N]` $\to$ `Author Biographies` $\to$ `\clearpage` $\to$ `Self-Contained Technical Appendices A–D` $\to$ `\clearpage`.

---

## Law VIII — Deterministic Provenance & Release Gate Scorecard

Every published number, curve, composite map, macro, table, and citation block must regenerate bit-identically from checksum-verified data on CPU with fixed random seeds. Before any release or push, the manuscript must pass all **12 Charter Gates**:

- [ ] **G1 (Zero Filler)**: Zero meta-roadmaps, zero duplicated multi-point conclusion lists, zero leaked referee/draft notes.
- [ ] **G2 (Single Home)**: Zero datasets or arguments narrated redundantly in both Main Text and Appendix.
- [ ] **G3 (Page-1 Contract)**: Title, `≤225`-word Abstract, and Page-1 Synthesis Figure convey paradox, law, scale, findings, and null within 60 seconds.
- [ ] **G4 (Epistemic Discipline)**: Derived, Measured, Accounting, and Conjectured claims explicitly distinguished; contested and null findings reported symmetrically with chronological ordering verified.
- [ ] **G5 (Unified 1:1 Graphics)**: 100% of Main Figures rendered by a single programmatic script at exact column widths (`0` legacy draft mismatches).
- [ ] **G6 (Semantic Palette)**: Every color invariant in meaning across all figures; monochrome redundancy verified.
- [ ] **G7 (Float Proximity)**: Every main figure and table lands within `0–1` pages of its first callout; `0` floats strand after Appendix D.
- [ ] **G8 (Numerical Tables Only)**: Zero prose-wall essay tables in Main Text; all `booktabs` tables carry explicit $N$s, units, and CIs.
- [ ] **G9 (Sequential Bibliography)**: Citations run `[1], [2], ..., [N]` in appearance order from Page 1; `0` unresolved `[?]` keys; `0` uncited bloat entries.
- [ ] **G10 (Clean Micro-Typography)**: `0` authored `Overfull \hbox` warnings; `0` missing font glyphs; all narrow-column equations wrapped cleanly.
- [ ] **G11 (Privacy & License Hygiene)**: Human composites obey strict minimum aggregation floors ($N \ge 50$); third-party datasets obey exact license terms.
- [ ] **G12 (Bit-Exact Reproducibility)**: Two consecutive runs produce `0` diff; schema/source-hash validator, unit tests, `--check` asset gate, and repository `make required` exit `0`.
