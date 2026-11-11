"""Build the rubric table of scorecard_v9.tex from one dictionary, so that totals cannot drift from the cells."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
D = ["V3", "CF", "FT", "PB", "V4", "V5", "V6", "V7", "F2", "S2", "V8", "VF", "V9"]
G = [
 ("A. New knowledge and evidence", [
  ("A1 Results from data", "Original analyses of open data, with intervals; more than one population", 6, [0.5, 1, 0.5, 1, 2, 3.5, 4, 4.5, 4.5, 1, 5, 5, 5.5]),
  ("A2 Formal results", "Stated exactly, proved, checked by independent code", 4, [2.5, 3, 0.5, 1, 3, 3.5, 3.5, 3.5, 2.5, 1, 3.5, 2.5, 3.5]),
  ("A3 Novelty", "Claims specialists would not already assert", 4, [1.5, 1.5, 1.5, 1.5, 2, 2.5, 2.5, 3, 3, 1.5, 3, 3, 3.5]),
  ("A4 Reproducibility", "One command rebuilds everything; sources verified", 4, [1.5, 2.5, 1, 2, 3.5, 3.5, 4, 4, 2.5, 2.5, 4, 2.5, 4])]),
 ("B. Correctness and discipline of inference", [
  ("B1 Claims at their level", "Nothing causal or historical beyond the evidence", 5, [3, 5, 2.5, 4, 3.5, 4, 4, 4, 3.5, 5, 4.5, 4, 4.5]),
  ("B2 Sources", "Every reference resolves; claims match sources", 3, [1.5, 2.5, 1.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2, 2.5, 2.5, 2, 2.5]),
  ("B3 Consistency", "Numbers agree across abstract, text, tables, figures", 3, [2, 2.5, 2, 2.5, 2, 2.5, 2.5, 2.5, 2, 2.5, 2.5, 2, 2.5]),
  ("B4 Falsifiability", "Predictions with weakening results; objections answered", 3, [2, 2.5, 1, 2, 3, 3, 3, 3, 2.5, 2.5, 3, 2.5, 3])]),
 ("C. Mate selection: criteria, the sexes, culture, and who decides", [
  ("C1 Said and chosen", "What each sex says it wants against what it chooses", 4, [0.5, 0.5, 1, 2.5, 1.5, 2, 2.5, 2.5, 3.5, 3, 3.5, 3.5, 4]),
  ("C2 Where selection acts", "Episode at which each sex is selected, measured", 4, [0.5, 1, 1, 1.5, 2, 3, 3, 3.5, 3, 2, 3.5, 3.5, 3.5]),
  ("C3 Culture", "Cross-cultural and cross-national evidence tied to mechanism", 4, [0.5, 0.5, 0.5, 1.5, 1, 1, 1.5, 1.5, 1, 2, 3, 3.5, 3.5]),
  ("C4 Who decides", "Consent, coercion, parents and kin treated with data", 4, [0.5, 0.5, 0.5, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3, 3.5])]),
 ("D. Answers to the hard questions raised in review", [
  ("D1 Who became an ancestor", "The `half of men' claim tested against theory and data", 2, [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0.5, 0.5, 1.5]),
  ("D2 Small-brained species", "Islands, isolation and timing stated correctly", 1, [0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1]),
  ("D3 Shrinking brains", "Recent decline, ancient DNA and efficiency reconciled", 2, [0, 0.5, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 0.5, 1.5]),
  ("D4 Variation", "Exploration and attention traits handled with evidence", 1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5]),
  ("D5 Stories, fire, ritual", "Storytelling, firelight and safe contests in the theory", 2, [0.5, 0, 1, 0.5, 1, 1, 1, 1, 1, 0.5, 1, 1, 1.5]),
  ("D6 Community and births", "Pairing, community and fertility tested, with nulls reported", 2, [0, 0, 0, 0, 0, 0.5, 0.5, 1, 0.5, 0.5, 1, 1, 1.5])]),
 ("E. Strength and importance of the conclusions", [
  ("E1 Stated plainly", "Conclusions a reader can repeat, each marked by status", 4, [1.5, 1, 2, 2, 2, 2.5, 2.5, 2.5, 3, 2, 2.5, 3, 3.5]),
  ("E2 Why it matters now", "Consequences for present-day life, supported by data", 4, [0.5, 0.5, 1, 1.5, 1, 2, 2, 2.5, 2, 1.5, 2.5, 3, 3.5])]),
 ("F. Clarity and read", [
  ("F1 One argument", "Can be retold in a few sentences after page one", 4, [1.5, 1.5, 3, 3, 2, 2.5, 3, 3, 3.5, 3, 3.5, 3.5, 3]),
  ("F2 Sentences", "Plain, concrete, short", 4, [2, 2, 3, 3.5, 2.5, 3, 3, 3, 3.5, 3.5, 3.5, 3.5, 3.5]),
  ("F3 Length and structure", "Journal-length main text; navigable supplement", 2, [1, 1, 1.5, 1.5, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1]),
  ("F4 Mixed audience", "Terms defined; light notation", 2, [0.5, 1, 1.5, 1.5, 1, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])]),
 ("G. Title, subtitle, abstract", [
  ("G1 Title and subtitle", "Memorable, true to the mechanism, no overclaim", 3, [1.5, 1, 1.5, 1.5, 1.5, 2, 2, 2, 2, 1.5, 2.5, 2.5, 2.5]),
  ("G2 Abstract", "Findings with numbers, in order, with the limit", 3, [1.5, 1.5, 1.5, 2, 1.5, 2, 2, 2, 2.5, 2, 2.5, 2.5, 2.5])]),
 ("H. First-page graphic and figures", [
  ("H1 First-page graphic", "Tells the argument in panels of data or exact results", 3, [1, 0.5, 0.5, 2, 2, 2, 2, 2, 2.5, 1.5, 2.5, 2.5, 2.5]),
  ("H2 Figure craft", "Legible, no collisions, uncertainty shown", 3, [1.5, 1.5, 1, 2, 2, 2, 2.5, 2.5, 2, 2, 2.5, 2.5, 2.5]),
  ("H3 Provenance", "All figures from code and data; nothing decorative as evidence", 2, [1, 1, 1, 1, 2, 2, 2, 2, 1.5, 1.5, 2, 1.5, 2])]),
 ("I. Readiness for a leading journal, and value to builders of AI", [
  ("I1 Format, limits, disclosure", "Article, methods, data statement; limits stated", 3, [1.5, 2, 1, 2, 2, 2.5, 2.5, 2.5, 2, 2.5, 2.5, 2.5, 2.5]),
  ("I2 Distance from submission", "Specialist review, second population, full-text checks", 2, [0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1]),
  ("I3 Value to AI builders", "Transferable results stated honestly", 3, [0.5, 0.5, 0, 0, 0.5, 0.5, 0.5, 1, 0.5, 0, 2.5, 2, 2.5])]),
]
f = lambda x: ("%g" % x)
tot = [0.0] * len(D); mx = 0.0; ncrit = 0
rows = []
for gname, crits in G:
    gm = sum(c[2] for c in crits); gs = [sum(c[3][i] for c in crits) for i in range(len(D))]
    rows.append("\\rowcolor{band}\\multicolumn{2}{@{}l}{\\sffamily\\bfseries %s} & \\textbf{%s} & %s\\\\" % (gname, f(gm), " & ".join("\\textbf{%s}" % f(v) for v in gs)))
    for name, need, m, sc in crits:
        assert len(sc) == len(D) and all(0 <= v <= m for v in sc), name
        rows.append("%s & %s & %s & %s\\\\" % (name, need, f(m), " & ".join(f(v) for v in sc)))
        ncrit += 1; mx += m
        for i, v in enumerate(sc):
            tot[i] += v
assert mx == 100, mx
rows.append("\\midrule\n\\textbf{Total} & & \\textbf{100} & " + " & ".join("\\textbf{%s}" % ("%.1f" % v) for v in tot) + "\\\\")
open(os.path.join(HERE, "rubric_v9_body.tex"), "w").write("\n".join(rows) + "\n")
open(os.path.join(HERE, "rubric_v9_totals.tex"), "w").write("".join("\\newcommand{\\Tot%s}{%.1f}\n" % (d.replace("3", "three").replace("4", "four").replace("5", "five").replace("6", "six").replace("7", "seven").replace("8", "eight").replace("9", "nine").replace("2", "two"), v) for d, v in zip(D, tot)) + "\\newcommand{\\NCrit}{%d}\n" % ncrit)
print(ncrit, "criteria;", dict(zip(D, ["%.1f" % v for v in tot])))
