#!/usr/bin/env python3
"""Write a single self-contained .tex (sections, generated numbers and tables,
and the bibliography inlined).  It needs only the figures/ folder to compile."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "paper")

def expand(text):
    def inp(m):
        f = m.group(1)
        f = f if f.endswith(".tex") else f + ".tex"
        return expand(open(os.path.join(P, f)).read().rstrip("\n"))
    text = re.sub(r"\\tabinput\{([^}]+)\}", inp, text)
    text = re.sub(r"^\\input\{([^}]+)\}", inp, text, flags=re.M)
    return text

src = open(os.path.join(P, "main.tex")).read()
src = src.replace("\\makeatletter\\newcommand{\\tabinput}[1]{\\@@input #1 }\\makeatother\n", "")
out = expand(src)
bbl = open(os.path.join(P, "main.bbl")).read()
out = re.sub(r"\\bibliographystyle\{unsrtnat\}\s*\\bibliography\{refs\}", lambda m: bbl.strip(), out)
out = out.replace("% Build: pdflatex main; bibtex main; pdflatex main; pdflatex main",
                  "% Single-file version: run pdflatex twice. Requires only the figures/ folder.\n"
                  "% The modular source, bibliography database, code, results and data are in the accompanying package.")
dst = sys.argv[1]
open(dst, "w").write(out)
print("wrote", dst, len(out.splitlines()), "lines")
