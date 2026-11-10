#!/usr/bin/env python3
"""Compile with no shell escape; fail on unresolved references, missing glyphs, font substitutions, or severe overfull boxes."""
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "target/paper-build"
output.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env["SOURCE_DATE_EPOCH"] = "1794340800"
env["FORCE_SOURCE_DATE"] = "1"
command = [
    "pdflatex",
    "-no-shell-escape",
    "-interaction=nonstopmode",
    "-halt-on-error",
    f"-output-directory={output}",
    "intelligence-is-not-authority.tex",
]
for iteration in range(3):
    result = subprocess.run(command, cwd=ROOT / "docs", capture_output=True, text=True, env=env)
    (output / f"pass-{iteration+1}.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        print(result.stdout[-8000:])
        raise SystemExit(result.returncode)
log = (output / "intelligence-is-not-authority.log").read_text(errors="replace")
if re.search(
    r"undefined references|Citation .+ undefined|Reference .+ undefined|Missing character:|multiply defined",
    log,
):
    raise SystemExit("Unresolved citations/references/glyphs: inspect target/paper-build/intelligence-is-not-authority.log")
if re.search(
    r"LaTeX Font Warning: Font shape .+ undefined|LaTeX Font Warning: Some font shapes were not available",
    log,
):
    raise SystemExit("Font substitution warning detected: inspect target/paper-build/intelligence-is-not-authority.log")
overfull = [float(m.group(1)) for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log)]
severe = [w for w in overfull if w > 5.0]
if severe:
    raise SystemExit(
        f"Severe Overfull \\hbox > 5.0pt (max {max(severe):.2f}pt across {len(severe)} box(es)): inspect target/paper-build/intelligence-is-not-authority.log"
    )
shutil.copy2(output / "intelligence-is-not-authority.pdf", ROOT / "docs/intelligence-is-not-authority.pdf")
print("built docs/intelligence-is-not-authority.pdf; all citations, references, fonts, and hboxes verified")
