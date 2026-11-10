#!/usr/bin/env python3
"""Compile with no shell escape; fail on unresolved references or missing glyphs."""
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "target/paper-build"
output.mkdir(parents=True, exist_ok=True)
command = ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
           f"-output-directory={output}", "intelligence-is-not-authority.tex"]
for iteration in range(3):
    result = subprocess.run(command, cwd=ROOT / "docs", capture_output=True, text=True)
    (output / f"pass-{iteration+1}.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        print(result.stdout[-8000:])
        raise SystemExit(result.returncode)
log = (output / "intelligence-is-not-authority.log").read_text(errors="replace")
if re.search(r"undefined references|Citation .+ undefined|Reference .+ undefined|Missing character:|multiply defined", log):
    raise SystemExit("Unresolved citations/references/glyphs: inspect target/paper-build/intelligence-is-not-authority.log")
shutil.copy2(output / "intelligence-is-not-authority.pdf", ROOT / "docs/intelligence-is-not-authority.pdf")
print("built docs/intelligence-is-not-authority.pdf; all citations and references resolved")
