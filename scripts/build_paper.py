#!/usr/bin/env python3
"""Compile with no shell escape; fail on unresolved references, missing glyphs, font substitutions, or severe overfull boxes."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from zipfile import ZipFile

from validate import originals

ROOT = Path(__file__).resolve().parents[1]


def build(source, directory, output, env=None):
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "pdflatex",
        "-no-shell-escape",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-recorder",
        f"-output-directory={output}",
        str(source),
    ]
    for iteration in range(3):
        result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, errors="replace")
        (output / f"pass-{iteration+1}.txt").write_text(result.stdout + result.stderr)
        if result.returncode:
            print(result.stdout[-8000:])
            raise SystemExit(result.returncode)
    log = output / f"{source.stem}.log"
    if re.search(
        r"undefined (?:citations|references)|(?:Citation|Reference) .+ undefined|Missing character:|multiply defined",
        log.read_text(errors="replace"),
    ):
        raise SystemExit(f"Unresolved citations/references/glyphs: inspect {log}")
    return output / f"{source.stem}.pdf"


def performance():
    study = "performance-before-outcomes"
    originals(study)
    archive_dir = ROOT / "project/brain/tips"
    package_name = "Performance_Before_Outcomes_v9"
    package = archive_dir / package_name / "paper"
    output = ROOT / "target" / study
    sections = ROOT / "docs/sections"
    env = dict(
        os.environ,
        TEXINPUTS=os.pathsep.join([str(sections), str(output / "generated"), ""]),
        SOURCE_DATE_EPOCH="1794355200",
    )
    pdf = build(sections / f"{study}.tex", package, output, env)
    # Verify the v9 assets and inputs actually loaded by this build against the supplied archive.
    inputs = (output / f"{study}.fls").read_text().splitlines()
    archive_inputs = set()
    for line in inputs:
        if line.startswith("INPUT "):
            path = (package / line[6:]).resolve()
            if path.is_relative_to(package):
                archive_inputs.add(path)
    with ZipFile(archive_dir / f"{package_name}_ALL.zip") as archive:
        for path in sorted(archive_inputs):
            entry = f"{package_name}/paper/{path.relative_to(package).as_posix()}"
            if archive.read(entry) != path.read_bytes():
                raise SystemExit(f"Extracted v9 input differs from the supplied archive: {path}")
    print(f"built {pdf.relative_to(ROOT)}; verified {len(archive_inputs)} v9 inputs and resolved citations and references")


def minds():
    originals("performance-before-outcomes")
    originals("minds-we-choose")
    output = ROOT / "target/minds-we-choose"
    sections = ROOT / "docs/sections/brain"
    env = dict(
        os.environ,
        TEXINPUTS=os.pathsep.join([str(sections), str(output / "generated"), ""]),
        SOURCE_DATE_EPOCH="1794398400",
    )
    counts = {}
    for name in ("minds-we-choose", "minds-we-choose-supplement", "minds-we-choose-combined"):
        directory = output / name
        pdf = build(sections / f"{name}.tex", ROOT, directory, env)
        shutil.copy2(pdf, output / pdf.name)
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        counts[name] = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE).group(1))
        if name == "minds-we-choose":
            aux = (directory / f"{name}.aux").read_text()
            last = re.search(r"\\newlabel\{minds-main-last\}\{\{[^}]*\}\{(\d+)\}", aux)
            if not last:
                raise SystemExit("Missing main-narrative page accounting label")
            counts["main_narrative_pages"] = int(last.group(1))
    (output / "build-report.json").write_text(
        json.dumps({"paper": "human-brain-and-behavior", "pages": counts}, indent=2) + "\n"
    )
    if not 18 <= counts["main_narrative_pages"] <= 22:
        raise SystemExit(
            f"Main narrative must have 18--22 pages; found {counts['main_narrative_pages']}. Inspect retained PDFs."
        )
    print(f"built human paper, supplement and combined copy in {output}; pages {counts}")


def safe_ai():
    output = ROOT / "target/paper-build"
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = "1794340800"
    env["FORCE_SOURCE_DATE"] = "1"
    pdf = build(Path("intelligence-is-not-authority.tex"), ROOT / "docs", output, env)
    log = (output / "intelligence-is-not-authority.log").read_text(errors="replace")
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
    shutil.copy2(pdf, ROOT / "docs/intelligence-is-not-authority.pdf")
    print("built docs/intelligence-is-not-authority.pdf; all citations, references, fonts, and hboxes verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["performance-before-outcomes", "minds-we-choose"])
    args = parser.parse_args()
    if args.study == "minds-we-choose":
        minds()
    elif args.study == "performance-before-outcomes":
        performance()
    else:
        safe_ai()
