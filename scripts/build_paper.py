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


PRIVATE_HOST_RE = re.compile(r"\b(?:xbabe\w*|neverhuman-lan)\b")
FORBIDDEN_GLOBAL_RE = re.compile(
    r"\.(?:json|jsonl|sqlite|rs|py|sh|tex|gguf|toml|lock)\b"
    r"|\b(?:xbabe\w*|neverhuman-lan|authority-lab|confine-v2|broker-v2|sink-v2|sink-v5(?:-shard)?"
    r"|policy-core|Exec[A-Z]\w*|DiscoveryLedger|SplitMix64|RegQDA|KNN15|MLP2L|Final\(2\)"
    r"|PR#1|12b5579|68783a0)\b"
    r"|make\s+required"
)
SNAKE_CASE_RE = re.compile(r"\b[A-Za-z0-9]+_[A-Za-z0-9_]+\b")


def check_general_audience_cleanliness(pdf):
    if shutil.which("pdftotext") is None:
        return
    raw_text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True, errors="replace")
    bad_hosts = [m.group(0) for m in PRIVATE_HOST_RE.finditer(raw_text)]
    if bad_hosts:
        raise SystemExit(
            f"General-audience cleanliness check failed: private hostnames in PDF: {list(dict.fromkeys(bad_hosts))}"
        )
    unwrapped = re.sub(r"([/._~-])\n(\S)", r"\1\2", raw_text)
    for _ in range(4):
        unwrapped = re.sub(r"(https?://\S+)\n(\S*(?:/|\.html|\.pdf)\S*)", r"\1\2", unwrapped)
    without_urls = re.sub(r"https?://\S+", " ", unwrapped)
    bad_global = [m.group(0) for m in FORBIDDEN_GLOBAL_RE.finditer(without_urls)]
    if bad_global:
        uniq = list(dict.fromkeys(bad_global))[:12]
        raise SystemExit(
            f"General-audience cleanliness check failed: forbidden internal/file tokens in PDF: {uniq}"
        )
    bad_snake = [m.group(0) for m in SNAKE_CASE_RE.finditer(without_urls)]
    if bad_snake:
        uniq = list(dict.fromkeys(bad_snake))[:12]
        raise SystemExit(
            f"General-audience cleanliness check failed: snake_case code tokens in PDF: {uniq}"
        )


# Terms earlier versions used for simulations or framing the evidence does not support (v6 review). "Jepsen" may appear
# only to say the partition schedules are not Jepsen, so it is checked by context in the source, not banned here.
SAFE_AI_BANNED_RE = re.compile(r"superintelligen|scaling law|emergent deception|120,000", re.IGNORECASE)


def check_safe_ai_terms(pdf):
    if shutil.which("pdftotext") is None:
        return
    text = re.sub(r"-\n", "", subprocess.check_output(["pdftotext", str(pdf), "-"], text=True, errors="replace"))
    found = sorted({m.group(0).lower() for m in SAFE_AI_BANNED_RE.finditer(re.sub(r"\s+", " ", text))})
    if found:
        raise SystemExit(f"Banned framing terms in the SAFE AI PDF: {found}")


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
    severe = [w for w in overfull if w > 2.0]
    if severe:
        raise SystemExit(
            f"Severe Overfull \\hbox > 2.0pt (max {max(severe):.2f}pt across {len(severe)} box(es)): inspect target/paper-build/intelligence-is-not-authority.log"
        )
    aux_text = (output / "intelligence-is-not-authority.aux").read_text(errors="replace")
    bibcite = {m.group(1): int(m.group(2)) for m in re.finditer(r"\\bibcite\{([^}]+)\}\{(\d+)\}", aux_text)}
    seen_cites = []
    for m in re.finditer(r"\\citation\{([^}]+)\}", aux_text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in seen_cites:
                seen_cites.append(k)
    cite_nums = [bibcite[k] for k in seen_cites if k in bibcite]
    if cite_nums != list(range(1, len(cite_nums) + 1)):
        bad = [(k, bibcite[k], idx + 1) for idx, k in enumerate(seen_cites) if k in bibcite and bibcite[k] != idx + 1]
        raise SystemExit(
            f"Citations out of sequential [1]-[N] order (key, assigned_num, expected_num): {bad[:8]}"
        )
    check_general_audience_cleanliness(pdf)
    check_safe_ai_terms(pdf)
    shutil.copy2(pdf, ROOT / "docs/intelligence-is-not-authority.pdf")
    print("built docs/intelligence-is-not-authority.pdf; all citations [1..N], references, fonts, hboxes, and general-audience cleanliness verified")


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
