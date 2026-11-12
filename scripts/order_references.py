#!/usr/bin/env python3
"""Order docs/references.tex by first citation, as recorded in a built paper's .aux, and drop uncited entries.

  python3 scripts/order_references.py target/paper-build/intelligence-is-not-authority.aux          # rewrite
  python3 scripts/order_references.py target/paper-build/intelligence-is-not-authority.aux --check  # verify only

The bibliography is a hand-ordered thebibliography; build_paper.py requires citations numbered [1..N] by first use.
Run a build, run this, then build again (the .aux of the second build will agree).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "docs/references.tex"


def cited_order(aux: str) -> list[str]:
    seen: list[str] = []
    for m in re.finditer(r"\\citation\{([^}]+)\}", aux):
        for k in (x.strip() for x in m.group(1).split(",")):
            if k and k not in seen:
                seen.append(k)
    return seen


def entries(tex: str) -> tuple[str, dict[str, str], str]:
    start = tex.index(r"\bibitem{")
    end = tex.index(r"\end{thebibliography}")
    body = tex[start:end]
    parts = re.split(r"(?=\\bibitem\{)", body)
    out = {}
    for part in parts:
        if not part.strip():
            continue
        key = re.match(r"\\bibitem\{([^}]+)\}", part).group(1)
        if key in out:
            raise SystemExit(f"duplicate bibitem {key}")
        out[key] = part.rstrip() + "\n"
    return tex[:start], out, tex[end:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aux", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    order = cited_order(args.aux.read_text(errors="replace"))
    tex = REFS.read_text()
    head, items, tail = entries(tex)
    missing = [k for k in order if k not in items]
    if missing:
        raise SystemExit(f"cited without a bibitem: {missing}")
    dropped = [k for k in items if k not in order]
    new = head + "".join(items[k] for k in order) + tail
    if args.check:
        if new != tex:
            raise SystemExit(f"references are not in first-citation order or include uncited entries {dropped}")
        print("references in first-citation order")
        return
    REFS.write_text(new)
    print(f"ordered {len(order)} references; dropped uncited: {dropped}")


if __name__ == "__main__":
    main()
