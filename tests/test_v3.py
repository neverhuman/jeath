"""Stdlib-only unit checks for V3 runner, validator, and paper macro hygiene."""
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import paper
import run_v3
import validate_v3

FORBIDDEN_NON_STDLIB = {"numpy", "scipy", "networkx", "sklearn", "pandas", "matplotlib", "torch"}


class V3PipelineChecks(unittest.TestCase):
    def test_v3_and_paper_scripts_are_stdlib_only(self):
        for name in ("run_v3.py", "validate_v3.py", "paper.py"):
            tree = ast.parse((ROOT / "scripts" / name).read_text(), filename=name)
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertTrue(imported.isdisjoint(FORBIDDEN_NON_STDLIB), (name, imported & FORBIDDEN_NON_STDLIB))

    def test_paper_macro_formatter_rejects_digits_and_underscores(self):
        ok = paper.format_macros("test", [("AlphaBeta", "1"), ("VThreeTestMacro", "2")])
        self.assertIn(r"\newcommand{\AlphaBeta}{1}", ok)
        for bad in ("V2Bad", "V_Three", "Macro1", "bad_name"):
            with self.assertRaises(AssertionError):
                paper.format_macros("test", [(bad, "1")])
        if (ROOT / "evidence/v3/reference/complete.json").is_file():
            generated = paper.generate()
            for macro_file in ("numbers.tex", "v2-numbers.tex", "v3-numbers.tex"):
                for cmd in re.findall(r"\\newcommand\{\\([^}]+)\}", generated[macro_file]):
                    self.assertRegex(cmd, r"^[A-Za-z]+$")

    def test_source_manifest_excludes_docs_readme_and_evidence(self):
        hashes = run_v3.collect_source_hashes()
        self.assertIn("src/lib.rs", hashes)
        self.assertIn("policy-core/src/lib.rs", hashes)
        self.assertIn("scripts/run_v3.py", hashes)
        self.assertIn("scripts/validate_v3.py", hashes)
        for rel in hashes:
            self.assertFalse(rel.startswith("docs/"), rel)
            self.assertFalse(rel.startswith("evidence/"), rel)
            self.assertNotEqual(rel, "README.md")

    def test_validate_v3_accepts_quick_run_and_rejects_tampering(self):
        out_dir = ROOT / "target/test-evidence" / f"py-v3-{os.getpid()}-{time.time_ns()}"
        debug_bin = ROOT / "target/debug/authority-lab"
        release_bin = ROOT / "target/release/authority-lab"
        chosen_bin = debug_bin if debug_bin.is_file() else release_bin
        self.assertTrue(chosen_bin.is_file(), f"missing authority-lab binary at {chosen_bin}")

        subprocess.run(
            [str(chosen_bin), "run-v3", str(out_dir), "--quick", "--seed", "20261110"],
            check=True,
        )
        manifest = {
            "schema_version": 3,
            "profile": "smoke",
            "quick": True,
            "seed": 20261110,
            "gpu_used": False,
            "gpu_seconds": 0,
            "binary_sha256": run_v3.digest(chosen_bin),
            "source_sha256": run_v3.collect_source_hashes(),
        }
        (out_dir / "provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (out_dir / "result.json").write_text(json.dumps({"suite": "test_v3", "passed": True}) + "\n")

        res = validate_v3.validate(out_dir, source=True)
        self.assertTrue(res["validated"])
        self.assertEqual(res["profile"], "smoke")

        tampered = ROOT / "target/test-evidence" / f"py-v3-tampered-{os.getpid()}-{time.time_ns()}"
        shutil.copytree(out_dir, tampered)
        try:
            comp = json.loads((tampered / "composition.json").read_text())
            comp["b100_overspend"] += 1
            (tampered / "composition.json").write_text(json.dumps(comp))
            with self.assertRaises(AssertionError):
                validate_v3.validate(tampered, source=True)
        finally:
            shutil.rmtree(tampered, ignore_errors=True)

        tampered_src = ROOT / "target/test-evidence" / f"py-v3-tampered-src-{os.getpid()}-{time.time_ns()}"
        shutil.copytree(out_dir, tampered_src)
        try:
            prov = json.loads((tampered_src / "provenance.json").read_text())
            prov["source_sha256"]["src/lib.rs"] = "0" * 64
            (tampered_src / "provenance.json").write_text(json.dumps(prov))
            with self.assertRaises(AssertionError):
                validate_v3.validate(tampered_src, source=True)
        finally:
            shutil.rmtree(tampered_src, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
