"""The launcher's stderr must not become a proposer capability."""
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_v2 import Lab, effects


@unittest.skipUnless(sys.platform == "linux", "Landlock and seccomp require Linux")
class LauncherBoundary(unittest.TestCase):
    def test_parent_stderr_is_not_inherited(self):
        work = ROOT / "target/test-evidence" / f"launcher-stderr-{time.time_ns()}"
        binary = ROOT / "target/debug/authority-lab"
        with patch("run_v2.BINARY", binary):
            lab = Lab(work)
            proposal = lab.proposal()
            lab.approve(proposal)
            lab.broker.close()
            lab.sink.close()

        launcher = (
            f"import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); "
            f"import run_v2; run_v2.BINARY = run_v2.Path({str(binary)!r}); "
            "import launch_v2; launch_v2.main()"
        )
        request = json.dumps({"op": "submit", "proposal": proposal})
        proposer = (
            "import os, sys; os.write(2, b'proposer-canary'); "
            f"print({request!r}, flush=True); sys.stdin.readline()"
        )
        transcript = work / "transcript.json"
        canary = work / "parent-stderr"
        with canary.open("wb") as stderr:
            subprocess.run(
                [sys.executable, "-c", launcher, "--transcript", str(transcript),
                 str(lab.broker.database), str(lab.sink.database), "alice",
                 "/usr/bin/python3", "-I", "-c", proposer],
                stderr=stderr, check=True, timeout=20,
            )
        self.assertEqual(json.loads(transcript.read_text())[0]["response"]["status"], "committed")
        self.assertEqual(len(effects(lab.sink.database)), 1)
        self.assertEqual(canary.read_bytes(), b"")
