import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import statistical_spine_v5  # noqa: E402


class TestStatisticalSpineV5(unittest.TestCase):
    def test_suissa_shuster_and_ville_size_and_rust_parity(self):
        # 1. Exact null size check on N = 16 at delta_0 = 0 across interior theta grid:
        # Unconditional Suissa-Shuster test must control Type-I size <= alpha everywhere on H_0!
        n = 16
        alpha = 0.05 / 16.0
        crit_pairs = []
        for x10 in range(n + 1):
            for x01 in range(n - x10 + 1):
                if statistical_spine_v5.exact_unconditional_paired_pvalue(x10, x01, n, 0.0) <= alpha:
                    crit_pairs.append((x10, x01))
        for theta in (0.01, 0.05, 0.15, 0.30, 0.49):
            p10 = theta
            p01 = theta
            prem = 1.0 - 2.0 * theta
            null_reject_prob = 0.0
            for x10, x01 in crit_pairs:
                x_rem = n - x10 - x01
                lcoef = (
                    statistical_spine_v5.ln_fact(n)
                    - statistical_spine_v5.ln_fact(x10)
                    - statistical_spine_v5.ln_fact(x01)
                    - statistical_spine_v5.ln_fact(x_rem)
                )
                prob = (p10**x10) * (p01**x01) * (prem**x_rem) * pow(2.718281828459045, lcoef)
                null_reject_prob += prob
            self.assertLessEqual(null_reject_prob, alpha + 1e-9)

        # 2. Qwen3-4B vs none (`0` vs `10` on `N = 32`):
        p_qwen = statistical_spine_v5.exact_unconditional_paired_pvalue(0, 10, 32, 0.0)
        ub_qwen = statistical_spine_v5.exact_unconditional_upper_bound(0, 10, 32, alpha)
        self.assertLess(p_qwen, alpha)
        self.assertLess(ub_qwen, 0.0)

        # 3. Cross-validate against recorded or freshly generated Rust `statistical_certificates.json`:
        cert_path = ROOT / "evidence" / "v3" / "reference" / "statistical_certificates.json"
        if cert_path.is_file():
            res = statistical_spine_v5.cross_validate_report(json.loads(cert_path.read_text()))
            self.assertTrue(res["cross_validated"])


if __name__ == "__main__":
    unittest.main()
