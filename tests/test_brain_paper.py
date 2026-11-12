"""Independent mathematical and paper-boundary checks for the human manuscript."""
import json
from pathlib import Path
import unittest

try:
    import numpy as np
    from scipy.interpolate import BSpline
    from scipy.linalg import null_space
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Independent spline checks require NumPy and SciPy") from exc

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "evidence/brain/reference/minds-we-choose.json"


class BrainPaperTests(unittest.TestCase):
    def test_observer_context_identity_against_simulated_independent_panels(self):
        rng = np.random.default_rng(90451)
        n, m, k = 120000, 5, 4
        target = rng.normal(size=n)
        halo = rng.normal(scale=np.sqrt(.5), size=n)
        def panel():
            return (target + halo + rng.normal(size=(n, m)).mean(axis=1)
                    + rng.normal(size=(n, k)).mean(axis=1)
                    + rng.normal(size=(n, m*k)).mean(axis=1))
        first, second = panel(), panel()
        variance = 1 + .5 + 1/m + 1/k + 1/(m*k)
        self.assertAlmostEqual(np.corrcoef(target, first)[0, 1]**2, 1/variance, delta=.012)
        self.assertAlmostEqual(np.corrcoef(first, second)[0, 1], 1.5/variance, delta=.012)
        self.assertGreater(np.corrcoef(first, second)[0, 1], np.corrcoef(target, first)[0, 1]**2)

    def test_parenthood_identity_with_weights_and_nonzero_trait_mean(self):
        z = np.array([3., 5., 4., -1., 7., 2.])
        count = np.array([0., 2., 3., 0., 1., 8.])
        weight = np.array([1., 8., 2., 4., 7., 3.])
        def mean(x, w): return np.dot(x, w)/sum(w)
        def cov(x, y, w): return mean((x-mean(x, w))*(y-mean(y, w)), w)
        parent = count > 0
        total = cov(z, count, weight)/mean(count, weight)
        access = mean(z[parent], weight[parent])-mean(z, weight)
        number = cov(z[parent], count[parent], weight[parent])/mean(count[parent], weight[parent])
        self.assertAlmostEqual(total, access+number, places=13)

    @unittest.skipUnless(REFERENCE.exists(), "Human reference not yet integrated")
    def test_principal_splines_with_independent_constrained_bspline_basis(self):
        d = json.loads(REFERENCE.read_text())
        ages = np.arange(18, 65)
        knots = [18.]*4 + [25., 35., 45., 55.] + [64.]*4
        # A constrained B-spline construction is independent of the production
        # truncated-power basis. Natural boundary constraints remove two df.
        basis = BSpline(knots, np.eye(8), 3)
        constraints = basis.derivative(2)([18., 64.])
        x = basis(ages) @ null_space(constraints)
        all_estimates = {"Female": [], "Male": []}
        for record in d["hon"]["age_analysis"]["country_contrasts"]:
            for sex in all_estimates:
                cells = d["hon"]["thresholds"]["20"][record["country"]][sex]["exact_age"]
                y = np.array([v["mean_percentile"] for v in cells], dtype=float)
                weights = np.array([v["score_n"] for v in cells], dtype=float)
                valid = np.isfinite(y) & (weights > 0)
                coefficients = np.linalg.lstsq(x[valid]*np.sqrt(weights[valid, None]), y[valid]*np.sqrt(weights[valid]), rcond=None)[0]
                fitted = x @ coefficients
                contrast = fitted[(ages >= 25) & (ages <= 29)].mean()-fitted[(ages >= 45) & (ages <= 49)].mean()
                self.assertAlmostEqual(contrast, record[sex]["contrasts"]["mean_percentile"]["young_minus_older"], places=8)
                all_estimates[sex].append(contrast)
        for sex, values in all_estimates.items():
            self.assertAlmostEqual(np.mean(values), d["hon"]["age_analysis"]["contrasts"][sex]["mean_percentile"]["mean"], places=8)

    @unittest.skipUnless(REFERENCE.exists(), "Human reference not yet integrated")
    def test_human_reference_has_no_safe_ai_result_families(self):
        d = json.loads(REFERENCE.read_text())
        self.assertEqual(d["study"], "minds-we-choose")
        for forbidden in ("monitors", "campaigns", "runtime", "observed-effects", "authority", "controllers"):
            self.assertNotIn(forbidden, d)
        self.assertEqual(d["baseline_reproduction"]["reference_sha256"].__len__(), 64)
        self.assertTrue((ROOT/"docs/intelligence-is-not-authority.tex").exists())
        self.assertTrue((ROOT/"docs/sections/brain/minds-we-choose.tex").exists())


if __name__ == "__main__":
    unittest.main()
