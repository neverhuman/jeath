import math
import unittest

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("The confidence fixtures require NumPy and pandas") from exc

from scripts import performance_confidence as pc


def synthetic_design(n_psu=40, per_psu=50, replicates=200, seed=7):
    """Two PSUs per stratum with a Rao--Wu style multiplier, as in minds_gss.design_multiplier."""
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_psu), per_psu)
    multiplier = np.zeros((replicates, n_psu))
    for h in range(n_psu // 2):
        pair = np.array([2 * h, 2 * h + 1])
        pick = rng.integers(0, 2, size=replicates)
        multiplier[np.arange(replicates), pair[pick]] += 2.0  # m/(m-1) with m=2, one draw
    weight = rng.uniform(0.5, 2.0, size=len(group))
    return pc.Replicated(weight, group, np.ones(len(group), bool), multiplier), rng


class ReplicatedEstimatorTests(unittest.TestCase):
    def test_unit_multiplier_reproduces_point_estimates(self):
        rep, rng = synthetic_design()
        rep.M = np.ones_like(rep.M)
        y = rng.normal(size=len(rep.w))
        mask = np.ones(len(y), bool)
        est, reps, n, supported = rep.mean(y, mask)
        self.assertTrue(supported)
        self.assertEqual(n, len(y))
        np.testing.assert_allclose(reps, est, atol=1e-12)
        X = np.column_stack([np.ones(len(y)), rng.normal(size=len(y))])
        beta, reps_b, _ = rep.ols(X, y, mask)
        np.testing.assert_allclose(reps_b, np.broadcast_to(beta, reps_b.shape), atol=1e-10)

    def test_unsupported_domain_returns_no_replicates(self):
        rep, rng = synthetic_design()
        rep.g = np.full(len(rep.w), -1)
        est, reps, n, supported = rep.mean(rng.normal(size=len(rep.w)), np.ones(len(rep.w), bool))
        self.assertFalse(supported)
        self.assertIsNone(reps)
        self.assertTrue(math.isfinite(est))


class FirebaughTests(unittest.TestCase):
    def test_exact_linear_world_has_zero_residual_and_true_components(self):
        rep, rng = synthetic_design(n_psu=60, per_psu=60)
        n = len(rep.w)
        year = rng.choice(np.array([1975.0, 1977.0, 1978.0, 1990.0, 2000.0, 2018.0, 2022.0, 2024.0]), size=n)
        cohort = year - rng.integers(18, 80, size=n)
        b_year, b_cohort = -0.03, -0.02  # per decade
        y = 0.4 + b_year * (year - 2000.0) / 10.0 + b_cohort * (cohort - 1950.0) / 10.0
        out = pc.firebaugh(rep, y, year, cohort, np.ones(n, bool), (1975, 1978), (2018, 2024))
        comp = out["components_pp"]
        self.assertAlmostEqual(comp["nonlinearity_residual"]["estimate"], 0.0, places=9)
        self.assertAlmostEqual(out["slope_per_decade_pp"]["year"], 100 * b_year, places=8)
        self.assertAlmostEqual(out["slope_per_decade_pp"]["cohort"], 100 * b_cohort, places=8)
        early = (year >= 1975) & (year <= 1978)
        late = (year >= 2018) & (year <= 2024)
        w = rep.w
        d_cohort = np.average(cohort[late], weights=w[late]) - np.average(cohort[early], weights=w[early])
        self.assertAlmostEqual(comp["cohort_replacement"]["estimate"], 100 * b_cohort * d_cohort / 10.0, places=8)
        self.assertAlmostEqual(
            comp["fitted_change"]["estimate"],
            comp["within_cohort_change"]["estimate"] + comp["cohort_replacement"]["estimate"], places=10)
        self.assertLess(out["identity_abs_error"], 1e-12)


def hcmst_frame(n=4000, effect=-0.10, seed=11):
    rng = np.random.default_rng(seed)
    sex = rng.integers(1, 3, size=n)
    q23 = rng.choice([1, 2, 3, 4], p=[0.45, 0.10, 0.40, 0.05], size=n)
    she_more = ((sex == 2) & (q23 == 1)) | ((sex == 1) & (q23 == 3))
    p_excellent = 0.6 + effect * she_more
    quality = np.where(rng.random(n) < p_excellent, 5, rng.integers(1, 5, size=n))
    return pd.DataFrame({
        "partnership_status": rng.choice([1, 2], size=n),
        "w6_same_sex_couple": np.zeros(n, int),
        "ppgender": sex,
        "w6_q4": 3 - sex,
        "ppage": rng.integers(25, 65, size=n),
        "weight1": rng.uniform(0.5, 1.5, size=n),
        "w6_relationship_quality": quality,
        "w6_q23": q23,
        "subject_yrsed": rng.integers(10, 20, size=n).astype(float),
        "partner_yrsed": rng.integers(10, 20, size=n).astype(float),
    })


class HcmstTests(unittest.TestCase):
    def test_recovers_planted_relative_income_effect(self):
        out = pc.hcmst_relative_income_from_frame(hcmst_frame())
        est = out["confirmatory"]["estimate_pp"]
        lo, hi = out["confirmatory"]["ci_95_pp"]
        self.assertLess(lo, est)
        self.assertLess(est, hi)
        self.assertLess(hi, 0.0)
        self.assertLess(abs(est - (-10.0)), 5.0)  # planted: she-earns-more lowers P(excellent) by 10 pp

    def test_null_world_is_not_significant_by_construction(self):
        out = pc.hcmst_relative_income_from_frame(hcmst_frame(effect=0.0, seed=3))
        lo, hi = out["confirmatory"]["ci_95_pp"]
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_same_sex_and_out_of_range_rows_are_excluded(self):
        df = hcmst_frame(n=1000)
        df.loc[:99, "w6_same_sex_couple"] = 1
        df.loc[100:149, "ppage"] = 70
        df.loc[150:159, "w6_q23"] = -1
        out = pc.hcmst_relative_income_from_frame(df)
        self.assertEqual(out["sample"]["n"], 840)


class PropositionTests(unittest.TestCase):
    def test_e25_threshold_zeroes_inward_selection(self):
        a, b, vo, vn, ve = 1.0, 2.0, 1.0, 0.25, 4.0
        k_star = pc.e25_threshold(a, b, vo, vn, ve)
        self.assertAlmostEqual(k_star, 4.0)
        cov_in, _, _ = pc.e25_closed_form(a, b, vo, vn, ve, k_star)
        self.assertAlmostEqual(cov_in, 0.0, places=12)
        self.assertGreater(pc.e25_closed_form(a, b, vo, vn, ve, 1)[0], 0.0)
        self.assertLess(pc.e25_closed_form(a, b, vo, vn, ve, 16)[0], 0.0)
        self.assertEqual(pc.e25_threshold(1.0, 8.0, vo, vn, ve), 0.0)  # strangers already penalize

    def test_e25_monte_carlo_matches_closed_form(self):
        out = pc.analyze_proposition_25_e25()
        self.assertLess(out["max_abs_error"], 0.02)
        signs = [r["selection_differential_inward_dominance"]["closed_form"] for r in out["by_k"]]
        self.assertGreater(signs[0], 0.0)
        self.assertLess(signs[-1], 0.0)

    def test_e26_shared_bias_survives_raters_and_outcome_test_flags_it(self):
        out = pc.analyze_proposition_26_e26()
        self.assertLess(out["max_abs_error"], 0.02)
        shares = [r["bias_share_of_rating_error_closed_form"] for r in out["by_raters"]]
        self.assertEqual(shares, sorted(shares))
        for r in out["by_raters"]:
            self.assertAlmostEqual(r["rating_gap_at_equal_quality"]["closed_form"], -0.30)
            self.assertGreater(r["outcome_test_quality_gap_funded"]["closed_form"], 0.0)
            self.assertLess(r["closed_form"]["penalized_group"]["funded_rate"],
                            r["closed_form"]["reference_group"]["funded_rate"])
        sym = out["symmetric_protocol_m16"]
        self.assertAlmostEqual(sym["penalized_group"]["funded_rate"], sym["reference_group"]["funded_rate"])
        ratios = [v["funding_rate_ratio_penalized_to_reference"] for v in out["communal_framing_m16"].values()]
        self.assertEqual(ratios, sorted(ratios))

    def test_simulations_are_deterministic(self):
        self.assertEqual(pc.analyze_proposition_25_e25(), pc.analyze_proposition_25_e25())
        self.assertEqual(pc.analyze_proposition_26_e26(), pc.analyze_proposition_26_e26())


GOOD_ROW = {
    "id": "x", "arena": "groups", "construct": "c", "stage": "entry", "metric": "r", "estimate": 0.2,
    "se": 0.05, "ci_95": [0.1, 0.3], "direction": "+", "k": 1, "n": 100, "source_key": "s", "doi": "10.1037/0022-3514.74.5.1197",
    "locator": "Table 2", "tier": "Tier 2", "status": "verified-fulltext", "note": "",
}


class LedgerAndScorecardTests(unittest.TestCase):
    def test_ledger_schema_accepts_good_row_and_rejects_bad_ones(self):
        pc.check_ledger([GOOD_ROW])
        pc.check_ledger([{**GOOD_ROW, "id": "y", "estimate": None, "se": None, "ci_95": None}])
        for change in ({"doi": "doi:10.1/x"}, {"locator": " "}, {"ci_95": [0.25, 0.3]},
                       {"stage": "vibes"}, {"status": "from memory"}, {"direction": "up"},
                       {"estimate": None}, {"n": 0}, {"extra": 1}):
            with self.assertRaises(ValueError):
                pc.check_ledger([{**GOOD_ROW, **change}])
        with self.assertRaises(ValueError):
            pc.check_ledger([GOOD_ROW, GOOD_ROW])

    def test_module_ledger_and_scorecard_validate(self):
        pc.check_ledger(pc.LEDGER)
        pc.check_scorecard(pc.SCORECARD)
        totals = pc.score(pc.SCORECARD)
        for theory in pc.SCORECARD["theories"]:
            t = totals[theory]
            self.assertEqual(t["predicted"] + t["contradicted"] + t["silent"], len(pc.SCORECARD["rows"]))
        # The scorecard must keep its honest losses: PBO is contradicted at least once, and some
        # regularities are won by other theories while PBO is silent.
        self.assertGreaterEqual(totals["PBO"]["contradicted"], 1)
        self.assertTrue(any(r["codes"]["PBO"] == "S" and "P" in r["codes"].values() for r in pc.SCORECARD["rows"]))
        # Every abstract-only row without a stated number carries a sign only.
        for row in pc.LEDGER:
            if row["estimate"] is None:
                self.assertIsNone(row["ci_95"])

    def test_scorecard_rejects_unknown_source(self):
        card = {**pc.SCORECARD, "rows": ({**pc.SCORECARD["rows"][0], "sources": ("nobody2099",)},)}
        with self.assertRaises(ValueError):
            pc.check_scorecard(card)

    def test_superseded_finding_is_kept_with_its_replication(self):
        ids = {row["id"]: row for row in pc.LEDGER}
        self.assertIn("SUPERSEDED", ids["livingston2012_original"]["note"])
        self.assertEqual(ids["liu2026_replication"]["n"], 1996)

    def test_scorecard_quotes_pbo_reference_looks_weighting(self):
        import json
        from pathlib import Path
        ref = json.loads((Path(__file__).resolve().parents[1] / "evidence/reference/performance-before-outcomes.json").read_text())
        attr = ref["speed_dating"]["adjusted_slopes"]["own"]["men_minus_women_pp"]["attr"]
        lo, hi = attr["interval_95"]
        text = {r["id"]: r["regularity"] for r in pc.SCORECARD["rows"]}["men_weight_looks_more"]
        self.assertIn(f"{attr['estimate']:.1f} points more than women's (95% CI {lo:.1f} to {hi:.1f})", text)

    def test_holm(self):
        adj = pc.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)


if __name__ == "__main__":
    unittest.main()


def halo_world(seed=11, sessions=21, per_side=10, halo=1.2):
    """Speed-dating sessions where yes depends on true agency only and the chooser's own warmth rating
    carries a halo from that same decision; other raters see true warmth plus independent noise."""
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("wave", "sex", "chooser", "partner", "yes")}
    own = []
    pid = 0
    for w in range(sessions):
        men = np.arange(pid, pid + per_side)
        women = np.arange(pid + per_side, pid + 2 * per_side)
        pid += 2 * per_side
        traits = {p: rng.normal(0.0, 1.0, 5) for p in [*men, *women]}  # attr, fun, intel, amb, sinc
        for choosers, targets, code in ((men, women, 1.0), (women, men, 0.0)):
            for c in choosers:
                bias = rng.normal(0.0, 0.5)
                for t in targets:
                    tr = traits[t]
                    agency = tr[[1, 2, 3]].mean()
                    latent = 0.9 * agency + 0.8 * tr[0] + bias + rng.logistic(0.0, 0.6)
                    y = float(latent > 0.3)
                    rating = tr + rng.normal(0.0, 0.7, 5)
                    rating[4] += halo * (y - 0.5)  # halo: the decision leaks into the warmth rating
                    own.append(np.clip(np.round(5.5 + 1.5 * rating), 1, 10))
                    for k, v in (("wave", w), ("sex", code), ("chooser", c), ("partner", t), ("yes", y)):
                        cols[k].append(v)
    return [np.array(cols[k], dtype=float) for k in ("wave", "sex", "chooser", "partner", "yes")] + [np.array(own)]


class WarmConfidenceTests(unittest.TestCase):
    def test_leave_one_out_panel_excludes_the_rater(self):
        wave = np.zeros(4)
        partner = np.zeros(4)
        own = np.array([[1.0], [2.0], [3.0], [10.0]])
        panel = pc.leave_one_out_panel(wave, partner, own, min_peers=3)
        np.testing.assert_allclose(panel[:, 0], [5.0, 14 / 3, 13 / 3, 2.0])
        self.assertTrue(np.isnan(pc.leave_one_out_panel(wave[:3], partner[:3], own[:3], min_peers=3)).all())

    def test_halo_appears_in_own_ratings_but_not_in_other_raters(self):
        wave, sex, chooser, partner, yes, own = halo_world()
        out = pc.warm_confidence_from_arrays(wave, sex, chooser, partner, yes, own, draws_n=400)
        for side in ("male_chooser_female_target", "female_chooser_male_target"):
            own_w = out["own"][side]["coefficients_pp_per_sd"]["warmth"]
            pan_w = out["panel"][side]["coefficients_pp_per_sd"]["warmth"]
            pan_a = out["panel"][side]["coefficients_pp_per_sd"]["agency"]
            self.assertGreater(own_w["interval_95"][0], 0.0, side)  # the halo is "found" in own ratings
            self.assertLess(pan_w["interval_95"][0], 0.0, side)      # and vanishes for independent raters
            self.assertGreater(pan_w["interval_95"][1], 0.0, side)
            self.assertGreater(pan_a["interval_95"][0], 0.0, side)   # true agency survives

    def test_estimator_is_deterministic(self):
        args = halo_world(seed=3, sessions=21, per_side=6)
        a = pc.warm_confidence_from_arrays(*args, draws_n=50)
        b = pc.warm_confidence_from_arrays(*args, draws_n=50)
        self.assertEqual(a, b)


class ConfidencePaperMacroTests(unittest.TestCase):
    def test_macros_are_alphabetic_and_do_not_shadow_v16_macros(self):
        import json
        import re
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        ref = root / "evidence/reference/performance-before-outcomes.json"
        data = json.loads(ref.read_text())
        if "confidence" not in data:
            self.skipTest("reference predates the confidence layer")
        from scripts import performance_confidence_paper as cpp
        macros = cpp.confidence_macros(data)
        self.assertGreater(len(macros), 200)
        for name in macros:
            self.assertRegex(name, r"^Pbo[A-Za-z]+$")
        paper_src = (root / "scripts/performance_paper.py").read_text()
        v16_names = set(re.findall(r'"(Pbo[A-Za-z]+)":', paper_src))
        self.assertFalse(v16_names & set(macros), "confidence macros shadow existing names")
