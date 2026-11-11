import unittest
from unittest.mock import patch

try:
    import numpy as np
    import pyarrow as pa
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("The platform study fixtures require NumPy and PyArrow") from exc

from scripts import performance_before_outcomes as study


class PlatformStudyTests(unittest.TestCase):
    def test_pov_age_parenthood_and_missing_response(self):
        cases = [
            (17, "No", "Want children"),
            (18, "No", "Want children"),
            (18, "No", "Undecided/Open"),
            (18, "No", "Does not want children"),
            (18, "No", "Prefer Not To Say"),
            (18, "No", ""),
            (18, "No", None),
            (18, "All my kids are over 18", "Want children"),
            (18, "Yes", "Does not want children"),
            (18, None, "Want children"),
            (79, "No", "Want children"),
            (80, "No", "Want children"),
        ]
        # Repeat each case to expose the denominator without releasing a small cell.
        age, parenthood, intention = map(list, zip(*(cases * 100)))
        n = len(age)
        data = pa.table({
            "source_record_no": list(range(1, n + 1)), "id": [str(i) for i in range(n)],
            "age": [str(a) for a in age], "gender": ["Female"] * n, "seeking": ["Man"] * n,
            "haschildren": parenthood, "wantchildren": intention, "maritalstatus": ["Single"] * n,
        })
        with patch.object(study, "read", return_value=data):
            result = study.pov()
        self.assertEqual(result["exclusions_in_order"]["age"], 200)
        self.assertEqual(result["response_audit_all_rows"]["parenthood"],
                         {"no": 900, "yes": 100, "adult_children": 100, "unknown": 100})
        cell = result["bands"]["18-19"]["Female"]
        self.assertEqual(cell["parenthood"], {"n": 800, "with_children": 200, "adult_children": 100, "share": .25})
        self.assertEqual(cell["without_children"]["counts"],
                         {"want": 100, "do_not_want": 100, "undecided_open": 100, "nonresponse": 300})
        self.assertEqual(result["bands"]["75-79"]["Female"]["without_children"]["n"], 100)
        self.assertIsNone(result["bands"]["75-79"]["Male"]["without_children"])

    def test_invalid_vote_counts_and_age_boundaries(self):
        self.assertEqual(study.eligible_age(np.array([17, 18, 79, 80])).tolist(), [False, True, True, False])
        self.assertEqual(study.valid_votes(np.array([0, 20, 1, -1, 0, 0, 21]),
                                           np.array([20, 20, 0, 20, 0, -1, 20])).tolist(),
                         [True, True, False, False, False, False, False])

    def test_suppression_at_100_profiles(self):
        self.assertIsNone(study.intent_cell([25, 25, 25, 24]))
        self.assertEqual(study.intent_cell([25] * 4)["n"], 100)
        self.assertIsNone(study.hon_rating(np.ones(99), np.full(99, 20)))
        self.assertEqual(study.hon_rating(np.ones(100), np.full(100, 20))["n"], 100)
        self.assertIsNone(study.beauty_cell(np.ones(99), np.full(99, .5), np.full(99, .5)))

    def test_percentile_ties_share_their_midpoint(self):
        scores = np.array([4., 1., 1., 2.])
        self.assertEqual(study.midranks(scores).tolist(), [.875, .25, .25, .625])

    def test_gini_lorenz_and_geography_mapping(self):
        gini_eq, dec_eq = study.gini_and_deciles(np.ones(200))
        self.assertAlmostEqual(gini_eq, 0.0, places=10)
        self.assertAlmostEqual(sum(dec_eq), 1.0, places=10)
        self.assertTrue(all(abs(d - 0.10) < 1e-10 for d in dec_eq))

        skewed = np.array([1.0] * 180 + [100.0] * 20)
        gini_sk, dec_sk = study.gini_and_deciles(skewed)
        self.assertGreater(gini_sk, 0.70)
        self.assertAlmostEqual(sum(dec_sk), 1.0, places=10)
        self.assertGreater(dec_sk[-1], 0.90)

        self.assertEqual(study.map_pov_country("California"), "United States")
        self.assertEqual(study.map_pov_country("UK"), "United Kingdom")
        self.assertEqual(study.map_pov_country("Ontario"), "Canada")
        self.assertEqual(study.map_pov_country("Sao Paulo"), "Brazil")
        self.assertEqual(study.map_pov_country("New South Wales"), "Australia")
        self.assertEqual(study.map_pov_country(""), "other_or_blank")

    def test_pov_height_cliff_and_two_way_jury_law(self):
        from scripts import performance_open as open_syn

        # Synthetic male height fixture with explicit 6'0" heap and <5' / 999cm audit cases
        heights = (
            ["5' 11\" (180cm)"] * 150
            + ["6' 0\" (183cm)"] * 200
            + ["6' 1\" (185cm)"] * 110
            + ["< 5' (< 152cm)"] * 20
            + ["5' 9\" (999cm)"] * 5
        )
        n = len(heights)
        data = pa.table({
            "source_record_no": list(range(1, n + 1)),
            "id": [str(i) for i in range(n)],
            "age": ["35"] * n,
            "gender": ["Male"] * n,
            "seeking": ["Woman"] * n,
            "haschildren": ["No"] * n,
            "wantchildren": ["Want children"] * n,
            "maritalstatus": ["Single"] * n,
            "intent": ["is looking for a relationship"] * n,
            "height": heights,
            "state_id": ["Texas"] * n,
        })
        with patch.object(study, "read", return_value=data):
            res = study.pov()
        hp_m = res["height_percentiles"]["by_sex"]["Male"]
        self.assertEqual(res["height_percentiles"]["audit"]["Male"], {
            "explicit_plausible": 460,
            "censored_under_5ft": 20,
            "implausible_999cm": 5,
            "other_or_blank": 0,
        })
        self.assertEqual(hp_m["cliff_5ft11_6ft0_6ft1"]["n_6ft0"], 200)
        self.assertEqual(hp_m["cliff_5ft11_6ft0_6ft1"]["n_6ft1"], 110)
        self.assertAlmostEqual(hp_m["cliff_5ft11_6ft0_6ft1"]["drop_6ft0_to_6ft1_pct"], -45.0, places=6)

        tw = open_syn.analyze_two_way_jury(np.random.default_rng(42))
        self.assertLess(tw["single_context_asymptote_r2"], tw["surface_r2"]["8"]["8"])
        for row in tw["monte_carlo_verification"].values():
            self.assertLess(abs(row["closed_form"] - row["monte_carlo"]), 0.02)

    def test_delaunay_canonical_warp_identity_and_zone_partition(self):
        from scripts import performance_face_attribution as fa

        self.assertEqual(fa.classify_region("United States"), "United States")
        self.assertEqual(fa.classify_region("Brazil"), "South America")
        self.assertEqual(fa.classify_region("Mexico"), "Mexico, Central Am. & Caribbean")
        self.assertEqual(fa.classify_region("Japan"), "Asia-Pacific & Oceania")
        self.assertEqual(fa.classify_region("France"), "Europe")
        self.assertEqual(fa.classify_region("Nigeria"), "Middle East & Africa")
        self.assertEqual(fa.classify_age_bucket(22), "18-24")
        self.assertEqual(fa.classify_age_bucket(60), "55+")

        yy, xx = np.mgrid[24:200:16, 24:200:16]
        canon = np.column_stack([xx.ravel(), yy.ravel()])[:106].astype(np.float32)
        lut = fa.build_delaunay_lut(canon, 224)
        src = np.full((224, 224, 3), 180, dtype=np.uint8)
        warped, log_j = fa.warp_with_lut(src, canon, lut)
        self.assertEqual(warped.shape, (224, 224, 3))
        self.assertTrue(np.allclose(warped, 180))
        self.assertTrue(np.allclose(log_j, 0.0, atol=1e-5))

        zones = fa.build_zone_masks_224(canon, 64)
        self.assertEqual(len(zones), 12)
        tot = sum(zones.values())
        self.assertAlmostEqual(float(tot[32, 32]), 1.0, places=4)

    def test_paleoclimate_plateau_and_holoface_invariants(self):
        from scripts import performance_holoface_benchmark as hfb
        from scripts import performance_paleoclimate_plateau as ppp

        pp = ppp.analyze_paleoclimate_plateau()
        self.assertEqual(pp["lr04_source"]["sha256"], ppp.EXPECTED_LR04_SHA256)
        self.assertGreater(pp["plateau_summary"]["linear_delta_aicc"], 40.0)
        self.assertLess(
            pp["mpt_amplification"]["pre_mpt_swing_permil"],
            pp["mpt_amplification"]["post_mpt_swing_permil"],
        )
        self.assertGreater(pp["timing_gap_summary"]["onset_to_mis34_cooling_kyr"], 300.0)
        self.assertGreater(pp["timing_gap_summary"]["onset_to_fitcoal_window_kyr"], 500.0)

        # Verify exact L1 simplex LP and SPD primal/dual Woodbury equivalence on synthetic matrix
        rng = np.random.default_rng(7)
        X_wide = rng.normal(size=(24, 40))
        Y_two = rng.normal(size=(24, 2))
        w_tr = rng.uniform(0.5, 1.5, size=24)
        sol_dual = hfb._spd_ridge_multi_rhs(X_wide, Y_two, w_tr, (5.0,))[5.0]
        # Compare against direct primal solve
        w_n = w_tr / w_tr.mean()
        mx = (w_n[:, None] * X_wide).sum(0) / w_n.sum()
        my = (w_n[:, None] * Y_two).sum(0) / w_n.sum()
        xw = (X_wide - mx) * np.sqrt(w_n)[:, None]
        yw = (Y_two - my) * np.sqrt(w_n)[:, None]
        beta_primal = np.linalg.solve(xw.T @ xw + 5.0 * np.eye(40), xw.T @ yw)
        self.assertTrue(np.allclose(sol_dual[0], beta_primal, atol=1e-8))

        w_lp, _ = hfb.solve_exact_l1_simplex(rng.normal(size=(50, 4)), rng.normal(size=50), max_w=0.5)
        self.assertAlmostEqual(float(w_lp.sum()), 1.0, places=6)
        self.assertTrue(np.all((w_lp >= -1e-9) & (w_lp <= 0.5 + 1e-6)))


if __name__ == "__main__":
    unittest.main()
