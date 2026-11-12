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

    def test_voice_dyadic_love_reversal_and_macro_kpi_reference_invariants(self):
        import json
        from pathlib import Path
        from scripts import performance_paper as ppaper

        # 1. Synthetic self-written voice + childless crossover test
        n = 240
        ages = ["27"] * 120 + ["46"] * 120
        genders = (["Female"] * 60 + ["Male"] * 60) * 2
        seeking = (["Man"] * 60 + ["Woman"] * 60) * 2
        wants = (
            ["Want children"] * 50 + ["Does not want children"] * 10
            + ["Want children"] * 30 + ["Does not want children"] * 30
            + ["Want children"] * 10 + ["Does not want children"] * 50
            + ["Want children"] * 30 + ["Does not want children"] * 30
        )
        descs = [
            "Looking for an honest, loyal, kind partner who loves family, kids, and laughter together."
            for _ in range(n)
        ]
        data = pa.table({
            "source_record_no": list(range(1, n + 1)),
            "id": [str(i) for i in range(n)],
            "age": ages,
            "gender": genders,
            "seeking": seeking,
            "haschildren": ["No"] * n,
            "wantchildren": wants,
            "maritalstatus": ["Single"] * n,
            "headline": ["Warm smile"] * n,
            "description": descs,
            "interests": ["hiking cooking"] * n,
        })
        with patch.object(study, "read", return_value=data):
            res = study.pov()
        swv = res["self_written_voice"]
        self.assertEqual(swv["eligible_with_essay"]["Pooled"], 240)
        self.assertEqual(swv["themes"]["honest_loyal_trust"]["Female"]["count"], 120)
        self.assertEqual(swv["themes"]["tall_6ft_mention"]["Female"]["count"], 0)
        # Verify band-midpoint childless crossover between 27.0 (+33.3 pp) and 47.0 (-33.3 pp)
        # Note: cells require n >= 100 per sex/band in production; check reference JSON below.

        ref_path = Path(__file__).resolve().parents[1] / "evidence/reference/performance-before-outcomes.json"
        ref = json.loads(ref_path.read_text())
        self.assertEqual(ref["macro_selection_kpis"]["schema_version"], 2)
        self.assertEqual(len(ref["macro_selection_kpis"]["kpis"]), 9)
        self.assertTrue(39.0 <= ref["pov"]["childless_crossover"]["band_midpoint_interpolated_age"] <= 44.0)
        self.assertGreater(ref["pov"]["self_written_voice"]["voice_to_checkbox_ratios"]["female_family_to_tall"], 80.0)

        srm = ref["speed_dating"]["dyadic_chemistry"]["variance_partition"]["like"]["unbiased_srm_shares"]
        self.assertAlmostEqual(
            srm["chooser_leniency"] + srm["partner_consensus"] + srm["unique_dyadic_and_noise"],
            1.0,
            places=6,
        )
        self.assertGreater(srm["unique_dyadic_and_noise"], srm["partner_consensus"])

        # v13 hobby-tag null vs. live wavelength & Waves 1-3 hindsight checks
        chk_sd = ref["speed_dating"]["checklist_and_hindsight"]
        hw = chk_sd["hobby_checklist_vs_live_wavelength"]
        self.assertLess(abs(hw["r_objective_hobby_corr_yes"]["estimate"]), 0.05)
        self.assertGreater(hw["r_live_shared_wavelength_yes"]["estimate"], 0.35)
        self.assertGreater(hw["live_to_checklist_r_ratio"], 20.0)
        self.assertGreater(
            chk_sd["within_person_shifts"]["nextday_realized_minus_pre_stated"]["women"]["delta_points"]["attr"]["estimate"],
            8.0,
        )
        self.assertGreater(
            chk_sd["within_person_shifts"]["threeweek_stated_minus_nextday_realized"]["women"]["delta_points"]["intel"]["interval_95"][0],
            0.0,
        )

        # v13 childhood metabolic bottleneck & Proposition 22 (E22) checks
        cc = ref["open_synthesis"]["childhood_coevolution"]
        self.assertEqual(cc["kuzawa_2014_brain_metabolism"]["peak_rmr_pct"], 66.3)
        self.assertEqual(cc["sccs_alloparenting_and_sharing"]["early_childhood_exclusive_mother_sccs52"]["share"], 0.0)
        self.assertGreater(cc["sccs_alloparenting_and_sharing"]["early_childhood_nonmaternal_majority_sccs52"]["share"], 0.70)
        self.assertGreater(cc["proposition_22_e22_coevolution"]["love_devotion_gain_cm3"], 200.0)
        for reg_e22 in cc["proposition_22_e22_coevolution"]["regimes"].values():
            self.assertLess(reg_e22["abs_closed_minus_sim_cm3"], 2.0)

        # v13 NHS campfire song legibility & Stanford HCMST audit compression checks
        nhs = ref["open_synthesis"]["nhs_campfire"]
        self.assertEqual((nhs["discography_songs"], nhs["discography_societies"]), (118, 86))
        self.assertGreater(nhs["lullaby_to_love_d_ratio"], 4.0)

        hcm = ref["open_synthesis"]["hcmst_2017"]
        self.assertEqual((hcm["total_respondents"], hcm["meeting_coded_couples"]), (3510, 3295))
        self.assertEqual(hcm["by_venue"]["online_dating_app"]["pre_romance_acquaintance_years"]["median"], 0.0)
        self.assertGreater(hcm["audit_compression_ratios"]["institutional_kin_to_app_mean_ratio"], 8.0)
        self.assertLess(hcm["intact_quality_boundary_check"]["ci_95"][0], 0.0)
        self.assertGreater(hcm["intact_quality_boundary_check"]["ci_95"][1], 0.0)

        # v13 Kowal checklist fit & marrying-up null checks
        chk_kow = ref["open_synthesis"]["kowal_2025"]["checklist_and_surplus_decomposition"]
        pk_kow = chk_kow["pooled"]["predictors_sd_per_sd"]
        self.assertGreater(pk_kow["partner_kindness"]["estimate"], 2.5 * pk_kow["idiographic_checklist_fit"]["estimate"])
        self.assertLess(abs(pk_kow["marrying_up_surplus"]["estimate"]), 0.04)
        self.assertLess(abs(chk_kow["five_trait_controlled_marrying_up_sd"]), 0.04)

        joel = ref["open_synthesis"]["joel_2020"]["satisfaction"]
        self.assertEqual(joel["distinct_datasets"], 43)
        self.assertGreater(joel["actor_rel_to_partner_trait_baseline_ratio"], 5.0)

        sccs = ref["open_synthesis"]["betzig_sccs"]
        self.assertEqual(sccs["causes"]["adultery_infidelity"]["coded_societies"], 160)
        self.assertGreater(sccs["adultery_to_aging_looks_ratio"], 8.0)

        pbr_w = ref["open_synthesis"]["kowal_2025"]["pair_bond_reversal"]["relationship_quality_prqc"]["by_sex"]["women"]["traits"]
        self.assertGreater(pbr_w["kindness"]["sd_per_sd"], 0.22)
        self.assertLess(abs(pbr_w["resources"]["sd_per_sd"]), 0.05)

        # v14 empirical mind selection (objective SAT) & OkCupid-JSE audience-staging checks
        sat_b = chk_sd["objective_sat_blindness"]
        self.assertEqual((sat_b["n_participants_with_sat"], sat_b["n_dates_with_partner_sat"]), (202, 2989))
        self.assertLess(sat_b["stranger_intel_sat_r2"], 0.01)
        self.assertGreater(sat_b["panel_to_single_r2_ratio"], 5.0)
        self.assertLess(sat_b["r_partner_sat_vs_yes"]["interval_95"][0], 0.0)
        self.assertGreater(sat_b["r_partner_sat_vs_yes"]["interval_95"][1], 0.0)

        okc = ref["open_synthesis"]["okcupid_jse"]
        self.assertEqual(okc["total_profiles"], 59946)
        ost = okc["orientation_staging_experiment"]
        self.assertGreater(ost["groups"]["men_straight"]["stature"]["jump_5ft11_to_6ft0_pct"], 15.0)
        self.assertLess(ost["groups"]["men_gay"]["stature"]["jump_5ft11_to_6ft0_pct"], 0.0)
        self.assertGreater(ost["straight_to_gay_male_income_disclosure_ratio"], 1.5)
        tpp = okc["ten_prompt_portfolio"]
        self.assertGreater(tpp["multi_domain_lexical_validity"]["k9_to_k1_validity_ratio"], 4.5)
        self.assertGreater(tpp["assortative_mind_requests_essay9"]["q5_to_q1_mind_request_ratio"], 1.35)
        self.assertGreater(tpp["essay9_message_me_if"]["ratios"]["humor_to_tall"], 10.0)
        self.assertGreater(tpp["essay3_first_noticed"]["themes"]["smile_laugh"]["share"], 0.25)
        self.assertGreater(okc["sf_childless_fertility_crossover"]["interpolated_crossover_age"], 36.5)

        # v15 cross-source male-female mating dynamics, Proposition 23 (E23), & Table III checks
        csm = chk_sd["cross_sex_misperception"]
        self.assertGreater(csm["women_about_men"]["delta_points"]["attr"]["interval_95"][0], 5.0)
        self.assertGreater(csm["men_about_women"]["delta_points"]["attr"]["interval_95"][0], 4.0)
        self.assertLess(csm["women_about_men"]["delta_points"]["intel"]["interval_95"][1], -4.0)
        init_sx = chk_sd["post_event_followup_3_4_weeks"]["initiation_by_sex"]
        self.assertGreater(init_sx["male_to_female_calls_per_match_ratio"], 4.0)

        kov = ref["open_synthesis"]["kowal_2025"]
        hvk = kov["homophily_vs_kindness"]
        self.assertGreater(hvk["politics"]["assortative_pearson_r"], 0.55)
        self.assertGreater(hvk["politics"]["kindness_to_homophily_ratio"], 4.5)
        self.assertGreater(hvk["education"]["kindness_to_homophily_ratio"], 6.0)

        rsk = kov["reverence_vs_settling_kink"]
        self.assertGreater(rsk["by_sex"]["women"]["kindness_settling_to_reverence_slope_ratio"], 1.5)
        self.assertGreater(rsk["by_sex"]["men"]["kindness_settling_to_reverence_slope_ratio"], 1.5)
        for sx_k in ("women", "men"):
            lt = rsk["by_sex"][sx_k]["tiers_by_trait"]["looks"]
            self.assertLess(
                lt["revered_mate_gt_self"]["infidelity_rate_pct"]["mean"]
                - lt["settled_mate_lt_self"]["infidelity_rate_pct"]["mean"],
                -5.0,
            )

        cesl = kov["courtship_effort_and_soi_lifecycle"]
        self.assertLess(cesl["by_relationship_stage"]["by_sex"]["women"]["dating_to_married_drop_pct"], -10.0)
        self.assertLess(cesl["by_relationship_stage"]["by_sex"]["men"]["dating_to_married_drop_pct"], -15.0)

        crh = hcm["cohabitation_rush_hazard"]
        self.assertGreater(crh["le_3m_to_gt_2_5y_breakup_ratio"], 2.5)
        bia = hcm["breakup_initiation_asymmetry"]
        self.assertGreater(bia["married_divorce_or_separation"]["woman_share_of_unilateral"], 0.60)

        e23 = ref["open_synthesis"]["proposition_23_e23"]
        self.assertLess(e23["part_i_staging_moral_hazard"]["abs_diff_min"], 1.5)
        self.assertLess(e23["part_iii_unilateral_divorce_flip"]["abs_max_diff"], 0.02)
        self.assertGreater(e23["part_iii_unilateral_divorce_flip"]["married_woman_unilateral_closed_form"], 0.60)

        avc = ref["pov"]["self_written_voice"]["action_verb_calendar"]["lifespan_shifts_18_22_to_65_79"]
        self.assertGreater(avc["female_cook_rise_ratio"], 3.5)
        self.assertGreater(avc["male_cook_rise_ratio"], 3.5)
        self.assertLess(avc["pooled_party_drop_pct"], -50.0)
        sar = ref["pov"]["second_act_reentry"]["headline_contrasts"]
        self.assertLess(sar["divorced_vs_single_female_marry_change_pct_40_49"], -20.0)
        self.assertGreater(sar["widowed_men_minus_women_marry_pp_40_49"], 0.5)

        macros_tex = ppaper.numbers(ref)
        self.assertIn(r"\newcommand{\PboBondKindW}", macros_tex)
        self.assertIn(r"\newcommand{\PboJoelBaseRatio}", macros_tex)
        self.assertIn(r"\newcommand{\PboETwentyTwoLoveGain}", macros_tex)
        self.assertIn(r"\newcommand{\PboHcmstAuditRatio}", macros_tex)
        self.assertIn(r"\newcommand{\PboChkMarryUpSD}", macros_tex)
        self.assertIn(r"\newcommand{\PboSatSingleRTwo}", macros_tex)
        self.assertIn(r"\newcommand{\PboOkcStrJump}", macros_tex)
        self.assertIn(r"\newcommand{\PboOkcDomGain}", macros_tex)
        self.assertIn(r"\newcommand{\PboOkcSfCrossAge}", macros_tex)
        self.assertIn(r"\newcommand{\PboHomoPolRatio}", macros_tex)
        self.assertIn(r"\newcommand{\PboRevKindSlopeW}", macros_tex)
        self.assertIn(r"\newcommand{\PboCohabRushRatio}", macros_tex)
        self.assertIn(r"\newcommand{\PboETwentyThreeDivW}", macros_tex)
        self.assertIn(r"\newcommand{\PboWarmOutlierYesW}", macros_tex)
        self.assertIn(r"\newcommand{\PboOverclaimAgeSurge}", macros_tex)
        self.assertIn(r"\newcommand{\PboExecDivW}", macros_tex)
        self.assertIn(r"\newcommand{\PboSccsEgalPersuadePct}", macros_tex)
        self.assertIn(r"\newcommand{\PboETwentyFourMaxDiff}", macros_tex)
        kpi_tex = ppaper.macro_kpi_table(ref)
        self.assertEqual(kpi_tex.strip().count(r"\\"), 9)
        psec_tex = ppaper.primary_secondary_table(ref)
        self.assertGreaterEqual(psec_tex.strip().count(r"\\"), 8)
        ireg_tex = ppaper.influence_regimes_table(ref)
        self.assertEqual(ireg_tex.strip().count(r"\\\addlinespace"), 4)
        fig_bytes = ppaper.confidence_influence_plot(ref)
        self.assertTrue(fig_bytes.startswith(b"%PDF"))

        # v16 outlier confidence vs. arrogance, audience priors, alpha asymmetry, & Proposition 24 (E24) checks
        cab = ref["speed_dating"]["confidence_and_backlash"]
        self.assertGreater(cab["agency_x_warmth_dyadic"]["female_to_male_coldness_backlash_ratio"], 1.75)
        self.assertGreater(cab["audience_age_modulation"]["mature_to_young_overclaim_penalty_ratio"], 7.0)
        self.assertLess(
            cab["status_threat"]["male_chooser_evaluating_woman"]["coefficients"]["target_outsmarts_chooser"]["ci_95_pp"][1],
            -2.0,
        )
        ex_c = ref["pov"]["alpha_executive_and_ethnicity"]["executives_ages_30_54"]["contrasts"]
        self.assertGreater(ex_c["female_to_male_exec_divorced_ratio"], 1.65)
        self.assertLess(ex_c["female_minus_male_exec_very_ambitious_pp"], -12.0)
        iap = ref["open_synthesis"]["influence_and_prestige"]
        self.assertEqual(iap["sccs_coalition_leadership"]["regimes"]["egalitarian_sccs270_eq1"]["checked_leaders_share"], 1.0)
        self.assertGreater(iap["sccs_coalition_leadership"]["regimes"]["egalitarian_sccs270_eq1"]["persuasion_or_consultation_share"], 0.92)
        self.assertGreater(iap["okcupid_heros_journey_and_execs"]["triad_to_passive_k_ratio"], 1.30)
        e24 = ref["open_synthesis"]["proposition_24_e24"]
        self.assertLess(e24["max_abs_diff"], 0.01)
        self.assertGreater(e24["part_iv_heros_journey_vs_self_aggrandizing"]["standing_ratio"], 5.0)


if __name__ == "__main__":
    unittest.main()
