"""Validate aggregate platform and speed-dating evidence."""
import math
import re


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-10), (a, b)


def accounted(cells, total, minimum):
    """Every withheld cell accounts for at most minimum - 1 profiles."""
    visible = sum(cell["n"] for cell in cells if cell is not None)
    suppressed = cells.count(None)
    assert 0 <= total - visible <= (minimum - 1) * suppressed


def check(data):
    minimum = data["cohorts"]["min_profiles_per_cell"]
    bands = data["cohorts"]["pov_bands"]
    assert minimum == 100
    assert [age for lo, hi in bands for age in range(lo, hi + 1)] == list(range(18, 80))
    assert [hi - lo + 1 for lo, hi in bands] == [2] + [5] * 12
    assert data["study"] == "performance-before-outcomes" and data["schema_version"] == 1
    assert data["source_host"].split(".")[0] == "xbabe3" and data["source_architecture"] == "x86_64"
    assert data["input_sha256"].keys() == data["input_sizes_bytes"].keys()
    assert all(size > 0 for size in data["input_sizes_bytes"].values())
    for digest in [*data["input_sha256"].values(), *data["analysis_sha256"].values()]:
        assert re.fullmatch(r"[a-f0-9]{64}", digest)
    archive = data["inventory"]
    hon = data["hon"]
    pov = data["pov"]
    assert pov["accounts"] == {"rows": 1208771, "distinct_nonnull_keys": 1208771, "blank_or_null_keys": 0}
    assert archive["sources"]["POV"]["records"] == pov["accounts"]["rows"]
    h = hon["inventory"]
    assert h["canonical_profiles"] == h["distinct_profile_keys"] == h["aligned_source_rows"] == 6727811
    assert archive["sources"]["HON"]["records"] == h["canonical_profiles"]
    assert archive["saved_index"]["tables"]["records"]["rows"] == 7936582
    assert archive["saved_index"]["passed"] and all(archive["saved_index"]["checks"].values())
    files = {row["kind"]: row["count_all"] for row in archive["saved_index"]["files_by_kind"]}
    assert sum(files.values()) == archive["saved_index"]["tables"]["files"]["rows"] == 5483581
    assert files["image"] == h["inventoried_images"] == 3685764
    assert sum(row["count_all"] for row in archive["saved_index"]["sidecars"]) == files["landmarks"] + files["face_size"] == 1797806
    assert h["profiles_with_inventoried_images"] == h["local_directories"] == 898903
    pilot = archive["cached_pilot"]
    assert pilot["selected_images"] == h["decoded_images"] == 19130
    assert sum(pilot["splits"]["counts"].values()) == pilot["profiles"] == 10000
    partial = archive["partial_image_audit"]
    assert sum(partial["image_status"].values()) == partial["counts"]["images"]
    assert sum(partial["jobs_by_state"].values()) == h["local_directories"]
    close(partial["coverage"], partial["counts"]["images"] / h["inventoried_images"])

    def response(cell):
        if cell is not None:
            n, counts, shares = cell["n"], cell["counts"], cell["shares"]
            assert n >= minimum and n == sum(counts.values()) and counts.keys() == shares.keys()
            for key, value in counts.items():
                assert value >= 0
                close(shares[key], value / n)

    e = pov["eligible"]
    x = pov["exclusions_in_order"]
    assert pov["accounts"]["rows"] - x["age"] == e["age_18_79"]
    assert e["age_18_79"] - x["gender"] == e["age_and_gender"]
    assert e["age_and_gender"] - x["seeking"] == e["opposite_gender"]
    assert e["with_children"] + e["without_children"] + e["unknown_parenthood"] == e["opposite_gender"]
    assert sum(pov["response_audit_all_rows"]["intent"].values()) == pov["accounts"]["rows"]
    assert sum(pov["response_audit_all_rows"]["parenthood"].values()) == pov["accounts"]["rows"]
    groups = [group for by_sex in pov["bands"].values() for group in by_sex.values()]
    accounted([{"n": g["all_profiles"]} if g["all_profiles"] is not None else None for g in groups],
              e["opposite_gender"], minimum)
    accounted([{"n": g["unknown_parenthood"]} if g["unknown_parenthood"] is not None else None for g in groups],
              e["unknown_parenthood"], minimum)
    accounted([{"n": g["parenthood"]["adult_children"]} if g["parenthood"] else None for g in groups],
              e["adult_children"], minimum)
    for g in groups:
        p = g["parenthood"]
        response(g["without_children"])
        response(g["with_children"])
        if p is None:
            assert not g["without_children"] and not g["with_children"]
            assert g["all_profiles"] is None or g["all_profiles"] - g["unknown_parenthood"] < minimum
            continue
        assert p["n"] == g["all_profiles"] - g["unknown_parenthood"]
        assert 0 <= p["adult_children"] <= p["with_children"] <= p["n"] and p["n"] >= minimum
        close(p["share"], p["with_children"] / p["n"])
        for key, expected in (("without_children", p["n"] - p["with_children"]), ("with_children", p["with_children"])):
            cell = g[key]
            assert cell["n"] == expected if cell else expected < minimum
    for key in ("with_children", "without_children"):
        accounted([g[key] for g in groups], e[key], minimum)
    for band, cases in pov["sensitivity"].items():
        assert set(cases) == {"main", "exact_age", "single", "all_seeking", "answered", "single_answered_exact_age"}
        main = cases["main"]["responses"]
        assert main == {sex: group["without_children"] for sex, group in pov["bands"][band].items()}
        for name, case in cases.items():
            by_sex = case["responses"]
            for cell in by_sex.values():
                response(cell)
                if "answered" in name and cell:
                    assert cell["counts"]["nonresponse"] == 0
            shares = case["want_shares"]
            if "exact_age_cells" in case:
                exact = case["exact_age_cells"]
                for day in exact.values():
                    for cell in day.values():
                        response(cell)
                weights = case["weights"]
                if weights is not None:
                    close(sum(weights.values()), 1)
                    for age, cells in exact.items():
                        close(weights[age], sum(c["n"] for c in cells.values()) / sum(c["n"] for c in by_sex.values()))
                    for sex in by_sex:
                        assert sum(day[sex]["n"] for day in exact.values()) == by_sex[sex]["n"]
                        close(shares[sex], sum(weights[age] * day[sex]["shares"]["want"] for age, day in exact.items()))
                else:
                    assert shares is None
            elif shares:
                assert shares == {sex: cell["shares"]["want"] for sex, cell in by_sex.items()}
            if shares:
                close(case["women_minus_men_pp"], 100 * (shares["Female"] - shares["Male"]))
    for band, expected in {"25-29": (46.0, 33.5), "45-49": (6.8, 12.8)}.items():
        shares = pov["sensitivity"][band]["main"]["want_shares"]
        assert tuple(round(100 * shares[sex], 1) for sex in ("Female", "Male")) == expected

    assert sum(h["labels"].values()) == h["canonical_profiles"]
    assert sum(h["recorded_gender"].values()) == h["canonical_profiles"]
    stages = hon["eligibility_stages"]
    exclusions = hon["exclusions_in_order"]
    previous = h["canonical_profiles"]
    for stage in ("age_18_79", "age_and_gender", "unconflicted", "valid_votes"):
        assert previous - exclusions[stage] == stages[stage]
        previous = stages[stage]
    ratings = hon["ratings"]
    beauty = hon["beauty_by_age"]
    exact_keys = [str(age) for age in range(18, 80)]
    band_keys = [f"{lo}-{hi}" for lo, hi in bands]
    for name, profile_set in ratings.items():
        threshold = {"votes_1": 1, "votes_20": 20, "votes_100": 100, "us_votes_20": 20}[name]
        assert set(profile_set) == {"all_ages", *exact_keys, *band_keys}
        for sex in ("Female", "Male"):
            for coverage in ("all", "inventoried"):
                aggregate = profile_set["all_ages"][sex][coverage]
                accounted([profile_set[key][sex][coverage] for key in exact_keys], aggregate["n"], minimum)
                accounted([profile_set[key][sex][coverage] for key in band_keys], aggregate["n"], minimum)
                for band, (lo, hi) in zip(band_keys, bands):
                    grouped = profile_set[band][sex][coverage]
                    ages = [profile_set[str(a)][sex][coverage] for a in range(lo, hi + 1)]
                    if grouped:
                        accounted(ages, grouped["n"], minimum)
                        if all(ages):
                            for k in ("likes", "votes"):
                                assert grouped[k] == sum(cell[k] for cell in ages)
                            close(grouped["mean_profile_like_rate"],
                                  sum(c["n"] * c["mean_profile_like_rate"] for c in ages) / grouped["n"])
                for row in profile_set.values():
                    cell = row[sex][coverage]
                    if cell:
                        assert cell["n"] >= minimum and 0 <= cell["likes"] <= cell["votes"]
                        assert cell["votes"] >= threshold * cell["n"] and cell["median_votes"] >= threshold
                        assert 0 <= cell["mean_profile_like_rate"] <= 1
                        close(cell["vote_pooled_like_rate"], cell["likes"] / cell["votes"])
                        if coverage == "inventoried":
                            assert cell["n"] <= row[sex]["all"]["n"]
    assert sum(g["all"]["n"] for g in ratings["votes_1"]["all_ages"].values()) == stages["valid_votes"]
    assert sum(g["all"]["n"] for g in ratings["votes_20"]["all_ages"].values()) == previous - exclusions["at_least_20_votes"]
    for sex in ("Female", "Male"):
        reference = hon["beauty_reference"][sex]
        assert -1 <= reference["score_like_rate_pearson_r"] <= 1
        assert reference["reference_n"] + reference["missing_or_invalid_score_among_primary"] == ratings["votes_20"]["all_ages"][sex]["all"]["n"]
        assert beauty["all_ages"][sex]["all"]["n"] == reference["reference_n"]
        close(beauty["all_ages"][sex]["all"]["mean_percentile"], .5)
        for coverage in ("all", "inventoried"):
            accounted([beauty[key][sex][coverage] for key in exact_keys], beauty["all_ages"][sex][coverage]["n"], minimum)
            accounted([beauty[key][sex][coverage] for key in band_keys], beauty["all_ages"][sex][coverage]["n"], minimum)
            for key, by_sex in beauty.items():
                cell = by_sex[sex][coverage]
                if cell:
                    assert minimum <= cell["n"] <= ratings["votes_20"][key][sex][coverage]["n"]
                    assert 0 <= cell["top_tenth_n"] <= cell["n"] and 0 <= cell["mean_score"] <= 10
                    assert 0 <= cell["percentile_iqr"][0] <= cell["percentile_iqr"][1] <= 1
                    assert 0 <= cell["mean_percentile"] <= 1 and 0 <= cell["mean_pooled_percentile"] <= 1

    speed = data["speed_dating"]
    assert (speed["records"], speed["participants"], speed["sessions"]) == (8378, 551, 21)
    assert speed["bootstrap"]["unit"] == "session"
    assert speed["counts"]["decisions_with_looks_panel"] <= speed["counts"]["valid_binary_decisions"] <= speed["records"]
    curve_n = 0
    for gender, series in speed["yes_by_panel_quintile"].items():
        rows = series["quintiles"]
        assert len(rows) == 5
        for row in rows:
            assert row["n"] >= minimum and 0 <= row["yes"] <= row["n"]
            close(row["estimate"], row["yes"] / row["n"])
            assert 0 <= row["interval_95"][0] <= row["interval_95"][1] <= 1
            curve_n += row["n"]
        close(series["top_minus_bottom"]["estimate"], rows[4]["estimate"] - rows[0]["estimate"])
        assert 0 <= speed["stated_looks"][gender]["mean_looks_points_of_100"] <= 100
    assert curve_n == speed["counts"]["decisions_with_looks_panel"]
    curves = speed["yes_by_panel_quintile"]
    close(speed["quintile_contrast_men_minus_women"]["estimate"],
          curves["men"]["top_minus_bottom"]["estimate"] - curves["women"]["top_minus_bottom"]["estimate"])
    for fit in speed["adjusted_slopes"].values():
        for key, contrast in fit["men_minus_women_pp"].items():
            close(contrast["estimate"], fit["men"]["coefficients_pp"][key]["estimate"] -
                  fit["women"]["coefficients_pp"][key]["estimate"])
            assert contrast["interval_95"][0] <= contrast["interval_95"][1]

    # --- Extended Platform, Multimodal Audit, Deep Facial Attribution & Open-Synthesis Validation ---
    assert set(data["analysis_sha256"]) == {
        "scripts/performance_before_outcomes.py",
        "scripts/performance_choice.py",
        "scripts/performance_face_attribution.py",
        "scripts/performance_holoface_benchmark.py",
        "scripts/performance_open.py",
        "scripts/performance_paleoclimate_plateau.py",
    }

    # 1. Completed 3,685,764-image faceflow audit & 20-wave multimodal benchmark
    ff = archive["full_faceflow_audit"]
    assert ff["complete"] and ff["counts_reconcile"] and ff["all_indexed_images_covered"]
    assert ff["source_image_count"] == ff["processed_image_count"] == h["inventoried_images"] == 3685764
    assert sum(row["n"] for row in ff["host_counts"]) == 3685764
    bm = archive["multimodal_benchmark"]
    assert (bm["sample_profiles"], bm["sample_images_total"], bm["sample_primary_images"]) == (10000, 20000, 19130)
    assert len(bm["waves"]) >= 20
    bm_by_id = {w["wave_id"]: w for w in bm["waves"]}
    assert bm["waves"][0]["wave_id"] == "H1" and "H20" in bm_by_id
    assert 0.20 < bm["waves"][0]["profile_test"]["spearman_r"] < 0.30
    assert 0.65 < bm_by_id["H20"]["profile_test"]["spearman_r"] < 0.70

    # 2. POV extended life-course commitment, traits, height percentiles & 5-country geography
    def check_rate(cell):
        if cell is not None:
            assert cell["n"] >= minimum and 0 <= cell["count"] <= cell["n"]
            close(cell["share"], cell["count"] / cell["n"])

    for w_entry in pov["market_and_commitment"].values():
        for s_cell in w_entry["by_sex"].values():
            if s_cell is not None:
                assert s_cell["n"] >= minimum
                for cat_key in ("intent", "for", "marital_status", "union_durability"):
                    response(s_cell[cat_key])
                for r_key in ("serious_commitment", "ever_partnered", "over_3y_union", "over_10y_union"):
                    check_rate(s_cell[r_key])

    for group_dict in pov["traits_and_postponement"].values():
        for by_s in group_dict.values():
            for t_cell in by_s.values():
                if t_cell is not None:
                    assert t_cell["n"] >= minimum
                    for r_key in ("parenthood", "childless_want_children", "want_children_all",
                                  "serious_commitment", "marry_intent", "casual_intent",
                                  "ever_partnered", "over_3y_union", "over_10y_union", "athletic_body"):
                        check_rate(t_cell[r_key])

    hp = pov["height_percentiles"]
    assert sum(sum(d.values()) for d in hp["audit"].values()) == e["opposite_gender"]
    for sex_label, h_sex in hp["by_sex"].items():
        assert h_sex is not None and h_sex["n_valid"] == hp["audit"][sex_label]["explicit_plausible"] >= minimum
        cliff = h_sex["cliff_5ft11_6ft0_6ft1"]
        if sex_label == "Male":
            assert cliff["n_6ft0"] > 90000 and cliff["n_6ft1"] < 65000 and cliff["drop_6ft0_to_6ft1_pct"] < -35.0
        inch_cells = [{"n": row["n_all_ages"]} for row in h_sex["by_inch"]] + [None] * (25 - len(h_sex["by_inch"]))
        accounted(inch_cells, h_sex["n_valid"], minimum)
        for row in h_sex["by_inch"]:
            assert 60 <= row["inches"] <= 84 and row["n_all_ages"] >= minimum
            close(row["observed_share"], row["n_all_ages"] / h_sex["n_valid"])
            assert 0 <= row["mean_height_percentile"] <= 100
            if row["outcomes_ages_30_49"] is not None:
                assert row["outcomes_ages_30_49"]["n"] >= minimum
            check_rate(row["very_ambitious_ages_30_49"])

    geo = pov["geography"]
    assert set(geo["replications"]) == {"United States", "United Kingdom", "Canada", "Brazil", "Australia"}
    for cname, rep in geo["replications"].items():
        assert rep["25-29"]["childless_want_women_minus_men_pp"] > 0, cname
        assert rep["45-49"]["childless_want_women_minus_men_pp"] < 0, cname
        assert rep["25-29"]["serious_women_minus_men_pp"] > 0, cname
        assert rep["45-49"]["serious_women_minus_men_pp"] > 0, cname

    # 3. HON attention skew, portfolio depth & cross-national coverage
    att = hon["attention_and_portfolio"]
    for sex in ("Female", "Male"):
        s_att = att["by_sex"][sex]
        assert s_att["n"] == hon["beauty_reference"][sex]["reference_n"]
        assert 0 < s_att["votes_gini"] < 1 and 0 < s_att["likes_gini"] < 1
        assert len(s_att["lorenz_deciles_votes"]) == 10 and len(s_att["lorenz_deciles_likes"]) == 10
        close(sum(s_att["lorenz_deciles_votes"]), 1.0)
        close(sum(s_att["lorenz_deciles_likes"]), 1.0)
        p_cells = list(att["by_photo_count"][sex].values())
        accounted(p_cells, s_att["n"], minimum)
    assert att["by_sex"]["Female"]["median_votes"] > 15 * att["by_sex"]["Male"]["median_votes"]

    cn = hon["cross_national"]
    assert cn["nonblank_country_profiles"] + cn["blank_country_profiles"] == sum(
        x["reference_n"] for x in hon["beauty_reference"].values()
    )
    assert cn["counts_by_threshold"]["ge_100_per_sex"] == len(cn["countries"]) >= 170
    assert cn["counts_by_threshold"]["ge_500_per_sex"] >= 70
    assert cn["summary_strict_age_contrast"]["countries"] >= 80
    assert cn["summary_strict_age_contrast"]["female_steeper_drop_countries"] > 0.8 * cn["summary_strict_age_contrast"]["countries"]

    # 4. Open-Data & Theory Synthesis Validation
    osyn = data["open_synthesis"]
    whyte = osyn["whyte_2021"]
    assert (whyte["records"], whyte["men"], whyte["women"]) == (7060, 4375, 2685)
    assert re.fullmatch(r"[a-f0-9]{64}", whyte["sha256"])
    for sex_label, expected_n in (("Men", 4375), ("Women", 2685)):
        assert sum(b["n"] for b in whyte["stated_importance"]["by_age_band"][sex_label].values()) == expected_n
        assert whyte["stated_importance"]["paired_contrasts"][sex_label]["trust_minus_intelligence"]["mean"] > 0
        assert whyte["stated_importance"]["paired_contrasts"][sex_label]["trust_minus_attractiveness"]["mean"] > 0
    m_ha = whyte["height_vs_attraction"]["Men"]
    assert m_ha["quadratic_fit_pct_per_sd"]["quadratic"]["ci_95"][1] < 0  # Statistically negative quadratic curvilinearity

    lid = osyn["lidborg_2022"]
    assert lid["height"]["total_effects"] == 97 and lid["body_strength"]["total_effects"] == 135
    assert lid["height"]["mating"]["re_ci_95"][0] > 0
    assert lid["height"]["reproduction_all"]["re_ci_95"][0] < 0 < lid["height"]["reproduction_all"]["re_ci_95"][1]
    assert lid["body_strength"]["reproduction_all"]["re_ci_95"][0] > 0

    kow = osyn["kowal_2025"]
    assert kow["inclusion_flow"]["source_rows"] == 117293
    assert kow["inclusion_flow"]["sum_equals_30"] == 67743
    assert kow["inclusion_flow"]["countries_ge_100_per_gender_count"] == 53
    assert kow["inclusion_flow"]["countries_ge_100_per_gender_respondents"] == 62185
    assert kow["age_standardized_32"]["countries"] == 32
    assert kow["cross_country_53"]["by_item"]["religion"]["cross_country_men_women_r"] > 0.93
    assert kow["cross_country_53"]["by_item"]["kindness"]["cross_country_men_women_r"] > 0.93
    assert kow["kowal_hon_bridge"]["overlapping_countries"] >= 50

    wal = osyn["walter_2020"]
    assert (wal["records"], wal["countries"]) == (14399, 45)

    tw = osyn["two_way_jury"]
    for check_row in tw["monte_carlo_verification"].values():
        assert abs(check_row["closed_form"] - check_row["monte_carlo"]) < 0.015

    # 5. Deep-Learning Facial Attribution, Super-Resolved Composites & Regional Invariance Validation
    fa = data["facial_attribution"]
    assert fa["schema_version"] == 1
    assert fa["portrait_resolution_px"] == 384 and fa["attribution_grid_px"] == 128
    assert fa["delaunay_landmarks"] == 106 and fa["delaunay_triangles"] >= 200
    assert fa["total_aligned_portraits_processed"] >= 4500
    assert re.fullmatch(r"[a-f0-9]{64}", fa["composites_npz_sha256"])
    assert fa["composites_npz_bytes"] > 100_000
    # Privacy floor: every stored composite averages at least 50 distinct aligned individuals
    assert all(cnt >= 50 for cnt in fa["composite_counts"].values())

    pf = fa["benchmark_10k_pure_face"]
    assert (pf["train_profiles"], pf["validation_profiles"], pf["test_overall"]["n"]) == (8000, 1000, 1000)
    assert 0.50 < pf["test_overall"]["spearman_r"] < 0.72
    assert 12.0 < pf["test_overall"]["gender_pct_mae"] < 24.0

    for sex in ("Female", "Male"):
        lv = fa["live_spatial_model_verification"][sex]
        assert lv["train_n"] >= 600 and lv["test_n"] >= 350 and lv["eval_natural_n"] >= 1400
        assert lv["test_spearman_r"] > (0.48 if sex == "Female" else 0.28)
        zd = fa["zone_drivers_and_rejectors"][sex]
        assert zd["top15_n"] >= 150 and zd["mid_n"] >= 150 and zd["bot15_n"] >= 150
        assert len(zd["zones"]) == 12
        for z in zd["zones"]:
            assert z["top15_ci95"][0] <= z["top15_driver_pct"] <= z["top15_ci95"][1]
            assert z["bot15_ci95"][0] <= z["bot15_rejector_pct"] <= z["bot15_ci95"][1]
            assert z["top15_driver_pct"] > z["bot15_rejector_pct"]
        ab_prog = fa["age_bucket_progression"][sex]
        assert set(ab_prog) == {"18-24", "25-34", "35-44", "45-54", "55+"}
        assert all(cell["n"] >= 150 for cell in ab_prog.values())
        assert ab_prog["18-24"]["mean_observed_pct"] > ab_prog["55+"]["mean_observed_pct"]
        wreg = fa["world_regional_synthesis"][sex]
        assert len(wreg["regions"]) >= 6
        assert all(rc["n"] >= 50 for rc in wreg["regions"].values())
        assert wreg["mean_pairwise_driver_corr"] > 0.35

    # 6. Synchronized LR04 Paleoclimate Stack & 1.5-0.7 Ma Plateau Audit Validation
    pp = data["paleoclimate_plateau"]
    assert pp["schema_version"] == 1
    assert pp["lr04_source"]["sha256"] == "ec3c4af8f80c4de1d36fb0ac9fabbde28520f77f0832958caacbae4957afb412"
    assert pp["lr04_source"]["points_0_to_3_4_ma"] >= 1700
    for fsha in pp["fossil_sources_sha256"].values():
        assert re.fullmatch(r"[a-f0-9]{64}", fsha)
    ps = pp["plateau_summary"]
    assert ps["gingerich_n"] == 233 and ps["linear_delta_aicc"] > 40.0
    assert 0.60 <= ps["fitted_plateau_ma"][0] <= 0.75 and 1.35 <= ps["fitted_plateau_ma"][1] <= 1.55
    assert 880.0 <= ps["plateau_level_cm3"] <= 940.0
    mpt = pp["mpt_amplification"]
    assert mpt["pre_mpt_swing_permil"] < mpt["mpt_crisis_swing_permil"] < mpt["post_mpt_swing_permil"]
    tgap = pp["timing_gap_summary"]
    assert 200.0 <= tgap["onset_to_mpt_kyr"] < tgap["onset_to_mis34_cooling_kyr"] < tgap["onset_to_fitcoal_window_kyr"] < tgap["onset_to_bolaven_impact_kyr"]
    assert len(pp["candidate_events"]) >= 9
    du_mc = pp["du2018_age_uncertainty_mc"]
    assert du_mc["specimens_total"] == 94 and du_mc["replicates"] == 2000
    assert du_mc["slopes"]["phase_III_1.5_0.7"]["median_slope_ln_per_myr"] < du_mc["slopes"]["phase_II_2.0_1.5"]["median_slope_ln_per_myr"]

    # 7. HoloFace-L1 12-Space Benchmark, 4-Family Shapley Decomposition & Psychometric Voter Ceiling Validation
    hb = data["holoface_benchmark"]
    assert hb["schema_version"] == 1
    assert re.fullmatch(r"[a-f0-9]{64}", hb["reference_json_sha256"])
    sc = hb["sample_counts"]
    assert (sc["total_profiles"], sc["train_profiles"], sc["validation_profiles"], sc["test_profiles"], sc["total_images"]) == (10000, 8000, 1000, 1000, 20000)
    hpure = hb["holoface_pure_zero_metadata"]
    assert 0.56 < hpure["test"]["spearman_r"] < 0.70 and hpure["test"]["beauty_mae"] < 0.39
    shap4 = hpure["shapley_4family_decomposition"]
    assert len(shap4) == 4
    assert abs(sum(r["shapley_test_spearman_r2"] for r in shap4) - hpure["test"]["spearman_r2"]) < 1e-3
    assert abs(sum(r["shapley_share_of_pure_face_pct"] for r in shap4) - 100.0) < 0.1
    hfull = hb["holoface_full_multimodal"]
    assert hfull["test_l1"]["beauty_mae"] <= 0.3362
    assert hfull["test_l1"]["spearman_r"] > 0.655
    assert hfull["test_l1"]["gender_pct_mae"] < 17.5
    assert abs(hfull["subgroups_test_l2_gender"]["Male"]["bias"]) < 0.04
    pvc = hb["psychometric_voter_ceiling"]["tiers"]
    assert (
        pvc["votes_20_49"]["oracle_finite_vote_spearman_r"]
        < pvc["votes_50_199"]["oracle_finite_vote_spearman_r"]
        < pvc["votes_200_plus"]["oracle_finite_vote_spearman_r"]
        < pvc["votes_500_plus"]["oracle_finite_vote_spearman_r"]
    )
    for tier_cell in pvc.values():
        assert tier_cell["holoface_pure_spearman_r"] <= tier_cell["holoface_full_spearman_r"] < tier_cell["oracle_finite_vote_spearman_r"]
    assert 0.30 < pvc["votes_500_plus"]["irreducible_single_context_gap_full"] < 0.60
    assert len(hb["waves_trajectory"]) >= 43

    # 8. Self-Written Voice, Dyadic Spark, Entry-to-Bond Love Reversal, Joel 2020, SCCS & 7 Macro KPIs
    cc = pov["childless_crossover"]
    assert 40.0 <= cc["band_midpoint_interpolated_age"] <= 43.0
    assert 40 <= cc["first_nonpositive_exact_age"] <= 43
    for ex_cell in cc["exact_ages_35_47"].values():
        check_rate(ex_cell["Female"])
        check_rate(ex_cell["Male"])

    swv = pov["self_written_voice"]
    assert swv["min_trimmed_chars"] == 30
    assert swv["eligible_with_essay"]["Pooled"] == swv["eligible_with_essay"]["Female"] + swv["eligible_with_essay"]["Male"] > 1_000_000
    for th in swv["themes"].values():
        for g_key in ("Pooled", "Female", "Male"):
            check_rate(th[g_key])
    for w_dict in swv["by_life_window"].values():
        for th_w in w_dict.values():
            check_rate(th_w["Female"])
            check_rate(th_w["Male"])
    assert swv["voice_to_checkbox_ratios"]["female_family_to_tall"] > 100.0
    assert swv["voice_to_checkbox_ratios"]["female_honesty_to_tall"] > 45.0
    assert swv["voice_to_checkbox_ratios"]["female_kindness_to_wealth"] > 25.0

    dc = speed["dyadic_chemistry"]
    for v_part in dc["variance_partition"].values():
        assert v_part["n"] >= 7000
        close(sum(v_part["unbiased_srm_shares"].values()), 1.0)
        close(sum(v_part["sample_mean_shares"].values()), 1.0)
    assert dc["variance_partition"]["like"]["unbiased_srm_shares"]["unique_dyadic_and_noise"] > 2.5 * dc["variance_partition"]["like"]["unbiased_srm_shares"]["partner_consensus"]
    rec_l = dc["mutual_dyadic_reciprocity_like"]
    assert rec_l["n_dyads"] >= 3900 and rec_l["interval_95"][0] > 0.15
    w_rec = dc["reciprocity_adjusted_slopes"]["women"]
    assert w_rec["dyadic_spark_sum_fun_shar_prob_pp"]["estimate"] > w_rec["coefficients_pp"]["attr"]["estimate"]

    hap = whyte["happiness_by_partnered"]
    for g_key in ("Men", "Women", "Pooled"):
        assert hap[g_key]["raw_lift_pts"]["ci_95"][0] > 4.0
        assert hap[g_key]["age_adjusted_lift_pts"]["ci_95"][0] > 4.0

    for b_age in kow["budget_by_age"].values():
        assert b_age["n_men"] >= minimum and b_age["n_women"] >= minimum
        for sex_k in ("men", "women"):
            ranked = sorted(b_age["items"], key=lambda it: b_age["items"][it][sex_k]["mean"], reverse=True)
            assert ranked[0] == "kindness"
            assert set(ranked[:3]) <= {"kindness", "health", "looks", "resources"}

    pag = kow["partner_age_gap_by_age"]["brackets"]
    assert pag["18-24"]["women_minus_men_years"] < pag["40-49"]["women_minus_men_years"] < pag["60-79"]["women_minus_men_years"]
    assert pag["60-79"]["women"]["mean"] > 0.5 and pag["60-79"]["men"]["mean"] < -3.5
    assert wal["age_gap_by_age"]["50-75"]["women"]["mean"] > 1.0 and wal["age_gap_by_age"]["50-75"]["men"]["mean"] < -3.5

    pbr = kow["pair_bond_reversal"]
    prqc_f = pbr["relationship_quality_prqc"]["by_sex"]
    for sx_k in ("men", "women"):
        # Inside the pair bond, partner kindness dominates while partner resources collapse near zero
        assert prqc_f[sx_k]["traits"]["kindness"]["sd_per_sd"] > 0.25
        assert abs(prqc_f[sx_k]["traits"]["resources"]["sd_per_sd"]) < 0.05
        assert pbr["reported_infidelity"]["by_sex"][sx_k]["traits"]["kindness"]["pp_per_sd"] < -1.0

    tla = kow["triangular_love_arc"]["by_duration"]
    assert tla["25y+"]["commitment"]["mean"] > tla["<=1y"]["commitment"]["mean"]
    assert tla["1-3y"]["passion"]["mean"] > tla["25y+"]["passion"]["mean"]
    assert kow["marry_without_love"]["by_sex"]["women"]["median"] < kow["marry_without_love"]["by_sex"]["men"]["median"]
    assert kow["assortative_mating"]["religion"]["pearson_r"] > 0.60

    sma = kow["social_media_and_alternatives"]
    assert sma["epistemic_tier"].startswith("observational_")
    assert sma["slopes_per_daily_hour"]["qoa_sd_per_hour"]["ci_95"][0] > 0.0
    assert sma["slopes_per_daily_hour"]["prqc_sd_per_hour"]["ci_95"][1] < 0.0
    assert sma["slopes_per_daily_hour"]["infidelity_pp_per_hour"]["ci_95"][0] > 1.0
    assert sma["by_qoa_tier"]["high_4.5_to_7.0"]["infidelity_rate_pct"]["mean"] > sma["by_qoa_tier"]["low_1_to_2.5"]["infidelity_rate_pct"]["mean"] + 10.0

    joel = osyn["joel_2020"]
    assert re.fullmatch(r"[a-f0-9]{64}", joel["satisfaction_sha256"])
    assert re.fullmatch(r"[a-f0-9]{64}", joel["commitment_sha256"])
    assert joel["satisfaction"]["distinct_datasets"] == 43
    assert joel["satisfaction"]["actor_rel_to_partner_trait_baseline_ratio"] > 5.0
    assert joel["satisfaction"]["actor_rel_to_partner_trait_followup_ratio"] > 2.0

    sccs = osyn["betzig_sccs"]
    assert re.fullmatch(r"[a-f0-9]{64}", sccs["data_sha256"])
    assert re.fullmatch(r"[a-f0-9]{64}", sccs["codes_sha256"])
    assert sccs["adultery_to_aging_looks_ratio"] > 8.0
    assert sccs["causes"]["old_age_loss_of_looks"]["share_mentioned"] < 0.08

    mk = data["macro_selection_kpis"]
    assert mk["schema_version"] == 2 and len(mk["kpis"]) == 8

    # 8. v13 Cross-Scale Causal Closure & Empirical Audit Validation
    chk_sd = data["speed_dating"]["checklist_and_hindsight"]
    hw = chk_sd["hobby_checklist_vs_live_wavelength"]
    assert hw["n_common_dyad_decisions"] == 7116
    assert abs(hw["r_objective_hobby_corr_yes"]["estimate"]) < 0.05
    assert hw["r_objective_hobby_corr_yes"]["interval_95"][0] < 0.0 < hw["r_objective_hobby_corr_yes"]["interval_95"][1]
    assert hw["r_live_shared_wavelength_yes"]["estimate"] > 0.35
    assert hw["live_to_checklist_r_ratio"] > 20.0
    assert hw["chooser_fe_slopes"]["women"]["coefficients_pp"]["int_corr"]["interval_95"][0] < 0.0 < hw["chooser_fe_slopes"]["women"]["coefficients_pp"]["int_corr"]["interval_95"][1]
    wps = chk_sd["within_person_shifts"]
    assert wps["nextday_realized_minus_pre_stated"]["women"]["delta_points"]["attr"]["estimate"] > 8.0
    assert wps["threeweek_stated_minus_nextday_realized"]["women"]["delta_points"]["intel"]["interval_95"][0] > 0.0
    assert 0.35 < chk_sd["post_event_followup_3_4_weeks"]["went_on_followup_date_share_among_matched"] < 0.50

    chk_kow = kow["checklist_and_surplus_decomposition"]
    assert chk_kow["n_total"] > 45000
    pk = chk_kow["pooled"]["predictors_sd_per_sd"]
    assert pk["partner_kindness"]["estimate"] > 0.25
    assert pk["idiographic_checklist_fit"]["estimate"] > 0.08
    assert pk["partner_kindness"]["estimate"] > 2.5 * pk["idiographic_checklist_fit"]["estimate"]
    assert abs(pk["marrying_up_surplus"]["estimate"]) < 0.04
    assert abs(chk_kow["five_trait_controlled_marrying_up_sd"]) < 0.04
    assert abs(chk_kow["signed_mate_minus_self_surplus_sd"]) < 0.04
    for reg_row in chk_kow["by_cultural_regime"].values():
        rp = reg_row["predictors_sd_per_sd"]
        assert rp["partner_kindness"]["estimate"] > 0.24
        assert rp["partner_kindness"]["estimate"] > 2.4 * rp["idiographic_checklist_fit"]["estimate"]
        assert rp["partner_kindness"]["estimate"] > 5.0 * rp["partner_resources"]["estimate"]
    assert kow["short_vs_long_term_ideals"]["by_sex"]["women"]["kindness"]["short_term_mean_1_to_11"] > kow["short_vs_long_term_ideals"]["by_sex"]["women"]["looks"]["short_term_mean_1_to_11"]
    assert kow["jealousy_asymmetry"]["by_sex"]["women"]["emotional_minus_sexual"]["ci_95"][0] > 0.20

    cc = osyn["childhood_coevolution"]
    assert cc["kuzawa_2014_brain_metabolism"]["peak_rmr_pct"] == 66.3
    assert cc["kuzawa_2014_brain_metabolism"]["peak_der_pct"] == 43.2
    sccs_cc = cc["sccs_alloparenting_and_sharing"]
    assert sccs_cc["early_childhood_exclusive_mother_sccs52"]["share"] == 0.0
    assert sccs_cc["early_childhood_nonmaternal_majority_sccs52"]["share"] > 0.70
    assert sccs_cc["father_close_early_childhood_sccs54"]["share"] > sccs_cc["father_close_infancy_sccs53"]["share"]
    assert sccs_cc["community_food_sharing_beyond_nuclear_sccs1718"]["share"] > 0.88
    e22 = cc["proposition_22_e22_coevolution"]
    for reg_e22 in e22["regimes"].values():
        assert reg_e22["abs_closed_minus_sim_cm3"] < 2.0
    assert e22["love_devotion_gain_cm3"] > 200.0

    nhs = osyn["nhs_campfire"]
    for sha_k in ("discography_sha256", "naive_listeners_sha256", "ethnography_sha256", "cultures_sha256"):
        assert re.fullmatch(r"[a-f0-9]{64}", nhs[sha_k])
    assert nhs["discography_songs"] == 118 and nhs["discography_societies"] == 86
    alg = nhs["acoustic_legibility_gradient"]
    assert alg["lullaby"]["cohen_d"] > 1.5 and alg["dance"]["cohen_d"] > 1.2 and alg["love"]["cohen_d"] < 0.6
    assert nhs["lullaby_to_love_d_ratio"] > 4.0

    hcm = osyn["hcmst_2017"]
    assert re.fullmatch(r"[a-f0-9]{64}", hcm["sha256"])
    assert hcm["total_respondents"] == 3510 and hcm["meeting_coded_couples"] == 3295
    assert hcm["by_venue"]["online_dating_app"]["pre_romance_acquaintance_years"]["median"] == 0.0
    assert hcm["audit_compression_ratios"]["institutional_kin_to_app_mean_ratio"] > 8.0
    assert hcm["intact_quality_boundary_check"]["ci_95"][0] < 0.0 < hcm["intact_quality_boundary_check"]["ci_95"][1]

    gss_all = osyn["v9_anchors"]["gss_three_episodes_all"]
    for g_k in ("income", "prestige", "schooling", "vocabulary"):
        assert gss_all[g_k]["men"]["pairing"][0] > 0.0 and gss_all[g_k]["men"]["pairing"][1] > 0.0
        assert gss_all[g_k]["men"]["number"][0] < 0.0 and gss_all[g_k]["men"]["number"][2] < 0.0

    # 9. v14 Empirical Mind Selection & OkCupid-JSE Audience-Staging Validation
    sat_b = chk_sd["objective_sat_blindness"]
    assert sat_b["n_participants_with_sat"] == 202 and sat_b["n_dates_with_partner_sat"] == 2989
    assert 0.0 < sat_b["r_stranger_intel_vs_partner_sat"]["estimate"] < 0.10
    assert sat_b["stranger_intel_sat_r2"] < 0.01
    assert sat_b["r_partner_sat_vs_yes"]["interval_95"][0] < 0.0 < sat_b["r_partner_sat_vs_yes"]["interval_95"][1]
    assert sat_b["n_targets_ge10_raters"] == 154
    assert sat_b["panel_to_single_r2_ratio"] > 5.0 and sat_b["panel_intel_sat_r2"] < 0.04

    okc = osyn["okcupid_jse"]
    for sha_k in ("profiles_sha256", "essays_sha256"):
        assert re.fullmatch(r"[a-f0-9]{64}", okc[sha_k])
    assert okc["total_profiles"] == 59946
    assert okc["sex_counts"] == {"men": 35829, "women": 24117}
    ost = okc["orientation_staging_experiment"]
    assert ost["groups"]["men_straight"]["stature"]["jump_5ft11_to_6ft0_pct"] > 15.0
    assert ost["groups"]["men_straight"]["stature"]["drop_6ft0_to_6ft1_pct"] < -40.0
    assert ost["groups"]["men_gay"]["stature"]["jump_5ft11_to_6ft0_pct"] < 0.0
    assert ost["straight_to_gay_male_income_disclosure_ratio"] > 1.5
    assert ost["straight_male_to_female_income_disclosure_ratio"] > 1.8

    tpp = okc["ten_prompt_portfolio"]
    assert tpp["total_rows"] == 59946 and tpp["total_nonempty_essays_ge5_chars"] > 480000
    mdv = tpp["multi_domain_lexical_validity"]
    assert mdv["cross_prompt_log_length_mean_r"] > 0.40
    assert 0.08 < mdv["single_prompt_reliability_r1"] < 0.15
    assert mdv["spearman_brown_curve_k_1_to_9"]["9"] > 0.50
    assert mdv["k9_to_k1_validity_ratio"] > 4.5
    amr = tpp["assortative_mind_requests_essay9"]
    assert amr["n_users_ge3_domains_with_essay9"] > 44000
    assert amr["q5_to_q1_mind_request_ratio"] > 1.35
    e9 = tpp["essay9_message_me_if"]
    assert e9["ratios"]["humor_to_tall"] > 10.0 and e9["ratios"]["humor_to_wealth"] > 20.0
    e3 = tpp["essay3_first_noticed"]["themes"]
    assert e3["smile_laugh"]["share"] > 0.25 and e3["eyes"]["share"] > 0.18
    sfc = okc["sf_childless_fertility_crossover"]
    assert sfc["by_age_band"]["25-29"]["women_minus_men_pp"] > 8.0
    assert sfc["by_age_band"]["40-44"]["women_minus_men_pp"] < -10.0
    assert 36.5 < sfc["interpolated_crossover_age"] < 40.0
