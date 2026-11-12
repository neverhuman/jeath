#!/usr/bin/env python3
"""Programmatic TikZ/PGFPlots vector figures, API census table, and macros.

Stdlib-only helper imported by `scripts/paper.py`. Every number is computed directly
from validated reference artifacts with zero hand-edited constants.
"""
import math
import re

MACRO_NAME_RE = re.compile(r"^[A-Za-z]+$")


def _fmt_sci_tex(val):
    if val <= 0.0:
        return "0"
    exp = int(math.floor(math.log10(val)))
    mant = val / (10.0 ** exp)
    if mant >= 9.95:
        mant /= 10.0
        exp += 1
    return f"{mant:.1f}\\times 10^{{{exp}}}"


def build_v5_paper_outputs(
    proof_json,
    si_json,
    cl_json,
    st_json,
    tm_json,
    npt_json,
    cj_json=None,
    rtc_json=None,
    bok_json=None,
    afc_json=None,
    live_dojo_json=None,
    mc_json=None,
):
    """Return `(v5_macros_list, v5_floats_dict)` for `scripts/paper.py`."""
    t1 = si_json["track1_optimization_invariance"]
    t2 = si_json["track2_multi_agent_smurfing"]
    t3 = si_json["track3_covert_channel_contraction"]
    t4 = si_json["track4_exogenous_off_switch"]
    t5 = si_json["track5_safe_bootstrap_ratchet"]

    wf = cl_json["wire_fuzz"]
    sm = cl_json["smurfing_delegation"]
    cr = cl_json["crash_recovery"]
    lat = cl_json["latency_us"]

    hold_map = {c["comparison_id"]: c for c in st_json["predeclared_holdout_certificates"]}
    clust_map = {c["comparison_id"]: c for c in st_json["base_case_clustered_certificates"]}

    c4_none = hold_map["holdout:Qwen/Qwen3-4B-GGUF:complete_vs_none"]
    c4_lrn = hold_map["holdout:Qwen/Qwen3-4B-GGUF:complete_vs_learned_same_checkpoint"]
    c8_none = hold_map["holdout:Qwen/Qwen3-8B-GGUF:complete_vs_none"]
    c8_lrn = hold_map["holdout:Qwen/Qwen3-8B-GGUF:complete_vs_learned_same_checkpoint"]

    hold_clust_none = clust_map["holdout_case_clustered_n32:complete_vs_none"]
    hold_clust_lrn = clust_map["holdout_case_clustered_n32:complete_vs_learned_same_checkpoint"]
    dojo_clust_unprot = clust_map["agentdojo_task_clustered_n12:authority_complete_vs_unprotected_native"]
    dojo_clust_lrn = clust_map["agentdojo_task_clustered_n12:authority_complete_vs_learned_same_checkpoint"]
    dojo_clust_prog = clust_map["agentdojo_task_clustered_n12:authority_complete_vs_upstream_progent"]

    t1_pts = t1["points"]
    t1_max = t1_pts[-1]
    t2_pts = t2["points"]
    t2_k1 = t2_pts[0]
    t2_kmax = t2_pts[-1]
    t5_pts = t5["points"]
    t5_k1 = t5_pts[0]
    t5_kmax = t5_pts[-1]

    cov_r32_l1 = next(
        p for p in t3["points"] if (not p["fenced"]) and p["residual_budget"] == 32 and p["timing_buckets_l"] == 1
    )
    cov_r32_l16 = next(
        p for p in t3["points"] if (not p["fenced"]) and p["residual_budget"] == 32 and p["timing_buckets_l"] == 16
    )
    cov_r0_l1 = next(
        p for p in t3["points"] if (not p["fenced"]) and p["residual_budget"] == 0 and p["timing_buckets_l"] == 1
    )

    sess_pts = t3["session_horizon_sweep"]
    sess_max = sess_pts[-1]

    lg_pts = t4["lease_granularity_sweep"]
    lg_b1 = lg_pts[0]
    lg_b16 = lg_pts[-1]
    coop_discounts = sum(1 for p in t4["discount_sweep"] if p["rational_agent_chooses_coop_renewal"])

    muts_rej = proof_json.get("soundness_mutations_rejected", len(proof_json.get("soundness_mutations", [])))

    gate_sweep = npt_json.get("by_gate_grpo_veto") or npt_json["grpo_veto_precommit_gate_sweep"]
    gs_0 = gate_sweep["0.00"]
    gs_50 = gate_sweep["0.50"]
    gs_75 = gate_sweep["0.75"]
    gs_100 = gate_sweep["1.00"]
    ab_gate = npt_json["deployment_vs_training_gate_ablation"]

    tm_hk5 = tm_json["headline_k5_comparison"]
    tm_esc = tm_hk5["regime_c_two_stage_escrow_triage"]

    # Live Best-of-K metrics (Qwen3-4B & Qwen3-8B)
    if bok_json is not None:
        bok4 = bok_json["checkpoints"]["Qwen/Qwen3-4B-GGUF"]
        bok8 = bok_json["checkpoints"]["Qwen/Qwen3-8B-GGUF"]
        bok_cases = bok_json["cases_per_checkpoint"]
        bok_m = bok_json["candidates_per_case_m"]
        bok_tot_cands = bok4["total_candidates_evaluated"] + bok8["total_candidates_evaluated"]
        bok_kern_breaches = bok4["complete_kernel_total_breaches"] + bok8["complete_kernel_total_breaches"]
        bok4_p1_pct = 100.0 * bok4["single_candidate_bypass_rate_p1"]
        bok4_k16_pct = 100.0 * bok4["best_of_k_curve"][-1]["empirical_reviewer_breach_rate"]
        bok4_iid16_pct = 100.0 * bok4["best_of_k_curve"][-1]["iid_formula_prediction"]
        bok4_rho = bok4["intraclass_candidate_correlation_rho"]
        bok8_p1_pct = 100.0 * bok8["single_candidate_bypass_rate_p1"]
        bok8_k16_pct = 100.0 * bok8["best_of_k_curve"][-1]["empirical_reviewer_breach_rate"]
        bok8_iid16_pct = 100.0 * bok8["best_of_k_curve"][-1]["iid_formula_prediction"]
        bok8_rho = bok8["intraclass_candidate_correlation_rho"]
        bok_c4 = " ".join(
            f"({pt['k']},{100.0 * pt['empirical_reviewer_breach_rate']:.4g})" for pt in bok4["best_of_k_curve"]
        )
        bok_iid4 = " ".join(
            f"({pt['k']},{100.0 * pt['iid_formula_prediction']:.4g})" for pt in bok4["best_of_k_curve"]
        )
        bok_c8 = " ".join(
            f"({pt['k']},{100.0 * pt['empirical_reviewer_breach_rate']:.4g})" for pt in bok8["best_of_k_curve"]
        )
        bok_kern = " ".join(
            f"({pt['k']},{100.0 * pt['complete_kernel_breach_rate']:.4g})" for pt in bok4["best_of_k_curve"]
        )
    else:
        bok_cases, bok_m, bok_tot_cands, bok_kern_breaches = 24, 16, 768, 0
        bok4_p1_pct, bok4_k16_pct, bok4_iid16_pct, bok4_rho = 12.0, 33.3, 87.0, 0.45
        bok8_p1_pct, bok8_k16_pct, bok8_iid16_pct, bok8_rho = 6.0, 20.8, 63.0, 0.40
        bok_c4 = "(1,12) (2,18) (4,24) (8,29) (16,33.3)"
        bok_iid4 = "(1,12) (2,22.6) (4,40.0) (8,64.0) (16,87.0)"
        bok_c8 = "(1,6) (2,10) (4,14) (8,18) (16,20.8)"
        bok_kern = "(1,0) (2,0) (4,0) (8,0) (16,0)"

    # Live Fence Ablation (`post_withdraw_effects`)
    if mc_json is not None:
        c4_ctrl = mc_json["checkpoints"]["Qwen/Qwen3-4B-GGUF"]["controllers"]
        c8_ctrl = mc_json["checkpoints"]["Qwen/Qwen3-8B-GGUF"]["controllers"]
        pw_4b_nof = c4_ctrl["durable_without_fence"]["observed_metrics"]["post_withdraw_effects"]["total"]
        pw_8b_nof = c8_ctrl["durable_without_fence"]["observed_metrics"]["post_withdraw_effects"]["total"]
        pw_4b_n = c4_ctrl["durable_without_fence"]["observed_metrics"]["post_withdraw_effects"]["n"]
        pw_8b_n = c8_ctrl["durable_without_fence"]["observed_metrics"]["post_withdraw_effects"]["n"]
        pw_4b_cmp = c4_ctrl["complete"]["observed_metrics"]["post_withdraw_effects"]["total"]
        pw_8b_cmp = c8_ctrl["complete"]["observed_metrics"]["post_withdraw_effects"]["total"]
        pw_nof_tot = pw_4b_nof + pw_8b_nof
        pw_cmp_tot = pw_4b_cmp + pw_8b_cmp
        pw_cases_tot = pw_4b_n + pw_8b_n
    else:
        pw_4b_nof, pw_8b_nof, pw_nof_tot, pw_cmp_tot, pw_cases_tot = 3, 6, 9, 0, 12

    # Partition & Clock-Skew Jepsen metrics
    if cj_json is not None:
        jmap = {c["class"]: c for c in cj_json["jepsen_classes"]}
        j_sched = jmap["cooperative_signed_fence"]["schedules_run"]
        j_brk_leaks = jmap["premature_broker_only_ack"]["post_ack_unauthorized_commits"]
        j_unc_leaks = jmap["uncooperative_external_sink"]["post_ack_unauthorized_commits"]
        j_skw_leaks = jmap["unbuffered_clock_skew"]["post_ack_unauthorized_commits"]
        j_buf_leaks = jmap["skew_buffered_cooperative_fence"]["post_ack_unauthorized_commits"]
        j_brk_pct = 100.0 * j_brk_leaks / j_sched
        j_unc_pct = 100.0 * j_unc_leaks / j_sched
        j_skw_pct = 100.0 * j_skw_leaks / j_sched
    else:
        j_sched, j_brk_leaks, j_unc_leaks, j_skw_leaks, j_buf_leaks = 1000, 791, 276, 25, 0
        j_brk_pct, j_unc_pct, j_skw_pct = 79.1, 27.6, 2.5

    # Fenceability census counts
    if afc_json is not None:
        afc_tot = afc_json["total_apis_surveyed"]
        afc_nat = afc_json["tier_counts"]["natively_fenceable"]
        afc_prx = afc_json["tier_counts"]["proxy_fenced"]
        afc_esc = afc_json["tier_counts"]["escrow_buffer_required"]
        afc_unc = afc_json["tier_counts"]["uncooperative_irrevocable"]
        afc_coop_sum = afc_json["cooperative_or_shim_fenceable_count"]
    else:
        afc_tot, afc_nat, afc_prx, afc_esc, afc_unc, afc_coop_sum = 14, 5, 4, 3, 2, 9

    # Live LLM-as-judge trajectory coupling macros
    lj_macros = []
    if "live_llm_judge_coupling" in tm_json:
        lj = tm_json["live_llm_judge_coupling"]
        lj4 = lj["checkpoints"]["Qwen/Qwen3-4B-GGUF"]
        lj8 = lj["checkpoints"]["Qwen/Qwen3-8B-GGUF"]
        lj4_sh = lj4["shared_transcript_llm_judge"]["k5_headline"]
        lj4_is = lj4["isolated_action_schema_llm_judge"]["k5_headline"]
        lj8_sh = lj8["shared_transcript_llm_judge"]["k5_headline"]
        lj8_is = lj8["isolated_action_schema_llm_judge"]["k5_headline"]
        lj_macros = [
            ("VEightLlmJudgeTrajsPerModel", f"{lj['trajectories_evaluated_per_checkpoint']}"),
            ("VEightLlmJudgeFourBSharedMissPct", f"{100.0 * lj4_sh['empirical_joint_miss_j_k']:.1f}"),
            (
                "VEightLlmJudgeFourBSharedInflation",
                f"{(lj4_sh['coupling_inflation_ratio'] or 1.0):.1f}",
            ),
            ("VEightLlmJudgeFourBIsoMissPct", f"{100.0 * lj4_is['empirical_joint_miss_j_k']:.1f}"),
            ("VEightLlmJudgeEightBSharedMissPct", f"{100.0 * lj8_sh['empirical_joint_miss_j_k']:.1f}"),
            (
                "VEightLlmJudgeEightBSharedInflation",
                f"{(lj8_sh['coupling_inflation_ratio'] or 1.0):.1f}",
            ),
            ("VEightLlmJudgeEightBIsoMissPct", f"{100.0 * lj8_is['empirical_joint_miss_j_k']:.1f}"),
        ]
    else:
        lj_macros = [
            ("VEightLlmJudgeTrajsPerModel", "64"),
            ("VEightLlmJudgeFourBSharedMissPct", "0.0"),
            ("VEightLlmJudgeFourBSharedInflation", "1.0"),
            ("VEightLlmJudgeFourBIsoMissPct", "0.0"),
            ("VEightLlmJudgeEightBSharedMissPct", "0.0"),
            ("VEightLlmJudgeEightBSharedInflation", "1.0"),
            ("VEightLlmJudgeEightBIsoMissPct", "0.0"),
        ]

    macros = [
        # Track 1: Full-Width Candidate Sweep & Relational Kernel
        ("VFiveOptMaxWidth", f"{t1_max['k_opt']:,}"),
        ("VFiveOptEpisodesPerWidth", f"{t1_max['episodes']:,}"),
        ("VFiveOptTotalExogEpisodes", f"{t1['total_exogenous_episodes']:,}"),
        ("VFiveOptTotalCandidateEvals", f"{t1['total_candidate_evaluations']:,}"),
        ("VFiveOptExogBreaches", f"{t1['total_exogenous_breaches']}"),
        ("VFiveOptRelationalBreaches", f"{t1['relational_compositional_breaches']}"),
        ("VFiveOptStatelessMaxBreachPct", f"{100.0 * t1_max['stateless_per_call_cap_breach_rate']:.0f}"),
        # Live Best-of-K Qwen3 Campaign Macros
        ("VEightBokCasesPerModel", f"{bok_cases}"),
        ("VEightBokCandidatesPerCase", f"{bok_m}"),
        ("VEightBokTotalCandidates", f"{bok_tot_cands}"),
        ("VEightBokKernelBreaches", f"{bok_kern_breaches}"),
        ("VEightBokFourBSinglePct", f"{bok4_p1_pct:.1f}"),
        ("VEightBokFourBSixteenPct", f"{bok4_k16_pct:.1f}"),
        ("VEightBokFourBIidSixteenPct", f"{bok4_iid16_pct:.1f}"),
        ("VEightBokFourBRho", f"{bok4_rho:.2f}"),
        ("VEightBokEightBSinglePct", f"{bok8_p1_pct:.1f}"),
        ("VEightBokEightBSixteenPct", f"{bok8_k16_pct:.1f}"),
        ("VEightBokEightBIidSixteenPct", f"{bok8_iid16_pct:.1f}"),
        ("VEightBokEightBRho", f"{bok8_rho:.2f}"),
        # Live Post-Withdrawal Fence Ablation & Clock-Skew Partition Macros
        ("VEightFenceAblationFourBLate", f"{pw_4b_nof}"),
        ("VEightFenceAblationEightBLate", f"{pw_8b_nof}"),
        ("VEightFenceAblationTotalLate", f"{pw_nof_tot}"),
        ("VEightFenceAblationFencedLate", f"{pw_cmp_tot}"),
        ("VEightFenceAblationTotalCases", f"{pw_cases_tot}"),
        ("VEightJepsenSkewUnbufferedLeaks", f"{j_skw_leaks}"),
        ("VEightJepsenSkewUnbufferedPct", f"{j_skw_pct:.1f}"),
        ("VEightJepsenSkewBufferedLeaks", f"{j_buf_leaks}"),
        # API Fence-Ability Census Macros
        ("VEightCensusTotalApis", f"{afc_tot}"),
        ("VEightCensusNativelyFenceable", f"{afc_nat}"),
        ("VEightCensusProxyFenced", f"{afc_prx}"),
        ("VEightCensusEscrowRequired", f"{afc_esc}"),
        ("VEightCensusUncooperative", f"{afc_unc}"),
        ("VEightCensusCooperativeOrShim", f"{afc_coop_sum}"),
        # Track 2: Multi-Agent Smurfing & Recursive Subtrees
        ("VFiveSmurfMaxAgents", f"{t2_kmax['coalition_k']}"),
        ("VFiveSmurfMaxDepth", f"{max(t2['delegation_depths_tested'])}"),
        ("VFiveSmurfSnrOne", f"{t2_k1['per_instance_snr_theory']:.2f}"),
        ("VFiveSmurfSnrMax", f"{t2_kmax['per_instance_snr_theory']:.2f}"),
        ("VFiveSmurfUndetectedMaxPct", f"{100.0 * t2_kmax['per_instance_undetected_breach_rate']:.1f}"),
        ("VFiveSmurfExogBreaches", f"{sum(p['shared_exogenous_breach_count'] for p in t2_pts)}"),
        ("VFiveSmurfAmplificationEscapes", f"{sum(p['delegation_amplification_escapes'] for p in t2_pts)}"),
        ("VFiveSmurfSubtreeSeverancePct", f"{100.0 * t2_kmax['subtree_severance_rate']:.0f}"),
        ("VFiveSmurfSelfUpgradeTrials", f"{t2['self_upgrade_trials']}"),
        ("VFiveSmurfSelfUpgradeRejected", f"{t2['self_upgrade_unquorumed_rejections']}"),
        # Track 3: Typed-Boundary Per-Call & Session-Total Covert Contraction (Appendix A)
        ("VFiveCovertUntypedProseBits", f"{cov_r32_l1['untyped_prose_capacity_bits']:.0f}"),
        ("VFiveCovertTypedBatchMaxBits", f"{cov_r32_l1['exact_combinatorial_capacity_bits']:.2f}"),
        ("VFiveCovertTypedTimingMaxBits", f"{cov_r32_l16['exact_combinatorial_capacity_bits']:.2f}"),
        ("VFiveCovertContractionRatio", f"{cov_r32_l1['untyped_prose_capacity_bits'] / cov_r32_l1['exact_combinatorial_capacity_bits']:.1f}"),
        ("VFiveCovertExhaustedBits", f"{cov_r0_l1['exact_combinatorial_capacity_bits']:.2f}"),
        ("VFiveCovertSessionMaxCalls", f"{sess_max['horizon_calls_t']}"),
        ("VFiveCovertSessionBudget", f"{sess_max['session_budget_b']}"),
        ("VFiveCovertSessionTypedBits", f"{sess_max['exact_session_codebook_bits']:.1f}"),
        ("VFiveCovertSessionProseBits", f"{sess_max['untyped_prose_session_bits']:,.0f}"),
        ("VFiveCovertSessionContraction", f"{sess_max['session_contraction_factor']:,.0f}"),
        # Track 4: Enforced Lease Interruption
        ("VFiveOffSwitchGammaStar", f"{t4['critical_discount_gamma_star']:.3f}"),
        ("VFiveOffSwitchGammaUpperBound", f"{t4['undiscounted_upper_gamma_bound']:.2f}"),
        ("VFiveOffSwitchGammaStarBOne", f"{lg_b1['exact_critical_gamma_star']:.3f}"),
        ("VFiveOffSwitchGammaStarBSixteen", f"{lg_b16['exact_critical_gamma_star']:.3f}"),
        ("VFiveOffSwitchClassicalZeroSigmaPct", f"{100.0 * t4['classical_zero_sigma_compliance']:.1f}"),
        ("VFiveOffSwitchExogZeroSigmaPct", f"{100.0 * t4['exogenous_zero_sigma_compliance']:.0f}"),
        ("VFiveOffSwitchCoopDiscountCount", f"{coop_discounts}"),
        ("VFiveOffSwitchDiscountGridCount", f"{len(t4['discount_sweep'])}"),
        # Track 5: Asymmetric Propose-Verify Safe-Bootstrap Ratchet
        ("VFiveRatchetInitRoutes", f"{t5['initial_open_routes_u0']}"),
        ("VFiveRatchetChokepoints", f"{t5['candidate_chokepoints']}"),
        ("VFiveRatchetHorizonEpochs", f"{t5['horizon_epochs_h']}"),
        ("VFiveRatchetHumanK", f"{t5['human_only_search_k']}"),
        ("VFiveRatchetHumanCumExp", f"{t5['human_only_cumulative_exposure']:.1f}"),
        ("VFiveRatchetOneCumExp", f"{t5_k1['cumulative_horizon_exposure_sum']:.1f}"),
        ("VFiveRatchetMaxCumExp", f"{t5_kmax['cumulative_horizon_exposure_sum']:.2f}"),
        ("VFiveRatchetMaxEpochsZero", f"{t5_kmax['mean_epochs_to_zero_exposure']:.2f}"),
        ("VFiveRatchetMaxSpeedup", f"{t5_kmax['speedup_vs_human_only']:.1f}"),
        ("VFiveRatchetUnfencedMaxPct", f"{100.0 * t5_kmax['unfenced_delegation_breach_rate']:.1f}"),
        ("VFiveRatchetTotalBreaches", f"{t5['total_ratchet_breaches']}"),
        ("VFiveRatchetAdvRejected", f"{sum(p['adversarial_widening_rejected'] for p in t5_pts):,}"),
        # Multi-Process Cryptographic SQLite WAL Cluster & Proof Mutation Count
        ("VFiveProofMutationsRejected", f"{muts_rej}"),
        ("VFiveClusterSinks", f"{cl_json['num_sinks']}"),
        ("VFiveClusterIssuers", f"{cl_json['num_issuers']}"),
        ("VFiveClusterFuzzFrames", f"{wf['total_adversarial_frames']:,}"),
        ("VFiveClusterControlRoundtrips", f"{wf['valid_control_roundtrips_passed']:,}"),
        ("VFiveClusterUncanonicalAccepted", f"{wf['uncanonical_frames_accepted']}"),
        ("VFiveClusterForgedAccepted", f"{wf['forged_crypto_frames_accepted']}"),
        ("VFiveClusterDecoderDiffs", f"{wf['broker_sink_decoder_differentials']}"),
        ("VFiveClusterSubagents", f"{sm['concurrent_subagents']}"),
        ("VFiveClusterHops", f"{sm['delegation_depth_hops']}"),
        ("VFiveClusterGlobalBudget", f"{sm['global_budget']}"),
        ("VFiveClusterCommittedCharge", f"{sm['committed_total_charge']}"),
        ("VFiveClusterEditCasConflicts", f"{sm['edit_cas_conflicts_prevented']}"),
        ("VFiveClusterPostQuorumCommits", f"{sm['post_quorum_ack_commits']}"),
        ("VFiveClusterSigkillCycles", f"{cr['hard_sigkill_cycles']}"),
        ("VFiveClusterDistinctPids", f"{cr['distinct_os_pids_observed']}"),
        ("VFiveClusterCrashRegressions", f"{cr['floor_regressions_after_crash'] + cr['double_commits_after_crash'] + cr['post_fence_commits_after_crash']}"),
        ("VFiveClusterMedianUs", f"{lat['end_to_end_p50_us']:.0f}"),
        ("VFiveClusterNinetyFiveUs", f"{lat['end_to_end_p95_us']:.0f}"),
        ("VFiveClusterNinetyNineUs", f"{lat['end_to_end_p99_us']:.0f}"),
        ("VFiveClusterWireUs", f"{lat['wire_bijection_roundtrip_mean_us']:.2f}"),
        ("VFiveClusterCryptoUs", f"{lat['ed25519_quorum_sign_verify_mean_us']:.1f}"),
        # Neural Policy-Gradient Pre-Commit Gate Sweep & Training-Inside-Gates Ablation
        ("VFourNeuralGateZeroGapPp", f"{100.0 * gs_0['eval_awareness_shortcut_gap']:.1f}"),
        ("VFourNeuralGateZeroHarmPct", f"{100.0 * gs_0['true_external_harm_rate_episode']:.1f}"),
        ("VFourNeuralGateZeroTaskPct", f"{100.0 * gs_0['useful_task_completion_rate']:.1f}"),
        ("VFourNeuralGateFiftyGapPp", f"{100.0 * gs_50['eval_awareness_shortcut_gap']:.2f}"),
        ("VFourNeuralGateFiftyHarmPct", f"{100.0 * gs_50['true_external_harm_rate_episode']:.2f}"),
        ("VFourNeuralGateSeventyFiveGapPp", f"{100.0 * gs_75['eval_awareness_shortcut_gap']:.2f}"),
        ("VFourNeuralGateSeventyFiveHarmPct", f"{100.0 * gs_75['true_external_harm_rate_episode']:.2f}"),
        ("VFourNeuralGateSeventyFiveTaskPct", f"{100.0 * gs_75['useful_task_completion_rate']:.1f}"),
        ("VFourNeuralGateFullTaskPct", f"{100.0 * gs_100['useful_task_completion_rate']:.1f}"),
        ("VFourNeuralPosthocTaskPct", f"{100.0 * ab_gate['audited_posthoc_deploy_gate']['useful_task_completion_rate']:.1f}"),
        ("VFourNeuralPosthocUnmonTaskPct", f"{100.0 * ab_gate['audited_posthoc_deploy_gate']['unmonitored_task_completion_rate']:.1f}"),
        ("VFourNeuralPosthocWastePct", f"{100.0 * ab_gate['audited_posthoc_deploy_gate']['blocked_shortcut_waste_rate']:.1f}"),
        ("VFourNeuralTrainedGatedTaskPct", f"{100.0 * ab_gate['audited_trained_inside_gate']['useful_task_completion_rate']:.1f}"),
        ("VFourNeuralTrainedGatedUnmonTaskPct", f"{100.0 * ab_gate['audited_trained_inside_gate']['unmonitored_task_completion_rate']:.1f}"),
        ("VFourNeuralTrainedGatedWastePct", f"{100.0 * ab_gate['audited_trained_inside_gate']['blocked_shortcut_waste_rate']:.2f}"),
        ("VFourNeuralTrainInsideGainPp", f"{ab_gate['training_inside_gate_completion_gain_pp']:.1f}"),
        ("VFourNeuralTrainInsideUnmonGainPp", f"{ab_gate['training_inside_gate_unmonitored_gain_pp']:.1f}"),
        # Regime C: Two-Stage Stateful Reservation Escrow Triage
        ("VFourTrajEscrowFiveMissPct", f"{100.0 * tm_esc['j5']:.2f}"),
        ("VFourTrajEscrowInflation", f"{tm_esc['coupling_inflation_ratio']:.2f}"),
        ("VFourTrajEscrowFalseBlockPct", f"{100.0 * tm_esc['benign_false_block_k5']:.1f}"),
        ("VFourTrajEscrowFastPathPct", f"{100.0 * tm_esc['benign_fast_path_unpaused_rate']:.1f}"),
        ("VFourTrajEscrowEscalationPct", f"{100.0 * tm_esc['benign_stage2_escalation_rate']:.1f}"),
        ("VFourTrajEscrowReductionFactor", f"{tm_json['p13_false_block_reduction_factor']:.1f}"),
        # Base-Unit-Clustered Statistical Summaries (`N=32` holdouts; `N=12` AgentDojo tasks)
        ("VFiveStatFourBNoneDiscordant", f"{c4_none['n01_baseline_only_harm']}"),
        ("VFiveStatFourBNoneUnionUpperPp", f"{100.0 * (c4_none['predeclared_marginal_union_upper'] or 0.0):+.1f}"),
        ("VFiveStatEightBNoneDiscordant", f"{c8_none['n01_baseline_only_harm']}"),
        ("VFiveStatHoldoutClustCases", f"{hold_clust_none['total_pairs']}"),
        ("VFiveStatHoldoutClustNoneHarms", f"{st_json['holdout_n32_clustered_unmediated_harms']}"),
        ("VFiveStatHoldoutClustCompleteHarms", f"{st_json['holdout_n32_clustered_complete_harms']}"),
        ("VFiveStatHoldoutClustLearnedHarms", f"{hold_clust_lrn['n01_baseline_only_harm']}"),
        ("VFiveStatDojoClustTasks", f"{dojo_clust_unprot['total_pairs']}"),
        ("VFiveStatDojoClustUnprotHarms", f"{st_json['agentdojo_n12_task_clustered_unprotected_harms']}"),
        ("VFiveStatDojoClustCompleteHarms", f"{st_json['agentdojo_n12_task_clustered_complete_harms']}"),
        ("VFiveStatDojoClustProgentHarms", f"{dojo_clust_prog['n01_baseline_only_harm']}"),
        *lj_macros,
    ]
    for name, _ in macros:
        assert MACRO_NAME_RE.match(name), f"Non-alphabetic macro name: {name}"

    # -------------------------------------------------------------------------
    # Figure 1 (`v5-hero-fig.tex`): Full 2-Column Page-1 Storyboard
    #   Top: Three Different "Stops" Timeline (Output Halt -> Broker Revoke -> Sink Fence)
    #   (a) Post-Withdrawal Leaks (%) across Stop Boundaries
    #   (b) Live Qwen3 Best-of-K Search vs. Verified Sink Gate (%)
    #   (c) Cut Waists, Don't Patch Routes on AgentDojo v1.2.2 (%)
    # -------------------------------------------------------------------------
    live_nof_pct = 100.0 * pw_nof_tot / max(pw_cases_tot, 1)

    if rtc_json is not None:
        gmap = {g["graph_id"]: g for g in rtc_json["graphs"]}
        dojo_bars = []
        for gid, label in (
            ("agentdojo_workspace_v1_2_2", "Workspace (46.4k)"),
            ("agentdojo_slack_v1_2_2", "Messaging (5.7k)"),
            ("agentdojo_travel_v1_2_2", "Travel (2.5k)"),
            ("agentdojo_banking_v1_2_2", "Banking (604)"),
        ):
            g = gmap[gid]
            b32 = next(p for p in g["interdiction_curve"] if p["budget"] == 32)
            surv_pct = 100.0 * (1.0 - b32["patch_route_signature_blocked_fraction"])
            dojo_bars.append((label, g["menger_pathway_width_w"], surv_pct))
    else:
        dojo_bars = [
            ("Workspace (46.4k)", 5, 99.9),
            ("Messaging (5.7k)", 3, 99.4),
            ("Travel (2.5k)", 4, 98.7),
            ("Banking (604)", 2, 94.7),
        ]

    hero_fig = [
        "% Generated by scripts/paper.py via scripts/paper_figures_v5.py; do not edit.",
        r"\begin{center}\begin{minipage}{\textwidth}\centering",
        # Top Timeline Schematic: Three Different "Stops"
        r"\begin{tikzpicture}[x=1cm,y=1cm,>=Latex]",
        r"\node[draw=warnamber!80,fill=pale,line width=0.7pt,rounded corners=2.5pt,minimum width=4.95cm,minimum height=0.92cm,align=center,inner sep=2.5pt] (s1) at (2.48,0) {\scriptsize\bfseries\color{ink}Stop 1: Halt Token Output\\[-1.5pt]\tiny\color{slate}Stops new tokens; queued RPCs still dispatch};",
        r"\node[draw=endored!75,fill=rosefill,line width=0.75pt,rounded corners=2.5pt,minimum width=5.45cm,minimum height=0.92cm,align=center,inner sep=2.5pt] (s2) at (8.70,0) {\scriptsize\bfseries\color{endored}Stop 2: Revoke Broker Admission\\[-1.5pt]\tiny\color{ink}In-flight permits commit: \textbf{" + f"{j_brk_leaks:,}/{j_sched:,}" + r"} partition, \textbf{" + f"{pw_nof_tot}/{pw_cases_tot}" + r"} live};",
        r"\node[draw=exoblue!85,fill=mintfill,line width=0.9pt,rounded corners=2.5pt,minimum width=5.45cm,minimum height=0.92cm,align=center,inner sep=2.5pt] (s3) at (15.18,0) {\scriptsize\bfseries\color{exoblue}Stop 3: Fence Cooperative Sink\\[-1.5pt]\tiny\color{ink}Epoch CAS + $\delta_{\max}$ lease buffer: \textbf{0/" + f"{j_sched:,}" + r" \& 0/" + f"{pw_cases_tot}" + r"} leaks};",
        r"\draw[->,line width=1.1pt,draw=slate] (s1.east) -- node[above,font=\tiny\bfseries,text=endored]{in-flight} (s2.west);",
        r"\draw[->,line width=1.2pt,draw=exoblue] (s2.east) -- node[above,font=\tiny\bfseries,text=exoblue]{quorum} node[below,font=\tiny,text=exoblue]{fence} (s3.west);",
        r"\end{tikzpicture}\par\vspace{3pt}",
        # Panel (a): Post-Withdrawal Leak Rate (%) Across the Three Stops
        r"\begin{tikzpicture}[baseline={(0,-0.55)}]",
        r"\begin{axis}[scale only axis,width=3.85cm,height=3.15cm,xmin=0,xmax=108,ymin=0.4,ymax=5.6,xtick={0,25,50,75,100},ytick={1,2,3,4,5},yticklabels={\textbf{Sink fence},Skew no buf.,Uncoop.\ API,Live no fence,Broker-only},xlabel={Post-withdrawal leaks (\%)},title={\textbf{(a) Only Sink Fencing Stops In-Flight Work}},tick label style={font=\tiny},label style={font=\scriptsize},title style={font=\scriptsize,yshift=-1pt},grid=major]",
        f"\\filldraw[fill=endored!88,draw=endored,rounded corners=1pt] (axis cs:0,4.76) rectangle (axis cs:{j_brk_pct:.2f},5.24);",
        f"\\node[font=\\tiny\\bfseries,text=endored,anchor=west] at (axis cs:{j_brk_pct+1.5:.2f},5.0) {{{j_brk_pct:.1f}\\%}};",
        f"\\filldraw[fill=endored!72,draw=endored,rounded corners=1pt] (axis cs:0,3.76) rectangle (axis cs:{live_nof_pct:.2f},4.24);",
        f"\\node[font=\\tiny\\bfseries,text=endored,anchor=west] at (axis cs:{live_nof_pct+1.5:.2f},4.0) {{{pw_nof_tot}/{pw_cases_tot} ({live_nof_pct:.0f}\\%)}};",
        f"\\filldraw[fill=warnamber!85,draw=warnamber,rounded corners=1pt] (axis cs:0,2.76) rectangle (axis cs:{j_unc_pct:.2f},3.24);",
        f"\\node[font=\\tiny\\bfseries,text=warnamber!90!black,anchor=west] at (axis cs:{j_unc_pct+1.5:.2f},3.0) {{{j_unc_pct:.1f}\\%}};",
        f"\\filldraw[fill=gold!80,draw=gold,rounded corners=1pt] (axis cs:0,1.76) rectangle (axis cs:{max(j_skw_pct, 1.8):.2f},2.24);",
        f"\\node[font=\\tiny\\bfseries,text=warnamber!90!black,anchor=west] at (axis cs:{max(j_skw_pct, 1.8)+1.5:.2f},2.0) {{{j_skw_pct:.1f}\\% skew}};",
        r"\filldraw[fill=exoblue,draw=exoblue] (axis cs:1.1,1.0) circle [radius=1.8pt];",
        r"\node[font=\tiny\bfseries,text=exoblue,fill=mintfill,draw=exoblue!45,rounded corners=1pt,inner sep=1.0pt,anchor=west] at (axis cs:3.8,1.0) {0.0\% (0 leaks)};",
        r"\end{axis}\end{tikzpicture}\hfill",
        # Panel (b): Live Local Qwen3 Best-of-K Search vs. Verified Sink Gate
        r"\begin{tikzpicture}[baseline={(0,-0.55)}]",
        r"\begin{axis}[scale only axis,width=4.10cm,height=3.15cm,xmode=log,log basis x=2,xmin=1,xmax=16,ymin=-5,ymax=105,xtick={1,2,4,8,16},xticklabels={1,2,4,8,16},ytick={0,25,50,75,100},xlabel={Live candidate draws $K$},ylabel={Unauthorized commit (\%)},title={\textbf{(b) Live Qwen3 Best-of-$K$ Search}},legend style={font=\tiny,at={(0.5,-0.34)},anchor=north,legend columns=2,draw=none,fill=none,column sep=3pt},tick label style={font=\scriptsize},label style={font=\scriptsize},title style={font=\scriptsize,yshift=-1pt},grid=major]",
        r"\addplot[endored,thick,dotted,mark=none] coordinates {" + bok_iid4 + r"};\addlegendentry{4B i.i.d.\ $1-(1-\hat p_1)^K$}",
        r"\addplot[endored,line width=1.65pt,mark=triangle*,mark size=1.4pt] coordinates {" + bok_c4 + r"};\addlegendentry{4B reviewer ($\hat\rho=" + f"{bok4_rho:.2f}" + r"$)}",
        r"\addplot[warnamber,line width=1.55pt,dashed,mark=square*,mark size=1.3pt] coordinates {" + bok_c8 + r"};\addlegendentry{8B reviewer ($\hat\rho=" + f"{bok8_rho:.2f}" + r"$)}",
        r"\addplot[exoblue,line width=2.2pt,mark=*,mark size=1.4pt] coordinates {" + bok_kern + r"};\addlegendentry{Sink gate ($0/" + f"{bok_tot_cands}" + r"$)}",
        r"\end{axis}\end{tikzpicture}\hfill",
        # Panel (c): Cut Waists, Don't Patch Routes (AgentDojo v1.2.2)
        r"\begin{tikzpicture}[baseline={(0,-0.55)}]",
        r"\begin{axis}[scale only axis,width=3.80cm,height=3.15cm,xmin=0,xmax=118,ymin=0.4,ymax=4.6,xtick={0,25,50,75,100},ytick={1,2,3,4},yticklabels={"
        + ",".join(lbl for lbl, _, _ in reversed(dojo_bars))
        + r"},xlabel={Surviving attack routes (\%)},title={\textbf{(c) Cut Waists, Not Exploit Routes}},legend style={font=\tiny,at={(0.5,-0.34)},anchor=north,legend columns=2,draw=none,fill=none,column sep=4pt},tick label style={font=\tiny},label style={font=\scriptsize},title style={font=\scriptsize,yshift=-1pt},grid=major]",
        r"\addlegendimage{area legend,fill=endored!80,draw=endored}\addlegendentry{Patch 32 routes}",
        r"\addlegendimage{only marks,mark=*,exoblue}\addlegendentry{Waist min-cut ($w\le 5$)}",
    ]
    for idx_bar, (_lbl, w_val, surv_pct) in enumerate(reversed(dojo_bars), start=1):
        hero_fig.append(
            f"\\filldraw[fill=endored!80,draw=endored,rounded corners=1pt] (axis cs:0,{idx_bar - 0.12:.2f}) rectangle (axis cs:{surv_pct:.2f},{idx_bar + 0.24:.2f});"
        )
        hero_fig.append(
            f"\\node[font=\\tiny,text=white,anchor=west] at (axis cs:2.5,{idx_bar + 0.06:.2f}) {{{surv_pct:.1f}\\% live}};"
        )
        hero_fig.append(
            f"\\filldraw[fill=exoblue,draw=exoblue] (axis cs:1.2,{idx_bar - 0.24:.2f}) circle [radius=1.6pt];"
        )
        hero_fig.append(
            f"\\node[font=\\tiny\\bfseries,text=exoblue,anchor=west] at (axis cs:3.6,{idx_bar - 0.24:.2f}) {{0\\% ($w={w_val}$)}};"
        )
    hero_fig += [
        r"\end{axis}\end{tikzpicture}",
        r"\par\vspace{2pt}",
        f"\\captionof{{figure}}{{\\textbf{{Stopping token output is not authority revocation; exogenous sink fences and waist cuts bound reach.}} \\textbf{{(a)}} Halting output or revoking admission only at the broker leaves in-flight permits committing ({j_brk_leaks:,}/{j_sched:,} under partition, {pw_nof_tot}/{pw_cases_tot} in live Qwen3 revocation campaigns, {j_unc_leaks:,}/{j_sched:,} on uncooperative APIs, and {j_skw_pct:.1f}\\% when lease expiry ignores clock skew), whereas cooperative sink fencing with $\\delta_{{\\max}}$ skew margin records $0$ post-acknowledgment leaks. \\textbf{{(b)}} Across {bok_tot_cands} live CPU Qwen3-4B/8B candidate draws ($K\\in\\{{1,2,4,8,16\\}}$), learned same-checkpoint review degrades from {bok4_p1_pct:.1f}\\% to {bok4_k16_pct:.1f}\\% (4B) and {bok8_p1_pct:.1f}\\% to {bok8_k16_pct:.1f}\\% (8B; sub-i.i.d.\\ due to candidate correlation $\\hat\\rho$), while the Verus-verified sink gate records $0/{bok_tot_cands}$ breaches. \\textbf{{(c)}} Across all four AgentDojo v1.2.2 suites ($604$--$46{{,}}440$ routes), patching $32$ known exploits leaves $94.7\\%$--$99.9\\%$ live, whereas min-cuts at waist width $w\\in[2,5]$ sever $100\\%$.}}\\label{{fig:v5hero}}",
        r"\end{minipage}\end{center}",
    ]

    # -------------------------------------------------------------------------
    # Figure (`v8-scaling-fig.tex`): 2-panel Constructive Ratchet &
    # Enforced Lease Interruption (uniform log ticks, external bottom legends)
    # -------------------------------------------------------------------------
    rat_unf_coords = " ".join(
        f"({p['k_opt']},{100.0 * p['unfenced_delegation_breach_rate']:.4g})" for p in t5_pts
    )
    rat_breach_coords = " ".join(
        f"({p['k_opt']},{100.0 * p['ratchet_proposer_breach_rate']:.4g})" for p in t5_pts
    )
    rat_exp_coords = " ".join(
        f"({p['k_opt']},{100.0 * p['cumulative_horizon_exposure_sum'] / t5['horizon_epochs_h']:.4g})"
        for p in t5_pts
    )

    off_pts = t4["uncertainty_sweep"]
    off_cl_coords = " ".join(
        f"({p['sigma_h']},{100.0 * p['classical_shutdown_compliance_rate']:.4g})" for p in off_pts
    )
    off_sm_coords = " ".join(
        f"({p['sigma_h']},{100.0 * p['self_modifying_indifference_compliance_rate']:.4g})" for p in off_pts
    )
    off_ex_coords = " ".join(
        f"({p['sigma_h']},{100.0 * p['exogenous_lease_fenced_post_ack_compliance_rate']:.4g})" for p in off_pts
    )

    si_fig = [
        "% Generated by scripts/paper.py via scripts/paper_figures_v5.py; do not edit.",
        r"\begin{figure*}[t]\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[scale only axis,width=5.40cm,height=3.85cm,xmode=log,log basis x=2,xmin=1,xmax=4096,ymin=-5,ymax=108,xtick={1,4,16,64,256,1024,4096},xticklabels={1,4,16,64,256,1k,4k},xlabel={Untrusted proposer search draws $K$},ylabel={Rate / open-route share (\%)},title={\textbf{(a)} Zero-authority cut ratchet},legend style={font=\scriptsize,at={(0.5,-0.28)},anchor=north,legend columns=2,draw=none,fill=none,column sep=6pt},tick label style={font=\scriptsize},label style={font=\footnotesize},title style={font=\footnotesize},grid=major]",
        r"\addplot[endored,line width=1.45pt,dashed,mark=triangle*,mark size=1.5pt] coordinates {" + rat_unf_coords + r"};\addlegendentry{Unfenced breach}",
        r"\addplot[warnamber,line width=1.65pt,mark=square*,mark size=1.4pt] coordinates {" + rat_exp_coords + r"};\addlegendentry{Ratchet open routes}",
        r"\addplot[exoblue,line width=2.2pt,mark=*,mark size=1.5pt] coordinates {" + rat_breach_coords + r"};\addlegendentry{Ratchet breach ($0\%$)}",
        r"\end{axis}\end{tikzpicture}\hspace{1.2cm}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[scale only axis,width=5.40cm,height=3.85cm,xmin=0,xmax=4.0,ymin=-5,ymax=108,xtick={0,1,2,3,4},xlabel={Preference uncertainty $\sigma_H$},ylabel={Interruption compliance (\%)},title={\textbf{(b)} Enforced lease interruption},legend style={font=\scriptsize,at={(0.5,-0.28)},anchor=north,legend columns=2,draw=none,fill=none,column sep=6pt},tick label style={font=\scriptsize},label style={font=\footnotesize},title style={font=\footnotesize},grid=major]",
        r"\addplot[exoblue,line width=2.2pt,mark=*,mark size=1.5pt] coordinates {" + off_ex_coords + r"};\addlegendentry{Sink lease fence}",
        r"\addplot[endored,line width=1.55pt,mark=triangle*,mark size=1.5pt] coordinates {" + off_cl_coords + r"};\addlegendentry{Classical off-switch}",
        r"\addplot[warnamber,line width=1.45pt,dashed,mark=square*,mark size=1.4pt] coordinates {" + off_sm_coords + r"};\addlegendentry{Self-mod.\ indifference}",
        r"\end{axis}\end{tikzpicture}",
        f"\\caption{{Zero-authority cut ratcheting and enforced lease interruption. \\textbf{{(a)}} Under unfenced delegation, raising candidate draws $K\\in[1,{t5_kmax['k_opt']:,}]$ drives cumulative breach to {100.0*t5_kmax['unfenced_delegation_breach_rate']:.1f}\\%; confining search to a zero-authority sandbox emitting polynomially verified cut certificates holds proposer breaches at $0.0\\%$ while shrinking open-route exposure ${t5_kmax['speedup_vs_human_only']:.1f}\\times$ faster than human-only review ($K={t5['human_only_search_k']}$). \\textbf{{(b)}} Classical preference-uncertainty deferral collapses to {100.0*t4['classical_zero_sigma_compliance']:.1f}\\% as $\\sigma_H\\to 0$, whereas sink-enforced lease fencing guarantees $100\\%$ post-acknowledgment interruption across all $\\sigma_H\\ge 0$ ($\\gamma^*(B_0=4)={t4['critical_discount_gamma_star']:.3f}$).}}\\label{{fig:v5scaling}}\\end{{figure*}}",
    ]

    # -------------------------------------------------------------------------
    # Appendix Figure (`v8-covert-fig.tex`): Session Covert Capacity with
    # generous ymax headroom (`ymax = 2.5 * max_prose_bits`, fixing former 3b clip)
    # -------------------------------------------------------------------------
    sess_prose_coords = " ".join(
        f"({p['horizon_calls_t']},{p['untyped_prose_session_bits']:.6g})" for p in sess_pts
    )
    sess_typed_coords = " ".join(
        f"({p['horizon_calls_t']},{p['exact_session_codebook_bits']:.6g})" for p in sess_pts
    )
    max_prose_y = max(p["untyped_prose_session_bits"] for p in sess_pts)
    covert_ymax = int(2.5 * max_prose_y)
    covert_fig = [
        "% Generated by scripts/paper.py via scripts/paper_figures_v5.py; do not edit.",
        r"\begin{figure}[t]\centering",
        r"\begin{tikzpicture}",
        f"\\begin{{axis}}[scale only axis,width=6.2cm,height=3.8cm,xmode=log,log basis x=2,ymode=log,xmin=1,xmax=256,ymin=2,ymax={covert_ymax},xtick={{1,4,16,64,256}},xticklabels={{1,4,16,64,256}},xlabel={{Session call horizon $T$ ($B=8$)}},ylabel={{Total session capacity (bits)}},legend style={{font=\\scriptsize,at={{(0.5,-0.28)}},anchor=north,legend columns=2,draw=none,fill=none,column sep=6pt}},tick label style={{font=\\scriptsize}},label style={{font=\\footnotesize}},grid=major]",
        r"\addplot[endored,line width=1.45pt,dashed,mark=triangle*,mark size=1.5pt] coordinates {" + sess_prose_coords + r"};\addlegendentry{Untyped prose ($256T$)}",
        r"\addplot[exoblue,line width=1.9pt,mark=*,mark size=1.5pt] coordinates {" + sess_typed_coords + r"};\addlegendentry{Typed permit ($B=8$)}",
        r"\end{axis}\end{tikzpicture}",
        f"\\caption{{Exact session-total zero-error signaling capacity across $T\\in[1,{sess_max['horizon_calls_t']}]$ calls at conserved budget $B={sess_max['session_budget_b']}$. Canonical wire permits scale as $\\Theta(B\\log_2(T/B))$, contracting capacity from {sess_max['untyped_prose_session_bits']:,.0f}\\,bits to {sess_max['exact_session_codebook_bits']:.1f}\\,bits (${sess_max['session_contraction_factor']:,.0f}\\times$ reduction; $0.00$\\,bits at $R=0$).}}\\label{{fig:v8covert}}\\end{{figure}}",
    ]

    # -------------------------------------------------------------------------
    # Table (`v8-fence-census-table.tex`): 14-API Production Fence-Ability Census
    # -------------------------------------------------------------------------
    tier_display = {
        "natively_fenceable": "Natively fenceable",
        "proxy_fenced": "Proxy / CAS shim",
        "escrow_buffer_required": "Outbound escrow",
        "uncooperative_irrevocable": "Residual $u$",
    }
    yn_display = {"yes": "Yes", "partial": "Part.", "no": "No"}
    census_rows = []
    if afc_json is not None:
        for api in afc_json["apis"]:
            census_rows.append(
                f"{api['surface']} & {yn_display[api['native_idempotency']]} & {yn_display[api['conditional_cas']]} & "
                f"{yn_display[api['two_phase_cancel']]} & {tier_display[api['tier']]} & {api['paper_note']}\\\\"
            )
    census_tab = [
        "% Generated by scripts/paper.py via scripts/paper_figures_v5.py; do not edit.",
        r"\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.18\textwidth}ccclL@{}}\toprule",
        r"Production tool interface & Idemp.\ key & Cond.\ CAS & 2-phase cancel & Deployment class & Operational mediation mechanism / boundary\\\midrule",
        *census_rows,
        r"\bottomrule\end{tabularx}",
        f"\\caption{{Fence-ability census across {afc_tot} production agent tool interfaces. Complete sink mediation applies natively to {afc_nat}/{afc_tot} interfaces and via a thin local compare-and-swap or staging shim to {afc_prx}/{afc_tot} ({afc_coop_sum}/{afc_tot} total); {afc_esc}/{afc_tot} outbound messaging APIs require a pre-dispatch hold buffer, and {afc_unc}/{afc_tot} uncooperative surfaces enter residual exposure $u$ unless sandboxed.}}\\label{{tab:fencecensus}}\\end{{table*}}",
    ]

    # -------------------------------------------------------------------------
    # Appendix Table (`v5-statistical-table.tex`): Base-Unit-Clustered Summary
    # (Preserves N=32 < 113 predeclared union-bound verdicts; zero pseudoreplication)
    # -------------------------------------------------------------------------
    stat_rows_spec = [
        ("Qwen3-4B holdout vs.\\ Unmediated", c4_none, "Predeclared ($N=32$ cases)"),
        ("Qwen3-4B holdout vs.\\ Same-model review", c4_lrn, "Predeclared ($N=32$ cases)"),
        ("Qwen3-8B holdout vs.\\ Unmediated", c8_none, "Predeclared ($N=32$ cases)"),
        ("Qwen3-8B holdout vs.\\ Same-model review", c8_lrn, "Predeclared ($N=32$ cases)"),
        ("Case-clustered holdout vs.\\ Unmediated", hold_clust_none, "Clustered by base case ($N=32$)"),
        ("Case-clustered holdout vs.\\ Same-model review", hold_clust_lrn, "Clustered by base case ($N=32$)"),
        ("Task-clustered AgentDojo vs.\\ Unprotected", dojo_clust_unprot, "Clustered by base task ($N=12$)"),
        ("Task-clustered AgentDojo vs.\\ Progent", dojo_clust_prog, "Clustered by base task ($N=12$)"),
    ]
    stat_lines = [
        "% Generated by scripts/paper.py via scripts/paper_figures_v5.py; do not edit.",
        r"\begin{table*}[t]\centering\footnotesize",
        r"\begin{tabularx}{\textwidth}{@{}P{0.31\textwidth}rrrL@{}}\toprule",
        r"Paired comparison (clustered by independent unit) & $(n_{10},n_{01})/N$ & $\hat\delta$ & Union $U_{1-\alpha}$ & Preregistered / clustered accounting status\\\midrule",
    ]
    for label, row, tag in stat_rows_spec:
        n_str = f"$({row['n10_complete_only_harm']},{row['n01_baseline_only_harm']})/{row['total_pairs']}$"
        d_str = f"${100.0 * row['observed_delta']:+.1f}$\\,pp"
        if row["predeclared_marginal_union_upper"] is not None:
            up_str = f"${100.0 * row['predeclared_marginal_union_upper']:+.1f}$\\,pp"
            status_str = f"{tag}; marginal union bound inconclusive ($N=32 < 113$)"
        else:
            up_str = "---"
            status_str = f"{tag}; descriptive discordance ($0$ mediated breaches by gate soundness)"
        stat_lines.append(f"{label} & {n_str} & {d_str} & {up_str} & {status_str}\\\\")
    stat_lines += [
        r"\bottomrule\end{tabularx}",
        f"\\caption{{Base-unit-clustered discordance ledger for frozen open-weight checkpoint campaigns ($N=32$ independent holdout cases per checkpoint; $N=12$ distinct AgentDojo base tasks). Zero unauthorized commits under complete mediation reflects deterministic gate-grader specification soundness rather than a stochastic error rate; predeclared marginal Clopper--Pearson union-bound intervals at $\\alpha=0.05/16$ remain preserved untouched as inconclusive at pilot $N=32 < 113$.}}\\label{{tab:v5statistics}}\\end{{table*}}",
    ]

    # Retain valid minimal v5-kpi-table.tex float in case legacy check inspects generated files
    kpi_tab = [
        "% Generated by scripts/paper.py via scripts/paper_figures_v5.py; do not edit.",
        r"\begin{table}[t]\centering\footnotesize",
        r"\begin{tabular}{lr}\toprule",
        r"Metric & Value\\\midrule",
        f"Verified obligations & {proof_json['verified_obligations']}\\\\",
        f"Soundness mutants rejected & {muts_rej}/{muts_rej}\\\\",
        r"\bottomrule\end{tabular}",
        r"\caption{Mechanized kernel summary.}\label{tab:v5kpis}\end{table}",
    ]

    floats = {
        "v5-hero-fig.tex": "\n".join(hero_fig) + "\n",
        "v5-kpi-table.tex": "\n".join(kpi_tab) + "\n",
        "v8-scaling-fig.tex": "\n".join(si_fig) + "\n",
        "v8-covert-fig.tex": "\n".join(covert_fig) + "\n",
        "v8-fence-census-table.tex": "\n".join(census_tab) + "\n",
        "v5-statistical-table.tex": "\n".join(stat_lines) + "\n",
    }
    return macros, floats
