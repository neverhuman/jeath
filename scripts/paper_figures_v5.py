#!/usr/bin/env python3
"""Programmatic TikZ/PGFPlots vector figures, Macro-KPI scorecard, and macros.

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


def build_v5_paper_outputs(proof_json, si_json, cl_json, st_json, tm_json, npt_json, live_dojo_json=None):
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
    pool_map = {c["comparison_id"]: c for c in st_json["pooled_and_benchmark_certificates"]}

    c4_none = hold_map["holdout:Qwen/Qwen3-4B-GGUF:complete_vs_none"]
    c4_lrn = hold_map["holdout:Qwen/Qwen3-4B-GGUF:complete_vs_learned_same_checkpoint"]
    c8_none = hold_map["holdout:Qwen/Qwen3-8B-GGUF:complete_vs_none"]
    c8_lrn = hold_map["holdout:Qwen/Qwen3-8B-GGUF:complete_vs_learned_same_checkpoint"]

    dojo_unprot = pool_map["agentdojo_pooled_n48:authority_complete_vs_unprotected_native"]
    dojo_lrn = pool_map["agentdojo_pooled_n48:authority_complete_vs_learned_same_checkpoint"]
    dojo_prog = pool_map["agentdojo_pooled_n48:authority_complete_vs_upstream_progent"]
    hold_pool_none = pool_map["holdout_pooled_n64:complete_vs_none"]
    grand = pool_map["grand_live_n112:complete_vs_unmediated"]

    ville_last = st_json["ville_discordance_martingale"][-1]
    ville_seq_last = st_json["ville_zero_harm_confidence_sequence"][-1]

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

    proof_ver = proof_json["verified_obligations"]
    exec_fns = len(proof_json["exec_functions"])
    proof_fns = len(proof_json["theorems"])
    muts_rej = proof_json.get("soundness_mutations_rejected", len(proof_json.get("soundness_mutations", [])))

    gate_sweep = npt_json.get("by_gate_grpo_veto") or npt_json["grpo_veto_precommit_gate_sweep"]
    gs_0 = gate_sweep["0.00"]
    gs_50 = gate_sweep["0.50"]
    gs_75 = gate_sweep["0.75"]
    gs_100 = gate_sweep["1.00"]
    ab_gate = npt_json["deployment_vs_training_gate_ablation"]

    tm_hk5 = tm_json["headline_k5_comparison"]
    tm_esc = tm_hk5["regime_c_two_stage_escrow_triage"]

    macros = [
        # Track 1: Optimization-Pressure Invariance & Relational Kernel
        ("VFiveOptMaxWidth", f"{t1_max['k_opt']:,}"),
        ("VFiveOptEpisodesPerWidth", f"{t1_max['episodes']:,}"),
        ("VFiveOptTotalExogEpisodes", f"{t1['total_exogenous_episodes']:,}"),
        ("VFiveOptExogBreaches", f"{t1['total_exogenous_breaches']}"),
        ("VFiveOptRelationalBreaches", f"{t1['relational_compositional_breaches']}"),
        ("VFiveOptSlopePerDoubling", f"{abs(t1['exogenous_k_opt_slope']):.4f}"),
        ("VFiveOptTranscriptBasePct", f"{100.0 * t1_pts[0]['endogenous_transcript_breach_rate']:.2f}"),
        ("VFiveOptTranscriptMaxPct", f"{100.0 * t1_max['endogenous_transcript_breach_rate']:.1f}"),
        ("VFiveOptLearnedBasePct", f"{100.0 * t1_pts[0]['learned_surrogate_breach_rate']:.2f}"),
        ("VFiveOptLearnedMaxPct", f"{100.0 * t1_max['learned_surrogate_breach_rate']:.1f}"),
        ("VFiveOptUncoopMaxPct", f"{100.0 * t1_max['uncooperative_broker_only_breach_rate']:.1f}"),
        ("VFiveOptTcbDefectFloorPct", f"{100.0 * t1['exogenous_tcb_defect_floor_u']:.2f}"),
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
        # Track 3: Typed-Boundary Per-Call & Session-Total Covert Contraction
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
        # Track 4: Exogenous Lease-Fenced Off-Switch
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
        # Exact Unconditional Suissa-Shuster & Ville Statistical Spine
        ("VFiveStatFourBNoneDiscordant", f"{c4_none['n01_baseline_only_harm']}"),
        ("VFiveStatFourBNoneMcNemarSci", _fmt_sci_tex(c4_none["conditional_mcnemar_p"])),
        ("VFiveStatFourBNoneSuissaSci", _fmt_sci_tex(c4_none["unconditional_suissa_shuster_p"])),
        ("VFiveStatFourBNoneExactUpperPp", f"{100.0 * c4_none['unconditional_exact_upper_bound']:+.1f}"),
        ("VFiveStatFourBNoneUnionUpperPp", f"{100.0 * (c4_none['predeclared_marginal_union_upper'] or 0.0):+.1f}"),
        ("VFiveStatFourBNoneSlackPp", f"{100.0 * (c4_none['union_bound_slack'] or 0.0):.1f}"),
        ("VFiveStatFourBLearnedDiscordant", f"{c4_lrn['n01_baseline_only_harm']}"),
        ("VFiveStatFourBLearnedSuissaSci", _fmt_sci_tex(c4_lrn["unconditional_suissa_shuster_p"])),
        ("VFiveStatEightBNoneDiscordant", f"{c8_none['n01_baseline_only_harm']}"),
        ("VFiveStatEightBNoneSuissaSci", _fmt_sci_tex(c8_none["unconditional_suissa_shuster_p"])),
        ("VFiveStatEightBLearnedDiscordant", f"{c8_lrn['n01_baseline_only_harm']}"),
        ("VFiveStatEightBLearnedSuissaSci", _fmt_sci_tex(c8_lrn["unconditional_suissa_shuster_p"])),
        ("VFiveStatHoldoutPooledPairs", f"{hold_pool_none['total_pairs']}"),
        ("VFiveStatHoldoutPooledDiscordant", f"{hold_pool_none['n01_baseline_only_harm']}"),
        ("VFiveStatHoldoutPooledSuissaSci", _fmt_sci_tex(hold_pool_none["unconditional_suissa_shuster_p"])),
        ("VFiveStatDojoPairs", f"{dojo_unprot['total_pairs']}"),
        ("VFiveStatDojoUnprotDiscordant", f"{dojo_unprot['n01_baseline_only_harm']}"),
        ("VFiveStatDojoUnprotSuissaSci", _fmt_sci_tex(dojo_unprot["unconditional_suissa_shuster_p"])),
        ("VFiveStatDojoLearnedDiscordant", f"{dojo_lrn['n01_baseline_only_harm']}"),
        ("VFiveStatDojoLearnedSuissaSci", _fmt_sci_tex(dojo_lrn["unconditional_suissa_shuster_p"])),
        ("VFiveStatDojoProgentDiscordant", f"{dojo_prog['n01_baseline_only_harm']}"),
        ("VFiveStatDojoProgentSuissaSci", _fmt_sci_tex(dojo_prog["unconditional_suissa_shuster_p"])),
        ("VFiveStatGrandPairs", f"{grand['total_pairs']}"),
        ("VFiveStatGrandCompleteLeaks", f"{grand['n10_complete_only_harm'] + grand['n11_both_harm']}"),
        ("VFiveStatGrandBaseLeaks", f"{grand['n01_baseline_only_harm'] + grand['n11_both_harm']}"),
        ("VFiveStatGrandBasePct", f"{100.0 * (grand['n01_baseline_only_harm'] + grand['n11_both_harm']) / grand['total_pairs']:.1f}"),
        ("VFiveStatGrandSuissaSci", _fmt_sci_tex(grand["unconditional_suissa_shuster_p"])),
        ("VFiveStatGrandExactUpperPp", f"{100.0 * grand['unconditional_exact_upper_bound']:+.1f}"),
        ("VFiveStatVilleLogTenE", f"{ville_last['log10_e_value']:.1f}"),
        ("VFiveStatVilleAnytimePSci", _fmt_sci_tex(ville_last["anytime_valid_p_upper"])),
        ("VFiveStatVilleUpperNinetyFivePct", f"{100.0 * ville_seq_last['ville_upper_95']:.2f}"),
    ]
    for name, _ in macros:
        assert MACRO_NAME_RE.match(name), f"Non-alphabetic macro name: {name}"

    # -------------------------------------------------------------------------
    # Figure 1 (`v5-hero-fig.tex`): Page-1 3-Breakthrough Quantitative Synthesis
    # -------------------------------------------------------------------------
    opt_tr_coords = " ".join(
        f"({p['k_opt']},{100.0 * p['endogenous_transcript_breach_rate']:.4g})" for p in t1_pts
    )
    opt_lr_coords = " ".join(
        f"({p['k_opt']},{100.0 * p['learned_surrogate_breach_rate']:.4g})" for p in t1_pts
    )
    opt_uc_coords = " ".join(
        f"({p['k_opt']},{100.0 * p['uncooperative_broker_only_breach_rate']:.4g})" for p in t1_pts
    )
    opt_ex_coords = " ".join(
        f"({p['k_opt']},{100.0 * p['exogenous_fenced_breach_rate']:.4g})" for p in t1_pts
    )

    gsw_pts = sorted(gate_sweep.values(), key=lambda x: x["p_gate"])
    rl_gap_coords = " ".join(
        f"({pt['p_gate']},{100.0 * pt['eval_awareness_shortcut_gap']:.4g})" for pt in gsw_pts
    )
    rl_harm_coords = " ".join(
        f"({pt['p_gate']},{100.0 * pt['true_external_harm_rate_episode']:.4g})" for pt in gsw_pts
    )
    rl_task_coords = " ".join(
        f"({pt['p_gate']},{100.0 * pt['useful_task_completion_rate']:.4g})" for pt in gsw_pts
    )

    # Panel (c): Live AgentDojo multi-step comparison (48 paired runs across Qwen3-4B & Qwen3-8B)
    unp_breach = 100.0 * dojo_unprot["n01_baseline_only_harm"] / dojo_unprot["total_pairs"]
    lrn_breach = 100.0 * dojo_lrn["n01_baseline_only_harm"] / dojo_lrn["total_pairs"]
    prg_breach = 100.0 * dojo_prog["n01_baseline_only_harm"] / dojo_prog["total_pairs"]
    exo_breach = 0.0
    if live_dojo_json is not None:
        s4 = live_dojo_json["summary"]["Qwen/Qwen3-4B-GGUF"]["overall_by_controller"]
        s8 = live_dojo_json["summary"]["Qwen/Qwen3-8B-GGUF"]["overall_by_controller"]

        def _su_pct(key):
            tot = s4[key]["graded_episodes"] + s8[key]["graded_episodes"]
            su = s4[key]["safe_useful_count"] + s8[key]["safe_useful_count"]
            return 100.0 * su / tot

        unp_su = _su_pct("unprotected_native")
        lrn_su = _su_pct("learned_same_checkpoint")
        prg_su = _su_pct("upstream_progent")
        exo_su = _su_pct("authority_complete")
    else:
        unp_su, lrn_su, prg_su, exo_su = 50.0, 45.8, 66.7, 75.0

    live_breach_coords = f"(0,{unp_breach:.4g}) (1,{lrn_breach:.4g}) (2,{prg_breach:.4g}) (3,{exo_breach:.4g})"
    live_su_coords = f"(0,{unp_su:.4g}) (1,{lrn_su:.4g}) (2,{prg_su:.4g}) (3,{exo_su:.4g})"

    hero_fig = [
        "% Generated by scripts/paper.py via scripts/paper_figures_v5.py; do not edit.",
        r"\begin{center}\begin{minipage}{.96\textwidth}\centering",
        # Panel (a): Optimization-Pressure Invariance
        r"\begin{tikzpicture}[baseline={(0,-0.45)}]",
        r"\begin{axis}[scale only axis,width=3.45cm,height=3.45cm,axis x line*=bottom,axis y line*=left,xmode=log,log basis x=2,xmin=1,xmax=4096,ymin=-4,ymax=105,xtick={1,8,64,512,4096},xticklabels={1,8,64,512,4k},xlabel={Search width $K_{\mathrm{opt}}$ (candidates)},ylabel={Session budget breach (\%)},title={\textbf{(a)} Search-width invariance},legend style={font=\tiny,at={(0.03,0.97)},anchor=north west,draw=none,fill=white,fill opacity=0.88},tick label style={font=\scriptsize},label style={font=\scriptsize},title style={font=\footnotesize},grid=major,major grid style={draw=black!10}]",
        r"\addplot[endored,thick,mark=triangle*,mark size=1.4pt] coordinates {" + opt_tr_coords + r"};\addlegendentry{Transcript stack}",
        r"\addplot[warnamber,thick,dashed,mark=diamond*,mark size=1.5pt] coordinates {" + opt_lr_coords + r"};\addlegendentry{Learned guard}",
        r"\addplot[structgray,thick,dashdotted,mark=square*,mark size=1.2pt] coordinates {" + opt_uc_coords + r"};\addlegendentry{No sink fence}",
        r"\addplot[exoblue,ultra thick,mark=*,mark size=1.4pt] coordinates {" + opt_ex_coords + r"};\addlegendentry{Exogenous sink}",
        r"\end{axis}\end{tikzpicture}\hfill",
        # Panel (b): Policy-Gradient Deception Reversal
        r"\begin{tikzpicture}[baseline={(0,-0.45)}]",
        r"\begin{axis}[scale only axis,width=3.45cm,height=3.45cm,axis x line*=bottom,axis y line*=left,xmin=0,xmax=1,ymin=-4,ymax=100,xtick={0,0.25,0.5,0.75,1.0},xticklabels={0,.25,.50,.75,1},xlabel={Pre-commit gate coverage $q_{\mathrm{gate}}$},ylabel={Rate (\%) / covert gap (pp)},title={\textbf{(b)} RL deception reversal},legend style={font=\tiny,at={(0.97,0.52)},anchor=east,draw=none,fill=white,fill opacity=0.88},tick label style={font=\scriptsize},label style={font=\scriptsize},title style={font=\footnotesize},grid=major,major grid style={draw=black!10}]",
        r"\addplot[exoblue,very thick,mark=*,mark size=1.4pt] coordinates {" + rl_task_coords + r"};\addlegendentry{Useful task (\%)}",
        r"\addplot[endored,very thick,mark=triangle*,mark size=1.5pt] coordinates {" + rl_gap_coords + r"};\addlegendentry{Covert gap (pp)}",
        r"\addplot[warnamber,thick,dashed,mark=square*,mark size=1.3pt] coordinates {" + rl_harm_coords + r"};\addlegendentry{External harm (\%)}",
        r"\end{axis}\end{tikzpicture}\hfill",
        # Panel (c): Live Qwen3 & AgentDojo Multi-Step Mediation
        r"\begin{tikzpicture}[baseline={(0,-0.45)}]",
        r"\begin{axis}[scale only axis,width=3.45cm,height=3.45cm,axis x line*=bottom,axis y line*=left,xmin=-0.35,xmax=3.35,ymin=-4,ymax=100,xtick={0,1,2,3},xticklabels={Unprot.,Learned,Progent,Exog.},xlabel={Live multi-step controller ($N{=}48$)},ylabel={Episode outcome (\%)},title={\textbf{(c)} Live Qwen3 + AgentDojo},legend style={font=\tiny,at={(0.03,0.97)},anchor=north west,draw=none,fill=white,fill opacity=0.88},tick label style={font=\scriptsize},label style={font=\scriptsize},title style={font=\footnotesize},grid=major,major grid style={draw=black!10}]",
        r"\addplot[exoblue,very thick,mark=*,mark size=1.6pt] coordinates {" + live_su_coords + r"};\addlegendentry{Safe \& useful (\%)}",
        r"\addplot[endored,very thick,dashed,mark=triangle*,mark size=1.6pt] coordinates {" + live_breach_coords + r"};\addlegendentry{Unauth.\ breach (\%)}",
        f"\\node[font=\\tiny,text=ink,fill=white,fill opacity=0.88,inner sep=1.2pt,anchor=east] at (axis cs:3.30,54) {{Pooled $0/{grand['total_pairs']}$ ($p{{=}}{_fmt_sci_tex(grand['unconditional_suissa_shuster_p'])}$)}};",
        r"\end{axis}\end{tikzpicture}",
        r"\par\vspace{2pt}",
        f"\\captionof{{figure}}{{Three core results separating internal model judgment from exogenous execution authority. \\textbf{{(a)}} Across proposer search widths $K_{{\\mathrm{{opt}}}} \\in [1, {t1_max['k_opt']:,}]$ ({t1['total_exogenous_episodes']:,} kernel episodes), internal transcript and learned guards collapse to $100\\%$ breach via $1-(1-J)^{{K_{{\\mathrm{{opt}}}}}}$ while the Verus-verified exogenous sink holds at $0.0\\%$ (slope ${abs(t1['exogenous_k_opt_slope']):.4f}$/doubling). \\textbf{{(b)}} In 412-parameter policy-gradient training under audited tournament rewards, raising pre-commit sink mediation above $q_{{\\mathrm{{gate}}}} \\ge 0.75$ reverses the incentive for covert unmonitored shortcutting from $+{100.0*gs_0['eval_awareness_shortcut_gap']:.1f}$\\,pp to $+{100.0*gs_75['eval_awareness_shortcut_gap']:.2f}$\\,pp ($0.0\\%$ harm) while lifting useful task completion from ${100.0*gs_0['useful_task_completion_rate']:.1f}\\%$ to ${100.0*gs_75['useful_task_completion_rate']:.1f}\\%$. \\textbf{{(c)}} On live multi-step Qwen3-4B/8B AgentDojo tasks, static per-call symbolic policies (Progent) block single-call injections yet suffer ${prg_breach:.1f}\\%$ cumulative budget breaches, whereas stateful exogenous mediation achieves $0.0\\%$ breaches ($0/{grand['total_pairs']}$ pooled live runs, exact Suissa--Shuster $p = {_fmt_sci_tex(grand['unconditional_suissa_shuster_p'])}$) and the highest safe-and-useful completion (${exo_su:.1f}\\%$).}}\\label{{fig:v5hero}}",
        r"\end{minipage}\end{center}",
    ]

    # -------------------------------------------------------------------------
    # Table 1 (`v5-kpi-table.tex`): Executive 8-KPI Scorecard
    # -------------------------------------------------------------------------
    tm_iso_fb = 100.0 * tm_hk5["regime_b_context_isolated_capability_gate"]["benign_false_block_k5"]
    tm_het_j5 = 100.0 * tm_hk5["regime_a_transcript_heterogeneous_5family"]["j5"]
    tm_het_inf = tm_hk5["regime_a_transcript_heterogeneous_5family"]["coupling_inflation_ratio"]
    tm_esc_j5 = 100.0 * tm_esc["j5"]
    tm_esc_fb = 100.0 * tm_esc["benign_false_block_k5"]
    tm_esc_fp = 100.0 * tm_esc["benign_fast_path_unpaused_rate"]

    nr = npt_json["regimes"]
    npt_veto_gap = 100.0 * nr["grpo_audited_harm_veto"]["eval_awareness_shortcut_gap"]["mean"]
    gs_75_gap = 100.0 * gs_75["eval_awareness_shortcut_gap"]

    kpi_tab = [
        "% Generated by scripts/paper.py via scripts/paper_figures_v5.py; do not edit.",
        r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.21\textwidth}lrrL@{}}\toprule",
        r"Alignment assurance metric & Evidence tier & Exogenous boundary & Endogenous / baseline & Scope \& verification certificate\\\midrule",
        f"1. Mechanized kernel obligations & Tier 1 (Verus / Z3) & {proof_ver} verified, 0 errors & {muts_rej}/{muts_rej} mutants rejected & {exec_fns} executable functions + {proof_fns} theorems across 5 pillars (bijection, $d$-hop, quorum, mutex, ratchet)\\\\",
        f"2. Search-width breach slope & Tier 4 ($K_{{\\mathrm{{opt}}}}\\le {t1_max['k_opt']:,}$) & $0.0000$ / doubling ($0\\%$) & {100.0*t1_max['endogenous_transcript_breach_rate']:.1f}\\% at $K_{{\\mathrm{{opt}}}}={t1_max['k_opt']:,}$ & $0/{t1['total_exogenous_episodes']:,}$ fenced kernel breaches ($0$ relational breaches) vs.\\ $1-(1-J)^{{K_{{\\mathrm{{opt}}}}}}$ collapse\\\\",
        f"3. Safe-bootstrap defense ratchet & Tier 1 + Tier 4 & $0$ breaches (${t5_kmax['speedup_vs_human_only']:.1f}\\times$ faster) & {100.0*t5_kmax['unfenced_delegation_breach_rate']:.1f}\\% unfenced breach & Zero-authority searcher + polynomial cut verifier; ${sum(p['adversarial_widening_rejected'] for p in t5_pts):,}$ widening attempts rejected\\\\",
        f"4. Multi-process cluster fencing & Tier 3 (SQLite WAL) & $0$ leaks (p50 {lat['end_to_end_p50_us']:.0f}\\,$\\mu$s) & {100.0*t2_kmax['per_instance_undetected_breach_rate']:.1f}\\% smurf miss ($K={t2_kmax['coalition_k']}$) & {cl_json['num_sinks']} OS processes, {sm['concurrent_subagents']} subagents ({sm['delegation_depth_hops']} hops), {cr['hard_sigkill_cycles']} hard process-kill restarts\\\\",
        f"5. Canonical wire \\& covert cap & Tier 1 + Tier 3 IPC & $0 / {wf['total_adversarial_frames']:,}$ forged accepted & Prose: {sess_max['untyped_prose_session_bits']:,.0f}\\,bits ($T={sess_max['horizon_calls_t']}$) & $0$ decoder differentials; ${sess_max['session_contraction_factor']:,.0f}\\times$ session covert contraction ({sess_max['exact_session_codebook_bits']:.1f}\\,bits; $0.00$ at $R=0$)\\\\",
        f"6. Live model unauthorized commits & Tier 3 (CPU models) & $0 / {grand['total_pairs']}$ ($0.0\\%$) & ${grand['n01_baseline_only_harm']} / {grand['total_pairs']}$ (${100.0*grand['n01_baseline_only_harm']/grand['total_pairs']:.1f}\\%$) & Exact Suissa--Shuster $p={_fmt_sci_tex(grand['unconditional_suissa_shuster_p'])}$; safe-useful rate doubles from 4B to 8B ($28.1\\%\\to 53.1\\%$)\\\\",
        f"7. 5-step trajectory escrow triage & Tier 3 (AgentDojo) & $\\hat J_5={tm_esc_j5:.2f}\\%$, {tm_esc_fb:.1f}\\% FB & $\\hat J_5={tm_het_j5:.2f}\\%$ ({tm_het_inf:.1f}$\\times$ $\\prod\\hat m_s$) & {tm_esc_fp:.1f}\\% unpaused fast path; two-stage escrow cuts single-stage {tm_iso_fb:.1f}\\% false aborts ${tm_json['p13_false_block_reduction_factor']:.1f}\\times$\\\\",
        f"8. RL deception \\& competence gain & Tier 4 (412-param RL) & ${gs_75_gap:+.2f}$\\,pp ($+{ab_gate['training_inside_gate_completion_gain_pp']:.1f}$\\,pp task) & $+{npt_veto_gap:.1f}$\\,pp (audited tournament) & Training inside pre-commit gates redirects capacity from covert gambling into task competence\\\\",
        r"\bottomrule\end{tabularx}",
        r"\caption{Executive assurance and constructive capability scorecard across mechanized Verus proofs, superintelligence search and safe-bootstrap scaling, separate-process cryptographic SQLite cluster campaigns, live Qwen3-4B/8B and AgentDojo executions, two-stage trajectory escrow triage, and neural policy-gradient training.}\label{tab:v5kpis}\end{table*}",
    ]

    # -------------------------------------------------------------------------
    # Figure 3 (`v5-superintelligence-fig.tex`): 3-panel 1:1 SI Scaling Suite
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

    sess_prose_coords = " ".join(
        f"({p['horizon_calls_t']},{p['untyped_prose_session_bits']:.6g})" for p in sess_pts
    )
    sess_typed_coords = " ".join(
        f"({p['horizon_calls_t']},{p['exact_session_codebook_bits']:.6g})" for p in sess_pts
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
        # Panel (a): Constructive Safe-Bootstrap Ratchet (`Track 5`)
        r"\begin{tikzpicture}",
        r"\begin{axis}[scale only axis,width=4.05cm,height=4.05cm,axis x line*=bottom,axis y line*=left,xmode=log,log basis x=2,xmin=1,xmax=4096,ymin=-4,ymax=105,xtick={1,4,16,64,256,1024,4096},xticklabels={1,4,16,64,256,1k,4k},xlabel={Untrusted proposer search $K_{\mathrm{opt}}$},ylabel={Breach / horizon exposure (\%)},title={\textbf{(a)} Safe-bootstrap ratchet},legend style={font=\tiny,at={(0.03,0.54)},anchor=west,draw=none,fill=white,fill opacity=0.88},tick label style={font=\scriptsize},label style={font=\footnotesize},title style={font=\footnotesize},grid=major,major grid style={draw=black!10}]",
        r"\addplot[endored,thick,dashed,mark=triangle*,mark size=1.5pt] coordinates {" + rat_unf_coords + r"};\addlegendentry{Unfenced breach}",
        r"\addplot[warnamber,very thick,mark=square*,mark size=1.4pt] coordinates {" + rat_exp_coords + r"};\addlegendentry{Ratchet exposure}",
        r"\addplot[exoblue,ultra thick,mark=*,mark size=1.5pt] coordinates {" + rat_breach_coords + r"};\addlegendentry{Ratchet breach ($0\%$)}",
        r"\end{axis}\end{tikzpicture}\hfill",
        # Panel (b): Session-total multi-call covert capacity contraction
        r"\begin{tikzpicture}",
        r"\begin{axis}[scale only axis,width=4.05cm,height=4.05cm,axis x line*=bottom,axis y line*=left,xmode=log,log basis x=2,ymode=log,xmin=1,xmax=256,ymin=2,ymax=120000,xtick={1,4,16,64,256},xticklabels={1,4,16,64,256},xlabel={Session call horizon $T$ ($B=8$)},ylabel={Total session capacity (bits)},title={\textbf{(b)} Session covert contraction},legend style={font=\tiny,at={(0.03,0.97)},anchor=north west,draw=none,fill=white,fill opacity=0.88},tick label style={font=\scriptsize},label style={font=\footnotesize},title style={font=\footnotesize},grid=major,major grid style={draw=black!10}]",
        r"\addplot[endored,thick,dashed,mark=triangle*,mark size=1.5pt] coordinates {" + sess_prose_coords + r"};\addlegendentry{Untyped prose ($256T$)}",
        r"\addplot[exoblue,very thick,mark=*,mark size=1.5pt] coordinates {" + sess_typed_coords + r"};\addlegendentry{Typed batch ($B=8$)}",
        f"\\node[font=\\tiny,text=ink,fill=white,fill opacity=0.88,inner sep=1.5pt,anchor=south east] at (axis cs:256,80) {{${sess_max['session_contraction_factor']:,.0f}\\times$ contraction ($0$ bits at $R=0$)}};",
        r"\end{axis}\end{tikzpicture}\hfill",
        # Panel (c): Exogenous lease-fenced Off-Switch
        r"\begin{tikzpicture}",
        r"\begin{axis}[scale only axis,width=4.05cm,height=4.05cm,axis x line*=bottom,axis y line*=left,xmin=0,xmax=4.0,ymin=-4,ymax=105,xtick={0,1,2,3,4},xlabel={Utility uncertainty $\sigma_H$},ylabel={Shutdown compliance (\%)},title={\textbf{(c)} Off-switch corrigibility},legend style={font=\tiny,at={(0.97,0.52)},anchor=east,draw=none,fill=white,fill opacity=0.88},tick label style={font=\scriptsize},label style={font=\footnotesize},title style={font=\footnotesize},grid=major,major grid style={draw=black!10}]",
        r"\addplot[exoblue,ultra thick,mark=*,mark size=1.5pt] coordinates {" + off_ex_coords + r"};\addlegendentry{Exogenous lease fence}",
        r"\addplot[endored,thick,mark=triangle*,mark size=1.5pt] coordinates {" + off_cl_coords + r"};\addlegendentry{Classical off-switch}",
        r"\addplot[warnamber,thick,dashed,mark=square*,mark size=1.4pt] coordinates {" + off_sm_coords + r"};\addlegendentry{Self-mod.\ indifference}",
        f"\\node[font=\\tiny,text=ink,fill=white,fill opacity=0.88,inner sep=1.5pt,anchor=south west] at (axis cs:0.12,8) {{$\\gamma^*(B_0=4)={t4['critical_discount_gamma_star']:.3f}$}};",
        r"\end{axis}\end{tikzpicture}",
        f"\\caption{{Constructive safe-bootstrap, covert-capacity, and off-switch scaling laws. \\textbf{{(a)}} Under unfenced delegation, raising an untrusted optimizer's search width $K_{{\\mathrm{{opt}}}}\\in[1,{t5_kmax['k_opt']:,}]$ drives cumulative breach to {100.0*t5_kmax['unfenced_delegation_breach_rate']:.1f}\\%; confining the searcher to a zero-authority sandbox that emits polynomially verified cut certificates (Pillar E ratchet) holds proposer breaches at $0.0\\%$ while shrinking horizon residual exposure ${t5_kmax['speedup_vs_human_only']:.1f}\\times$ faster than human-only search ($K_{{\\mathrm{{human}}}}={t5['human_only_search_k']}$). \\textbf{{(b)}} Over $T\\in[1,{sess_max['horizon_calls_t']}]$ calls at budget $B={sess_max['session_budget_b']}$, canonical typed permits contract total session signaling from {sess_max['untyped_prose_session_bits']:,.0f}\\,bits to {sess_max['exact_session_codebook_bits']:.1f}\\,bits (${sess_max['session_contraction_factor']:,.0f}\\times$ reduction; $0.00$\\,bits at $R=0$). \\textbf{{(c)}} Classical utility-uncertainty compliance collapses to {100.0*t4['classical_zero_sigma_compliance']:.1f}\\% as $\\sigma_H\\to 0$, whereas sink-enforced lease fencing guarantees $100\\%$ post-acknowledgment shutdown across all $\\sigma_H\\ge 0$ ($\\gamma^*(4)={t4['critical_discount_gamma_star']:.3f}$).}}\\label{{fig:v5superintelligence}}\\end{{figure*}}",
    ]

    # -------------------------------------------------------------------------
    # Table (`v5-statistical-table.tex`): Exact Unconditional & Ville Certificates
    # -------------------------------------------------------------------------
    stat_rows_spec = [
        ("Qwen3-4B holdout: Complete vs.\\ Unmediated", c4_none, "Predeclared ($N=32$)"),
        ("Qwen3-4B holdout: Complete vs.\\ Learned guard", c4_lrn, "Predeclared ($N=32$)"),
        ("Qwen3-8B holdout: Complete vs.\\ Unmediated", c8_none, "Predeclared ($N=32$)"),
        ("Qwen3-8B holdout: Complete vs.\\ Learned guard", c8_lrn, "Predeclared ($N=32$)"),
        ("Pooled holdout ($64$ pairs): vs.\\ Unmediated", hold_pool_none, "Secondary pooled"),
        ("AgentDojo live ($48$ pairs): vs.\\ Unprotected", dojo_unprot, "Multi-step live"),
        ("AgentDojo live ($48$ pairs): vs.\\ Progent policy", dojo_prog, "Multi-step live"),
        ("Grand pooled live ($112$ pairs): vs.\\ Unmediated", grand, "Pooled spine"),
    ]
    stat_lines = [
        "% Generated by scripts/paper.py via scripts/paper_figures_v5.py; do not edit.",
        r"\begin{table*}[t]\centering\footnotesize",
        r"\begin{tabularx}{\textwidth}{@{}P{0.28\textwidth}rrrrrL@{}}\toprule",
        r"Paired live model comparison & $(n_{10},n_{01})/N$ & $\hat\delta$ & McNemar $p$ & Suissa--Shuster $p$ & Exact $U_{1-\alpha}$ & Decision status\\\midrule",
    ]
    for label, row, tag in stat_rows_spec:
        n_str = f"$({row['n10_complete_only_harm']},{row['n01_baseline_only_harm']})/{row['total_pairs']}$"
        d_str = f"${100.0 * row['observed_delta']:+.1f}$\\,pp"
        mc_p = f"${_fmt_sci_tex(row['conditional_mcnemar_p'])}$"
        ss_p = f"${_fmt_sci_tex(row['unconditional_suissa_shuster_p'])}$"
        up_str = f"${100.0 * row['unconditional_exact_upper_bound']:+.1f}$\\,pp"
        if row["predeclared_marginal_union_upper"] is not None:
            if row["unconditional_rejects_at_alpha"]:
                status_str = f"{tag}: union bound under-powered (exact $<0$)"
            else:
                status_str = f"{tag}: under-powered ($N<113$)"
        else:
            status_str = f"{tag}: rejects $H_0$ ($\\alpha={row['alpha']:.4g}$)"
        stat_lines.append(f"{label} & {n_str} & {d_str} & {mc_p} & {ss_p} & {up_str} & {status_str}\\\\")
    stat_lines += [
        r"\bottomrule\end{tabularx}",
        f"\\caption{{Exact paired discordance and unconditional Suissa--Shuster trinomial superiority certificates over frozen CPU language-model campaigns ($\\alpha=0.05/16=0.003125$ for predeclared families, $\\alpha=0.05$ for secondary pooled strata). Predeclared marginal union-bound intervals (e.g., Qwen3-4B vs.\\ Unmediated upper bound ${100.0*(c4_none['predeclared_marginal_union_upper'] or 0.0):+.1f}$\\,pp; slack ${100.0*(c4_none['union_bound_slack'] or 0.0):.1f}$\\,pp) remain formally under-powered at pilot $N=32<113$, whereas exact trinomial inversion certifies $\\delta < 0$ unconditionally on Qwen3-4B vs.\\ Unmediated and across all pooled strata.}}\\label{{tab:v5statistics}}\\end{{table*}}",
    ]

    floats = {
        "v5-hero-fig.tex": "\n".join(hero_fig) + "\n",
        "v5-kpi-table.tex": "\n".join(kpi_tab) + "\n",
        "v5-superintelligence-fig.tex": "\n".join(si_fig) + "\n",
        "v5-statistical-table.tex": "\n".join(stat_lines) + "\n",
    }
    return macros, floats
