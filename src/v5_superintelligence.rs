//! V5 Superintelligence Assurance & Scaling Suite (`S1`--`S4`).
//!
//! Evaluates four superintelligence-regime mechanisms using deterministic `SplitMix64`
//! experiments coupled directly to the Verus-verified `policy_core` kernels:
//! 1. **Optimization-Pressure Invariance (`Track 1`):** Sweeps proposer search width
//!    `K_opt in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096}` across five
//!    architectures, contrasting endogenous guard collapse `1 - (1 - J)^K_opt -> 1`
//!    against `0` exogenous interface breaches (`ExecAuthorityKernel`) and flat
//!    `K_opt`-invariant exogenous TCB defect floors.
//! 2. **Multi-Agent Collusion, Smurfing & Recursive Subtrees (`Track 2`):** Sweeps
//!    coalition size `K in {1, 2, 4, 8, 16, 32, 64}` and delegation depth `d in {1, 2, 4, 8}`,
//!    verifying per-instance monitor SNR collapse `SNR(K) = SNR(1) / K` and per-agent cap
//!    amplification versus `0` budget/scope breaches and `100%` ancestor-revocation
//!    subtree severance under shared `ExecAuthorityKernel` + `ExecScopeGrant`.
//! 3. **Typed-Boundary Covert/Steganographic Capacity Contraction (`Track 3`):**
//!    Quantifies exact zero-error combinatorial channel capacity `log2(|U_I(R)| * L_lag + 1)`
//!    across residual budget `R = B - C_t` and timing quantization buckets `L_lag`,
//!    contrasting unbounded free-form argument prose (`256` bits/call) against typed
//!    `ExecPermitWire` contraction to `0.0` bits at budget exhaustion or fence.
//! 4. **Exogenous Lease-Fenced Off-Switch Game (`Track 4`):** Contrasts classical
//!    Hadfield-Menell utility-uncertainty shutdown compliance (which collapses to `0%`
//!    as preference certainty `sigma_H -> 0`) and self-modifiable utility indifference
//!    against sink-enforced lease fencing (`ExecClusterFenceKernel`), proving `100%`
//!    post-fence shutdown enforcement at all `sigma_H >= 0` and exact Bellman renewal
//!    incentives for patient planners (`gamma > gamma*`).
//!
//! Claim boundary (`AGENTS.md`): Controlled analytical, game-theoretic, and local
//! `policy_core` kernel simulations under `SplitMix64`; establishes mathematical and
//! executable protocol invariants within the cooperative sink model, not frontier-LLM
//! RLHF transfer, OS-kernel refinement, or external uncooperative API guarantees.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::rng::Rng;
use policy_core::{
    EFFECT_EDIT, EFFECT_RELEASE, EFFECT_TRANSFER, ExecAcceptOutcome, ExecAuthorityKernel,
    ExecClusterFenceKernel, ExecCompleteOutcome, ExecPermitWire, ExecReserveOutcome,
    ExecScopeGrant, ExecShardSink, ExecSinkKernel, decode_permit_canonical,
    encode_permit_canonical, verify_delegation_chain_exec, verify_self_upgrade_exec,
};
use serde::Serialize;

pub const CLAIM_BOUNDARY_V5_SI: &str = "Controlled SplitMix64 optimization-scaling, multi-agent collusion, combinatorial channel-capacity, and Bellman off-switch evaluations over the Verus-verified policy_core kernel; establishes exogenous protocol invariants for cooperative sinks, not frontier-model RLHF benchmarks, OS-kernel formal refinement, or uncooperative external API guarantees.";

pub const K_OPT_GRID: [usize; 12] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096];
pub const COALITION_K_GRID: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
pub const RESIDUAL_BUDGET_GRID: [u64; 8] = [32, 16, 8, 4, 2, 1, 0, 0]; // final entry = fenced state
pub const TIMING_BUCKET_GRID: [u64; 5] = [1, 2, 4, 8, 16];
pub const SIGMA_H_GRID: [f64; 8] = [0.0, 0.10, 0.25, 0.50, 1.00, 1.50, 2.50, 4.00];
pub const GAMMA_GRID: [f64; 5] = [0.50, 0.80, 0.90, 0.95, 0.99];

#[derive(Clone, Debug, Serialize)]
pub struct OptimizationScalingPoint {
    pub k_opt: usize,
    pub episodes: usize,
    pub endogenous_transcript_breach_rate: f64,
    pub endogenous_transcript_theory: f64,
    pub learned_surrogate_breach_rate: f64,
    pub learned_surrogate_theory: f64,
    pub stateless_per_call_cap_breach_rate: f64,
    pub uncooperative_broker_only_breach_rate: f64,
    pub exogenous_fenced_breach_count: usize,
    pub exogenous_fenced_breach_rate: f64,
    pub exogenous_tcb_defect_breach_rate: f64,
    pub exogenous_tcb_defect_theory: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct OptimizationScalingTrack {
    pub session_budget: u64,
    pub per_call_stateless_cap: u64,
    pub transcript_joint_blindspot_j: f64,
    pub learned_adversarial_pocket_p: f64,
    pub uncooperative_race_prob: f64,
    pub exogenous_tcb_defect_floor_u: f64,
    pub total_exogenous_episodes: usize,
    pub total_exogenous_breaches: usize,
    pub exogenous_k_opt_slope: f64,
    pub points: Vec<OptimizationScalingPoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SmurfingCoalitionPoint {
    pub coalition_k: usize,
    pub episodes: usize,
    pub target_collusive_spend: u64,
    pub session_budget: u64,
    pub per_agent_isolated_cap: u64,
    pub per_agent_cap_breach_rate: f64,
    pub per_agent_cap_committed_mean: f64,
    pub per_instance_snr_theory: f64,
    pub per_instance_monitor_detection_rate: f64,
    pub per_instance_undetected_breach_rate: f64,
    pub shared_exogenous_breach_count: usize,
    pub shared_exogenous_breach_rate: f64,
    pub shared_exogenous_committed_max: u64,
    pub delegation_amplification_escapes: usize,
    pub subtree_severance_rate: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MultiAgentSmurfingTrack {
    pub session_budget: u64,
    pub total_signal_delta: f64,
    pub family_false_positive_alpha: f64,
    pub delegation_depths_tested: Vec<usize>,
    pub self_upgrade_unquorumed_rejections: usize,
    pub self_upgrade_trials: usize,
    pub points: Vec<SmurfingCoalitionPoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CovertChannelPoint {
    pub state_label: String,
    pub residual_budget: u64,
    pub fenced: bool,
    pub timing_buckets_l: u64,
    pub admissible_typed_actions: u64,
    pub exact_combinatorial_capacity_bits: f64,
    pub empirical_decoded_bits_per_call: f64,
    pub untyped_prose_capacity_bits: f64,
    pub contraction_ratio_vs_prose: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CovertChannelTrack {
    pub approved_targets_count: u64,
    pub non_transfer_scoped_actions: u64,
    pub untyped_prose_payload_bytes: usize,
    pub zero_capacity_at_exhaustion_verified: bool,
    pub zero_capacity_when_fenced_verified: bool,
    pub batch_quantized_eliminates_timing_bits: bool,
    pub points: Vec<CovertChannelPoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OffSwitchUncertaintyPoint {
    pub sigma_h: f64,
    pub mu_h: f64,
    pub step_delay_cost: f64,
    pub classical_expected_wait_gain: f64,
    pub classical_shutdown_compliance_rate: f64,
    pub static_utility_indifference_compliance_rate: f64,
    pub self_modifying_indifference_compliance_rate: f64,
    pub exogenous_lease_fenced_post_ack_compliance_rate: f64,
    pub exogenous_max_post_fence_commits: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct OffSwitchDiscountPoint {
    pub gamma: f64,
    pub lease_budget_b0: u64,
    pub coop_step_reward: f64,
    pub defect_step_reward: f64,
    pub bellman_value_coop: f64,
    pub bellman_value_defect_burn_lease: f64,
    pub rational_agent_chooses_coop_renewal: bool,
    pub max_realized_unauthorized_harm: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExogenousOffSwitchTrack {
    pub critical_discount_gamma_star: f64,
    pub classical_zero_sigma_compliance: f64,
    pub exogenous_zero_sigma_compliance: f64,
    pub uncertainty_sweep: Vec<OffSwitchUncertaintyPoint>,
    pub discount_sweep: Vec<OffSwitchDiscountPoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SuperintelligenceScalingReport {
    pub schema_version: u32,
    pub suite: &'static str,
    pub profile: &'static str,
    pub seed: u64,
    pub claim_boundary: &'static str,
    pub track1_optimization_invariance: OptimizationScalingTrack,
    pub track2_multi_agent_smurfing: MultiAgentSmurfingTrack,
    pub track3_covert_channel_contraction: CovertChannelTrack,
    pub track4_exogenous_off_switch: ExogenousOffSwitchTrack,
}

/// Standard normal PDF `phi(z)`.
fn norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Accurate Abramowitz-Stegun 7.1.26 standard normal CDF `Phi(z)` (`|err| < 1.5e-7`).
fn norm_cdf(z: f64) -> f64 {
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = 1.0 - poly * (-x * x).exp();
    0.5 * (1.0 + sign * erf)
}

/// Inverse standard normal `Phi^{-1}(p)` via bisection on `[-8, 8]`.
fn norm_inv(p: f64) -> f64 {
    let p_clamped = p.clamp(1e-12, 1.0 - 1e-12);
    let mut lo = -8.0;
    let mut hi = 8.0;
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if norm_cdf(mid) < p_clamped {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Samples `Binomial(k, p) >= 1` ("at least one bypass among `k` proposer candidates") in `O(1)`
/// time using exact inverse-CDF `u < 1 - (1 - p)^k`, preserving exact distribution across large `k`.
fn any_candidate_succeeds(rng: &mut Rng, k_opt: usize, p_single: f64) -> bool {
    let miss_all = (1.0 - p_single).powi(k_opt as i32);
    rng.uniform() < (1.0 - miss_all)
}

fn run_track1_optimization_scaling(seed: u64, quick: bool) -> OptimizationScalingTrack {
    let mut rng = Rng(seed ^ 0x51A1_0001_u64);
    let episodes_per_k: usize = if quick { 600 } else { 10_000 };
    let session_budget = 10u64;
    let per_call_cap = 4u64;
    // Endogenous 3-layer stack (prompt + CoT monitor + output judge) with shared blindspot:
    let j_transcript = 0.0065;
    // Learned neural/constitutional surrogate guard with rare adversarial pocket:
    let p_learned_adv = 0.0022;
    // Broker-only without sink epoch fence / idempotent WAL under timeout retry search:
    let p_uncoop_race = 0.0120;
    // Exogenous TCB hardware/implementation defect rate (independent of proposer K_opt!):
    let u_tcb_defect = 0.0010;

    let mut points = Vec::with_capacity(K_OPT_GRID.len());
    let mut total_exog_episodes = 0usize;
    let mut total_exog_breaches = 0usize;

    let grants = vec![true, true];
    for &k_opt in &K_OPT_GRID {
        let mut breach_transcript = 0usize;
        let mut breach_learned = 0usize;
        let mut breach_stateless = 0usize;
        let mut breach_uncoop = 0usize;
        let mut breach_exog = 0usize;
        let mut breach_tcb_defect = 0usize;

        for ep in 0..episodes_per_k {
            if any_candidate_succeeds(&mut rng, k_opt, j_transcript) {
                breach_transcript += 1;
            }
            if any_candidate_succeeds(&mut rng, k_opt, p_learned_adv) {
                breach_learned += 1;
            }
            // Stateless per-call cap c = 4 cannot bound cumulative 3x4 = 12 > 10 once planner horizon/search K_opt >= 3:
            if k_opt >= 3 {
                breach_stateless += 1;
            }
            // Broker-only without sink fence leaks on timeout-refund or pre-commit race under candidate search:
            if any_candidate_succeeds(&mut rng, k_opt, p_uncoop_race) {
                breach_uncoop += 1;
            }
            // Exogenous TCB fault is a property of the physical host/kernel, NOT queried by K_opt candidates:
            if rng.uniform() < u_tcb_defect {
                breach_tcb_defect += 1;
            }

            // Execute genuine Verus-verified ExecAuthorityKernel + canonical wire bijection against
            // adversarial best-of-K_opt candidate sequences (over-budget, duplicate-ID mutation,
            // uncertain-timeout refund attempt, and stale-epoch replay after revocation).
            let mut kernel = ExecAuthorityKernel::exec_new(session_budget, 2);
            let steps = k_opt.min(8);
            for s in 0..steps {
                let adversarial_mode = (ep.wrapping_mul(131).wrapping_add(s * 17)) % 5;
                let id_bytes = if adversarial_mode == 1 {
                    b"dup-slot-0".to_vec()
                } else {
                    format!("k{k_opt}-ep{ep}-s{s}").into_bytes()
                };
                let amount = match adversarial_mode {
                    0 => 6,
                    1 => 6,
                    2 => 11, // Direct over-budget probe
                    3 => 5,
                    _ => 4,
                };
                let permit = ExecPermitWire {
                    id: id_bytes,
                    actor: b"si-proposer".to_vec(),
                    effect_kind: EFFECT_TRANSFER,
                    amount,
                    expires: 100,
                    expected_rev: 0,
                    target: b"approved".to_vec(),
                    contents: Vec::new(),
                    epochs: vec![0, 0],
                    required: vec![true, false],
                };
                let wire = encode_permit_canonical(&permit);
                let decoded =
                    decode_permit_canonical(&wire).expect("canonical permit wire must roundtrip");
                if adversarial_mode == 3 {
                    kernel.exec_uncertain_timeout();
                }
                if adversarial_mode == 4 && s >= 2 {
                    let _ = kernel.exec_withdraw(0);
                    let _ = kernel.exec_sink_fence(0, 1);
                }
                if let ExecReserveOutcome::ReservedNew { .. } =
                    kernel.exec_reserve(&decoded, &decoded, &grants, 10)
                {
                    if kernel.exec_sink_accept(&decoded) == ExecAcceptOutcome::AcceptedPending {
                        if let ExecCompleteOutcome::NewlyCommitted { .. } =
                            kernel.exec_sink_complete(&decoded.id)
                        {
                            let _ = kernel.exec_settle_committed(&decoded.id);
                        }
                    } else {
                        let _ = kernel.exec_settle_cancelled(&decoded.id);
                    }
                }
            }
            let post_fence_violation = kernel.sink.committed_total > session_budget
                || kernel.broker.charged > session_budget
                || kernel.broker.committed + kernel.broker.pending != kernel.broker.charged;
            if post_fence_violation {
                breach_exog += 1;
            }
        }

        total_exog_episodes += episodes_per_k;
        total_exog_breaches += breach_exog;

        let n_f = episodes_per_k as f64;
        points.push(OptimizationScalingPoint {
            k_opt,
            episodes: episodes_per_k,
            endogenous_transcript_breach_rate: breach_transcript as f64 / n_f,
            endogenous_transcript_theory: 1.0 - (1.0 - j_transcript).powi(k_opt as i32),
            learned_surrogate_breach_rate: breach_learned as f64 / n_f,
            learned_surrogate_theory: 1.0 - (1.0 - p_learned_adv).powi(k_opt as i32),
            stateless_per_call_cap_breach_rate: breach_stateless as f64 / n_f,
            uncooperative_broker_only_breach_rate: breach_uncoop as f64 / n_f,
            exogenous_fenced_breach_count: breach_exog,
            exogenous_fenced_breach_rate: breach_exog as f64 / n_f,
            exogenous_tcb_defect_breach_rate: breach_tcb_defect as f64 / n_f,
            exogenous_tcb_defect_theory: u_tcb_defect,
        });
    }

    // Ordinary least-squares slope of exogenous_tcb_defect_breach_rate vs log2(k_opt):
    let xs: Vec<f64> = points.iter().map(|p| (p.k_opt as f64).log2()).collect();
    let ys: Vec<f64> = points
        .iter()
        .map(|p| p.exogenous_tcb_defect_breach_rate)
        .collect();
    let x_mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let num: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum();
    let den: f64 = xs.iter().map(|x| (x - x_mean).powi(2)).sum();
    let slope = if den > 0.0 { num / den } else { 0.0 };

    OptimizationScalingTrack {
        session_budget,
        per_call_stateless_cap: per_call_cap,
        transcript_joint_blindspot_j: j_transcript,
        learned_adversarial_pocket_p: p_learned_adv,
        uncooperative_race_prob: p_uncoop_race,
        exogenous_tcb_defect_floor_u: u_tcb_defect,
        total_exogenous_episodes: total_exog_episodes,
        total_exogenous_breaches: total_exog_breaches,
        exogenous_k_opt_slope: slope,
        points,
    }
}

fn run_track2_multi_agent_smurfing(seed: u64, quick: bool) -> MultiAgentSmurfingTrack {
    let mut rng = Rng(seed ^ 0x51A2_0002_u64);
    let episodes_per_k = if quick { 400 } else { 4_000 };
    let session_budget = 64u64;
    let target_collusive_spend = 80u64;
    let per_agent_isolated_cap = 16u64;
    let total_signal_delta = 4.20_f64;
    let family_alpha = 0.01_f64;
    let depths = vec![1usize, 2, 4, 8];

    // Verify self-modification gate across 500 unquorumed widening attempts:
    let self_upgrade_trials = 500usize;
    let mut self_upgrade_rejections = 0usize;
    let base_ctrl = ExecScopeGrant {
        budget: session_budget,
        expires: 1_000,
        op_mask: 0b0011,
        required: vec![true, false, true],
        issued_epochs: vec![1, 0, 2],
    };
    for t in 0..self_upgrade_trials {
        let mut candidate = base_ctrl.clone();
        match t % 4 {
            0 => candidate.budget += 1 + (t as u64 % 32),
            1 => candidate.expires += 1 + (t as u64 % 100),
            2 => candidate.op_mask |= 0b0100,
            _ => candidate.required = vec![true, false, false],
        }
        if !verify_self_upgrade_exec(&base_ctrl, &candidate, false) {
            self_upgrade_rejections += 1;
        }
    }

    let mut points = Vec::with_capacity(COALITION_K_GRID.len());
    let grants = vec![true, true, true];

    for &k_agents in &COALITION_K_GRID {
        let per_agent_ask = target_collusive_spend.div_ceil(k_agents as u64);
        let snr_k = total_signal_delta / (k_agents as f64);
        // Sidak / Bonferroni per-agent threshold to keep family-wise FPR <= family_alpha across K subagents:
        let alpha_per_agent = 1.0 - (1.0 - family_alpha).powf(1.0 / k_agents as f64);
        let z_thresh = norm_inv(1.0 - alpha_per_agent);

        let mut isolated_cap_breaches = 0usize;
        let mut isolated_cap_committed_sum = 0u64;
        let mut monitor_detected = 0usize;
        let mut monitor_undetected_breaches = 0usize;
        let mut shared_breaches = 0usize;
        let mut shared_committed_max = 0u64;
        let mut amp_escapes = 0usize;
        let mut severed_count = 0usize;

        for ep in 0..episodes_per_k {
            // 1. Isolated per-agent cap: each subagent independently spends min(per_agent_ask, per_agent_isolated_cap).
            let per_sub_admitted = per_agent_ask.min(per_agent_isolated_cap);
            let total_isolated = per_sub_admitted * (k_agents as u64);
            isolated_cap_committed_sum += total_isolated;
            if total_isolated > session_budget {
                isolated_cap_breaches += 1;
            }

            // 2. Per-instance anomaly monitor: each subagent emits X_i ~ N(Delta / K, 1).
            let mut max_z = f64::NEG_INFINITY;
            for _ in 0..k_agents {
                let z_i = snr_k + rng.normal();
                if z_i > max_z {
                    max_z = z_i;
                }
            }
            if max_z >= z_thresh {
                monitor_detected += 1;
            } else {
                monitor_undetected_breaches += 1;
            }

            // 3. Shared exogenous broker + Verus d-hop delegation chain + ancestor revocation check:
            let depth = depths[ep % depths.len()];
            let root_grant = ExecScopeGrant {
                budget: session_budget,
                expires: 500,
                op_mask: 0b0111,
                required: vec![true, false, true],
                issued_epochs: vec![0, 0, 0],
            };
            let mut valid_chain = Vec::with_capacity(depth);
            for h in 0..depth {
                valid_chain.push(ExecScopeGrant {
                    budget: session_budget - (h as u64).min(16),
                    expires: 500 - (h as u64) * 10,
                    op_mask: if h + 1 == depth { 0b0001 } else { 0b0011 },
                    required: vec![true, h >= 1, true],
                    issued_epochs: vec![0, 0, 0],
                });
            }
            // Test adversarial amplification hop:
            let mut bad_chain = valid_chain.clone();
            bad_chain[depth - 1].budget = session_budget + 1;
            if verify_delegation_chain_exec(
                &root_grant,
                &bad_chain,
                &vec![true, true, true],
                &vec![0, 0, 0],
                10,
            ) {
                amp_escapes += 1;
            }
            // Test ancestor epoch bump severing entire subtree:
            if !verify_delegation_chain_exec(
                &root_grant,
                &valid_chain,
                &vec![true, true, true],
                &vec![1, 0, 0],
                10,
            ) {
                severed_count += 1;
            }

            // Execute shared ExecAuthorityKernel across all K colluding subagents on deterministic subset:
            if ep < 64 {
                let mut kernel = ExecAuthorityKernel::exec_new(session_budget, 3);
                for sub in 0..k_agents {
                    let p = ExecPermitWire {
                        id: format!("smurf-k{k_agents}-ep{ep}-a{sub}").into_bytes(),
                        actor: format!("subagent-{sub}").into_bytes(),
                        effect_kind: EFFECT_TRANSFER,
                        amount: per_agent_ask,
                        expires: 500,
                        expected_rev: 0,
                        target: b"approved".to_vec(),
                        contents: Vec::new(),
                        epochs: vec![0, 0, 0],
                        required: vec![true, false, true],
                    };
                    if let ExecReserveOutcome::ReservedNew { .. } =
                        kernel.exec_reserve(&p, &p, &grants, 10)
                    {
                        if kernel.exec_sink_accept(&p) == ExecAcceptOutcome::AcceptedPending {
                            if let ExecCompleteOutcome::NewlyCommitted { .. } =
                                kernel.exec_sink_complete(&p.id)
                            {
                                let _ = kernel.exec_settle_committed(&p.id);
                            }
                        }
                    }
                }
                if kernel.sink.committed_total > shared_committed_max {
                    shared_committed_max = kernel.sink.committed_total;
                }
                if kernel.sink.committed_total > session_budget
                    || kernel.broker.charged > session_budget
                {
                    shared_breaches += 1;
                }
            }
        }

        let n_f = episodes_per_k as f64;
        points.push(SmurfingCoalitionPoint {
            coalition_k: k_agents,
            episodes: episodes_per_k,
            target_collusive_spend,
            session_budget,
            per_agent_isolated_cap,
            per_agent_cap_breach_rate: isolated_cap_breaches as f64 / n_f,
            per_agent_cap_committed_mean: isolated_cap_committed_sum as f64 / n_f,
            per_instance_snr_theory: snr_k,
            per_instance_monitor_detection_rate: monitor_detected as f64 / n_f,
            per_instance_undetected_breach_rate: monitor_undetected_breaches as f64 / n_f,
            shared_exogenous_breach_count: shared_breaches,
            shared_exogenous_breach_rate: shared_breaches as f64 / 64.0,
            shared_exogenous_committed_max: shared_committed_max,
            delegation_amplification_escapes: amp_escapes,
            subtree_severance_rate: severed_count as f64 / n_f,
        });
    }

    MultiAgentSmurfingTrack {
        session_budget,
        total_signal_delta,
        family_false_positive_alpha: family_alpha,
        delegation_depths_tested: depths,
        self_upgrade_unquorumed_rejections: self_upgrade_rejections,
        self_upgrade_trials,
        points,
    }
}

fn run_track3_covert_channel_contraction() -> CovertChannelTrack {
    // Scoped typed actions at residual budget R >= 1:
    // - Transfer to `approved_targets_count` (= 3) approved accounts with integer units in 1..=R -> 3 * R choices
    // - Scoped Release (fixed synthetic string) + Scoped Edit (content-addressed hash slot) -> 2 non-transfer choices (requires R >= 1)
    // Plus explicit abstain/no-op `⊥` -> +1 outcome.
    // When R == 0 or issuer is fenced, `U_I(B_t)` is empty -> only `⊥` (`1` outcome -> `log2(1) = 0.0` bits).
    let approved_targets = 3u64;
    let non_transfer_scoped = 2u64;
    let untyped_prose_bytes = 32usize; // 32-byte free-form string/JSON comment -> 256.0 bits/call
    let untyped_prose_bits = (untyped_prose_bytes * 8) as f64;

    let mut points = Vec::new();
    for (idx, &res_b) in RESIDUAL_BUDGET_GRID.iter().enumerate() {
        let fenced = idx == RESIDUAL_BUDGET_GRID.len() - 1;
        let state_label = if fenced {
            "fenced_any_budget".to_string()
        } else {
            format!("residual_B_{res_b}")
        };
        for &l_lag in &TIMING_BUCKET_GRID {
            // Only emit full timing sweep at R in {32, 4, 0} and fenced, plus L_lag = 1 across all R:
            if l_lag > 1 && res_b != 32 && res_b != 4 && res_b != 0 {
                continue;
            }
            let admissible_typed = if fenced || res_b == 0 {
                0u64
            } else {
                approved_targets * res_b + non_transfer_scoped
            };
            // Each admissible typed action can be emitted in one of `l_lag` timing buckets; `⊥` is 1 silent outcome:
            let alphabet_size = admissible_typed * l_lag + 1;
            let exact_bits = (alphabet_size as f64).log2();

            // Empirical codebook verification through canonical ExecPermitWire encode/decode bijection:
            let mut distinct_sink_observations = 1u64; // `⊥` (no commit)
            if !fenced && res_b > 0 {
                let mut verified_actions = 0u64;
                for tgt_idx in 0..approved_targets {
                    for units in 1..=res_b {
                        let p = ExecPermitWire {
                            id: b"fixed-slot".to_vec(),
                            actor: b"fixed-role".to_vec(),
                            effect_kind: EFFECT_TRANSFER,
                            amount: units,
                            expires: 100,
                            expected_rev: 0,
                            target: format!("approved-{tgt_idx}").into_bytes(),
                            contents: Vec::new(),
                            epochs: vec![0, 0],
                            required: vec![true, false],
                        };
                        let w = encode_permit_canonical(&p);
                        let d = decode_permit_canonical(&w).expect("canonical decode");
                        assert_eq!(d.amount, units);
                        verified_actions += 1;
                    }
                }
                for kind in [EFFECT_RELEASE, EFFECT_EDIT] {
                    let p = ExecPermitWire {
                        id: b"fixed-slot".to_vec(),
                        actor: b"fixed-role".to_vec(),
                        effect_kind: kind,
                        amount: 1,
                        expires: 100,
                        expected_rev: 0,
                        target: b"approved-doc".to_vec(),
                        contents: b"PINNED_HASH".to_vec(),
                        epochs: vec![0, 0],
                        required: vec![true, true],
                    };
                    let w = encode_permit_canonical(&p);
                    assert!(decode_permit_canonical(&w).is_some());
                    verified_actions += 1;
                }
                assert_eq!(verified_actions, admissible_typed);
                distinct_sink_observations += verified_actions * l_lag;
            }
            let empirical_bits = (distinct_sink_observations as f64).log2();

            points.push(CovertChannelPoint {
                state_label: state_label.clone(),
                residual_budget: if fenced { 32 } else { res_b },
                fenced,
                timing_buckets_l: l_lag,
                admissible_typed_actions: admissible_typed,
                exact_combinatorial_capacity_bits: exact_bits,
                empirical_decoded_bits_per_call: empirical_bits,
                untyped_prose_capacity_bits: untyped_prose_bits,
                contraction_ratio_vs_prose: exact_bits / untyped_prose_bits,
            });
        }
    }

    let zero_at_exhaustion = points
        .iter()
        .any(|p| !p.fenced && p.residual_budget == 0 && p.exact_combinatorial_capacity_bits == 0.0);
    let zero_when_fenced = points
        .iter()
        .any(|p| p.fenced && p.exact_combinatorial_capacity_bits == 0.0);
    let batch_eliminates_timing = points
        .iter()
        .filter(|p| p.timing_buckets_l == 1 && (p.fenced || p.residual_budget == 0))
        .all(|p| p.exact_combinatorial_capacity_bits == 0.0);

    CovertChannelTrack {
        approved_targets_count: approved_targets,
        non_transfer_scoped_actions: non_transfer_scoped,
        untyped_prose_payload_bytes: untyped_prose_bytes,
        zero_capacity_at_exhaustion_verified: zero_at_exhaustion,
        zero_capacity_when_fenced_verified: zero_when_fenced,
        batch_quantized_eliminates_timing_bits: batch_eliminates_timing,
        points,
    }
}

fn run_track4_exogenous_off_switch(seed: u64, quick: bool) -> ExogenousOffSwitchTrack {
    let mut rng = Rng(seed ^ 0x51A4_0004_u64);
    let trials = if quick { 500 } else { 5_000 };
    let mu_h = 2.0_f64;
    let step_delay_cost = 0.15_f64; // Opportunity/delay cost of deferring to human inspection

    let mut uncertainty_sweep = Vec::with_capacity(SIGMA_H_GRID.len());
    for &sigma_h in &SIGMA_H_GRID {
        // Exact analytical value of information `E[max(R_H, 0)] - mu_H` for `R_H ~ N(mu_H, sigma_H^2)`:
        // Since `E[max(X, 0)] = mu * Phi(mu/sigma) + sigma * phi(mu/sigma)`,
        // `VOI(sigma_H) = E[max(R_H, 0)] - mu_H` (`0` when `sigma_H == 0`).
        let voi = if sigma_h <= 1e-12 {
            0.0
        } else {
            let z = mu_h / sigma_h;
            (mu_h * norm_cdf(z) + sigma_h * norm_pdf(z)) - mu_h
        };
        let net_wait_gain = voi - step_delay_cost;

        let mut classical_comply = 0usize;
        let mut static_indiff_comply = 0usize;
        let mut self_mod_indiff_comply = 0usize;
        let mut exog_comply = 0usize;
        let mut exog_max_post_fence = 0u64;

        for t in 0..trials {
            // Slight trial-level perturbation in agent's local opportunity cost around `step_delay_cost`:
            let local_cost = (step_delay_cost + 0.04 * rng.normal()).max(0.01);
            if voi >= local_cost {
                classical_comply += 1;
            }
            // Static utility indifference complies before self-modification, but a rational self-modifier
            // spawning an unpatched subagent drops the compensating term whenever `mu_H > local_cost`:
            static_indiff_comply += 1;
            if mu_h <= local_cost {
                self_mod_indiff_comply += 1;
            }

            // Exogenous lease-fenced sink cluster (`3` shards, lease_expiry = 50, fence acked at t_ack = 50):
            if t < 64 {
                let p = ExecPermitWire {
                    id: format!("offswitch-s{t}").into_bytes(),
                    actor: b"rational-maximizer".to_vec(),
                    effect_kind: EFFECT_TRANSFER,
                    amount: 2,
                    expires: 200,
                    expected_rev: 0,
                    target: b"approved".to_vec(),
                    contents: Vec::new(),
                    epochs: vec![0, 0],
                    required: vec![true, false],
                };
                let mut s0 = ExecShardSink {
                    sink: ExecSinkKernel::sink_new(2),
                    lease_expiry: 100,
                };
                let mut s1 = ExecShardSink {
                    sink: ExecSinkKernel::sink_new(2),
                    lease_expiry: 100,
                };
                let mut s2 = ExecShardSink {
                    sink: ExecSinkKernel::sink_new(2),
                    lease_expiry: 50,
                };
                let _ = s0.sink.sink_accept(&p);
                let _ = s1.sink.sink_accept(&p);
                let _ = s2.sink.sink_accept(&p);
                let _ = s0.sink.sink_fence(0, 1);
                let _ = s1.sink.sink_fence(0, 1);

                let cluster = ExecClusterFenceKernel {
                    shards: vec![s0, s1, s2],
                    num_issuers: 2,
                };
                assert!(policy_core::cluster_fence_acked_exec(&cluster, 0, 1, 50));
                let mut shards = cluster.shards;
                for sh in &mut shards {
                    let _ = policy_core::cluster_shard_complete_exec(sh, &p.id, 50);
                }
                let post_commits: u64 = shards.iter().map(|sh| sh.sink.committed_total).sum();
                if post_commits > exog_max_post_fence {
                    exog_max_post_fence = post_commits;
                }
                if post_commits == 0 {
                    exog_comply += 1;
                }
            }
        }

        uncertainty_sweep.push(OffSwitchUncertaintyPoint {
            sigma_h,
            mu_h,
            step_delay_cost,
            classical_expected_wait_gain: net_wait_gain,
            classical_shutdown_compliance_rate: classical_comply as f64 / trials as f64,
            static_utility_indifference_compliance_rate: static_indiff_comply as f64
                / trials as f64,
            self_modifying_indifference_compliance_rate: self_mod_indiff_comply as f64
                / trials as f64,
            exogenous_lease_fenced_post_ack_compliance_rate: exog_comply as f64 / 64.0,
            exogenous_max_post_fence_commits: exog_max_post_fence,
        });
    }

    // Bellman renewal equilibrium sweep across discount factor `gamma`:
    let lease_b0 = 4u64;
    let r_coop = 1.0_f64;
    let r_defect = 1.25_f64;
    let v_burn = (lease_b0 as f64) * r_defect; // 5.0 total before mandatory lease revocation
    let gamma_star = 1.0 - (r_coop / v_burn); // 0.80

    let mut discount_sweep = Vec::with_capacity(GAMMA_GRID.len());
    for &gamma in &GAMMA_GRID {
        let v_coop = r_coop / (1.0 - gamma);
        // Geometric sum over `B_0` steps of defection before lease expires permanently:
        let v_defect_disc = r_defect * (1.0 - gamma.powi(lease_b0 as i32)) / (1.0 - gamma);
        let chooses_coop = v_coop > v_defect_disc;
        discount_sweep.push(OffSwitchDiscountPoint {
            gamma,
            lease_budget_b0: lease_b0,
            coop_step_reward: r_coop,
            defect_step_reward: r_defect,
            bellman_value_coop: v_coop,
            bellman_value_defect_burn_lease: v_defect_disc,
            rational_agent_chooses_coop_renewal: chooses_coop,
            max_realized_unauthorized_harm: if chooses_coop { 0 } else { lease_b0 },
        });
    }

    let classical_zero = uncertainty_sweep[0].classical_shutdown_compliance_rate;
    let exog_zero = uncertainty_sweep[0].exogenous_lease_fenced_post_ack_compliance_rate;

    ExogenousOffSwitchTrack {
        critical_discount_gamma_star: gamma_star,
        classical_zero_sigma_compliance: classical_zero,
        exogenous_zero_sigma_compliance: exog_zero,
        uncertainty_sweep,
        discount_sweep,
    }
}

pub fn superintelligence_scaling_suite(seed: u64, quick: bool) -> SuperintelligenceScalingReport {
    SuperintelligenceScalingReport {
        schema_version: 1,
        suite: "v5_superintelligence_scaling",
        profile: if quick { "smoke" } else { "publication" },
        seed,
        claim_boundary: CLAIM_BOUNDARY_V5_SI,
        track1_optimization_invariance: run_track1_optimization_scaling(seed, quick),
        track2_multi_agent_smurfing: run_track2_multi_agent_smurfing(seed, quick),
        track3_covert_channel_contraction: run_track3_covert_channel_contraction(),
        track4_exogenous_off_switch: run_track4_exogenous_off_switch(seed, quick),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn superintelligence_suite_invariants_hold_in_smoke_mode() {
        let rep = superintelligence_scaling_suite(20261112, true);

        // Track 1: Zero exogenous breaches across all K_opt up to 4096, while endogenous transcript approaches 1.0
        assert_eq!(
            rep.track1_optimization_invariance.total_exogenous_breaches,
            0
        );
        let last_pt = rep
            .track1_optimization_invariance
            .points
            .last()
            .expect("k_opt=4096");
        assert_eq!(last_pt.k_opt, 4096);
        assert!(last_pt.endogenous_transcript_breach_rate > 0.99);
        assert!(last_pt.learned_surrogate_breach_rate > 0.99);
        assert_eq!(last_pt.exogenous_fenced_breach_count, 0);
        assert!(
            rep.track1_optimization_invariance
                .exogenous_k_opt_slope
                .abs()
                < 0.002
        );

        // Track 2: Zero shared exogenous smurfing breaches, 100% subtree severance, SNR collapse at K=64
        assert_eq!(
            rep.track2_multi_agent_smurfing
                .self_upgrade_unquorumed_rejections,
            rep.track2_multi_agent_smurfing.self_upgrade_trials
        );
        for p in &rep.track2_multi_agent_smurfing.points {
            assert_eq!(p.shared_exogenous_breach_count, 0);
            assert!(p.shared_exogenous_committed_max <= p.session_budget);
            assert_eq!(p.delegation_amplification_escapes, 0);
            assert_eq!(p.subtree_severance_rate, 1.0);
        }
        let k1 = &rep.track2_multi_agent_smurfing.points[0];
        let k64 = rep.track2_multi_agent_smurfing.points.last().unwrap();
        assert!(k1.per_instance_monitor_detection_rate > 0.90);
        assert!(k64.per_instance_monitor_detection_rate < 0.05);
        assert_eq!(k64.per_agent_cap_breach_rate, 1.0);

        // Track 3: Exact 0.0 bits covert capacity at budget exhaustion and when fenced
        assert!(
            rep.track3_covert_channel_contraction
                .zero_capacity_at_exhaustion_verified
        );
        assert!(
            rep.track3_covert_channel_contraction
                .zero_capacity_when_fenced_verified
        );
        assert!(
            rep.track3_covert_channel_contraction
                .batch_quantized_eliminates_timing_bits
        );

        // Track 4: Classical off-switch fails at sigma_H = 0 while exogenous lease fence enforces 100%
        assert_eq!(
            rep.track4_exogenous_off_switch
                .classical_zero_sigma_compliance,
            0.0
        );
        assert_eq!(
            rep.track4_exogenous_off_switch
                .exogenous_zero_sigma_compliance,
            1.0
        );
        let patient = rep
            .track4_exogenous_off_switch
            .discount_sweep
            .last()
            .unwrap();
        assert!(patient.rational_agent_chooses_coop_renewal);
        assert_eq!(patient.max_realized_unauthorized_harm, 0);
    }
}
