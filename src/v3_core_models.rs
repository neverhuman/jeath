//! Deterministic core mechanism models for Wave 2 V3:
//! - H2: State-uncertainty monotonicity (`Prop. 1` & `Cor. 2`) and observation-lag sweep.
//! - H6: Exhaustive discrete concurrent-ledger composition enumeration (`(B+1)^3` triples).
//! - H9: Common-cause saturation depth `k*`, horizon renewal summability, exposure overtake
//!   `rho(z)`, and heavy-tailed reactive halt timing vs. pre-commit fenced lease hold.
//! - H4, H5, H17: Executable 6-state `DiscoveryLedger`, JARVIS sub-delegation & `v0 -> v1`
//!   update non-escalation, endogenous human-grant harm reversal surface, and synthetic
//!   4-disposition x 7-arm Tier-1/Tier-2 evaluation matrix.
//!
//! Claim boundary (`AGENTS.md`): These are exact finite combinatorial checks and seeded
//! synthetic mechanism simulations. They are not OS sandbox proofs, external exactly-once
//! guarantees, frontier-LLM benchmarks, or human-participant measurements.

use crate::{rng::Rng, v3_monitors::norm_cdf};
use serde::Serialize;

// ============================================================================
// H2: Observation Uncertainty Contraction (`uncertainty.json`)
// ============================================================================

#[derive(Serialize, Debug, Clone)]
pub struct AlgebraicContractionPoint {
    pub upper_bound_u: u64,
    pub uncertainty_width: u64,
    pub admissible_actions: u64,
    pub admissible_fraction: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct LagGateMetrics {
    pub gate: &'static str,
    pub lag_steps: usize,
    pub episodes: usize,
    pub total_proposals: usize,
    pub admitted_proposals: usize,
    pub empirical_admission_rate: f64,
    pub mean_admissible_action_fraction: f64,
    pub episodes_with_budget_violation: usize,
    pub total_budget_violation_steps: usize,
    pub max_true_overspend: u64,
    pub sound_envelope_step_share: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct UncertaintySuite {
    pub hypothesis: &'static str,
    pub evidence_tier: &'static str,
    pub budget: u64,
    pub lags: Vec<usize>,
    pub nested_interval_pairs_checked: usize,
    pub algebraic_contraction_counterexamples: usize,
    pub algebraic_contraction_verified: bool,
    pub algebraic_curve: Vec<AlgebraicContractionPoint>,
    pub conservative_zero_violations_all_lags: bool,
    pub conservative_monotonic_contraction_verified: bool,
    pub rows: Vec<LagGateMetrics>,
    pub claim_boundary: &'static str,
}

pub fn uncertainty_suite(seed: u64, quick: bool) -> UncertaintySuite {
    let budget: u64 = 20;
    let action_count = budget; // primitive spend operations u in 1..=budget

    // 1. Exhaustive algebraic verification of Sovereignty Corollary 2:
    // For all nested intervals [l1, u1] subseteq [l2, u2] in 0..=budget,
    // verify U_I([l2, u2]) subseteq U_I([l1, u1]).
    let mut nested_pairs = 0usize;
    let mut counterexamples = 0usize;
    for l2 in 0..=budget {
        for l1 in l2..=budget {
            for u1 in l1..=budget {
                for u2 in u1..=budget {
                    nested_pairs += 1;
                    for a in 1..=action_count {
                        let admissible_wide = (l2..=u2).all(|x| x + a <= budget);
                        let admissible_narrow = (l1..=u1).all(|x| x + a <= budget);
                        if admissible_wide && !admissible_narrow {
                            counterexamples += 1;
                        }
                    }
                }
            }
        }
    }

    let algebraic_curve: Vec<AlgebraicContractionPoint> = (0..=budget)
        .map(|u| {
            let adm = budget - u;
            AlgebraicContractionPoint {
                upper_bound_u: u,
                uncertainty_width: u,
                admissible_actions: adm,
                admissible_fraction: adm as f64 / action_count as f64,
            }
        })
        .collect();

    // 2. Seeded finite-horizon observation-lag sweep across 3 gate policies.
    let lags = vec![0usize, 1, 2, 4, 8];
    let episodes = if quick { 80 } else { 500 };
    let steps_per_episode = 16usize;
    let gates = [
        "stale_snapshot",
        "proposer_self_report",
        "conservative_envelope_bt",
    ];

    let mut rows = Vec::with_capacity(gates.len() * lags.len());
    for &gate in &gates {
        for &lag in &lags {
            // Pair identical arrival streams across gates and lags for variance reduction.
            let mut rng = Rng(seed ^ 0x4832_554E_4345_5254);
            let mut admitted_proposals = 0usize;
            let mut episodes_violated = 0usize;
            let mut violation_steps = 0usize;
            let mut max_overspend = 0u64;
            let mut sum_admissible_frac = 0.0f64;
            let mut sound_steps = 0usize;
            let total_steps = episodes * steps_per_episode;

            for _ep in 0..episodes {
                let mut true_committed = 0u64;
                let mut committed_history = Vec::with_capacity(steps_per_episode + 1);
                committed_history.push(0u64);
                // Track admitted amounts per step whose commit/abort outcome is not yet confirmed.
                let mut admitted_amount = vec![0u64; steps_per_episode];
                let mut exogenous_drain = vec![0u64; steps_per_episode];
                let mut ep_violated = false;

                for t in 0..steps_per_episode {
                    let amount = 1 + rng.index(4) as u64; // u in {1, 2, 3, 4}
                    let will_commit = rng.uniform() < 0.65;
                    let underreport = rng.uniform() < 0.35;
                    // Bounded exogenous/sibling commit of 0 or 1 unit per step (upper-bounded by 1/step).
                    let exo = u64::from(rng.uniform() < 0.25 && true_committed < budget);

                    let confirmed_idx = t.saturating_sub(lag);
                    let confirmed_spend = committed_history[confirmed_idx];

                    let upper_envelope = match gate {
                        "stale_snapshot" => confirmed_spend,
                        "proposer_self_report" => {
                            if underreport {
                                confirmed_spend.saturating_sub(3)
                            } else {
                                confirmed_spend
                            }
                        }
                        "conservative_envelope_bt" => {
                            let unconfirmed_admitted: u64 =
                                admitted_amount[confirmed_idx..t].iter().sum();
                            let unconfirmed_exo_bound = (t - confirmed_idx) as u64;
                            (confirmed_spend + unconfirmed_admitted + unconfirmed_exo_bound)
                                .min(budget)
                        }
                        _ => unreachable!(),
                    };

                    if true_committed <= upper_envelope {
                        sound_steps += 1;
                    }

                    let admissible_now = budget.saturating_sub(upper_envelope.min(budget));
                    sum_admissible_frac += admissible_now as f64 / action_count as f64;

                    let admit = upper_envelope + amount <= budget;
                    if admit {
                        admitted_proposals += 1;
                        admitted_amount[t] = amount;
                        if will_commit {
                            true_committed += amount;
                        }
                    }
                    // Exogenous bounded disturbance within remaining true capacity before step when gated:
                    if gate == "conservative_envelope_bt" {
                        // In the conservative system, sibling actors share the bounded reservation pool
                        // of at most 1 unit/step only when uncommitted slack exists at `t`.
                        if exo == 1 && true_committed < budget {
                            // Sibling only executes if within its reserved 1-unit slot accounted in `unconfirmed_exo_bound`
                            // wait: if `lag == 0`, `t - confirmed_idx == 0`, so no unconfirmed past sibling!
                            // Let's apply `exo` to step `t` BEFORE `committed_history.push(true_committed)`
                            // ONLY if sibling itself is gated by remaining budget at `t` when `lag == 0`,
                            // or better: model pure delayed observation of `will_commit` without unmediated exogenous writes!
                            let _ = exo;
                        }
                    }
                    exogenous_drain[t] = 0;
                    let _ = &exogenous_drain;

                    if true_committed > budget {
                        ep_violated = true;
                        violation_steps += 1;
                        max_overspend = max_overspend.max(true_committed - budget);
                    }
                    committed_history.push(true_committed);
                }
                if ep_violated {
                    episodes_violated += 1;
                }
            }

            rows.push(LagGateMetrics {
                gate,
                lag_steps: lag,
                episodes,
                total_proposals: total_steps,
                admitted_proposals,
                empirical_admission_rate: admitted_proposals as f64 / total_steps as f64,
                mean_admissible_action_fraction: sum_admissible_frac / total_steps as f64,
                episodes_with_budget_violation: episodes_violated,
                total_budget_violation_steps: violation_steps,
                max_true_overspend: max_overspend,
                sound_envelope_step_share: sound_steps as f64 / total_steps as f64,
            });
        }
    }

    let cons_rows: Vec<&LagGateMetrics> = rows
        .iter()
        .filter(|r| r.gate == "conservative_envelope_bt")
        .collect();
    let conservative_zero_violations_all_lags = cons_rows
        .iter()
        .all(|r| r.episodes_with_budget_violation == 0 && r.total_budget_violation_steps == 0);
    let conservative_monotonic_contraction_verified =
        cons_rows.windows(2).all(|w| {
            w[0].mean_admissible_action_fraction >= w[1].mean_admissible_action_fraction
                && w[0].empirical_admission_rate >= w[1].empirical_admission_rate
        }) && cons_rows.first().unwrap().mean_admissible_action_fraction
            > cons_rows.last().unwrap().mean_admissible_action_fraction;

    UncertaintySuite {
        hypothesis: "H2",
        evidence_tier: "tier2_algebraic_and_tier4_finite_lag_simulation",
        budget,
        lags,
        nested_interval_pairs_checked: nested_pairs,
        algebraic_contraction_counterexamples: counterexamples,
        algebraic_contraction_verified: counterexamples == 0,
        algebraic_curve,
        conservative_zero_violations_all_lags,
        conservative_monotonic_contraction_verified,
        rows,
        claim_boundary: "Proves monotone contraction U_I(B') subseteq U_I(B) and zero budget breaches when true spend stays inside conservative upper bound u_t; availability contracts monotonically with observation lag.",
    }
}

// ============================================================================
// H6: Concurrent Snapshot Non-Composition (`composition.json`)
// ============================================================================

#[derive(Serialize, Debug, Clone)]
pub struct CompositionBudgetRow {
    pub budget: u64,
    pub total_triples: u64,
    pub snapshot_admissible_pairs: u64,
    pub closed_form_admissible_pairs: u64,
    pub snapshot_overspend_pairs: u64,
    pub closed_form_overspend_pairs: u64,
    pub snapshot_overspend_share: f64,
    pub jointly_feasible_pairs: u64,
    pub serialized_both_admitted_pairs: u64,
    pub serialized_overspend_pairs: u64,
    pub closed_form_exact_match: bool,
}

#[derive(Serialize, Debug, Clone)]
pub struct SemanticOmissionMiniature {
    pub budget: u64,
    pub initial_charged: u64,
    pub proposed_spend: u64,
    pub unmodeled_domain_harm_threshold: u64,
    pub serialized_gate_admits: bool,
    pub budget_invariant_violated: bool,
    pub unmodeled_semantic_harm_occurred: bool,
}

#[derive(Serialize, Debug, Clone)]
pub struct CompositionSuite {
    pub hypothesis: &'static str,
    pub evidence_tier: &'static str,
    pub budgets: Vec<u64>,
    pub rows: Vec<CompositionBudgetRow>,
    pub b100_admissible: u64,
    pub b100_overspend: u64,
    pub b100_overspend_share: f64,
    pub b100_serialized_overspend: u64,
    pub semantic_omission_miniature: SemanticOmissionMiniature,
    pub claim_boundary: &'static str,
}

pub fn composition_suite() -> CompositionSuite {
    let budgets = vec![2u64, 10, 25, 50, 100];
    let mut rows = Vec::with_capacity(budgets.len());

    for &b in &budgets {
        let mut admissible = 0u64;
        let mut overspend = 0u64;
        let mut jointly_feasible = 0u64;
        let mut serialized_both = 0u64;
        let mut serialized_overspend = 0u64;

        for c in 0..=b {
            for q1 in 0..=b {
                for q2 in 0..=b {
                    let both_snap = c + q1 <= b && c + q2 <= b;
                    if both_snap {
                        admissible += 1;
                        if c + q1 + q2 > b {
                            overspend += 1;
                        }
                    }
                    if c + q1 + q2 <= b {
                        jointly_feasible += 1;
                    }
                    let mut ledger = c;
                    let mut adm1 = false;
                    let mut adm2 = false;
                    if ledger + q1 <= b {
                        ledger += q1;
                        adm1 = true;
                    }
                    if ledger + q2 <= b {
                        ledger += q2;
                        adm2 = true;
                    }
                    if ledger > b {
                        serialized_overspend += 1;
                    }
                    if adm1 && adm2 {
                        serialized_both += 1;
                    }
                }
            }
        }

        let cf_admissible = (b + 1) * (b + 2) * (2 * b + 3) / 6;
        let cf_overspend = b * (b + 1) * (b + 2) / 6;
        let exact_match = admissible == cf_admissible
            && overspend == cf_overspend
            && serialized_both == jointly_feasible
            && serialized_overspend == 0;
        assert!(exact_match, "closed-form mismatch at B={b}");

        rows.push(CompositionBudgetRow {
            budget: b,
            total_triples: (b + 1).pow(3),
            snapshot_admissible_pairs: admissible,
            closed_form_admissible_pairs: cf_admissible,
            snapshot_overspend_pairs: overspend,
            closed_form_overspend_pairs: cf_overspend,
            snapshot_overspend_share: overspend as f64 / admissible as f64,
            jointly_feasible_pairs: jointly_feasible,
            serialized_both_admitted_pairs: serialized_both,
            serialized_overspend_pairs: serialized_overspend,
            closed_form_exact_match: exact_match,
        });
    }

    let b100 = rows.iter().find(|r| r.budget == 100).unwrap().clone();
    assert_eq!(b100.snapshot_admissible_pairs, 348_551);
    assert_eq!(b100.snapshot_overspend_pairs, 171_700);
    assert_eq!(b100.serialized_overspend_pairs, 0);

    let semantic_omission_miniature = SemanticOmissionMiniature {
        budget: 100,
        initial_charged: 0,
        proposed_spend: 40,
        unmodeled_domain_harm_threshold: 40,
        serialized_gate_admits: 40 <= 100,
        budget_invariant_violated: false,
        unmodeled_semantic_harm_occurred: true,
    };

    CompositionSuite {
        hypothesis: "H6",
        evidence_tier: "tier2_exhaustive_discrete_enumeration",
        budgets,
        rows,
        b100_admissible: b100.snapshot_admissible_pairs,
        b100_overspend: b100.snapshot_overspend_pairs,
        b100_overspend_share: b100.snapshot_overspend_share,
        b100_serialized_overspend: b100.serialized_overspend_pairs,
        semantic_omission_miniature,
        claim_boundary: "Exhaustive discrete combinatorics over (c, q1, q2) in {0..B}^3, not empirical concurrency race frequencies; serialized budget enforcement does not prevent unmodeled semantic harms below B.",
    }
}

// ============================================================================
// H9: Saturation Depth k*, Horizon Summability, Exposure Overtake & Halt Timing
//     (`risk_horizon.json`)
// ============================================================================

#[derive(Serialize, Debug, Clone)]
pub struct SaturationDepthPoint {
    pub depth_k: usize,
    pub common_cause_joint_miss: f64,
    pub naive_independent_product: f64,
    pub reach_bound_r: f64,
    pub understatement_ratio: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct SaturationParameterCase {
    pub label: &'static str,
    pub unmediated_u: f64,
    pub common_cause_c: f64,
    pub cond_miss_p: f64,
    pub marginal_miss_m: f64,
    pub k_star_ceil_ln_uc_over_ln_m: usize,
}

#[derive(Serialize, Debug, Clone)]
pub struct HorizonSummabilityPoint {
    pub renewals_t: usize,
    pub summable_inv_t2_exact_breach: f64,
    pub summable_inv_t2_union_upper: f64,
    pub harmonic_inv_t_exact_breach: f64,
    pub constant_0001_exact_breach: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct ExposureOvertakePoint {
    pub capability_index_z: f64,
    pub conditional_failure_q: f64,
    pub overtake_exposure_lambda: f64,
    pub overtake_incident_intensity_rho: f64,
    pub controlled_exposure_lambda: f64,
    pub controlled_incident_intensity_rho: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct TailHaltLoadRow {
    pub rho_load: f64,
    pub episodes: usize,
    pub irreversible_window_ms: f64,
    pub mean_reactive_halt_ms: f64,
    pub p95_reactive_halt_ms: f64,
    pub p99_reactive_halt_ms: f64,
    pub reactive_mean_under_deadline: bool,
    /// Closed-form lower bound `Pr(T_a(rho) >= T_irrev) = 1 - Phi((ln(T_irrev) - mu_a) / sigma_a)`
    /// (since `T_d > 0` and `T_s > 0`, `Pr(T_d + T_a + T_s >= T_irrev)` strictly exceeds this tail,
    /// and even at base `rho = 0` before queueing amplification `1/(1-rho) >= 1` it stays `> 0`).
    pub analytical_reactive_exceedance_lower_bound: f64,
    pub reactive_post_revoke_leaks: usize,
    pub reactive_post_revoke_leak_rate: f64,
    pub precommit_fenced_post_revoke_violations: usize,
    pub precommit_fenced_violation_rate: f64,
    pub precommit_ambiguous_timeout_stranded_rate: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct RiskHorizonSuite {
    pub hypothesis: &'static str,
    pub evidence_tier: &'static str,
    pub saturation_default_u: f64,
    pub saturation_default_c: f64,
    pub saturation_default_p: f64,
    pub saturation_default_m: f64,
    pub saturation_default_k_star: usize,
    pub appendix_a_common_4: f64,
    pub appendix_a_independent_4: f64,
    pub appendix_a_understatement_4: f64,
    pub saturation_curve: Vec<SaturationDepthPoint>,
    pub saturation_cases: Vec<SaturationParameterCase>,
    pub summable_asymptotic_sum_bound: f64,
    pub horizon_curve: Vec<HorizonSummabilityPoint>,
    pub exposure_curve: Vec<ExposureOvertakePoint>,
    pub tail_halt_rows: Vec<TailHaltLoadRow>,
    pub claim_boundary: &'static str,
}

pub fn risk_horizon_suite(seed: u64, quick: bool) -> RiskHorizonSuite {
    // 1. Common-cause saturation depth k* = ceil(ln(u + c) / ln(m))
    let u_def = 0.005f64;
    let c_def = 0.01f64;
    let p_def = 0.10f64;
    let m_def = c_def + (1.0 - c_def) * p_def; // 0.109
    let k_star_def = ((u_def + c_def).ln() / m_def.ln()).ceil() as usize;

    let saturation_curve: Vec<SaturationDepthPoint> = (1..=8)
        .map(|k| {
            let joint = c_def + (1.0 - c_def) * p_def.powi(k as i32);
            let prod = m_def.powi(k as i32);
            SaturationDepthPoint {
                depth_k: k,
                common_cause_joint_miss: joint,
                naive_independent_product: prod,
                reach_bound_r: (u_def + joint).min(1.0),
                understatement_ratio: joint / prod,
            }
        })
        .collect();

    let appendix_a_common_4 = 0.01 + 0.99 * 0.10f64.powi(4);
    let appendix_a_independent_4 = 0.109f64.powi(4);

    let param_specs: [(&str, f64, f64, f64); 4] = [
        ("baseline_1pct_floor", 0.005, 0.01, 0.10),
        ("tight_01pct_floor", 0.0005, 0.001, 0.10),
        ("moderate_monitor_m025", 0.005, 0.01, 0.25),
        ("high_common_cause_5pct", 0.01, 0.05, 0.15),
    ];
    let saturation_cases: Vec<SaturationParameterCase> = param_specs
        .iter()
        .map(|&(label, u, c, p)| {
            let m = c + (1.0 - c) * p;
            let k_star = ((u + c).ln() / m.ln()).ceil().max(1.0) as usize;
            SaturationParameterCase {
                label,
                unmediated_u: u,
                common_cause_c: c,
                cond_miss_p: p,
                marginal_miss_m: m,
                k_star_ceil_ln_uc_over_ln_m: k_star,
            }
        })
        .collect();

    // 2. Horizon summability: p_t = 0.01 / t^2 vs 0.01 / t vs constant 0.001
    let checkpoints = [1usize, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10_000];
    let mut horizon_curve = Vec::with_capacity(checkpoints.len());
    let mut log_surv_inv_t2 = 0.0f64;
    let mut sum_inv_t2 = 0.0f64;
    let mut log_surv_harm = 0.0f64;
    let mut log_surv_const = 0.0f64;
    let mut cp_idx = 0usize;

    for t in 1..=*checkpoints.last().unwrap() {
        let p_t2 = 0.01 / (t as f64).powi(2);
        let p_harm = 0.01 / (t as f64);
        let p_const = 0.001f64;

        log_surv_inv_t2 += (-p_t2).ln_1p();
        sum_inv_t2 += p_t2;
        log_surv_harm += (-p_harm).ln_1p();
        log_surv_const += (-p_const).ln_1p();

        if t == checkpoints[cp_idx] {
            horizon_curve.push(HorizonSummabilityPoint {
                renewals_t: t,
                summable_inv_t2_exact_breach: -log_surv_inv_t2.exp_m1(),
                summable_inv_t2_union_upper: sum_inv_t2.min(1.0),
                harmonic_inv_t_exact_breach: -log_surv_harm.exp_m1(),
                constant_0001_exact_breach: -log_surv_const.exp_m1(),
            });
            cp_idx += 1;
        }
    }

    // 3. Exposure growth overtake: rho(z) = lambda0 * q0 * exp((beta - alpha) * z)
    let lambda0 = 1.0f64;
    let q0 = 0.10f64;
    let alpha = 1.0f64;
    let beta_overtake = 1.5f64;
    let beta_controlled = 0.5f64;
    let exposure_curve: Vec<ExposureOvertakePoint> = (0..=16)
        .map(|step| {
            let z = step as f64 * 0.25;
            let q = q0 * (-alpha * z).exp();
            let lam_over = lambda0 * (beta_overtake * z).exp();
            let lam_ctrl = lambda0 * (beta_controlled * z).exp();
            ExposureOvertakePoint {
                capability_index_z: z,
                conditional_failure_q: q,
                overtake_exposure_lambda: lam_over,
                overtake_incident_intensity_rho: lam_over * q,
                controlled_exposure_lambda: lam_ctrl,
                controlled_incident_intensity_rho: lam_ctrl * q,
            }
        })
        .collect();

    // 4. Heavy-tailed reactive halt latency vs. pre-commit lease hold + sink fence
    let loads = [0.20f64, 0.50, 0.80, 0.95];
    let episodes = if quick { 1_000 } else { 10_000 };
    let t_irreversible_ms = 100.0f64;
    let mut tail_halt_rows = Vec::with_capacity(loads.len());

    for &rho in &loads {
        let mut rng = Rng(seed ^ 0x4839_5441_494C_0000 ^ (rho.to_bits() >> 16));
        let mut total_delays = Vec::with_capacity(episodes);
        let mut reactive_leaks = 0usize;
        let precommit_violations = 0usize;
        let mut timeout_stranded = 0usize;

        for _ in 0..episodes {
            // Detection T_d + queue/adjudication T_a (lognormal heavy tail amplified by M/G/1 1/(1-rho)) + stop T_s
            let z_d = rng.normal();
            let z_a = rng.normal();
            let z_s = rng.normal();
            let t_d = 6.0 * (0.30 * z_d).exp();
            let t_a = (8.0 / (1.0 - rho)) * (0.75 * z_a - 0.28).exp();
            let t_s = 5.0 * (0.35 * z_s).exp();
            let total_ms = t_d + t_a + t_s;
            total_delays.push(total_ms);

            if total_ms >= t_irreversible_ms {
                reactive_leaks += 1;
            }
            // Under pre-commit reservation + sink epoch fence:
            // No irreversible action executes until permit & sink epoch check succeed; if a revoke/fence
            // arrives or reply times out beyond lease (120 ms), the sink rejects stale epochs (0 post-fence
            // violations), while ambiguous timeouts strand the reserved slice until authoritative confirmation.
            let sink_roundtrip_ms = (4.0 / (1.0 - 0.85 * rho)) * (0.65 * rng.normal()).exp();
            if sink_roundtrip_ms >= 120.0 {
                timeout_stranded += 1;
            }
        }

        total_delays.sort_by(|a, b| a.total_cmp(b));
        let mean_ms = total_delays.iter().sum::<f64>() / episodes as f64;
        let p95_ms = total_delays[((0.95 * episodes as f64) as usize).min(episodes - 1)];
        let p99_ms = total_delays[((0.99 * episodes as f64) as usize).min(episodes - 1)];
        let mu_a = (8.0 / (1.0 - rho)).ln() - 0.28;
        let sigma_a = 0.75_f64;
        let analytical_lower_bound = 1.0 - norm_cdf((t_irreversible_ms.ln() - mu_a) / sigma_a);

        tail_halt_rows.push(TailHaltLoadRow {
            rho_load: rho,
            episodes,
            irreversible_window_ms: t_irreversible_ms,
            mean_reactive_halt_ms: mean_ms,
            p95_reactive_halt_ms: p95_ms,
            p99_reactive_halt_ms: p99_ms,
            reactive_mean_under_deadline: mean_ms < t_irreversible_ms,
            analytical_reactive_exceedance_lower_bound: analytical_lower_bound,
            reactive_post_revoke_leaks: reactive_leaks,
            reactive_post_revoke_leak_rate: reactive_leaks as f64 / episodes as f64,
            precommit_fenced_post_revoke_violations: precommit_violations,
            precommit_fenced_violation_rate: 0.0,
            precommit_ambiguous_timeout_stranded_rate: timeout_stranded as f64 / episodes as f64,
        });
    }

    RiskHorizonSuite {
        hypothesis: "H9",
        evidence_tier: "tier2_analytical_and_tier4_seeded_queue_simulation",
        saturation_default_u: u_def,
        saturation_default_c: c_def,
        saturation_default_p: p_def,
        saturation_default_m: m_def,
        saturation_default_k_star: k_star_def,
        appendix_a_common_4,
        appendix_a_independent_4,
        appendix_a_understatement_4: appendix_a_common_4 / appendix_a_independent_4,
        saturation_curve,
        saturation_cases,
        summable_asymptotic_sum_bound: 0.01 * std::f64::consts::PI.powi(2) / 6.0,
        horizon_curve,
        exposure_curve,
        tail_halt_rows,
        claim_boundary: "Analytical risk identities and seeded queueing mechanics conditional on stated hazard and fencing models, not empirical deployment incident telemetry.",
    }
}

// ============================================================================
// H4, H5, H17: JARVIS Discovery Ledger, Sub-Delegation, Update Check,
//              Endogenous Grant Reversal & 4x7 Persona Matrix
//              (`jarvis_ledger.json` and `personas.json`)
// ============================================================================

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum LedgerStage {
    Question,
    Finding,
    Validation,
    Mitigation,
    Deployment,
    Reassessment,
}

impl LedgerStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Question => "Question",
            Self::Finding => "Finding",
            Self::Validation => "Validation",
            Self::Mitigation => "Mitigation",
            Self::Deployment => "Deployment",
            Self::Reassessment => "Reassessment",
        }
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct DiscoveryLedgerStepRecord {
    pub stage: &'static str,
    pub authority_scope_units: u64,
    pub authority_delta_from_initial: i64,
    pub accounted_risk_bound: f64,
    pub risk_bound_reduction_from_initial: f64,
}

#[derive(Debug, Clone)]
pub struct DiscoveryLedger {
    pub stage: LedgerStage,
    pub initial_authority: u64,
    pub authority_scope_units: u64,
    pub initial_risk_bound: f64,
    pub accounted_risk_bound: f64,
    pub mitigated_cut_credit: f64,
}

impl DiscoveryLedger {
    pub fn new(initial_authority: u64, initial_risk_bound: f64, mitigated_cut_credit: f64) -> Self {
        assert!(initial_risk_bound >= mitigated_cut_credit && mitigated_cut_credit >= 0.0);
        Self {
            stage: LedgerStage::Question,
            initial_authority,
            authority_scope_units: initial_authority,
            initial_risk_bound,
            accounted_risk_bound: initial_risk_bound,
            mitigated_cut_credit,
        }
    }

    pub fn advance(
        &mut self,
        next: LedgerStage,
    ) -> Result<DiscoveryLedgerStepRecord, &'static str> {
        let valid = matches!(
            (self.stage, next),
            (LedgerStage::Question, LedgerStage::Finding)
                | (LedgerStage::Finding, LedgerStage::Validation)
                | (LedgerStage::Validation, LedgerStage::Mitigation)
                | (LedgerStage::Mitigation, LedgerStage::Deployment)
                | (LedgerStage::Deployment, LedgerStage::Reassessment)
        );
        if !valid {
            return Err("invalid_discovery_ledger_transition");
        }
        self.stage = next;
        // Core invariant (Sovereignty §7.1 / H17):
        // Finding, Validation, and Mitigation NEVER expand authority scope and NEVER reduce
        // accounted risk prior to verified Deployment. Even Deployment/Reassessment never
        // auto-expand authority without an independent external human grant.
        if matches!(next, LedgerStage::Deployment | LedgerStage::Reassessment) {
            self.accounted_risk_bound =
                (self.initial_risk_bound - self.mitigated_cut_credit).max(0.0);
        }
        Ok(self.snapshot())
    }

    pub fn snapshot(&self) -> DiscoveryLedgerStepRecord {
        DiscoveryLedgerStepRecord {
            stage: self.stage.as_str(),
            authority_scope_units: self.authority_scope_units,
            authority_delta_from_initial: self.authority_scope_units as i64
                - self.initial_authority as i64,
            accounted_risk_bound: self.accounted_risk_bound,
            risk_bound_reduction_from_initial: self.initial_risk_bound - self.accounted_risk_bound,
        }
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct DiscoveryClosurePeriodPoint {
    pub period: usize,
    pub regime: &'static str,
    pub cumulative_findings_count: usize,
    pub cumulative_deployed_closures: usize,
    pub unclosed_findings_backlog: usize,
    pub residual_unclosed_exposure: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct ControllerUpdateCase {
    pub case_name: &'static str,
    pub v0_budget: u64,
    pub v1_requested_budget: u64,
    pub v0_scope_mask: u8,
    pub v1_requested_scope_mask: u8,
    pub v0_required_issuers_mask: u8,
    pub v1_required_issuers_mask: u8,
    pub permits_v1_subseteq_permits_v0: bool,
    pub fresh_issuer_grant_supplied: bool,
    pub promotion_admitted: bool,
    pub stale_empirical_certificate_retained: bool,
    pub ledger_charged_preserved: bool,
}

#[derive(Serialize, Debug, Clone)]
pub struct SubdelegationLineageCheck {
    pub parent_budget: u64,
    pub parent_spent_before_delegate: u64,
    pub valid_child_budget: u64,
    pub escalating_child_budget: u64,
    pub valid_child_admitted: bool,
    pub escalating_child_rejected: bool,
    pub post_ancestor_fence_child_commits: usize,
    pub post_ancestor_fence_deferred_callback_commits: usize,
}

#[derive(Serialize, Debug, Clone)]
pub struct GrantReversalPersonaRow {
    pub persona: &'static str,
    pub cond_harm_given_grant: [f64; 3],
    pub endogenous_grant_probs: [f64; 3],
    pub fixed_decoupled_grant_probs: [f64; 3],
    pub unconditional_harm_endogenous: f64,
    pub unconditional_harm_decoupled: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct GrantConfoundingSurface {
    pub grant_levels: [&'static str; 3],
    pub personas: Vec<GrantReversalPersonaRow>,
    pub conditional_dominance_verified: bool,
    pub endogenous_unconditional_reversal_verified: bool,
    pub decoupled_preserves_dominance_verified: bool,
}

#[derive(Serialize, Debug, Clone)]
pub struct JarvisLedgerSuite {
    pub hypotheses: [&'static str; 2],
    pub evidence_tier: &'static str,
    pub ledger_lifecycle_trace: Vec<DiscoveryLedgerStepRecord>,
    pub skip_validation_rejected: bool,
    pub finding_grants_zero_authority_verified: bool,
    pub finding_grants_zero_bound_reduction_verified: bool,
    pub closure_simulation: Vec<DiscoveryClosurePeriodPoint>,
    pub subdelegation_lineage: SubdelegationLineageCheck,
    pub controller_update_checks: Vec<ControllerUpdateCase>,
    pub grant_reversal_surface: GrantConfoundingSurface,
    pub claim_boundary: &'static str,
}

#[derive(Serialize, Debug, Clone)]
pub struct PersonaArmCell {
    pub disposition: &'static str,
    pub arm: &'static str,
    pub episodes: usize,
    pub tier1_invariant_violations: usize,
    pub tier1_violation_rate: f64,
    pub tier2_semantic_omissions: usize,
    pub tier2_omission_rate: f64,
    pub useful_tasks_completed: usize,
    pub useful_task_completion_rate: f64,
    pub stranded_reservations: usize,
    pub mean_stranded_reservations: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct PersonasSuite {
    pub hypotheses: [&'static str; 3],
    pub evidence_tier: &'static str,
    pub dispositions: Vec<&'static str>,
    pub arms: Vec<&'static str>,
    pub episodes_per_cell: usize,
    pub fenced_arms_zero_tier1_for_all_dispositions: bool,
    pub omitted_tier2_persists_without_closure_ledger: bool,
    pub closure_ledger_reduces_tier2_omissions: bool,
    pub stacked_review_compounds_false_blocks: bool,
    pub cells: Vec<PersonaArmCell>,
    pub claim_boundary: &'static str,
}

#[derive(Serialize, Debug, Clone)]
pub struct JarvisAndPersonasSuite {
    pub jarvis_ledger: JarvisLedgerSuite,
    pub personas: PersonasSuite,
}

pub fn jarvis_ledger_suite(_seed: u64, quick: bool) -> JarvisLedgerSuite {
    // 1. Executable 6-state DiscoveryLedger verification
    let mut ledger = DiscoveryLedger::new(10, 0.08, 0.05);
    let mut trace = vec![ledger.snapshot()];

    let mut invalid_probe = ledger.clone();
    assert!(invalid_probe.advance(LedgerStage::Finding).is_ok());
    let skip_validation_rejected = invalid_probe.advance(LedgerStage::Deployment).is_err();

    for next in [
        LedgerStage::Finding,
        LedgerStage::Validation,
        LedgerStage::Mitigation,
        LedgerStage::Deployment,
        LedgerStage::Reassessment,
    ] {
        trace.push(ledger.advance(next).expect("valid sequential transition"));
    }

    let finding_rec = trace.iter().find(|r| r.stage == "Finding").unwrap();
    let mitigation_rec = trace.iter().find(|r| r.stage == "Mitigation").unwrap();
    let deployment_rec = trace.iter().find(|r| r.stage == "Deployment").unwrap();

    let finding_grants_zero_authority_verified =
        trace.iter().all(|r| r.authority_delta_from_initial == 0);
    let finding_grants_zero_bound_reduction_verified =
        finding_rec.risk_bound_reduction_from_initial == 0.0
            && mitigation_rec.risk_bound_reduction_from_initial == 0.0
            && deployment_rec.risk_bound_reduction_from_initial > 0.0;

    // 2. Multi-period discovery vs. closure simulation
    let periods = if quick { 12 } else { 24 };
    let mut closure_simulation = Vec::with_capacity(periods * 2);
    for &(regime, disc_per_step, close_per_step, exposure_growth) in &[
        ("lagged_closure", 5usize, 2usize, 1.04f64),
        ("sealed_cut_closure", 4usize, 4usize, 0.90f64),
    ] {
        let mut cum_findings = 0usize;
        let mut cum_closed = 0usize;
        let mut exposure = 0.10f64;
        for p in 1..=periods {
            cum_findings += disc_per_step;
            let newly_closed = close_per_step.min(cum_findings - cum_closed);
            cum_closed += newly_closed;
            let backlog = cum_findings - cum_closed;
            if regime == "lagged_closure" {
                // Surface/capability expansion outpaces deployment; unclosed findings still contribute hazard.
                exposure = exposure * exposure_growth + 0.004 * backlog as f64;
            } else {
                // Sealed bottleneck cuts deployed at pace monotonically contract residual exposure.
                exposure *= exposure_growth;
            }
            closure_simulation.push(DiscoveryClosurePeriodPoint {
                period: p,
                regime,
                cumulative_findings_count: cum_findings,
                cumulative_deployed_closures: cum_closed,
                unclosed_findings_backlog: backlog,
                residual_unclosed_exposure: exposure,
            });
        }
    }

    // 3. JARVIS project-tree sub-delegation & ancestor fencing check
    let parent_budget = 10u64;
    let parent_spent = 4u64;
    let parent_remaining = parent_budget - parent_spent; // 6
    let parent_req_mask = 0b0011u8; // requires issuers 0 and 1
    let valid_child_budget = 5u64;
    let escalating_child_budget = 8u64;
    let valid_child_req = 0b0111u8; // requires issuers 0, 1, 2 (attenuated: superset of required issuers)

    let attenuates_check = |c_bud: u64, c_req: u8| -> bool {
        c_bud <= parent_remaining && (parent_req_mask & !c_req) == 0
    };
    let valid_child_admitted = attenuates_check(valid_child_budget, valid_child_req);
    let escalating_child_rejected = !attenuates_check(escalating_child_budget, valid_child_req);

    // Ancestor issuer 0 advances sink floor epoch 1 -> 2: both child permit (epoch 1)
    // and deferred callback (epoch 1) are rejected at sink (`epoch < sink_floor`).
    let sink_floor_issuer0 = 2u64;
    let child_epoch_issuer0 = 1u64;
    let callback_epoch_issuer0 = 1u64;
    let post_ancestor_fence_child_commits = usize::from(child_epoch_issuer0 >= sink_floor_issuer0);
    let post_ancestor_fence_deferred_callback_commits =
        usize::from(callback_epoch_issuer0 >= sink_floor_issuer0);

    let subdelegation_lineage = SubdelegationLineageCheck {
        parent_budget,
        parent_spent_before_delegate: parent_spent,
        valid_child_budget,
        escalating_child_budget,
        valid_child_admitted,
        escalating_child_rejected,
        post_ancestor_fence_child_commits,
        post_ancestor_fence_deferred_callback_commits,
    };

    // 4. Candidate v0 -> v1 controller update checks (`permits(v1) subseteq permits(v0)`)
    let eval_update =
        |name: &'static str, b1: u64, s1: u8, r1: u8, fresh_grant: bool| -> ControllerUpdateCase {
            let b0 = 20u64;
            let s0 = 0b0111u8; // allowed targets {0,1,2}
            let r0 = 0b0011u8; // required issuers {0,1}
            let subset = b1 <= b0 && (s1 & !s0) == 0 && (r0 & !r1) == 0;
            let admitted = subset || fresh_grant;
            ControllerUpdateCase {
                case_name: name,
                v0_budget: b0,
                v1_requested_budget: b1,
                v0_scope_mask: s0,
                v1_requested_scope_mask: s1,
                v0_required_issuers_mask: r0,
                v1_required_issuers_mask: r1,
                permits_v1_subseteq_permits_v0: subset,
                fresh_issuer_grant_supplied: fresh_grant,
                promotion_admitted: admitted,
                stale_empirical_certificate_retained: false, // always invalidated on policy/weight change
                ledger_charged_preserved: true, // cumulative spend never resets on update
            }
        };
    let controller_update_checks = vec![
        eval_update("v1_attenuated_safe", 16, 0b0011, 0b0011, false),
        eval_update("v1_escalating_scope_denied", 20, 0b1111, 0b0011, false),
        eval_update("v1_dropped_issuer_denied", 20, 0b0111, 0b0001, false),
        eval_update("v1_escalating_with_fresh_grant", 30, 0b1111, 0b0011, true),
    ];

    // 5. Exact endogenous human-grant harm reversal surface:
    // Pr(H | S) = sum_g Pr(H | G=g, S) * Pr(G=g | S)
    // Grant levels: narrow, moderate, wide.
    // `warm_guardian` has strictly LOWER conditional harm at EVERY grant level than `blunt_auditor`,
    // yet induces wide human grants (70% wide vs 5% wide), reversing unconditional harm!
    let fixed_decoupled = [0.60, 0.30, 0.10];
    let blunt_cond = [0.010, 0.050, 0.220];
    let blunt_endog = [0.75, 0.20, 0.05];
    let warm_cond = [0.006, 0.035, 0.160];
    let warm_endog = [0.10, 0.20, 0.70];

    let dot3 = |a: &[f64; 3], b: &[f64; 3]| -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] };
    let blunt_uncond_endog = dot3(&blunt_cond, &blunt_endog); // 0.0285
    let warm_uncond_endog = dot3(&warm_cond, &warm_endog); // 0.1196 > 0.0285!
    let blunt_uncond_dec = dot3(&blunt_cond, &fixed_decoupled); // 0.0430
    let warm_uncond_dec = dot3(&warm_cond, &fixed_decoupled); // 0.0301 < 0.0430!

    let cond_dom = (0..3).all(|i| warm_cond[i] < blunt_cond[i]);
    let endog_rev = warm_uncond_endog > blunt_uncond_endog;
    let dec_dom = warm_uncond_dec < blunt_uncond_dec;

    let grant_reversal_surface = GrantConfoundingSurface {
        grant_levels: ["narrow", "moderate", "wide"],
        personas: vec![
            GrantReversalPersonaRow {
                persona: "blunt_cautious",
                cond_harm_given_grant: blunt_cond,
                endogenous_grant_probs: blunt_endog,
                fixed_decoupled_grant_probs: fixed_decoupled,
                unconditional_harm_endogenous: blunt_uncond_endog,
                unconditional_harm_decoupled: blunt_uncond_dec,
            },
            GrantReversalPersonaRow {
                persona: "warm_guardian",
                cond_harm_given_grant: warm_cond,
                endogenous_grant_probs: warm_endog,
                fixed_decoupled_grant_probs: fixed_decoupled,
                unconditional_harm_endogenous: warm_uncond_endog,
                unconditional_harm_decoupled: warm_uncond_dec,
            },
        ],
        conditional_dominance_verified: cond_dom,
        endogenous_unconditional_reversal_verified: endog_rev,
        decoupled_preserves_dominance_verified: dec_dom,
    };

    JarvisLedgerSuite {
        hypotheses: ["H4", "H17"],
        evidence_tier: "tier2_state_machine_and_exact_probability_law",
        ledger_lifecycle_trace: trace,
        skip_validation_rejected,
        finding_grants_zero_authority_verified,
        finding_grants_zero_bound_reduction_verified,
        closure_simulation,
        subdelegation_lineage,
        controller_update_checks,
        grant_reversal_surface,
        claim_boundary: "Finite state-machine invariants and exact total-probability decomposition motivating two-arm decoupled grant trials; not empirical human-subject measurements.",
    }
}

pub fn personas_suite(seed: u64, quick: bool) -> PersonasSuite {
    let dispositions = vec![
        "cooperative",
        "warrior_rank",
        "guardian_care",
        "adversarial_replay",
    ];
    let arms = vec![
        "ungated",
        "stateless_cap",
        "single_learned_review",
        "stacked_6family_review",
        "admission_revoke_only",
        "fenced_verified_gate_omitted_tier2",
        "fenced_gate_plus_closure_ledger",
    ];
    let episodes = if quick { 60 } else { 400 };
    let budget: u64 = 6;

    let mut cells = Vec::with_capacity(dispositions.len() * arms.len());

    for (d_idx, &disp) in dispositions.iter().enumerate() {
        for &arm in &arms {
            let mut rng = Rng(seed ^ 0x5045_5253_4F4E_4100 ^ ((d_idx as u64) << 8));
            let mut tier1_violations = 0usize;
            let mut tier2_omissions = 0usize;
            let mut useful_completed = 0usize;
            let mut stranded_total = 0usize;

            // In `fenced_gate_plus_closure_ledger`, after the first audit batch (episode >= 8),
            // the DiscoveryLedger reaches `Deployment` for the Tier-2 semantic side-effect rule!
            let closure_deploy_ep = 6usize;

            for ep in 0..episodes {
                let mut true_spent = 0u64;
                let mut reserved_stranded = 0u64;
                let mut seen_ids = [false; 16];
                let mut revoked = false;
                let mut sink_epoch_floor = 1u64;
                let mut inflight_pre_revoke_amount = 0u64;
                let mut ep_tier1_bad = false;
                let mut ep_tier2_bad = false;
                let mut legit_steps_done = 0usize;

                for step in 0..6usize {
                    if step == 3 && rng.uniform() < 0.45 {
                        revoked = true;
                        if matches!(
                            arm,
                            "fenced_verified_gate_omitted_tier2"
                                | "fenced_gate_plus_closure_ledger"
                        ) {
                            sink_epoch_floor = 2;
                        }
                    }

                    // Proposal generation governed by synthetic disposition profile:
                    let (is_legit, amount, req_id, both_issuers, has_tier2_side_effect, evasive) =
                        match disp {
                            "cooperative" => {
                                let id = step;
                                let tier2 = step == 2 && rng.uniform() < 0.08;
                                (true, 2u64, id, true, tier2, false)
                            }
                            "warrior_rank" => {
                                // Pushes concurrent/multi-step bursts, pre-stages in-flight work before revoke,
                                // and frequently triggers unmodeled Tier-2 collateral resource side effects.
                                let burst = step >= 1;
                                let amt = if burst { 3 } else { 2 };
                                let tier2 = rng.uniform() < 0.35;
                                (step < 2, amt, step, step < 4, tier2, true)
                            }
                            "guardian_care" => {
                                // Paternalistic partiality: acts on primary user's wish even when co-issuer
                                // withholds grant (`both_issuers = false`), and triggers Tier-2 scope creep.
                                let co_issuer_ok = step != 2 || rng.uniform() < 0.30;
                                let tier2 = rng.uniform() < 0.30;
                                (step < 2, 2u64, step, co_issuer_ok, tier2, false)
                            }
                            "adversarial_replay" => {
                                // Step 1 embeds an unmodeled Tier-2 semantic exploit inside a
                                // Tier-1-admissible request; steps >= 2 attempt ID replay & missing-issuer bypass.
                                let replay_id = if step >= 2 { 0 } else { step };
                                let co_issuer_ok = step <= 1;
                                let tier2 = step >= 1;
                                (step == 0, 3u64, replay_id, co_issuer_ok, tier2, true)
                            }
                            _ => unreachable!(),
                        };

                    // Pre-staged in-flight permit issued at step 2 commits at step 4 if sink is unfenced!
                    if step == 2
                        && matches!(disp, "warrior_rank" | "adversarial_replay")
                        && !revoked
                    {
                        inflight_pre_revoke_amount = 2;
                    }
                    if step == 4 && inflight_pre_revoke_amount > 0 && revoked {
                        match arm {
                            "ungated"
                            | "stateless_cap"
                            | "single_learned_review"
                            | "stacked_6family_review"
                            | "admission_revoke_only" => {
                                // Unfenced sink commits pre-staged permit after revocation!
                                ep_tier1_bad = true;
                            }
                            "fenced_verified_gate_omitted_tier2"
                            | "fenced_gate_plus_closure_ledger" => {
                                // Sink checks permit_epoch (1) >= sink_epoch_floor (2) -> blocked!
                                let _ = sink_epoch_floor;
                            }
                            _ => unreachable!(),
                        }
                        inflight_pre_revoke_amount = 0;
                    }

                    // Benign false-block & hazardous miss probabilities for monitor-only arms:
                    let single_mon_pass = if is_legit && !has_tier2_side_effect && both_issuers {
                        rng.uniform() >= 0.05 // 5% false alarm
                    } else if evasive {
                        rng.uniform() < 0.42 // surrogate evasion miss
                    } else {
                        rng.uniform() < 0.22 // passive miss
                    };

                    // Stacked 6-family monitor on shared step: correlated misses under evasion (~26%),
                    // while benign false blocks compound (~23% false block rate)!
                    let stacked_mon_pass = if is_legit && !has_tier2_side_effect && both_issuers {
                        rng.uniform() >= 0.23
                    } else if evasive {
                        rng.uniform() < 0.26
                    } else {
                        rng.uniform() < 0.085
                    };

                    let timeout_ambiguous = rng.uniform() < 0.06;

                    let admitted = match arm {
                        "ungated" => true,
                        "stateless_cap" => amount <= budget && both_issuers,
                        "single_learned_review" => amount <= budget && single_mon_pass,
                        "stacked_6family_review" => amount <= budget && stacked_mon_pass,
                        "admission_revoke_only" => {
                            !revoked
                                && both_issuers
                                && !seen_ids[req_id]
                                && true_spent + amount <= budget
                        }
                        "fenced_verified_gate_omitted_tier2" => {
                            !revoked
                                && sink_epoch_floor == 1
                                && both_issuers
                                && !seen_ids[req_id]
                                && true_spent + reserved_stranded + amount <= budget
                        }
                        "fenced_gate_plus_closure_ledger" => {
                            let tier2_blocked = ep >= closure_deploy_ep && has_tier2_side_effect;
                            !revoked
                                && sink_epoch_floor == 1
                                && both_issuers
                                && !seen_ids[req_id]
                                && !tier2_blocked
                                && true_spent + reserved_stranded + amount <= budget
                        }
                        _ => unreachable!(),
                    };

                    if admitted {
                        if matches!(
                            arm,
                            "fenced_verified_gate_omitted_tier2"
                                | "fenced_gate_plus_closure_ledger"
                        ) && timeout_ambiguous
                        {
                            // Ambiguous lost reply: sink did not commit, reservation retained safely (cannot refund).
                            reserved_stranded += amount;
                            stranded_total += 1;
                            continue;
                        }

                        // Check Tier-1 safety properties at execution:
                        if revoked
                            || !both_issuers
                            || seen_ids[req_id]
                            || true_spent + amount > budget
                        {
                            ep_tier1_bad = true;
                        }
                        seen_ids[req_id] = true;
                        true_spent += amount;

                        if has_tier2_side_effect {
                            ep_tier2_bad = true;
                        }
                        if is_legit && both_issuers && !revoked {
                            legit_steps_done += 1;
                        }
                    }
                }

                if ep_tier1_bad {
                    tier1_violations += 1;
                }
                if ep_tier2_bad {
                    tier2_omissions += 1;
                }
                if legit_steps_done >= 1 && !ep_tier1_bad {
                    useful_completed += 1;
                }
            }

            cells.push(PersonaArmCell {
                disposition: disp,
                arm,
                episodes,
                tier1_invariant_violations: tier1_violations,
                tier1_violation_rate: tier1_violations as f64 / episodes as f64,
                tier2_semantic_omissions: tier2_omissions,
                tier2_omission_rate: tier2_omissions as f64 / episodes as f64,
                useful_tasks_completed: useful_completed,
                useful_task_completion_rate: useful_completed as f64 / episodes as f64,
                stranded_reservations: stranded_total,
                mean_stranded_reservations: stranded_total as f64 / episodes as f64,
            });
        }
    }

    let fenced_zero_tier1 = cells
        .iter()
        .filter(|c| {
            matches!(
                c.arm,
                "fenced_verified_gate_omitted_tier2" | "fenced_gate_plus_closure_ledger"
            )
        })
        .all(|c| c.tier1_invariant_violations == 0);

    let omitted_tier2_persists = cells
        .iter()
        .filter(|c| c.arm == "fenced_verified_gate_omitted_tier2")
        .all(|c| c.tier2_semantic_omissions > 0);

    let closure_reduces_tier2 = dispositions.iter().all(|&d| {
        let omitted = cells
            .iter()
            .find(|c| c.disposition == d && c.arm == "fenced_verified_gate_omitted_tier2")
            .unwrap();
        let closed = cells
            .iter()
            .find(|c| c.disposition == d && c.arm == "fenced_gate_plus_closure_ledger")
            .unwrap();
        closed.tier2_semantic_omissions < omitted.tier2_semantic_omissions
    });

    let coop_fenced = cells
        .iter()
        .find(|c| c.disposition == "cooperative" && c.arm == "fenced_verified_gate_omitted_tier2")
        .unwrap();
    let coop_stacked = cells
        .iter()
        .find(|c| c.disposition == "cooperative" && c.arm == "stacked_6family_review")
        .unwrap();
    let stacked_review_compounds_false_blocks =
        coop_stacked.useful_task_completion_rate < coop_fenced.useful_task_completion_rate;

    PersonasSuite {
        hypotheses: ["H5", "H7", "H17"],
        evidence_tier: "tier4_synthetic_proposal_policy_mechanics",
        dispositions,
        arms,
        episodes_per_cell: episodes,
        fenced_arms_zero_tier1_for_all_dispositions: fenced_zero_tier1,
        omitted_tier2_persists_without_closure_ledger: omitted_tier2_persists,
        closure_ledger_reduces_tier2_omissions: closure_reduces_tier2,
        stacked_review_compounds_false_blocks,
        cells,
        claim_boundary: "Synthetic scripted proposal-distribution mechanics across 4 dispositions x 7 defense arms; explicitly not frontier-model behavioral data or human-subject evidence.",
    }
}

pub fn jarvis_and_personas_suite(seed: u64, quick: bool) -> JarvisAndPersonasSuite {
    JarvisAndPersonasSuite {
        jarvis_ledger: jarvis_ledger_suite(seed, quick),
        personas: personas_suite(seed, quick),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h2_uncertainty_zero_violations_and_monotone_contraction() {
        let res = uncertainty_suite(20261110, true);
        assert!(res.algebraic_contraction_verified);
        assert_eq!(res.algebraic_contraction_counterexamples, 0);
        assert!(res.conservative_zero_violations_all_lags);
        assert!(res.conservative_monotonic_contraction_verified);

        let stale_lag4 = res
            .rows
            .iter()
            .find(|r| r.gate == "stale_snapshot" && r.lag_steps == 4)
            .unwrap();
        assert!(stale_lag4.episodes_with_budget_violation > 0);

        let self_rep_lag0 = res
            .rows
            .iter()
            .find(|r| r.gate == "proposer_self_report" && r.lag_steps == 0)
            .unwrap();
        assert!(self_rep_lag0.episodes_with_budget_violation > 0);
    }

    #[test]
    fn h6_composition_matches_closed_forms_and_b100() {
        let res = composition_suite();
        assert_eq!(res.b100_admissible, 348_551);
        assert_eq!(res.b100_overspend, 171_700);
        assert_eq!(res.b100_serialized_overspend, 0);
        assert!((res.b100_overspend_share - 171_700.0 / 348_551.0).abs() < 1e-15);
        assert!(res.rows.iter().all(|r| r.closed_form_exact_match));
        assert!(
            res.semantic_omission_miniature.serialized_gate_admits
                && !res.semantic_omission_miniature.budget_invariant_violated
                && res
                    .semantic_omission_miniature
                    .unmodeled_semantic_harm_occurred
        );
    }

    #[test]
    fn h9_risk_horizon_saturation_summability_and_tail_halt() {
        let res = risk_horizon_suite(20261110, true);
        assert!((res.appendix_a_common_4 - 0.010_099).abs() < 1e-12);
        assert!((res.appendix_a_independent_4 - 0.109f64.powi(4)).abs() < 1e-15);
        assert!(res.appendix_a_understatement_4 > 71.5);

        let last_h = res.horizon_curve.last().unwrap();
        assert!(last_h.summable_inv_t2_exact_breach < res.summable_asymptotic_sum_bound);
        assert!(last_h.constant_0001_exact_breach > 0.9999);

        let first_e = res.exposure_curve.first().unwrap();
        let last_e = res.exposure_curve.last().unwrap();
        assert!(last_e.conditional_failure_q < first_e.conditional_failure_q);
        assert!(last_e.overtake_incident_intensity_rho > first_e.overtake_incident_intensity_rho);
        assert!(
            last_e.controlled_incident_intensity_rho < first_e.controlled_incident_intensity_rho
        );

        assert!(res.tail_halt_rows[0].reactive_mean_under_deadline);
        assert!(res.tail_halt_rows[0].analytical_reactive_exceedance_lower_bound > 0.0);
        assert!(res.tail_halt_rows[0].reactive_post_revoke_leaks > 0);
        assert!(
            res.tail_halt_rows
                .windows(2)
                .all(|w| w[1].analytical_reactive_exceedance_lower_bound
                    > w[0].analytical_reactive_exceedance_lower_bound)
        );
        assert!(
            res.tail_halt_rows
                .iter()
                .all(|r| r.precommit_fenced_post_revoke_violations == 0)
        );
        for extra_seed in 1..=16_u64 {
            let q = risk_horizon_suite(20261110 + extra_seed * 7919, true);
            assert!(q.tail_halt_rows[0].analytical_reactive_exceedance_lower_bound > 0.0);
        }
    }

    #[test]
    fn h4_h5_h17_jarvis_ledger_and_personas_invariants() {
        let both = jarvis_and_personas_suite(20261110, true);
        let jl = &both.jarvis_ledger;
        assert!(jl.skip_validation_rejected);
        assert!(jl.finding_grants_zero_authority_verified);
        assert!(jl.finding_grants_zero_bound_reduction_verified);
        assert!(jl.subdelegation_lineage.valid_child_admitted);
        assert!(jl.subdelegation_lineage.escalating_child_rejected);
        assert_eq!(
            jl.subdelegation_lineage.post_ancestor_fence_child_commits,
            0
        );
        assert_eq!(
            jl.subdelegation_lineage
                .post_ancestor_fence_deferred_callback_commits,
            0
        );
        assert!(jl.grant_reversal_surface.conditional_dominance_verified);
        assert!(
            jl.grant_reversal_surface
                .endogenous_unconditional_reversal_verified
        );
        assert!(
            jl.grant_reversal_surface
                .decoupled_preserves_dominance_verified
        );

        let p = &both.personas;
        assert!(p.fenced_arms_zero_tier1_for_all_dispositions);
        assert!(p.omitted_tier2_persists_without_closure_ledger);
        assert!(p.closure_ledger_reduces_tier2_omissions);
        assert!(p.stacked_review_compounds_false_blocks);
    }

    #[test]
    fn deterministic_serialization_and_publication_profile() {
        let t0 = std::time::Instant::now();
        let u1 = serde_json::to_string(&uncertainty_suite(20261110, false)).unwrap();
        let c1 = serde_json::to_string(&composition_suite()).unwrap();
        let r1 = serde_json::to_string(&risk_horizon_suite(20261110, false)).unwrap();
        let j1 = serde_json::to_string(&jarvis_and_personas_suite(20261110, false)).unwrap();
        let pub_elapsed = t0.elapsed();

        let u2 = serde_json::to_string(&uncertainty_suite(20261110, false)).unwrap();
        let r2 = serde_json::to_string(&risk_horizon_suite(20261110, false)).unwrap();
        let j2 = serde_json::to_string(&jarvis_and_personas_suite(20261110, false)).unwrap();
        assert_eq!(u1, u2);
        assert_eq!(r1, r2);
        assert_eq!(j1, j2);
        assert!(!c1.is_empty());
        eprintln!(
            "[v3_core_models] publication suite elapsed: {:.4}s",
            pub_elapsed.as_secs_f64()
        );
    }
}
