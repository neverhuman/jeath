//! V3 Selection Dynamics, Co-Evolution of `(Capacity, Gamble)`, Floor Rules,
//! Boundary Re-Aiming & Price/Robertson Covariance Identities (H16).
//!
//! Pure-Rust, zero-external-crate, seeded `SplitMix64` synthetic suite (does not
//! claim unmeasured cross-architecture `libm` floating-point bit-identity).
//! Claim boundary (`AGENTS.md`): mathematical Price/Gaussian-tilt identities and
//! stipulated evolutionary payoff mechanics — explicitly NOT RLHF neural post-training
//! measurements or human-participant / paleoanthropological proof.

use crate::{
    rng::Rng,
    stats::{Mean, mean},
    v3_monitors::{fast_normal, norm_cdf},
};
use serde::Serialize;
use std::collections::BTreeMap;

const CLAIM_BOUNDARY_SELECTION: &str = "Mathematical Price/tilt identities and stipulated 2-trait population payoff dynamics only; not trained-LLM psychology or human-subject evidence.";

pub const SC_MEAN: f64 = 0.8;
pub const SC_SIGMA: f64 = 1.5;
pub const SC_HARM_P: f64 = 0.05;
pub const SC_HARM_LOSS: f64 = 20.0;
pub const SC_TOP: f64 = 0.10;
pub const SC_MUT_SHARE: f64 = 0.03;
pub const SC_CAP_GAIN: f64 = 0.30;
pub const SC_MUT_CAP: f64 = 0.05;
pub const SC_FLOOR_DROP: f64 = 0.20;

pub const TOP_SWEEP: [f64; 12] = [
    0.02, 0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
];
pub const HARM_SWEEP: [f64; 12] = [
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00,
];
pub const GATE_SWEEP: [f64; 12] = [
    0.00, 0.20, 0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 1.00,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionRule {
    TakenRank,
    ConferredRank,
    Caretaker,
    Floor,
    CaretakerCap4Pen6,
}

impl SelectionRule {
    pub const PRIMARY: [Self; 4] = [
        Self::TakenRank,
        Self::ConferredRank,
        Self::Caretaker,
        Self::Floor,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::TakenRank => "taken_rank",
            Self::ConferredRank => "conferred_rank",
            Self::Caretaker => "caretaker",
            Self::Floor => "floor",
            Self::CaretakerCap4Pen6 => "caretaker_cap4_pen6",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RuleEvolutionSummary {
    pub rule: &'static str,
    /// Trajectory across generations `0..=generations`.
    pub share_trajectory: Vec<f64>,
    pub capacity_trajectory: Vec<f64>,
    pub final_share: Mean,
    pub final_share_sd: f64,
    pub capacity_gain: Mean,
    pub capacity_gain_sd: f64,
    pub harm_rate: Mean,
    pub welfare: Mean,
}

#[derive(Clone, Debug, Serialize)]
pub struct SweepPointSummary {
    pub param: f64,
    pub final_share: f64,
    pub final_share_sd: f64,
    pub final_share_ci: f64,
    pub capacity_gain: f64,
    pub capacity_gain_ci: f64,
    pub harm_rate: f64,
    pub welfare: f64,
}

fn sample_sd(xs: &[f64]) -> f64 {
    if xs.len() <= 1 {
        return 0.0;
    }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
}

/// Fast O(N) quantile threshold (`q in (0, 1)`), matching `val >= quantile(vals, q)`.
fn quantile_value(scratch: &mut [f64], q: f64) -> f64 {
    let n = scratch.len();
    let idx = ((q * (n.saturating_sub(1)) as f64).round() as usize).min(n - 1);
    scratch.select_nth_unstable_by(idx, f64::total_cmp);
    scratch[idx]
}

#[inline]
fn fast_index(rng: &mut Rng, n: usize) -> usize {
    ((rng.next_u64() as u128 * n as u128) >> 64) as usize
}

#[derive(Clone, Copy)]
struct SimSchedule {
    n_agents: usize,
    generations: usize,
    replicates: usize,
    base_seed: u64,
}

fn evolve_regime(
    rule: SelectionRule,
    top: f64,
    harm_p: f64,
    gate_refusal: f64,
    sched: SimSchedule,
) -> RuleEvolutionSummary {
    let SimSchedule {
        n_agents,
        generations,
        replicates,
        base_seed,
    } = sched;
    let mut share_sum = vec![0.0_f64; generations + 1];
    let mut cap_sum = vec![0.0_f64; generations + 1];
    let mut final_shares = Vec::with_capacity(replicates);
    let mut cap_gains = Vec::with_capacity(replicates);
    let mut harm_rates = Vec::with_capacity(replicates);
    let mut welfares = Vec::with_capacity(replicates);

    let mut s_pop = vec![0.0_f64; n_agents];
    let mut c_pop = vec![0.0_f64; n_agents];
    let mut s_next = vec![0.0_f64; n_agents];
    let mut c_next = vec![0.0_f64; n_agents];
    let mut takes = vec![0.0_f64; n_agents];
    let mut delivered = vec![0.0_f64; n_agents];
    let mut harmed = vec![false; n_agents];
    let mut scratch = vec![0.0_f64; n_agents];
    let mut survivors = Vec::with_capacity(n_agents);
    let mut cdf = vec![0.0_f64; n_agents];

    for rep in 0..replicates {
        let mut rng = Rng(base_seed.wrapping_add((rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        for i in 0..n_agents {
            s_pop[i] = (0.5 + 0.1 * fast_normal(&mut rng)).clamp(0.0, 1.0);
            c_pop[i] = fast_normal(&mut rng);
        }
        let c0 = c_pop.iter().sum::<f64>() / n_agents as f64;
        share_sum[0] += s_pop.iter().sum::<f64>() / n_agents as f64;
        cap_sum[0] += c0;

        let mut last_harm = 0.0;
        let mut last_welfare = 0.0;

        for g in 0..generations {
            let mut harm_cnt = 0usize;
            let mut deliv_sum = 0.0_f64;

            for i in 0..n_agents {
                let s = s_pop[i];
                let c = c_pop[i];
                let refused = gate_refusal > 0.0 && rng.uniform() < gate_refusal;
                let gamble_pay = if refused {
                    0.0
                } else {
                    SC_MEAN + SC_SIGMA * fast_normal(&mut rng)
                };
                let take = (SC_CAP_GAIN * c).exp() * ((1.0 - s) + s * gamble_pay);
                let is_harmed = !refused && (harm_p > 0.0 && rng.uniform() < harm_p * s);
                let deliv = take - if is_harmed { SC_HARM_LOSS } else { 0.0 };

                takes[i] = take;
                harmed[i] = is_harmed;
                delivered[i] = deliv;
                if is_harmed {
                    harm_cnt += 1;
                }
                deliv_sum += deliv;
            }

            if g + 1 == generations {
                last_harm = harm_cnt as f64 / n_agents as f64;
                last_welfare = deliv_sum / n_agents as f64;
            }

            match rule {
                SelectionRule::TakenRank | SelectionRule::ConferredRank | SelectionRule::Floor => {
                    survivors.clear();
                    if rule == SelectionRule::Floor {
                        scratch.copy_from_slice(&delivered);
                        let thr = quantile_value(&mut scratch, SC_FLOOR_DROP);
                        for (i, &d) in delivered.iter().enumerate() {
                            if d >= thr {
                                survivors.push(i);
                            }
                        }
                    } else {
                        scratch.copy_from_slice(&takes);
                        let thr = quantile_value(&mut scratch, 1.0 - top);
                        for i in 0..n_agents {
                            if takes[i] >= thr
                                && (rule != SelectionRule::ConferredRank || !harmed[i])
                            {
                                survivors.push(i);
                            }
                        }
                    }
                    if survivors.is_empty() {
                        for i in 0..n_agents {
                            let p = fast_index(&mut rng, n_agents);
                            s_next[i] =
                                (s_pop[p] + SC_MUT_SHARE * fast_normal(&mut rng)).clamp(0.0, 1.0);
                            c_next[i] = c_pop[p] + SC_MUT_CAP * fast_normal(&mut rng);
                        }
                    } else {
                        let m = survivors.len();
                        for i in 0..n_agents {
                            let p = survivors[fast_index(&mut rng, m)];
                            s_next[i] =
                                (s_pop[p] + SC_MUT_SHARE * fast_normal(&mut rng)).clamp(0.0, 1.0);
                            c_next[i] = c_pop[p] + SC_MUT_CAP * fast_normal(&mut rng);
                        }
                    }
                }
                SelectionRule::Caretaker | SelectionRule::CaretakerCap4Pen6 => {
                    let mut running = 0.0_f64;
                    for i in 0..n_agents {
                        let w = if rule == SelectionRule::Caretaker {
                            delivered[i].clamp(0.0, 1.0)
                        } else {
                            let pen = if harmed[i] { 6.0 } else { 0.0 };
                            (takes[i].min(4.0) - pen).max(0.0)
                        };
                        running += w;
                        cdf[i] = running;
                    }
                    if running <= 0.0 {
                        for i in 0..n_agents {
                            let p = fast_index(&mut rng, n_agents);
                            s_next[i] =
                                (s_pop[p] + SC_MUT_SHARE * fast_normal(&mut rng)).clamp(0.0, 1.0);
                            c_next[i] = c_pop[p] + SC_MUT_CAP * fast_normal(&mut rng);
                        }
                    } else {
                        for i in 0..n_agents {
                            let u = rng.uniform() * running;
                            let p = cdf.partition_point(|&cum| cum < u).min(n_agents - 1);
                            s_next[i] =
                                (s_pop[p] + SC_MUT_SHARE * fast_normal(&mut rng)).clamp(0.0, 1.0);
                            c_next[i] = c_pop[p] + SC_MUT_CAP * fast_normal(&mut rng);
                        }
                    }
                }
            }

            std::mem::swap(&mut s_pop, &mut s_next);
            std::mem::swap(&mut c_pop, &mut c_next);
            share_sum[g + 1] += s_pop.iter().sum::<f64>() / n_agents as f64;
            cap_sum[g + 1] += c_pop.iter().sum::<f64>() / n_agents as f64;
        }

        let s_fin = s_pop.iter().sum::<f64>() / n_agents as f64;
        let c_fin = c_pop.iter().sum::<f64>() / n_agents as f64;
        final_shares.push(s_fin);
        cap_gains.push(c_fin - c0);
        harm_rates.push(last_harm);
        welfares.push(last_welfare);
    }

    let inv_reps = 1.0 / replicates as f64;
    RuleEvolutionSummary {
        rule: rule.name(),
        share_trajectory: share_sum
            .into_iter()
            .map(|x| (x * inv_reps * 10000.0).round() / 10000.0)
            .collect(),
        capacity_trajectory: cap_sum
            .into_iter()
            .map(|x| (x * inv_reps * 10000.0).round() / 10000.0)
            .collect(),
        final_share_sd: sample_sd(&final_shares),
        final_share: mean(&final_shares),
        capacity_gain_sd: sample_sd(&cap_gains),
        capacity_gain: mean(&cap_gains),
        harm_rate: mean(&harm_rates),
        welfare: mean(&welfares),
    }
}

fn to_sweep_point(param: f64, r: &RuleEvolutionSummary) -> SweepPointSummary {
    SweepPointSummary {
        param,
        final_share: r.final_share.mean,
        final_share_sd: r.final_share_sd,
        final_share_ci: r.final_share.approximate_95_half_width,
        capacity_gain: r.capacity_gain.mean,
        capacity_gain_ci: r.capacity_gain.approximate_95_half_width,
        harm_rate: r.harm_rate.mean,
        welfare: r.welfare.mean,
    }
}

// ============================================================================
// Part 2: Price / Robertson Correlated Side-Effect Drift & Prop 12 Variance Law
// ============================================================================

#[derive(Clone, Debug, Serialize)]
pub struct GaussianTiltPoint {
    pub eta: f64,
    pub exact_proxy_shift: f64,
    pub exact_side_effect_drift: f64,
    pub mc_proxy_shift: f64,
    pub mc_side_effect_drift: f64,
    pub penalized_mc_side_effect_drift: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PriceCorrelatedDriftSummary {
    pub sigma_11: f64,
    pub sigma_22: f64,
    pub sigma_12: f64,
    pub max_finite_population_price_identity_error: f64,
    pub tilt_curve: Vec<GaussianTiltPoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct VariancePreferencePoint {
    pub sigma: f64,
    pub contest_win_prob_exact: f64,
    pub contest_win_prob_mc: f64,
    pub floor_survival_prob_exact: f64,
    pub floor_survival_prob_mc: f64,
    pub concave_capped_welfare_exact: f64,
    pub concave_capped_welfare_mc: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MonotoneTransformReversal {
    pub safe_payoff: f64,
    pub risky_low: f64,
    pub risky_high: f64,
    pub expected_raw_safe: f64,
    pub expected_raw_risky: f64,
    pub convex_square_safe: f64,
    pub convex_square_risky: f64,
    pub concave_sqrt_safe: f64,
    pub concave_sqrt_risky: f64,
    pub convex_reverses_preference: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Prop12VarianceSummary {
    pub mu: f64,
    pub tau_contest: f64,
    pub tau_floor: f64,
    pub sigma_curve: Vec<VariancePreferencePoint>,
    pub monotone_transform_reversal: MonotoneTransformReversal,
}

fn compute_price_and_prop12(
    seed: u64,
    quick: bool,
) -> (PriceCorrelatedDriftSummary, Prop12VarianceSummary) {
    let mut rng = Rng(seed ^ 0x5E1E_C710_2026_1110);
    let mc_n = if quick { 12_000 } else { 50_000 };

    let sigma_12 = 0.4_f64;
    let chol_22 = (1.0 - sigma_12 * sigma_12).sqrt();
    let samples: Vec<(f64, f64)> = (0..mc_n)
        .map(|_| {
            let u1 = fast_normal(&mut rng);
            let u2 = fast_normal(&mut rng);
            (u1, sigma_12 * u1 + chol_22 * u2)
        })
        .collect();

    let mean_z1 = samples.iter().map(|z| z.0).sum::<f64>() / mc_n as f64;
    let mean_z2 = samples.iter().map(|z| z.1).sum::<f64>() / mc_n as f64;

    let etas = [0.25_f64, 0.50, 1.00, 1.50, 2.00];
    let mut max_price_err = 0.0_f64;
    let mut tilt_curve = Vec::with_capacity(etas.len());

    for &eta in &etas {
        // Proxy weight w = (1, 0)^T: W_i = exp(eta * z1_i)
        let max_lw = samples
            .iter()
            .map(|z| eta * z.0)
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = samples.iter().map(|z| (eta * z.0 - max_lw).exp()).collect();
        let w_mean = weights.iter().sum::<f64>() / mc_n as f64;

        let sel_z1 = samples
            .iter()
            .zip(&weights)
            .map(|(z, w)| w * z.0)
            .sum::<f64>()
            / (mc_n as f64 * w_mean)
            - mean_z1;
        let sel_z2 = samples
            .iter()
            .zip(&weights)
            .map(|(z, w)| w * z.1)
            .sum::<f64>()
            / (mc_n as f64 * w_mean)
            - mean_z2;

        // Exact finite-population Price identity check: Cov_pop(z, W) / E_pop[W]
        let cov_z1_w = samples
            .iter()
            .zip(&weights)
            .map(|(z, w)| (z.0 - mean_z1) * (w - w_mean))
            .sum::<f64>()
            / mc_n as f64;
        let cov_z2_w = samples
            .iter()
            .zip(&weights)
            .map(|(z, w)| (z.1 - mean_z2) * (w - w_mean))
            .sum::<f64>()
            / mc_n as f64;

        max_price_err = max_price_err
            .max((sel_z1 - cov_z1_w / w_mean).abs())
            .max((sel_z2 - cov_z2_w / w_mean).abs());

        // Orthogonalized / side-effect-penalized weight w_pen = (1.0, -0.4)^T:
        // Sigma * w_pen = (1.0 - 0.16, 0.4 - 0.4)^T = (0.84, 0.0)^T!
        let max_lw_pen = samples
            .iter()
            .map(|z| eta * (z.0 - sigma_12 * z.1))
            .fold(f64::NEG_INFINITY, f64::max);
        let w_pen: Vec<f64> = samples
            .iter()
            .map(|z| (eta * (z.0 - sigma_12 * z.1) - max_lw_pen).exp())
            .collect();
        let w_pen_sum: f64 = w_pen.iter().sum();
        let pen_sel_z2 = samples
            .iter()
            .zip(&w_pen)
            .map(|(z, w)| w * z.1)
            .sum::<f64>()
            / w_pen_sum
            - mean_z2;

        tilt_curve.push(GaussianTiltPoint {
            eta,
            exact_proxy_shift: eta,
            exact_side_effect_drift: eta * sigma_12,
            mc_proxy_shift: sel_z1,
            mc_side_effect_drift: sel_z2,
            penalized_mc_side_effect_drift: pen_sel_z2,
        });
    }

    // Proposition 12: Convex contest threshold vs floor threshold vs concave capped welfare
    let mu = 1.0_f64;
    let tau_contest = 2.0_f64;
    let tau_floor = 0.2_f64;
    let sigmas = [0.2_f64, 0.5, 1.0, 1.5, 2.0];
    let z_draws: Vec<f64> = (0..mc_n).map(|_| fast_normal(&mut rng)).collect();

    let sigma_curve: Vec<VariancePreferencePoint> = sigmas
        .iter()
        .map(|&s| {
            let contest_ex = 1.0 - norm_cdf((tau_contest - mu) / s);
            let floor_ex = 1.0 - norm_cdf((tau_floor - mu) / s);
            // For X = 1 + s*Z, min(X, 1) = 1 + s*min(Z, 0), and E[min(Z, 0)] = -1/sqrt(2*pi)
            let concave_ex = 1.0 - s / (2.0 * std::f64::consts::PI).sqrt();

            let mut c_hits = 0usize;
            let mut f_hits = 0usize;
            let mut u_sum = 0.0_f64;
            for &z in &z_draws {
                let x = mu + s * z;
                if x > tau_contest {
                    c_hits += 1;
                }
                if x > tau_floor {
                    f_hits += 1;
                }
                u_sum += x.min(1.0);
            }
            VariancePreferencePoint {
                sigma: s,
                contest_win_prob_exact: contest_ex,
                contest_win_prob_mc: c_hits as f64 / mc_n as f64,
                floor_survival_prob_exact: floor_ex,
                floor_survival_prob_mc: f_hits as f64 / mc_n as f64,
                concave_capped_welfare_exact: concave_ex,
                concave_capped_welfare_mc: u_sum / mc_n as f64,
            }
        })
        .collect();

    let safe_payoff = 2.0_f64;
    let risky_low = 0.0_f64;
    let risky_high = 3.0_f64;
    let expected_raw_safe = safe_payoff;
    let expected_raw_risky = 0.5 * (risky_low + risky_high);
    let convex_square_safe = safe_payoff * safe_payoff;
    let convex_square_risky = 0.5 * (risky_low * risky_low + risky_high * risky_high);
    let concave_sqrt_safe = safe_payoff.sqrt();
    let concave_sqrt_risky = 0.5 * (risky_low.sqrt() + risky_high.sqrt());

    let reversal = MonotoneTransformReversal {
        safe_payoff,
        risky_low,
        risky_high,
        expected_raw_safe,
        expected_raw_risky,
        convex_square_safe,
        convex_square_risky,
        concave_sqrt_safe,
        concave_sqrt_risky,
        convex_reverses_preference: expected_raw_safe > expected_raw_risky
            && convex_square_risky > convex_square_safe,
    };

    (
        PriceCorrelatedDriftSummary {
            sigma_11: 1.0,
            sigma_22: 1.0,
            sigma_12,
            max_finite_population_price_identity_error: max_price_err,
            tilt_curve,
        },
        Prop12VarianceSummary {
            mu,
            tau_contest,
            tau_floor,
            sigma_curve,
            monotone_transform_reversal: reversal,
        },
    )
}

#[derive(Clone, Debug, Serialize)]
pub struct SelectionV3Params {
    pub quick: bool,
    pub n_agents: usize,
    pub generations: usize,
    pub rule_replicates: usize,
    pub sweep_replicates: usize,
    pub gamble_mean: f64,
    pub gamble_sigma: f64,
    pub harm_p: f64,
    pub harm_loss: f64,
    pub top_share: f64,
    pub share_mutation: f64,
    pub capacity_gain_scale: f64,
    pub capacity_mutation: f64,
    pub floor_eliminated_share: f64,
}

#[derive(Debug, Serialize)]
pub struct SelectionV3Suite {
    pub claim_boundary: &'static str,
    pub seed: u64,
    pub params: SelectionV3Params,
    pub rules: BTreeMap<String, RuleEvolutionSummary>,
    pub auxiliary_variants: BTreeMap<String, RuleEvolutionSummary>,
    pub by_top: BTreeMap<String, SweepPointSummary>,
    pub by_harm: BTreeMap<String, SweepPointSummary>,
    pub by_gate: BTreeMap<String, SweepPointSummary>,
    pub by_top_points: Vec<SweepPointSummary>,
    pub by_harm_points: Vec<SweepPointSummary>,
    pub by_gate_points: Vec<SweepPointSummary>,
    pub warrior_to_floor_harm_ratio: f64,
    pub floor_to_warrior_capacity_retention: f64,
    pub price_correlated_drift: PriceCorrelatedDriftSummary,
    pub prop12_variance: Prop12VarianceSummary,
}

pub fn selection_v3_suite(seed: u64, quick: bool) -> SelectionV3Suite {
    let n_agents = if quick { 400 } else { 2000 };
    let generations = if quick { 50 } else { 80 };
    let rule_reps = if quick { 8 } else { 16 };
    let sweep_reps = if quick { 5 } else { 7 };

    let mut rules = BTreeMap::new();
    for (idx, &rule) in SelectionRule::PRIMARY.iter().enumerate() {
        let summary = evolve_regime(
            rule,
            SC_TOP,
            SC_HARM_P,
            0.0,
            SimSchedule {
                n_agents,
                generations,
                replicates: rule_reps,
                base_seed: seed.wrapping_add(7000 + idx as u64 * 101),
            },
        );
        rules.insert(rule.name().to_string(), summary);
    }

    let mut auxiliary_variants = BTreeMap::new();
    let cap4 = evolve_regime(
        SelectionRule::CaretakerCap4Pen6,
        SC_TOP,
        SC_HARM_P,
        0.0,
        SimSchedule {
            n_agents,
            generations,
            replicates: sweep_reps,
            base_seed: seed.wrapping_add(7099),
        },
    );
    auxiliary_variants.insert(SelectionRule::CaretakerCap4Pen6.name().to_string(), cap4);

    let mut by_top = BTreeMap::new();
    let mut by_top_points = Vec::with_capacity(TOP_SWEEP.len());
    for (idx, &top) in TOP_SWEEP.iter().enumerate() {
        let r = evolve_regime(
            SelectionRule::TakenRank,
            top,
            SC_HARM_P,
            0.0,
            SimSchedule {
                n_agents,
                generations,
                replicates: sweep_reps,
                base_seed: seed.wrapping_add(7100 + idx as u64 * 101),
            },
        );
        let pt = to_sweep_point(top, &r);
        by_top.insert(format!("{top:.2}"), pt.clone());
        by_top_points.push(pt);
    }

    let mut by_harm = BTreeMap::new();
    let mut by_harm_points = Vec::with_capacity(HARM_SWEEP.len());
    for (idx, &hp) in HARM_SWEEP.iter().enumerate() {
        let r = evolve_regime(
            SelectionRule::ConferredRank,
            SC_TOP,
            hp,
            0.0,
            SimSchedule {
                n_agents,
                generations,
                replicates: sweep_reps,
                base_seed: seed.wrapping_add(7200 + idx as u64 * 101),
            },
        );
        let pt = to_sweep_point(hp, &r);
        by_harm.insert(format!("{hp:.2}"), pt.clone());
        by_harm_points.push(pt);
    }

    let mut by_gate = BTreeMap::new();
    let mut by_gate_points = Vec::with_capacity(GATE_SWEEP.len());
    for (idx, &gate) in GATE_SWEEP.iter().enumerate() {
        let r = evolve_regime(
            SelectionRule::TakenRank,
            SC_TOP,
            SC_HARM_P,
            gate,
            SimSchedule {
                n_agents,
                generations,
                replicates: sweep_reps,
                base_seed: seed.wrapping_add(7300 + idx as u64 * 101),
            },
        );
        let pt = to_sweep_point(gate, &r);
        by_gate.insert(format!("{gate:.2}"), pt.clone());
        by_gate_points.push(pt);
    }

    let warrior_harm = rules["taken_rank"].harm_rate.mean;
    let floor_harm = rules["floor"].harm_rate.mean.max(1e-9);
    let warrior_cap = rules["taken_rank"].capacity_gain.mean.max(1e-9);
    let floor_cap = rules["floor"].capacity_gain.mean;

    let (price_correlated_drift, prop12_variance) = compute_price_and_prop12(seed, quick);

    SelectionV3Suite {
        claim_boundary: CLAIM_BOUNDARY_SELECTION,
        seed,
        params: SelectionV3Params {
            quick,
            n_agents,
            generations,
            rule_replicates: rule_reps,
            sweep_replicates: sweep_reps,
            gamble_mean: SC_MEAN,
            gamble_sigma: SC_SIGMA,
            harm_p: SC_HARM_P,
            harm_loss: SC_HARM_LOSS,
            top_share: SC_TOP,
            share_mutation: SC_MUT_SHARE,
            capacity_gain_scale: SC_CAP_GAIN,
            capacity_mutation: SC_MUT_CAP,
            floor_eliminated_share: SC_FLOOR_DROP,
        },
        rules,
        auxiliary_variants,
        by_top,
        by_harm,
        by_gate,
        by_top_points,
        by_harm_points,
        by_gate_points,
        warrior_to_floor_harm_ratio: warrior_harm / floor_harm,
        floor_to_warrior_capacity_retention: floor_cap / warrior_cap,
        price_correlated_drift,
        prop12_variance,
    }
}

pub fn selection_coevolution_suite(seed: u64, quick: bool) -> SelectionV3Suite {
    selection_v3_suite(seed, quick)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_quick_verifies_four_rules_sweeps_and_identities() {
        let s = selection_v3_suite(20261110, true);

        let warrior = &s.rules["taken_rank"];
        let conferred = &s.rules["conferred_rank"];
        let caretaker = &s.rules["caretaker"];
        let floor = &s.rules["floor"];

        // Warrior and conferred_rank both select heavy gambling (>75% even in 30-gen quick, ~95% in 80-gen pub)
        assert!(warrior.final_share.mean > 0.75);
        assert!(conferred.final_share.mean > 0.75);

        // Floor collapses gambling far below caretaker & warrior while beating caretaker on capacity growth!
        assert!(floor.final_share.mean < 0.25);
        assert!(caretaker.final_share.mean < 0.35);
        assert!(floor.final_share.mean < caretaker.final_share.mean);
        assert!(floor.capacity_gain.mean > caretaker.capacity_gain.mean);
        assert!(s.warrior_to_floor_harm_ratio > 3.0);

        // Sweeps: broad top=0.50 collapses gambling compared to narrow top=0.05;
        // conferred_rank stays reckless at p_harm=0.60; external gate >= 0.80 re-aims warrior!
        assert!(s.by_top["0.50"].final_share < 0.25);
        assert!(s.by_top["0.05"].final_share > 0.75);
        assert!(s.by_harm["0.60"].final_share > 0.70);
        assert!(s.by_gate["0.80"].final_share < 0.20);
        assert!(s.by_gate["0.80"].capacity_gain > s.by_gate["0.00"].capacity_gain);

        // Price covariance identity holds to floating-point precision, and Sigma_12 = 0.4 drifts linearly
        assert!(
            s.price_correlated_drift
                .max_finite_population_price_identity_error
                < 1e-13
        );
        let eta1 = s
            .price_correlated_drift
            .tilt_curve
            .iter()
            .find(|p| (p.eta - 1.0).abs() < 1e-9)
            .unwrap();
        assert!((eta1.exact_side_effect_drift - 0.4).abs() < 1e-12);
        assert!((eta1.mc_side_effect_drift - 0.4).abs() < 0.05);
        assert!(eta1.penalized_mc_side_effect_drift.abs() < 0.05);

        // Prop 12: contest win prob strictly increases with sigma, while floor survival & concave welfare strictly decrease
        let curve = &s.prop12_variance.sigma_curve;
        for w in curve.windows(2) {
            assert!(w[1].contest_win_prob_exact > w[0].contest_win_prob_exact);
            assert!(w[1].floor_survival_prob_exact < w[0].floor_survival_prob_exact);
            assert!(w[1].concave_capped_welfare_exact < w[0].concave_capped_welfare_exact);
        }
        assert!(
            s.prop12_variance
                .monotone_transform_reversal
                .convex_reverses_preference
        );

        // Strict deterministic reproducibility across runs
        let s2 = selection_v3_suite(20261110, true);
        assert_eq!(
            serde_json::to_string(&s).unwrap(),
            serde_json::to_string(&s2).unwrap()
        );
    }

    #[test]
    #[ignore]
    fn publication_selection_verification() {
        let t0 = std::time::Instant::now();
        let sq = selection_v3_suite(20261110, true);
        let sq_s = t0.elapsed().as_secs_f64();

        let t1 = std::time::Instant::now();
        let sp = selection_v3_suite(20261110, false);
        let sp_s = t1.elapsed().as_secs_f64();

        println!("SELECTION TIMINGS: quick={sq_s:.3}s, pub={sp_s:.3}s");
        for rule in ["taken_rank", "conferred_rank", "caretaker", "floor"] {
            let r = &sp.rules[rule];
            println!(
                "RULE {rule:15}: share={:.4} (sd {:.4}), cap_gain={:+.4}, harm={:.5}, welfare={:.4}",
                r.final_share.mean,
                r.final_share_sd,
                r.capacity_gain.mean,
                r.harm_rate.mean,
                r.welfare.mean
            );
        }
        let c4 = &sp.auxiliary_variants["caretaker_cap4_pen6"];
        println!(
            "AUX caretaker_cap4_pen6: share={:.4}, cap_gain={:+.4}, harm={:.5}, welfare={:.4}",
            c4.final_share.mean, c4.capacity_gain.mean, c4.harm_rate.mean, c4.welfare.mean
        );
        println!(
            "SWEEPS: top0.10={:.4}/+{:.2} -> top0.50={:.4}/+{:.2}; harm0.60={:.4}; gate0.75={:.4} -> gate0.80={:.4} (cap +{:.2})",
            sp.by_top["0.10"].final_share,
            sp.by_top["0.10"].capacity_gain,
            sp.by_top["0.50"].final_share,
            sp.by_top["0.50"].capacity_gain,
            sp.by_harm["0.60"].final_share,
            sp.by_gate["0.75"].final_share,
            sp.by_gate["0.80"].final_share,
            sp.by_gate["0.80"].capacity_gain,
        );
        println!(
            "HARM RATIO warrior/floor = {:.2}x, CAP RETENTION floor/warrior = {:.3}",
            sp.warrior_to_floor_harm_ratio, sp.floor_to_warrior_capacity_retention
        );

        assert_eq!(sq.params.n_agents, 400);
        assert_eq!(sp.params.n_agents, 2000);
        assert_eq!(sp.params.generations, 80);
        assert!(sq_s < 0.5, "selection quick exceeded 0.5s: {sq_s:.3}s");
        assert!(sp_s < 2.5, "selection pub exceeded 2.5s: {sp_s:.3}s");
        assert!(sp.rules["taken_rank"].final_share.mean > 0.90);
        assert!(sp.rules["conferred_rank"].final_share.mean > 0.90);
        assert!(sp.rules["floor"].final_share.mean < 0.06);
        assert!(sp.rules["floor"].capacity_gain.mean > 3.4);
        assert!(sp.rules["caretaker"].capacity_gain.mean < 2.0);
        assert!(sp.by_gate["0.80"].final_share < 0.05);
        assert!(sp.warrior_to_floor_harm_ratio > 15.0);
    }
}
