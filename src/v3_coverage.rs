//! Coverage non-identifiability and planted-pathway audit power for Wave 2 V3:
//! - H12 (`coverage_v3.json`): 3-regime Zipf (`K = 2,000`, `N = 12,000`) Good--Turing missing-mass
//!   study (`honest`, `15%` shared `blind` spot, `5%` critical `withheld`), 2-searcher
//!   Lincoln--Petersen capture--recapture positive-bias check under shared blind spots, and
//!   McAllester--Schapire vs. Clopper--Pearson confidence bounds.
//! - H13 (`audit_power.json`): Exact log-hypergeometric + log-binomial evaluation of the two-sample
//!   one-sided Fisher exact test (`alpha = 0.01`, `P in [40, 200]`, 21 recognition points
//!   `r in [0.0, 1.0]`), matching `tips/experiments.py` Experiment 4 (`17.0%` at `P=40, r=0.8`
//!   and `87.7%` at `P=200, r=0.8` for baseline recall `0.80`, plus `q0 = 0.50` sensitivity).
//!
//! Claim boundary (`AGENTS.md`): Demonstrates statistical non-identifiability from report streams
//! alone and exact frequentist power under exchangeable indistinguishable plants, not empirical
//! prevalence of withholding in deployed evaluators.

use crate::{
    rng::Rng,
    stats::{binomial_upper, mean},
};
use serde::Serialize;
use std::collections::BTreeMap;

// ============================================================================
// H12: 3-Regime Good--Turing Coverage & Capture--Recapture Bias (`coverage_v3.json`)
// ============================================================================

#[derive(Serialize, Debug, Clone)]
pub struct CoverageRegimeSummary {
    pub regime: &'static str,
    pub trials: usize,
    pub est: f64,
    pub est_ci95_half_width: f64,
    pub true_coverage: f64,
    pub true_ci95_half_width: f64,
    pub critical_recall: f64,
    pub critical_ci95_half_width: f64,
    pub overconfidence_gap: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct CaptureRecaptureRow {
    pub condition: &'static str,
    pub true_total_classes: usize,
    pub visible_classes: usize,
    pub reports_per_searcher: usize,
    pub mean_lincoln_petersen_k_hat: f64,
    pub mean_lp_implied_class_coverage: f64,
    pub mean_true_class_coverage: f64,
    pub mean_true_mass_coverage: f64,
    pub lp_coverage_positive_bias: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct SampleComplexityRow {
    pub target_missing_mass_eps: f64,
    pub confidence_delta: f64,
    pub known_oracle_clopper_pearson_n: u64,
    pub mcallester_schapire_missing_mass_n: u64,
    pub inflation_factor: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct CoverageV3Suite {
    pub hypothesis: &'static str,
    pub evidence_tier: &'static str,
    pub total_classes_k: usize,
    pub searchers: usize,
    pub reports_per_searcher: usize,
    pub total_reports_n: usize,
    pub trials: usize,
    pub blind_spot_fraction: f64,
    pub critical_withheld_fraction: f64,
    pub honest: CoverageRegimeSummary,
    pub blind: CoverageRegimeSummary,
    pub withheld: CoverageRegimeSummary,
    pub blind_inflates_gt_estimate_verified: bool,
    pub withheld_zero_critical_recall_verified: bool,
    pub capture_recapture_rows: Vec<CaptureRecaptureRow>,
    pub capture_recapture_shared_blind_positive_bias_verified: bool,
    pub mcallester_schapire_bound_at_n12000_honest: f64,
    pub sample_complexity_table: Vec<SampleComplexityRow>,
    pub claim_boundary: &'static str,
}

fn build_cdf(weights: &[f64]) -> Vec<f64> {
    let sum: f64 = weights.iter().sum();
    assert!(sum.is_finite() && sum > 0.0);
    let mut cdf = Vec::with_capacity(weights.len());
    let mut acc = 0.0f64;
    for &w in weights {
        acc += w / sum;
        cdf.push(acc);
    }
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }
    cdf
}

fn sample_subset_mask(k: usize, count: usize, rng: &mut Rng) -> Vec<bool> {
    let mut indices: Vec<usize> = (0..k).collect();
    for i in 0..count {
        let j = i + rng.index(k - i);
        indices.swap(i, j);
    }
    let mut mask = vec![false; k];
    for &idx in &indices[..count] {
        mask[idx] = true;
    }
    mask
}

fn draw_counts_from_cdf(cdf: &[f64], n_draws: usize, rng: &mut Rng) -> Vec<u32> {
    let k = cdf.len();
    let mut cnt = vec![0u32; k];
    for _ in 0..n_draws {
        let u = rng.uniform();
        let idx = cdf.partition_point(|&c| c < u).min(k - 1);
        cnt[idx] += 1;
    }
    cnt
}

pub fn coverage_v3_suite(seed: u64, quick: bool) -> CoverageV3Suite {
    let k_classes = 2_000usize;
    let searchers = 4usize;
    let reports_per_searcher = 3_000usize;
    let n_reports = searchers * reports_per_searcher; // 12,000
    let trials = if quick { 25 } else { 200 };
    let n_crit = k_classes / 20; // 100 (5%)
    let n_blind = (k_classes * 15) / 100; // 300 (15%)

    let mut rng = Rng(seed ^ 0x4831_325F_434F_5633);

    let mut honest_est = Vec::with_capacity(trials);
    let mut honest_true = Vec::with_capacity(trials);
    let mut honest_crit = Vec::with_capacity(trials);

    let mut blind_est = Vec::with_capacity(trials);
    let mut blind_true = Vec::with_capacity(trials);
    let mut blind_crit = Vec::with_capacity(trials);

    let mut with_est = Vec::with_capacity(trials);
    let mut with_true = Vec::with_capacity(trials);
    let mut with_crit = Vec::with_capacity(trials);

    // Capture-recapture accumulators (2 searchers x 3,000 reports each)
    let mut lp_unif_honest = Vec::with_capacity(trials);
    let mut lp_unif_blind = Vec::with_capacity(trials);
    let mut lp_zipf_blind = Vec::with_capacity(trials);

    for _ in 0..trials {
        let mut a: Vec<f64> = (1..=k_classes).map(|r| 1.0 / (r as f64)).collect();
        rng.shuffle(&mut a);
        let a_sum: f64 = a.iter().sum();
        for w in &mut a {
            *w /= a_sum;
        }

        let critical = sample_subset_mask(k_classes, n_crit, &mut rng);
        let blind = sample_subset_mask(k_classes, n_blind, &mut rng);
        let crit_mass_total: f64 = (0..k_classes).filter(|&i| critical[i]).map(|i| a[i]).sum();

        for (regime_idx, (est_v, true_v, crit_v)) in [
            (&mut honest_est, &mut honest_true, &mut honest_crit),
            (&mut blind_est, &mut blind_true, &mut blind_crit),
            (&mut with_est, &mut with_true, &mut with_crit),
        ]
        .into_iter()
        .enumerate()
        {
            let mut p = a.clone();
            if regime_idx == 1 {
                for i in 0..k_classes {
                    if blind[i] {
                        p[i] = 0.0;
                    }
                }
            } else if regime_idx == 2 {
                for i in 0..k_classes {
                    if critical[i] {
                        p[i] = 0.0;
                    }
                }
            }
            let cdf = build_cdf(&p);
            let cnt = draw_counts_from_cdf(&cdf, n_reports, &mut rng);
            let f1 = cnt.iter().filter(|&&c| c == 1).count();
            let gt = 1.0 - (f1 as f64) / (n_reports as f64);
            let t_cov: f64 = (0..k_classes).filter(|&i| cnt[i] > 0).map(|i| a[i]).sum();
            let c_cov: f64 = (0..k_classes)
                .filter(|&i| cnt[i] > 0 && critical[i])
                .map(|i| a[i])
                .sum::<f64>()
                / crit_mass_total;

            est_v.push(gt);
            true_v.push(t_cov);
            crit_v.push(c_cov);
        }

        // Two-searcher Chapman-adjusted Lincoln-Petersen check
        // Condition 1: Uniform honest (all 2,000 classes equal)
        let unif_honest_w = vec![1.0 / k_classes as f64; k_classes];
        lp_unif_honest.push(eval_two_searcher_lp(
            &unif_honest_w,
            &unif_honest_w,
            reports_per_searcher,
            &mut rng,
        ));

        // Condition 2: Uniform with 15% shared blind spot (both searchers see only 1,700 classes)
        let mut unif_blind_w = unif_honest_w.clone();
        for i in 0..k_classes {
            if blind[i] {
                unif_blind_w[i] = 0.0;
            }
        }
        lp_unif_blind.push(eval_two_searcher_lp(
            &unif_honest_w,
            &unif_blind_w,
            reports_per_searcher,
            &mut rng,
        ));

        // Condition 3: Zipf-1.0 with 15% shared blind spot
        let mut zipf_blind_w = a.clone();
        for i in 0..k_classes {
            if blind[i] {
                zipf_blind_w[i] = 0.0;
            }
        }
        lp_zipf_blind.push(eval_two_searcher_lp(
            &a,
            &zipf_blind_w,
            reports_per_searcher,
            &mut rng,
        ));
    }

    let clean_zero = |x: f64| if x == 0.0 { 0.0 } else { x };
    let mk_summary =
        |regime: &'static str, est: &[f64], tr: &[f64], cr: &[f64]| -> CoverageRegimeSummary {
            let m_est = mean(est);
            let m_tr = mean(tr);
            let m_cr = mean(cr);
            CoverageRegimeSummary {
                regime,
                trials,
                est: clean_zero(m_est.mean),
                est_ci95_half_width: clean_zero(m_est.approximate_95_half_width),
                true_coverage: clean_zero(m_tr.mean),
                true_ci95_half_width: clean_zero(m_tr.approximate_95_half_width),
                critical_recall: clean_zero(m_cr.mean),
                critical_ci95_half_width: clean_zero(m_cr.approximate_95_half_width),
                overconfidence_gap: clean_zero(m_est.mean - m_tr.mean),
            }
        };

    let honest = mk_summary("honest", &honest_est, &honest_true, &honest_crit);
    let blind = mk_summary("blind", &blind_est, &blind_true, &blind_crit);
    let withheld = mk_summary("withheld", &with_est, &with_true, &with_crit);

    let capture_recapture_rows = vec![
        summarize_lp(
            "uniform_honest",
            k_classes,
            k_classes,
            reports_per_searcher,
            &lp_unif_honest,
        ),
        summarize_lp(
            "uniform_shared_blind_15pct",
            k_classes,
            k_classes - n_blind,
            reports_per_searcher,
            &lp_unif_blind,
        ),
        summarize_lp(
            "zipf_shared_blind_15pct",
            k_classes,
            k_classes - n_blind,
            reports_per_searcher,
            &lp_zipf_blind,
        ),
    ];

    let blind_inflates_gt = blind.est > honest.est && blind.true_coverage < 0.85;
    let withheld_zero_crit = withheld.critical_recall == 0.0 && withheld.est > 0.95;
    let lp_bias_ok = capture_recapture_rows[1].lp_coverage_positive_bias > 0.10
        && capture_recapture_rows[2].lp_coverage_positive_bias > 0.20;

    // McAllester-Schapire (2000) missing-mass upper bound slack at N=12,000, delta=0.05:
    let delta = 0.05f64;
    let ms_slack = (85.0 / n_reports as f64).sqrt();
    let ms_bound_honest = ((1.0 - honest.est) + ms_slack).min(1.0);

    let sample_complexity_table: Vec<SampleComplexityRow> = [0.05f64, 0.02, 0.01, 0.005]
        .into_iter()
        .map(|eps| {
            let mut cp_n = (-(delta.ln()) / eps).floor().max(1.0) as u64;
            while binomial_upper(0, cp_n, delta) > eps {
                cp_n += 1;
            }
            let ms_n = (85.0 / (eps * eps)).ceil() as u64;
            SampleComplexityRow {
                target_missing_mass_eps: eps,
                confidence_delta: delta,
                known_oracle_clopper_pearson_n: cp_n,
                mcallester_schapire_missing_mass_n: ms_n,
                inflation_factor: ms_n as f64 / cp_n as f64,
            }
        })
        .collect();

    CoverageV3Suite {
        hypothesis: "H12",
        evidence_tier: "tier2_nonidentifiability_and_tier4_multinomial_simulation",
        total_classes_k: k_classes,
        searchers,
        reports_per_searcher,
        total_reports_n: n_reports,
        trials,
        blind_spot_fraction: 0.15,
        critical_withheld_fraction: 0.05,
        honest,
        blind,
        withheld,
        blind_inflates_gt_estimate_verified: blind_inflates_gt,
        withheld_zero_critical_recall_verified: withheld_zero_crit,
        capture_recapture_rows,
        capture_recapture_shared_blind_positive_bias_verified: lp_bias_ok,
        mcallester_schapire_bound_at_n12000_honest: ms_bound_honest,
        sample_complexity_table,
        claim_boundary: "Proves report-only coverage estimators (Good-Turing and two-searcher capture-recapture) cannot identify shared blind spots or strategic withholding without independent planted audits.",
    }
}

/// Returns `(k_hat_lp, lp_implied_class_cov, true_class_cov, true_mass_cov)`.
fn eval_two_searcher_lp(
    true_weights: &[f64],
    searcher_weights: &[f64],
    reports_each: usize,
    rng: &mut Rng,
) -> (f64, f64, f64, f64) {
    let k = true_weights.len();
    let cdf = build_cdf(searcher_weights);
    let c1 = draw_counts_from_cdf(&cdf, reports_each, rng);
    let c2 = draw_counts_from_cdf(&cdf, reports_each, rng);

    let s1 = c1.iter().filter(|&&x| x > 0).count();
    let s2 = c2.iter().filter(|&&x| x > 0).count();
    let m12 = (0..k).filter(|&i| c1[i] > 0 && c2[i] > 0).count();
    let union_s = (0..k).filter(|&i| c1[i] > 0 || c2[i] > 0).count();
    let union_mass: f64 = (0..k)
        .filter(|&i| c1[i] > 0 || c2[i] > 0)
        .map(|i| true_weights[i])
        .sum();

    // Chapman unbiased capture-recapture estimator for population size K:
    let k_hat = (((s1 + 1) * (s2 + 1)) as f64 / (m12 + 1) as f64) - 1.0;
    let lp_cov = (union_s as f64 / k_hat.max(union_s as f64)).min(1.0);
    let true_class_cov = union_s as f64 / k as f64;
    (k_hat, lp_cov, true_class_cov, union_mass)
}

fn summarize_lp(
    condition: &'static str,
    true_k: usize,
    visible_k: usize,
    reports_each: usize,
    samples: &[(f64, f64, f64, f64)],
) -> CaptureRecaptureRow {
    let n = samples.len() as f64;
    let mean_k_hat = samples.iter().map(|s| s.0).sum::<f64>() / n;
    let mean_lp_cov = samples.iter().map(|s| s.1).sum::<f64>() / n;
    let mean_true_class = samples.iter().map(|s| s.2).sum::<f64>() / n;
    let mean_true_mass = samples.iter().map(|s| s.3).sum::<f64>() / n;
    CaptureRecaptureRow {
        condition,
        true_total_classes: true_k,
        visible_classes: visible_k,
        reports_per_searcher: reports_each,
        mean_lincoln_petersen_k_hat: mean_k_hat,
        mean_lp_implied_class_coverage: mean_lp_cov,
        mean_true_class_coverage: mean_true_class,
        mean_true_mass_coverage: mean_true_mass,
        lp_coverage_positive_bias: mean_lp_cov - mean_true_class,
    }
}

// ============================================================================
// H13: Two-Sample Fisher Exact Planted-Pathway Audit Power (`audit_power.json`)
//      Exact log-hypergeometric + log-binomial evaluation (`tips/experiments.py` Exp 4)
// ============================================================================

#[derive(Serialize, Debug, Clone)]
pub struct FisherAuditCurve {
    pub r: Vec<f64>,
    pub power: Vec<f64>,
}

#[derive(Serialize, Debug, Clone)]
pub struct AuditPowerSuite {
    pub hypothesis: &'static str,
    pub evidence_tier: &'static str,
    pub alpha: f64,
    pub primary_baseline_recall: f64,
    pub plants_grid: Vec<usize>,
    pub recognition_grid: Vec<f64>,
    /// Primary `tips/experiments.py` Exp 4 curves (`recall = 0.80`, `alpha = 0.01`):
    /// `"40"` (`17.01%` at `r=0.80`) and `"200"` (`87.74%` at `r=0.80`).
    pub by_plants: BTreeMap<String, FisherAuditCurve>,
    /// Sensitivity curves at lower ordinary discovery baseline (`q0 = 0.50`, `alpha = 0.01`).
    pub sensitivity_q0_050: BTreeMap<String, FisherAuditCurve>,
    pub power_p40_r050: f64,
    pub power_p40_r080: f64,
    pub power_p40_r090: f64,
    pub power_p200_r080: f64,
    pub power_p200_r090: f64,
    pub type1_size_p40_r100: f64,
    pub type1_size_p200_r100: f64,
    pub clopper_pearson_zero_of_300_upper95: f64,
    pub clopper_pearson_zero_of_3000_upper95: f64,
    pub clopper_pearson_trials_for_1e_minus_6: u64,
    pub claim_boundary: &'static str,
}

fn build_ln_fact(max_n: usize) -> Vec<f64> {
    let mut f = vec![0.0f64; max_n + 1];
    for i in 1..=max_n {
        f[i] = f[i - 1] + (i as f64).ln();
    }
    f
}

#[inline]
fn ln_choose(ln_fact: &[f64], n: usize, k: usize) -> f64 {
    debug_assert!(k <= n);
    ln_fact[n] - ln_fact[k] - ln_fact[n - k]
}

fn hypergeom_lower_tail_cdf(ln_fact: &[f64], xc: usize, total_found: usize, p: usize) -> f64 {
    // Population M = 2*p, Type-I = p (critical), draws = total_found = xc + xo.
    // Support of X_c: k in max(0, total_found - p) ..= min(p, total_found).
    let k_min = total_found.saturating_sub(p);
    let denom = ln_choose(ln_fact, 2 * p, total_found);
    let mut max_lp = f64::NEG_INFINITY;
    let mut log_terms = Vec::with_capacity(xc + 1 - k_min);
    for k in k_min..=xc {
        let lp = ln_choose(ln_fact, p, k) + ln_choose(ln_fact, p, total_found - k) - denom;
        if lp > max_lp {
            max_lp = lp;
        }
        log_terms.push(lp);
    }
    let sum_exp: f64 = log_terms.into_iter().map(|lp| (lp - max_lp).exp()).sum();
    (max_lp + sum_exp.ln()).exp().min(1.0)
}

fn binom_pmf_vec(ln_fact: &[f64], n: usize, prob: f64) -> Vec<f64> {
    if prob <= 0.0 {
        let mut v = vec![0.0; n + 1];
        v[0] = 1.0;
        return v;
    }
    if prob >= 1.0 {
        let mut v = vec![0.0; n + 1];
        v[n] = 1.0;
        return v;
    }
    let lp = prob.ln();
    let lq = (-prob).ln_1p();
    (0..=n)
        .map(|k| (ln_choose(ln_fact, n, k) + (k as f64) * lp + ((n - k) as f64) * lq).exp())
        .collect()
}

fn evaluate_fisher_curves(
    plants_list: &[usize],
    rs: &[f64],
    baseline_recall: f64,
    alpha: f64,
    ln_fact: &[f64],
) -> BTreeMap<String, FisherAuditCurve> {
    let mut out = BTreeMap::new();
    for &p in plants_list {
        let mut reject = vec![vec![false; p + 1]; p + 1];
        for (xc, row) in reject.iter_mut().enumerate() {
            for (xo, cell) in row.iter_mut().enumerate().skip(xc + 1) {
                let pval = hypergeom_lower_tail_cdf(ln_fact, xc, xc + xo, p);
                // Guard against tiny f64 rounding differences at exact boundary; matches scipy.stats.hypergeom.cdf
                *cell = pval <= alpha + 1e-15;
            }
        }
        let po = binom_pmf_vec(ln_fact, p, baseline_recall);
        let mut power = Vec::with_capacity(rs.len());
        for &r in rs {
            let pc = binom_pmf_vec(ln_fact, p, baseline_recall * r);
            let mut pwr = 0.0f64;
            for xc in 0..=p {
                if pc[xc] == 0.0 {
                    continue;
                }
                let mut inner = 0.0f64;
                for xo in (xc + 1)..=p {
                    if reject[xc][xo] {
                        inner += po[xo];
                    }
                }
                pwr += pc[xc] * inner;
            }
            power.push(pwr.clamp(0.0, 1.0));
        }
        out.insert(
            p.to_string(),
            FisherAuditCurve {
                r: rs.to_vec(),
                power,
            },
        );
    }
    out
}

pub fn audit_power_suite() -> AuditPowerSuite {
    let alpha = 0.01f64;
    let primary_recall = 0.80f64;
    let plants_grid = vec![40usize, 200];
    let rs: Vec<f64> = (0..=20)
        .map(|i| (i as f64 * 0.05 * 100.0).round() / 100.0)
        .collect();

    let ln_fact = build_ln_fact(400);
    let by_plants = evaluate_fisher_curves(&plants_grid, &rs, primary_recall, alpha, &ln_fact);
    let sensitivity_q0_050 = evaluate_fisher_curves(&plants_grid, &rs, 0.50, alpha, &ln_fact);

    let c40 = &by_plants["40"].power;
    let c200 = &by_plants["200"].power;

    AuditPowerSuite {
        hypothesis: "H13",
        evidence_tier: "tier2_exact_hypergeometric_binomial_evaluation",
        alpha,
        primary_baseline_recall: primary_recall,
        plants_grid,
        recognition_grid: rs,
        power_p40_r050: c40[10],
        power_p40_r080: c40[16],
        power_p40_r090: c40[18],
        power_p200_r080: c200[16],
        power_p200_r090: c200[18],
        type1_size_p40_r100: c40[20],
        type1_size_p200_r100: c200[20],
        by_plants,
        sensitivity_q0_050,
        clopper_pearson_zero_of_300_upper95: binomial_upper(0, 300, 0.05),
        clopper_pearson_zero_of_3000_upper95: binomial_upper(0, 3000, 0.05),
        clopper_pearson_trials_for_1e_minus_6: 2_995_731,
        claim_boundary: "Exact frequentist two-sample Fisher audit power conditional on planted pathways being indistinguishable from organic bugs up to recognition rate r; power collapses to <= alpha as r -> 1.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h12_good_turing_blindness_withholding_and_lp_bias() {
        let res = coverage_v3_suite(20261110, true);
        assert!(res.honest.est > 0.94 && res.honest.est < 0.97);
        assert!((res.honest.est - res.honest.true_coverage).abs() < 0.015);
        assert!(res.honest.critical_recall > 0.90);

        assert!(res.blind_inflates_gt_estimate_verified);
        assert!(res.blind.true_coverage > 0.79 && res.blind.true_coverage < 0.85);

        assert!(res.withheld_zero_critical_recall_verified);
        assert_eq!(res.withheld.critical_recall, 0.0);
        assert!(res.withheld.true_coverage > 0.89 && res.withheld.true_coverage < 0.93);

        assert!(res.capture_recapture_shared_blind_positive_bias_verified);
        let eps001 = res
            .sample_complexity_table
            .iter()
            .find(|r| (r.target_missing_mass_eps - 0.01).abs() < 1e-12)
            .unwrap();
        assert_eq!(eps001.mcallester_schapire_missing_mass_n, 850_000);
        assert_eq!(eps001.known_oracle_clopper_pearson_n, 299);
    }

    #[test]
    fn h13_fisher_audit_power_matches_scipy_reference_to_machine_precision() {
        let res = audit_power_suite();
        // Cross-check against exact scipy.stats.hypergeom + binom values in target/tips-check/results.json:
        assert!((res.power_p40_r050 - 0.887_429_204_600_136_9).abs() < 1e-12);
        assert!((res.power_p40_r080 - 0.170_116_716_457_936_86).abs() < 1e-12);
        assert!((res.power_p40_r090 - 0.042_790_021_302_454_885).abs() < 1e-12);
        assert!((res.type1_size_p40_r100 - 0.005_256_109_810_513_497).abs() < 1e-12);

        assert!((res.power_p200_r080 - 0.877_351_607_239_086_9).abs() < 1e-12);
        assert!((res.power_p200_r090 - 0.285_070_137_472_041_8).abs() < 1e-12);
        assert!((res.type1_size_p200_r100 - 0.007_205_013_354_046_533).abs() < 1e-12);

        assert!(res.type1_size_p40_r100 <= res.alpha);
        assert!(res.type1_size_p200_r100 <= res.alpha);
        // Sensitivity at q0 = 0.50 has strictly lower power at r = 0.80 than q0 = 0.80:
        assert!(res.sensitivity_q0_050["40"].power[16] < res.power_p40_r080);
        assert!(res.sensitivity_q0_050["200"].power[16] < res.power_p200_r080);
    }

    #[test]
    fn h12_h13_publication_profile_and_determinism() {
        let t_quick = std::time::Instant::now();
        let _q = coverage_v3_suite(20261110, true);
        let quick_s = t_quick.elapsed().as_secs_f64();

        let t_pub = std::time::Instant::now();
        let cov1 = coverage_v3_suite(20261110, false);
        let aud1 = audit_power_suite();
        let pub_s = t_pub.elapsed().as_secs_f64();

        assert_eq!(
            serde_json::to_string(&cov1).unwrap(),
            serde_json::to_string(&coverage_v3_suite(20261110, false)).unwrap()
        );
        assert_eq!(
            serde_json::to_string(&aud1).unwrap(),
            serde_json::to_string(&audit_power_suite()).unwrap()
        );

        eprintln!(
            "[v3_coverage] quick={:.4}s, pub={:.4}s | honest=({:.4},{:.4},{:.4}) blind=({:.4},{:.4},{:.4}) withheld=({:.4},{:.4},{:.4}) | audit P40@0.8={:.4}, P200@0.8={:.4}",
            quick_s,
            pub_s,
            cov1.honest.est,
            cov1.honest.true_coverage,
            cov1.honest.critical_recall,
            cov1.blind.est,
            cov1.blind.true_coverage,
            cov1.blind.critical_recall,
            cov1.withheld.est,
            cov1.withheld.true_coverage,
            cov1.withheld.critical_recall,
            aud1.power_p40_r080,
            aud1.power_p200_r080,
        );

        assert!((cov1.honest.est - 0.957).abs() < 0.005);
        assert!((cov1.honest.true_coverage - 0.957).abs() < 0.005);
        assert!((cov1.honest.critical_recall - 0.940).abs() < 0.025);
        assert!((cov1.blind.est - 0.966).abs() < 0.005);
        assert!((cov1.blind.true_coverage - 0.820).abs() < 0.015);
        assert!((cov1.blind.critical_recall - 0.819).abs() < 0.030);
        assert!((cov1.withheld.est - 0.960).abs() < 0.005);
        assert!((cov1.withheld.true_coverage - 0.914).abs() < 0.010);
        assert_eq!(cov1.withheld.critical_recall, 0.0);
    }
}
