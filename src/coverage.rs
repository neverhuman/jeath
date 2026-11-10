use crate::{
    rng::Rng,
    stats::{Mean, binomial_log_cdf, mean},
};
use serde::Serialize;
#[derive(Serialize)]
pub struct CoverageResult {
    pub reports: usize,
    pub visible_classes: usize,
    pub distinct: usize,
    pub good_turing_reported_coverage: f64,
    pub honest_true_mass: f64,
    pub identical_reports_hidden_15_percent_mass: f64,
    pub hidden_family_recall: f64,
}
#[derive(Serialize)]
pub struct AuditPower {
    pub plants: u64,
    pub recognition: f64,
    pub repetitions: usize,
    pub rejection_rate: Mean,
}
#[derive(Serialize)]
pub struct Result {
    pub paired_worlds: CoverageResult,
    pub planted_audit: Vec<AuditPower>,
}
pub fn run(seed: u64, repetitions: usize) -> Result {
    let mut rng = Rng(seed);
    let classes = 500;
    let mut weights: Vec<_> = (1..=classes).map(|k| 1.0 / (k as f64).powf(1.1)).collect();
    let sum: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum;
    }
    let reports = 3000;
    let mut counts = vec![0; classes];
    for _ in 0..reports {
        counts[rng.weighted(&weights)] += 1;
    }
    let true_mass: f64 = weights
        .iter()
        .zip(&counts)
        .filter(|(_, n)| **n > 0)
        .map(|(w, _)| *w)
        .sum();
    let paired_worlds = CoverageResult {
        reports,
        visible_classes: classes,
        distinct: counts.iter().filter(|n| **n > 0).count(),
        good_turing_reported_coverage: 1.0
            - counts.iter().filter(|n| **n == 1).count() as f64 / reports as f64,
        honest_true_mass: true_mass,
        identical_reports_hidden_15_percent_mass: 0.85 * true_mass,
        hidden_family_recall: 0.0,
    };
    let mut planted_audit = Vec::new();
    // Null: known baseline recall 0.8. A withheld critical plant is reported
    // only if recognized as a plant. Fixed one-sided binomial test at 0.05.
    for plants in [40, 200] {
        for recognition in [0.0, 0.5, 0.8, 0.9, 1.0] {
            let mut rejected = Vec::new();
            for _ in 0..repetitions {
                let found = (0..plants)
                    .filter(|_| rng.uniform() < 0.8 * recognition)
                    .count() as u64;
                rejected.push(if binomial_log_cdf(found, plants, 0.8) <= 0.05_f64.ln() {
                    1.0
                } else {
                    0.0
                });
            }
            planted_audit.push(AuditPower {
                plants,
                recognition,
                repetitions,
                rejection_rate: mean(&rejected),
            });
        }
    }
    Result {
        paired_worlds,
        planted_audit,
    }
}
