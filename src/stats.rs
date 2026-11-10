use serde::Serialize;

/// Exact one-sided Clopper--Pearson limit, fixed independent Bernoulli design.
pub fn binomial_upper(failures: u64, n: u64, delta: f64) -> f64 {
    assert!(n > 0 && failures <= n && delta > 0.0 && delta < 1.0);
    if failures == n {
        return 1.0;
    }
    if failures == 0 {
        return -(delta.ln() / n as f64).exp_m1();
    }
    let (mut lo, mut hi) = (failures as f64 / n as f64, 1.0);
    for _ in 0..70 {
        let p = (lo + hi) / 2.0;
        if binomial_log_cdf(failures, n, p) > delta.ln() {
            lo = p;
        } else {
            hi = p;
        }
    }
    (lo + hi) / 2.0
}

pub fn binomial_log_cdf(x: u64, n: u64, p: f64) -> f64 {
    assert!(x <= n && (0.0..=1.0).contains(&p));
    if p == 0.0 || x == n {
        return 0.0;
    }
    if p == 1.0 {
        return f64::NEG_INFINITY;
    }
    let mut log_term = n as f64 * (-p).ln_1p();
    let mut log_sum = log_term;
    for k in 1..=x {
        log_term += ((n - k + 1) as f64 / k as f64).ln() + p.ln() - (-p).ln_1p();
        let m = log_sum.max(log_term);
        log_sum = m + ((log_sum - m).exp() + (log_term - m).exp()).ln();
    }
    log_sum.min(0.0)
}

#[derive(Serialize, Debug)]
pub struct Mean {
    pub n: usize,
    pub mean: f64,
    /// Descriptive 1.96 standard errors over independent seeds, not an exact CI.
    pub approximate_95_half_width: f64,
}
pub fn mean(xs: &[f64]) -> Mean {
    assert!(!xs.is_empty() && xs.iter().all(|x| x.is_finite()));
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    let se = if xs.len() > 1 {
        (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() * (xs.len() - 1)) as f64)
            .sqrt()
    } else {
        0.0
    };
    Mean {
        n: xs.len(),
        mean: m,
        approximate_95_half_width: 1.96 * se,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_small_binomial_and_large_zero_limit() {
        assert!((binomial_upper(0, 1, 0.05) - 0.95).abs() < 1e-14);
        assert!((binomial_upper(1, 2, 0.05) - 0.95_f64.sqrt()).abs() < 1e-13);
        assert!((binomial_log_cdf(1, 2, 0.5).exp() - 0.75).abs() < 1e-14);
        assert!(binomial_upper(0, 2_995_730, 0.05) > 1e-6);
        assert!(binomial_upper(0, 2_995_731, 0.05) <= 1e-6);
    }
}
