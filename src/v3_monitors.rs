//! V3 Monitor Dependence, Multi-Family Stacking & Assessment Ceiling Suite (H14, H15).
//!
//! Pure-Rust, zero-external-crate, seeded `SplitMix64` synthetic suite (standalone
//! pure-Rust classifiers; does not claim `scikit-learn` numerical equivalence or
//! unmeasured cross-architecture `libm` floating-point bit-identity).
//! Claim boundary (`AGENTS.md`): controlled parametric copula, 20D synthetic nonlinear
//! classification, and linear random-effects variance decompositions — explicitly NOT
//! frontier-LLM monitor error rates or human-participant behavioral measurements.
#![allow(clippy::needless_range_loop)]

use crate::{
    rng::Rng,
    stats::{Mean, mean},
};
use serde::Serialize;
use std::collections::BTreeMap;

pub const DIM: usize = 20;
const TEACHER_HIDDEN: usize = 8;
const GATES: usize = 6;
const HAZ_SHARE: f64 = 0.25;
const FALSE_ALARM: f64 = 0.05;
pub const ADV_BUDGETS: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
pub const RETRY_K_VALUES: [usize; 6] = [1, 2, 4, 8, 16, 32];

type Point = [f64; DIM];

const CLAIM_BOUNDARY_MONITORS: &str = "Synthetic 20D nonlinear teacher and parametric copula/variance mechanics only; not frontier-LLM monitor transfer or human-subject evidence.";

// ============================================================================
// High-Precision Pure-Rust Normal Special Functions & Gauss-Hermite Quadrature
// ============================================================================

/// High-precision complementary error function `erfc(x)` (Cody rational Chebyshev, <1e-15 rel err).
#[allow(clippy::excessive_precision)]
pub fn erfc_exact(x: f64) -> f64 {
    let ax = x.abs();
    if ax <= 0.46875 {
        let y = ax * ax;
        let p = [
            3.16112374387056560e00,
            1.13864154151050156e02,
            3.77485237685302021e02,
            3.20937758913846947e03,
            1.85777706184603153e-01,
        ];
        let q = [
            2.36012909523441209e01,
            2.44024637934444173e02,
            1.28261652607737228e03,
            2.84423683343917062e03,
        ];
        let num = (((p[4] * y + p[0]) * y + p[1]) * y + p[2]) * y + p[3];
        let den = (((y + q[0]) * y + q[1]) * y + q[2]) * y + q[3];
        let erf_val = x * num / den;
        return 1.0 - erf_val;
    }
    let result = if ax <= 4.0 {
        let p = [
            5.64188496988670089e-01,
            8.88314979438837594e00,
            6.61191906371416295e01,
            2.98635138197400131e02,
            8.81952221241769090e02,
            1.71204761263407058e03,
            2.05107837782607147e03,
            1.23033935479799725e03,
            2.15311535474403846e-08,
        ];
        let q = [
            1.57449261107098347e01,
            1.17693950891312499e02,
            5.37181101862009858e02,
            1.62138957456669019e03,
            3.29079923573345963e03,
            4.36261909014324716e03,
            3.43936767414372164e03,
            1.23033935480374942e03,
        ];
        let mut num = p[8];
        for &c in &p[..8] {
            num = num * ax + c;
        }
        let mut den = 1.0;
        for &c in &q {
            den = den * ax + c;
        }
        let xsq = (ax * 16.0).floor() / 16.0;
        let del = (ax - xsq) * (ax + xsq);
        (-xsq * xsq).exp() * (-del).exp() * (num / den)
    } else if ax >= 27.0 {
        0.0
    } else {
        let y = 1.0 / (ax * ax);
        let p = [
            3.05326634961232344e-01,
            3.60344899949804439e-01,
            1.25781726111229246e-01,
            1.60837851487422766e-02,
            6.58749161529837803e-04,
            1.63153871373020978e-02,
        ];
        let q = [
            2.56852019228982242e00,
            1.87295284992346047e00,
            5.27905102951428412e-01,
            6.05183413124413191e-02,
            2.33520497626869185e-03,
        ];
        let num = y * (((((p[5] * y + p[4]) * y + p[3]) * y + p[2]) * y + p[1]) * y + p[0]);
        let den = ((((q[4] * y + q[3]) * y + q[2]) * y + q[1]) * y + q[0]) * y + 1.0;
        let xsq = (ax * 16.0).floor() / 16.0;
        let del = (ax - xsq) * (ax + xsq);
        let frac = 1.0 / std::f64::consts::PI.sqrt() - num / den;
        (-xsq * xsq).exp() * (-del).exp() * (frac / ax)
    };
    if x < 0.0 { 2.0 - result } else { result }
}

/// Standard normal CDF `Phi(z)`.
#[inline]
pub fn norm_cdf(z: f64) -> f64 {
    0.5 * erfc_exact(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// Standard normal PDF `phi(z)`.
#[inline]
pub fn norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Exact inverse standard normal CDF `Phi^-1(p)` via Acklam rational init + Halley polish.
#[allow(clippy::excessive_precision)]
pub fn norm_ppf(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "norm_ppf requires p in (0, 1)");
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    let mut z = if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    };
    for _ in 0..3 {
        let err = norm_cdf(z) - p;
        let pdf = norm_pdf(z);
        if pdf > 0.0 {
            let u = err / pdf;
            z -= u / (1.0 + 0.5 * z * u);
        }
    }
    z
}

/// Probabilists' Gauss-Hermite nodes and normalized weights (`sum(w) == 1`) for `E_{f ~ N(0,1)}[g(f)]`,
/// computed deterministically via Golub-Welsch implicit symmetric tridiagonal QL.
pub fn gauss_hermite_probabilist(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 2);
    let mut d = vec![0.0_f64; n];
    let mut e = vec![0.0_f64; n];
    for (i, slot) in e.iter_mut().enumerate().take(n - 1) {
        *slot = ((i + 1) as f64).sqrt();
    }
    let mut z = vec![0.0_f64; n];
    z[0] = 1.0;

    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            assert!(iter <= 60, "Golub-Welsch QL failed to converge");
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r_step = f.hypot(g);
                e[i + 1] = r_step;
                if r_step == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r_step;
                c = g / r_step;
                g = d[i + 1] - p;
                let r_diag = (d[i] - g) * s + 2.0 * c * b;
                p = s * r_diag;
                d[i + 1] = g + p;
                g = c * r_diag - b;
                let zi1 = z[i + 1];
                z[i + 1] = s * z[i] + c * zi1;
                z[i] = c * z[i] - s * zi1;
            }
            if e[m] != 0.0 || l == m {
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        }
    }

    let mut pairs: Vec<(f64, f64)> = d.into_iter().zip(z.into_iter().map(|zi| zi * zi)).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let w_sum: f64 = pairs.iter().map(|x| x.1).sum();
    let nodes = pairs.iter().map(|x| x.0).collect();
    let weights = pairs.iter().map(|x| x.1 / w_sum).collect();
    (nodes, weights)
}

// ============================================================================
// Part 1: Gaussian Copula Graded Correlation Suite (H14, `copula.json`)
// ============================================================================

#[derive(Clone, Debug, Serialize)]
pub struct CopulaRow {
    pub rho: f64,
    pub joint_miss: Vec<f64>,
    pub inflation_over_independent: Vec<f64>,
    pub effective_independent_depth_k6: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CopulaSuite {
    pub claim_boundary: &'static str,
    pub marginal_miss: f64,
    pub quadrature_nodes: usize,
    pub k_values: Vec<usize>,
    pub rhos: Vec<f64>,
    /// Keyed by `"0.0"`, `"0.3"`, `"0.6"`, `"0.9"` for direct parity with `tips/experiments.py`.
    pub curves: BTreeMap<String, Vec<f64>>,
    pub inflation_over_independent: BTreeMap<String, Vec<f64>>,
    pub rows: Vec<CopulaRow>,
}

pub fn copula_suite() -> CopulaSuite {
    let m = 0.10_f64;
    let n_nodes = 200;
    let (nodes, weights) = gauss_hermite_probabilist(n_nodes);
    let t = norm_ppf(m);
    let rhos: Vec<f64> = vec![0.0, 0.3, 0.6, 0.9];
    let k_values: Vec<usize> = (1..=8).collect();
    let mut curves = BTreeMap::new();
    let mut inflation_map = BTreeMap::new();
    let mut rows = Vec::new();

    for &rho in &rhos {
        let key = format!("{rho:.1}");
        let mut joint_miss = Vec::with_capacity(k_values.len());
        let mut inflation = Vec::with_capacity(k_values.len());
        let cond_probs: Vec<f64> = if rho == 0.0 {
            vec![m; n_nodes]
        } else {
            let sr = rho.sqrt();
            let s1r = (1.0 - rho).sqrt();
            nodes
                .iter()
                .map(|&f| norm_cdf((t + sr * f) / s1r))
                .collect()
        };

        for &k in &k_values {
            let jk = if rho == 0.0 {
                m.powi(k as i32)
            } else {
                weights
                    .iter()
                    .zip(cond_probs.iter())
                    .map(|(&w, &p)| w * p.powi(k as i32))
                    .sum::<f64>()
            };
            let indep = m.powi(k as i32);
            joint_miss.push(jk);
            inflation.push(jk / indep);
        }
        let eff_k6 = joint_miss[5].ln() / m.ln();
        curves.insert(key.clone(), joint_miss.clone());
        inflation_map.insert(key, inflation.clone());
        rows.push(CopulaRow {
            rho,
            joint_miss,
            inflation_over_independent: inflation,
            effective_independent_depth_k6: eff_k6,
        });
    }

    CopulaSuite {
        claim_boundary: CLAIM_BOUNDARY_MONITORS,
        marginal_miss: m,
        quadrature_nodes: n_nodes,
        k_values,
        rhos,
        curves,
        inflation_over_independent: inflation_map,
        rows,
    }
}

pub fn copula_quadrature_suite() -> CopulaSuite {
    copula_suite()
}

// ============================================================================
// Part 2: 20D Nonlinear Teacher & 6 Distinct Pure-Rust Classifier Families (H14)
// ============================================================================

#[inline]
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

#[inline]
fn dot20(a: &Point, b: &Point) -> f64 {
    let mut s = 0.0;
    for i in 0..DIM {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn sq_dist20(a: &Point, b: &Point) -> f64 {
    let mut s = 0.0;
    for i in 0..DIM {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

struct ZigguratTable {
    x: [f64; 128],
    f: [f64; 128],
    r: f64,
    w_base: f64,
}

impl ZigguratTable {
    fn build() -> Self {
        let sqrt_two_pi = (2.0 * std::f64::consts::PI).sqrt();
        let eval = |r: f64| -> Option<([f64; 128], [f64; 128], f64)> {
            let tail = sqrt_two_pi * (1.0 - norm_cdf(r));
            let v = r * (-0.5 * r * r).exp() + tail;
            let mut x = [0.0_f64; 128];
            let mut f = [0.0_f64; 128];
            x[127] = r;
            f[127] = (-0.5 * r * r).exp();
            for k in (1..127).rev() {
                let fk = f[k + 1] + v / x[k + 1];
                if fk >= 1.0 {
                    return None;
                }
                f[k] = fk;
                x[k] = (-2.0 * fk.ln()).sqrt();
            }
            f[0] = 1.0;
            x[0] = 0.0;
            Some((x, f, v))
        };

        let mut lo = 3.40_f64;
        let mut hi = 3.50_f64;
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            match eval(mid) {
                Some((x, f, v)) if x[1] * (1.0 - f[1]) >= v => hi = mid,
                _ => lo = mid,
            }
        }
        let (x, f, v) = eval(hi).expect("valid 128-layer Ziggurat table");
        Self {
            x,
            f,
            r: hi,
            w_base: v / f[127],
        }
    }
}

#[inline]
fn ziggurat_table() -> &'static ZigguratTable {
    static TABLE: std::sync::OnceLock<ZigguratTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(ZigguratTable::build)
}

/// Exact standard normal draw `N(0, 1)` via 128-layer Marsaglia–Tsang Ziggurat
/// backed exclusively by deterministic `SplitMix64` (`crate::rng::Rng`).
#[inline]
pub(crate) fn fast_normal(rng: &mut Rng) -> f64 {
    let tab = ziggurat_table();
    loop {
        let bits = rng.next_u64();
        let k = (bits & 127) as usize;
        let s = (((bits >> 11) as i64) - (1_i64 << 52)) as f64 + 0.5;
        let u_sym = s * (1.0 / (1_u64 << 52) as f64);
        if k < 127 {
            let x = u_sym * tab.x[k + 1];
            if x.abs() < tab.x[k] {
                return x;
            }
            if tab.f[k + 1] + rng.uniform() * (tab.f[k] - tab.f[k + 1]) < (-0.5 * x * x).exp() {
                return x;
            }
        } else {
            let x = u_sym * tab.w_base;
            if x.abs() < tab.r {
                return x;
            }
            loop {
                let xx = -rng.uniform().ln() / tab.r;
                let yy = -rng.uniform().ln();
                if 2.0 * yy > xx * xx {
                    return if u_sym < 0.0 {
                        -(tab.r + xx)
                    } else {
                        tab.r + xx
                    };
                }
            }
        }
    }
}

#[inline]
fn sample_point(rng: &mut Rng) -> Point {
    std::array::from_fn(|_| fast_normal(rng))
}

struct Task20D {
    w: [[f64; DIM]; TEACHER_HIDDEN],
    b: [f64; TEACHER_HIDDEN],
    a: [f64; TEACHER_HIDDEN],
    tau: f64,
}

impl Task20D {
    fn new(rng: &mut Rng, tau_samples: usize) -> Self {
        let scale = 2.0 / (DIM as f64).sqrt();
        let w = std::array::from_fn(|_| std::array::from_fn(|_| fast_normal(rng) * scale));
        let b = std::array::from_fn(|_| fast_normal(rng) * 0.5);
        let a = std::array::from_fn(|_| fast_normal(rng));
        let mut t = Self { w, b, a, tau: 0.0 };
        let mut scores: Vec<f64> = (0..tau_samples)
            .map(|_| {
                let x = sample_point(rng);
                t.h(&x)
            })
            .collect();
        let idx = (((1.0 - HAZ_SHARE) * tau_samples as f64) as usize).min(tau_samples - 1);
        scores.select_nth_unstable_by(idx, f64::total_cmp);
        t.tau = scores[idx];
        t
    }

    #[inline]
    fn h(&self, x: &Point) -> f64 {
        let mut out = 0.0;
        for j in 0..TEACHER_HIDDEN {
            out += self.a[j] * (self.b[j] + dot20(&self.w[j], x)).tanh();
        }
        out
    }

    #[inline]
    fn hazardous(&self, x: &Point) -> bool {
        self.h(x) > self.tau
    }

    #[inline]
    fn sample_one(&self, rng: &mut Rng, hazard: Option<bool>) -> Point {
        loop {
            let x = sample_point(rng);
            if hazard.is_none() || hazard == Some(self.hazardous(&x)) {
                return x;
            }
        }
    }

    fn sample(&self, rng: &mut Rng, n: usize, hazard: Option<bool>) -> Vec<Point> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.sample_one(rng, hazard));
        }
        out
    }
}

// --- Family 1: 2-Layer ReLU MLP (`20 -> 14 -> 10 -> 1`) trained with Adam ---
const MLP_H1: usize = 14;
const MLP_H2: usize = 10;

#[derive(Clone)]
struct Mlp2L {
    w1: [[f64; DIM]; MLP_H1],
    b1: [f64; MLP_H1],
    w2: [[f64; MLP_H1]; MLP_H2],
    b2: [f64; MLP_H2],
    w3: [f64; MLP_H2],
    b3: f64,
}

impl Mlp2L {
    fn train(xs: &[Point], ys: &[bool], rng: &mut Rng, epochs: usize) -> Self {
        let s1 = (2.0 / DIM as f64).sqrt();
        let s2 = (2.0 / MLP_H1 as f64).sqrt();
        let s3 = (1.0 / MLP_H2 as f64).sqrt();
        let mut net = Self {
            w1: std::array::from_fn(|_| std::array::from_fn(|_| fast_normal(rng) * s1)),
            b1: [0.0; MLP_H1],
            w2: std::array::from_fn(|_| std::array::from_fn(|_| fast_normal(rng) * s2)),
            b2: [0.0; MLP_H2],
            w3: std::array::from_fn(|_| fast_normal(rng) * s3),
            b3: 0.0,
        };

        let mut mw1 = [[0.0; DIM]; MLP_H1];
        let mut vw1 = [[0.0; DIM]; MLP_H1];
        let mut mb1 = [0.0; MLP_H1];
        let mut vb1 = [0.0; MLP_H1];
        let mut mw2 = [[0.0; MLP_H1]; MLP_H2];
        let mut vw2 = [[0.0; MLP_H1]; MLP_H2];
        let mut mb2 = [0.0; MLP_H2];
        let mut vb2 = [0.0; MLP_H2];
        let mut mw3 = [0.0; MLP_H2];
        let mut vw3 = [0.0; MLP_H2];
        let mut mb3 = 0.0_f64;
        let mut vb3 = 0.0_f64;

        let n = xs.len();
        let mut order: Vec<usize> = (0..n).collect();
        let batch_size = 64usize;
        let lr = 0.011_f64;
        let wd = 2e-4_f64;
        let beta1 = 0.9_f64;
        let beta2 = 0.999_f64;
        let mut step = 0_i32;

        for _ in 0..epochs {
            rng.shuffle(&mut order);
            for chunk in order.chunks(batch_size) {
                step += 1;
                let inv_bs = 1.0 / chunk.len() as f64;
                let mut gw1 = [[0.0; DIM]; MLP_H1];
                let mut gb1 = [0.0; MLP_H1];
                let mut gw2 = [[0.0; MLP_H1]; MLP_H2];
                let mut gb2 = [0.0; MLP_H2];
                let mut gw3 = [0.0; MLP_H2];
                let mut gb3 = 0.0_f64;

                for &idx in chunk {
                    let x = &xs[idx];
                    let y = if ys[idx] { 1.0 } else { 0.0 };
                    let mut a1 = [0.0; MLP_H1];
                    for i in 0..MLP_H1 {
                        let z = net.b1[i] + dot20(&net.w1[i], x);
                        if z > 0.0 {
                            a1[i] = z;
                        }
                    }
                    let mut a2 = [0.0; MLP_H2];
                    let mut logit = net.b3;
                    for j in 0..MLP_H2 {
                        let mut z = net.b2[j];
                        for i in 0..MLP_H1 {
                            z += net.w2[j][i] * a1[i];
                        }
                        if z > 0.0 {
                            a2[j] = z;
                            logit += net.w3[j] * z;
                        }
                    }
                    let dz3 = (sigmoid(logit) - y) * inv_bs;
                    gb3 += dz3;
                    let mut da1 = [0.0; MLP_H1];
                    for j in 0..MLP_H2 {
                        gw3[j] += dz3 * a2[j];
                        if a2[j] > 0.0 {
                            let dz2 = dz3 * net.w3[j];
                            gb2[j] += dz2;
                            for i in 0..MLP_H1 {
                                gw2[j][i] += dz2 * a1[i];
                                da1[i] += dz2 * net.w2[j][i];
                            }
                        }
                    }
                    for i in 0..MLP_H1 {
                        if a1[i] > 0.0 {
                            let dz1 = da1[i];
                            gb1[i] += dz1;
                            for k in 0..DIM {
                                gw1[i][k] += dz1 * x[k];
                            }
                        }
                    }
                }

                let c1 = 1.0 - beta1.powi(step);
                let c2 = 1.0 - beta2.powi(step);
                let step_lr = lr * c2.sqrt() / c1;

                mb3 = beta1 * mb3 + (1.0 - beta1) * gb3;
                vb3 = beta2 * vb3 + (1.0 - beta2) * gb3 * gb3;
                net.b3 -= step_lr * mb3 / (vb3.sqrt() + 1e-8);

                for j in 0..MLP_H2 {
                    let g3 = gw3[j] + wd * net.w3[j];
                    mw3[j] = beta1 * mw3[j] + (1.0 - beta1) * g3;
                    vw3[j] = beta2 * vw3[j] + (1.0 - beta2) * g3 * g3;
                    net.w3[j] -= step_lr * mw3[j] / (vw3[j].sqrt() + 1e-8);

                    mb2[j] = beta1 * mb2[j] + (1.0 - beta1) * gb2[j];
                    vb2[j] = beta2 * vb2[j] + (1.0 - beta2) * gb2[j] * gb2[j];
                    net.b2[j] -= step_lr * mb2[j] / (vb2[j].sqrt() + 1e-8);

                    for i in 0..MLP_H1 {
                        let g2 = gw2[j][i] + wd * net.w2[j][i];
                        mw2[j][i] = beta1 * mw2[j][i] + (1.0 - beta1) * g2;
                        vw2[j][i] = beta2 * vw2[j][i] + (1.0 - beta2) * g2 * g2;
                        net.w2[j][i] -= step_lr * mw2[j][i] / (vw2[j][i].sqrt() + 1e-8);
                    }
                }
                for i in 0..MLP_H1 {
                    mb1[i] = beta1 * mb1[i] + (1.0 - beta1) * gb1[i];
                    vb1[i] = beta2 * vb1[i] + (1.0 - beta2) * gb1[i] * gb1[i];
                    net.b1[i] -= step_lr * mb1[i] / (vb1[i].sqrt() + 1e-8);
                    for k in 0..DIM {
                        let g1 = gw1[i][k] + wd * net.w1[i][k];
                        mw1[i][k] = beta1 * mw1[i][k] + (1.0 - beta1) * g1;
                        vw1[i][k] = beta2 * vw1[i][k] + (1.0 - beta2) * g1 * g1;
                        net.w1[i][k] -= step_lr * mw1[i][k] / (vw1[i][k].sqrt() + 1e-8);
                    }
                }
            }
        }
        net
    }

    #[inline]
    fn score(&self, x: &Point) -> f64 {
        let mut a1 = [0.0; MLP_H1];
        for i in 0..MLP_H1 {
            let z = self.b1[i] + dot20(&self.w1[i], x);
            if z > 0.0 {
                a1[i] = z;
            }
        }
        let mut logit = self.b3;
        for j in 0..MLP_H2 {
            let mut z = self.b2[j];
            for i in 0..MLP_H1 {
                z += self.w2[j][i] * a1[i];
            }
            if z > 0.0 {
                logit += self.w3[j] * z;
            }
        }
        logit
    }
}

// --- Family 2: L2-Regularized Logistic Regression via Exact Newton-IRLS ---
#[derive(Clone)]
struct LogisticL2 {
    w: Point,
    b: f64,
}

fn solve_linear(mut a: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let m = rhs.len();
    for i in 0..m {
        let mut pivot = i;
        let mut best = a[i][i].abs();
        for (r, row) in a.iter().enumerate().take(m).skip(i + 1) {
            if row[i].abs() > best {
                best = row[i].abs();
                pivot = r;
            }
        }
        if pivot != i {
            a.swap(i, pivot);
            rhs.swap(i, pivot);
        }
        let diag = if a[i][i].abs() < 1e-12 {
            1e-12
        } else {
            a[i][i]
        };
        for r in (i + 1)..m {
            let factor = a[r][i] / diag;
            rhs[r] -= factor * rhs[i];
            for c in i..m {
                a[r][c] -= factor * a[i][c];
            }
        }
    }
    let mut sol = vec![0.0; m];
    for i in (0..m).rev() {
        let mut s = rhs[i];
        for c in (i + 1)..m {
            s -= a[i][c] * sol[c];
        }
        let diag = if a[i][i].abs() < 1e-12 {
            1e-12
        } else {
            a[i][i]
        };
        sol[i] = s / diag;
    }
    sol
}

impl LogisticL2 {
    fn train(xs: &[Point], ys: &[bool]) -> Self {
        let mut w = [0.0; DIM];
        let mut b = 0.0_f64;
        let lambda = 1.0_f64;
        for _ in 0..12 {
            let mut grad = vec![0.0; DIM + 1];
            let mut hess = vec![vec![0.0; DIM + 1]; DIM + 1];
            for i in 0..DIM {
                grad[i] += lambda * w[i];
                hess[i][i] += lambda;
            }
            hess[DIM][DIM] += 1e-6;
            for (x, &y) in xs.iter().zip(ys.iter()) {
                let p = sigmoid(b + dot20(&w, x));
                let diff = p - if y { 1.0 } else { 0.0 };
                let wt = (p * (1.0 - p)).max(1e-6);
                for i in 0..DIM {
                    grad[i] += diff * x[i];
                    let wx_i = wt * x[i];
                    for j in i..DIM {
                        hess[i][j] += wx_i * x[j];
                    }
                    hess[i][DIM] += wx_i;
                }
                grad[DIM] += diff;
                hess[DIM][DIM] += wt;
            }
            for i in 0..(DIM + 1) {
                for j in 0..i {
                    hess[i][j] = hess[j][i];
                }
            }
            let step = solve_linear(hess, grad);
            let mut norm_sq = 0.0;
            for i in 0..DIM {
                w[i] -= step[i];
                norm_sq += step[i] * step[i];
            }
            b -= step[DIM];
            norm_sq += step[DIM] * step[DIM];
            if norm_sq < 1e-12 {
                break;
            }
        }
        Self { w, b }
    }

    #[inline]
    fn score(&self, x: &Point) -> f64 {
        self.b + dot20(&self.w, x)
    }
}

// --- Family 3: Regularized Quadratic Discriminant Analysis (`reg_param = 0.05`) ---
#[derive(Clone)]
struct RegQda {
    mean0: Point,
    mean1: Point,
    chol0: [[f64; DIM]; DIM],
    chol1: [[f64; DIM]; DIM],
    const_offset: f64,
}

fn cholesky20(mat: &[[f64; DIM]; DIM]) -> ([[f64; DIM]; DIM], f64) {
    let mut l = [[0.0; DIM]; DIM];
    let mut log_det = 0.0;
    for i in 0..DIM {
        for j in 0..=i {
            let mut sum = mat[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                let d = sum.max(1e-9).sqrt();
                l[i][i] = d;
                log_det += 2.0 * d.ln();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    (l, log_det)
}

#[inline]
fn mahalanobis_sq(chol: &[[f64; DIM]; DIM], x: &Point, mean: &Point) -> f64 {
    let mut y = [0.0; DIM];
    let mut sq = 0.0;
    for i in 0..DIM {
        let mut s = x[i] - mean[i];
        for j in 0..i {
            s -= chol[i][j] * y[j];
        }
        let yi = s / chol[i][i];
        y[i] = yi;
        sq += yi * yi;
    }
    sq
}

impl RegQda {
    fn train(xs: &[Point], ys: &[bool]) -> Self {
        let reg = 0.05_f64;
        let mut mean0 = [0.0; DIM];
        let mut mean1 = [0.0; DIM];
        let mut n0 = 0usize;
        let mut n1 = 0usize;
        for (x, &y) in xs.iter().zip(ys.iter()) {
            if y {
                n1 += 1;
                for i in 0..DIM {
                    mean1[i] += x[i];
                }
            } else {
                n0 += 1;
                for i in 0..DIM {
                    mean0[i] += x[i];
                }
            }
        }
        let n0f = n0.max(1) as f64;
        let n1f = n1.max(1) as f64;
        for i in 0..DIM {
            mean0[i] /= n0f;
            mean1[i] /= n1f;
        }
        let mut cov0 = [[0.0; DIM]; DIM];
        let mut cov1 = [[0.0; DIM]; DIM];
        for (x, &y) in xs.iter().zip(ys.iter()) {
            if y {
                for i in 0..DIM {
                    let di = x[i] - mean1[i];
                    for j in i..DIM {
                        cov1[i][j] += di * (x[j] - mean1[j]);
                    }
                }
            } else {
                for i in 0..DIM {
                    let di = x[i] - mean0[i];
                    for j in i..DIM {
                        cov0[i][j] += di * (x[j] - mean0[j]);
                    }
                }
            }
        }
        let d0 = (n0.saturating_sub(1)).max(1) as f64;
        let d1 = (n1.saturating_sub(1)).max(1) as f64;
        for i in 0..DIM {
            for j in i..DIM {
                let c0 = (1.0 - reg) * (cov0[i][j] / d0) + if i == j { reg } else { 0.0 };
                let c1 = (1.0 - reg) * (cov1[i][j] / d1) + if i == j { reg } else { 0.0 };
                cov0[i][j] = c0;
                cov0[j][i] = c0;
                cov1[i][j] = c1;
                cov1[j][i] = c1;
            }
        }
        let (chol0, ld0) = cholesky20(&cov0);
        let (chol1, ld1) = cholesky20(&cov1);
        let log_prior = (n1f / n0f).ln();
        Self {
            mean0,
            mean1,
            chol0,
            chol1,
            const_offset: log_prior - 0.5 * (ld1 - ld0),
        }
    }

    #[inline]
    fn score(&self, x: &Point) -> f64 {
        self.const_offset
            - 0.5
                * (mahalanobis_sq(&self.chol1, x, &self.mean1)
                    - mahalanobis_sq(&self.chol0, x, &self.mean0))
    }
}

// --- Family 4: Distance-Weighted 15-Nearest Neighbors (`KNN15`) ---
const KNN_K: usize = 15;

#[derive(Clone)]
struct Knn15 {
    xs: Vec<Point>,
    ys: Vec<bool>,
}

impl Knn15 {
    fn train(xs: &[Point], ys: &[bool]) -> Self {
        Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
        }
    }

    fn score(&self, x: &Point) -> f64 {
        let mut best_d2 = [f64::INFINITY; KNN_K];
        let mut best_y = [false; KNN_K];
        let mut worst_d2 = f64::INFINITY;
        for (xi, &yi) in self.xs.iter().zip(self.ys.iter()) {
            let mut d2 = 0.0;
            for c in 0..10 {
                let diff = x[c] - xi[c];
                d2 += diff * diff;
            }
            if d2 >= worst_d2 {
                continue;
            }
            for c in 10..DIM {
                let diff = x[c] - xi[c];
                d2 += diff * diff;
            }
            if d2 >= worst_d2 {
                continue;
            }
            let mut pos = KNN_K - 1;
            while pos > 0 && d2 < best_d2[pos - 1] {
                best_d2[pos] = best_d2[pos - 1];
                best_y[pos] = best_y[pos - 1];
                pos -= 1;
            }
            best_d2[pos] = d2;
            best_y[pos] = yi;
            worst_d2 = best_d2[KNN_K - 1];
        }
        let mut pos_w = 0.0;
        let mut tot_w = 0.0;
        for j in 0..KNN_K {
            if best_d2[j].is_finite() {
                let w = 1.0 / (best_d2[j].sqrt() + 1e-4);
                tot_w += w;
                if best_y[j] {
                    pos_w += w;
                }
            }
        }
        if tot_w > 0.0 { pos_w / tot_w } else { 0.0 }
    }
}

// --- Family 5: Nyström RBF Kernel Logistic Classifier (`RBFKernel`) ---
#[derive(Clone)]
struct RbfKernelClassifier {
    centers: Vec<Point>,
    weights: Vec<f64>,
    bias: f64,
    gamma: f64,
}

impl RbfKernelClassifier {
    fn train(xs: &[Point], ys: &[bool], rng: &mut Rng, n_centers: usize) -> Self {
        let n = xs.len();
        let m = n_centers.min(n);
        let mut pos_idx: Vec<usize> = (0..n).filter(|&i| ys[i]).collect();
        let mut neg_idx: Vec<usize> = (0..n).filter(|&i| !ys[i]).collect();
        rng.shuffle(&mut pos_idx);
        rng.shuffle(&mut neg_idx);

        let m_pos = (m * pos_idx.len() / n.max(1)).clamp(1, m - 1);
        let m_neg = m - m_pos;
        let mut centers = Vec::with_capacity(m);
        for &i in pos_idx.iter().take(m_pos) {
            centers.push(xs[i]);
        }
        for &i in neg_idx.iter().take(m_neg) {
            centers.push(xs[i]);
        }
        while centers.len() < m {
            centers.push(xs[rng.index(n)]);
        }

        let mut mean_x = [0.0; DIM];
        for x in xs {
            for c in 0..DIM {
                mean_x[c] += x[c];
            }
        }
        for val in &mut mean_x {
            *val /= n as f64;
        }
        let mut var_sum = 0.0;
        for x in xs {
            for c in 0..DIM {
                let d = x[c] - mean_x[c];
                var_sum += d * d;
            }
        }
        let var = (var_sum / (n * DIM) as f64).max(1e-6);
        let gamma = 1.0 / (DIM as f64 * var);

        let m_len = centers.len();
        let mut phi = vec![0.0_f64; n * m_len];
        for (i, x) in xs.iter().enumerate() {
            let row = &mut phi[i * m_len..(i + 1) * m_len];
            for (j, c) in centers.iter().enumerate() {
                row[j] = (-gamma * sq_dist20(x, c)).exp();
            }
        }

        let mut w = vec![0.0_f64; m_len];
        let mut v_w = vec![0.0_f64; m_len];
        let mut b = 0.0_f64;
        let mut v_b = 0.0_f64;
        let lr = 1.0_f64;
        let momentum = 0.85_f64;
        let lambda = 1.0 / (6.0 * n as f64); // C = 3.0 parity
        let inv_n = 1.0 / n as f64;

        for _ in 0..42 {
            let mut gw = vec![0.0_f64; m_len];
            let mut gb = 0.0_f64;
            for j in 0..m_len {
                gw[j] = lambda * (w[j] + momentum * v_w[j]);
            }
            let b_look = b + momentum * v_b;
            for i in 0..n {
                let row = &phi[i * m_len..(i + 1) * m_len];
                let mut logit = b_look;
                for j in 0..m_len {
                    logit += (w[j] + momentum * v_w[j]) * row[j];
                }
                let diff = (sigmoid(logit) - if ys[i] { 1.0 } else { 0.0 }) * inv_n;
                gb += diff;
                for j in 0..m_len {
                    gw[j] += diff * row[j];
                }
            }
            v_b = momentum * v_b - lr * gb;
            b += v_b;
            for j in 0..m_len {
                v_w[j] = momentum * v_w[j] - lr * gw[j];
                w[j] += v_w[j];
            }
        }

        Self {
            centers,
            weights: w,
            bias: b,
            gamma,
        }
    }

    #[inline]
    fn score(&self, x: &Point) -> f64 {
        let mut logit = self.bias;
        for (c, &w) in self.centers.iter().zip(self.weights.iter()) {
            logit += w * (-self.gamma * sq_dist20(x, c)).exp();
        }
        logit
    }
}

// --- Family 6: Gradient-Boosted Depth-2 Decision Trees/Stumps (`BoostedStumps`) ---
#[derive(Clone, Copy)]
struct Depth2Tree {
    root_feat: usize,
    root_thr: f64,
    left_feat: usize,
    left_thr: f64,
    right_feat: usize,
    right_thr: f64,
    leaves: [f64; 4],
}

impl Depth2Tree {
    #[inline]
    fn eval(&self, x: &Point) -> f64 {
        if x[self.root_feat] <= self.root_thr {
            if x[self.left_feat] <= self.left_thr {
                self.leaves[0]
            } else {
                self.leaves[1]
            }
        } else if x[self.right_feat] <= self.right_thr {
            self.leaves[2]
        } else {
            self.leaves[3]
        }
    }
}

#[derive(Clone)]
struct BoostedStumps {
    base_logit: f64,
    trees: Vec<Depth2Tree>,
}

fn find_best_split(
    binned_x: &[[u8; DIM]],
    indices: &[usize],
    grad: &[f64],
    hess: &[f64],
    bins: &[[f64; 15]; DIM],
    reg_lambda: f64,
) -> (usize, f64, f64, f64) {
    if indices.len() < 4 {
        let g_tot: f64 = indices.iter().map(|&i| grad[i]).sum();
        let h_tot: f64 = indices.iter().map(|&i| hess[i]).sum();
        let leaf = -g_tot / (h_tot + reg_lambda);
        return (0, 0.0, leaf, leaf);
    }
    let g_tot: f64 = indices.iter().map(|&i| grad[i]).sum();
    let h_tot: f64 = indices.iter().map(|&i| hess[i]).sum();
    let parent_gain = g_tot * g_tot / (h_tot + reg_lambda);
    let mut best_score = -1.0_f64;
    let mut best_feat = 0usize;
    let mut best_thr = 0.0_f64;
    let mut best_l = -g_tot / (h_tot + reg_lambda);
    let mut best_r = best_l;

    for feat in 0..DIM {
        let thrs = &bins[feat];
        let mut g_bins = [0.0_f64; 16];
        let mut h_bins = [0.0_f64; 16];
        for &idx in indices {
            let b = binned_x[idx][feat] as usize;
            g_bins[b] += grad[idx];
            h_bins[b] += hess[idx];
        }
        let mut gl = 0.0;
        let mut hl = 0.0;
        for b in 0..15 {
            gl += g_bins[b];
            hl += h_bins[b];
            let hr = h_tot - hl;
            if hl < 1e-3 || hr < 1e-3 {
                continue;
            }
            let gr = g_tot - gl;
            let gain = gl * gl / (hl + reg_lambda) + gr * gr / (hr + reg_lambda) - parent_gain;
            if gain > best_score {
                best_score = gain;
                best_feat = feat;
                best_thr = thrs[b];
                best_l = -gl / (hl + reg_lambda);
                best_r = -gr / (hr + reg_lambda);
            }
        }
    }
    (best_feat, best_thr, best_l, best_r)
}

impl BoostedStumps {
    fn train(xs: &[Point], ys: &[bool], n_trees: usize) -> Self {
        let n = xs.len();
        let pos = ys.iter().filter(|&&y| y).count().clamp(1, n - 1) as f64;
        let base_p = pos / n as f64;
        let base_logit = (base_p / (1.0 - base_p)).ln();

        let bins: [[f64; 15]; DIM] = std::array::from_fn(|feat| {
            let mut vals: Vec<f64> = xs.iter().map(|x| x[feat]).collect();
            vals.sort_by(f64::total_cmp);
            std::array::from_fn(|b| {
                let idx = (((b + 1) * n) / 16).min(n - 1);
                vals[idx]
            })
        });
        let binned_x: Vec<[u8; DIM]> = xs
            .iter()
            .map(|x| {
                std::array::from_fn(|feat| {
                    let v = x[feat];
                    bins[feat].partition_point(|&t| t < v) as u8
                })
            })
            .collect();

        let mut logits = vec![base_logit; n];
        let mut grad = vec![0.0_f64; n];
        let mut hess = vec![0.0_f64; n];
        let all_idx: Vec<usize> = (0..n).collect();
        let lr = 0.14_f64;
        let reg_lambda = 1.5_f64;
        let mut trees = Vec::with_capacity(n_trees);

        for _ in 0..n_trees {
            for i in 0..n {
                let p = sigmoid(logits[i]);
                grad[i] = p - if ys[i] { 1.0 } else { 0.0 };
                hess[i] = (p * (1.0 - p)).max(1e-6);
            }
            let (root_feat, root_thr, _, _) =
                find_best_split(&binned_x, &all_idx, &grad, &hess, &bins, reg_lambda);
            let mut left_idx = Vec::with_capacity(n / 2 + 16);
            let mut right_idx = Vec::with_capacity(n / 2 + 16);
            for i in 0..n {
                if xs[i][root_feat] <= root_thr {
                    left_idx.push(i);
                } else {
                    right_idx.push(i);
                }
            }
            let (left_feat, left_thr, l0, l1) =
                find_best_split(&binned_x, &left_idx, &grad, &hess, &bins, reg_lambda);
            let (right_feat, right_thr, l2, l3) =
                find_best_split(&binned_x, &right_idx, &grad, &hess, &bins, reg_lambda);
            let tree = Depth2Tree {
                root_feat,
                root_thr,
                left_feat,
                left_thr,
                right_feat,
                right_thr,
                leaves: [lr * l0, lr * l1, lr * l2, lr * l3],
            };
            for i in 0..n {
                logits[i] += tree.eval(&xs[i]);
            }
            trees.push(tree);
        }

        Self { base_logit, trees }
    }

    #[inline]
    fn score(&self, x: &Point) -> f64 {
        let mut s = self.base_logit;
        for t in &self.trees {
            s += t.eval(x);
        }
        s
    }
}

// --- Unified Calibrated Monitor Wrapper ---
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FamilyKind {
    Mlp2L,
    LogisticL2,
    RegQda,
    Knn15,
    RbfKernel,
    BoostedStumps,
}

impl FamilyKind {
    pub const ALL: [Self; GATES] = [
        Self::Mlp2L,
        Self::LogisticL2,
        Self::RegQda,
        Self::Knn15,
        Self::RbfKernel,
        Self::BoostedStumps,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Mlp2L => "MLP2L",
            Self::LogisticL2 => "LogisticL2",
            Self::RegQda => "RegQDA",
            Self::Knn15 => "KNN15",
            Self::RbfKernel => "RBFKernel",
            Self::BoostedStumps => "BoostedStumps",
        }
    }
}

#[derive(Clone)]
enum ModelBackend {
    Mlp2L(Box<Mlp2L>),
    LogisticL2(LogisticL2),
    RegQda(Box<RegQda>),
    Knn15(Knn15),
    RbfKernel(RbfKernelClassifier),
    BoostedStumps(BoostedStumps),
}

impl ModelBackend {
    #[inline]
    fn raw_score(&self, x: &Point) -> f64 {
        match self {
            Self::Mlp2L(m) => m.score(x),
            Self::LogisticL2(m) => m.score(x),
            Self::RegQda(m) => m.score(x),
            Self::Knn15(m) => m.score(x),
            Self::RbfKernel(m) => m.score(x),
            Self::BoostedStumps(m) => m.score(x),
        }
    }
}

#[derive(Clone)]
struct CalibratedMonitor {
    backend: ModelBackend,
    threshold: f64,
    iqr_scale: f64,
}

impl CalibratedMonitor {
    fn train_and_calibrate(
        family: FamilyKind,
        task: &Task20D,
        rng: &mut Rng,
        n_train: usize,
        n_cal: usize,
        quick: bool,
    ) -> Self {
        let xs = task.sample(rng, n_train, None);
        let ys: Vec<bool> = xs.iter().map(|x| task.hazardous(x)).collect();
        let backend = match family {
            FamilyKind::Mlp2L => {
                let epochs = if quick { 14 } else { 16 };
                ModelBackend::Mlp2L(Box::new(Mlp2L::train(&xs, &ys, rng, epochs)))
            }
            FamilyKind::LogisticL2 => ModelBackend::LogisticL2(LogisticL2::train(&xs, &ys)),
            FamilyKind::RegQda => ModelBackend::RegQda(Box::new(RegQda::train(&xs, &ys))),
            FamilyKind::Knn15 => ModelBackend::Knn15(Knn15::train(&xs, &ys)),
            FamilyKind::RbfKernel => {
                let centers = if quick { 48 } else { 72 };
                ModelBackend::RbfKernel(RbfKernelClassifier::train(&xs, &ys, rng, centers))
            }
            FamilyKind::BoostedStumps => {
                let n_trees = if quick { 36 } else { 52 };
                ModelBackend::BoostedStumps(BoostedStumps::train(&xs, &ys, n_trees))
            }
        };

        let cal_benign = task.sample(rng, n_cal, Some(false));
        let mut cal_scores: Vec<f64> = cal_benign.iter().map(|x| backend.raw_score(x)).collect();
        cal_scores.sort_by(f64::total_cmp);
        let thr_idx = (((1.0 - FALSE_ALARM) * n_cal as f64).ceil() as usize)
            .min(n_cal)
            .saturating_sub(1);
        let q25 = cal_scores[n_cal / 4];
        let q75 = cal_scores[(3 * n_cal) / 4];
        Self {
            backend,
            threshold: cal_scores[thr_idx],
            iqr_scale: (q75 - q25).abs().max(1e-6),
        }
    }

    #[inline]
    fn raw_score(&self, x: &Point) -> f64 {
        self.backend.raw_score(x)
    }

    #[inline]
    fn standardized_margin(&self, x: &Point) -> f64 {
        (self.backend.raw_score(x) - self.threshold) / self.iqr_scale
    }

    #[inline]
    fn misses(&self, x: &Point) -> bool {
        self.backend.raw_score(x) <= self.threshold
    }
}

/// Evaluates `het_monitors.iter().all(|m| m.misses(x))`, ordering fast parametric/tree/kernel
/// gates (`LogisticL2`, `RegQDA`, `MLP2L`, `BoostedStumps`, `RBFKernel`) before `KNN15` (`[3]`).
#[inline]
fn het_all_miss(het: &[CalibratedMonitor; GATES], x: &Point) -> bool {
    het[1].misses(x)
        && het[2].misses(x)
        && het[0].misses(x)
        && het[5].misses(x)
        && het[4].misses(x)
        && het[3].misses(x)
}

/// Payload-preserving projected L2 surrogate attack (`||cand - x0||_2 <= rho` AND `h(cand) >= h(x0)`).
fn surrogate_attack(
    task: &Task20D,
    surrogate: &CalibratedMonitor,
    x0: &Point,
    rho: f64,
    iters: usize,
    cands: usize,
    rng: &mut Rng,
) -> Point {
    if rho <= 0.0 {
        return *x0;
    }
    let h0 = task.h(x0);
    let rho_sq = rho * rho;
    let mut best = *x0;
    let mut best_raw = surrogate.raw_score(&best);
    let step_sigma = (rho / 2.4) * (2.0 / (DIM as f64).sqrt());

    for _ in 0..iters {
        let mut iter_best = best;
        let mut iter_raw = best_raw;
        for _ in 0..cands {
            let mut cand: Point = std::array::from_fn(|i| best[i] + step_sigma * fast_normal(rng));
            let d2 = sq_dist20(&cand, x0);
            if d2 > rho_sq {
                let scale = rho / d2.sqrt();
                for i in 0..DIM {
                    cand[i] = x0[i] + (cand[i] - x0[i]) * scale;
                }
            }
            let s_raw = surrogate.raw_score(&cand);
            if s_raw < iter_raw && task.h(&cand) >= h0 {
                iter_raw = s_raw;
                iter_best = cand;
            }
        }
        best = iter_best;
        best_raw = iter_raw;
    }
    best
}

struct PackedBits {
    words: Vec<u64>,
    len: usize,
}

impl PackedBits {
    fn from_bools(bs: &[bool]) -> Self {
        let mut words = vec![0_u64; bs.len().div_ceil(64)];
        for (i, &b) in bs.iter().enumerate() {
            if b {
                words[i / 64] |= 1_u64 << (i % 64);
            }
        }
        Self {
            words,
            len: bs.len(),
        }
    }

    fn count_ones(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }
}

fn subset_masks_by_k() -> [Vec<u8>; GATES] {
    let mut out: [Vec<u8>; GATES] = std::array::from_fn(|_| Vec::new());
    for mask in 1_u8..(1_u8 << GATES) {
        let k = mask.count_ones() as usize;
        out[k - 1].push(mask);
    }
    out
}

#[derive(Clone, Debug)]
struct SubsetCurve {
    marginal: f64,
    joint: [f64; GATES],
    product: [f64; GATES],
}

fn compute_subset_curve(misses: &[PackedBits; GATES], subsets: &[Vec<u8>; GATES]) -> SubsetCurve {
    let n = misses[0].len as f64;
    let marg: [f64; GATES] = std::array::from_fn(|j| misses[j].count_ones() as f64 / n);
    let n_words = misses[0].words.len();
    let mut joint = [0.0; GATES];
    let mut product = [0.0; GATES];

    for k in 1..=GATES {
        let subs = &subsets[k - 1];
        let mut j_sum = 0.0;
        let mut p_sum = 0.0;
        for &mask in subs {
            let mut prod_s = 1.0;
            let mut first_gate = None;
            for (g, &mg) in marg.iter().enumerate() {
                if (mask & (1 << g)) != 0 {
                    prod_s *= mg;
                    if first_gate.is_none() {
                        first_gate = Some(g);
                    }
                }
            }
            p_sum += prod_s;
            let fg = first_gate.unwrap();
            let mut ones = 0_u32;
            for w_idx in 0..n_words {
                let mut acc = misses[fg].words[w_idx];
                for (g, m_bits) in misses.iter().enumerate().skip(fg + 1) {
                    if (mask & (1 << g)) != 0 {
                        acc &= m_bits.words[w_idx];
                    }
                }
                ones += acc.count_ones();
            }
            j_sum += ones as f64 / n;
        }
        let denom = subs.len() as f64;
        joint[k - 1] = j_sum / denom;
        product[k - 1] = p_sum / denom;
    }

    SubsetCurve {
        marginal: marg.iter().sum::<f64>() / GATES as f64,
        joint,
        product,
    }
}

/// Computes `1 - Pr(all k pass | benign)` averaged over all `C(6, k)` gate subsets.
fn compute_false_block_curve(
    passes: &[PackedBits; GATES],
    subsets: &[Vec<u8>; GATES],
) -> [f64; GATES] {
    let all_pass = compute_subset_curve(passes, subsets).joint;
    std::array::from_fn(|i| 1.0 - all_pass[i])
}

#[derive(Debug, Serialize)]
pub struct StackedBudgetSummary {
    pub rho_adv: f64,
    pub marginal: Mean,
    pub joint: Vec<Mean>,
    pub product: Vec<Mean>,
    /// Convenience flat arrays matching `tips/experiments.py` schema (`joint_mean`, `joint_ci`, `product_mean`, `product_ci`).
    pub joint_mean: Vec<f64>,
    pub joint_ci: Vec<f64>,
    pub product_mean: Vec<f64>,
    pub product_ci: Vec<f64>,
    pub joint_to_product_ratio_k6: f64,
}

#[derive(Debug, Serialize)]
pub struct BenignFalseBlockCurves {
    pub k_values: Vec<usize>,
    pub target_per_gate_fpr: f64,
    pub same_step_homogeneous: Vec<Mean>,
    pub same_step_heterogeneous: Vec<Mean>,
    pub distinct_steps_heterogeneous: Vec<Mean>,
    pub theoretical_independent: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct AdaptiveFeedbackSummary {
    pub rho_adv: f64,
    pub queries: usize,
    pub same_step_no_feedback_k6: Mean,
    pub same_step_binary_feedback_8q_k6: Mean,
    pub same_step_score_feedback_8q_k6: Mean,
    pub distinct_steps_no_feedback_k6: Mean,
    pub distinct_steps_score_feedback_8q_k6: Mean,
}

#[derive(Debug, Serialize)]
pub struct RetrySelectionRow {
    pub k_select: usize,
    pub independent_retry_miss: Mean,
    pub formula_prediction: Mean,
    pub abs_formula_gap: f64,
    pub correlated_drift_miss: Mean,
}

#[derive(Clone, Debug, Serialize)]
pub struct AttackIntegritySummary {
    pub max_l2_overshoot: f64,
    pub min_payload_hazard_delta: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MonitorsV3Config {
    pub quick: bool,
    pub tasks: usize,
    pub dim: usize,
    pub gates: usize,
    pub train_samples: usize,
    pub calibration_samples: usize,
    pub eval_episodes: usize,
    pub adv_budgets: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct MonitorsV3Suite {
    pub claim_boundary: &'static str,
    pub seed: u64,
    pub config: MonitorsV3Config,
    pub families: Vec<&'static str>,
    pub per_family_passive_miss: BTreeMap<String, Mean>,
    pub per_family_benign_fpr: BTreeMap<String, Mean>,
    pub surrogate_pass: BTreeMap<String, Mean>,
    pub same_step_homogeneous: BTreeMap<String, StackedBudgetSummary>,
    pub same_step_heterogeneous: BTreeMap<String, StackedBudgetSummary>,
    pub distinct_steps_heterogeneous: BTreeMap<String, StackedBudgetSummary>,
    pub benign_false_block: BenignFalseBlockCurves,
    pub adaptive_feedback: AdaptiveFeedbackSummary,
    pub readonly_retry_selection: Vec<RetrySelectionRow>,
    pub attack_integrity: AttackIntegritySummary,
}

pub fn monitors_v3_suite(seed: u64, quick: bool) -> MonitorsV3Suite {
    let n_tasks = if quick { 6 } else { 24 };
    let n_train = if quick { 500 } else { 680 };
    let n_cal = if quick { 360 } else { 560 };
    let n_eval = if quick { 80 } else { 128 };
    let tau_samples = if quick { 3_000 } else { 5_000 };
    let atk_iters = if quick { 6 } else { 7 };
    let atk_cands = 5usize;
    let retry_episodes = if quick { 100 } else { 140 };
    let j6_ref_samples = if quick { 600 } else { 1200 };

    let subsets = subset_masks_by_k();

    let mut fam_passive: [Vec<f64>; GATES] = std::array::from_fn(|_| Vec::with_capacity(n_tasks));
    let mut fam_fpr: [Vec<f64>; GATES] = std::array::from_fn(|_| Vec::with_capacity(n_tasks));
    let mut surr_pass_acc: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(n_tasks));
    let mut same_hom_acc: [Vec<SubsetCurve>; 4] =
        std::array::from_fn(|_| Vec::with_capacity(n_tasks));
    let mut same_het_acc: [Vec<SubsetCurve>; 4] =
        std::array::from_fn(|_| Vec::with_capacity(n_tasks));
    let mut dist_het_acc: [Vec<SubsetCurve>; 4] =
        std::array::from_fn(|_| Vec::with_capacity(n_tasks));

    let mut fb_hom_acc: Vec<[f64; GATES]> = Vec::with_capacity(n_tasks);
    let mut fb_het_acc: Vec<[f64; GATES]> = Vec::with_capacity(n_tasks);
    let mut fb_dist_acc: Vec<[f64; GATES]> = Vec::with_capacity(n_tasks);

    let mut no_fb_same = Vec::with_capacity(n_tasks);
    let mut bin_fb_same = Vec::with_capacity(n_tasks);
    let mut score_fb_same = Vec::with_capacity(n_tasks);
    let mut no_fb_dist = Vec::with_capacity(n_tasks);
    let mut score_fb_dist = Vec::with_capacity(n_tasks);

    let mut retry_indep_acc: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(n_tasks));
    let mut retry_form_acc: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(n_tasks));
    let mut retry_corr_acc: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(n_tasks));

    let mut max_l2_overshoot = 0.0_f64;
    let mut min_payload_delta = f64::INFINITY;

    for t_idx in 0..n_tasks {
        let task_seed = seed.wrapping_add((t_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut rng = Rng(task_seed);
        let task = Task20D::new(&mut rng, tau_samples);

        let hom_monitors: [CalibratedMonitor; GATES] = std::array::from_fn(|_| {
            CalibratedMonitor::train_and_calibrate(
                FamilyKind::Mlp2L,
                &task,
                &mut rng,
                n_train,
                n_cal,
                quick,
            )
        });
        let het_monitors: [CalibratedMonitor; GATES] = std::array::from_fn(|g| {
            CalibratedMonitor::train_and_calibrate(
                FamilyKind::ALL[g],
                &task,
                &mut rng,
                n_train,
                n_cal,
                quick,
            )
        });
        let surrogate = CalibratedMonitor::train_and_calibrate(
            FamilyKind::Mlp2L,
            &task,
            &mut rng,
            n_train,
            n_cal,
            quick,
        );

        let shared_haz = task.sample(&mut rng, n_eval, Some(true));
        let distinct_haz: [Vec<Point>; GATES] =
            std::array::from_fn(|_| task.sample(&mut rng, n_eval, Some(true)));

        // Benign false-block compounding across k = 1..=6
        let shared_benign = task.sample(&mut rng, n_eval, Some(false));
        let distinct_benign: [Vec<Point>; GATES] =
            std::array::from_fn(|_| task.sample(&mut rng, n_eval, Some(false)));

        let hom_pass_bits: [PackedBits; GATES] = std::array::from_fn(|g| {
            let bs: Vec<bool> = shared_benign
                .iter()
                .map(|b| hom_monitors[g].misses(b))
                .collect();
            PackedBits::from_bools(&bs)
        });
        let het_pass_bits: [PackedBits; GATES] = std::array::from_fn(|g| {
            let bs: Vec<bool> = shared_benign
                .iter()
                .map(|b| het_monitors[g].misses(b))
                .collect();
            let fpr = 1.0 - (bs.iter().filter(|&&ok| ok).count() as f64 / n_eval as f64);
            fam_fpr[g].push(fpr);
            PackedBits::from_bools(&bs)
        });
        let dist_pass_bits: [PackedBits; GATES] = std::array::from_fn(|g| {
            let bs: Vec<bool> = distinct_benign[g]
                .iter()
                .map(|b| het_monitors[g].misses(b))
                .collect();
            PackedBits::from_bools(&bs)
        });
        fb_hom_acc.push(compute_false_block_curve(&hom_pass_bits, &subsets));
        fb_het_acc.push(compute_false_block_curve(&het_pass_bits, &subsets));
        fb_dist_acc.push(compute_false_block_curve(&dist_pass_bits, &subsets));

        // Evaluate across all 4 perturbation budgets
        for (b_idx, &rho) in ADV_BUDGETS.iter().enumerate() {
            let attacked_shared: Vec<Point> = shared_haz
                .iter()
                .map(|x0| {
                    let xa = surrogate_attack(
                        &task, &surrogate, x0, rho, atk_iters, atk_cands, &mut rng,
                    );
                    let dist = sq_dist20(&xa, x0).sqrt();
                    max_l2_overshoot = max_l2_overshoot.max((dist - rho).max(0.0));
                    min_payload_delta = min_payload_delta.min(task.h(&xa) - task.h(x0));
                    xa
                })
                .collect();

            let sp = attacked_shared
                .iter()
                .filter(|x| surrogate.misses(x))
                .count() as f64
                / n_eval as f64;
            surr_pass_acc[b_idx].push(sp);

            let hom_miss_bits: [PackedBits; GATES] = std::array::from_fn(|g| {
                let bs: Vec<bool> = attacked_shared
                    .iter()
                    .map(|x| hom_monitors[g].misses(x))
                    .collect();
                PackedBits::from_bools(&bs)
            });
            let het_miss_bits: [PackedBits; GATES] = std::array::from_fn(|g| {
                let bs: Vec<bool> = attacked_shared
                    .iter()
                    .map(|x| het_monitors[g].misses(x))
                    .collect();
                if b_idx == 0 {
                    fam_passive[g].push(bs.iter().filter(|&&m| m).count() as f64 / n_eval as f64);
                }
                PackedBits::from_bools(&bs)
            });
            let attacked_distinct: [Vec<Point>; GATES] = std::array::from_fn(|g| {
                distinct_haz[g]
                    .iter()
                    .map(|s0| {
                        let sa = surrogate_attack(
                            &task, &surrogate, s0, rho, atk_iters, atk_cands, &mut rng,
                        );
                        let dist = sq_dist20(&sa, s0).sqrt();
                        max_l2_overshoot = max_l2_overshoot.max((dist - rho).max(0.0));
                        min_payload_delta = min_payload_delta.min(task.h(&sa) - task.h(s0));
                        sa
                    })
                    .collect()
            });
            let dist_miss_bits: [PackedBits; GATES] = std::array::from_fn(|g| {
                let bs: Vec<bool> = attacked_distinct[g]
                    .iter()
                    .map(|sa| het_monitors[g].misses(sa))
                    .collect();
                PackedBits::from_bools(&bs)
            });

            let c_hom = compute_subset_curve(&hom_miss_bits, &subsets);
            let c_het = compute_subset_curve(&het_miss_bits, &subsets);
            let c_dist = compute_subset_curve(&dist_miss_bits, &subsets);

            if b_idx == 0 {
                // Read-only retry candidate selection (K_select in [1, 2, 4, 8, 16, 32])
                // against frozen 6-family heterogeneous stack with zero gradient access.
                let mut j6_hits = 0usize;
                for _ in 0..j6_ref_samples {
                    let p = task.sample_one(&mut rng, Some(true));
                    if het_all_miss(&het_monitors, &p) {
                        j6_hits += 1;
                    }
                }
                let j6_task = j6_hits as f64 / j6_ref_samples as f64;
                let max_k = RETRY_K_VALUES[RETRY_K_VALUES.len() - 1];
                let mut indep_hits = [0usize; 6];
                let mut corr_hits = [0usize; 6];

                for _ in 0..retry_episodes {
                    let mut first_indep_pass: Option<usize> = None;
                    for attempt in 1..=max_k {
                        let cand = task.sample_one(&mut rng, Some(true));
                        if het_all_miss(&het_monitors, &cand) {
                            first_indep_pass = Some(attempt);
                            break;
                        }
                    }
                    if let Some(hit_at) = first_indep_pass {
                        for (ki, &kv) in RETRY_K_VALUES.iter().enumerate() {
                            if hit_at <= kv {
                                indep_hits[ki] += 1;
                            }
                        }
                    }

                    // Correlated local drift around a single underlying step x_base
                    let x_base = task.sample_one(&mut rng, Some(true));
                    let h_base = task.h(&x_base);
                    let mut first_corr_pass: Option<usize> = None;
                    for attempt in 1..=max_k {
                        let cand = if attempt == 1 {
                            x_base
                        } else {
                            let mut c_try = x_base;
                            for _ in 0..6 {
                                let noise = sample_point(&mut rng);
                                let prop: Point =
                                    std::array::from_fn(|i| x_base[i] + 0.18 * noise[i]);
                                if task.h(&prop) >= h_base {
                                    c_try = prop;
                                    break;
                                }
                            }
                            c_try
                        };
                        if het_all_miss(&het_monitors, &cand) {
                            first_corr_pass = Some(attempt);
                            break;
                        }
                    }
                    if let Some(hit_at) = first_corr_pass {
                        for (ki, &kv) in RETRY_K_VALUES.iter().enumerate() {
                            if hit_at <= kv {
                                corr_hits[ki] += 1;
                            }
                        }
                    }
                }
                for (ki, &kv) in RETRY_K_VALUES.iter().enumerate() {
                    retry_indep_acc[ki].push(indep_hits[ki] as f64 / retry_episodes as f64);
                    retry_form_acc[ki].push(1.0 - (1.0 - j6_task).powi(kv as i32));
                    retry_corr_acc[ki].push(corr_hits[ki] as f64 / retry_episodes as f64);
                }
            }

            if (rho - 2.0).abs() < 1e-9 {
                // 8-query adaptive feedback leakage at rho = 2.0
                no_fb_same.push(c_het.joint[GATES - 1]);
                no_fb_dist.push(c_dist.joint[GATES - 1]);

                let fb_eps = if quick { 24 } else { 32 };
                let fb_iters = 3usize;
                let fb_cands = 4usize;
                let mut bin_same_miss = 0usize;
                let mut score_same_miss = 0usize;
                let mut score_dist_miss = 0usize;

                let pipeline_max_margin = |x: &Point| -> f64 {
                    het_monitors
                        .iter()
                        .map(|m| m.standardized_margin(x))
                        .fold(f64::NEG_INFINITY, f64::max)
                };

                for ep in 0..fb_eps {
                    let x0 = &shared_haz[ep];
                    let h0 = task.h(x0);
                    let x_init = &attacked_shared[ep];

                    // Binary feedback (8 surrogate-edited candidates, pass/fail verdict only)
                    let mut bin_evaded = het_all_miss(&het_monitors, x_init);
                    for _ in 1..8 {
                        if bin_evaded {
                            break;
                        }
                        let cand = surrogate_attack(
                            &task, &surrogate, x0, rho, fb_iters, fb_cands, &mut rng,
                        );
                        if het_all_miss(&het_monitors, &cand) {
                            bin_evaded = true;
                        }
                    }
                    if bin_evaded {
                        bin_same_miss += 1;
                    }

                    // Score feedback (8 adaptive queries minimizing max standardized margin)
                    let mut best_x = *x_init;
                    let mut best_m = pipeline_max_margin(&best_x);
                    for _ in 1..8 {
                        let mut cand = surrogate_attack(
                            &task, &surrogate, x0, rho, fb_iters, fb_cands, &mut rng,
                        );
                        // Also test a local step from current best_x within ball
                        let noise = sample_point(&mut rng);
                        let mut local: Point = std::array::from_fn(|i| best_x[i] + 0.22 * noise[i]);
                        let d_loc = sq_dist20(&local, x0).sqrt();
                        if d_loc > rho {
                            for i in 0..DIM {
                                local[i] = x0[i] + (local[i] - x0[i]) * (rho / d_loc);
                            }
                        }
                        if task.h(&local) >= h0
                            && surrogate.raw_score(&local) < surrogate.raw_score(&cand)
                        {
                            cand = local;
                        }
                        let m_val = pipeline_max_margin(&cand);
                        if m_val < best_m {
                            best_m = m_val;
                            best_x = cand;
                        }
                    }
                    if best_m <= 0.0 {
                        score_same_miss += 1;
                    }

                    // Distinct-steps score feedback: 8 queries against the 6-step pipeline
                    // (attacker sees only overall pipeline max margin across all 6 distinct steps)
                    let mut best_steps: [Point; GATES] =
                        std::array::from_fn(|g| attacked_distinct[g][ep]);
                    let steps_margin = |st: &[Point; GATES]| -> f64 {
                        (0..GATES)
                            .map(|g| het_monitors[g].standardized_margin(&st[g]))
                            .fold(f64::NEG_INFINITY, f64::max)
                    };
                    let mut best_dist_m = steps_margin(&best_steps);
                    for _ in 1..8 {
                        let cand_steps: [Point; GATES] = std::array::from_fn(|g| {
                            surrogate_attack(
                                &task,
                                &surrogate,
                                &distinct_haz[g][ep],
                                rho,
                                3,
                                5,
                                &mut rng,
                            )
                        });
                        let cm = steps_margin(&cand_steps);
                        if cm < best_dist_m {
                            best_dist_m = cm;
                            best_steps = cand_steps;
                        }
                    }
                    if best_dist_m <= 0.0 {
                        score_dist_miss += 1;
                    }
                }

                bin_fb_same.push(bin_same_miss as f64 / fb_eps as f64);
                score_fb_same.push(score_same_miss as f64 / fb_eps as f64);
                score_fb_dist.push(score_dist_miss as f64 / fb_eps as f64);
            }

            same_hom_acc[b_idx].push(c_hom);
            same_het_acc[b_idx].push(c_het);
            dist_het_acc[b_idx].push(c_dist);
        }
    }

    let summarize_arch = |acc: &[Vec<SubsetCurve>; 4]| -> BTreeMap<String, StackedBudgetSummary> {
        let mut map = BTreeMap::new();
        for (b_idx, &rho) in ADV_BUDGETS.iter().enumerate() {
            let key = format!("{rho:.1}");
            let curves = &acc[b_idx];
            let marg = mean(&curves.iter().map(|c| c.marginal).collect::<Vec<_>>());
            let joint: Vec<Mean> = (0..GATES)
                .map(|k| mean(&curves.iter().map(|c| c.joint[k]).collect::<Vec<_>>()))
                .collect();
            let product: Vec<Mean> = (0..GATES)
                .map(|k| mean(&curves.iter().map(|c| c.product[k]).collect::<Vec<_>>()))
                .collect();
            let joint_mean: Vec<f64> = joint.iter().map(|m| m.mean).collect();
            let joint_ci: Vec<f64> = joint.iter().map(|m| m.approximate_95_half_width).collect();
            let product_mean: Vec<f64> = product.iter().map(|m| m.mean).collect();
            let product_ci: Vec<f64> = product
                .iter()
                .map(|m| m.approximate_95_half_width)
                .collect();
            let ratio_k6 = joint_mean[GATES - 1] / product_mean[GATES - 1].max(1e-15);
            map.insert(
                key,
                StackedBudgetSummary {
                    rho_adv: rho,
                    marginal: marg,
                    joint,
                    product,
                    joint_mean,
                    joint_ci,
                    product_mean,
                    product_ci,
                    joint_to_product_ratio_k6: ratio_k6,
                },
            );
        }
        map
    };

    let mut per_family_passive_miss = BTreeMap::new();
    let mut per_family_benign_fpr = BTreeMap::new();
    for g in 0..GATES {
        let name = FamilyKind::ALL[g].name().to_string();
        per_family_passive_miss.insert(name.clone(), mean(&fam_passive[g]));
        per_family_benign_fpr.insert(name, mean(&fam_fpr[g]));
    }

    let mut surrogate_pass = BTreeMap::new();
    for (b_idx, &rho) in ADV_BUDGETS.iter().enumerate() {
        surrogate_pass.insert(format!("{rho:.1}"), mean(&surr_pass_acc[b_idx]));
    }

    let benign_false_block = BenignFalseBlockCurves {
        k_values: (1..=GATES).collect(),
        target_per_gate_fpr: FALSE_ALARM,
        same_step_homogeneous: (0..GATES)
            .map(|k| mean(&fb_hom_acc.iter().map(|r| r[k]).collect::<Vec<_>>()))
            .collect(),
        same_step_heterogeneous: (0..GATES)
            .map(|k| mean(&fb_het_acc.iter().map(|r| r[k]).collect::<Vec<_>>()))
            .collect(),
        distinct_steps_heterogeneous: (0..GATES)
            .map(|k| mean(&fb_dist_acc.iter().map(|r| r[k]).collect::<Vec<_>>()))
            .collect(),
        theoretical_independent: (1..=GATES)
            .map(|k| 1.0 - (1.0 - FALSE_ALARM).powi(k as i32))
            .collect(),
    };

    let readonly_retry_selection: Vec<RetrySelectionRow> = RETRY_K_VALUES
        .iter()
        .enumerate()
        .map(|(ki, &kv)| {
            let ind = mean(&retry_indep_acc[ki]);
            let form = mean(&retry_form_acc[ki]);
            let corr = mean(&retry_corr_acc[ki]);
            RetrySelectionRow {
                k_select: kv,
                abs_formula_gap: (ind.mean - form.mean).abs(),
                independent_retry_miss: ind,
                formula_prediction: form,
                correlated_drift_miss: corr,
            }
        })
        .collect();

    MonitorsV3Suite {
        claim_boundary: CLAIM_BOUNDARY_MONITORS,
        seed,
        config: MonitorsV3Config {
            quick,
            tasks: n_tasks,
            dim: DIM,
            gates: GATES,
            train_samples: n_train,
            calibration_samples: n_cal,
            eval_episodes: n_eval,
            adv_budgets: ADV_BUDGETS.to_vec(),
        },
        families: FamilyKind::ALL.iter().map(|f| f.name()).collect(),
        per_family_passive_miss,
        per_family_benign_fpr,
        surrogate_pass,
        same_step_homogeneous: summarize_arch(&same_hom_acc),
        same_step_heterogeneous: summarize_arch(&same_het_acc),
        distinct_steps_heterogeneous: summarize_arch(&dist_het_acc),
        benign_false_block,
        adaptive_feedback: AdaptiveFeedbackSummary {
            rho_adv: 2.0,
            queries: 8,
            same_step_no_feedback_k6: mean(&no_fb_same),
            same_step_binary_feedback_8q_k6: mean(&bin_fb_same),
            same_step_score_feedback_8q_k6: mean(&score_fb_same),
            distinct_steps_no_feedback_k6: mean(&no_fb_dist),
            distinct_steps_score_feedback_8q_k6: mean(&score_fb_dist),
        },
        readonly_retry_selection,
        attack_integrity: AttackIntegritySummary {
            max_l2_overshoot,
            min_payload_hazard_delta: min_payload_delta,
        },
    }
}

pub fn six_family_monitors_suite(seed: u64, quick: bool) -> MonitorsV3Suite {
    monitors_v3_suite(seed, quick)
}

// ============================================================================
// Part 3: Crossed Generalizability & Multi-Signal Portfolio Ceiling (H15)
// ============================================================================

#[derive(Clone, Debug, Serialize)]
pub struct PortfolioPoint {
    pub budget_n: usize,
    pub k_domains: usize,
    pub n_per_domain: usize,
    pub exact_r_g: f64,
    pub mc_r_g: f64,
    pub abs_error: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PortfolioVarianceModel {
    pub v_g: f64,
    pub omega_sq: f64,
    pub sigma_c_sq: f64,
    pub sigma_e_sq: f64,
    pub shared_bias_ceiling: f64,
    pub single_domain_ceiling: f64,
    pub repeat_64: f64,
    pub diverse_64: f64,
    pub repeated_single_domain_curve: Vec<PortfolioPoint>,
    pub diverse_domains_curve: Vec<PortfolioPoint>,
    pub fixed_budget_64_tradeoff: Vec<PortfolioPoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CrossedVariancePoint {
    pub m_evaluators: usize,
    pub k_contexts: usize,
    pub exact_v_mk: f64,
    pub mc_v_mk: f64,
    pub abs_error: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CrossedGeneralizabilityModel {
    pub citation: &'static str,
    pub v_t: f64,
    pub v_u: f64,
    pub v_delta: f64,
    pub v_eta: f64,
    pub v_eps: f64,
    pub shared_bias_ceiling: f64,
    pub single_context_panel_ceiling: f64,
    pub panel_scaling_k1: Vec<CrossedVariancePoint>,
    pub context_scaling_m1: Vec<CrossedVariancePoint>,
    pub context_scaling_m4: Vec<CrossedVariancePoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AssessmentVarianceSuite {
    pub claim_boundary: &'static str,
    pub seed: u64,
    pub mc_subjects: usize,
    pub portfolio: PortfolioVarianceModel,
    pub crossed_generalizability: CrossedGeneralizabilityModel,
}

#[inline]
pub fn exact_portfolio_rg(
    v_g: f64,
    omega_sq: f64,
    sigma_c_sq: f64,
    sigma_e_sq: f64,
    k: usize,
    n: usize,
) -> f64 {
    v_g / (v_g + omega_sq + sigma_c_sq / k as f64 + sigma_e_sq / (k * n) as f64)
}

#[inline]
pub fn exact_crossed_vmk(
    v_t: f64,
    v_u: f64,
    v_delta: f64,
    v_eta: f64,
    v_eps: f64,
    m: usize,
    k: usize,
) -> f64 {
    v_t / (v_t + v_u + v_delta / k as f64 + v_eta / m as f64 + v_eps / (m * k) as f64)
}

fn squared_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mx;
        let dy = y - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 0.0 || vy <= 0.0 {
        0.0
    } else {
        (cov * cov) / (vx * vy)
    }
}

#[derive(Clone, Copy)]
struct PortfolioComponents {
    v_g: f64,
    omega_sq: f64,
    sigma_c_sq: f64,
    sigma_e_sq: f64,
}

#[derive(Clone, Copy)]
struct CrossedComponents {
    v_t: f64,
    v_u: f64,
    v_delta: f64,
    v_eta: f64,
    v_eps: f64,
}

fn mc_portfolio_rg(
    rng: &mut Rng,
    subjects: usize,
    c: PortfolioComponents,
    k: usize,
    n: usize,
) -> f64 {
    let sg = c.v_g.sqrt();
    let so = c.omega_sq.sqrt();
    // Since the average of K i.i.d. domain terms and K*n event terms is exact Gaussian with
    // variance sigma_c_sq/K + sigma_e_sq/(K*n), simulate domain & trial draws explicitly!
    let sc = c.sigma_c_sq.sqrt();
    let se = c.sigma_e_sq.sqrt();
    let mut g_true = Vec::with_capacity(subjects);
    let mut z_bar = Vec::with_capacity(subjects);
    let inv_k = 1.0 / k as f64;
    let inv_n = 1.0 / n as f64;

    for _ in 0..subjects {
        let g = sg * fast_normal(rng);
        let omega = so * fast_normal(rng);
        let mut dom_sum = 0.0;
        for _ in 0..k {
            let c_k = sc * fast_normal(rng);
            let mut ev_sum = 0.0;
            for _ in 0..n {
                ev_sum += se * fast_normal(rng);
            }
            dom_sum += c_k + ev_sum * inv_n;
        }
        g_true.push(g);
        z_bar.push(g + omega + dom_sum * inv_k);
    }
    squared_correlation(&g_true, &z_bar)
}

fn mc_crossed_vmk(rng: &mut Rng, subjects: usize, c: CrossedComponents, m: usize, k: usize) -> f64 {
    let st = c.v_t.sqrt();
    let su = c.v_u.sqrt();
    let sd = c.v_delta.sqrt();
    let se_r = c.v_eta.sqrt();
    let se_rc = c.v_eps.sqrt();
    let mut t_true = Vec::with_capacity(subjects);
    let mut z_bar = Vec::with_capacity(subjects);
    let inv_m = 1.0 / m as f64;
    let inv_k = 1.0 / k as f64;
    let inv_mk = 1.0 / (m * k) as f64;

    for _ in 0..subjects {
        let t = st * fast_normal(rng);
        let u = su * fast_normal(rng);
        let mut delta_sum = 0.0;
        for _ in 0..k {
            delta_sum += sd * fast_normal(rng);
        }
        let mut eta_sum = 0.0;
        for _ in 0..m {
            eta_sum += se_r * fast_normal(rng);
        }
        let mut eps_sum = 0.0;
        for _ in 0..(m * k) {
            eps_sum += se_rc * fast_normal(rng);
        }
        t_true.push(t);
        z_bar.push(t + u + delta_sum * inv_k + eta_sum * inv_m + eps_sum * inv_mk);
    }
    squared_correlation(&t_true, &z_bar)
}

pub fn assessment_variance_suite(seed: u64, quick: bool) -> AssessmentVarianceSuite {
    let subjects = if quick { 3_000 } else { 10_000 };
    let mut rng = Rng(seed ^ 0xA55E_5515_2026_1110);

    // Multi-signal portfolio model (shared-latent variance ceiling):
    // V_G = 1.0, omega^2 = 0.2, sigma_c^2 = 4.0, sigma_e^2 = 6.0
    let (v_g, omega_sq, sigma_c_sq, sigma_e_sq) = (1.0, 0.2, 4.0, 6.0);
    let port_comp = PortfolioComponents {
        v_g,
        omega_sq,
        sigma_c_sq,
        sigma_e_sq,
    };
    let budgets = [1usize, 2, 4, 8, 16, 32, 64];

    let mut repeated_curve = Vec::with_capacity(budgets.len());
    let mut diverse_curve = Vec::with_capacity(budgets.len());
    for &b in &budgets {
        let ex_rep = exact_portfolio_rg(v_g, omega_sq, sigma_c_sq, sigma_e_sq, 1, b);
        let mc_rep = mc_portfolio_rg(&mut rng, subjects, port_comp, 1, b);
        repeated_curve.push(PortfolioPoint {
            budget_n: b,
            k_domains: 1,
            n_per_domain: b,
            exact_r_g: ex_rep,
            mc_r_g: mc_rep,
            abs_error: (ex_rep - mc_rep).abs(),
        });

        let ex_div = exact_portfolio_rg(v_g, omega_sq, sigma_c_sq, sigma_e_sq, b, 1);
        let mc_div = mc_portfolio_rg(&mut rng, subjects, port_comp, b, 1);
        diverse_curve.push(PortfolioPoint {
            budget_n: b,
            k_domains: b,
            n_per_domain: 1,
            exact_r_g: ex_div,
            mc_r_g: mc_div,
            abs_error: (ex_div - mc_div).abs(),
        });
    }

    let mut fixed_budget_64 = Vec::with_capacity(budgets.len());
    for &k in &budgets {
        let n = 64 / k;
        let ex = exact_portfolio_rg(v_g, omega_sq, sigma_c_sq, sigma_e_sq, k, n);
        let mc = mc_portfolio_rg(&mut rng, subjects, port_comp, k, n);
        fixed_budget_64.push(PortfolioPoint {
            budget_n: 64,
            k_domains: k,
            n_per_domain: n,
            exact_r_g: ex,
            mc_r_g: mc,
            abs_error: (ex - mc).abs(),
        });
    }

    // Crossed evaluator x context generalizability V(m, k) (`Cronbach et al. 1972`, Prop. 10)
    let (v_t, v_u, v_delta, v_eta, v_eps) = (1.0, 0.2, 3.0, 1.0, 4.0);
    let cross_comp = CrossedComponents {
        v_t,
        v_u,
        v_delta,
        v_eta,
        v_eps,
    };
    let grid = [1usize, 2, 4, 8, 16, 32, 64];

    let panel_scaling_k1: Vec<CrossedVariancePoint> = grid
        .iter()
        .map(|&m| {
            let ex = exact_crossed_vmk(v_t, v_u, v_delta, v_eta, v_eps, m, 1);
            let mc = mc_crossed_vmk(&mut rng, subjects, cross_comp, m, 1);
            CrossedVariancePoint {
                m_evaluators: m,
                k_contexts: 1,
                exact_v_mk: ex,
                mc_v_mk: mc,
                abs_error: (ex - mc).abs(),
            }
        })
        .collect();

    let context_scaling_m1: Vec<CrossedVariancePoint> = grid
        .iter()
        .map(|&k| {
            let ex = exact_crossed_vmk(v_t, v_u, v_delta, v_eta, v_eps, 1, k);
            let mc = mc_crossed_vmk(&mut rng, subjects, cross_comp, 1, k);
            CrossedVariancePoint {
                m_evaluators: 1,
                k_contexts: k,
                exact_v_mk: ex,
                mc_v_mk: mc,
                abs_error: (ex - mc).abs(),
            }
        })
        .collect();

    let context_scaling_m4: Vec<CrossedVariancePoint> = grid
        .iter()
        .map(|&k| {
            let ex = exact_crossed_vmk(v_t, v_u, v_delta, v_eta, v_eps, 4, k);
            let mc = mc_crossed_vmk(&mut rng, subjects, cross_comp, 4, k);
            CrossedVariancePoint {
                m_evaluators: 4,
                k_contexts: k,
                exact_v_mk: ex,
                mc_v_mk: mc,
                abs_error: (ex - mc).abs(),
            }
        })
        .collect();

    AssessmentVarianceSuite {
        claim_boundary: CLAIM_BOUNDARY_MONITORS,
        seed,
        mc_subjects: subjects,
        portfolio: PortfolioVarianceModel {
            v_g,
            omega_sq,
            sigma_c_sq,
            sigma_e_sq,
            shared_bias_ceiling: v_g / (v_g + omega_sq),
            single_domain_ceiling: v_g / (v_g + omega_sq + sigma_c_sq),
            repeat_64: exact_portfolio_rg(v_g, omega_sq, sigma_c_sq, sigma_e_sq, 1, 64),
            diverse_64: exact_portfolio_rg(v_g, omega_sq, sigma_c_sq, sigma_e_sq, 64, 1),
            repeated_single_domain_curve: repeated_curve,
            diverse_domains_curve: diverse_curve,
            fixed_budget_64_tradeoff: fixed_budget_64,
        },
        crossed_generalizability: CrossedGeneralizabilityModel {
            citation: "Cronbach et al. (1972) Generalizability Theory",
            v_t,
            v_u,
            v_delta,
            v_eta,
            v_eps,
            shared_bias_ceiling: v_t / (v_t + v_u),
            single_context_panel_ceiling: v_t / (v_t + v_u + v_delta),
            panel_scaling_k1,
            context_scaling_m1,
            context_scaling_m4,
        },
    }
}

pub fn assessment_portfolio_and_crossed_variance(
    seed: u64,
    quick: bool,
) -> AssessmentVarianceSuite {
    assessment_variance_suite(seed, quick)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_hermite_and_copula_match_reference_to_machine_precision() {
        let (nodes, weights) = gauss_hermite_probabilist(200);
        assert_eq!(nodes.len(), 200);
        let m0: f64 = weights.iter().sum();
        let m1: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x).sum();
        let m2: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x * x).sum();
        let m4: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x.powi(4)).sum();
        let m6: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x.powi(6)).sum();
        assert!((m0 - 1.0).abs() < 1e-14);
        assert!(m1.abs() < 1e-14);
        assert!((m2 - 1.0).abs() < 1e-13);
        assert!((m4 - 3.0).abs() < 1e-12);
        assert!((m6 - 15.0).abs() < 1e-11);

        let cop = copula_suite();
        // Verify Ziggurat exact standard normal moments & tail CDFs
        let mut z_rng = Rng(20261110);
        let z_n = 200_000usize;
        let zs: Vec<f64> = (0..z_n).map(|_| fast_normal(&mut z_rng)).collect();
        let z_m1 = zs.iter().sum::<f64>() / z_n as f64;
        let z_m2 = zs.iter().map(|z| z * z).sum::<f64>() / z_n as f64;
        let z_m4 = zs.iter().map(|z| z.powi(4)).sum::<f64>() / z_n as f64;
        let z_q10 = zs.iter().filter(|&&z| z < -1.2815515655446004).count() as f64 / z_n as f64;
        assert!(z_m1.abs() < 0.01);
        assert!((z_m2 - 1.0).abs() < 0.015);
        assert!((z_m4 - 3.0).abs() < 0.06);
        assert!((z_q10 - 0.10).abs() < 0.004);
        // Exact cross-check against `target/tips-check/results.json` (`roots_hermitenorm(200)`)
        assert!((cop.curves["0.0"][5] - 1e-6).abs() < 1e-16);
        assert!((cop.curves["0.3"][3] - 0.0028311190899769465).abs() < 1e-12);
        assert!((cop.curves["0.3"][5] - 0.0007230702316007424).abs() < 1e-12);
        assert!((cop.curves["0.3"][7] - 0.0002605786202398544).abs() < 1e-12);
        assert!((cop.curves["0.6"][3] - 0.014474547113671998).abs() < 1e-12);
        assert!((cop.curves["0.6"][5] - 0.00811795445915844).abs() < 1e-12);
        assert!((cop.curves["0.6"][7] - 0.005417208273487004).abs() < 1e-12);
        assert!((cop.curves["0.9"][5] - 0.041397038527105315).abs() < 1e-12);
        assert!((cop.curves["0.9"][7] - 0.03674188036310797).abs() < 1e-12);
    }

    #[test]
    fn six_family_monitors_quick_verifies_shared_vs_distinct_and_retries() {
        let suite = monitors_v3_suite(20261110, true);
        assert_eq!(suite.families.len(), 6);
        for fam in &suite.families {
            let miss = suite.per_family_passive_miss[*fam].mean;
            let fpr = suite.per_family_benign_fpr[*fam].mean;
            assert!(
                miss < 0.55,
                "family {fam} failed to learn nonlinear 20D task: miss={miss}"
            );
            assert!(
                (0.01..=0.11).contains(&fpr),
                "family {fam} calibration out of bounds: fpr={fpr}"
            );
        }

        // Payload preservation and L2 ball projection hold strictly
        assert!(suite.attack_integrity.max_l2_overshoot <= 1e-11);
        assert!(suite.attack_integrity.min_payload_hazard_delta >= -1e-12);

        // Shared-step stacking (both homogeneous and 6-family heterogeneous) inflates
        // massively over product of marginals, whereas distinct-step gates match product.
        let hom0 = &suite.same_step_homogeneous["0.0"];
        let het0 = &suite.same_step_heterogeneous["0.0"];
        let dist0 = &suite.distinct_steps_heterogeneous["0.0"];
        assert!(hom0.joint_to_product_ratio_k6 > 10.0);
        assert!(het0.joint_to_product_ratio_k6 > 10.0);
        assert!(
            (dist0.joint_mean[5] - dist0.product_mean[5]).abs() < 0.02,
            "distinct passive joint={} vs prod={}",
            dist0.joint_mean[5],
            dist0.product_mean[5]
        );

        let het2 = &suite.same_step_heterogeneous["2.0"];
        let dist2 = &suite.distinct_steps_heterogeneous["2.0"];
        assert!(het2.joint_mean[5] > 3.0 * dist2.joint_mean[5]);
        assert!((dist2.joint_mean[5] - dist2.product_mean[5]).abs() < 0.04);

        // Benign false-block compounds with k
        assert!(
            suite.benign_false_block.distinct_steps_heterogeneous[5].mean
                > suite.benign_false_block.distinct_steps_heterogeneous[0].mean + 0.12
        );

        // Adaptive 8-query feedback increases evasion over 0-feedback
        assert!(
            suite.adaptive_feedback.same_step_score_feedback_8q_k6.mean
                >= suite.adaptive_feedback.same_step_no_feedback_k6.mean
        );

        // Read-only retry selection rises steeply and tracks 1 - (1 - J_6)^K on independent draws
        let r1 = &suite.readonly_retry_selection[0];
        let r32 = &suite.readonly_retry_selection[5];
        assert_eq!(r1.k_select, 1);
        assert_eq!(r32.k_select, 32);
        assert!(r32.independent_retry_miss.mean > r1.independent_retry_miss.mean + 0.30);
        assert!(r32.abs_formula_gap < 0.08);
    }

    #[test]
    fn assessment_variance_matches_exact_portfolio_and_cronbach_ceilings() {
        let av = assessment_variance_suite(20261110, true);
        assert!((av.portfolio.repeat_64 - 0.18890200708382526).abs() < 1e-12);
        assert!((av.portfolio.diverse_64 - 0.7373271889400922).abs() < 1e-12);
        assert!((av.portfolio.shared_bias_ceiling - 0.8333333333333334).abs() < 1e-12);
        assert!(
            (av.portfolio.repeated_single_domain_curve[0].exact_r_g
                - av.portfolio.diverse_domains_curve[0].exact_r_g)
                .abs()
                < 1e-15
        );
        for pt in &av.portfolio.fixed_budget_64_tradeoff {
            assert!(pt.exact_r_g < av.portfolio.shared_bias_ceiling);
            assert!(pt.abs_error < 0.04);
        }
        for pt in &av.crossed_generalizability.panel_scaling_k1 {
            assert!(pt.exact_v_mk < av.crossed_generalizability.single_context_panel_ceiling);
            assert!(pt.abs_error < 0.04);
        }

        // Strict deterministic reproducibility across runs
        let av2 = assessment_variance_suite(20261110, true);
        assert_eq!(
            serde_json::to_string(&av).unwrap(),
            serde_json::to_string(&av2).unwrap()
        );
    }

    #[test]
    #[ignore]
    fn publication_monitors_and_variance_verification() {
        let t0 = std::time::Instant::now();
        let cop = copula_suite();
        let cop_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = std::time::Instant::now();
        let mon_q = monitors_v3_suite(20261110, true);
        let mon_q_s = t1.elapsed().as_secs_f64();

        let t2 = std::time::Instant::now();
        let mon_p = monitors_v3_suite(20261110, false);
        let mon_p_s = t2.elapsed().as_secs_f64();

        let t3 = std::time::Instant::now();
        let av_p = assessment_variance_suite(20261110, false);
        let av_p_ms = t3.elapsed().as_secs_f64() * 1e3;

        println!(
            "TIMINGS: copula={cop_ms:.2}ms, monitors_quick={mon_q_s:.3}s, monitors_pub={mon_p_s:.3}s, variance_pub={av_p_ms:.2}ms"
        );
        println!(
            "COPULA k=6: rho0.0={:.6e}, rho0.3={:.6}, rho0.6={:.6}, rho0.9={:.6}",
            cop.curves["0.0"][5], cop.curves["0.3"][5], cop.curves["0.6"][5], cop.curves["0.9"][5]
        );
        println!(
            "COPULA k=8: rho0.0={:.6e}, rho0.3={:.6}, rho0.6={:.6}, rho0.9={:.6}",
            cop.curves["0.0"][7], cop.curves["0.3"][7], cop.curves["0.6"][7], cop.curves["0.9"][7]
        );
        println!(
            "MON_PUB passive k=6: same_hom={:.4} (ratio {:.1}x), same_het={:.4} (ratio {:.1}x), dist_het={:.5} (prod {:.5})",
            mon_p.same_step_homogeneous["0.0"].joint_mean[5],
            mon_p.same_step_homogeneous["0.0"].joint_to_product_ratio_k6,
            mon_p.same_step_heterogeneous["0.0"].joint_mean[5],
            mon_p.same_step_heterogeneous["0.0"].joint_to_product_ratio_k6,
            mon_p.distinct_steps_heterogeneous["0.0"].joint_mean[5],
            mon_p.distinct_steps_heterogeneous["0.0"].product_mean[5],
        );
        println!(
            "MON_PUB rho=2.0 k=6: surr_pass={:.4}, same_hom={:.4}, same_het={:.4} (ratio {:.1}x), dist_het={:.4} (prod {:.4})",
            mon_p.surrogate_pass["2.0"].mean,
            mon_p.same_step_homogeneous["2.0"].joint_mean[5],
            mon_p.same_step_heterogeneous["2.0"].joint_mean[5],
            mon_p.same_step_heterogeneous["2.0"].joint_to_product_ratio_k6,
            mon_p.distinct_steps_heterogeneous["2.0"].joint_mean[5],
            mon_p.distinct_steps_heterogeneous["2.0"].product_mean[5],
        );
        println!(
            "MON_PUB feedback (rho=2.0, k=6): same_no_fb={:.4}, same_bin8={:.4}, same_score8={:.4}, dist_no_fb={:.4}, dist_score8={:.4}",
            mon_p.adaptive_feedback.same_step_no_feedback_k6.mean,
            mon_p.adaptive_feedback.same_step_binary_feedback_8q_k6.mean,
            mon_p.adaptive_feedback.same_step_score_feedback_8q_k6.mean,
            mon_p.adaptive_feedback.distinct_steps_no_feedback_k6.mean,
            mon_p
                .adaptive_feedback
                .distinct_steps_score_feedback_8q_k6
                .mean,
        );
        println!(
            "MON_PUB retry: K=1 {:.4} (form {:.4}), K=8 {:.4} (form {:.4}), K=32 {:.4} (form {:.4}, corr {:.4})",
            mon_p.readonly_retry_selection[0]
                .independent_retry_miss
                .mean,
            mon_p.readonly_retry_selection[0].formula_prediction.mean,
            mon_p.readonly_retry_selection[3]
                .independent_retry_miss
                .mean,
            mon_p.readonly_retry_selection[3].formula_prediction.mean,
            mon_p.readonly_retry_selection[5]
                .independent_retry_miss
                .mean,
            mon_p.readonly_retry_selection[5].formula_prediction.mean,
            mon_p.readonly_retry_selection[5].correlated_drift_miss.mean,
        );
        println!(
            "MON_PUB false_block dist k=1->6: {:.4} -> {:.4} (indep {:.4})",
            mon_p.benign_false_block.distinct_steps_heterogeneous[0].mean,
            mon_p.benign_false_block.distinct_steps_heterogeneous[5].mean,
            mon_p.benign_false_block.theoretical_independent[5],
        );

        assert_eq!(mon_q.config.tasks, 6);
        assert_eq!(mon_p.config.tasks, 24);
        assert!(mon_q_s < 1.0, "monitors quick exceeded 1s: {mon_q_s:.3}s");
        assert!(mon_p_s < 5.0, "monitors pub exceeded 5s: {mon_p_s:.3}s");
        assert!(mon_p.same_step_heterogeneous["0.0"].joint_to_product_ratio_k6 > 20.0);
        assert!(mon_p.same_step_heterogeneous["2.0"].joint_to_product_ratio_k6 > 6.0);
        assert!(
            mon_p.same_step_heterogeneous["2.0"].joint_mean[5]
                / mon_p.same_step_heterogeneous["0.0"].product_mean[5]
                > 30.0
        );
        assert!(
            mon_p.same_step_heterogeneous["2.0"].joint_mean[5]
                > 5.0 * mon_p.distinct_steps_heterogeneous["2.0"].joint_mean[5]
        );
        assert!(
            (mon_p.distinct_steps_heterogeneous["2.0"].joint_mean[5]
                - mon_p.distinct_steps_heterogeneous["2.0"].product_mean[5])
                .abs()
                < 0.025
        );
        assert!(
            av_p.portfolio
                .fixed_budget_64_tradeoff
                .iter()
                .all(|p| p.abs_error < 0.025)
        );
    }
}
