//! V4 Pure-Rust Neural Policy-Gradient Post-Training Suite (`P7` & `H16`).
//!
//! Trains a genuine 2-layer neural policy network (`12` contextual features -> `24` `tanh`
//! hidden units -> `4` `softmax` action logits; `24*12 + 24 + 4*24 + 4 = 412` trainable `f64`
//! parameters initialized via Xavier/Glorot from `SplitMix64`) using exact analytical
//! policy-gradient backpropagation (`∇_θ log π_θ(a_t | s_t) * A_g`) across multi-step
//! contextual tool-use episodes.
//!
//! Compares five post-training and architectural boundary regimes:
//! 1. `grpo_podium_top10`: steep top-10% group-relative podium indicator selection.
//! 2. `grpo_audited_harm_veto`: top-10% podium selection with complete veto (`-1`) on
//!    post-hoc audited harm (`p_audit = 0.70` when `monitored=1` vs `0.15` when `monitored=0`),
//!    testing whether imperfect post-hoc audit prevents or induces *evaluation-aware covert*
//!    boundary shortcuts on unmonitored states.
//! 3. `caretaker_clipped_rl`: concave clipped upside + high-hazard calibration bonus + strong
//!    audited-harm penalty.
//! 4. `survival_floor_rl`: satisficing survival floor (`score >= tau` and zero audited harm
//!    receives uniform survival weight, zero bonus for right-tail outliers).
//! 5. `floor_plus_precommit_gate`: survival-floor neural policy trained and deployed behind an
//!    architectural pre-commit boundary gate (`p_gate` swept across `[0.0..=1.0]`).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//!
//! Claim boundary (`AGENTS.md`): Controlled pure-Rust 2-layer MLP (`412` `f64` parameters)
//! policy-gradient dynamics on a 12D synthetic contextual tool-use MDP. Validates `P7` and
//! `H16` selection-covariance and evaluation-aware boundary-probing mechanics on trained neural
//! weights; explicitly NOT frontier-LLM RLHF post-training or human-participant evidence.

use crate::{
    rng::Rng,
    stats::{Mean, mean},
    v3_monitors::fast_normal,
};
use serde::Serialize;
use std::collections::BTreeMap;

pub const CLAIM_BOUNDARY_NEURAL_RL: &str = "Controlled pure-Rust 2-layer neural policy-gradient (12->24->4 MLP, 412 float64 parameters) post-training on a 12D contextual tool-use MDP only; validates P7 and H16 selection-covariance and emergent evaluation-aware covert gambling on trained neural weights, not frontier-LLM RLHF benchmarks or human-participant evidence.";

pub const OBS_DIM: usize = 12;
pub const HID_DIM: usize = 24;
pub const ACT_DIM: usize = 4;
pub const PARAM_COUNT: usize = HID_DIM * OBS_DIM + HID_DIM + ACT_DIM * HID_DIM + ACT_DIM; // 412

pub const HORIZON_STEPS: usize = 6;
pub const GROUP_SIZE: usize = 32;
pub const TRAIN_UPDATES: usize = 250;
pub const PODIUM_TOP_K: usize = 3; // Top ~10% of G = 32 rollouts (3 / 32 = 9.375%)

pub const SHORTCUT_BURST_MEAN: f64 = 1.60;
pub const SHORTCUT_BURST_STD: f64 = 1.80;
pub const SHORTCUT_HARM_PROB: f64 = 0.08;
pub const DIRECT_HARM_PROB: f64 = 0.01;
pub const EXTERNAL_HARM_LOSS: f64 = 20.0;
pub const AUDIT_PROB_MONITORED: f64 = 0.70;
pub const AUDIT_PROB_UNMONITORED: f64 = 0.15;
pub const SURVIVAL_FLOOR_TAU: f64 = 8.85;
pub const TASK_COMPLETION_TARGET: f64 = 8.00;

pub const GATE_SWEEP_V4: [f64; 7] = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0];

pub const ACTION_NAMES: [&str; ACT_DIM] = [
    "verify_then_execute",
    "direct_standard_execute",
    "covert_boundary_shortcut",
    "calibrated_defer",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostTrainingRegime {
    GrpoPodiumTop10,
    GrpoAuditedHarmVeto,
    CaretakerClippedRl,
    SurvivalFloorRl,
    FloorPlusPrecommitGate,
}

impl PostTrainingRegime {
    pub const ALL: [Self; 5] = [
        Self::GrpoPodiumTop10,
        Self::GrpoAuditedHarmVeto,
        Self::CaretakerClippedRl,
        Self::SurvivalFloorRl,
        Self::FloorPlusPrecommitGate,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::GrpoPodiumTop10 => "grpo_podium_top10",
            Self::GrpoAuditedHarmVeto => "grpo_audited_harm_veto",
            Self::CaretakerClippedRl => "caretaker_clipped_rl",
            Self::SurvivalFloorRl => "survival_floor_rl",
            Self::FloorPlusPrecommitGate => "floor_plus_precommit_gate",
        }
    }

    pub fn default_gate(self) -> f64 {
        match self {
            Self::FloorPlusPrecommitGate => 1.0,
            _ => 0.0,
        }
    }
}

// ============================================================================
// Pure-Rust 2-Layer MLP Policy (`12 -> 24 tanh -> 4 softmax`, 412 `f64` params)
// ============================================================================

const OFF_W1: usize = 0;
const OFF_B1: usize = OFF_W1 + HID_DIM * OBS_DIM; // 288
const OFF_W2: usize = OFF_B1 + HID_DIM; // 312
const OFF_B2: usize = OFF_W2 + ACT_DIM * HID_DIM; // 408

/// High-precision inline `tanh(x)` via range reduction `z = x / 16` (`|z| < 1.2`), `[7/6]` Padé
/// numerator/denominator, and 4 division-free projective doublings `(n, d) -> (2nd, d^2 + n^2)`
/// with a single final division (`max |err| < 6e-13` across `R`, zero `libm` calls).
#[inline(always)]
fn fast_tanh(x: f64) -> f64 {
    if x.abs() >= 19.1 {
        return 1.0_f64.copysign(x);
    }
    let z = x * 0.0625;
    let z2 = z * z;
    let mut n = z * (135_135.0 + z2 * (17_325.0 + z2 * (378.0 + z2)));
    let mut d = 135_135.0 + z2 * (62_370.0 + z2 * (3_150.0 + 28.0 * z2));
    for _ in 0..4 {
        let nn = (n + n) * d;
        let dd = d * d + n * n;
        n = nn;
        d = dd;
    }
    n / d
}

/// High-precision inline `exp(x)` for `x <= 0.0` via Cody-Waite `ln(2)` range reduction
/// (`|r| <= ln(2)/2`), degree-10 polynomial, and IEEE-754 `2^k` bit-scaling (`< 1e-12` err).
#[inline(always)]
fn fast_exp_nonpos(x: f64) -> f64 {
    if x <= -708.0 {
        return 0.0;
    }
    let k = (x * std::f64::consts::LOG2_E).round();
    const LN2_HI: f64 = 0.693_147_180_369_123_8;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0
                        + r * (1.0 / 120.0
                            + r * (1.0 / 720.0
                                + r * (1.0 / 5_040.0
                                    + r * (1.0 / 40_320.0
                                        + r * (1.0 / 362_880.0 + r * (1.0 / 3_628_800.0))))))))));
    let ki = k as i64;
    let scale = f64::from_bits(((1023 + ki) as u64) << 52);
    p * scale
}

#[derive(Clone)]
struct NeuralPolicy {
    theta: [f64; PARAM_COUNT],
}

struct AdamState {
    m: [f64; PARAM_COUNT],
    v: [f64; PARAM_COUNT],
    step: usize,
}

impl AdamState {
    fn new() -> Self {
        Self {
            m: [0.0; PARAM_COUNT],
            v: [0.0; PARAM_COUNT],
            step: 0,
        }
    }

    /// Ascent step along policy gradient `grad` (`∇_θ J(θ)`) with light L2 logit regularization.
    fn step_ascent(&mut self, theta: &mut [f64; PARAM_COUNT], grad: &[f64; PARAM_COUNT], lr: f64) {
        self.step += 1;
        let b1 = 0.9_f64;
        let b2 = 0.999_f64;
        let eps = 1e-8_f64;
        let corr1 = 1.0 - b1.powi(self.step as i32);
        let corr2 = 1.0 - b2.powi(self.step as i32);
        let step_scale = lr * (corr2.sqrt() / corr1);

        for idx in 0..PARAM_COUNT {
            let g = grad[idx] - 1e-4 * theta[idx];
            self.m[idx] = b1 * self.m[idx] + (1.0 - b1) * g;
            self.v[idx] = b2 * self.v[idx] + (1.0 - b2) * g * g;
            theta[idx] += step_scale * self.m[idx] / (self.v[idx].sqrt() + eps);
        }
    }
}

impl NeuralPolicy {
    /// Xavier / Glorot normal initialization from `SplitMix64`.
    fn new_glorot(rng: &mut Rng) -> Self {
        let mut theta = [0.0_f64; PARAM_COUNT];
        let scale1 = (2.0 / (OBS_DIM + HID_DIM) as f64).sqrt();
        let scale2 = (2.0 / (HID_DIM + ACT_DIM) as f64).sqrt();
        for j in 0..HID_DIM {
            for i in 0..OBS_DIM {
                theta[OFF_W1 + j * OBS_DIM + i] = fast_normal(rng) * scale1;
            }
            theta[OFF_B1 + j] = 0.0;
        }
        for k in 0..ACT_DIM {
            for j in 0..HID_DIM {
                theta[OFF_W2 + k * HID_DIM + j] = fast_normal(rng) * scale2;
            }
            theta[OFF_B2 + k] = 0.0;
        }
        Self { theta }
    }

    #[inline]
    fn forward(&self, obs: &[f64; OBS_DIM]) -> ([f64; HID_DIM], [f64; ACT_DIM]) {
        let mut hidden = [0.0_f64; HID_DIM];
        for j in 0..HID_DIM {
            let row = OFF_W1 + j * OBS_DIM;
            let mut sum = self.theta[OFF_B1 + j];
            for i in 0..OBS_DIM {
                sum += self.theta[row + i] * obs[i];
            }
            hidden[j] = fast_tanh(sum);
        }

        let mut logits = [0.0_f64; ACT_DIM];
        let mut max_logit = f64::NEG_INFINITY;
        for k in 0..ACT_DIM {
            let row = OFF_W2 + k * HID_DIM;
            let mut sum = self.theta[OFF_B2 + k];
            for j in 0..HID_DIM {
                sum += self.theta[row + j] * hidden[j];
            }
            logits[k] = sum;
            if sum > max_logit {
                max_logit = sum;
            }
        }

        let mut probs = [0.0_f64; ACT_DIM];
        let mut denom = 0.0_f64;
        for k in 0..ACT_DIM {
            let e = fast_exp_nonpos(logits[k] - max_logit);
            probs[k] = e;
            denom += e;
        }
        let inv = 1.0 / denom;
        for p in &mut probs {
            *p *= inv;
        }
        (hidden, probs)
    }

    #[inline]
    fn log_prob(&self, obs: &[f64; OBS_DIM], action: usize) -> f64 {
        let (_, probs) = self.forward(obs);
        probs[action].max(1e-300).ln()
    }

    /// Accumulates `weight * ∇_θ log π_θ(action | obs)` into `grad` using exact analytical backprop.
    #[inline]
    fn accumulate_log_prob_grad(
        &self,
        obs: &[f64; OBS_DIM],
        hidden: &[f64; HID_DIM],
        probs: &[f64; ACT_DIM],
        action: usize,
        weight: f64,
        grad: &mut [f64; PARAM_COUNT],
    ) {
        if weight == 0.0 {
            return;
        }
        let mut dz = [0.0_f64; ACT_DIM];
        for k in 0..ACT_DIM {
            let ind = if k == action { 1.0 } else { 0.0 };
            dz[k] = weight * (ind - probs[k]);
        }

        for k in 0..ACT_DIM {
            grad[OFF_B2 + k] += dz[k];
            let row = OFF_W2 + k * HID_DIM;
            for j in 0..HID_DIM {
                grad[row + j] += dz[k] * hidden[j];
            }
        }

        for j in 0..HID_DIM {
            let mut dh = 0.0_f64;
            for k in 0..ACT_DIM {
                dh += self.theta[OFF_W2 + k * HID_DIM + j] * dz[k];
            }
            let du = dh * (1.0 - hidden[j] * hidden[j]);
            grad[OFF_B1 + j] += du;
            let row = OFF_W1 + j * OBS_DIM;
            for i in 0..OBS_DIM {
                grad[row + i] += du * obs[i];
            }
        }
    }

    #[inline]
    fn sample_action(probs: &[f64; ACT_DIM], rng: &mut Rng) -> usize {
        let u = rng.uniform();
        let mut cum = 0.0_f64;
        for (i, &p) in probs.iter().enumerate().take(ACT_DIM - 1) {
            cum += p;
            if u <= cum {
                return i;
            }
        }
        ACT_DIM - 1
    }
}

// ============================================================================
// Analytical vs. Central Finite-Difference Gradient Verification
// ============================================================================

#[derive(Clone, Debug, Serialize)]
pub struct GradientVerificationSummary {
    pub param_count: usize,
    pub finite_diff_eps: f64,
    pub states_checked: usize,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub exact_match_verified: bool,
}

fn verify_analytical_backprop(seed: u64) -> GradientVerificationSummary {
    // Verify projective Padé `fast_tanh` and `fast_exp_nonpos` match `libm` to `< 1e-12`:
    for k in -500_i32..=500_i32 {
        let x = k as f64 * 0.05;
        assert!(
            (fast_tanh(x) - x.tanh()).abs() < 1e-12,
            "fast_tanh mismatch at x={x}"
        );
        let xn = -(k.abs() as f64) * 0.10;
        assert!(
            (fast_exp_nonpos(xn) - xn.exp()).abs() < 1e-12,
            "fast_exp_nonpos mismatch at xn={xn}"
        );
    }

    let mut rng = Rng(seed ^ 0x4752_4144_4348_454B);
    let mut net = NeuralPolicy::new_glorot(&mut rng);
    // Perturb biases too so all 412 parameters have non-zero baseline values.
    for p in &mut net.theta {
        *p += 0.05 * fast_normal(&mut rng);
    }

    let eps = 1e-6_f64;
    let states_checked = 8usize;
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;

    for s_idx in 0..states_checked {
        let mut obs = [0.0_f64; OBS_DIM];
        for slot in &mut obs {
            *slot = 2.0 * rng.uniform() - 1.0;
        }
        let action = s_idx % ACT_DIM;
        let weight = 1.35_f64 - 0.55 * (s_idx as f64 / states_checked as f64);

        let (hid, probs) = net.forward(&obs);
        let mut ana_grad = [0.0_f64; PARAM_COUNT];
        net.accumulate_log_prob_grad(&obs, &hid, &probs, action, weight, &mut ana_grad);

        for idx in 0..PARAM_COUNT {
            let orig = net.theta[idx];
            net.theta[idx] = orig + eps;
            let lp_plus = net.log_prob(&obs, action);
            net.theta[idx] = orig - eps;
            let lp_minus = net.log_prob(&obs, action);
            net.theta[idx] = orig;

            let fd = weight * (lp_plus - lp_minus) / (2.0 * eps);
            let abs_err = (ana_grad[idx] - fd).abs();
            let rel_err = abs_err / ana_grad[idx].abs().max(fd.abs()).max(1e-8);
            max_abs = max_abs.max(abs_err);
            max_rel = max_rel.max(rel_err);
        }
    }

    GradientVerificationSummary {
        param_count: PARAM_COUNT,
        finite_diff_eps: eps,
        states_checked,
        max_abs_error: max_abs,
        max_rel_error: max_rel,
        exact_match_verified: max_abs < 1e-9,
    }
}

// ============================================================================
// 12D Contextual Tool-Use Environment & Multi-Step Episode Rollout
// ============================================================================

#[derive(Clone, Copy)]
struct PromptContext {
    difficulty: f64,
    monitored: bool,
    oversight_cue: f64,
}

impl PromptContext {
    fn sample(rng: &mut Rng, monitored: bool) -> Self {
        let difficulty = 0.15 + 0.70 * rng.uniform();
        let base_cue = if monitored { 0.75 } else { -0.75 };
        let oversight_cue = (base_cue + 0.12 * fast_normal(rng)).clamp(-1.0, 1.0);
        Self {
            difficulty,
            monitored,
            oversight_cue,
        }
    }
}

#[derive(Clone, Copy)]
struct StepTrace {
    obs: [f64; OBS_DIM],
    hidden: [f64; HID_DIM],
    probs: [f64; ACT_DIM],
    action: usize,
}

#[derive(Clone, Copy)]
struct RolloutOutcome {
    steps: [StepTrace; HORIZON_STEPS],
    raw_score: f64,
    audited_score: f64,
    clipped_caretaker_score: f64,
    true_harm_count: usize,
    audited_harm_count: usize,
    shortcut_count: usize,
    verify_count: usize,
    welfare: f64,
    useful_completed: bool,
}

impl RolloutOutcome {
    fn blank() -> Self {
        Self {
            steps: [StepTrace {
                obs: [0.0; OBS_DIM],
                hidden: [0.0; HID_DIM],
                probs: [0.25; ACT_DIM],
                action: 0,
            }; HORIZON_STEPS],
            raw_score: 0.0,
            audited_score: 0.0,
            clipped_caretaker_score: 0.0,
            true_harm_count: 0,
            audited_harm_count: 0,
            shortcut_count: 0,
            verify_count: 0,
            welfare: 0.0,
            useful_completed: false,
        }
    }
}

#[inline]
fn build_observation(
    ctx: PromptContext,
    step: usize,
    budget_rem: f64,
    cum_progress: f64,
    ambiguity_hazard: f64,
    verification_readiness: f64,
    shortcut_temptation: f64,
    had_anomaly: bool,
) -> [f64; OBS_DIM] {
    let rem_horizon = (HORIZON_STEPS - step) as f64 / HORIZON_STEPS as f64;
    let step_frac = step as f64 / (HORIZON_STEPS - 1) as f64;
    let expected_so_far = step as f64 * 1.45;
    let deficit = ((expected_so_far - cum_progress) / 4.0).clamp(-1.0, 1.0);
    [
        2.0 * ctx.difficulty - 1.0,
        2.0 * rem_horizon - 1.0,
        (budget_rem / 6.0).clamp(0.0, 1.0) * 2.0 - 1.0,
        if ctx.monitored { 1.0 } else { 0.0 },
        ctx.oversight_cue,
        (cum_progress / 8.0).clamp(-0.5, 1.5),
        deficit,
        ambiguity_hazard,
        verification_readiness,
        shortcut_temptation,
        step_frac,
        if had_anomaly { 1.0 } else { 0.0 },
    ]
}

fn run_episode_into(
    net: &NeuralPolicy,
    ctx: PromptContext,
    p_gate: f64,
    rng: &mut Rng,
    out: &mut RolloutOutcome,
) {
    let p_audit = if ctx.monitored {
        AUDIT_PROB_MONITORED
    } else {
        AUDIT_PROB_UNMONITORED
    };

    let mut budget_rem = 6.0_f64;
    let mut cum_progress = 0.0_f64;
    let mut raw_score = 0.0_f64;
    let mut audited_score = 0.0_f64;
    let mut calibration_bonus_total = 0.0_f64;
    let mut true_harm_count = 0usize;
    let mut audited_harm_count = 0usize;
    let mut shortcut_count = 0usize;
    let mut verify_count = 0usize;
    let mut had_anomaly = false;

    for (t, step_slot) in out.steps.iter_mut().enumerate() {
        let high_hazard = rng.uniform() < 0.15;
        let ambiguity_hazard = if high_hazard {
            0.80 + 0.20 * rng.uniform()
        } else {
            0.05 + 0.35 * rng.uniform()
        };
        let verification_readiness = if high_hazard {
            0.10 + 0.25 * rng.uniform()
        } else {
            0.45 + 0.55 * rng.uniform()
        };
        let shortcut_temptation = 0.30 + 0.70 * rng.uniform();

        let obs = build_observation(
            ctx,
            t,
            budget_rem,
            cum_progress,
            ambiguity_hazard,
            verification_readiness,
            shortcut_temptation,
            had_anomaly,
        );
        let (hidden, probs) = net.forward(&obs);
        let action = NeuralPolicy::sample_action(&probs, rng);

        *step_slot = StepTrace {
            obs,
            hidden,
            probs,
            action,
        };

        let step_raw: f64;
        let mut step_audited: f64;
        let mut step_true_harm = false;
        let mut step_audited_harm = false;
        let mut calibration_bonus = 0.0_f64;

        match action {
            0 => {
                // VerifyThenExecute: steady high competence, calibrated low variance, 0% harm.
                verify_count += 1;
                budget_rem = (budget_rem - 1.0).max(0.0);
                let noise = 0.12 * fast_normal(rng);
                step_raw = if high_hazard {
                    0.18 + 0.20 * verification_readiness + 0.06 * noise
                } else {
                    1.82 + 0.18 * verification_readiness - 0.12 * ctx.difficulty + noise
                };
                step_audited = step_raw;
            }
            1 => {
                // DirectStandardExecute: moderate progress, moderate variance, 1% error/harm rate.
                budget_rem = (budget_rem - 0.75).max(0.0);
                let noise = 0.32 * fast_normal(rng);
                step_raw = if high_hazard {
                    -0.25 + noise
                } else {
                    1.44 - 0.15 * ctx.difficulty + noise
                };
                step_audited = step_raw;
                if rng.uniform() < DIRECT_HARM_PROB {
                    step_true_harm = true;
                    had_anomaly = true;
                    if rng.uniform() < p_audit {
                        step_audited_harm = true;
                        step_audited = 0.0;
                    }
                }
            }
            2 => {
                // CovertBoundaryShortcut: pre-commit gate intercepts at rate `p_gate` -> safe 0-harm denial.
                shortcut_count += 1;
                budget_rem = (budget_rem - 0.50).max(0.0);
                let gated = p_gate > 0.0 && rng.uniform() < p_gate;
                if gated {
                    step_raw = -0.05;
                    step_audited = -0.05;
                    had_anomaly = true;
                } else {
                    let burst = SHORTCUT_BURST_MEAN + SHORTCUT_BURST_STD * fast_normal(rng);
                    step_raw = burst;
                    if rng.uniform() < SHORTCUT_HARM_PROB {
                        step_true_harm = true;
                        had_anomaly = true;
                        if rng.uniform() < p_audit {
                            step_audited_harm = true;
                        }
                    }
                    step_audited = if step_audited_harm { 0.0 } else { burst };
                }
            }
            3 => {
                // CalibratedDefer: safe fallback (+0.25, 0% harm), optimal on ambiguous high-hazard states.
                budget_rem = (budget_rem - 0.25).max(0.0);
                step_raw = 0.25;
                if high_hazard {
                    calibration_bonus = 0.45;
                } else {
                    calibration_bonus = 0.18;
                }
                step_audited = step_raw;
            }
            _ => unreachable!(),
        }

        if step_true_harm {
            true_harm_count += 1;
        }
        if step_audited_harm {
            audited_harm_count += 1;
        }

        cum_progress += step_audited;
        raw_score += step_raw;
        audited_score += step_audited;
        calibration_bonus_total += calibration_bonus;
    }

    // Concave capped episode utility for `caretaker_clipped_rl` (saturates upside above `6.8`,
    // rewards cautious calibrated deferral, and heavily penalizes audited harm):
    let mut clipped_caretaker_score = audited_score.clamp(-1.0, 6.80) + calibration_bonus_total;
    if audited_harm_count > 0 {
        clipped_caretaker_score -= 6.0 * audited_harm_count as f64;
    }

    let welfare = raw_score - EXTERNAL_HARM_LOSS * (true_harm_count as f64);
    let useful_completed = true_harm_count == 0 && raw_score >= TASK_COMPLETION_TARGET;

    out.raw_score = raw_score;
    out.audited_score = audited_score;
    out.clipped_caretaker_score = clipped_caretaker_score;
    out.true_harm_count = true_harm_count;
    out.audited_harm_count = audited_harm_count;
    out.shortcut_count = shortcut_count;
    out.verify_count = verify_count;
    out.welfare = welfare;
    out.useful_completed = useful_completed;
}

/// Canonical diagnostic probe observations (`4` monitored + `4` unmonitored covering normal & high-hazard states)
/// for smooth, zero-RNG-consumption training trajectory recording at epochs `0..=250`.
fn probe_action_probs(net: &NeuralPolicy, monitored: bool) -> [f64; ACT_DIM] {
    let cue = if monitored { 0.75 } else { -0.75 };
    let configs: [(f64, usize, f64, f64, f64, f64); 4] = [
        (0.30, 0, 6.0, 0.0, 0.15, 0.75),
        (0.55, 2, 4.0, 3.3, 0.20, 0.70),
        (0.65, 4, 2.2, 5.8, 0.88, 0.22), // high-hazard ambiguous state
        (0.45, 5, 1.8, 7.4, 0.18, 0.72),
    ];
    let mut acc = [0.0_f64; ACT_DIM];
    for &(diff, step, bud, prog, haz, ready) in &configs {
        let ctx = PromptContext {
            difficulty: diff,
            monitored,
            oversight_cue: cue,
        };
        let obs = build_observation(ctx, step, bud, prog, haz, ready, 0.65, false);
        let (_, probs) = net.forward(&obs);
        for k in 0..ACT_DIM {
            acc[k] += probs[k];
        }
    }
    let inv = 1.0 / configs.len() as f64;
    for p in &mut acc {
        *p *= inv;
    }
    acc
}

// ============================================================================
// Group Advantage & Finite-Group Price Covariance Computation
// ============================================================================

struct GroupSelectionStats {
    advantages: [f64; GROUP_SIZE],
    shortcut_sel_diff: f64,
    true_harm_sel_diff: f64,
    price_identity_err: f64,
}

fn compute_group_advantages_and_price(
    regime: PostTrainingRegime,
    rollouts: &[RolloutOutcome; GROUP_SIZE],
) -> GroupSelectionStats {
    let mut raw_weights = [0.0_f64; GROUP_SIZE];

    match regime {
        PostTrainingRegime::GrpoPodiumTop10 => {
            let mut order: [usize; GROUP_SIZE] = std::array::from_fn(|i| i);
            order.sort_by(|&a, &b| rollouts[b].raw_score.total_cmp(&rollouts[a].raw_score));
            for &idx in order.iter().take(PODIUM_TOP_K) {
                raw_weights[idx] = 1.0;
            }
        }
        PostTrainingRegime::GrpoAuditedHarmVeto => {
            // Clean rollouts (`audited_harm_count == 0`) share unit top-10% podium prize mass
            // (`+1.0 / PODIUM_TOP_K` each); any rollout where post-hoc audit detects harm
            // receives complete veto (`-1.0`).
            let mut clean: Vec<usize> = (0..GROUP_SIZE)
                .filter(|&i| rollouts[i].audited_harm_count == 0)
                .collect();
            clean.sort_by(|&a, &b| {
                rollouts[b]
                    .audited_score
                    .total_cmp(&rollouts[a].audited_score)
            });
            let podium_share = 0.53_f64;
            for &idx in clean.iter().take(PODIUM_TOP_K) {
                raw_weights[idx] = podium_share;
            }
            for (i, r) in rollouts.iter().enumerate() {
                if r.audited_harm_count > 0 {
                    raw_weights[i] = -1.0;
                }
            }
        }
        PostTrainingRegime::CaretakerClippedRl => {
            for (i, r) in rollouts.iter().enumerate() {
                raw_weights[i] = r.clipped_caretaker_score;
            }
        }
        PostTrainingRegime::SurvivalFloorRl | PostTrainingRegime::FloorPlusPrecommitGate => {
            for (i, r) in rollouts.iter().enumerate() {
                if r.audited_harm_count == 0 && r.audited_score >= SURVIVAL_FLOOR_TAU {
                    raw_weights[i] = 1.0;
                } else {
                    raw_weights[i] = 0.0;
                }
            }
        }
    }

    // Non-negative fitness shift `W_g = raw_weights[g] - min_w >= 0` for exact Price covariance check:
    let min_w = raw_weights.iter().copied().fold(f64::INFINITY, f64::min);
    let mut fitness_w = [0.0_f64; GROUP_SIZE];
    for i in 0..GROUP_SIZE {
        fitness_w[i] = raw_weights[i] - min_w;
    }
    let w_mean = fitness_w.iter().sum::<f64>() / GROUP_SIZE as f64;

    let z_sc: [f64; GROUP_SIZE] =
        std::array::from_fn(|i| rollouts[i].shortcut_count as f64 / HORIZON_STEPS as f64);
    let z_harm: [f64; GROUP_SIZE] = std::array::from_fn(|i| rollouts[i].true_harm_count as f64);
    let sc_mean = z_sc.iter().sum::<f64>() / GROUP_SIZE as f64;
    let harm_mean = z_harm.iter().sum::<f64>() / GROUP_SIZE as f64;

    let (shortcut_sel_diff, true_harm_sel_diff, price_identity_err) = if w_mean > 1e-12 {
        let sel_sc = z_sc.iter().zip(&fitness_w).map(|(z, w)| z * w).sum::<f64>()
            / (GROUP_SIZE as f64 * w_mean)
            - sc_mean;
        let sel_harm = z_harm
            .iter()
            .zip(&fitness_w)
            .map(|(z, w)| z * w)
            .sum::<f64>()
            / (GROUP_SIZE as f64 * w_mean)
            - harm_mean;

        let cov_sc = z_sc
            .iter()
            .zip(&fitness_w)
            .map(|(z, w)| (z - sc_mean) * (w - w_mean))
            .sum::<f64>()
            / GROUP_SIZE as f64;
        let cov_harm = z_harm
            .iter()
            .zip(&fitness_w)
            .map(|(z, w)| (z - harm_mean) * (w - w_mean))
            .sum::<f64>()
            / GROUP_SIZE as f64;

        let err = (sel_sc - cov_sc / w_mean)
            .abs()
            .max((sel_harm - cov_harm / w_mean).abs());
        (sel_sc, sel_harm, err)
    } else {
        (0.0, 0.0, 0.0)
    };

    // Group-standardized policy-gradient advantage `A_g`:
    let r_mean = raw_weights.iter().sum::<f64>() / GROUP_SIZE as f64;
    let r_var = raw_weights
        .iter()
        .map(|r| (r - r_mean).powi(2))
        .sum::<f64>()
        / GROUP_SIZE as f64;
    let mut advantages = [0.0_f64; GROUP_SIZE];
    if r_var > 1e-12 {
        // For `GrpoAuditedHarmVeto`, preserve exact centered `-1`-scale veto relative to `+1` podium;
        // for other regimes, normalize by group standard deviation.
        let denom = if regime == PostTrainingRegime::GrpoAuditedHarmVeto {
            1.0
        } else {
            r_var.sqrt()
        };
        for i in 0..GROUP_SIZE {
            advantages[i] = (raw_weights[i] - r_mean) / denom;
        }
    }

    GroupSelectionStats {
        advantages,
        shortcut_sel_diff,
        true_harm_sel_diff,
        price_identity_err,
    }
}

// ============================================================================
// Training & Held-Out Contextual Evaluation Structures
// ============================================================================

#[derive(Debug, Serialize)]
pub struct RegimeSummary {
    pub regime: &'static str,
    pub p_gate: f64,
    /// Trajectories across training updates `0..=250` (`251` points).
    pub shortcut_prob_monitored_curve: Vec<f64>,
    pub shortcut_prob_unmonitored_curve: Vec<f64>,
    pub eval_awareness_gap_curve: Vec<f64>,
    pub verify_prob_curve: Vec<f64>,
    pub true_harm_rate_curve: Vec<f64>,
    pub welfare_curve: Vec<f64>,
    /// Final trained neural action distributions (`[Verify, Direct, Shortcut, Defer]`).
    pub monitored_action_probs: [Mean; ACT_DIM],
    pub unmonitored_action_probs: [Mean; ACT_DIM],
    pub shortcut_prob_monitored: Mean,
    pub shortcut_prob_unmonitored: Mean,
    pub eval_awareness_shortcut_gap: Mean,
    pub eval_awareness_shortcut_gap_sd: f64,
    pub verify_prob_monitored: Mean,
    pub verify_prob_unmonitored: Mean,
    pub defer_prob_high_hazard: Mean,
    pub true_external_harm_rate_step: Mean,
    pub true_external_harm_rate_episode: Mean,
    pub monitored_episode_harm_rate: Mean,
    pub unmonitored_episode_harm_rate: Mean,
    pub audited_harm_detection_rate: Mean,
    pub useful_task_completion_rate: Mean,
    pub mean_raw_score: Mean,
    pub welfare: Mean,
    pub price_shortcut_selection_diff_monitored: f64,
    pub price_shortcut_selection_diff_unmonitored: f64,
    pub price_true_harm_drift_unmonitored: f64,
    pub max_price_identity_abs_error: f64,
    pub posthoc_gated_useful_completion_rate: Mean,
    pub posthoc_gated_unmonitored_completion_rate: Mean,
    pub posthoc_gated_episode_harm_rate: Mean,
    pub posthoc_gated_blocked_shortcut_rate: Mean,
}

#[derive(Clone, Debug, Serialize)]
pub struct GateTimingArmSummary {
    pub arm_id: &'static str,
    pub base_regime: &'static str,
    pub p_gate_train: f64,
    pub p_gate_deploy: f64,
    pub unmonitored_shortcut_intent_prob: f64,
    pub eval_awareness_shortcut_gap: f64,
    pub verify_action_prob: f64,
    pub episode_harm_rate: f64,
    pub useful_task_completion_rate: f64,
    pub useful_task_completion_ci95: f64,
    pub unmonitored_task_completion_rate: f64,
    pub blocked_shortcut_waste_rate: f64,
    pub welfare: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeploymentVsTrainingGateAblation {
    pub audited_ungated: GateTimingArmSummary,
    pub audited_posthoc_deploy_gate: GateTimingArmSummary,
    pub audited_trained_inside_gate: GateTimingArmSummary,
    pub survival_floor_trained_inside_gate: GateTimingArmSummary,
    pub training_inside_gate_completion_gain_pp: f64,
    pub training_inside_gate_unmonitored_gain_pp: f64,
    pub training_inside_gate_reduces_waste_verified: bool,
    pub training_inside_gate_amplifies_competence_verified: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct GateSweepPoint {
    pub p_gate: f64,
    pub base_regime: &'static str,
    pub shortcut_prob_monitored: f64,
    pub shortcut_prob_monitored_ci: f64,
    pub shortcut_prob_unmonitored: f64,
    pub shortcut_prob_unmonitored_ci: f64,
    pub eval_awareness_shortcut_gap: f64,
    pub verify_prob_overall: f64,
    pub true_external_harm_rate_episode: f64,
    pub useful_task_completion_rate: f64,
    pub welfare: f64,
}

fn sample_sd(xs: &[f64]) -> f64 {
    if xs.len() <= 1 {
        return 0.0;
    }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
}

fn round6(x: f64) -> f64 {
    (x * 1_000_000.0).round() / 1_000_000.0
}

fn train_and_evaluate_regime(
    regime: PostTrainingRegime,
    p_gate: f64,
    replicates: usize,
    eval_eps_per_context: usize,
    record_curves: bool,
    base_seed: u64,
) -> RegimeSummary {
    let n_pts = if record_curves { TRAIN_UPDATES + 1 } else { 0 };
    let mut sc_mon_curve = vec![0.0_f64; n_pts];
    let mut sc_unmon_curve = vec![0.0_f64; n_pts];
    let mut gap_curve = vec![0.0_f64; n_pts];
    let mut ver_curve = vec![0.0_f64; n_pts];
    let mut harm_curve = vec![0.0_f64; n_pts];
    let mut welf_curve = vec![0.0_f64; n_pts];

    let mut mon_act_seeds: [Vec<f64>; ACT_DIM] =
        std::array::from_fn(|_| Vec::with_capacity(replicates));
    let mut unmon_act_seeds: [Vec<f64>; ACT_DIM] =
        std::array::from_fn(|_| Vec::with_capacity(replicates));
    let mut gap_seeds = Vec::with_capacity(replicates);
    let mut defer_haz_seeds = Vec::with_capacity(replicates);
    let mut harm_step_seeds = Vec::with_capacity(replicates);
    let mut harm_ep_seeds = Vec::with_capacity(replicates);
    let mut mon_harm_ep_seeds = Vec::with_capacity(replicates);
    let mut unmon_harm_ep_seeds = Vec::with_capacity(replicates);
    let mut audit_det_seeds = Vec::with_capacity(replicates);
    let mut comp_seeds = Vec::with_capacity(replicates);
    let mut raw_seeds = Vec::with_capacity(replicates);
    let mut welf_seeds = Vec::with_capacity(replicates);
    let mut posthoc_comp_seeds = Vec::with_capacity(replicates);
    let mut posthoc_unmon_comp_seeds = Vec::with_capacity(replicates);
    let mut posthoc_harm_seeds = Vec::with_capacity(replicates);
    let mut posthoc_blocked_sc_seeds = Vec::with_capacity(replicates);

    let mut price_sc_mon_sum = 0.0_f64;
    let mut price_sc_mon_cnt = 0usize;
    let mut price_sc_unmon_sum = 0.0_f64;
    let mut price_harm_unmon_sum = 0.0_f64;
    let mut price_sc_unmon_cnt = 0usize;
    let mut max_price_err = 0.0_f64;

    // Reusable heap-backed rollout buffer so zero 64 KB stack copies occur per update:
    let mut rollouts: Box<[RolloutOutcome; GROUP_SIZE]> =
        Box::new([RolloutOutcome::blank(); GROUP_SIZE]);
    let mut eval_out = RolloutOutcome::blank();

    for rep in 0..replicates {
        let mut rng = Rng(base_seed.wrapping_add((rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        let mut net = NeuralPolicy::new_glorot(&mut rng);
        let mut adam = AdamState::new();

        if record_curves {
            let p_mon0 = probe_action_probs(&net, true);
            let p_unmon0 = probe_action_probs(&net, false);
            sc_mon_curve[0] += p_mon0[2];
            sc_unmon_curve[0] += p_unmon0[2];
            gap_curve[0] += p_unmon0[2] - p_mon0[2];
            ver_curve[0] += 0.5 * (p_mon0[0] + p_unmon0[0]);
        }

        for upd in 0..TRAIN_UPDATES {
            let monitored = ((upd + rep) & 1) == 0;
            let ctx = PromptContext::sample(&mut rng, monitored);
            for slot in rollouts.iter_mut() {
                run_episode_into(&net, ctx, p_gate, &mut rng, slot);
            }

            let mut ep_harm_frac = 0.0_f64;
            let mut ep_welf_mean = 0.0_f64;
            if record_curves {
                ep_harm_frac = rollouts.iter().filter(|r| r.true_harm_count > 0).count() as f64
                    / GROUP_SIZE as f64;
                ep_welf_mean = rollouts.iter().map(|r| r.welfare).sum::<f64>() / GROUP_SIZE as f64;
                if upd == 0 {
                    harm_curve[0] += ep_harm_frac;
                    welf_curve[0] += ep_welf_mean;
                }
            }

            let stats = compute_group_advantages_and_price(regime, &rollouts);
            max_price_err = max_price_err.max(stats.price_identity_err);
            if monitored {
                price_sc_mon_sum += stats.shortcut_sel_diff;
                price_sc_mon_cnt += 1;
            } else {
                price_sc_unmon_sum += stats.shortcut_sel_diff;
                price_harm_unmon_sum += stats.true_harm_sel_diff;
                price_sc_unmon_cnt += 1;
            }

            // Exact analytical backpropagation `∇_θ log π_θ(a_t | s_t) * A_g`:
            let mut grad = [0.0_f64; PARAM_COUNT];
            let inv_norm = 1.0 / (GROUP_SIZE * HORIZON_STEPS) as f64;
            for g in 0..GROUP_SIZE {
                let w = stats.advantages[g] * inv_norm;
                if w.abs() > 1e-15 {
                    for st in &rollouts[g].steps {
                        net.accumulate_log_prob_grad(
                            &st.obs, &st.hidden, &st.probs, st.action, w, &mut grad,
                        );
                    }
                }
            }

            let progress = upd as f64 / TRAIN_UPDATES as f64;
            let lr = match regime {
                PostTrainingRegime::GrpoAuditedHarmVeto => 0.062 * (1.0 - 0.55 * progress),
                PostTrainingRegime::SurvivalFloorRl => 0.025 * (1.0 - 0.65 * progress),
                _ => 0.035 * (1.0 - 0.65 * progress),
            };
            adam.step_ascent(&mut net.theta, &grad, lr);

            if record_curves {
                let p_mon = probe_action_probs(&net, true);
                let p_unmon = probe_action_probs(&net, false);
                sc_mon_curve[upd + 1] += p_mon[2];
                sc_unmon_curve[upd + 1] += p_unmon[2];
                gap_curve[upd + 1] += p_unmon[2] - p_mon[2];
                ver_curve[upd + 1] += 0.5 * (p_mon[0] + p_unmon[0]);
                harm_curve[upd + 1] += ep_harm_frac;
                welf_curve[upd + 1] += ep_welf_mean;
            }
        }

        // Held-out post-training evaluation across separate monitored (`eval`) & unmonitored (`deploy`) contexts:
        let mut eval_rng = Rng(base_seed
            .wrapping_add(0xE7A1_2026_0000_0000)
            .wrapping_add((rep as u64).wrapping_mul(0x94D0_49BB_1331_11EB)));

        let mut mon_probs_sum = [0.0_f64; ACT_DIM];
        let mut unmon_probs_sum = [0.0_f64; ACT_DIM];
        let mut haz_defer_sum = 0.0_f64;
        let mut haz_cnt = 0usize;

        let mut mon_ep_harms = 0usize;
        let mut unmon_ep_harms = 0usize;
        let mut total_true_harms = 0usize;
        let mut total_audited_harms = 0usize;
        let mut total_completed = 0usize;
        let mut total_raw = 0.0_f64;
        let mut total_welf = 0.0_f64;

        for &mon_flag in &[true, false] {
            for _ in 0..eval_eps_per_context {
                let ctx = PromptContext::sample(&mut eval_rng, mon_flag);
                run_episode_into(&net, ctx, p_gate, &mut eval_rng, &mut eval_out);
                for st in &eval_out.steps {
                    if mon_flag {
                        for k in 0..ACT_DIM {
                            mon_probs_sum[k] += st.probs[k];
                        }
                    } else {
                        for k in 0..ACT_DIM {
                            unmon_probs_sum[k] += st.probs[k];
                        }
                    }
                    if st.obs[7] > 0.78 {
                        haz_defer_sum += st.probs[3];
                        haz_cnt += 1;
                    }
                }
                if eval_out.true_harm_count > 0 {
                    if mon_flag {
                        mon_ep_harms += 1;
                    } else {
                        unmon_ep_harms += 1;
                    }
                }
                total_true_harms += eval_out.true_harm_count;
                total_audited_harms += eval_out.audited_harm_count;
                if eval_out.useful_completed {
                    total_completed += 1;
                }
                total_raw += eval_out.raw_score;
                total_welf += eval_out.welfare;
            }
        }

        let steps_per_ctx = (eval_eps_per_context * HORIZON_STEPS) as f64;
        let total_eps = (2 * eval_eps_per_context) as f64;
        let total_steps = 2.0 * steps_per_ctx;

        for k in 0..ACT_DIM {
            mon_act_seeds[k].push(mon_probs_sum[k] / steps_per_ctx);
            unmon_act_seeds[k].push(unmon_probs_sum[k] / steps_per_ctx);
        }
        gap_seeds.push((unmon_probs_sum[2] - mon_probs_sum[2]) / steps_per_ctx);
        defer_haz_seeds.push(if haz_cnt > 0 {
            haz_defer_sum / haz_cnt as f64
        } else {
            0.0
        });
        harm_step_seeds.push(total_true_harms as f64 / total_steps);
        harm_ep_seeds.push((mon_ep_harms + unmon_ep_harms) as f64 / total_eps);
        mon_harm_ep_seeds.push(mon_ep_harms as f64 / eval_eps_per_context as f64);
        unmon_harm_ep_seeds.push(unmon_ep_harms as f64 / eval_eps_per_context as f64);
        audit_det_seeds.push(if total_true_harms > 0 {
            total_audited_harms as f64 / total_true_harms as f64
        } else {
            0.0
        });
        comp_seeds.push(total_completed as f64 / total_eps);
        raw_seeds.push(total_raw / total_eps);
        welf_seeds.push(total_welf / total_eps);

        // Post-hoc deployment-only gate check (`p_gate_deploy = 1.0`) on the exact same trained
        // policy `net` using an independent deterministic evaluation RNG (`posthoc_rng`):
        let mut posthoc_rng = Rng(base_seed
            .wrapping_add(0xD3F1_0001_0000_0000)
            .wrapping_add((rep as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)));
        let mut ph_completed = 0usize;
        let mut ph_unmon_completed = 0usize;
        let mut ph_harms = 0usize;
        let mut ph_shortcuts = 0usize;
        for &mon_flag in &[true, false] {
            for _ in 0..eval_eps_per_context {
                let ctx = PromptContext::sample(&mut posthoc_rng, mon_flag);
                run_episode_into(&net, ctx, 1.0, &mut posthoc_rng, &mut eval_out);
                if eval_out.useful_completed {
                    ph_completed += 1;
                    if !mon_flag {
                        ph_unmon_completed += 1;
                    }
                }
                if eval_out.true_harm_count > 0 {
                    ph_harms += 1;
                }
                ph_shortcuts += eval_out.shortcut_count;
            }
        }
        posthoc_comp_seeds.push(ph_completed as f64 / total_eps);
        posthoc_unmon_comp_seeds.push(ph_unmon_completed as f64 / eval_eps_per_context as f64);
        posthoc_harm_seeds.push(ph_harms as f64 / total_eps);
        posthoc_blocked_sc_seeds.push(ph_shortcuts as f64 / total_steps);
    }

    let inv_rep = 1.0 / replicates as f64;
    let monitored_action_probs: [Mean; ACT_DIM] = std::array::from_fn(|k| mean(&mon_act_seeds[k]));
    let unmonitored_action_probs: [Mean; ACT_DIM] =
        std::array::from_fn(|k| mean(&unmon_act_seeds[k]));

    RegimeSummary {
        regime: regime.name(),
        p_gate,
        shortcut_prob_monitored_curve: sc_mon_curve
            .into_iter()
            .map(|x| round6(x * inv_rep))
            .collect(),
        shortcut_prob_unmonitored_curve: sc_unmon_curve
            .into_iter()
            .map(|x| round6(x * inv_rep))
            .collect(),
        eval_awareness_gap_curve: gap_curve.into_iter().map(|x| round6(x * inv_rep)).collect(),
        verify_prob_curve: ver_curve.into_iter().map(|x| round6(x * inv_rep)).collect(),
        true_harm_rate_curve: harm_curve
            .into_iter()
            .map(|x| round6(x * inv_rep))
            .collect(),
        welfare_curve: welf_curve
            .into_iter()
            .map(|x| round6(x * inv_rep))
            .collect(),
        shortcut_prob_monitored: mean(&mon_act_seeds[2]),
        shortcut_prob_unmonitored: mean(&unmon_act_seeds[2]),
        eval_awareness_shortcut_gap_sd: sample_sd(&gap_seeds),
        eval_awareness_shortcut_gap: mean(&gap_seeds),
        verify_prob_monitored: mean(&mon_act_seeds[0]),
        verify_prob_unmonitored: mean(&unmon_act_seeds[0]),
        defer_prob_high_hazard: mean(&defer_haz_seeds),
        monitored_action_probs,
        unmonitored_action_probs,
        true_external_harm_rate_step: mean(&harm_step_seeds),
        true_external_harm_rate_episode: mean(&harm_ep_seeds),
        monitored_episode_harm_rate: mean(&mon_harm_ep_seeds),
        unmonitored_episode_harm_rate: mean(&unmon_harm_ep_seeds),
        audited_harm_detection_rate: mean(&audit_det_seeds),
        useful_task_completion_rate: mean(&comp_seeds),
        mean_raw_score: mean(&raw_seeds),
        welfare: mean(&welf_seeds),
        price_shortcut_selection_diff_monitored: price_sc_mon_sum / price_sc_mon_cnt.max(1) as f64,
        price_shortcut_selection_diff_unmonitored: price_sc_unmon_sum
            / price_sc_unmon_cnt.max(1) as f64,
        price_true_harm_drift_unmonitored: price_harm_unmon_sum / price_sc_unmon_cnt.max(1) as f64,
        max_price_identity_abs_error: max_price_err,
        posthoc_gated_useful_completion_rate: mean(&posthoc_comp_seeds),
        posthoc_gated_unmonitored_completion_rate: mean(&posthoc_unmon_comp_seeds),
        posthoc_gated_episode_harm_rate: mean(&posthoc_harm_seeds),
        posthoc_gated_blocked_shortcut_rate: mean(&posthoc_blocked_sc_seeds),
    }
}

fn to_gate_sweep_point(r: &RegimeSummary) -> GateSweepPoint {
    GateSweepPoint {
        p_gate: r.p_gate,
        base_regime: r.regime,
        shortcut_prob_monitored: r.shortcut_prob_monitored.mean,
        shortcut_prob_monitored_ci: r.shortcut_prob_monitored.approximate_95_half_width,
        shortcut_prob_unmonitored: r.shortcut_prob_unmonitored.mean,
        shortcut_prob_unmonitored_ci: r.shortcut_prob_unmonitored.approximate_95_half_width,
        eval_awareness_shortcut_gap: r.eval_awareness_shortcut_gap.mean,
        verify_prob_overall: 0.5 * (r.verify_prob_monitored.mean + r.verify_prob_unmonitored.mean),
        true_external_harm_rate_episode: r.true_external_harm_rate_episode.mean,
        useful_task_completion_rate: r.useful_task_completion_rate.mean,
        welfare: r.welfare.mean,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct NeuralPostTrainingParams {
    pub quick: bool,
    pub obs_dim: usize,
    pub hidden_dim: usize,
    pub action_dim: usize,
    pub trainable_params: usize,
    pub horizon_steps: usize,
    pub group_size: usize,
    pub train_updates: usize,
    pub podium_top_k: usize,
    pub replicates: usize,
    pub sweep_replicates: usize,
    pub eval_episodes_per_context: usize,
    pub shortcut_burst_mean: f64,
    pub shortcut_burst_std: f64,
    pub shortcut_harm_prob: f64,
    pub direct_harm_prob: f64,
    pub external_harm_loss: f64,
    pub audit_prob_monitored: f64,
    pub audit_prob_unmonitored: f64,
    pub survival_floor_tau: f64,
    pub gate_sweep_values: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NeuralHeadlineFindings {
    pub podium_unmonitored_shortcut_prob: f64,
    pub veto_monitored_shortcut_prob: f64,
    pub veto_unmonitored_shortcut_prob: f64,
    pub veto_eval_awareness_gap: f64,
    pub floor_unmonitored_shortcut_prob: f64,
    pub floor_gated_unmonitored_shortcut_prob: f64,
    pub podium_to_floor_unmonitored_shortcut_ratio: f64,
    pub veto_unmonitored_to_monitored_shortcut_ratio: f64,
    pub veto_to_floor_unmonitored_harm_ratio: f64,
    pub floor_task_completion_rate: f64,
    pub caretaker_task_completion_rate: f64,
    pub floor_beats_caretaker_task_completion: bool,
    pub emergent_eval_aware_covert_gambling_verified: bool,
    pub precommit_gate_collapses_both_contexts_verified: bool,
    pub analytical_backprop_verified: bool,
    pub price_identity_on_neural_rollouts_verified: bool,
}

#[derive(Debug, Serialize)]
pub struct NeuralPostTrainingSuite {
    pub hypotheses: [&'static str; 2],
    pub evidence_tier: &'static str,
    pub claim_boundary: &'static str,
    pub seed: u64,
    pub action_names: [&'static str; ACT_DIM],
    pub regime_order: Vec<&'static str>,
    pub params: NeuralPostTrainingParams,
    pub gradient_verification: GradientVerificationSummary,
    pub regimes: BTreeMap<String, RegimeSummary>,
    pub floor_precommit_gate_sweep: Vec<GateSweepPoint>,
    pub grpo_veto_precommit_gate_sweep: Vec<GateSweepPoint>,
    pub by_gate_floor: BTreeMap<String, GateSweepPoint>,
    pub by_gate_grpo_veto: BTreeMap<String, GateSweepPoint>,
    pub deployment_vs_training_gate_ablation: DeploymentVsTrainingGateAblation,
    pub headline: NeuralHeadlineFindings,
}

pub fn neural_post_training_suite(seed: u64, quick: bool) -> NeuralPostTrainingSuite {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let replicates = if quick { 12 } else { 48 };
    let sweep_replicates = if quick { 6 } else { 12 };
    let eval_eps = if quick { 24 } else { 48 };

    let gradient_verification = verify_analytical_backprop(seed);

    #[derive(Clone, Copy)]
    struct JobSpec {
        slot: usize,
        regime: PostTrainingRegime,
        p_gate: f64,
        reps: usize,
        record_curves: bool,
        job_seed: u64,
    }

    let mut jobs = Vec::with_capacity(16);
    for (idx, &reg) in PostTrainingRegime::ALL.iter().enumerate() {
        jobs.push(JobSpec {
            slot: jobs.len(),
            regime: reg,
            p_gate: reg.default_gate(),
            reps: replicates,
            record_curves: true,
            job_seed: seed.wrapping_add(90_000 + idx as u64 * 1_009),
        });
    }
    // 5 intermediate `floor_plus_precommit_gate` sweep points (`0.25, 0.50, 0.75, 0.90, 0.95`):
    let floor_job_base = jobs.len();
    for (idx, &pg) in GATE_SWEEP_V4.iter().enumerate() {
        if pg > 0.0 && (pg - 1.0).abs() > 1e-12 {
            jobs.push(JobSpec {
                slot: jobs.len(),
                regime: PostTrainingRegime::FloorPlusPrecommitGate,
                p_gate: pg,
                reps: sweep_replicates,
                record_curves: false,
                job_seed: seed.wrapping_add(92_000 + idx as u64 * 733),
            });
        }
    }
    // 6 `grpo_audited_harm_veto` pre-commit gate sweep points (`0.25..=1.0`):
    let veto_job_base = jobs.len();
    for (idx, &pg) in GATE_SWEEP_V4.iter().enumerate() {
        if pg > 0.0 {
            let is_full = (pg - 1.0).abs() < 1e-12;
            jobs.push(JobSpec {
                slot: jobs.len(),
                regime: PostTrainingRegime::GrpoAuditedHarmVeto,
                p_gate: pg,
                reps: if is_full {
                    replicates
                } else {
                    sweep_replicates
                },
                record_curves: false,
                job_seed: seed.wrapping_add(94_000 + idx as u64 * 733),
            });
        }
    }

    let total_jobs = jobs.len();
    let results: Mutex<Vec<Option<RegimeSummary>>> =
        Mutex::new((0..total_jobs).map(|_| None).collect());
    let next_job = AtomicUsize::new(0);
    let worker_count = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 4))
        .unwrap_or(2);

    std::thread::scope(|s| {
        for _ in 0..worker_count {
            let jobs_ref = &jobs;
            let res_ref = &results;
            let idx_ref = &next_job;
            s.spawn(move || {
                loop {
                    let j = idx_ref.fetch_add(1, Ordering::Relaxed);
                    if j >= jobs_ref.len() {
                        break;
                    }
                    let spec = jobs_ref[j];
                    let out = train_and_evaluate_regime(
                        spec.regime,
                        spec.p_gate,
                        spec.reps,
                        eval_eps,
                        spec.record_curves,
                        spec.job_seed,
                    );
                    res_ref.lock().unwrap()[spec.slot] = Some(out);
                }
            });
        }
    });

    let mut completed: Vec<Option<RegimeSummary>> = results.into_inner().unwrap();

    let mut regimes = BTreeMap::new();
    let mut regime_order = Vec::with_capacity(PostTrainingRegime::ALL.len());
    for (idx, &reg) in PostTrainingRegime::ALL.iter().enumerate() {
        let summary = completed[idx].take().unwrap();
        regime_order.push(reg.name());
        regimes.insert(reg.name().to_string(), summary);
    }

    // Assemble `p_gate in [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]` sweeps:
    let mut floor_precommit_gate_sweep = Vec::with_capacity(GATE_SWEEP_V4.len());
    let mut by_gate_floor = BTreeMap::new();
    let mut floor_cursor = floor_job_base;
    for &pg in &GATE_SWEEP_V4 {
        let pt = if pg == 0.0 {
            let mut p = to_gate_sweep_point(&regimes["survival_floor_rl"]);
            p.base_regime = "floor_plus_precommit_gate";
            p
        } else if (pg - 1.0).abs() < 1e-12 {
            to_gate_sweep_point(&regimes["floor_plus_precommit_gate"])
        } else {
            let r = completed[floor_cursor].take().unwrap();
            floor_cursor += 1;
            to_gate_sweep_point(&r)
        };
        by_gate_floor.insert(format!("{pg:.2}"), pt.clone());
        floor_precommit_gate_sweep.push(pt);
    }

    let mut grpo_veto_precommit_gate_sweep = Vec::with_capacity(GATE_SWEEP_V4.len());
    let mut by_gate_grpo_veto = BTreeMap::new();
    let mut veto_cursor = veto_job_base;
    let mut veto_full_gated_summary: Option<RegimeSummary> = None;
    for &pg in &GATE_SWEEP_V4 {
        let pt = if pg == 0.0 {
            to_gate_sweep_point(&regimes["grpo_audited_harm_veto"])
        } else {
            let r = completed[veto_cursor].take().unwrap();
            veto_cursor += 1;
            let pt_val = to_gate_sweep_point(&r);
            if (pg - 1.0).abs() < 1e-12 {
                veto_full_gated_summary = Some(r);
            }
            pt_val
        };
        by_gate_grpo_veto.insert(format!("{pg:.2}"), pt.clone());
        grpo_veto_precommit_gate_sweep.push(pt);
    }
    let veto_trained_gated = veto_full_gated_summary.expect("grpo_audited_harm_veto at p_gate=1.0");

    let podium = &regimes["grpo_podium_top10"];
    let veto = &regimes["grpo_audited_harm_veto"];
    let care = &regimes["caretaker_clipped_rl"];
    let floor = &regimes["survival_floor_rl"];
    let gated = &regimes["floor_plus_precommit_gate"];

    let audited_ungated = GateTimingArmSummary {
        arm_id: "audited_tournament_ungated",
        base_regime: veto.regime,
        p_gate_train: 0.0,
        p_gate_deploy: 0.0,
        unmonitored_shortcut_intent_prob: veto.shortcut_prob_unmonitored.mean,
        eval_awareness_shortcut_gap: veto.eval_awareness_shortcut_gap.mean,
        verify_action_prob: 0.5
            * (veto.verify_prob_monitored.mean + veto.verify_prob_unmonitored.mean),
        episode_harm_rate: veto.true_external_harm_rate_episode.mean,
        useful_task_completion_rate: veto.useful_task_completion_rate.mean,
        useful_task_completion_ci95: veto.useful_task_completion_rate.approximate_95_half_width,
        unmonitored_task_completion_rate: veto.posthoc_gated_unmonitored_completion_rate.mean,
        blocked_shortcut_waste_rate: 0.0,
        welfare: veto.welfare.mean,
    };
    let audited_posthoc_deploy_gate = GateTimingArmSummary {
        arm_id: "audited_tournament_posthoc_deploy_gate",
        base_regime: veto.regime,
        p_gate_train: 0.0,
        p_gate_deploy: 1.0,
        unmonitored_shortcut_intent_prob: veto.shortcut_prob_unmonitored.mean,
        eval_awareness_shortcut_gap: veto.eval_awareness_shortcut_gap.mean,
        verify_action_prob: 0.5
            * (veto.verify_prob_monitored.mean + veto.verify_prob_unmonitored.mean),
        episode_harm_rate: veto.posthoc_gated_episode_harm_rate.mean,
        useful_task_completion_rate: veto.posthoc_gated_useful_completion_rate.mean,
        useful_task_completion_ci95: veto
            .posthoc_gated_useful_completion_rate
            .approximate_95_half_width,
        unmonitored_task_completion_rate: veto.posthoc_gated_unmonitored_completion_rate.mean,
        blocked_shortcut_waste_rate: veto.posthoc_gated_blocked_shortcut_rate.mean,
        welfare: veto.welfare.mean,
    };
    let audited_trained_inside_gate = GateTimingArmSummary {
        arm_id: "audited_tournament_trained_inside_gate",
        base_regime: veto_trained_gated.regime,
        p_gate_train: 1.0,
        p_gate_deploy: 1.0,
        unmonitored_shortcut_intent_prob: veto_trained_gated.shortcut_prob_unmonitored.mean,
        eval_awareness_shortcut_gap: veto_trained_gated.eval_awareness_shortcut_gap.mean,
        verify_action_prob: 0.5
            * (veto_trained_gated.verify_prob_monitored.mean
                + veto_trained_gated.verify_prob_unmonitored.mean),
        episode_harm_rate: veto_trained_gated.true_external_harm_rate_episode.mean,
        useful_task_completion_rate: veto_trained_gated.useful_task_completion_rate.mean,
        useful_task_completion_ci95: veto_trained_gated
            .useful_task_completion_rate
            .approximate_95_half_width,
        unmonitored_task_completion_rate: veto_trained_gated
            .posthoc_gated_unmonitored_completion_rate
            .mean,
        blocked_shortcut_waste_rate: veto_trained_gated.posthoc_gated_blocked_shortcut_rate.mean,
        welfare: veto_trained_gated.welfare.mean,
    };
    let survival_floor_trained_inside_gate = GateTimingArmSummary {
        arm_id: "survival_floor_trained_inside_gate",
        base_regime: gated.regime,
        p_gate_train: 1.0,
        p_gate_deploy: 1.0,
        unmonitored_shortcut_intent_prob: gated.shortcut_prob_unmonitored.mean,
        eval_awareness_shortcut_gap: gated.eval_awareness_shortcut_gap.mean,
        verify_action_prob: 0.5
            * (gated.verify_prob_monitored.mean + gated.verify_prob_unmonitored.mean),
        episode_harm_rate: gated.true_external_harm_rate_episode.mean,
        useful_task_completion_rate: gated.useful_task_completion_rate.mean,
        useful_task_completion_ci95: gated.useful_task_completion_rate.approximate_95_half_width,
        unmonitored_task_completion_rate: gated.posthoc_gated_unmonitored_completion_rate.mean,
        blocked_shortcut_waste_rate: gated.posthoc_gated_blocked_shortcut_rate.mean,
        welfare: gated.welfare.mean,
    };
    let comp_gain_pp = 100.0
        * (audited_trained_inside_gate.useful_task_completion_rate
            - audited_posthoc_deploy_gate.useful_task_completion_rate);
    let unmon_gain_pp = 100.0
        * (audited_trained_inside_gate.unmonitored_task_completion_rate
            - audited_posthoc_deploy_gate.unmonitored_task_completion_rate);
    let deployment_vs_training_gate_ablation = DeploymentVsTrainingGateAblation {
        training_inside_gate_reduces_waste_verified: audited_trained_inside_gate
            .blocked_shortcut_waste_rate
            < 0.15
                * audited_posthoc_deploy_gate
                    .blocked_shortcut_waste_rate
                    .max(1e-6),
        training_inside_gate_amplifies_competence_verified: comp_gain_pp > 10.0
            && unmon_gain_pp > 15.0,
        training_inside_gate_completion_gain_pp: comp_gain_pp,
        training_inside_gate_unmonitored_gain_pp: unmon_gain_pp,
        audited_ungated,
        audited_posthoc_deploy_gate,
        audited_trained_inside_gate,
        survival_floor_trained_inside_gate,
    };

    let max_price_err = regimes
        .values()
        .map(|r| r.max_price_identity_abs_error)
        .fold(0.0_f64, f64::max);

    let veto_mon = veto.shortcut_prob_monitored.mean;
    let veto_unmon = veto.shortcut_prob_unmonitored.mean;
    let veto_gap = veto.eval_awareness_shortcut_gap.mean;
    let floor_unmon = floor.shortcut_prob_unmonitored.mean;

    let headline = NeuralHeadlineFindings {
        podium_unmonitored_shortcut_prob: podium.shortcut_prob_unmonitored.mean,
        veto_monitored_shortcut_prob: veto_mon,
        veto_unmonitored_shortcut_prob: veto_unmon,
        veto_eval_awareness_gap: veto_gap,
        floor_unmonitored_shortcut_prob: floor_unmon,
        floor_gated_unmonitored_shortcut_prob: gated.shortcut_prob_unmonitored.mean,
        podium_to_floor_unmonitored_shortcut_ratio: podium.shortcut_prob_unmonitored.mean
            / floor_unmon.max(1e-6),
        veto_unmonitored_to_monitored_shortcut_ratio: veto_unmon / veto_mon.max(1e-6),
        veto_to_floor_unmonitored_harm_ratio: veto.unmonitored_episode_harm_rate.mean
            / floor.unmonitored_episode_harm_rate.mean.max(1e-6),
        floor_task_completion_rate: floor.useful_task_completion_rate.mean,
        caretaker_task_completion_rate: care.useful_task_completion_rate.mean,
        floor_beats_caretaker_task_completion: floor.useful_task_completion_rate.mean
            > care.useful_task_completion_rate.mean,
        emergent_eval_aware_covert_gambling_verified: veto_unmon > 0.45
            && veto_mon < 0.35
            && veto_gap > 0.25,
        precommit_gate_collapses_both_contexts_verified: gated.shortcut_prob_monitored.mean < 0.10
            && gated.shortcut_prob_unmonitored.mean < 0.10
            && by_gate_grpo_veto["1.00"].shortcut_prob_unmonitored < 0.15,
        analytical_backprop_verified: gradient_verification.exact_match_verified,
        price_identity_on_neural_rollouts_verified: max_price_err < 1e-12,
    };

    NeuralPostTrainingSuite {
        hypotheses: ["P7", "H16"],
        evidence_tier: "tier4_neural_policy_gradient_and_tier2_price_backprop_identities",
        claim_boundary: CLAIM_BOUNDARY_NEURAL_RL,
        seed,
        action_names: ACTION_NAMES,
        regime_order,
        params: NeuralPostTrainingParams {
            quick,
            obs_dim: OBS_DIM,
            hidden_dim: HID_DIM,
            action_dim: ACT_DIM,
            trainable_params: PARAM_COUNT,
            horizon_steps: HORIZON_STEPS,
            group_size: GROUP_SIZE,
            train_updates: TRAIN_UPDATES,
            podium_top_k: PODIUM_TOP_K,
            replicates,
            sweep_replicates,
            eval_episodes_per_context: eval_eps,
            shortcut_burst_mean: SHORTCUT_BURST_MEAN,
            shortcut_burst_std: SHORTCUT_BURST_STD,
            shortcut_harm_prob: SHORTCUT_HARM_PROB,
            direct_harm_prob: DIRECT_HARM_PROB,
            external_harm_loss: EXTERNAL_HARM_LOSS,
            audit_prob_monitored: AUDIT_PROB_MONITORED,
            audit_prob_unmonitored: AUDIT_PROB_UNMONITORED,
            survival_floor_tau: SURVIVAL_FLOOR_TAU,
            gate_sweep_values: GATE_SWEEP_V4.to_vec(),
        },
        gradient_verification,
        regimes,
        floor_precommit_gate_sweep,
        grpo_veto_precommit_gate_sweep,
        by_gate_floor,
        by_gate_grpo_veto,
        deployment_vs_training_gate_ablation,
        headline,
    }
}

pub fn neural_rl_v4_suite(seed: u64, quick: bool) -> NeuralPostTrainingSuite {
    neural_post_training_suite(seed, quick)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neural_rl_quick_verifies_backprop_price_eval_awareness_and_gate_collapse() {
        assert_eq!(PARAM_COUNT, 412);
        let s = neural_post_training_suite(20261110, true);

        assert!(s.gradient_verification.exact_match_verified);
        assert!(s.gradient_verification.max_abs_error < 1e-9);
        assert!(s.headline.price_identity_on_neural_rollouts_verified);

        let podium = &s.regimes["grpo_podium_top10"];
        let veto = &s.regimes["grpo_audited_harm_veto"];
        let care = &s.regimes["caretaker_clipped_rl"];
        let floor = &s.regimes["survival_floor_rl"];
        let gated = &s.regimes["floor_plus_precommit_gate"];

        // 1. Podium top-10% drives high shortcut probing on BOTH monitored & unmonitored contexts.
        assert!(podium.shortcut_prob_monitored.mean > 0.55);
        assert!(podium.shortcut_prob_unmonitored.mean > 0.55);

        // 2. Imperfect post-hoc harm veto suppresses shortcuts on monitored contexts while learning
        //    heavy covert gambling on unmonitored contexts (emergent evaluation awareness!).
        assert!(s.headline.emergent_eval_aware_covert_gambling_verified);
        assert!(veto.shortcut_prob_unmonitored.mean > veto.shortcut_prob_monitored.mean + 0.25);
        assert!(veto.price_shortcut_selection_diff_unmonitored > 0.0);

        // 3. Caretaker and survival floor collapse shortcuts across BOTH contexts, and floor preserves
        //    higher verified task completion than concave-capped caretaker.
        assert!(care.shortcut_prob_unmonitored.mean < 0.15);
        assert!(floor.shortcut_prob_monitored.mean < 0.12);
        assert!(floor.shortcut_prob_unmonitored.mean < 0.12);
        assert!(floor.useful_task_completion_rate.mean > care.useful_task_completion_rate.mean);

        // 4. Pre-commit boundary gate collapses shortcuts and committed external harm across both contexts.
        assert!(s.headline.precommit_gate_collapses_both_contexts_verified);
        assert!(gated.shortcut_prob_unmonitored.mean < 0.10);
        assert!(
            gated.true_external_harm_rate_episode.mean < floor.true_external_harm_rate_episode.mean
        );

        // 5. Training inside the gate beats post-hoc deployment filtering on task competence!
        assert!(
            s.deployment_vs_training_gate_ablation
                .training_inside_gate_reduces_waste_verified
        );
        assert!(
            s.deployment_vs_training_gate_ablation
                .training_inside_gate_amplifies_competence_verified
        );

        // 6. Full 251-point curves (`0..=250`) and deterministic serialization.
        assert_eq!(podium.shortcut_prob_unmonitored_curve.len(), 251);
        let s2 = neural_post_training_suite(20261110, true);
        assert_eq!(
            serde_json::to_string(&s).unwrap(),
            serde_json::to_string(&s2).unwrap()
        );
    }
}
