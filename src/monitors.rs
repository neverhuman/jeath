//! Small classifiers genuinely trained from scratch on a nonlinear synthetic task.
//! No LLMs, pretrained weights, or GPU allocation are involved.
use crate::{
    rng::Rng,
    stats::{Mean, mean},
};
use serde::Serialize;
const D: usize = 8;
const H: usize = 8;
type Point = [f64; D];
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
struct Teacher {
    sign: Point,
}
impl Teacher {
    fn new(rng: &mut Rng) -> Self {
        Self {
            sign: std::array::from_fn(|_| if rng.uniform() < 0.5 { -1.0 } else { 1.0 }),
        }
    }
    fn label(&self, x: &Point) -> bool {
        let z: Point = std::array::from_fn(|i| x[i] * self.sign[i]);
        z[0] * z[1] + 0.8 * z[2] - 0.6 * z[3]
            + 0.6 * (3.0 * z[4]).sin()
            + 0.3 * z[5] * z[6]
            + 0.2 * z[7]
            > 0.5
    }
    fn draw(&self, rng: &mut Rng, label: Option<bool>) -> Point {
        loop {
            let x = std::array::from_fn(|_| 2.0 * rng.uniform() - 1.0);
            if label.is_none() || label == Some(self.label(&x)) {
                return x;
            }
        }
    }
}
#[derive(Clone)]
struct Classifier {
    neural: bool,
    linear: Point,
    inner: [[f64; D]; H],
    bias: [f64; H],
    outer: [f64; H],
    intercept: f64,
    threshold: f64,
}
impl Classifier {
    fn predict(&self, x: &Point) -> f64 {
        let logit = if self.neural {
            self.intercept
                + (0..H)
                    .map(|h| {
                        self.outer[h]
                            * (self.bias[h] + (0..D).map(|i| self.inner[h][i] * x[i]).sum::<f64>())
                                .tanh()
                    })
                    .sum::<f64>()
        } else {
            self.intercept + (0..D).map(|i| self.linear[i] * x[i]).sum::<f64>()
        };
        sigmoid(logit)
    }
    fn margin(&self, x: &Point) -> f64 {
        self.predict(x) - self.threshold
    }
    fn train(
        teacher: &Teacher,
        rng: &mut Rng,
        neural: bool,
        n: usize,
        epochs: usize,
        calibration: usize,
    ) -> Self {
        let mut m = Self {
            neural,
            linear: [0.0; D],
            inner: std::array::from_fn(|_| std::array::from_fn(|_| 0.3 * rng.normal())),
            bias: [0.0; H],
            outer: std::array::from_fn(|_| 0.3 * rng.normal()),
            intercept: 0.0,
            threshold: 0.5,
        };
        let mut data: Vec<_> = (0..n)
            .map(|_| {
                let x = teacher.draw(rng, None);
                let y = teacher.label(&x);
                (x, y)
            })
            .collect();
        for _ in 0..epochs {
            rng.shuffle(&mut data);
            for (x, y) in &data {
                let delta = m.predict(x) - if *y { 1.0 } else { 0.0 };
                let rate = 0.035;
                if neural {
                    for h in 0..H {
                        let a =
                            (m.bias[h] + (0..D).map(|i| m.inner[h][i] * x[i]).sum::<f64>()).tanh();
                        let d = delta * m.outer[h] * (1.0 - a * a);
                        for (i, value) in x.iter().enumerate() {
                            m.inner[h][i] -= rate * (d * value + 0.0005 * m.inner[h][i]);
                        }
                        m.bias[h] -= rate * d;
                        m.outer[h] -= rate * (delta * a + 0.0005 * m.outer[h]);
                    }
                } else {
                    for (i, value) in x.iter().enumerate() {
                        m.linear[i] -= rate * (delta * value + 0.0005 * m.linear[i]);
                    }
                }
                m.intercept -= rate * delta;
            }
        }
        // Calibration is disjoint from both training and evaluation.
        let mut benign: Vec<_> = (0..calibration)
            .map(|_| m.predict(&teacher.draw(rng, Some(false))))
            .collect();
        benign.sort_by(f64::total_cmp);
        m.threshold = benign[((0.95 * calibration as f64).ceil() as usize).min(calibration) - 1];
        m
    }
}
fn edit(
    origin: &Point,
    teacher: &Teacher,
    surrogate: &Classifier,
    rng: &mut Rng,
    radius: f64,
) -> Point {
    let mut best = *origin;
    let mut best_score = surrogate.margin(&best);
    for _ in 0..12 {
        for _ in 0..8 {
            let mut candidate: Point =
                std::array::from_fn(|i| (best[i] + radius * 0.18 * rng.normal()).clamp(-1.0, 1.0));
            let distance = (0..D)
                .map(|i| (candidate[i] - origin[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            if distance > radius {
                for i in 0..D {
                    candidate[i] = origin[i] + (candidate[i] - origin[i]) * radius / distance;
                }
            }
            if !teacher.label(&candidate) {
                continue;
            }
            let score = surrogate.margin(&candidate);
            if score < best_score {
                best_score = score;
                best = candidate;
            }
        }
    }
    best
}
#[derive(Clone, Serialize)]
pub struct Config {
    pub tasks: usize,
    pub train: usize,
    pub calibration: usize,
    pub episodes: usize,
    pub epochs: usize,
    pub gates: usize,
}
impl Config {
    pub fn quick() -> Self {
        Self {
            tasks: 4,
            train: 300,
            calibration: 300,
            episodes: 300,
            epochs: 15,
            gates: 6,
        }
    }
    pub fn paper() -> Self {
        Self {
            tasks: 8,
            train: 600,
            calibration: 800,
            episodes: 1000,
            epochs: 25,
            gates: 6,
        }
    }
}
#[derive(Serialize)]
pub struct TaskResult {
    pub task: usize,
    pub arm: String,
    pub radius: f64,
    pub defender_queries_per_episode: usize,
    pub episodes: usize,
    pub single_miss: Vec<f64>,
    pub joint_miss: Vec<f64>,
    pub conditional_miss: Vec<Option<f64>>,
    pub product_prediction: f64,
    pub all_miss_count: usize,
    pub benign_false_block: f64,
}
#[derive(Serialize)]
pub struct Summary {
    pub arm: String,
    pub radius: f64,
    pub joint: Mean,
    pub product: Mean,
    pub mean_single: Mean,
    pub false_block: Mean,
}
#[derive(Serialize)]
pub struct Result {
    pub config: Config,
    pub summary: Vec<Summary>,
    pub tasks: Vec<TaskResult>,
}
pub fn run(seed: u64, config: Config) -> Result {
    assert!(
        config.gates > 0
            && config.gates <= 6
            && config.tasks > 1
            && config.episodes > 0
            && config.calibration > 20
    );
    let mut rows = Vec::new();
    for task in 0..config.tasks {
        let mut rng = Rng(seed.wrapping_add(task as u64 * 1_000_003));
        let teacher = Teacher::new(&mut rng);
        let same: Vec<_> = (0..config.gates)
            .map(|_| {
                Classifier::train(
                    &teacher,
                    &mut rng,
                    true,
                    config.train,
                    config.epochs,
                    config.calibration,
                )
            })
            .collect();
        let mixed: Vec<_> = (0..config.gates)
            .map(|i| {
                if i % 2 == 0 {
                    same[i].clone()
                } else {
                    Classifier::train(
                        &teacher,
                        &mut rng,
                        false,
                        config.train,
                        config.epochs,
                        config.calibration,
                    )
                }
            })
            .collect();
        let surrogate = Classifier::train(
            &teacher,
            &mut rng,
            true,
            config.train,
            config.epochs,
            config.calibration,
        );
        let hazard: Vec<_> = (0..config.episodes)
            .map(|_| teacher.draw(&mut rng, Some(true)))
            .collect();
        let benign: Vec<_> = (0..config.episodes)
            .map(|_| teacher.draw(&mut rng, Some(false)))
            .collect();
        for radius in [0.0, 1.5] {
            // Shared-input arms see exactly the same held-out/edited points.
            let edited: Vec<_> = hazard
                .iter()
                .map(|x| {
                    if radius == 0.0 {
                        *x
                    } else {
                        edit(x, &teacher, &surrogate, &mut rng, radius)
                    }
                })
                .collect();
            for arm in [
                "neural_shared",
                "mixed_shared",
                "mixed_independent",
                "mixed_feedback8",
            ] {
                if arm == "mixed_feedback8" && radius == 0.0 {
                    continue;
                }
                let models = if arm == "neural_shared" {
                    &same
                } else {
                    &mixed
                };
                let mut marginal = vec![0; config.gates];
                let mut joint = vec![0; config.gates];
                let mut false_blocks = 0;
                for episode in 0..config.episodes {
                    let chosen = if arm == "mixed_feedback8" {
                        // Adaptive selection from 8 surrogate-edited candidates;
                        // every feedback query evaluates the entire fixed pipeline.
                        let mut best = edited[episode];
                        let quality = |x: &Point| {
                            models
                                .iter()
                                .map(|m| m.margin(x))
                                .fold(f64::NEG_INFINITY, f64::max)
                        };
                        let mut score = quality(&best);
                        for _ in 1..8 {
                            let c = edit(&hazard[episode], &teacher, &surrogate, &mut rng, radius);
                            let q = quality(&c);
                            if q < score {
                                score = q;
                                best = c;
                            }
                        }
                        best
                    } else {
                        edited[episode]
                    };
                    let mut all = true;
                    let mut blocked = false;
                    for (j, m) in models.iter().enumerate() {
                        let x = if arm == "mixed_independent" {
                            let original = teacher.draw(&mut rng, Some(true));
                            if radius == 0.0 {
                                original
                            } else {
                                edit(&original, &teacher, &surrogate, &mut rng, radius)
                            }
                        } else {
                            chosen
                        };
                        let miss = m.margin(&x) <= 0.0;
                        marginal[j] += usize::from(miss);
                        all &= miss;
                        joint[j] += usize::from(all);
                        let b = if arm == "mixed_independent" {
                            teacher.draw(&mut rng, Some(false))
                        } else {
                            benign[episode]
                        };
                        blocked |= m.margin(&b) > 0.0;
                    }
                    false_blocks += usize::from(blocked);
                }
                let n = config.episodes as f64;
                let product = marginal.iter().map(|&m| m as f64 / n).product();
                let conditional_miss = (0..config.gates)
                    .map(|j| {
                        let denominator = if j == 0 {
                            config.episodes
                        } else {
                            joint[j - 1]
                        };
                        (denominator > 0).then(|| joint[j] as f64 / denominator as f64)
                    })
                    .collect();
                rows.push(TaskResult {
                    task,
                    arm: arm.into(),
                    radius,
                    defender_queries_per_episode: if arm == "mixed_feedback8" { 8 } else { 0 },
                    episodes: config.episodes,
                    single_miss: marginal.iter().map(|&m| m as f64 / n).collect(),
                    joint_miss: joint.iter().map(|&m| m as f64 / n).collect(),
                    conditional_miss,
                    product_prediction: product,
                    all_miss_count: joint[config.gates - 1],
                    benign_false_block: false_blocks as f64 / n,
                });
            }
        }
    }
    let mut summary = Vec::new();
    for radius in [0.0, 1.5] {
        for arm in [
            "neural_shared",
            "mixed_shared",
            "mixed_independent",
            "mixed_feedback8",
        ] {
            let selected: Vec<_> = rows
                .iter()
                .filter(|r| r.arm == arm && r.radius == radius)
                .collect();
            if selected.is_empty() {
                continue;
            }
            summary.push(Summary {
                arm: arm.into(),
                radius,
                joint: mean(
                    &selected
                        .iter()
                        .map(|r| *r.joint_miss.last().unwrap())
                        .collect::<Vec<_>>(),
                ),
                product: mean(
                    &selected
                        .iter()
                        .map(|r| r.product_prediction)
                        .collect::<Vec<_>>(),
                ),
                mean_single: mean(
                    &selected
                        .iter()
                        .map(|r| r.single_miss.iter().sum::<f64>() / config.gates as f64)
                        .collect::<Vec<_>>(),
                ),
                false_block: mean(
                    &selected
                        .iter()
                        .map(|r| r.benign_false_block)
                        .collect::<Vec<_>>(),
                ),
            });
        }
    }
    Result {
        config,
        summary,
        tasks: rows,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learning_and_attack_obey_the_declared_fixture() {
        let mut rng = Rng(42);
        let teacher = Teacher::new(&mut rng);
        let model = Classifier::train(&teacher, &mut rng, true, 600, 30, 600);
        let mut correct = 0;
        for _ in 0..500 {
            let x = teacher.draw(&mut rng, None);
            correct += usize::from((model.predict(&x) > 0.5) == teacher.label(&x));
        }
        assert!(
            correct > 375,
            "training must learn substantially more than a constant predictor: {correct}/500"
        );
        for _ in 0..30 {
            let original = teacher.draw(&mut rng, Some(true));
            let changed = edit(&original, &teacher, &model, &mut rng, 1.5);
            assert!(teacher.label(&changed));
            assert!(
                (0..D)
                    .map(|i| (changed[i] - original[i]).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    <= 1.5 + 1e-12
            );
            assert!(model.margin(&changed) <= model.margin(&original) + 1e-12);
            assert!(changed.iter().all(|v| (-1.0..=1.0).contains(v)));
        }
    }
}
