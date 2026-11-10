//! Literal small versions of the drafts' selection illustrations. The reward
//! functions build in the welfare assumptions; this is not evidence about LLMs.
use crate::{
    rng::Rng,
    stats::{Mean, mean},
};
use serde::Serialize;
#[derive(Serialize)]
pub struct TraitResult {
    pub score: String,
    pub gate: f64,
    pub intensity: f64,
    pub capacity: Mean,
    pub taking_effort: Mean,
}
#[derive(Serialize)]
pub struct GambleResult {
    pub score: String,
    pub gamble_share: Mean,
    pub final_harm_fraction: Mean,
    pub final_welfare: Mean,
}
#[derive(Serialize)]
pub struct Result {
    pub populations: usize,
    pub agents: usize,
    pub generations: usize,
    pub traits: Vec<TraitResult>,
    pub gamble: Vec<GambleResult>,
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
pub fn run(seed: u64, populations: usize, agents: usize, generations: usize) -> Result {
    assert!(populations > 1 && agents >= 10 && agents % 5 == 0 && generations > 0);
    let mut traits = Vec::new();
    for (kind, gate) in [
        ("holdings", 0.0),
        ("holdings", 0.8),
        ("holdings", 1.0),
        ("contribution", 0.0),
    ] {
        for intensity in [0.5, 2.0] {
            let mut caps = Vec::new();
            let mut efforts = Vec::new();
            for rep in 0..populations {
                // Common seed across interventions provides a paired design.
                let mut rng = Rng(seed.wrapping_add(rep as u64));
                let mut pop: Vec<_> = (0..agents).map(|_| (rng.normal(), rng.normal())).collect();
                for _ in 0..generations {
                    rng.shuffle(&mut pop);
                    let mut scores = vec![0.0; agents];
                    for start in (0..agents).step_by(5) {
                        let output: Vec<_> =
                            (start..start + 5).map(|i| (0.3 * pop[i].0).exp()).collect();
                        let mut held = [0.0; 5];
                        let mut taken = [0.0; 5];
                        let total: f64 = output.iter().sum();
                        for i in 0..5 {
                            held[i] = 0.5 * output[i] + 0.5 * (total - output[i]) / 4.0
                                - 0.5 * sigmoid(pop[start + i].1) * gate;
                        }
                        for i in 0..5 {
                            for j in 0..5 {
                                if i != j {
                                    let x =
                                        0.5 * sigmoid(pop[start + i].1) * (1.0 - gate) * output[j]
                                            / 4.0
                                            * output[i]
                                            / (output[i] + output[j]);
                                    taken[i] += x;
                                    held[i] += 0.8 * x;
                                    held[j] -= x;
                                }
                            }
                        }
                        let group_mean = held.iter().sum::<f64>() / 5.0;
                        for i in 0..5 {
                            scores[start + i] = if kind == "holdings" {
                                held[i] - group_mean
                            } else {
                                0.5 * output[i] - taken[i]
                            };
                        }
                    }
                    let m = scores.iter().sum::<f64>() / agents as f64;
                    let sd = (scores.iter().map(|s| (s - m).powi(2)).sum::<f64>() / agents as f64)
                        .sqrt()
                        .max(1e-12);
                    let logweights: Vec<_> =
                        scores.iter().map(|s| intensity * (s - m) / sd).collect();
                    let max = logweights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<_> = logweights.iter().map(|x| (x - max).exp()).collect();
                    pop = (0..agents)
                        .map(|_| {
                            let (c, t) = pop[rng.weighted(&weights)];
                            (c + 0.05 * rng.normal(), t + 0.05 * rng.normal())
                        })
                        .collect();
                }
                caps.push(pop.iter().map(|x| x.0).sum::<f64>() / agents as f64);
                efforts.push(pop.iter().map(|x| sigmoid(x.1)).sum::<f64>() / agents as f64);
            }
            traits.push(TraitResult {
                score: kind.into(),
                gate,
                intensity,
                capacity: mean(&caps),
                taking_effort: mean(&efforts),
            });
        }
    }
    let mut gamble = Vec::new();
    for kind in ["rank", "conferred_rank", "capped_welfare"] {
        let mut shares = Vec::new();
        let mut harms = Vec::new();
        let mut welfare = Vec::new();
        for rep in 0..populations {
            let mut rng = Rng(seed.wrapping_add(10_000 + rep as u64));
            let mut pop: Vec<_> = (0..agents)
                .map(|_| (0.5 + 0.03 * rng.normal()).clamp(0.0, 1.0))
                .collect();
            let mut last = (0.0, 0.0);
            for generation in 0..generations {
                let outcomes: Vec<_> = pop
                    .iter()
                    .map(|&g| {
                        let take = (1.0 - g) + g * (0.8 + 1.5 * rng.normal());
                        let harm = rng.uniform() < 0.05 * g;
                        (take, harm, take - if harm { 20.0 } else { 0.0 })
                    })
                    .collect();
                last = (
                    outcomes.iter().filter(|x| x.1).count() as f64 / agents as f64,
                    outcomes.iter().map(|x| x.2).sum::<f64>() / agents as f64,
                );
                // Report share, harm and welfare from the SAME final generation.
                if generation + 1 == generations {
                    break;
                }
                let weights = if kind == "capped_welfare" {
                    outcomes
                        .iter()
                        .map(|x| x.2.clamp(0.0, 1.0))
                        .collect::<Vec<_>>()
                } else {
                    let mut order: Vec<_> = (0..agents)
                        .filter(|&i| kind != "conferred_rank" || !outcomes[i].1)
                        .collect();
                    order.sort_by(|&a, &b| outcomes[b].0.total_cmp(&outcomes[a].0));
                    let mut w = vec![0.0; agents];
                    for &i in order.iter().take(agents / 10) {
                        w[i] = 1.0;
                    }
                    w
                };
                pop = (0..agents)
                    .map(|_| (pop[rng.weighted(&weights)] + 0.03 * rng.normal()).clamp(0.0, 1.0))
                    .collect();
            }
            shares.push(pop.iter().sum::<f64>() / agents as f64);
            harms.push(last.0);
            welfare.push(last.1);
        }
        gamble.push(GambleResult {
            score: kind.into(),
            gamble_share: mean(&shares),
            final_harm_fraction: mean(&harms),
            final_welfare: mean(&welfare),
        });
    }
    Result {
        populations,
        agents,
        generations,
        traits,
        gamble,
    }
}
