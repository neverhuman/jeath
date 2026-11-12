#!/usr/bin/env python3
"""E8: simulated power for the Stage-2 information x ability interaction.
Generative assumptions (stated in the paper): ability a ~ N(0,1), measured with
reliability 0.8; each target has a baseline appeal (SD 1, logit scale); each
clip of a target yields one impression of competence shared by everyone who sees
that clip, correlated r_c with ability (0.35 informative, 0.15 control); each
decision adds an observer intercept (SD 0.8); choice is Bernoulli with logit
0.8 per SD of perceived competence.  Analysis: per-target difference in choice
rates (informative minus control) regressed on measured ability, two-sided 5%.
"""
import numpy as np
from scipy import stats
from common import rng_for, save

def one(rng, T, clips, obs, r_info=0.35, r_ctrl=0.15, slope=0.8, sd_app=1.0, sd_obs=0.8, rel=0.8):
    a = rng.normal(size=T)
    m = np.sqrt(rel) * a + np.sqrt(1 - rel) * rng.normal(size=T)
    app = sd_app * rng.normal(size=T)
    rates = []
    per = obs // clips
    for rc in (r_info, r_ctrl):
        p = rc * a[:, None] + np.sqrt(1 - rc ** 2) * rng.normal(size=(T, clips))       # clip-level impressions
        eta = app[:, None, None] + slope * p[:, :, None] + sd_obs * rng.normal(size=(T, clips, per))
        ch = rng.random((T, clips, per)) < 1 / (1 + np.exp(-eta))
        rates.append(ch.mean(axis=(1, 2)))
    d = rates[0] - rates[1]
    res = stats.linregress(m, d)
    return res.pvalue < 0.05, res.slope

def e8(seed=20261116, reps=500):
    rows = []
    for ci, clips in enumerate((1, 4)):
        for oi, obs in enumerate((20, 40)):
            for ti, T in enumerate((100, 200, 400, 600, 800, 1000, 1200)):
                rng = rng_for(seed, ci, oi, ti)
                out = [one(rng, T, clips, obs) for _ in range(reps)]
                pw = float(np.mean([o[0] for o in out]))
                rows.append(dict(clips=clips, observers=obs, targets=T, power=pw,
                                 se=float(np.sqrt(pw * (1 - pw) / reps)), slope=float(np.mean([o[1] for o in out])),
                                 decisions=2 * T * obs))
    def need(clips, obs, target=0.9):
        xs = [r for r in rows if r["clips"] == clips and r["observers"] == obs]
        for lo, hi in zip(xs[:-1], xs[1:]):
            if lo["power"] < target <= hi["power"]:
                return lo["targets"] + (target - lo["power"]) / (hi["power"] - lo["power"]) * (hi["targets"] - lo["targets"])
        return None if xs[-1]["power"] < target else xs[0]["targets"]
    return save("e8_power", dict(seed=seed, reps=reps, rows=rows,
                                 targets_for_90={f"{c}_{o}": need(c, o) for c in (1, 4) for o in (20, 40)},
                                 mean_slope=float(np.mean([r["slope"] for r in rows]))))

if __name__ == "__main__":
    r = e8()
    for x in r["rows"]:
        print(x["clips"], x["observers"], x["targets"], round(x["power"], 3), round(x["slope"], 4))
    print(r["targets_for_90"])
