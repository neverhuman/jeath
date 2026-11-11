"""E21  Numerical checks of the results added in version 9.

  ancestors : effective size of a sex equals (1 - q) / (1 + alpha) times its number (Proposition 18), by direct
              simulation of drift at a uniparental marker
  waiting   : time to reach a required reliability about delivery is inversely proportional to the number of independent
              witnesses and to the unstaged share of occasions (Proposition 19)
  ritual    : repetitions of a safe trial needed to match one dangerous trial (Corollary 9)
  narrow    : a trait that helps in half the domains and hurts in the other half is neutral under broad comparison and
              penalized under a test confined to one half (Corollary 10)
  coercion  : selection through assessment scales with the assessed share of matings (Eq. coercion)

Run:  python3 code/exp_notes.py
"""
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20261129


def ne_ratio(q, alpha):
    return (1.0 - q) / (1.0 + alpha)


def drift_ne(q, alpha, rng, n=4000, reps=6000):
    """Ne/N from the variance of allele-frequency change at a uniparental marker in one generation."""
    p0 = 0.5
    dp = np.empty(reps)
    k = n // 2
    for r in range(reps):
        w = np.zeros(n)
        rep = rng.random(n) >= q
        m = rep.sum()
        w[rep] = rng.gamma(1.0 / alpha, alpha, m) if alpha > 0 else 1.0
        carriers = rng.permutation(n)[:k]
        pc = w[carriers].sum() / w.sum()
        dp[r] = rng.binomial(n, pc) / n - p0
    return p0 * (1 - p0) / dp.var() / n


def ancestors(rng):
    cells = []
    for q, a in ((0.0, 0.0), (0.2, 0.0), (0.6, 0.0), (0.2, 0.5), (0.2, 1.0), (0.5, 0.5), (0.4, 2.0)):
        sim = drift_ne(q, a, rng)
        cells.append({"q": q, "alpha": a, "theory": ne_ratio(q, a), "sim": float(sim)})
    worst = max(abs(c["sim"] / c["theory"] - 1) for c in cells)
    # readings of a twofold female-to-male ratio
    qf = 0.2
    readings = {"headcount_equal_alpha": {"qf": qf, "qm": 1 - (1 - qf) / 2},                      # 80% and 40%
                "skew_only": {"qf": qf, "qm": qf, "alpha_f": 0.0, "alpha_m": 1.0},                # same share, skewed fathers
                "mixed": {"qf": qf, "qm": 0.4, "alpha_f": 0.0, "alpha_m": 2 * (1 - 0.4) / (1 - qf) - 1}}
    curve = {"qm": [float(x) for x in np.linspace(0.0, 0.9, 91)]}
    for a in (0.0, 0.5, 1.0, 2.0):
        curve[f"alpha_{a}"] = [float(ne_ratio(qf, 0.0) / ne_ratio(x, a)) for x in curve["qm"]]
    return {"cells": cells, "max_rel_err": float(worst), "readings": readings, "curve": curve, "qf": qf}


def wait(k, lam, d, r1, rstar):
    """Periods needed for k witnesses, lam occasions each per period, share d staged, to reach reliability rstar."""
    need = rstar * (1 - r1) / (r1 * (1 - rstar))
    return need / (k * lam * (1 - d))


def waiting(rng):
    r1, rstar, lam = 0.05, 0.8, 4.0
    rows = []
    for k, d in ((30, 0.2), (15, 0.2), (5, 0.2), (2, 0.2), (1, 0.2), (1, 0.8), (30, 0.8)):
        t = wait(k, lam, d, r1, rstar)
        # simulate: true delivery z ~ N(0, r1 share), each unstaged observation adds independent noise
        nobs = max(1, int(round(t * k * lam * (1 - d))))
        z = rng.normal(0, np.sqrt(r1), 60000)
        mean = z + rng.normal(0, np.sqrt((1 - r1) / nobs), 60000)
        rows.append({"k": k, "d": d, "periods": float(t), "sim_reliability": float(np.corrcoef(z, mean)[0, 1] ** 2)})
    worst = max(abs(r["sim_reliability"] - rstar) for r in rows)
    grid = {"k": list(range(1, 41)), "periods_d02": [float(wait(k, lam, 0.2, r1, rstar)) for k in range(1, 41)],
            "periods_d08": [float(wait(k, lam, 0.8, r1, rstar)) for k in range(1, 41)]}
    return {"r1": r1, "rstar": rstar, "lam": lam, "rows": rows, "max_abs_err": float(worst), "grid": grid}


def ritual(rng):
    out = []
    for rs, rd in ((0.10, 0.50), (0.20, 0.50), (0.05, 0.60), (0.30, 0.70)):
        n = rd * (1 - rs) / (rs * (1 - rd))
        nint = int(np.ceil(n))
        t = rng.normal(0, np.sqrt(rs), 80000)
        safe = t + rng.normal(0, np.sqrt((1 - rs) / nint), 80000)
        out.append({"r_safe": rs, "r_danger": rd, "n_needed": float(n), "n_used": nint,
                    "sim_reliability": float(np.corrcoef(t, safe)[0, 1] ** 2),
                    "theory_reliability": nint * rs / (1 + (nint - 1) * rs)})
    return {"rows": out, "max_abs_err": float(max(abs(r["sim_reliability"] - r["theory_reliability"]) for r in out))}


def narrow(rng):
    n, K = 200000, 10
    x = rng.normal(size=n)
    g = rng.normal(size=n)
    load = np.array([+0.3] * (K // 2) + [-0.3] * (K // 2))
    perf = g[:, None] + x[:, None] * load[None, :] + rng.normal(size=(n, K))
    out = {"K": K, "loading": 0.3}
    for name, cols in (("broad", list(range(K))), ("half_that_hurts", list(range(K // 2, K))), ("half_that_helps", list(range(K // 2)))):
        score = perf[:, cols].mean(axis=1)
        out[name] = {"slope_on_x": float(np.cov(score, x)[0, 1] / x.var()), "theory": float(load[cols].mean()),
                     "slope_on_g": float(np.cov(score, g)[0, 1] / g.var())}
    out["max_abs_err"] = float(max(abs(out[k]["slope_on_x"] - out[k]["theory"]) for k in ("broad", "half_that_hurts", "half_that_helps")))
    return out


def coercion(rng):
    """Selection differential on an assessed trait when a share kappa of matings is settled without assessing it."""
    n = 400000
    z = rng.normal(size=n)                       # assessed trait (standardized)
    body = rng.normal(size=n)                    # what a contest of force rewards, independent of z
    rows = []
    for kappa in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        forced = rng.random(n) < kappa
        key = np.where(forced, body, z)
        w = (key > np.quantile(key, 0.5)).astype(float)       # the upper half reproduces
        rows.append({"kappa": kappa, "S_assessed": float(np.cov(w / w.mean(), z)[0, 1]),
                     "S_body": float(np.cov(w / w.mean(), body)[0, 1])})
    s0 = rows[0]["S_assessed"]
    for r in rows:
        r["theory"] = (1 - r["kappa"]) * s0
    return {"rows": rows, "max_abs_err": float(max(abs(r["S_assessed"] - r["theory"]) for r in rows))}


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    out = {"_seed": SEED, "_written": time.strftime("%Y-%m-%d %H:%M:%S"), "ancestors": ancestors(rng), "waiting": waiting(rng),
           "ritual": ritual(rng), "narrow": narrow(rng), "coercion": coercion(rng)}
    json.dump(out, open(os.path.join(ROOT, "results", "e21_notes.json"), "w"), indent=1)
    print("E21", {"ancestors_rel_err": round(out["ancestors"]["max_rel_err"], 4), "waiting_err": round(out["waiting"]["max_abs_err"], 4),
                  "ritual_err": round(out["ritual"]["max_abs_err"], 4), "narrow_err": round(out["narrow"]["max_abs_err"], 4),
                  "coercion_err": round(out["coercion"]["max_abs_err"], 4)})
