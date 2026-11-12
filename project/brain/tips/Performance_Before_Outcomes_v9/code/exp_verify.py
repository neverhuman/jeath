"""E16: tests, enforcement and verification (Model VIII; Propositions 16 and 17).

1. Staging.  An assessor sees n occasions, of which a share d is recognizable to the candidate
   as assessment.  Capacity is displayed at full effort on those occasions; a conditional
   candidate never defaults on them.  Monte Carlo validities for capacity and for delivery are
   compared with the closed forms
       R_b = V_B / (V_B + sigma^2 / (d n)),      R_p = V_P / (V_P + nu / ((1 - d) n)),
   where nu = E[q (1 - q)] and q is the default rate on unrecognized occasions.  Also: the
   chance that a defaulter keeps a clean record, and the staging share that maximizes R_b R_p.

2. Enforcement.  A share e of every candidate's product reaches the assessors whatever the
   candidate's disposition.  Equilibrium capacity and evolved commitment from the closed form,
   from iteration of the recursion for the means, and from an individual-based simulation.

Run with:  python3 code/exp_verify.py        (about two minutes; fixed seed)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import save, mcse

SEED = 20261123

# ---- staging -------------------------------------------------------------------------------
VB, SIG2 = 1.0, 4.0              # capacity variance; noise variance of one performance
QA, QB = 2.0, 6.0                # default rate q ~ Beta(2, 6): mean 0.25
N_OCC = 40                       # occasions observed
D_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def beta_moments(a=QA, b=QB):
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    nu = mean * (1 - mean) - var          # E[q(1-q)]
    return mean, var, nu


def r_capacity(n, d, vb=VB, sig2=SIG2):
    m = d * n
    return 0.0 if m <= 0 else vb / (vb + sig2 / m)


def r_delivery(n, d):
    _, vp, nu = beta_moments()
    m = (1 - d) * n
    return 0.0 if m <= 0 else vp / (vp + nu / m)


def d_star(n):
    """Staging share that maximizes R_b * R_p (closed form)."""
    _, vp, nu = beta_moments()
    A, C = SIG2 / (n * VB), nu / (n * vp)
    u, v = math.sqrt(A * (1 + C)), math.sqrt(C * (1 + A))
    return u / (u + v)


def staging(reps=20, N=300_000):
    rng = np.random.default_rng(SEED)
    out = {}
    for d in D_GRID:
        ms = int(round(d * N_OCC)); mu = N_OCC - ms
        rb, rp = [], []
        for _ in range(reps):
            b = rng.normal(0, math.sqrt(VB), N)
            q = rng.beta(QA, QB, N)
            if ms > 0:
                yb = b + rng.normal(0, math.sqrt(SIG2 / ms), N)      # mean of ms full-effort performances
                rb.append(np.corrcoef(yb, b)[0, 1] ** 2)
            else:
                rb.append(0.0)
            if mu > 0:
                f = rng.binomial(mu, q)                               # defaults seen on unrecognized occasions
                rp.append(np.corrcoef(f, q)[0, 1] ** 2)
            else:
                rp.append(0.0)
        eb, ep = r_capacity(N_OCC, d), r_delivery(N_OCC, d)
        out[str(d)] = dict(staged=ms, unstaged=mu,
                           R_b=[float(np.mean(rb)), float(mcse(rb))], R_b_exact=eb,
                           R_p=[float(np.mean(rp)), float(mcse(rp))], R_p_exact=ep,
                           z_b=float((np.mean(rb) - eb) / mcse(rb)) if ms > 0 else 0.0,
                           z_p=float((np.mean(rp) - ep) / mcse(rp)) if mu > 0 else 0.0)
    return out


def clean_record():
    """Chance that a candidate who defaults on a quarter of unrecognized occasions shows no default."""
    rng = np.random.default_rng(SEED + 1)
    q0, out = 0.25, {}
    for d in D_GRID[:-1]:
        mu = N_OCC - int(round(d * N_OCC))
        sim = float(np.mean(rng.binomial(mu, q0, 2_000_000) == 0))
        out[str(d)] = dict(exact=(1 - q0) ** mu, sim=sim,
                           upper95=1 - 0.05 ** (1 / mu),                     # limit on the default rate after a clean record
                           n_half=math.log(2) / (-(1 - d) * math.log(1 - q0)))  # occasions for an even chance of one default
    return out


def staging_optimum():
    grid = np.linspace(0.001, 0.999, 9981)
    out = {}
    for n in (10, 40, 200):
        prod = np.array([r_capacity(n, d) * r_delivery(n, d) for d in grid])
        out[str(n)] = dict(closed=d_star(n), grid=float(grid[prod.argmax()]), product=float(prod.max()))
    return out


# ---- enforcement ---------------------------------------------------------------------------
ENF = dict(s=1.0, Rb=0.5, Rp=0.3, cB=0.5, cP=0.5)        # take-off number T = s^2 Rb Rp / (cB cP) = 0.6


def takeoff(par=ENF):
    return par["s"] ** 2 * par["Rb"] * par["Rp"] / (par["cB"] * par["cP"])


def equilibrium(e, par=ENF):
    """Closed form (b*, p*, delivered share) for enforced share e."""
    T = takeoff(par); bmax = par["s"] * par["Rb"] / par["cB"]
    if e >= 1:
        return bmax, 0.0, 1.0
    if T * (1 - e) >= 1:                                   # delivery reaches its bound
        return bmax, 1.0, 1.0
    den = 1 - T * (1 - e) ** 2
    p = T * e * (1 - e) / den
    delta = e / den
    return bmax * delta, p, delta


def iterate(e, par=ENF, gB=0.05, gP=0.02, steps=200_000, start=(1e-6, 1e-6)):
    b, p = start
    for _ in range(steps):
        delta = e + (1 - e) * p
        b += gB * (par["s"] * delta * par["Rb"] - par["cB"] * b)
        p += gP * (par["s"] * (1 - e) * b * par["Rp"] - par["cP"] * p)
        p = min(max(p, 0.0), 1.0)
    return b, p


def e_star(par=ENF):
    """Enforced share at which evolved commitment is greatest (T < 1)."""
    T = takeoff(par)
    return 1 - (1 - math.sqrt(1 - T)) / T


def ibm(e, rng, par=ENF, N=20_000, gens=2000, tail=400, vb=0.05, vp=0.02):
    """Individual-based check: phenotype equals breeding value; assessors shrink noisy signals."""
    s, Rb, Rp, cB, cP = (par[k] for k in ("s", "Rb", "Rp", "cB", "cP"))
    sb, sp = math.sqrt(vb * (1 - Rb) / Rb), math.sqrt(vp * (1 - Rp) / Rp)
    b = rng.normal(0, math.sqrt(vb), N); p = rng.normal(0, math.sqrt(vp), N)
    tb, tp = [], []
    for t in range(gens):
        yb = b + rng.normal(0, sb, N); yp = p + rng.normal(0, sp, N)
        bh = b.mean() + Rb * (yb - yb.mean()); ph = p.mean() + Rp * (yp - yp.mean())
        vhat = (e + (1 - e) * ph) * bh
        w = 1 + s * (vhat - vhat.mean()) - 0.5 * cB * (b ** 2 - (b ** 2).mean()) - 0.5 * cP * (p ** 2 - (p ** 2).mean())
        w = np.clip(w, 1e-9, None); pr = w / w.sum()
        ma, pa = rng.choice(N, N, p=pr), rng.choice(N, N, p=pr)
        b = 0.5 * (b[ma] + b[pa]) + rng.normal(0, math.sqrt(vb / 2), N)
        p = 0.5 * (p[ma] + p[pa]) + rng.normal(0, math.sqrt(vp / 2), N)
        if t >= gens - tail:
            tb.append(b.mean()); tp.append(p.mean())
    return float(np.mean(tb)), float(np.mean(tp))


def enforcement(reps=3):
    grid = [round(x, 3) for x in np.linspace(0, 1, 41)]
    closed = [equilibrium(e) for e in grid]
    worst = 0.0
    for e in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        b, p = iterate(e); cb, cp, _ = equilibrium(e)
        worst = max(worst, abs(b - cb), abs(p - cp))
    high = dict(ENF, Rp=0.75)                              # T = 1.5: choice alone already takes off
    worst_high = 0.0
    for e in (0.1, 0.25, 0.5, 0.75, 1.0):
        b, p = iterate(e, high); cb, cp, _ = equilibrium(e, high)
        worst_high = max(worst_high, abs(b - cb), abs(p - cp))
    rng = np.random.default_rng(SEED + 2)
    sim = {}
    for e in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        runs = np.array([ibm(e, rng) for _ in range(reps)])
        cb, cp, _ = equilibrium(e)
        sim[str(e)] = dict(b=[float(runs[:, 0].mean()), float(mcse(runs[:, 0]))], b_exact=cb,
                           p=[float(runs[:, 1].mean()), float(mcse(runs[:, 1]))], p_exact=cp)
    es = e_star()
    fine = np.linspace(0, 1, 100_001)
    pf = np.array([equilibrium(float(x))[1] for x in fine])
    dh = np.array([equilibrium(float(x), high)[2] for x in fine])          # delivered share when choice alone takes off
    crowd = dict(T=takeoff(high), min_delivered=float(dh.min()), at=float(fine[dh.argmin()]))
    return dict(params=dict(ENF, T=takeoff(), bmax=ENF["s"] * ENF["Rb"] / ENF["cB"]), crowding=crowd,
                grid=grid, b=[c[0] for c in closed], p=[c[1] for c in closed], delivered=[c[2] for c in closed],
                recursion_max_err=worst, recursion_max_err_T15=worst_high, ibm=sim,
                e_star=es, e_star_numeric=float(fine[pf.argmax()]), p_max=float(pf.max()),
                b_at_half=equilibrium(0.5)[0], p_at_half=equilibrium(0.5)[1], b_at_fifth=equilibrium(0.2)[0])


if __name__ == "__main__":
    mean, vp, nu = beta_moments()
    res = dict(params=dict(VB=VB, sig2=SIG2, q_mean=mean, VP=vp, nu=nu, n=N_OCC),
               staging=staging(), clean=clean_record(), optimum=staging_optimum(), enforcement=enforcement())
    save("e16_verify", res)
    for d, v in res["staging"].items():
        print("d=%s  R_b %.4f (%.4f, z %+.1f)   R_p %.4f (%.4f, z %+.1f)" % (
            d, v["R_b"][0], v["R_b_exact"], v["z_b"], v["R_p"][0], v["R_p_exact"], v["z_p"]))
    for d, v in res["clean"].items():
        print("d=%s  clean record %.4f (sim %.4f)  95%% limit on default rate %.3f  n for even chance %.1f" % (
            d, v["exact"], v["sim"], v["upper95"], v["n_half"]))
    print("d*:", {n: (round(v["closed"], 4), round(v["grid"], 4)) for n, v in res["optimum"].items()})
    E = res["enforcement"]
    print("T %.2f  recursion err %.1e / %.1e  e* %.4f (%.4f)  p_max %.4f" % (
        E["params"]["T"], E["recursion_max_err"], E["recursion_max_err_T15"], E["e_star"], E["e_star_numeric"], E["p_max"]))
    print("crowding out at T %.1f: delivered share falls to %.3f at e = %.3f" % (
        E["crowding"]["T"], E["crowding"]["min_delivered"], E["crowding"]["at"]))
    for e, v in E["ibm"].items():
        print("e=%s  b %.3f±%.3f (%.3f)   p %.3f±%.3f (%.3f)" % (e, v["b"][0], v["b"][1], v["b_exact"], v["p"][0], v["p"][1], v["p_exact"]))
