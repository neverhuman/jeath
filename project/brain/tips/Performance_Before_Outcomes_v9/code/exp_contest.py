#!/usr/bin/env python3
"""E10-E12: contests, coalitions and the aim of selection (Model V).
 E10  Proposition 8: the loading of the score, not the intensity, sets the aim.
 E11  Proposition 9: repeatable (low-severity) contests yield more information.
 E12  Proposition 10: the despot's dilemma - rival strength limits reproductive skew.
Loadings and costs are illustrative assumptions, not estimates.
"""
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from common import rng_for, save, mcse, zscore, wmean

# ------------------------------------------------------------------ E10
VENUES = {"contest": (1.0, 0.25), "hunt": (0.6, 0.6), "verbal": (0.0, 1.0)}   # (loading on F, loading on G)
REGIMES = {"dominance": dict(contest=1.0, hunt=0.0, verbal=0.0),
           "mixed": dict(contest=1 / 3, hunt=1 / 3, verbal=1 / 3),
           "prestige": dict(contest=0.0, hunt=0.5, verbal=0.5)}
H2F = H2G = 0.5
VF = VG = 1.0
VNOISE = 1.0     # per-venue error variance

def regime_moments(w):
    lF = sum(w[k] * VENUES[k][0] for k in w); lG = sum(w[k] * VENUES[k][1] for k in w)
    vy = lF ** 2 * VF + lG ** 2 * VG + sum(w[k] ** 2 for k in w) * VNOISE
    return lF, lG, np.sqrt(vy)

def e10(seed=20261120, reps=20, N=200000):
    rows, zs = [], []
    rules = {"exponential": lambda z: np.exp(0.2 * z), "top_decile": lambda z: (z > norm.ppf(0.9)).astype(float)}
    inten = {"exponential": 0.2, "top_decile": norm.pdf(norm.ppf(0.9)) / 0.1}
    for gi, (rg, w) in enumerate(REGIMES.items()):
        lF, lG, sd = regime_moments(w)
        for ri, (rn, rule) in enumerate(rules.items()):
            sF, sG = [], []
            for r in range(reps):
                rng = rng_for(seed, gi, ri, r)
                aF = rng.normal(0, np.sqrt(H2F * VF), N); F = aF + rng.normal(0, np.sqrt((1 - H2F) * VF), N)
                aG = rng.normal(0, np.sqrt(H2G * VG), N); G = aG + rng.normal(0, np.sqrt((1 - H2G) * VG), N)
                Y = np.zeros(N)
                for k, wk in w.items():
                    if wk > 0:
                        Y += wk * (VENUES[k][0] * F + VENUES[k][1] * G + rng.normal(0, np.sqrt(VNOISE), N))
                z = (Y - Y.mean()) / Y.std(); ww = rule(z)
                sF.append(wmean(aF, ww) - aF.mean()); sG.append(wmean(aG, ww) - aG.mean())
            exF = inten[rn] * H2F * lF * VF / sd; exG = inten[rn] * H2G * lG * VG / sd
            zF = zscore(sF, exF) if exF > 0 else 0.0; zG = zscore(sG, exG)
            zs += [abs(zF), abs(zG)]
            rows.append(dict(regime=rg, rule=rn, lF=lF, lG=lG, sd=sd, exF=exF, exG=exG, mcF=float(np.mean(sF)),
                             mcG=float(np.mean(sG)), seF=mcse(sF), seG=mcse(sG), zF=zF, zG=zG,
                             ratio_exact=(exG / exF if exF > 0 else float("inf")),
                             ratio_mc=(float(np.mean(sG) / np.mean(sF)) if exF > 0 else float("inf"))))
    # a cue dominated by a transient state (age) carries little heritable selection and none on capacity
    cue = []
    for r in range(reps):
        rng = rng_for(seed, 9, r)
        aQ = rng.normal(0, np.sqrt(0.10), N); state = rng.normal(0, np.sqrt(0.80), N)
        Q = aQ + state + rng.normal(0, np.sqrt(0.10), N)
        aG = rng.normal(0, np.sqrt(H2G * VG), N)
        ww = np.exp(0.2 * (Q - Q.mean()) / Q.std())
        cue.append((wmean(aQ, ww) - aQ.mean(), wmean(aG, ww) - aG.mean()))
    cue = np.array(cue)
    dom = [x for x in rows if x["regime"] == "dominance" and x["rule"] == "exponential"][0]
    pre = [x for x in rows if x["regime"] == "prestige" and x["rule"] == "exponential"][0]
    return save("e10_aim", dict(seed=seed, reps=reps, N=N, venues=VENUES, regimes=REGIMES, rows=rows,
                                max_abs_z=max(zs), n_comparisons=len(zs),
                                cue_SQ=float(cue[:, 0].mean()), cue_SQ_exact=0.2 * 0.10 / 1.0,
                                cue_SG=float(cue[:, 1].mean()), cue_SG_se=mcse(cue[:, 1]),
                                aim_gain_G=pre["exG"] / dom["exG"], aim_loss_F=dom["exF"] / max(pre["exF"], 1e-12)))

# ------------------------------------------------------------------ E11
REP = dict(lam=1.0, VP=1.0, tau2=0.5, sig2=4.0, rho=0.25)

def R_of_n(n, p=REP):
    a = p["lam"] ** 2 * p["VP"]
    return a / (a + p["tau2"] + p["sig2"] * (p["rho"] + (1 - p["rho"]) / n))

def n_star(d_over_B, p=REP):
    """Maximise sqrt(R(n)) - (delta/B) n over n > 0; marginal benefit is strictly decreasing."""
    a = p["lam"] ** 2 * p["VP"]; b = p["tau2"] + p["sig2"] * p["rho"]; c = p["sig2"] * (1 - p["rho"])
    mb = lambda n: 0.5 * np.sqrt(R_of_n(n, p)) * c / (n * ((a + b) * n + c)) - d_over_B
    return brentq(mb, 1e-9, 1e9)

def e11(seed=20261121, reps=20, N=100000):
    p = REP; rows, zs = [], []
    for ni, n in enumerate((1, 2, 4, 8, 16, 32, 64)):
        v = []
        for r in range(reps):
            rng = rng_for(seed, ni, r)
            c = rng.normal(0, 1, N)
            y = p["lam"] * c + rng.normal(0, np.sqrt(p["tau2"]), N) + rng.normal(0, np.sqrt(p["sig2"] * p["rho"]), N)
            e = np.zeros(N)
            for _ in range(n):
                e += rng.normal(0, np.sqrt(p["sig2"] * (1 - p["rho"])), N)
            y = y + e / n
            v.append(np.corrcoef(c, y)[0, 1] ** 2)
        z = zscore(v, R_of_n(n)); zs.append(abs(z))
        rows.append(dict(n=n, exact=R_of_n(n), mc=float(np.mean(v)), z=z))
    ceil = 1.0 / (1.0 + p["tau2"] + p["sig2"] * p["rho"])
    opt = []
    for d in (0.2, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        nc = n_star(d); ne = max(1.0, nc)
        # brute-force check on a fine grid
        grid = np.linspace(0.01, 400, 400000); obj = np.sqrt(R_of_n(grid)) - d * grid
        nb = grid[np.argmax(obj)]
        opt.append(dict(d=d, n_cont=nc, n_brute=float(nb), n_eff=ne, R=R_of_n(ne), frac_ceiling=R_of_n(ne) / ceil,
                        gain_vs_single=float(np.sqrt(R_of_n(ne) / R_of_n(1)))))
    mono = all(opt[i]["n_cont"] < opt[i + 1]["n_cont"] for i in range(len(opt) - 1))
    # risk-budget form: equal cumulative risk D spread over n = D/delta events
    budget = [dict(delta=dl, n=0.2 / dl, R=R_of_n(max(1.0, 0.2 / dl))) for dl in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005)]
    return save("e11_repeat", dict(seed=seed, reps=reps, N=N, params=p, validity=rows, max_abs_z=max(zs),
                                   ceiling=ceil, R1=R_of_n(1), optimum=opt, monotone=bool(mono),
                                   max_root_vs_brute=max(abs(o["n_cont"] - o["n_brute"]) / o["n_cont"] for o in opt),
                                   budget=budget))

# ------------------------------------------------------------------ E12
def p_win(k, M, th):
    return (1 + k) ** th / ((1 + k) ** th + M ** th)

def U(k, M, q, th):
    return (1 - k * q) * p_win(k, M, th)

def k_star(M, q, th):
    foc = lambda k: -q / (1 - k * q) + th * (1 - p_win(k, M, th)) / (1 + k)
    if foc(0.0) <= 0:
        return 0.0
    return brentq(foc, 0.0, 1 / q - 1e-9)

def e12(Npool=20):
    rows, maxdiff, concave_ok, mono_ok = [], 0.0, True, True
    for th in (1.0, 2.0):
        for q in (0.02, 0.05, 0.10):
            prev = -1.0
            for M in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64):
                ks = k_star(M, q, th)
                kk = np.linspace(0, 1 / q - 1e-6, 200001)
                uu = U(kk, M, q, th); kb = kk[np.argmax(uu)]
                maxdiff = max(maxdiff, abs(kb - ks))
                d2 = np.diff(np.log(uu[1:-1]), 2)
                concave_ok &= bool(np.all(d2 < 1e-9))
                mono_ok &= ks >= prev - 1e-9; prev = ks
                kint = int(np.argmax([U(i, M, q, th) for i in range(0, int(1 / q))]))
                kpool = min(kint, Npool - 1)
                sd_ = 1 - kpool * q
                I = Npool * (sd_ ** 2 + kpool * q ** 2) - 1
                rows.append(dict(theta=th, q=q, M=M, k_cont=ks, k_int=kint, share=1 - ks * q, p=p_win(ks, M, th),
                                 I_pool=I, I_despot=Npool - 1))
    asym = {f"{th}_{q}": dict(k=(th - q) / (q * (1 + th)), share=(1 + q) / (1 + th),
                              k_num=k_star(1e7, q, th), share_num=1 - q * k_star(1e7, q, th))
            for th in (1.0, 2.0) for q in (0.02, 0.05, 0.10)}
    thr = {f"{th}_{q}": brentq(lambda M: th * M ** th / (1 + M ** th) - q, 1e-9, 1e3) for th in (1.0, 2.0) for q in (0.02, 0.05, 0.10)}
    return save("e12_despot", dict(rows=rows, asymptote=asym, tolerance_threshold_M=thr, Npool=Npool,
                                   max_root_vs_brute=maxdiff, log_concave=bool(concave_ok), monotone_in_M=bool(mono_ok),
                                   asym_err=max(abs(v["share"] - v["share_num"]) for v in asym.values())))

if __name__ == "__main__":
    for k in (sys.argv[1:] or ["e10", "e11", "e12"]):
        r = globals()[k]()
        print(k, {x: r[x] for x in ("max_abs_z", "aim_gain_G", "aim_loss_F", "monotone", "max_root_vs_brute", "log_concave", "monotone_in_M", "asym_err") if x in r}, flush=True)
