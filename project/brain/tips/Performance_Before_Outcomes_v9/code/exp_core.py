#!/usr/bin/env python3
"""Experiments E1-E6 and E9: Monte Carlo and individual-based checks of
Propositions 1-7.  Implemented directly from the equations in the paper.  Parameters are dimensionless illustrations.
"""
import sys
import numpy as np
from common import rng_for, save, mcse, zscore, wmean

VA, VE = 0.4, 0.6
VP = VA + VE

def draw_people(rng, n, va=VA, ve=VE, mu=0.0):
    a = rng.normal(mu, np.sqrt(va), n)
    c = a + rng.normal(0, np.sqrt(ve), n)
    return a, c

def score(rng, c, R, vp=VP):
    """Score with target validity R = Corr(C,Y)^2; R=0 gives pure noise."""
    if R >= 1.0:
        return c.copy()
    if R <= 0.0:
        return rng.normal(0, 1, c.size)
    return c + rng.normal(0, np.sqrt(vp * (1 - R) / R), c.size)

def zstd(y):
    return (y - y.mean()) / y.std()

# ---------------------------------------------------------------- E1
def e1(seed=20261109, reps=20, n=20000, b=0.2, sig2=4.0):
    """Pair-output model: W = exp(b (Cf+Cm)/2), rank matching on a
    score averaged over n events with shared error fraction rho."""
    out = []
    conds = [(nn, rho) for rho in (0.0, 0.25, 0.5) for nn in (1, 2, 4, 8, 16, 32)]
    for ci, (nn, rho) in enumerate(conds + [(None, None)]):
        vals = []
        for r in range(reps):
            rng = rng_for(seed, ci, r)
            af, cf = draw_people(rng, n); am, cm = draw_people(rng, n)
            if nn is None:
                R = 0.0; i, j = rng.permutation(n), rng.permutation(n)
            else:
                ev = sig2 * (rho + (1 - rho) / nn)
                R = VP / (VP + ev)
                yf = cf + rng.normal(0, np.sqrt(ev), n); ym = cm + rng.normal(0, np.sqrt(ev), n)
                i, j = np.argsort(yf), np.argsort(ym)
            w = np.exp(b * (cf[i] + cm[j]) / 2)
            mid = (af[i] + am[j]) / 2
            vals.append(wmean(mid, w) - mid.mean())
        exact = b * VA * (1 + R) / 2
        out.append(dict(n=nn, rho=rho, R=R, exact=exact, mc=float(np.mean(vals)), mcse=mcse(vals)))
    md = max(abs(o["mc"] - o["exact"]) for o in out)
    return save("e1_baseline", dict(seed=seed, reps=reps, pairs=n, b=b, rows=out, max_abs_diff=md))

# ---------------------------------------------------------------- E2
def e2(seed=20261111, reps=20, n=20000):
    """Three routes (Proposition 2).  Scenarios: pair output only; pair output plus
    standing; standing in men only without assortment."""
    Rs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    scen = {
        "pair": dict(f=(0.1, 0.1, 0.0), m=(0.1, 0.1, 0.0), assort=True),
        "pair_standing": dict(f=(0.1, 0.1, 0.1), m=(0.1, 0.1, 0.1), assort=True),
        "men_standing": dict(f=(0.1, 0.0, 0.0), m=(0.1, 0.0, 0.2), assort=False),
    }
    rows, zs = [], []
    for si, (name, sc) in enumerate(scen.items()):
        for ri, R in enumerate(Rs):
            vals = []
            for r in range(reps):
                rng = rng_for(seed, si, ri, r)
                af, cf = draw_people(rng, n); am, cm = draw_people(rng, n)
                yf, ym = score(rng, cf, R), score(rng, cm, R)
                zf, zm = zstd(yf), zstd(ym)
                if sc["assort"]:
                    i, j = np.argsort(yf), np.argsort(ym)
                else:
                    i, j = rng.permutation(n), rng.permutation(n)
                S = []
                for (bN, bD, bS), a_, c_, cp_, z_ in ((sc["f"], af[i], cf[i], cm[j], zf[i]),
                                                     (sc["m"], am[j], cm[j], cf[i], zm[j])):
                    w = np.exp(bN * c_ + bD * cp_ + bS * z_)
                    S.append(wmean(a_, w) - a_.mean())
                vals.append(0.5 * (S[0] + S[1]))
            def Sx(bN, bD, bS, assort):
                return VA * (bN + (bD * R if assort else 0.0) + bS * np.sqrt(R / VP))
            exact = 0.5 * (Sx(*sc["f"], sc["assort"]) + Sx(*sc["m"], sc["assort"]))
            z = zscore(vals, exact); zs.append(abs(z))
            rows.append(dict(scenario=name, R=R, exact=exact, mc=float(np.mean(vals)), mcse=mcse(vals), z=z))
    return save("e2_routes", dict(seed=seed, reps=reps, per_sex=n, rows=rows, max_abs_z=max(zs)))

# ---------------------------------------------------------------- E3
PORT = dict(VG=0.5, VS=0.5, tau2=0.5, sig2=4.0, rho=0.25, om2=0.1)

def RG_exact(K, n, p=PORT):
    return p["VG"] / (p["VG"] + (p["VS"] + p["tau2"] + p["sig2"] * (p["rho"] + (1 - p["rho"]) / n)) / K + p["om2"])

def VY_exact(K, n, p=PORT):
    return p["VG"] + (p["VS"] + p["tau2"] + p["sig2"] * (p["rho"] + (1 - p["rho"]) / n)) / K + p["om2"]

def sim_portfolio(rng, N, K, n, h2=0.5, p=PORT):
    aG = rng.normal(0, np.sqrt(h2 * p["VG"]), N); G = aG + rng.normal(0, np.sqrt((1 - h2) * p["VG"]), N)
    aS = rng.normal(0, np.sqrt(h2 * p["VS"]), (N, K)); S = aS + rng.normal(0, np.sqrt((1 - h2) * p["VS"]), (N, K))
    aO = rng.normal(0, np.sqrt(h2 * p["om2"]), N); Om = aO + rng.normal(0, np.sqrt((1 - h2) * p["om2"]), N)
    U = rng.normal(0, np.sqrt(p["tau2"]), (N, K))
    shared = rng.normal(0, np.sqrt(p["sig2"] * p["rho"]), (N, K))
    ebar = np.zeros((N, K))
    for _ in range(n):  # explicit events, accumulated to limit memory
        ebar += rng.normal(0, np.sqrt(p["sig2"] * (1 - p["rho"])), (N, K))
    ebar = shared + ebar / n
    Y = (G[:, None] + S + U + ebar).mean(axis=1) + Om
    return G, aG, aS[:, 0], aO, Y

def e3(seed=20261112, reps=20, N=100000, bS=0.2, h2=0.5):
    rows_v, rows_s, zs = [], [], []
    for ti, T in enumerate((1, 2, 4, 8, 16, 32, 64)):
        for mode, (K, n) in (("diverse", (T, 1)), ("repeat", (1, T))):
            v = []
            for r in range(reps):
                rng = rng_for(seed, 0, ti, 0 if mode == "diverse" else 1, r)
                G, *_ , Y = sim_portfolio(rng, 40000, K, n)
                v.append(np.corrcoef(G, Y)[0, 1] ** 2)
            ex = RG_exact(K, n); z = zscore(v, ex); zs.append(abs(z))
            rows_v.append(dict(T=T, mode=mode, K=K, n=n, exact=ex, mc=float(np.mean(v)), mcse=mcse(v), z=z))
    for si, (K, n) in enumerate(((1, 24), (2, 12), (3, 8), (4, 6), (6, 4), (8, 3), (12, 2), (24, 1))):
        acc = {"G": [], "S": [], "O": []}
        for r in range(reps):
            rng = rng_for(seed, 1, si, r)
            G, aG, aS1, aO, Y = sim_portfolio(rng, N, K, n, h2)
            w = np.exp(bS * (Y - Y.mean()) / Y.std())
            acc["G"].append(wmean(aG, w) - aG.mean()); acc["S"].append(wmean(aS1, w) - aS1.mean())
            acc["O"].append(wmean(aO, w) - aO.mean())
        sd = np.sqrt(VY_exact(K, n))
        ex = dict(G=bS * h2 * PORT["VG"] / sd, S=bS * h2 * PORT["VS"] / (K * sd), O=bS * h2 * PORT["om2"] / sd)
        row = dict(K=K, n=n, RG=RG_exact(K, n))
        for k in acc:
            z = zscore(acc[k], ex[k]); zs.append(abs(z))
            row.update({"ex_" + k: ex[k], "mc_" + k: float(np.mean(acc[k])), "se_" + k: mcse(acc[k]), "z_" + k: z})
        rows_s.append(row)
    p = PORT
    ceil_single = p["VG"] / (p["VG"] + p["VS"] + p["tau2"] + p["rho"] * p["sig2"] + p["om2"])
    ceil_port = p["VG"] / (p["VG"] + p["om2"])
    return save("e3_portfolio", dict(seed=seed, reps=reps, N=N, bS=bS, h2=h2, params=PORT, validity=rows_v,
                                     selection=rows_s, ceiling_single=ceil_single, ceiling_portfolio=ceil_port,
                                     max_abs_z=max(zs), n_comparisons=len(zs)))

# ---------------------------------------------------------------- E4
def vstar(m, vle, ve):
    if m >= 1.0:
        return vle * ve / (ve - vle) if ve > vle else float("inf")
    a, b, c = (1 - m), (ve - vle), -vle * ve
    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

def e4(seed=20261113, reps=10, N=5000, gens=150, b=0.05, vle=0.4, ve=0.6):
    """Assortment across generations (Proposition 5): individual-based infinitesimal
    model versus the deterministic recursion."""
    conds = [("random", None), ("R0=0.40", 0.40), ("R0=0.67", 2 / 3), ("R0=0.91", 10 / 11), ("R0=1", 1.0)]
    out = []
    for ci, (name, R0) in enumerate(conds):
        nu = None if R0 is None else (vle + ve) * (1 - R0) / R0
        M = np.zeros((reps, gens + 1)); V = np.zeros((reps, gens + 1))
        for r in range(reps):
            rng = rng_for(seed, ci, r)
            af = rng.normal(0, np.sqrt(vle), N); am = rng.normal(0, np.sqrt(vle), N)
            for t in range(gens + 1):
                M[r, t] = 0.5 * (af.mean() + am.mean()); V[r, t] = 0.5 * (af.var() + am.var())
                if t == gens:
                    break
                cf = af + rng.normal(0, np.sqrt(ve), N); cm = am + rng.normal(0, np.sqrt(ve), N)
                if R0 is None:
                    i, j = rng.permutation(N), rng.permutation(N)
                else:
                    yf = cf + (rng.normal(0, np.sqrt(nu), N) if nu > 0 else 0.0)
                    ym = cm + (rng.normal(0, np.sqrt(nu), N) if nu > 0 else 0.0)
                    i, j = np.argsort(yf), np.argsort(ym)
                w = np.exp(b * (cf[i] + cm[j]) / 2); pw = w / w.sum()
                mid = (af[i] + am[j]) / 2
                kf = rng.choice(N, N, p=pw); km = rng.choice(N, N, p=pw)
                af = mid[kf] + rng.normal(0, np.sqrt(vle / 2), N)
                am = mid[km] + rng.normal(0, np.sqrt(vle / 2), N)
        # deterministic recursion
        va, mu = vle, 0.0; dm = [mu]; dv = [va]
        for t in range(gens):
            vp = va + ve
            m = 0.0 if R0 is None else vp / (vp + nu)
            mu += 0.5 * b * va * (1 + m)
            va = 0.5 * va * (1 + m * va / vp) + 0.5 * vle
            dm.append(mu); dv.append(va)
        # equilibrium: solve jointly for V* and m (validity changes with VP when noise is fixed)
        if R0 is None:
            veq, meq = vle, 0.0
        else:
            veq = vle
            for _ in range(20000):
                meq = (veq + ve) / (veq + ve + nu)
                new = vstar(meq, vle, ve)
                if not np.isfinite(new):
                    veq = new; break
                if abs(new - veq) < 1e-13:
                    veq = new; break
                veq = 0.5 * (veq + new)
        F = veq / vle * (1 + meq) if np.isfinite(veq) else float("inf")
        out.append(dict(name=name, R0=R0, mean=M.mean(0), mean_sd=M.std(0, ddof=1), va=V.mean(0),
                        det_mean=dm, det_va=dv, final_mean=float(M[:, -1].mean()), final_va=float(V[:, -10:].mean()),
                        det_final_mean=dm[-1], det_final_va=dv[-1], v_eq=veq, m_eq=meq, F_eq=F))
    base = out[0]["final_mean"]
    for o in out:
        o["ratio_vs_random"] = o["final_mean"] / base
        o["rel_err_mean"] = abs(o["final_mean"] / o["det_final_mean"] - 1)
        o["rel_err_va"] = abs(o["final_va"] / o["det_final_va"] - 1)
    # table of F(m) for fixed m
    tab = {str(h): [vstar(m, h, 1 - h) / h * (1 + m) for m in (0.2, 0.4, 0.6, 0.8, 1.0)] for h in (0.2, 0.4, 0.6, 0.8)}
    F044 = [vstar(0.44, h, 1 - h) / h * 1.44 for h in (0.4, 0.5)]
    return save("e4_multigen", dict(seed=seed, reps=reps, N=N, gens=gens, b=b, vle=vle, ve=ve, conds=out,
                                    F_table=tab, F_m044=F044,
                                    max_rel_err=max(max(o["rel_err_mean"], o["rel_err_va"]) for o in out)))

# ---------------------------------------------------------------- E5
def deferred_acceptance(pm, pw):
    """Men-proposing deferred acceptance, vectorised by rounds.
    pm[i] = women ordered from most to least preferred by man i.
    pw[j, i] = rank that woman j gives man i (lower is better)."""
    n = pm.shape[0]
    nxt = np.zeros(n, int); wife = -np.ones(n, int); husb = -np.ones(n, int)
    free = np.arange(n)
    while free.size:
        prop = pm[free, nxt[free]]; nxt[free] += 1
        # each proposed-to woman keeps the best of (current partner, proposers)
        order = np.lexsort((pw[prop, free], prop))
        pr, fr = prop[order], free[order]
        first = np.r_[True, pr[1:] != pr[:-1]]
        bw, bm = pr[first], fr[first]                  # best proposer for each woman
        cur = husb[bw]
        better = (cur < 0) | (pw[bw, bm] < pw[bw, np.where(cur < 0, 0, cur)])
        dumped = cur[better & (cur >= 0)]
        wife[dumped] = -1
        husb[bw[better]] = bm[better]; wife[bm[better]] = bw[better]
        free = np.flatnonzero(wife < 0)
    return wife

def e5(seed=20261114, reps=15, n=1000, R=0.5, b=0.2):
    rows = []
    for si, s in enumerate((0.0, 0.25, 0.5, 1.0, 2.0)):
        ms, resp, pred = [], [], []
        for r in range(reps):
            rng = rng_for(seed, 0, si, r)
            af, cf = draw_people(rng, n); am, cm = draw_people(rng, n)
            zf, zm = zstd(score(rng, cf, R)), zstd(score(rng, cm, R))
            um = zf[None, :] + s * rng.normal(size=(n, n))      # men's utilities over women
            uw = zm[None, :] + s * rng.normal(size=(n, n))      # women's utilities over men
            pm = np.argsort(-um, axis=1)
            pw = np.argsort(np.argsort(-uw, axis=1), axis=1)
            wife = deferred_acceptance(pm, pw)
            cfm = cf[wife]; afm = af[wife]
            m = np.corrcoef(cfm, cm)[0, 1]
            w = np.exp(b * (cfm + cm) / 2); mid = (afm + am) / 2
            ms.append(m); resp.append(wmean(mid, w) - mid.mean()); pred.append(b * VA * (1 + m) / 2)
        rows.append(dict(s=s, m=float(np.mean(ms)), m_se=mcse(ms), resp=float(np.mean(resp)), resp_se=mcse(resp),
                         pred=float(np.mean(pred))))
    hid = []
    N = 200000
    for qi, q in enumerate((0.1, 0.2, 0.4)):
        rng = rng_for(seed, 1, qi)
        _, c = draw_people(rng, N); y = score(rng, c, R)
        hide = rng.random(N) < q
        yr = np.where(hide, y[~hide].mean(), y)
        thr = np.quantile(y, q); low = y < thr
        yt = np.where(low, y[low].mean(), y)
        hid.append(dict(q=q, naive=(1 - q) * R, random=float(np.corrcoef(c, yr)[0, 1] ** 2),
                        threshold=float(np.corrcoef(c, yt)[0, 1] ** 2)))
    return save("e5_matching", dict(seed=seed, reps=reps, per_sex=n, R=R, b=b, matching=rows, hiding=hid,
                                    max_resp_diff=max(abs(x["resp"] - x["pred"]) for x in rows)))

# ---------------------------------------------------------------- E6
def e6(seed=20261115, reps=6, N=5000, gens=1500, c=0.02, vle=0.4, ve=0.6, nu=1.0):
    """Positional persistence (Proposition 6) in a finite population, and the
    deterministic tissue dynamics of Proposition 7."""
    bS = 0.04 / np.sqrt(0.5)
    M = np.zeros((reps, gens + 1)); eq = []
    for r in range(reps):
        rng = rng_for(seed, 0, r)
        a = rng.normal(0, np.sqrt(vle), 2 * N)
        Rr, Vr = [], []
        for t in range(gens + 1):
            M[r, t] = a.mean()
            if t == gens:
                break
            cph = a + rng.normal(0, np.sqrt(ve), 2 * N)
            y = cph + rng.normal(0, np.sqrt(nu), 2 * N)
            z = (y - y.mean()) / y.std()
            if t >= gens - 300:
                Rr.append(np.corrcoef(cph, y)[0, 1] ** 2); Vr.append(cph.var())
            w = np.exp(bS * z - 0.5 * c * cph ** 2)
            w[:N] /= w[:N].sum(); w[N:] /= w[N:].sum()
            mo = rng.choice(N, 2 * N, p=w[:N]); fa = N + rng.choice(N, 2 * N, p=w[N:])
            a = 0.5 * (a[mo] + a[fa]) + rng.normal(0, np.sqrt(vle / 2), 2 * N)
        eq.append(dict(sim=float(M[r, -300:].mean()), pred=float(bS * np.sqrt(np.mean(Rr) / np.mean(Vr)) / c),
                       R=float(np.mean(Rr)), VP=float(np.mean(Vr))))
    # recursion of Proposition 6 with freely evolving variance (deterministic)
    va, mu = vle, 0.0; det = [mu]
    for t in range(gens):
        vp = va + ve; R = vp / (vp + nu)
        mu += va * (bS * np.sqrt(R / vp) - c * mu) / (1 + c * vp)
        vsel = va - c * va * va / (1 + c * vp)          # variance among selected parents
        va = 0.5 * vsel + 0.5 * vle
        det.append(mu)
    # Proposition 7: deterministic bivariate dynamics
    G = np.diag([0.7, 0.4]); a_c = 0.5; kK = 0.01; beff = 0.04; kap = 1.0
    phases = [0.02, 0.01, 0.005, 0.0025, 0.0025]; scale = [1, 1, 1, 1, 0.5]; L = 3000
    traj = {}
    for kind in ("positional", "ecological", "both"):
        z = np.zeros(2); tr = [z.copy()]
        for kB, sc in zip(phases, scale):
            for t in range(L):
                C = a_c * z[0] + z[1]
                gp = sc * beff
                ge = beff * np.exp(-C / kap)
                g = gp if kind == "positional" else ge if kind == "ecological" else 0.5 * (gp + ge)
                beta = np.array([a_c * g - kB * z[0], g - kK * z[1]])
                z = z + G @ beta
                tr.append(z.copy())
        traj[kind] = np.array(tr)
    ends = {k: [float(v[(i + 1) * L, 0]) for i in range(5)] for k, v in traj.items()}
    # approach time for tissue at the smallest cost: 1/(G_BB kappa_B)
    return save("e6_persistence", dict(seed=seed, reps=reps, N=N, gens=gens, c=c, bS=bS, nu=nu,
                                       sim_mean=M.mean(0), det=det, eq=eq,
                                       eq_sim_mean=float(np.mean([e["sim"] for e in eq])),
                                       eq_sim_sd=float(np.std([e["sim"] for e in eq], ddof=1)),
                                       eq_pred=float(np.mean([e["pred"] for e in eq])),
                                       tissue_positional=traj["positional"][::20, 0], tissue_ecological=traj["ecological"][::20, 0],
                                       tissue_both=traj["both"][::20, 0], org_positional=traj["positional"][::20, 1],
                                       phase_len=L, kappaB=phases, phase_ends=ends,
                                       approach_time=[1 / (0.7 * k) for k in phases[:4]]))

# ---------------------------------------------------------------- E9
def allocation(name, z, u):
    if name == "exponential":
        return np.exp(0.3 * z)
    if name == "percentile":
        return 1.0 + 0.5 * (2 * u - 1)
    if name == "truncation":
        return (u > 0.5).astype(float)
    if name == "top_sixth_x4":
        return np.where(z > 1.0, 4.0, 1.0)
    raise ValueError(name)

def e9(seed=20261118, reps=20, N=200000):
    """Information-opportunity identity (Proposition 1) for four allocation shapes."""
    from math import erf, sqrt, pi, exp
    Rs = [0.1, 0.25, 0.5, 0.75, 1.0]
    names = ["exponential", "percentile", "truncation", "top_sixth_x4"]
    phi = lambda x: exp(-x * x / 2) / sqrt(2 * pi); Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    # exact intensity i = Cov(Z,W)/E[W] for a standard normal Z
    i_exact = dict(exponential=0.3, percentile=0.5 / sqrt(pi), truncation=phi(0) / 0.5,
                   top_sixth_x4=3 * phi(1.0) / (1 + 3 * (1 - Phi(1.0))))
    rows, zs = [], []
    for ai, nm in enumerate(names):
        for ri, R in enumerate(Rs):
            resp, cv = [], []
            for r in range(reps):
                rng = rng_for(seed, 0, ai, ri, r)
                a, c = draw_people(rng, N); y = score(rng, c, R)
                z = zstd(y); u = (np.argsort(np.argsort(y)) + 0.5) / N
                w = allocation(nm, z, u)
                resp.append(wmean(a, w) - a.mean()); cv.append(w.std() / w.mean())
            exact = np.sqrt(VA) * np.sqrt(VA / VP) * np.sqrt(R) * i_exact[nm]
            zval = zscore(resp, exact); zs.append(abs(zval))
            rows.append(dict(rule=nm, R=R, resp=float(np.mean(resp)), se=mcse(resp), exact=exact, z=zval,
                             cv=float(np.mean(cv)), resp_over_sqrtR=float(np.mean(resp) / np.sqrt(R))))
    # Cauchy-Schwarz efficiency of the percentile rule: Corr(Z, W)
    eff = (0.5 / np.sqrt(np.pi)) / (0.5 / np.sqrt(3))
    # heavy tails: Laplace breeding values, environments and errors
    lap, viol = [], 0
    for ai, nm in enumerate(names):
        for ri, R in enumerate(Rs):
            rng = rng_for(seed, 1, ai, ri)
            Nl = 1000000
            a = rng.laplace(0, np.sqrt(VA / 2), Nl); c = a + rng.laplace(0, np.sqrt(VE / 2), Nl)
            y = c if R >= 1 else c + rng.laplace(0, np.sqrt(VP * (1 - R) / R / 2), Nl)
            z = zstd(y); order = np.argsort(y); u = np.empty(Nl); u[order] = (np.arange(Nl) + 0.5) / Nl
            w = allocation(nm, z, u)
            S = wmean(a, w) - a.mean()
            lin = np.sqrt(VA) * np.sqrt(VA / VP) * np.sqrt(R) * (np.cov(z, w)[0, 1] / w.mean())
            # bound from a binned conditional expectation E[A|Y]
            nb = 400; idx = np.minimum((u * nb).astype(int), nb - 1)
            cm = np.bincount(idx, a, nb) / np.bincount(idx, minlength=nb)
            bound = np.sqrt(np.average((cm - a.mean()) ** 2)) * (w.std() / w.mean())
            # within-bin variation of W adds nothing to Cov(A,W) beyond the bin means up to O(1/nb)
            if abs(S) > bound * (1 + 2e-3):
                viol += 1
            lap.append(dict(rule=nm, R=R, S=float(S), linear=float(lin), rel_err=float(abs(lin / S - 1)),
                            bound=float(bound)))
    worst = max(lap, key=lambda d: d["rel_err"])
    # pair-output model with Var(ln W) held fixed
    b1 = 0.2; fixed = []
    for R in (0.0, 1.0):
        bR = b1 * np.sqrt(2 / (1 + R))
        vals = []
        for r in range(reps):
            rng = rng_for(seed, 2, int(R), r)
            n = 100000
            af, cf = draw_people(rng, n); am, cm = draw_people(rng, n)
            if R == 0:
                i, j = rng.permutation(n), rng.permutation(n)
            else:
                i, j = np.argsort(cf), np.argsort(cm)
            w = np.exp(bR * (cf[i] + cm[j]) / 2); mid = (af[i] + am[j]) / 2
            vals.append(wmean(mid, w) - mid.mean())
        fixed.append(dict(R=R, b=bR, mc=float(np.mean(vals)), se=mcse(vals), exact=bR * VA * (1 + R) / 2))
    return save("e9_identity", dict(seed=seed, reps=reps, N=N, rows=rows, max_abs_z=max(zs), n_comparisons=len(zs),
                                    percentile_efficiency=eff, laplace=lap, laplace_worst=worst,
                                    laplace_bound_violations=viol, fixed_opportunity=fixed,
                                    fixed_ratio=fixed[1]["mc"] / fixed[0]["mc"]))

if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e2", "e3", "e4", "e5", "e6", "e9"]
    for k in which:
        r = globals()[k]()
        keys = [x for x in ("max_abs_diff", "max_abs_z", "max_rel_err", "max_resp_diff", "eq_sim_mean", "eq_pred", "fixed_ratio") if x in r]
        print(k, {x: (round(r[x], 5) if isinstance(r[x], float) else r[x]) for x in keys}, flush=True)
