#!/usr/bin/env python3
"""Independent audit A1-A11.  Each proposition is re-derived by a construction
different from the one used in the experiments; this file imports no model code.
"""
import json, os, time
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(np.random.SeedSequence(20261117))
out = {}

# A1  Proposition 2: matched pairs built from the conditional Gaussian law of capacity given a shared score
VA, VE = 0.4, 0.6; VP = VA + VE; h2 = VA / VP
bN, bD, bS = 0.1, 0.1, 0.1
zs = []
for R in (0.1, 0.25, 0.5, 0.75, 0.9):
    VY = VP / R; kap = VP / VY; vres = VP - kap ** 2 * VY
    est = []
    for blk in range(40):
        n = 200000
        y = rng.normal(0, np.sqrt(VY), n)
        c = kap * y + rng.normal(0, np.sqrt(vres), n)          # focal individual
        cp = kap * y + rng.normal(0, np.sqrt(vres), n)         # partner with the same score
        a = h2 * c + rng.normal(0, np.sqrt(VA * (1 - h2)), n)
        w = np.exp(bN * c + bD * cp + bS * y / np.sqrt(VY))
        est.append(np.sum(a * w) / np.sum(w))
    exact = VA * (bN + bD * R + bS * np.sqrt(R / VP))
    zs.append((np.mean(est) - exact) / (np.std(est, ddof=1) / np.sqrt(len(est))))
out["A1_prop2_max_abs_z"] = float(np.max(np.abs(zs)))

# A2  Proposition 6: Gauss-Hermite quadrature of the tilted density
xg, wg = np.polynomial.hermite_e.hermegauss(120)
err = 0.0
for mu in (0.0, 0.7, 1.5, 2.0, 3.0):
    for c in (0.01, 0.02, 0.05):
        for R in (0.3, 0.5, 0.8):
            bS_, bN_ = 0.0566, 0.01
            k = bN_ - c * mu + bS_ * np.sqrt(R / VP)           # coefficient of x after integrating the score noise
            x = np.sqrt(VP) * xg
            f = np.exp(k * x - 0.5 * c * x ** 2)          # the linear term -c*mu*x is already inside k
            tilt = np.sum(wg * x * f) / np.sum(wg * f)
            err = max(err, abs(h2 * tilt - VA * k / (1 + c * VP)))
out["A2_prop6_max_abs_err"] = float(err)

# A3  Proposition 5: quadratic root versus fixed-point iteration; divergence where no equilibrium exists
err, div_ok = 0.0, True
for vle in (0.2, 0.4, 0.6):
    ve = 1 - vle
    for m in (0.0, 0.2, 0.44, 0.8, 1.0):
        v = vle
        for _ in range(200000):
            v = 0.5 * v * (1 + m * v / (v + ve)) + 0.5 * vle
            if v > 1e6:
                break
        if m < 1:
            root = (-(ve - vle) + np.sqrt((ve - vle) ** 2 + 4 * (1 - m) * vle * ve)) / (2 * (1 - m))
            err = max(err, abs(v - root))
        elif ve > vle:
            err = max(err, abs(v - vle * ve / (ve - vle)) / v)
        else:
            div_ok &= v > 1e3
out["A3_prop5_max_abs_err"] = float(err); out["A3_divergence_confirmed"] = bool(div_ok)

# A4  Proposition 3: explicit covariance algebra on the loading matrix
p = dict(VG=0.5, VS=0.5, tau2=0.5, sig2=4.0, rho=0.25, om2=0.1)
err = 0.0
for K in (1, 2, 5, 24):
    for n in (1, 3, 24):
        # latent vector: G, S_1..S_K, U_1..U_K, shared_1..K, event_{k,j}, Omega
        dim = 1 + 3 * K + K * n + 1
        var = np.r_[p["VG"], np.full(K, p["VS"]), np.full(K, p["tau2"]), np.full(K, p["sig2"] * p["rho"]),
                    np.full(K * n, p["sig2"] * (1 - p["rho"])), p["om2"]]
        load = np.r_[1.0, np.full(3 * K, 1 / K), np.full(K * n, 1 / (K * n)), 1.0]
        vy = np.sum(load ** 2 * var); cov = load[0] * var[0]
        rg = cov ** 2 / (p["VG"] * vy)
        ex = p["VG"] / (p["VG"] + (p["VS"] + p["tau2"] + p["sig2"] * (p["rho"] + (1 - p["rho"]) / n)) / K + p["om2"])
        err = max(err, abs(rg - ex))
out["A4_prop3_max_abs_err"] = float(err)

# A5  Proposition 8: selection differentials from explicit covariance algebra on a loading matrix
err = 0.0
for _ in range(500):
    lF, lG = rng.uniform(0, 1, 3), rng.uniform(0, 1, 3); w = rng.dirichlet(np.ones(3))
    h2F, h2G, VF, VG, vn = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), rng.uniform(0.5, 2), rng.uniform(0.5, 2), rng.uniform(0.3, 2)
    # latent vector: A_F, E_F, A_G, E_G, venue errors 1..3
    D = np.diag([h2F * VF, (1 - h2F) * VF, h2G * VG, (1 - h2G) * VG, vn, vn, vn])
    ly = np.r_[w @ lF, w @ lF, w @ lG, w @ lG, w]
    sdy = np.sqrt(ly @ D @ ly)
    for col, form in ((0, h2F * (w @ lF) * VF), (2, h2G * (w @ lG) * VG)):
        e = np.zeros(7); e[col] = 1.0
        sd_formula = np.sqrt((w @ lF) ** 2 * VF + (w @ lG) ** 2 * VG + np.sum(w ** 2) * vn)
        err = max(err, abs((e @ D @ ly) / sdy - form / sd_formula))
out["A5_prop8_max_abs_err"] = float(err)

# A6  Proposition 9: optimum number of events by direct maximisation, random parameters
ok_mono, err = True, 0.0
for _ in range(200):
    a, b, c = rng.uniform(0.5, 2), rng.uniform(0.2, 3), rng.uniform(0.5, 6)
    Rn = lambda n: a / (a + b + c / n)
    last = None
    for d in (0.05, 0.02, 0.01, 0.005):
        res = minimize_scalar(lambda n: -(np.sqrt(Rn(n)) - d * n), bounds=(1e-6, 1e5), method="bounded",
                              options=dict(xatol=1e-10))
        n1 = res.x
        foc = 0.5 * np.sqrt(Rn(n1)) * c / (n1 * ((a + b) * n1 + c)) - d
        err = max(err, abs(foc) / d)
        if last is not None:
            ok_mono &= n1 > last
        last = n1
out["A6_prop9_foc_rel_err"] = float(err); out["A6_prop9_monotone"] = bool(ok_mono)

# A7  Proposition 10: direct maximisation of the dominant's payoff, random parameters
ok_mono, err_as, err_thr = True, 0.0, 0.0
for _ in range(200):
    q = rng.uniform(0.01, 0.2); th = rng.uniform(0.5, 3.0)
    last = -1.0
    for M in (0.1, 0.5, 2.0, 10.0, 50.0, 1e6):
        Uf = lambda k: -(1 - k * q) * (1 + k) ** th / ((1 + k) ** th + M ** th)
        res = minimize_scalar(Uf, bounds=(0.0, 1 / q), method="bounded", options=dict(xatol=1e-12))
        k1 = res.x if -res.fun > -Uf(0.0) else 0.0
        ok_mono &= k1 >= last - 1e-6; last = k1
    err_as = max(err_as, abs((1 - q * last) - (1 + q) / (1 + th)))
    Mthr = (q / (th - q)) ** (1 / th) if th > q else np.inf   # theta M^theta/(1+M^theta) = q
    if np.isfinite(Mthr):
        for M, want in ((Mthr * 0.9, False), (Mthr * 1.1, True)):
            g = th * M ** th / (1 + M ** th) - q
            err_thr = max(err_thr, 0.0 if (g > 0) == want else 1.0)
out["A7_prop10_monotone"] = bool(ok_mono); out["A7_prop10_asymptote_err"] = float(err_as)
out["A7_prop10_threshold_ok"] = bool(err_thr == 0.0)

# A8  Proposition 7: equilibrium does not depend on G; a = 0 returns tissue to its cost optimum
err, err0 = 0.0, 0.0
for _ in range(20):
    L = rng.normal(size=(2, 2)); G = L @ L.T * 0.2 + 0.05 * np.eye(2)
    for a in (0.5, 0.0):
        kB, kK, be = 0.005, 0.01, 0.04
        z = np.array([3.0, 1.0])
        for _ in range(400000):
            z = z + G @ np.array([a * be - kB * z[0], be - kK * z[1]])
        if a > 0:
            err = max(err, abs(z[0] - a * be / kB), abs(z[1] - be / kK))
        else:
            err0 = max(err0, abs(z[0]))
out["A8_prop7_eq_err"] = float(err); out["A8_prop7_a0_tissue"] = float(err0)

# A9  Generalist target: covariance algebra, Monte Carlo order statistics, and dynamics with random G
VG_, VS_, hG, hS = 0.5, 0.5, 0.5, 0.5
err_aim = 0.0
for K in (1, 2, 4, 8, 16):
    load = np.concatenate(([1.0], np.full(K, 1.0 / K)))            # loadings of the mean score on (G, S_1..S_K)
    covA = load * np.concatenate(([hG * VG_], np.full(K, hS * VS_)))  # covariances of breeding values with the score
    err_aim = max(err_aim, abs(covA[0] / covA[1] - K * hG * VG_ / (hS * VS_)))
e15 = json.load(open(os.path.join(ROOT, "results", "e15_polymath.json")))
err_c = 0.0
for K in (2, 5, 10, 20):
    mx = rng.standard_normal((1_000_000, K)).max(axis=1)
    err_c = max(err_c, abs(mx.var() - e15["aim_division"][str(K)]["c_K"]))
err_share = 0.0
for _ in range(20):
    K = int(rng.integers(2, 12)); cG, cS, grad = float(rng.uniform(0.3, 2.0)), float(rng.uniform(0.05, 0.3)), 0.3
    L = rng.normal(size=(K + 1, K + 1)); Gm = (L @ L.T) * 0.005 + 0.02 * np.eye(K + 1)
    z = np.zeros(K + 1)
    cost = np.concatenate(([cG], np.full(K, cS))); push = np.concatenate(([grad], np.full(K, grad / K)))
    for _ in range(200000):
        z = z + Gm @ (push - cost * z)
    share = cG * z[0] ** 2 / (cG * z[0] ** 2 + cS * np.sum(z[1:] ** 2))
    err_share = max(err_share, abs(share - K * cS / (K * cS + cG)))
out["A9_generalist_aim_err"] = float(err_aim); out["A9_cK_mc_err"] = float(err_c); out["A9_share_err"] = float(err_share)

# A10  Present-day decomposition recomputed by a different code path (np.average / np.cov with weights)
npz = os.path.join(ROOT, "data", "gss", "gss_extract.npz")
if os.path.exists(npz) and os.path.exists(os.path.join(ROOT, "results", "gss.json")):
    D = np.load(npz); ref = json.load(open(os.path.join(ROOT, "results", "gss.json")))["decomposition"]
    age, sex, coh, ch, wt = D["age"], D["sex"], D["cohort"], D["childs"], D["wtssps"]
    ok = np.isfinite(age) & np.isfinite(sex) & np.isfinite(coh) & np.isfinite(ch) & np.isfinite(wt) & (wt > 0)
    ok &= (age >= 45) & (age <= 79) & (coh >= 1900) & (coh <= 1979)
    with np.errstate(invalid="ignore", divide="ignore"):
        inc = np.where(age <= 64, np.log(D["realrinc"]), np.nan)
    worst = 0.0
    for tname, x in (("schooling", D["educ"]), ("income", inc)):
        for sname, sv in (("men", 1.0), ("women", 2.0)):
            m = ok & (sex == sv) & np.isfinite(x)
            dec = (coh // 10)
            tot = acc = num = W = 0.0
            for d in np.unique(dec[m]):
                i = m & (dec == d)
                if i.sum() < 60:
                    continue
                w_, c_, x_ = wt[i], ch[i], x[i]
                mu = np.average(x_, weights=w_); sd = np.sqrt(np.average((x_ - mu) ** 2, weights=w_))
                z = (x_ - mu) / sd; par = c_ > 0
                if par.sum() < 5 or (~par).sum() < 1:
                    continue
                rel = c_ / np.average(c_, weights=w_)
                t_ = np.average(z * rel, weights=w_)
                a_ = np.average(z[par], weights=w_[par])
                n_ = np.cov(z[par], c_[par], aweights=w_[par], bias=True)[0, 1] / np.average(c_[par], weights=w_[par])
                sw = w_.sum(); tot += t_ * sw; acc += a_ * sw; num += n_ * sw; W += sw
            e = ref[tname][sname]
            worst = max(worst, abs(tot / W - e["total"][0]), abs(acc / W - e["access"][0]), abs(num / W - e["number"][0]),
                        abs(tot / W - acc / W - num / W))
    out["A10_gss_recompute_err"] = float(worst)
else:
    out["A10_gss_recompute_err"] = None

# A11  Capacity and delivery: Monte Carlo covariances against the closed form, and the take-off threshold by eigenvalue
VAB, VEB, VAP, VEP = 0.3, 0.3, 0.02, 0.02
bbar, pbar, sig_b, sig_p, s_ = 2.0, 0.5, 0.6, 0.15, 0.2
n = 2_000_000
AB = rng.normal(0, np.sqrt(VAB), n); Bv = AB + rng.normal(0, np.sqrt(VEB), n)
AP = rng.normal(0, np.sqrt(VAP), n); Pv = AP + rng.normal(0, np.sqrt(VEP), n)
Yb = Bv + rng.normal(0, sig_b, n); Yp = Pv + rng.normal(0, sig_p, n)
Rb = (VAB + VEB) / (VAB + VEB + sig_b ** 2); Rp = (VAP + VEP) / (VAP + VEP + sig_p ** 2)
vhat = (pbar + Rp * Yp) * (bbar + Rb * Yb)                      # product of posterior means
w = 1 + s_ * (vhat - vhat.mean())
d_b = np.mean(AB * w) / np.mean(w); d_p = np.mean(AP * w) / np.mean(w)
se_b = np.std(AB * w) / np.sqrt(n); se_p = np.std(AP * w) / np.sqrt(n)
out["A11_delivery_z"] = float(max(abs(d_b - s_ * pbar * Rb * VAB) / se_b, abs(d_p - s_ * bbar * Rp * VAP) / se_p))
bad = 0
for _ in range(200):
    cB, cP, sv, rb, rp = rng.uniform(0.05, 1.0), rng.uniform(0.05, 1.0), rng.uniform(0.05, 1.0), rng.uniform(0.1, 1), rng.uniform(0.1, 1)
    gB, gP = rng.uniform(0.05, 0.5), rng.uniform(0.05, 0.5)
    Jm = np.array([[-gB * cB, gB * sv * rb], [gP * sv * rp, -gP * cP]])
    unstable = np.max(np.linalg.eigvals(Jm).real) > 0
    bad += int(unstable != (sv * sv * rb * rp > cB * cP))
out["A11_threshold_mismatches"] = int(bad)

# A12  Tests and enforcement (Propositions 16 and 17), by constructions that share no code with exp_verify.py:
#      (a) validity for delivery under staging by exact summation over the default rate and the binomial count;
#      (b) the staging share that maximizes R_b R_p by bounded minimization; (c) the selection differentials under
#      enforcement by Monte Carlo covariances; (d) the equilibrium by solving the stationarity conditions directly,
#      and its stability by eigenvalue.
from scipy import stats as _st, optimize as _opt
qa, qb = 2.0, 6.0
qg = (np.arange(4000) + 0.5) / 4000
fq = _st.beta.pdf(qg, qa, qb); fq /= fq.sum()
Eq = float(np.sum(fq * qg)); VPq = float(np.sum(fq * (qg - Eq) ** 2)); nuq = float(np.sum(fq * qg * (1 - qg)))
worst = 0.0
for n_occ, d_ in [(40, 0.0), (40, 0.25), (40, 0.5), (40, 0.9), (10, 0.5), (200, 0.75), (200, 0.95)]:
    mu = int(round((1 - d_) * n_occ))
    k = np.arange(mu + 1)
    pmf = _st.binom.pmf(k[None, :], mu, qg[:, None])                 # rows: default rate; columns: defaults seen
    Ef = float(np.sum(fq[:, None] * pmf * k[None, :]))
    Ef2 = float(np.sum(fq[:, None] * pmf * k[None, :] ** 2))
    Efq = float(np.sum(fq[:, None] * pmf * k[None, :] * qg[:, None]))
    r2 = (Efq - Ef * Eq) ** 2 / ((Ef2 - Ef ** 2) * VPq)
    worst = max(worst, abs(r2 - VPq / (VPq + nuq / mu)))
out["A12_staging_validity_err"] = float(worst)
worst = 0.0
for n_occ in (10, 40, 200):
    A_, C_ = 4.0 / (n_occ * 1.0), nuq / (n_occ * VPq)
    neg = lambda d_: -(1 / (1 + A_ / d_)) * (1 / (1 + C_ / (1 - d_)))
    r = _opt.minimize_scalar(neg, bounds=(1e-6, 1 - 1e-6), method="bounded", options=dict(xatol=1e-10))
    u_, v_ = np.sqrt(A_ * (1 + C_)), np.sqrt(C_ * (1 + A_))
    worst = max(worst, abs(r.x - u_ / (u_ + v_)))
out["A12_staging_optimum_err"] = float(worst)
e_ = 0.35
vhat = (e_ + (1 - e_) * (pbar + Rp * Yp)) * (bbar + Rb * Yb)        # same draws as A11, with an enforced share
w = 1 + s_ * (vhat - vhat.mean())
d_b = np.mean(AB * w) / np.mean(w); d_p = np.mean(AP * w) / np.mean(w)
se_b = np.std(AB * w) / np.sqrt(n); se_p = np.std(AP * w) / np.sqrt(n)
out["A12_enforcement_z"] = float(max(abs(d_b - s_ * (e_ + (1 - e_) * pbar) * Rb * VAB) / se_b,
                                     abs(d_p - s_ * (1 - e_) * bbar * Rp * VAP) / se_p))
bad, worst, n_int = 0, 0.0, 0
for _ in range(400):
    cB, cP, sv, rb, rp, e_ = (rng.uniform(0.05, 1.0), rng.uniform(0.05, 1.0), rng.uniform(0.05, 1.0),
                              rng.uniform(0.1, 1), rng.uniform(0.1, 1), rng.uniform(0.02, 0.98))
    T_ = sv * sv * rb * rp / (cB * cP)
    gB, gP = rng.uniform(0.05, 0.5), rng.uniform(0.05, 0.5)
    Jm = np.array([[-gB * cB, gB * sv * rb * (1 - e_)], [gP * sv * (1 - e_) * rp, -gP * cP]])
    bad += int((np.max(np.linalg.eigvals(Jm).real) < 0) != (T_ * (1 - e_) ** 2 < 1))
    if T_ * (1 - e_) < 1:                                           # interior equilibrium: solve the two linear conditions
        M_ = np.array([[cB, -sv * rb * (1 - e_)], [-sv * (1 - e_) * rp, cP]])
        b_, p_ = np.linalg.solve(M_, np.array([sv * rb * e_, 0.0]))
        den = 1 - T_ * (1 - e_) ** 2
        worst = max(worst, abs(b_ - (sv * rb / cB) * e_ / den) / (sv * rb / cB), abs(p_ - T_ * e_ * (1 - e_) / den))
        n_int += 1
out["A12_enforcement_stability_mismatches"] = int(bad)
out["A12_enforcement_equilibrium_err"] = float(worst)
out["A12_enforcement_interior_cases"] = int(n_int)
worst = 0.0
for T_ in (0.2, 0.6, 0.9):
    r = _opt.minimize_scalar(lambda e_: -T_ * e_ * (1 - e_) / (1 - T_ * (1 - e_) ** 2), bounds=(1e-6, 1 - 1e-6),
                             method="bounded", options=dict(xatol=1e-10))
    worst = max(worst, abs(r.x - (1 - (1 - np.sqrt(1 - T_)) / T_)))
out["A12_enforcement_peak_err"] = float(worst)

# A13  Speed-dating results recomputed from the raw file by constructions that share no code with exp_speed.py:
#      (a) single-observer agreement as the correlation between two randomly drawn observers of the same person
#      (a different estimator from the intraclass correlation used in the experiment);
#      (b) the linear probability model with explicit chooser dummies instead of within-chooser centring;
#      (c) expected and actual yeses by direct tabulation.
sd_path = os.path.join(ROOT, "data", "speeddating", "speed_dating.csv")
sd_ref = os.path.join(ROOT, "results", "e17_speed.json")
if os.path.exists(sd_path) and os.path.exists(sd_ref):
    import csv as _csv
    with open(sd_path, encoding="latin-1") as f:
        _rows = list(_csv.reader(f))
    _h = _rows[0]
    def _col(k):
        j = _h.index(k); o = []
        for r_ in _rows[1:]:
            try:
                o.append(float(r_[j].replace(",", "")))
            except ValueError:
                o.append(np.nan)
        return np.array(o)
    _iid, _pid, _sex, _dec, _deco, _prob = (_col(k) for k in ("iid", "pid", "gender", "dec", "dec_o", "prob"))
    _ref = json.load(open(sd_ref))
    worst_r = 0.0
    for a_ in ("attr", "fun", "intel", "amb", "sinc"):
        x = _col(a_); x = np.where((x >= 0) & (x <= 10), x, np.nan)
        xc = np.full(len(x), np.nan)
        for r_ in np.unique(_iid):
            m_ = (_iid == r_) & np.isfinite(x)
            if m_.sum() >= 2:
                xc[m_] = x[m_] - x[m_].mean()
        tg = {}
        for t_, v_ in zip(_pid, xc):
            if np.isfinite(t_) and np.isfinite(v_):
                tg.setdefault(int(t_), []).append(v_)
        tg = [np.array(v_) for v_ in tg.values() if len(v_) >= 2]
        rs = []
        for _ in range(300):
            a1, a2 = [], []
            for v_ in tg:
                i_, j_ = rng.choice(len(v_), 2, replace=False)
                a1.append(v_[i_]); a2.append(v_[j_])
            rs.append(np.corrcoef(a1, a2)[0, 1])
        worst_r = max(worst_r, abs(float(np.mean(rs)) - _ref["jury"][a_]["r1"]))
    out["A13_agreement_gap"] = float(worst_r)
    names = ["attr", "fun", "intel", "amb", "sinc", "shar"]
    X = np.array([np.where((_col(k) >= 0) & (_col(k) <= 10), _col(k), np.nan) for k in names]).T
    full = np.isfinite(X).all(axis=1)
    Z = (X - X[full].mean(axis=0)) / X[full].std(axis=0)
    worst_b = 0.0
    for sv_, nm_ in ((0.0, "women"), (1.0, "men")):
        ok = full & (_sex == sv_) & np.isfinite(_dec)
        ids = _iid[ok]; uq = np.unique(ids)
        cnt = {u: int((ids == u).sum()) for u in uq}
        keep = np.array([cnt[u] >= 2 for u in ids])
        ids = ids[keep]; yy = _dec[ok][keep]; zz = Z[ok][keep]
        uq = np.unique(ids)
        Dm = (ids[:, None] == uq[None, :]).astype(float)                     # one dummy per chooser
        beta = np.linalg.lstsq(np.hstack([zz, Dm]), yy, rcond=None)[0][:6] * 100
        refc = [_ref["did"]["own"][nm_]["coef"][k][0] for k in ["attr", "fun", "intel", "amb", "sinc", "shar"]]
        worst_b = max(worst_b, float(np.max(np.abs(beta - np.array(refc)))))
    out["A13_lpm_err"] = float(worst_b)
    worst_i = 0.0
    for sv_, nm_ in ((0.0, "women"), (1.0, "men")):
        m_ = (_sex == sv_) & np.isfinite(_prob) & np.isfinite(_deco) & (_prob >= 1) & (_prob <= 10)
        worst_i = max(worst_i, abs(float(((_prob[m_] - 1) / 9).mean()) - _ref["interest"][nm_]["expected"][0]),
                      abs(float(_deco[m_].mean()) - _ref["interest"][nm_]["actual"][0]))
    out["A13_interest_err"] = float(worst_i)
else:
    out["A13_agreement_gap"] = out["A13_lpm_err"] = out["A13_interest_err"] = None

# A13d  Shared error caps a panel: with single-observer agreement r1 and error correlation c among observers, the squared
#       correlation of a k-observer mean with the truth is r1 / (r1 + (1 - r1) (c + (1 - c) / k)), with limit r1 / (r1 + c (1 - r1)).
_rng = np.random.default_rng(20261125)
worst_c = 0.0
for r1_, c_, k_ in ((0.11, 0.3, 50), (0.40, 0.1, 20), (0.26, 0.5, 80)):
    n_ = 60000
    t_ = _rng.normal(0, np.sqrt(r1_), n_)
    u_ = _rng.normal(0, np.sqrt(c_ * (1 - r1_)), n_)
    v_ = _rng.normal(0, np.sqrt((1 - c_) * (1 - r1_)), (n_, k_)).mean(axis=1)
    got_ = np.corrcoef(t_, t_ + u_ + v_)[0, 1] ** 2
    worst_c = max(worst_c, abs(got_ - r1_ / (r1_ + (1 - r1_) * (c_ + (1 - c_) / k_))))
out["A13_ceiling_err"] = float(worst_c)

# ---------------------------------------------------------------------------------------------------------------------
# A14  Who becomes an ancestor (Proposition 18): exact enumeration, no simulation.  For a sex with a share q childless and a
#      discrete family-size distribution among parents, Ne/N = kbar / (kbar - 1 + V / kbar) must equal (1 - q) / (1 + alpha).
worst14 = 0.0
for q_, sizes_, probs_ in ((0.2, [1, 2, 3, 6], [0.3, 0.4, 0.2, 0.1]), (0.6, [1, 2, 9], [0.5, 0.3, 0.2]), (0.0, [2], [1.0])):
    ks = np.array([0] + sizes_, float)
    ps = np.array([q_] + [(1 - q_) * p for p in probs_])
    kb = float((ks * ps).sum()); V = float((ps * (ks - kb) ** 2).sum())
    m_ = float((np.array(sizes_) * np.array(probs_)).sum()); vp = float((np.array(probs_) * (np.array(sizes_) - m_) ** 2).sum())
    al = (vp - m_) / m_ ** 2
    worst14 = max(worst14, abs(kb / (kb - 1 + V / kb) - (1 - q_) / (1 + al)))
out["A14_ancestors_identity"] = float(worst14)

# A15  The decomposition of the change in children per adult into marriage and within-status parts sums to the change.
_pv = os.path.join(ROOT, "results", "e19_village.json")
if os.path.exists(_pv):
    _v = json.load(open(_pv))
    out["A15_births_sum"] = float(max(abs(c["marriage"] + c["within_ever"] + c["within_never"] - c["change"])
                                     for lab in ("long", "recent") for c in _v["births"][lab].values()))
    out["A15_ne_identity_data"] = float(max(c["identity_gap"] for c in _v["ancestors"].values()))
else:
    out["A15_births_sum"] = out["A15_ne_identity_data"] = None

# A16  Cross-cultural codes recounted from the raw file by a second reader (pattern matching on lines, no csv module).
_pd = os.path.join(ROOT, "data", "dplace", "mirror", "datasets", "SCCS", "data.csv")
_pc = os.path.join(ROOT, "results", "e18_cultures.json")
if os.path.exists(_pd) and os.path.exists(_pc):
    _c = json.load(open(_pc))
    import re as _re
    seen, cnt = set(), {"SCCS740": {}, "SCCS174": {}}
    pat = _re.compile(r",(SCCS740|SCCS174),([^,]*),")
    with open(_pd, encoding="utf-8") as f:
        next(f)
        for line in f:
            mm = pat.search(line)
            if not mm or mm.group(2) in ("NA", ""):
                continue
            key = (line.split(",", 1)[0], mm.group(1))
            if key in seen:
                continue
            seen.add(key)
            cnt[mm.group(1)][mm.group(2)] = cnt[mm.group(1)].get(mm.group(2), 0) + 1
    d1 = sum(abs(cnt["SCCS740"].get(k, 0) - v) for k, v in _c["arrangement"]["women"]["counts"].items())
    d2 = sum(abs(cnt["SCCS174"].get(k, 0) - v) for k, v in _c["rape"]["frequency"]["counts"].items())
    out["A16_cultures_recount"] = int(d1 + d2)
else:
    out["A16_cultures_recount"] = None

# A17  The 175-country file: women's minus men's ideal for a partner's financial prospects, recomputed country by country
#      with a different reader and a Welch-type standardizer; the count of countries with the expected sign must agree.
_pk = os.path.join(ROOT, "data", "kowal2025", "Final_dataset.csv")
_pw = os.path.join(ROOT, "results", "e20_world.json")
if os.path.exists(_pk) and os.path.exists(_pw):
    _w = json.load(open(_pw))
    with open(_pk, encoding="utf-8", errors="replace") as f:
        head = [h.strip().strip('"') for h in f.readline().rstrip("\n").split(";")]
        ic, isx, ir = head.index("Country_live"), head.index("Sex01"), head.index("ideallt_resources")
        acc = {}
        for line in f:
            p_ = line.split(";")
            if len(p_) != len(head):          # a text answer containing the separator: skip the row
                continue
            c_, s_, r_ = p_[ic].strip('"'), p_[isx].strip('"'), p_[ir].strip('"')
            if c_ in ("NA", "") or s_ not in ("0", "1"):
                continue
            try:
                acc.setdefault(c_, ([], []))[int(s_)].append(float(r_))
            except ValueError:
                pass
    pos = tot = 0
    for c_, (m_, w_) in acc.items():
        if len(m_) >= 50 and len(w_) >= 50:
            tot += 1
            pos += int(np.mean(w_) > np.mean(m_))
    out["A17_world_countries"] = int(abs(tot - _w["ideals"]["resources"]["countries"]))
    out["A17_world_sign"] = int(abs(pos - _w["ideals"]["resources"]["positive"]))
else:
    out["A17_world_countries"] = out["A17_world_sign"] = None

out["_written"] = time.strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(ROOT, "results", "audit.json"), "w") as f:
    json.dump(out, f, indent=1)
for k, v in out.items():
    print(k, v)
