"""E15: the generalist target.

Checks four statements about what a portfolio of commonly performed domains selects:
  (a) aim: the selection differential on general capacity relative to one specialty is
      K * V_AG / V_AS when everyone is compared on the same K domains;
  (b) division of labour: when each person performs only the domain of their best specific
      talent, the ratio falls to V_AG / (c_K * V_AS), c_K = variance of the largest of K
      independent standard normal variables;
  (c) breadth buys time: observations per domain needed to reach a target validity for
      general capacity, as a function of K;
  (d) expenditure: at the evolutionary equilibrium with quadratic costs, the share of
      investment cost devoted to general capacity is K c_S / (K c_S + c_G).
Also verifies the Stein identity for a 'best performance' rule.

Run:  python3 code/exp_polymath.py
"""
import numpy as np
from scipy import integrate, stats

from common import mcse, rng_for, save

PORT = dict(VG=0.5, VS=0.5, tau2=0.5, sig2=4.0, rho=0.25, om2=0.1)
H2G, H2S = 0.5, 0.5
BETA = 0.3
SEED = 20261121


def c_max(K):
    """Variance of the maximum of K independent standard normal variables."""
    if K == 1:
        return 1.0
    f1 = lambda x: x * K * stats.norm.pdf(x) * stats.norm.cdf(x) ** (K - 1)
    f2 = lambda x: x * x * K * stats.norm.pdf(x) * stats.norm.cdf(x) ** (K - 1)
    m1 = integrate.quad(f1, -12, 12, epsabs=1e-12, epsrel=1e-12)[0]
    m2 = integrate.quad(f2, -12, 12, epsabs=1e-12, epsrel=1e-12)[0]
    return m2 - m1 * m1


def draw(rng, N, K, n):
    p = PORT
    AG = rng.normal(0, np.sqrt(H2G * p["VG"]), N)
    G = AG + rng.normal(0, np.sqrt((1 - H2G) * p["VG"]), N)
    AS = rng.normal(0, np.sqrt(H2S * p["VS"]), (N, K))
    S = AS + rng.normal(0, np.sqrt((1 - H2S) * p["VS"]), (N, K))
    U = rng.normal(0, np.sqrt(p["tau2"]), (N, K))
    ebar = rng.normal(0, np.sqrt(p["sig2"] * (p["rho"] + (1 - p["rho"]) / n)), (N, K))
    Om = rng.normal(0, np.sqrt(p["om2"]), N)
    return AG, G, AS, S, U, ebar, Om


def sdiff(a, w):
    return float(np.mean(a * w) / np.mean(w) - np.mean(a))


def RG(K, n):
    p = PORT
    return p["VG"] / (p["VG"] + (p["VS"] + p["tau2"] + p["sig2"] * (p["rho"] + (1 - p["rho"]) / n)) / K + p["om2"])


def n_star(K, target):
    """Events per domain needed for validity `target`; inf if unreachable."""
    p = PORT
    D = p["VG"] * (1 / target - 1) - p["om2"]
    floor_ = p["VS"] + p["tau2"] + p["rho"] * p["sig2"]
    denom = K * D - floor_
    return float((1 - p["rho"]) * p["sig2"] / denom) if denom > 0 else float("inf")


def aim_common(reps=20, N=300_000, n=4):
    out = {}
    for K in (1, 2, 4, 8, 16):
        rg, rs, ratio, val = [], [], [], []
        for r in range(reps):
            rng = rng_for(SEED, 1, K, r)
            AG, G, AS, S, U, ebar, Om = draw(rng, N, K, n)
            Y = (G[:, None] + S + U + ebar).mean(axis=1) + Om
            Z = (Y - Y.mean()) / Y.std()
            w = np.exp(BETA * Z)
            sg, ss = sdiff(AG, w), np.mean([sdiff(AS[:, k], w) for k in range(K)])
            rg.append(sg); rs.append(ss); ratio.append(sg / ss); val.append(np.corrcoef(G, Y)[0, 1] ** 2)
        sdY = np.sqrt(PORT["VG"] + (PORT["VS"] + PORT["tau2"] + PORT["sig2"] * (PORT["rho"] + (1 - PORT["rho"]) / n)) / K + PORT["om2"])
        out[K] = dict(S_G=[float(np.mean(rg)), mcse(rg)], S_S=[float(np.mean(rs)), mcse(rs)],
                      exact_G=BETA * H2G * PORT["VG"] / sdY, exact_S=BETA * H2S * PORT["VS"] / (K * sdY),
                      ratio=[float(np.mean(rg) / np.mean(rs)), float(np.std(ratio, ddof=1) / np.sqrt(reps))],
                      ratio_exact=K * H2G * PORT["VG"] / (H2S * PORT["VS"]),
                      validity=[float(np.mean(val)), mcse(val)], validity_exact=RG(K, n))
    return out


def aim_division(reps=20, N=400_000, n=4):
    out = {}
    for K in (1, 2, 5, 10, 20):
        ratio, sgs, sss, vs = [], [], [], []
        for r in range(reps):
            rng = rng_for(SEED, 2, K, r)
            AG, G, AS, S, U, ebar, Om = draw(rng, N, K, n)
            ks = S.argmax(axis=1)
            rows = np.arange(N)
            Sk, ASk = S[rows, ks], AS[rows, ks]
            Y = G + Sk + U[rows, 0] + ebar[rows, 0] + Om
            sg = ss = 0.0
            for k in range(K):
                m = ks == k
                Z = (Y[m] - Y[m].mean()) / Y[m].std()
                w = np.exp(BETA * Z)
                sg += sdiff(AG[m], w) * m.mean()
                ss += sdiff(ASk[m], w) * m.mean()
            sgs.append(sg); sss.append(ss); ratio.append(sg / ss)
            vs.append(np.mean([Sk[ks == k].var() for k in range(K)]))
        cK = c_max(K)
        out[K] = dict(ratio=[float(np.mean(sgs) / np.mean(sss)), float(np.std(ratio, ddof=1) / np.sqrt(reps))],
                      ratio_exact=H2G * PORT["VG"] / (H2S * cK * PORT["VS"]), c_K=cK,
                      var_S_within=[float(np.mean(vs)), mcse(vs)], var_S_exact=cK * PORT["VS"],
                      common_over_division=K * cK)
    return out


def stein(reps=20, N=400_000, n=4):
    out = {}
    for K in (2, 5, 10):
        cg, cs, vmax, vmean = [], [], [], []
        for r in range(reps):
            rng = rng_for(SEED, 3, K, r)
            AG, G, AS, S, U, ebar, Om = draw(rng, N, K, n)
            X = G[:, None] + S + U + ebar
            M = X.max(axis=1) + Om
            A = X.mean(axis=1) + Om
            cg.append(np.cov(G, M)[0, 1]); cs.append(np.cov(S[:, 0], M)[0, 1])
            vmax.append(np.corrcoef(G, M)[0, 1] ** 2); vmean.append(np.corrcoef(G, A)[0, 1] ** 2)
        out[K] = dict(cov_G=[float(np.mean(cg)), mcse(cg)], cov_G_exact=PORT["VG"],
                      cov_S=[float(np.mean(cs)), mcse(cs)], cov_S_exact=PORT["VS"] / K,
                      validity_max=[float(np.mean(vmax)), mcse(vmax)], validity_mean=[float(np.mean(vmean)), mcse(vmean)])
    return out


def breadth_time():
    Ks = list(range(1, 25))
    curves = {K: [RG(K, n) for n in range(1, 41)] for K in (1, 3, 6, 12, 24)}
    return dict(target=0.5, K=Ks, n_star=[n_star(K, 0.5) for K in Ks],
                K_min=float((PORT["VS"] + PORT["tau2"] + PORT["rho"] * PORT["sig2"]) / (PORT["VG"] * (1 / 0.5 - 1) - PORT["om2"])),
                ceiling_one=PORT["VG"] / (PORT["VG"] + PORT["VS"] + PORT["tau2"] + PORT["rho"] * PORT["sig2"] + PORT["om2"]),
                ceiling_many=PORT["VG"] / (PORT["VG"] + PORT["om2"]), curves=curves)


def evolve(K, cG, cS, rng, N=20_000, T=700, v=0.2, noise=1.0):
    """Individual-based evolution of a general investment g and K specific investments s_k."""
    g = rng.normal(0, np.sqrt(v), N)
    s = rng.normal(0, np.sqrt(v), (N, K))
    tail_g, tail_s, tail_sd = [], [], []
    for t in range(T):
        Y = (g[:, None] + s + rng.normal(0, np.sqrt(noise), (N, K))).mean(axis=1)
        sd = Y.std()
        logw = BETA * (Y - Y.mean()) / sd - 0.5 * cG * g ** 2 - 0.5 * cS * (s ** 2).sum(axis=1)
        w = np.exp(logw - logw.max())
        pr = w / w.sum()
        ma, pa = rng.choice(N, N, p=pr), rng.choice(N, N, p=pr)
        g = 0.5 * (g[ma] + g[pa]) + rng.normal(0, np.sqrt(v / 2), N)
        s = 0.5 * (s[ma] + s[pa]) + rng.normal(0, np.sqrt(v / 2), (N, K))
        if t >= T - 200:
            tail_g.append(g.mean()); tail_s.append(s.mean()); tail_sd.append(sd)
    gbar, sbar, sd = float(np.mean(tail_g)), float(np.mean(tail_s)), float(np.mean(tail_sd))
    share = cG * gbar ** 2 / (cG * gbar ** 2 + cS * K * sbar ** 2)
    return gbar, sbar, sd, share


def expenditure(reps=4):
    out = {}
    cS = 0.1
    for cG in (0.4, 0.8, 1.6):
        out[str(cG / cS)] = {}
        for K in (2, 4, 8, 16, 32):
            runs = [evolve(K, cG, cS, rng_for(SEED, 4, int(cG * 10), K, r)) for r in range(reps)]
            a = np.array(runs)
            sd = a[:, 2].mean()
            out[str(cG / cS)][K] = dict(g=[float(a[:, 0].mean()), mcse(a[:, 0])], g_exact=BETA / (cG * sd),
                                         s=[float(a[:, 1].mean()), mcse(a[:, 1])], s_exact=BETA / (K * cS * sd),
                                         share=[float(a[:, 3].mean()), mcse(a[:, 3])], share_exact=K * cS / (K * cS + cG))
    return out


if __name__ == "__main__":
    res = dict(params=dict(PORT, h2G=H2G, h2S=H2S, beta=BETA), aim_common=aim_common(), aim_division=aim_division(),
               stein=stein(), breadth=breadth_time(), expenditure=expenditure())
    save("e15_polymath", res)
    for K, v in res["aim_common"].items():
        print("common K=%2d ratio %.2f±%.2f (exact %.1f)  S_G %.4f (%.4f)  R_G %.3f (%.3f)" % (
            K, v["ratio"][0], v["ratio"][1], v["ratio_exact"], v["S_G"][0], v["exact_G"], v["validity"][0], v["validity_exact"]))
    for K, v in res["aim_division"].items():
        print("division K=%2d ratio %.3f±%.3f (exact %.3f) c_K %.4f varS %.4f (%.4f) common/division %.2f" % (
            K, v["ratio"][0], v["ratio"][1], v["ratio_exact"], v["c_K"], v["var_S_within"][0], v["var_S_exact"], v["common_over_division"]))
    for K, v in res["stein"].items():
        print("stein K=%2d covG %.4f (%.2f) covS %.4f (%.4f) R_max %.3f R_mean %.3f" % (
            K, v["cov_G"][0], v["cov_G_exact"], v["cov_S"][0], v["cov_S_exact"], v["validity_max"][0], v["validity_mean"][0]))
    b = res["breadth"]
    print("K_min %.2f; n* at K=6,8,12,16: %s" % (b["K_min"], [round(b["n_star"][k - 1], 2) for k in (6, 8, 12, 16)]))
    for r, d in res["expenditure"].items():
        print("cost ratio", r, " ".join("K=%d %.3f(%.3f) g %.3f(%.3f)" % (K, v["share"][0], v["share_exact"], v["g"][0], v["g_exact"]) for K, v in d.items()))
