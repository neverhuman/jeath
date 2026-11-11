#!/usr/bin/env python3
"""E7: rate audit, neutral benchmarks and the plausibility envelope (analytic
grids from published summary values), and E13: reanalysis of open fossil data.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # allows isolated mode (python -I)
import numpy as np
from common import save, rng_for

SLOPE, SPAN, SD0 = 0.416, 3.19, 0.10        # Gingerich (2022): ln ECV per Myr, Myr, within-population SD of ln ECV

def e7():
    total_sd = SLOPE * SPAN / SD0
    gens25 = SPAN * 1e6 / 25
    tab2 = {str(g): [total_sd / (SPAN * 1e6 / g * f) for f in (1.0, 0.5, 0.25, 0.1)] for g in (20, 25, 30)}
    tab3 = {str(h): [total_sd / np.sqrt(h * gens25 / ne) for ne in (1000, 2000, 5000, 10000, 20000)] for h in (0.3, 0.5, 0.8)}
    mut = [total_sd / np.sqrt(1e-3 * (1 - h) * gens25) for h in (0.3, 0.5, 0.8)]
    rate = total_sd / gens25
    req = {f"{hc}_{hb}": [rate / (rg * np.sqrt(hc * hb)) for rg in (0.1, 0.24, 0.5)] for hc, hb in ((0.3, 0.5), (0.5, 0.8))}
    base = rate / (0.24 * np.sqrt(0.5 * 0.8))
    avail = {str(I): [0.5 * rho * 0.19 * np.sqrt(I) for rho in (0.05, 0.1, 0.3)] for I in (0.3, 0.6, 1.0)}
    mult = {k: [x / base for x in v] for k, v in avail.items()}
    # if standing predicts reproduction equally in both sexes the factor 1/2 disappears
    phase_rate = total_sd / (1.2e6 / 25)
    return save("e7_rates", dict(total_sd=total_sd, generations=gens25, rate=rate, table2=tab2, table3=tab3,
                                 mutational_z=mut, required=req, base_required=base, available=avail,
                                 multiples=mult, phase_rate=phase_rate,
                                 max_grid_rate=max(max(v) for v in tab2.values()) * 1.25 / 1.0))

# ------------------------------------------------------------------ E13
def _ols(x, y):
    X = np.column_stack([np.ones_like(x), x]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; s2 = r @ r / (len(y) - 2); cov = s2 * np.linalg.inv(X.T @ X)
    return b, np.sqrt(np.diag(cov)), r

def _aicc(rss, n, k):
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)

def _fit_pw(t, y, bps, flat=()):
    """Continuous piecewise-linear fit with breakpoints bps (in time, increasing);
    segments listed in `flat` are constrained to zero slope."""
    nseg = len(bps) + 1
    edges = [t.min()] + list(bps) + [t.max()]
    cols = [np.ones_like(t)]
    for s in range(nseg):
        if s in flat:
            continue
        lo, hi = edges[s], edges[s + 1]
        cols.append(np.clip(t, lo, hi) - lo)        # contributes slope only inside segment s
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return float(r @ r), b, X.shape[1]

def _search(t, y, nbp, flat=(), grid=None, minw=0.2, minn=5):
    import itertools
    best = (np.inf, None, None, None)
    for bps in itertools.combinations(grid, nbp):
        e = [t.min()] + list(bps) + [t.max()]
        if any(e[i + 1] - e[i] < minw for i in range(len(e) - 1)):
            continue
        if any(((t >= e[i]) & (t <= e[i + 1])).sum() < minn for i in range(len(e) - 1)):
            continue
        rss, b, k = _fit_pw(t, y, bps, flat)
        if rss < best[0]:
            best = (rss, bps, b, k)
    return best

def e13(seed=20261119, nboot=400):
    from load_fossils import gingerich, will_full
    G = gingerich()
    age = np.array([g[2] for g in G]); y = np.log(np.array([g[3] for g in G])); t = -age
    n = len(y)
    b, se, r = _ols(t, y)
    bins = np.floor(age / 0.1).astype(int)
    ks = sorted(set(bins))
    bt = np.array([-(k + 0.5) * 0.1 for k in ks]); bm = np.array([y[bins == k].mean() for k in ks])
    bn = np.array([(bins == k).sum() for k in ks]); bs = np.array([y[bins == k].std(ddof=1) if (bins == k).sum() > 1 else np.nan for k in ks])
    bb, bse, _ = _ols(bt, bm)
    ok = bn >= 3
    sd_within = float(np.sqrt(np.nansum((bn[ok] - 1) * bs[ok] ** 2) / np.sum(bn[ok] - 1)))
    span = float(age.max() - age.min())
    rates = {}
    for nm, sl in (("specimen", b[1]), ("binned", bb[1]), ("published", SLOPE)):
        for sdn, sd in (("0.10", 0.10), ("reanalysis", sd_within)):
            rates[f"{nm}_sd{sdn}"] = sl * span / sd / (span * 1e6 / 25)
    # model comparison on specimens
    grid = np.round(np.arange(-3.0, -0.05, 0.05), 2)
    models = {}
    rss1 = float(r @ r); models["linear"] = dict(k=3, rss=rss1, aicc=_aicc(rss1, n, 3), bps=[])
    for nbp in (1, 2, 3):
        rss, bps, coef, kk = _search(t, y, nbp, (), grid)
        k = kk + nbp + 1
        models[f"pw{nbp}"] = dict(k=k, rss=rss, aicc=_aicc(rss, n, k), bps=[-x for x in bps], coef=list(coef))
    rss, bps, coef, kk = _search(t, y, 3, (0, 2), grid)
    k = kk + 3 + 1
    models["stasis_step"] = dict(k=k, rss=rss, aicc=_aicc(rss, n, k), bps=[-x for x in bps], coef=list(coef))
    # published phase boundaries (2.0, 1.5, 0.7 Ma) with stasis in phases I and III
    rss, coef, kk = _fit_pw(t, y, (-2.0, -1.5, -0.7), (0, 2))
    models["published_phases"] = dict(k=kk + 1, rss=rss, aicc=_aicc(rss, n, kk + 1), bps=[2.0, 1.5, 0.7], coef=list(coef))
    amin = min(m["aicc"] for m in models.values())
    for m in models.values():
        m["daicc"] = m["aicc"] - amin
        m["bic"] = n * np.log(m["rss"] / n) + m["k"] * np.log(n)
    bmin = min(m["bic"] for m in models.values())
    for m in models.values():
        m["dbic"] = m["bic"] - bmin
    # within-phase slopes for the free three-breakpoint model and the step model
    ss = models["stasis_step"]; lev1 = ss["coef"][0]; s2, s4 = ss["coef"][1], ss["coef"][2]
    e = [3.2] + ss["bps"] + [0.01]
    step = dict(level_I=float(np.exp(lev1)), slope_II=s2, slope_IV=s4,
                level_III=float(np.exp(lev1 + s2 * (e[1] - e[2]))),
                end=float(np.exp(lev1 + s2 * (e[1] - e[2]) + s4 * (e[3] - e[4]))),
                rate_II=s2 / 0.10 / (1e6 / 25), rate_IV=s4 / 0.10 / (1e6 / 25))
    # bootstrap of the step-model breakpoints (coarser grid for speed)
    rng = rng_for(seed)
    gridb = np.round(np.arange(-3.0, -0.05, 0.1), 2)
    bo = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        rs, bp, *_ = _search(t[idx], y[idx], 3, (0, 2), gridb)
        if bp is not None:
            bo.append([-x for x in bp])
    bo = np.array(bo)
    ci = [[float(np.percentile(bo[:, j], 2.5)), float(np.percentile(bo[:, j], 97.5))] for j in range(3)]
    # is there residual trend inside the two 'stasis' phases?
    stasis = {}
    for nm, lo, hi in (("I", e[1], 3.3), ("III", e[3], e[2])):
        msk = (age <= hi) & (age >= lo)
        b_, se_, _ = _ols(t[msk], y[msk])
        stasis[nm] = dict(n=int(msk.sum()), slope=float(b_[1]), se=float(se_[1]), lo=lo, hi=hi)
    # -------- taxon comparisons (Will et al. 2021)
    W = will_full()
    def grp(f):
        return np.array([w["ecv"] for w in W if f(w)]), np.array([w["age"] for w in W if f(w)])
    groups = {}
    for nm, f in (("neanderthal", lambda w: w["taxon"] == "Neanderthals"),
                  ("sapiens_pleistocene", lambda w: w["taxon"] == "Pleistocene Homo sapiens"),
                  ("mid_pleistocene", lambda w: w["taxon"] == "Mid-Pleistocene Homo"),
                  ("sapiens_over_100ka", lambda w: w["taxon"] == "Pleistocene Homo sapiens" and w["age"] > 0.1),
                  ("sapiens_100_30ka", lambda w: w["taxon"] == "Pleistocene Homo sapiens" and 0.03 < w["age"] <= 0.1),
                  ("sapiens_30_10ka", lambda w: w["taxon"] == "Pleistocene Homo sapiens" and w["age"] <= 0.03),
                  ("mp_europe", lambda w: w["taxon"] == "Mid-Pleistocene Homo" and w["region"] == "Europe/W Asia"),
                  ("mp_africa", lambda w: w["taxon"] == "Mid-Pleistocene Homo" and w["region"] == "Africa"),
                  ("mp_asia", lambda w: w["taxon"] == "Mid-Pleistocene Homo" and w["region"] == "E/SE Asia")):
        ev, ag = grp(f)
        groups[nm] = dict(n=int(len(ev)), mean=float(ev.mean()), sd=float(ev.std(ddof=1)), median=float(np.median(ev)),
                          age_min=float(ag.min()), age_max=float(ag.max()), sd_ln=float(np.log(ev).std(ddof=1)))
    a, _ = grp(lambda w: w["taxon"] == "Neanderthals"); c, _ = grp(lambda w: w["taxon"] == "Pleistocene Homo sapiens")
    la, lc = np.log(a), np.log(c)
    tw = (lc.mean() - la.mean()) / np.sqrt(la.var(ddof=1) / len(la) + lc.var(ddof=1) / len(lc))
    df = (la.var(ddof=1) / len(la) + lc.var(ddof=1) / len(lc)) ** 2 / ((la.var(ddof=1) / len(la)) ** 2 / (len(la) - 1) + (lc.var(ddof=1) / len(lc)) ** 2 / (len(lc) - 1))
    from scipy import stats
    welch = dict(diff_cc=float(c.mean() - a.mean()), ratio=float(np.exp(lc.mean() - la.mean())), t=float(tw), df=float(df),
                 p=float(2 * stats.t.sf(abs(tw), df)), cohen_d=float((lc.mean() - la.mean()) / np.sqrt(0.5 * (la.var(ddof=1) + lc.var(ddof=1)))))
    # contemporaneous comparison: both groups restricted to the Neanderthal time range
    c2, _ = grp(lambda w: w["taxon"] == "Pleistocene Homo sapiens" and 0.039 <= w["age"] <= 0.25)
    welch["sapiens_same_window_n"] = int(len(c2)); welch["sapiens_same_window_mean"] = float(c2.mean()) if len(c2) else None
    # regional series: slope of ln ECV on time (per Myr)
    series = {}
    for nm, f in (("europe_w_asia", lambda w: w["region"] == "Europe/W Asia" and w["taxon"] != "Pleistocene Homo sapiens"),
                  ("africa_to_sapiens", lambda w: (w["taxon"] == "Mid-Pleistocene Homo" and w["region"] == "Africa") or w["taxon"] == "Pleistocene Homo sapiens"),
                  ("east_asia_archaic", lambda w: w["taxon"] == "Mid-Pleistocene Homo" and w["region"] == "E/SE Asia")):
        ev, ag = grp(f)
        b_, se_, _ = _ols(-ag, np.log(ev))
        bs_ = []
        for _ in range(2000):
            ix = rng.integers(0, len(ev), len(ev))
            if np.ptp(ag[ix]) > 0:
                bs_.append(_ols(-ag[ix], np.log(ev[ix]))[0][1])
        series[nm] = dict(n=int(len(ev)), slope=float(b_[1]), se=float(se_[1]), lo=float(np.percentile(bs_, 2.5)),
                          hi=float(np.percentile(bs_, 97.5)), age_min=float(ag.min()), age_max=float(ag.max()),
                          haldanes=float(b_[1] / 0.10 / (1e6 / 25)))
    return save("e13_fossils", dict(n=n, span=span, slope_specimen=float(b[1]), se_specimen=float(se[1]),
                                    slope_binned=float(bb[1]), se_binned=float(bse[1]), nbins=int(len(ks)),
                                    resid_sd=float(r.std(ddof=2)), sd_within=sd_within, rates=rates, models=models,
                                    step=step, bp_ci=ci, nboot=int(len(bo)), stasis=stasis, groups=groups, welch=welch,
                                    series=series, bins=dict(t=(-bt).tolist(), mean=np.exp(bm).tolist(), n=bn.tolist()),
                                    seed=seed))

if __name__ == "__main__":
    for k in (sys.argv[1:] or ["e7", "e13"]):
        r = globals()[k]()
        print(k, "done", {x: r[x] for x in ("rate", "slope_specimen", "slope_binned", "sd_within") if x in r})
