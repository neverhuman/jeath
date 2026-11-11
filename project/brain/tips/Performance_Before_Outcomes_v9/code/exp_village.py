"""E19  Who becomes an ancestor, where the births went, and how thin the village has become.

General Social Survey 1972-2024 (public cumulative file; fetched by fetch_gss.py, not redistributed).
  1. ancestors : effective-size identity Ne/N = (1 - q) / (1 + alpha) by sex and cohort (completed families, ages 45-79)
  2. births    : decomposition of the change in children per adult aged 25-39 into marriage and within-status parts
  3. ideals    : ideal against actual number of children, by period and marital status
  4. village   : trends in seeing neighbours, relatives and friends, in religious attendance, hours online, and in
                 having no sexual partner; association of those ties with being partnered among adults aged 22-34

Associations only.  Run:  python3 -I code/exp_village.py
"""
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = dict(np.load(os.path.join(ROOT, "data", "gss", "gss_extract2.npz")))
SEED = 20261127
NB = 500
PERIODS = ((1972, 1984), (1985, 1999), (2000, 2011), (2012, 2018), (2021, 2024))


def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def wvar(x, w):
    m = np.sum(x * w) / np.sum(w)
    return float(np.sum(w * (x - m) ** 2) / np.sum(w))


def boot(fn, idx, rng, n=NB):
    """Percentile interval of fn(indices) over resamples of the respondents in idx."""
    vals = np.array([fn(rng.choice(idx, len(idx))) for _ in range(n)], float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def ne_parts(i):
    """Mean, variance, childless share q, overdispersion alpha among parents, and Ne/N two ways."""
    k, w = D["childs"][i], D["wtssps"][i]
    kbar, v = wmean(k, w), wvar(k, w)
    par = k > 0
    q = 1.0 - float(np.sum(w[par]) / np.sum(w))
    m, vp = wmean(k[par], w[par]), wvar(k[par], w[par])
    alpha = (vp - m) / m ** 2
    return {"n": int(len(i)), "mean": kbar, "var": v, "I": v / kbar ** 2, "q": q, "alpha": alpha,
            "ne_direct": kbar / (kbar - 1 + v / kbar), "ne_identity": (1 - q) / (1 + alpha)}


def ancestors(rng):
    age, sex, coh, ch, wt = D["age"], D["sex"], D["cohort"], D["childs"], D["wtssps"]
    ok = np.isfinite(age) & np.isfinite(sex) & np.isfinite(coh) & np.isfinite(ch) & np.isfinite(wt) & (age >= 45) & (age <= 79)
    out = {}
    for name, (a, b) in (("1900-1939", (1900, 1939)), ("1940-1959", (1940, 1959)), ("1960-1979", (1960, 1979)),
                         ("all", (1900, 1979))):
        im = np.where(ok & (sex == 1) & (coh >= a) & (coh <= b))[0]
        iw = np.where(ok & (sex == 2) & (coh >= a) & (coh <= b))[0]
        pm, pw = ne_parts(im), ne_parts(iw)
        both = np.concatenate([im, iw])

        def ratio(j):
            jm, jw = j[D["sex"][j] == 1], j[D["sex"][j] == 2]
            return ne_parts(jw)["ne_direct"] / ne_parts(jm)["ne_direct"]

        out[name] = {"men": pm, "women": pw, "ratio": pw["ne_direct"] / pm["ne_direct"], "ratio_ci": boot(ratio, both, rng),
                     "ratio_headcount": (1 - pw["q"]) / (1 - pm["q"]),
                     "identity_gap": max(abs(pm["ne_direct"] - pm["ne_identity"]), abs(pw["ne_direct"] - pw["ne_identity"]))}
    return out


def births(rng):
    """Kitagawa decomposition of the change in mean children ever born at ages 25-39 between two periods."""
    age, yr, ch, wt, ms, sex = D["age"], D["year"], D["childs"], D["wtssps"], D["marital"], D["sex"]
    ok = np.isfinite(age) & np.isfinite(ch) & np.isfinite(wt) & np.isfinite(ms) & np.isfinite(sex) & (age >= 25) & (age <= 39)

    def cell(i):
        w, k, never = wt[i], ch[i], (ms[i] == 5)
        p = float(np.sum(w[never]) / np.sum(w))
        return p, wmean(k[never], w[never]), wmean(k[~never], w[~never]), wmean(k, w), \
            float(np.sum(w[never & (k > 0)]) / np.sum(w[never])), float(np.sum(w[~never & (k > 0)]) / np.sum(w[~never]))

    def decomp(i1, i2):
        p1, n1, e1, t1, _, _ = cell(i1)
        p2, n2, e2, t2, _, _ = cell(i2)
        comp = (p2 - p1) * ((n1 + n2) / 2 - (e1 + e2) / 2)        # more people never married
        rate_e = ((1 - p1) + (1 - p2)) / 2 * (e2 - e1)            # fewer children among the ever married
        rate_n = (p1 + p2) / 2 * (n2 - n1)                        # change among the never married
        return t2 - t1, comp, rate_e, rate_n

    out = {"periods": {}}
    for a, b in PERIODS:
        for sname, sv in (("all", None), ("men", 1), ("women", 2)):
            i = np.where(ok & (yr >= a) & (yr <= b) & ((sex == sv) if sv else True))[0]
            p, n, e, t, pn, pe = cell(i)
            out["periods"].setdefault(f"{a}-{b}", {})[sname] = {"n": int(len(i)), "never_married": p, "children": t,
                                                                 "children_never": n, "children_ever": e,
                                                                 "parent_never": pn, "parent_ever": pe}
    for label, (pa, pb) in (("long", (PERIODS[0], PERIODS[-1])), ("recent", (PERIODS[2], PERIODS[-1]))):
        for sname, sv in (("all", None), ("men", 1), ("women", 2)):
            i1 = np.where(ok & (yr >= pa[0]) & (yr <= pa[1]) & ((sex == sv) if sv else True))[0]
            i2 = np.where(ok & (yr >= pb[0]) & (yr <= pb[1]) & ((sex == sv) if sv else True))[0]
            tot, comp, re_, rn = decomp(i1, i2)
            share = lambda j1, j2: decomp(j1, j2)[1] / decomp(j1, j2)[0]
            vals = np.array([share(rng.choice(i1, len(i1)), rng.choice(i2, len(i2))) for _ in range(NB)])
            out.setdefault(label, {})[sname] = {"from": f"{pa[0]}-{pa[1]}", "to": f"{pb[0]}-{pb[1]}", "change": tot,
                                                "marriage": comp, "within_ever": re_, "within_never": rn,
                                                "marriage_share": comp / tot,
                                                "marriage_share_ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}
    return out


def ideals(rng):
    age, yr, ch, wt, ms, idl = D["age"], D["year"], D["childs"], D["wtssps"], D["marital"], D["chldidel"]
    okw = np.isfinite(age) & np.isfinite(wt)
    out = {}
    for a, b in PERIODS:
        per = (yr >= a) & (yr <= b)
        ii = np.where(okw & per & np.isfinite(idl) & (idl <= 7) & (age >= 18) & (age <= 44))[0]
        ia = np.where(okw & per & np.isfinite(ch) & (age >= 40) & (age <= 49))[0]
        if len(ii) < 200:
            continue
        r = {"n_ideal": int(len(ii)), "ideal": wmean(idl[ii], wt[ii]), "ideal_ci": boot(lambda j: wmean(idl[j], wt[j]), ii, rng),
             "n_actual": int(len(ia)), "actual": wmean(ch[ia], wt[ia]), "actual_ci": boot(lambda j: wmean(ch[j], wt[j]), ia, rng)}
        r["gap"] = r["ideal"] - r["actual"]
        for nm, msk in (("never", ms == 5), ("ever", (ms != 5) & np.isfinite(ms))):
            j = np.where(okw & per & np.isfinite(idl) & (idl <= 7) & (age >= 25) & (age <= 39) & msk)[0]
            r["ideal_" + nm] = wmean(idl[j], wt[j]) if len(j) > 50 else None
            r["n_ideal_" + nm] = int(len(j))
            j2 = np.where(okw & per & np.isfinite(ch) & (age >= 40) & (age <= 49) & msk)[0]
            r["actual_" + nm] = wmean(ch[j2], wt[j2]) if len(j2) > 50 else None
        out[f"{a}-{b}"] = r
    return out


def village(rng):
    age, yr, wt, sex, ms = D["age"], D["year"], D["wtssps"], D["sex"], D["marital"]
    young = np.isfinite(age) & np.isfinite(wt) & (age >= 22) & (age <= 34)
    out = {"trend": {}}
    measures = {"neighbours_monthly": (D["socommun"], lambda x: x <= 4), "relatives_monthly": (D["socrel"], lambda x: x <= 4),
                "friends_monthly": (D["socfrend"], lambda x: x <= 4), "bar_monthly": (D["socbar"], lambda x: x <= 4),
                "worship_weekly": (D["attend"], lambda x: x >= 6), "no_partner_year": (D["partners"], lambda x: x == 0),
                "never_married": (ms, lambda x: x == 5), "no_steady_partner": (D["posslq"], lambda x: x == 4)}
    for a, b in PERIODS:
        per = (yr >= a) & (yr <= b)
        row = {}
        for nm, (x, f) in measures.items():
            for sname, sv in (("all", None), ("men", 1), ("women", 2)):
                i = np.where(young & per & np.isfinite(x) & ((sex == sv) if sv else True))[0]
                if len(i) < 150:
                    continue
                y = f(x).astype(float)
                row.setdefault(nm, {})[sname] = {"n": int(len(i)), "pct": 100 * wmean(y[i], wt[i]),
                                                 "ci": [100 * v for v in boot(lambda j: wmean(y[j], wt[j]), i, rng, 300)]}
        i = np.where(young & per & np.isfinite(D["wwwhr"]))[0]
        if len(i) >= 150:
            row["hours_online"] = {"all": {"n": int(len(i)), "mean": wmean(D["wwwhr"][i], wt[i]),
                                           "ci": boot(lambda j: wmean(D["wwwhr"][j], wt[j]), i, rng, 300)}}
        out["trend"][f"{a}-{b}"] = row

    # Association among adults aged 22-34: having a steady partner (married or steady partner) on community ties,
    # holding age, sex, schooling, income rank within year, and survey year fixed (linear probability model).
    part = np.where(np.isfinite(D["posslq"]), (D["posslq"] != 4).astype(float), np.nan)
    inc = D["realinc"].copy()
    rank = np.full(len(inc), np.nan)
    for y in np.unique(yr[np.isfinite(yr)]):
        m = (yr == y) & np.isfinite(inc)
        if m.sum() > 10:
            rank[m] = (np.argsort(np.argsort(inc[m])) + 0.5) / m.sum()
    out["assoc"] = {}
    ties = {"neighbours_monthly": (D["socommun"] <= 4), "relatives_monthly": (D["socrel"] <= 4),
            "friends_monthly": (D["socfrend"] <= 4), "worship_weekly": (D["attend"] >= 6)}
    rawv = {"neighbours_monthly": D["socommun"], "relatives_monthly": D["socrel"], "friends_monthly": D["socfrend"],
            "worship_weekly": D["attend"]}
    for nm, t in ties.items():
        m = young & np.isfinite(part) & np.isfinite(rawv[nm]) & np.isfinite(sex) & np.isfinite(D["educ"]) & np.isfinite(rank)
        i = np.where(m)[0]

        def fit(j):
            yrs = yr[j]
            uy = np.unique(yrs)
            X = np.column_stack([t[j].astype(float), age[j] - 28, (sex[j] == 2).astype(float), D["educ"][j] - 13, rank[j] - 0.5] +
                                [(yrs == u).astype(float) for u in uy])
            w = np.sqrt(wt[j])
            beta = np.linalg.lstsq(X * w[:, None], part[j] * w, rcond=None)[0]
            return 100 * beta[0]

        out["assoc"][nm] = {"n": int(len(i)), "points": fit(i), "ci": boot(fit, i, rng, 300),
                            "partnered_pct": 100 * wmean(part[i], wt[i]), "tie_pct": 100 * wmean(t[i].astype(float), wt[i])}
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    out = {"_seed": SEED, "_written": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": int(len(D["year"]))}
    out["ancestors"] = ancestors(rng)
    out["births"] = births(rng)
    out["ideals"] = ideals(rng)
    out["village"] = village(rng)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "e19_village.json"), "w"), indent=1)
    a = out["ancestors"]["all"]
    print("E19", {"ne_ratio": round(a["ratio"], 3), "q_men": round(a["men"]["q"], 3), "q_women": round(a["women"]["q"], 3),
                  "recent_marriage_share": round(out["births"]["recent"]["all"]["marriage_share"], 2)})
