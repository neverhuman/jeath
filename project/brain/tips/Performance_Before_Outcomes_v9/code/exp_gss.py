"""E14: where selection sits in a contemporary population.

Decomposes the selection differential on four measured traits into an access component
(becoming a parent at all) and a number component (family size among parents), by sex and
birth cohort, in the US General Social Survey 1972-2024. Also: the share of the opportunity
for selection that is due to childlessness, children by lifetime number of sexual partners,
partnerlessness by income, and spousal resemblance in schooling.

Identity (exact, within any cohort cell, with relative fitness w = c / mean(c)):
    S = Cov(z, w) = [mean(z | parent) - mean(z)]  +  Cov(z, c | parent) / mean(c | parent)
                     access ("floor")               number ("ladder")
and for the opportunity for selection, with p the fraction who are parents,
    I = Var(w) = (1 - p)/p + I_parents / p.

Run with:  python3 -I code/exp_gss.py     (reads third-party data, so isolated mode)
Requires data/gss/gss_extract.npz, written by code/fetch_gss.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import save

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = dict(np.load(os.path.join(ROOT, "data", "gss", "gss_extract.npz")))
NBOOT = 2000
SEED = 20261110
SEXES = {"men": 1.0, "women": 2.0}
COHORTS = [(1900, 1919), (1920, 1939), (1940, 1959), (1960, 1979)]


def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def wcov(x, y, w):
    mx, my = wmean(x, w), wmean(y, w)
    return float(np.sum(w * (x - mx) * (y - my)) / np.sum(w))


def wcorr(x, y, w):
    return wcov(x, y, w) / np.sqrt(wcov(x, x, w) * wcov(y, y, w))


def base_frame():
    """Respondents aged 45-79 with known sex, cohort, number of children and weight."""
    age, sex, coh, ch, wt = D["age"], D["sex"], D["cohort"], D["childs"], D["wtssps"]
    ok = np.isfinite(age) & np.isfinite(sex) & np.isfinite(coh) & np.isfinite(ch) & np.isfinite(wt) & (wt > 0)
    ok &= (age >= 45) & (age <= 79) & (coh >= 1900) & (coh <= 1979)
    return ok


def trait(name):
    if name == "income":
        x = np.log(D["realrinc"])
        x[(D["age"] > 64)] = np.nan            # earnings are not comparable after retirement age
        return x
    if name == "prestige":
        return D["prestg10"].copy()
    if name == "schooling":
        return D["educ"].copy()
    if name == "vocabulary":
        return D["wordsum"].copy()
    raise KeyError(name)


def cells(mask, sexval):
    """Sex x birth-decade cells inside which traits are standardised and fitness is relative."""
    dec = np.floor(D["cohort"] / 10.0)
    for d in np.unique(dec[mask & (D["sex"] == sexval)]):
        yield mask & (D["sex"] == sexval) & (dec == d)


def decompose_cell(z, c, w):
    """Return (total, access, number, weight) for one cell; z is the raw trait."""
    sd = np.sqrt(wcov(z, z, w))
    if not np.isfinite(sd) or sd == 0:
        return None
    zs = (z - wmean(z, w)) / sd
    par = c > 0
    if par.sum() < 5 or (~par).sum() < 1:
        return None
    cbar = wmean(c, w)
    total = wcov(zs, c, w) / cbar
    access = wmean(zs[par], w[par])
    number = wcov(zs[par], c[par], w[par]) / wmean(c[par], w[par])
    return total, access, number, float(np.sum(w))


def decompose(mask, sexval, tname, rng=None):
    x = trait(tname)
    c, wt = D["childs"], D["wtssps"]          # read at call time (robustness variants swap the weights)
    out = []
    for cell in cells(mask & np.isfinite(x), sexval):
        idx = np.flatnonzero(cell)
        if len(idx) < 60:
            continue
        if rng is not None:
            idx = rng.choice(idx, size=len(idx), replace=True)
        r = decompose_cell(x[idx], c[idx], wt[idx])
        if r is not None:
            out.append(r)
    a = np.array(out)
    wts = a[:, 3] / a[:, 3].sum()
    return [float(np.sum(a[:, k] * wts)) for k in range(3)]


def with_ci(fn, rng):
    point = np.array(fn(None))
    boots = np.array([fn(rng) for _ in range(NBOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return point, lo, hi, boots


def with_ci_n(fn, rng, n):
    point = np.array(fn(None))
    boots = np.array([fn(rng) for _ in range(n)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return point, lo, hi, boots


def main():
    rng = np.random.default_rng(SEED)
    base = base_frame()
    res = {"n_base": int(base.sum()), "nboot": NBOOT, "years": [int(D["year"][base].min()), int(D["year"][base].max())]}

    # ---- 1. access / number decomposition, pooled over cohorts and by cohort
    dec = {}
    for tname in ["schooling", "vocabulary", "prestige", "income"]:
        dec[tname] = {}
        for sname, sv in SEXES.items():
            x = trait(tname)
            n = int((base & np.isfinite(x) & (D["sex"] == sv)).sum())
            pt, lo, hi, boots = with_ci(lambda r: decompose(base, sv, tname, r), rng)
            share = boots[:, 1] / boots[:, 0]
            entry = dict(n=n, total=[pt[0], lo[0], hi[0]], access=[pt[1], lo[1], hi[1]], number=[pt[2], lo[2], hi[2]],
                         p_access_pos=float(np.mean(boots[:, 1] > 0)), p_number_neg=float(np.mean(boots[:, 2] < 0)),
                         check=float(pt[0] - pt[1] - pt[2]))
            entry["by_cohort"] = {}
            for (a, b) in COHORTS:
                m = base & (D["cohort"] >= a) & (D["cohort"] <= b)
                nn = int((m & np.isfinite(x) & (D["sex"] == sv)).sum())
                if nn < 300:
                    continue
                p2, l2, h2, _ = with_ci(lambda r: decompose(m, sv, tname, r), rng)
                entry["by_cohort"][f"{a}-{b}"] = dict(n=nn, total=[p2[0], l2[0], h2[0]], access=[p2[1], l2[1], h2[1]],
                                                     number=[p2[2], l2[2], h2[2]])
            dec[tname][sname] = entry
    res["decomposition"] = dec

    # ---- 2. sex difference in the access component (men minus women), bootstrap
    diff = {}
    for tname in dec:
        bm = np.array([decompose(base, 1.0, tname, rng) for _ in range(NBOOT)])
        bw = np.array([decompose(base, 2.0, tname, rng) for _ in range(NBOOT)])
        d = bm - bw
        diff[tname] = dict(access=[float(np.mean(d[:, 1])), *[float(v) for v in np.percentile(d[:, 1], [2.5, 97.5])]],
                           number=[float(np.mean(d[:, 2])), *[float(v) for v in np.percentile(d[:, 2], [2.5, 97.5])]],
                           total=[float(np.mean(d[:, 0])), *[float(v) for v in np.percentile(d[:, 0], [2.5, 97.5])]])
    res["sex_difference"] = diff

    # ---- 3. childlessness and family size by trait third (for the figure)
    thirds = {}
    c, wt = D["childs"], D["wtssps"]
    for tname in ["schooling", "prestige", "income"]:
        thirds[tname] = {}
        x = trait(tname)
        for sname, sv in SEXES.items():
            rank = np.full(len(x), np.nan)
            for cell in cells(base & np.isfinite(x), sv):
                idx = np.flatnonzero(cell)
                if len(idx) < 60:
                    continue
                order = np.argsort(np.argsort(x[idx] + 1e-9 * rng.standard_normal(len(idx))))
                rank[idx] = (order + 0.5) / len(idx)
            rows = []
            for k in range(3):
                m = np.isfinite(rank) & (rank >= k / 3) & (rank < (k + 1) / 3)
                idx = np.flatnonzero(m)

                def stat(ii):
                    cc, ww = c[ii], wt[ii]
                    par = cc > 0
                    return [1 - wmean(par.astype(float), ww), wmean(cc[par], ww[par]), wmean(cc, ww)]
                pt = stat(idx)
                bs = np.array([stat(rng.choice(idx, len(idx))) for _ in range(600)])
                lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
                rows.append(dict(n=int(len(idx)), childless=[pt[0], lo[0], hi[0]], kids_parents=[pt[1], lo[1], hi[1]],
                                 kids_all=[pt[2], lo[2], hi[2]]))
            thirds[tname][sname] = rows
    res["thirds"] = thirds

    # ---- 4. opportunity for selection and the share due to childlessness
    opp = {}
    for sname, sv in SEXES.items():
        opp[sname] = {}
        for (a, b) in COHORTS + [(1900, 1979)]:
            m = base & (D["sex"] == sv) & (D["cohort"] >= a) & (D["cohort"] <= b)
            idx = np.flatnonzero(m)

            def stat(ii):
                cc, ww = c[ii], wt[ii]
                p = wmean((cc > 0).astype(float), ww)
                cbar = wmean(cc, ww)
                I = wcov(cc, cc, ww) / cbar ** 2
                return [p, cbar, I, ((1 - p) / p) / I]
            pt = stat(idx)
            bs = np.array([stat(rng.choice(idx, len(idx))) for _ in range(600)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            opp[sname][f"{a}-{b}"] = dict(n=int(len(idx)), parent=[pt[0], lo[0], hi[0]], mean_children=[pt[1], lo[1], hi[1]],
                                          I=[pt[2], lo[2], hi[2]], share_childless=[pt[3], lo[3], hi[3]])
    res["opportunity"] = opp

    # ---- 5. children by lifetime number of opposite-sex partners (a Bateman curve)
    bins = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 988)]
    bat = {}
    for sname, sv, var in [("men", 1.0, "numwomen"), ("women", 2.0, "nummen")]:
        k = D[var].copy()
        k[k > 988] = np.nan                               # codes above 988 are not counts
        m = base & (D["sex"] == sv) & np.isfinite(k)
        rows = []
        for (a, b) in bins:
            idx = np.flatnonzero(m & (k >= a) & (k <= b))
            if len(idx) < 20:
                rows.append(dict(lo=a, hi=b, n=int(len(idx))))
                continue
            pt = [wmean(c[idx], wt[idx]), 1 - wmean((c[idx] > 0).astype(float), wt[idx])]
            bs = np.array([[wmean(c[ii], wt[ii]), 1 - wmean((c[ii] > 0).astype(float), wt[ii])]
                           for ii in (rng.choice(idx, len(idx)) for _ in range(600))])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            rows.append(dict(lo=a, hi=b, n=int(len(idx)), share=float(wt[idx].sum() / wt[m].sum()),
                             children=[pt[0], lo[0], hi[0]], childless=[pt[1], lo[1], hi[1]]))
        idx = np.flatnonzero(m)
        kk = np.minimum(k, 50)                            # cap the long tail before computing a slope

        def grad(ii):
            wrel = c[ii] / wmean(c[ii], wt[ii])
            mrel = kk[ii] / wmean(kk[ii], wt[ii])
            slope_all = wcov(mrel, wrel, wt[ii]) / wcov(mrel, mrel, wt[ii])
            pos = kk[ii] > 0
            j = ii[pos]
            wrel2 = c[j] / wmean(c[j], wt[j])
            mrel2 = kk[j] / wmean(kk[j], wt[j])
            slope_pos = wcov(mrel2, wrel2, wt[j]) / wcov(mrel2, mrel2, wt[j])
            return [slope_all, slope_pos, wcorr(np.log1p(kk[ii]), c[ii], wt[ii])]
        pt = grad(idx)
        bs = np.array([grad(rng.choice(idx, len(idx))) for _ in range(600)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        bat[sname] = dict(n=int(len(idx)), rows=rows, median=float(np.median(k[m])),
                          gradient_all=[pt[0], lo[0], hi[0]], gradient_partnered=[pt[1], lo[1], hi[1]],
                          corr_log=[pt[2], lo[2], hi[2]])
    res["bateman"] = bat

    # ---- 6. no sexual partner in the past year, ages 22-34, by own income third and period
    age, sex, yr, pt_ = D["age"], D["sex"], D["year"], D["partners"]
    inc = D["realrinc"]
    sl = {}
    for sname, sv in SEXES.items():
        sl[sname] = {}
        for (a, b) in [(1989, 2002), (2004, 2014), (2016, 2024)]:
            m = (sex == sv) & (age >= 22) & (age <= 34) & (yr >= a) & (yr <= b) & np.isfinite(pt_) & np.isfinite(D["wtssps"])
            idx = np.flatnonzero(m)
            rate = lambda ii: wmean((pt_[ii] == 0).astype(float), wt[ii])
            bs = [rate(rng.choice(idx, len(idx))) for _ in range(600)]
            entry = dict(n=int(len(idx)), all=[rate(idx), *[float(v) for v in np.percentile(bs, [2.5, 97.5])]])
            mi = m & np.isfinite(inc)
            ii = np.flatnonzero(mi)
            r = np.empty(len(ii))
            for y in np.unique(yr[ii]):
                s = yr[ii] == y
                o = np.argsort(np.argsort(inc[ii][s] + 1e-6 * rng.standard_normal(s.sum())))
                r[s] = (o + 0.5) / s.sum()
            for k, lab in enumerate(["low", "mid", "high"]):
                jj = ii[(r >= k / 3) & (r < (k + 1) / 3)]
                bs = [rate(rng.choice(jj, len(jj))) for _ in range(600)]
                entry[lab] = [rate(jj), *[float(v) for v in np.percentile(bs, [2.5, 97.5])], int(len(jj))]
            noinc = np.flatnonzero(m & ~np.isfinite(inc))
            bs = [rate(rng.choice(noinc, len(noinc))) for _ in range(600)]
            entry["no_earnings"] = [rate(noinc), *[float(v) for v in np.percentile(bs, [2.5, 97.5])], int(len(noinc))]
            sl[sname][f"{a}-{b}"] = entry
    res["partnerless"] = sl

    # ---- 7. never married by age 45+, by trait third (pairing), and spousal resemblance
    mar = D["marital"]
    pair = {}
    for tname in ["schooling", "prestige", "income"]:
        pair[tname] = {}
        x = trait(tname)
        for sname, sv in SEXES.items():
            acc = []
            for cell in cells(base & np.isfinite(x) & np.isfinite(mar), sv):
                idx = np.flatnonzero(cell)
                if len(idx) < 60:
                    continue
                zs = (x[idx] - wmean(x[idx], wt[idx])) / np.sqrt(wcov(x[idx], x[idx], wt[idx]))
                ever = mar[idx] != 5
                acc.append((wmean(zs[ever], wt[idx][ever]), wmean((~ever).astype(float), wt[idx]), wt[idx].sum()))
            a_ = np.array(acc)
            ww = a_[:, 2] / a_[:, 2].sum()
            pair[tname][sname] = dict(S_ever_married=float(np.sum(a_[:, 0] * ww)), never_married=float(np.sum(a_[:, 1] * ww)))
    res["pairing"] = pair

    hom = {}
    ed, sp = D["educ"], D["speduc"]
    for (a, b) in [(1972, 1989), (1990, 2006), (2008, 2024)]:
        m = np.isfinite(ed) & np.isfinite(sp) & (mar == 1) & (yr >= a) & (yr <= b) & np.isfinite(wt)
        idx = np.flatnonzero(m)
        bs = [wcorr(ed[ii], sp[ii], wt[ii]) for ii in (rng.choice(idx, len(idx)) for _ in range(600))]
        hom[f"{a}-{b}"] = dict(n=int(len(idx)), r=[wcorr(ed[idx], sp[idx], wt[idx]), *[float(v) for v in np.percentile(bs, [2.5, 97.5])]])
    res["homogamy"] = hom

    # ---- 8. timing: age at first birth and schooling (parents aged 45+, 1994 onward)
    akb = D["agekdbrn"]
    tim = {}
    for sname, sv in SEXES.items():
        m = base & (D["sex"] == sv) & np.isfinite(akb) & np.isfinite(ed) & (c > 0)
        idx = np.flatnonzero(m)

        def stat(ii):
            return [wcorr(ed[ii], akb[ii], wt[ii]), wcorr(akb[ii], c[ii], wt[ii]), wmean(akb[ii], wt[ii])]
        pt = stat(idx)
        bs = np.array([stat(rng.choice(idx, len(idx))) for _ in range(600)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        tim[sname] = dict(n=int(len(idx)), r_school_agefirst=[pt[0], lo[0], hi[0]], r_agefirst_children=[pt[1], lo[1], hi[1]],
                          mean_age_first=[pt[2], lo[2], hi[2]])
    res["timing"] = tim

    # ---- 9. how much of the number component runs through the age at first birth (covariance accounting)
    acct = {}
    for sname, sv in SEXES.items():
        m = base & (D["sex"] == sv) & np.isfinite(akb) & np.isfinite(ed) & (c > 0)
        idx0 = np.flatnonzero(m)
        dec_ = np.floor(D["cohort"] / 10.0)

        def stat(ii):
            z = np.empty(len(ii)); wr = np.empty(len(ii)); t = np.empty(len(ii))
            for d in np.unique(dec_[ii]):
                s_ = dec_[ii] == d
                j = ii[s_]
                if len(j) < 30:
                    z[s_] = np.nan; wr[s_] = np.nan; t[s_] = np.nan
                    continue
                z[s_] = (ed[j] - wmean(ed[j], wt[j])) / np.sqrt(wcov(ed[j], ed[j], wt[j]))
                wr[s_] = c[j] / wmean(c[j], wt[j])
                t[s_] = akb[j] - wmean(akb[j], wt[j])
            g = np.isfinite(z)
            z, wr, t, w_ = z[g], wr[g], t[g], wt[ii][g]
            s_num = wcov(z, wr, w_)
            b_t = wcov(t, wr, w_) / wcov(t, t, w_)
            via = b_t * wcov(z, t, w_)
            return [s_num, via, via / s_num, b_t]
        pt = stat(idx0)
        bs = np.array([stat(rng.choice(idx0, len(idx0))) for _ in range(600)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        acct[sname] = dict(n=int(len(idx0)), number=[pt[0], lo[0], hi[0]], via_timing=[pt[1], lo[1], hi[1]],
                           share=[pt[2], lo[2], hi[2]], slope_per_year=[pt[3], lo[3], hi[3]])
    res["timing_account"] = acct

    # ---- 10. robustness of the two headline contrasts
    rob = {}
    wt_saved = D["wtssps"].copy()
    variants = {
        "main": (base, None),
        "unweighted": (base, "unw"),
        "ages 50-74": (base & (D["age"] >= 50) & (D["age"] <= 74), None),
        "surveys 1990-2024": (base & (D["year"] >= 1990), None),
        "surveys 1972-1989": (base & (D["year"] < 1990), None),
    }
    for vname, (mask, mode) in variants.items():
        if mode == "unw":
            D["wtssps"] = np.where(np.isfinite(wt_saved), 1.0, np.nan)
        rob[vname] = {}
        for tname in ["schooling", "income"]:
            for sname, sv in SEXES.items():
                pt, lo, hi, _b = with_ci_n(lambda r: decompose(mask, sv, tname, r), rng, 400)
                rob[vname][f"{tname}|{sname}"] = dict(access=[pt[1], lo[1], hi[1]], number=[pt[2], lo[2], hi[2]],
                                                      total=[pt[0], lo[0], hi[0]])
        D["wtssps"] = wt_saved
    res["robustness"] = rob

    # ---- 11. three episodes: pairing (ever married), conversion (parenthood given pairing), number
    #      S = E[z | ever married] + (E[z | parent] - E[z | ever married]) + Cov(z, c | parent)/E[c | parent]
    def three(sv, tname, r=None):
        x = trait(tname); mar_ = D["marital"]
        out = []
        for cell in cells(base & np.isfinite(x) & np.isfinite(mar_), sv):
            idx = np.flatnonzero(cell)
            if len(idx) < 60:
                continue
            if r is not None:
                idx = r.choice(idx, size=len(idx), replace=True)
            z0, c_, w_, ev = x[idx], c[idx], wt[idx], mar_[idx] != 5
            sd = np.sqrt(wcov(z0, z0, w_))
            if not np.isfinite(sd) or sd == 0:
                continue
            z = (z0 - wmean(z0, w_)) / sd
            par = c_ > 0
            if par.sum() < 5 or ev.sum() < 5 or (~par).sum() < 1:
                continue
            pair_ = wmean(z[ev], w_[ev]); acc_ = wmean(z[par], w_[par])
            num_ = wcov(z[par], c_[par], w_[par]) / wmean(c_[par], w_[par])
            out.append((pair_, acc_ - pair_, num_, acc_ + num_, float(w_.sum())))
        a = np.array(out); ww = a[:, 4] / a[:, 4].sum()
        return [float(np.sum(a[:, k] * ww)) for k in range(4)]
    te = {}
    for tname in ["schooling", "vocabulary", "prestige", "income"]:
        te[tname] = {}
        for sname, sv in SEXES.items():
            pt, lo, hi, _b = with_ci_n(lambda r: three(sv, tname, r), rng, 1000)
            te[tname][sname] = dict(pairing=[pt[0], lo[0], hi[0]], conversion=[pt[1], lo[1], hi[1]],
                                    number=[pt[2], lo[2], hi[2]], total=[pt[3], lo[3], hi[3]])
    res["three_episodes"] = te

    # ---- 12. shape of the floor: childlessness and never marrying by fifth of the trait
    fifths = {}
    for tname in ["income", "schooling", "prestige"]:
        fifths[tname] = {}
        x = trait(tname)
        for sname, sv in SEXES.items():
            rank = np.full(len(x), np.nan)
            for cell in cells(base & np.isfinite(x) & np.isfinite(mar), sv):
                idx = np.flatnonzero(cell)
                if len(idx) < 60:
                    continue
                order = np.argsort(np.argsort(x[idx] + 1e-9 * rng.standard_normal(len(idx))))
                rank[idx] = (order + 0.5) / len(idx)
            idx_all = np.flatnonzero(np.isfinite(rank))
            q = np.minimum((rank * 5).astype("float"), 4.999)

            def stat(ii):
                qq = np.floor(q[ii]); cc, ww = c[ii], wt[ii]; nm = (mar[ii] == 5).astype(float)
                cl = [1 - wmean((cc[qq == k] > 0).astype(float), ww[qq == k]) for k in range(5)]
                nv = [wmean(nm[qq == k], ww[qq == k]) for k in range(5)]
                return cl + nv + [(cl[0] - cl[2]) - (cl[2] - cl[4]), cl[0] - cl[4]]
            pt = np.array(stat(idx_all))
            bs = np.array([stat(rng.choice(idx_all, len(idx_all))) for _ in range(1000)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            fifths[tname][sname] = dict(n=int(len(idx_all)),
                                        childless=[[pt[k], lo[k], hi[k]] for k in range(5)],
                                        never_married=[[pt[5 + k], lo[5 + k], hi[5 + k]] for k in range(5)],
                                        concentration=[pt[10], lo[10], hi[10]], range=[pt[11], lo[11], hi[11]])
    res["fifths"] = fifths

    # ---- 13. the pairing gate: how much reproduction passes through marriage, by sex and birth cohort
    #      outside = share of all children reported by people who never married;
    #      parents_nm = share of never-married people who are parents; never = share never married.
    gate = {}
    for sname, sv in SEXES.items():
        gate[sname] = {}
        for lo_, hi_ in COHORTS + [(1900, 1979)]:
            idx_all = np.flatnonzero(base & np.isfinite(mar) & (D["sex"] == sv) & (D["cohort"] >= lo_) & (D["cohort"] <= hi_))

            def gstat(ii):
                cc, ww, nm = c[ii], wt[ii], mar[ii] == 5
                return [float(np.sum(ww * cc * nm) / np.sum(ww * cc)),
                        wmean((cc[nm] > 0).astype(float), ww[nm]),
                        wmean(nm.astype(float), ww)]
            pt = np.array(gstat(idx_all))
            bs = np.array([gstat(rng.choice(idx_all, len(idx_all))) for _ in range(1000)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            gate[sname][f"{lo_}-{hi_}"] = dict(n=int(len(idx_all)), outside=[pt[0], lo[0], hi[0]],
                                               parents_nm=[pt[1], lo[1], hi[1]], never=[pt[2], lo[2], hi[2]])
    res["gate"] = gate

    # ---- 14. the three episodes for income by birth cohort (is pairing more or less selective as the gate opens?)
    def three_c(sv, lo_, hi_, r=None):
        x = trait("income"); out = []
        m_ = base & np.isfinite(x) & np.isfinite(mar) & (D["cohort"] >= lo_) & (D["cohort"] <= hi_)
        for cell in cells(m_, sv):
            idx = np.flatnonzero(cell)
            if len(idx) < 60:
                continue
            if r is not None:
                idx = r.choice(idx, size=len(idx), replace=True)
            z0, c_, w_, ev = x[idx], c[idx], wt[idx], mar[idx] != 5
            sd = np.sqrt(wcov(z0, z0, w_))
            z = (z0 - wmean(z0, w_)) / sd
            par = c_ > 0
            pair_ = wmean(z[ev], w_[ev]); acc_ = wmean(z[par], w_[par])
            num_ = wcov(z[par], c_[par], w_[par]) / wmean(c_[par], w_[par])
            out.append((pair_, acc_ - pair_, num_, acc_ + num_, float(w_.sum())))
        a = np.array(out); ww = a[:, 4] / a[:, 4].sum()
        return [float(np.sum(a[:, k] * ww)) for k in range(4)]
    tc = {}
    for sname, sv in SEXES.items():
        tc[sname] = {}
        for lo_, hi_ in COHORTS[1:]:                       # the 1900-1919 cohort has too few incomes at ages 45-64
            pt, lo, hi, bs = with_ci_n(lambda r: three_c(sv, lo_, hi_, r), rng, 1000)
            tc[sname][f"{lo_}-{hi_}"] = dict(pairing=[pt[0], lo[0], hi[0]], conversion=[pt[1], lo[1], hi[1]],
                                             number=[pt[2], lo[2], hi[2]], total=[pt[3], lo[3], hi[3]], _boot_pairing=bs[:, 0].tolist())
        d_ = np.array(tc[sname]["1960-1979"]["_boot_pairing"]) - np.array(tc[sname]["1920-1939"]["_boot_pairing"])
        tc[sname]["pairing_change"] = [tc[sname]["1960-1979"]["pairing"][0] - tc[sname]["1920-1939"]["pairing"][0],
                                       float(np.percentile(d_, 2.5)), float(np.percentile(d_, 97.5))]
        for k in list(tc[sname]):
            if isinstance(tc[sname][k], dict):
                tc[sname][k].pop("_boot_pairing")
    res["three_by_cohort"] = tc

    # ---- 15. who marries up: wife's schooling relative to husband's among the currently married, by birth cohort
    educ, sp = D["educ"], D["speduc"]
    wife = np.where(D["sex"] == 2.0, educ, sp); husb = np.where(D["sex"] == 2.0, sp, educ)
    hyp = {}
    for lo_, hi_ in COHORTS + [(1980, 2004)]:
        idx_all = np.flatnonzero(np.isfinite(educ) & np.isfinite(sp) & (sp <= 20) & (mar == 1) & np.isfinite(wt) & (wt > 0)
                                 & (D["age"] >= 25) & (D["cohort"] >= lo_) & (D["cohort"] <= hi_))

        def hstat(ii):
            ww = wt[ii]; more = wmean((wife[ii] > husb[ii]).astype(float), ww); less = wmean((wife[ii] < husb[ii]).astype(float), ww)
            return [more, less, more - less, wcorr(wife[ii], husb[ii], ww)]
        pt = np.array(hstat(idx_all))
        bs = np.array([hstat(rng.choice(idx_all, len(idx_all))) for _ in range(500)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        hyp[f"{lo_}-{hi_}"] = dict(n=int(len(idx_all)), wife_more=[pt[0], lo[0], hi[0]], wife_less=[pt[1], lo[1], hi[1]],
                                   more_minus_less=[pt[2], lo[2], hi[2]], r=[pt[3], lo[3], hi[3]])
    res["hypergamy"] = hyp

    # ---- 16. marital history and family size (relative to the mean of one's sex and birth decade), ages 45-79
    dv = D["divorce"]
    relc = np.full(len(c), np.nan)
    for sv in SEXES.values():
        for cell in cells(base, sv):
            relc[cell] = c[cell] / wmean(c[cell], wt[cell])
    groups = {"never": mar == 5, "intact": ((mar == 1) | (mar == 2)) & (dv == 2), "remarried": ((mar == 1) | (mar == 2)) & (dv == 1),
              "divorced": (mar == 3) | (mar == 4)}
    hist = {}
    boots_gap = {}
    for sname, sv in SEXES.items():
        idx_all = np.flatnonzero(base & (D["sex"] == sv) & np.isfinite(mar) & np.isfinite(relc) & (groups["never"] | groups["intact"] | groups["remarried"] | groups["divorced"]))

        def mstat(ii):
            out = []
            for g in ("never", "intact", "remarried", "divorced"):
                m_ = groups[g][ii]
                out += [wmean(relc[ii][m_], wt[ii][m_]), wmean((c[ii][m_] == 0).astype(float), wt[ii][m_]), wmean(c[ii][m_], wt[ii][m_])]
            out += [out[6] - out[3], out[9] - out[3]]            # remarried minus intact; divorced minus intact (relative family size)
            return out
        pt = np.array(mstat(idx_all))
        bs = np.array([mstat(rng.choice(idx_all, len(idx_all))) for _ in range(500)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        hist[sname] = {g: dict(n=int(groups[g][idx_all].sum()), rel=[pt[3 * k], lo[3 * k], hi[3 * k]],
                               childless=[pt[3 * k + 1], lo[3 * k + 1], hi[3 * k + 1]], children=float(pt[3 * k + 2]))
                       for k, g in enumerate(("never", "intact", "remarried", "divorced"))}
        hist[sname]["remarried_minus_intact"] = [pt[12], lo[12], hi[12]]
        hist[sname]["divorced_minus_intact"] = [pt[13], lo[13], hi[13]]
        boots_gap[sname] = bs[:, 12:14]
    d_ = boots_gap["men"] - boots_gap["women"]
    hist["sex_difference_remarried"] = [hist["men"]["remarried_minus_intact"][0] - hist["women"]["remarried_minus_intact"][0],
                                        float(np.percentile(d_[:, 0], 2.5)), float(np.percentile(d_[:, 0], 97.5))]
    hist["sex_difference_divorced"] = [hist["men"]["divorced_minus_intact"][0] - hist["women"]["divorced_minus_intact"][0],
                                       float(np.percentile(d_[:, 1], 2.5)), float(np.percentile(d_[:, 1], 97.5))]
    res["marital_history"] = hist

    # ---- 17. standing and the number of sexual partners: rank correlation of each trait with lifetime partners of the other sex
    def rank(v):
        return np.argsort(np.argsort(v)).astype(float)
    part = {}
    for sname, sv, key in (("men", 1.0, "numwomen"), ("women", 2.0, "nummen")):
        pc_ = D[key]; part[sname] = {}
        for tname in ["schooling", "vocabulary", "prestige", "income"]:
            x = trait(tname)
            z = np.full(len(x), np.nan)
            m_ = base & (D["sex"] == sv) & np.isfinite(x) & np.isfinite(pc_) & (pc_ < 989)
            for cell in cells(m_, sv):
                ii = np.flatnonzero(cell)
                if len(ii) >= 30 and np.std(x[ii]) > 0:
                    z[ii] = (x[ii] - np.mean(x[ii])) / np.std(x[ii])
            idx_all = np.flatnonzero(m_ & np.isfinite(z))

            def pstat(ii):
                return float(np.corrcoef(rank(z[ii]), rank(pc_[ii] + 1e-9 * np.arange(len(ii))))[0, 1])
            pt = pstat(idx_all)
            bs = [pstat(rng.choice(idx_all, len(idx_all))) for _ in range(500)]
            part[sname][tname] = dict(n=int(len(idx_all)), rho=[pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                      median=float(np.median(pc_[idx_all])))
    res["partners_by_trait"] = part
    save("gss", res)
    return res


if __name__ == "__main__":
    r = main()
    for t, v in r["decomposition"].items():
        for s, e in v.items():
            print(f"{t:10s} {s:5s} n={e['n']:6d} total={e['total'][0]:+.3f} [{e['total'][1]:+.3f},{e['total'][2]:+.3f}] "
                  f"access={e['access'][0]:+.3f} [{e['access'][1]:+.3f},{e['access'][2]:+.3f}] "
                  f"number={e['number'][0]:+.3f} [{e['number'][1]:+.3f},{e['number'][2]:+.3f}] chk={e['check']:.1e}")
