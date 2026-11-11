"""E17: what four minutes can read, and what people then do (Columbia speed-dating experiment).

Data: the public file of Fisman, Iyengar, Kamenica and Simonson (2006): 8,378 rows, one per person per four-minute
date; 551 graduate students in 21 sessions.  Each person rated each partner on six attributes (1-10), said yes or no to
meeting again, and guessed how likely the partner was to say yes.  Race and religion variables are not used.

1. Jury law.  Agreement between two disjoint panels of k raters about the same person, for each attribute, against the
   Spearman-Brown prediction from k = 1; single-rater agreement by sex of the person rated.
2. Said and did.  Stated importance before the event against the change in the probability of a yes per standard
   deviation of the rating (chooser fixed effects), using the chooser's own rating and the rating of everyone else.
3. Legibility and weight.  Whether attributes that one meeting reads well are the ones that earn a yes.
4. Access after four minutes.  Inequality in the share of yeses received, by sex.
5. Reading interest.  Expected against actual yeses, by sex, and how well people know who said yes.
6. Self-knowledge and an outside criterion.  Self-ratings against the panel; panel ratings against the median
   entrance-test score of the person's college.

Run with:  python3 -I code/exp_speed.py     (reads a third-party file, so isolated mode; about one minute)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import save

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "speeddating", "speed_dating.csv")
SEED = 20261124
ATTRS = ["attr", "fun", "intel", "amb", "sinc", "shar"]            # looks, fun, intelligence, ambition, sincerity, shared interests
PERSON = ["attr", "fun", "intel", "amb", "sinc"]                    # attributes of a person (shared interests belong to a pair)
NAMES = dict(attr="looks", fun="fun", intel="intelligence", amb="ambition", sinc="sincerity", shar="shared interests")
KMAX = 8


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return np.nan


def load():
    with open(PATH, encoding="latin-1") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    keep = ["iid", "pid", "gender", "wave", "dec", "dec_o", "match", "prob", "like", "met", "age", "mn_sat", "order",
            "attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"] + ATTRS + [a + "1_1" for a in ATTRS] + [a + "7_2" for a in ATTRS]
    col = {k: np.array([num(r[head.index(k)]) for r in rows[1:]]) for k in keep}
    for a in ATTRS:                                                 # ratings are on a 1-10 scale; a few stray values are dropped
        col[a] = np.where((col[a] >= 0) & (col[a] <= 10), col[a], np.nan)
    return col


def rater_centered(D, a):
    """Rating minus the rater's own mean rating of everyone met (removes leniency and session differences)."""
    x = D[a].copy()
    out = np.full(len(x), np.nan)
    for r in np.unique(D["iid"]):
        m = (D["iid"] == r) & np.isfinite(x)
        if m.sum() >= 2:
            out[m] = x[m] - x[m].mean()
    return out


def by_target(D, x, target_sex=None):
    """Lists of centered ratings received by each person (optionally only people of one sex)."""
    out = {}
    ok = np.isfinite(x) & np.isfinite(D["pid"])
    if target_sex is not None:
        ok &= D["gender"] == (1 - target_sex)                       # raters are of the other sex
    for t, v in zip(D["pid"][ok], x[ok]):
        out.setdefault(int(t), []).append(v)
    return out


def icc1(groups):
    """One-way random-effects intraclass correlation: agreement between two single raters of the same person."""
    g = [np.asarray(v) for v in groups if len(v) >= 2]
    n = np.array([len(v) for v in g]); N = n.sum(); a = len(g)
    grand = np.concatenate(g).mean()
    msb = sum(len(v) * (v.mean() - grand) ** 2 for v in g) / (a - 1)
    msw = sum(((v - v.mean()) ** 2).sum() for v in g) / (N - a)
    k0 = (N - (n ** 2).sum() / N) / (a - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def split_panels(groups, k, rng, draws=400):
    """Mean correlation, over random draws, between two disjoint panels of k raters of the same person."""
    g = [np.asarray(v) for v in groups if len(v) >= 2 * k]
    if len(g) < 30:
        return np.nan, len(g)
    rs = []
    for _ in range(draws):
        a, b = [], []
        for v in g:
            p = rng.permutation(len(v))
            a.append(v[p[:k]].mean()); b.append(v[p[k:2 * k]].mean())
        rs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(rs)), len(g)


def jury(D, rng):
    out = {}
    for a in PERSON:
        x = rater_centered(D, a)
        G = by_target(D, x)
        groups = list(G.values())
        r1 = icc1(groups)
        keys = list(G)
        boots = [icc1([G[keys[i]] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(500)]
        curve = [split_panels(groups, k, rng)[0] for k in range(1, KMAX + 1)]
        sb = [k * curve[0] / (1 + (k - 1) * curve[0]) for k in range(1, KMAX + 1)]
        sex = {}
        for s, name in ((1, "men"), (0, "women")):                  # the people being rated
            Gs = by_target(D, x, target_sex=s); ks = list(Gs)
            bs = [icc1([Gs[ks[i]] for i in rng.integers(0, len(ks), len(ks))]) for _ in range(500)]
            sex[name] = dict(r1=icc1(list(Gs.values())), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                             n=len(ks), boots=bs)
        d = np.array(sex["women"]["boots"]) - np.array(sex["men"]["boots"])
        for v in sex.values():
            v.pop("boots")
        out[a] = dict(name=NAMES[a], r1=r1, ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                      n_people=len(groups), mean_raters=float(np.mean([len(v) for v in groups])),
                      curve=curve, spearman_brown=sb, max_gap=float(np.nanmax(np.abs(np.array(curve) - np.array(sb)))),
                      k_for_08=float(4 * (1 - r1) / r1), by_sex=sex,
                      women_minus_men=[sex["women"]["r1"] - sex["men"]["r1"], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    return out


def first_per_person(D, key):
    ids, idx = np.unique(D["iid"], return_index=True)
    return ids, D[key][idx], D["gender"][idx]


def said(D):
    ids, _, sex = first_per_person(D, "attr1_1")
    vals = np.array([first_per_person(D, a + "1_1")[1] for a in ATTRS]).T          # people x attributes
    ok = np.isfinite(vals).all(axis=1) & (vals.sum(axis=1) > 0)
    share = 100 * vals[ok] / vals[ok].sum(axis=1, keepdims=True)
    out = {}
    for s, name in ((0, "women"), (1, "men")):
        m = sex[ok] == s
        out[name] = {a: float(share[m, j].mean()) for j, a in enumerate(ATTRS)}
        out[name]["n"] = int(m.sum())
    return out


def after(D):
    """What people said, the day after, had actually mattered in their decisions (asked in some sessions)."""
    ids, _, sex = first_per_person(D, "attr7_2")
    vals = np.array([first_per_person(D, a + "7_2")[1] for a in ATTRS]).T
    ok = np.isfinite(vals).all(axis=1) & (vals.sum(axis=1) > 0)
    share = 100 * vals[ok] / vals[ok].sum(axis=1, keepdims=True)
    out = {}
    for s, name in ((0, "women"), (1, "men")):
        m = sex[ok] == s
        out[name] = {a: float(share[m, j].mean()) for j, a in enumerate(ATTRS)}
        out[name]["n"] = int(m.sum())
    return out


def panel_rating(D, a):
    """Mean rating of the partner by every other person who met that partner (leave the chooser out)."""
    x = D[a]
    tot, cnt = {}, {}
    ok = np.isfinite(x) & np.isfinite(D["pid"])
    for t, v in zip(D["pid"][ok], x[ok]):
        tot[t] = tot.get(t, 0.0) + v; cnt[t] = cnt.get(t, 0) + 1
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        t = D["pid"][i]
        if t in tot:
            own = x[i] if np.isfinite(x[i]) else 0.0
            n = cnt[t] - (1 if np.isfinite(x[i]) else 0)
            if n >= 3:
                out[i] = (tot[t] - own) / n
    return out


def lpm(D, X, sexval, rng, boots=600):
    """Linear probability model of the yes/no decision on standardized ratings with chooser fixed effects."""
    y = D["dec"]
    ok = (D["gender"] == sexval) & np.isfinite(y) & np.isfinite(X).all(axis=1)
    Z = (X - np.nanmean(X[np.isfinite(X).all(axis=1)], axis=0)) / np.nanstd(X[np.isfinite(X).all(axis=1)], axis=0)
    ids = D["iid"][ok]; yy = y[ok].copy(); ZZ = Z[ok].copy()
    uniq = np.unique(ids)
    rows = {u: np.flatnonzero(ids == u) for u in uniq}

    def fit(sel):
        ys, zs = [], []
        for u in sel:
            r = rows[u]
            if len(r) < 2:
                continue
            ys.append(yy[r] - yy[r].mean()); zs.append(ZZ[r] - ZZ[r].mean(axis=0))
        ys = np.concatenate(ys); zs = np.vstack(zs)
        return 100 * np.linalg.lstsq(zs, ys, rcond=None)[0]
    pt = fit(uniq)
    bs = np.array([fit(rng.choice(uniq, len(uniq))) for _ in range(boots)])
    return pt, bs, int(ok.sum()), int(len(uniq))


def did(D, rng):
    own = np.array([D[a] for a in ATTRS]).T
    pan = np.array([panel_rating(D, a) for a in ATTRS]).T
    out = {}
    for label, X in (("own", own), ("panel", pan)):
        res = {}
        for s, name in ((0, "women"), (1, "men")):
            pt, bs, n, nch = lpm(D, X, s, rng)
            res[name] = dict(coef={a: [float(pt[j]), float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
                                   for j, a in enumerate(ATTRS)}, n_dates=n, n_choosers=nch, _b=bs)
        d = res["men"]["_b"] - res["women"]["_b"]
        res["men_minus_women"] = {a: [res["men"]["coef"][a][0] - res["women"]["coef"][a][0],
                                      float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))] for j, a in enumerate(ATTRS)}
        for name in ("women", "men"):
            res[name].pop("_b")
        out[label] = res
    return out


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def access(D):
    out = {}
    for s, name in ((0, "women"), (1, "men")):
        m = (D["gender"] == s) & np.isfinite(D["dec_o"]) & np.isfinite(D["dec"])
        ids = D["iid"][m]
        pop = np.array([D["dec_o"][m][ids == u].mean() for u in np.unique(ids)])     # share of partners who said yes to this person
        srt = np.sort(pop); n = len(srt)
        gini = float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum()))
        out[name] = dict(n=int(n), says_yes=float(100 * D["dec"][m].mean()), receives_yes=float(100 * D["dec_o"][m].mean()),
                         cv=float(pop.std(ddof=1) / pop.mean()), none=float(100 * np.mean(pop == 0)), gini=gini,
                         match=float(100 * D["match"][m].mean()))
    return out


def interest(D, rng):
    """Expected against actual interest: the guess (1-10) that the partner will say yes, and whether the partner did."""
    out = {}
    stats = {}
    for s, name in ((0, "women"), (1, "men")):
        m = (D["gender"] == s) & np.isfinite(D["prob"]) & np.isfinite(D["dec_o"]) & (D["prob"] >= 1) & (D["prob"] <= 10)
        ids = D["iid"][m]; p = (D["prob"][m] - 1) / 9.0; y = D["dec_o"][m]
        uniq = np.unique(ids)
        per = {u: (p[ids == u], y[ids == u]) for u in uniq}

        def stat(sel):
            pp = np.concatenate([per[u][0] for u in sel]); yy = np.concatenate([per[u][1] for u in sel])
            acc = []
            for u in sel:
                a, b = per[u]
                if len(a) >= 5 and a.std() > 0 and b.std() > 0:
                    acc.append(np.corrcoef(a, b)[0, 1])
            return np.array([pp.mean(), yy.mean(), pp.mean() - yy.mean(), np.mean(acc)])
        pt = stat(uniq)
        bs = np.array([stat(rng.choice(uniq, len(uniq))) for _ in range(600)])
        stats[name] = bs
        ci = lambda j: [float(pt[j]), float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
        out[name] = dict(n_dates=int(m.sum()), n_people=int(len(uniq)), expected=ci(0), actual=ci(1), gap=ci(2), accuracy=ci(3))
    d = stats["men"][:, 2] - stats["women"][:, 2]
    out["gap_men_minus_women"] = [out["men"]["gap"][0] - out["women"]["gap"][0], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out


def self_and_criterion(D, rng):
    ids, idx = np.unique(D["iid"], return_index=True)
    out = {"self": {}, "sat": {}}
    for a in PERSON:
        G = by_target(D, rater_centered(D, a))
        panel = np.array([np.mean(G[int(i)]) if int(i) in G and len(G[int(i)]) >= 5 else np.nan for i in ids])
        selfr = D[a + "3_1"][idx]
        ok = np.isfinite(panel) & np.isfinite(selfr)
        out["self"][a] = dict(r=float(np.corrcoef(panel[ok], selfr[ok])[0, 1]), n=int(ok.sum()))
        if a in ("intel", "attr"):
            sat = D["mn_sat"][idx]
            ok = np.isfinite(panel) & np.isfinite(sat)
            r = float(np.corrcoef(panel[ok], sat[ok])[0, 1])
            ii = np.flatnonzero(ok)
            bs = [np.corrcoef(panel[j], sat[j])[0, 1] for j in (rng.choice(ii, len(ii)) for _ in range(1000))]
            out["sat"][a] = dict(r=r, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=int(ok.sum()))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    D = load()
    ids, idx = np.unique(D["iid"], return_index=True)
    res = dict(rows=int(len(D["iid"])), people=int(len(ids)), women=int((D["gender"][idx] == 0).sum()), men=int((D["gender"][idx] == 1).sum()),
               sessions=int(len(np.unique(D["wave"]))), mean_age=float(np.nanmean(D["age"][idx])),
               jury=jury(D, rng), said=said(D), after=after(D), did=did(D, rng), access=access(D), interest=interest(D, rng),
               checks=self_and_criterion(D, rng))
    J = res["jury"]
    r1 = [J[a]["r1"] for a in PERSON]
    for label in ("own", "panel"):
        w = [0.5 * (res["did"][label]["women"]["coef"][a][0] + res["did"][label]["men"]["coef"][a][0]) for a in PERSON]
        res["legibility_" + label] = dict(r1=r1, weight=w, spearman=spearman(np.array(r1), np.array(w)),
                                          pearson=float(np.corrcoef(r1, w)[0, 1]))
    save("e17_speed", res)
    print("rows %d people %d (women %d, men %d) sessions %d mean age %.1f" % (res["rows"], res["people"], res["women"], res["men"], res["sessions"], res["mean_age"]))
    for a in PERSON:
        v = J[a]
        print("%-13s r1 %.3f [%.3f, %.3f]  k for 0.8: %5.1f  max gap to Spearman-Brown %.3f  | rated men %.3f, rated women %.3f, diff %+.3f [%+.3f, %+.3f]" % (
            v["name"], v["r1"], v["ci"][0], v["ci"][1], v["k_for_08"], v["max_gap"], v["by_sex"]["men"]["r1"], v["by_sex"]["women"]["r1"], *v["women_minus_men"]))
    for name in ("women", "men"):
        print("said  ", name, {a: round(res["said"][name][a], 1) for a in ATTRS}, "| after", {a: round(res["after"][name][a], 1) for a in ATTRS}, res["after"][name]["n"])
        for label in ("own", "panel"):
            print("did   ", name, label, {a: round(res["did"][label][name]["coef"][a][0], 1) for a in ATTRS})
    for label in ("own", "panel"):
        print("men - women", label, {a: [round(x, 1) for x in res["did"][label]["men_minus_women"][a]] for a in ATTRS})
        print("legibility vs weight", label, res["legibility_" + label])
    print("access", res["access"])
    print("interest", {k: v for k, v in res["interest"].items()})
    print("checks", res["checks"])
