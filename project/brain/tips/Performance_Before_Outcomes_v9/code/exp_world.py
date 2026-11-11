"""E20  Stated ideals, partners and pairing in 175 countries (Kowal et al. 2025, Scientific Data; CC BY 4.0).

The file is fetched by fetch_open.py and is not redistributed.  Countries are used as units of replication; none is
named or ranked.  Self-reports from convenience samples: associations only.

  1. ideals   : sex differences in the ideal long-term partner, country by country (direction and size)
  2. budget   : thirty points divided among six qualities, by sex
  3. door     : does a person's own standing go with having a partner, by sex (country fixed effects)
  4. ages     : actual age difference between partners, country by country
  5. screens  : time on social media and being single
  6. equality : do the sex differences shrink where attitudes are more equal

Run:  python3 -I code/exp_world.py
"""
import csv
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "kowal2025", "Final_dataset.csv")
SEED = 20261128
NB = 1000
MIN_SEX = 50          # minimum men and minimum women for a country to enter a country-level analysis
csv.field_size_limit(10**9)
NUM = ["Age", "Sex01", "Relationship_status01", "Children", "mate_age", "Time_social_media_1", "Gender_Equality_Mean",
       "self_physatt", "self_resources", "self_kind", "self_health",
       "ideallt_physatt", "ideallt_resources", "ideallt_kind", "ideallt_health",
       "preference_imp_health", "preference_imp_kind", "preference_imp_physat", "preference_imp_relig",
       "preference_imp_financ", "preference_imp_age", "Pref_Sum"]


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def load():
    cols = {k: [] for k in NUM}
    country = []
    with open(PATH, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            country.append(r.get("Country_live") or "NA")
            for k in NUM:
                cols[k].append(fnum(r.get(k)))
    D = {k: np.array(v, float) for k, v in cols.items()}
    names, idx = np.unique(np.array(country), return_inverse=True)
    D["c"] = np.where(names[idx] == "NA", -1, idx).astype(int)
    D["n_countries"] = int(np.sum(names != "NA"))
    return D


def cohen_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def by_country(D, var, extra=None):
    """Per-country (id, men's values, women's values) for a variable, countries with enough of each sex."""
    ok = (D["c"] >= 0) & np.isfinite(D[var]) & np.isfinite(D["Sex01"])
    if extra is not None:
        ok &= extra
    out = []
    for c in np.unique(D["c"][ok]):
        m = ok & (D["c"] == c)
        men, women = D[var][m & (D["Sex01"] == 0)], D[var][m & (D["Sex01"] == 1)]
        if len(men) >= MIN_SEX and len(women) >= MIN_SEX:
            out.append((int(c), men, women))
    return out


def summarize(vals, rng):
    v = np.array(vals, float)
    bs = np.array([np.median(rng.choice(v, len(v))) for _ in range(NB)])
    return {"countries": int(len(v)), "median": float(np.median(v)), "median_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "q10": float(np.percentile(v, 10)), "q90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max()),
            "positive": int(np.sum(v > 0)), "positive_pct": float(100 * np.mean(v > 0)), "values": [float(x) for x in v]}


def ideals(D, rng):
    out = {}
    for name, var, sign in (("resources", "ideallt_resources", +1), ("looks", "ideallt_physatt", -1),
                            ("kindness", "ideallt_kind", +1), ("health", "ideallt_health", +1)):
        rows = by_country(D, var)
        # d > 0 means the sex expected to want more of it does: women for resources, men for looks; women minus men otherwise
        d = [sign * cohen_d(w, m) for _, m, w in rows]
        out[name] = summarize(d, rng)
        out[name]["people"] = int(sum(len(m) + len(w) for _, m, w in rows))
        out[name]["ids"] = [c for c, _, _ in rows]
        out[name]["mean_men"] = float(np.mean(np.concatenate([m for _, m, _ in rows])))
        out[name]["mean_women"] = float(np.mean(np.concatenate([w for _, _, w in rows])))
    return out


def budget(D, rng):
    items = {"kindness": "preference_imp_kind", "health": "preference_imp_health", "looks": "preference_imp_physat",
             "resources": "preference_imp_financ", "age": "preference_imp_age", "religion": "preference_imp_relig"}
    ok = (D["Pref_Sum"] == 30) & np.isfinite(D["Sex01"])
    out = {"n_men": int(np.sum(ok & (D["Sex01"] == 0))), "n_women": int(np.sum(ok & (D["Sex01"] == 1))), "points": {}}
    for k, v in items.items():
        out["points"][k] = {"men": float(np.nanmean(D[v][ok & (D["Sex01"] == 0)])), "women": float(np.nanmean(D[v][ok & (D["Sex01"] == 1)]))}
    # in how many countries is kindness given the most points by each sex
    top = {"men": 0, "women": 0, "countries": 0}
    for c in np.unique(D["c"][ok & (D["c"] >= 0)]):
        m = ok & (D["c"] == c)
        if np.sum(m & (D["Sex01"] == 0)) < MIN_SEX or np.sum(m & (D["Sex01"] == 1)) < MIN_SEX:
            continue
        top["countries"] += 1
        for sname, sv in (("men", 0), ("women", 1)):
            means = {k: np.nanmean(D[v][m & (D["Sex01"] == sv)]) for k, v in items.items()}
            top[sname] += int(max(means, key=means.get) == "kindness")
    out["kindness_top"] = top
    return out


def within(x, groups):
    """Demean x within groups (integer labels)."""
    out = x.astype(float).copy()
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1, len(g)]
    for a, b in zip(starts[:-1], starts[1:]):
        ii = order[a:b]
        out[ii] -= out[ii].mean()
    return out


def zwithin(x, groups):
    out = within(x, groups)
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1, len(g)]
    for a, b in zip(starts[:-1], starts[1:]):
        ii = order[a:b]
        s = out[ii].std()
        out[ii] = out[ii] / s if s > 0 else 0.0
    return out


def door(D, rng):
    """Change in the probability of having a partner per standard deviation of a self-rated quality, by sex."""
    traits = {"resources": "self_resources", "looks": "self_physatt", "kindness": "self_kind", "health": "self_health"}
    ok = (D["c"] >= 0) & np.isfinite(D["Sex01"]) & np.isfinite(D["Relationship_status01"]) & np.isfinite(D["Age"]) & (D["Age"] >= 18) & (D["Age"] <= 60)
    for v in traits.values():
        ok &= np.isfinite(D[v])
    out = {"n": int(ok.sum()), "coef": {}, "by_country": {}}
    res = {}
    for sname, sv in (("men", 0), ("women", 1)):
        i = np.where(ok & (D["Sex01"] == sv))[0]
        g = D["c"][i]
        age = D["Age"][i]
        X = np.column_stack([zwithin(D[v][i], g) for v in traits.values()] + [within(age, g), within(age ** 2, g)])
        y = within(D["Relationship_status01"][i], g)
        res[sname] = (i, g, X, y)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        out["coef"][sname] = {k: float(100 * beta[j]) for j, k in enumerate(traits)}
        out["n_" + sname] = int(len(i))
        out["partnered_" + sname] = float(100 * D["Relationship_status01"][i].mean())
    # cluster bootstrap over countries for the coefficients and for the sex difference
    cs = np.unique(np.concatenate([res["men"][1], res["women"][1]]))
    by = {s: {c: np.where(res[s][1] == c)[0] for c in cs} for s in res}
    draws = {s: [] for s in res}
    for _ in range(NB):
        pick = rng.choice(cs, len(cs))
        for s in res:
            _, _, X, y = res[s]
            j = np.concatenate([by[s][c] for c in pick])
            draws[s].append(100 * np.linalg.lstsq(X[j], y[j], rcond=None)[0][:len(traits)])
    dm, dw = np.array(draws["men"]), np.array(draws["women"])
    out["ci"] = {s: {k: [float(np.percentile(a[:, j], 2.5)), float(np.percentile(a[:, j], 97.5))] for j, k in enumerate(traits)}
                 for s, a in (("men", dm), ("women", dw))}
    out["diff"] = {k: {"value": out["coef"]["men"][k] - out["coef"]["women"][k],
                       "ci": [float(np.percentile(dm[:, j] - dw[:, j], 2.5)), float(np.percentile(dm[:, j] - dw[:, j], 97.5))]}
                   for j, k in enumerate(traits)}
    # country by country: men's minus women's coefficient on resources (countries with at least 150 of each sex)
    gaps = []
    for c in cs:
        jm, jw = by["men"][c], by["women"][c]
        if len(jm) >= 150 and len(jw) >= 150:
            bm = np.linalg.lstsq(res["men"][2][jm], res["men"][3][jm], rcond=None)[0][0]
            bw = np.linalg.lstsq(res["women"][2][jw], res["women"][3][jw], rcond=None)[0][0]
            gaps.append(100 * (bm - bw))
    out["by_country"] = summarize(gaps, rng)
    return out


def ages(D, rng):
    ok = np.isfinite(D["mate_age"]) & np.isfinite(D["Age"]) & (D["Relationship_status01"] == 1) & (D["mate_age"] >= 18)
    rows = by_country(D, "Age", ok)
    gap = []
    for c, _, _ in rows:
        m = ok & (D["c"] == c)
        men = (D["Age"] - D["mate_age"])[m & (D["Sex01"] == 0)]         # man's age minus his partner's
        women = (D["mate_age"] - D["Age"])[m & (D["Sex01"] == 1)]       # partner's age minus the woman's
        gap.append(float(np.mean(np.concatenate([men, women]))))
    out = summarize(gap, rng)
    allm = ok & (D["c"] >= 0)
    out["people"] = int(allm.sum())
    return out


def screens(D, rng):
    ok = (D["c"] >= 0) & np.isfinite(D["Sex01"]) & np.isfinite(D["Relationship_status01"]) & np.isfinite(D["Age"]) & \
        np.isfinite(D["Time_social_media_1"]) & (D["Age"] >= 18) & (D["Age"] <= 60)
    hours = D["Time_social_media_1"] / 60.0
    out = {"n": int(ok.sum()), "mean_hours": float(hours[ok].mean()), "median_hours": float(np.median(hours[ok]))}
    for sname, sv in (("men", 0), ("women", 1), ("young", None)):
        m = ok & ((D["Sex01"] == sv) if sv is not None else (D["Age"] <= 34))
        i = np.where(m)[0]
        g = D["c"][i] * 2 + D["Sex01"][i].astype(int)
        X = np.column_stack([within(hours[i], g), within(D["Age"][i], g), within(D["Age"][i] ** 2, g)])
        y = within(1.0 - D["Relationship_status01"][i], g)
        b = 100 * np.linalg.lstsq(X, y, rcond=None)[0][0]
        cs = np.unique(D["c"][i])
        by = {c: np.where(D["c"][i] == c)[0] for c in cs}
        bs = []
        for _ in range(NB // 2):
            j = np.concatenate([by[c] for c in rng.choice(cs, len(cs))])
            bs.append(100 * np.linalg.lstsq(X[j], y[j], rcond=None)[0][0])
        out[sname] = {"n": int(len(i)), "points_per_hour": float(b), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      "single_pct": float(100 * (1 - D["Relationship_status01"][i].mean()))}
    return out


def equality(D, I, rng):
    """Across countries: mean attitude to gender equality against the size of each sex difference in ideals."""
    out = {}
    ge = {}
    ok = (D["c"] >= 0) & np.isfinite(D["Gender_Equality_Mean"])
    for c in np.unique(D["c"][ok]):
        ge[int(c)] = float(D["Gender_Equality_Mean"][ok & (D["c"] == c)].mean())

    def spearman(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])

    for name in ("resources", "looks"):
        ids, d = I[name]["ids"], np.array(I[name]["values"])
        keep = [k for k, c in enumerate(ids) if c in ge]
        x, y = np.array([ge[ids[k]] for k in keep]), d[keep]
        bs = []
        for _ in range(NB):
            j = rng.integers(0, len(x), len(x))
            bs.append(spearman(x[j], y[j]))
        out[name] = {"countries": int(len(x)), "rho": spearman(x, y), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                     "x": [float(v) for v in x], "y": [float(v) for v in y]}
    return out


def walter(rng):
    """Walter et al. 2020: ideals for five qualities, intelligence among them, in 45 countries (1-7 scales)."""
    path = os.path.join(ROOT, "data", "walter2020", "processed_data.csv")
    traits = ["kindness", "intelligence", "health", "physatt", "resources"]
    rows = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            rows.append([fnum(r["CIN"]), fnum(r["sex"]), fnum(r["age"]), fnum(r["mate_age"]), fnum(r["gggi"]), fnum(r["relstat"])] +
                        [fnum(r["ideal_" + t]) for t in traits])
    A = np.array(rows, float)
    cin, sex, age, mage, gggi = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]
    T = {t: A[:, 6 + k] for k, t in enumerate(traits)}
    # which code is men: the sex whose partners are younger on average
    okm = np.isfinite(mage) & np.isfinite(age) & np.isfinite(sex)
    male = 1.0 if np.mean((mage - age)[okm & (sex == 1)]) < np.mean((mage - age)[okm & (sex == 0)]) else 0.0
    out = {"n": int(len(A)), "countries": int(len(np.unique(cin[np.isfinite(cin)]))), "men": int(np.sum(sex == male)),
           "women": int(np.sum(sex == 1 - male)), "male_code": male, "means": {}, "d": {}, "country_means": {}}
    cs = np.unique(cin[np.isfinite(cin)])
    for t in traits:
        ok = np.isfinite(T[t]) & np.isfinite(sex)
        out["means"][t] = {"men": float(T[t][ok & (sex == male)].mean()), "women": float(T[t][ok & (sex != male)].mean())}
        d, cm = [], {"men": [], "women": []}
        for c in cs:
            m, w = T[t][ok & (cin == c) & (sex == male)], T[t][ok & (cin == c) & (sex != male)]
            if len(m) >= 30 and len(w) >= 30:
                sign = -1 if t == "physatt" else +1          # women minus men, except looks (men minus women)
                d.append(sign * cohen_d(w, m))
                cm["men"].append(float(m.mean()))
                cm["women"].append(float(w.mean()))
        out["d"][t] = summarize(d, rng)
        out["country_means"][t] = cm
    # rank of intelligence among the five, country by country and sex by sex
    rank = {"men": [], "women": []}
    above = {"men": 0, "women": 0, "cells": 0}
    for c in cs:
        for sname, sel in (("men", sex == male), ("women", sex != male)):
            mm = {t: np.nanmean(T[t][(cin == c) & sel]) for t in traits}
            order = sorted(traits, key=lambda t: -mm[t])
            rank[sname].append(order.index("intelligence") + 1)
            above[sname] += int(mm["intelligence"] > mm["physatt"] and mm["intelligence"] > mm["resources"])
        above["cells"] += 1
    out["intelligence_rank"] = {s: {"median": float(np.median(v)), "top2": int(np.sum(np.array(v) <= 2)), "countries": len(v)} for s, v in rank.items()}
    out["intelligence_above_looks_and_money"] = above
    # age difference between partners, and gender equality
    gap, gg = [], []
    for c in cs:
        m = okm & (cin == c)
        if m.sum() >= 50:
            diff = np.where(sex[m] == male, age[m] - mage[m], mage[m] - age[m])
            gap.append(float(diff.mean()))
            gg.append(float(np.nanmean(gggi[cin == c])))
    out["age_gap"] = summarize(gap, rng)
    x, y = np.array(gg), np.array(gap)
    keep = np.isfinite(x)

    def spearman(a, b):
        return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])

    bs = []
    for _ in range(NB):
        j = rng.integers(0, keep.sum(), keep.sum())
        bs.append(spearman(x[keep][j], y[keep][j]))
    out["age_gap_equality"] = {"countries": int(keep.sum()), "rho": spearman(x[keep], y[keep]),
                               "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                               "x": [float(v) for v in x[keep]], "y": [float(v) for v in y[keep]]}
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    D = load()
    out = {"_seed": SEED, "_written": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": int(len(D["Age"])), "countries": D["n_countries"],
           "men": int(np.sum(D["Sex01"] == 0)), "women": int(np.sum(D["Sex01"] == 1))}
    out["ideals"] = ideals(D, rng)
    out["budget"] = budget(D, rng)
    out["door"] = door(D, rng)
    out["ages"] = ages(D, rng)
    out["screens"] = screens(D, rng)
    out["equality"] = equality(D, out["ideals"], rng)
    out["walter"] = walter(rng)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "e20_world.json"), "w"))
    i = out["ideals"]
    print("E20", {"rows": out["rows"], "countries_res": i["resources"]["countries"], "res_pos_pct": round(i["resources"]["positive_pct"], 1),
                  "looks_pos_pct": round(i["looks"]["positive_pct"], 1), "door_men_res": round(out["door"]["coef"]["men"]["resources"], 2),
                  "door_women_res": round(out["door"]["coef"]["women"]["resources"], 2)})
