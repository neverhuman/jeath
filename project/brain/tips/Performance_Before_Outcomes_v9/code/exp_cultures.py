"""E18  Who decides a marriage, and how much coercion is coded, in the ethnographic record.

Reads the open D-PLACE release of the Standard Cross-Cultural Sample (SCCS), the Ethnographic Atlas (EA) and Binford's
hunter-gatherer data (fetched by fetch_dplace.py; not redistributed) and tabulates published codes.  Societies share
ancestry and neighbours, so the output is counts and shares; no significance is claimed across societies.

Run:  python3 -I code/exp_cultures.py        (isolated mode: it reads third-party data)
"""
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "data", "dplace", "mirror", "datasets")
SEED = 20261126
csv.field_size_limit(10**9)


def load(ds, want):
    """{var: {soc_id: code}} for the wanted variables; the first record of a society is kept."""
    want = set(want)
    out = defaultdict(dict)
    with open(os.path.join(BASE, ds, "data.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = r["var_id"]
            if v in want and r["code"] not in ("NA", "") and r["soc_id"] not in out[v]:
                out[v][r["soc_id"]] = r["code"]
    return out


def labels(ds, want):
    out = defaultdict(dict)
    with open(os.path.join(BASE, ds, "codes.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["var_id"] in want and r["code"] != "NA":
                out[r["var_id"]][r["code"]] = (r.get("name") or r.get("description") or "").strip()
    return out


def societies(ds):
    with open(os.path.join(BASE, ds, "societies.csv"), encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def dist(codes, groups=None):
    """Counts and shares of a {soc: code} map, optionally merged into named groups of codes."""
    c = Counter(codes.values())
    n = sum(c.values())
    out = {"n": n, "counts": {k: c[k] for k in sorted(c, key=lambda x: float(x))}}
    if groups:
        out["groups"] = {g: sum(c[k] for k in ks) for g, ks in groups.items()}
        out["share"] = {g: (100.0 * v / n if n else None) for g, v in out["groups"].items()}
    else:
        out["share"] = {k: 100.0 * v / n for k, v in out["counts"].items()}
    return out


def boot_mean_diff(a, b, rng, n=4000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = (a[idx] - b[idx]).mean(axis=1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def boot_corr(x, y, rng, n=4000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = rng.integers(0, len(x), (n, len(x)))
    r = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    return [float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))]


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    S_VARS = ["SCCS173", "SCCS174", "SCCS667", "SCCS739", "SCCS740", "SCCS971", "SCCS972", "SCCS870", "SCCS610", "SCCS744",
              "SCCS754", "SCCS612", "SCCS903", "SCCS239", "SCCS872"]
    E_VARS = ["EA035", "EA042", "EA006", "EA009"]
    B_VARS = ["B019", "B020", "B021"]
    S, E, B = load("SCCS", S_VARS), load("EA", E_VARS), load("Binford", B_VARS)
    LS, LE = labels("SCCS", set(S_VARS)), labels("EA", set(E_VARS))
    soc_s, soc_e, soc_b = societies("SCCS"), societies("EA"), societies("Binford")

    # Foragers: dominant subsistence gathering, fishing or hunting (EA042 codes 1-3), matched to the SCCS by cross-dataset id.
    xd_forager = {soc_e[s]["xd_id"] for s, c in E["EA042"].items() if c in ("1", "2", "3")}
    xd_coded = {soc_e[s]["xd_id"] for s in E["EA042"]}
    s_forager = {s for s, r in soc_s.items() if r["xd_id"] in xd_forager}
    s_coded = {s for s, r in soc_s.items() if r["xd_id"] in xd_coded}
    e_forager = {s for s, c in E["EA042"].items() if c in ("1", "2", "3")}

    def sub(m, keep):
        return {k: v for k, v in m.items() if k in keep}

    out = {"_seed": SEED, "_written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n": {"sccs": len(soc_s), "ea": len(soc_e), "binford": len(soc_b), "sccs_foragers": len(s_forager),
                 "ea_foragers": len(e_forager), "sccs_with_subsistence": len(s_coded)},
           "labels": {v: LS[v] for v in S_VARS if v in LS} | {v: LE[v] for v in E_VARS if v in LE}}

    # 1. Who chooses a spouse (Broude and Greene 1983): 1 alone; 2-3 individual with approval; 4 either; 5-6 parents.
    G_ARR = {"alone": ["1"], "with_approval": ["2", "3"], "either": ["4"], "parents": ["5", "6"]}
    arr = {}
    for sex, v in (("men", "SCCS739"), ("women", "SCCS740")):
        arr[sex] = dist(S[v], G_ARR)
        arr[sex + "_foragers"] = dist(sub(S[v], s_forager), G_ARR)
        arr[sex + "_cannot_object"] = sum(1 for c in S[v].values() if c == "6")
    both = [s for s in S["SCCS739"] if s in S["SCCS740"]]
    fm = [(float(S["SCCS739"][s]), float(S["SCCS740"][s])) for s in both]
    arr["paired"] = {"n": len(both), "daughter_less_free": sum(1 for m, w in fm if w > m),
                     "same": sum(1 for m, w in fm if w == m), "son_less_free": sum(1 for m, w in fm if w < m)}
    out["arrangement"] = arr

    # 2. Consent to marry (Frayser 1985): 1 necessary; 2 consulted; 3 not consulted.
    out["consent"] = {"women": dist(S["SCCS971"]), "men": dist(S["SCCS972"]),
                      "women_foragers": dist(sub(S["SCCS971"], s_forager))}

    # 3. Rape as coded by two teams: frequency (Broude and Greene 1976) and presence (Sanday 1981); attitude toward it.
    out["rape"] = {"frequency": dist(S["SCCS174"]), "frequency_foragers": dist(sub(S["SCCS174"], s_forager)),
                   "attitude": dist(S["SCCS173"]), "sanday": dist(S["SCCS667"]),
                   "sanday_foragers": dist(sub(S["SCCS667"], s_forager))}
    # frequency of rape against who chooses a daughter's spouse (counts only)
    xt = defaultdict(Counter)
    for s, c in S["SCCS174"].items():
        if s in S["SCCS740"]:
            g = next(k for k, ks in G_ARR.items() if S["SCCS740"][s] in ks)
            xt[g][c] += 1
    out["rape"]["by_arrangement"] = {g: dict(c) for g, c in xt.items()}

    # 4. Captives, wife-beating, divorce
    out["captives"] = dist(S["SCCS870"])
    out["wife_beating"] = dist(S["SCCS754"])
    out["divorce_access"] = dist(S["SCCS610"])
    out["divorce_access_foragers"] = dist(sub(S["SCCS610"], s_forager))
    out["divorce_frequency"] = dist(S["SCCS744"], {"common": ["1", "2", "4"], "moderate": ["3"], "rare": ["5"]})
    out["warrior_prestige"] = dist(S["SCCS903"])

    # 5. Age at first marriage
    out["relative_age"] = dist(S["SCCS612"], {"women_older": ["1"], "equal": ["2"], "men_older": ["3", "4", "5"]})
    ages = [(float(B["B020"][s]), float(B["B021"][s])) for s in B["B020"] if s in B["B021"]]
    am, af = np.array([a for a, _ in ages]), np.array([b for _, b in ages])
    out["binford_age"] = {"n": len(ages), "men": float(am.mean()), "women": float(af.mean()), "gap": float((am - af).mean()),
                          "gap_ci": boot_mean_diff(am, af, rng), "men_older_pct": float(100 * np.mean(am > af)),
                          "pairs": [[float(a_), float(b_)] for a_, b_ in ages]}
    tri = [(float(B["B020"][s]) - float(B["B021"][s]), float(B["B019"][s])) for s in B["B020"]
           if s in B["B021"] and s in B["B019"]]
    g, p = np.array([t[0] for t in tri]), np.array([t[1] for t in tri])
    out["binford_polygyny"] = {"n": len(tri), "median_pct": float(np.median(p)), "r_gap": float(np.corrcoef(g, p)[0, 1]),
                               "r_gap_ci": boot_corr(g, p, rng), "gap_low": float(g[p <= np.median(p)].mean()),
                               "gap_high": float(g[p > np.median(p)].mean())}

    # 6. Games (Roberts, Arth and Bush 1959; EA035): safe contests in the record
    G_GAMES = {"none": ["1"], "physical_skill": ["2", "4", "6", "8"], "chance": ["3", "4", "7", "8"], "strategy": ["5", "6", "7", "8"]}
    out["games"] = {"all": dist(E["EA035"], G_GAMES), "foragers": dist(sub(E["EA035"], e_forager), G_GAMES),
                    "others": dist({k: v for k, v in E["EA035"].items() if k in E["EA042"] and k not in e_forager}, G_GAMES)}

    # 7. Transfers at marriage and marriage system (Ethnographic Atlas)
    G_TR = {"bridewealth": ["1", "3"], "bride_service": ["2"], "exchange": ["4", "5"], "none": ["6"], "dowry": ["7"]}
    out["transfers"] = {"all": dist(E["EA006"], G_TR), "foragers": dist(sub(E["EA006"], e_forager), G_TR),
                        "others": dist({k: v for k, v in E["EA006"].items() if k in E["EA042"] and k not in e_forager}, G_TR)}
    out["marriage_system"] = dist(E["EA009"], {"monogamous": ["1"], "limited_polygyny": ["2"], "polygyny": ["3", "4", "5", "6"],
                                               "polyandrous": ["7"]})
    pw = np.array([float(v) for v in S["SCCS872"].values()])
    out["polygyny_women_pct"] = {"n": int(len(pw)), "median": float(np.median(pw)), "q25": float(np.percentile(pw, 25)),
                                 "q75": float(np.percentile(pw, 75))}

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "e18_cultures.json"), "w"), indent=1)
    a = out["arrangement"]
    print("E18", {"sccs": out["n"]["sccs"], "women_alone_pct": round(a["women"]["share"]["alone"], 1),
                  "men_alone_pct": round(a["men"]["share"]["alone"], 1), "rape_freq": out["rape"]["frequency"]["counts"],
                  "binford_gap": round(out["binford_age"]["gap"], 2)})
