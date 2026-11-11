"""Deterministic open-data and theoretical synthesis for performance-before-outcomes."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd

SEED = 20261111
OPEN_INPUTS = {
    "whyte2021_sexiness.dta": {
        "sha256": "a7c621f9aa6e68719f92010a354500d38bd3a26013368e6a87a2c87070f87f00",
        "url": "https://osf.io/download/f84pw/",
    },
    "lidborg2022_height.csv": {
        "sha256": "a4863503c2a14d1246e97eb39ab870967e0c260c0be4c49935234ed2d47cfdef",
        "url": "https://osf.io/download/chk7j/",
    },
    "lidborg2022_body.csv": {
        "sha256": "ced886958f8e7b06b71aebb2d6d339f3bb9f35bb876f8ab1ff6f356058311f53",
        "url": "https://osf.io/download/ytkhw/",
    },
    "kowal2025_Final_dataset.csv": {
        "sha256": "00f3292fa94280ad126724700fb439570fbada271595c4235968d00aa295120a",
        "url": "https://osf.io/download/67ceccfc02944524c51b23c2/",
    },
    "walter2020_processed_data.csv": {
        "sha256": "0509eb7bbaae75c08533d19e1e6fa3a343a4775d825d644068be63824dff5933",
        "url": "https://osf.io/download/5e277ef4350f6c000c20ba3f/",
    },
}

WHYTE_TRAITS = (
    ("trust", "trust_import", "Trust"),
    ("openness", "openness_import", "Openness"),
    ("emotional_connection", "emotionalconn_import", "Emotional connection"),
    ("intelligence", "intelligence_import", "Intelligence"),
    ("attractiveness", "attractive_import", "Physical attractiveness"),
    ("physical_build", "physicalbuild_import", "Physical build"),
    ("age", "age_import", "Age"),
    ("education", "education_import", "Education"),
    ("income", "income_import", "Income"),
)
WHYTE_AGE_BANDS = ((18, 29, "18-29"), (30, 39, "30-39"), (40, 49, "40-49"), (50, 64, "50-64"))

KOWAL_BUDGET_ITEMS = (
    ("kindness", "preference_imp_kind"),
    ("health", "preference_imp_health"),
    ("looks", "preference_imp_physat"),
    ("resources", "preference_imp_financ"),
    ("age", "preference_imp_age"),
    ("religion", "preference_imp_relig"),
)

# Published meta-analytic random-effects summaries from Havlicek et al. (2020, Proc. R. Soc. B, Results 3a-d & Fig. 2).
HAVLICEK_2020_MHC = (
    {"domain": "Genomic disassortative pairing", "k_effects": 54, "fisher_z": -0.015, "ci_95": [-0.044, 0.015]},
    {"domain": "Odor dissimilarity preference (non-HC)", "k_effects": 30, "fisher_z": 0.044, "ci_95": [-0.036, 0.123]},
    {"domain": "Sexual satisfaction with MHC-dissimilar partner", "k_effects": 13, "fisher_z": -0.078, "ci_95": [-0.180, 0.023]},
    {"domain": "Combined cross-domain synthesis", "k_effects": 63, "fisher_z": -0.004, "ci_95": [-0.038, 0.030]},
)

# Published composite comparisons from Eastwick et al. (2025, JPSP, Table 5; N = 2,933-2,935 men, 5,408-5,410 women).
EASTWICK_2025_COMPOSITES = (
    {"attribute": "Physical attractiveness composite", "stated_men": 6.91, "stated_women": 6.23, "eval_beta_men": 0.46, "eval_beta_women": 0.43},
    {"attribute": "Earning prospects composite", "stated_men": 5.18, "stated_women": 6.19, "eval_beta_men": 0.19, "eval_beta_women": 0.24},
    {"attribute": "Warmth & trustworthiness", "stated_men": 7.64, "stated_women": 8.01, "eval_beta_men": 0.49, "eval_beta_women": 0.52},
    {"attribute": "Intelligence & competence", "stated_men": 7.12, "stated_women": 7.42, "eval_beta_men": 0.34, "eval_beta_women": 0.37},
    {"attribute": "Charisma & humor", "stated_men": 7.05, "stated_women": 7.31, "eval_beta_men": 0.44, "eval_beta_women": 0.46},
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def midranks(values: np.ndarray) -> np.ndarray:
    _, inv, freq = np.unique(values, return_inverse=True, return_counts=True)
    return ((freq.cumsum() - 0.5 * freq) / len(values))[inv]


def resolve_input(name: str, inputs_dir: Path, fallback_dirs=()) -> Path:
    spec = OPEN_INPUTS[name]
    for folder in (inputs_dir, *fallback_dirs):
        candidate = folder / name
        if candidate.exists():
            digest = sha256_file(candidate)
            if digest != spec["sha256"]:
                raise ValueError(f"Checksum mismatch for {candidate}: {digest} != {spec['sha256']}")
            return candidate
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
    digest = sha256_bytes(payload)
    if digest != spec["sha256"]:
        raise ValueError(f"Downloaded checksum mismatch for {name}: {digest} != {spec['sha256']}")
    inputs_dir.mkdir(parents=True, exist_ok=True)
    dest = inputs_dir / name
    dest.write_bytes(payload)
    return dest


def mean_se_ci(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    n = int(len(v))
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "ci_95": [mean - 1.96 * se, mean + 1.96 * se]}


def analyze_whyte(dta_path: Path, rng: np.random.Generator) -> dict:
    df = pd.read_stata(dta_path, convert_categoricals=False)
    assert len(df) == 7060 and int(df["id"].nunique()) == 7060
    male = df["male_dummy"].to_numpy(dtype=np.float64) == 1.0
    hetero = df["hetero_dummy"].to_numpy(dtype=np.float64) == 1.0
    age = df["age"].to_numpy(dtype=np.float64)
    height = df["height"].to_numpy(dtype=np.float64)
    attr = df["attractiveness"].to_numpy(dtype=np.float64)
    health = df["health"].to_numpy(dtype=np.float64)
    edu = df["education"].to_numpy(dtype=np.float64)
    wage = df["wage_individ"].to_numpy(dtype=np.float64)
    partnered = 1.0 - df["marital_nopartner"].to_numpy(dtype=np.float64)
    offspring = df["offspring_dummy"].to_numpy(dtype=np.float64)
    assert int(male.sum()) == 4375 and int((~male).sum()) == 2685
    assert age.min() >= 18 and age.max() <= 64

    # 1. Stated importance profiles across 9 traits and 4 age bands
    trait_arrays = {key: df[col].to_numpy(dtype=np.float64) for key, col, _ in WHYTE_TRAITS}
    person_mean = np.mean(np.column_stack(list(trait_arrays.values())), axis=1)

    stated = {"pooled": {}, "by_age_band": {}, "paired_contrasts": {}, "hetero_sensitivity": {}}
    for sex_label, smask in (("Men", male), ("Women", ~male)):
        stated["pooled"][sex_label] = {
            key: mean_se_ci(trait_arrays[key][smask]) for key, _, _ in WHYTE_TRAITS
        }
        by_band = {}
        for lo, hi, bname in WHYTE_AGE_BANDS:
            bmask = smask & (age >= lo) & (age <= hi)
            by_band[bname] = {
                "n": int(bmask.sum()),
                "raw": {key: mean_se_ci(trait_arrays[key][bmask]) for key, _, _ in WHYTE_TRAITS},
                "person_centered": {
                    key: mean_se_ci(trait_arrays[key][bmask] - person_mean[bmask]) for key, _, _ in WHYTE_TRAITS
                },
            }
        stated["by_age_band"][sex_label] = by_band
        stated["paired_contrasts"][sex_label] = {
            "trust_minus_intelligence": mean_se_ci(trait_arrays["trust"][smask] - trait_arrays["intelligence"][smask]),
            "trust_minus_attractiveness": mean_se_ci(trait_arrays["trust"][smask] - trait_arrays["attractiveness"][smask]),
        }
        hmask = smask & hetero
        stated["hetero_sensitivity"][sex_label] = {
            "n": int(hmask.sum()),
            "trust_minus_intelligence": mean_se_ci(trait_arrays["trust"][hmask] - trait_arrays["intelligence"][hmask]),
            "trust_minus_attractiveness": mean_se_ci(trait_arrays["trust"][hmask] - trait_arrays["attractiveness"][hmask]),
        }

    # 2. Height Percentile vs. Attraction Percentile, Partnering, Offspring & Standardized Horse-Race
    height_vs_attraction = {}
    for sex_label, smask in (("Men", male), ("Women", ~male)):
        h_sex = height[smask]
        a_sex = attr[smask]
        p_sex = partnered[smask]
        o_sex = offspring[smask]
        hl_sex = health[smask]
        ed_sex = edu[smask]
        wg_sex = wage[smask]
        ag_sex = age[smask]

        h_pct = midranks(h_sex) * 100.0
        a_pct = midranks(a_sex) * 100.0

        brackets = []
        for code in sorted(np.unique(h_sex).astype(int).tolist()):
            cmask = h_sex == code
            nc = int(cmask.sum())
            if nc < 10:
                continue
            ap_vals = a_pct[cmask]
            boot_idx = rng.integers(0, nc, size=(2000, nc))
            boot_ap = ap_vals[boot_idx].mean(axis=1)
            boot_pt = p_sex[cmask][boot_idx].mean(axis=1)
            brackets.append({
                "height_code": int(code),
                "n": nc,
                "height_percentile": float(h_pct[cmask].mean()),
                "mean_attraction_percentile": float(ap_vals.mean()),
                "attraction_percentile_ci_95": [float(np.quantile(boot_ap, 0.025)), float(np.quantile(boot_ap, 0.975))],
                "mean_raw_attractiveness": float(a_sex[cmask].mean()),
                "partnered_share": float(p_sex[cmask].mean()),
                "partnered_ci_95": [float(np.quantile(boot_pt, 0.025)), float(np.quantile(boot_pt, 0.975))],
                "offspring_share": float(o_sex[cmask].mean()),
                "mean_health": float(hl_sex[cmask].mean()),
            })

        # Quadratic fit of attraction_percentile on standardized height (testing curvilinearity)
        hz = (h_sex - h_sex.mean()) / h_sex.std(ddof=1)
        X_quad = np.column_stack([np.ones_like(hz), hz, hz ** 2])
        beta_quad = np.linalg.lstsq(X_quad, a_pct, rcond=None)[0]
        n_s = len(hz)
        quad_draws = np.empty((2000, 3), dtype=np.float64)
        for b in range(2000):
            ii = rng.integers(0, n_s, n_s)
            quad_draws[b] = np.linalg.lstsq(X_quad[ii], a_pct[ii], rcond=None)[0]

        # Multivariate standardized comparison (height, health, income, education, age) on complete wage rows
        ok_w = np.isfinite(wg_sex)
        preds_raw = np.column_stack([h_sex[ok_w], hl_sex[ok_w], wg_sex[ok_w], ed_sex[ok_w], ag_sex[ok_w]])
        preds_z = (preds_raw - preds_raw.mean(axis=0)) / preds_raw.std(axis=0, ddof=1)
        X_mv = np.column_stack([np.ones(int(ok_w.sum())), preds_z])
        y_ap = a_pct[ok_w]
        y_pt = 100.0 * p_sex[ok_w]
        b_ap = np.linalg.lstsq(X_mv, y_ap, rcond=None)[0][1:]
        b_pt = np.linalg.lstsq(X_mv, y_pt, rcond=None)[0][1:]
        mv_ap_draws = np.empty((2000, 5), dtype=np.float64)
        mv_pt_draws = np.empty((2000, 5), dtype=np.float64)
        n_mv = int(ok_w.sum())
        for b in range(2000):
            ii = rng.integers(0, n_mv, n_mv)
            mv_ap_draws[b] = np.linalg.lstsq(X_mv[ii], y_ap[ii], rcond=None)[0][1:]
            mv_pt_draws[b] = np.linalg.lstsq(X_mv[ii], y_pt[ii], rcond=None)[0][1:]
        pred_names = ("height", "health", "income", "education", "age")

        height_vs_attraction[sex_label] = {
            "n": int(smask.sum()),
            "pearson_r_pct": float(np.corrcoef(h_pct, a_pct)[0, 1]),
            "spearman_r": float(np.corrcoef(midranks(h_sex), midranks(a_sex))[0, 1]),
            "quadratic_fit_pct_per_sd": {
                "linear": {"estimate": float(beta_quad[1]), "ci_95": np.quantile(quad_draws[:, 1], [0.025, 0.975]).tolist()},
                "quadratic": {"estimate": float(beta_quad[2]), "ci_95": np.quantile(quad_draws[:, 2], [0.025, 0.975]).tolist()},
            },
            "multivariate_standardized": {
                "n_complete": n_mv,
                "attraction_pct_per_sd": {
                    pname: {"estimate": float(b_ap[j]), "ci_95": np.quantile(mv_ap_draws[:, j], [0.025, 0.975]).tolist()}
                    for j, pname in enumerate(pred_names)
                },
                "partnered_pp_per_sd": {
                    pname: {"estimate": float(b_pt[j]), "ci_95": np.quantile(mv_pt_draws[:, j], [0.025, 0.975]).tolist()}
                    for j, pname in enumerate(pred_names)
                },
            },
            "by_bracket": brackets,
        }

    return {
        "sha256": OPEN_INPUTS["whyte2021_sexiness.dta"]["sha256"],
        "records": int(len(df)),
        "men": int(male.sum()),
        "women": int((~male).sum()),
        "age_range": [int(age.min()), int(age.max())],
        "stated_importance": stated,
        "height_vs_attraction": height_vs_attraction,
    }


def dersimonian_laird_r(rs: np.ndarray, ns: np.ndarray) -> dict:
    valid = (ns > 3) & (np.abs(rs) < 0.999)
    z = np.arctanh(rs[valid])
    v = 1.0 / (ns[valid] - 3.0)
    w = 1.0 / v
    z_fe = float(np.sum(w * z) / np.sum(w))
    q = float(np.sum(w * (z - z_fe) ** 2))
    df = max(1, len(z) - 1)
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    w_re = 1.0 / (v + tau2)
    z_re = float(np.sum(w_re * z) / np.sum(w_re))
    se_re = math.sqrt(1.0 / float(np.sum(w_re)))
    return {
        "re_mean_r": float(math.tanh(z_re)),
        "re_ci_95": [float(math.tanh(z_re - 1.96 * se_re)), float(math.tanh(z_re + 1.96 * se_re))],
        "tau": float(math.sqrt(tau2)),
    }


def summarize_lidborg_subset(rows: list) -> dict:
    rs_oriented = np.array([r["r_oriented"] for r in rows], dtype=np.float64)
    rs_raw = np.array([r["r_raw"] for r in rows], dtype=np.float64)
    ns = np.array([r["n"] for r in rows], dtype=np.float64)
    dl = dersimonian_laird_r(rs_oriented, ns)
    return {
        "k": int(len(rows)),
        "total_n": int(ns.sum()),
        "median_r": float(np.median(rs_oriented)),
        "iqr_r": np.quantile(rs_oriented, [0.25, 0.75]).tolist(),
        "n_weighted_r": float(np.sum(ns * rs_oriented) / ns.sum()),
        "raw_unflipped_median_r": float(np.median(rs_raw)),
        **dl,
    }


def analyze_lidborg(height_path: Path, body_path: Path) -> dict:
    out = {
        "height_sha256": OPEN_INPUTS["lidborg2022_height.csv"]["sha256"],
        "body_sha256": OPEN_INPUTS["lidborg2022_body.csv"]["sha256"],
    }
    for label, path in (("height", height_path), ("body_strength", body_path)):
        parsed = []
        with path.open(encoding="latin-1", newline="") as stream:
            for row in csv.DictReader(stream, delimiter=";"):
                if row["r"] == "" or row["N"] == "":
                    continue
                r_raw = float(row["r"])
                direction = float(row["Direction"]) if row["Direction"] else 1.0
                parsed.append({
                    "authors": row["Names"],
                    "predictor": row["Predictor"],
                    "outcome": row["Outcome"],
                    "grouping": row["Outcome_grouping"],
                    "sub_type": row["MAT_type"] if row["Outcome_grouping"] == "MAT" else row["REP_type"],
                    "high_fertility": row["High_fertility"] == "1",
                    "n": int(row["N"]),
                    "r_raw": r_raw,
                    "r_oriented": r_raw * direction,
                })
        mat_rows = [r for r in parsed if r["grouping"] == "MAT"]
        rep_rows = [r for r in parsed if r["grouping"] == "REP"]
        rep_ft = [r for r in rep_rows if r["sub_type"] == "FT"]
        rep_rs = [r for r in rep_rows if r["sub_type"] == "RS"]
        out[label] = {
            "total_effects": len(parsed),
            "mating": summarize_lidborg_subset(mat_rows),
            "reproduction_all": summarize_lidborg_subset(rep_rows),
            "reproduction_fertility": summarize_lidborg_subset(rep_ft) if rep_ft else None,
            "reproduction_surviving_offspring": summarize_lidborg_subset(rep_rs) if rep_rs else None,
            "effects": parsed,
        }
    return out


def harmonize_country_name(name: str) -> str:
    aliases = {
        "USA": "United States",
        "US": "United States",
        "United States of America": "United States",
        "UK": "United Kingdom",
        "Great Britain": "United Kingdom",
        "Korea, South": "South Korea",
        "Republic of Korea": "South Korea",
        "Russian Federation": "Russia",
        "Czechia": "Czech Republic",
        "Viet Nam": "Vietnam",
        "Turkiye": "Turkey",
    }
    cleaned = name.strip()
    return aliases.get(cleaned, cleaned)


def analyze_kowal(csv_path: Path, hon_countries: dict, rng: np.random.Generator) -> dict:
    flow_counts = [0] * 9
    sex_list, age_list, ctry_list, budget_list = [], [], [], []
    # Also retain broader rows for partnering ("door") check
    door_rows = []
    door_cols = ("self_resources", "self_physatt", "self_kind", "self_health")

    with csv_path.open(encoding="latin-1", newline="") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            flow_counts[0] += 1
            try:
                age = float(row["Age"])
            except (TypeError, ValueError):
                age = math.nan

            # Partnering ("door") cohort check (all adults 18-60 with Sex01, Relationship_status01, Country_live, 4 self-ratings)
            ctry_raw = (row.get("Country_live") or "").strip()
            if ctry_raw and ctry_raw != "NA" and np.isfinite(age) and 18 <= age <= 60:
                try:
                    s01 = float(row["Sex01"])
                    rel01 = float(row["Relationship_status01"])
                    s_vals = [float(row[c]) for c in door_cols]
                    ge_val = float(row["Gender_Equality_Mean"]) if row.get("Gender_Equality_Mean") not in ("", "NA", None) else math.nan
                    if s01 in (0.0, 1.0) and rel01 in (0.0, 1.0) and all(np.isfinite(s_vals)):
                        door_rows.append((harmonize_country_name(ctry_raw), int(s01), age, rel01, ge_val, *s_vals))
                except (TypeError, ValueError):
                    pass

            if not (np.isfinite(age) and 18 <= age <= 90):
                continue
            flow_counts[1] += 1
            if row["Sex"] not in ("0", "1"):
                continue
            flow_counts[2] += 1
            if row["Sexual_Orientation"] != "1":
                continue
            flow_counts[3] += 1
            try:
                bvals = [float(row[col]) for _, col in KOWAL_BUDGET_ITEMS]
            except (TypeError, ValueError):
                continue
            if not all(np.isfinite(bvals)):
                continue
            flow_counts[4] += 1
            if not all(0.0 <= v <= 30.0 for v in bvals):
                continue
            flow_counts[5] += 1
            if abs(sum(bvals) - 30.0) > 1e-6:
                continue
            flow_counts[6] += 1
            has_country = bool(ctry_raw and ctry_raw != "NA")
            if has_country:
                flow_counts[7] += 1
            sex_list.append(int(row["Sex"]))
            age_list.append(age)
            ctry_list.append(harmonize_country_name(ctry_raw) if has_country else "")
            budget_list.append(bvals)

    sex_arr = np.array(sex_list, dtype=np.int8)   # 0 = Men, 1 = Women
    age_arr = np.array(age_list, dtype=np.float64)
    ctry_arr = np.array(ctry_list, dtype=object)
    bud_arr = np.array(budget_list, dtype=np.float64)

    assert flow_counts[:8] == [117293, 107539, 105963, 86744, 67817, 67817, 67743, 67615]
    men_mask = sex_arr == 0
    women_mask = sex_arr == 1
    assert int(men_mask.sum()) == 23059 and int(women_mask.sum()) == 44684

    item_names = [k for k, _ in KOWAL_BUDGET_ITEMS]
    pooled = {
        "n_total": int(len(sex_arr)),
        "n_men": int(men_mask.sum()),
        "n_women": int(women_mask.sum()),
        "means": {
            name: {
                "men": float(bud_arr[men_mask, j].mean()),
                "women": float(bud_arr[women_mask, j].mean()),
                "women_minus_men": float(bud_arr[women_mask, j].mean() - bud_arr[men_mask, j].mean()),
            }
            for j, name in enumerate(item_names)
        },
    }

    # 53-country >=100 per gender analysis
    unique_ctries = sorted(set(c for c in ctry_list if c))
    eligible_53 = []
    country_details = {}
    for c in unique_ctries:
        cm = (ctry_arr == c) & men_mask
        cw = (ctry_arr == c) & women_mask
        nm, nw = int(cm.sum()), int(cw.sum())
        if nm >= 100 and nw >= 100:
            eligible_53.append(c)
            flow_counts[8] += nm + nw
            m_means = bud_arr[cm].mean(axis=0)
            w_means = bud_arr[cw].mean(axis=0)
            country_details[c] = {
                "n_men": nm,
                "n_women": nw,
                "men": {name: float(m_means[j]) for j, name in enumerate(item_names)},
                "women": {name: float(w_means[j]) for j, name in enumerate(item_names)},
                "women_minus_men": {name: float(w_means[j] - m_means[j]) for j, name in enumerate(item_names)},
            }
    assert len(eligible_53) == 53 and flow_counts[8] == 62185

    cross_country = {}
    for j, name in enumerate(item_names):
        m_vec = np.array([country_details[c]["men"][name] for c in eligible_53])
        w_vec = np.array([country_details[c]["women"][name] for c in eligible_53])
        diff_vec = w_vec - m_vec
        cross_country[name] = {
            "equal_country_mean_men": float(m_vec.mean()),
            "equal_country_mean_women": float(w_vec.mean()),
            "median_women_minus_men": float(np.median(diff_vec)),
            "iqr_women_minus_men": np.quantile(diff_vec, [0.25, 0.75]).tolist(),
            "women_greater_countries": int((diff_vec > 0).sum()),
            "men_greater_countries": int((diff_vec < 0).sum()),
            "cross_country_men_women_r": float(np.corrcoef(m_vec, w_vec)[0, 1]),
        }

    kindness_top_men = sum(1 for c in eligible_53 if max(country_details[c]["men"], key=country_details[c]["men"].get) == "kindness")
    kindness_top_women = sum(1 for c in eligible_53 if max(country_details[c]["women"], key=country_details[c]["women"].get) == "kindness")

    # 32-country common-age-distribution sensitivity (18-29, 30-44, 45-90, >=20 per gender per band)
    age_bands = ((18, 29), (30, 44), (45, 90))
    std_countries = []
    std_diffs = {name: [] for name in item_names}
    for c in eligible_53:
        cm = (ctry_arr == c) & men_mask
        cw = (ctry_arr == c) & women_mask
        if all(((cm & (age_arr >= lo) & (age_arr <= hi)).sum() >= 20 and
                (cw & (age_arr >= lo) & (age_arr <= hi)).sum() >= 20) for lo, hi in age_bands):
            std_countries.append(c)
            tot_c = (ctry_arr == c) & (men_mask | women_mask)
            weights = np.array([((tot_c & (age_arr >= lo) & (age_arr <= hi)).sum()) / tot_c.sum() for lo, hi in age_bands])
            for j, name in enumerate(item_names):
                m_std = sum(weights[b] * bud_arr[cm & (age_arr >= lo) & (age_arr <= hi), j].mean() for b, (lo, hi) in enumerate(age_bands))
                w_std = sum(weights[b] * bud_arr[cw & (age_arr >= lo) & (age_arr <= hi), j].mean() for b, (lo, hi) in enumerate(age_bands))
                std_diffs[name].append(float(w_std - m_std))
    assert len(std_countries) == 32

    # Partnering ("door") within-country standardized FE regressions
    d_ctry = np.array([r[0] for r in door_rows], dtype=object)
    d_sex = np.array([r[1] for r in door_rows], dtype=np.int8)
    d_age = np.array([r[2] for r in door_rows], dtype=np.float64)
    d_rel = np.array([r[3] for r in door_rows], dtype=np.float64)
    d_ge = np.array([r[4] for r in door_rows], dtype=np.float64)
    d_traits = np.array([r[5:] for r in door_rows], dtype=np.float64)
    _, d_cidx = np.unique(d_ctry, return_inverse=True)

    def demean_by_group(vals: np.ndarray, groups: np.ndarray, standardize: bool = False) -> np.ndarray:
        res = vals.astype(np.float64).copy()
        order = np.argsort(groups, kind="stable")
        g_sorted = groups[order]
        cuts = np.r_[0, np.flatnonzero(np.diff(g_sorted)) + 1, len(g_sorted)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            idx = order[a:b]
            res[idx] -= res[idx].mean()
            if standardize:
                sd = res[idx].std()
                res[idx] = res[idx] / sd if sd > 0 else 0.0
        return res

    door_trait_names = ("resources", "looks", "kindness", "health")
    door_summary = {"n": int(len(door_rows)), "by_sex": {}}
    common_cs = np.unique(d_cidx)
    c_to_pos = {int(c): pos for pos, c in enumerate(common_cs)}
    n_cs = len(common_cs)
    xx_by_sex, xy_by_sex, point_by_sex = {}, {}, {}
    for s_label, s_code in (("men", 0), ("women", 1)):
        sm = d_sex == s_code
        gc = d_cidx[sm]
        ag = d_age[sm]
        X = np.column_stack([demean_by_group(d_traits[sm, j], gc, True) for j in range(4)] +
                            [demean_by_group(ag, gc, False), demean_by_group(ag ** 2, gc, False)])
        y = demean_by_group(d_rel[sm], gc, False)
        xx = np.zeros((n_cs, 6, 6), dtype=np.float64)
        xy = np.zeros((n_cs, 6), dtype=np.float64)
        order = np.argsort(gc, kind="stable")
        g_sorted = gc[order]
        cuts = np.r_[0, np.flatnonzero(np.diff(g_sorted)) + 1, len(g_sorted)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            idx = order[a:b]
            pos = c_to_pos[int(g_sorted[a])]
            xx[pos] = X[idx].T @ X[idx]
            xy[pos] = X[idx].T @ y[idx]
        point_by_sex[s_label] = 100.0 * np.linalg.solve(xx.sum(axis=0), xy.sum(axis=0))[:4]
        xx_by_sex[s_label] = xx
        xy_by_sex[s_label] = xy

    resamples = rng.integers(0, n_cs, size=(1000, n_cs))
    boot_m = 100.0 * np.linalg.solve(xx_by_sex["men"][resamples].sum(axis=1), xy_by_sex["men"][resamples].sum(axis=1)[..., None])[..., 0][:, :4]
    boot_w = 100.0 * np.linalg.solve(xx_by_sex["women"][resamples].sum(axis=1), xy_by_sex["women"][resamples].sum(axis=1)[..., None])[..., 0][:, :4]
    for s_label, b_arr in (("men", boot_m), ("women", boot_w)):
        beta = point_by_sex[s_label]
        door_summary["by_sex"][s_label] = {
            tname: {"pp_per_sd": float(beta[j]), "ci_95": np.quantile(b_arr[:, j], [0.025, 0.975]).tolist()}
            for j, tname in enumerate(door_trait_names)
        }
    door_summary["men_minus_women"] = {
        tname: {
            "pp_per_sd": float(point_by_sex["men"][j] - point_by_sex["women"][j]),
            "ci_95": np.quantile(boot_m[:, j] - boot_w[:, j], [0.025, 0.975]).tolist(),
        }
        for j, tname in enumerate(door_trait_names)
    }

    # Cross-National Bridge between Kowal (53 countries) and HON (>=100/sex countries)
    ge_by_country = {}
    for c in eligible_53:
        m_ge = (d_ctry == c) & np.isfinite(d_ge)
        if m_ge.sum() >= 20:
            ge_by_country[c] = float(d_ge[m_ge].mean())

    hon_harmonized = {harmonize_country_name(k): v for k, v in hon_countries.items()}
    bridge_rows = []
    for c in eligible_53:
        if c in hon_harmonized:
            hc = hon_harmonized[c]
            bridge_rows.append({
                "country": c,
                "kowal_looks_men_minus_women": float(-country_details[c]["women_minus_men"]["looks"]),
                "kowal_resources_women_minus_men": float(country_details[c]["women_minus_men"]["resources"]),
                "kowal_religion_mean": float(0.5 * (country_details[c]["men"]["religion"] + country_details[c]["women"]["religion"])),
                "kowal_gender_equality": ge_by_country.get(c),
                "hon_female_to_male_ratio": hc["female_to_male_profile_ratio"],
                "hon_female_minus_male_like_rate_pp": hc["female_minus_male_like_rate_pp"],
                "hon_female_minus_male_age_drop_pp": hc["female_minus_male_age_drop_pp"],
            })

    # Also compute HON participation ratio vs like-rate sex gap correlation across all HON >=500/sex countries
    hon_500_rows = [
        (math.log10(v["female_to_male_profile_ratio"]), v["female_minus_male_like_rate_pp"])
        for v in hon_countries.values()
        if v["by_sex"]["Female"]["n"] >= 500 and v["by_sex"]["Male"]["n"] >= 500
    ]
    x_rat = np.array([r[0] for r in hon_500_rows])
    y_gap = np.array([r[1] for r in hon_500_rows])

    return {
        "sha256": OPEN_INPUTS["kowal2025_Final_dataset.csv"]["sha256"],
        "inclusion_flow": {
            "source_rows": flow_counts[0],
            "age_18_90": flow_counts[1],
            "man_or_woman": flow_counts[2],
            "heterosexual": flow_counts[3],
            "six_items_observed": flow_counts[4],
            "items_in_0_30": flow_counts[5],
            "sum_equals_30": flow_counts[6],
            "country_observed": flow_counts[7],
            "countries_ge_100_per_gender_respondents": flow_counts[8],
            "countries_ge_100_per_gender_count": len(eligible_53),
        },
        "pooled_budget": pooled,
        "cross_country_53": {
            "countries": len(eligible_53),
            "kindness_top_men_countries": kindness_top_men,
            "kindness_top_women_countries": kindness_top_women,
            "by_item": cross_country,
            "country_means": country_details,
        },
        "age_standardized_32": {
            "countries": len(std_countries),
            "looks_men_greater_countries": sum(1 for v in std_diffs["looks"] if v < 0),
            "resources_women_greater_countries": sum(1 for v in std_diffs["resources"] if v > 0),
            "median_women_minus_men": {name: float(np.median(std_diffs[name])) for name in item_names},
        },
        "partnering_slopes": door_summary,
        "kowal_hon_bridge": {
            "overlapping_countries": len(bridge_rows),
            "hon_500_log_participation_vs_like_gap_r": float(np.corrcoef(x_rat, y_gap)[0, 1]),
            "rows": bridge_rows,
        },
    }


def analyze_walter(csv_path: Path) -> dict:
    traits = ("kindness", "intelligence", "health", "physatt", "resources")
    rows = []
    with csv_path.open(encoding="latin-1", newline="") as stream:
        for r in csv.DictReader(stream):
            try:
                sex = int(r["sex"])
            except (TypeError, ValueError):
                continue
            if sex not in (0, 1):
                continue
            cin = int(r["CIN"])
            vals = []
            for t in traits:
                try:
                    vals.append(float(r["ideal_" + t]))
                except (TypeError, ValueError):
                    vals.append(math.nan)
            try:
                age_gap = float(r["mate_age"]) - float(r["age"])
            except (TypeError, ValueError):
                age_gap = math.nan
            rows.append((cin, sex, age_gap, *vals))
    arr = np.array(rows, dtype=np.float64)
    # Verified above: sex == 0 is Women (mate_age - age = +1.45), sex == 1 is Men (mate_age - age = -3.04).
    women = arr[:, 1] == 0.0
    men = arr[:, 1] == 1.0
    means = {}
    for j, t in enumerate(traits):
        col = arr[:, 3 + j]
        means[t] = {
            "women": float(np.nanmean(col[women])),
            "men": float(np.nanmean(col[men])),
            "women_minus_men": float(np.nanmean(col[women]) - np.nanmean(col[men])),
        }
    return {
        "sha256": OPEN_INPUTS["walter2020_processed_data.csv"]["sha256"],
        "records": int(len(arr)),
        "countries": int(len(np.unique(arr[:, 0]))),
        "n_women": int(women.sum()),
        "n_men": int(men.sum()),
        "mean_mate_minus_own_age_years": {
            "women": float(np.nanmean(arr[women, 2])),
            "men": float(np.nanmean(arr[men, 2])),
        },
        "means_1_to_7": means,
    }


def analyze_two_way_jury(rng: np.random.Generator) -> dict:
    # Proposition 20 benchmark parameterization:
    # Target variance = 0.30, shared persistent framing/halo = 0.10, observer-target = 0.25, context-target = 0.25, residual = 0.10 (sum = 1.00).
    s_theta, s_h, s_b, s_c, s_eps = 0.30, 0.10, 0.25, 0.25, 0.10

    def closed_r2(m: float, k: float, sh: float = s_h) -> float:
        return s_theta / (s_theta + sh + s_b / m + s_c / k + s_eps / (m * k))

    m_grid = [1, 2, 4, 8, 16, 32, 64, 1000]
    k_grid = [1, 2, 4, 8, 16, 32]
    surface = {
        str(k): {str(m): float(closed_r2(m, k)) for m in m_grid}
        for k in k_grid
    }
    # Monte Carlo check at (m=8, k=1) vs (m=8, k=8) on N=25,000 targets
    n_sim = 25000
    theta = rng.normal(0.0, math.sqrt(s_theta), size=n_sim)
    halo = rng.normal(0.0, math.sqrt(s_h), size=n_sim)
    mc_checks = {}
    for m, k in ((1, 1), (8, 1), (64, 1), (8, 8)):
        b_mean = rng.normal(0.0, math.sqrt(s_b / m), size=n_sim)
        c_mean = rng.normal(0.0, math.sqrt(s_c / k), size=n_sim)
        e_mean = rng.normal(0.0, math.sqrt(s_eps / (m * k)), size=n_sim)
        y_bar = theta + halo + b_mean + c_mean + e_mean
        r2_mc = float(np.corrcoef(theta, y_bar)[0, 1] ** 2)
        mc_checks[f"m{m}_k{k}"] = {"closed_form": float(closed_r2(m, k)), "monte_carlo": r2_mc}

    return {
        "parameters": {"sigma2_theta": s_theta, "sigma2_h": s_h, "sigma2_b": s_b, "sigma2_c": s_c, "sigma2_eps": s_eps},
        "single_context_asymptote_r2": float(s_theta / (s_theta + s_h + s_c)),
        "multi_context_halo_ceiling_r2": float(s_theta / (s_theta + s_h)),
        "halo_free_ceiling_r2": 1.0,
        "surface_r2": surface,
        "monte_carlo_verification": mc_checks,
    }


def load_v9_anchors(root_dir: Path) -> dict:
    v9_dir = root_dir / "project/brain/tips/Performance_Before_Outcomes_v9"
    res_dir = v9_dir / "results"
    code_dir = v9_dir / "code"
    needed = ("e13_fossils.json", "e15_polymath.json", "e16_verify.json", "e17_speed.json", "e18_cultures.json", "e19_village.json", "gss.json")
    hashes = {name: sha256_file(res_dir / name) for name in needed}
    loaded = {name[:-5]: json.loads((res_dir / name).read_text()) for name in needed}

    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from load_fossils import du, will_full  # type: ignore

    early = [u for u in du() if u[1] >= 0.95]
    fossils_grouped = {
        "aus": [[float(u[1]), float(u[3])] for u in early if u[2].startswith(("Au.", "P."))],
        "eh": [[float(u[1]), float(u[3])] for u in early if u[2].startswith("H.")],
        "mp": [[float(w["age"]), float(w["ecv"])] for w in will_full() if w["taxon"] == "Mid-Pleistocene Homo"],
        "nea": [[float(w["age"]), float(w["ecv"])] for w in will_full() if w["taxon"] == "Neanderthals"],
        "sap": [[float(w["age"]), float(w["ecv"])] for w in will_full() if w["taxon"] == "Pleistocene Homo sapiens"],
    }

    return {
        "results_sha256": hashes,
        "fossils": {
            "step": loaded["e13_fossils"]["step"],
            "bps": loaded["e13_fossils"]["models"]["stasis_step"]["bps"],
            "bp_ci": loaded["e13_fossils"]["bp_ci"],
            "groups": fossils_grouped,
        },
        "speed_jury_curves": {
            k: loaded["e17_speed"]["jury"][k]["curve"]
            for k in ("attr", "fun", "intel", "sinc", "amb")
        },
        "polymath": {
            "aim_common": {k: float(v["ratio"][0]) for k, v in loaded["e15_polymath"]["aim_common"].items()},
            "aim_division": {k: float(v["ratio"][0]) for k, v in loaded["e15_polymath"]["aim_division"].items()},
        },
        "staging": {
            k: {"R_capacity": float(v["R_b"][0]), "R_delivery": float(v["R_p"][0])}
            for k, v in loaded["e16_verify"]["staging"].items()
        },
        "gss_three_episodes_income": loaded["gss"]["three_episodes"]["income"],
        "village_births_periods": {
            p: {
                "never_married": float(v["all"]["never_married"]),
                "children": float(v["all"]["children"]),
            }
            for p, v in loaded["e19_village"]["births"]["periods"].items()
        },
        "ethnographic_arranged": {
            "sccs_societies": int(loaded["e18_cultures"]["n"]["sccs"]),
            "binford_foragers": int(loaded["e18_cultures"]["n"]["binford"]),
        },
    }


def analyze_open(inputs_dir: Path, root_dir: Path, hon_countries: dict) -> dict:
    rng = np.random.default_rng(SEED)
    shared_root_inputs = Path("/home/ubuntu/jeath/target/performance-before-outcomes/inputs")
    fallbacks = (shared_root_inputs,) if shared_root_inputs != inputs_dir else ()

    whyte_path = resolve_input("whyte2021_sexiness.dta", inputs_dir, fallbacks)
    lidborg_h_path = resolve_input("lidborg2022_height.csv", inputs_dir, fallbacks)
    lidborg_b_path = resolve_input("lidborg2022_body.csv", inputs_dir, fallbacks)
    kowal_path = resolve_input("kowal2025_Final_dataset.csv", inputs_dir, fallbacks)
    walter_path = resolve_input("walter2020_processed_data.csv", inputs_dir, fallbacks)

    return {
        "seed": SEED,
        "whyte_2021": analyze_whyte(whyte_path, rng),
        "lidborg_2022": analyze_lidborg(lidborg_h_path, lidborg_b_path),
        "kowal_2025": analyze_kowal(kowal_path, hon_countries, rng),
        "walter_2020": analyze_walter(walter_path),
        "two_way_jury": analyze_two_way_jury(rng),
        "published_benchmarks": {
            "havlicek_2020_mhc": list(HAVLICEK_2020_MHC),
            "eastwick_2025_composites": list(EASTWICK_2025_COMPOSITES),
        },
        "v9_anchors": load_v9_anchors(root_dir),
    }
