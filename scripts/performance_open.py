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
    "joel2020_SatisfactionMetaAnalyses.csv": {
        "sha256": "8872155d4bc961a91eb66ddf3b7155542e22467d1ebdf6a7bd307eeaa4bf5d25",
        "url": "https://osf.io/download/vtwhc/",
    },
    "joel2020_CommitmentMetaAnalyses.csv": {
        "sha256": "464eb9d7c89c5c63c254b8f42fe6287b9cd806402e5440562063c071fc9af664",
        "url": "https://osf.io/download/4zypd/",
    },
    "dplace_sccs_data.csv": {
        "sha256": "1e37ad6af1ae62c2108470137ca4eb03d07247a9fd12da2251de3673e8307541",
        "url": "https://raw.githubusercontent.com/D-PLACE/dplace-data/9bfed2c8c206be00f55f71516f262bbca2234e5a/datasets/SCCS/data.csv",
    },
    "dplace_sccs_codes.csv": {
        "sha256": "3d2396b9026719866f480671fab1389dade026139a5e251454b212c6ef31f7d5",
        "url": "https://raw.githubusercontent.com/D-PLACE/dplace-data/9bfed2c8c206be00f55f71516f262bbca2234e5a/datasets/SCCS/codes.csv",
    },
    "hcmst2017_public_v1.1.dta": {
        "sha256": "8f21fe52e1d9fb5ad286fc68ef9ac74c4a4ade8fbeca5b45a64087083b06a64c",
        "url": "https://raw.githubusercontent.com/shivi-a/how_couples_meet/master/HCMST%202017%20fresh%20sample%20for%20public%20sharing%20draft%20v1.1.dta",
    },
    "nhs_discography_metadata.csv": {
        "sha256": "36fb9f0b6d76dd5c1b3568650bb80488c9b2c8c37240232aaee741bfc6004842",
        "url": "https://raw.githubusercontent.com/themusiclab/nhs/master/data/nhs/NHSDiscography_Metadata.csv",
    },
    "nhs_naive_listeners.csv": {
        "sha256": "78299732962b46f6dd56412d04825e9a621f136ae666db53364602adb72a5e7c",
        "url": "https://raw.githubusercontent.com/themusiclab/nhs/master/data/nhs/NHSDiscography_NaiveListeners.csv",
    },
    "nhs_ethnography_prim.csv": {
        "sha256": "16ee6511b4ad01fcf0ce51d6e1a7156f4c3a00887077bf7a5313f6367d09fdbd",
        "url": "https://raw.githubusercontent.com/themusiclab/nhs/master/data/nhs/NHSEthnography_AnnotatePrim.csv",
    },
    "nhs_cultures_metadata.csv": {
        "sha256": "a7cfaba8a29276237e17e988f85858b4057c99852fba0997c45ee92fe955f33f",
        "url": "https://raw.githubusercontent.com/themusiclab/nhs/master/data/nhs/NHSCultures_Metadata.csv",
    },
    "okcupid_profiles_revised.csv.zip": {
        "sha256": "a31dbac13ba2e0fc6169c286cfa9751513cc73b888cd7fe08f311463847f075c",
        "url": "https://raw.githubusercontent.com/rudeboybert/JSE_OkCupid/master/profiles_revised.csv.zip",
    },
    "okcupid_essays_revised_and_shuffled.csv.zip": {
        "sha256": "3de4b64da2d4e8db555c8a9b2357b01c17a881b6267125d55565ea0e758090ef",
        "url": "https://raw.githubusercontent.com/rudeboybert/JSE_OkCupid/master/essays_revised_and_shuffled.csv.zip",
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

# Published ethnographic romantic-love benchmark from Jankowiak & Fischer (1992, Ethnology 31(2):149-155; 166 SCCS societies).
JANKOWIAK_FISCHER_1992 = {
    "societies_sampled": 166,
    "romantic_love_documented": 147,
    "share_documented": 147 / 166,
}

# Published developmental brain glucose metabolism trajectory from Kuzawa et al. (2014, PNAS 111(36):13010-13015, Fig. 1 & Table S2).
KUZAWA_2014_BRAIN_METABOLISM = (
    {"stage": "Neonate (birth)", "age_years": 0.0, "brain_rmr_pct": 52.5, "brain_der_pct": 35.6},
    {"stage": "Infancy (6 mo)", "age_years": 0.5, "brain_rmr_pct": 36.8, "brain_der_pct": 26.4},
    {"stage": "Toddler (1.5 yr)", "age_years": 1.5, "brain_rmr_pct": 53.2, "brain_der_pct": 36.5},
    {"stage": "Early childhood peak (4.5 yr)", "age_years": 4.5, "brain_rmr_pct": 66.3, "brain_der_pct": 43.2},
    {"stage": "Juvenile (8 yr)", "age_years": 8.0, "brain_rmr_pct": 54.8, "brain_der_pct": 34.0},
    {"stage": "Adolescence (14 yr)", "age_years": 14.0, "brain_rmr_pct": 33.5, "brain_der_pct": 19.8},
    {"stage": "Adult (22 yr)", "age_years": 22.0, "brain_rmr_pct": 20.5, "brain_der_pct": 11.8},
)

# Comparative primate & hominin energetic ceilings (Isler & van Schaik 2012 Current Anthropology / J. Hum. Evol.; Navarrete et al. 2011 Nature).
PRIMATE_GRAY_CEILING_BENCHMARKS = {
    "great_ape_solitary_mother_mean_cm3": {"Pan_troglodytes": 383.0, "Pongo_pygmaeus": 405.0, "Gorilla_gorilla": 504.0},
    "solitary_maternal_gray_ceiling_cm3": 650.0,
    "early_homo_raw_allomaternal_plateau_cm3": 908.0,
    "late_homo_cooking_biparental_mean_cm3": 1385.0,
}

KOWAL_LIFE_BRACKETS = (
    (18, 24, "18-24"),
    (25, 29, "25-29"),
    (30, 34, "30-34"),
    (35, 39, "35-39"),
    (40, 49, "40-49"),
    (50, 59, "50-59"),
    (60, 79, "60-79"),
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

    # 3. Partnered vs. Single Happiness Lift (0-100 scale, raw & age-adjusted; dedicated RNG preserves legacy draws)
    rng_w_ext = np.random.default_rng(SEED + 201)
    happiness = df["happiness"].to_numpy(dtype=np.float64)
    happiness_by_partnered = {}
    for g_label, g_mask in (("Men", male), ("Women", ~male), ("Pooled", np.ones(len(df), dtype=bool))):
        ok_h = g_mask & np.isfinite(happiness) & np.isfinite(partnered) & np.isfinite(age)
        h_g = happiness[ok_h]
        p_g = partnered[ok_h]
        a_g = age[ok_h]
        part_vals = h_g[p_g == 1.0]
        sing_vals = h_g[p_g == 0.0]
        raw_diff = float(part_vals.mean() - sing_vals.mean())
        Xh = np.column_stack([np.ones(len(h_g)), p_g, a_g, a_g ** 2])
        adj_lift = float(np.linalg.lstsq(Xh, h_g, rcond=None)[0][1])
        adj_draws = np.empty(1500, dtype=np.float64)
        raw_draws = np.empty(1500, dtype=np.float64)
        nh = len(h_g)
        for b in range(1500):
            ii = rng_w_ext.integers(0, nh, nh)
            pb = p_g[ii] == 1.0
            raw_draws[b] = h_g[ii][pb].mean() - h_g[ii][~pb].mean()
            adj_draws[b] = np.linalg.lstsq(Xh[ii], h_g[ii], rcond=None)[0][1]
        happiness_by_partnered[g_label] = {
            "partnered": mean_se_ci(part_vals),
            "single": mean_se_ci(sing_vals),
            "raw_lift_pts": {"estimate": raw_diff, "ci_95": np.quantile(raw_draws, [0.025, 0.975]).tolist()},
            "age_adjusted_lift_pts": {"estimate": adj_lift, "ci_95": np.quantile(adj_draws, [0.025, 0.975]).tolist()},
        }

    return {
        "sha256": OPEN_INPUTS["whyte2021_sexiness.dta"]["sha256"],
        "records": int(len(df)),
        "men": int(male.sum()),
        "women": int((~male).sum()),
        "age_range": [int(age.min()), int(age.max())],
        "stated_importance": stated,
        "height_vs_attraction": height_vs_attraction,
        "happiness_by_partnered": happiness_by_partnered,
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
    # Also retain broader rows for partnering ("door") check and inside-bond / love / life-course analyses
    door_rows = []
    door_cols = ("self_resources", "self_physatt", "self_kind", "self_health")
    mate_cols = ("mate_resources", "mate_physatt", "mate_kind", "mate_health")
    love_cols = tuple(f"Love_{i}" for i in range(1, 16))
    assort_Self = ("self_religion", "self_health", "self_resources", "self_physatt", "self_kind")
    assort_Mate = ("mate_religion", "mate_health", "mate_resources", "mate_physatt", "mate_kind")

    bond_rows = []
    love_rows = []
    mwl_rows = []
    gap_rows = []
    assort_rows = []
    sm_rows = []
    qoa_rows = []
    chk_surplus_rows = []
    lt_arm_rows = []
    st_arm_rows = []
    jeal_rows = []
    hom_rows = []
    rev_rows = []
    eff_soi_rows = []

    pref5_cols = ("preference_imp_physat", "preference_imp_financ", "preference_imp_kind", "preference_imp_health", "preference_imp_relig")
    mate5_cols = ("mate_physatt", "mate_resources", "mate_kind", "mate_health", "mate_religion")
    self5_cols = ("self_physatt", "self_resources", "self_kind", "self_health", "self_religion")
    lt6_cols = ("ideallt_kind", "ideallt_health", "ideallt_physatt", "ideallt_resources", "ideallt_age", "ideallt_religion")
    st6_cols = ("idealst_kind", "idealst_health", "idealst_physatt", "idealst_resources", "idealst_age", "idealst_religion")

    def fnum(val):
        if val in ("", "NA", None):
            return math.nan
        try:
            return float(val)
        except (TypeError, ValueError):
            return math.nan

    with csv_path.open(encoding="latin-1", newline="") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            flow_counts[0] += 1
            age = fnum(row.get("Age"))
            ctry_raw = (row.get("Country_live") or "").strip()
            has_ctry_any = bool(ctry_raw and ctry_raw != "NA")
            ctry_clean = harmonize_country_name(ctry_raw) if has_ctry_any else ""

            # Partnering ("door") cohort check (all adults 18-60 with Sex01, Relationship_status01, Country_live, 4 self-ratings)
            if has_ctry_any and np.isfinite(age) and 18 <= age <= 60:
                s01 = fnum(row.get("Sex01"))
                rel01 = fnum(row.get("Relationship_status01"))
                s_vals = [fnum(row.get(c)) for c in door_cols]
                ge_val = fnum(row.get("Gender_Equality_Mean"))
                if s01 in (0.0, 1.0) and rel01 in (0.0, 1.0) and all(np.isfinite(s_vals)):
                    door_rows.append((ctry_clean, int(s01), age, rel01, ge_val, *s_vals))

            # Extended inside-bond, love-arc, social-media, assortative & partner-age analyses (adults 18-79)
            if np.isfinite(age) and 18 <= age <= 79:
                s01_ext = fnum(row.get("Sex01"))
                prqc_v = fnum(row.get("PRQC_mean"))
                if not (1.0 <= prqc_v <= 7.0):
                    prqc_v = math.nan
                inf_v = fnum(row.get("infidelity"))
                if inf_v not in (0.0, 1.0):
                    inf_v = math.nan
                qoa_v = fnum(row.get("qoa"))
                if not (1.0 <= qoa_v <= 7.0):
                    qoa_v = math.nan

                if np.isfinite(qoa_v):
                    qoa_rows.append((qoa_v, prqc_v, inf_v))

                if has_ctry_any and s01_ext in (0.0, 1.0):
                    m_vals = [fnum(row.get(c)) for c in mate_cols]
                    if all(np.isfinite(v) and 1.0 <= v <= 11.0 for v in m_vals):
                        bond_rows.append((ctry_clean, int(s01_ext), age, prqc_v, inf_v, *m_vals))
                    sm_min = fnum(row.get("Time_social_media_1"))
                    if np.isfinite(sm_min) and 0.0 <= sm_min <= 360.0:
                        sm_rows.append((ctry_clean, int(s01_ext), age, sm_min / 60.0, qoa_v, prqc_v, inf_v))

                    pf5 = [fnum(row.get(c)) for c in pref5_cols]
                    mt5 = [fnum(row.get(c)) for c in mate5_cols]
                    sf5 = [fnum(row.get(c)) for c in self5_cols]
                    if np.isfinite(prqc_v) and all(0.0 <= v <= 30.0 for v in pf5) and all(1.0 <= v <= 11.0 for v in mt5) and all(1.0 <= v <= 11.0 for v in sf5):
                        chk_surplus_rows.append((
                            ctry_clean,
                            int(s01_ext),
                            age,
                            prqc_v,
                            inf_v,
                            fnum(row.get("Gender_Equality_Mean")),
                            fnum(row.get("Individualism_Mean")),
                            *pf5,
                            *mt5,
                            *sf5,
                        ))

                    lt6 = [fnum(row.get(c)) for c in lt6_cols]
                    if all(np.isfinite(v) and 1.0 <= v <= 11.0 for v in lt6):
                        lt_arm_rows.append((int(s01_ext), *lt6))
                    st6 = [fnum(row.get(c)) for c in st6_cols]
                    if all(np.isfinite(v) and 1.0 <= v <= 11.0 for v in st6):
                        st_arm_rows.append((int(s01_ext), *st6))
                    je_v, js_v = fnum(row.get("jealemo")), fnum(row.get("jealsex"))
                    if np.isfinite(je_v) and np.isfinite(js_v) and 1.0 <= je_v <= 7.0 and 1.0 <= js_v <= 7.0:
                        jeal_rows.append((int(s01_ext), je_v, js_v))

                if row.get("Sex") in ("0", "1"):
                    sx_i = int(row["Sex"])
                    rstat_rec = (row.get("Relationship_status_recoded") or "").strip()
                    eff_soi_rows.append((
                        sx_i,
                        age,
                        rstat_rec,
                        fnum(row.get("Attractiveness_Time_Sum")),
                        fnum(row.get("Time_A_exercise_1")),
                        fnum(row.get("Time_A_makeup_1")),
                        fnum(row.get("Time_A_dress_1")),
                        fnum(row.get("soi_desire")),
                        fnum(row.get("soi_attitude")),
                        fnum(row.get("soi_behave")),
                    ))
                    if has_ctry_any and np.isfinite(prqc_v):
                        mk_v = fnum(row.get("mate_kind"))
                        if np.isfinite(mk_v) and 1.0 <= mk_v <= 11.0:
                            hom_rows.append((
                                ctry_clean,
                                sx_i,
                                age,
                                prqc_v,
                                mk_v,
                                fnum(row.get("cot_politics")),
                                fnum(row.get("cpt_politics")),
                                fnum(row.get("cot_educ")),
                                fnum(row.get("cpt_educ")),
                                fnum(row.get("cot_class")),
                                fnum(row.get("cpt_class")),
                            ))
                        sk_v, mp_v, sp_v, mr_v, sr_v = (
                            fnum(row.get("self_kind")),
                            fnum(row.get("mate_physatt")),
                            fnum(row.get("self_physatt")),
                            fnum(row.get("mate_resources")),
                            fnum(row.get("self_resources")),
                        )
                        if all(np.isfinite(v) and 1.0 <= v <= 11.0 for v in (mk_v, sk_v, mp_v, sp_v, mr_v, sr_v)):
                            rev_rows.append((
                                ctry_clean, sx_i, age, prqc_v, inf_v,
                                sk_v, mk_v, sp_v, mp_v, sr_v, mr_v,
                            ))
                    rl_v = fnum(row.get("Relationship_length"))
                    if np.isfinite(rl_v) and 0.0 < rl_v <= 60.0 and rl_v <= age - 14.0 and np.isfinite(prqc_v):
                        l_vals = [fnum(row.get(c)) for c in love_cols]
                        if all(np.isfinite(v) and 1.0 <= v <= 5.0 for v in l_vals):
                            love_rows.append((
                                sx_i,
                                age,
                                rl_v,
                                sum(l_vals[0:5]) / 5.0,
                                sum(l_vals[5:10]) / 5.0,
                                sum(l_vals[10:15]) / 5.0,
                                prqc_v,
                            ))
                    if row.get("Sexual_Orientation") == "1":
                        mwl_v = fnum(row.get("Love_would_you_marry_1"))
                        if np.isfinite(mwl_v) and 0.0 <= mwl_v <= 100.0:
                            mwl_rows.append((sx_i, mwl_v))
                        mage_v = fnum(row.get("mate_age"))
                        if np.isfinite(mage_v) and 18.0 <= mage_v <= 90.0:
                            gap_rows.append((sx_i, age, mage_v - age))
                        a_pair = [fnum(row.get(c)) for c in (*assort_Self, *assort_Mate)]
                        if any(np.isfinite(a_pair[j]) and np.isfinite(a_pair[j + 5]) for j in range(5)):
                            assort_rows.append(a_pair)

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

    # --- Extended Kowal Life-Course, Pair-Bond Reversal, Love Arc, & Option-Illusion Summaries ---
    rng_k_ext = np.random.default_rng(SEED + 301)

    # 1. 30-Point Partner Budgets Across Adult Age (7 brackets x sex, n >= 100)
    budget_by_age = {}
    for lo, hi, bname in KOWAL_LIFE_BRACKETS:
        b_m = men_mask & (age_arr >= lo) & (age_arr <= hi)
        b_w = women_mask & (age_arr >= lo) & (age_arr <= hi)
        if int(b_m.sum()) >= 100 and int(b_w.sum()) >= 100:
            budget_by_age[bname] = {
                "n_men": int(b_m.sum()),
                "n_women": int(b_w.sum()),
                "items": {
                    name: {
                        "men": mean_se_ci(bud_arr[b_m, j]),
                        "women": mean_se_ci(bud_arr[b_w, j]),
                        "women_minus_men": float(bud_arr[b_w, j].mean() - bud_arr[b_m, j].mean()),
                    }
                    for j, name in enumerate(item_names)
                },
            }

    # 2. Realized Partner Age Gap (mate_age - Age) Across Adult Age
    gap_arr = np.array(gap_rows, dtype=np.float64)
    partner_age_gap_by_age = {"n_total": int(len(gap_arr)), "brackets": {}}
    for lo, hi, bname in KOWAL_LIFE_BRACKETS:
        gm = (gap_arr[:, 0] == 0.0) & (gap_arr[:, 1] >= lo) & (gap_arr[:, 1] <= hi)
        gw = (gap_arr[:, 0] == 1.0) & (gap_arr[:, 1] >= lo) & (gap_arr[:, 1] <= hi)
        if int(gm.sum()) >= 100 and int(gw.sum()) >= 100:
            cm_s = mean_se_ci(gap_arr[gm, 2])
            cw_s = mean_se_ci(gap_arr[gw, 2])
            partner_age_gap_by_age["brackets"][bname] = {
                "men": cm_s,
                "women": cw_s,
                "women_minus_men_years": float(cw_s["mean"] - cm_s["mean"]),
            }

    # 3. Inside-the-Pair-Bond Reversal (PRQC quality SD & Infidelity pp on 4 standardized partner traits, country-cluster FE)
    b_ctry = np.array([r[0] for r in bond_rows], dtype=object)
    b_sex = np.array([r[1] for r in bond_rows], dtype=np.int8)
    b_age = np.array([r[2] for r in bond_rows], dtype=np.float64)
    b_prqc = np.array([r[3] for r in bond_rows], dtype=np.float64)
    b_inf = np.array([r[4] for r in bond_rows], dtype=np.float64)
    b_mtraits = np.array([r[5:] for r in bond_rows], dtype=np.float64)

    def fit_bond_fe(mask_valid: np.ndarray, y_vec: np.ndarray, scale_mult: float, std_y: bool) -> dict:
        sub_c = b_ctry[mask_valid]
        sub_s = b_sex[mask_valid]
        sub_a = b_age[mask_valid]
        sub_m = b_mtraits[mask_valid]
        sub_y = y_vec[mask_valid]
        uniq_c, c_idx = np.unique(sub_c, return_inverse=True)
        nc_u = len(uniq_c)
        c_pos = {int(c): p for p, c in enumerate( np.arange(nc_u) )}
        res_out = {"n": int(mask_valid.sum()), "countries": int(nc_u), "by_sex": {}}
        xx_s, xy_s, pt_s, r2_s = {}, {}, {}, {}
        for s_label, s_code in (("men", 0), ("women", 1)):
            sm = sub_s == s_code
            gc = c_idx[sm]
            ag = sub_a[sm]
            ys = sub_y[sm].copy()
            if std_y:
                ys = (ys - ys.mean()) / ys.std(ddof=1)
            # Pooled standardized OLS R^2 on the 4 partner traits
            Xz_pool = (sub_m[sm] - sub_m[sm].mean(axis=0)) / sub_m[sm].std(axis=0, ddof=1)
            Xp = np.column_stack([np.ones(int(sm.sum())), Xz_pool])
            yhat_p = Xp @ np.linalg.lstsq(Xp, ys, rcond=None)[0]
            r2_s[s_label] = float(1.0 - np.sum((ys - yhat_p) ** 2) / np.sum((ys - ys.mean()) ** 2))

            X = np.column_stack(
                [demean_by_group(sub_m[sm, j], gc, True) for j in range(4)]
                + [demean_by_group(ag, gc, False), demean_by_group(ag ** 2, gc, False)]
            )
            yd = demean_by_group(ys, gc, False)
            xx = np.zeros((nc_u, 6, 6), dtype=np.float64)
            xy = np.zeros((nc_u, 6), dtype=np.float64)
            order = np.argsort(gc, kind="stable")
            gs = gc[order]
            cuts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1, len(gs)]
            for a_i, b_i in zip(cuts[:-1], cuts[1:]):
                idx = order[a_i:b_i]
                pos = c_pos[int(gs[a_i])]
                xx[pos] = X[idx].T @ X[idx]
                xy[pos] = X[idx].T @ yd[idx]
            pt_s[s_label] = scale_mult * np.linalg.solve(xx.sum(axis=0), xy.sum(axis=0))[:4]
            xx_s[s_label] = xx
            xy_s[s_label] = xy

        rs_b = rng_k_ext.integers(0, nc_u, size=(1000, nc_u))
        bm_d = scale_mult * np.linalg.solve(xx_s["men"][rs_b].sum(axis=1), xy_s["men"][rs_b].sum(axis=1)[..., None])[..., 0][:, :4]
        bw_d = scale_mult * np.linalg.solve(xx_s["women"][rs_b].sum(axis=1), xy_s["women"][rs_b].sum(axis=1)[..., None])[..., 0][:, :4]
        unit_key = "sd_per_sd" if std_y else "pp_per_sd"
        for s_label, s_code, b_draws in (("men", 0, bm_d), ("women", 1, bw_d)):
            sm = sub_s == s_code
            res_out["by_sex"][s_label] = {
                "n": int(sm.sum()),
                "pooled_four_trait_r2": r2_s[s_label],
                "mean_outcome": float(sub_y[sm].mean()),
                "traits": {
                    tname: {
                        unit_key: float(pt_s[s_label][j]),
                        "ci_95": np.quantile(b_draws[:, j], [0.025, 0.975]).tolist(),
                    }
                    for j, tname in enumerate(door_trait_names)
                },
            }
        return res_out

    pair_bond_reversal = {
        "relationship_quality_prqc": fit_bond_fe(np.isfinite(b_prqc), b_prqc, 1.0, True),
        "reported_infidelity": fit_bond_fe(np.isfinite(b_inf), b_inf, 100.0, False),
    }

    # 4. Sternberg 25-Year Triangular Love Arc (6 relationship-duration brackets)
    love_arr = np.array(love_rows, dtype=np.float64)
    dur_brackets = (
        (0.0, 1.0, "<=1y", True),
        (1.0, 3.0, "1-3y", False),
        (3.0, 7.0, "3-7y", False),
        (7.0, 15.0, "7-15y", False),
        (15.0, 25.0, "15-25y", False),
        (25.0, 60.0, "25y+", False),
    )
    triangular_love_arc = {"n_total": int(len(love_arr)), "by_duration": {}}
    for d_lo, d_hi, d_label, inc_lo in dur_brackets:
        dmask = ((love_arr[:, 2] >= d_lo) if inc_lo else (love_arr[:, 2] > d_lo)) & (love_arr[:, 2] <= d_hi)
        if int(dmask.sum()) >= 100:
            triangular_love_arc["by_duration"][d_label] = {
                "n": int(dmask.sum()),
                "intimacy": mean_se_ci(love_arr[dmask, 3]),
                "passion": mean_se_ci(love_arr[dmask, 4]),
                "commitment": mean_se_ci(love_arr[dmask, 5]),
                "prqc": mean_se_ci(love_arr[dmask, 6]),
                "by_sex": {
                    s_lbl: {
                        "n": int((dmask & (love_arr[:, 0] == s_c)).sum()),
                        "intimacy": mean_se_ci(love_arr[dmask & (love_arr[:, 0] == s_c), 3]),
                        "passion": mean_se_ci(love_arr[dmask & (love_arr[:, 0] == s_c), 4]),
                        "commitment": mean_se_ci(love_arr[dmask & (love_arr[:, 0] == s_c), 5]),
                        "prqc": mean_se_ci(love_arr[dmask & (love_arr[:, 0] == s_c), 6]),
                    }
                    for s_lbl, s_c in (("men", 0.0), ("women", 1.0))
                    if int((dmask & (love_arr[:, 0] == s_c)).sum()) >= 100
                },
            }

    # 5. Refusal of Loveless Marriage (0-100 willingness scale)
    mwl_arr = np.array(mwl_rows, dtype=np.float64)
    marry_without_love = {"n_total": int(len(mwl_arr)), "by_sex": {}}
    for s_lbl, s_c in (("men", 0.0), ("women", 1.0)):
        vals = mwl_arr[mwl_arr[:, 0] == s_c, 1]
        marry_without_love["by_sex"][s_lbl] = {
            **mean_se_ci(vals),
            "median": float(np.median(vals)),
            "iqr": np.quantile(vals, [0.25, 0.75]).tolist(),
            "share_le_20": float((vals <= 20.0).mean()),
            "share_eq_0": float((vals == 0.0).mean()),
        }
    marry_without_love["women_minus_men_points"] = float(
        marry_without_love["by_sex"]["women"]["mean"] - marry_without_love["by_sex"]["men"]["mean"]
    )

    # 6. Assortative Mating (self vs. mate correlations across 5 traits)
    assort_arr = np.array(assort_rows, dtype=np.float64)
    assort_names = ("religion", "health", "resources", "looks", "kindness")
    assortative_mating = {}
    for j, aname in enumerate(assort_names):
        sv = assort_arr[:, j]
        mv = assort_arr[:, j + 5]
        ok_a = np.isfinite(sv) & np.isfinite(mv) & (sv >= 1.0) & (sv <= 11.0) & (mv >= 1.0) & (mv <= 11.0)
        na = int(ok_a.sum())
        r_val = float(np.corrcoef(sv[ok_a], mv[ok_a])[0, 1])
        z_a = math.atanh(r_val)
        se_z = 1.0 / math.sqrt(na - 3.0)
        assortative_mating[aname] = {
            "n": na,
            "pearson_r": r_val,
            "ci_95": [float(math.tanh(z_a - 1.96 * se_z)), float(math.tanh(z_a + 1.96 * se_z))],
        }

    # 7. Observational Social-Media Feed Gradient & Perceived Quality of Alternatives (QOA)
    sm_ctry = np.array([r[0] for r in sm_rows], dtype=object)
    sm_sex = np.array([r[1] for r in sm_rows], dtype=np.float64)
    sm_age = np.array([r[2] for r in sm_rows], dtype=np.float64)
    sm_hrs = np.array([r[3] for r in sm_rows], dtype=np.float64)
    sm_qoa = np.array([r[4] for r in sm_rows], dtype=np.float64)
    sm_prqc = np.array([r[5] for r in sm_rows], dtype=np.float64)
    sm_inf = np.array([r[6] for r in sm_rows], dtype=np.float64)

    def fit_sm_slope(y_vec: np.ndarray, mult: float, std_y: bool) -> dict:
        ok_s = np.isfinite(y_vec)
        sc = sm_ctry[ok_s]
        uc, c_idx = np.unique(sc, return_inverse=True)
        nc_s = len(uc)
        yv = y_vec[ok_s].copy()
        if std_y:
            yv = (yv - yv.mean()) / yv.std(ddof=1)
        ag = sm_age[ok_s]
        sx = sm_sex[ok_s]
        hr = sm_hrs[ok_s]
        X = np.column_stack([
            demean_by_group(hr, c_idx, False),
            demean_by_group(sx, c_idx, False),
            demean_by_group(ag, c_idx, False),
            demean_by_group(ag ** 2, c_idx, False),
        ])
        yd = demean_by_group(yv, c_idx, False)
        xx = np.zeros((nc_s, 4, 4), dtype=np.float64)
        xy = np.zeros((nc_s, 4), dtype=np.float64)
        order = np.argsort(c_idx, kind="stable")
        gs = c_idx[order]
        cuts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1, len(gs)]
        for a_i, b_i in zip(cuts[:-1], cuts[1:]):
            idx = order[a_i:b_i]
            pos = int(gs[a_i])
            xx[pos] = X[idx].T @ X[idx]
            xy[pos] = X[idx].T @ yd[idx]
        est = float(mult * np.linalg.solve(xx.sum(axis=0), xy.sum(axis=0))[0])
        rs_s = rng_k_ext.integers(0, nc_s, size=(1000, nc_s))
        b_draws = mult * np.linalg.solve(xx[rs_s].sum(axis=1), xy[rs_s].sum(axis=1)[..., None])[..., 0][:, 0]
        return {
            "n": int(ok_s.sum()),
            "countries": int(nc_s),
            "estimate": est,
            "ci_95": np.quantile(b_draws, [0.025, 0.975]).tolist(),
        }

    qoa_arr = np.array(qoa_rows, dtype=np.float64)
    qoa_tiers = {}
    for t_lbl, q_lo, q_hi, inc_lo in (
        ("low_1_to_2.5", 1.0, 2.5, True),
        ("moderate_2.5_to_4.5", 2.5, 4.5, False),
        ("high_4.5_to_7.0", 4.5, 7.0, False),
    ):
        qm = ((qoa_arr[:, 0] >= q_lo) if inc_lo else (qoa_arr[:, 0] > q_lo)) & (qoa_arr[:, 0] <= q_hi)
        inf_ok = qm & np.isfinite(qoa_arr[:, 2])
        prq_ok = qm & np.isfinite(qoa_arr[:, 1])
        qoa_tiers[t_lbl] = {
            "n_qoa": int(qm.sum()),
            "infidelity_rate_pct": mean_se_ci(100.0 * qoa_arr[inf_ok, 2]),
            "prqc_mean": mean_se_ci(qoa_arr[prq_ok, 1]),
        }

    social_media_and_alternatives = {
        "epistemic_tier": "observational_cross_sectional_country_age_sex_adjusted",
        "mean_daily_hours": mean_se_ci(sm_hrs),
        "slopes_per_daily_hour": {
            "qoa_sd_per_hour": fit_sm_slope(sm_qoa, 1.0, True),
            "prqc_sd_per_hour": fit_sm_slope(sm_prqc, 1.0, True),
            "infidelity_pp_per_hour": fit_sm_slope(sm_inf, 100.0, False),
        },
        "by_qoa_tier": qoa_tiers,
    }

    # 8. Conroy-Beam Idiographic Checklist Alignment & "Marrying-Up" Surplus vs. Unconditional Kindness
    rng_k_v13 = np.random.default_rng(SEED + 401)
    cs_ctry = np.array([r[0] for r in chk_surplus_rows], dtype=object)
    cs_num = np.array([r[1:] for r in chk_surplus_rows], dtype=np.float64)
    cs_sex = cs_num[:, 0]
    cs_age = cs_num[:, 1]
    cs_prqc = cs_num[:, 2]
    cs_ge = cs_num[:, 4]
    cs_ind = cs_num[:, 5]
    cs_pf = cs_num[:, 6:11]
    cs_mt = cs_num[:, 11:16]
    cs_sf = cs_num[:, 16:21]

    def zscore_cols(mat: np.ndarray) -> np.ndarray:
        return (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=1)

    z_pf = zscore_cols(cs_pf)
    z_mt = zscore_cols(cs_mt)
    euclid_align = -np.sqrt(np.mean((z_pf - z_mt) ** 2, axis=1))
    pos_marry_up = np.maximum(0.0, np.mean(cs_mt - cs_sf, axis=1))
    signed_surplus = np.mean(cs_mt - cs_sf, axis=1)
    y_prqc_z = (cs_prqc - cs_prqc.mean()) / cs_prqc.std(ddof=1)

    # Predictors: [mate_kind, mate_looks, mate_resources, idiographic_checklist_fit, marrying_up_surplus]
    pred_5 = np.column_stack([
        z_mt[:, 2],
        z_mt[:, 0],
        z_mt[:, 1],
        (euclid_align - euclid_align.mean()) / euclid_align.std(ddof=1),
        (pos_marry_up - pos_marry_up.mean()) / pos_marry_up.std(ddof=1),
    ])
    pred_names_5 = ("partner_kindness", "partner_looks", "partner_resources", "idiographic_checklist_fit", "marrying_up_surplus")

    def fit_checklist_subset(sub_mask: np.ndarray, include_sex_ctrl: bool) -> dict:
        sc = cs_ctry[sub_mask]
        uc, c_idx = np.unique(sc, return_inverse=True)
        nc_c = len(uc)
        ag = cs_age[sub_mask]
        sx = cs_sex[sub_mask]
        P = pred_5[sub_mask]
        yv = y_prqc_z[sub_mask]
        cols = [demean_by_group(P[:, j], c_idx, False) for j in range(5)]
        if include_sex_ctrl:
            cols.append(demean_by_group(sx, c_idx, False))
        cols.append(demean_by_group(ag, c_idx, False))
        cols.append(demean_by_group(ag ** 2, c_idx, False))
        X = np.column_stack(cols)
        yd = demean_by_group(yv, c_idx, False)
        k_dim = X.shape[1]
        xx = np.zeros((nc_c, k_dim, k_dim), dtype=np.float64)
        xy = np.zeros((nc_c, k_dim), dtype=np.float64)
        order = np.argsort(c_idx, kind="stable")
        gs = c_idx[order]
        cuts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1, len(gs)]
        for a_i, b_i in zip(cuts[:-1], cuts[1:]):
            idx = order[a_i:b_i]
            pos = int(gs[a_i])
            xx[pos] = X[idx].T @ X[idx]
            xy[pos] = X[idx].T @ yd[idx]
        beta = np.linalg.solve(xx.sum(axis=0), xy.sum(axis=0))
        rs_c = rng_k_v13.integers(0, nc_c, size=(1000, nc_c))
        b_draws = np.linalg.solve(xx[rs_c].sum(axis=1), xy[rs_c].sum(axis=1)[..., None])[..., 0]
        yhat = X[:, :5] @ beta[:5]
        r2_within = float(1.0 - np.sum((yd - X @ beta) ** 2) / np.sum(yd ** 2))
        return {
            "n": int(sub_mask.sum()),
            "countries": int(nc_c),
            "within_country_r2": r2_within,
            "predictors_sd_per_sd": {
                pname: {
                    "estimate": float(beta[j]),
                    "ci_95": np.quantile(b_draws[:, j], [0.025, 0.975]).tolist(),
                }
                for j, pname in enumerate(pred_names_5)
            },
            "kindness_to_checklist_ratio": float(beta[0] / beta[3]),
        }

    # Sensitivity controlling all 5 partner traits simultaneously alongside marrying-up surplus
    uc_all, ci_all = np.unique(cs_ctry, return_inverse=True)
    X_full5 = np.column_stack([
        *[demean_by_group(z_mt[:, j], ci_all, False) for j in range(5)],
        demean_by_group(pred_5[:, 3], ci_all, False),
        demean_by_group(pred_5[:, 4], ci_all, False),
        demean_by_group(cs_sex, ci_all, False),
        demean_by_group(cs_age, ci_all, False),
        demean_by_group(cs_age ** 2, ci_all, False),
    ])
    X_signed5 = np.column_stack([
        *[demean_by_group(pred_5[:, j], ci_all, False) for j in range(4)],
        demean_by_group((signed_surplus - signed_surplus.mean()) / signed_surplus.std(ddof=1), ci_all, False),
        demean_by_group(cs_sex, ci_all, False),
        demean_by_group(cs_age, ci_all, False),
        demean_by_group(cs_age ** 2, ci_all, False),
    ])
    yd_all = demean_by_group(y_prqc_z, ci_all, False)
    full5_beta_marry_up = float(np.linalg.lstsq(X_full5, yd_all, rcond=None)[0][6])
    signed_surplus_beta = float(np.linalg.lstsq(X_signed5, yd_all, rcond=None)[0][4])

    ge_med = float(np.nanmedian(cs_ge))
    ind_med = float(np.nanmedian(cs_ind))
    checklist_and_surplus_decomposition = {
        "n_total": int(len(cs_num)),
        "countries": int(len(uc_all)),
        "pooled": fit_checklist_subset(np.ones(len(cs_num), dtype=bool), True),
        "by_sex": {
            "women": fit_checklist_subset(cs_sex == 1.0, False),
            "men": fit_checklist_subset(cs_sex == 0.0, False),
        },
        "by_cultural_regime": {
            "high_gender_equality": fit_checklist_subset(np.isfinite(cs_ge) & (cs_ge >= ge_med), True),
            "low_gender_equality": fit_checklist_subset(np.isfinite(cs_ge) & (cs_ge < ge_med), True),
            "individualist": fit_checklist_subset(np.isfinite(cs_ind) & (cs_ind >= ind_med), True),
            "collectivist": fit_checklist_subset(np.isfinite(cs_ind) & (cs_ind < ind_med), True),
        },
        "five_trait_controlled_marrying_up_sd": full5_beta_marrying_up if (full5_beta_marrying_up := full5_beta_marry_up) is not None else 0.0,
        "signed_mate_minus_self_surplus_sd": signed_surplus_beta,
    }

    # 9. Short-Term vs. Long-Term Ideal Ratings (1-11 scale) & Emotional vs. Sexual Jealousy (1-7 scale)
    lt_arr = np.array(lt_arm_rows, dtype=np.float64)
    st_arr = np.array(st_arm_rows, dtype=np.float64)
    ideal_trait_names = ("kindness", "health", "looks", "resources", "age", "religion")
    short_vs_long_term_ideals = {"n_long_term": int(len(lt_arr)), "n_short_term": int(len(st_arr)), "by_sex": {}}
    for s_lbl, s_c in (("women", 1.0), ("men", 0.0)):
        lm = lt_arr[:, 0] == s_c
        sm_m = st_arr[:, 0] == s_c
        short_vs_long_term_ideals["by_sex"][s_lbl] = {
            tname: {
                "long_term_mean_1_to_11": float(lt_arr[lm, j + 1].mean()),
                "short_term_mean_1_to_11": float(st_arr[sm_m, j + 1].mean()),
                "long_minus_short": float(lt_arr[lm, j + 1].mean() - st_arr[sm_m, j + 1].mean()),
            }
            for j, tname in enumerate(ideal_trait_names)
        }

    jeal_arr = np.array(jeal_rows, dtype=np.float64)
    jealousy_asymmetry = {"n_total": int(len(jeal_arr)), "by_sex": {}}
    for s_lbl, s_c in (("women", 1.0), ("men", 0.0)):
        jm = jeal_arr[:, 0] == s_c
        emo_v = jeal_arr[jm, 1]
        sex_v = jeal_arr[jm, 2]
        jealousy_asymmetry["by_sex"][s_lbl] = {
            "n": int(jm.sum()),
            "emotional_1_to_7": mean_se_ci(emo_v),
            "sexual_1_to_7": mean_se_ci(sex_v),
            "emotional_minus_sexual": mean_se_ci(emo_v - sex_v),
        }

    # 10. Primary Kindness vs. Political/Educational/Class Homophily, Reverence vs. Settling Kink, & Daily Effort/SOI Lifecycle
    rng_k_v15 = np.random.default_rng(SEED + 451)
    h_ctry = np.array([r[0] for r in hom_rows], dtype=object)
    h_num = np.array([r[1:] for r in hom_rows], dtype=np.float64)
    homophily_vs_kindness = {}
    for dom_name, idx_own, idx_mate in (("politics", 4, 5), ("education", 6, 7), ("social_class", 8, 9)):
        ok_h = np.isfinite(h_num[:, idx_own]) & np.isfinite(h_num[:, idx_mate])
        sub_c = h_ctry[ok_h]
        uc_h, ci_h = np.unique(sub_c, return_inverse=True)
        nc_h = len(uc_h)
        own_v = h_num[ok_h, idx_own]
        mate_v = h_num[ok_h, idx_mate]
        r_sort = float(np.corrcoef(own_v, mate_v)[0, 1])
        exact_m = float((own_v == mate_v).mean())
        sim_raw = -np.abs(own_v - mate_v)
        sim_z = (sim_raw - sim_raw.mean()) / sim_raw.std(ddof=1)
        kind_v = h_num[ok_h, 3]
        kind_z = (kind_v - kind_v.mean()) / kind_v.std(ddof=1)
        prqc_v_h = h_num[ok_h, 2]
        prqc_z_h = (prqc_v_h - prqc_v_h.mean()) / prqc_v_h.std(ddof=1)
        sx_h = h_num[ok_h, 0]
        ag_h = h_num[ok_h, 1]
        Xh = np.column_stack([
            demean_by_group(sim_z, ci_h, False),
            demean_by_group(kind_z, ci_h, False),
            demean_by_group(sx_h, ci_h, False),
            demean_by_group(ag_h, ci_h, False),
            demean_by_group(ag_h ** 2, ci_h, False),
        ])
        yh = demean_by_group(prqc_z_h, ci_h, False)
        xx_h = np.zeros((nc_h, 5, 5), dtype=np.float64)
        xy_h = np.zeros((nc_h, 5), dtype=np.float64)
        ord_h = np.argsort(ci_h, kind="stable")
        gs_h = ci_h[ord_h]
        cuts_h = np.r_[0, np.flatnonzero(np.diff(gs_h)) + 1, len(gs_h)]
        for a_i, b_i in zip(cuts_h[:-1], cuts_h[1:]):
            idx = ord_h[a_i:b_i]
            pos = int(gs_h[a_i])
            xx_h[pos] = Xh[idx].T @ Xh[idx]
            xy_h[pos] = Xh[idx].T @ yh[idx]
        beta_h = np.linalg.solve(xx_h.sum(axis=0), xy_h.sum(axis=0))
        rs_h = rng_k_v15.integers(0, nc_h, size=(1000, nc_h))
        bh_draws = np.linalg.solve(xx_h[rs_h].sum(axis=1), xy_h[rs_h].sum(axis=1)[..., None])[..., 0]
        homophily_vs_kindness[dom_name] = {
            "n": int(ok_h.sum()),
            "countries": int(nc_h),
            "assortative_pearson_r": r_sort,
            "exact_match_share": exact_m,
            "homophily_prqc_sd_per_sd": {
                "estimate": float(beta_h[0]),
                "ci_95": np.quantile(bh_draws[:, 0], [0.025, 0.975]).tolist(),
            },
            "partner_kindness_prqc_sd_per_sd": {
                "estimate": float(beta_h[1]),
                "ci_95": np.quantile(bh_draws[:, 1], [0.025, 0.975]).tolist(),
            },
            "kindness_to_homophily_ratio": float(beta_h[1] / max(abs(beta_h[0]), 1e-6)),
        }

    rv_ctry = np.array([r[0] for r in rev_rows], dtype=object)
    rv_num = np.array([r[1:] for r in rev_rows], dtype=np.float64)
    reverence_vs_settling_kink = {"n_total": int(len(rv_num)), "by_sex": {}}
    for s_lbl, s_c in (("women", 1.0), ("men", 0.0)):
        sm_r = rv_num[:, 0] == s_c
        sc_r = rv_ctry[sm_r]
        uc_r, ci_r = np.unique(sc_r, return_inverse=True)
        nc_r = len(uc_r)
        sub_r = rv_num[sm_r]
        tiers_by_trait = {}
        for tname, i_sf, i_mt in (("kindness", 4, 5), ("looks", 6, 7), ("resources", 8, 9)):
            diff_v = sub_r[:, i_mt] - sub_r[:, i_sf]
            t_entry = {}
            for tier_k, tmask in (
                ("settled_mate_lt_self", diff_v < 0.0),
                ("equal", diff_v == 0.0),
                ("revered_mate_gt_self", diff_v > 0.0),
            ):
                inf_ok_t = tmask & np.isfinite(sub_r[:, 3])
                t_entry[tier_k] = {
                    "n": int(tmask.sum()),
                    "share": float(tmask.mean()),
                    "prqc_mean": mean_se_ci(sub_r[tmask, 2]),
                    "infidelity_rate_pct": mean_se_ci(100.0 * sub_r[inf_ok_t, 3]),
                }
            tiers_by_trait[tname] = t_entry

        prqc_z_r = (sub_r[:, 2] - sub_r[:, 2].mean()) / sub_r[:, 2].std(ddof=1)
        ag_r = sub_r[:, 1]
        raw_cols_r = [
            sub_r[:, 4],
            np.maximum(0.0, sub_r[:, 4] - sub_r[:, 5]),
            np.maximum(0.0, sub_r[:, 5] - sub_r[:, 4]),
            sub_r[:, 6],
            np.maximum(0.0, sub_r[:, 6] - sub_r[:, 7]),
            np.maximum(0.0, sub_r[:, 7] - sub_r[:, 6]),
            sub_r[:, 8],
            np.maximum(0.0, sub_r[:, 8] - sub_r[:, 9]),
            np.maximum(0.0, sub_r[:, 9] - sub_r[:, 8]),
            ag_r,
            ag_r ** 2,
        ]
        Xr = np.column_stack([demean_by_group(col_v, ci_r, False) for col_v in raw_cols_r])
        yr = demean_by_group(prqc_z_r, ci_r, False)
        xx_r = np.zeros((nc_r, 11, 11), dtype=np.float64)
        xy_r = np.zeros((nc_r, 11), dtype=np.float64)
        ord_r = np.argsort(ci_r, kind="stable")
        gs_r = ci_r[ord_r]
        cuts_r = np.r_[0, np.flatnonzero(np.diff(gs_r)) + 1, len(gs_r)]
        for a_i, b_i in zip(cuts_r[:-1], cuts_r[1:]):
            idx = ord_r[a_i:b_i]
            pos = int(gs_r[a_i])
            xx_r[pos] = Xr[idx].T @ Xr[idx]
            xy_r[pos] = Xr[idx].T @ yr[idx]
        beta_r = np.linalg.solve(xx_r.sum(axis=0), xy_r.sum(axis=0))
        rs_r = rng_k_v15.integers(0, nc_r, size=(1000, nc_r))
        br_draws = np.linalg.solve(xx_r[rs_r].sum(axis=1), xy_r[rs_r].sum(axis=1)[..., None])[..., 0]
        spline_names = (
            "self_kindness", "settling_deficit_kindness", "reverence_surplus_kindness",
            "self_looks", "settling_deficit_looks", "reverence_surplus_looks",
            "self_resources", "settling_deficit_resources", "reverence_surplus_resources",
        )
        reverence_vs_settling_kink["by_sex"][s_lbl] = {
            "n": int(sm_r.sum()),
            "countries": int(nc_r),
            "tiers_by_trait": tiers_by_trait,
            "country_fe_spline_sd_per_pt": {
                sname: {
                    "estimate": float(beta_r[j]),
                    "ci_95": np.quantile(br_draws[:, j], [0.025, 0.975]).tolist(),
                }
                for j, sname in enumerate(spline_names)
            },
            "kindness_settling_to_reverence_slope_ratio": float(abs(beta_r[1]) / max(abs(beta_r[2]), 1e-6)),
        }

    ef_rstat = np.array([r[2] for r in eff_soi_rows], dtype=object)
    ef_num = np.array([[r[0], r[1], r[3], r[4], r[5], r[6], r[7], r[8], r[9]] for r in eff_soi_rows], dtype=np.float64)
    tot_ok = np.isfinite(ef_num[:, 2]) & (ef_num[:, 2] >= 0.0) & (ef_num[:, 2] <= 600.0)
    ex_ok = np.isfinite(ef_num[:, 3]) & (ef_num[:, 3] >= 0.0) & (ef_num[:, 3] <= 300.0)
    mk_ok = np.isfinite(ef_num[:, 4]) & (ef_num[:, 4] >= 0.0) & (ef_num[:, 4] <= 180.0)
    dr_ok = np.isfinite(ef_num[:, 5]) & (ef_num[:, 5] >= 0.0) & (ef_num[:, 5] <= 180.0)

    effort_by_stage = {"epistemic_tier": "observational_cross_sectional_relationship_stage", "by_sex": {}}
    stage_specs = (
        ("single", "Single"),
        ("dating", "Dating"),
        ("committed", "In a committed relationship"),
        ("married", "Married"),
    )
    for s_lbl, s_c in (("women", 1.0), ("men", 0.0)):
        sm_e = ef_num[:, 0] == s_c
        st_out = {}
        for st_key, st_raw in stage_specs:
            st_m = sm_e & (ef_rstat == st_raw)
            st_out[st_key] = {
                "total_effort_min": mean_se_ci(ef_num[st_m & tot_ok, 2]),
                "exercise_min": mean_se_ci(ef_num[st_m & ex_ok, 3]),
                "makeup_min": mean_se_ci(ef_num[st_m & mk_ok, 4]),
                "dress_min": mean_se_ci(ef_num[st_m & dr_ok, 5]),
                "mean_age": float(ef_num[st_m & tot_ok, 1].mean()),
            }
        dt_m = st_out["dating"]["total_effort_min"]["mean"]
        mr_m = st_out["married"]["total_effort_min"]["mean"]
        st_out["dating_to_married_drop_min"] = float(mr_m - dt_m)
        st_out["dating_to_married_drop_pct"] = float(100.0 * (mr_m - dt_m) / dt_m)
        effort_by_stage["by_sex"][s_lbl] = st_out

    soi_and_effort_by_age = {}
    for lo_a, hi_a, b_lbl in ((18, 24, "18-24"), (25, 29, "25-29"), (30, 39, "30-39"), (40, 49, "40-49"), (50, 59, "50-59"), (60, 79, "60-79")):
        am_e = (ef_num[:, 1] >= lo_a) & (ef_num[:, 1] <= hi_a)
        b_cell = {}
        for s_lbl, s_c in (("women", 1.0), ("men", 0.0)):
            sg_e = am_e & (ef_num[:, 0] == s_c)
            sd_ok = sg_e & np.isfinite(ef_num[:, 6]) & (ef_num[:, 6] >= 1.0) & (ef_num[:, 6] <= 9.0)
            sa_ok = sg_e & np.isfinite(ef_num[:, 7]) & (ef_num[:, 7] >= 1.0) & (ef_num[:, 7] <= 9.0)
            sb_ok = sg_e & np.isfinite(ef_num[:, 8]) & (ef_num[:, 8] >= 1.0) & (ef_num[:, 8] <= 9.0)
            b_cell[s_lbl] = {
                "soi_desire": mean_se_ci(ef_num[sd_ok, 6]),
                "soi_attitude": mean_se_ci(ef_num[sa_ok, 7]),
                "soi_behavior": mean_se_ci(ef_num[sb_ok, 8]),
                "total_effort_min": mean_se_ci(ef_num[sg_e & tot_ok, 2]),
            }
        b_cell["men_to_women_soi_desire_ratio"] = float(
            b_cell["men"]["soi_desire"]["mean"] / b_cell["women"]["soi_desire"]["mean"]
        )
        soi_and_effort_by_age[b_lbl] = b_cell

    courtship_effort_and_soi_lifecycle = {
        "by_relationship_stage": effort_by_stage,
        "by_age_bracket": soi_and_effort_by_age,
    }

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
        "budget_by_age": budget_by_age,
        "partner_age_gap_by_age": partner_age_gap_by_age,
        "pair_bond_reversal": pair_bond_reversal,
        "triangular_love_arc": triangular_love_arc,
        "marry_without_love": marry_without_love,
        "assortative_mating": assortative_mating,
        "social_media_and_alternatives": social_media_and_alternatives,
        "checklist_and_surplus_decomposition": checklist_and_surplus_decomposition,
        "short_vs_long_term_ideals": short_vs_long_term_ideals,
        "jealousy_asymmetry": jealousy_asymmetry,
        "homophily_vs_kindness": homophily_vs_kindness,
        "reverence_vs_settling_kink": reverence_vs_settling_kink,
        "courtship_effort_and_soi_lifecycle": courtship_effort_and_soi_lifecycle,
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
                own_age = float(r["age"])
            except (TypeError, ValueError):
                own_age = math.nan
            try:
                age_gap = float(r["mate_age"]) - own_age
            except (TypeError, ValueError):
                age_gap = math.nan
            rows.append((cin, sex, age_gap, *vals, own_age))
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
    own_ages = arr[:, 8]
    gaps = arr[:, 2]
    age_gap_by_age = {}
    for lo, hi, bname in ((18, 24, "18-24"), (25, 29, "25-29"), (30, 34, "30-34"), (35, 39, "35-39"), (40, 49, "40-49"), (50, 75, "50-75")):
        wm = women & np.isfinite(own_ages) & np.isfinite(gaps) & (own_ages >= lo) & (own_ages <= hi)
        mm = men & np.isfinite(own_ages) & np.isfinite(gaps) & (own_ages >= lo) & (own_ages <= hi)
        if int(wm.sum()) >= 50 and int(mm.sum()) >= 50:
            ws = mean_se_ci(gaps[wm])
            ms = mean_se_ci(gaps[mm])
            age_gap_by_age[bname] = {
                "women": ws,
                "men": ms,
                "women_minus_men_years": float(ws["mean"] - ms["mean"]),
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
        "age_gap_by_age": age_gap_by_age,
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
        "gss_three_episodes_all": {
            trait_k: {
                sex_k: {
                    ep_k: [float(x) for x in loaded["gss"]["three_episodes"][trait_k][sex_k][ep_k]]
                    for ep_k in ("total", "pairing", "parenthood", "number")
                    if ep_k in loaded["gss"]["three_episodes"][trait_k][sex_k]
                }
                for sex_k in ("men", "women")
            }
            for trait_k in ("income", "prestige", "schooling", "vocabulary")
        },
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


def wilson_ci_95(k: int, n: int) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def analyze_joel_2020(sat_path: Path, comm_path: Path) -> dict:
    def summarize_file(path: Path) -> dict:
        df = pd.read_csv(path)
        models = {}
        for mname, g in df.groupby("Outcome", sort=True):
            r = g["Correlation"].to_numpy(dtype=np.float64)
            n = g["SampleSize"].to_numpy(dtype=np.float64)
            r2_pct = 100.0 * (np.maximum(0.0, r) ** 2)
            models[str(mname)] = {
                "k_studies": int(len(g)),
                "total_couples": int(n.sum()),
                "unweighted_mean_r2_pct": float(r2_pct.mean()),
                "n_weighted_mean_r2_pct": float(np.sum(r2_pct * n) / np.sum(n)),
                "median_r2_pct": float(np.median(r2_pct)),
                "iqr_r2_pct": np.quantile(r2_pct, [0.25, 0.75]).tolist(),
            }
        return {
            "total_models": int(len(df)),
            "distinct_datasets": int(df["Dataset"].nunique()),
            "models": models,
            "actor_rel_to_partner_trait_baseline_ratio": float(
                models["ActorRelBaseline"]["unweighted_mean_r2_pct"]
                / models["PartnerTraitBaseline"]["unweighted_mean_r2_pct"]
            ),
            "actor_rel_to_partner_trait_followup_ratio": float(
                models["ActorRelFollowup"]["unweighted_mean_r2_pct"]
                / models["PartnerTraitFollowup"]["unweighted_mean_r2_pct"]
            ),
        }

    return {
        "satisfaction_sha256": OPEN_INPUTS["joel2020_SatisfactionMetaAnalyses.csv"]["sha256"],
        "commitment_sha256": OPEN_INPUTS["joel2020_CommitmentMetaAnalyses.csv"]["sha256"],
        "satisfaction": summarize_file(sat_path),
        "commitment": summarize_file(comm_path),
    }


def analyze_betzig_sccs(data_path: Path, codes_path: Path) -> dict:
    causes_spec = (
        ("SCCS1136", "adultery_infidelity", "Adultery / infidelity"),
        ("SCCS1144", "sterility_childlessness", "Sterility / childlessness"),
        ("SCCS1139", "cruelty_maltreatment", "Cruelty / maltreatment"),
        ("SCCS1156", "laziness_economic_neglect", "Laziness / economic neglect"),
        ("SCCS1174", "desertion", "Desertion / abandonment"),
        ("SCCS1157", "inadequate_support", "Inadequate material support"),
        ("SCCS1148", "old_age_loss_of_looks", "Old age / loss of physical appeal"),
    )
    wanted = {vid for vid, _, _ in causes_spec}
    code_names = {}
    with codes_path.open(encoding="utf-8", newline="") as stream:
        for r in csv.DictReader(stream):
            if r["var_id"] in wanted and r["code"] not in ("", "NA", None):
                code_names[(r["var_id"], int(float(r["code"])))] = r["name"]

    raw_codes = {vid: [] for vid in wanted}
    with data_path.open(encoding="utf-8", newline="") as stream:
        for r in csv.DictReader(stream):
            vid = r["var_id"]
            if vid in wanted and r["code"] not in ("", "NA", None):
                raw_codes[vid].append(int(float(r["code"])))

    causes = {}
    for vid, key, label in causes_spec:
        vals = raw_codes[vid]
        n_soc = len(vals)
        mentioned = sum(1 for c in vals if c > 0)
        causes[key] = {
            "var_id": vid,
            "label": label,
            "coded_societies": n_soc,
            "mentioned_societies": mentioned,
            "share_mentioned": float(mentioned / n_soc),
            "wilson_ci_95": wilson_ci_95(mentioned, n_soc),
            "by_spouse": {
                "husband_only": sum(1 for c in vals if c == 1),
                "wife_only": sum(1 for c in vals if c == 4),
                "both_or_unspecified": sum(1 for c in vals if c in (2, 3)),
            },
        }
    return {
        "data_sha256": OPEN_INPUTS["dplace_sccs_data.csv"]["sha256"],
        "codes_sha256": OPEN_INPUTS["dplace_sccs_codes.csv"]["sha256"],
        "causes": causes,
        "adultery_to_aging_looks_ratio": float(
            causes["adultery_infidelity"]["share_mentioned"]
            / causes["old_age_loss_of_looks"]["share_mentioned"]
        ),
    }


def analyze_childhood_coevolution(data_path: Path, codes_path: Path) -> dict:
    """Empirical SCCS caregiving/food-sharing synthesis + Proposition 22 (E22) simulation."""
    care_vars = ("SCCS51", "SCCS52", "SCCS53", "SCCS54", "SCCS1718")
    raw_codes = {vid: [] for vid in care_vars}
    with data_path.open(encoding="utf-8", newline="") as stream:
        for r in csv.DictReader(stream):
            vid = r["var_id"]
            if vid in raw_codes and r["code"] not in ("", "NA", None):
                raw_codes[vid].append(int(float(r["code"])))

    def share_stat(vals: list[int], pred) -> dict:
        n_s = len(vals)
        k_s = sum(1 for c in vals if pred(c))
        return {
            "coded_societies": n_s,
            "matching_societies": k_s,
            "share": float(k_s / n_s),
            "wilson_ci_95": wilson_ci_95(k_s, n_s),
        }

    sccs_care = {
        "infancy_exclusive_mother_sccs51": share_stat(raw_codes["SCCS51"], lambda c: c == 1),
        "infancy_any_nonmaternal_sccs51": share_stat(raw_codes["SCCS51"], lambda c: c >= 2),
        "infancy_major_nonmaternal_sccs51": share_stat(raw_codes["SCCS51"], lambda c: c >= 3),
        "early_childhood_exclusive_mother_sccs52": share_stat(raw_codes["SCCS52"], lambda c: c == 1),
        "early_childhood_nonmaternal_majority_sccs52": share_stat(raw_codes["SCCS52"], lambda c: c >= 3),
        "father_close_infancy_sccs53": share_stat(raw_codes["SCCS53"], lambda c: c >= 4),
        "father_close_early_childhood_sccs54": share_stat(raw_codes["SCCS54"], lambda c: c >= 4),
        "father_occasional_or_closer_early_childhood_sccs54": share_stat(raw_codes["SCCS54"], lambda c: c >= 3),
        "community_food_sharing_beyond_nuclear_sccs1718": share_stat(raw_codes["SCCS1718"], lambda c: c >= 2),
    }

    # Proposition 22 (E22): Childhood encephalization, alloparenting, cooking, and pair-bond commitment love.
    # Adult selection slope beta_S = 0.045, viability curvature lambda_V = 0.0015 (adult selection lift = +30 cm3 above juvenile ceiling).
    # Juvenile provisioning budget M_tot = M_mom + Delta_allo + p_bond * Delta_pair supports juvenile brain cost kappa_1 * (B - B_base)^2.
    rng_e22 = np.random.default_rng(SEED + 701)
    b_base = 400.0
    m_mom = 193.6       # Solitary maternal surplus -> sqrt(193.6 / 0.0040) = 220 -> 400 + 220 + 30 = 650 cm3
    delta_allo = 540.0  # Kin/grandmaternal alloparenting -> +208 cm3 raw ceiling lift
    delta_pair = 980.0  # Biparental multi-year provisioning potential when pair bond holds
    beta_s = 0.045
    lam_v = 0.0015

    # Pair-bond hold-up game across T_J juvenile period: outside temptation shocks O_hat ~ Normal(0.55, 0.25^2);
    # partner stays through childhood iff pair-bond commitment love L + baseline stay value u0 >= O_hat.
    u0_stay = 0.18
    temptation_shocks = rng_e22.normal(0.55, 0.25, size=20000)

    regimes_spec = (
        ("solitary_ape_mother", 0.0040, 0.0, 0.0, False),
        ("early_homo_allomaternal_raw", 0.0040, delta_allo, 0.16, True),
        ("cooking_without_pair_bond_love", 0.00205, delta_allo, 0.0, True),
        ("late_homo_cooking_plus_pair_bond_love", 0.00205, delta_allo, 0.72, True),
    )
    e22_regimes = {}
    for r_key, kappa_1, d_allo, love_commit, pair_channel in regimes_spec:
        if not pair_channel:
            p_bond_sim = 0.0
            p_bond_cf = 0.0
        else:
            z_stay = (u0_stay + love_commit - 0.55) / 0.25
            p_bond_cf = float(0.5 * (1.0 + math.erf(z_stay / math.sqrt(2.0))))
            p_bond_sim = float(np.mean(u0_stay + love_commit >= temptation_shocks))
        m_tot_cf = m_mom + d_allo + p_bond_cf * (delta_pair if pair_channel else 0.0)
        b_cap_cf = b_base + math.sqrt(m_tot_cf / kappa_1)
        b_star_cf = b_cap_cf + beta_s / lam_v

        # 20,000-agent Lande evolutionary trajectory over 120 generations (heritability * additive variance G = 220 cm3^2)
        pop_b = rng_e22.normal(620.0, 45.0, size=20000)
        m_tot_sim = m_mom + d_allo + p_bond_sim * (delta_pair if pair_channel else 0.0)
        b_cap_sim = b_base + math.sqrt(m_tot_sim / kappa_1)
        for _ in range(120):
            # Individual fitness gradient dW/dB = beta_S - lam_V * (B - B_cap) + developmental noise
            grad = beta_s - lam_v * (pop_b - b_cap_sim)
            pop_b = pop_b + 180.0 * float(grad.mean()) + rng_e22.normal(0.0, 4.0, size=20000)
            pop_b = np.clip(pop_b, 350.0, 2000.0)
        e22_regimes[r_key] = {
            "kappa_1_juvenile_cost": kappa_1,
            "allomaternal_transfer": d_allo,
            "pair_bond_commitment_L": love_commit,
            "p_bond_sustained_closed_form": p_bond_cf,
            "p_bond_sustained_simulated": p_bond_sim,
            "juvenile_ceiling_cm3": float(b_cap_cf),
            "equilibrium_cm3_closed_form": float(b_star_cf),
            "equilibrium_cm3_simulated": float(pop_b.mean()),
            "abs_closed_minus_sim_cm3": float(abs(b_star_cf - float(pop_b.mean()))),
        }

    kuzawa_list = list(KUZAWA_2014_BRAIN_METABOLISM)
    peak_stage = max(kuzawa_list, key=lambda d: d["brain_rmr_pct"])
    adult_stage = kuzawa_list[-1]

    return {
        "sccs_data_sha256": OPEN_INPUTS["dplace_sccs_data.csv"]["sha256"],
        "kuzawa_2014_brain_metabolism": {
            "trajectory": kuzawa_list,
            "peak_age_years": float(peak_stage["age_years"]),
            "peak_rmr_pct": float(peak_stage["brain_rmr_pct"]),
            "peak_der_pct": float(peak_stage["brain_der_pct"]),
            "adult_rmr_pct": float(adult_stage["brain_rmr_pct"]),
            "child_to_adult_rmr_ratio": float(peak_stage["brain_rmr_pct"] / adult_stage["brain_rmr_pct"]),
        },
        "primate_gray_ceiling_benchmarks": dict(PRIMATE_GRAY_CEILING_BENCHMARKS),
        "sccs_alloparenting_and_sharing": sccs_care,
        "proposition_22_e22_coevolution": {
            "agents_simulated": 20000,
            "regimes": e22_regimes,
            "love_devotion_gain_cm3": float(
                e22_regimes["late_homo_cooking_plus_pair_bond_love"]["equilibrium_cm3_closed_form"]
                - e22_regimes["cooking_without_pair_bond_love"]["equilibrium_cm3_closed_form"]
            ),
            "cooking_plus_love_total_gain_over_plateau_cm3": float(
                e22_regimes["late_homo_cooking_plus_pair_bond_love"]["equilibrium_cm3_closed_form"]
                - e22_regimes["early_homo_allomaternal_raw"]["equilibrium_cm3_closed_form"]
            ),
        },
    }


def analyze_nhs_campfire(disc_meta_path: Path, naive_path: Path, eth_prim_path: Path, cult_meta_path: Path) -> dict:
    """Harvard Music Lab Natural History of Song (Mehr et al., Science 2019) acoustic legibility & hearth context."""
    rng_nhs = np.random.default_rng(SEED + 601)
    df_meta = pd.read_csv(disc_meta_path)
    df_naive = pd.read_csv(naive_path)
    df_eth = pd.read_csv(eth_prim_path, low_memory=False)
    df_cult = pd.read_csv(cult_meta_path)

    df_meta["song_num"] = df_meta["song"].astype(str).map(lambda s: int(s.split("-")[-1]))
    df_naive["song_num"] = df_naive["song"].astype(str).map(lambda s: int(s.split("-")[-1]))
    merged = df_meta.merge(df_naive, on="song_num", suffixes=("_meta", "_naive")).sort_values("song_num")
    assert len(merged) == 118

    cat_specs = (
        ("Lullaby", "func_baby", "Infant soothing / lullaby"),
        ("Dance", "func_danc", "Communal motor dance"),
        ("Healing", "func_heal", "Ceremonial healing"),
        ("Love", "func_love", "Romantic love song"),
    )
    legibility = {}
    for cat_name, rating_col, label in cat_specs:
        in_mask = (merged["type"] == cat_name).to_numpy()
        vals = merged[rating_col].to_numpy(dtype=np.float64)
        v_in = vals[in_mask]
        v_out = vals[~in_mask]
        n1, n0 = len(v_in), len(v_out)
        s_pool = math.sqrt(((n1 - 1) * v_in.var(ddof=1) + (n0 - 1) * v_out.var(ddof=1)) / (n1 + n0 - 2))
        d_est = float((v_in.mean() - v_out.mean()) / s_pool)
        # Mann-Whitney U / AUROC probability naive listeners rate target song > non-target song
        auroc = float(np.mean(v_in[:, None] > v_out[None, :]) + 0.5 * np.mean(v_in[:, None] == v_out[None, :]))
        d_draws = np.empty(2000, dtype=np.float64)
        for b in range(2000):
            bi = v_in[rng_nhs.integers(0, n1, size=n1)]
            bo = v_out[rng_nhs.integers(0, n0, size=n0)]
            sp = math.sqrt(((n1 - 1) * bi.var(ddof=1) + (n0 - 1) * bo.var(ddof=1)) / (n1 + n0 - 2))
            d_draws[b] = (bi.mean() - bo.mean()) / sp
        legibility[cat_name.lower()] = {
            "category": cat_name,
            "label": label,
            "rating_column": rating_col,
            "n_target_songs": int(n1),
            "n_other_songs": int(n0),
            "target_mean": float(v_in.mean()),
            "other_mean": float(v_out.mean()),
            "cohen_d": d_est,
            "cohen_d_ci_95": np.quantile(d_draws, [0.025, 0.975]).tolist(),
            "auroc": auroc,
        }

    def eth_coded_share(col_name: str) -> dict:
        s = df_eth[col_name].astype(str).str.strip()
        coded = s[s.isin(["0", "1", "0.0", "1.0"])]
        pos = int(coded.isin(["1", "1.0"]).sum())
        tot = int(len(coded))
        return {
            "coded_performances": tot,
            "positive_performances": pos,
            "share_of_coded": float(pos / tot),
            "share_of_all_4709": float(pos / len(df_eth)),
            "wilson_ci_95": wilson_ci_95(pos, tot),
        }

    sd_s = df_eth["singers_dance"].astype(str).str.strip()
    ad_s = df_eth["audience_dance"].astype(str).str.strip()
    any_dance_coded = (sd_s.isin(["0", "1"])) | (ad_s.isin(["0", "1"]))
    any_dance_pos = int(((sd_s == "1") | (ad_s == "1")).sum())
    any_dance_tot = int(any_dance_coded.sum())

    def valid_link_count(col: str) -> int:
        vals = df_cult[col].astype(str).str.strip()
        return int((~vals.isin(["", ".", "nan", "None", "0"])).sum())

    return {
        "discography_sha256": OPEN_INPUTS["nhs_discography_metadata.csv"]["sha256"],
        "naive_listeners_sha256": OPEN_INPUTS["nhs_naive_listeners.csv"]["sha256"],
        "ethnography_sha256": OPEN_INPUTS["nhs_ethnography_prim.csv"]["sha256"],
        "cultures_sha256": OPEN_INPUTS["nhs_cultures_metadata.csv"]["sha256"],
        "discography_songs": int(len(merged)),
        "discography_societies": int(merged["id_nhs"].nunique()),
        "naive_listener_raters_published": 29357,
        "ethnography_performances": int(len(df_eth)),
        "cultures_total": int(len(df_cult)),
        "cross_linked_cultures": {
            "dplace_linked": valid_link_count("id_dplace"),
            "ethnographic_atlas_linked": valid_link_count("id_ea"),
            "sccs_linked": valid_link_count("id_sccs"),
            "binford_forager_linked": valid_link_count("id_binford"),
        },
        "acoustic_legibility_gradient": legibility,
        "lullaby_to_love_d_ratio": float(legibility["lullaby"]["cohen_d"] / legibility["love"]["cohen_d"]),
        "dance_to_love_d_ratio": float(legibility["dance"]["cohen_d"] / legibility["love"]["cohen_d"]),
        "ethnographic_hearth_contexts": {
            "singers_dance": eth_coded_share("singers_dance"),
            "audience_dance": eth_coded_share("audience_dance"),
            "any_communal_dance": {
                "coded_performances": any_dance_tot,
                "positive_performances": any_dance_pos,
                "share_of_coded": float(any_dance_pos / any_dance_tot),
                "wilson_ci_95": wilson_ci_95(any_dance_pos, any_dance_tot),
            },
            "informal_everyday": eth_coded_share("informal"),
            "ceremonial": eth_coded_share("ceremony"),
            "infant_child_directed": eth_coded_share("child_for"),
            "child_performed": eth_coded_share("child_by"),
            "live_improvisation": eth_coded_share("improv"),
        },
    }


def analyze_hcmst_2017(dta_path: Path) -> dict:
    """Stanford How Couples Meet and Stay Together 2017 (Rosenfeld, Thomas & Hausen) courtship audit."""
    rng_h = np.random.default_rng(SEED + 501)
    df = pd.read_stata(dta_path, convert_categoricals=False)
    assert len(df) == 3510

    pstat = df["partnership_status"].to_numpy(dtype=np.float64)
    ever_partnered = np.isin(pstat, [1.0, 2.0, 3.0])
    intact_vec = np.isin(pstat, [1.0, 2.0]).astype(np.float64)
    married_vec = df["w6_married"].to_numpy(dtype=np.float64)
    qual_vec = df["w6_relationship_quality"].to_numpy(dtype=np.float64)
    t_met_rel = df["time_from_met_to_rel"].to_numpy(dtype=np.float64)
    yr_met = df["year_fraction_met"].to_numpy(dtype=np.float64)
    rel_dur = df["relate_duration_at_w6_years"].to_numpy(dtype=np.float64)
    age_resp = df["ppage"].to_numpy(dtype=np.float64)
    no_prior_tie = df["w6_friend_connect_4_all"].to_numpy(dtype=np.float64)
    q24_valid = df["hcm2017q24_summary_all_codes"].notna().to_numpy()

    def col_bool(cname: str) -> np.ndarray:
        return (df[cname].to_numpy(dtype=np.float64) == 1.0) & q24_valid

    venue_masks = {
        "online_dating_app": col_bool("hcm2017q24_internet_dating"),
        "all_online": (df["hcm2017q24_met_online"].to_numpy(dtype=np.float64) == 1.0) & q24_valid,
        "bar_restaurant": col_bool("hcm2017q24_bar_restaurant"),
        "friends": col_bool("hcm2017q24_met_through_friend"),
        "coworkers": col_bool("hcm2017q24_met_as_through_cowork"),
        "church_worship": col_bool("hcm2017q24_church"),
        "family": col_bool("hcm2017q24_met_through_family"),
        "school_college": (col_bool("hcm2017q24_school") | col_bool("hcm2017q24_college")),
    }

    def summarize_group(gmask: np.ndarray) -> dict:
        tm_ok = gmask & np.isfinite(t_met_rel) & (t_met_rel >= 0.0)
        t_vals = t_met_rel[tm_ok]
        nt = len(t_vals)
        t_draws = np.array([t_vals[rng_h.integers(0, nt, size=nt)].mean() for _ in range(1500)], dtype=np.float64)
        mar_ok = gmask & np.isfinite(married_vec)
        tie_ok = gmask & np.isfinite(no_prior_tie)
        int_ok = gmask & ever_partnered
        q_ok = gmask & (intact_vec == 1.0) & np.isfinite(qual_vec) & (qual_vec >= 1.0) & (qual_vec <= 5.0)
        return {
            "n_couples": int(gmask.sum()),
            "pre_romance_acquaintance_years": {
                "n": int(nt),
                "mean": float(t_vals.mean()),
                "ci_95": np.quantile(t_draws, [0.025, 0.975]).tolist(),
                "median": float(np.median(t_vals)),
                "iqr": np.quantile(t_vals, [0.25, 0.75]).tolist(),
                "share_zero_years": float((t_vals == 0.0).mean()),
                "share_ge_1_year": float((t_vals >= 1.0).mean()),
            },
            "zero_prior_social_tie_share": float(no_prior_tie[tie_ok].mean()),
            "married_share": float(married_vec[mar_ok].mean()),
            "intact_share": float(intact_vec[int_ok].mean()),
            "intact_relationship_quality_1_to_5": mean_se_ci(qual_vec[q_ok]),
        }

    by_venue = {v_key: summarize_group(v_mask) for v_key, v_mask in venue_masks.items()}
    inst_kin_mask = (
        venue_masks["school_college"]
        | venue_masks["family"]
        | venue_masks["church_worship"]
        | venue_masks["coworkers"]
    )
    inst_kin_summary = summarize_group(inst_kin_mask)

    by_witness_bridge = {
        "zero_prior_tie": summarize_group(q24_valid & (no_prior_tie == 1.0)),
        "prior_friend_or_network_bridge": summarize_group(q24_valid & (no_prior_tie == 0.0)),
    }

    # Historical meeting eras (1940s-2017): disintermediation of family/school/church by online stranger markets
    eras = (
        ("pre_1980", 1900.0, 1980.0),
        ("1980_1994", 1980.0, 1995.0),
        ("1995_2004", 1995.0, 2005.0),
        ("2005_2011", 2005.0, 2012.0),
        ("2012_2017", 2012.0, 2018.0),
    )
    historical_eras = {}
    for e_key, y_lo, y_hi in eras:
        em = q24_valid & np.isfinite(yr_met) & (yr_met >= y_lo) & (yr_met < y_hi)
        ne = int(em.sum())
        tm_e = em & np.isfinite(t_met_rel) & (t_met_rel >= 0.0)
        tie_e = em & np.isfinite(no_prior_tie)
        historical_eras[e_key] = {
            "n_couples": ne,
            "online_all_share": float(venue_masks["all_online"][em].mean()),
            "online_dating_app_share": float(venue_masks["online_dating_app"][em].mean()),
            "friends_share": float(venue_masks["friends"][em].mean()),
            "family_share": float(venue_masks["family"][em].mean()),
            "school_college_share": float(venue_masks["school_college"][em].mean()),
            "coworkers_share": float(venue_masks["coworkers"][em].mean()),
            "church_share": float(venue_masks["church_worship"][em].mean()),
            "zero_prior_social_tie_share": float(no_prior_tie[tie_e].mean()),
            "mean_pre_romance_years": float(t_met_rel[tm_e].mean()),
        }

    # Duration- and age-adjusted OLS boundary check on intact relationship quality (1-5 scale)
    q_reg_mask = (
        q24_valid
        & (intact_vec == 1.0)
        & np.isfinite(qual_vec)
        & (qual_vec >= 1.0)
        & (qual_vec <= 5.0)
        & np.isfinite(rel_dur)
        & np.isfinite(age_resp)
    )
    app_ind = venue_masks["online_dating_app"][q_reg_mask].astype(np.float64)
    y_q = qual_vec[q_reg_mask]
    X_q = np.column_stack([
        np.ones(int(q_reg_mask.sum()), dtype=np.float64),
        app_ind,
        rel_dur[q_reg_mask],
        age_resp[q_reg_mask],
    ])
    beta_q = float(np.linalg.lstsq(X_q, y_q, rcond=None)[0][1])
    nq = int(q_reg_mask.sum())
    bq_draws = np.empty(1500, dtype=np.float64)
    for b in range(1500):
        bi = rng_h.integers(0, nq, size=nq)
        bq_draws[b] = np.linalg.lstsq(X_q[bi], y_q[bi], rcond=None)[0][1]

    app_mean_t = by_venue["online_dating_app"]["pre_romance_acquaintance_years"]["mean"]

    # Cohabitation Rush Hazard & Marital-vs.-Nonmarital Breakup Initiation Asymmetry
    rng_h_v15 = np.random.default_rng(SEED + 551)
    df_cat = pd.read_stata(dta_path, convert_categoricals=True)
    t_rel_cohab = pd.to_numeric(df["time_from_rel_to_cohab"], errors="coerce").to_numpy(dtype=np.float64)
    yr_relstart = pd.to_numeric(df["year_fraction_relstart"], errors="coerce").to_numpy(dtype=np.float64)
    ended_mar = df_cat["w6_relationship_end_mar"].isin(["divorce", "separation with no divorce"]).to_numpy()
    ended_nonmar = (df_cat["w6_relationship_end_nonmar"] == "We broke up").to_numpy()
    broke_up_vec = (ended_mar | ended_nonmar).astype(np.float64)

    cohab_ok = np.isfinite(t_rel_cohab) & (t_rel_cohab >= 0.0)
    cohab_windows = {}
    for w_key, c_lo, c_hi, inc_lo in (
        ("le_3m_rush", 0.0, 0.25, True),
        ("gt_3m_to_1y", 0.25, 1.0, False),
        ("gt_1y_to_2_5y", 1.0, 2.5, False),
        ("gt_2_5y", 2.5, 100.0, False),
    ):
        cm_w = cohab_ok & ((t_rel_cohab >= c_lo) if inc_lo else (t_rel_cohab > c_lo)) & (t_rel_cohab <= c_hi)
        nw_c = int(cm_w.sum())
        kb_c = int(broke_up_vec[cm_w].sum())
        cohab_windows[w_key] = {
            "n_couples": nw_c,
            "breakups": kb_c,
            "breakup_share": float(kb_c / nw_c),
            "wilson_ci_95": wilson_ci_95(kb_c, nw_c),
        }

    reg_cohab_m = cohab_ok & np.isfinite(yr_relstart) & np.isfinite(age_resp)
    X_coh = np.column_stack([
        np.ones(int(reg_cohab_m.sum()), dtype=np.float64),
        (t_rel_cohab[reg_cohab_m] <= 0.25).astype(np.float64),
        yr_relstart[reg_cohab_m] - 2000.0,
        age_resp[reg_cohab_m] - 45.0,
    ])
    y_coh = broke_up_vec[reg_cohab_m]
    beta_coh_pp = float(100.0 * np.linalg.lstsq(X_coh, y_coh, rcond=None)[0][1])
    nc_reg = int(reg_cohab_m.sum())
    b_coh_draws = np.empty(1500, dtype=np.float64)
    for b_i in range(1500):
        bi_c = rng_h_v15.integers(0, nc_reg, size=nc_reg)
        b_coh_draws[b_i] = 100.0 * np.linalg.lstsq(X_coh[bi_c], y_coh[bi_c], rcond=None)[0][1]

    cohabitation_rush_hazard = {
        "epistemic_tier": "observational_retrospective_cohort_age_adjusted",
        "n_ever_cohabited": int(cohab_ok.sum()),
        "by_romance_to_cohab_window": cohab_windows,
        "le_3m_to_gt_2_5y_breakup_ratio": float(
            cohab_windows["le_3m_rush"]["breakup_share"] / cohab_windows["gt_2_5y"]["breakup_share"]
        ),
        "cohort_age_adjusted_rush_le_3m_pp": {
            "n": nc_reg,
            "estimate": beta_coh_pp,
            "ci_95": np.quantile(b_coh_draws, [0.025, 0.975]).tolist(),
        },
    }

    opp_sex_m = pd.to_numeric(df["w6_same_sex_couple"], errors="coerce").to_numpy(dtype=np.float64) == 0.0
    resp_gender_s = df_cat["ppgender"].astype(str).to_numpy()

    def summarize_who_wanted_breakup(col_name: str) -> dict:
        col_vals = df_cat[col_name]
        w_more = m_more = eq_cnt = 0
        m_resp_wife_more = m_resp_self_more = 0
        f_resp_self_more = f_resp_husb_more = 0
        for is_opp, rg, raw_w in zip(opp_sex_m, resp_gender_s, col_vals):
            if not is_opp or pd.isna(raw_w):
                continue
            ws = str(raw_w)
            if "Refused" in ws or ws in ("", "nan"):
                continue
            if "equally" in ws:
                eq_cnt += 1
            elif ws.startswith("I wanted"):
                if rg == "Female":
                    w_more += 1
                    f_resp_self_more += 1
                elif rg == "Male":
                    m_more += 1
                    m_resp_self_more += 1
            elif "Partner" in ws:
                if rg == "Female":
                    m_more += 1
                    f_resp_husb_more += 1
                elif rg == "Male":
                    w_more += 1
                    m_resp_wife_more += 1
        tot_b = w_more + m_more + eq_cnt
        unil_b = w_more + m_more
        m_unil = m_resp_wife_more + m_resp_self_more
        f_unil = f_resp_self_more + f_resp_husb_more
        return {
            "n_total": int(tot_b),
            "woman_wanted_more": int(w_more),
            "man_wanted_more": int(m_more),
            "equal_mutual": int(eq_cnt),
            "equal_mutual_share": float(eq_cnt / tot_b),
            "woman_share_of_all": float(w_more / tot_b),
            "woman_share_of_unilateral": float(w_more / unil_b),
            "woman_unilateral_wilson_ci_95": wilson_ci_95(w_more, unil_b),
            "by_respondent_gender_unilateral": {
                "male_respondents_wife_wanted_more": int(m_resp_wife_more),
                "male_respondents_self_wanted_more": int(m_resp_self_more),
                "male_respondents_wife_share": float(m_resp_wife_more / m_unil),
                "female_respondents_self_wanted_more": int(f_resp_self_more),
                "female_respondents_partner_wanted_more": int(f_resp_husb_more),
                "female_respondents_self_share": float(f_resp_self_more / f_unil),
            },
        }

    breakup_initiation_asymmetry = {
        "married_divorce_or_separation": summarize_who_wanted_breakup("w6_who_breakup"),
        "nonmarital_breakup": summarize_who_wanted_breakup("w6_breakup_nonmar"),
    }

    return {
        "sha256": OPEN_INPUTS["hcmst2017_public_v1.1.dta"]["sha256"],
        "total_respondents": int(len(df)),
        "ever_partnered_respondents": int(ever_partnered.sum()),
        "meeting_coded_couples": int(q24_valid.sum()),
        "by_venue": by_venue,
        "institutional_and_kin_pooled": inst_kin_summary,
        "by_witness_bridge": by_witness_bridge,
        "historical_eras": historical_eras,
        "audit_compression_ratios": {
            "school_to_app_mean_ratio": float(by_venue["school_college"]["pre_romance_acquaintance_years"]["mean"] / app_mean_t),
            "institutional_kin_to_app_mean_ratio": float(inst_kin_summary["pre_romance_acquaintance_years"]["mean"] / app_mean_t),
            "friends_to_app_mean_ratio": float(by_venue["friends"]["pre_romance_acquaintance_years"]["mean"] / app_mean_t),
        },
        "intact_quality_boundary_check": {
            "epistemic_tier": "observational_surviving_union_duration_age_adjusted",
            "n_intact_scored": nq,
            "online_app_minus_offline_pts_1_to_5": beta_q,
            "ci_95": np.quantile(bq_draws, [0.025, 0.975]).tolist(),
        },
        "cohabitation_rush_hazard": cohabitation_rush_hazard,
        "breakup_initiation_asymmetry": breakup_initiation_asymmetry,
    }


def analyze_okcupid_jse(profiles_zip_path: Path, essays_zip_path: Path) -> dict:
    """Authorized JSE_OkCupid (Kim & Escobedo-Land 2015, N=59,946 profiles + 10-prompt essays)."""
    import re as _re
    import zipfile

    with zipfile.ZipFile(profiles_zip_path) as zfp:
        df_p = pd.read_csv(zfp.open("profiles_revised.csv"))
    with zipfile.ZipFile(essays_zip_path) as zfe:
        df_e = pd.read_csv(zfe.open("essays_revised_and_shuffled.csv"))
    assert len(df_p) == 59946 and len(df_e) == 59946

    # 1. Sexual-orientation natural experiment on 6'0" stature staging & strategic income disclosure (min cell n >= 100)
    ori_groups = {}
    for sx_code, sx_name in (("m", "men"), ("f", "women")):
        for ori in ("straight", "gay", "bisexual"):
            sub = df_p[(df_p["sex"] == sx_code) & (df_p["orientation"] == ori)]
            n_sub = int(len(sub))
            assert n_sub >= 100
            inc = sub["income"].to_numpy(dtype=np.float64)
            n_conceal = int((inc == -1).sum())
            n_disc = int((inc > 0).sum())
            n_100k = int((inc >= 100000).sum())
            grp_entry = {
                "n": n_sub,
                "income_disclosure": {
                    "concealed_minus1_count": n_conceal,
                    "concealed_minus1_share": float(n_conceal / n_sub),
                    "disclosed_positive_count": n_disc,
                    "disclosed_positive_share": float(n_disc / n_sub),
                    "disclosed_wilson_ci_95": wilson_ci_95(n_disc, n_sub),
                    "ge_100k_count": n_100k,
                    "ge_100k_all_share": float(n_100k / n_sub),
                    "ge_100k_among_disclosed_share": float(n_100k / n_disc) if n_disc >= 100 else None,
                },
            }
            if sx_code == "m":
                hv = sub["height"].dropna()
                hv_valid = hv[(hv >= 58) & (hv <= 82)]
                n_hv = int(len(hv_valid))
                hc = hv_valid.value_counts()
                n71, n72, n73 = int(hc.get(71.0, 0)), int(hc.get(72.0, 0)), int(hc.get(73.0, 0))
                by_inch = {}
                for inc_i in range(67, 76):
                    cnt_i = int(hc.get(float(inc_i), 0))
                    by_inch[str(inc_i)] = (
                        {"n": cnt_i, "share": float(cnt_i / n_hv)} if cnt_i >= 100 else None
                    )
                grp_entry["stature"] = {
                    "n_valid_58_to_82": n_hv,
                    "by_inch_67_to_75": by_inch,
                    "n_5ft11": n71,
                    "n_6ft0": n72,
                    "n_6ft1": n73,
                    "jump_5ft11_to_6ft0_pct": float(100.0 * (n72 - n71) / n71),
                    "drop_6ft0_to_6ft1_pct": float(100.0 * (n73 - n72) / n72),
                }
            ori_groups[f"{sx_name}_{ori}"] = grp_entry

    disc_str_m = ori_groups["men_straight"]["income_disclosure"]["disclosed_positive_share"]
    disc_gay_m = ori_groups["men_gay"]["income_disclosure"]["disclosed_positive_share"]
    disc_str_w = ori_groups["women_straight"]["income_disclosure"]["disclosed_positive_share"]

    # 2. 10-Prompt Multi-Domain Essay Portfolio & Assortative Mind Law (59,946 rows x 10 prompts)
    prompt_labels = (
        ("essay0", "My self summary"),
        ("essay1", "What I'm doing with my life"),
        ("essay2", "I'm really good at"),
        ("essay3", "First thing people notice about me"),
        ("essay4", "Favorite books, movies, shows, music, food"),
        ("essay5", "Six things I could never do without"),
        ("essay6", "I spend a lot of time thinking about"),
        ("essay7", "On a typical Friday night I am"),
        ("essay8", "Most private thing I'm willing to admit"),
        ("essay9", "You should message me if..."),
    )
    prompt_stats = {}
    total_nonempty = 0
    for col_k, p_lbl in prompt_labels:
        s_clean = df_e[col_k].fillna("").astype(str).str.strip()
        ne_cnt = int((s_clean.str.len() >= 5).sum())
        total_nonempty += ne_cnt
        prompt_stats[col_k] = {
            "label": p_lbl,
            "nonempty_ge5_chars": ne_cnt,
            "share": float(ne_cnt / len(df_e)),
        }

    word_re = _re.compile(r"[a-zA-Z]{2,}")
    lens_08 = np.zeros((len(df_e), 9), dtype=np.float64)
    long_word_rate = np.full((len(df_e), 9), np.nan, dtype=np.float64)
    for k in range(9):
        for idx, txt in enumerate(df_e[f"essay{k}"].fillna("").astype(str)):
            ws = word_re.findall(txt)
            lens_08[idx, k] = len(ws)
            if len(ws) >= 8:
                long_word_rate[idx, k] = sum(len(w) >= 7 for w in ws) / len(ws)

    corr_len = np.corrcoef(np.log1p(lens_08).T)
    mean_cross_len_r = float((corr_len.sum() - 9.0) / 72.0)
    valid_all9 = ~np.isnan(long_word_rate).any(axis=1)
    corr_lex = np.corrcoef(long_word_rate[valid_all9].T)
    r1_lex = float((corr_lex.sum() - 9.0) / 72.0)
    sb_curve = {str(k): float(k * r1_lex / (1.0 + (k - 1) * r1_lex)) for k in range(1, 10)}

    e9_raw = df_e["essay9"].fillna("").astype(str).str.lower()
    has_e9_10 = (e9_raw.str.strip().str.len() >= 10).to_numpy()
    valid_dom_cnt = (~np.isnan(long_word_rate)).sum(axis=1)
    mean_lex = np.zeros(len(df_e), dtype=np.float64)
    ge3_mask = valid_dom_cnt >= 3
    mean_lex[ge3_mask] = np.nansum(long_word_rate[ge3_mask], axis=1) / valid_dom_cnt[ge3_mask]
    mean_lex[~ge3_mask] = np.nan

    ask_mind_arr = e9_raw.str.contains(
        r"\b(?:smart|intelligen|curious|intellectual|conversation|philosoph|science|book|read|nerd|geek|witty|banter|clever|thoughtful)\b",
        regex=True,
    ).to_numpy()
    ask_kind_arr = e9_raw.str.contains(
        r"\b(?:kind|genuine|honest|sincere|authentic|caring|warm|compassion|empath|good heart)\b",
        regex=True,
    ).to_numpy()
    ask_surf_arr = e9_raw.str.contains(
        r"\b(?:tall|6'|6 ft|6ft|six foot|six pack|6 pack|hot|rich|wealthy|millionaire|sugar daddy)\b",
        regex=True,
    ).to_numpy()

    sub_assort = has_e9_10 & np.isfinite(mean_lex)
    q_idx = pd.qcut(mean_lex[sub_assort], 5, labels=False)
    assort_quintiles = {}
    for q in range(5):
        qm = q_idx == q
        nq_c = int(qm.sum())
        assert nq_c >= 100
        cm = int(ask_mind_arr[sub_assort][qm].sum())
        ck = int(ask_kind_arr[sub_assort][qm].sum())
        cs = int(ask_surf_arr[sub_assort][qm].sum())
        assort_quintiles[f"Q{q + 1}"] = {
            "n": nq_c,
            "ask_mind_curiosity_share": float(cm / nq_c),
            "ask_mind_wilson_ci_95": wilson_ci_95(cm, nq_c),
            "ask_kindness_sincerity_share": float(ck / nq_c),
            "ask_kindness_wilson_ci_95": wilson_ci_95(ck, nq_c),
            "ask_height_money_looks_share": float(cs / nq_c),
        }

    def lexical_scan(series_clean: pd.Series, spec_dict: dict) -> dict:
        n_tot = int(len(series_clean))
        out = {"n_answered": n_tot, "themes": {}}
        for t_key, pat in spec_dict.items():
            cnt = int(series_clean.str.contains(pat, regex=True).sum())
            out["themes"][t_key] = {
                "count": cnt,
                "share": float(cnt / n_tot),
                "wilson_ci_95": wilson_ci_95(cnt, n_tot),
            }
        return out

    e9_s = e9_raw[e9_raw.str.strip().str.len() >= 5]
    e9_res = lexical_scan(e9_s, {
        "humor_banter": r"\b(?:laugh|funny|humor|humour|wit|witty|banter|sarcas|joke|smile)\b",
        "kindness_sincerity": r"\b(?:kind|sweet|genuine|honest|sincere|authentic|caring|warm|nice|good heart|down to earth|real)\b",
        "curiosity_intelligence": r"\b(?:smart|intelligen|curious|intellectual|conversation|talk about|deep|thoughtful|nerd|geek|clever)\b",
        "adventure_outdoors": r"\b(?:adventur|travel|explor|spontaneous|hike|outdoors|try new|dance)\b",
        "tall_6ft_height": r"\b(?:tall|taller|6'|6 ft|6ft|six foot|six feet|height)\b",
        "hot_sixpack_looks": r"\b(?:six pack|6 pack|abs|muscular|jacked|supermodel|hot|gorgeous)\b",
        "rich_wealthy_sixfig": r"\b(?:rich|wealthy|millionaire|six figure|6 figure|money|financially)\b",
    })
    e9_t = e9_res["themes"]
    e9_res["ratios"] = {
        "humor_to_tall": float(e9_t["humor_banter"]["share"] / e9_t["tall_6ft_height"]["share"]),
        "kindness_to_tall": float(e9_t["kindness_sincerity"]["share"] / e9_t["tall_6ft_height"]["share"]),
        "mind_to_tall": float(e9_t["curiosity_intelligence"]["share"] / e9_t["tall_6ft_height"]["share"]),
        "humor_to_wealth": float(e9_t["humor_banter"]["share"] / e9_t["rich_wealthy_sixfig"]["share"]),
    }

    e3_raw = df_e["essay3"].fillna("").astype(str).str.lower()
    e3_s = e3_raw[e3_raw.str.strip().str.len() >= 3]
    e3_res = lexical_scan(e3_s, {
        "smile_laugh": r"\b(?:smile|smiling|laugh|laughter|grin)\b",
        "eyes": r"\b(?:eyes|eye)\b",
        "hair": r"\b(?:hair|curls|beard)\b",
        "height": r"\b(?:tall|height|short)\b",
        "voice_energy": r"\b(?:voice|accent|energy|enthusiasm|loud)\b",
    })

    e5_raw = df_e["essay5"].fillna("").astype(str).str.lower()
    e5_s = e5_raw[e5_raw.str.strip().str.len() >= 5]
    e5_res = lexical_scan(e5_s, {
        "family_friends_loved_ones": r"\b(?:family|friends|mom|dad|sister|brother|dog|loved ones|community)\b",
        "food_coffee_water": r"\b(?:food|coffee|water|tea|chocolate|wine|cheese|tacos)\b",
        "music_books_art": r"\b(?:music|book|books|reading|art|guitar|spotify)\b",
        "laughter_affection": r"\b(?:laugh|laughter|humor|hugs|affection|love|sex|cuddle)\b",
        "money_luxury": r"\b(?:money|cash|wealth|bmw|mercedes|rolex|designer)\b",
    })
    e5_res["family_to_money_ratio"] = float(
        e5_res["themes"]["family_friends_loved_ones"]["share"] / e5_res["themes"]["money_luxury"]["share"]
    )

    # 3. Open-Science San Francisco Childless Fertility-Desire Crossover
    s_str = df_p[df_p["orientation"] == "straight"].copy()
    off = s_str["offspring"].fillna("")
    is_childless = (
        off.str.contains(r"doesn&rsquo;t have kids|doesn&rsquo;t want kids|^wants kids$|^might want kids$", regex=True)
        & (~off.str.contains(r"has a kid|has kids", regex=True))
    )
    wants_any = off.str.contains(r"wants them|might want them|^wants kids$|^might want kids$", regex=True)
    s_str["is_childless"] = is_childless
    s_str["wants_any"] = wants_any
    sf_bands = {}
    band_defs = ((20, 24, 22.0), (25, 29, 27.0), (30, 34, 32.0), (35, 39, 37.0), (40, 44, 42.0), (45, 54, 49.5))
    prev_mid, prev_gap = None, None
    cross_age_sf = None
    for lo, hi, mid_a in band_defs:
        sub_b = s_str[(s_str["age"] >= lo) & (s_str["age"] <= hi) & s_str["is_childless"]]
        sub_w = sub_b[sub_b["sex"] == "f"]
        sub_m = sub_b[sub_b["sex"] == "m"]
        nw_b, nm_b = int(len(sub_w)), int(len(sub_m))
        assert nw_b >= 100 and nm_b >= 100
        cw_b, cm_b = int(sub_w["wants_any"].sum()), int(sub_m["wants_any"].sum())
        pw_b, pm_b = float(cw_b / nw_b), float(cm_b / nm_b)
        gap_pp = 100.0 * (pw_b - pm_b)
        if prev_gap is not None and prev_gap > 0.0 and gap_pp <= 0.0 and cross_age_sf is None:
            cross_age_sf = float(prev_mid + (mid_a - prev_mid) * prev_gap / (prev_gap - gap_pp))
        prev_mid, prev_gap = mid_a, gap_pp
        sf_bands[f"{lo}-{hi}"] = {
            "midpoint_age": mid_a,
            "women": {"n": nw_b, "want_or_might_count": cw_b, "share": pw_b, "wilson_ci_95": wilson_ci_95(cw_b, nw_b)},
            "men": {"n": nm_b, "want_or_might_count": cm_b, "share": pm_b, "wilson_ci_95": wilson_ci_95(cm_b, nm_b)},
            "women_minus_men_pp": gap_pp,
        }

    return {
        "profiles_sha256": OPEN_INPUTS["okcupid_profiles_revised.csv.zip"]["sha256"],
        "essays_sha256": OPEN_INPUTS["okcupid_essays_revised_and_shuffled.csv.zip"]["sha256"],
        "privacy_revision_note": "2021 JSE privacy revision: essay rows randomly permuted relative to demographic rows, ages jittered, social URLs scrubbed; all cells enforce n >= 100.",
        "total_profiles": int(len(df_p)),
        "sex_counts": {"men": int((df_p["sex"] == "m").sum()), "women": int((df_p["sex"] == "f").sum())},
        "orientation_staging_experiment": {
            "groups": ori_groups,
            "straight_to_gay_male_income_disclosure_ratio": float(disc_str_m / disc_gay_m),
            "straight_male_to_female_income_disclosure_ratio": float(disc_str_m / disc_str_w),
        },
        "ten_prompt_portfolio": {
            "total_rows": int(len(df_e)),
            "total_nonempty_essays_ge5_chars": int(total_nonempty),
            "prompts": prompt_stats,
            "multi_domain_lexical_validity": {
                "cross_prompt_log_length_mean_r": mean_cross_len_r,
                "n_complete_9_domain_writers": int(valid_all9.sum()),
                "single_prompt_reliability_r1": r1_lex,
                "spearman_brown_curve_k_1_to_9": sb_curve,
                "k9_to_k1_validity_ratio": float(sb_curve["9"] / sb_curve["1"]),
            },
            "assortative_mind_requests_essay9": {
                "n_users_ge3_domains_with_essay9": int(sub_assort.sum()),
                "by_lexical_quintile": assort_quintiles,
                "q5_to_q1_mind_request_ratio": float(
                    assort_quintiles["Q5"]["ask_mind_curiosity_share"] / assort_quintiles["Q1"]["ask_mind_curiosity_share"]
                ),
            },
            "essay9_message_me_if": e9_res,
            "essay3_first_noticed": e3_res,
            "essay5_six_essentials": e5_res,
        },
        "sf_childless_fertility_crossover": {
            "n_childless_straight_coded": int(s_str["is_childless"].sum()),
            "by_age_band": sf_bands,
            "interpolated_crossover_age": float(cross_age_sf),
        },
    }


def build_macro_selection_kpis(result: dict) -> dict:
    pp = result["paleoclimate_plateau"]
    hb = result["holoface_benchmark"]
    sd = result["speed_dating"]
    pov = result["pov"]
    hon = result["hon"]
    osyn = result["open_synthesis"]
    kow = osyn["kowal_2025"]
    joel = osyn["joel_2020"]
    lid = osyn["lidborg_2022"]
    v9 = osyn["v9_anchors"]
    cc = osyn["childhood_coevolution"]
    nhs = osyn["nhs_campfire"]
    hcm = osyn["hcmst_2017"]
    okc = osyn["okcupid_jse"]

    pvc_500 = hb["psychometric_voter_ceiling"]["tiers"]["votes_500_plus"]
    gss_inc = v9["gss_three_episodes_income"]
    gss_voc = v9["gss_three_episodes_all"]["vocabulary"]
    chk_hw = sd["checklist_and_hindsight"]["hobby_checklist_vs_live_wavelength"]
    sat_b = sd["checklist_and_hindsight"]["objective_sat_blindness"]
    chk_kow = kow["checklist_and_surplus_decomposition"]["pooled"]["predictors_sd_per_sd"]

    return {
        "schema_version": 2,
        "kpis": [
            {
                "id": "kpi1_campfire_childhood_release",
                "name": "Campfire & Childhood Brain Release",
                "question": "Why did hominin brains stall at 908 cm3 (1.50-0.70 Ma) before surging to ~1,385 cm3, and why did big brains require pair-bond love?",
                "scale": f"{pp['plateau_summary']['gingerich_n']} dated crania + 57-core LR04 + {cc['sccs_alloparenting_and_sharing']['early_childhood_nonmaternal_majority_sccs52']['coded_societies']} SCCS childhood-care societies",
                "headline_value": float(pp["plateau_summary"]["plateau_level_cm3"]),
                "childhood_peak_rmr_pct": float(cc["kuzawa_2014_brain_metabolism"]["peak_rmr_pct"]),
                "sccs_early_childhood_nonmaternal_majority_share": float(cc["sccs_alloparenting_and_sharing"]["early_childhood_nonmaternal_majority_sccs52"]["share"]),
                "lead_over_crises_kyr": [
                    float(pp["timing_gap_summary"]["onset_to_mis34_cooling_kyr"]),
                    float(pp["timing_gap_summary"]["onset_to_bolaven_impact_kyr"]),
                ],
                "post_cooking_acceleration_pct_myr": 50.0,
                "boundary_check": (
                    f"Exclusive maternal early-childhood care is 0/{cc['sccs_alloparenting_and_sharing']['early_childhood_exclusive_mother_sccs52']['coded_societies']} (0.0%) in SCCS; "
                    f"cooking without pair-bond devotion stalls at ~{cc['proposition_22_e22_coevolution']['regimes']['cooking_without_pair_bond_love']['equilibrium_cm3_closed_form']:,.0f} cm3 in E22."
                ),
                "formal_ref": "Propositions 11-12 & 22",
            },
            {
                "id": "kpi2_multimodal_legibility_wall",
                "name": "Multi-Modal Legibility Wall",
                "question": "How much latent human character and romantic devotion remains invisible to zero-context strangers across photos, talk, and song?",
                "scale": "20,000 HoloFace-L1 portraits + 8,378 speed dates (2,989 w/ SAT) + 118 NHS songs (29,357 raters)",
                "photo_blindness_share_500v": float(pvc_500["irreducible_single_context_gap_full"]),
                "four_min_sincerity_blindness": float(1.0 - v9["speed_jury_curves"]["sinc"][0]),
                "four_min_objective_sat_r2": float(sat_b["stranger_intel_sat_r2"]),
                "nhs_lullaby_dance_vs_love_cohen_d": [
                    float(nhs["acoustic_legibility_gradient"]["lullaby"]["cohen_d"]),
                    float(nhs["acoustic_legibility_gradient"]["dance"]["cohen_d"]),
                    float(nhs["acoustic_legibility_gradient"]["love"]["cohen_d"]),
                ],
                "boundary_check": (
                    f"4-min stranger intel explains 0.34% of objective SAT (r={sat_b['r_stranger_intel_vs_partner_sat']['estimate']:+.2f}); "
                    f"lullabies (+{nhs['acoustic_legibility_gradient']['lullaby']['cohen_d']:.2f}) beat love songs (+{nhs['acoustic_legibility_gradient']['love']['cohen_d']:.2f}) {nhs['lullaby_to_love_d_ratio']:.1f}x."
                ),
                "formal_ref": "Proposition 20",
            },
            {
                "id": "kpi3_play_vs_checklist_spark",
                "name": "Conversational Play vs. Hobby-Tag Null",
                "question": "Does live romantic spark come from matching profile hobby tags or from unique two-person conversational play?",
                "scale": f"{sd['dyadic_chemistry']['variance_partition']['like']['n']:,} speed-date ratings + {chk_hw['n_common_dyad_decisions']:,} objective 17-hobby dyad decisions + Waves 1-3 follow-up",
                "unbiased_partner_consensus_share": float(sd["dyadic_chemistry"]["variance_partition"]["like"]["unbiased_srm_shares"]["partner_consensus"]),
                "unbiased_unique_dyadic_share": float(sd["dyadic_chemistry"]["variance_partition"]["like"]["unbiased_srm_shares"]["unique_dyadic_and_noise"]),
                "objective_hobby_corr_yes_r": float(chk_hw["r_objective_hobby_corr_yes"]["estimate"]),
                "live_shared_wavelength_yes_r": float(chk_hw["r_live_shared_wavelength_yes"]["estimate"]),
                "wavelength_to_hobby_tag_ratio": float(chk_hw["live_to_checklist_r_ratio"]),
                "women_next_day_looks_jump_pts": float(sd["checklist_and_hindsight"]["within_person_shifts"]["nextday_realized_minus_pre_stated"]["women"]["delta_points"]["attr"]["estimate"]),
                "boundary_check": "Objective 17-hobby overlap is null for Yes (r=+0.014); women catch +10.3 pt 24h looks hijack before intelligence snaps back (+3.7 pts).",
                "formal_ref": "Proposition 21(ii)",
            },
            {
                "id": "kpi4_voice_vs_checkbox_inversion",
                "name": "Voice-vs.-Checkbox Inversion",
                "question": "What do humans ask for in their own words vs. how searchable platform filters route attention?",
                "scale": f"{pov['self_written_voice']['eligible_with_essay']['Pooled']:,} POV essays + {okc['ten_prompt_portfolio']['total_nonempty_essays_ge5_chars']:,} OkCupid 10-prompt essays + {hon['beauty_reference']['Male']['reference_n'] + hon['beauty_reference']['Female']['reference_n']:,} HON profiles",
                "female_family_vs_tall_ratio": float(pov["self_written_voice"]["voice_to_checkbox_ratios"]["female_family_to_tall"]),
                "female_honesty_vs_tall_ratio": float(pov["self_written_voice"]["voice_to_checkbox_ratios"]["female_honesty_to_tall"]),
                "male_6ft_cliff_drop_pct": float(pov["height_percentiles"]["by_sex"]["Male"]["cliff_5ft11_6ft0_6ft1"]["drop_6ft0_to_6ft1_pct"]),
                "okcupid_straight_vs_gay_6ft_jump_pct": [
                    float(okc["orientation_staging_experiment"]["groups"]["men_straight"]["stature"]["jump_5ft11_to_6ft0_pct"]),
                    float(okc["orientation_staging_experiment"]["groups"]["men_gay"]["stature"]["jump_5ft11_to_6ft0_pct"]),
                ],
                "male_likes_gini": float(hon["attention_and_portfolio"]["by_sex"]["Male"]["likes_gini"]),
                "boundary_check": (
                    f"Straight men surge +17.1% at 6'0\" (24.4% income disc.) vs -1.3% in gay men (14.4% disc.) on same UI; "
                    f"male stature fertility is null (r=+0.006)."
                ),
                "formal_ref": "Propositions 8 & 13",
            },
            {
                "id": "kpi5_app_witness_audit_compression",
                "name": "10x App Witness-Audit Compression",
                "question": "How sharply has digital courtship compressed the pre-romantic everyday witness observation that historically screened character?",
                "scale": f"{hcm['total_respondents']:,} Stanford HCMST U.S. respondents ({hcm['meeting_coded_couples']:,} meeting-coded couples, 1940s-2017)",
                "app_pre_romance_mean_years": float(hcm["by_venue"]["online_dating_app"]["pre_romance_acquaintance_years"]["mean"]),
                "app_pre_romance_median_years": float(hcm["by_venue"]["online_dating_app"]["pre_romance_acquaintance_years"]["median"]),
                "institutional_kin_pre_romance_mean_years": float(hcm["institutional_and_kin_pooled"]["pre_romance_acquaintance_years"]["mean"]),
                "institutional_kin_to_app_ratio": float(hcm["audit_compression_ratios"]["institutional_kin_to_app_mean_ratio"]),
                "intact_app_vs_offline_quality": [
                    float(hcm["by_venue"]["online_dating_app"]["intact_relationship_quality_1_to_5"]["mean"]),
                    float(hcm["institutional_and_kin_pooled"]["intact_relationship_quality_1_to_5"]["mean"]),
                ],
                "boundary_check": (
                    f"Conditional on surviving intact, app unions match offline quality "
                    f"({hcm['by_venue']['online_dating_app']['intact_relationship_quality_1_to_5']['mean']:.2f} vs "
                    f"{hcm['institutional_and_kin_pooled']['intact_relationship_quality_1_to_5']['mean']:.2f}; "
                    f"adj diff {hcm['intact_quality_boundary_check']['online_app_minus_offline_pts_1_to_5']:+.2f}, null) — apps strip pre-entry screening."
                ),
                "formal_ref": "Proposition 19",
            },
            {
                "id": "kpi6_entry_to_bond_love_reversal",
                "name": "Love Reversal & Marrying-Up Null",
                "question": "Do money, checklists, or 'marrying out of your league' build lasting love, or does unconditional kindness govern the pair bond?",
                "scale": f"117,293 adults ({kow['checklist_and_surplus_decomposition']['n_total']:,} couples, 175 countries) + 43 longitudinal studies (11,196 couples)",
                "male_resources_entry_pp_per_sd": float(kow["partnering_slopes"]["by_sex"]["men"]["resources"]["pp_per_sd"]),
                "partner_resources_bond_sd_women": float(kow["pair_bond_reversal"]["relationship_quality_prqc"]["by_sex"]["women"]["traits"]["resources"]["sd_per_sd"]),
                "partner_kindness_bond_sd_women": float(kow["pair_bond_reversal"]["relationship_quality_prqc"]["by_sex"]["women"]["traits"]["kindness"]["sd_per_sd"]),
                "idiographic_checklist_fit_sd": float(chk_kow["idiographic_checklist_fit"]["estimate"]),
                "marrying_up_surplus_five_trait_sd": float(kow["checklist_and_surplus_decomposition"]["five_trait_controlled_marrying_up_sd"]),
                "joel_dyadic_vs_partner_trait_r2_pct": [
                    float(joel["satisfaction"]["models"]["ActorRelBaseline"]["unweighted_mean_r2_pct"]),
                    float(joel["satisfaction"]["models"]["PartnerTraitBaseline"]["unweighted_mean_r2_pct"]),
                ],
                "boundary_check": (
                    f"Kindness (+{chk_kow['partner_kindness']['estimate']:.2f} SD) beats checklist fit "
                    f"(+{chk_kow['idiographic_checklist_fit']['estimate']:.2f} SD), money (+{chk_kow['partner_resources']['estimate']:.2f} SD), "
                    f"and marrying-up ({chk_kow['marrying_up_surplus']['estimate']:+.2f} SD) across egalitarian & collectivist regimes."
                ),
                "formal_ref": "Propositions 21 & 22",
            },
            {
                "id": "kpi7_horizon_crossover_vs_devotion",
                "name": "Age-41 Horizon Flip vs. Unbroken Devotion",
                "question": "Which male-female decisions flip with biological clocks, and which stay invariant from 18 to 75?",
                "scale": "1,141,106 POV + 59,946 OkCupid-JSE + 67,743 Kowal + 46,853 age-gap records + 7,060 Whyte",
                "childless_want_crossover_age": float(pov["childless_crossover"]["band_midpoint_interpolated_age"]),
                "okcupid_sf_crossover_age": float(okc["sf_childless_fertility_crossover"]["interpolated_crossover_age"]),
                "partner_age_scissors_60_79_years": float(kow["partner_age_gap_by_age"]["brackets"]["60-79"]["women_minus_men_years"]),
                "serious_commitment_gap_25_29_and_45_49_pp": [
                    float(100.0 * (pov["market_and_commitment"]["25-29"]["by_sex"]["Female"]["serious_commitment"]["share"] - pov["market_and_commitment"]["25-29"]["by_sex"]["Male"]["serious_commitment"]["share"])),
                    float(100.0 * (pov["market_and_commitment"]["45-49"]["by_sex"]["Female"]["serious_commitment"]["share"] - pov["market_and_commitment"]["45-49"]["by_sex"]["Male"]["serious_commitment"]["share"])),
                ],
                "partnered_happiness_lift_pts": [
                    float(osyn["whyte_2021"]["happiness_by_partnered"]["Women"]["raw_lift_pts"]["estimate"]),
                    float(osyn["whyte_2021"]["happiness_by_partnered"]["Men"]["raw_lift_pts"]["estimate"]),
                ],
                "boundary_check": (
                    f"Commitment at 25+ yrs ({kow['triangular_love_arc']['by_duration']['25y+']['commitment']['mean']:.2f}/5) exceeds Year 1 "
                    f"({kow['triangular_love_arc']['by_duration']['<=1y']['commitment']['mean']:.2f}); women weight emotional over sexual betrayal "
                    f"({kow['jealousy_asymmetry']['by_sex']['women']['emotional_1_to_7']['mean']:.2f} vs {kow['jealousy_asymmetry']['by_sex']['women']['sexual_1_to_7']['mean']:.2f}/7)."
                ),
                "formal_ref": "Propositions 9 & 21",
            },
            {
                "id": "kpi8_modern_pairing_gate",
                "name": "The 94% Modern Pairing Gate",
                "name_short": "94% Pairing Gate",
                "question": "Did modern fertility fall because humans stopped wanting children or because single-context markets stall union entry?",
                "scale": "34,092 U.S. GSS respondents (1972-2024, incl. objective WORDSUM) + Lidborg 232 meta-analytic effects",
                "non_marriage_share_of_birth_drop_pct": 94.0,
                "height_vs_vigor_reproduction_r": [
                    float(lid["height"]["reproduction_all"]["re_mean_r"]),
                    float(lid["body_strength"]["reproduction_all"]["re_mean_r"]),
                ],
                "gss_male_income_pairing_vs_within_marriage": [
                    float(gss_inc["men"]["pairing"][0]),
                    float(gss_inc["men"]["number"][0]),
                ],
                "gss_male_wordsum_pairing_vs_within_marriage": [
                    float(gss_voc["men"]["pairing"][0]),
                    float(gss_voc["men"]["number"][0]),
                ],
                "boundary_check": "Marriage pairing is positive on income (+0.046), schooling (+0.015), and WORDSUM (+0.011), while within-marriage births invert all three.",
                "formal_ref": "Propositions 16-19",
            },
        ],
    }


def analyze_proposition_23_e23() -> dict:
    """Proposition 23 (E23): Secondary courtship staging, witness-rushed lock-in, unilateral marital exit, & second-act re-entry."""
    rng_e23 = np.random.default_rng(SEED + 801)
    n_sim = 40000

    def norm_cdf_scalar(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # (i) Secondary courtship staging moral hazard (Dating open competition vs. Married lock-in shield)
    gamma_mkt_mu, gamma_mkt_sd = 2.20, 0.30
    c_effort = 0.01
    lam_shield = 0.17
    gammas = rng_e23.normal(gamma_mkt_mu, gamma_mkt_sd, size=n_sim)
    e_date_cf = gamma_mkt_mu / c_effort
    e_mar_cf = (1.0 - lam_shield) * gamma_mkt_mu / c_effort
    e_date_sim = float(np.mean(gammas / c_effort + rng_e23.normal(0.0, 4.0, size=n_sim)))
    e_mar_sim = float(np.mean((1.0 - lam_shield) * gammas / c_effort + rng_e23.normal(0.0, 4.0, size=n_sim)))

    # (ii) Pre-cohabitation witness rush hazard (months m of unstaged observation before cohabitation)
    mu_p, sigma_p, sigma_eps = 0.50, 0.60, 1.20
    theta_enter, theta_stay = 0.35, 0.17
    p_true = rng_e23.normal(mu_p, sigma_p, size=n_sim)
    witness_curves = {}
    z_grid = np.linspace(theta_enter, mu_p + 6.0 * sigma_p, 800)
    dz = float(z_grid[1] - z_grid[0])
    erf_vec = np.vectorize(math.erf)
    for m_months in (2, 6, 18, 30):
        inv_tau = 1.0 / (1.0 / (sigma_p ** 2) + m_months / (sigma_eps ** 2))
        v_hat = max(sigma_p ** 2 - inv_tau, 1e-9)
        sd_hat = math.sqrt(v_hat)
        sd_post = math.sqrt(inv_tau)
        p_enter_cf = 1.0 - norm_cdf_scalar((theta_enter - mu_p) / sd_hat)
        dens_z = np.exp(-0.5 * ((z_grid - mu_p) / sd_hat) ** 2) / (math.sqrt(2.0 * math.pi) * sd_hat)
        cond_fail_z = 0.5 * (1.0 + erf_vec((theta_stay - z_grid) / (sd_post * math.sqrt(2.0))))
        breakup_cf = float(np.sum(cond_fail_z * dens_z) * dz / p_enter_cf)

        y_bar = p_true + rng_e23.normal(0.0, sigma_eps / math.sqrt(m_months), size=n_sim)
        p_hat = inv_tau * (mu_p / (sigma_p ** 2) + (m_months / (sigma_eps ** 2)) * y_bar)
        entered = p_hat >= theta_enter
        breakup_sim = float(np.mean(p_true[entered] < theta_stay))
        witness_curves[f"m_{m_months}"] = {
            "months": m_months,
            "breakup_closed_form": breakup_cf,
            "breakup_sim": breakup_sim,
            "abs_diff": float(abs(breakup_cf - breakup_sim)),
        }

    # (iii) Symmetric non-marital breakups vs. asymmetric female unilateral divorce under marital lock-in
    mu_u, sigma_u = 0.65, 0.55
    q_def, delta_f, delta_m = 0.35, 0.72, 0.32
    pf0 = norm_cdf_scalar(-mu_u / sigma_u)
    pm0 = norm_cdf_scalar(-mu_u / sigma_u)
    pf1 = norm_cdf_scalar(-(mu_u - delta_f) / sigma_u)
    pm1 = norm_cdf_scalar(-(mu_u - delta_m) / sigma_u)
    nonmar_w_unil_cf = 0.5
    mar_w_only_cf = (1.0 - q_def) * pf0 * (1.0 - pm0) + q_def * pf1 * (1.0 - pm1)
    mar_m_only_cf = (1.0 - q_def) * pm0 * (1.0 - pf0) + q_def * pm1 * (1.0 - pf1)
    mar_w_unil_cf = float(mar_w_only_cf / (mar_w_only_cf + mar_m_only_cf))

    uf_nm = rng_e23.normal(mu_u, sigma_u, size=n_sim) < 0.0
    um_nm = rng_e23.normal(mu_u, sigma_u, size=n_sim) < 0.0
    nonmar_w_unil_sim = float(np.sum(uf_nm & (~um_nm)) / np.sum(uf_nm ^ um_nm))

    def_shock = rng_e23.random(size=n_sim) < q_def
    uf_m = (rng_e23.normal(mu_u, sigma_u, size=n_sim) - def_shock * delta_f) < 0.0
    um_m = (rng_e23.normal(mu_u, sigma_u, size=n_sim) - def_shock * delta_m) < 0.0
    mar_w_unil_sim = float(np.sum(uf_m & (~um_m)) / np.sum(uf_m ^ um_m))

    # (iv) Second-act re-entry divergence (Single vs. Divorced vs. Widowed formal marriage intent)
    reentry_specs = {
        "single": (-1.445, -1.495),
        "divorced": (-1.590, -1.535),
        "widowed": (-1.360, -1.287),
    }
    reentry_e23 = {}
    for st_name, (zf, zm) in reentry_specs.items():
        cf_f, cf_m = norm_cdf_scalar(zf), norm_cdf_scalar(zm)
        sim_f = float(np.mean(zf + rng_e23.normal(0.0, 1.0, size=n_sim) > 0.0))
        sim_m = float(np.mean(zm + rng_e23.normal(0.0, 1.0, size=n_sim) > 0.0))
        reentry_e23[st_name] = {
            "women_marry_closed_form": cf_f,
            "women_marry_sim": sim_f,
            "men_marry_closed_form": cf_m,
            "men_marry_sim": sim_m,
            "men_minus_women_closed_form_pp": float(100.0 * (cf_m - cf_f)),
            "abs_max_diff": float(max(abs(cf_f - sim_f), abs(cf_m - sim_m))),
        }

    return {
        "n_sim": n_sim,
        "part_i_staging_moral_hazard": {
            "dating_min_closed_form": float(e_date_cf),
            "dating_min_sim": e_date_sim,
            "married_min_closed_form": float(e_mar_cf),
            "married_min_sim": e_mar_sim,
            "drop_pct_closed_form": float(100.0 * (e_mar_cf - e_date_cf) / e_date_cf),
            "abs_diff_min": float(max(abs(e_date_cf - e_date_sim), abs(e_mar_cf - e_mar_sim))),
        },
        "part_ii_witness_rush_hazard": {
            "by_months": witness_curves,
            "rush_m2_to_patient_m30_ratio": float(
                witness_curves["m_2"]["breakup_closed_form"] / witness_curves["m_30"]["breakup_closed_form"]
            ),
        },
        "part_iii_unilateral_divorce_flip": {
            "nonmarital_woman_unilateral_closed_form": nonmar_w_unil_cf,
            "nonmarital_woman_unilateral_sim": nonmar_w_unil_sim,
            "married_woman_unilateral_closed_form": mar_w_unil_cf,
            "married_woman_unilateral_sim": mar_w_unil_sim,
            "abs_max_diff": float(
                max(abs(nonmar_w_unil_cf - nonmar_w_unil_sim), abs(mar_w_unil_cf - mar_w_unil_sim))
            ),
        },
        "part_iv_second_act_reentry": reentry_e23,
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
    joel_sat_path = resolve_input("joel2020_SatisfactionMetaAnalyses.csv", inputs_dir, fallbacks)
    joel_comm_path = resolve_input("joel2020_CommitmentMetaAnalyses.csv", inputs_dir, fallbacks)
    sccs_data_path = resolve_input("dplace_sccs_data.csv", inputs_dir, fallbacks)
    sccs_codes_path = resolve_input("dplace_sccs_codes.csv", inputs_dir, fallbacks)
    hcmst_path = resolve_input("hcmst2017_public_v1.1.dta", inputs_dir, fallbacks)
    nhs_disc_path = resolve_input("nhs_discography_metadata.csv", inputs_dir, fallbacks)
    nhs_naive_path = resolve_input("nhs_naive_listeners.csv", inputs_dir, fallbacks)
    nhs_eth_path = resolve_input("nhs_ethnography_prim.csv", inputs_dir, fallbacks)
    nhs_cult_path = resolve_input("nhs_cultures_metadata.csv", inputs_dir, fallbacks)
    okc_prof_path = resolve_input("okcupid_profiles_revised.csv.zip", inputs_dir, fallbacks)
    okc_ess_path = resolve_input("okcupid_essays_revised_and_shuffled.csv.zip", inputs_dir, fallbacks)

    return {
        "seed": SEED,
        "whyte_2021": analyze_whyte(whyte_path, rng),
        "lidborg_2022": analyze_lidborg(lidborg_h_path, lidborg_b_path),
        "kowal_2025": analyze_kowal(kowal_path, hon_countries, rng),
        "walter_2020": analyze_walter(walter_path),
        "joel_2020": analyze_joel_2020(joel_sat_path, joel_comm_path),
        "betzig_sccs": analyze_betzig_sccs(sccs_data_path, sccs_codes_path),
        "childhood_coevolution": analyze_childhood_coevolution(sccs_data_path, sccs_codes_path),
        "nhs_campfire": analyze_nhs_campfire(nhs_disc_path, nhs_naive_path, nhs_eth_path, nhs_cult_path),
        "hcmst_2017": analyze_hcmst_2017(hcmst_path),
        "okcupid_jse": analyze_okcupid_jse(okc_prof_path, okc_ess_path),
        "two_way_jury": analyze_two_way_jury(rng),
        "proposition_23_e23": analyze_proposition_23_e23(),
        "published_benchmarks": {
            "havlicek_2020_mhc": list(HAVLICEK_2020_MHC),
            "eastwick_2025_composites": list(EASTWICK_2025_COMPOSITES),
            "jankowiak_fischer_1992_love": dict(JANKOWIAK_FISCHER_1992),
        },
        "v9_anchors": load_v9_anchors(root_dir),
    }
