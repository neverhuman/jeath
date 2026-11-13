"""Confidence Before Outcomes: audience, relative-status, and cross-arena evidence for PBO.

This module is deliberately separate from ``performance_open.py`` so that it composes with
concurrent v16 work there.  It reads only open, checksum-verified inputs, uses its own random
streams (``SEED + 1001`` upward), and never touches the ``pov``/``hon``/``speed_dating`` blocks
that *The Minds We Choose* requires to stay equal to the PBO reference.

Contents
--------
G1  GSS (1972--2024) audience for women leaders: ``fepol`` trend by respondent sex, audience
    composition (sex, age, race, degree), and a Firebaugh within-cohort / cohort-replacement
    accounting split with Rao--Wu design replicates (reused from ``scripts/brain/minds_gss.py``).
H1  HCMST 2017: who earned more versus "excellent" relationship quality in different-sex unions.
Q1  Speed dating: the institution-median SAT (``mn_sat``) legibility estimate re-bootstrapped by
    institution, partner, and session clusters (relabelling support; v15 numbers untouched).
E25 Proposition 25 (dominance direction): closed-form witness threshold versus Monte Carlo.
E26 Proposition 26 (shared-stereotype ceiling and outcome test): closed forms versus Monte Carlo.
D1  Speed dating: the chooser's pre-outcome forecast of the partner's decision versus the partner's
    record with other people, and calibration before meeting and at the end of the night.
E27 Proposition 27 (agreement breeds confidence): closed-form agreement and validity versus Monte Carlo.
L1  Cross-arena entry-versus-outcome ledger of verified published estimates (``LEDGER``).
S1  Cross-theory prediction scorecard (``SCORECARD``; Tier-4 synthesis, explicitly not a test).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261111
SCHEMA = "pbo-confidence-1"

CONFIDENCE_INPUTS = {
    "gss_all.rda": {
        "sha256": "38712b98fe23598856f757c096e5d4be908f6a9f644f695fce62ba0db4e4627f",
        "url": "https://raw.githubusercontent.com/kjhealy/gssr/9ed92cf9e065396fedb7cc60a4590ddb25b64179/data/gss_all.rda",
        "citation": "gss2024",
        "license": "GSS public-use data (NORC); redistributed via kjhealy/gssr",
    },
    "hcmst2017_public_v1.1.dta": {
        "sha256": "8f21fe52e1d9fb5ad286fc68ef9ac74c4a4ade8fbeca5b45a64087083b06a64c",
        "url": "https://raw.githubusercontent.com/shivi-a/how_couples_meet/master/HCMST%202017%20fresh%20sample%20for%20public%20sharing%20draft%20v1.1.dta",
        "citation": "rosenfeld2019",
        "license": "Stanford HCMST 2017 public release",
    },
    "speed_dating.csv": {
        "sha256": "fda9a924b18880fabe6c7a71961cff20aa3c7cc2a5aa8d6e823c2fc41893814c",
        "url": "https://raw.githubusercontent.com/tsizzle/speed_dating_experiment/master/Speed_Dating_Data.csv",
        "citation": "fisman2006",
        "license": "Fisman et al. (2006) Columbia speed-dating public data",
    },
}

# Pre-specified confirmatory family (Holm-adjusted).  Everything else is exploratory.
CONFIRMATORY = (
    "h1_she_earns_more_vs_he_earns_more_excellent_pp",
    "g1_recent_audience_men_minus_women_agree_pp",
    "g1_cohort_replacement_component_pp",
)
GSS_EARLY = (1975, 1978)  # first fully design-supported fepol window (1972-74 lack VSTRAT)
GSS_LATE = (2018, 2024)
GSS_LATE_PRE_MODE = (2010, 2018)  # sensitivity: before the 2021+ web/mixed-mode transition
GSS_AUDIENCE_EARLY = (1975, 1985)
GSS_AUDIENCE_RECENT = (2016, 2024)


# ----------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve(name: str, inputs_dir: Path) -> Path:
    spec = CONFIDENCE_INPUTS[name]
    path = Path(inputs_dir) / name
    if not path.exists():
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != spec["sha256"]:
            raise ValueError(f"Downloaded checksum mismatch for {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise ValueError(f"Checksum mismatch for {path}: {digest} != {spec['sha256']}")
    return path


def quantile_ci(draws: np.ndarray) -> list[float]:
    return [float(v) for v in np.quantile(np.asarray(draws, dtype=np.float64), [0.025, 0.975])]


def bootstrap_p(draws: np.ndarray, estimate: float) -> float:
    """Two-sided p from the replicate distribution's normal approximation (SD of replicates)."""
    sd = float(np.std(draws, ddof=1))
    if sd <= 0.0:
        return 0.0 if estimate != 0.0 else 1.0
    return float(math.erfc(abs(estimate) / sd / math.sqrt(2.0)))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m = len(keys)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(keys):
        running = max(running, min(1.0, (m - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def normal_cdf(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z, dtype=np.float64) / math.sqrt(2.0)))


def normal_pdf(z):
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def inverse_mills(z):
    return normal_pdf(z) / (1.0 - normal_cdf(z))


def weighted_ols_hc1(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Survey-weighted least squares with HC1 (pweight) sandwich covariance."""
    XtWX = X.T @ (X * w[:, None])
    bread = np.linalg.inv(XtWX)
    beta = bread @ (X.T @ (w * y))
    resid = y - X @ beta
    meat = (X * (w * resid)[:, None]).T @ (X * (w * resid)[:, None])
    n, p = X.shape
    cov = bread @ meat @ bread * n / (n - p)
    return beta, cov


# ----------------------------------------------------------------------------- G1: GSS audience
GSS_VARIABLES = "year sex race age cohort degree fepol fepres fepresch fehire wtssps vstrat vpsu sample".split()


def load_gss(path: Path, root: Path):
    sys.path.insert(0, str(root / "project/brain/tips/Performance_Before_Outcomes_v9/code"))
    import rda_reader  # frozen v9 archive reader (read-only)

    data, labels, missing = rda_reader.read_columns(path, GSS_VARIABLES)
    if missing:
        raise ValueError("Missing GSS variables " + repr(missing))
    return data, labels


def design_replicates(data: dict, weight: np.ndarray, root: Path):
    sys.path.insert(0, str(root / "scripts/brain"))
    import minds_gss  # reuse the Rao--Wu year-by-stratum PSU bootstrap exactly

    valid, group, multiplier, info = minds_gss.design_multiplier(data, weight)
    return valid, group, multiplier.astype(np.float64), info, minds_gss.REPLICATES


class Replicated:
    """Weighted means and least squares with Rao--Wu replicate weights via PSU sufficient sums."""

    def __init__(self, weight, group, valid, multiplier):
        self.w = weight
        self.g = np.where(valid, group, -1)
        self.M = multiplier
        self.n_psu = multiplier.shape[1]

    def _psu_sum(self, values, mask):
        ok = mask & (self.g >= 0)
        return np.bincount(self.g[ok], weights=values[ok], minlength=self.n_psu)

    def mean(self, y, mask):
        """Weighted mean on the design-supported part of the domain.

        Returns (estimate, replicates, n_used, supported).  When no row of the domain is design
        supported (1972-74, which lack VSTRAT) the estimate uses every row and replicates are None.
        """
        ok = mask & np.isfinite(y)
        sup = ok & (self.g >= 0)
        if not sup.any():
            return float(np.sum(self.w[ok] * y[ok]) / np.sum(self.w[ok])), None, int(ok.sum()), False
        est = float(np.sum(self.w[sup] * y[sup]) / np.sum(self.w[sup]))
        sw = self._psu_sum(self.w, sup)
        swy = self._psu_sum(self.w * np.where(sup, y, 0.0), sup)
        reps = (self.M @ swy) / (self.M @ sw)
        return est, reps, int(sup.sum()), True

    def ols(self, X, y, mask):
        ok = mask & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        w = self.w[ok]
        Xo, yo = X[ok], y[ok]
        beta = np.linalg.solve(Xo.T @ (Xo * w[:, None]), Xo.T @ (w * yo))
        g = self.g[ok]
        if np.any(g < 0):
            raise ValueError("Replicated OLS requires design-supported rows only")
        p = X.shape[1]
        xtx = np.zeros((self.n_psu, p, p))
        xty = np.zeros((self.n_psu, p))
        np.add.at(xtx, g, (Xo[:, :, None] * Xo[:, None, :]) * w[:, None, None])
        np.add.at(xty, g, Xo * (w * yo)[:, None])
        rep_xtx = (self.M @ xtx.reshape(self.n_psu, p * p)).reshape(-1, p, p)
        rep_xty = self.M @ xty
        reps = np.linalg.solve(rep_xtx, rep_xty[..., None])[..., 0]
        return beta, reps, int(ok.sum())


def kish_interval(y, w):
    p = float(np.sum(w * y) / np.sum(w))
    neff = float(np.sum(w) ** 2 / np.sum(w * w))
    se = math.sqrt(max(p * (1.0 - p), 1e-12) / neff)
    return [p - 1.96 * se, p + 1.96 * se], neff


def in_window(year, window):
    return (year >= window[0]) & (year <= window[1])


def firebaugh(rep: Replicated, y, year, cohort, domain, early_window, late_window) -> dict:
    """Firebaugh (1989) linear decomposition of social change with design replicates.

    Fits y = b0 + b1*year + b2*cohort on the design-supported domain between the two windows
    (no separate age effect), then splits the fitted change between windows into within-cohort
    change b1*d(mean year) and cohort replacement b2*d(mean cohort).  The observed minus fitted
    change is reported as the nonlinearity residual rather than absorbed.
    """
    base = (domain & np.isfinite(y) & np.isfinite(cohort) & (year >= early_window[0])
            & (year <= late_window[1]) & (rep.g >= 0))
    cohort0 = np.where(np.isfinite(cohort), cohort, 1950.0)
    Xf = np.column_stack([np.ones(len(year)), (year - 2000.0) / 10.0, (cohort0 - 1950.0) / 10.0])
    beta_f, reps_f, n_f = rep.ols(Xf, y, base)
    early = base & in_window(year, early_window)
    late = base & in_window(year, late_window)

    def point_mean(v, m):
        return float(np.sum(rep.w[m] * v[m]) / np.sum(rep.w[m]))

    def rep_mean(v, m):
        return (rep.M @ rep._psu_sum(rep.w * np.where(m, v, 0.0), m)) / (rep.M @ rep._psu_sum(rep.w, m))

    def components(mean, coefs):
        dyear = mean(Xf[:, 1], late) - mean(Xf[:, 1], early)
        dcoh = mean(Xf[:, 2], late) - mean(Xf[:, 2], early)
        dobs = mean(y, late) - mean(y, early)
        within, replacement = coefs[..., 1] * dyear, coefs[..., 2] * dcoh
        return np.stack(np.broadcast_arrays(dobs, within, replacement, within + replacement,
                                            dobs - within - replacement), axis=-1)

    point = components(point_mean, beta_f)
    out_rep = components(rep_mean, reps_f)
    yhat = Xf @ beta_f
    direct_fitted = point_mean(yhat, late) - point_mean(yhat, early)
    keys = ("observed_change", "within_cohort_change", "cohort_replacement", "fitted_change", "nonlinearity_residual")
    return {
        "n": n_f,
        "early_window": list(early_window),
        "late_window": list(late_window),
        "fit_years": [int(early_window[0]), int(late_window[1])],
        "slope_per_decade_pp": {"year": 100.0 * float(beta_f[1]), "cohort": 100.0 * float(beta_f[2])},
        "components_pp": {
            key: {"estimate": 100.0 * float(point[j]), "ci_95": quantile_ci(100.0 * out_rep[:, j])}
            for j, key in enumerate(keys)
        },
        "replacement_share_of_fitted": float(point[2] / point[3]) if point[3] != 0 else 0.0,
        "identity_abs_error": float(abs(direct_fitted - point[3])),
        "assumption": "linear year and birth-cohort terms with no separate age effect (Firebaugh 1989); accounting split, not causal",
        "_replacement_reps_pp": 100.0 * out_rep[:, 2],
    }


def analyze_gss_audience(gss_path: Path, root: Path) -> dict:
    data, labels = load_gss(gss_path, root)
    year = data["year"]
    sex = data["sex"]  # 1 male, 2 female
    weight = np.where(np.isfinite(data["wtssps"]) & (data["wtssps"] > 0), data["wtssps"], np.nan)
    valid_d, group, multiplier, design_info, replicates = design_replicates(data, weight, root)
    rep = Replicated(np.nan_to_num(weight), group, valid_d, multiplier)

    fepol = data["fepol"]
    answered = np.isin(fepol, [1.0, 2.0]) & np.isfinite(weight)
    agree = np.where(answered, (fepol == 1.0).astype(np.float64), np.nan)
    years = sorted({int(v) for v in year[answered]})

    trend = {}
    for yr in years:
        cell = {}
        for s_lbl, s_code in (("men", 1.0), ("women", 2.0)):
            mask = answered & (year == yr) & (sex == s_code)
            est, reps, n, supported = rep.mean(agree, mask)
            if supported:
                ci, method = quantile_ci(reps), "rao_wu_design_replicates"
            else:
                ci, _ = kish_interval(agree[mask], weight[mask])
                method = "kish_effective_n_srs_approximation"
            cell[s_lbl] = {"n": n, "n_answered": int(mask.sum()), "agree_share": est, "ci_95": ci, "variance_method": method}
        trend[str(yr)] = cell

    # Oversample sensitivity (1982 Black oversample, sample codes 4/5/7).
    over = np.isin(data["sample"], [4.0, 5.0, 7.0])
    oversample_check = {}
    for yr in (1982,):
        diffs = {}
        for s_lbl, s_code in (("men", 1.0), ("women", 2.0)):
            base = answered & (year == yr) & (sex == s_code)
            full = float(np.average(agree[base], weights=weight[base]))
            excl = float(np.average(agree[base & ~over], weights=weight[base & ~over]))
            diffs[s_lbl] = {"with_oversample": full, "without_oversample": excl, "difference": full - excl}
        oversample_check[str(yr)] = diffs

    # Audience composition: early vs recent windows, by sex, age band, race, degree.
    age = data["age"]
    age_band = np.select([age < 35, age < 55, age >= 55], [0, 1, 2], default=-1)
    age_band = np.where(np.isfinite(age), age_band, -1)
    race = data["race"]
    ba = np.where(np.isfinite(data["degree"]), (data["degree"] >= 3.0).astype(np.float64), np.nan)
    segments = {
        "men": sex == 1.0,
        "women": sex == 2.0,
        "age_18_34": age_band == 0,
        "age_35_54": age_band == 1,
        "age_55_plus": age_band == 2,
        "white": race == 1.0,
        "black": race == 2.0,
        "other_race": race == 3.0,
        "bachelors_plus": ba == 1.0,
        "below_bachelors": ba == 0.0,
        "men_age_55_plus": (sex == 1.0) & (age_band == 2),
        "women_age_18_34": (sex == 2.0) & (age_band == 0),
    }
    audience = {}
    for w_lbl, window in (("early_1975_1985", GSS_AUDIENCE_EARLY), ("recent_2016_2024", GSS_AUDIENCE_RECENT)):
        win = answered & in_window(year, window)
        seg_out = {}
        for s_lbl, s_mask in segments.items():
            est, reps, n, supported = rep.mean(agree, win & s_mask)
            if not supported:
                raise ValueError("audience window lacks design support: " + s_lbl)
            seg_out[s_lbl] = {"n": n, "agree_share": est, "ci_95": quantile_ci(reps)}
        audience[w_lbl] = seg_out

    recent = answered & in_window(year, GSS_AUDIENCE_RECENT)
    e_m, r_m, n_m, _ = rep.mean(agree, recent & (sex == 1.0))
    e_w, r_w, n_w, _ = rep.mean(agree, recent & (sex == 2.0))
    sex_gap = {
        "estimate_pp": 100.0 * (e_m - e_w),
        "ci_95_pp": quantile_ci(100.0 * (r_m - r_w)),
        "p_two_sided": bootstrap_p(100.0 * (r_m - r_w), 100.0 * (e_m - e_w)),
        "n_men": n_m,
        "n_women": n_w,
    }

    # Mutually adjusted recent-window linear probability model (exploratory).
    rmask = recent & (age_band >= 0) & np.isin(race, [1.0, 2.0, 3.0]) & np.isfinite(ba)
    X = np.column_stack([
        np.ones(len(year)), (sex == 2.0).astype(float), (age_band == 1).astype(float),
        (age_band == 2).astype(float), (race == 2.0).astype(float), (race == 3.0).astype(float),
        np.nan_to_num(ba),
    ])
    names = ("intercept", "female", "age_35_54", "age_55_plus", "black", "other_race", "bachelors_plus")
    beta, reps_b, n_lpm = rep.ols(X, agree, rmask & (rep.g >= 0))
    adjusted = {
        "n": n_lpm,
        "window": list(GSS_AUDIENCE_RECENT),
        "coefficients_pp": {
            nm: {"estimate": 100.0 * float(beta[j]), "ci_95": quantile_ci(100.0 * reps_b[:, j])}
            for j, nm in enumerate(names)
        },
        "epistemic_tier": "exploratory",
    }

    decomposition = {}
    for s_lbl, s_mask in (("pooled", np.ones(len(year), bool)), ("men", sex == 1.0), ("women", sex == 2.0)):
        decomposition[s_lbl] = firebaugh(rep, agree, year, data["cohort"], answered & s_mask, GSS_EARLY, GSS_LATE)
    decomposition["pooled_pre_mode_change_sensitivity"] = firebaugh(
        rep, agree, year, data["cohort"], answered, GSS_EARLY, GSS_LATE_PRE_MODE)
    repl_reps = decomposition["pooled"].pop("_replacement_reps_pp")
    for key in list(decomposition):
        decomposition[key].pop("_replacement_reps_pp", None)
    repl = decomposition["pooled"]["components_pp"]["cohort_replacement"]
    repl["p_two_sided"] = bootstrap_p(repl_reps, repl["estimate"])

    # Exploratory companion items.
    fepres = data["fepres"]
    fepres_ok = np.isin(fepres, [1.0, 2.0]) & np.isfinite(weight)  # code 5 "wouldn't vote" -> missing
    would_not = np.where(fepres_ok, (fepres == 2.0).astype(float), np.nan)
    fepres_trend = {}
    for yr in sorted({int(v) for v in year[fepres_ok]}):
        cell = {}
        for s_lbl, s_code in (("men", 1.0), ("women", 2.0)):
            m = fepres_ok & (year == yr) & (sex == s_code)
            cell[s_lbl] = {"n": int(m.sum()), "would_not_vote_share": float(np.average(would_not[m], weights=weight[m]))}
        fepres_trend[str(yr)] = cell

    def agree_scale_trend(var, agree_codes, codes):
        ok = np.isin(data[var], codes) & np.isfinite(weight)
        val = np.where(ok, np.isin(data[var], agree_codes).astype(float), np.nan)
        out = {}
        for yr in sorted({int(v) for v in year[ok]}):
            m = ok & (year == yr)
            out[str(yr)] = {"n": int(m.sum()), "agree_share": float(np.average(val[m], weights=weight[m]))}
        return out

    exploratory = {
        "fepres_would_not_vote_for_qualified_woman": {"by_year_sex": fepres_trend, "note": "code 5 (would not vote) set missing; asked 1972-2010"},
        "fepresch_preschool_child_suffers_if_mother_works": agree_scale_trend("fepresch", [1.0, 2.0], [1.0, 2.0, 3.0, 4.0]),
        "fehire_employers_special_efforts_to_hire_women": agree_scale_trend("fehire", [1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]),
        "wording_note": "question wording from the GSS codebook (not stored in the .rda labels): FEPOL men better suited emotionally for politics; FEPRES vote for a qualified woman nominated for President; FEPRESCH preschool child likely to suffer if mother works; FEHIRE employers should make special efforts to hire and promote qualified women because of past discrimination",
        "epistemic_tier": "exploratory",
    }

    return {
        "source": {
            "file": "gss_all.rda",
            "sha256": CONFIDENCE_INPUTS["gss_all.rda"]["sha256"],
            "url": CONFIDENCE_INPUTS["gss_all.rda"]["url"],
            "reader_sha256": sha256_file(root / "project/brain/tips/Performance_Before_Outcomes_v9/code/rda_reader.py"),
            "rows": int(len(year)),
        },
        "item": "fepol: 'Most men are better suited emotionally for politics than are most women' (1 agree, 2 disagree)",
        "n_answered": int(answered.sum()),
        "years": years,
        "design": {k: v for k, v in design_info.items() if k in ("psus", "strata", "replicates", "seed", "bootstrap", "supported_records")},
        "trend_by_year_sex": trend,
        "oversample_sensitivity": oversample_check,
        "audience_composition": audience,
        "recent_audience_sex_gap": sex_gap,
        "recent_adjusted_lpm": adjusted,
        "firebaugh_decomposition": decomposition,
        "exploratory": exploratory,
        "epistemic_tier": "measured weighted survey shares (Tier 2) with accounting decomposition (Tier 3)",
        "limitations": [
            "Don't-know answers are missing in this release; shares are among substantive answers.",
            "1972-74 lack VSTRAT; 1974 fepol shares use a Kish effective-n approximation and are excluded from the decomposition.",
            "Firebaugh decomposition assumes no independent age effect; it is an accounting split, not a causal estimate.",
            "Attitude items measure stated beliefs of the audience, not behaviour toward specific women.",
        ],
    }


# ----------------------------------------------------------------------------- H1: HCMST relative income
def analyze_hcmst_relative_income(dta_path: Path) -> dict:
    df = pd.read_stata(dta_path, convert_categoricals=False)
    if len(df) != 3510:
        raise ValueError(f"HCMST 2017 public file should have 3510 respondents, found {len(df)}")
    return hcmst_relative_income_from_frame(df)


def hcmst_relative_income_from_frame(df: pd.DataFrame) -> dict:
    current = df["partnership_status"].isin([1, 2])
    different_sex = (df["w6_same_sex_couple"] == 0) & (df["ppgender"] != df["w6_q4"]) & df["w6_q4"].isin([1, 2])
    ages = df["ppage"].between(25, 64)
    weighted = df["weight1"].notna() & (df["weight1"] > 0)
    quality_ok = df["w6_relationship_quality"].between(1, 5)
    income_ok = df["w6_q23"].isin([1, 2, 3, 4])
    educ_ok = df["subject_yrsed"].notna() & df["partner_yrsed"].notna()
    sample = df[current & different_sex & ages & weighted & quality_ok & income_ok & educ_ok].copy()

    female = (sample["ppgender"] == 2).to_numpy()
    q23 = sample["w6_q23"].to_numpy()
    she_more = (female & (q23 == 1)) | (~female & (q23 == 3))
    he_more = (female & (q23 == 3)) | (~female & (q23 == 1))
    same = q23 == 2
    he_not_working = female & (q23 == 4)
    she_not_working = ~female & (q23 == 4)
    quality = sample["w6_relationship_quality"].to_numpy(dtype=np.float64)
    excellent = (quality == 5).astype(np.float64)
    w = sample["weight1"].to_numpy(dtype=np.float64)
    age = sample["ppage"].to_numpy(dtype=np.float64)
    married = (sample["partnership_status"] == 1).to_numpy(dtype=np.float64)
    ed_s = sample["subject_yrsed"].to_numpy(dtype=np.float64)
    ed_p = sample["partner_yrsed"].to_numpy(dtype=np.float64)

    def design(mask=None, with_female=True):
        cols = [np.ones(len(w)), she_more, same, he_not_working, she_not_working]
        names = ["intercept", "she_earns_more", "about_the_same", "male_partner_not_working", "female_partner_not_working"]
        if with_female:
            cols.append(female.astype(float))
            names.append("respondent_female")
        cols += [(age - 45.0) / 10.0, ((age - 45.0) / 10.0) ** 2, married, ed_s - 14.0, ed_p - 14.0]
        names += ["age_decades", "age_decades_sq", "married", "own_years_schooling", "partner_years_schooling"]
        X = np.column_stack(cols).astype(np.float64)
        if mask is not None:
            X = X[mask]
        keep = np.ptp(X, axis=0) > 0
        keep[0] = True
        return X[:, keep], [n for n, k in zip(names, keep) if k]

    def fit(y, mask=None, with_female=True):
        X, names = design(mask, with_female)
        yy = y if mask is None else y[mask]
        ww = w if mask is None else w[mask]
        beta, cov = weighted_ols_hc1(X, yy, ww)
        se = np.sqrt(np.diag(cov))
        return {
            nm: {"estimate": float(beta[j]), "se": float(se[j]), "ci_95": [float(beta[j] - 1.96 * se[j]), float(beta[j] + 1.96 * se[j])]}
            for j, nm in enumerate(names)
        }

    pooled = fit(excellent)
    by_sex = {"women_respondents": fit(excellent, female, False), "men_respondents": fit(excellent, ~female, False)}
    ordered_sensitivity = fit(quality)

    cells = {}
    for lbl, m in (("he_earns_more", he_more), ("she_earns_more", she_more), ("about_the_same", same),
                   ("male_partner_not_working", he_not_working), ("female_partner_not_working", she_not_working)):
        sub = {}
        for s_lbl, s_m in (("women_respondents", female), ("men_respondents", ~female)):
            mm = m & s_m
            if mm.sum() > 0:
                sub[s_lbl] = {"n": int(mm.sum()), "excellent_share": float(np.average(excellent[mm], weights=w[mm]))}
        cells[lbl] = sub

    est = 100.0 * pooled["she_earns_more"]["estimate"]
    se = 100.0 * pooled["she_earns_more"]["se"]
    return {
        "source": {"file": "hcmst2017_public_v1.1.dta", "sha256": CONFIDENCE_INPUTS["hcmst2017_public_v1.1.dta"]["sha256"]},
        "sample": {
            "definition": "current married/partnered (partnership_status 1-2), different-sex, ages 25-64, general-population weight1, valid w6_q23 and w6_q34, schooling observed",
            "n": int(len(sample)),
            "n_women_respondents": int(female.sum()),
            "n_men_respondents": int((~female).sum()),
            "n_she_earns_more": int(she_more.sum()),
            "n_he_earns_more": int(he_more.sum()),
        },
        "outcome": "excellent relationship quality (w6_relationship_quality == 5); ordered 1-5 sensitivity",
        "excellent_share_cells": cells,
        "pooled_lpm_excellent": pooled,
        "by_respondent_sex_lpm_excellent": by_sex,
        "ordered_1_to_5_lpm": ordered_sensitivity,
        "confirmatory": {
            "estimand": "she earns more vs he earns more, excellent quality, percentage points (weight1, HC1)",
            "estimate_pp": est,
            "ci_95_pp": [est - 1.96 * se, est + 1.96 * se],
            "p_two_sided": float(math.erfc(abs(est) / se / math.sqrt(2.0))),
        },
        "epistemic_tier": "measured cross-sectional association (Tier 2); intact unions only",
        "limitations": [
            "Cross-section of surviving unions: couples dissolved by income conflict are absent (survivorship).",
            "Relationship quality is near ceiling; the excellent-vs-not split is pre-specified.",
            "Who-earned-more is self-reported by one partner; no causal identification.",
        ],
    }


# ----------------------------------------------------------------------------- Q1: mn_sat re-bootstrap
def analyze_sat_proxy(csv_path: Path) -> dict:
    data = csv_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CONFIDENCE_INPUTS["speed_dating.csv"]["sha256"]:
        raise ValueError("speed-dating checksum mismatch")
    rows = list(csv.DictReader(data.decode("latin-1").splitlines()))

    def num(v):
        v = str(v).replace(",", "").strip()
        return float(v) if v else np.nan

    first = {}
    for r in rows:
        first.setdefault(int(float(r["iid"])), r)
    sat, inst = {}, {}
    for iid, r in sorted(first.items()):
        v = num(r.get("mn_sat", ""))
        if np.isfinite(v) and v > 400.0:
            sat[iid] = v
            inst[iid] = str(r.get("undergra", "")).strip().lower()
    wave = np.array([num(r["wave"]) for r in rows])
    partner = np.array([num(r["pid"]) for r in rows])
    yes = np.array([num(r["dec"]) for r in rows])
    intel = np.array([num(r["intel"]) for r in rows])
    intel[(intel < 1) | (intel > 10)] = np.nan
    psat = np.array([sat.get(int(p), np.nan) if np.isfinite(p) else np.nan for p in partner])
    mask = np.isfinite(psat) & np.isfinite(intel) & ((yes == 0) | (yes == 1))
    x, y_int, y_yes = psat[mask], intel[mask], yes[mask]
    r_intel = float(np.corrcoef(x, y_int)[0, 1])
    r_yes = float(np.corrcoef(x, y_yes)[0, 1])
    # Each institution maps to one mn_sat value, while free-text `undergra` spellings vary; the
    # value itself is therefore the conservative institution cluster.
    clusters = {
        "institution_sat_value": x.astype(str),
        "partner": partner[mask].astype(int).astype(str),
        "session": wave[mask].astype(int).astype(str),
    }
    rng = np.random.default_rng(SEED + 1021)
    intervals = {}
    for c_lbl, c_vec in clusters.items():
        uniq, inv = np.unique(c_vec, return_inverse=True)
        members = [np.flatnonzero(inv == j) for j in range(len(uniq))]
        draws_i, draws_y = np.empty(2000), np.empty(2000)
        for b in range(2000):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            idx = np.concatenate([members[j] for j in pick])
            draws_i[b] = np.corrcoef(x[idx], y_int[idx])[0, 1]
            draws_y[b] = np.corrcoef(x[idx], y_yes[idx])[0, 1]
        intervals[c_lbl] = {
            "clusters": int(len(uniq)),
            "r_stranger_intel_vs_partner_institution_sat_ci_95": quantile_ci(draws_i),
            "r_partner_institution_sat_vs_yes_ci_95": quantile_ci(draws_y),
        }
    # Panel (>=10 raters per target) version, re-bootstrapped by institution value.
    part_m = partner[mask]
    panel = []
    for pid_u in sorted(set(int(p) for p in part_m)):
        rows_p = part_m == pid_u
        if int(rows_p.sum()) >= 10:
            panel.append((sat[pid_u], float(y_int[rows_p].mean())))
    panel = np.array(panel)
    r_panel = float(np.corrcoef(panel[:, 0], panel[:, 1])[0, 1])
    uniq_v, inv_v = np.unique(panel[:, 0], return_inverse=True)
    members_v = [np.flatnonzero(inv_v == j) for j in range(len(uniq_v))]
    draws_p = np.empty(2000)
    for b in range(2000):
        idx = np.concatenate([members_v[j] for j in rng.integers(0, len(uniq_v), size=len(uniq_v))])
        draws_p[b] = np.corrcoef(panel[idx, 0], panel[idx, 1])[0, 1]
    values, counts = np.unique(np.array(list(sat.values())), return_counts=True)
    return {
        "variable": "mn_sat = median SAT of the participant's undergraduate institution (Fisman et al. 2006 data key); an ecological proxy, not an individual score",
        "n_dates": int(mask.sum()),
        "n_participants_with_mn_sat": int(len(sat)),
        "n_distinct_mn_sat_values": int(len(values)),
        "n_distinct_undergra_strings": int(len(set(inst.values()))),
        "largest_shared_value_block": int(counts.max()),
        "r_stranger_intel_vs_partner_institution_sat": r_intel,
        "r_partner_institution_sat_vs_yes": r_yes,
        "cluster_bootstrap": intervals,
        "panel_ge10_raters": {
            "n_targets": int(len(panel)),
            "institution_values": int(len(uniq_v)),
            "r_panel_intel_vs_institution_sat": r_panel,
            "institution_cluster_ci_95": quantile_ci(draws_p),
        },
        "relabel": "Text and Fig. generalist_verify caption should read 'institution-median SAT (ecological proxy)'; v15 point estimates are reproduced exactly, only intervals and wording change.",
    }


# ----------------------------------------------------------------------------- E25: dominance direction
def e25_closed_form(a, b, var_out, var_in, var_err, k):
    """Covariance of the chooser's best linear selection index with inward dominance.

    Display s = o + n (o outward, n inward); k witnessed contexts give zbar = n + e, Var(e)=var_err/k.
    Chooser welfare U = a*o - b*n.  Returns (Cov(I, n), Var(I), Cov(I, o)).
    """
    A = var_out + var_in
    B = var_in
    if k == 0:
        cov_in = (a * var_out - b * var_in) * var_in / A
        var_i = (a * var_out - b * var_in) ** 2 / A
        cov_out = (a * var_out - b * var_in) * var_out / A
        return cov_in, var_i, cov_out
    v = var_err / k
    D = var_in + v
    det = A * D - B * B
    c = np.array([a * var_out - b * B, -b * B])
    inv = np.array([[D, -B], [-B, A]]) / det
    beta = c @ inv
    S = np.array([[A, B], [B, D]])
    var_i = float(beta @ S @ beta)
    cov_in = float(beta @ np.array([B, B]))
    cov_out = float(beta @ np.array([var_out, 0.0]))
    return cov_in, var_i, cov_out


def e25_threshold(a, b, var_out, var_in, var_err):
    gap = a * var_out - b * var_in
    if gap <= 0:
        return 0.0
    return var_err * gap / (b * var_in * var_out)


def analyze_proposition_25_e25() -> dict:
    a, b, var_out, var_in, var_err = 1.0, 2.0, 1.0, 0.25, 4.0
    top = 0.10
    z = 1.2815515655446004  # upper 10% standard-normal cut
    intensity = float(normal_pdf(z) / top)
    k_star = e25_threshold(a, b, var_out, var_in, var_err)
    rng = np.random.default_rng(SEED + 1001)
    n = 400_000
    rows = []
    for k in (0, 1, 2, 4, 8, 16, 32):
        o = rng.normal(0.0, math.sqrt(var_out), n)
        nn = rng.normal(0.0, math.sqrt(var_in), n)
        s = o + nn
        if k == 0:
            index = (a * var_out - b * var_in) / (var_out + var_in) * s
        else:
            zbar = nn + rng.normal(0.0, math.sqrt(var_err / k), n)
            X = np.column_stack([s, zbar])
            U = a * o - b * nn
            coef = np.linalg.lstsq(X, U, rcond=None)[0]
            index = X @ coef
        cut = np.quantile(index, 1.0 - top)
        chosen = index >= cut
        sim_in = float(nn[chosen].mean() - nn.mean())
        sim_out = float(o[chosen].mean() - o.mean())
        cov_in, var_i, cov_out = e25_closed_form(a, b, var_out, var_in, var_err, k)
        rows.append({
            "k_contexts": k,
            "selection_differential_inward_dominance": {"closed_form": intensity * cov_in / math.sqrt(var_i), "monte_carlo": sim_in},
            "selection_differential_outward_formidability": {"closed_form": intensity * cov_out / math.sqrt(var_i), "monte_carlo": sim_out},
        })
    max_err = max(
        max(abs(r[key]["closed_form"] - r[key]["monte_carlo"]) for key in ("selection_differential_inward_dominance", "selection_differential_outward_formidability"))
        for r in rows
    )
    return {
        "proposition": 25,
        "parameters": {"a_outward_value": a, "b_inward_cost": b, "var_outward": var_out, "var_inward": var_in, "var_context_noise": var_err, "selected_top_share": top, "agents_per_k": n},
        "witness_threshold_k_star": k_star,
        "closed_form_threshold": "k* = sigma_e^2 (a sigma_o^2 - b sigma_n^2) / (b sigma_n^2 sigma_o^2), positive iff strangers reward the display bundle (a sigma_o^2 > b sigma_n^2)",
        "by_k": rows,
        "max_abs_error": float(max_err),
        "seed": SEED + 1001,
        "epistemic_tier": "derived + Monte Carlo check",
    }


# ----------------------------------------------------------------------------- E26: shared stereotype ceiling
def e26_closed_form(bias, tau, var_noise, m):
    """Group with bias penalty: rating mean = q - bias*g + ebar, Var(ebar)=var_noise/m, q~N(0,1)."""
    sd = math.sqrt(1.0 + var_noise / m)
    out = {}
    for g_lbl, g in (("reference_group", 0.0), ("penalized_group", 1.0)):
        zc = (tau + bias * g) / sd
        rate = float(1.0 - normal_cdf(zc))
        mean_q = float(inverse_mills(zc) / sd)
        out[g_lbl] = {"funded_rate": rate, "mean_true_quality_if_funded": mean_q}
    bias_share = bias * bias * 0.25 / (bias * bias * 0.25 + var_noise / m)
    return out, float(bias_share)


def analyze_proposition_26_e26() -> dict:
    bias, var_noise, tau = 0.30, 1.0, 1.2815515655446004
    rng = np.random.default_rng(SEED + 1011)
    n = 400_000
    rows = []
    for m in (1, 4, 16, 64, 256):
        closed, bias_share = e26_closed_form(bias, tau, var_noise, m)
        q = rng.normal(0.0, 1.0, n)
        g = (rng.random(n) < 0.5).astype(float)
        ebar = rng.normal(0.0, math.sqrt(var_noise / m), n)
        rating = q - bias * g + ebar
        funded = rating > tau
        sim = {}
        for g_lbl, gv in (("reference_group", 0.0), ("penalized_group", 1.0)):
            sel = g == gv
            sim[g_lbl] = {"funded_rate": float(funded[sel].mean()), "mean_true_quality_if_funded": float(q[sel & funded].mean())}
        gap_rating = float(rating[g == 1].mean() - rating[g == 0].mean() - (q[g == 1].mean() - q[g == 0].mean()))
        rows.append({
            "raters_m": m,
            "closed_form": closed,
            "monte_carlo": sim,
            "rating_gap_at_equal_quality": {"closed_form": -bias, "monte_carlo": gap_rating},
            "bias_share_of_rating_error_closed_form": bias_share,
            "outcome_test_quality_gap_funded": {
                "closed_form": closed["penalized_group"]["mean_true_quality_if_funded"] - closed["reference_group"]["mean_true_quality_if_funded"],
                "monte_carlo": sim["penalized_group"]["mean_true_quality_if_funded"] - sim["reference_group"]["mean_true_quality_if_funded"],
            },
        })
    symmetric, _ = e26_closed_form(0.0, tau, var_noise, 16)
    framing = {}
    for h in (0.0, 0.5, 1.0):
        res, _ = e26_closed_form(bias * (1.0 - h), tau, var_noise, 16)
        framing[f"h_{h:.1f}"] = {"effective_bias": bias * (1.0 - h), "funding_rate_ratio_penalized_to_reference": res["penalized_group"]["funded_rate"] / res["reference_group"]["funded_rate"]}
    errs = []
    for r in rows:
        for g_lbl in ("reference_group", "penalized_group"):
            for key in ("funded_rate", "mean_true_quality_if_funded"):
                errs.append(abs(r["closed_form"][g_lbl][key] - r["monte_carlo"][g_lbl][key]))
        errs.append(abs(r["rating_gap_at_equal_quality"]["closed_form"] - r["rating_gap_at_equal_quality"]["monte_carlo"]))
    return {
        "proposition": 26,
        "parameters": {"shared_bias_sd": bias, "rater_noise_variance": var_noise, "funding_threshold_sd": tau, "agents_per_m": n},
        "by_raters": rows,
        "symmetric_protocol_m16": symmetric,
        "communal_framing_m16": framing,
        "max_abs_error": float(max(errs)),
        "seed": SEED + 1011,
        "interpretation": "A bias common to all raters is a shared error: it survives any number of raters (gap = -b for all m) and becomes the dominant error as m grows; only removing b (symmetric protocols, outcome audits) closes the funding and outcome-test gaps. Communal framing h lowers the effective bias b(1-h) but is a tax paid by the penalized group, not an audience correction.",
        "epistemic_tier": "derived + Monte Carlo check",
    }


# ----------------------------------------------------------------------------- L1 ledger and S1 scorecard
try:  # imported as scripts.performance_confidence (tests) or from scripts/ on sys.path (run.py)
    from scripts.performance_confidence_ledger import LEDGER, SCORECARD
except ModuleNotFoundError:
    from performance_confidence_ledger import LEDGER, SCORECARD

DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")
LEDGER_FIELDS = frozenset(("id", "arena", "construct", "stage", "metric", "estimate", "se", "ci_95", "direction",
                           "k", "n", "source_key", "doi", "locator", "tier", "status", "note"))
LEDGER_STATUS = ("verified-fulltext", "verified-fulltext-preprint", "verified-abstract-only")
LOCAL_SOURCE_PREFIXES = ("gss:", "hcmst:", "pbo:")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_ledger(rows) -> None:
    seen = set()
    for row in rows:
        rid = row.get("id", "?")
        _require(set(row) == LEDGER_FIELDS, f"{rid}: fields {sorted(set(row) ^ LEDGER_FIELDS)}")
        _require(rid not in seen, f"duplicate ledger id {rid}")
        seen.add(rid)
        _require(bool(DOI_PATTERN.match(row["doi"])), f"{rid}: DOI {row['doi']!r}")
        _require(bool(row["locator"].strip()), f"{rid}: empty locator")
        _require(row["stage"] in ("entry", "outcome", "benchmark", "boundary"), f"{rid}: stage")
        _require(row["status"] in LEDGER_STATUS, f"{rid}: status {row['status']!r}")
        _require(row["direction"] in ("+", "-", "0"), f"{rid}: direction")
        est, se, ci = row["estimate"], row["se"], row["ci_95"]
        _require(est is None or (isinstance(est, (int, float)) and math.isfinite(est)), f"{rid}: estimate")
        _require(se is None or (est is not None and se > 0), f"{rid}: se")
        if ci is not None:
            _require(est is not None and ci[0] <= est <= ci[1], f"{rid}: CI must bracket the estimate")
        for key in ("k", "n"):
            _require(row[key] is None or (isinstance(row[key], int) and row[key] > 0), f"{rid}: {key}")


def check_scorecard(card, ledger=LEDGER) -> None:
    theories = card["theories"]
    ids = {row["id"] for row in ledger}
    for row in card["rows"]:
        _require(set(row["codes"]) == set(theories), f"{row['id']}: theory columns")
        _require(all(v in ("P", "C", "S") for v in row["codes"].values()), f"{row['id']}: codes")
        _require(bool(row["sources"]) and bool(row["justification"].strip()), f"{row['id']}: sources/justification")
        for src in row["sources"]:
            _require(src in ids or src.startswith(LOCAL_SOURCE_PREFIXES), f"{row['id']}: unknown source {src}")


def check_scorecard_numbers(card, gss: dict, hcmst: dict) -> None:
    """The scorecard quotes a few measured numbers in prose; fail loudly if a rerun moves them."""
    text = {row["id"]: row["regularity"] for row in card["rows"]}
    early = gss["audience_composition"]["early_1975_1985"]
    decomposition = gss["firebaugh_decomposition"]["pooled"]
    expected = {
        "stereotype_shared_by_both_sexes": (
            f"{100 * early['men']['agree_share']:.1f} vs {100 * early['women']['agree_share']:.1f}%",
            f"gap {gss['recent_audience_sex_gap']['estimate_pp']:.1f} points",
        ),
        "leader_stereotype_declines_by_cohort": (
            f"about {-decomposition['components_pp']['observed_change']['estimate']:.0f} points",
            f"roughly {100 * decomposition['replacement_share_of_fitted']:.0f}% of the fitted fall",
        ),
        "wife_earns_more_lower_marital_quality": (
            f"{-hcmst['confirmatory']['estimate_pp']:.1f} points lower",
            f"raw p={hcmst['confirmatory']['p_two_sided']:.3f}".replace("=0.", "=."),
        ),
    }
    for row_id, fragments in expected.items():
        for fragment in fragments:
            _require(fragment in text[row_id], f"scorecard prose for {row_id} lacks measured value {fragment!r}")


def score(card) -> dict:
    out = {}
    for t in card["theories"]:
        codes = [row["codes"][t] for row in card["rows"]]
        out[t] = {"predicted": codes.count("P"), "contradicted": codes.count("C"), "silent": codes.count("S")}
    return out


# ----------------------------------------------------------------------------- W1: warm confidence, own vs other raters
WARM_ATTRS = ("attr", "fun", "intel", "amb", "sinc")


def analyze_warm_confidence(csv_path: Path) -> dict:
    """Is the v16 'warm outlier confidence' complementarity a same-rater halo?

    Within-chooser linear probability model of the yes decision on standardized agency (mean of the
    intel, fun and amb ratings, the v16 definition), warmth (sinc), their product, and attractiveness.
    Ratings come either from the deciding chooser (``own``, halo-prone) or from the other choosers who
    met the same partner in the same session (``panel``, leave-one-out, at least three peers).
    Intervals resample the 21 sessions. Exploratory: outside the confirmatory family.
    """
    data = csv_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CONFIDENCE_INPUTS["speed_dating.csv"]["sha256"]:
        raise ValueError("speed-dating checksum mismatch")
    rows = list(csv.DictReader(data.decode("latin-1").splitlines()))

    def num(v):
        v = str(v).replace(",", "").strip()
        return float(v) if v else np.nan

    wave = np.array([num(r["wave"]) for r in rows])
    sex = np.array([num(r["gender"]) for r in rows])  # 1 = male chooser
    chooser = np.array([num(r["iid"]) for r in rows])
    partner = np.array([num(r["pid"]) for r in rows])
    yes = np.array([num(r["dec"]) for r in rows])
    own = np.array([[num(r[a]) for a in WARM_ATTRS] for r in rows])
    if len(np.unique(wave)) != 21 or len(rows) != 8378:
        raise ValueError("unexpected speed-dating layout")
    return warm_confidence_from_arrays(wave, sex, chooser, partner, yes, own)


def leave_one_out_panel(wave, partner, own, min_peers=3):
    """Mean rating of each partner by the *other* choosers in the same session (NaN below ``min_peers``)."""
    panel = np.full_like(own, np.nan)
    for event in np.unique(wave):
        for person in np.unique(partner[wave == event]):
            idx = np.flatnonzero((wave == event) & (partner == person))
            vals = own[idx]
            tot = np.nansum(vals, axis=0) - np.nan_to_num(vals)
            peers = np.isfinite(vals).sum(axis=0) - np.isfinite(vals)
            panel[idx] = np.divide(tot, peers, out=np.full_like(tot, np.nan), where=peers >= min_peers)
    return panel


def warm_confidence_from_arrays(wave, sex, chooser, partner, yes, own, draws_n=4000, seed=SEED + 1031):
    """Estimator behind ``analyze_warm_confidence``; columns of ``own`` follow ``WARM_ATTRS``."""
    own = np.array(own, dtype=np.float64)
    own[(own < 1) | (own > 10)] = np.nan
    panel = leave_one_out_panel(wave, partner, own)
    sessions, s_idx = np.unique(wave, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(sessions), size=(draws_n, len(sessions)))
    terms = ("agency", "warmth", "agency_x_warmth", "attractiveness")
    out = {"definition": "agency = mean(intel, fun, amb); warmth = sinc; z-scored over complete rows; chooser fixed effects",
           "terms": list(terms)}
    boots = {}
    for source, ratings in (("own", own), ("panel", panel)):
        ok = np.isfinite(ratings).all(axis=1) & ((yes == 0) | (yes == 1))
        ag = ratings[:, [1, 2, 3]].mean(axis=1)
        z = lambda v: (v - v[ok].mean()) / v[ok].std()
        za, zw, zt = z(ag), z(ratings[:, 4]), z(ratings[:, 0])
        design = np.column_stack([za, zw, za * zw, zt])
        res = {}
        for code, key in ((1.0, "male_chooser_female_target"), (0.0, "female_chooser_male_target")):
            sel = ok & (sex == code)
            xx = np.zeros((len(sessions), 4, 4))
            xy = np.zeros((len(sessions), 4))
            n_rows = n_people = 0
            for person in np.unique(chooser[sel]):
                r_ = np.flatnonzero(sel & (chooser == person))
                if len(r_) < 2:
                    continue
                x = design[r_] - design[r_].mean(axis=0)
                y = yes[r_] - yes[r_].mean()
                e = s_idx[r_[0]]
                xx[e] += x.T @ x
                xy[e] += x.T @ y
                n_rows += len(r_)
                n_people += 1
            coef = np.linalg.solve(xx.sum(axis=0), xy.sum(axis=0))
            bt = np.linalg.solve(xx[draws].sum(axis=1), xy[draws].sum(axis=1)[..., None])[..., 0]
            boots[(source, key)] = bt
            res[key] = {"n_decisions": n_rows, "n_choosers": n_people,
                        "coefficients_pp_per_sd": {t: {"estimate": float(100 * c), "interval_95": [float(v) for v in 100 * np.quantile(b, [.025, .975])]}
                                                   for t, c, b in zip(terms, coef, bt.T)}}
        diff = 100 * (boots[(source, "male_chooser_female_target")] - boots[(source, "female_chooser_male_target")])
        res["female_minus_male_target_pp_per_sd"] = {
            t: {"estimate": float(res["male_chooser_female_target"]["coefficients_pp_per_sd"][t]["estimate"]
                                  - res["female_chooser_male_target"]["coefficients_pp_per_sd"][t]["estimate"]),
                "interval_95": [float(v) for v in np.quantile(diff[:, i], [.025, .975])]} for i, t in enumerate(terms)}
        out[source] = res
    return out


# ----------------------------------------------------------------------------- D1: decision confidence before outcomes
FORECAST_LEVELS = tuple(range(1, 11))
CAPPED_WAVE = 12  # Fisman et al. data key: wave 12 limited how many partners a participant could accept


def auc_rank(score, label) -> float:
    """Mann--Whitney AUC with average ranks for ties (0.5 = chance); NaN when one class is empty."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    n1 = int(label.sum())
    n0 = int(label.size - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    s_sorted = score[order]
    i = 0
    while i < score.size:
        j = i
        while j + 1 < score.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [mid - half, mid + half]


def partner_acceptance_record(iid, pid, dec, keep, min_others=2):
    """For each row (chooser i, partner j): the share of the *other* people j met whom j accepted.

    j's decision about i is left out, so the record never contains the outcome it is used to predict.
    NaN when j has fewer than ``min_others`` other decisions among the ``keep`` rows.
    """
    acc_sum, acc_n, said = {}, {}, {}
    for person, other, d in zip(iid[keep], pid[keep], dec[keep]):
        acc_sum[person] = acc_sum.get(person, 0.0) + d
        acc_n[person] = acc_n.get(person, 0) + 1
        said[(person, other)] = d
    record = np.full(np.shape(pid), np.nan)
    for r_, (person, other) in enumerate(zip(iid, pid)):
        if other not in acc_n:
            continue
        own_row = said.get((other, person))
        tot = acc_sum[other] - (own_row if own_row is not None else 0.0)
        cnt = acc_n[other] - (1 if own_row is not None else 0)
        if cnt >= min_others:
            record[r_] = tot / cnt
    return record


def analyze_decision_confidence(csv_path: Path) -> dict:
    data = csv_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CONFIDENCE_INPUTS["speed_dating.csv"]["sha256"]:
        raise ValueError("speed-dating checksum mismatch")
    rows = list(csv.DictReader(data.decode("latin-1").splitlines()))

    def num(v):
        v = str(v).replace(",", "").strip()
        return float(v) if v else np.nan

    cols = {k: np.array([num(r.get(k, "")) for r in rows]) for k in
            ("wave", "gender", "iid", "pid", "dec", "dec_o", "match", "prob", "expnum", "match_es")}
    if len(np.unique(cols["wave"])) != 21 or len(rows) != 8378:
        raise ValueError("unexpected speed-dating layout")
    return decision_confidence_from_arrays(**cols)


def decision_confidence_from_arrays(wave, gender, iid, pid, dec, dec_o, match, prob, expnum, match_es,
                                    draws_n=4000, seed=SEED + 1041) -> dict:
    """Does a chooser's pre-outcome forecast predict the partner's decision, and how does it compare with
    the partner's record with other people?  ``gender`` 1 = male chooser; ``prob`` is the 1--10 answer to
    "how probable is it that this person will say yes to you?", given before any decision is revealed.
    """
    wave, gender, iid, pid = (np.asarray(a, dtype=np.float64) for a in (wave, gender, iid, pid))
    dec, dec_o, match = (np.asarray(a, dtype=np.float64) for a in (dec, dec_o, match))
    prob, expnum, match_es = (np.array(a, dtype=np.float64) for a in (prob, expnum, match_es))
    prob[(prob < 1) | (prob > 10)] = np.nan
    keep = (wave != CAPPED_WAVE) & np.isin(dec, (0.0, 1.0)) & np.isin(dec_o, (0.0, 1.0)) & np.isfinite(pid)
    partner_record = partner_acceptance_record(iid, pid, dec, keep)
    ok = keep & np.isfinite(prob) & np.isfinite(partner_record)
    if not ok.any():
        raise ValueError("no decisions with both a forecast and a partner record")
    level = np.floor(prob + 0.5)
    sessions = np.unique(wave[ok])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(sessions), size=(draws_n, len(sessions)))
    out = {
        "definition": ("Within-chooser AUC (Mann-Whitney, ties = 1/2) averaged over choosers with both outcomes; "
                       "partner record = partner's acceptance rate of the other people met that night (leave-one-out, >=2 others); "
                       f"wave {CAPPED_WAVE} excluded (acceptances capped); forecast answers outside 1-10 set missing"),
        "n_sessions": int(len(sessions)),
        "bootstrap": f"{draws_n} session-cluster resamples",
    }
    by_sex = {}
    for code, key in ((0.0, "women"), (1.0, "men")):
        sel = ok & (gender == code)
        people = np.unique(iid[sel])
        stats = {m: [] for m in ("forecast_partner", "record_partner", "forecast_own")}
        sess_of = []
        for person in people:
            r_ = np.flatnonzero(sel & (iid == person))
            a1 = auc_rank(prob[r_], dec_o[r_] == 1)
            a2 = auc_rank(partner_record[r_], dec_o[r_] == 1)
            a3 = auc_rank(prob[r_], dec[r_] == 1)
            stats["forecast_partner"].append(a1)
            stats["record_partner"].append(a2)
            stats["forecast_own"].append(a3)
            sess_of.append(np.searchsorted(sessions, wave[r_[0]]))
        sess_of = np.array(sess_of)
        arr = {m: np.array(v) for m, v in stats.items()}
        both = np.isfinite(arr["forecast_partner"]) & np.isfinite(arr["record_partner"])
        res = {"n_decisions": int(sel.sum()), "n_choosers": int(len(people))}
        # Per-session sums let each bootstrap draw re-weight sessions without recomputing AUCs.
        def boot_mean(vals, mask):
            s = np.bincount(sess_of[mask], weights=vals[mask], minlength=len(sessions))
            c = np.bincount(sess_of[mask], minlength=len(sessions)).astype(np.float64)
            return float(s.sum() / c.sum()), s[draws].sum(axis=1) / c[draws].sum(axis=1)
        for m, lbl in (("forecast_partner", "forecast_predicts_partner_yes"),
                       ("record_partner", "partner_record_predicts_partner_yes"),
                       ("forecast_own", "forecast_predicts_own_yes")):
            mask = np.isfinite(arr[m]) if m == "forecast_own" else both
            est, bt = boot_mean(arr[m], mask)
            res[lbl] = {"auc": est, "ci_95": quantile_ci(bt), "n_choosers": int(mask.sum())}
        est_d, bt_d = boot_mean(np.where(both, arr["record_partner"] - arr["forecast_partner"], 0.0), both)
        res["record_minus_forecast_auc"] = {"estimate": est_d, "ci_95": quantile_ci(bt_d)}
        # Projection: within-chooser regression of the forecast on own and partner decisions.
        xx = np.zeros((len(sessions), 2, 2))
        xy = np.zeros((len(sessions), 2))
        for person in people:
            r_ = np.flatnonzero(sel & (iid == person))
            if len(r_) < 2:
                continue
            X = np.column_stack([dec[r_], dec_o[r_]])
            X = X - X.mean(axis=0)
            y = prob[r_] - prob[r_].mean()
            e = np.searchsorted(sessions, wave[r_[0]])
            xx[e] += X.T @ X
            xy[e] += X.T @ y
        coef = np.linalg.solve(xx.sum(axis=0), xy.sum(axis=0))
        bt = np.linalg.solve(xx[draws].sum(axis=1), xy[draws].sum(axis=1)[..., None])[..., 0]
        res["forecast_points_per_decision"] = {
            "own_yes": {"estimate": float(coef[0]), "ci_95": quantile_ci(bt[:, 0])},
            "partner_yes": {"estimate": float(coef[1]), "ci_95": quantile_ci(bt[:, 1])},
            "own_over_partner_ratio": {"estimate": float(coef[0] / coef[1]), "ci_95": quantile_ci(bt[:, 0] / bt[:, 1])},
        }
        res["mean_forecast_tenths_exploratory"] = float(np.mean(prob[sel]) / 10.0)
        res["partner_yes_rate"] = float(np.mean(dec_o[sel]))
        by_sex[key] = res
    out["by_chooser_sex"] = by_sex
    curve = []
    for lv in FORECAST_LEVELS:
        s = ok & (level == lv)
        n = int(s.sum())
        k_p = int(dec_o[s].sum())
        k_o = int(dec[s].sum())
        curve.append({"forecast": lv, "n": n,
                      "partner_yes": k_p / n if n else float("nan"), "partner_yes_wilson_95": wilson(k_p, n),
                      "own_yes": k_o / n if n else float("nan"), "own_yes_wilson_95": wilson(k_o, n)})
    out["by_forecast_level_pooled"] = curve
    out["n_decisions"] = int(ok.sum())
    out["n_choosers"] = int(len(np.unique(iid[ok])))
    # Calibration in the large, before meeting and at the end of the night (same people).
    people = np.unique(iid[keep])
    per = []
    for person in people:
        r_ = np.flatnonzero(keep & (iid == person))
        e0 = expnum[r_][np.isfinite(expnum[r_])]
        e1 = match_es[r_][np.isfinite(match_es[r_])]
        per.append((gender[r_[0]], len(r_), dec_o[r_].mean(), np.nansum(match[r_]),
                    e0[0] if e0.size else np.nan, e1[0] if e1.size else np.nan))
    per = np.array(per, dtype=np.float64)
    rng_p = np.random.default_rng(seed + 1)
    cal = {}
    for code, key in ((0.0, "women"), (1.0, "men")):
        g = per[(per[:, 0] == code) & np.isfinite(per[:, 4]) & np.isfinite(per[:, 5]) & (per[:, 4] <= 20)]
        n_met = g[:, 1]
        before = 100.0 * (g[:, 4] / 20.0 - g[:, 2])
        after = 100.0 * (g[:, 5] - g[:, 3]) / n_met
        idx = rng_p.integers(len(g), size=(draws_n, len(g)))
        cal[key] = {
            "n_people": int(len(g)),
            "before_meeting_expected_minus_actual_share_accepting_pp": {"estimate": float(before.mean()), "ci_95": quantile_ci(before[idx].mean(axis=1))},
            "end_of_night_expected_minus_actual_matches_pp_of_met": {"estimate": float(after.mean()), "ci_95": quantile_ci(after[idx].mean(axis=1))},
            "expected_share_accepting_before": float((g[:, 4] / 20.0).mean()),
            "actual_share_accepting": float(g[:, 2].mean()),
            "expected_matches_end_of_night": float(g[:, 5].mean()),
            "actual_matches": float(g[:, 3].mean()),
        }
        a = per[(per[:, 0] == code) & np.isfinite(per[:, 5])]
        cal[key]["all_waves_end_of_night"] = {"n_people": int(len(a)), "expected_matches": float(a[:, 5].mean()), "actual_matches": float(a[:, 3].mean())}
    out["calibration_before_and_after_meeting"] = cal
    out["epistemic_tier"] = "measured (exploratory; outside the confirmatory family)"
    return out


# ----------------------------------------------------------------------------- E27: agreement breeds confidence
def e27_closed_form(var_t, var_h, var_c, var_o, var_e, m, k):
    """Agreement-based confidence L and validity R^2 for the two-way jury mean (Eq. twoway)."""
    den = var_t + var_h + var_c / k + var_o / m + var_e / (m * k)
    agreement = (var_t + var_h + var_c / k) / den
    validity = var_t / den
    return agreement, validity


def analyze_proposition_27_e27() -> dict:
    var_t, var_h, var_c, var_o, var_e = 1.0, 0.5, 1.0, 1.5, 2.0
    rng = np.random.default_rng(SEED + 1051)
    n = 100_000
    rows = []
    for k in (1, 8):
        for m in (1, 2, 4, 8, 16, 32, 64):
            t = rng.normal(0, math.sqrt(var_t), n)
            h = rng.normal(0, math.sqrt(var_h), n)
            c = rng.normal(0, math.sqrt(var_c / k), n)  # mean of k context states
            jury = [t + h + c + rng.normal(0, math.sqrt(var_o / m + var_e / (m * k)), n) for _ in range(2)]
            sim_l = float(np.corrcoef(jury[0], jury[1])[0, 1])
            sim_r = float(np.corrcoef(jury[0], t)[0, 1] ** 2)
            cf_l, cf_r = e27_closed_form(var_t, var_h, var_c, var_o, var_e, m, k)
            rows.append({"judges_m": m, "contexts_k": k,
                         "agreement_confidence": {"closed_form": cf_l, "monte_carlo": sim_l},
                         "validity_r2": {"closed_form": cf_r, "monte_carlo": sim_r},
                         "gap": cf_l - cf_r})
    lone = []
    for k in (1, 8):
        t = rng.normal(0, math.sqrt(var_t), n)
        shared = t + rng.normal(0, math.sqrt(var_h), n) + rng.normal(0, math.sqrt(var_c / k), n) + rng.normal(0, math.sqrt(var_o), n)
        imp = [shared + rng.normal(0, math.sqrt(var_e / k), n) for _ in range(2)]
        den = var_t + var_h + var_c / k + var_o + var_e / k
        lone.append({"contexts_k": k,
                     "self_consistency": {"closed_form": (den - var_e / k) / den, "monte_carlo": float(np.corrcoef(imp[0], imp[1])[0, 1])},
                     "excess_over_jury_agreement": var_o / den})
    err = max(max(abs(r[q]["closed_form"] - r[q]["monte_carlo"]) for q in ("agreement_confidence", "validity_r2")) for r in rows)
    err = max(err, max(abs(r["self_consistency"]["closed_form"] - r["self_consistency"]["monte_carlo"]) for r in lone))
    lim1 = (var_h + var_c) / (var_t + var_h + var_c)
    lim_inf = var_h / (var_t + var_h)
    return {
        "proposition": 27,
        "parameters": {"var_trait": var_t, "var_halo": var_h, "var_context": var_c, "var_observer": var_o, "var_residual": var_e, "agents_per_cell": n},
        "grid": rows,
        "lone_judge_self_consistency": lone,
        "gap_limit_many_judges_one_context": lim1,
        "gap_limit_many_judges_many_contexts": lim_inf,
        "max_abs_error": float(err),
        "seed": SEED + 1051,
        "epistemic_tier": "derived + Monte Carlo check",
    }


# ----------------------------------------------------------------------------- entry point
def analyze_confidence(inputs_dir: Path, root: Path = ROOT) -> dict:
    inputs_dir = Path(inputs_dir)
    gss = analyze_gss_audience(resolve("gss_all.rda", inputs_dir), root)
    hcmst = analyze_hcmst_relative_income(resolve("hcmst2017_public_v1.1.dta", inputs_dir))
    sat = analyze_sat_proxy(resolve("speed_dating.csv", inputs_dir))
    warm = analyze_warm_confidence(resolve("speed_dating.csv", inputs_dir))
    decision = analyze_decision_confidence(resolve("speed_dating.csv", inputs_dir))
    e25 = analyze_proposition_25_e25()
    e26 = analyze_proposition_26_e26()
    e27 = analyze_proposition_27_e27()
    check_ledger(LEDGER)
    check_scorecard(SCORECARD)
    result = {
        "schema": SCHEMA,
        "module_sha256": sha256_file(Path(__file__)),
        "ledger_module_sha256": sha256_file(Path(__file__).with_name("performance_confidence_ledger.py")),
        "inputs": {k: {"sha256": v["sha256"], "url": v["url"], "citation": v["citation"], "license": v["license"]} for k, v in CONFIDENCE_INPUTS.items()},
        "gss_audience_women_leaders": gss,
        "hcmst_relative_income": hcmst,
        "speed_dating_sat_proxy": sat,
        "speed_dating_warm_confidence": warm,
        "proposition_25_e25": e25,
        "proposition_26_e26": e26,
        "speed_dating_decision_confidence": decision,
        "proposition_27_e27": e27,
        "ledger": list(LEDGER),
    }
    check_scorecard_numbers(SCORECARD, gss, hcmst)
    result["scorecard"] = {**SCORECARD, "totals": score(SCORECARD)}
    result["ledger_summary"] = {
        "rows": len(LEDGER),
        "by_status": {st: sum(r["status"] == st for r in LEDGER) for st in LEDGER_STATUS},
        "by_stage": {sg: sum(r["stage"] == sg for r in LEDGER) for sg in ("entry", "outcome", "benchmark", "boundary")},
    }
    pvals = {
        CONFIRMATORY[0]: hcmst["confirmatory"]["p_two_sided"],
        CONFIRMATORY[1]: gss["recent_audience_sex_gap"]["p_two_sided"],
        CONFIRMATORY[2]: gss["firebaugh_decomposition"]["pooled"]["components_pp"]["cohort_replacement"]["p_two_sided"],
    }
    result["multiplicity"] = {
        "confirmatory_family": list(CONFIRMATORY),
        "raw_p": pvals,
        "holm_adjusted_p": holm(pvals),
        "rule": "Holm over the pre-specified confirmatory family; all other estimates are exploratory and reported with intervals only",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, default=ROOT / "target/performance-before-outcomes/inputs")
    parser.add_argument("--out", type=Path, default=ROOT / "target/confidence/confidence.json")
    args = parser.parse_args()
    result = analyze_confidence(args.inputs, ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.new")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
