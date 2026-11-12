#!/usr/bin/env python3
"""Synchronized Plio-Pleistocene hominin endocranial stasis-step audit, NOAA/Lisiecki LR04
57-core benthic d18O paleoclimate coupling, Du et al. (2018) radiometric age-uncertainty Monte Carlo,
and chronological/epistemic evaluation of Mid-Pleistocene global crises across the 1.5-0.7 Ma plateau.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
LR04_PATH = ROOT / "evidence/paleoclimate/lr04_benthic_stack.txt"
V9_ROOT = ROOT / "project/brain/tips/Performance_Before_Outcomes_v9"
V9_DATA = V9_ROOT / "data"
V9_FOSSILS_JSON = V9_ROOT / "results/e13_fossils.json"

EXPECTED_LR04_SHA256 = "ec3c4af8f80c4de1d36fb0ac9fabbde28520f77f0832958caacbae4957afb412"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_lr04(max_age_ma: float = 3.40) -> dict[str, Any]:
    raw_bytes = LR04_PATH.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    ages_ka: list[float] = []
    d18o: list[float] = []
    se: list[float] = []
    for line in raw_bytes.decode("utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("age_"):
            continue
        parts = s.split()
        if len(parts) >= 3:
            ak = float(parts[0])
            if ak <= max_age_ma * 1000.0:
                ages_ka.append(ak)
                d18o.append(float(parts[1]))
                se.append(float(parts[2]))
    age_ma = np.asarray(ages_ka, dtype=np.float64) / 1000.0
    val = np.asarray(d18o, dtype=np.float64)
    err = np.asarray(se, dtype=np.float64)
    return {
        "sha256": sha,
        "bytes": len(raw_bytes),
        "age_ma": age_ma,
        "d18o": val,
        "d18o_se": err,
    }


def _load_xlsx_rows(path: Path, sheet: int = 0) -> list[list[Any]]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.worksheets[sheet]
        return [list(r) for r in ws.iter_rows(values_only=True)]
    finally:
        wb.close()


def load_du2018_with_bounds() -> list[dict[str, Any]]:
    rows = _load_xlsx_rows(V9_DATA / "du2018/du2018_ecv.xlsx", 0)
    hdr = [str(x).strip() if x is not None else "" for x in rows[0]]
    out: list[dict[str, Any]] = []
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        d = dict(zip(hdr, r))
        ecvs = [_num(d.get(f"ecv{k}")) for k in range(1, 7)]
        ecvs = [x for x in ecvs if math.isfinite(x)] if False else [x for x in ecvs if x == x and x > 0]
        if not ecvs:
            continue
        lo, hi, mu = _num(d.get("min.date")), _num(d.get("max.date")), _num(d.get("mean.date"))
        age_mid = mu if mu == mu else 0.5 * (lo + hi)
        if not (age_mid == age_mid):
            continue
        age_lo = lo if lo == lo else age_mid
        age_hi = hi if hi == hi else age_mid
        if age_lo > age_hi:
            age_lo, age_hi = age_hi, age_lo
        out.append({
            "id": str(d["ID"]).strip(),
            "taxon": str(d.get("lump.taxon", "")).strip(),
            "grade": str(d.get("grade", "")).strip(),
            "age_mid": float(age_mid),
            "age_lo": float(age_lo),
            "age_hi": float(age_hi),
            "ecvs": [float(v) for v in ecvs],
            "ecv_mean": float(np.mean(ecvs)),
        })
    return out


def _ols_slope(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(t) < 3 or float(np.ptp(t)) < 1e-9:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones_like(t), t])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - 2, 1)
    s2 = float((resid @ resid) / dof)
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(beta[1]), float(np.sqrt(max(cov[1, 1], 0.0)))


def analyze_paleoclimate_plateau(seed: int = 20261119) -> dict[str, Any]:
    lr = load_lr04(max_age_ma=3.40)
    age_ma = lr["age_ma"]
    d18o = lr["d18o"]

    v9_fos = json.loads(V9_FOSSILS_JSON.read_text())
    bp_pub = [2.00, 1.50, 0.70]
    bp_fit = [float(x) for x in v9_fos["models"]["stasis_step"]["bps"]]  # [2.05, 1.45, 0.65]
    plateau_onset_pub = 1.50
    plateau_onset_fit = bp_fit[1]  # 1.45 Ma

    # 1. LR04 Paleoclimate Regime Statistics across Hominin Phases & MPT
    regime_Windows = [
        ("phase_I_australopith_stasis", "Phase I Stasis (3.20–2.05 Ma)", 2.05, 3.20),
        ("phase_II_early_homo_surge", "Phase II Early Homo Surge (2.05–1.50 Ma)", 1.50, 2.05),
        ("plateau_early_pre_mpt", "Early Phase III Plateau, Pre-MPT (1.50–1.25 Ma)", 1.25, 1.50),
        ("plateau_late_mpt_crisis", "Late Phase III Plateau, MPT Crisis (1.25–0.70 Ma)", 0.70, 1.25),
        ("phase_IV_post_mpt_surge", "Phase IV Parallel Surge, 100-kyr Cycles (0.70–0.01 Ma)", 0.01, 0.70),
    ]
    regimes: dict[str, dict[str, Any]] = {}
    for key, label, lo, hi in regime_Windows:
        m = (age_ma >= lo) & (age_ma <= hi)
        sub = d18o[m]
        p10, p90 = float(np.percentile(sub, 10)), float(np.percentile(sub, 90))
        regimes[key] = {
            "label": label,
            "age_lo_ma": lo,
            "age_hi_ma": hi,
            "n_points": int(m.sum()),
            "mean_d18o_permil": round(float(sub.mean()), 4),
            "sd_d18o_permil": round(float(sub.std(ddof=1)), 4),
            "p90_glacial_permil": round(p90, 4),
            "p10_interglacial_permil": round(p10, 4),
            "glacial_interglacial_swing_p90_p10": round(p90 - p10, 4),
            "max_glacial_permil": round(float(sub.max()), 4),
            "max_glacial_age_ma": round(float(age_ma[m][int(np.argmax(sub))]), 4),
        }

    # Sliding 200-kyr envelope (every 25 kyr from 0.05 to 3.30 Ma) for synchronized plotting
    env_centers = np.round(np.arange(0.05, 3.31, 0.025), 4)
    env_mean: list[float] = []
    env_p10: list[float] = []
    env_p90: list[float] = []
    for c in env_centers:
        m = (age_ma >= c - 0.10) & (age_ma <= c + 0.10)
        sub = d18o[m]
        env_mean.append(round(float(sub.mean()), 4))
        env_p10.append(round(float(np.percentile(sub, 10)), 4))
        env_p90.append(round(float(np.percentile(sub, 90)), 4))

    # Compact 5-kyr decimated LR04 curve (0.00 to 3.35 Ma, ~671 points) for fast crisp vector plotting
    plot_grid = np.round(np.arange(0.0, 3.351, 0.005), 4)
    d18o_grid = np.interp(plot_grid, age_ma, d18o)

    # Specific MIS excursion checks around ~1.12 Ma (Margari 2023 MIS 34) and ~0.95-0.86 Ma (MIS 24-22)
    m_mis34 = (age_ma >= 1.09) & (age_ma <= 1.16)
    mis34_peak = float(d18o[m_mis34].max())
    mis34_age = float(age_ma[m_mis34][int(np.argmax(d18o[m_mis34]))])

    m_mis22 = (age_ma >= 0.85) & (age_ma <= 0.96)
    mis22_peak = float(d18o[m_mis22].max())
    mis22_age = float(age_ma[m_mis22][int(np.argmax(d18o[m_mis22]))])

    amp_pre_mpt = regimes["plateau_early_pre_mpt"]["glacial_interglacial_swing_p90_p10"]
    amp_mpt = regimes["plateau_late_mpt_crisis"]["glacial_interglacial_swing_p90_p10"]
    amp_post_mpt = regimes["phase_IV_post_mpt_surge"]["glacial_interglacial_swing_p90_p10"]

    # 2. Cross-dataset fossil robustness: Du et al. (2018) radiometric-uncertainty Monte Carlo
    du_rows = load_du2018_with_bounds()
    rng = np.random.default_rng(seed)
    n_mc = 2000
    mc_slopes = {"phase_II_2.0_1.5": [], "phase_III_1.5_0.7": [], "early_homo_all_2.0_0.7": []}
    for _ in range(n_mc):
        sim_ages = np.array([rng.uniform(r["age_lo"], r["age_hi"]) for r in du_rows], dtype=np.float64)
        sim_ecvs = np.array([r["ecvs"][int(rng.integers(0, len(r["ecvs"])))] for r in du_rows], dtype=np.float64)
        sim_ln = np.log(sim_ecvs)
        is_homo = np.array([r["taxon"].startswith("H.") for r in du_rows], dtype=bool)
        for key, lo, hi, homo_only in (
            ("phase_II_2.0_1.5", 1.50, 2.05, False),
            ("phase_III_1.5_0.7", 0.70, 1.50, True),
            ("early_homo_all_2.0_0.7", 0.70, 2.05, True),
        ):
            msk = (sim_ages >= lo) & (sim_ages <= hi)
            if homo_only:
                msk &= is_homo
            sl, _ = _ols_slope(-sim_ages[msk], sim_ln[msk])
            if np.isfinite(sl):
                mc_slopes[key].append(sl)

    du_mc_summary = {}
    for key, vals in mc_slopes.items():
        arr = np.asarray(vals, dtype=np.float64)
        du_mc_summary[key] = {
            "n_replicates": int(len(arr)),
            "median_slope_ln_per_myr": round(float(np.median(arr)), 4),
            "ci95_ln_per_myr": [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)],
            "prob_positive": round(float(np.mean(arr > 0.0)), 4),
        }

    # 3. Structured Global-Event & Hypothesis Ledger (1.5-0.7 Ma Plateau & Phase IV Transition)
    raw_events = [
        {
            "id": "mpt_intensification",
            "short_label": "MPT ice-cycle shift (41→100 kyr)",
            "window_ma": [0.70, 1.25],
            "point_ma": 1.25,
            "tier": "measured_climate",
            "tier_label": "Measured global climate proxy",
            "observable": "57-core benthic d18O stack (LR04), Mg/Ca deep-water T, boron CO2, Atlantic Nd",
            "evidence_summary": f"Glacial P90-P10 swing widens from {amp_pre_mpt:.2f} to {amp_mpt:.2f} permil across MPT ({amp_post_mpt:.2f} post-MPT); CO2 falls and deep-Atlantic circulation reorganizes.",
            "hominin_scope": "Fragments habitats & amplifies ecological variability in late plateau; begins ~250 kyr AFTER ~1.5 Ma stasis onset.",
            "cite_keys": ["lisiecki2005", "elderfield2012", "chalk2017", "pena2014"],
        },
        {
            "id": "mis34_europe_cooling",
            "short_label": "MIS 34 extreme cooling (Eur. depop.)",
            "window_ma": [1.10, 1.15],
            "point_ma": 1.12,
            "tier": "measured_climate",
            "tier_label": "Measured regional climate + habitat model",
            "observable": f"Iberian margin alkenone/pollen cores + envelope modeling (MIS 34, LR04 d18O = {mis34_peak:.2f} permil)",
            "evidence_summary": "Severe North Atlantic/Mediterranean cooling renders western Europe climatically unsuitable after ~1.4 Ma occupation.",
            "hominin_scope": "Supports regional European depopulation/turnover, NOT global extinction; occurs ~380 kyr after plateau onset.",
            "cite_keys": ["margari2023", "carbonell2008", "garba2024", "huguet2025"],
        },
        {
            "id": "bosumtwi_impact",
            "short_label": "Bosumtwi crater (10.5 km, Ghana)",
            "window_ma": [1.06, 1.08],
            "point_ma": 1.07,
            "tier": "unlinked_physical",
            "tier_label": "Dated impact — no hominin mortality link",
            "observable": "ICDP drill cores, shocked quartz, Ivory Coast microtektites",
            "evidence_summary": "Confirmed 10.5 km West African impact structure dated near 1.07 Ma.",
            "hominin_scope": "Regional blast hazard only; zero paleontological or demographic evidence of continental hominin die-off.",
            "cite_keys": ["koeberl2007", "kenkmann2021"],
        },
        {
            "id": "wonderwerk_fire",
            "short_label": "Wonderwerk Cave fire (~1.0 Ma)",
            "window_ma": [0.95, 1.05],
            "point_ma": 1.00,
            "tier": "energetic_release",
            "tier_label": "Dated archaeological energetic marker",
            "observable": "Microstratigraphic wood ash & calcined bone inside Acheulean stratum 10",
            "evidence_summary": "In-situ combustion features ~1.0 Ma inside deep cave context.",
            "hominin_scope": "Early thermal processing candidate; precedes Phase IV expansion as fire use generalizes.",
            "cite_keys": ["berna2012"],
        },
        {
            "id": "fitcoal_bottleneck",
            "short_label": "Proposed Ne≈1,280 bottleneck (contested)",
            "window_ma": [0.813, 0.930],
            "point_ma": 0.872,
            "tier": "contested_genomic",
            "tier_label": "Contested coalescent genomic inference",
            "observable": "Site-frequency spectra of 3,154 modern genomes (FitCoal vs PHLASH / structured stem)",
            "evidence_summary": "Hu et al. (2023) & Zhou et al. (2026) infer Ne≈1,280; Deng et al. (2025), Cousins & Durvasula (2025), and Terhorst (2025) find SFS artifact / structured-stem alternatives fit without collapse.",
            "hominin_scope": "Coalescent Ne is NOT a census headcount; occurs ~570 kyr after plateau onset; potentially reflects metapopulation fragmentation across MIS 24–22.",
            "cite_keys": ["hu2023", "deng2025bottleneck", "cousins2025", "terhorst2025", "zhou2026", "ragsdale2023"],
        },
        {
            "id": "pantasma_impact",
            "short_label": "Pantasma crater (14 km, Nicaragua)",
            "window_ma": [0.804, 0.826],
            "point_ma": 0.815,
            "tier": "unlinked_physical",
            "tier_label": "Dated impact — no hominin mortality link",
            "observable": "Shocked zircon/reidite, Cr isotopes, 40Ar/39Ar impact glass (815 ± 11 ka)",
            "evidence_summary": "Confirmed 14 km Central American impact crater in uninhabited Americas.",
            "hominin_scope": "No hominins in Americas; impact scale below global winter threshold.",
            "cite_keys": ["rochette2019"],
        },
        {
            "id": "oldest_toba_tuff",
            "short_label": "Oldest Toba Tuff eruption (~800 ka)",
            "window_ma": [0.79, 0.81],
            "point_ma": 0.800,
            "tier": "unlinked_physical",
            "tier_label": "Dated super-eruption — no hominin die-off shown",
            "observable": "Zircon U–Pb & Ar dating of Oldest Toba Tuff (distinct from ~74 ka Youngest Toba)",
            "evidence_summary": "Major Quaternary silicic eruption in Sumatra near 0.80 Ma.",
            "hominin_scope": "Occurs near end of plateau (~700 kyr after onset); Southeast Asian hominin occupation persists across interval.",
            "cite_keys": ["ito2020"],
        },
        {
            "id": "australasian_tektite_impact",
            "short_label": "Australasian Bolaven tektite impact (~789 ka)",
            "window_ma": [0.785, 0.793],
            "point_ma": 0.789,
            "tier": "unlinked_physical",
            "tier_label": "Dated impact glass — crater debated, no kill link",
            "observable": "Strewn tektites across SE Asia–Antarctica (788.1 ± 3 ka); Bolaven gravity/ejecta vs critiques",
            "evidence_summary": "Largest Quaternary tektite field; Bolaven source proposed (Sieh 2020, 2024) and disputed (Mizera 2022, 2024); Bose/Vietnam bifaces co-occur.",
            "hominin_scope": "Occurs ~710 kyr after plateau onset right before Phase IV rise; bifacial toolmakers persist in Bose & Vietnam.",
            "cite_keys": ["jourdan2019", "sieh2020", "mizera2024", "sieh2024", "michel2024tektite"],
        },
        {
            "id": "gby_cooking",
            "short_label": "Gesher Benot Ya'aqov cooking (~780 ka)",
            "window_ma": [0.77, 0.79],
            "point_ma": 0.780,
            "tier": "energetic_release",
            "tier_label": "Dated archaeological cooking evidence",
            "observable": "Heated pharyngeal teeth of giant carp in hearth zones (~780 ka)",
            "evidence_summary": "Earliest well-dated controlled thermal cooking of aquatic/animal tissue.",
            "hominin_scope": "Relaxes gut/metabolic budget ceiling right at the ~0.7 Ma Phase IV parallel brain-expansion onset.",
            "cite_keys": ["zohar2022"],
        },
        {
            "id": "matuyama_brunhes",
            "short_label": "Matuyama–Brunhes magnetic reversal (~773 ka)",
            "window_ma": [0.770, 0.776],
            "point_ma": 0.773,
            "tier": "unlinked_physical",
            "tier_label": "Dated geomagnetic horizon — no biological kill",
            "observable": "40Ar/39Ar lava + 10Be ice/marine synchronization at ~773 ka",
            "evidence_summary": "Last full geomagnetic polarity reversal; key chronostratigraphic anchor for Middle Pleistocene.",
            "hominin_scope": "Chronological marker only; no demonstrated hominin or mammalian extinction.",
            "cite_keys": ["singer2019"],
        },
    ]

    events = []
    for ev in raw_events:
        earliest_crisis = float(ev["window_ma"][1])
        events.append({
            **ev,
            "lag_after_published_onset_kyr": round((plateau_onset_pub - earliest_crisis) * 1000.0, 1),
            "lag_after_fitted_onset_kyr": round((plateau_onset_fit - earliest_crisis) * 1000.0, 1),
        })

    return {
        "schema_version": 1,
        "seed": seed,
        "lr04_source": {
            "path": "evidence/paleoclimate/lr04_benthic_stack.txt",
            "sha256": lr["sha256"],
            "bytes": lr["bytes"],
            "citation": "Lisiecki & Raymo (2005), Paleoceanography 20:PA1003, doi:10.1029/2004PA001071 (NOAA Study 5847)",
            "points_0_to_3_4_ma": int(len(age_ma)),
        },
        "fossil_sources_sha256": {
            "gingerich2022": _sha256(V9_DATA / "gingerich2022/41598_2022_15481_MOESM1_ESM.xlsx"),
            "will2021": _sha256(V9_DATA / "will2021/41467_2021_24290_MOESM5_ESM.xlsx"),
            "du2018": _sha256(V9_DATA / "du2018/du2018_ecv.xlsx"),
            "v9_e13_fossils": _sha256(V9_FOSSILS_JSON),
        },
        "plateau_summary": {
            "gingerich_n": int(v9_fos["n"]),
            "published_plateau_ma": [0.70, 1.50],
            "fitted_plateau_ma": [round(bp_fit[2], 2), round(bp_fit[1], 2)],
            "breakpoint_ci95_onset_ma": [round(float(x), 2) for x in v9_fos["bp_ci"][1]],
            "breakpoint_ci95_exit_ma": [round(float(x), 2) for x in v9_fos["bp_ci"][2]],
            "plateau_level_cm3": round(float(v9_fos["step"]["level_III"]), 1),
            "phase_I_level_cm3": round(float(v9_fos["step"]["level_I"]), 1),
            "phase_IV_end_cm3": round(float(v9_fos["step"]["end"]), 1),
            "phase_II_slope_ln_per_myr": round(float(v9_fos["step"]["slope_II"]), 4),
            "phase_III_free_slope_ln_per_myr": round(float(v9_fos["stasis"]["III"]["slope"]), 4),
            "phase_III_free_se_ln_per_myr": round(float(v9_fos["stasis"]["III"]["se"]), 4),
            "phase_III_specimens": int(v9_fos["stasis"]["III"]["n"]),
            "phase_IV_slope_ln_per_myr": round(float(v9_fos["step"]["slope_IV"]), 4),
            "linear_delta_aicc": round(float(v9_fos["models"]["linear"]["daicc"]), 2),
            "linear_delta_bic": round(float(v9_fos["models"]["linear"]["dbic"]), 2),
        },
        "du2018_age_uncertainty_mc": {
            "specimens_total": len(du_rows),
            "replicates": n_mc,
            "slopes": du_mc_summary,
        },
        "lr04_regimes": regimes,
        "mpt_amplification": {
            "pre_mpt_swing_permil": amp_pre_mpt,
            "mpt_crisis_swing_permil": amp_mpt,
            "post_mpt_swing_permil": amp_post_mpt,
            "mpt_vs_pre_mpt_ratio": round(amp_mpt / max(amp_pre_mpt, 1e-6), 3),
            "post_vs_pre_mpt_ratio": round(amp_post_mpt / max(amp_pre_mpt, 1e-6), 3),
            "mis34_peak_permil": round(mis34_peak, 3),
            "mis34_age_ma": round(mis34_age, 3),
            "mis22_peak_permil": round(mis22_peak, 3),
            "mis22_age_ma": round(mis22_age, 3),
        },
        "timing_gap_summary": {
            "onset_to_mpt_kyr": 250.0,
            "onset_to_mis34_cooling_kyr": 380.0,
            "onset_to_fitcoal_window_kyr": 570.0,
            "onset_to_bolaven_impact_kyr": 711.0,
        },
        "candidate_events": events,
        "lr04_plot_series": {
            "age_ma": [round(float(x), 4) for x in plot_grid],
            "d18o_permil": [round(float(x), 4) for x in d18o_grid],
            "envelope_age_ma": [round(float(x), 4) for x in env_centers],
            "envelope_mean": env_mean,
            "envelope_p10": env_p10,
            "envelope_p90": env_p90,
        },
    }


def main() -> None:
    out = analyze_paleoclimate_plateau()
    assert out["lr04_source"]["sha256"] == EXPECTED_LR04_SHA256
    dest = ROOT / "evidence/reference/performance-paleoclimate-plateau.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {dest} ({dest.stat().st_size:,} bytes)")
    print("Plateau summary:", json.dumps(out["plateau_summary"], indent=2))
    print("MPT amplification:", json.dumps(out["mpt_amplification"], indent=2))
    print("Du 2018 MC:", json.dumps(out["du2018_age_uncertainty_mc"], indent=2))


if __name__ == "__main__":
    main()
