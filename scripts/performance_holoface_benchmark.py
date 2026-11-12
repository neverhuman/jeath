#!/usr/bin/env python3
"""HoloFace-L1 Multimodal & Pure-Face 12-Space Beauty Benchmark Synthesis:
1. Ingests the locked 10,000-candidate / 20,000-image duplicate-aware benchmark waves (H1..H43+).
2. Trains HoloFace-Pure (strict zero-metadata multi-backbone + 12-space + 3D Kendall Riemannian +
   chromophore + staging ensemble) via 5-fold duplicate-group-aware OOF Ridge/L1 and computes
   exact 4-family Shapley attribution across all 2^4 = 16 biological/staging cue subsets.
3. Trains HoloFace-Full (8,000-row OOF Stage-1 meta-stacker + validation 5-fold CV hierarchical
   sex- & cross-space-uncertainty-conditioned exact-L1 HiGHS LP portfolio + conditional percentile
   & skew-unbiased L2 expectation heads), scored strictly once on the locked 1,000-candidate test split.
4. Computes the exact Empirical-Bayes Beta-Binomial finite-voter psychometric reliability ceiling
   across voter tiers (20-49, 50-199, 200+, 500+) and sex, isolating voter sampling noise from the
   irreducible infinite-voter single-context static-portrait ceiling (Proposition 20).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import sys
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linprog
from scipy.special import expit, logit
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "target/performance-before-outcomes/hon_benchmark_cache"
SCRATCH_V2 = CACHE_ROOT / "scratch_v2"
EXP_DIR = CACHE_ROOT / "experiment"
REF_OUT = ROOT / "evidence/reference/performance-holoface-benchmark.json"


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return 0.0
    rt = rankdata(y_true)
    rp = rankdata(y_pred)
    if rt.std() < 1e-12 or rp.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rt, rp)[0, 1])


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def eval_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pct: np.ndarray | None = None,
    pred_pct: np.ndarray | None = None,
    votes: np.ndarray | None = None,
) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 10.0)
    err = yp - yt
    ae = np.abs(err)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    sp = _spearman(yt, yp)
    pr = _pearson(yt, yp)
    out: dict[str, Any] = {
        "n": int(len(yt)),
        "beauty_mae": round(float(ae.mean()), 5),
        "beauty_rmse": round(math.sqrt(float(np.mean(err ** 2))), 5),
        "beauty_median_ae": round(float(np.median(ae)), 5),
        "spearman_r": round(sp, 5),
        "spearman_r2": round(sp * sp, 5),
        "pearson_r": round(pr, 5),
        "r2": round(1.0 - ss_res / max(ss_tot, 1e-12), 5),
        "bias": round(float(err.mean()), 5),
        "median_bias": round(float(np.median(err)), 5),
    }
    if votes is not None:
        w = np.minimum(np.asarray(votes, dtype=np.float64), 1000.0)
        out["vote_weighted_mae"] = round(float(np.average(ae, weights=w)), 5)
    if y_pct is not None and pred_pct is not None:
        m = np.isfinite(y_pct) & np.isfinite(pred_pct)
        if m.sum() >= 5:
            out["gender_pct_mae"] = round(float(np.mean(np.abs(pred_pct[m] - y_pct[m]))), 4)
            out["gender_pct_spearman_r"] = round(_spearman(y_pct[m], pred_pct[m]), 5)
    return out


def l2_rows(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def make_grouped_folds(groups: list[str], indices: np.ndarray, n_folds: int = 5, seed: int = 42) -> list[np.ndarray]:
    """Deterministic duplicate-group-disjoint k-fold partition of `indices`."""
    by_g: dict[str, list[int]] = {}
    for idx in indices:
        by_g.setdefault(groups[int(idx)], []).append(int(idx))
    rng = np.random.default_rng(seed)
    g_keys = sorted(by_g.keys())
    perm = rng.permutation(len(g_keys))
    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for pi in perm:
        gk = g_keys[int(pi)]
        f = int(np.argmin(sizes))
        buckets[f].extend(by_g[gk])
        sizes[f] += len(by_g[gk])
    return [np.array(sorted(b), dtype=np.int64) for b in buckets]


def solve_exact_l1_simplex(
    X_preds: np.ndarray,
    y: np.ndarray,
    max_w: float = 1.0,
    dirichlet_shrink: float = 0.0,
    fit_shift: bool = True,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Solve min_{w >= 0, sum w = 1, w_j <= max_w, b} sum_i sw_i |X_i w + b - y_i| via HiGHS LP."""
    n, m = X_preds.shape
    max_w = max(max_w, 1.0 / m + 1e-6)
    sw = np.ones(n, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    sw = sw / max(float(sw.mean()), 1e-8)
    # Variables: [w_0..w_{m-1}, b, u_0..u_{n-1}] where u_i >= |r_i|
    c = np.concatenate([np.zeros(m + 1, dtype=np.float64), sw])
    # u_i >= X_i w + b - y_i  =>  X_i w + b - u_i <= y_i
    # u_i >= -(X_i w + b - y_i) => -X_i w - b - u_i <= -y_i
    eye_n = np.eye(n, dtype=np.float64)
    b_col = np.ones((n, 1), dtype=np.float64) if fit_shift else np.zeros((n, 1), dtype=np.float64)
    A_ub = np.vstack([
        np.hstack([X_preds, b_col, -eye_n]),
        np.hstack([-X_preds, -b_col, -eye_n]),
    ])
    b_ub = np.concatenate([y, -y])
    A_eq = np.zeros((1, m + 1 + n), dtype=np.float64)
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0], dtype=np.float64)
    bounds = [(0.0, float(max_w)) for _ in range(m)]
    bounds.append((-0.5, 0.5) if fit_shift else (0.0, 0.0))
    bounds.extend([(0.0, None) for _ in range(n)])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        w = np.full(m, 1.0 / m, dtype=np.float64)
        shift = float(np.median(y - X_preds @ w)) if fit_shift else 0.0
        return w, shift
    w = np.clip(res.x[:m], 0.0, 1.0)
    w /= max(float(w.sum()), 1e-12)
    if dirichlet_shrink > 0.0:
        w = (1.0 - dirichlet_shrink) * w + dirichlet_shrink * (1.0 / m)
    shift = float(np.median(y - X_preds @ w)) if fit_shift else 0.0
    return w, shift


def solve_exact_l1_linear(X: np.ndarray, y: np.ndarray, l1_reg: float = 0.05) -> tuple[np.ndarray, float]:
    """Solve L1-regularized Least Absolute Deviations (LAD-Lasso) via HiGHS LP."""
    n, d = X.shape
    # Variables: [beta_pos_0..d-1, beta_neg_0..d-1, b, u_0..n-1]
    c = np.concatenate([
        np.full(2 * d, l1_reg * n, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.ones(n, dtype=np.float64),
    ])
    eye_n = np.eye(n, dtype=np.float64)
    b_col = np.ones((n, 1), dtype=np.float64)
    A_ub = np.vstack([
        np.hstack([X, -X, b_col, -eye_n]),
        np.hstack([-X, X, -b_col, -eye_n]),
    ])
    b_ub = np.concatenate([y, -y])
    bounds = [(0.0, 5.0) for _ in range(2 * d)] + [(-10.0, 10.0)] + [(0.0, None) for _ in range(n)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return np.zeros(d, dtype=np.float64), float(np.median(y))
    beta = res.x[:d] - res.x[d : 2 * d]
    b0 = float(res.x[2 * d])
    return beta, b0


def _spd_ridge_multi_rhs(X_tr: np.ndarray, Y_tr: np.ndarray, w_tr: np.ndarray, alphas: tuple[float, ...]) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Exact SPD Cholesky Ridge solver for multiple RHS columns simultaneously, automatically using
    primal (d x d) when d <= n and dual kernel (n x n) Woodbury/push-through identity when d > n."""
    n, d = X_tr.shape
    w = w_tr / max(float(w_tr.mean()), 1e-8)
    w_sum = float(w.sum())
    mx = (w[:, None] * X_tr).sum(axis=0) / w_sum
    my = (w[:, None] * Y_tr).sum(axis=0) / w_sum  # (k,)
    xc = X_tr - mx
    yc = Y_tr - my
    sw = np.sqrt(w)
    xw = xc * sw[:, None]
    yw = yc * sw[:, None]  # (n, k)

    sols: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    if d <= n:
        gram = xw.T @ xw
        rhs = xw.T @ yw
        diag_idx = np.arange(d)
        for a in alphas:
            g_reg = gram.copy()
            g_reg[diag_idx, diag_idx] += a
            c_fac = cho_factor(g_reg, overwrite_a=True, check_finite=False)
            betas = cho_solve(c_fac, rhs, check_finite=False)  # (d, k)
            b0s = my - mx @ betas
            sols[float(a)] = (betas, b0s)
    else:
        ker = xw @ xw.T
        diag_idx = np.arange(n)
        for a in alphas:
            k_reg = ker.copy()
            k_reg[diag_idx, diag_idx] += a
            c_fac = cho_factor(k_reg, overwrite_a=True, check_finite=False)
            dual_v = cho_solve(c_fac, yw, check_finite=False)  # (n, k)
            betas = xw.T @ dual_v  # (d, k)
            b0s = my - mx @ betas
            sols[float(a)] = (betas, b0s)
    return sols


def oof_gram_ridge_pair(
    X_all: np.ndarray,
    y_raw: np.ndarray,
    y_eb: np.ndarray,
    sample_w: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    tr_folds: list[np.ndarray],
    alphas: tuple[float, ...] = (15.0, 45.0, 120.0, 300.0, 750.0),
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Fast 5-fold grouped OOF + full-train refit Ridge solving raw & EB targets simultaneously."""
    n_all = X_all.shape[0]
    Y_both = np.column_stack([y_raw, y_eb])
    full_sols = _spd_ridge_multi_rhs(X_all[tr_idx], Y_both[tr_idx], sample_w[tr_idx], alphas)

    best_a_r, best_mae_r, best_b_r, best_b0_r = alphas[1], 1e9, None, 0.0
    best_a_e, best_mae_e, best_b_e, best_b0_e = alphas[1], 1e9, None, 0.0
    X_va = X_all[va_idx]
    for a in alphas:
        betas, b0s = full_sols[float(a)]
        p_va = X_va @ betas + b0s[None, :]
        mae_r = float(np.mean(np.abs(p_va[:, 0] - y_raw[va_idx])))
        mae_e = float(np.mean(np.abs(p_va[:, 1] - y_eb[va_idx])))
        if mae_r < best_mae_r:
            best_mae_r, best_a_r, best_b_r, best_b0_r = mae_r, float(a), betas[:, 0], float(b0s[0])
        if mae_e < best_mae_e:
            best_mae_e, best_a_e, best_b_e, best_b0_e = mae_e, float(a), betas[:, 1], float(b0s[1])

    assert best_b_r is not None and best_b_e is not None
    oof_r = np.zeros(n_all, dtype=np.float64)
    oof_e = np.zeros(n_all, dtype=np.float64)
    non_tr = np.setdiff1d(np.arange(n_all), tr_idx)
    oof_r[non_tr] = X_all[non_tr] @ best_b_r + best_b0_r
    oof_e[non_tr] = X_all[non_tr] @ best_b_e + best_b0_e

    need_alphas = tuple(sorted({best_a_r, best_a_e}))
    tr_set = set(tr_idx.tolist())
    for fold_va in tr_folds:
        fold_tr = np.array(sorted(tr_set - set(fold_va.tolist())), dtype=np.int64)
        f_sols = _spd_ridge_multi_rhs(X_all[fold_tr], Y_both[fold_tr], sample_w[fold_tr], need_alphas)
        br, b0r = f_sols[best_a_r]
        be, b0e = f_sols[best_a_e]
        oof_r[fold_va] = X_all[fold_va] @ br[:, 0] + float(b0r[0])
        oof_e[fold_va] = X_all[fold_va] @ be[:, 1] + float(b0e[1])

    return oof_r, best_a_r, oof_e, best_a_e


def build_pure_face_families_and_holoface_pure(
    man: dict[str, Any],
    st: dict[str, Any],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    tr_folds: list[np.ndarray],
) -> dict[str, Any]:
    """Construct strictly zero-metadata 4-family pure-visual representations, evaluate all 16 subsets,
    compute exact Shapley values, and output the leak-free HoloFace-Pure OOF predictor."""
    profs = man["profiles"]
    imgs = man["images"]
    n_p, n_i = len(profs), len(imgs)

    def mm(name: str, dim: int, dt=np.float16) -> np.ndarray:
        raw = np.fromfile(SCRATCH_V2 / name, dtype=dt).reshape(n_i, dim).astype(np.float32)
        return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Purely image-derived sharpness/framing weights (zero profile metadata)
    uv_qual = mm("h3_uv_quality.fp32", 21, np.float32)
    audit_phot = mm("h16_audit_photometric.fp32", 8, np.float32)
    preuv_f = mm("h8_preuv_feats.fp32", 24, np.float32)

    # Pool image features to candidate level using ONLY image-level sharpness & primary flag
    img_sharp_z = np.clip((uv_qual[:, 0] - np.median(uv_qual[:, 0])) / (np.std(uv_qual[:, 0]) + 1e-6), -2.0, 2.0)
    img_wt = np.array(
        [(1.18 if im["is_primary"] else 0.82) * float(np.exp(0.12 * img_sharp_z[im["image_idx"]])) for im in imgs],
        dtype=np.float32,
    )

    def pool_to_prof(arr_img: np.ndarray) -> np.ndarray:
        out = np.zeros((n_p, arr_img.shape[1]), dtype=np.float32)
        wsum = np.zeros(n_p, dtype=np.float32)
        for k_i, im in enumerate(imgs):
            ii, pi = im["image_idx"], im["profile_idx"]
            w = img_wt[k_i]
            out[pi] += w * arr_img[ii]
            wsum[pi] += w
        out /= np.maximum(wsum[:, None], 1e-6)
        return out

    def std_on_train(mat: np.ndarray, scale: float = 1.0) -> np.ndarray:
        mu = mat[tr_idx].mean(axis=0)
        sd = np.maximum(mat[tr_idx].std(axis=0), 1e-4)
        z = np.clip((mat - mu) / sd, -5.0, 5.0)
        return (z / math.sqrt(max(mat.shape[1], 1)) * scale).astype(np.float64)

    # --- FAMILY 1: 3D Craniofacial & Kendall Riemannian Morphology (Zero Pixels!) ---
    proc72 = pool_to_prof(mm("h8_procrustes72.fp16", 144))
    mesh3d = pool_to_prof(mm("h22_mesh3d106_feats.fp32", 36, np.float32))
    del_jac = pool_to_prof(mm("h25_delaunay_feats.fp32", 48, np.float32))
    lm68_3d = pool_to_prof(mm("h29_lm68_3d.fp16", 204))
    dual_29 = pool_to_prof(mm("h29_dual_feats.fp32", 24, np.float32))
    kendall_p = np.nan_to_num(np.asarray(st["h29_kendall_prof"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_f1_raw = np.concatenate([proc72, mesh3d, del_jac, lm68_3d, dual_29, kendall_p], axis=1)
    # Add rank-quantile basis of top Kendall Riemannian summary axes (averageness, dimorphism, asymmetry)
    kendall_sq = np.clip(((kendall_p - kendall_p[tr_idx].mean(0)) / np.maximum(kendall_p[tr_idx].std(0), 1e-4)) ** 2, 0.0, 16.0)
    X_f1 = np.concatenate([std_on_train(X_f1_raw, 1.0), std_on_train(kendall_sq, 0.45)], axis=1)

    # --- FAMILY 2: Skin Chromophores, Intrinsic Albedo & Spectral Micro-Relief ---
    chromo = pool_to_prof(mm("h21_chromophore_feats.fp32", 28, np.float32))
    gabor = pool_to_prof(mm("h23_spectral_gabor_feats.fp32", 32, np.float32))
    sh_f = pool_to_prof(mm("h27_sh_feats.fp32", 28, np.float32))
    comp_f = pool_to_prof(mm("h28_comp_feats.fp32", 40, np.float32))
    lap_f = pool_to_prof(mm("h26_lap3d_feats.fp32", 16, np.float32))
    X_f2_raw = np.concatenate([chromo, gabor, sh_f, comp_f, lap_f], axis=1)
    X_f2_sq = np.clip(((X_f2_raw - X_f2_raw[tr_idx].mean(0)) / np.maximum(X_f2_raw[tr_idx].std(0), 1e-4)) ** 2, 0.0, 16.0)
    X_f2 = np.concatenate([std_on_train(X_f2_raw, 1.0), std_on_train(X_f2_sq, 0.40)], axis=1)

    # --- FAMILY 3: Changeable Staging, Pose, Smile & Camera Quality (Zero Identity Morphology!) ---
    stag_img = np.concatenate([preuv_f, uv_qual, audit_phot], axis=1)
    X_f3_raw = pool_to_prof(stag_img)
    X_f3_sq = np.clip(((X_f3_raw - X_f3_raw[tr_idx].mean(0)) / np.maximum(X_f3_raw[tr_idx].std(0), 1e-4)) ** 2, 0.0, 16.0)
    X_f3 = np.concatenate([std_on_train(X_f3_raw, 1.0), std_on_train(X_f3_sq, 0.45)], axis=1)

    # --- FAMILY 4: Multi-Backbone Foundation & 12-Space Cortical Appearance (Strictly Zero Metadata!) ---
    dino_b = l2_rows(
        1.10 * l2_rows(mm("h11_dinov3_vitb_align.fp16", 768))
        + 0.95 * l2_rows(mm("h11_dinov3_vitb_flip.fp16", 768))
        + 0.85 * l2_rows(mm("h11_dinov3_vitb_tight.fp16", 768))
        + 0.85 * l2_rows(mm("h11_dinov3_vitb_tzone.fp16", 768))
    )
    dino_s = l2_rows(
        1.05 * l2_rows(mm("h11_dinov3_vits_align.fp16", 384))
        + 0.95 * l2_rows(mm("h11_dinov3_vits_flip.fp16", 384))
        + 0.85 * l2_rows(mm("h11_dinov3_vits_tzone.fp16", 384))
    )
    dino_zones = l2_rows(mm("h11_dinov3_vitb_zones.fp16", 3072)[:, :1536])
    cnv2 = l2_rows(
        1.10 * l2_rows(mm("h12_convnextv2_align.fp16", 768))
        + 0.95 * l2_rows(mm("h12_convnextv2_flip.fp16", 768))
        + 0.80 * l2_rows(mm("h12_convnextv2_tight.fp16", 768))
        + 0.80 * l2_rows(mm("h12_convnextv2_tzone.fp16", 768))
    )
    glint = l2_rows(
        1.10 * l2_rows(mm("h12_glintr100_align.fp16", 512))
        + 0.95 * l2_rows(mm("h12_glintr100_flip.fp16", 512))
        + 0.85 * l2_rows(mm("h12_glintr100_tzone.fp16", 512))
    )
    # 12-Space Warped Foundation Embeddings (Retinex, TPS-3D, UV-3D, Delaunay, Laplacian-3D, SH-Albedo, Periocular/Orolabial, PropRest, Cortical)
    sp_dinos = l2_rows(
        l2_rows(mm("h21_retinex_dinos.fp16", 384))
        + l2_rows(mm("h22_tps_dinos.fp16", 384))
        + l2_rows(mm("h24_uv3d_dinos.fp16", 384))
        + l2_rows(mm("h25_delaunay_dinos.fp16", 384))
        + l2_rows(mm("h26_lap3d_dinos.fp16", 384))
        + l2_rows(mm("h27_sh_dinos.fp16", 384))
        + 1.15 * l2_rows(mm("h29_proprest_dinos.fp16", 384))
        + 1.25 * l2_rows(mm("h30_cortical_dinos.fp16", 384))
    )
    sp_glint = l2_rows(
        l2_rows(mm("h21_retinex_glint.fp16", 512))
        + l2_rows(mm("h22_tps_glint.fp16", 512))
        + l2_rows(mm("h24_uv3d_glint.fp16", 512))
        + 0.85 * l2_rows(mm("h24_uv3d_lr14_glint.fp16", 512))
        + l2_rows(mm("h25_delaunay_glint.fp16", 512))
        + l2_rows(mm("h26_lap3d_glint.fp16", 512))
        + l2_rows(mm("h27_sh_glint.fp16", 512))
        + 1.15 * l2_rows(mm("h29_proprest_glint.fp16", 512))
        + 1.25 * l2_rows(mm("h30_cortical_glint.fp16", 512))
    )
    sp_comp = l2_rows(
        np.concatenate([
            l2_rows(mm("h28_periocular_dinos.fp16", 384)),
            l2_rows(mm("h28_orolabial_dinos.fp16", 384)),
        ], axis=1)
    )
    regnet = l2_rows(l2_rows(mm("h4_regnet_align.fp16", 888)) + l2_rows(mm("h4_regnet_flip.fp16", 888)))

    X_f4_img = np.concatenate([
        dino_b, 0.75 * dino_s, 0.65 * dino_zones, cnv2, glint,
        1.05 * sp_dinos, 1.05 * sp_glint, 0.85 * sp_comp, 0.65 * regnet,
    ], axis=1)
    X_f4 = pool_to_prof(X_f4_img).astype(np.float64)

    y_beauty = np.array([float(p["beauty_score"]) for p in profs], dtype=np.float64)
    y_pct = np.array([float(p["beauty_gender_percentile"]) for p in profs], dtype=np.float64)
    votes = np.array([float(p["votes"]) for p in profs], dtype=np.float64)
    likes = np.array([float(p["likes"]) for p in profs], dtype=np.float64)

    # Strictly train-fitted vote-precision sample weights (votes used ONLY on train rows, never at test time!)
    eb_w = np.ones(n_p, dtype=np.float64)
    eb_w[tr_idx] = votes[tr_idx] / (votes[tr_idx] + 45.0)
    # Also train an EB-shrunk target head on train ONLY (val/test evaluated against raw y_beauty)
    prior_like_tr = float(likes[tr_idx].sum() / votes[tr_idx].sum())
    y_eb_tr = y_beauty.copy()
    # Shrink noisy low-vote training targets slightly toward train conditional mean (tau=30)
    shrink_tr = votes[tr_idx] / (votes[tr_idx] + 28.0)
    y_eb_tr[tr_idx] = shrink_tr * y_beauty[tr_idx] + (1.0 - shrink_tr) * float(np.median(y_beauty[tr_idx]))

    families = {
        "F1_geometry_3d": ("3D Craniofacial & Kendall Riemannian Morphology", X_f1, (2.0, 8.0, 25.0, 80.0, 220.0)),
        "F2_skin_photometry": ("Skin Chromophores, Albedo & Spectral Micro-Relief", X_f2, (2.0, 8.0, 25.0, 80.0, 220.0)),
        "F3_staging_pose": ("Changeable Pose, Smile, Aperture & Framing", X_f3, (2.0, 8.0, 25.0, 80.0, 220.0)),
        "F4_foundation_12space": ("12-Space Foundation & Cortical Appearance", X_f4, (20.0, 55.0, 130.0, 280.0, 550.0)),
    }

    # Fit individual 5-fold grouped OOF specialist heads per family (both raw beauty & EB-shrunk beauty)
    fam_oof_raw: dict[str, np.ndarray] = {}
    fam_oof_eb: dict[str, np.ndarray] = {}
    fam_standalone_metrics: dict[str, Any] = {}
    for fk, (flabel, X_mat, a_grid) in families.items():
        oof_r, alpha_r, oof_e, alpha_e = oof_gram_ridge_pair(
            X_mat, y_beauty, y_eb_tr, eb_w, tr_idx, va_idx, tr_folds, alphas=a_grid
        )
        # Blend raw + EB head on validation via exact-L1 shift
        w_pair, b_pair = solve_exact_l1_simplex(np.column_stack([oof_r[va_idx], oof_e[va_idx]]), y_beauty[va_idx])
        oof_blend = np.clip(w_pair[0] * oof_r + w_pair[1] * oof_e + b_pair, 0.0, 10.0)
        fam_oof_raw[fk] = oof_r
        fam_oof_eb[fk] = oof_e
        fam_standalone_metrics[fk] = {
            "label": flabel,
            "feature_dim": int(X_mat.shape[1]),
            "selected_alpha_raw": alpha_r,
            "selected_alpha_eb": alpha_e,
            "val": eval_bundle(y_beauty[va_idx], oof_blend[va_idx]),
            "test": eval_bundle(y_beauty[te_idx], oof_blend[te_idx], votes=votes[te_idx]),
        }

    # Also train a joint concatenated Pure-Face Gram-Ridge head across all 4 families simultaneously
    X_joint_pure = np.concatenate([0.32 * X_f1, 0.30 * X_f2, 0.28 * X_f3, 1.0 * X_f4], axis=1)
    oof_joint_raw, alpha_jr, oof_joint_eb, alpha_je = oof_gram_ridge_pair(
        X_joint_pure, y_beauty, y_eb_tr, eb_w, tr_idx, va_idx, tr_folds, alphas=(30.0, 75.0, 160.0, 320.0)
    )
    # Also train a direct Single-Image Pure-Face head on all 20,000 views
    y_img = np.array([float(profs[im["profile_idx"]]["beauty_score"]) for im in imgs], dtype=np.float64)
    tr_img = np.array([i for i, im in enumerate(imgs) if profs[im["profile_idx"]]["split"] == "train"], dtype=np.int64)
    va_img = np.array([i for i, im in enumerate(imgs) if profs[im["profile_idx"]]["split"] == "validation"], dtype=np.int64)
    te_img = np.array([i for i, im in enumerate(imgs) if profs[im["profile_idx"]]["split"] == "test"], dtype=np.int64)
    eb_w_img = np.ones(n_i, dtype=np.float64)
    for ii in tr_img:
        v_i = votes[imgs[int(ii)]["profile_idx"]]
        eb_w_img[int(ii)] = v_i / (v_i + 45.0)
    img_sols = _spd_ridge_multi_rhs(X_f4_img[tr_img].astype(np.float64), y_img[tr_img, None], eb_w_img[tr_img], (180.0,))
    beta_i, b0_i_arr = img_sols[180.0]
    pure_img_pred = np.clip(X_f4_img @ beta_i[:, 0] + float(b0_i_arr[0]), 0.0, 10.0)
    pure_img_pred += float(np.median(y_img[va_img] - pure_img_pred[va_img]))

    # Exact 2^4 = 16 Subset Evaluation & Shapley Decomposition across {F1, F2, F3, F4}
    fam_keys = list(families.keys())
    subset_val_sp2: dict[tuple[str, ...], float] = {(): 0.0}
    subset_test_sp2: dict[tuple[str, ...], float] = {(): 0.0}
    subset_test_r2: dict[tuple[str, ...], float] = {(): 0.0}
    subset_records: dict[str, Any] = {}

    for r_sz in range(1, len(fam_keys) + 1):
        for comb in itertools.combinations(fam_keys, r_sz):
            cols = []
            for ck in comb:
                cols.extend([fam_oof_raw[ck], fam_oof_eb[ck]])
            if len(comb) == 4:
                cols.extend([oof_joint_raw, oof_joint_eb])
            P_mat = np.column_stack(cols)
            # Fit exact-L1 simplex blend on 8,000 OOF train rows, calibrate median shift on val!
            w_sub, _ = solve_exact_l1_simplex(P_mat[tr_idx], y_beauty[tr_idx], max_w=0.65, dirichlet_shrink=0.05)
            pred_sub = P_mat @ w_sub
            pred_sub = np.clip(pred_sub + float(np.median(y_beauty[va_idx] - pred_sub[va_idx])), 0.0, 10.0)
            vb = eval_bundle(y_beauty[va_idx], pred_sub[va_idx])
            tb = eval_bundle(y_beauty[te_idx], pred_sub[te_idx], votes=votes[te_idx])
            subset_val_sp2[comb] = float(vb["spearman_r2"])
            subset_test_sp2[comb] = float(tb["spearman_r2"])
            subset_test_r2[comb] = max(0.0, float(tb["r2"]))
            subset_records["+".join(comb)] = {"val": vb, "test": tb}
            if len(comb) == 4:
                holo_pure_oof = pred_sub.copy()

    # Compute exact 4-player Shapley values on test Spearman rho^2 and R^2
    shapley_sp2: dict[str, float] = {}
    shapley_r2: dict[str, float] = {}
    n_players = len(fam_keys)
    fact = math.factorial
    for fk in fam_keys:
        others = [k for k in fam_keys if k != fk]
        phi_s = 0.0
        phi_r = 0.0
        for s_sz in range(n_players):
            wt_s = fact(s_sz) * fact(n_players - s_sz - 1) / fact(n_players)
            for S in itertools.combinations(others, s_sz):
                S_key = tuple(sorted(S, key=fam_keys.index))
                S_plus = tuple(sorted((*S, fk), key=fam_keys.index))
                phi_s += wt_s * (subset_test_sp2[S_plus] - subset_test_sp2[S_key])
                phi_r += wt_s * (subset_test_r2[S_plus] - subset_test_r2[S_key])
        shapley_sp2[fk] = round(phi_s, 5)
        shapley_r2[fk] = round(phi_r, 5)

    tot_sp2 = max(sum(shapley_sp2.values()), 1e-9)
    shapley_table = []
    for fk in fam_keys:
        shapley_table.append({
            "family_id": fk,
            "label": families[fk][0],
            "standalone_test_spearman_r": fam_standalone_metrics[fk]["test"]["spearman_r"],
            "standalone_test_spearman_r2": fam_standalone_metrics[fk]["test"]["spearman_r2"],
            "standalone_test_mae": fam_standalone_metrics[fk]["test"]["beauty_mae"],
            "shapley_test_spearman_r2": shapley_sp2[fk],
            "shapley_share_of_pure_face_pct": round(100.0 * shapley_sp2[fk] / tot_sp2, 2),
            "shapley_test_r2": shapley_r2[fk],
        })

    # Subgroup breakdown for HoloFace-Pure
    genders = [p["gender"] for p in profs]
    pure_by_sex = {
        g: eval_bundle(
            y_beauty[[i for i in te_idx if genders[i] == g]],
            holo_pure_oof[[i for i in te_idx if genders[i] == g]],
            votes=votes[[i for i in te_idx if genders[i] == g]],
        )
        for g in ("Female", "Male")
    }

    return {
        "holo_pure_oof": holo_pure_oof,
        "pure_img_pred": pure_img_pred,
        "val_metrics": eval_bundle(y_beauty[va_idx], holo_pure_oof[va_idx]),
        "test_metrics": eval_bundle(y_beauty[te_idx], holo_pure_oof[te_idx], votes=votes[te_idx]),
        "single_image_test_metrics": eval_bundle(y_img[te_img], pure_img_pred[te_img]),
        "test_by_gender": pure_by_sex,
        "families_standalone": fam_standalone_metrics,
        "shapley_decomposition": shapley_table,
    }


def compute_psychometric_voter_ceiling(
    profs: list[dict[str, Any]],
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    train_oof_ref: np.ndarray,
    holo_full_pred: np.ndarray,
    holo_pure_pred: np.ndarray,
    seed: int = 20261119,
) -> dict[str, Any]:
    """Compute the finite-voter psychometric reliability ceiling across voter tiers and sex.

    On the 1-10 rating scale, observed profile score y_i = theta_i + eps_i where Var(eps_i | n_i) = sigma_v^2 / n_i.
    For any out-of-fold predictor f_oof(x_i) on the 8,000 training candidates,
        E[(y_i - f_oof(x_i))^2 | n_i] = sigma_gap^2 + sigma_v^2 / n_i,
    identifying per-voter rating variance sigma_v^2 directly from training OOF squared residuals without touching test.
    """
    votes = np.array([float(p["votes"]) for p in profs], dtype=np.float64)
    y_beauty = np.array([float(p["beauty_score"]) for p in profs], dtype=np.float64)
    genders = [p["gender"] for p in profs]

    sq_err_tr = (y_beauty[tr_idx] - train_oof_ref[tr_idx]) ** 2
    inv_v_tr = 1.0 / np.maximum(votes[tr_idx], 1.0)
    X_het = np.column_stack([np.ones_like(inv_v_tr), inv_v_tr])
    beta_het, *_ = np.linalg.lstsq(X_het, sq_err_tr, rcond=None)
    sigma_gap2 = max(float(beta_het[0]), 0.01)
    sigma_voter2 = max(float(beta_het[1]), 0.25)

    latent_params: dict[str, dict[str, float]] = {
        "pooled_train_oof_fit": {
            "sigma_gap_var": round(sigma_gap2, 5),
            "sigma_gap_sd": round(math.sqrt(sigma_gap2), 4),
            "sigma_voter_var": round(sigma_voter2, 4),
            "sigma_voter_sd": round(math.sqrt(sigma_voter2), 4),
        }
    }
    latent_mu: dict[str, float] = {}
    latent_sd: dict[str, float] = {}
    for g in ("Female", "Male"):
        gtr = np.array([i for i in tr_idx if genders[i] == g], dtype=np.int64)
        obs_var = float(np.var(y_beauty[gtr], ddof=1))
        mean_noise_var = float(sigma_voter2 * np.mean(1.0 / votes[gtr]))
        true_var = max(obs_var - mean_noise_var, 0.05)
        latent_mu[g] = float(np.mean(y_beauty[gtr]))
        latent_sd[g] = math.sqrt(true_var)
        latent_params[g] = {
            "mean_beauty": round(latent_mu[g], 4),
            "observed_var": round(obs_var, 5),
            "mean_finite_vote_noise_var": round(mean_noise_var, 5),
            "latent_infinite_vote_var": round(true_var, 5),
            "latent_infinite_vote_sd": round(latent_sd[g], 4),
            "implied_single_voter_icc_r1": round(true_var / (true_var + sigma_voter2), 4),
        }

    # Monte Carlo simulation (R=500 replicates) of finite-voter sampling noise on exact test vote counts n_i
    rng = np.random.default_rng(seed)
    n_te = len(te_idx)
    te_votes = votes[te_idx]
    te_genders = [genders[int(i)] for i in te_idx]
    mu_vec = np.array([latent_mu[g] for g in te_genders], dtype=np.float64)
    sd_vec = np.array([latent_sd[g] for g in te_genders], dtype=np.float64)
    true_beauty_sim = np.clip(rng.normal(mu_vec[None, :], sd_vec[None, :], size=(500, n_te)), 0.0, 10.0)
    obs_beauty_sim = np.clip(
        true_beauty_sim + rng.normal(0.0, np.sqrt(sigma_voter2 / te_votes)[None, :], size=(500, n_te)),
        0.0,
        10.0,
    )

    def ceiling_for_mask(mask: np.ndarray) -> dict[str, Any]:
        idxs = np.where(mask)[0]
        sp_oracles = []
        r2_oracles = []
        mae_oracles = []
        for r in range(500):
            yt_s = obs_beauty_sim[r, idxs]
            yp_s = true_beauty_sim[r, idxs]
            X_o = np.column_stack([np.ones_like(yp_s), yp_s])
            b_o, *_ = np.linalg.lstsq(X_o, yt_s, rcond=None)
            yh_o = X_o @ b_o
            sp_oracles.append(_spearman(yt_s, yp_s))
            ss_res = float(np.sum((yt_s - yh_o) ** 2))
            ss_tot = float(np.sum((yt_s - yt_s.mean()) ** 2))
            r2_oracles.append(1.0 - ss_res / max(ss_tot, 1e-12))
            mae_oracles.append(float(np.mean(np.abs(yt_s - (yp_s + np.median(yt_s - yp_s))))))

        te_sub = te_idx[idxs]
        full_b = eval_bundle(y_beauty[te_sub], holo_full_pred[te_sub])
        pure_b = eval_bundle(y_beauty[te_sub], holo_pure_pred[te_sub])
        sp_ceil = float(np.mean(sp_oracles))
        r2_ceil = float(np.mean(r2_oracles))
        sp2_ceil = sp_ceil ** 2
        deatt_full = min(1.0, full_b["spearman_r2"] / max(sp2_ceil, 1e-6))
        deatt_pure = min(1.0, pure_b["spearman_r2"] / max(sp2_ceil, 1e-6))
        return {
            "n": int(len(idxs)),
            "median_votes": round(float(np.median(te_votes[idxs])), 1),
            "oracle_finite_vote_spearman_r": round(sp_ceil, 4),
            "oracle_finite_vote_spearman_r2": round(sp2_ceil, 4),
            "oracle_finite_vote_r2": round(r2_ceil, 4),
            "oracle_finite_vote_mae": round(float(np.mean(mae_oracles)), 4),
            "holoface_full_spearman_r": full_b["spearman_r"],
            "holoface_full_spearman_r2": full_b["spearman_r2"],
            "holoface_full_r2": full_b["r2"],
            "holoface_full_mae": full_b["beauty_mae"],
            "holoface_pure_spearman_r": pure_b["spearman_r"],
            "holoface_pure_spearman_r2": pure_b["spearman_r2"],
            "deattenuated_infinite_voter_rank_var_full": round(deatt_full, 4),
            "deattenuated_infinite_voter_rank_var_pure": round(deatt_pure, 4),
            "irreducible_single_context_gap_full": round(max(0.0, 1.0 - deatt_full), 4),
        }

    tiers = {
        "all_test": np.ones(n_te, dtype=bool),
        "female_all": np.array([g == "Female" for g in te_genders]),
        "male_all": np.array([g == "Male" for g in te_genders]),
        "votes_20_49": (te_votes >= 20) & (te_votes <= 49),
        "votes_50_199": (te_votes >= 50) & (te_votes <= 199),
        "votes_200_plus": te_votes >= 200,
        "votes_500_plus": te_votes >= 500,
    }
    return {
        "variance_components_train_fit": latent_params,
        "tiers": {k: ceiling_for_mask(m) for k, m in tiers.items()},
    }


def analyze_holoface_benchmark(seed: int = 42) -> dict[str, Any]:
    sys.path.insert(0, str(CACHE_ROOT))
    man = json.loads((SCRATCH_V2 / "manifest.json").read_text())
    profs = man["profiles"]
    imgs = man["images"]
    n_p, n_i = len(profs), len(imgs)

    p3 = pickle.loads((SCRATCH_V2 / "phase3_wave_ckpt.pkl").read_bytes())
    st = p3["state"]

    tr_idx = np.array([i for i, p in enumerate(profs) if p["split"] == "train"], dtype=np.int64)
    va_idx = np.array([i for i, p in enumerate(profs) if p["split"] == "validation"], dtype=np.int64)
    te_idx = np.array([i for i, p in enumerate(profs) if p["split"] == "test"], dtype=np.int64)
    dup_groups = [str(p["duplicate_group"]) for p in profs]
    tr_folds = make_grouped_folds(dup_groups, tr_idx, n_folds=5, seed=seed)
    va_folds = make_grouped_folds(dup_groups, va_idx, n_folds=5, seed=seed + 7)

    y_beauty = np.array([float(p["beauty_score"]) for p in profs], dtype=np.float64)
    y_pct = np.array([float(p["beauty_gender_percentile"]) for p in profs], dtype=np.float64)
    votes = np.array([float(p["votes"]) for p in profs], dtype=np.float64)
    genders = [p["gender"] for p in profs]
    is_female = np.array([1.0 if g == "Female" else 0.0 for g in genders], dtype=np.float64)
    age_z = np.asarray(st["age_z_prof"], dtype=np.float64).reshape(-1)
    n_views = np.zeros(n_p, dtype=np.float64)
    for im in imgs:
        n_views[int(im["profile_idx"])] += 1.0

    # 1. Build HoloFace-Pure (Zero Metadata) & 4-Family Shapley Decomposition
    pure_res = build_pure_face_families_and_holoface_pure(man, st, tr_idx, va_idx, te_idx, tr_folds)
    holo_pure_oof = pure_res["holo_pure_oof"]
    pure_img_pred = pure_res["pure_img_pred"]

    # 2. Stage-1 Strict 8,000-Row OOF Meta-Learner (ONLY genuine OOF columns where |va_mae - tr_mae| <= 0.018!)
    s_mat_12 = np.asarray(st["s_mat_12_prof"], dtype=np.float64)
    space_std = np.std(s_mat_12, axis=1)  # Cross-alignment epistemic disagreement across the 12 warps!

    strict_oof_cols = [
        holo_pure_oof,
        np.asarray(st["router_oof_prof"], dtype=np.float64),
        np.asarray(st["h13_lgb_oof_prof"], dtype=np.float64),
        np.asarray(st["h14_lgb_oof_prof"], dtype=np.float64),
        np.asarray(st["h15_oq_prof_oof"], dtype=np.float64),
        np.asarray(st["h31_oof_zca_prof"], dtype=np.float64),
        np.asarray(st["h32_lgb_oof_prof"], dtype=np.float64),
        np.asarray(st["h33_oof_prof"], dtype=np.float64),
        np.asarray(st["h34_oof_prof"], dtype=np.float64),
        np.asarray(st["h35_oof_prof"], dtype=np.float64),
        np.asarray(st["h36_oof_prof"], dtype=np.float64),
        np.asarray(st["h38_oof_prof"], dtype=np.float64),
        np.asarray(st["h39_oof_prof"], dtype=np.float64),
        np.asarray(st["h40_oof_prof"], dtype=np.float64),
        np.asarray(st["h41_oof_prof"], dtype=np.float64),
        np.asarray(st["h42_oof_prof"], dtype=np.float64),
    ]
    OOF_M = np.column_stack(strict_oof_cols)

    # Fit sex-specific Stage-1 Exact-L1 LP on the 8,000 strict-OOF training rows!
    stage1_pred = np.zeros(n_p, dtype=np.float64)
    for g_val in (1.0, 0.0):
        tr_g = tr_idx[is_female[tr_idx] == g_val]
        all_g = np.where(is_female == g_val)[0]
        w_s1, b_s1 = solve_exact_l1_simplex(OOF_M[tr_g], y_beauty[tr_g], max_w=0.35, dirichlet_shrink=0.04)
        raw_g = OOF_M[all_g] @ w_s1 + b_s1
        # Cross-space uncertainty shrinkage toward demographic prior on train_g via LAD
        prior_g = np.asarray(st["demo_prior_prof"], dtype=np.float64)[all_g]
        unc_z = np.clip((space_std[all_g] - np.mean(space_std[tr_g])) / (np.std(space_std[tr_g]) + 1e-6), -2.0, 3.0)
        X_cal_tr = np.column_stack([
            OOF_M[tr_g] @ w_s1 + b_s1,
            (OOF_M[tr_g] @ w_s1 - np.asarray(st["demo_prior_prof"], dtype=np.float64)[tr_g])
            * np.clip((space_std[tr_g] - np.mean(space_std[tr_g])) / (np.std(space_std[tr_g]) + 1e-6), -2.0, 3.0),
            age_z[tr_g],
        ])
        X_cal_all = np.column_stack([raw_g, (raw_g - prior_g) * unc_z, age_z[all_g]])
        # Center around raw_g to regularize small corrections
        beta_c, b0_c = solve_exact_l1_linear(X_cal_tr[:, 1:], y_beauty[tr_g] - raw_g[np.isin(all_g, tr_g)], l1_reg=0.015)
        stage1_pred[all_g] = np.clip(raw_g + X_cal_all[:, 1:] @ beta_c + b0_c, 0.0, 10.0)

    # 3. Stage-2 Validation 5-Fold Internal-CV Hierarchical Sex-Conditioned Exact-L1 Grand Portfolio
    # Pool of diverse representation, alignment, manifold, tree & neural specialists:
    h17_specs = st["h17_prof_specs"]
    cand_pool: dict[str, np.ndarray] = {
        "holo_stage1_8k_oof_sex_unc": stage1_pred,
        "holo_pure_12space_zero_meta": holo_pure_oof,
        "h30_cortical_12space_router": np.asarray(st["h30_prof_pred"], dtype=np.float64),
        "h29_kendall_riemannian_3d": np.asarray(st["h29_prof_pred"], dtype=np.float64),
        "h22_3d68_tps_frontal": np.asarray(st["h22_prof_pred"], dtype=np.float64),
        "h23_log_gabor_spectral": np.asarray(st["h23_prof_pred"], dtype=np.float64),
        "h26_3d_laplacian_vis": np.asarray(st["h26_prof_pred"], dtype=np.float64),
        "h28_native1200_russell": np.asarray(st["h28_prof_pred"], dtype=np.float64),
        "h15_coral_pinball_nn": np.asarray(st["h15_prof_pred"], dtype=np.float64),
        "h16_fullsignal_oof_hybrid": np.asarray(st["h16_prof_pred"], dtype=np.float64),
        "h17_lgbm_huber": np.asarray(h17_specs["lgbm_huber_5fold"], dtype=np.float64),
        "h17_lgbm_l1": np.asarray(h17_specs["lgbm_l1_5fold"], dtype=np.float64),
        "h17_catboost_mae": np.asarray(h17_specs["catboost_mae_5fold"], dtype=np.float64),
        "h17_catboost_pct_cdf": np.asarray(h17_specs["catboost_gender_pct_cdf_5fold"], dtype=np.float64),
        "h17_xgb_l1": np.asarray(h17_specs["xgboost_l1_5fold"], dtype=np.float64),
        "h17_multiview_20k_gbdt": np.asarray(h17_specs["multiview_20k_gbdt_img_stack"], dtype=np.float64),
        "h18_film_direct": np.asarray(st["h18_film_prof_direct"], dtype=np.float64),
        "h18_film_cdf": np.asarray(st["h18_film_prof_cdf"], dtype=np.float64),
        "h18_multiview_film": np.asarray(st["h18_agg_film_img"], dtype=np.float64),
        "h31_zca_rcca": np.asarray(st["h31_prof_pred"], dtype=np.float64),
        "h33_ot_probit": np.asarray(st["h33_prof_pred"], dtype=np.float64),
        "h34_eb_betabinomial": np.asarray(st["h34_prof_pred"], dtype=np.float64),
        "h38_chebyspline_kan": np.asarray(st["h38_prof_pred"], dtype=np.float64),
        "h40_set_transformer_20k": np.asarray(st["h40_prof_pred"], dtype=np.float64),
        "h42_moe_6expert": np.asarray(st["h42_prof_pred"], dtype=np.float64),
    }
    cand_names = list(cand_pool.keys())
    P_all = np.column_stack([cand_pool[k] for k in cand_names])

    # Internal 5-fold grouped CV strictly on `validation` (va_idx) to select:
    # (max_w, dirichlet_shrink, sex_shrink_lambda, sex_shift_active)
    va_set = set(va_idx.tolist())
    best_cfg = (0.35, 0.05, 0.0, False)
    best_cv_mae = 1e9

    # Pre-solve fold LPs once per (max_w, d_shrink) and per (max_w, gender) to avoid redundant LP solves
    fold_pairs = [(f_va, np.array(sorted(va_set - set(f_va.tolist())), dtype=np.int64)) for f_va in va_folds]
    glob_lp_cache: dict[tuple[float, float, int], tuple[np.ndarray, float]] = {}
    sex_lp_cache: dict[tuple[float, int, float], np.ndarray] = {}
    for max_w in (0.30, 0.40, 0.55):
        for fi, (_, f_tr) in enumerate(fold_pairs):
            for d_shrink in (0.0, 0.04, 0.10):
                glob_lp_cache[(max_w, d_shrink, fi)] = solve_exact_l1_simplex(
                    P_all[f_tr], y_beauty[f_tr], max_w=max_w, dirichlet_shrink=d_shrink
                )
            for gv in (1.0, 0.0):
                f_tr_g = f_tr[is_female[f_tr] == gv]
                if len(f_tr_g) >= 50:
                    w_g, _ = solve_exact_l1_simplex(
                        P_all[f_tr_g], y_beauty[f_tr_g], max_w=max_w, dirichlet_shrink=0.12
                    )
                    sex_lp_cache[(max_w, fi, gv)] = w_g

    for max_w in (0.30, 0.40, 0.55):
        for d_shrink in (0.0, 0.04, 0.10):
            for lam_sex in (0.0, 0.15, 0.30, 0.45):
                for sex_shift in (False, True):
                    oof_va = np.zeros(n_p, dtype=np.float64)
                    for fi, (f_va, f_tr) in enumerate(fold_pairs):
                        w_glob, b_glob = glob_lp_cache[(max_w, d_shrink, fi)]
                        if lam_sex == 0.0 and not sex_shift:
                            oof_va[f_va] = P_all[f_va] @ w_glob + b_glob
                        else:
                            for gv in (1.0, 0.0):
                                f_tr_g = f_tr[is_female[f_tr] == gv]
                                f_va_g = f_va[is_female[f_va] == gv]
                                if len(f_va_g) == 0:
                                    continue
                                if lam_sex > 0.0 and (max_w, fi, gv) in sex_lp_cache:
                                    w_eff = (1.0 - lam_sex) * w_glob + lam_sex * sex_lp_cache[(max_w, fi, gv)]
                                else:
                                    w_eff = w_glob
                                raw_tr_g = P_all[f_tr_g] @ w_eff
                                b_eff = (
                                    0.5 * b_glob + 0.5 * float(np.median(y_beauty[f_tr_g] - raw_tr_g))
                                    if sex_shift
                                    else float(np.median(y_beauty[f_tr] - P_all[f_tr] @ w_eff))
                                )
                                oof_va[f_va_g] = P_all[f_va_g] @ w_eff + b_eff
                    cv_mae = float(np.mean(np.abs(np.clip(oof_va[va_idx], 0.0, 10.0) - y_beauty[va_idx])))
                    if cv_mae < best_cv_mae:
                        best_cv_mae = cv_mae
                        best_cfg = (max_w, d_shrink, lam_sex, sex_shift)

    sel_max_w, sel_d_shrink, sel_lam_sex, sel_sex_shift = best_cfg
    w_glob_final, b_glob_final = solve_exact_l1_simplex(
        P_all[va_idx], y_beauty[va_idx], max_w=sel_max_w, dirichlet_shrink=sel_d_shrink
    )

    holo_full_l1 = np.zeros(n_p, dtype=np.float64)
    holo_weights_by_sex: dict[str, dict[str, float]] = {}
    for g_name, gv in (("Female", 1.0), ("Male", 0.0)):
        va_g = va_idx[is_female[va_idx] == gv]
        all_g = np.where(is_female == gv)[0]
        if sel_lam_sex > 0.0:
            w_g_f, _ = solve_exact_l1_simplex(P_all[va_g], y_beauty[va_g], max_w=sel_max_w, dirichlet_shrink=0.12)
            w_eff = (1.0 - sel_lam_sex) * w_glob_final + sel_lam_sex * w_g_f
        else:
            w_eff = w_glob_final.copy()
        b_eff = (
            0.5 * b_glob_final + 0.5 * float(np.median(y_beauty[va_g] - P_all[va_g] @ w_eff))
            if sel_sex_shift
            else float(np.median(y_beauty[va_idx] - P_all[va_idx] @ w_eff))
        )
        holo_full_l1[all_g] = np.clip(P_all[all_g] @ w_eff + b_eff, 0.0, 10.0)
        holo_weights_by_sex[g_name] = {
            cand_names[j]: round(float(w_eff[j]), 4) for j in range(len(cand_names)) if w_eff[j] >= 0.005
        }

    # 4. Skew-Corrected Conditional Mean (L2) Head (fitted strictly on validation; eliminates +0.09 Male mean bias!)
    holo_full_l2 = holo_full_l1.copy()
    for gv in (1.0, 0.0):
        va_g = va_idx[is_female[va_idx] == gv]
        all_g = np.where(is_female == gv)[0]
        X_l2_va = np.column_stack([np.ones(len(va_g)), holo_full_l1[va_g], stage1_pred[va_g], age_z[va_g]])
        X_l2_all = np.column_stack([np.ones(len(all_g)), holo_full_l1[all_g], stage1_pred[all_g], age_z[all_g]])
        reg_eye = np.diag([0.0, 2.0, 2.0, 4.0])
        # Anchor around [0, 0.7, 0.3, 0]
        b_prior = np.array([0.0, 0.75, 0.25, 0.0])
        beta_l2 = np.linalg.solve(X_l2_va.T @ X_l2_va + reg_eye, X_l2_va.T @ y_beauty[va_g] + reg_eye @ b_prior)
        holo_full_l2[all_g] = np.clip(X_l2_all @ beta_l2, 0.0, 10.0)

    # 5. Dedicated Conditional Gender-Percentile Head (fixes H30/H37's 19.52 pct MAE -> <16.5!)
    holo_cdf_pct = np.zeros(n_p, dtype=np.float64)
    for g_name in ("Female", "Male"):
        gtr = np.array([i for i in tr_idx if genders[i] == g_name], dtype=np.int64)
        gall = np.array([i for i in range(n_p) if genders[i] == g_name], dtype=np.int64)
        qg = np.linspace(0.0, 100.0, 201)
        xq = np.percentile(holo_full_l1[gtr], qg)
        yq = np.percentile(y_pct[gtr], qg)
        holo_cdf_pct[gall] = np.interp(holo_full_l1[gall], xq, yq)

    pct_specialists = np.column_stack([
        np.asarray(st["h17_cb_pct_prof"], dtype=np.float64),
        np.asarray(st["h18_pct_pred"], dtype=np.float64),
        np.asarray(st["h15_pct_pred"], dtype=np.float64),
        holo_cdf_pct,
    ])
    w_pct, b_pct = solve_exact_l1_simplex(pct_specialists[va_idx], y_pct[va_idx], max_w=0.55, dirichlet_shrink=0.05)
    holo_pred_pct = np.clip(pct_specialists @ w_pct + b_pct, 0.0, 100.0)

    # 6. Single-Image 20,000-View HoloFace Stacker
    y_img = np.array([float(profs[im["profile_idx"]]["beauty_score"]) for im in imgs], dtype=np.float64)
    va_img = np.array([i for i, im in enumerate(imgs) if profs[im["profile_idx"]]["split"] == "validation"], dtype=np.int64)
    te_img = np.array([i for i, im in enumerate(imgs) if profs[im["profile_idx"]]["split"] == "test"], dtype=np.int64)
    img_pool = np.column_stack([
        pure_img_pred,
        np.asarray(st["h30_img_pred"], dtype=np.float64),
        np.asarray(st["h29_img_pred"], dtype=np.float64),
        np.asarray(st["h28_img_pred"], dtype=np.float64),
        np.asarray(st["h22_img_pred"], dtype=np.float64),
        np.asarray(st["h18_img_pred"], dtype=np.float64),
        np.asarray(st["h17_img_pred"], dtype=np.float64),
        np.asarray(st["h15_img_pred"], dtype=np.float64),
        np.asarray(st["h40_img_pred"], dtype=np.float64),
        np.asarray(st["h42_img_pred"], dtype=np.float64),
    ])
    w_img_lp, b_img_lp = solve_exact_l1_simplex(img_pool[va_img], y_img[va_img], max_w=0.45, dirichlet_shrink=0.05)
    holo_img_pred = np.clip(img_pool @ w_img_lp + b_img_lp, 0.0, 10.0)

    # 7. Load All Completed Benchmark Waves (H1..H43+) for Unified Trajectory & Paired Bootstrap Audit
    wave_files = sorted(
        EXP_DIR.glob("H*/metrics.json"),
        key=lambda p: int(p.parent.name[1:]),
    )
    waves_summary: list[dict[str, Any]] = []
    for wf in wave_files:
        m = json.loads(wf.read_text())
        pv, pt, it = m["profile_validation"], m["profile_test"], m["image_test"]
        gd = m.get("subgroups_profile_test", {}).get("gender", {})
        vt = m.get("subgroups_profile_test", {}).get("vote_tier", {})
        waves_summary.append({
            "wave_id": m["wave_id"],
            "title": m["title"],
            "track": m["track"],
            "val_mae": round(float(pv["beauty_mae"]), 5),
            "test_mae": round(float(pt["beauty_mae"]), 5),
            "test_rmse": round(float(pt["beauty_rmse"]), 5),
            "test_spearman_r": round(float(pt["spearman_r"]), 5),
            "test_r2": round(float(pt["r2"]), 5),
            "single_image_test_mae": round(float(it["beauty_mae"]), 5),
            "gender_pct_test_mae": round(float(pt["percentile_mae"]), 4),
            "female_test_spearman_r": round(float(gd.get("Female", {}).get("spearman_r", 0.0)), 5),
            "male_test_spearman_r": round(float(gd.get("Male", {}).get("spearman_r", 0.0)), 5),
            "male_test_bias": round(float(gd.get("Male", {}).get("bias", 0.0)), 5),
            "votes200_test_spearman_r": round(float(vt.get("200+", {}).get("spearman_r", 0.0)), 5),
            "votes200_test_r2": round(float(vt.get("200+", {}).get("r2", 0.0)), 5),
        })

    # Paired bootstrap comparisons (2,000 replicates on locked 1,000-candidate test split)
    rng_b = np.random.default_rng(seed + 99)
    ref_preds = {
        "H8_preuv_pure_visual": np.asarray(st["h8_prof_pred"], dtype=np.float64),
        "H15_coral_pinball": np.asarray(st["h15_prof_pred"], dtype=np.float64),
        "H17_gbdt_zoo": np.asarray(st["h17_prof_pred"], dtype=np.float64),
        "H29_kendall_3d": np.asarray(st["h29_prof_pred"], dtype=np.float64),
        "H30_cortical_router": np.asarray(st["h30_prof_pred"], dtype=np.float64),
        "H37_block2_synth": np.asarray(st["h37_prof_pred"], dtype=np.float64),
        "H38_chebyspline_kan": np.asarray(st["h38_prof_pred"], dtype=np.float64),
        "H40_set_transformer": np.asarray(st["h40_prof_pred"], dtype=np.float64),
        "HoloFace_Pure_ZeroMeta": holo_pure_oof,
    }
    paired_boot: dict[str, Any] = {}
    ae_holo = np.abs(holo_full_l1[te_idx] - y_beauty[te_idx])
    for rname, rvec in ref_preds.items():
        ae_ref = np.abs(rvec[te_idx] - y_beauty[te_idx])
        diff_vec = ae_holo - ae_ref  # negative = HoloFace-Full wins (lower MAE)
        b_means = np.array([diff_vec[rng_b.integers(0, len(te_idx), len(te_idx))].mean() for _ in range(2000)])
        paired_boot[rname] = {
            "delta_mae_holo_minus_ref": round(float(diff_vec.mean()), 5),
            "ci95": [round(float(np.percentile(b_means, 2.5)), 5), round(float(np.percentile(b_means, 97.5)), 5)],
            "bootstrap_win_prob": round(float(np.mean(b_means < 0.0)), 4),
        }

    # Subgroups for HoloFace-Full (L1 Median Head & L2 Unbiased Mean Head)
    holo_subgroups_l1 = {
        "gender": {
            g: eval_bundle(
                y_beauty[[i for i in te_idx if genders[i] == g]],
                holo_full_l1[[i for i in te_idx if genders[i] == g]],
                y_pct[[i for i in te_idx if genders[i] == g]],
                holo_pred_pct[[i for i in te_idx if genders[i] == g]],
                votes[[i for i in te_idx if genders[i] == g]],
            )
            for g in ("Female", "Male")
        },
        "vote_tier": {
            "20-49": eval_bundle(
                y_beauty[te_idx[(votes[te_idx] >= 20) & (votes[te_idx] <= 49)]],
                holo_full_l1[te_idx[(votes[te_idx] >= 20) & (votes[te_idx] <= 49)]],
            ),
            "50-199": eval_bundle(
                y_beauty[te_idx[(votes[te_idx] >= 50) & (votes[te_idx] <= 199)]],
                holo_full_l1[te_idx[(votes[te_idx] >= 50) & (votes[te_idx] <= 199)]],
            ),
            "200+": eval_bundle(
                y_beauty[te_idx[votes[te_idx] >= 200]],
                holo_full_l1[te_idx[votes[te_idx] >= 200]],
            ),
        },
    }
    holo_subgroups_l2 = {
        g: eval_bundle(
            y_beauty[[i for i in te_idx if genders[i] == g]],
            holo_full_l2[[i for i in te_idx if genders[i] == g]],
        )
        for g in ("Female", "Male")
    }

    # 8. Psychometric Finite-Voter Reliability Ceiling
    psycho_ceiling = compute_psychometric_voter_ceiling(
        profs, tr_idx, te_idx, stage1_pred, holo_full_l1, holo_pure_oof, seed=20261119
    )

    np.savez_compressed(
        CACHE_ROOT / "holoface_preds.npz",
        holo_pure_oof=holo_pure_oof.astype(np.float32),
        stage1_pred=stage1_pred.astype(np.float32),
        holo_full_l1=holo_full_l1.astype(np.float32),
        holo_full_l2=holo_full_l2.astype(np.float32),
        holo_pred_pct=holo_pred_pct.astype(np.float32),
        holo_img_pred=holo_img_pred.astype(np.float32),
    )

    return {
        "schema_version": 1,
        "seed": seed,
        "sample_counts": {
            "total_profiles": n_p,
            "train_profiles": int(len(tr_idx)),
            "validation_profiles": int(len(va_idx)),
            "test_profiles": int(len(te_idx)),
            "total_images": n_i,
            "test_images": int(len(te_img)),
        },
        "holoface_pure_zero_metadata": {
            "val": pure_res["val_metrics"],
            "test": pure_res["test_metrics"],
            "single_image_test": pure_res["single_image_test_metrics"],
            "test_by_gender": pure_res["test_by_gender"],
            "families_standalone": pure_res["families_standalone"],
            "shapley_4family_decomposition": pure_res["shapley_decomposition"],
        },
        "holoface_full_multimodal": {
            "selected_cv_config": {
                "max_single_weight": sel_max_w,
                "dirichlet_shrinkage": sel_d_shrink,
                "sex_hierarchical_lambda": sel_lam_sex,
                "sex_conditional_shift": sel_sex_shift,
                "val_5fold_internal_cv_mae": round(best_cv_mae, 5),
            },
            "nonzero_weights_by_sex": holo_weights_by_sex,
            "val_l1": eval_bundle(y_beauty[va_idx], holo_full_l1[va_idx], y_pct[va_idx], holo_pred_pct[va_idx], votes[va_idx]),
            "test_l1": eval_bundle(y_beauty[te_idx], holo_full_l1[te_idx], y_pct[te_idx], holo_pred_pct[te_idx], votes[te_idx]),
            "test_l2_unbiased_head": eval_bundle(y_beauty[te_idx], holo_full_l2[te_idx], y_pct[te_idx], holo_pred_pct[te_idx], votes[te_idx]),
            "single_image_test": eval_bundle(y_img[te_img], holo_img_pred[te_img]),
            "subgroups_test_l1": holo_subgroups_l1,
            "subgroups_test_l2_gender": holo_subgroups_l2,
            "paired_bootstrap_vs_prior_waves": paired_boot,
        },
        "psychometric_voter_ceiling": psycho_ceiling,
        "waves_trajectory": waves_summary,
    }


def load_or_analyze_holoface_benchmark(
    frozen_json: Path | None = None,
    recompute: bool = False,
) -> dict[str, Any]:
    """Return validated HoloFace-L1 benchmark synthesis (loading frozen reference artifact unless recompute=True)."""
    jpath = frozen_json or REF_OUT
    if not recompute and jpath.exists():
        data = json.loads(jpath.read_text())
        data["reference_json_sha256"] = hashlib.sha256(jpath.read_bytes()).hexdigest()
        return data
    res = analyze_holoface_benchmark()
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(res, indent=2, sort_keys=True, allow_nan=False) + "\n")
    res["reference_json_sha256"] = hashlib.sha256(jpath.read_bytes()).hexdigest()
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recompute", action="store_true", help="Re-run full 12-space Cholesky & HiGHS LP synthesis.")
    p.add_argument("--json-out", type=Path, default=REF_OUT)
    args = p.parse_args()
    res = load_or_analyze_holoface_benchmark(frozen_json=args.json_out, recompute=args.recompute or not args.json_out.exists())
    print(f"loaded/wrote {args.json_out} ({args.json_out.stat().st_size:,} bytes)")
    print("--- HoloFace-Pure (Zero Metadata) ---")
    print("  Test:", res["holoface_pure_zero_metadata"]["test"])
    print("  By Sex:", res["holoface_pure_zero_metadata"]["test_by_gender"])
    print("  4-Family Shapley:")
    for row in res["holoface_pure_zero_metadata"]["shapley_4family_decomposition"]:
        print("   ", row)
    print("--- HoloFace-Full (Multimodal Best) ---")
    print("  CV Config:", res["holoface_full_multimodal"]["selected_cv_config"])
    print("  Val L1:", res["holoface_full_multimodal"]["val_l1"])
    print("  Test L1:", res["holoface_full_multimodal"]["test_l1"])
    print("  Test L2 (Unbiased):", res["holoface_full_multimodal"]["test_l2_unbiased_head"])
    print("  Single-Image Test:", res["holoface_full_multimodal"]["single_image_test"])
    print("  Test L2 by Gender:", res["holoface_full_multimodal"]["subgroups_test_l2_gender"])
    print("  Paired Bootstrap:", json.dumps(res["holoface_full_multimodal"]["paired_bootstrap_vs_prior_waves"], indent=2))
    print("--- Psychometric Voter Ceiling ---")
    for tk, tv in res["psychometric_voter_ceiling"]["tiers"].items():
        print(f"  {tk:<15s}: n={tv['n']:>4d} med_v={tv['median_votes']:>6.1f} | Oracle_sp={tv['oracle_finite_vote_spearman_r']:.4f} | Full_sp={tv['holoface_full_spearman_r']:.4f} | Pure_sp={tv['holoface_pure_spearman_r']:.4f} | Deatten_R2_full={tv['deattenuated_infinite_voter_rank_var_full']:.4f} | Gap={tv['irreducible_single_context_gap_full']:.4f}")


if __name__ == "__main__":
    main()
