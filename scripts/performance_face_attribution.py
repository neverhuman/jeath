#!/usr/bin/env python3
"""Deep-learning canonical facial attribution, super-resolved 106-landmark Delaunay composites,
and within-gender beauty percentile driver/rejector maps across sex, age buckets, and world regions.

Uses the locally staged H25 + H34 + H20 multi-backbone stack in
`target/performance-before-outcomes/face-models/` (DINOv3 ViT-B/16 & ViT-S/16 LVD-1689M,
ConvNeXt V2 FCMAE+GRN, AuraFace-v1 Glint360K-R100 & 2d106det 106-pt landmarks, YuNet 5-pt,
and 218-triangle Delaunay canonical atlas) strictly on CPU (`CUDA_VISIBLE_DEVICES=""`).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import logit, expit, ndtri
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "target/performance-before-outcomes/face-models"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
MATTE_RGB = np.array([0, 38, 77], dtype=np.int16)
CANONICAL_UV_5PT = np.array(
    [[388.0, 312.0], [794.0, 312.0], [599.0, 553.0], [420.0, 757.0], [778.0, 757.0]],
    dtype=np.float32,
)

AGE_BUCKETS = ("18-24", "25-34", "35-44", "45-54", "55+")
WORLD_REGIONS = (
    "United States",
    "South America",
    "Mexico, Central Am. & Caribbean",
    "Canada",
    "Asia-Pacific & Oceania",
    "Europe",
    "Middle East & Africa",
)

EUR_COUNTRIES = {
    "France", "Spain", "Italy", "United Kingdom", "Germany", "Netherlands", "Sweden",
    "Switzerland", "Belgium", "Portugal", "Greece", "Poland", "Russia", "Ireland",
    "Norway", "Denmark", "Austria", "Finland", "Romania", "Czech Republic", "Hungary",
    "Ukraine", "Croatia", "Iceland", "Bulgaria", "Serbia", "Slovakia", "Slovenia",
    "Lithuania", "Latvia", "Estonia", "Luxembourg", "Malta", "Cyprus", "Moldova", "Belarus",
}
SAM_COUNTRIES = {
    "Argentina", "Brazil", "Chile", "Colombia", "Venezuela", "Ecuador", "Peru",
    "Uruguay", "Paraguay", "Bolivia", "Guyana", "Suriname", "French Guiana",
}
MCAC_COUNTRIES = {
    "Mexico", "Puerto Rico", "Dominican Republic", "Costa Rica", "Panama", "Guatemala",
    "Bermuda", "Curacao", "Dominica", "Saint Kitts and Nevis", "Turks and Caicos Islands",
    "El Salvador", "Honduras", "Nicaragua", "Barbados", "Trinidad and Tobago",
    "Cayman Islands", "Jamaica", "Guadeloupe", "Martinique", "Saint Lucia", "Belize",
    "Bahamas", "Aruba", "Cuba", "Haiti", "Antigua and Barbuda", "Grenada",
    "Saint Vincent and the Grenadines", "British Virgin Islands", "U.S. Virgin Islands",
}
APAC_COUNTRIES = {
    "Australia", "New Zealand", "Thailand", "Taiwan", "Malaysia", "Singapore", "Japan",
    "Philippines", "Indonesia", "India", "China", "South Korea", "Hong Kong", "Vietnam",
    "Northern Mariana Islands", "New Caledonia", "Vanuatu", "Kiribati", "Guam", "Fiji",
    "Papua New Guinea", "Cambodia", "Sri Lanka", "Pakistan", "Bangladesh", "Nepal", "Macau",
}

# 12 Canonical Anatomical Zones in 224x224 ArcFace/Delaunay space
ANATOMICAL_ZONES = [
    ("eyes_iris", "Eyes, iris & palpebral aperture"),
    ("brows", "Eyebrow arch & spacing"),
    ("tear_trough", "Infraorbital / tear trough"),
    ("lips_vermilion", "Vermilion lips & smile contour"),
    ("nose_alar_tip", "Nasal tip & alar width"),
    ("nose_bridge", "Nasal dorsum / bridge"),
    ("malar_cheek", "Malar cheekbone prominence"),
    ("nasolabial", "Nasolabial & perioral fold"),
    ("jaw_chin", "Jawline taper & chin"),
    ("philtrum", "Philtrum & upper perioral"),
    ("forehead_glabella", "Forehead & glabella"),
    ("lateral_skin", "Lateral cheek skin & tone"),
]


def classify_region(country: str | None) -> str | None:
    if not country:
        return None
    if country == "United States":
        return "United States"
    if country == "Canada":
        return "Canada"
    if country in SAM_COUNTRIES:
        return "South America"
    if country in MCAC_COUNTRIES:
        return "Mexico, Central Am. & Caribbean"
    if country in EUR_COUNTRIES:
        return "Europe"
    if country in APAC_COUNTRIES:
        return "Asia-Pacific & Oceania"
    return "Middle East & Africa"


def classify_age_bucket(age: int) -> str:
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    return "55+"


def arcface_template(size: int = 224) -> np.ndarray:
    base = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32,
    )
    return base * (float(size) / 112.0)


def portrait_template(size: int = 384) -> tuple[np.ndarray, float, float, float]:
    """Padded portrait coordinate system mapping 224-space [-10..234, -2..242] -> [0..size, 0..size]
    so forehead, cheeks, chin (y=219), and submental contour all sit cleanly inside the frame."""
    x0, y0, span = -10.0, -2.0, 244.0
    scale = float(size) / span
    t224 = arcface_template(224)
    out = np.empty_like(t224)
    out[:, 0] = (t224[:, 0] - x0) * scale
    out[:, 1] = (t224[:, 1] - y0) * scale
    return out, x0, y0, scale


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src64 = np.asarray(src, dtype=np.float64)
    dst64 = np.asarray(dst, dtype=np.float64)
    sx = src64.mean(axis=0)
    tx = dst64.mean(axis=0)
    a = src64 - sx
    b = dst64 - tx
    u, s, vt = np.linalg.svd((b.T @ a) / len(a))
    det_sign = 1.0 if np.linalg.det(u @ vt) > 0 else -1.0
    sign = np.array([1.0, det_sign], dtype=np.float64)
    rot = u @ np.diag(sign) @ vt
    var = max(float(np.sum(a * a) / len(a)), 1e-8)
    scale = float(np.sum(s * sign) / var)
    m = scale * rot
    t = tx - m @ sx
    return np.column_stack([m, t]).astype(np.float32)


def neutralize_uv_matte(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    non_matte = np.max(np.abs(rgb.astype(np.int16) - MATTE_RGB), axis=2) > 3
    clean_mask = cv2.erode(non_matte.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)).astype(bool)
    skin_fill = np.median(rgb[clean_mask], axis=0).astype(np.uint8) if clean_mask.any() else np.array([128, 128, 128], dtype=np.uint8)
    out = rgb.copy()
    out[~clean_mask] = skin_fill
    return out, clean_mask


def detect_5pt(clean_rgb: np.ndarray, yunet: Any) -> np.ndarray:
    h, w = clean_rgb.shape[:2]
    sx, sy = w / 1200.0, h / 1200.0
    canon = CANONICAL_UV_5PT.copy()
    canon[:, 0] *= sx
    canon[:, 1] *= sy
    if yunet is None:
        return canon
    small_bgr = cv2.resize(cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR), (320, 320), interpolation=cv2.INTER_AREA)
    yunet.setInputSize((320, 320))
    _, det = yunet.detect(small_bgr)
    if det is None or len(det) == 0:
        return canon
    best = det[int(np.argmax(det[:, -1]))]
    lm = best[4:14].reshape(5, 2).astype(np.float32)
    lm[:, 0] *= w / 320.0
    lm[:, 1] *= h / 320.0
    lmn = lm.copy()
    lmn[:, 0] /= sx
    lmn[:, 1] /= sy
    sane = (
        220.0 <= float(lmn[1, 0] - lmn[0, 0]) <= 580.0
        and abs(float(lmn[1, 1] - lmn[0, 1])) <= 140.0
        and 140.0 <= float(lmn[4, 0] - lmn[3, 0]) <= 540.0
        and lmn[0, 1] < lmn[2, 1] < lmn[3, 1]
        and float(np.sqrt(np.mean(np.sum((lmn - CANONICAL_UV_5PT) ** 2, axis=1)))) <= 140.0
    )
    return (0.65 * lm + 0.35 * canon) if sane else canon


def build_delaunay_lut(canon_verts: np.ndarray, size: int) -> dict[str, np.ndarray]:
    """Build deterministic 106-landmark + 16-border-pin Delaunay triangulation and per-pixel barycentric lookup."""
    s = float(size - 1)
    q1, q2, q3 = 0.25 * s, 0.50 * s, 0.75 * s
    pins = np.array(
        [
            [0.0, 0.0], [q1, 0.0], [q2, 0.0], [q3, 0.0], [s, 0.0],
            [s, q1], [s, q2], [s, q3], [s, s],
            [q3, s], [q2, s], [q1, s], [0.0, s],
            [0.0, q3], [0.0, q2], [0.0, q1],
        ],
        dtype=np.float32,
    )
    verts = np.vstack([np.clip(canon_verts, 2.0, s - 2.0), pins])
    subdiv = cv2.Subdiv2D((0, 0, size, size))
    for pt in verts:
        subdiv.insert((float(pt[0]), float(pt[1])))
    tris: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for t in subdiv.getTriangleList():
        tri_xy = np.asarray(t, dtype=np.float32).reshape(3, 2)
        if np.any(tri_xy < -0.5) or np.any(tri_xy > size + 0.5):
            continue
        idxs = [int(np.argmin(np.sum((verts - v) ** 2, axis=1))) for v in tri_xy]
        if len(set(idxs)) < 3:
            continue
        if any(float(np.linalg.norm(verts[idx] - v)) > 0.5 for idx, v in zip(idxs, tri_xy)):
            continue
        a, b, c = verts[idxs[0]], verts[idxs[1]], verts[idxs[2]]
        area2 = float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if abs(area2) < 1.0:
            continue
        if area2 < 0:
            idxs = [idxs[0], idxs[2], idxs[1]]
        key = tuple(sorted(idxs))
        if key not in seen:
            seen.add(key)
            tris.append((idxs[0], idxs[1], idxs[2]))
    tris_arr = np.array(tris, dtype=np.int32)
    tri_id = np.full((size, size), -1, dtype=np.int32)
    for tid, (i0, i1, i2) in enumerate(tris_arr):
        cv2.fillConvexPoly(tri_id, np.round(verts[[i0, i1, i2]]).astype(np.int32), int(tid))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    valid = tri_id >= 0
    tids = np.maximum(tri_id, 0)
    v0 = verts[tris_arr[tids, 0]]
    v1 = verts[tris_arr[tids, 1]]
    v2 = verts[tris_arr[tids, 2]]
    d00 = v1[..., 0] - v0[..., 0]
    d01 = v2[..., 0] - v0[..., 0]
    d10 = v1[..., 1] - v0[..., 1]
    d11 = v2[..., 1] - v0[..., 1]
    denom = d00 * d11 - d01 * d10
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    px = xx - v0[..., 0]
    py = yy - v0[..., 1]
    w1 = (px * d11 - d01 * py) / denom
    w2 = (d00 * py - px * d10) / denom
    w0 = 1.0 - w1 - w2
    return {
        "size": size,
        "pins": pins,
        "verts": verts,
        "tris": tris_arr,
        "tids": tids,
        "valid": valid,
        "w0": w0.astype(np.float32),
        "w1": w1.astype(np.float32),
        "w2": w2.astype(np.float32),
        "xx": xx,
        "yy": yy,
        "interior": np.all(tris_arr < 106, axis=1),
        "centroids": (verts[tris_arr[:, 0]] + verts[tris_arr[:, 1]] + verts[tris_arr[:, 2]]) / 3.0,
    }


def warp_with_lut(src_img: np.ndarray, src_lm106: np.ndarray, lut: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    s = float(lut["size"] - 1)
    src_verts = np.vstack([np.clip(src_lm106, 1.0, s - 1.0), lut["pins"]])
    tris = lut["tris"]
    tids = lut["tids"]
    sv0, sv1, sv2 = src_verts[tris[:, 0]], src_verts[tris[:, 1]], src_verts[tris[:, 2]]
    mx = np.where(lut["valid"], lut["w0"] * sv0[tids, 0] + lut["w1"] * sv1[tids, 0] + lut["w2"] * sv2[tids, 0], lut["xx"]).astype(np.float32)
    my = np.where(lut["valid"], lut["w0"] * sv0[tids, 1] + lut["w1"] * sv1[tids, 1] + lut["w2"] * sv2[tids, 1], lut["yy"]).astype(np.float32)
    warped = cv2.remap(src_img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    cv0, cv1, cv2_v = lut["verts"][tris[:, 0]].astype(np.float64), lut["verts"][tris[:, 1]].astype(np.float64), lut["verts"][tris[:, 2]].astype(np.float64)
    dc1, dc2 = cv1 - cv0, cv2_v - cv0
    ds1, ds2 = (sv1 - sv0).astype(np.float64), (sv2 - sv0).astype(np.float64)
    det_c = dc1[:, 0] * dc2[:, 1] - dc1[:, 1] * dc2[:, 0]
    det_s = ds1[:, 0] * ds2[:, 1] - ds1[:, 1] * ds2[:, 0]
    log_det_j = np.clip(np.log(np.maximum(det_s, 0.25) / np.maximum(np.abs(det_c), 0.25)), -2.2, 2.2).astype(np.float32)
    return warped, log_det_j


def build_zone_masks_224(canon106: np.ndarray, grid_n: int = 224) -> dict[str, np.ndarray]:
    """Construct smooth, mutually normalized anatomical zone weights in 224x224 canonical space."""
    scale = grid_n / 224.0
    yy, xx = np.mgrid[0:grid_n, 0:grid_n].astype(np.float32) / scale

    def gauss_pts(pts: np.ndarray, sx: float, sy: float) -> np.ndarray:
        acc = np.zeros((grid_n, grid_n), dtype=np.float32)
        for px, py in pts:
            acc = np.maximum(acc, np.exp(-0.5 * (((xx - px) / sx) ** 2 + ((yy - py) / sy) ** 2)))
        return acc

    eyes = gauss_pts(canon106[list(range(33, 43)) + list(range(87, 97))], 8.5, 5.5)
    brows = gauss_pts(canon106[list(range(43, 52)) + list(range(97, 106))], 9.0, 5.2)
    tear_pts = canon106[[35, 36, 37, 39, 89, 90, 91, 93]].copy()
    tear_pts[:, 1] += 11.5
    tear = gauss_pts(tear_pts, 10.0, 5.8) * (1.0 - 0.65 * eyes)
    lips = gauss_pts(canon106[list(range(52, 72))], 9.0, 6.2)
    nose_bridge = gauss_pts(canon106[[72, 73, 74, 75, 81]], 8.0, 8.5)
    nose_alar = gauss_pts(canon106[[76, 77, 78, 79, 80, 82, 83, 84, 85, 86]], 8.5, 6.8)
    malar_pts = np.array([[62.0, 134.0], [74.0, 138.0], [150.0, 138.0], [162.0, 134.0]], dtype=np.float32)
    malar = gauss_pts(malar_pts, 13.5, 11.0)
    nlf_pts = np.array([[84.0, 156.0], [80.0, 170.0], [140.0, 156.0], [144.0, 170.0]], dtype=np.float32)
    nasolabial = gauss_pts(nlf_pts, 8.5, 9.5) * (1.0 - 0.5 * lips)
    philtrum = gauss_pts(np.array([[113.0, 168.0]], dtype=np.float32), 9.5, 5.5) * (1.0 - 0.5 * lips)
    jaw_pts = canon106[[0, 3, 4, 5, 6, 7, 8, 19, 20, 21, 22, 23, 24]]
    jaw = gauss_pts(jaw_pts, 11.0, 8.5) * (1.0 - 0.5 * lips)
    forehead = np.exp(-0.5 * (((xx - 112.0) / 46.0) ** 2 + ((yy - 52.0) / 19.0) ** 2)) * (1.0 - 0.6 * brows)
    oval = (((xx - 112.0) / 76.0) ** 2 + ((yy - 126.0) / 94.0) ** 2) <= 1.05
    raw = {
        "eyes_iris": eyes,
        "brows": brows,
        "tear_trough": tear,
        "lips_vermilion": lips,
        "nose_alar_tip": nose_alar,
        "nose_bridge": nose_bridge,
        "malar_cheek": malar,
        "nasolabial": nasolabial,
        "jaw_chin": jaw,
        "philtrum": philtrum,
        "forehead_glabella": forehead,
    }
    used = np.clip(sum(raw.values()), 0.0, 1.5)
    raw["lateral_skin"] = np.clip(oval.astype(np.float32) - 0.65 * used, 0.0, 1.0)
    tot = sum(raw.values()) + 1e-6
    return {k: (v / tot) * oval.astype(np.float32) for k, v in raw.items()}


def fit_ridge(x: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    w = w / np.maximum(w.mean(), 1e-8)
    mx = (w[:, None] * x).sum(axis=0) / w.sum()
    my = float((w * y).sum() / w.sum())
    xc = x - mx
    yc = y - my
    sw = np.sqrt(w)[:, None]
    xw = xc * sw
    yw = yc * np.sqrt(w)
    d = x.shape[1]
    a = xw.T @ xw + alpha * np.eye(d, dtype=np.float64)
    b = xw.T @ yw
    beta = np.linalg.solve(a, b).astype(np.float32)
    intercept = float(my - mx @ beta)
    return beta, intercept


def l2_rows(arr: np.ndarray) -> np.ndarray:
    arr32 = np.asarray(arr, dtype=np.float32)
    return arr32 / np.maximum(np.linalg.norm(arr32, axis=1, keepdims=True), 1e-8)


def evaluate_10k_pure_face_benchmark() -> dict[str, Any]:
    """Fit & audit the H25+H34+H20 Pure-Face Multi-Backbone Percentile Ensemble on all 10,000 profiles (20,000 images)."""
    manifest = json.loads((MODEL_ROOT / "manifest.json").read_text())
    profs = manifest["profiles"]
    imgs = manifest["images"]
    n_p, n_i = len(profs), len(imgs)
    arr_dir = MODEL_ROOT / "arrays"

    def mm(name: str, dim: int, dt=np.float16) -> np.ndarray:
        return np.fromfile(arr_dir / name, dtype=dt).reshape(n_i, dim).astype(np.float32)

    b_dino = l2_rows(
        1.15 * l2_rows(mm("h11_dinov3_vitb_align.fp16", 768))
        + 0.95 * l2_rows(mm("h11_dinov3_vitb_flip.fp16", 768))
        + 0.90 * l2_rows(mm("h11_dinov3_vitb_tzone.fp16", 768))
    )
    b_zones = l2_rows(mm("h11_dinov3_vitb_zones.fp16", 3072))
    b_cnv2 = l2_rows(mm("h12_convnextv2_align.fp16", 768))
    b_glint = l2_rows(
        1.10 * l2_rows(mm("h12_glintr100_align.fp16", 512))
        + 0.90 * l2_rows(mm("h12_glintr100_flip.fp16", 512))
    )
    b_del = l2_rows(
        np.concatenate(
            [
                l2_rows(mm("h25_delaunay_glint.fp16", 512)),
                0.85 * l2_rows(mm("h25_delaunay_dinos.fp16", 384)),
            ],
            axis=1,
        )
    )
    b_comp = l2_rows(
        np.concatenate(
            [
                l2_rows(mm("h28_periocular_dinos.fp16", 384)),
                l2_rows(mm("h28_orolabial_dinos.fp16", 384)),
            ],
            axis=1,
        )
    )
    f_del = np.nan_to_num(mm("h25_delaunay_feats.fp32", 48, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    f_comp = np.nan_to_num(mm("h28_comp_feats.fp32", 40, np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    x_img = np.concatenate([b_dino, 0.55 * b_zones[:, :1536], b_cnv2, b_glint, b_del, b_comp], axis=1)
    # Pool primary + secondary views to profile level
    x_prof = np.zeros((n_p, x_img.shape[1]), dtype=np.float32)
    f_prof = np.zeros((n_p, 88), dtype=np.float32)
    cnt = np.zeros(n_p, dtype=np.float32)
    for im in imgs:
        ii, pi = im["image_idx"], im["profile_idx"]
        wt = 1.15 if im["is_primary"] else 0.85
        x_prof[pi] += wt * x_img[ii]
        f_prof[pi] += wt * np.concatenate([f_del[ii], f_comp[ii]])
        cnt[pi] += wt
    x_prof /= np.maximum(cnt[:, None], 1e-6)
    f_prof /= np.maximum(cnt[:, None], 1e-6)

    f_tr_mask = np.array([p["split"] == "train" for p in profs])
    f_mean = f_prof[f_tr_mask].mean(axis=0)
    f_std = np.maximum(f_prof[f_tr_mask].std(axis=0), 1e-3)
    f_norm = np.clip((f_prof - f_mean) / f_std, -5.0, 5.0) * 0.06
    x_all = np.concatenate([x_prof, f_norm], axis=1).astype(np.float64)

    tr_idx = np.array([i for i, p in enumerate(profs) if p["split"] == "train"])
    va_idx = np.array([i for i, p in enumerate(profs) if p["split"] == "validation"])
    te_idx = np.array([i for i, p in enumerate(profs) if p["split"] == "test"])

    y_beauty = np.array([float(p["beauty_score"]) for p in profs], dtype=np.float64)
    y_pct = np.array([float(p["beauty_gender_percentile"]) for p in profs], dtype=np.float64)
    votes = np.array([float(p["votes"]) for p in profs], dtype=np.float64)
    likes = np.array([float(p["likes"]) for p in profs], dtype=np.float64)
    genders = [p["gender"] for p in profs]

    # H34 Empirical-Bayes Beta-Binomial vote-precision latent target (strictly fitted on train)
    eb_w = votes / (votes + 42.0)
    prior_rate = {
        g: float(likes[[i for i in tr_idx if genders[i] == g]].sum() / votes[[i for i in tr_idx if genders[i] == g]].sum())
        for g in ("Female", "Male")
    }
    theta = np.array(
        [
            logit(np.clip((likes[i] + 42.0 * prior_rate[genders[i]]) / (votes[i] + 42.0), 1e-3, 1.0 - 1e-3))
            for i in range(n_p)
        ],
        dtype=np.float64,
    )

    # Precompute weighted Gram matrix once so multi-alpha tuning takes <2s instead of 150s
    w_tr = eb_w[tr_idx] / np.maximum(eb_w[tr_idx].mean(), 1e-8)
    mx_tr = (w_tr[:, None] * x_all[tr_idx]).sum(axis=0) / w_tr.sum()
    my_tr = float((w_tr * theta[tr_idx]).sum() / w_tr.sum())
    xc_tr = x_all[tr_idx] - mx_tr
    yc_tr = theta[tr_idx] - my_tr
    xw_tr = xc_tr * np.sqrt(w_tr)[:, None]
    yw_tr = yc_tr * np.sqrt(w_tr)
    gram_tr = xw_tr.T @ xw_tr
    rhs_tr = xw_tr.T @ yw_tr
    d_tr = x_all.shape[1]

    best_alpha, best_val, best_beta, best_b0 = 80.0, 1e9, None, 0.0
    for alpha in (25.0, 60.0, 120.0, 240.0):
        beta_cand = np.linalg.solve(gram_tr + alpha * np.eye(d_tr, dtype=np.float64), rhs_tr).astype(np.float32)
        b0_cand = float(my_tr - mx_tr @ beta_cand)
        pred_va = x_all[va_idx] @ beta_cand + b0_cand
        mae_va = float(np.mean(np.abs(pred_va - theta[va_idx])))
        if mae_va < best_val:
            best_val, best_alpha, best_beta, best_b0 = mae_va, alpha, beta_cand, b0_cand

    assert best_beta is not None
    raw_pred = x_all @ best_beta + best_b0

    # Map latent score -> within-gender empirical CDF percentile strictly from train split
    pred_pct = np.zeros(n_p, dtype=np.float64)
    pred_beauty = np.zeros(n_p, dtype=np.float64)
    for g in ("Female", "Male"):
        gtr = np.array([i for i in tr_idx if genders[i] == g])
        gall = np.array([i for i in range(n_p) if genders[i] == g])
        order = np.argsort(raw_pred[gtr])
        xs = raw_pred[gtr][order]
        ps = rankdata(y_pct[gtr][order], method="average") / len(gtr) * 100.0
        # Monotone binned mapping
        q_grid = np.linspace(0.0, 100.0, 101)
        x_q = np.percentile(xs, q_grid)
        y_q = np.percentile(y_pct[gtr], q_grid)
        b_q = np.percentile(y_beauty[gtr], q_grid)
        pred_pct[gall] = np.interp(raw_pred[gall], x_q, y_q)
        pred_beauty[gall] = np.interp(raw_pred[gall], x_q, b_q)

    def metrics_on(idxs: np.ndarray) -> dict[str, float]:
        rho = float(spearmanr(pred_beauty[idxs], y_beauty[idxs]).statistic)
        rho_pct = float(spearmanr(pred_pct[idxs], y_pct[idxs]).statistic)
        return {
            "n": int(len(idxs)),
            "beauty_mae": round(float(np.mean(np.abs(pred_beauty[idxs] - y_beauty[idxs]))), 4),
            "spearman_r": round(rho, 4),
            "explained_rank_var": round(rho * rho, 4),
            "gender_pct_mae": round(float(np.mean(np.abs(pred_pct[idxs] - y_pct[idxs]))), 2),
            "gender_pct_spearman_r": round(rho_pct, 4),
        }

    by_sex_test = {}
    for g in ("Female", "Male"):
        gte = np.array([i for i in te_idx if genders[i] == g])
        by_sex_test[g] = metrics_on(gte)

    return {
        "model_name": "Pure-Face Multi-Backbone Spatial & Delaunay Ensemble (DINOv3-ViT-B/16 + ConvNeXtV2 + AuraFace-GlintR100 + 106-pt Delaunay + H34 EB-BetaBinomial)",
        "selected_ridge_alpha": best_alpha,
        "train_profiles": int(len(tr_idx)),
        "validation_profiles": int(len(va_idx)),
        "test_overall": metrics_on(te_idx),
        "test_by_gender": by_sex_test,
    }


def select_stratified_cohorts(seed: int = 20261111) -> list[dict[str, Any]]:
    """Query manifest.json + xbabe3 hon.sqlite3 (read-only) to select ~4,800 stratified primary portraits."""
    manifest = json.loads((MODEL_ROOT / "manifest.json").read_text())
    profs = manifest["profiles"]
    prim_by_pid: dict[str, str] = {}
    for im in manifest["images"]:
        if im["is_primary"]:
            pid = im["profile_id"]
            prim_by_pid[pid] = im["path"]

    rng = np.random.default_rng(seed)
    selected: dict[str, dict[str, Any]] = {}

    # 1. Include ALL test primary portraits (~956) for leak-free held-out verification
    #    and stratified train portraits (1,400: 700 Female, 700 Male across deciles & ages)
    for g in ("Female", "Male"):
        te_cands = [p for p in profs if p["split"] == "test" and p["gender"] == g and p["profile_id"] in prim_by_pid and 18 <= int(p["age"]) <= 79]
        for p in te_cands:
            pid = p["profile_id"]
            selected[pid] = {
                "profile_id": pid,
                "path": prim_by_pid[pid],
                "gender": g,
                "age": int(p["age"]),
                "age_bucket": classify_age_bucket(int(p["age"])),
                "country": p["country"] or "United States",
                "region": classify_region(p["country"] or "United States") or "United States",
                "votes": int(p["votes"]),
                "likes": int(p["likes"]),
                "beauty_score": float(p["beauty_score"]),
                "beauty_pct": float(p["beauty_gender_percentile"]),
                "split": "test",
            }

        tr_cands = [p for p in profs if p["split"] == "train" and p["gender"] == g and p["profile_id"] in prim_by_pid and 18 <= int(p["age"]) <= 79]
        # Stratify across 10 beauty percentile deciles (75 per decile = 750 per sex)
        tr_cands.sort(key=lambda r: float(r["beauty_gender_percentile"]))
        bins = np.array_split(np.arange(len(tr_cands)), 10)
        for b in bins:
            pick = rng.choice(b, size=min(75, len(b)), replace=False)
            for idx in pick:
                p = tr_cands[int(idx)]
                pid = p["profile_id"]
                selected[pid] = {
                    "profile_id": pid,
                    "path": prim_by_pid[pid],
                    "gender": g,
                    "age": int(p["age"]),
                    "age_bucket": classify_age_bucket(int(p["age"])),
                    "country": p["country"] or "United States",
                    "region": classify_region(p["country"] or "United States") or "United States",
                    "votes": int(p["votes"]),
                    "likes": int(p["likes"]),
                    "beauty_score": float(p["beauty_score"]),
                    "beauty_pct": float(p["beauty_gender_percentile"]),
                    "split": "train",
                }

    # 2. Query xbabe3 hon.sqlite3 (read-only) to compute exact unified saved-image within-gender beauty percentiles
    #    for ALL selected profiles (train, test, eval_world, eval_age) so every cohort shares the identical [0, 100] CDF!
    train_pids = {pid for pid, r in selected.items() if r["split"] == "train"}
    existing_pids_json = json.dumps(sorted(selected.keys()))
    remote_py = r"""
import json, sqlite3, sys, numpy as np
from scipy.stats import rankdata

existing_pids = set(json.loads(sys.stdin.read()))
con = sqlite3.connect("file:/home/ubuntu/HON_INDEX/hon.sqlite3?mode=ro", uri=True)
rows = con.execute('''
    SELECT p.profile_id, p.gender, p.age, p.country, l.votes, l.likes, l.beauty_score
    FROM profiles p
    JOIN labels l ON l.profile_id = p.profile_id
    WHERE p.image_count > 0 AND p.age BETWEEN 18 AND 79
      AND p.gender IN ('Female', 'Male')
      AND l.votes >= 20 AND l.likes BETWEEN 0 AND l.votes
      AND p.metadata_conflict = 0 AND p.label_conflict = 0
''').fetchall()

by_g = {"Female": [], "Male": []}
for r in rows:
    by_g[r[1]].append(r)

pct_map = {}
for g, glist in by_g.items():
    bs = np.array([float(x[6]) for x in glist], dtype=np.float64)
    pcts = (rankdata(bs, method="average") - 0.5) / len(bs) * 100.0
    for x, pct in zip(glist, pcts):
        pct_map[x[0]] = float(pct)

existing_pct = {pid: pct_map[pid] for pid in existing_pids if pid in pct_map}

# Select candidate pools: non-US with country, plus US ages 35+ and young supplement
cands = []
rng = np.random.default_rng(20261111)
us_by_ga = {}
for r in rows:
    pid, g, age, country, votes, likes, bscore = r
    if country and country != "United States":
        cands.append((pid, g, int(age), country, int(votes), int(likes), float(bscore), pct_map[pid]))
    elif country == "United States":
        ab = "18-24" if age<=24 else ("25-34" if age<=34 else ("35-44" if age<=44 else ("45-54" if age<=54 else "55+")))
        us_by_ga.setdefault((g, ab), []).append((pid, g, int(age), country, int(votes), int(likes), float(bscore), pct_map[pid]))

for k, lst in us_by_ga.items():
    lst.sort(key=lambda x: x[0])
    pick_n = 280 if k[1] in ("35-44", "45-54", "55+") else 120
    idxs = rng.choice(len(lst), size=min(pick_n, len(lst)), replace=False)
    for i in idxs:
        cands.append(lst[int(i)])

# Resolve primary image path in one indexed query via batched IN
pid_set = sorted({c[0] for c in cands})
path_map = {}
for i in range(0, len(pid_set), 900):
    chunk = pid_set[i:i+900]
    q = "SELECT profile_id, MIN(path) FROM files WHERE kind='image' AND profile_id IN (" + ",".join("?"*len(chunk)) + ") GROUP BY profile_id"
    for pid, pth in con.execute(q, chunk):
        path_map[pid] = pth

out = [ (*c, path_map[c[0]]) for c in cands if c[0] in path_map ]
print(json.dumps({"existing_pct": existing_pct, "ext_rows": out}))
"""
    import base64 as _b64
    b64_py = _b64.b64encode(remote_py.encode("utf-8")).decode("ascii")
    raw_out = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "xbabe3", f"python3 -c 'import base64;exec(base64.b64decode(\"{b64_py}\").decode())'"],
        input=existing_pids_json.encode("utf-8"),
    )
    payload_remote = json.loads(raw_out)
    existing_pct = payload_remote["existing_pct"]
    ext_rows = payload_remote["ext_rows"]
    for pid, rec in list(selected.items()):
        if pid in existing_pct:
            rec["beauty_pct_full_archive"] = rec["beauty_pct"]
            rec["beauty_pct"] = float(existing_pct[pid])

    # Organize non-US by (gender, region) and US age supplements
    by_gr: dict[tuple[str, str], list[tuple]] = {}
    by_us_age: dict[tuple[str, str], list[tuple]] = {}
    for row in ext_rows:
        pid, g, age, country, votes, likes, bscore, bpct, rel_path = row
        if pid in selected or pid in train_pids:
            continue
        reg = classify_region(country)
        if not reg:
            continue
        if reg != "United States":
            by_gr.setdefault((g, reg), []).append(row)
        else:
            by_us_age.setdefault((g, classify_age_bucket(age)), []).append(row)

    # Cap up to 260 per (gender, non-US region), deterministic sample
    for (g, reg), lst in sorted(by_gr.items()):
        lst.sort(key=lambda x: x[0])
        if len(lst) > 260:
            idxs = sorted(rng.choice(len(lst), size=260, replace=False).tolist())
            lst = [lst[i] for i in idxs]
        for pid, gg, age, country, votes, likes, bscore, bpct, rel_path in lst:
            selected[pid] = {
                "profile_id": pid,
                "path": rel_path,
                "gender": gg,
                "age": age,
                "age_bucket": classify_age_bucket(age),
                "country": country,
                "region": reg,
                "votes": votes,
                "likes": likes,
                "beauty_score": bscore,
                "beauty_pct": bpct,
                "split": "eval_world",
            }

    # Ensure every (gender, age_bucket) non-train pool has at least 220 profiles
    for g in ("Female", "Male"):
        for ab in AGE_BUCKETS:
            cur_n = sum(1 for r in selected.values() if r["split"] != "train" and r["gender"] == g and r["age_bucket"] == ab)
            need = max(0, 220 - cur_n)
            if need > 0:
                pool = by_us_age.get((g, ab), [])
                pool.sort(key=lambda x: x[0])
                for row in pool[:need]:
                    pid, gg, age, country, votes, likes, bscore, bpct, rel_path = row
                    selected[pid] = {
                        "profile_id": pid,
                        "path": rel_path,
                        "gender": gg,
                        "age": age,
                        "age_bucket": ab,
                        "country": country,
                        "region": "United States",
                        "votes": votes,
                        "likes": likes,
                        "beauty_score": bscore,
                        "beauty_pct": bpct,
                        "split": "eval_age",
                    }

    records = sorted(selected.values(), key=lambda r: r["path"])
    return records


def run_streaming_alignment_and_spatial_features(
    records: list[dict[str, Any]],
    portrait_size: int = 384,
    workers: int = 12,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Stream selected 1200x1200 PNGs sequentially from xbabe3 and run 106-pt Delaunay super-res
    warping + DINOv3 LVD-1689M spatial tokens + SFace + strictly local 12-zone descriptors on CPU."""
    canon106_224 = np.load(MODEL_ROOT / "canon_lm106_224.npy").astype(np.float32)
    lut_224 = build_delaunay_lut(canon106_224, 224)

    pt_tmpl, px0, py0, pscale = portrait_template(portrait_size)
    canon106_port = np.empty_like(canon106_224)
    canon106_port[:, 0] = (canon106_224[:, 0] - px0) * pscale
    canon106_port[:, 1] = (canon106_224[:, 1] - py0) * pscale
    lut_port = build_delaunay_lut(canon106_port, portrait_size)

    cache_npz = MODEL_ROOT / "live_5353_cache.npz"
    cache_meta = MODEL_ROOT / "live_5353_metas.json"
    if use_cache and cache_npz.exists() and cache_meta.exists():
        cmeta = json.loads(cache_meta.read_text())
        if len(cmeta.get("metas", [])) >= 4500:
            print(f"[face-attr] Loading cached live features for {len(cmeta['metas']):,} aligned portraits from {cache_npz}...", flush=True)
            cd = np.load(cache_npz)
            composites_u8 = {k[len("comp__"):]: cd[k] for k in cd.files if k.startswith("comp__")}
            sample_224_by_tier = {k[len("tier224__"):]: list(cd[k]) for k in cd.files if k.startswith("tier224__")}
            return {
                "metas": cmeta["metas"],
                "patches": cd["patches"],
                "pools": cd["pools"],
                "sface": cd["sface"],
                "genderage": cd["genderage"],
                "local_zone_feats": cd["local_zone_feats"],
                "log_det_j": cd["log_det_j"],
                "composites_u8": composites_u8,
                "silhouette_u8": cd["silhouette_u8"],
                "comp_counts": cmeta["comp_counts"],
                "sample_224_by_tier": sample_224_by_tier,
                "lut_224": lut_224,
                "lut_port": lut_port,
                "canon106_224": canon106_224,
                "canon106_port": canon106_port,
                "portrait_frame": {"size": portrait_size, "x0": px0, "y0": py0, "scale": pscale},
            }

    zone_masks_224 = build_zone_masks_224(canon106_224, 224)
    z_norm_224 = {zk: zone_masks_224[zk] / max(float(zone_masks_224[zk].sum()), 1e-6) for zk, _ in ANATOMICAL_ZONES}
    tids_224 = lut_224["tids"]
    valid_224 = lut_224["valid"]

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.log_severity_level = 3

    det106_sess = ort.InferenceSession(
        str(MODEL_ROOT / "weights/auraface/2d106det.onnx"), sess_options=so, providers=["CPUExecutionProvider"]
    )
    dinos_sess = ort.InferenceSession(
        str(MODEL_ROOT / "weights/dinov3_vits16/model.onnx"), sess_options=so, providers=["CPUExecutionProvider"]
    )
    sface_sess = ort.InferenceSession(
        str(MODEL_ROOT / "weights/sface.onnx"), sess_options=so, providers=["CPUExecutionProvider"]
    )
    sface_in_name = sface_sess.get_inputs()[0].name
    ga_sess = ort.InferenceSession(
        str(MODEL_ROOT / "weights/auraface/genderage.onnx"), sess_options=so, providers=["CPUExecutionProvider"]
    )
    yunet_path = str(MODEL_ROOT / "weights/yunet.onnx")

    import threading
    tls = threading.local()

    def get_yunet():
        if not hasattr(tls, "yunet"):
            tls.yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.30, 0.3, 5000)
        return tls.yunet

    def process_one(item: tuple[int, dict[str, Any], bytes]) -> dict[str, Any] | None:
        idx, meta, img_bytes = item
        try:
            with Image.open(io.BytesIO(img_bytes)) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        except Exception:
            return None
        clean_rgb, clean_mask = neutralize_uv_matte(rgb)
        pts5 = detect_5pt(clean_rgb, get_yunet())

        # 1. 224x224 similarity crop -> 106-landmark detection -> 224x224 Delaunay atlas
        mat224 = umeyama_similarity(pts5, arcface_template(224))
        align_224 = cv2.warpAffine(clean_rgb, mat224, (224, 224), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        in192 = np.transpose(cv2.resize(align_224, (192, 192), interpolation=cv2.INTER_AREA).astype(np.float32), (2, 0, 1))[None]
        out106 = det106_sess.run(None, {"data": in192})[0][0, :212].reshape(106, 2).astype(np.float32)
        lm106_224 = (out106 + 1.0) * 112.0

        # Regularize extreme detector outliers gently toward canonical UV topology
        rms_dev = float(np.sqrt(np.mean(np.sum((lm106_224 - canon106_224) ** 2, axis=1))))
        if rms_dev > 28.0:
            lm106_224 = 0.55 * lm106_224 + 0.45 * canon106_224

        atlas_224, log_det_j = warp_with_lut(align_224, lm106_224, lut_224)

        # 2. Native 1200x1200 -> 384x384 high-resolution padded portrait Delaunay warp (never downsampled through 224!)
        mat_port = umeyama_similarity(pts5, pt_tmpl)
        align_port = cv2.warpAffine(
            clean_rgb, mat_port, (portrait_size, portrait_size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        mask_port_aff = cv2.warpAffine(
            clean_mask.astype(np.float32), mat_port, (portrait_size, portrait_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        lm106_port = np.empty_like(lm106_224)
        lm106_port[:, 0] = (lm106_224[:, 0] - px0) * pscale
        lm106_port[:, 1] = (lm106_224[:, 1] - py0) * pscale
        atlas_port, _ = warp_with_lut(align_port, lm106_port, lut_port)
        mask_port, _ = warp_with_lut(mask_port_aff, lm106_port, lut_port)
        mask_port = np.clip(mask_port, 0.0, 1.0)

        # 3. DINOv3 LVD-1689M 14x14 spatial tokens + pooler + SFace 128-d + AuraFace genderage 3-d
        chw = (np.transpose(atlas_224.astype(np.float32) / 255.0, (2, 0, 1)) - IMAGENET_MEAN) / IMAGENET_STD
        seq, pool = dinos_sess.run(None, {"pixel_values": chw[None]})
        patches = seq[0, 5:, :].reshape(14, 14, 384).astype(np.float16)

        bgr112 = cv2.cvtColor(cv2.resize(align_224, (112, 112), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        sf_vec = sface_sess.run(None, {sface_in_name: np.transpose(bgr112.astype(np.float32), (2, 0, 1))[None]})[0][0].astype(np.float16)
        bgr96 = cv2.cvtColor(cv2.resize(align_224, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        ga_vec = ga_sess.run(None, {"data": np.transpose(bgr96.astype(np.float32), (2, 0, 1))[None]})[0][0].astype(np.float16)

        # 4. Strictly local per-zone CIE-Lab, luminance contrast, sharpness, wrinkle energy, symmetry & Delaunay strain
        lab = cv2.cvtColor(atlas_224, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_ch = lab[:, :, 0]
        gx = cv2.Sobel(l_ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l_ch, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        lap_abs = np.abs(cv2.Laplacian(l_ch, cv2.CV_32F, ksize=3))
        asym_lab = np.mean(np.abs(lab - lab[:, ::-1, :]), axis=2)
        lj_map = np.where(valid_224, log_det_j[tids_224], 0.0).astype(np.float32)
        lj_asym = np.abs(lj_map - lj_map[:, ::-1])

        loc_z = np.zeros((len(ANATOMICAL_ZONES), 10), dtype=np.float32)
        l_mir = l_ch[:, ::-1]
        for zi, (zk, _) in enumerate(ANATOMICAL_ZONES):
            zn = z_norm_224[zk]
            mu_l = float((l_ch * zn).sum())
            mu_a = float((lab[:, :, 1] * zn).sum())
            mu_b = float((lab[:, :, 2] * zn).sum())
            var_l = max(float(((l_ch - mu_l) ** 2 * zn).sum()), 1e-4)
            sd_l = math.sqrt(var_l)
            mu_mr = float((l_mir * zn).sum())
            cov_lr = float(((l_ch - mu_l) * (l_mir - mu_mr) * zn).sum())
            sym_r = cov_lr / max(var_l, 1e-3)
            loc_z[zi] = [
                mu_l,
                mu_a,
                mu_b,
                sd_l,
                float((grad_mag * zn).sum()),
                float((lap_abs * zn).sum()),
                float((asym_lab * zn).sum()),
                sym_r,
                float((lj_map * zn).sum()),
                float((lj_asym * zn).sum()),
            ]

        return {
            "idx": idx,
            "meta": meta,
            "atlas_port": atlas_port,
            "mask_port": mask_port,
            "atlas_224": atlas_224,
            "patches": patches,
            "pool": pool[0].astype(np.float16),
            "sface": sf_vec,
            "genderage": ga_vec,
            "local_zone_feats": loc_z.astype(np.float16),
            "log_det_j": log_det_j,
            "lm106_224": lm106_224.astype(np.float16),
        }

    # Accumulators for linear-sRGB super-resolved composites & canonical UV silhouette
    comp_sums: dict[str, np.ndarray] = {}
    comp_counts: dict[str, int] = {}
    silhouette_sum = np.zeros((portrait_size, portrait_size), dtype=np.float64)
    silhouette_count = 0

    def accum_comp(key: str, port_u8: np.ndarray, symmetrize: bool = True) -> None:
        lin = np.power(port_u8.astype(np.float32) / 255.0, 2.2)
        if symmetrize:
            lin = 0.5 * (lin + lin[:, ::-1, :])
        if key not in comp_sums:
            comp_sums[key] = np.zeros_like(lin, dtype=np.float64)
            comp_counts[key] = 0
        comp_sums[key] += lin
        comp_counts[key] += 1

    path_to_rec = {r["path"]: (idx, r) for idx, r in enumerate(records)}
    tar_list = "\n".join(r["path"] for r in records) + "\n"

    valid_metas: list[dict[str, Any]] = []
    patches_list: list[np.ndarray] = []
    pools_list: list[np.ndarray] = []
    sface_list: list[np.ndarray] = []
    ga_list: list[np.ndarray] = []
    locz_list: list[np.ndarray] = []
    logj_list: list[np.ndarray] = []
    sample_224_by_tier: dict[str, list[np.ndarray]] = {}

    cmd = ["ssh", "-o", "BatchMode=yes", "xbabe3", "tar", "-cf", "-", "-C", "/mnt/xbabe3-external-4tb/HON", "--ignore-failed-read", "-T", "-"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None and proc.stdout is not None

    import threading as _th
    def _feed():
        assert proc.stdin is not None
        proc.stdin.write(tar_list.encode())
        proc.stdin.close()
    feeder = _th.Thread(target=_feed, daemon=True)
    feeder.start()

    batch_items: list[tuple[int, dict[str, Any], bytes]] = []
    t0 = time.time()

    def flush_batch(executor: ThreadPoolExecutor, items: list[tuple[int, dict[str, Any], bytes]]) -> None:
        nonlocal silhouette_count
        for res in executor.map(process_one, items):
            if res is None:
                continue
            m = res["meta"]
            g = m["gender"]
            pct = m["beauty_pct"]
            ab = m["age_bucket"]
            reg = m["region"]
            split = m["split"]
            port = res["atlas_port"]
            mp = res["mask_port"]
            silhouette_sum[:] += 0.5 * (mp + mp[:, ::-1])
            silhouette_count += 1

            # Accumulate composites on non-train cohorts:
            # - Natural-age splits (test + eval_world) populate overall sex, beauty tiers, and world regions
            #   so United States & beauty tiers are never skewed by age-35+ oversampling.
            # - All non-train splits (including eval_age) populate the 5 life-course age buckets (18-24 .. 55+).
            if split != "train":
                accum_comp(f"age:{g}:{ab}", port)
                if split in ("test", "eval_world"):
                    accum_comp(f"sex_all:{g}", port)
                    accum_comp(f"region:{g}:{reg}", port)
                    if pct >= 85.0:
                        accum_comp(f"tier:{g}:top15", port)
                        sample_224_by_tier.setdefault(f"{g}:top15", [])
                        if len(sample_224_by_tier[f"{g}:top15"]) < 48:
                            sample_224_by_tier[f"{g}:top15"].append(res["atlas_224"])
                    elif 42.0 <= pct <= 58.0:
                        accum_comp(f"tier:{g}:mid", port)
                        sample_224_by_tier.setdefault(f"{g}:mid", [])
                        if len(sample_224_by_tier[f"{g}:mid"]) < 48:
                            sample_224_by_tier[f"{g}:mid"].append(res["atlas_224"])
                    elif pct <= 15.0:
                        accum_comp(f"tier:{g}:bot15", port)
                        sample_224_by_tier.setdefault(f"{g}:bot15", [])
                        if len(sample_224_by_tier[f"{g}:bot15"]) < 48:
                            sample_224_by_tier[f"{g}:bot15"].append(res["atlas_224"])

            valid_metas.append(m)
            patches_list.append(res["patches"])
            pools_list.append(res["pool"])
            sface_list.append(res["sface"])
            ga_list.append(res["genderage"])
            locz_list.append(res["local_zone_feats"])
            logj_list.append(res["log_det_j"].astype(np.float16))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                rel = member.name.lstrip("./")
                if rel not in path_to_rec:
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                data_bytes = fobj.read()
                idx, meta = path_to_rec[rel]
                batch_items.append((idx, meta, data_bytes))
                if len(batch_items) >= 32:
                    flush_batch(ex, batch_items)
                    batch_items.clear()
                    if len(valid_metas) % 640 == 0:
                        print(f"[face-attr] processed {len(valid_metas):,} aligned portraits ({time.time()-t0:.1f}s)", flush=True)
            if batch_items:
                flush_batch(ex, batch_items)
                batch_items.clear()
    proc.wait()

    # Decode linear-sRGB averages back to crisp sRGB composites with gentle micro-contrast restoration
    composites_u8: dict[str, np.ndarray] = {}
    for k, s_arr in comp_sums.items():
        n_k = max(comp_counts[k], 1)
        mean_lin = np.clip(s_arr / float(n_k), 0.0, 1.0)
        srgb = np.power(mean_lin, 1.0 / 2.2)
        # Subtle multi-image super-resolution unsharp restoration (preserves natural luminance while restoring averaged micro-edges)
        blur = gaussian_filter(srgb, sigma=(1.15, 1.15, 0.0))
        sharp = np.clip(srgb + 0.24 * (srgb - blur), 0.0, 1.0)
        composites_u8[k] = np.round(sharp * 255.0).astype(np.uint8)

    silhouette_u8 = np.round(
        np.clip(gaussian_filter(silhouette_sum / max(silhouette_count, 1), sigma=2.2), 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    patches_arr = np.stack(patches_list, axis=0)
    pools_arr = np.stack(pools_list, axis=0)
    sface_arr = np.stack(sface_list, axis=0)
    ga_arr = np.stack(ga_list, axis=0)
    locz_arr = np.stack(locz_list, axis=0)
    logj_arr = np.stack(logj_list, axis=0)

    # Cache extracted embeddings & composites in target/ so Step 4 can be iterated deterministically in <2s
    cache_save: dict[str, np.ndarray] = {
        "patches": patches_arr,
        "pools": pools_arr,
        "sface": sface_arr,
        "genderage": ga_arr,
        "local_zone_feats": locz_arr,
        "log_det_j": logj_arr,
        "silhouette_u8": silhouette_u8,
    }
    for k, arr_u8 in composites_u8.items():
        cache_save[f"comp__{k}"] = arr_u8
    for k, lst_224 in sample_224_by_tier.items():
        if lst_224:
            cache_save[f"tier224__{k}"] = np.stack(lst_224, axis=0)
    np.savez_compressed(cache_npz, **cache_save)
    cache_meta.write_text(json.dumps({"metas": valid_metas, "comp_counts": comp_counts}))

    return {
        "metas": valid_metas,
        "patches": patches_arr,
        "pools": pools_arr,
        "sface": sface_arr,
        "genderage": ga_arr,
        "local_zone_feats": locz_arr,
        "log_det_j": logj_arr,
        "composites_u8": composites_u8,
        "silhouette_u8": silhouette_u8,
        "comp_counts": comp_counts,
        "sample_224_by_tier": sample_224_by_tier,
        "lut_224": lut_224,
        "lut_port": lut_port,
        "canon106_224": canon106_224,
        "canon106_port": canon106_port,
        "portrait_frame": {"size": portrait_size, "x0": px0, "y0": py0, "scale": pscale},
    }


def compute_highres_3backbone_counterfactuals(
    sample_224_by_tier: dict[str, list[np.ndarray]],
    zone_masks_224: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Run the 3-Backbone Heavy Suite (DINOv3-ViT-B/16 at 336x336 [21x21 patches] + ConvNeXt V2 + AuraFace Glint360K-R100)
    on canonical tier composites & archetypes to verify nonlinear counterfactual zone impacts in percentile points."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.log_severity_level = 3

    dinob = ort.InferenceSession(str(MODEL_ROOT / "weights/dinov3_vitb16/model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    cnv2 = ort.InferenceSession(str(MODEL_ROOT / "weights/convnextv2/model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    glint = ort.InferenceSession(str(MODEL_ROOT / "weights/auraface/glintr100.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    pinv_data = np.load(MODEL_ROOT / "convnextv2_head_pinv.npz")
    cn_pinv = pinv_data["pinv_wt"].astype(np.float32)
    cn_bias = pinv_data["bias"].astype(np.float32)

    def embed_3backbone(imgs_224: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        n = len(imgs_224)
        chw224 = np.stack([(np.transpose(im.astype(np.float32) / 255.0, (2, 0, 1)) - IMAGENET_MEAN) / IMAGENET_STD for im in imgs_224])
        chw336 = np.stack(
            [
                (np.transpose(cv2.resize(im, (336, 336), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0, (2, 0, 1)) - IMAGENET_MEAN)
                / IMAGENET_STD
                for im in imgs_224
            ]
        )
        seq336, pool_b = dinob.run(None, {"pixel_values": chw336})
        patches_21 = seq336[:, 5:, :].reshape(n, 21, 21, 768)
        cn_log = cnv2.run(None, {"pixel_values": chw224})[0]
        pool_cn = (cn_log - cn_bias) @ cn_pinv
        gl_list = []
        for im in imgs_224:
            im112 = cv2.resize(im, (112, 112), interpolation=cv2.INTER_AREA)
            gin = np.transpose((im112.astype(np.float32) - 127.5) / 127.5, (2, 0, 1))[None]
            gl_list.append(glint.run(None, {"data": gin})[0][0])
        pool_gl = np.stack(gl_list, axis=0)
        cat = np.concatenate([l2_rows(pool_b), l2_rows(pool_cn), l2_rows(pool_gl)], axis=1)
        return cat, patches_21

    out_cf: dict[str, Any] = {}
    for g in ("Female", "Male"):
        top_imgs = sample_224_by_tier.get(f"{g}:top15", [])[:28]
        mid_imgs = sample_224_by_tier.get(f"{g}:mid", [])[:28]
        bot_imgs = sample_224_by_tier.get(f"{g}:bot15", [])[:28]
        if len(top_imgs) < 8 or len(bot_imgs) < 8 or len(mid_imgs) < 8:
            continue
        all_g = top_imgs + mid_imgs + bot_imgs
        targets = np.array([92.0] * len(top_imgs) + [50.0] * len(mid_imgs) + [8.0] * len(bot_imgs), dtype=np.float64)
        feats, patches_21 = embed_3backbone(all_g)
        beta3, b0_3 = fit_ridge(feats.astype(np.float64), targets, np.ones_like(targets), alpha=12.0)

        # Construct neutral 50th-percentile template and top/bottom canonical 224 composites
        tmpl_mid = np.round(np.mean(np.stack(mid_imgs).astype(np.float32), axis=0)).astype(np.uint8)
        tmpl_top = np.round(np.mean(np.stack(top_imgs).astype(np.float32), axis=0)).astype(np.uint8)
        tmpl_bot = np.round(np.mean(np.stack(bot_imgs).astype(np.float32), axis=0)).astype(np.uint8)

        # Evaluate counterfactual zone inpainting on top-decile and bottom-decile archetypes
        cf_imgs = [tmpl_top, tmpl_bot, tmpl_mid]
        zone_keys = [z[0] for z in ANATOMICAL_ZONES]
        for base_im in (tmpl_top, tmpl_bot):
            for zk in zone_keys:
                m = np.clip(zone_masks_224[zk] * 2.4, 0.0, 1.0)[..., None]
                pert = np.round((1.0 - m) * base_im.astype(np.float32) + m * tmpl_mid.astype(np.float32)).astype(np.uint8)
                cf_imgs.append(pert)
        cf_feats, _ = embed_3backbone(cf_imgs)
        cf_preds = cf_feats @ beta3 + b0_3
        p_top, p_bot, p_mid = float(cf_preds[0]), float(cf_preds[1]), float(cf_preds[2])
        n_z = len(zone_keys)
        top_zone_cf = {zone_keys[i]: round(float(p_top - cf_preds[3 + i]), 3) for i in range(n_z)}
        bot_zone_cf = {zone_keys[i]: round(float(p_bot - cf_preds[3 + n_z + i]), 3) for i in range(n_z)}

        # Also compute 21x21 high-res DINOv3-ViT-B/16 site-specific contrast map (Top15 vs Bot15)
        p21_top = patches_21[: len(top_imgs)]
        p21_mid = patches_21[len(top_imgs) : len(top_imgs) + len(mid_imgs)]
        p21_bot = patches_21[len(top_imgs) + len(mid_imgs) :]
        grid21_top = np.zeros((21, 21), dtype=np.float32)
        grid21_bot = np.zeros((21, 21), dtype=np.float32)
        for r in range(21):
            for c in range(21):
                z_all = patches_21[:, r, c, :].astype(np.float64)
                z_mu = p21_mid[:, r, c, :].mean(axis=0).astype(np.float64)
                zc = z_all - z_mu
                bt, _ = fit_ridge(zc, targets - 50.0, np.ones_like(targets), alpha=80.0)
                grid21_top[r, c] = float(((p21_top[:, r, c, :].astype(np.float64) - z_mu) @ bt).mean())
                grid21_bot[r, c] = float(((p21_bot[:, r, c, :].astype(np.float64) - z_mu) @ bt).mean())

        out_cf[g] = {
            "archetype_scores_pct": {"top15": round(p_top, 2), "mid": round(p_mid, 2), "bot15": round(p_bot, 2)},
            "top15_counterfactual_zone_delta_pct": top_zone_cf,
            "bot15_counterfactual_zone_delta_pct": bot_zone_cf,
            "grid21_top": grid21_top,
            "grid21_bot": grid21_bot,
        }
    return out_cf


def fit_and_project_spatial_percentile_maps(
    stream_out: dict[str, Any],
    cf_out: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit full live multi-backbone beauty head + Joint 12-Zone Interventional Shapley attribution strictly on `train`,
    evaluate held-out accuracy on `test`, and produce 128x128 / 384x384 canonical percentile heatmaps for all cohorts."""
    metas = stream_out["metas"]
    patches = stream_out["patches"].astype(np.float32)             # (N, 14, 14, 384)
    pools = stream_out["pools"].astype(np.float32)                 # (N, 384)
    sfaces = stream_out["sface"].astype(np.float32)                # (N, 128)
    gages = stream_out["genderage"].astype(np.float32)             # (N, 3)
    local_zone_feats = stream_out["local_zone_feats"].astype(np.float32)  # (N, 12, 10)
    log_det_j = stream_out["log_det_j"].astype(np.float32)         # (N, T)
    canon106 = stream_out["canon106_224"]
    lut_224 = stream_out["lut_224"]
    pframe = stream_out["portrait_frame"]
    psize = int(pframe["size"])
    px0, py0, pscale = float(pframe["x0"]), float(pframe["y0"]), float(pframe["scale"])

    zone_keys = [z[0] for z in ANATOMICAL_ZONES]
    n_zones = len(zone_keys)
    zone_masks_224 = build_zone_masks_224(canon106, 224)
    # Soft canonical face interior mask in 224x224, 56x56, and 14x14
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
    face_oval_224 = np.clip(1.18 - (((xx - 112.0) / 76.0) ** 2 + ((yy - 125.0) / 93.0) ** 2), 0.0, 1.0)
    face_oval_224 = gaussian_filter(face_oval_224, sigma=4.5)
    face_oval_56 = cv2.resize(face_oval_224, (56, 56), interpolation=cv2.INTER_AREA)
    face_oval_14 = cv2.resize(face_oval_224, (14, 14), interpolation=cv2.INTER_AREA)

    # Map any 224x224 field onto the padded 384x384 portrait frame (and 128x128 compact storage grid)
    def field224_to_portrait(f224: np.ndarray, out_sz: int = 128) -> np.ndarray:
        sc = pscale * (out_sz / float(psize))
        yy_o, xx_o = np.mgrid[0:out_sz, 0:out_sz].astype(np.float32)
        mx = xx_o / sc + px0
        my = yy_o / sc + py0
        return cv2.remap(f224.astype(np.float32), mx, my, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    oval_128 = field224_to_portrait(face_oval_224, 128)
    zone_masks_56 = {k: cv2.resize(v, (56, 56), interpolation=cv2.INTER_AREA) for k, v in zone_masks_224.items()}
    zone_masks_14 = {k: cv2.resize(v, (14, 14), interpolation=cv2.INTER_AREA) for k, v in zone_masks_224.items()}

    # Precompute 12x12 dual-basis spatial reconstruction operator on 56x56 and 224x224 grids
    # so <Z_k_norm, M_i> matches the 12-zone Shapley attribution vector phi_i!
    z_norm_56 = np.stack([zone_masks_56[zk] / max(float(zone_masks_56[zk].sum()), 1e-6) for zk in zone_keys], axis=0)  # (12, 56, 56)
    basis_56 = np.stack([zone_masks_56[zk] * face_oval_56 for zk in zone_keys], axis=0)                                # (12, 56, 56)
    gram_12_56 = np.einsum("kxy,lxy->kl", z_norm_56, basis_56).astype(np.float64)
    dual_inv_56 = np.linalg.inv(gram_12_56 + 0.012 * np.eye(n_zones, dtype=np.float64))

    z_norm_224 = np.stack([zone_masks_224[zk] / max(float(zone_masks_224[zk].sum()), 1e-6) for zk in zone_keys], axis=0)
    basis_224 = np.stack([zone_masks_224[zk] * face_oval_224 for zk in zone_keys], axis=0)
    gram_12_224 = np.einsum("kxy,lxy->kl", z_norm_224, basis_224).astype(np.float64)
    dual_inv_224 = np.linalg.inv(gram_12_224 + 0.012 * np.eye(n_zones, dtype=np.float64))

    # Map each Delaunay triangle centroid to anatomical zone weights
    tris = lut_224["tris"]
    tri_centroids = lut_224["centroids"]  # (T, 2) in 224x224 coords
    tri_zone_w = np.zeros((n_zones, len(tris)), dtype=np.float32)
    for k_idx, zk in enumerate(zone_keys):
        zm = zone_masks_224[zk]
        cx = np.clip(np.round(tri_centroids[:, 0]).astype(int), 0, 223)
        cy = np.clip(np.round(tri_centroids[:, 1]).astype(int), 0, 223)
        tri_zone_w[k_idx] = zm[cy, cx] * lut_224["interior"].astype(np.float32)

    n_all = len(metas)
    indiv_maps_224 = np.zeros((n_all, 56, 56), dtype=np.float32)
    indiv_geom_224 = np.zeros((n_all, 56, 56), dtype=np.float32)
    indiv_zone_phi = np.zeros((n_all, n_zones), dtype=np.float64)
    indiv_pred_pct = np.zeros(n_all, dtype=np.float64)

    live_eval_metrics: dict[str, Any] = {}
    zone_summary_by_sex: dict[str, Any] = {}
    age_maps_summary: dict[str, Any] = {}
    region_maps_summary: dict[str, Any] = {}
    npz_payload: dict[str, np.ndarray] = {
        "oval_128": oval_128.astype(np.float16),
        "silhouette_384": stream_out["silhouette_u8"],
        "canon106_port": stream_out["canon106_port"].astype(np.float32),
        "interior_tris": stream_out["lut_port"]["tris"][stream_out["lut_port"]["interior"]].astype(np.int16),
    }

    # Store all composites into npz_payload
    for k, arr_u8 in stream_out["composites_u8"].items():
        safe_k = "comp__" + k.replace(":", "__").replace(",", "").replace(" ", "_").replace(".", "").replace("&", "and").replace("+", "plus")
        npz_payload[safe_k] = arr_u8

    tids_56 = cv2.resize(lut_224["tids"].astype(np.float32), (56, 56), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    valid_56 = cv2.resize(lut_224["valid"].astype(np.uint8), (56, 56), interpolation=cv2.INTER_NEAREST).astype(bool)

    y_pct = np.array([float(m["beauty_pct"]) for m in metas], dtype=np.float64)
    votes = np.array([float(m["votes"]) for m in metas], dtype=np.float64)
    eb_w = votes / (votes + 42.0)

    for g in ("Female", "Male"):
        tr_i = np.array([i for i, m in enumerate(metas) if m["gender"] == g and m["split"] == "train"])
        te_i = np.array([i for i, m in enumerate(metas) if m["gender"] == g and m["split"] == "test"])
        ev_all_i = np.array([i for i, m in enumerate(metas) if m["gender"] == g and m["split"] != "train"])
        ev_nat_i = np.array([i for i, m in enumerate(metas) if m["gender"] == g and m["split"] in ("test", "eval_world")])
        gall = np.array([i for i, m in enumerate(metas) if m["gender"] == g])
        mid_tr_i = np.array([int(i) for i in tr_i if 42.0 <= y_pct[i] <= 58.0])
        if len(mid_tr_i) < 20:
            mid_tr_i = tr_i

        # 1. Full Live Multi-Backbone Facial Beauty Head via 5-Block Out-of-Fold Stacked Ridge strictly on tr_i
        #    (Each modality block gets its own optimal CV regularization so high-dim pixel/mesh blocks never drown L2 embeddings)
        pmean_all = l2_rows(patches.mean(axis=(1, 2))).astype(np.float64)
        pmir_all = l2_rows(np.abs(patches - patches[:, :, ::-1, :]).mean(axis=(1, 2))).astype(np.float64)
        dino_blk = np.hstack([l2_rows(pools).astype(np.float64), pmean_all, pmir_all * 0.5])
        sface_blk = l2_rows(sfaces).astype(np.float64)

        loc_flat = local_zone_feats[:, :, 1:].reshape(n_all, -1).astype(np.float64)  # exclude col 0 raw L* exposure
        loc_mu, loc_sd = loc_flat[tr_i].mean(axis=0), np.maximum(loc_flat[tr_i].std(axis=0), 1e-5)
        loc_z = np.clip((loc_flat - loc_mu) / loc_sd, -4.5, 4.5)

        ga_mu, ga_sd = gages[tr_i].mean(axis=0).astype(np.float64), np.maximum(gages[tr_i].std(axis=0).astype(np.float64), 1e-5)
        ga_z = np.clip((gages.astype(np.float64) - ga_mu) / ga_sd, -4.5, 4.5)

        lj_mu_tr = (eb_w[tr_i, None] * log_det_j[tr_i]).sum(axis=0) / eb_w[tr_i].sum()
        lj_sd_tr = np.maximum(log_det_j[tr_i].std(axis=0).astype(np.float64), 1e-4)
        lj_z = np.clip((log_det_j.astype(np.float64) - lj_mu_tr) / lj_sd_tr, -4.5, 4.5)
        lj_blk = np.hstack([lj_z, -np.abs(lj_z[:, lut_224["interior"]]) * 0.35])

        mod_blocks = [
            ("dinov3", dino_blk, (0.5, 1.5, 4.0, 10.0, 25.0)),
            ("sface", sface_blk, (0.5, 1.5, 4.0, 10.0, 25.0)),
            ("genderage", ga_z, (2.0, 10.0, 40.0)),
            ("local_anat", loc_z, (30.0, 100.0, 300.0)),
            ("delaunay_222", lj_blk, (50.0, 150.0, 400.0)),
        ]
        rng_cv = np.random.default_rng(101 + (0 if g == "Female" else 1))
        folds = np.array_split(rng_cv.permutation(len(tr_i)), 5)
        oof_mat = np.zeros((len(tr_i), len(mod_blocks)), dtype=np.float64)
        all_blk_preds = np.zeros((n_all, len(mod_blocks)), dtype=np.float64)
        chosen_alphas: dict[str, float] = {}
        for b_idx, (b_name, x_blk, cand_alphas) in enumerate(mod_blocks):
            best_a, best_r = cand_alphas[0], -2.0
            best_oof = np.zeros(len(tr_i), dtype=np.float64)
            for ca in cand_alphas:
                oof_c = np.zeros(len(tr_i), dtype=np.float64)
                for f_idx in range(5):
                    va_f = folds[f_idx]
                    tr_f = np.concatenate([folds[j] for j in range(5) if j != f_idx])
                    bt_f, b0_f = fit_ridge(x_blk[tr_i[tr_f]], y_pct[tr_i[tr_f]] - 50.0, eb_w[tr_i[tr_f]], alpha=ca)
                    oof_c[va_f] = x_blk[tr_i[va_f]] @ bt_f + b0_f
                r_c = float(spearmanr(oof_c, y_pct[tr_i]).statistic)
                if r_c > best_r:
                    best_r, best_a, best_oof = r_c, ca, oof_c
            oof_mat[:, b_idx] = best_oof
            chosen_alphas[b_name] = float(best_a)
            bt_all, b0_all = fit_ridge(x_blk[tr_i], y_pct[tr_i] - 50.0, eb_w[tr_i], alpha=best_a)
            all_blk_preds[:, b_idx] = x_blk @ bt_all + b0_all

        w_stack, b0_stack = fit_ridge(oof_mat, y_pct[tr_i] - 50.0, eb_w[tr_i], alpha=5.0)
        w_stack = np.maximum(w_stack, 0.0)
        best_cv_rho = float(spearmanr(oof_mat @ w_stack + b0_stack, y_pct[tr_i]).statistic)
        raw_full_pred = all_blk_preds @ w_stack + b0_stack

        # Empirical quantile calibration strictly on tr_i so predicted percentiles span [1, 99] faithfully
        q_grid = np.linspace(0.0, 100.0, 101)
        x_q = np.percentile(raw_full_pred[tr_i], q_grid)
        y_q = np.percentile(y_pct[tr_i], q_grid)
        indiv_pred_pct[gall] = np.interp(raw_full_pred[gall], x_q, y_q)

        rho_te = float(spearmanr(indiv_pred_pct[te_i], y_pct[te_i]).statistic)
        live_eval_metrics[g] = {
            "train_n": int(len(tr_i)),
            "test_n": int(len(te_i)),
            "eval_natural_n": int(len(ev_nat_i)),
            "eval_non_train_n": int(len(ev_all_i)),
            "block_cv_alphas": chosen_alphas,
            "train_cv_spearman_r": round(best_cv_rho, 4),
            "test_spearman_r": round(rho_te, 4),
            "test_pct_mae": round(float(np.mean(np.abs(indiv_pred_pct[te_i] - y_pct[te_i]))), 2),
        }

        # 2. Joint 12-Zone Interventional Shapley Attribution (strictly fitted on tr_i, centered on 42-58th pct mid_tr_i)
        #    Uses exposure-invariant local anatomical descriptors (a*, b*, contrast, sharpness, wrinkle, bilateral asym, Jacobian)
        #    + localized DINOv3 symmetric & asymmetry tokens, preventing ambient photo flash on wide cheek skin from dominating.
        patches_mir = patches[:, :, ::-1, :]
        patches_asym = np.abs(patches - patches_mir) * 0.5

        s_strict = np.zeros((n_all, n_zones), dtype=np.float64)
        s_deep = np.zeros((n_all, n_zones), dtype=np.float64)

        for k_idx, zk in enumerate(zone_keys):
            # A. Exposure-invariant zero-halo anatomical descriptors for zone k (cols 1..9: a*, b*, contrast, sharpness, wrinkle, asym, Jacobian)
            lf_k = local_zone_feats[:, k_idx, 1:].astype(np.float64)
            lf_mu_mid = lf_k[mid_tr_i].mean(axis=0)
            lf_sd = np.maximum(lf_k[tr_i].std(axis=0), 1e-5)
            lf_norm = np.clip((lf_k - lf_mu_mid) / lf_sd, -4.0, 4.0)
            # Mild koinophilia / averageness deviation on chromaticity & local Jacobian area (cols 0, 1, 6 of lf_norm)
            av_raw = -np.abs(lf_norm[:, [0, 1, 6]])
            av_c = av_raw - av_raw[mid_tr_i].mean(axis=0)
            tw_k = tri_zone_w[k_idx] > 0.12
            if int(tw_k.sum()) > 0:
                lj_k = lj_z[:, tw_k] - lj_z[mid_tr_i][:, tw_k].mean(axis=0)
                lj_k_av = -np.abs(lj_z[:, tw_k])
                lj_k_av = lj_k_av - lj_k_av[mid_tr_i].mean(axis=0)
                x_strict_k = np.concatenate([lf_norm, av_c * 0.25, lj_k * 0.35, lj_k_av * 0.20], axis=1)
            else:
                x_strict_k = np.concatenate([lf_norm, av_c * 0.25], axis=1)

            bt_st, _ = fit_ridge(x_strict_k[tr_i], y_pct[tr_i] - 50.0, eb_w[tr_i], alpha=65.0)
            raw_st = x_strict_k @ bt_st
            s_strict[:, k_idx] = raw_st - float(raw_st[mid_tr_i].mean())

            # B. Zone-masked DINOv3 symmetric + bilateral-asymmetry token descriptor
            zm14 = 0.5 * (zone_masks_14[zk] + zone_masks_14[zk][:, ::-1])
            zm14_n = zm14 / max(float(zm14.sum()), 1e-6)
            tok_sym = np.einsum("nxyc,xy->nc", patches, zm14_n).astype(np.float64)
            tok_asy = np.einsum("nxyc,xy->nc", patches_asym, zm14_n).astype(np.float64)
            x_deep_k = np.concatenate([tok_sym, tok_asy], axis=1)
            x_deep_k = x_deep_k - x_deep_k[mid_tr_i].mean(axis=0)
            bt_dp, _ = fit_ridge(x_deep_k[tr_i], y_pct[tr_i] - 50.0, eb_w[tr_i], alpha=220.0)
            raw_dp = x_deep_k @ bt_dp
            s_deep[:, k_idx] = raw_dp - float(raw_dp[mid_tr_i].mean())

        # Combine exposure-invariant physical/morphometric attribution (45%) + localized DINOv3 feature attribution (55%)
        # with gentle downweight (0.55) on peripheral framing boundary zones (forehead_glabella, lateral_skin) where hair/pose occlude
        periph_factor = np.array(
            [0.55 if zk in ("lateral_skin", "forehead_glabella") else 1.0 for zk in zone_keys], dtype=np.float64
        )
        s_zone = (0.45 * s_strict + 0.55 * s_deep) * periph_factor[None, :]

        # Shapley-Ridge Coalition Combiner across all 12 zones strictly on tr_i:
        # Solves min_{w >= 0.03} || S_tr @ w - (indiv_pred_pct[tr_i] - mid_pred) ||_W^2 + alpha_c || w - mean(w) ||^2
        # so all 12 zones sum additively to the centered ensemble beauty percentile with zero constant offset subtraction!
        target_tr = indiv_pred_pct[tr_i] - float(indiv_pred_pct[mid_tr_i].mean())
        sw_tr = s_zone[tr_i] * np.sqrt(eb_w[tr_i])[:, None]
        yw_tr = target_tr * np.sqrt(eb_w[tr_i])
        gram_s = sw_tr.T @ sw_tr
        rhs_s = sw_tr.T @ yw_tr
        reg_scale = float(np.trace(gram_s)) / float(n_zones)
        # Projected ridge solver onto w >= 0.03 (prevents collinearity sign-inversions while preserving true data-driven zone leverage)
        eye_z = np.eye(n_zones, dtype=np.float64)
        center_z = eye_z - np.ones((n_zones, n_zones), dtype=np.float64) / float(n_zones)
        a_mat = gram_s + 0.45 * reg_scale * (center_z.T @ center_z) + 0.08 * reg_scale * eye_z
        w_zone = np.linalg.solve(a_mat, rhs_s)
        if np.any(w_zone < 0.03):
            # Active-set projection onto w >= 0.03
            w_zone = np.maximum(w_zone, 0.03)
            for _ in range(80):
                grad = a_mat @ w_zone - rhs_s
                step = 0.25 / max(float( np.linalg.norm(a_mat, ord=2) ), 1e-6)
                w_zone = np.maximum(w_zone - step * grad, 0.03)

        # Calibrate positive scalar multiplier on tr_i so total Top15 - Bot15 sum across 12 zones matches ensemble delta
        phi_unscaled = s_zone * w_zone[None, :]
        sum_tr = phi_unscaled[tr_i].sum(axis=1)
        gamma_pos = float((sum_tr * eb_w[tr_i] * target_tr).sum() / max(float((sum_tr * eb_w[tr_i] * sum_tr).sum()), 1e-6))
        gamma_pos = float(np.clip(gamma_pos, 0.25, 4.0))
        phi_all = phi_unscaled * gamma_pos
        indiv_zone_phi[gall] = phi_all[gall]

        # 3. Also fit 222-triangle morphometric contribution map + fine 14x14 local texture detail strictly on tr_i
        lj_c = (log_det_j - log_det_j[mid_tr_i].mean(axis=0)).astype(np.float64)
        beta_geom, _ = fit_ridge(lj_c[tr_i], y_pct[tr_i] - 50.0, eb_w[tr_i], alpha=55.0)
        tri_contrib_all = lj_c * beta_geom[None, :]

        site_preds_all = np.zeros((n_all, 14, 14), dtype=np.float32)
        for r in range(14):
            for c in range(14):
                if face_oval_14[r, c] < 0.08:
                    continue
                z_loc = patches[:, r, c, :]
                z_mir = patches[:, r, 13 - c, :]
                feat_rc = np.concatenate([z_loc, np.abs(z_loc - z_mir) * 0.5], axis=1).astype(np.float64)
                fc = feat_rc - feat_rc[mid_tr_i].mean(axis=0)
                beta_rc, _ = fit_ridge(fc[tr_i], y_pct[tr_i] - 50.0, eb_w[tr_i], alpha=220.0)
                site_preds_all[:, r, c] = (fc @ beta_rc).astype(np.float32) * face_oval_14[r, c]

        # 4. Dual-Basis Spatial Reconstruction + Zero-Zone-Mean Fine Anatomical Detail for every face i in gall
        c_base_gall = phi_all[gall] @ dual_inv_56.T  # (Ng, 12)
        base_maps_gall = np.einsum("nk,kxy->nxy", c_base_gall, basis_56).astype(np.float32)

        for idx_pos, i in enumerate(gall):
            tex_56 = cv2.resize(site_preds_all[i], (56, 56), interpolation=cv2.INTER_CUBIC) * face_oval_56
            tex_56 = gaussian_filter(tex_56, sigma=1.0) * face_oval_56
            tri_vec = tri_contrib_all[i].astype(np.float32)
            gm_56 = np.where(valid_56, tri_vec[tids_56], 0.0) * face_oval_56
            gm_56 = gaussian_filter(gm_56, sigma=1.15) * face_oval_56

            fine_raw = (0.55 * tex_56 + 0.45 * gm_56 * 18.0).astype(np.float64)
            # Project out zone averages from fine_raw so <Z_k_norm, indiv_maps_224[i]> == phi_all[i, k] identically!
            fine_zmeans = np.einsum("kxy,xy->k", z_norm_56, fine_raw)
            fine_proj_c = dual_inv_56 @ fine_zmeans
            fine_ortho = (fine_raw - np.einsum("k,kxy->xy", fine_proj_c, basis_56)).astype(np.float32) * face_oval_56

            indiv_maps_224[i] = base_maps_gall[idx_pos] + 0.22 * fine_ortho
            indiv_geom_224[i] = gm_56 * 32.0

        # Incorporate zero-zone-mean 21x21 3-Backbone (DINOv3-ViT-B/16 + ConvNeXtV2 + GlintR100) micro-detail on Top/Bot deciles
        def mean_map_128(idx_list: list[int], highres_21: np.ndarray | None = None) -> np.ndarray:
            if not idx_list:
                return np.zeros((128, 128), dtype=np.float32)
            m56 = indiv_maps_224[idx_list].mean(axis=0)
            m224 = cv2.resize(m56, (224, 224), interpolation=cv2.INTER_CUBIC) * face_oval_224
            if highres_21 is not None:
                hr224 = cv2.resize(highres_21.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC) * face_oval_224
                hr224 = gaussian_filter(hr224, sigma=2.8) * face_oval_224
                s_m = max(float(np.std(m224[face_oval_224 > 0.3])), 1e-4)
                s_h = max(float(np.std(hr224[face_oval_224 > 0.3])), 1e-4)
                hr_scaled = (hr224 * (s_m / s_h)).astype(np.float64)
                hr_zmeans = np.einsum("kxy,xy->k", z_norm_224, hr_scaled)
                hr_proj_c = dual_inv_224 @ hr_zmeans
                hr_ortho = (hr_scaled - np.einsum("k,kxy->xy", hr_proj_c, basis_224)).astype(np.float32) * face_oval_224
                m224 = (m224 + 0.18 * hr_ortho) * face_oval_224
            return field224_to_portrait(m224, 128)

        top_idx = [int(i) for i in ev_nat_i if y_pct[i] >= 85.0]
        mid_idx = [int(i) for i in ev_nat_i if 42.0 <= y_pct[i] <= 58.0]
        bot_idx = [int(i) for i in ev_nat_i if y_pct[i] <= 15.0]

        hr_top = cf_out.get(g, {}).get("grid21_top")
        hr_bot = cf_out.get(g, {}).get("grid21_bot")

        map_top = mean_map_128(top_idx, hr_top)
        map_mid = mean_map_128(mid_idx, None)
        map_bot = mean_map_128(bot_idx, hr_bot)
        # Pure anatomical leverage (|ΔP| driver/rejector sensitivity across natural-age evaluation faces)
        abs_56 = np.abs(indiv_maps_224[ev_nat_i]).mean(axis=0)
        map_lev = field224_to_portrait(cv2.resize(abs_56, (224, 224), interpolation=cv2.INTER_CUBIC) * face_oval_224, 128)
        # Pure 222-triangle geometric difference (Top15 minus Bot15)
        geom_diff_56 = indiv_geom_224[top_idx].mean(axis=0) - indiv_geom_224[bot_idx].mean(axis=0)
        map_geom = field224_to_portrait(cv2.resize(geom_diff_56, (224, 224), interpolation=cv2.INTER_CUBIC) * face_oval_224, 128)

        npz_payload[f"map__tier__{g}__top15"] = map_top.astype(np.float16)
        npz_payload[f"map__tier__{g}__mid"] = map_mid.astype(np.float16)
        npz_payload[f"map__tier__{g}__bot15"] = map_bot.astype(np.float16)
        npz_payload[f"map__leverage__{g}"] = map_lev.astype(np.float16)
        npz_payload[f"map__geom_diff__{g}"] = map_geom.astype(np.float16)

        # 12-Zone Quantitative Drivers (+Δ pct in Top15) & Rejectors (-Δ pct in Bot15) with 600-draw bootstrap CIs
        rng_b = np.random.default_rng(77 + (0 if g == "Female" else 1))
        z_rows = []
        for k_idx, (zk, zlabel) in enumerate(ANATOMICAL_ZONES):
            vals_top = indiv_zone_phi[top_idx, k_idx]
            vals_bot = indiv_zone_phi[bot_idx, k_idx]
            vals_all_abs = np.abs(indiv_zone_phi[ev_nat_i, k_idx])

            b_top = [float(vals_top[rng_b.integers(0, len(vals_top), len(vals_top))].mean()) for _ in range(600)]
            b_bot = [float(vals_bot[rng_b.integers(0, len(vals_bot), len(vals_bot))].mean()) for _ in range(600)]
            z_rows.append(
                {
                    "zone_key": zk,
                    "zone_label": zlabel,
                    "top15_driver_pct": round(float(vals_top.mean()), 2),
                    "top15_ci95": [round(float(np.percentile(b_top, 2.5)), 2), round(float(np.percentile(b_top, 97.5)), 2)],
                    "bot15_rejector_pct": round(float(vals_bot.mean()), 2),
                    "bot15_ci95": [round(float(np.percentile(b_bot, 2.5)), 2), round(float(np.percentile(b_bot, 97.5)), 2)],
                    "spread_top_minus_bot_pct": round(float(vals_top.mean() - vals_bot.mean()), 2),
                    "mean_abs_leverage_pct": round(float(vals_all_abs.mean()), 2),
                    "counterfactual_3backbone_top_pct": cf_out.get(g, {}).get("top15_counterfactual_zone_delta_pct", {}).get(zk, 0.0),
                    "counterfactual_3backbone_bot_pct": cf_out.get(g, {}).get("bot15_counterfactual_zone_delta_pct", {}).get(zk, 0.0),
                }
            )
        z_rows.sort(key=lambda r: r["spread_top_minus_bot_pct"], reverse=True)
        zone_summary_by_sex[g] = {
            "top15_n": len(top_idx),
            "mid_n": len(mid_idx),
            "bot15_n": len(bot_idx),
            "eval_n": int(len(ev_nat_i)),
            "zones": z_rows,
        }

        # Age-bucket maps & zone trajectories (18-24 -> 55+) across all non-train cohorts
        age_maps_summary[g] = {}
        for ab in AGE_BUCKETS:
            ab_idx = [int(i) for i in ev_all_i if metas[i]["age_bucket"] == ab]
            mp_ab = mean_map_128(ab_idx, None)
            safe_ab = ab.replace("+", "plus")
            npz_payload[f"map__age__{g}__{safe_ab}"] = mp_ab.astype(np.float16)
            z_ab = {
                zk: round(float(indiv_zone_phi[ab_idx, k_idx].mean()), 2) if ab_idx else 0.0
                for k_idx, (zk, _) in enumerate(ANATOMICAL_ZONES)
            }
            age_maps_summary[g][ab] = {
                "n": len(ab_idx),
                "mean_observed_pct": round(float(y_pct[ab_idx].mean()), 2) if ab_idx else 50.0,
                "mean_predicted_pct": round(float(indiv_pred_pct[ab_idx].mean()), 2) if ab_idx else 50.0,
                "zone_deltas_pct": z_ab,
            }

        # World-region maps, leverage maps & cross-regional spatial consistency matrix (natural-age cohorts)
        region_maps_summary[g] = {"regions": {}}
        reg_lev_vecs: dict[str, np.ndarray] = {}
        mask_flat = oval_128.ravel() > 0.25
        for reg in WORLD_REGIONS:
            r_idx = [int(i) for i in ev_nat_i if metas[i]["region"] == reg]
            if len(r_idx) < 25:
                continue
            mp_reg_signed = mean_map_128(r_idx, None)
            # Also compute within-region Top-Half minus Bottom-Half beauty driver map!
            med_r = float(np.median(y_pct[r_idx]))
            hi_r = [i for i in r_idx if y_pct[i] >= med_r]
            lo_r = [i for i in r_idx if y_pct[i] < med_r]
            diff_56 = indiv_maps_224[hi_r].mean(axis=0) - indiv_maps_224[lo_r].mean(axis=0)
            mp_reg_driver = field224_to_portrait(cv2.resize(diff_56, (224, 224), interpolation=cv2.INTER_CUBIC) * face_oval_224, 128)
            safe_reg = reg.replace(",", "").replace(" ", "_").replace(".", "").replace("&", "and")
            npz_payload[f"map__region_signed__{g}__{safe_reg}"] = mp_reg_signed.astype(np.float16)
            npz_payload[f"map__region_driver__{g}__{safe_reg}"] = mp_reg_driver.astype(np.float16)
            reg_lev_vecs[reg] = mp_reg_driver.ravel()[mask_flat]
            region_maps_summary[g]["regions"][reg] = {
                "n": len(r_idx),
                "mean_observed_pct": round(float(y_pct[r_idx].mean()), 2),
                "mean_predicted_pct": round(float(indiv_pred_pct[r_idx].mean()), 2),
                "mean_age": round(float(np.mean([metas[i]["age"] for i in r_idx])), 2),
            }

        rnames = list(reg_lev_vecs.keys())
        corr_mat = {}
        off_vals = []
        for r1 in rnames:
            corr_mat[r1] = {}
            for r2 in rnames:
                cval = float(np.corrcoef(reg_lev_vecs[r1], reg_lev_vecs[r2])[0, 1])
                corr_mat[r1][r2] = round(cval, 4)
                if r1 < r2:
                    off_vals.append(cval)
        region_maps_summary[g]["cross_region_driver_corr"] = corr_mat
        region_maps_summary[g]["mean_pairwise_driver_corr"] = round(float(np.mean(off_vals)), 4) if off_vals else 0.0
        region_maps_summary[g]["min_pairwise_driver_corr"] = round(float(np.min(off_vals)), 4) if off_vals else 0.0

    summary = {
        "schema_version": 1,
        "portrait_resolution_px": psize,
        "attribution_grid_px": 128,
        "delaunay_landmarks": 106,
        "delaunay_triangles": int(len(lut_224["tris"])),
        "total_aligned_portraits_processed": int(len(metas)),
        "composite_counts": {k: int(v) for k, v in sorted(stream_out["comp_counts"].items())},
        "live_spatial_model_verification": live_eval_metrics,
        "three_backbone_counterfactual_verification": {
            g: {
                "archetype_scores_pct": cf_out[g]["archetype_scores_pct"],
                "top15_counterfactual_zone_delta_pct": cf_out[g]["top15_counterfactual_zone_delta_pct"],
                "bot15_counterfactual_zone_delta_pct": cf_out[g]["bot15_counterfactual_zone_delta_pct"],
            }
            for g in cf_out
        },
        "zone_drivers_and_rejectors": zone_summary_by_sex,
        "age_bucket_progression": age_maps_summary,
        "world_regional_synthesis": region_maps_summary,
    }
    return summary, npz_payload


def analyze_face_attribution(
    frozen_json: Path | None = None,
    frozen_npz: Path | None = None,
    recompute: bool = False,
    use_stream_cache: bool = False,
) -> dict[str, Any]:
    """Return validated facial attribution summary (loading frozen reference artifact unless recompute=True)."""
    default_json = ROOT / "evidence/reference/performance-face-attribution.json"
    default_npz = ROOT / "evidence/reference/performance-face-composites.npz"
    jpath = frozen_json or default_json
    npath = frozen_npz or default_npz

    if not recompute and jpath.exists() and npath.exists():
        summary = json.loads(jpath.read_text())
        summary["composites_npz_sha256"] = hashlib.sha256(npath.read_bytes()).hexdigest()
        summary["composites_npz_bytes"] = npath.stat().st_size
        return summary

    t0 = time.time()
    print("[face-attr] Step 1/4: Auditing 10,000-profile (20,000-image) Pure-Face Multi-Backbone Ensemble...", flush=True)
    bench_10k = evaluate_10k_pure_face_benchmark()
    print(f"[face-attr] 10k Pure-Face Test Spearman={bench_10k['test_overall']['spearman_r']:.4f}, Pct MAE={bench_10k['test_overall']['gender_pct_mae']:.2f}", flush=True)

    print("[face-attr] Step 2/4: Selecting stratified Sex x Age x Percentile x 7-World-Region portrait cohorts...", flush=True)
    records = select_stratified_cohorts(seed=20261111)
    print(f"[face-attr] Selected {len(records):,} unique primary aligned portraits.", flush=True)

    print("[face-attr] Step 3/4: Streaming 1200x1200 portraits -> 106-pt Delaunay super-resolution & DINOv3+SFace extraction...", flush=True)
    stream_out = run_streaming_alignment_and_spatial_features(
        records, portrait_size=384, workers=12, use_cache=use_stream_cache
    )

    print("[face-attr] Step 4/4: Running 3-Backbone counterfactuals, Joint 12-Zone Shapley attribution & zone CIs...", flush=True)
    zone_masks_224 = build_zone_masks_224(stream_out["canon106_224"], 224)
    cf_out = compute_highres_3backbone_counterfactuals(stream_out["sample_224_by_tier"], zone_masks_224)
    summary, npz_payload = fit_and_project_spatial_percentile_maps(stream_out, cf_out)
    summary["benchmark_10k_pure_face"] = bench_10k
    summary["elapsed_seconds_recorded"] = round(time.time() - t0, 1)

    npath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npath, **npz_payload)
    # Remove wall-clock elapsed field before deterministic JSON serialization
    summary.pop("elapsed_seconds_recorded", None)
    summary["composites_npz_sha256"] = hashlib.sha256(npath.read_bytes()).hexdigest()
    summary["composites_npz_bytes"] = npath.stat().st_size

    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"[face-attr] Wrote {jpath} and {npath} ({npath.stat().st_size / 1024 / 1024:.2f} MiB)", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recompute", action="store_true", help="Re-run full live model alignment & attribution from xbabe3.")
    p.add_argument("--reproject-from-cache", action="store_true", help="Re-run Steps 1, 2, 4 from cached 5,353 aligned features.")
    p.add_argument("--json-out", type=Path, default=ROOT / "evidence/reference/performance-face-attribution.json")
    p.add_argument("--npz-out", type=Path, default=ROOT / "evidence/reference/performance-face-composites.npz")
    args = p.parse_args()
    analyze_face_attribution(
        frozen_json=args.json_out,
        frozen_npz=args.npz_out,
        recompute=args.recompute or args.reproject_from_cache,
        use_stream_cache=args.reproject_from_cache,
    )


if __name__ == "__main__":
    main()
