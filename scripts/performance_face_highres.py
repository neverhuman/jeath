#!/usr/bin/env python3
"""High-resolution canonical facial evidence maps from real, human-rated HON portraits.

The published v16 maps resolve 14x14 DINOv3-ViT-S/16 cells on a 224 px atlas. This module asks whether finer,
model-resolved spatial evidence exists and draws it only where the source pixels and held-out human ratings
support it:

* Stage E (``extract``, CPU worker host): stream the same 5,353 stratified 1200x1200 portraits used by
  ``performance_face_attribution.py``, align each to the identical 106-landmark Delaunay atlas at 672 px
  (median source inter-ocular distance is about 430 px, so 672 px is a downsample, never an upscale), and run
  DINOv3-ViT-B/16 natively at 672 px, which yields 42x42 patch tokens. Linear-sRGB composites are accumulated
  at 768 px for cohorts of at least ``MIN_COMPOSITE_N`` faces; no individual face leaves the worker.
* Stage P (``project``): fit a label-free 96-component PCA basis on face-oval tokens and project every token.
* Stage F (``faithfulness-extract``): re-extract a held-out subsample after replacing selected cells with the
  sex-specific training mean atlas, so the deletion test passes through the backbone.
* Stage A (``analyze``, local and deterministic): sex-specific weighted dual ridge over all face-oval cells,
  5-fold cross-fitting, exact additive cell contributions (they sum to each held-out prediction), a
  protocol-matched resolution and backbone ablation against the published 14x14 maps, bootstrap cell
  intervals, a 12-zone roll-up, a label-permutation null, and the pixel-space deletion test.

Run ``python3 scripts/performance_face_highres.py analyze`` after the worker stages to refresh the frozen
reference artifacts in ``evidence/reference/``. ``analyze_face_highres()`` loads them for ``scripts/run.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import threading
import time
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HR_ROOT = ROOT / "target/performance-before-outcomes/face-highres"
REF_JSON = ROOT / "evidence/reference/performance-face-highres.json"
REF_NPZ = ROOT / "evidence/reference/performance-face-highres.npz"
PUBLISHED_FACE_JSON = ROOT / "evidence/reference/performance-face-attribution.json"
LEGACY_CACHE = ROOT / "target/performance-before-outcomes/face-models/live_5353_cache.npz"

SCHEMA = "pbo-face-highres-1"
SEED = 20261113
GRID = 42
ATLAS = 672
PORTRAIT = 768
NPC = 96
FOLDS = 5
MIN_COMPOSITE_N = 50
VOTE_PRIOR = 42.0  # identical to the published module's reliability weight votes / (votes + 42)
BOOTSTRAP = 2000
SOURCE_HOST = "xbabe3"
SOURCE_ROOT = "/mnt/xbabe3-external-4tb/HON"

INPUT_SHA256 = {
    "live_5353_metas.json": "0a6f83dc5dfea10ef07087b54c18a6d08e89174306936a2b2c90f646bfc00cec",
    "canon_lm106_224.npy": "5547f3b047d45e3b57bb8ed432fdb3b74ac585d21f3f8db2cc39ea71fedd42c9",
    "weights/dinov3_vitb16/model.onnx": "0d68aa8f33d21a5616a468239dbe80608abe53619a2663f05e269785eaceb746",
    "weights/dinov3_vitb16/model.onnx_data": "6f292af27497d258257d256bf5eee4151edb67d92f586196beaa182767060353",
    "weights/2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
    "weights/yunet.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
}
GENDERS = ("Female", "Male")
TIERS = {"top15": (85.0, 100.0), "mid": (42.0, 58.0), "bot15": (0.0, 15.0)}
COMPOSITE_SPLITS = ("test", "eval_world")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def verify_inputs(root: Path) -> dict[str, str]:
    out = {}
    for rel, want in INPUT_SHA256.items():
        got = sha256_file(root / rel)
        if got != want:
            raise ValueError(f"{rel}: sha256 {got} != pinned {want}")
        out[rel] = got
    return out


def load_metas(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / "live_5353_metas.json").read_text())["metas"]


def oval_224() -> np.ndarray:
    """Face oval identical to the published module (before its Gaussian softening)."""
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
    return np.clip(1.18 - (((xx - 112.0) / 76.0) ** 2 + ((yy - 125.0) / 93.0) ** 2), 0.0, 1.0)


def oval_cells(grid: int = GRID) -> np.ndarray:
    """Boolean grid of cells whose 224-space centre lies inside the face oval used for modelling."""
    step = 224.0 / grid
    centres = (np.arange(grid, dtype=np.float64) + 0.5) * step
    cx, cy = np.meshgrid(centres, centres)
    return (((cx - 112.0) / 76.0) ** 2 + ((cy - 125.0) / 93.0) ** 2) <= 1.0


# ---------------------------------------------------------------------------
# Stage E: extraction on a CPU worker host
# ---------------------------------------------------------------------------
_W: dict[str, Any] = {}


def _worker_init(root: str, threads: int) -> None:
    import cv2
    import onnxruntime as ort

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import performance_face_attribution as fa

    cv2.setNumThreads(1)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.log_severity_level = 3
    r = Path(root)
    canon224 = np.load(r / "canon_lm106_224.npy").astype(np.float32)
    pt_tmpl, px0, py0, pscale = fa.portrait_template(PORTRAIT)
    canon_port = np.empty_like(canon224)
    canon_port[:, 0] = (canon224[:, 0] - px0) * pscale
    canon_port[:, 1] = (canon224[:, 1] - py0) * pscale
    _W.update(
        fa=fa,
        cv2=cv2,
        canon224=canon224,
        lut_atlas=fa.build_delaunay_lut(canon224 * (ATLAS / 224.0), ATLAS),
        lut_port=fa.build_delaunay_lut(canon_port, PORTRAIT),
        pt_tmpl=pt_tmpl,
        pframe=(px0, py0, pscale),
        det106=ort.InferenceSession(str(r / "weights/2d106det.onnx"), sess_options=so, providers=["CPUExecutionProvider"]),
        dino=ort.InferenceSession(str(r / "weights/dinov3_vitb16/model.onnx"), sess_options=so, providers=["CPUExecutionProvider"]),
        yunet=cv2.FaceDetectorYN.create(str(r / "weights/yunet.onnx"), "", (320, 320), 0.30, 0.3, 5000),
    )


def _align(img_bytes: bytes) -> dict[str, Any]:
    """Reproduce the published alignment exactly, then warp natively to the 672 px atlas and 768 px portrait."""
    from PIL import Image

    fa, cv2 = _W["fa"], _W["cv2"]
    with Image.open(io.BytesIO(img_bytes)) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    clean, clean_mask = fa.neutralize_uv_matte(rgb)
    pts5 = fa.detect_5pt(clean, _W["yunet"])
    canon5 = fa.CANONICAL_UV_5PT * np.array([rgb.shape[1] / 1200.0, rgb.shape[0] / 1200.0], dtype=np.float32)
    fallback = bool(np.allclose(pts5, canon5))
    mat224 = fa.umeyama_similarity(pts5, fa.arcface_template(224))
    align224 = cv2.warpAffine(clean, mat224, (224, 224), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    in192 = np.transpose(cv2.resize(align224, (192, 192), interpolation=cv2.INTER_AREA).astype(np.float32), (2, 0, 1))[None]
    out106 = _W["det106"].run(None, {"data": in192})[0][0, :212].reshape(106, 2).astype(np.float32)
    lm224 = (out106 + 1.0) * 112.0
    if float(np.sqrt(np.mean(np.sum((lm224 - _W["canon224"]) ** 2, axis=1)))) > 28.0:
        lm224 = 0.55 * lm224 + 0.45 * _W["canon224"]
    k = ATLAS / 224.0
    mat_atlas = fa.umeyama_similarity(pts5, fa.arcface_template(ATLAS))
    align_atlas = cv2.warpAffine(clean, mat_atlas, (ATLAS, ATLAS), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    atlas, _ = fa.warp_with_lut(align_atlas, lm224 * k, _W["lut_atlas"])
    px0, py0, pscale = _W["pframe"]
    mat_port = fa.umeyama_similarity(pts5, _W["pt_tmpl"])
    align_port = cv2.warpAffine(clean, mat_port, (PORTRAIT, PORTRAIT), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    lm_port = np.empty_like(lm224)
    lm_port[:, 0] = (lm224[:, 0] - px0) * pscale
    lm_port[:, 1] = (lm224[:, 1] - py0) * pscale
    port, _ = fa.warp_with_lut(align_port, lm_port, _W["lut_port"])
    mask_aff = cv2.warpAffine(clean_mask.astype(np.float32), mat_port, (PORTRAIT, PORTRAIT), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    mask_port, _ = fa.warp_with_lut(mask_aff, lm_port, _W["lut_port"])
    iod = float(np.linalg.norm(pts5[1] - pts5[0]))
    return {"atlas": atlas, "port": port, "mask": np.clip(mask_port, 0.0, 1.0), "lm224": lm224, "fallback": fallback, "iod": iod}


def _tokens(atlas_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chw = (np.transpose(atlas_u8.astype(np.float32) / 255.0, (2, 0, 1)) - IMAGENET_MEAN) / IMAGENET_STD
    seq, pool = _W["dino"].run(None, {"pixel_values": chw[None]})
    return seq[0, 5:, :].reshape(GRID, GRID, 768).astype(np.float16), pool[0].astype(np.float16)


def _extract_one(item: tuple[int, bytes]) -> dict[str, Any] | None:
    idx, data = item
    try:
        a = _align(data)
    except Exception as exc:  # corrupt or unreadable source image: recorded, never imputed
        return {"idx": idx, "error": f"{type(exc).__name__}"}
    tok, pool = _tokens(a["atlas"])
    return {"idx": idx, "tokens": tok, "pool": pool, "port": a["port"], "mask": (a["mask"] * 255).astype(np.uint8),
            "atlas": a["atlas"], "lm224": a["lm224"].astype(np.float32), "fallback": a["fallback"], "iod": a["iod"]}


def _stream(paths: list[str]):
    tar = ["tar", "-cf", "-", "-C", SOURCE_ROOT, "--ignore-failed-read", "-T", "-"]
    # On the source host itself read locally: an ssh alias for the host's own name may point elsewhere.
    cmd = tar if socket.gethostname() == SOURCE_HOST else ["ssh", "-o", "BatchMode=yes", SOURCE_HOST, *tar]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    payload = ("\n".join(paths) + "\n").encode()

    def feed() -> None:
        proc.stdin.write(payload)
        proc.stdin.close()

    threading.Thread(target=feed, daemon=True).start()
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
        for member in tf:
            if member.isfile():
                fh = tf.extractfile(member)
                if fh is not None:
                    yield member.name.lstrip("./"), fh.read()
    proc.wait()


def run_extract(root: Path, work: Path, procs: int, threads: int, limit: int | None) -> None:
    import multiprocessing as mp

    verify_inputs(root)
    metas = load_metas(root)
    if limit:
        metas = metas[:limit]
    work.mkdir(parents=True, exist_ok=True)
    n = len(metas)
    path_to_idx = {m["path"]: i for i, m in enumerate(metas)}
    tokens = np.lib.format.open_memmap(work / "tokens_f16.npy", mode="w+", dtype=np.float16, shape=(n, GRID, GRID, 768))
    pools = np.zeros((n, 768), dtype=np.float16)
    lms = np.zeros((n, 106, 2), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    fallback = np.zeros(n, dtype=bool)
    iod = np.zeros(n, dtype=np.float32)
    errors: dict[str, str] = {}
    comp_sum: dict[str, np.ndarray] = {}
    comp_n: dict[str, int] = {}
    sil_sum = np.zeros((PORTRAIT, PORTRAIT), dtype=np.float64)
    train_atlas_sum = {g: np.zeros((ATLAS, ATLAS, 3), dtype=np.float64) for g in GENDERS}
    train_atlas_n = {g: 0 for g in GENDERS}

    def add(key: str, port: np.ndarray) -> None:
        lin = np.power(port.astype(np.float64) / 255.0, 2.2)
        lin = 0.5 * (lin + lin[:, ::-1, :])
        if key not in comp_sum:
            comp_sum[key] = np.zeros_like(lin)
            comp_n[key] = 0
        comp_sum[key] += lin
        comp_n[key] += 1

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(procs, initializer=_worker_init, initargs=(str(root), threads)) as pool:
        gen = ((path_to_idx[p], b) for p, b in _stream([m["path"] for m in metas]) if p in path_to_idx)
        for done, res in enumerate(pool.imap(_extract_one, gen, chunksize=1), start=1):
            i = res["idx"]
            if "error" in res:
                errors[metas[i]["path"]] = res["error"]
                continue
            m = metas[i]
            tokens[i] = res["tokens"]
            pools[i] = res["pool"]
            lms[i] = res["lm224"]
            ok[i] = True
            fallback[i] = res["fallback"]
            iod[i] = res["iod"]
            mk = res["mask"].astype(np.float64) / 255.0
            sil_sum += 0.5 * (mk + mk[:, ::-1])
            g = m["gender"]
            if m["split"] == "train":
                train_atlas_sum[g] += np.power(res["atlas"].astype(np.float64) / 255.0, 2.2)
                train_atlas_n[g] += 1
            else:
                add(f"age:{g}:{m['age_bucket']}", res["port"])
                if m["split"] in COMPOSITE_SPLITS:
                    add(f"sex_all:{g}", res["port"])
                    pct = float(m["beauty_pct"])
                    for tier, (lo, hi) in TIERS.items():
                        if lo <= pct <= hi:
                            add(f"tier:{g}:{tier}", res["port"])
            if done % 250 == 0:
                print(f"[face-hr] {done}/{n} portraits in {time.time() - t0:.0f}s", flush=True)
    tokens.flush()
    del tokens

    def decode(lin_sum: np.ndarray, count: int) -> np.ndarray:
        return np.round(255.0 * np.power(np.clip(lin_sum / count, 0.0, 1.0), 1.0 / 2.2)).astype(np.uint8)

    comps = {k: decode(v, comp_n[k]) for k, v in comp_sum.items() if comp_n[k] >= MIN_COMPOSITE_N}
    np.savez(work / "extract_side.npz", pools=pools, lm224=lms, ok=ok, fallback=fallback, iod=iod,
             silhouette_u8=np.round(255.0 * sil_sum / max(int(ok.sum()), 1)).astype(np.uint8),
             **{f"comp__{k}": v for k, v in comps.items()},
             **{f"train_mean_atlas__{g}": decode(train_atlas_sum[g], train_atlas_n[g]) for g in GENDERS if train_atlas_n[g]})
    import onnxruntime as ort
    import cv2

    (work / "extract_meta.json").write_text(json.dumps({
        "n_records": n, "n_ok": int(ok.sum()), "n_detector_fallback": int(fallback.sum()), "errors": errors,
        "composite_counts": {k: comp_n[k] for k in sorted(comp_n)},
        "composites_kept_min_n": MIN_COMPOSITE_N,
        "train_mean_atlas_n": train_atlas_n,
        "source_interocular_px": {"min": float(iod[ok].min()), "median": float(np.median(iod[ok])), "max": float(iod[ok].max())},
        "atlas_px": ATLAS, "grid": GRID, "portrait_px": PORTRAIT, "procs": procs, "ort_intra_threads": threads,
        "onnxruntime": ort.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
        "host": os.uname().nodename, "machine": os.uname().machine, "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, sort_keys=True) + "\n")
    print(f"[face-hr] extract done: {int(ok.sum())}/{n} ok, {len(errors)} errors, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Stage P: label-free PCA token basis
# ---------------------------------------------------------------------------
def fit_token_pca(tokens: np.ndarray, ok: np.ndarray, cells: np.ndarray, npc: int = NPC, stride: int = 3,
                  chunk: int = 64) -> dict[str, np.ndarray]:
    """Covariance PCA over face-oval tokens of every ``stride``-th usable face; deterministic sign convention."""
    idx = np.flatnonzero(ok)[::stride]
    d = tokens.shape[-1]
    s1 = np.zeros(d, dtype=np.float64)
    s2 = np.zeros((d, d), dtype=np.float64)
    cnt = 0
    for start in range(0, len(idx), chunk):
        block = np.asarray(tokens[idx[start:start + chunk]], dtype=np.float64)[:, cells, :].reshape(-1, d)
        s1 += block.sum(axis=0)
        s2 += block.T @ block
        cnt += block.shape[0]
    mean = s1 / cnt
    cov = s2 / cnt - np.outer(mean, mean)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:npc]
    basis = evecs[:, order]
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(npc)])
    basis = basis * signs
    return {"mean": mean, "basis": basis, "evals": evals[order], "explained": evals[order].sum() / evals.sum(),
            "n_tokens": np.array(cnt), "n_faces": np.array(len(idx))}


def project_tokens(tokens: np.ndarray, pca: dict[str, np.ndarray], chunk: int = 64) -> np.ndarray:
    n = tokens.shape[0]
    out = np.zeros(tokens.shape[:3] + (pca["basis"].shape[1],), dtype=np.float16)
    mean, basis = pca["mean"].astype(np.float32), pca["basis"].astype(np.float32)
    for start in range(0, n, chunk):
        block = np.asarray(tokens[start:start + chunk], dtype=np.float32)
        out[start:start + chunk] = ((block - mean) @ basis).astype(np.float16)
    return out


def run_project(work: Path) -> None:
    side = np.load(work / "extract_side.npz")
    tokens = np.load(work / "tokens_f16.npy", mmap_mode="r")
    pca = fit_token_pca(tokens, side["ok"], oval_cells())
    x96 = project_tokens(tokens, pca)
    np.save(work / "x96_f16.npy", x96)
    np.savez(work / "token_pca.npz", **pca)
    print(f"[face-hr] PCA {NPC} comps explain {float(pca['explained']):.3f} of face-oval token variance", flush=True)


# ---------------------------------------------------------------------------
# Stage F: pixel-space deletion re-extraction
# ---------------------------------------------------------------------------
def cell_mask_672(cells: list[int], feather: int = 2) -> np.ndarray:
    """Pixel mask of the given 42x42 cell indices on the 672 px atlas, lightly feathered to avoid hard seams."""
    import cv2

    step = ATLAS // GRID
    m = np.zeros((ATLAS, ATLAS), dtype=np.float32)
    for c in cells:
        r, q = divmod(int(c), GRID)
        m[r * step:(r + 1) * step, q * step:(q + 1) * step] = 1.0
    if feather:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return np.clip(m, 0.0, 1.0)


def _faith_one(item: tuple[int, bytes, list[list[int]], np.ndarray]) -> dict[str, Any]:
    idx, data, variants, mean_atlas = item
    atlas = _align(data)["atlas"].astype(np.float32)
    outs = []
    for cells in variants:
        m = cell_mask_672(cells)[..., None]
        pert = np.round((1.0 - m) * atlas + m * mean_atlas.astype(np.float32)).astype(np.uint8)
        outs.append(_tokens(pert)[0])
    return {"idx": idx, "tokens": np.stack(outs)}


def run_faithfulness_extract(root: Path, work: Path, plan_path: Path, procs: int, threads: int) -> None:
    import multiprocessing as mp

    verify_inputs(root)
    metas = load_metas(root)
    plan = json.loads(plan_path.read_text())
    side = np.load(work / "extract_side.npz")
    pca = np.load(work / "token_pca.npz")
    items = {int(k): v for k, v in plan["variants"].items()}
    path_to_idx = {metas[i]["path"]: i for i in items}
    n_var = len(next(iter(items.values())))
    out = np.zeros((len(items), n_var, GRID, GRID, NPC), dtype=np.float16)
    order = sorted(items)
    pos = {i: p for p, i in enumerate(order)}
    mean_atlas = {g: side[f"train_mean_atlas__{g}"] for g in GENDERS}
    ctx = mp.get_context("fork")
    with ctx.Pool(procs, initializer=_worker_init, initargs=(str(root), threads)) as pool:
        gen = ((path_to_idx[p], b, items[path_to_idx[p]], mean_atlas[metas[path_to_idx[p]]["gender"]])
               for p, b in _stream([metas[i]["path"] for i in order]) if p in path_to_idx)
        for res in pool.imap(_faith_one, gen, chunksize=1):
            tok = res["tokens"].astype(np.float32)
            out[pos[res["idx"]]] = ((tok - pca["mean"].astype(np.float32)) @ pca["basis"].astype(np.float32)).astype(np.float16)
    np.savez(work / "faithfulness_x96.npz", order=np.array(order), x96=out)
    print(f"[face-hr] faithfulness re-extraction done for {len(order)} faces x {n_var} variants", flush=True)


# ---------------------------------------------------------------------------
# Stage A: deterministic analysis
# ---------------------------------------------------------------------------
ALPHAS = np.logspace(-3, 3, 25)


def rng_for(*keys: str) -> np.random.Generator:
    """Independent, named random streams so adding one analysis never shifts another's draws."""
    import zlib

    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=[zlib.crc32(k.encode()) for k in keys]))


def gcv_dual_ridge(xc: np.ndarray, yc: np.ndarray, w: np.ndarray, alphas: np.ndarray) -> tuple[np.ndarray, float]:
    """Weighted ridge in dual form with a GCV-selected penalty; returns primal coefficients and the chosen alpha."""
    sw = np.sqrt(w)
    xs = xc * sw[:, None]
    ys = yc * sw
    evals, evecs = np.linalg.eigh(xs @ xs.T)
    evals = np.maximum(evals, 0.0)
    uy = evecs.T @ ys
    n = len(ys)
    scale = max(float(np.mean(evals)), 1e-12)
    best_g, best_a = math.inf, float(alphas[0] * scale)
    for a in alphas * scale:
        shrink = evals / (evals + a)
        resid = ys - evecs @ (shrink * uy)
        g = n * float(resid @ resid) / max((n - float(shrink.sum())) ** 2, 1e-12)
        if g < best_g:
            best_g, best_a = g, float(a)
    return xs.T @ (evecs @ (uy / (evals + best_a))), best_a


def fit_spatial(x: np.ndarray, y: np.ndarray, w: np.ndarray, tr: np.ndarray) -> dict[str, Any]:
    """x: (n, cells, d). Weighted dual ridge on rows ``tr`` with weighted centring."""
    n, c, d = x.shape
    mu = np.average(x[tr].astype(np.float64), axis=0, weights=w[tr])
    ybar = float(np.average(y[tr], weights=w[tr]))
    xc = (x[tr] - mu).reshape(len(tr), c * d)
    beta, a = gcv_dual_ridge(xc, y[tr] - ybar, w[tr], ALPHAS)
    return {"mu": mu, "ybar": ybar, "beta": beta.reshape(c, d), "alpha": a}


def contributions(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Exact additive cell contributions (percentile points); their sum plus ``ybar`` is the prediction."""
    return np.einsum("ncd,cd->nc", x.astype(np.float64) - model["mu"], model["beta"])


def predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return model["ybar"] + contributions(model, x).sum(axis=1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).correlation)


def paired_bootstrap_rho(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    from scipy.stats import rankdata

    n = len(y)
    d = np.empty(BOOTSTRAP)
    for k in range(BOOTSTRAP):
        i = rng.integers(0, n, n)
        ry, r1, r2 = rankdata(y[i]), rankdata(p1[i]), rankdata(p2[i])
        d[k] = np.corrcoef(ry, r1)[0, 1] - np.corrcoef(ry, r2)[0, 1]
    return {"delta_spearman": round(spearman(y, p1) - spearman(y, p2), 4),
            "ci_95": [round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)]}


def mean_ci(v: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    v = np.asarray(v, dtype=np.float64)
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(BOOTSTRAP)])
    return {"estimate": round(float(v.mean()), 3), "ci_95": [round(float(np.quantile(b, 0.025)), 3), round(float(np.quantile(b, 0.975)), 3)],
            "n": int(len(v))}


def fold_ids(n: int, rng: np.random.Generator) -> np.ndarray:
    f = np.empty(n, dtype=np.int64)
    f[rng.permutation(n)] = np.arange(n) % FOLDS
    return f


def zone_partition(cells: np.ndarray, root: Path = HR_ROOT) -> tuple[list[str], np.ndarray]:
    """Rows: the published 12 anatomical zones; columns: modelled oval cells; each column sums to 1."""
    import cv2

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import performance_face_attribution as fa

    canon = np.load(root / "canon_lm106_224.npy").astype(np.float32)
    masks = fa.build_zone_masks_224(canon, 224)
    keys = [z[0] for z in fa.ANATOMICAL_ZONES]
    grid = np.stack([cv2.resize(masks[k].astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA)
                     for k in keys]).reshape(len(keys), GRID * GRID)[:, cells.ravel()]
    grid[keys.index("lateral_skin"), grid.sum(axis=0) <= 1e-9] = 1.0
    return keys, grid / grid.sum(axis=0, keepdims=True)


def pca_rows(tokens_2d: np.ndarray, npc: int) -> tuple[np.ndarray, np.ndarray, float]:
    t = tokens_2d.astype(np.float64)
    mean = t.mean(axis=0)
    cov = np.cov(t, rowvar=False, bias=True)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:npc]
    basis = evecs[:, order]
    basis = basis * np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(npc)])
    return mean, basis, float(evals[order].sum() / evals.sum())


def split_half_r(contrib: np.ndarray, sel: np.ndarray, rng: np.random.Generator, reps: int = 200) -> dict[str, Any]:
    """Reliability of a tier map: correlation between maps from two disjoint random halves of its faces."""
    rs = []
    for _ in range(reps):
        p = rng.permutation(sel)
        h = len(p) // 2
        rs.append(np.corrcoef(contrib[p[:h]].mean(axis=0), contrib[p[h:2 * h]].mean(axis=0))[0, 1])
    rs = np.array(rs)
    r = float(np.median(rs))
    return {"split_half_r": round(r, 4), "spearman_brown_full": round(2 * r / (1 + r), 4) if r > -1 else None}


def analyze(work: Path, root: Path = HR_ROOT) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    verify_inputs(root)
    metas = load_metas(root)
    side = np.load(work / "extract_side.npz")
    meta_x = json.loads((work / "extract_meta.json").read_text())
    pca = np.load(work / "token_pca.npz")
    x96 = np.load(work / "x96_f16.npy", mmap_mode="r")
    ok = side["ok"]
    if len(ok) != len(metas):
        raise ValueError("extraction does not cover the pinned 5,353-record cohort")
    cells = oval_cells()
    cell_idx = np.flatnonzero(cells.ravel())
    cells14 = oval_cells(14)
    cell14_idx = np.flatnonzero(cells14.ravel())
    zone_keys, zone_w = zone_partition(cells, root)
    zone_area = zone_w.sum(axis=1)
    pct = np.array([m["beauty_pct"] for m in metas], dtype=np.float64)
    votes = np.array([m["votes"] for m in metas], dtype=np.float64)
    w_all = votes / (votes + VOTE_PRIOR)
    split = np.array([m["split"] for m in metas])
    gender = np.array([m["gender"] for m in metas])
    published = json.loads(PUBLISHED_FACE_JSON.read_text())

    legacy = None
    if LEGACY_CACHE.exists():
        lc = np.load(LEGACY_CACHE, mmap_mode="r")
        lt = np.asarray(lc["patches"], dtype=np.float32).reshape(len(metas), 196, 384)[:, cell14_idx, :]
        lmean, lbasis, lexpl = pca_rows(lt[ok].reshape(-1, 384), NPC)
        legacy = {"x": ((lt - lmean) @ lbasis).astype(np.float32), "explained": lexpl,
                  "sha256": sha256_file(LEGACY_CACHE)}

    summary: dict[str, Any] = {
        "schema": SCHEMA, "module_sha256": sha256_file(Path(__file__)), "inputs": dict(INPUT_SHA256),
        "extraction": {k: meta_x[k] for k in ("n_records", "n_ok", "n_detector_fallback", "source_interocular_px", "atlas_px",
                                              "grid", "portrait_px", "onnxruntime", "opencv", "numpy", "host", "machine",
                                              "procs", "ort_intra_threads")},
        "extraction_errors": len(meta_x["errors"]),
        "token_pca": {"components": NPC, "explained_variance": round(float(pca["explained"]), 4), "faces": int(pca["n_faces"]),
                      "tokens": int(pca["n_tokens"])},
        "modelled_cells": {"grid": GRID, "face_oval_cells": int(cells.sum()), "legacy_grid": 14,
                           "legacy_face_oval_cells": int(cells14.sum())},
        "legacy_cache_sha256": legacy["sha256"] if legacy else None,
        "legacy_pca_explained_variance": round(legacy["explained"], 4) if legacy else None,
        "vote_weight": f"votes/(votes+{VOTE_PRIOR:g})", "seed": SEED, "folds": FOLDS, "bootstrap": BOOTSTRAP,
        "zone_keys": zone_keys, "by_sex": {},
    }
    payload: dict[str, np.ndarray] = {"cells_mask_42": cells.astype(np.uint8)}
    plan: dict[str, list[list[int]]] = {}
    state: dict[str, Any] = {}

    for g in GENDERS:
        rows = np.flatnonzero((gender == g) & ok)
        xg = np.asarray(x96[rows], dtype=np.float32).reshape(len(rows), GRID * GRID, NPC)[:, cell_idx, :]
        yg, wg, sg = pct[rows], w_all[rows], split[rows]
        tr, te = np.flatnonzero(sg == "train"), np.flatnonzero(sg == "test")
        res: dict[str, Any] = {"n_faces": int(len(rows)), "n_train_protocol": int(len(tr)), "n_test_protocol": int(len(te))}

        # (1) Protocol-matched: published train split -> published test split, identical learner for every arm
        full = np.zeros((len(rows), GRID * GRID, NPC), dtype=np.float32)
        full[:, cell_idx, :] = xg
        pooled = full.reshape(len(rows), 14, 3, 14, 3, NPC).mean(axis=(2, 4)).reshape(len(rows), 196, NPC)[:, cell14_idx, :]
        del full
        cls = side["pools"][rows].astype(np.float32)[:, None, :]
        arms = {"vitb16_672_grid42": xg, "vitb16_672_pooled_grid14": pooled, "vitb16_672_global_pool_only": cls}
        if legacy is not None:
            arms["published_vits16_224_grid14"] = legacy["x"][rows]
        preds = {}
        alphas = {}
        for name, xa in arms.items():
            m = fit_spatial(xa, yg, wg, tr)
            preds[name] = predict(m, xa[te])
            alphas[name] = m["alpha"]
        yt = yg[te]
        res["protocol_matched_test"] = {name: {"spearman_r": round(spearman(yt, p), 4),
                                               "pct_mae": round(float(np.mean(np.abs(yt - p))), 2)} for name, p in preds.items()}
        pub = published["live_spatial_model_verification"][g]
        res["published_multiblock_reference"] = {"spearman_r": pub["test_spearman_r"], "pct_mae": pub["test_pct_mae"],
                                                 "note": "published model adds Delaunay, SFace, gender-age and local-anatomy blocks"}
        rb = rng_for("paired", g)
        res["paired_bootstrap_grid42_minus"] = {name: paired_bootstrap_rho(yt, preds["vitb16_672_grid42"], p, rb)
                                                for name, p in preds.items() if name != "vitb16_672_grid42"}
        res["gcv_alpha"] = {k: float(f"{v:.6g}") for k, v in alphas.items()}

        # (2) 5-fold cross-fitting over all faces of this sex: every map value comes from a model blind to that face
        f = fold_ids(len(rows), rng_for("folds", g))
        oof = np.zeros((len(rows), len(cell_idx)))
        oof_pred = np.zeros(len(rows))
        ybars = np.zeros(len(rows))
        models = []
        for k in range(FOLDS):
            trk, tek = np.flatnonzero(f != k), np.flatnonzero(f == k)
            mk = fit_spatial(xg, yg, wg, trk)
            oof[tek] = contributions(mk, xg[tek])
            ybars[tek] = mk["ybar"]
            oof_pred[tek] = ybars[tek] + oof[tek].sum(axis=1)
            models.append(mk)
        res["cross_fitted_all_faces"] = {"spearman_r": round(spearman(yg, oof_pred), 4),
                                         "pct_mae": round(float(np.mean(np.abs(yg - oof_pred))), 2),
                                         "additivity_max_abs_error": float(np.max(np.abs(ybars + oof.sum(axis=1) - oof_pred)))}

        # (3) Label-permutation null through the identical pipeline
        yperm = yg[rng_for("perm", g).permutation(len(yg))]
        null_oof = np.zeros_like(oof)
        for k in range(FOLDS):
            trk, tek = np.flatnonzero(f != k), np.flatnonzero(f == k)
            null_oof[tek] = contributions(fit_spatial(xg, yperm, wg, trk), xg[tek])

        # (4) Tier maps, cell intervals, reliability, zone roll-up
        in_comp = np.isin(sg, COMPOSITE_SPLITS)
        tiers: dict[str, Any] = {}
        rt = rng_for("tiers", g)
        for tier, (lo, hi) in TIERS.items():
            sel = np.flatnonzero(in_comp & (yg >= lo) & (yg <= hi))
            mean_map = oof[sel].mean(axis=0)
            boots = np.empty((BOOTSTRAP, len(cell_idx)))
            zb = np.empty((BOOTSTRAP, len(zone_keys)))
            zone_faces = oof[sel] @ zone_w.T
            for b in range(BOOTSTRAP):
                ii = rt.integers(0, len(sel), len(sel))
                boots[b] = oof[sel[ii]].mean(axis=0)
                zb[b] = zone_faces[ii].mean(axis=0)
            lo_ci, hi_ci = np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)
            sig = (lo_ci > 0) | (hi_ci < 0)
            null_map = null_oof[sel].mean(axis=0)
            for name, vec, dtype in (("map42", mean_map, np.float32), ("sig42", sig, np.uint8), ("null42", null_map, np.float32)):
                grid = np.zeros(GRID * GRID, dtype=dtype)
                grid[cell_idx] = vec
                payload[f"{name}__{g}__{tier}"] = grid.reshape(GRID, GRID)
            actual = float(np.mean(yg[sel] - ybars[sel]))
            tiers[tier] = {
                "n_faces": int(len(sel)),
                "mean_actual_deviation_pct": round(actual, 3),
                "mean_predicted_deviation_pct": round(float(mean_map.sum()), 3),
                "recovered_share_of_actual": round(float(mean_map.sum()) / actual, 4) if abs(actual) > 1 else None,
                "cells_ci_excluding_zero_share": round(float(sig.mean()), 4),
                "abs_cell_mean_pct": round(float(np.abs(mean_map).mean()), 5),
                "null_abs_cell_mean_pct": round(float(np.abs(null_map).mean()), 5),
                "null_map_corr_with_real": round(float(np.corrcoef(mean_map, null_map)[0, 1]), 4),
                "reliability": split_half_r(oof, sel, rng_for("halves", g, tier)),
                "zones_pct": {zk: {"estimate": round(float(zone_faces[:, zi].mean()), 3),
                                   "ci_95": [round(float(np.quantile(zb[:, zi], 0.025)), 3),
                                             round(float(np.quantile(zb[:, zi], 0.975)), 3)],
                                   # Zone totals scale with area; per-cell weight compares zones of unequal size.
                                   "cells": round(float(zone_area[zi]), 2),
                                   "per_cell_pct": round(float(zone_faces[:, zi].mean() / zone_area[zi]), 5),
                                   "per_cell_ci_95": [round(float(np.quantile(zb[:, zi], 0.025) / zone_area[zi]), 5),
                                                      round(float(np.quantile(zb[:, zi], 0.975) / zone_area[zi]), 5)]}
                              for zi, zk in enumerate(zone_keys)},
            }
            dens = zone_faces.mean(axis=0) / zone_area
            top_zone = int(np.argmax(np.abs(dens)))
            tiers[tier]["densest_zone"] = {"zone_key": zone_keys[top_zone],
                                           "per_cell_over_oval_mean": round(float(dens[top_zone] / mean_map.mean()), 3)
                                           if abs(float(mean_map.mean())) > 1e-12 else None}
        res["tiers"] = tiers
        payload[f"spread42__{g}"] = payload[f"map42__{g}__top15"] - payload[f"map42__{g}__bot15"]
        top, bot = tiers["top15"]["recovered_share_of_actual"], tiers["bot15"]["recovered_share_of_actual"]
        res["tail_legibility"] = {"top15_recovered_share": top, "bot15_recovered_share": bot,
                                  "bottom_over_top": round(bot / top, 3) if top and bot else None}

        pz = published.get("zone_drivers_and_rejectors", {}).get(g, {}).get("zones", [])
        pub_top = {r["zone_key"]: r["top15_driver_pct"] for r in pz}
        pub_bot = {r["zone_key"]: r["bot15_rejector_pct"] for r in pz}
        keys = [k for k in zone_keys if k in pub_top]
        res["published_zone_agreement"] = {
            "zones": len(keys),
            "top15_rank_spearman": round(spearman(np.array([tiers["top15"]["zones_pct"][k]["estimate"] for k in keys]),
                                                  np.array([pub_top[k] for k in keys])), 3),
            "bot15_rank_spearman": round(spearman(np.array([tiers["bot15"]["zones_pct"][k]["estimate"] for k in keys]),
                                                  np.array([pub_bot[k] for k in keys])), 3)}

        # (5) Deletion plan for a deterministic held-out subsample: drivers, rejectors, three random equal-area sets
        rp = rng_for("deletion-plan", g)
        te_rows = np.flatnonzero(sg == "test")
        pick = te_rows[np.sort(rp.choice(len(te_rows), size=min(150, len(te_rows)), replace=False))]
        kdel = max(1, int(round(0.10 * len(cell_idx))))
        for r in pick:
            c = oof[r]
            plan[str(int(rows[r]))] = ([cell_idx[np.argsort(-c, kind="stable")[:kdel]].tolist(),
                                        cell_idx[np.argsort(c, kind="stable")[:kdel]].tolist()]
                                       + [np.sort(cell_idx[rp.choice(len(cell_idx), kdel, replace=False)]).tolist() for _ in range(3)])
        state[g] = {"rows": rows, "folds": f, "models": models, "oof_pred": oof_pred}
        summary["by_sex"][g] = res

    zp = np.zeros((len(zone_keys), GRID * GRID), dtype=np.float32)
    zp[:, cell_idx] = zone_w
    payload["zone_partition_42"] = zp
    comps = {k[len("comp__"):]: side[k] for k in side.files if k.startswith("comp__")}
    for g in GENDERS:
        for key in (f"tier:{g}:top15", f"tier:{g}:mid", f"tier:{g}:bot15", f"sex_all:{g}"):
            if key in comps:
                payload["comp768__" + key.replace(":", "__")] = comps[key]
    payload["silhouette_768"] = side["silhouette_u8"]
    summary["composite_counts"] = {k: v for k, v in meta_x["composite_counts"].items() if v >= MIN_COMPOSITE_N}

    fpath = work / "faithfulness_x96.npz"
    summary["deletion_test"] = _deletion_test(np.load(fpath), plan, state, gender, cell_idx) if fpath.exists() else None
    return summary, payload, {"variants": plan}


def _deletion_test(faith: Any, plan: dict[str, list[list[int]]], state: dict[str, Any], gender: np.ndarray,
                   cell_idx: np.ndarray) -> dict[str, Any]:
    """Pixel-space deletion: cells replaced by the training mean atlas, re-encoded by DINOv3, re-scored by the fold model."""
    order = [int(i) for i in faith["order"]]
    if sorted(order) != sorted(int(k) for k in plan):
        raise ValueError("faithfulness re-extraction does not match the current deletion plan")
    x = faith["x96"]
    out: dict[str, Any] = {"cells_deleted_share": 0.10, "variants": ["drivers", "rejectors", "random_x3"]}
    for g in GENDERS:
        st = state[g]
        pos = {int(r): i for i, r in enumerate(st["rows"])}
        d_drv, d_rej, d_rnd = [], [], []
        for j, gi in enumerate(order):
            if gender[gi] != g:
                continue
            i = pos[gi]
            mk = st["models"][int(st["folds"][i])]
            xv = np.asarray(x[j], dtype=np.float32).reshape(x.shape[1], GRID * GRID, NPC)[:, cell_idx, :]
            delta = predict(mk, xv) - st["oof_pred"][i]
            d_drv.append(delta[0])
            d_rej.append(delta[1])
            d_rnd.append(float(np.mean(delta[2:])))
        d_drv, d_rej, d_rnd = map(np.array, (d_drv, d_rej, d_rnd))
        r = rng_for("deletion-ci", g)
        out[g] = {"drivers_deleted_delta_pct": mean_ci(d_drv, r), "rejectors_deleted_delta_pct": mean_ci(d_rej, r),
                  "random_deleted_delta_pct": mean_ci(d_rnd, r),
                  "drivers_minus_random_pct": mean_ci(d_drv - d_rnd, r), "rejectors_minus_random_pct": mean_ci(d_rej - d_rnd, r)}
    return out


def write_reference(summary: dict[str, Any], payload: dict[str, np.ndarray]) -> None:
    REF_NPZ.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: payload[k] for k in sorted(payload)})
    REF_NPZ.write_bytes(buf.getvalue())
    summary["npz_sha256"] = hashlib.sha256(buf.getvalue()).hexdigest()
    summary["npz_bytes"] = len(buf.getvalue())
    REF_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")


def analyze_face_highres(json_path: Path = REF_JSON, npz_path: Path = REF_NPZ) -> dict[str, Any] | None:
    """Load the frozen high-resolution reference for ``scripts/run.py`` and verify the NPZ hash."""
    if not (json_path.exists() and npz_path.exists()):
        return None
    summary = json.loads(json_path.read_text())
    got = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    if got != summary["npz_sha256"]:
        raise ValueError(f"{npz_path}: sha256 {got} != recorded {summary['npz_sha256']}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--root", type=Path, default=HR_ROOT)
    e.add_argument("--work", type=Path, default=HR_ROOT / "work")
    e.add_argument("--procs", type=int, default=4)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("project")
    p.add_argument("--work", type=Path, default=HR_ROOT / "work")
    f = sub.add_parser("faithfulness-extract")
    f.add_argument("--root", type=Path, default=HR_ROOT)
    f.add_argument("--work", type=Path, default=HR_ROOT / "work")
    f.add_argument("--plan", type=Path, default=HR_ROOT / "work/faithfulness_plan.json")
    f.add_argument("--procs", type=int, default=4)
    f.add_argument("--threads", type=int, default=4)
    a = sub.add_parser("analyze")
    a.add_argument("--root", type=Path, default=HR_ROOT)
    a.add_argument("--work", type=Path, default=HR_ROOT / "work")
    a.add_argument("--write-reference", action="store_true")
    args = ap.parse_args()
    if args.cmd == "extract":
        run_extract(args.root, args.work, args.procs, args.threads, args.limit)
    elif args.cmd == "project":
        run_project(args.work)
    elif args.cmd == "faithfulness-extract":
        run_faithfulness_extract(args.root, args.work, args.plan, args.procs, args.threads)
    else:
        summary, payload, plan = analyze(args.work, args.root)
        (args.work / "faithfulness_plan.json").write_text(json.dumps(plan, sort_keys=True) + "\n")
        if args.write_reference:
            write_reference(summary, payload)
        print(json.dumps({g: summary["by_sex"][g]["protocol_matched_test"] for g in GENDERS}, indent=1))


if __name__ == "__main__":
    main()
