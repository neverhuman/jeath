"""Shared helpers for the numerical experiments."""
import json, os, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

def rng_for(seed, *spawn):
    """Independent generator from a SeedSequence with a fixed entropy and spawn key."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=tuple(spawn)))

def save(name, obj):
    obj = dict(obj)
    obj["_written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    obj["_numpy"] = np.__version__
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    return obj

def load(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)

def mcse(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(len(x))

def zscore(x, exact):
    x = np.asarray(x, float)
    se = mcse(x)
    return float((x.mean() - exact) / se) if se > 0 else 0.0

def rank_match(yf, ym):
    """Ideal reciprocal rank matching: k-th ranked woman pairs with k-th ranked man.
    Returns index arrays (females, males) of the pairs."""
    return np.argsort(yf), np.argsort(ym)

def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))
