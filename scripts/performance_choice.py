"""Measure how attractiveness ratings relate to a yes in the Columbia speed dates."""
import csv
import hashlib
from pathlib import Path
import urllib.request

import numpy as np

SHA256 = "fda9a924b18880fabe6c7a71961cff20aa3c7cc2a5aa8d6e823c2fc41893814c"
MIRROR = "https://raw.githubusercontent.com/tsizzle/speed_dating_experiment/master/Speed_Dating_Data.csv"
ATTRIBUTES = ("attr", "fun", "intel", "amb", "sinc", "shar")
SEXES = ("women", "men")
BOOTSTRAPS = 4000
SEED = 20261111


def number(value):
    return float(value) if value else np.nan


def load_data(path):
    if not path.exists():
        with urllib.request.urlopen(MIRROR, timeout=45) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("Columbia speed-dating CSV checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError(f"Columbia speed-dating CSV checksum mismatch: {path}")
    fields = ("wave", "gender", "iid", "pid", "dec", *ATTRIBUTES, *(a + "1_1" for a in ATTRIBUTES))
    return np.array([[number(row[key]) for key in fields]
                     for row in csv.DictReader(data.decode("latin-1").splitlines())])


def outcome(value, draws):
    return {"estimate": float(value), "interval_95": np.quantile(draws, [.025, .975]).tolist()}


def analyze(path: Path):
    data = load_data(path)
    wave, sex, chooser, partner, yes = data[:, :5].T
    own = data[:, 5:11].copy()
    own[(own < 1) | (own > 10)] = np.nan
    panel = np.full_like(own, np.nan)
    # A partner's peers are the other choosers who saw that partner in the same session.
    for event in np.unique(wave):
        for person in np.unique(partner[wave == event]):
            rows = np.flatnonzero((wave == event) & (partner == person))
            values = own[rows]
            totals = np.nansum(values, axis=0) - np.nan_to_num(values)
            peers = np.isfinite(values).sum(axis=0) - np.isfinite(values)
            panel[rows] = np.divide(totals, peers, out=np.full_like(totals, np.nan), where=peers >= 3)

    sessions, session_index = np.unique(wave, return_inverse=True)
    assert len(sessions) == 21 and len(data) == 8378 and len(np.unique(chooser)) == 551
    rng = np.random.default_rng(SEED)
    resamples = rng.integers(len(sessions), size=(BOOTSTRAPS, len(sessions)))
    valid_decision = (yes == 0) | (yes == 1)
    quintile = np.full(len(yes), -1, dtype=int)
    curve_rows = valid_decision & np.isfinite(panel[:, 0])
    for person in np.unique(chooser[curve_rows]):
        rows = np.flatnonzero(curve_rows & (chooser == person))
        _, order, counts = np.unique(panel[rows, 0], return_inverse=True, return_counts=True)
        rank = ((counts.cumsum() - .5 * counts) / len(rows))[order]
        quintile[rows] = (rank * 5).astype(int)

    curves, curves_boot = {}, {}
    for index, gender in enumerate(SEXES):
        session_quintile = (session_index * 5 + quintile)[curve_rows & (sex == index)]
        sample = yes[curve_rows & (sex == index)]
        n = np.bincount(session_quintile, minlength=len(sessions) * 5).reshape(-1, 5)
        y = np.bincount(session_quintile, weights=sample, minlength=len(sessions) * 5).reshape(-1, 5)
        rates = y.sum(axis=0) / n.sum(axis=0)
        boot = y[resamples].sum(axis=1) / n[resamples].sum(axis=1)
        curves_boot[gender] = boot[:, 4] - boot[:, 0]
        curves[gender] = {
            "quintiles": [dict(n=int(count), yes=int(yes_count), **outcome(rate, samples))
                          for count, yes_count, rate, samples in zip(n.sum(axis=0), y.sum(axis=0), rates, boot.T)],
            "top_minus_bottom": outcome(rates[4] - rates[0], curves_boot[gender]),
            "participants": len(np.unique(chooser[curve_rows & (sex == index)])),
        }
    curve_gap = outcome(curves["men"]["top_minus_bottom"]["estimate"] -
                        curves["women"]["top_minus_bottom"]["estimate"], curves_boot["men"] - curves_boot["women"])

    slopes = {}
    for method, ratings in (("own", own), ("panel", panel)):
        complete = np.isfinite(ratings).all(axis=1)
        scale = ratings[complete].std(axis=0)
        predictors = ratings / scale
        fits, boot_fits = {}, {}
        for index, gender in enumerate(SEXES):
            selected = complete & valid_decision & (sex == index)
            xx = np.zeros((len(sessions), 6, 6))
            xy = np.zeros((len(sessions), 6))
            used_rows = used_people = 0
            for person in np.unique(chooser[selected]):
                rows = np.flatnonzero(selected & (chooser == person))
                if len(rows) < 2:
                    continue
                event = session_index[rows[0]]
                assert (session_index[rows] == event).all()
                x = predictors[rows] - predictors[rows].mean(axis=0)
                y = yes[rows] - yes[rows].mean()
                xx[event] += x.T @ x
                xy[event] += x.T @ y
                used_rows += len(rows)
                used_people += 1
            coef = np.linalg.solve(xx.sum(axis=0), xy.sum(axis=0))
            boot = np.linalg.solve(xx[resamples].sum(axis=1), xy[resamples].sum(axis=1)[..., None])[..., 0]
            boot_fits[gender] = boot
            fits[gender] = {"n_decisions": used_rows, "n_choosers": used_people,
                            "coefficients_pp": {key: outcome(100 * value, 100 * samples)
                                                for key, value, samples in zip(ATTRIBUTES, coef, boot.T)}}
        slopes[method] = {**fits, "men_minus_women_pp": {
            key: outcome(fits["men"]["coefficients_pp"][key]["estimate"] -
                         fits["women"]["coefficients_pp"][key]["estimate"], 100 * samples)
            for key, samples in zip(ATTRIBUTES, (boot_fits["men"] - boot_fits["women"]).T)}}

    stated = {}
    for index, gender in enumerate(SEXES):
        rows = np.flatnonzero(sex == index)
        _, first = np.unique(chooser[rows], return_index=True)
        allocations = data[rows[first], 11:]
        totals = allocations.sum(axis=1)
        usable = np.isfinite(allocations).all(axis=1) & (totals > 0)
        stated[gender] = {"n": int(usable.sum()),
                          "mean_looks_points_of_100": float(np.mean(100 * allocations[usable, 0] / totals[usable]))}

    return {
        "input_sha256": SHA256, "mirror": MIRROR, "records": len(data), "sessions": len(sessions),
        "participants": len(np.unique(chooser)), "bootstrap": {"unit": "session", "draws": BOOTSTRAPS, "seed": SEED},
        "cohorts": {
            "quintiles": "Within each chooser, midpoint ranks of partners' mean 1–10 looks ratings from at least three other session peers. Quintiles are 20-point rank bands; ties remain together. Rates count yes decisions among rows with an observed binary decision and available panel rating.",
            "slopes": "Within-chooser linear probability models for yes, by chooser gender, adjusting for looks, fun, intelligence, ambition, sincerity and shared interests. Own and leave-one-out panel ratings use separate, common-across-gender SD scales. All six ratings and two usable decisions per chooser are required.",
            "stated": "First row per participant. Mean looks points among complete allocations after rescaling each total to 100.",
        },
        "counts": {"valid_binary_decisions": int(valid_decision.sum()), "decisions_with_looks_panel": int(curve_rows.sum())},
        "yes_by_panel_quintile": curves, "quintile_contrast_men_minus_women": curve_gap,
        "adjusted_slopes": slopes, "stated_looks": stated,
    }
