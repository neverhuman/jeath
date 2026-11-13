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


def load_raw_rows(path: Path) -> list[dict[str, str]]:
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
    return list(csv.DictReader(data.decode("latin-1").splitlines()))


def load_data(path):
    raw_rows = load_raw_rows(path)
    fields = ("wave", "gender", "iid", "pid", "dec", *ATTRIBUTES, *(a + "1_1" for a in ATTRIBUTES), "like", "prob")
    return np.array([[number(row[key]) for key in fields] for row in raw_rows]), raw_rows


def outcome(value, draws):
    return {"estimate": float(value), "interval_95": np.quantile(draws, [.025, .975]).tolist()}


def analyze(path: Path):
    data, raw_rows = load_data(path)
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
        allocations = data[rows[first], 11:17]
        totals = allocations.sum(axis=1)
        usable = np.isfinite(allocations).all(axis=1) & (totals > 0)
        stated[gender] = {"n": int(usable.sum()),
                          "mean_looks_points_of_100": float(np.mean(100 * allocations[usable, 0] / totals[usable]))}

    # --- Dyadic Chemistry: Bipartite Social Relations Variance Partition, Reciprocity & Chosen-Back Slopes ---
    like_raw = data[:, 17].copy()
    like_raw[(like_raw < 0) | (like_raw > 10)] = np.nan
    prob_raw = data[:, 18].copy()
    prob_raw[(prob_raw < 0) | (prob_raw > 10)] = np.nan

    srm_cols = {
        "attr": own[:, 0],
        "like": like_raw,
        "fun": own[:, 1],
        "shar": own[:, 5],
        "sinc": own[:, 4],
        "intel": own[:, 2],
    }
    variance_partition = {}
    like_residuals = np.full(len(data), np.nan)
    for vname, vec in srm_cols.items():
        ok = np.isfinite(vec) & np.isfinite(chooser) & np.isfinite(partner)
        idx_ok = np.flatnonzero(ok)
        y = vec[ok]
        w_ok, s_ok, c_ok, p_ok = wave[ok], sex[ok], chooser[ok], partner[ok]
        c_eff = np.zeros_like(y)
        p_eff = np.zeros_like(y)
        b_mean = np.zeros_like(y)
        n_blocks = 0
        wc_groups = 0
        wp_groups = 0
        for ev in sessions:
            for sx in (0.0, 1.0):
                bm = (w_ok == ev) & (s_ok == sx)
                if not bm.any():
                    continue
                n_blocks += 1
                yb = y[bm]
                mu = float(yb.mean())
                b_mean[bm] = mu
                uc, inv_c = np.unique(c_ok[bm], return_inverse=True)
                up, inv_p = np.unique(p_ok[bm], return_inverse=True)
                wc_groups += len(uc)
                wp_groups += len(up)
                cnt_c = np.bincount(inv_c)
                cnt_p = np.bincount(inv_p)
                ce = np.zeros_like(yb)
                pe = np.zeros_like(yb)
                for _ in range(35):
                    ce = (np.bincount(inv_c, weights=yb - mu - pe) / cnt_c)[inv_c]
                    pe = (np.bincount(inv_p, weights=yb - mu - ce) / cnt_p)[inv_p]
                c_eff[bm] = ce
                p_eff[bm] = pe
        res_v = y - b_mean - c_eff - p_eff
        if vname == "like":
            like_residuals[idx_ok] = res_v
        n_v = len(y)
        df_r = max(1, n_v - wc_groups - wp_groups + n_blocks)
        sig2_r = float(np.sum(res_v ** 2) / df_r)
        sig2_c = max(0.0, float((np.sum(c_eff ** 2) - (wc_groups - n_blocks) * sig2_r) / n_v))
        sig2_p = max(0.0, float((np.sum(p_eff ** 2) - (wp_groups - n_blocks) * sig2_r) / n_v))
        tot_u = sig2_p + sig2_c + sig2_r
        # Sample-mean (unadjusted) variance shares for reference
        raw_p, raw_c, raw_r = float(np.var(p_eff)), float(np.var(c_eff)), float(np.var(res_v))
        tot_raw = raw_p + raw_c + raw_r
        variance_partition[vname] = {
            "n": int(n_v),
            "unbiased_srm_shares": {
                "partner_consensus": float(sig2_p / tot_u),
                "chooser_leniency": float(sig2_c / tot_u),
                "unique_dyadic_and_noise": float(sig2_r / tot_u),
            },
            "sample_mean_shares": {
                "partner_consensus": float(raw_p / tot_raw),
                "chooser_leniency": float(raw_c / tot_raw),
                "unique_dyadic_and_noise": float(raw_r / tot_raw),
            },
        }

    # Exact within-session mutual dyadic reciprocity on residual liking
    dyad_map = {}
    for row_i in np.flatnonzero(np.isfinite(like_residuals)):
        dyad_map[(int(wave[row_i]), int(chooser[row_i]), int(partner[row_i]))] = float(like_residuals[row_i])
    dyad_pairs = []
    for (ev_i, i_id, j_id), r_ij in sorted(dyad_map.items()):
        if i_id < j_id and (ev_i, j_id, i_id) in dyad_map:
            dyad_pairs.append((session_index[np.flatnonzero(wave == ev_i)[0]], r_ij, dyad_map[(ev_i, j_id, i_id)]))
    dyad_arr = np.array(dyad_pairs, dtype=np.float64)
    d_sess = dyad_arr[:, 0].astype(int)
    d_x, d_y = dyad_arr[:, 1], dyad_arr[:, 2]
    recip_r = float(np.corrcoef(d_x, d_y)[0, 1])
    # Session-cluster bootstrap for mutual dyadic reciprocity r
    s_n = np.bincount(d_sess, minlength=len(sessions))
    s_x = np.bincount(d_sess, weights=d_x, minlength=len(sessions))
    s_y = np.bincount(d_sess, weights=d_y, minlength=len(sessions))
    s_xx = np.bincount(d_sess, weights=d_x ** 2, minlength=len(sessions))
    s_yy = np.bincount(d_sess, weights=d_y ** 2, minlength=len(sessions))
    s_xy = np.bincount(d_sess, weights=d_x * d_y, minlength=len(sessions))
    bn = s_n[resamples].sum(axis=1)
    bx = s_x[resamples].sum(axis=1) / bn
    by = s_y[resamples].sum(axis=1) / bn
    cov_xy = s_xy[resamples].sum(axis=1) / bn - bx * by
    var_x = np.maximum(1e-12, s_xx[resamples].sum(axis=1) / bn - bx ** 2)
    var_y = np.maximum(1e-12, s_yy[resamples].sum(axis=1) / bn - by ** 2)
    boot_recip_r = cov_xy / np.sqrt(var_x * var_y)

    # Reciprocity-adjusted within-chooser LPM adding perceived reciprocal liking ("prob")
    ext_attrs = (*ATTRIBUTES, "prob")
    ext_ratings = np.column_stack([own, prob_raw])
    ext_complete = np.isfinite(ext_ratings).all(axis=1)
    ext_scale = ext_ratings[ext_complete].std(axis=0)
    ext_preds = ext_ratings / ext_scale
    recip_slopes = {}
    for index, gender in enumerate(SEXES):
        selected = ext_complete & valid_decision & (sex == index)
        xx7 = np.zeros((len(sessions), 7, 7))
        xy7 = np.zeros((len(sessions), 7))
        used_rows = used_people = 0
        for person in np.unique(chooser[selected]):
            rows_p = np.flatnonzero(selected & (chooser == person))
            if len(rows_p) < 2:
                continue
            ev_p = session_index[rows_p[0]]
            xp = ext_preds[rows_p] - ext_preds[rows_p].mean(axis=0)
            yp = yes[rows_p] - yes[rows_p].mean()
            xx7[ev_p] += xp.T @ xp
            xy7[ev_p] += xp.T @ yp
            used_rows += len(rows_p)
            used_people += 1
        coef7 = np.linalg.solve(xx7.sum(axis=0), xy7.sum(axis=0))
        boot7 = np.linalg.solve(xx7[resamples].sum(axis=1), xy7[resamples].sum(axis=1)[..., None])[..., 0]
        sum_dyad_est = 100.0 * float(coef7[1] + coef7[5] + coef7[6])  # fun + shar + prob
        sum_dyad_boot = 100.0 * (boot7[:, 1] + boot7[:, 5] + boot7[:, 6])
        recip_slopes[gender] = {
            "n_decisions": used_rows,
            "n_choosers": used_people,
            "coefficients_pp": {
                key: outcome(100.0 * val, 100.0 * smp)
                for key, val, smp in zip(ext_attrs, coef7, boot7.T)
            },
            "dyadic_spark_sum_fun_shar_prob_pp": outcome(sum_dyad_est, sum_dyad_boot),
        }

    dyadic_chemistry = {
        "variance_partition": variance_partition,
        "mutual_dyadic_reciprocity_like": {
            "n_dyads": int(len(dyad_arr)),
            **outcome(recip_r, boot_recip_r),
        },
        "reciprocity_adjusted_slopes": recip_slopes,
    }

    # --- Checklist Hobby Correlation (int_corr) vs. Live Conversational Wavelength (shar) & Longitudinal Hindsight ---
    int_corr = np.array([number(r.get("int_corr", "")) for r in raw_rows], dtype=np.float64)
    int_corr[(int_corr < -1.0) | (int_corr > 1.0)] = np.nan
    shar_live = own[:, 5]

    def session_boot_corr(x_vec: np.ndarray, y_vec: np.ndarray, mask: np.ndarray) -> dict:
        m = mask & np.isfinite(x_vec) & np.isfinite(y_vec)
        xv, yv, sv = x_vec[m], y_vec[m], session_index[m]
        r_est = float(np.corrcoef(xv, yv)[0, 1])
        sn = np.bincount(sv, minlength=len(sessions))
        sx = np.bincount(sv, weights=xv, minlength=len(sessions))
        sy = np.bincount(sv, weights=yv, minlength=len(sessions))
        sxx = np.bincount(sv, weights=xv ** 2, minlength=len(sessions))
        syy = np.bincount(sv, weights=yv ** 2, minlength=len(sessions))
        sxy = np.bincount(sv, weights=xv * yv, minlength=len(sessions))
        bn_c = sn[resamples].sum(axis=1)
        bx_c = sx[resamples].sum(axis=1) / bn_c
        by_c = sy[resamples].sum(axis=1) / bn_c
        cov_c = sxy[resamples].sum(axis=1) / bn_c - bx_c * by_c
        vx_c = np.maximum(1e-12, sxx[resamples].sum(axis=1) / bn_c - bx_c ** 2)
        vy_c = np.maximum(1e-12, syy[resamples].sum(axis=1) / bn_c - by_c ** 2)
        return {"n": int(m.sum()), **outcome(r_est, cov_c / np.sqrt(vx_c * vy_c))}

    common_cs_mask = valid_decision & np.isfinite(int_corr) & np.isfinite(shar_live)
    r_ic_yes = session_boot_corr(int_corr, yes, common_cs_mask)
    r_sh_yes = session_boot_corr(shar_live, yes, common_cs_mask)
    r_ic_sh = session_boot_corr(int_corr, shar_live, common_cs_mask)

    # Head-to-head chooser-FE LPM: objective 17-hobby checklist correlation (int_corr) + 6 live attributes
    chk_attrs = ("int_corr", *ATTRIBUTES)
    chk_mat = np.column_stack([int_corr, own])
    chk_complete = np.isfinite(chk_mat).all(axis=1)
    chk_scale = chk_mat[chk_complete].std(axis=0)
    chk_preds = chk_mat / chk_scale
    checklist_fe_slopes = {}
    for index, gender in enumerate(SEXES):
        selected = chk_complete & valid_decision & (sex == index)
        xx_c = np.zeros((len(sessions), 7, 7))
        xy_c = np.zeros((len(sessions), 7))
        used_r = used_p = 0
        for person in np.unique(chooser[selected]):
            rows_p = np.flatnonzero(selected & (chooser == person))
            if len(rows_p) < 2:
                continue
            ev_p = session_index[rows_p[0]]
            xp = chk_preds[rows_p] - chk_preds[rows_p].mean(axis=0)
            yp = yes[rows_p] - yes[rows_p].mean()
            xx_c[ev_p] += xp.T @ xp
            xy_c[ev_p] += xp.T @ yp
            used_r += len(rows_p)
            used_p += 1
        coef_c = np.linalg.solve(xx_c.sum(axis=0), xy_c.sum(axis=0))
        boot_c = np.linalg.solve(xx_c[resamples].sum(axis=1), xy_c[resamples].sum(axis=1)[..., None])[..., 0]
        checklist_fe_slopes[gender] = {
            "n_decisions": int(used_r),
            "n_choosers": int(used_p),
            "coefficients_pp": {
                key: outcome(100.0 * val, 100.0 * smp)
                for key, val, smp in zip(chk_attrs, coef_c, boot_c.T)
            },
        }

    # Longitudinal Hindsight Awakening across Waves 1 -> 2 -> 3 (normalized 100-point allocations per participant)
    rng_hindsight = np.random.default_rng(SEED + 77)
    first_by_iid = {}
    match_flag = np.array([number(r.get("match", "")) for r in raw_rows], dtype=np.float64)
    for row_idx, r in enumerate(raw_rows):
        iid_i = int(float(r["iid"]))
        if iid_i not in first_by_iid:
            first_by_iid[iid_i] = r

    wave_specs = (
        ("t1_pre_stated", "1_1"),
        ("t1_perceived_opp_sex", "2_1"),
        ("t1_perceived_same_sex", "4_1"),
        ("t2_nextday_stated", "1_2"),
        ("t2_nextday_realized", "7_2"),
        ("t3_threeweek_stated", "1_3"),
        ("t3_threeweek_realized", "7_3"),
    )
    alloc_by_iid = {wkey: {} for wkey, _ in wave_specs}
    sex_by_iid = {}
    followup_rows = []
    for iid_i, r in sorted(first_by_iid.items()):
        sx_i = int(float(r["gender"]))
        sex_by_iid[iid_i] = sx_i
        for wkey, sfx in wave_specs:
            vals = np.array([number(r.get(a + sfx, "")) for a in ATTRIBUTES], dtype=np.float64)
            if np.isfinite(vals).all() and vals.sum() > 0:
                alloc_by_iid[wkey][iid_i] = 100.0 * vals / vals.sum()
        d3_v = number(r.get("date_3", ""))
        yc_v = number(r.get("you_call", ""))
        tc_v = number(r.get("them_cal", ""))
        n_matches_i = int(np.sum((chooser == iid_i) & (match_flag == 1.0)))
        if d3_v in (0.0, 1.0):
            followup_rows.append((sx_i, n_matches_i, d3_v, yc_v, tc_v))

    hindsight_waves = {}
    wave_boot_draws = {}
    for wkey, _ in wave_specs:
        hindsight_waves[wkey] = {}
        wave_boot_draws[wkey] = {}
        for index, gender in enumerate(SEXES):
            pts = np.array([vec for iid_i, vec in alloc_by_iid[wkey].items() if sex_by_iid[iid_i] == index], dtype=np.float64)
            nw = len(pts)
            b_idx = rng_hindsight.integers(0, nw, size=(2000, nw))
            b_means = pts[b_idx].mean(axis=1)
            wave_boot_draws[wkey][gender] = (pts.mean(axis=0), b_means)
            hindsight_waves[wkey][gender] = {
                "n_participants": int(nw),
                "points_of_100": {
                    attr_k: outcome(pts[:, j].mean(), b_means[:, j])
                    for j, attr_k in enumerate(ATTRIBUTES)
                },
            }

    # Cross-sex mind-reading blindness (what each sex thinks opposite sex wants [2_1] minus opposite sex actual stated [1_1])
    cross_sex_misperception = {}
    for perceiver, target in (("women", "men"), ("men", "women")):
        p_pt, p_boot = wave_boot_draws["t1_perceived_opp_sex"][perceiver]
        t_pt, t_boot = wave_boot_draws["t1_pre_stated"][target]
        diff_pt = p_pt - t_pt
        diff_boot = p_boot - t_boot
        cross_sex_misperception[f"{perceiver}_about_{target}"] = {
            "n_perceivers": hindsight_waves["t1_perceived_opp_sex"][perceiver]["n_participants"],
            "n_targets": hindsight_waves["t1_pre_stated"][target]["n_participants"],
            "delta_points": {
                attr_k: outcome(diff_pt[j], diff_boot[:, j])
                for j, attr_k in enumerate(ATTRIBUTES)
            },
        }

    # Within-person pluralistic ignorance (what fellow same-sex peers want [4_1] minus own pre-stated [1_1])
    # Added after paired_shifts below so common_iids pairing is preserved.

    # Within-person paired shifts: T2 realized - T1 pre-stated, and T3 stated - T2 realized
    paired_shifts = {}
    for shift_name, w_from, w_to in (
        ("nextday_realized_minus_pre_stated", "t1_pre_stated", "t2_nextday_realized"),
        ("nextday_stated_minus_pre_stated", "t1_pre_stated", "t2_nextday_stated"),
        ("threeweek_stated_minus_nextday_realized", "t2_nextday_realized", "t3_threeweek_stated"),
    ):
        paired_shifts[shift_name] = {}
        for index, gender in enumerate(SEXES):
            common_iids = [
                iid_i for iid_i in sorted(alloc_by_iid[w_from])
                if iid_i in alloc_by_iid[w_to] and sex_by_iid[iid_i] == index
            ]
            diffs = np.array([alloc_by_iid[w_to][iid_i] - alloc_by_iid[w_from][iid_i] for iid_i in common_iids], dtype=np.float64)
            nd = len(diffs)
            b_idx = rng_hindsight.integers(0, nd, size=(2000, nd))
            b_means = diffs[b_idx].mean(axis=1)
            paired_shifts[shift_name][gender] = {
                "n_paired": int(nd),
                "delta_points": {
                    attr_k: outcome(diffs[:, j].mean(), b_means[:, j])
                    for j, attr_k in enumerate(ATTRIBUTES)
                },
            }

    # Within-person pluralistic ignorance (what fellow same-sex peers want [4_1] minus own pre-stated [1_1])
    rng_mind = np.random.default_rng(SEED + 99)
    pluralistic_ignorance = {}
    for index, gender in enumerate(SEXES):
        common_p = [
            iid_i for iid_i in sorted(alloc_by_iid["t1_pre_stated"])
            if iid_i in alloc_by_iid["t1_perceived_same_sex"] and sex_by_iid[iid_i] == index
        ]
        diffs_p = np.array(
            [alloc_by_iid["t1_perceived_same_sex"][iid_i] - alloc_by_iid["t1_pre_stated"][iid_i] for iid_i in common_p],
            dtype=np.float64,
        )
        np_i = len(diffs_p)
        bp_idx = rng_mind.integers(0, np_i, size=(2000, np_i))
        bp_means = diffs_p[bp_idx].mean(axis=1)
        pluralistic_ignorance[gender] = {
            "n_paired": int(np_i),
            "delta_points": {
                attr_k: outcome(diffs_p[:, j].mean(), bp_means[:, j])
                for j, attr_k in enumerate(ATTRIBUTES)
            },
        }

    fu_arr = np.array(followup_rows, dtype=np.float64)
    matched_fu = fu_arr[fu_arr[:, 1] >= 1.0]
    initiation_by_sex = {}
    for index, gender in enumerate(SEXES):
        mf_g = matched_fu[(matched_fu[:, 0] == float(index)) & np.isfinite(matched_fu[:, 3]) & np.isfinite(matched_fu[:, 4])]
        tot_matches_g = float(mf_g[:, 1].sum())
        tot_you_call_g = float(mf_g[:, 3].sum())
        tot_them_cal_g = float(mf_g[:, 4].sum())
        initiation_by_sex[gender] = {
            "n_matched_respondents": int(len(mf_g)),
            "total_mutual_matches": int(tot_matches_g),
            "placed_any_call_share": float((mf_g[:, 3] > 0.0).mean()),
            "zero_calls_placed_share": float((mf_g[:, 3] == 0.0).mean()),
            "received_any_call_share": float((mf_g[:, 4] > 0.0).mean()),
            "calls_placed_per_match_share": float(tot_you_call_g / tot_matches_g),
            "calls_received_per_match_share": float(tot_them_cal_g / tot_matches_g),
            "went_on_followup_date_share": float(mf_g[:, 2].mean()),
        }
    initiation_by_sex["male_to_female_calls_per_match_ratio"] = float(
        initiation_by_sex["men"]["calls_placed_per_match_share"]
        / initiation_by_sex["women"]["calls_placed_per_match_share"]
    )
    initiation_by_sex["total_mutual_match_dyads"] = int(np.sum(match_flag == 1.0) // 2)

    # Objective college median SAT (mn_sat) blindness in 4-minute stranger dates vs 15-rater single-context panel
    rng_sat = np.random.default_rng(SEED + 91)
    sat_by_iid = {}
    for iid_i, r in sorted(first_by_iid.items()):
        raw_s = str(r.get("mn_sat", "")).replace(",", "").strip()
        val_s = number(raw_s)
        if np.isfinite(val_s) and val_s > 400.0:
            sat_by_iid[iid_i] = float(val_s)
    own_sat_vec = np.array([sat_by_iid.get(int(c), np.nan) if np.isfinite(c) else np.nan for c in chooser], dtype=np.float64)
    part_sat_vec = np.array([sat_by_iid.get(int(p), np.nan) if np.isfinite(p) else np.nan for p in partner], dtype=np.float64)
    intel_own_vec = own[:, 2]

    sat_date_mask = np.isfinite(part_sat_vec) & np.isfinite(intel_own_vec) & valid_decision
    sd_psat = part_sat_vec[sat_date_mask]
    sd_intl = intel_own_vec[sat_date_mask]
    sd_yes = yes[sat_date_mask]
    n_sd_sat = len(sd_psat)
    r_sat_intel_pt = float(np.corrcoef(sd_psat, sd_intl)[0, 1])
    r_sat_yes_pt = float(np.corrcoef(sd_psat, sd_yes)[0, 1])
    b_sat_intel = np.empty(2000, dtype=np.float64)
    b_sat_yes = np.empty(2000, dtype=np.float64)
    for b in range(2000):
        bi = rng_sat.integers(0, n_sd_sat, size=n_sd_sat)
        b_sat_intel[b] = np.corrcoef(sd_psat[bi], sd_intl[bi])[0, 1]
        b_sat_yes[b] = np.corrcoef(sd_psat[bi], sd_yes[bi])[0, 1]

    # 15-rater panel mean intelligence vs partner's objective SAT (k=1 single-context jury bound)
    panel_sat_rows = []
    for pid_u in sorted(set(int(p) for p in partner[sat_date_mask])):
        m_p = sat_date_mask & (partner == pid_u)
        if int(m_p.sum()) >= 10:
            panel_sat_rows.append((sat_by_iid[pid_u], float(intel_own_vec[m_p].mean())))
    ps_arr = np.array(panel_sat_rows, dtype=np.float64)
    n_ps_targets = len(ps_arr)
    r_panel_sat_pt = float(np.corrcoef(ps_arr[:, 0], ps_arr[:, 1])[0, 1])
    b_panel_sat = np.empty(2000, dtype=np.float64)
    for b in range(2000):
        bi = rng_sat.integers(0, n_ps_targets, size=n_ps_targets)
        b_panel_sat[b] = np.corrcoef(ps_arr[bi, 0], ps_arr[bi, 1])[0, 1]

    both_sat_match = np.isfinite(own_sat_vec) & np.isfinite(part_sat_vec) & (match_flag == 1.0)
    r_match_sat = float(np.corrcoef(own_sat_vec[both_sat_match], part_sat_vec[both_sat_match])[0, 1])
    chooser_grade_mask = np.isfinite(own_sat_vec) & np.isfinite(intel_own_vec)
    r_chooser_strict = float(np.corrcoef(own_sat_vec[chooser_grade_mask], intel_own_vec[chooser_grade_mask])[0, 1])

    # --- v16: Outlier Confidence vs. Gross Arrogance, Audience Priors, & Status-Threat Backlash ---
    def _ols_pp(y_vec: np.ndarray, x_cols: list[np.ndarray], names: list[str]) -> dict:
        yv = y_vec.astype(np.float64)
        X = np.column_stack([np.ones(len(yv), dtype=np.float64), *[c.astype(np.float64) for c in x_cols]])
        beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        dof = max(1, len(yv) - X.shape[1])
        s2 = float((resid @ resid) / dof)
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        out = {}
        for idx_c, cname in enumerate(names, start=1):
            b_val = float(beta[idx_c])
            s_val = float(se[idx_c])
            t_val = float(b_val / s_val) if s_val > 1e-12 else 0.0
            out[cname] = {
                "slope_pp": 100.0 * b_val,
                "se_pp": 100.0 * s_val,
                "t_stat": t_val,
                "ci_95_pp": [100.0 * (b_val - 1.96 * s_val), 100.0 * (b_val + 1.96 * s_val)],
            }
        return {"n": int(len(yv)), "coefficients": out}

    first_by_iid: dict[int, dict[str, str]] = {}
    rec_by_iid: dict[int, list[list[float]]] = {}
    for row in raw_rows:
        iid_str = row.get("iid", "").strip()
        if not iid_str:
            continue
        iid_int = int(float(iid_str))
        if iid_int not in first_by_iid:
            first_by_iid[iid_int] = row
        rec_vals = [number(row.get(k, "")) for k in ("attr_o", "sinc_o", "intel_o", "fun_o", "amb_o", "dec_o", "like_o")]
        rec_by_iid.setdefault(iid_int, []).append(rec_vals)

    person_self_intel: dict[int, float] = {}
    person_self_amb: dict[int, float] = {}
    person_jury_intel: dict[int, float] = {}
    person_overclaim: dict[int, float] = {}
    person_age: dict[int, float] = {}
    person_race: dict[int, int] = {}
    person_archetype_rows = []

    for iid_int, frow in sorted(first_by_iid.items()):
        s_attr = number(frow.get("attr3_1", ""))
        s_sinc = number(frow.get("sinc3_1", ""))
        s_intel = number(frow.get("intel3_1", ""))
        s_fun = number(frow.get("fun3_1", ""))
        s_amb = number(frow.get("amb3_1", ""))
        a_val = number(frow.get("age", ""))
        r_val = number(frow.get("race", ""))
        g_val = int(float(frow.get("gender", "0")))
        if np.isfinite(s_intel):
            person_self_intel[iid_int] = float(s_intel)
        if np.isfinite(s_amb):
            person_self_amb[iid_int] = float(s_amb)
        if np.isfinite(a_val):
            person_age[iid_int] = float(a_val)
        if np.isfinite(r_val):
            person_race[iid_int] = int(r_val)

        rmat = np.array(rec_by_iid[iid_int], dtype=np.float64)
        for col_j in range(5):
            bad = (rmat[:, col_j] < 1.0) | (rmat[:, col_j] > 10.0)
            rmat[bad, col_j] = np.nan
        j_means = np.nanmean(rmat[:, :5], axis=0)
        dec_ok = rmat[(rmat[:, 5] == 0.0) | (rmat[:, 5] == 1.0), 5]
        like_ok = rmat[(rmat[:, 6] >= 0.0) & (rmat[:, 6] <= 10.0), 6]
        if np.isfinite(j_means[2]):
            person_jury_intel[iid_int] = float(j_means[2])
        s_ag_vec = np.array([s_intel, s_fun, s_amb, s_attr], dtype=np.float64)
        j_ag_vec = np.array([j_means[2], j_means[3], j_means[4], j_means[0]], dtype=np.float64)
        if np.isfinite(s_ag_vec).all() and np.isfinite(j_ag_vec).all():
            s_ag = float(s_ag_vec.mean())
            j_ag = float(j_ag_vec.mean())
            oc_val = s_ag - j_ag
            person_overclaim[iid_int] = oc_val
            if len(dec_ok) > 0 and np.isfinite(j_means[1]):
                person_archetype_rows.append(
                    (g_val, s_ag, j_ag, float(j_means[1]), oc_val, float(dec_ok.mean()), float(np.nanmean(like_ok)))
                )

    ch_age_vec = np.array([person_age.get(int(c), np.nan) for c in chooser], dtype=np.float64)
    ch_race_vec = np.array([person_race.get(int(c), -1) for c in chooser], dtype=np.int32)
    ch_s_intel_vec = np.array([person_self_intel.get(int(c), np.nan) for c in chooser], dtype=np.float64)
    ch_s_amb_vec = np.array([person_self_amb.get(int(c), np.nan) for c in chooser], dtype=np.float64)
    ch_j_intel_vec = np.array([person_jury_intel.get(int(c), np.nan) for c in chooser], dtype=np.float64)
    tg_overclaim_vec = np.array([person_overclaim.get(int(p), np.nan) if np.isfinite(p) else np.nan for p in partner], dtype=np.float64)

    # 1. Dyadic Agency x Warmth (Sincerity) 4-Archetype Matrix
    five_ok = valid_decision & np.isfinite(own[:, :5]).all(axis=1)
    dyad_agency = (own[:, 2] + own[:, 1] + own[:, 3]) / 3.0
    sinc_vec = own[:, 4]
    agency_x_warmth = {}
    for ch_sx, dir_key in ((1.0, "male_chooser_to_female_target"), (0.0, "female_chooser_to_male_target")):
        base_m = five_ok & (sex == ch_sx)
        cells = {}
        for ckey, cmask in (
            ("warm_outlier_agency", base_m & (dyad_agency >= 8.0) & (sinc_vec >= 8.0)),
            ("cold_arrogant_high_agency", base_m & (dyad_agency >= 8.0) & (sinc_vec <= 6.0)),
            ("warm_humble_low_agency", base_m & (dyad_agency <= 6.0) & (sinc_vec >= 8.0)),
            ("low_agency_low_warmth", base_m & (dyad_agency <= 6.0) & (sinc_vec <= 6.0)),
        ):
            n_c = int(cmask.sum())
            y_c = int(yes[cmask].sum())
            lk_sub = like_raw[cmask & np.isfinite(like_raw)]
            cells[ckey] = {
                "n": n_c,
                "yes": y_c,
                "yes_share": float(y_c / n_c),
                "mean_like": float(lk_sub.mean()) if len(lk_sub) else 0.0,
            }
        warm_hi = cells["warm_outlier_agency"]["yes_share"]
        cold_hi = cells["cold_arrogant_high_agency"]["yes_share"]
        warm_lo = cells["warm_humble_low_agency"]["yes_share"]
        cells["coldness_backlash_drop_pp"] = 100.0 * (warm_hi - cold_hi)
        cells["warm_outlier_over_humble_low_pp"] = 100.0 * (warm_hi - warm_lo)
        cells["warm_outlier_to_humble_low_ratio"] = float(warm_hi / warm_lo)
        agency_x_warmth[dir_key] = cells

    agency_x_warmth["female_to_male_coldness_backlash_ratio"] = float(
        agency_x_warmth["male_chooser_to_female_target"]["coldness_backlash_drop_pp"]
        / agency_x_warmth["female_chooser_to_male_target"]["coldness_backlash_drop_pp"]
    )

    # 2. Audience Age & Race/Ethnicity Modulation of Unbacked Overclaiming & Ambition
    aud_ok = five_ok & np.isfinite(tg_overclaim_vec)
    pred_names = ["attr", "sinc", "intel", "fun", "amb", "target_overclaim"]
    pred_vecs = [own[:, 0], own[:, 4], own[:, 2], own[:, 1], own[:, 3], tg_overclaim_vec]
    by_chooser_age = {}
    for age_key, amask in (
        ("age_le_24", aud_ok & (ch_age_vec <= 24.0)),
        ("age_25_28", aud_ok & (ch_age_vec >= 25.0) & (ch_age_vec <= 28.0)),
        ("age_ge_29", aud_ok & (ch_age_vec >= 29.0)),
    ):
        by_chooser_age[age_key] = _ols_pp(yes[amask], [v[amask] for v in pred_vecs], pred_names)

    oc_le24 = abs(by_chooser_age["age_le_24"]["coefficients"]["target_overclaim"]["slope_pp"])
    oc_ge29 = abs(by_chooser_age["age_ge_29"]["coefficients"]["target_overclaim"]["slope_pp"])
    by_chooser_age["mature_to_young_overclaim_penalty_ratio"] = float(oc_ge29 / max(oc_le24, 1e-6))

    by_chooser_race = {}
    for r_code, r_key in ((1, "black"), (2, "white_euro"), (3, "latino"), (4, "asian"), (6, "other")):
        rmask = aud_ok & (ch_race_vec == r_code)
        if int(rmask.sum()) >= 200:
            by_chooser_race[r_key] = _ols_pp(yes[rmask], [v[rmask] for v in pred_vecs], pred_names)

    # 3. Asymmetric Status Threat ("Outsmarting / Out-Ambitioning the Chooser")
    st_ok = five_ok & np.isfinite(ch_s_intel_vec) & np.isfinite(ch_s_amb_vec)
    outsmart_vec = (own[:, 2] > ch_s_intel_vec).astype(np.float64)
    outamb_vec = (own[:, 3] > ch_s_amb_vec).astype(np.float64)
    st_names = ["attr", "sinc", "intel", "fun", "amb", "target_outsmarts_chooser", "target_outambitions_chooser"]
    st_vecs = [own[:, 0], own[:, 4], own[:, 2], own[:, 1], own[:, 3], outsmart_vec, outamb_vec]
    status_threat = {}
    for ch_sx, dir_key in ((1.0, "male_chooser_evaluating_woman"), (0.0, "female_chooser_evaluating_man")):
        smask = st_ok & (sex == ch_sx)
        status_threat[dir_key] = _ols_pp(yes[smask], [v[smask] for v in st_vecs], st_names)

    # Male chooser insecurity stratification on partner ambition & intelligence
    dm_ok = five_ok & (sex == 1.0) & np.isfinite(ch_s_amb_vec) & np.isfinite(ch_j_intel_vec)
    med_m_jintel = float(np.median(ch_j_intel_vec[dm_ok]))
    male_insecurity_strata = {}
    base_names_5 = ["attr", "sinc", "intel", "fun", "amb"]
    base_vecs_5 = [own[:, 0], own[:, 4], own[:, 2], own[:, 1], own[:, 3]]
    for skey, smask in (
        ("ambition_insecure_men_le7", dm_ok & (ch_s_amb_vec <= 7.0)),
        ("high_ambition_men_ge9", dm_ok & (ch_s_amb_vec >= 9.0)),
        ("lower_jury_intel_men", dm_ok & (ch_j_intel_vec < med_m_jintel)),
        ("higher_jury_intel_men", dm_ok & (ch_j_intel_vec >= med_m_jintel)),
    ):
        male_insecurity_strata[skey] = _ols_pp(yes[smask], [v[smask] for v in base_vecs_5], base_names_5)
    status_threat["male_chooser_strata"] = male_insecurity_strata

    confidence_and_backlash = {
        "n_person_archetypes": len(person_archetype_rows),
        "agency_x_warmth_dyadic": agency_x_warmth,
        "audience_age_modulation": by_chooser_age,
        "audience_race_modulation": by_chooser_race,
        "status_threat": status_threat,
    }

    checklist_and_hindsight = {
        "hobby_checklist_vs_live_wavelength": {
            "n_common_dyad_decisions": int(common_cs_mask.sum()),
            "r_objective_hobby_corr_yes": r_ic_yes,
            "r_live_shared_wavelength_yes": r_sh_yes,
            "r_objective_hobby_vs_live_wavelength": r_ic_sh,
            "live_to_checklist_r_ratio": float(r_sh_yes["estimate"] / max(abs(r_ic_yes["estimate"]), 1e-6)),
            "chooser_fe_slopes": checklist_fe_slopes,
        },
        "objective_sat_blindness": {
            "n_participants_with_sat": int(len(sat_by_iid)),
            "n_dates_with_partner_sat": int(n_sd_sat),
            "r_stranger_intel_vs_partner_sat": outcome(r_sat_intel_pt, b_sat_intel),
            "stranger_intel_sat_r2": float(r_sat_intel_pt ** 2),
            "r_partner_sat_vs_yes": outcome(r_sat_yes_pt, b_sat_yes),
            "n_targets_ge10_raters": int(n_ps_targets),
            "r_panel_intel_vs_partner_sat": outcome(r_panel_sat_pt, b_panel_sat),
            "panel_intel_sat_r2": float(r_panel_sat_pt ** 2),
            "panel_to_single_r2_ratio": float((r_panel_sat_pt ** 2) / max(r_sat_intel_pt ** 2, 1e-9)),
            "n_matched_rows_both_sat": int(both_sat_match.sum()),
            "r_matched_couple_sat": r_match_sat,
            "r_chooser_sat_grading_strictness": r_chooser_strict,
        },
        "longitudinal_hindsight_waves": hindsight_waves,
        "within_person_shifts": paired_shifts,
        "cross_sex_misperception": cross_sex_misperception,
        "same_sex_pluralistic_ignorance": pluralistic_ignorance,
        "post_event_followup_3_4_weeks": {
            "n_wave3_respondents": int(len(fu_arr)),
            "n_wave3_with_at_least_one_match": int(len(matched_fu)),
            "went_on_followup_date_share_among_matched": float(matched_fu[:, 2].mean()),
            "called_or_was_called_share_among_matched": float(
                np.mean((np.nan_to_num(matched_fu[:, 3], nan=0.0) > 0) | (np.nan_to_num(matched_fu[:, 4], nan=0.0) > 0))
            ),
            "initiation_by_sex": initiation_by_sex,
        },
    }

    return {
        "input_sha256": SHA256, "mirror": MIRROR, "records": len(data), "sessions": len(sessions),
        "participants": len(np.unique(chooser)), "bootstrap": {"unit": "session", "draws": BOOTSTRAPS, "seed": SEED},
        "cohorts": {
            "quintiles": "Within each chooser, midpoint ranks of partners' mean 1–10 looks ratings from at least three other session peers. Quintiles are 20-point rank bands; ties remain together. Rates count yes decisions among rows with an observed binary decision and available panel rating.",
            "slopes": "Within-chooser linear probability models for yes, by chooser gender, adjusting for looks, fun, intelligence, ambition, sincerity and shared interests. Own and leave-one-out panel ratings use separate, common-across-gender SD scales. All six ratings and two usable decisions per chooser are required.",
            "stated": "First row per participant. Mean looks points among complete allocations after rescaling each total to 100.",
            "dyadic_chemistry": "Two-way bipartite Social Relations Model variance partition within session x chooser-sex blocks (with exact degrees-of-freedom finite-sample correction), residual mutual dyadic liking correlation across matched pairs, and within-chooser yes slopes adding perceived reciprocal liking (prob).",
            "checklist_and_hindsight": "Comparison of pre-event objective 17-hobby profile correlation (int_corr) against live 4-minute shared wavelength (shar), plus normalized 100-point preference trajectories across pre-event (T1), 24-hour hindsight (T2), and 3-4-week post-event follow-up (T3).",
            "confidence_and_backlash": "Dyadic agency x sincerity decision rates, chooser age/race OLS modulation on unbacked self-overclaiming (self_agency - jury_agency), and chooser status-comparison regressions when target intelligence or ambition exceeds chooser self-ratings.",
        },
        "counts": {"valid_binary_decisions": int(valid_decision.sum()), "decisions_with_looks_panel": int(curve_rows.sum())},
        "yes_by_panel_quintile": curves, "quintile_contrast_men_minus_women": curve_gap,
        "adjusted_slopes": slopes, "stated_looks": stated,
        "dyadic_chemistry": dyadic_chemistry,
        "checklist_and_hindsight": checklist_and_hindsight,
        "confidence_and_backlash": confidence_and_backlash,
    }
