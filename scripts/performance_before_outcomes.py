#!/usr/bin/env python3
"""Read the saved HON_INDEX snapshot in place and emit aggregate evidence."""
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MIN_CELL = 100
SEXES = ("Female", "Male")
BANDS = [(18, 19), *((start, start + 4) for start in range(20, 80, 5))]
LIFE_WINDOWS = [(18, 24), (25, 29), (30, 39), (40, 49), (50, 59), (60, 69)]
HEADLINES = [(25, 29), (45, 49)]
ANSWERS = ("want", "do_not_want", "undecided_open", "nonresponse")
EXPORTS = Path("exports")
PARTIAL = Path("faceflow/output/partial_exports")
FULL_FACEFLOW = Path("faceflow/output/summary.json")
BENCHMARK = Path("experiment/LEADERBOARD.json")
REVIEW = Path("reports/beauty_review")

US_STATES = frozenset({
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "District of Columbia", "Washington DC", "Florida", "Georgia", "Hawaii",
    "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
    "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
    "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin",
    "Wyoming",
})
UK_STATES = frozenset({"UK", "United Kingdom", "England", "Scotland", "Wales", "Northern Ireland"})
CA_PROVINCES = frozenset({
    "Ontario", "British Columbia", "Alberta", "Quebec", "Manitoba", "Saskatchewan",
    "Nova Scotia", "New Brunswick", "Newfoundland", "Newfoundland and Labrador",
    "Prince Edward Island",
})
BR_STATES = frozenset({
    "Sao Paulo", "Rio de Janeiro", "Minas Gerais", "Parana", "Rio Grande do Sul",
    "Bahia", "Santa Catarina", "Pernambuco", "Ceara", "Goias", "Distrito Federal",
    "Espirito Santo", "Para", "Amazonas", "Mato Grosso", "Mato Grosso do Sul",
})
AU_STATES = frozenset({
    "New South Wales", "Victoria", "Queensland", "Western Australia", "South Australia",
    "Tasmania", "Australian Capital Territory", "Northern Territory",
})

POV_INTENT_LABELS = (
    ("marry", "wants to find someone to marry"),
    ("actively_seeking", "is actively seeking a relationship"),
    ("looking", "is looking for a relationship"),
    ("casual", "wants to date but nothing serious"),
    ("not_seeking", "isn't seeking a relationship or any kind of commitment"),
    ("anything", "anything"),
)
POV_MARITAL_LABELS = (
    ("single", "Single"),
    ("divorced", "Divorced"),
    ("separated", "Separated"),
    ("widowed", "Widowed"),
    ("living_together", "Living Together"),
    ("not_single", "Not Single/Not Looking"),
    ("married", "Married"),
)
POV_EDU_ORDER = (
    "High school",
    "Some college",
    "Associates Degree",
    "Some University",
    "Bachelors Degree",
    "Graduate Degree",
    "Masters Degree",
    "PhD / Post Doctoral",
)
POV_AMBITION_ORDER = (
    "Not Ambitious",
    "Somewhat Ambitious",
    "Ambitious",
    "Very Ambitious",
)
POV_BODY_ORDER = (
    "Thin body type",
    "Athletic body type",
    "Average body type",
    "A Few Extra Pounds body type",
    "Big & Tall/BBW body type",
    "Prefer Not To Say body type",
)
POV_CAR_ORDER = ("Yes", "No", "Prefer Not To Say")
FT_IN_RE = re.compile(r"^(\d)'\s*(\d{1,2})\"\s*\((\d{2,3})\s*cm\)$")


def read(name, columns, folder=EXPORTS):
    return pq.read_table(folder / f"{name}.parquet", columns=columns)


def saved(path):
    return json.loads(Path(path).read_text())


def counts(table, *keys):
    rows = table.group_by(keys, use_threads=False).aggregate([([], "count_all")]).to_pylist()
    return sorted(rows, key=lambda row: tuple(str(row[key]) for key in keys))


def tally(table, key):
    return {row[key]: row["count_all"] for row in counts(table, key)}


def age_band(lo, hi):
    return f"{lo}-{hi}"


def eligible_age(age):
    return (age >= 18) & (age <= 79)


def response_codes(values):
    return pc.fill_null(pc.index_in(values, value_set=pa.array(
        ["Want children", "Does not want children", "Undecided/Open"])), 3)


def intent_cell(values):
    n = sum(values)
    if n < MIN_CELL:
        return None
    return {"n": n, "counts": dict(zip(ANSWERS, values)),
            "shares": {key: v / n for key, v in zip(ANSWERS, values)}}


def categorical_cell(keys, bucket_counts):
    n = int(sum(bucket_counts))
    if n < MIN_CELL:
        return None
    return {
        "n": n,
        "counts": {k: int(v) for k, v in zip(keys, bucket_counts)},
        "shares": {k: float(v / n) for k, v in zip(keys, bucket_counts)},
    }


def rate_cell(num, den):
    den_i = int(den)
    if den_i < MIN_CELL:
        return None
    num_i = int(num)
    return {"n": den_i, "count": num_i, "share": float(num_i / den_i)}


def gini_and_deciles(values):
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = len(x)
    total_sum = float(x.sum())
    idx = np.arange(1, n + 1, dtype=np.float64)
    gini = float((2.0 * np.sum(idx * x) / (n * total_sum)) - (n + 1.0) / n)
    edges = np.linspace(0, n, 11, dtype=np.int64)
    deciles = [float(x[edges[i]:edges[i + 1]].sum() / total_sum) for i in range(10)]
    return gini, deciles


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def map_pov_country(state_str):
    if not state_str:
        return "other_or_blank"
    s = state_str.strip()
    if s in US_STATES:
        return "United States"
    if s in UK_STATES:
        return "United Kingdom"
    if s in CA_PROVINCES:
        return "Canada"
    if s in BR_STATES:
        return "Brazil"
    if s in AU_STATES:
        return "Australia"
    return "other_or_blank"


def pov():
    columns = [
        "source_record_no", "id", "age", "gender", "seeking", "haschildren",
        "wantchildren", "maritalstatus", "intent", "for", "ambition",
        "college_id", "relationshipage_id", "height", "body", "car", "state_id",
        "headline", "description", "interests",
    ]
    table = read("POV_source_fields", columns)
    keys = table["id"]
    assert sequence(table["source_record_no"], len(table))
    accounts = {
        "rows": len(table), "distinct_nonnull_keys": pc.count_distinct(keys).as_py(),
        "blank_or_null_keys": total(pc.or_(pc.is_null(keys), pc.fill_null(pc.equal(keys, ""), False))),
    }
    assert accounts["rows"] == accounts["distinct_nonnull_keys"] and accounts["blank_or_null_keys"] == 0
    age = pc.cast(pc.if_else(pc.match_substring_regex(table["age"], r"^[0-9]{1,3}$"),
                             table["age"], None), pa.int16())
    sex = table["gender"]
    opposite = pc.or_kleene(
        pc.and_kleene(pc.equal(sex, "Female"), pc.equal(table["seeking"], "Man")),
        pc.and_kleene(pc.equal(sex, "Male"), pc.equal(table["seeking"], "Woman")))
    parent = pc.fill_null(pc.index_in(table["haschildren"],
                          value_set=pa.array(["No", "Yes", "All my kids are over 18"])), 3)
    answer = response_codes(table["wantchildren"])
    single = pc.fill_null(pc.equal(table["maritalstatus"], "Single"), False)
    full = pa.table({"age": age, "sex": sex, "opposite": pc.fill_null(opposite, False),
                     "parent": parent, "answer": answer, "single": single})
    age_ok = pc.fill_null(pc.and_(pc.greater_equal(age, 18), pc.less_equal(age, 79)), False)
    sex_ok = pc.is_in(sex, value_set=pa.array(SEXES))
    base = full.filter(pc.and_(age_ok, sex_ok))
    scoped_mask = pc.and_(pc.and_(age_ok, sex_ok), full["opposite"])
    scoped = base.filter(base["opposite"])
    groups = counts(base, *base.column_names)
    responses = Counter()
    for row in groups:
        key = tuple(row[k] for k in ("age", "sex", "opposite", "parent", "answer", "single"))
        responses[key] = row["count_all"]

    def intent(lo, hi, gender, parents, *, restrict=True, only_single=False, answered=False):
        buckets = [0] * 4
        for (rec_age, rec_sex, seeking, rec_parent, rec_answer, rec_single), n in responses.items():
            if (lo <= rec_age <= hi and rec_sex == gender and rec_parent in parents and
                    (seeking or not restrict) and (rec_single or not only_single) and
                    (rec_answer != 3 or not answered)):
                buckets[rec_answer] += n
        return intent_cell(buckets)

    bins = {}
    for lo, hi in BANDS:
        by_sex = {}
        for gender in SEXES:
            parenthood = Counter()
            for (recorded_age, recorded_sex, seeking, parent_code, _, _), n in responses.items():
                if lo <= recorded_age <= hi and recorded_sex == gender and seeking:
                    parenthood[parent_code] += n
            known = sum(parenthood[i] for i in (0, 1, 2))
            total_profiles = parenthood.total()
            by_sex[gender] = {
                "parenthood": ({"n": known, "with_children": parenthood[1] + parenthood[2],
                                "adult_children": parenthood[2],
                                "share": (parenthood[1] + parenthood[2]) / known}
                               if known >= MIN_CELL else None),
                "all_profiles": total_profiles if total_profiles >= MIN_CELL else None,
                "unknown_parenthood": parenthood[3] if total_profiles >= MIN_CELL else None,
                "without_children": intent(lo, hi, gender, (0,)),
                "with_children": intent(lo, hi, gender, (1, 2)),
            }
        bins[age_band(lo, hi)] = by_sex

    sensitivity = {}
    for lo, hi in HEADLINES:
        scenarios = {}
        for name, options in {
            "main": {}, "single": {"only_single": True}, "all_seeking": {"restrict": False},
            "answered": {"answered": True}, "exact_age": {},
            "single_answered_exact_age": {"only_single": True, "answered": True},
        }.items():
            by_sex = {sex_label: intent(lo, hi, sex_label, (0,), **options) for sex_label in SEXES}
            exact = None
            if "exact_age" in name:
                exact = {str(a): {sex_label: intent(a, a, sex_label, (0,), **options) for sex_label in SEXES}
                         for a in range(lo, hi + 1)}
                if all(cell for day in exact.values() for cell in day.values()):
                    age_n = {a: sum(cell["n"] for cell in day.values()) for a, day in exact.items()}
                    weights = {a: n / sum(age_n.values()) for a, n in age_n.items()}
                    shares = {sex_label: sum(weights[a] * day[sex_label]["shares"]["want"] for a, day in exact.items())
                              for sex_label in SEXES}
                else:
                    weights = shares = None
            else:
                weights = None
                shares = {sex_label: cell["shares"]["want"] for sex_label, cell in by_sex.items()} if all(by_sex.values()) else None
            scenarios[name] = {
                "responses": by_sex, "want_shares": shares,
                "women_minus_men_pp": (100 * (shares["Female"] - shares["Male"]) if shares else None),
            }
            if exact is not None:
                scenarios[name].update(exact_age_cells=exact, weights=weights)
        sensitivity[age_band(lo, hi)] = scenarios

    parent_counts = {int(k): n for k, n in tally(scoped, "parent").items()}
    eligible = {"age_18_79": total(age_ok), "age_and_gender": len(base), "opposite_gender": len(scoped),
                "without_children": parent_counts.get(0, 0),
                "with_children": parent_counts.get(1, 0) + parent_counts.get(2, 0),
                "adult_children": parent_counts.get(2, 0), "unknown_parenthood": parent_counts.get(3, 0)}

    # --- Extended Mate-Market, Commitment, Union Durability, Trait, Height & Geography Analyses ---
    sub = table.filter(scoped_mask)
    n_sub = len(sub)
    age_np = pc.cast(sub["age"], pa.int16()).to_numpy()
    is_f = pc.equal(sub["gender"], "Female").to_numpy()
    parent_np = full.filter(scoped_mask)["parent"].to_numpy()
    answer_np = full.filter(scoped_mask)["answer"].to_numpy()

    def optional_pylist(tbl, col_name):
        return tbl[col_name].to_pylist() if col_name in tbl.column_names else [None] * len(tbl)

    intent_keys = tuple(k for k, _ in POV_INTENT_LABELS) + ("other_or_blank",)
    intent_raw = optional_pylist(sub, "intent")
    intent_map = {raw: idx for idx, (_, raw) in enumerate(POV_INTENT_LABELS)}
    intent_code = np.array([intent_map.get(v, len(POV_INTENT_LABELS)) for v in intent_raw], dtype=np.int8)
    serious_mask = (intent_code == 0) | (intent_code == 1) | (intent_code == 2)
    marry_mask = intent_code == 0
    casual_mask = intent_code == 3

    marital_keys = tuple(k for k, _ in POV_MARITAL_LABELS) + ("other_or_blank",)
    marital_map = {raw: idx for idx, (_, raw) in enumerate(POV_MARITAL_LABELS)}
    marital_code = np.array([marital_map.get(v, len(POV_MARITAL_LABELS)) for v in optional_pylist(sub, "maritalstatus")], dtype=np.int8)
    ever_partnered_mask = np.isin(marital_code, [1, 2, 3, 4, 6])

    dur_keys = ("under_1y", "over_1_to_2y", "over_3_to_5y", "over_6_to_9y", "over_10y", "other_or_blank")
    dur_map = {
        "Under 1 year": 0,
        "Over 1 year": 1, "Over 2 years": 1,
        "Over 3 years": 2, "Over 4 years": 2, "Over 5 years": 2,
        "Over 6 years": 3, "Over 7 years": 3, "Over 8 years": 3, "Over 9 years": 3,
        "Over 10 years": 4,
    }
    dur_code = np.array([dur_map.get(v, 5) for v in optional_pylist(sub, "relationshipage_id")], dtype=np.int8)
    over_3y_mask = (dur_code == 2) | (dur_code == 3) | (dur_code == 4)
    over_10y_mask = dur_code == 4

    for_keys = ("long_term", "dating", "friends", "hang_out", "other_or_blank")
    for_map = {"Long Term": 0, "Dating": 1, "Friends": 2, "Hang Out": 3}
    for_code = np.array([for_map.get(v, 4) for v in optional_pylist(sub, "for")], dtype=np.int8)

    market_windows = {}
    for lo, hi in [*BANDS, *LIFE_WINDOWS]:
        wkey = age_band(lo, hi)
        if wkey in market_windows:
            continue
        in_band = (age_np >= lo) & (age_np <= hi)
        nf = int((in_band & is_f).sum())
        nm = int((in_band & ~is_f).sum())
        w_entry = {
            "male_to_female_ratio": float(nm / nf) if nf >= MIN_CELL and nm >= MIN_CELL else None,
            "by_sex": {},
        }
        for sex_label, smask in (("Female", is_f), ("Male", ~is_f)):
            sel = in_band & smask
            nsel = int(sel.sum())
            if nsel < MIN_CELL:
                w_entry["by_sex"][sex_label] = None
                continue
            ic = np.bincount(intent_code[sel], minlength=len(intent_keys))
            mc = np.bincount(marital_code[sel], minlength=len(marital_keys))
            dc = np.bincount(dur_code[sel], minlength=len(dur_keys))
            fc = np.bincount(for_code[sel], minlength=len(for_keys))
            w_entry["by_sex"][sex_label] = {
                "n": nsel,
                "intent": categorical_cell(intent_keys, ic),
                "for": categorical_cell(for_keys, fc),
                "marital_status": categorical_cell(marital_keys, mc),
                "union_durability": categorical_cell(dur_keys, dc),
                "serious_commitment": rate_cell(serious_mask[sel].sum(), nsel),
                "ever_partnered": rate_cell(ever_partnered_mask[sel].sum(), nsel),
                "over_3y_union": rate_cell(over_3y_mask[sel].sum(), nsel),
                "over_10y_union": rate_cell(over_10y_mask[sel].sum(), nsel),
            }
        market_windows[wkey] = w_entry

    edu_list = optional_pylist(sub, "college_id")
    amb_list = optional_pylist(sub, "ambition")
    body_list = optional_pylist(sub, "body")
    car_list = optional_pylist(sub, "car")
    athletic_mask = np.array([v == "Athletic body type" for v in body_list], dtype=bool)
    very_amb_mask = np.array([v == "Very Ambitious" for v in amb_list], dtype=bool)

    def summarize_trait_slice(sel):
        nsel = int(sel.sum())
        if nsel < MIN_CELL:
            return None
        known_par = sel & (parent_np != 3)
        childless = sel & (parent_np == 0)
        return {
            "n": nsel,
            "parenthood": rate_cell(((parent_np == 1) | (parent_np == 2))[known_par].sum(), known_par.sum()),
            "childless_want_children": rate_cell((answer_np[childless] == 0).sum(), childless.sum()),
            "want_children_all": rate_cell((answer_np[sel] == 0).sum(), nsel),
            "serious_commitment": rate_cell(serious_mask[sel].sum(), nsel),
            "marry_intent": rate_cell(marry_mask[sel].sum(), nsel),
            "casual_intent": rate_cell(casual_mask[sel].sum(), nsel),
            "ever_partnered": rate_cell(ever_partnered_mask[sel].sum(), nsel),
            "over_3y_union": rate_cell(over_3y_mask[sel].sum(), nsel),
            "over_10y_union": rate_cell(over_10y_mask[sel].sum(), nsel),
            "athletic_body": rate_cell(athletic_mask[sel].sum(), nsel),
        }

    def build_trait_group(values_list, ordered_cats, age_lo, age_hi):
        val_map = {cat: i for i, cat in enumerate(ordered_cats)}
        codes = np.array([val_map.get(v, len(ordered_cats)) for v in values_list], dtype=np.int16)
        in_age = (age_np >= age_lo) & (age_np <= age_hi)
        out = {}
        for idx_cat, cat in enumerate(ordered_cats):
            cmask = in_age & (codes == idx_cat)
            out[cat] = {
                sex_label: summarize_trait_slice(cmask & smask)
                for sex_label, smask in (("Female", is_f), ("Male", ~is_f))
            }
        return out

    traits_and_postponement = {
        "education_ages_30_49": build_trait_group(edu_list, POV_EDU_ORDER, 30, 49),
        "ambition_ages_25_49": build_trait_group(amb_list, POV_AMBITION_ORDER, 25, 49),
        "body_ages_25_49": build_trait_group(body_list, POV_BODY_ORDER, 25, 49),
        "car_ages_25_49": build_trait_group(car_list, POV_CAR_ORDER, 25, 49),
    }

    # --- Height Percentile, 6'0" Staging Cliff & Outcome Gradients ---
    height_raw = optional_pylist(sub, "height")
    height_in = np.full(n_sub, -1, dtype=np.int16)
    height_cm = np.full(n_sub, -1, dtype=np.int16)
    height_audit_code = np.zeros(n_sub, dtype=np.int8)  # 0=explicit_plausible, 1=censored_under_5ft, 2=implausible_999cm, 3=other_or_blank
    for idx_h, h_str in enumerate(height_raw):
        if not h_str:
            height_audit_code[idx_h] = 3
            continue
        m = FT_IN_RE.match(h_str.strip())
        if m:
            ft_v, in_v, cm_v = int(m.group(1)), int(m.group(2)), int(m.group(3))
            tot_in = ft_v * 12 + in_v
            if 60 <= tot_in <= 84 and 150 <= cm_v <= 215:
                height_in[idx_h] = tot_in
                height_cm[idx_h] = cm_v
                height_audit_code[idx_h] = 0
            elif cm_v == 999:
                height_audit_code[idx_h] = 2
            else:
                height_audit_code[idx_h] = 3
        elif h_str.startswith("< 5'"):
            height_audit_code[idx_h] = 1
        else:
            height_audit_code[idx_h] = 3

    height_percentiles = {"audit": {}, "by_sex": {}}
    audit_names = ("explicit_plausible", "censored_under_5ft", "implausible_999cm", "other_or_blank")
    for sex_label, smask in (("Female", is_f), ("Male", ~is_f)):
        ac = np.bincount(height_audit_code[smask], minlength=4)
        height_percentiles["audit"][sex_label] = {k: int(v) for k, v in zip(audit_names, ac)}

        valid_h = smask & (height_audit_code == 0)
        vals_in = height_in[valid_h].astype(np.float64)
        vals_cm = height_cm[valid_h].astype(np.float64)
        n_valid_h = len(vals_in)
        if n_valid_h < MIN_CELL:
            height_percentiles["by_sex"][sex_label] = None
            continue
        ranks_pct = midranks(vals_in) * 100.0
        full_pct = np.full(n_sub, np.nan)
        full_pct[valid_h] = ranks_pct

        # Robust Gaussian reference from discrete median & IQR (resistant to local 6'0" rounding heap)
        q25, q50, q75 = np.quantile(vals_in, [0.25, 0.50, 0.75])
        mu_fit = float(vals_in.mean())
        sd_fit = float(vals_in.std(ddof=1))
        rows_by_inch = []
        for inch in range(60, 85):
            imask = valid_h & (height_in == inch)
            ni = int(imask.sum())
            if ni < MIN_CELL:
                continue
            ft_v, rem_v = divmod(inch, 12)
            gauss_p = float(norm_cdf((inch + 0.5 - mu_fit) / sd_fit) - norm_cdf((inch - 0.5 - mu_fit) / sd_fit)) if sd_fit > 0 else 0.0
            obs_p = float(ni / n_valid_h)
            prime_sel = imask & (age_np >= 30) & (age_np <= 49)
            rows_by_inch.append({
                "inches": inch,
                "label": f"{ft_v}'{rem_v}\"",
                "cm": int(np.median(height_cm[imask])),
                "n_all_ages": ni,
                "observed_share": obs_p,
                "gaussian_expected_share": gauss_p,
                "observed_to_gaussian_ratio": float(obs_p / gauss_p) if gauss_p > 0 else None,
                "mean_height_percentile": float(full_pct[imask].mean()),
                "outcomes_ages_30_49": summarize_trait_slice(prime_sel),
                "very_ambitious_ages_30_49": rate_cell(very_amb_mask[prime_sel].sum(), prime_sel.sum()),
            })
        n_71 = int((valid_h & (height_in == 71)).sum())
        n_72 = int((valid_h & (height_in == 72)).sum())
        n_73 = int((valid_h & (height_in == 73)).sum())
        height_percentiles["by_sex"][sex_label] = {
            "n_valid": n_valid_h,
            "mean_inches": mu_fit,
            "sd_inches": sd_fit,
            "mean_cm": float(vals_cm.mean()),
            "quantiles_inches": [float(q25), float(q50), float(q75)],
            "cliff_5ft11_6ft0_6ft1": {
                "n_5ft11": n_71,
                "n_6ft0": n_72,
                "n_6ft1": n_73,
                "drop_6ft0_to_6ft1_pct": float(100.0 * (n_73 - n_72) / n_72) if n_72 >= MIN_CELL else None,
            },
            "by_inch": rows_by_inch,
        }

    # --- 5-Country Geography Replication via unambiguous state_id ---
    geo_countries = ("United States", "United Kingdom", "Canada", "Brazil", "Australia")
    geo_mapped = [map_pov_country(s) for s in optional_pylist(sub, "state_id")]
    geo_map_idx = {c: i for i, c in enumerate(geo_countries)}
    geo_code = np.array([geo_map_idx.get(c, len(geo_countries)) for c in geo_mapped], dtype=np.int8)
    geography = {"country_totals": {}, "replications": {}}
    for idx_c, cname in enumerate(geo_countries):
        cmask = geo_code == idx_c
        geography["country_totals"][cname] = {
            sex_label: int((cmask & smask).sum()) if int((cmask & smask).sum()) >= MIN_CELL else None
            for sex_label, smask in (("Female", is_f), ("Male", ~is_f))
        }
        rep = {}
        for lo, hi in ((25, 29), (40, 49), (45, 49)):
            bkey = age_band(lo, hi)
            bmask = cmask & (age_np >= lo) & (age_np <= hi)
            by_s = {}
            for sex_label, smask in (("Female", is_f), ("Male", ~is_f)):
                sel = bmask & smask
                childless = sel & (parent_np == 0)
                by_s[sex_label] = {
                    "all_n": int(sel.sum()) if int(sel.sum()) >= MIN_CELL else None,
                    "serious_commitment": rate_cell(serious_mask[sel].sum(), sel.sum()),
                    "childless_want_children": rate_cell((answer_np[childless] == 0).sum(), childless.sum()),
                }
            f_cw = by_s["Female"]["childless_want_children"]
            m_cw = by_s["Male"]["childless_want_children"]
            f_sc = by_s["Female"]["serious_commitment"]
            m_sc = by_s["Male"]["serious_commitment"]
            rep[bkey] = {
                "by_sex": by_s,
                "childless_want_women_minus_men_pp": (
                    float(100.0 * (f_cw["share"] - m_cw["share"])) if (f_cw and m_cw) else None
                ),
                "serious_women_minus_men_pp": (
                    float(100.0 * (f_sc["share"] - m_sc["share"])) if (f_sc and m_sc) else None
                ),
            }
        geography["replications"][cname] = rep

    # --- Exact Childless Fertility-Intent Age Crossover ---
    band_diffs = []
    for lo, hi in BANDS:
        bkey = age_band(lo, hi)
        fc = bins[bkey]["Female"]["without_children"]
        mc = bins[bkey]["Male"]["without_children"]
        if fc and mc:
            band_diffs.append((0.5 * (lo + hi), bkey, 100.0 * (fc["shares"]["want"] - mc["shares"]["want"])))
    crossover_age_band_interp = None
    for (m0, _, d0), (m1, _, d1) in zip(band_diffs[:-1], band_diffs[1:]):
        if d0 > 0.0 >= d1:
            crossover_age_band_interp = float(m0 + (0.0 - d0) * (m1 - m0) / (d1 - d0))
            break

    single_year_childless = {}
    for exact_a in range(35, 48):
        amask = (age_np == exact_a) & (parent_np == 0)
        fc_a = rate_cell((answer_np[amask & is_f] == 0).sum(), (amask & is_f).sum())
        mc_a = rate_cell((answer_np[amask & ~is_f] == 0).sum(), (amask & ~is_f).sum())
        if fc_a and mc_a:
            single_year_childless[str(exact_a)] = {
                "Female": fc_a,
                "Male": mc_a,
                "women_minus_men_pp": float(100.0 * (fc_a["share"] - mc_a["share"])),
            }
    first_negative_exact_age = next(
        (int(a) for a, row in single_year_childless.items() if row["women_minus_men_pp"] <= 0.0),
        None,
    )
    childless_crossover = {
        "band_midpoint_interpolated_age": crossover_age_band_interp,
        "first_nonpositive_exact_age": first_negative_exact_age,
        "exact_ages_35_47": single_year_childless,
    }

    # --- Self-Written Human Voice Essays (headline + description + interests, >=30 chars) ---
    empty_txt = pa.array([""] * len(sub), type=pa.string())
    txt_head = pc.fill_null(sub["headline"], "") if "headline" in sub.column_names else empty_txt
    txt_desc = pc.fill_null(sub["description"], "") if "description" in sub.column_names else empty_txt
    txt_intr = pc.fill_null(sub["interests"], "") if "interests" in sub.column_names else empty_txt
    txt_trimmed = pc.utf8_trim_whitespace(pc.binary_join_element_wise(txt_head, txt_desc, txt_intr, " "))
    has_essay = pc.greater_equal(pc.utf8_length(txt_trimmed), 30).to_numpy(zero_copy_only=False)
    txt_lower = pc.utf8_lower(txt_trimmed)

    voice_themes = (
        ("family_kids_home", "Family, children, or home", r"\b(family|kids|children|marriage|husband|wife|home)\b"),
        ("honest_loyal_trust", "Honesty, loyalty, or trust", r"\b(honest|honesty|loyal|loyalty|trust|trustworthy|faithful)\b"),
        ("laugh_humor_smile", "Laughter, humor, or smile", r"\b(laugh|laughing|laughter|humor|humour|funny|smile)\b"),
        ("kind_caring_loving", "Kindness, warmth, or caring", r"\b(kind|kindness|caring|sweet|warm|compassionate|loving|affectionate|gentle)\b"),
        ("best_friend_soulmate", "Best friend or soulmate", r"\b(soulmate|soul mate|best friend|partner in crime|grow old)\b"),
        ("no_games_no_drama", "No games, drama, or cheating", r"\b(no games|no drama|cheat|cheater|liar)\b"),
        ("tall_6ft_mention", "6 ft or tall stature", r"(6\s*ft|6'0|6\s*foot|\btall\s+(man|guy|men)\b)"),
        ("rich_six_figure", "Wealth or six-figure status", r"\b(rich|wealthy|millionaire|six figure|6 figure|generous)\b"),
    )
    voice_overall = {}
    voice_by_window = {age_band(lo, hi): {} for lo, hi in LIFE_WINDOWS}
    for tkey, tlabel, pattern in voice_themes:
        hit = pc.fill_null(pc.match_substring_regex(txt_lower, pattern), False).to_numpy(zero_copy_only=False)
        voice_overall[tkey] = {
            "label": tlabel,
            "Pooled": rate_cell(hit[has_essay].sum(), has_essay.sum()),
            "Female": rate_cell(hit[has_essay & is_f].sum(), (has_essay & is_f).sum()),
            "Male": rate_cell(hit[has_essay & ~is_f].sum(), (has_essay & ~is_f).sum()),
        }
        for lo, hi in LIFE_WINDOWS:
            wkey = age_band(lo, hi)
            wmask = has_essay & (age_np >= lo) & (age_np <= hi)
            voice_by_window[wkey][tkey] = {
                "Female": rate_cell(hit[wmask & is_f].sum(), (wmask & is_f).sum()),
                "Male": rate_cell(hit[wmask & ~is_f].sum(), (wmask & ~is_f).sum()),
            }

    def _safe_ratio(num_cell: dict | None, den_cell: dict | None) -> float | None:
        if num_cell and den_cell and den_cell["share"] > 0.0:
            return float(num_cell["share"] / den_cell["share"])
        return None

    self_written_voice = {
        "min_trimmed_chars": 30,
        "eligible_with_essay": {
            "Pooled": int(has_essay.sum()),
            "Female": int((has_essay & is_f).sum()),
            "Male": int((has_essay & ~is_f).sum()),
        },
        "themes": voice_overall,
        "by_life_window": voice_by_window,
        "voice_to_checkbox_ratios": {
            "female_family_to_tall": _safe_ratio(
                voice_overall["family_kids_home"]["Female"],
                voice_overall["tall_6ft_mention"]["Female"],
            ),
            "female_honesty_to_tall": _safe_ratio(
                voice_overall["honest_loyal_trust"]["Female"],
                voice_overall["tall_6ft_mention"]["Female"],
            ),
            "female_kindness_to_wealth": _safe_ratio(
                voice_overall["kind_caring_loving"]["Female"],
                voice_overall["rich_six_figure"]["Female"],
            ),
            "pooled_honesty_to_tall": _safe_ratio(
                voice_overall["honest_loyal_trust"]["Pooled"],
                voice_overall["tall_6ft_mention"]["Pooled"],
            ),
            "pooled_family_to_wealth": _safe_ratio(
                voice_overall["family_kids_home"]["Pooled"],
                voice_overall["rich_six_figure"]["Pooled"],
            ),
        },
    }

    return {
        "accounts": accounts, "eligible": eligible, "bands": bins, "sensitivity": sensitivity,
        "exclusions_in_order": {"age": len(table) - total(age_ok),
                                "gender": total(age_ok) - len(base), "seeking": len(base) - len(scoped),
                                "unknown_parenthood": eligible["unknown_parenthood"]},
        "response_audit_all_rows": {
            "intent": dict(zip(ANSWERS, (total(pc.equal(answer, i)) for i in range(4)))),
            "intent_prefer_not_to_say": total(pc.equal(table["wantchildren"], "Prefer Not To Say")),
            "intent_blank_or_null": total(pc.or_(pc.is_null(table["wantchildren"]),
                                                 pc.fill_null(pc.equal(table["wantchildren"], ""), False))),
            "parenthood": dict(zip(("no", "yes", "adult_children", "unknown"),
                                   (total(pc.equal(parent, i)) for i in range(4)))),
        },
        "market_and_commitment": market_windows,
        "traits_and_postponement": traits_and_postponement,
        "height_percentiles": height_percentiles,
        "geography": geography,
        "childless_crossover": childless_crossover,
        "self_written_voice": self_written_voice,
    }


def total(mask):
    return int(pc.sum(mask).as_py() or 0)


def sequence(column, n):
    return len(column) == n and np.array_equal(column.to_numpy(), np.arange(1, n + 1))


def valid_votes(likes, votes):
    return (votes > 0) & (likes >= 0) & (likes <= votes)


def hon_rating(likes, votes):
    if len(votes) < MIN_CELL:
        return None
    likes_total = int(likes.sum())
    votes_total = int(votes.sum())
    return {"n": len(votes), "likes": likes_total, "votes": votes_total,
            "mean_profile_like_rate": float(np.mean(likes / votes)),
            "vote_pooled_like_rate": likes_total / votes_total, "median_votes": float(np.median(votes))}


def midranks(scores):
    _, inverse, frequency = np.unique(scores, return_inverse=True, return_counts=True)
    return ((frequency.cumsum() - 0.5 * frequency) / len(scores))[inverse]


def beauty_cell(score, rank, pooled_rank):
    if len(score) < MIN_CELL:
        return None
    return {"n": len(score), "mean_percentile": float(rank.mean()),
            "percentile_iqr": np.quantile(rank, [.25, .75]).tolist(),
            "top_tenth_n": int((rank >= .90).sum()), "mean_score": float(score.mean()),
            "mean_pooled_percentile": float(pooled_rank.mean())}


def hon():
    p = read("profiles", ["profile_id", "age", "gender", "country", "metadata_conflict",
                          "label_conflict", "dir_present", "image_count", "decoded_count", "record_count"])
    labels = read("labels", ["profile_id", "likes", "votes", "like_rate", "beauty_score", "status"])
    source = read("HON_source_fields", ["profile_id", "source_record_no"])
    assert len(p) == len(labels) == len(source) and sequence(source["source_record_no"], len(p))
    assert pc.count_distinct(p["profile_id"]).as_py() == len(p) and not p["profile_id"].null_count
    # Both exports are ordered by the same canonical identifier in this saved snapshot.
    assert pc.all(pc.equal(p["profile_id"], labels["profile_id"])).as_py()
    assert pc.all(pc.equal(p["profile_id"], source["profile_id"])).as_py()
    assert total(pc.equal(p["record_count"], 1)) == len(p)
    age = pc.fill_null(p["age"], -1).to_numpy()
    gender = p["gender"]
    woman = pc.fill_null(pc.equal(gender, "Female"), False).to_numpy()
    man = pc.fill_null(pc.equal(gender, "Male"), False).to_numpy()
    us = pc.fill_null(pc.equal(p["country"], "United States"), False).to_numpy()
    img_counts = pc.fill_null(p["image_count"], 0).to_numpy()
    images = img_counts > 0
    likes = pc.fill_null(labels["likes"], -1).to_numpy()
    votes = pc.fill_null(labels["votes"], -1).to_numpy()
    rate = labels["like_rate"].to_numpy()
    score = labels["beauty_score"].to_numpy()
    metadata_clear = pc.fill_null(pc.equal(p["metadata_conflict"], 0), False).to_numpy()
    label_clear = pc.fill_null(pc.equal(p["label_conflict"], 0), False).to_numpy()
    valid = pc.fill_null(pc.equal(labels["status"], "valid"), False).to_numpy()
    valid_counts = valid_votes(likes, votes)
    assert np.array_equal(valid, valid_counts)
    assert np.allclose(rate[valid], likes[valid] / votes[valid], rtol=0, atol=1e-12)
    stages = [eligible_age(age), eligible_age(age) & (woman | man)]
    stages.append(stages[-1] & metadata_clear & label_clear)
    stages.append(stages[-1] & valid & valid_counts)
    eligible = stages[-1]
    primary = eligible & (votes >= 20)
    all_counts = {
        "canonical_profiles": len(p), "aligned_source_rows": len(source), "distinct_profile_keys": len(p),
        "recorded_gender": {sex: n for sex, n in zip(SEXES, (int(woman.sum()), int(man.sum())))},
        "labels": tally(labels, "status"), "metadata_conflicts": int((~metadata_clear).sum()),
        "label_conflicts": int((~label_clear).sum()),
        "local_directories": total(p["dir_present"]), "profiles_with_inventoried_images": int(images.sum()),
        "inventoried_images": total(p["image_count"]), "decoded_images": total(p["decoded_count"]),
        "valid_likes": int(likes[valid].sum()), "valid_votes": int(votes[valid].sum()),
    }
    stage_names = ("age_18_79", "age_and_gender", "unconflicted", "valid_votes")
    stage_counts = dict(zip(stage_names, (int(mask.sum()) for mask in stages)))
    exclusions = dict(zip(stage_names + ("at_least_20_votes",),
                          (a - b for a, b in zip([len(p), *stage_counts.values()],
                                                 [*stage_counts.values(), int(primary.sum())]))))

    # Sort eligible profiles once so exact ages and grouped age ranges share the same records.
    order = np.flatnonzero(eligible)
    order = order[np.argsort(age[order], kind="stable")]
    ages = age[order]
    ratings = {}
    for name, minimum, us_only in [("votes_1", 1, False), ("votes_20", 20, False),
                                   ("votes_100", 100, False), ("us_votes_20", 20, True)]:
        result = {}
        for lo, hi, key in [(18, 79, "all_ages"),
                            *((a, a, str(a)) for a in range(18, 80)),
                            *((a, b, age_band(a, b)) for a, b in BANDS)]:
            left, right = np.searchsorted(ages, [lo, hi + 1])
            idx = order[left:right]
            idx = idx[(votes[idx] >= minimum) & (us[idx] if us_only else True)]
            result[key] = {}
            for sex, mask in zip(SEXES, (woman, man)):
                selected = idx[mask[idx]]
                inventoried = selected[images[selected]]
                result[key][sex] = {"all": hon_rating(likes[selected], votes[selected]),
                                    "inventoried": hon_rating(likes[inventoried], votes[inventoried])}
        ratings[name] = result

    score_ok = np.isfinite(score) & (score >= 0) & (score <= 10)
    beauty_eligible = primary & score_ok
    pooled = np.full(len(p), np.nan)
    pooled[beauty_eligible] = midranks(score[beauty_eligible])
    within_sex_rank = np.full(len(p), np.nan)
    beauty = {}
    beauty_summary = {}
    for sex, mask in zip(SEXES, (woman, man)):
        idx = order[beauty_eligible[order] & mask[order]]
        scores = score[idx]
        ranks = midranks(scores)
        within_sex_rank[idx] = ranks
        beauty_summary[sex] = {
            "reference_n": len(idx), "score_like_rate_pearson_r": float(np.corrcoef(scores, rate[idx])[0, 1]),
            "missing_or_invalid_score_among_primary": int((primary & mask & ~score_ok).sum()),
        }
        for lo, hi, key in [(18, 79, "all_ages"), *((a, a, str(a)) for a in range(18, 80)),
                            *((a, b, age_band(a, b)) for a, b in BANDS)]:
            left, right = np.searchsorted(age[idx], [lo, hi + 1])
            rows = idx[left:right]
            all_rank = ranks[left:right]
            subset = images[rows]
            beauty.setdefault(key, {})[sex] = {
                "all": beauty_cell(score[rows], all_rank, pooled[rows]),
                "inventoried": beauty_cell(score[rows][subset], all_rank[subset], pooled[rows][subset]),
            }

    # --- Attention Skew (Gini & Lorenz Deciles) + Portfolio Image-Count Depth ---
    attention_and_portfolio = {"by_sex": {}, "by_photo_count": {}}
    img_clipped = np.clip(img_counts, 0, 5)
    photo_labels = ("0", "1", "2", "3", "4", "5_plus")
    for sex, mask in zip(SEXES, (woman, man)):
        bmask = beauty_eligible & mask
        v_sex = votes[bmask]
        l_sex = likes[bmask]
        v_gini, v_dec = gini_and_deciles(v_sex)
        l_gini, l_dec = gini_and_deciles(l_sex)
        attention_and_portfolio["by_sex"][sex] = {
            "n": int(bmask.sum()),
            "votes_gini": v_gini,
            "likes_gini": l_gini,
            "median_votes": float(np.median(v_sex)),
            "mean_votes": float(v_sex.mean()),
            "p90_votes": float(np.quantile(v_sex, 0.90)),
            "top_10pct_votes_share": v_dec[-1],
            "top_10pct_likes_share": l_dec[-1],
            "lorenz_deciles_votes": v_dec,
            "lorenz_deciles_likes": l_dec,
        }
        by_photo = {}
        for k_img, k_label in enumerate(photo_labels):
            pmask = bmask & (img_clipped == k_img)
            if int(pmask.sum()) < MIN_CELL:
                by_photo[k_label] = None
            else:
                by_photo[k_label] = {
                    "n": int(pmask.sum()),
                    "mean_profile_like_rate": float(rate[pmask].mean()),
                    "vote_pooled_like_rate": float(likes[pmask].sum() / votes[pmask].sum()),
                    "median_votes": float(np.median(votes[pmask])),
                    "mean_within_sex_percentile": float(within_sex_rank[pmask].mean()),
                    "mean_pooled_percentile": float(pooled[pmask].mean()),
                }
        attention_and_portfolio["by_photo_count"][sex] = by_photo

    # --- Cross-National HON Attraction & Age-Drop Gradients ---
    b_idx = np.flatnonzero(beauty_eligible)
    country_trimmed = pc.utf8_trim_whitespace(pc.fill_null(p["country"], "")).take(b_idx)
    nonblank = pc.not_equal(country_trimmed, "")
    b_idx_c = b_idx[nonblank.to_numpy()]
    c_col = country_trimmed.filter(nonblank)
    sex_c = pa.array(woman[b_idx_c])
    age_c = age[b_idx_c]
    lr_c = rate[b_idx_c]
    votes_c = votes[b_idx_c]
    likes_c = likes[b_idx_c]
    win_c = within_sex_rank[b_idx_c]
    pool_c = pooled[b_idx_c]

    ct_table = pa.table({
        "country": c_col,
        "is_female": sex_c,
        "age": pa.array(age_c),
        "likes": pa.array(likes_c),
        "votes": pa.array(votes_c),
        "lr": pa.array(lr_c),
        "win": pa.array(win_c),
        "pool": pa.array(pool_c),
    })
    grp_all = ct_table.group_by(["country", "is_female"], use_threads=False).aggregate([
        ("age", "count"), ("likes", "sum"), ("votes", "sum"),
        ("lr", "mean"), ("win", "mean"), ("pool", "mean"),
    ]).to_pylist()
    grp_y = ct_table.filter(pc.and_(pc.greater_equal(ct_table["age"], 18), pc.less_equal(ct_table["age"], 29))).group_by(
        ["country", "is_female"], use_threads=False
    ).aggregate([("age", "count"), ("win", "mean"), ("lr", "mean")]).to_pylist()
    grp_o = ct_table.filter(pc.and_(pc.greater_equal(ct_table["age"], 40), pc.less_equal(ct_table["age"], 59))).group_by(
        ["country", "is_female"], use_threads=False
    ).aggregate([("age", "count"), ("win", "mean"), ("lr", "mean")]).to_pylist()

    all_map = defaultdict(dict)
    for r in grp_all:
        all_map[r["country"]]["Female" if r["is_female"] else "Male"] = r
    y_map = {(r["country"], "Female" if r["is_female"] else "Male"): r for r in grp_y}
    o_map = {(r["country"], "Female" if r["is_female"] else "Male"): r for r in grp_o}

    qual_100 = sorted(
        c for c, d in all_map.items()
        if d.get("Female", {}).get("age_count", 0) >= MIN_CELL and d.get("Male", {}).get("age_count", 0) >= MIN_CELL
    )
    qual_500 = [c for c in qual_100 if all_map[c]["Female"]["age_count"] >= 500 and all_map[c]["Male"]["age_count"] >= 500]
    qual_1000 = [c for c in qual_100 if all_map[c]["Female"]["age_count"] >= 1000 and all_map[c]["Male"]["age_count"] >= 1000]

    country_rows = {}
    age_contrast_countries = []
    for cname in qual_100:
        d = all_map[cname]
        nf = int(d["Female"]["age_count"])
        nm = int(d["Male"]["age_count"])
        by_s = {}
        has_all_age_cells = True
        for sex_label in SEXES:
            rs = d[sex_label]
            ry = y_map.get((cname, sex_label))
            ro = o_map.get((cname, sex_label))
            y_cell = (
                {"n": int(ry["age_count"]), "mean_within_sex_percentile": float(ry["win_mean"]), "mean_like_rate": float(ry["lr_mean"])}
                if (ry and ry["age_count"] >= MIN_CELL) else None
            )
            o_cell = (
                {"n": int(ro["age_count"]), "mean_within_sex_percentile": float(ro["win_mean"]), "mean_like_rate": float(ro["lr_mean"])}
                if (ro and ro["age_count"] >= MIN_CELL) else None
            )
            if not (y_cell and o_cell):
                has_all_age_cells = False
            by_s[sex_label] = {
                "n": int(rs["age_count"]),
                "likes": int(rs["likes_sum"]),
                "votes": int(rs["votes_sum"]),
                "mean_profile_like_rate": float(rs["lr_mean"]),
                "vote_pooled_like_rate": float(rs["likes_sum"] / rs["votes_sum"]),
                "mean_within_sex_percentile": float(rs["win_mean"]),
                "mean_pooled_percentile": float(rs["pool_mean"]),
                "age_18_29": y_cell,
                "age_40_59": o_cell,
                "age_drop_pp": (
                    float(100.0 * (y_cell["mean_within_sex_percentile"] - o_cell["mean_within_sex_percentile"]))
                    if (y_cell and o_cell) else None
                ),
            }
        if has_all_age_cells:
            age_contrast_countries.append(cname)
        country_rows[cname] = {
            "female_to_male_profile_ratio": float(nf / nm),
            "female_minus_male_like_rate_pp": float(100.0 * (by_s["Female"]["mean_profile_like_rate"] - by_s["Male"]["mean_profile_like_rate"])),
            "female_minus_male_pooled_percentile_pp": float(100.0 * (by_s["Female"]["mean_pooled_percentile"] - by_s["Male"]["mean_pooled_percentile"])),
            "female_minus_male_age_drop_pp": (
                float(by_s["Female"]["age_drop_pp"] - by_s["Male"]["age_drop_pp"]) if has_all_age_cells else None
            ),
            "by_sex": by_s,
        }

    f_drops_strict = [country_rows[c]["by_sex"]["Female"]["age_drop_pp"] for c in age_contrast_countries]
    m_drops_strict = [country_rows[c]["by_sex"]["Male"]["age_drop_pp"] for c in age_contrast_countries]
    cross_national = {
        "nonblank_country_profiles": int(len(b_idx_c)),
        "blank_country_profiles": int(len(b_idx) - len(b_idx_c)),
        "counts_by_threshold": {
            "ge_100_per_sex": len(qual_100),
            "ge_500_per_sex": len(qual_500),
            "ge_1000_per_sex": len(qual_1000),
            "strict_age_contrast_ge_100_all_4_cells": len(age_contrast_countries),
        },
        "summary_ge_500_per_sex": {
            "countries": len(qual_500),
            "female_higher_like_rate_countries": sum(
                1 for c in qual_500 if country_rows[c]["female_minus_male_like_rate_pp"] > 0
            ),
            "female_higher_pooled_percentile_countries": sum(
                1 for c in qual_500 if country_rows[c]["female_minus_male_pooled_percentile_pp"] > 0
            ),
        },
        "summary_strict_age_contrast": {
            "countries": len(age_contrast_countries),
            "female_steeper_drop_countries": sum(
                1 for c in age_contrast_countries if country_rows[c]["female_minus_male_age_drop_pp"] > 0
            ),
            "median_female_drop_pp": float(np.median(f_drops_strict)) if f_drops_strict else None,
            "median_male_drop_pp": float(np.median(m_drops_strict)) if m_drops_strict else None,
            "median_female_minus_male_drop_pp": (
                float(np.median([f - m for f, m in zip(f_drops_strict, m_drops_strict)])) if f_drops_strict else None
            ),
        },
        "countries": country_rows,
    }

    return {
        "inventory": all_counts, "eligibility_stages": stage_counts, "exclusions_in_order": exclusions,
        "ratings": ratings, "beauty_reference": beauty_summary, "beauty_by_age": beauty,
        "attention_and_portfolio": attention_and_portfolio,
        "cross_national": cross_national,
    }


def snapshot_inventory(hon_data, pov_data):
    tables = saved("exports/manifest.json")
    raw = saved("exports/source_fields_manifest.json")
    face = saved(PARTIAL / "manifest.json")
    full_faceflow = saved(FULL_FACEFLOW)
    leaderboard = saved(BENCHMARK)
    validation = saved("reports/validation.json")
    report = saved("reports/analysis.json")
    partial_report = saved("faceflow/reports/partial_audit.json")
    pilot = saved("pilot/metrics.json")
    for name, row in tables.items():
        path = EXPORTS / f"{name}.parquet"
        assert row["rows"] == pq.ParquetFile(path).metadata.num_rows == validation["export_counts"][name]["database"]
        assert path.stat().st_size == row["bytes"]
    for row in [*raw.values(), *face["tables"].values()]:
        path = (EXPORTS if "source_id" in row else PARTIAL) / row["file"]
        assert path.stat().st_size == row["bytes"]
        assert row["rows"] == pq.ParquetFile(path).metadata.num_rows
    assert validation["passed"] and all(validation["checks"].values())
    sources = {row["kind"]: {"csv_sha256": row["sha256"], "records": row["records"], "complete": bool(row["completed"])}
               for row in read("sources", ["kind", "sha256", "records", "completed"]).to_pylist()}
    records = read("records", ["source_id", "record_no", "profile_id", "parse_status"])
    assert tally(records, "parse_status") == {"ok": len(records)}
    for kind, source_id, n in [("HON", 1, hon_data["inventory"]["canonical_profiles"]),
                                ("POV", 2, pov_data["accounts"]["rows"])]:
        subset = records.filter(pc.equal(records["source_id"], source_id))
        assert sources[kind]["records"] == raw[kind]["rows"] == n
        assert sources[kind]["complete"] and sequence(subset["record_no"], n)
        assert subset["profile_id"].null_count == (n if kind == "POV" else 0)
    assert len(records) == sum(row["records"] for row in sources.values())
    for key, value in [("profiles", "canonical_profiles"), ("images", "inventoried_images"),
                       ("image_backed_profiles", "profiles_with_inventoried_images"),
                       ("local_directories", "local_directories"), ("decoded_images", "decoded_images"),
                       ("votes", "valid_votes"), ("likes", "valid_likes")]:
        assert report["cohorts"]["all_metadata"][key] == hon_data["inventory"][value]
    files = read("files", ["kind", "validation", "size"])
    by_kind = files.group_by("kind", use_threads=False).aggregate([([], "count_all"), ("size", "sum")]).to_pylist()
    by_kind = sorted(by_kind, key=lambda row: row["kind"])
    assert by_kind == [{"kind": r["kind"], "count_all": r["count"], "size_sum": r["bytes"]} for r in report["files"]]
    directories = tally(read("directories", ["state"]), "state")
    assert directories == {"done": hon_data["inventory"]["local_directories"] + 1}
    sidecars = counts(read("quality", ["kind", "status"]), "kind", "status")
    assert sidecars == [{"kind": r["kind"], "status": r["status"], "count_all": r["count"]} for r in report["quality"]]
    by_file_status = counts(files, "kind", "validation")
    assert {r["validation"]: r["count_all"] for r in by_file_status if r["kind"] == "image"} == {
        r["validation"]: r["count"] for r in report["image_validation"]}
    properties = counts(read("image_properties", ["status", "format", "mode"], PARTIAL), "status", "format", "mode")
    jobs = read("jobs", ["state", "expected_images", "images", "selected"], PARTIAL)
    assert total(jobs["expected_images"]) == hon_data["inventory"]["inventoried_images"]
    images = read("images", ["status", "width", "height", "sha256", "pixel_sha256", "detected_faces"], PARTIAL)
    image_status = tally(images, "status")
    assert len(images) == total(jobs["images"]) == partial_report["counts"]["images"]
    assert image_status == {r["status"]: r["images"] for r in partial_report["image_status"]}
    assert sum(r["count_all"] for r in properties) == len(images)
    assert len(images) - images["pixel_sha256"].null_count == partial_report["counts"]["pixel_hashes"]
    assert full_faceflow["complete"] and full_faceflow["counts_reconcile"] and full_faceflow["all_indexed_images_covered"]
    assert (full_faceflow["source_image_count"] == full_faceflow["global_indexed_image_count"] ==
            full_faceflow["processed_image_count"] == hon_data["inventory"]["inventoried_images"])
    assert sum(row["n"] for row in full_faceflow["host_counts"]) == hon_data["inventory"]["inventoried_images"]
    assert (leaderboard["sample_profiles"] == 10000 and leaderboard["sample_images_total"] == 20000 and
            leaderboard["sample_primary_images"] == hon_data["inventory"]["decoded_images"] and
            len(leaderboard["waves"]) >= 20)
    pilot_profiles = read("pilot_profiles", ["split", "status", "images"])
    assert len(pilot_profiles) == pilot["profiles"] and tally(pilot_profiles, "split") == pilot["splits"]["counts"]
    assert total(pilot_profiles["images"]) == pilot["selected_images"] == hon_data["inventory"]["decoded_images"]
    assert all(pilot["verification"].values())
    review = saved(REVIEW / "verification.json")
    assert review["profile_rows"] == hon_data["inventory"]["canonical_profiles"]
    return {
        "sources": sources,
        "saved_index": {"tables": tables, "passed": validation["passed"], "checks": validation["checks"],
                        "logical_records_by_source": {kind: data["records"] for kind, data in sources.items()},
                        "files_by_kind": by_kind, "files_by_status": by_file_status,
                        "directory_states_including_root": directories, "sidecars": sidecars},
        "partial_image_audit": {
            "tables": face["tables"], "complete": partial_report["complete"], "jobs_by_state": tally(jobs, "state"),
            "expected_images": total(jobs["expected_images"]), "image_status": image_status,
            "coverage": len(images) / total(jobs["expected_images"]), "counts": partial_report["counts"],
            "properties": properties, "dimensions": counts(images, "width", "height"),
            "selected_candidates": total(jobs["selected"]),
        },
        "full_faceflow_audit": {
            "complete": full_faceflow["complete"],
            "counts_reconcile": full_faceflow["counts_reconcile"],
            "all_indexed_images_covered": full_faceflow["all_indexed_images_covered"],
            "source_image_count": full_faceflow["source_image_count"],
            "processed_image_count": full_faceflow["processed_image_count"],
            "jobs": full_faceflow["jobs"],
            "image_status": full_faceflow["image_status"],
            "host_counts": full_faceflow["host_counts"],
        },
        "multimodal_benchmark": {
            "benchmark": leaderboard["benchmark"],
            "sample_profiles": leaderboard["sample_profiles"],
            "sample_images_total": leaderboard["sample_images_total"],
            "sample_primary_images": leaderboard["sample_primary_images"],
            "waves": [
                {
                    "wave_id": w["wave_id"],
                    "title": w["title"],
                    "track": w["track"],
                    "profile_validation": w["profile_validation"],
                    "profile_test": w["profile_test"],
                    "paired_bootstrap_comparisons": w["paired_bootstrap_comparisons"],
                }
                for w in leaderboard["waves"]
            ],
        },
        "cached_pilot": {
            "profiles": len(pilot_profiles), "selected_images": total(pilot_profiles["images"]),
            "splits": pilot["splits"], "seed": pilot["seed"], "epochs": pilot["epochs"],
            "duplicate_check_coverage": pilot["duplicate_check_coverage"], "verification": pilot["verification"],
            "test_metrics": {key: {k: value for k, value in model["test"].items() if k != "calibration_bins"}
                             for key, model in pilot["models"].items()},
        },
        "saved_beauty_review": {"profile_rows": review["profile_rows"],
                                "image_backed_profiles": review["image_backed_profiles"],
                                "all_percentiles_match_reference": review["all_percentiles_match_reference"],
                                "image_snapshot": saved(REVIEW / "audited_image_quality.json"),
                                "pilot_delta_mae": review["pilot_delta_mae"],
                                "pilot_paired_bootstrap_95ci": review["pilot_paired_bootstrap_95ci"]},
        "artifact_use": {
            "source_fields_and_canonical_labels": "Support descriptive POV intentions/traits/height/geography and HON ratings, attention skew, portfolio depth, and cross-national ranks. The sources are separate populations.",
            "records_and_index_validation": "Reconcile source checkpoints, record counts, profile linkage and saved export totals.",
            "directories_and_file_inventory": "Support coverage of associated image-extension files, independently of decoding.",
            "face_size_and_landmark_sidecars": "Historical coordinate and parse coverage. Face size is in landmark coordinate pixels. No appearance or owner claims use these sidecars.",
            "partial_image_audit": "Frozen provisional export of an earlier partial decode pass, retained alongside the completed 3,685,764-image faceflow audit.",
            "full_faceflow_audit": "Completed 100%-coverage (3,685,764/3,685,764) processed UV texture audit across all 898,903 image-backed directories.",
            "multimodal_benchmark": "Audited 20-wave (H1-H20) progressive facial alignment, nuisance-matte removal, multi-photo pooling, and candidate context benchmark on the 10,000-profile zero-leakage split.",
            "cached_pilot": "Selected 10,000-profile execution baseline with cached embeddings and duplicate checks within the selected images.",
            "saved_beauty_review": "Earlier age/rank and quality checks with different reference eligibility. The manuscript ranks are recomputed from canonical labels.",
            "legacy_assets": "Inventoried as legacy files only. No pickle or model objects were loaded.",
        },
    }


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def main():
    assert platform.node().split(".")[0] == "xbabe3" and platform.machine() == "x86_64"
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    files = sorted([*EXPORTS.glob("*.parquet"), *EXPORTS.glob("*manifest.json"),
                    *PARTIAL.glob("*.parquet"), PARTIAL / "manifest.json",
                    FULL_FACEFLOW, BENCHMARK,
                    *Path("pilot").glob("*.json"), *Path("pilot").glob("*.npz"), *Path("pilot").glob("*.pt"),
                    REVIEW / "verification.json", REVIEW / "audited_image_quality.json", REVIEW / "manifest.json",
                    Path("reports/validation.json"), Path("reports/analysis.json"),
                    Path("faceflow/reports/partial_audit.json")])
    before = {p: identity(p) for p in files}
    hashes = {str(p): sha256(p) for p in files}
    pov_data = pov()
    hon_data = hon()
    inventory = snapshot_inventory(hon_data, pov_data)
    assert before == {p: identity(p) for p in files}
    json.dump({
        "schema_version": 1, "study": "performance-before-outcomes",
        "source_host": platform.node(), "source_architecture": platform.machine(),
        "input_sha256": hashes, "input_sizes_bytes": {str(p): before[p][2] for p in files},
        "cohorts": {
            "ages": [18, 79], "pov_bands": [list(band) for band in BANDS], "min_profiles_per_cell": MIN_CELL,
            "pov": "Exact Female/Male and Man/Woman pairings in gender/seeking. Parenthood uses Yes or All my kids are over 18 versus No. All other parenthood responses are unknown. Stated desire denominator includes nonresponse except in answered-only analyses. Nonresponse includes all values outside the three recognized English responses.",
            "pov_sensitivity": "Single uses the exact Single label. All-seeking lifts the seeking filter, including blanks. Exact-age standardization weights both sexes by their pooled eligible exact-age distribution within each band. The combined check uses single and answered profiles for its weights.",
            "pov_extensions": "Commitment horizon uses exact English intent categories; serious commitment groups marry, actively seeking, and looking for a relationship. Height percentiles use explicit uncensored strings from 5'0\" (152cm) to 7'0\" (213cm), keeping <5' and 999cm audited separately. Geography maps unambiguous US states, UK, Canadian provinces, Brazilian states, and Australian states.",
            "hon": "Equal weight per profile, recorded Female or Male, age 18–79, unconflicted metadata and label, valid parsed likes and votes. Main threshold 20; 1 and 100 are sensitivities. Recorded U.S. uses exact United States in parsed country. Inventoried means at least one associated image file. Vote pooling weights by recorded vote total.",
            "beauty": "Supplied beauty score on 0–10 scale, among main HON rating profiles with a finite in-range score. Reference is all eligible ages within each recorded gender; midrank is (number below + half tied) / reference N. Image subset reuses the same reference. A pooled-gender reference is retained as a sensitivity.",
            "hon_cross_national": "Countries enter overall cross-national summaries at >=100 primary beauty-eligible profiles per recorded gender (with >=500 and >=1000 flagged) and strict age-drop contrasts only when all four gender x {18-29, 40-59} cells contain >=100 profiles.",
        },
        "inventory": inventory, "pov": pov_data, "hon": hon_data,
    }, sys.stdout, allow_nan=False, sort_keys=True, indent=2)
    print()


if __name__ == "__main__":
    main()
