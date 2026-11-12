"""Loaders for the three open fossil datasets (read-only use of third-party data).
 Gingerich 2022, Sci. Rep. 12:11216, Suppl. Dataset 1 (CC BY 4.0)
 Will et al. 2021, Nat. Commun. 12:4116, SOM File 2 (CC BY 4.0)
 Du et al. 2018, Proc. R. Soc. B 285:20172738, Dryad doi:10.5061/dryad.c30g9 (CC0)
"""
import os, re, unicodedata
import numpy as np
import openpyxl
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")

def _rows(path, sheet=0):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[sheet]
    return [list(r) for r in ws.iter_rows(values_only=True)]

def _num(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def keyname(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def gingerich():
    R = _rows(os.path.join(D, "gingerich2022", "41598_2022_15481_MOESM1_ESM.xlsx"))
    out = []
    for r in R:
        if r and r[0] and _num(r[2]) == _num(r[2]) and _num(r[3]) == _num(r[3]):
            out.append((str(r[0]).strip(), int(_num(r[1])), abs(_num(r[2])), _num(r[3])))
    return out  # specimen, n case studies, age (Ma, positive), ECV (cm3)

def will():
    R = _rows(os.path.join(D, "will2021", "41467_2021_24290_MOESM5_ESM.xlsx"))
    out = []
    for r in R[1:]:
        if r and r[0] and _num(r[1]) == _num(r[1]) and _num(r[5]) == _num(r[5]):
            out.append((str(r[0]).strip(), _num(r[1]), str(r[4]).strip(), _num(r[5]), str(r[6])))
    return out  # specimen, age (Ma), taxon, ECV, age group

def du():
    R = _rows(os.path.join(D, "du2018", "du2018_ecv.xlsx"))
    H = R[0]; out = []
    for r in R[1:]:
        if not r or not r[0]:
            continue
        d = dict(zip(H, r))
        e = [_num(d[k]) for k in ("ecv1", "ecv2", "ecv3", "ecv4", "ecv5", "ecv6")]
        e = [x for x in e if x == x]
        lo, hi, mu = _num(d["min.date"]), _num(d["max.date"]), _num(d["mean.date"])
        age = mu if mu == mu else 0.5 * (lo + hi)
        out.append((str(d["ID"]).strip(), age, str(d["lump.taxon"]).strip(), float(np.mean(e)), str(d["grade"])))
    return out  # specimen, age (Ma), taxon, mean ECV, grade


def region(lon, lat):
    """Coarse region from site coordinates (used only to split the Mid-Pleistocene sample)."""
    try:
        lon, lat = float(lon), float(lat)
    except Exception:
        return "NA"
    if lon > 60:
        return "E/SE Asia"
    if lon < -30:
        return "Americas"
    if (lat >= 36 and -12 <= lon <= 60) or (lat >= 29 and 30 <= lon <= 60):
        return "Europe/W Asia"
    return "Africa"

def will_full():
    R = _rows(os.path.join(D, "will2021", "41467_2021_24290_MOESM5_ESM.xlsx"))
    out = []
    for r in R[1:]:
        if r and r[0] and _num(r[1]) == _num(r[1]) and _num(r[5]) == _num(r[5]):
            out.append(dict(id=str(r[0]).strip(), age=_num(r[1]), taxon=str(r[4]).strip(), ecv=_num(r[5]),
                            region=region(r[2], r[3])))
    return out
