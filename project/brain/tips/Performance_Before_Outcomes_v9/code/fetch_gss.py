"""Download the public General Social Survey cumulative file (as packaged for R in
kjhealy/gssr) and extract the variables used by exp_gss.py into data/gss/gss_extract.npz.

The survey file itself is not redistributed with this package.
Run:  python3 -I code/fetch_gss.py        (isolated mode: it reads third-party data)
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import rda_reader as rr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "gss")
REPO = os.path.join(DEST, "gssr")
WANT = ("year id age sex cohort childs educ degree wordsum realrinc realinc conrinc coninc prestg10 sei10 marital agewed "
        "agekdbrn partners partnrs5 numwomen nummen sexfreq health wtssps ballot speduc spdeg wrkstat divorce sibs paeduc maeduc "
        "vstrat vpsu sample").split()
# Second extract (version 9): social life, ideals and partnership, for exp_village.py.  Kept in a separate file so that
# the first extract, and every result computed from it, stays byte-identical to earlier versions.
WANT2 = ("year id age sex cohort childs marital wtssps educ realinc chldidel socrel socommun socfrend socbar attend wwwhr "
         "partners sexfreq posslq").split()

if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    if not os.path.exists(os.path.join(REPO, "data", "gss_all.rda")):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/kjhealy/gssr.git", REPO], check=True)
    cols, labels, missing = rr.read_columns(os.path.join(REPO, "data", "gss_all.rda"), WANT)
    np.savez_compressed(os.path.join(DEST, "gss_extract.npz"), **cols)
    cols2, labels2, missing2 = rr.read_columns(os.path.join(REPO, "data", "gss_all.rda"), WANT2)
    np.savez_compressed(os.path.join(DEST, "gss_extract2.npz"), **cols2)
    with open(os.path.join(DEST, "gss_labels2.json"), "w") as f:
        json.dump({k: {str(a): b for a, b in v.items()} for k, v in labels2.items()}, f)
    print("second extract: variables", len(cols2), "missing", missing2)
    with open(os.path.join(DEST, "gss_labels.json"), "w") as f:
        json.dump({k: {str(a): b for a, b in v.items()} for k, v in labels.items()}, f)
    print("rows", len(cols["year"]), "variables", len(cols), "missing", missing)
