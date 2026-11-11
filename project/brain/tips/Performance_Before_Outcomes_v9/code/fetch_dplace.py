"""Fetch the open D-PLACE release of the Ethnographic Atlas, the Standard Cross-Cultural Sample and Binford's
hunter-gatherer data (Kirby et al. 2016, PLoS ONE) into data/dplace/mirror and check the commit.  Not redistributed.

Run:  python3 -I code/fetch_dplace.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "dplace", "mirror")
COMMIT = "9bfed2c8c206be00f55f71516f262bbca2234e5a"      # 2022-02-07

if __name__ == "__main__":
    if not os.path.exists(os.path.join(DEST, "datasets", "SCCS", "data.csv")):
        os.makedirs(os.path.dirname(DEST), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/D-PLACE/dplace-data.git", DEST], check=True)
        subprocess.run(["git", "-C", DEST, "sparse-checkout", "set", "datasets/EA", "datasets/SCCS", "datasets/Binford"], check=True)
    head = subprocess.run(["git", "-C", DEST, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("commit", head, "(expected)" if head == COMMIT else "(DIFFERENT from the commit analysed: " + COMMIT + ")")
    sys.exit(0 if head == COMMIT else 1)
