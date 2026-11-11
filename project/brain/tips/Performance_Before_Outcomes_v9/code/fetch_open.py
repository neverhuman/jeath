"""Download the open data sets added in version 9 and check their SHA-256 digests.  Nothing here is redistributed.

  kowal2025   Kowal et al. 2025, Scientific Data (OSF k54fx / svfuh), CC BY 4.0: 117,293 people in 175 countries
  walter2020  Walter et al. 2020, Psychological Science (OSF gb5cn), public deposit: 14,399 people in 45 countries

Run:  python3 -I code/fetch_open.py
"""
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "refcheck/0.1 (research manuscript reference verification)"}
FILES = [
    ("kowal2025", "Final_dataset.csv", "https://osf.io/download/eyn65/", "00f3292fa94280ad126724700fb439570fbada271595c4235968d00aa295120a"),
    ("kowal2025", "Codebook.xlsx", "https://osf.io/download/4e5nr/", "5150b30737bd6ae3484e30fbe908ee06bd1750001da4a7e3b8397f5a1470ade7"),
    ("walter2020", "processed_data.csv", "https://osf.io/download/5fmyu/", "0509eb7bbaae75c08533d19e1e6fa3a343a4775d825d644068be63824dff5933"),
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    bad = 0
    for folder, name, url, want in FILES:
        dest = os.path.join(ROOT, "data", folder)
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(dest, name)
        if not os.path.exists(path):
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(path, "wb") as f:
                for b in iter(lambda: r.read(1 << 20), b""):
                    f.write(b)
        got = sha(path)
        ok = (got == want)
        bad += (not ok)
        print(("ok   " if ok else "DIFF "), folder + "/" + name, os.path.getsize(path), got)
    sys.exit(1 if bad else 0)
