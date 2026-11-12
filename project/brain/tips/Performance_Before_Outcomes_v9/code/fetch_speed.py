"""Fetch the public file of the Columbia speed-dating experiment (Fisman, Iyengar, Kamenica and Simonson 2006).

The file is not redistributed with this package.  It is mirrored unchanged in several public repositories; this script
clones one of them, copies the CSV to data/speeddating/speed_dating.csv and checks its SHA-256 digest, so that a changed
or corrupted mirror is detected.

Run with:  python3 -I code/fetch_speed.py
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "speeddating", "speed_dating.csv")
SHA256 = "fda9a924b18880fabe6c7a71961cff20aa3c7cc2a5aa8d6e823c2fc41893814c"
MIRRORS = [("https://github.com/tsizzle/speed_dating_experiment.git", "Speed_Dating_Data.csv"),
           ("https://github.com/XanderGriff/speed-dating-data-mining.git", "speed_dating_data.csv"),
           ("https://github.com/dafrd/speed-dating.git", "SpeedDatingData.csv")]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if os.path.exists(DEST) and digest(DEST) == SHA256:
        print("already present and verified:", DEST)
        return 0
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    for url, name in MIRRORS:
        tmp = tempfile.mkdtemp(prefix="speed_")
        try:
            subprocess.run(["git", "clone", "--depth", "1", "-q", url, tmp], check=True, timeout=300)
            src = os.path.join(tmp, name)
            if os.path.exists(src) and digest(src) == SHA256:
                shutil.copyfile(src, DEST)
                print("fetched and verified from", url)
                return 0
            print("digest mismatch or file missing in", url, file=sys.stderr)
        except Exception as e:  # noqa
            print("could not use", url, repr(e)[:80], file=sys.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print("no mirror supplied the expected file", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
