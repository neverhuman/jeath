#!/bin/sh
# Rebuild every result, figure and manuscript input, then compile the paper.
# The present-day analysis needs the public General Social Survey file; fetch_gss.py downloads it (about 30 MB) once.
# The speed-dating analysis needs the public Columbia file; fetch_speed.py copies it from a mirror and checks its digest.
set -e
cd "$(dirname "$0")"
python3 exp_core.py
python3 -I exp_rates.py
python3 exp_power.py
python3 exp_contest.py
python3 exp_polymath.py
python3 exp_verify.py
[ -f ../data/gss/gss_extract.npz ] || python3 -I fetch_gss.py
python3 -I exp_gss.py
[ -f ../data/speeddating/speed_dating.csv ] || python3 -I fetch_speed.py
python3 -I exp_speed.py
[ -f ../data/gss/gss_extract2.npz ] || python3 -I fetch_gss.py
python3 -I exp_village.py
[ -f ../data/dplace/mirror/datasets/SCCS/data.csv ] || python3 -I fetch_dplace.py
python3 -I exp_cultures.py
python3 -I fetch_open.py
python3 -I exp_world.py
python3 exp_notes.py
python3 -I audit.py
python3 -I make_figures.py
python3 make_tex.py
cd ../paper
pdflatex -interaction=nonstopmode main >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main >/dev/null
pdflatex -interaction=nonstopmode main >/dev/null
echo "built paper/main.pdf"
