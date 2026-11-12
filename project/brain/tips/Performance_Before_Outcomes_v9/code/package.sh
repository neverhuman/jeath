#!/bin/sh
# Assemble the release folder and the source package (run after run_all.sh).
set -e
cd "$(dirname "$0")/.."
REL=release_v9
NAME=Performance_Before_Outcomes_v9
mkdir -p $REL/figures
python3 code/flatten.py $REL/$NAME.tex
cp paper/main.pdf $REL/$NAME.pdf
cp paper/figures/*.pdf $REL/figures/
cp scorecard/scorecard_v9.pdf $REL/Scorecard_and_Review_v9.pdf
cp scorecard/Scorecard_and_Review_v9.tex $REL/Scorecard_and_Review_v9.tex
cp README.md $REL/README.md
cp MASTER_PLAN_v9.md $REL/MASTER_PLAN_v9.md
cp data/README.md $REL/DATA_SOURCES.md
mkdir -p $REL/code && cp code/*.py code/*.sh $REL/code/
cp paper/figures/author_*.* $REL/figures/ 2>/dev/null || true
# check that the single-file source compiles on its own
TMP=$(mktemp -d)
cp $REL/$NAME.tex $TMP/ && cp -r $REL/figures $TMP/
(cd $TMP && pdflatex -interaction=nonstopmode $NAME.tex >/dev/null && pdflatex -interaction=nonstopmode $NAME.tex >/dev/null)
echo "single-file build: $(pdfinfo $TMP/$NAME.pdf | grep Pages)"
# source package: everything needed to rebuild, without the survey file, the speed-dating file, or caches
PKG=$REL/${NAME}_package
STAGE=$(mktemp -d)/$NAME
mkdir -p $STAGE
cp README.md $STAGE/
cp MASTER_PLAN_v9.md $STAGE/
cp $REL/$NAME.pdf $REL/$NAME.tex $STAGE/
cp $REL/Scorecard_and_Review_v9.pdf $STAGE/
mkdir -p $STAGE/scorecard && cp scorecard/scorecard_v9.tex scorecard/make_scorecard_v9.py scorecard/rubric_v9_body.tex scorecard/rubric_v9_totals.tex $STAGE/scorecard/
cp -r $REL/figures $STAGE/figures
mkdir -p $STAGE/paper $STAGE/code $STAGE/results $STAGE/data
cp paper/main.tex paper/refs.bib paper/main.bbl $STAGE/paper/
cp -r paper/sections paper/generated paper/figures $STAGE/paper/
cp code/*.py code/*.sh $STAGE/code/
for f in results/*.json; do
  case "$(basename $f)" in abstracts_cache.json|gss_colnames.json) ;; *) cp "$f" $STAGE/results/ ;; esac
done
cp data/README.md $STAGE/data/
for d in gingerich2022 will2021 du2018; do
  mkdir -p $STAGE/data/$d
  find data/$d -maxdepth 1 -type f ! -name '*.zip' -exec cp {} $STAGE/data/$d/ \;
done
(cd $(dirname $STAGE) && zip -qr pkg.zip $NAME)
cp $(dirname $STAGE)/pkg.zip $PKG.zip
echo "package: $(unzip -l $PKG.zip | tail -1)"
