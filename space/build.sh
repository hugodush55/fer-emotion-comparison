#!/usr/bin/env bash
# Assemble the Hugging Face Space in space/build/ (app, inference code, weights, Space README/requirements).
# Training scripts, reports, dataset and demo video are not needed to run the app, so they are left out.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=space/build
rm -rf "$OUT"
mkdir -p "$OUT/src" "$OUT/models"

cp app.py "$OUT/"
cp src/__init__.py src/config.py src/data.py src/features.py src/cnn.py src/pipeline.py "$OUT/src/"
cp models/lbp_knn.joblib models/hog_svm.joblib models/mini_xception.pt "$OUT/models/"
cp space/README.md space/requirements.txt "$OUT/"

echo "Space assembled in $OUT:"
find "$OUT" -type f | sort
du -sh "$OUT"
