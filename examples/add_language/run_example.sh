#!/usr/bin/env bash
# Add-language worked example, end to end. Run from anywhere:
#   bash examples/add_language/run_example.sh
# Uses the pure-Python training paths only (no SentencePiece build needed).
# See README.md in this directory for the step-by-step explanation.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
# Make the repository importable for the `python -m unilid...` steps when the
# package is not pip-installed (a pip install -e . makes this redundant).
REPO="$(cd ../.. && pwd)"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== step 1: generate the toy corpora (deterministic) ==="
"$PY" make_data.py

echo "=== step 2: train the base model on three toy languages ==="
"$PY" ../../train.py \
    --fasttext work/train_base.txt \
    --vocab-size 300 \
    --byte-level \
    --per-lang-counts-method soft \
    --results-dir work/results \
    --no-reuse-corpus

echo "=== step 2b: pack the trained tokenizers into a version-1 .unilid ==="
"$PY" ../../convert.py work/results -o work/toy_base.unilid

echo "=== step 3: build and bundle the calibration (version-2 .unilid) ==="
"$PY" build_calibration.py

echo "=== step 4: add the fourth language from its own data alone ==="
# With the package pip-installed, the console script form is:
#   unilid-add-language work/toy_calibrated.unilid ddd_Latn data/ddd_Latn_train.txt \
#       -o work/toy_extended.unilid --method em
"$PY" -m unilid.add_language work/toy_calibrated.unilid ddd_Latn \
    data/ddd_Latn_train.txt -o work/toy_extended.unilid --method em

echo "=== step 5: compare predictions before and after ==="
"$PY" predict_demo.py
