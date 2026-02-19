#!/bin/bash
#SBATCH --job-name=tok-test
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/tok_test_%j.log
#SBATCH --error=logs/tok_test_%j.err

# Quick test job to verify setup and tune SLURM config.
#
# Runs a small subset: apertus × per_lang with minimal sampling.
# Adjust SLURM headers above, then:
#   sbatch test_tokenizer.sh
#
# Or run interactively (no SLURM):
#   bash test_tokenizer.sh

set -euo pipefail

# ── Cluster config (adjust these) ──
VENV_DIR="${VENV_DIR:-/users/ayavuz/venvs/unilid}"
UNILID_DIR="${UNILID_DIR:-/users/ayavuz/UNILID}"

# ── Activate environment ──
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${UNILID_DIR}:${PYTHONPATH:-}"

mkdir -p logs

echo "=============================="
echo "TEST JOB"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Start: $(date)"
echo "=============================="

# Quick sanity checks
echo "--- Python ---"
which python
python --version
echo "--- Imports ---"
python -c "
from unilid.trainers.standard_trainer import StandardUnigramLMTokenizer
from unilid.trainers.language_specific_trainer import LanguageSpecificUnigramLMTokenizer
import pyarrow.parquet as pq
import yaml
print('All imports OK')
"

echo "--- spm_train ---"
which spm_train && spm_train --help 2>&1 | head -3 || echo "WARNING: spm_train not found"

echo "--- Data check ---"
DATA_DIR="/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset"
echo "fineweb2 parquets: $(ls ${DATA_DIR}/fineweb2/*.parquet 2>/dev/null | wc -l)"
echo "fineweb shards:    $(ls ${DATA_DIR}/fineweb/*.parquet 2>/dev/null | wc -l)"
echo "starcoder parquets:$(ls ${DATA_DIR}/starcoder/*.parquet 2>/dev/null | wc -l)"
for d in finemath infimath megamath; do
    echo "${d}: $(ls ${DATA_DIR}/${d}/data.parquet 2>/dev/null && echo 'OK' || echo 'MISSING')"
done

echo "--- Apertus tokenizer ---"
APERTUS="/users/ayavuz/tokenizer_train/tokenizer.json"
python -c "
from tokenizers import Tokenizer
t = Tokenizer.from_file('${APERTUS}')
v = t.get_vocab()
print(f'Apertus vocab size: {len(v)}')
print(f'Pre-tokenizer: {t.pre_tokenizer}')
print(f'Decoder: {t.decoder}')
"

echo ""
echo "--- Running minimal training test ---"

# Run with very small settings to verify the pipeline works end-to-end
python "${UNILID_DIR}/train_tokenizer.py" \
    --base-method apertus \
    --grouping per_lang \
    --results-dir /users/ayavuz/test_output \
    --data-dir "${DATA_DIR}" \
    --apertus-path "${APERTUS}" \
    --coarse-families /users/ayavuz/tokenizer_train/coarse_families.json \
    --fine-families /users/ayavuz/tokenizer_train/fine_families.json \
    --code-families /users/ayavuz/tokenizer_train/code_lang_families.yaml \
    --batch-size 2 \
    --base-samples 100 \
    --num-em-iterations 2 \
    --seed 42

echo ""
echo "--- Test results ---"
ls -la /users/ayavuz/test_output/tokenizers/ | head -20
cat /users/ayavuz/test_output/training_report.json

echo ""
echo "=============================="
echo "TEST COMPLETE: $(date)"
echo "=============================="
