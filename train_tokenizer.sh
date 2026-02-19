#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=logs/tok_%x_%j.log
#SBATCH --error=logs/tok_%x_%j.err

# Usage:
#   sbatch --job-name=tok-apertus-per_lang train_tokenizer.sh apertus per_lang
#   sbatch --job-name=tok-bpe-fine_family   train_tokenizer.sh bpe fine_family

set -euo pipefail

BASE_METHOD=${1:?"Usage: train_tokenizer.sh <base-method> <grouping>"}
GROUPING=${2:?"Usage: train_tokenizer.sh <base-method> <grouping>"}

# ── Cluster config (adjust these) ──
VENV_DIR="${VENV_DIR:-/users/ayavuz/venvs/unilid}"
UNILID_DIR="${UNILID_DIR:-/users/ayavuz/UNILID}"
RESULTS_BASE="${RESULTS_BASE:-/iopsstor/scratch/cscs/ayavuz/tokenizer_results}"

# ── Activate environment ──
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${UNILID_DIR}:${PYTHONPATH:-}"

# ── Create log dir ──
mkdir -p logs

echo "=============================="
echo "Job: ${BASE_METHOD} x ${GROUPING}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Mem:  ${SLURM_MEM_PER_NODE:-N/A}"
echo "Start: $(date)"
echo "=============================="

python "${UNILID_DIR}/train_tokenizer.py" \
    --base-method "${BASE_METHOD}" \
    --grouping "${GROUPING}" \
    --results-dir "${RESULTS_BASE}/results_${BASE_METHOD}_${GROUPING}" \
    --data-dir /capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset \
    --apertus-path /users/ayavuz/tokenizer_train/tokenizer.json \
    --coarse-families /users/ayavuz/tokenizer_train/coarse_families.json \
    --fine-families /users/ayavuz/tokenizer_train/fine_families.json \
    --code-families /users/ayavuz/tokenizer_train/code_lang_families.yaml \
    --base-samples 10000 \
    --batch-size 10 \
    --num-em-iterations 10 \
    --seed 42

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
