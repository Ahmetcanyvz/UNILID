#!/bin/bash
#SBATCH --job-name=unilid-fineweb2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=unilid_train_%j.log
#SBATCH --error=unilid_train_%j.err

# ──────────────────────────────────────────────────────────────────────
# UNILID FineWeb2 Training on SLURM
#
# Submit: sbatch train_cluster.sh
# Monitor: tail -f unilid_train_<jobid>.log
# Resume: just resubmit — --skip-existing-langs picks up where it left off
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Adjust these paths ──
PARQUET_DIR="${PARQUET_DIR:-/path/to/fineweb2}"
RESULTS_DIR="${RESULTS_DIR:-results_100k}"
VENV_DIR="${VENV_DIR:-.venv}"

# ── Load modules (adjust for your cluster) ──
# module load python/3.10
# module load cmake gcc

# ── Activate virtual environment ──
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# ── Verify spm_train is available ──
if ! command -v spm_train &> /dev/null; then
    echo "ERROR: spm_train not found in PATH"
    echo "Build SentencePiece first or add it to PATH"
    exit 1
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Parquet dir: $PARQUET_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Python: $(which python)"
echo "Start: $(date)"
echo "──────────────────────────────────────────"

python train_cluster.py \
    --parquet-dir "$PARQUET_DIR" \
    --results-dir "$RESULTS_DIR" \
    --vocab-size 100000 \
    --base-samples 10000 \
    --lang-batch-size 10 \
    --base-training-method hf \
    --per-lang-method sp \
    --byte-level \
    --reuse-base \
    --skip-existing-langs \
    --seed 42

echo "──────────────────────────────────────────"
echo "End: $(date)"
echo "Done."
