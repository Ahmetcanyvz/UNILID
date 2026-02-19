#!/bin/bash
# Submit all 9 tokenizer training jobs (3 base methods × 3 grouping levels).
#
# Usage:
#   bash submit_all.sh
#   bash submit_all.sh --dry-run    # print commands without submitting

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Commands that would be submitted:"
    echo
fi

mkdir -p logs

for method in apertus bpe unigram; do
    for grouping in per_lang fine_family coarse_family; do
        JOB_NAME="tok-${method}-${grouping}"
        CMD="sbatch --job-name=${JOB_NAME} train_tokenizer.sh ${method} ${grouping}"

        if $DRY_RUN; then
            echo "  ${CMD}"
        else
            echo "Submitting: ${method} × ${grouping}"
            ${CMD}
        fi
    done
done

if ! $DRY_RUN; then
    echo
    echo "All 9 jobs submitted. Monitor with: squeue -u \$USER"
fi
