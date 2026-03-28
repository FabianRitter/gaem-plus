#!/bin/bash
# Submit all downstream evaluation jobs in parallel (1 GPU each).
# Run AFTER create_all_merges.sh has completed.
#
# Usage: bash scripts/submit_all_evals.sh

PROJECT="/data/projects/12004380/fabian/generalized_model_merging"
SCRIPT="${PROJECT}/scripts/eval_single_task.sh"
LOGDIR="${PROJECT}/logfiles/downstream_results"

mkdir -p "$LOGDIR"

submit() {
    local METHOD=$1
    local TASK=$2
    local MODEL_BASE=$3
    local NAME="E_${METHOD}_${TASK}"
    # Truncate PBS job name to 15 chars
    local SHORT_NAME=$(echo "$NAME" | cut -c1-15)
    local LOG="${LOGDIR}/${METHOD}/${TASK}.log"
    mkdir -p "$(dirname $LOG)"

    # Skip if results already exist
    local RESULT_DIR="${PROJECT}/results/downstream_results/${METHOD}/${TASK}"
    if [ -f "${RESULT_DIR}/test_predict.txt" ]; then
        echo "SKIP: $METHOD / $TASK (already done)"
        return
    fi

    echo "Submitting: $METHOD / $TASK"
    qsub -N "$SHORT_NAME" \
         -o "$LOG" \
         -v "method=${METHOD},task=${TASK},model_base=${MODEL_BASE}" \
         "$SCRIPT"
}

echo "============================================"
echo "Submitting parallel eval jobs"
echo "============================================"

# Global Procrustes methods (from results/exp1_merge/hf_models/)
for METHOD in procrustes_avg_09_01 procrustes_avg_07_03; do
    for TASK in asr genre_gtzan vocalset_singer_id vocalset_technique_id; do
        submit "$METHOD" "$TASK" "results/exp1_merge/hf_models"
    done
done

# Simple average baseline
for TASK in asr vocalset_singer_id vocalset_technique_id; do
    submit "simple_avg" "$TASK" "results/exp1_merge/hf_models"
done

# Per-layer Procrustes methods (from results/exp1b_perlayer/hf_models/)
for METHOD in perlayer_procrustes_09_01 perlayer_procrustes_07_03; do
    for TASK in asr genre_gtzan vocalset_singer_id vocalset_technique_id; do
        submit "$METHOD" "$TASK" "results/exp1b_perlayer/hf_models"
    done
done

echo ""
echo "All jobs submitted. Monitor with: qstat -u \$USER"
