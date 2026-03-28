#!/bin/bash
#PBS -N gaem_perlayer
#PBS -P 12004380
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /data/projects/12004380/fabian/generalized_model_merging/logfiles/gaem_perlayer.log
#PBS -l select=1:ncpus=14:mem=253gb:ngpus=1:container_engine=enroot
#PBS -l container_image=~/images/fabianritterg_torch2_nocuda.sqsh
#PBS -l container_name=fabianritterg_torch2_nocuda
#PBS -l enroot_env_file=~/sample_jobs/container_env.conf

# ============================================================================
# GAEM+ Per-Layer Procrustes: Create merges + downstream evaluation
#
# Step 1: Run exp1b to create per-layer aligned merged models
# Step 2: Evaluate on ASR, singer_id, technique_id (same tasks as batch v2)
# ============================================================================

PROJECT="/data/projects/12004380/fabian/generalized_model_merging"
S3PRL="${PROJECT}/ssl-phase1"

LIBRISPEECH="/data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech"
GTZAN="/data/projects/12004380/datasets/superb/superb/GTZAN"
VOCALSET_AUDIO="/data/projects/12004380/fabian/task-arithmetic-speech-audio/s3prl/data/vocalset_processed/audio"

PROXY="http://10.104.4.124:10104"

echo "============================================"
echo "GAEM+ Per-Layer Procrustes Pipeline"
echo "Start: $(date)"
echo "============================================"

# Helper function
run_enroot() {
    enroot start \
        --mount /data/projects/12004380:/data/projects/12004380 \
        --mount /app/apps/cuda/12.2.2:/usr/local/cuda \
        --mount /usr/lib/x86_64-linux-gnu/nvidia:/usr/lib/x86_64-linux-gnu/nvidia \
        --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        --env CUDA_VISIBLE_DEVICES=0 \
        --env http_proxy=$PROXY \
        --env https_proxy=$PROXY \
        --env HTTP_PROXY=$PROXY \
        --env HTTPS_PROXY=$PROXY \
        fabianritterg_torch2_nocuda \
        bash -c "$1"
}

# ============================================================================
# Step 1: Create per-layer aligned merged models
# ============================================================================
echo ""
echo "========== Step 1: Create Per-Layer Procrustes Merged Models =========="
run_enroot "
source /opt/conda/bin/activate s3prl_old_cuda
cd ${PROJECT}
export PYTHONPATH=${PROJECT}:\$PYTHONPATH
pip install soundfile -q 2>/dev/null
python experiments/exp1b_perlayer_procrustes/run_exp1b.py
"

# Check if models were created
if [ ! -d "${PROJECT}/results/exp1b_perlayer/hf_models/perlayer_procrustes_09_01" ]; then
    echo "ERROR: Per-layer merged models not created. Aborting."
    exit 1
fi

echo ""
echo "========== Step 2: Downstream Evaluation =========="

METHODS="perlayer_procrustes_09_01 perlayer_procrustes_07_03"

run_downstream() {
    local METHOD=$1
    local TASK=$2
    local CONFIG=$3
    local OVERRIDE=$4
    local BEST_CKPT=$5
    local EVAL_ARGS=$6

    local HF_MODEL="${PROJECT}/results/exp1b_perlayer/hf_models/${METHOD}"
    local RESULT_DIR="${PROJECT}/results/exp1b_downstream/${METHOD}/${TASK}"
    local LOG="${PROJECT}/logfiles/exp1b_downstream/${METHOD}/${TASK}.log"
    mkdir -p "$(dirname $LOG)"

    if [ -f "${RESULT_DIR}/${BEST_CKPT}" ] && [ -f "${RESULT_DIR}/test_predict.txt" ]; then
        echo "SKIP: $METHOD / $TASK (already completed)"
        return
    fi

    echo ""
    echo "--- $METHOD / $TASK ---"
    echo "Start: $(date)"

    run_enroot "
source /opt/conda/bin/activate s3prl_old_cuda
cd ${S3PRL}/s3prl
export PYTHONPATH=${S3PRL}/s3prl:${S3PRL}:\$PYTHONPATH
pip install soundfile modelscope fvcore addict 2>&1 | tail -1

python run_downstream.py -m train \
    -u hf_hubert_custom \
    -k ${HF_MODEL} \
    -d ${TASK} \
    -c ${CONFIG} \
    -p ${RESULT_DIR} \
    -s hidden_states \
    --verbose \
    ${OVERRIDE}

if [ \$? -eq 0 ] && [ -f ${RESULT_DIR}/${BEST_CKPT} ]; then
    python run_downstream.py -m evaluate --verbose \
        -e ${RESULT_DIR}/${BEST_CKPT} \
        -u hf_hubert_custom \
        -k ${HF_MODEL} \
        -d ${TASK} \
        -c ${CONFIG} \
        -s hidden_states \
        ${EVAL_ARGS} \
        ${OVERRIDE}
fi
" 2>&1 | tee -a "$LOG"

    echo "End: $(date)"
}

for METHOD in $METHODS; do
    # Genre GTZAN
    run_downstream "$METHOD" "genre_gtzan" \
        "./downstream/genre_gtzan/config.yaml" \
        "-o config.downstream_expert.datarc.file_path=${GTZAN},,config.downstream_expert.datarc.meta_data=${GTZAN},,config.downstream_expert.datarc.num_workers=8" \
        "valid-best.ckpt" ""

    # Singer ID
    run_downstream "$METHOD" "vocalset_singer_id" \
        "./downstream/vocalset_singer_id/config.yaml" \
        "-o config.downstream_expert.datarc.file_path=${VOCALSET_AUDIO},,config.downstream_expert.datarc.meta_data=./downstream/vocalset_singer_id/data,,config.downstream_expert.datarc.num_workers=8" \
        "dev-best.ckpt" ""

    # Technique ID
    run_downstream "$METHOD" "vocalset_technique_id" \
        "./downstream/vocalset_technique_id/config.yaml" \
        "-o config.downstream_expert.datarc.file_path=${VOCALSET_AUDIO},,config.downstream_expert.datarc.meta_data=./downstream/vocalset_technique_id/data,,config.downstream_expert.datarc.num_workers=8" \
        "dev-best.ckpt" ""

    # ASR (longest — run last)
    run_downstream "$METHOD" "asr" \
        "./downstream/asr/config.yaml" \
        "-o config.downstream_expert.datarc.libri_root=${LIBRISPEECH},,config.downstream_expert.datarc.bucket_file=./data/len_for_bucket,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32" \
        "dev-clean-best.ckpt" "-t test-clean"
done

echo ""
echo "============================================"
echo "Per-Layer Procrustes Pipeline Complete: $(date)"
echo "Results: ${PROJECT}/results/exp1b_downstream/"
echo "============================================"
