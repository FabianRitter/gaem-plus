#!/bin/bash
# Run downstream evaluation for merged GAEM+ models inside enroot.
# This runs directly on the hold_node (not qsub).
#
# Usage:
#   bash scripts/run_downstream_merged.sh <method> <task> [gpu_id]
#
# Examples:
#   bash scripts/run_downstream_merged.sh procrustes_avg_05 asr 0
#   bash scripts/run_downstream_merged.sh simple_avg genre_gtzan 0

METHOD="${1:?Usage: $0 <method> <task> [gpu_id]}"
TASK="${2:?Usage: $0 <method> <task> [gpu_id]}"
GPU="${3:-0}"

PROJECT="/data/projects/12004380/fabian/generalized_model_merging"
S3PRL="/data/projects/12004380/fabian/generalized_model_merging/ssl-phase1"
HF_MODEL="${PROJECT}/results/exp1_merge/hf_models/${METHOD}"
RESULT_DIR="${PROJECT}/results/exp1_downstream/${METHOD}/${TASK}"
LOG_DIR="${PROJECT}/logfiles/exp1_downstream/${METHOD}"
LOG_FILE="${LOG_DIR}/${TASK}.log"

LIBRISPEECH="/data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech"
GTZAN="/data/projects/12004380/datasets/superb/superb/GTZAN"
VOCALSET="/data/projects/12004380/datasets/superb/superb/VocalSet"
NSYNTH="/data/projects/12004380/datasets/superb/superb/NSynth"
ESC50="/data/projects/12004380/datasets/superb/superb/esc50-v2.0.0-full"

PROXY="http://10.104.4.124:10104"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "GAEM+ Downstream Evaluation"
echo "  Method: $METHOD"
echo "  Task: $TASK"
echo "  GPU: $GPU"
echo "  Model: $HF_MODEL"
echo "  Results: $RESULT_DIR"
echo "  Log: $LOG_FILE"
echo "============================================"

if [ ! -d "$HF_MODEL" ]; then
    echo "ERROR: HF model not found at $HF_MODEL"
    exit 1
fi

# Build task-specific arguments
case "$TASK" in
    asr)
        DOWNSTREAM="asr"
        CONFIG="./downstream/asr/config.yaml"
        EXTRA_ARGS="-o config.downstream_expert.datarc.libri_root=${LIBRISPEECH},,config.downstream_expert.datarc.bucket_file=./data/len_for_bucket,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32"
        EVAL_SPLIT="test-clean"
        ;;
    genre_gtzan)
        DOWNSTREAM="genre_gtzan"
        CONFIG="./downstream/genre_gtzan/config.yaml"
        EXTRA_ARGS="-o config.downstream_expert.datarc.file_path=${GTZAN},,config.downstream_expert.datarc.meta_data=${GTZAN},,config.downstream_expert.datarc.num_workers=8"
        EVAL_SPLIT=""
        ;;
    vocalset_singer_id)
        DOWNSTREAM="vocalset_singer_id"
        CONFIG="./downstream/vocalset_singer_id/config.yaml"
        EXTRA_ARGS="-o config.downstream_expert.datarc.file_path=${VOCALSET},,config.downstream_expert.datarc.meta_data=${VOCALSET},,config.downstream_expert.datarc.num_workers=8"
        EVAL_SPLIT=""
        ;;
    *)
        echo "ERROR: Unknown task '$TASK'. Supported: asr, genre_gtzan, vocalset_singer_id"
        exit 1
        ;;
esac

enroot start \
    --mount /data/projects/12004380:/data/projects/12004380 \
    --mount /app/apps/cuda/12.2.2:/usr/local/cuda \
    --mount /usr/lib/x86_64-linux-gnu/nvidia:/usr/lib/x86_64-linux-gnu/nvidia \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env CUDA_VISIBLE_DEVICES=$GPU \
    --env http_proxy=$PROXY \
    --env https_proxy=$PROXY \
    --env HTTP_PROXY=$PROXY \
    --env HTTPS_PROXY=$PROXY \
    fabianritterg_torch2_nocuda \
    bash -c "
source /opt/conda/bin/activate s3prl_old_cuda
cd ${S3PRL}/s3prl
export PYTHONPATH=${S3PRL}/s3prl:${S3PRL}:\$PYTHONPATH

echo '=== Installing packages ==='
pip install soundfile modelscope fvcore addict 2>&1 | tail -3

echo '=== Environment ==='
python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'

echo '=== Training downstream: $TASK ==='
python run_downstream.py -m train \
    -u hf_hubert_custom \
    -k ${HF_MODEL} \
    -d ${DOWNSTREAM} \
    -c ${CONFIG} \
    -p ${RESULT_DIR} \
    -s hidden_states \
    --verbose \
    ${EXTRA_ARGS}

TRAIN_EXIT=\$?

if [ \$TRAIN_EXIT -eq 0 ]; then
    echo '=== Evaluating ==='
    # Find best checkpoint
    if [ -f ${RESULT_DIR}/dev-clean-best.ckpt ]; then
        EVAL_CKPT=${RESULT_DIR}/dev-clean-best.ckpt
    elif [ -f ${RESULT_DIR}/dev-best.ckpt ]; then
        EVAL_CKPT=${RESULT_DIR}/dev-best.ckpt
    elif [ -f ${RESULT_DIR}/valid-best.ckpt ]; then
        EVAL_CKPT=${RESULT_DIR}/valid-best.ckpt
    else
        echo 'No best checkpoint found, skipping eval'
        exit 0
    fi

    EVAL_ARGS=\"\"
    if [ -n \"${EVAL_SPLIT}\" ]; then
        EVAL_ARGS=\"-t ${EVAL_SPLIT}\"
    fi

    python run_downstream.py -m evaluate --verbose \
        -e \$EVAL_CKPT \
        -u hf_hubert_custom \
        -k ${HF_MODEL} \
        -d ${DOWNSTREAM} \
        -c ${CONFIG} \
        -s hidden_states \
        \$EVAL_ARGS \
        ${EXTRA_ARGS}
fi

echo '=== Done: $TASK ==='
" 2>&1 | tee "$LOG_FILE"
