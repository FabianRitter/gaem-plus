#!/bin/bash
#PBS -N gaem_downstream
#PBS -P 12004380
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /data/projects/12004380/fabian/generalized_model_merging/logfiles/gaem_downstream_batch.log
#PBS -l select=1:ncpus=14:mem=253gb:ngpus=1:container_engine=enroot
#PBS -l container_image=~/images/fabianritterg_torch2_nocuda.sqsh
#PBS -l container_name=fabianritterg_torch2_nocuda
#PBS -l enroot_env_file=~/sample_jobs/container_env.conf

# ============================================================================
# GAEM+ Batch Downstream Evaluation
#
# Evaluates Procrustes-aligned merged HuBERT+MERT on genre_gtzan and singer_id.
# Each task runs sequentially on 1 GPU.
#
# Usage:
#   qsub scripts/run_gaem_downstream_batch.sh
# ============================================================================

PROJECT="/data/projects/12004380/fabian/generalized_model_merging"
S3PRL="${PROJECT}/ssl-phase1"

LIBRISPEECH="/data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech"
GTZAN="/data/projects/12004380/datasets/superb/superb/GTZAN"
VOCALSET="/data/projects/12004380/datasets/superb/superb/VocalSet"

PROXY="http://10.104.4.124:10104"

# Models to evaluate
METHODS="procrustes_avg_05 procrustes_avg_07_03 simple_avg"

# Tasks to run
TASKS="genre_gtzan vocalset_singer_id"

echo "============================================"
echo "GAEM+ Batch Downstream Evaluation"
echo "Methods: $METHODS"
echo "Tasks: $TASKS"
echo "Start: $(date)"
echo "============================================"

for METHOD in $METHODS; do
    HF_MODEL="${PROJECT}/results/exp1_merge/hf_models/${METHOD}"

    if [ ! -d "$HF_MODEL" ]; then
        echo "SKIP: $METHOD (model not found)"
        continue
    fi

    for TASK in $TASKS; do
        RESULT_DIR="${PROJECT}/results/exp1_downstream/${METHOD}/${TASK}"
        LOG="${PROJECT}/logfiles/exp1_downstream/${METHOD}/${TASK}.log"
        mkdir -p "$(dirname $LOG)"

        echo ""
        echo "========== $METHOD / $TASK =========="
        echo "Start: $(date)"

        # Build task-specific args
        case "$TASK" in
            genre_gtzan)
                CONFIG="./downstream/genre_gtzan/config.yaml"
                OVERRIDE="-o config.downstream_expert.datarc.file_path=${GTZAN},,config.downstream_expert.datarc.meta_data=${GTZAN},,config.downstream_expert.datarc.num_workers=8"
                BEST_CKPT="valid-best.ckpt"
                ;;
            vocalset_singer_id)
                CONFIG="./downstream/vocalset_singer_id/config.yaml"
                OVERRIDE="-o config.downstream_expert.datarc.file_path=${VOCALSET},,config.downstream_expert.datarc.meta_data=${VOCALSET},,config.downstream_expert.datarc.num_workers=8"
                BEST_CKPT="dev-best.ckpt"
                ;;
        esac

        # Run inside enroot
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
            bash -c "
source /opt/conda/bin/activate s3prl_old_cuda
cd ${S3PRL}/s3prl
export PYTHONPATH=${S3PRL}/s3prl:${S3PRL}:\$PYTHONPATH
pip install soundfile modelscope fvcore addict 2>&1 | tail -1

echo '=== Training: $METHOD / $TASK ==='
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
    echo '=== Evaluating: $METHOD / $TASK ==='
    python run_downstream.py -m evaluate --verbose \
        -e ${RESULT_DIR}/${BEST_CKPT} \
        -u hf_hubert_custom \
        -k ${HF_MODEL} \
        -d ${TASK} \
        -c ${CONFIG} \
        -s hidden_states \
        ${OVERRIDE}
fi
echo '=== Done: $METHOD / $TASK ==='
" 2>&1 | tee -a "$LOG"

        echo "End: $(date)"
    done
done

echo ""
echo "============================================"
echo "GAEM+ Batch Complete: $(date)"
echo "Results: ${PROJECT}/results/exp1_downstream/"
echo "============================================"
