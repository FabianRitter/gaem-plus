#!/bin/bash
#PBS -P 12004380
#PBS -q normal
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -l select=1:ncpus=14:mem=253gb:ngpus=1:container_engine=enroot
#PBS -l container_image=~/images/fabianritterg_torch2_nocuda.sqsh
#PBS -l container_name=fabianritterg_torch2_nocuda
#PBS -l enroot_env_file=~/sample_jobs/container_env.conf

# Single-task downstream evaluation for a merged model.
# Submit with: qsub -N <name> -o <log> -v method=X,task=Y,model_base=Z eval_single_task.sh
#
# Required vars:
#   method   - e.g. procrustes_avg_09_01, perlayer_procrustes_09_01
#   task     - asr, genre_gtzan, vocalset_singer_id, vocalset_technique_id
#   model_base - path prefix: results/exp1_merge/hf_models or results/exp1b_perlayer/hf_models

PROJECT="/data/projects/12004380/fabian/generalized_model_merging"
S3PRL="${PROJECT}/ssl-phase1"
PROXY="http://10.104.4.124:10104"

# Data paths
LIBRISPEECH="/data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech"
GTZAN="/data/projects/12004380/datasets/superb/superb/GTZAN"
VOCALSET="/data/projects/12004380/fabian/task-arithmetic-speech-audio/s3prl/data/vocalset_processed"
S3PRL_ORIG="/data/projects/12004380/fabian/task-arithmetic-speech-audio/s3prl"

HF_MODEL="${PROJECT}/${model_base}/${method}"
RESULT_DIR="${PROJECT}/results/downstream_results/${method}/${task}"
LOG_DIR="${PROJECT}/logfiles/downstream_results/${method}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "method=$method task=$task"
echo "model=$HF_MODEL"
echo "result=$RESULT_DIR"
echo "Start: $(date)"
echo "============================================"

# Task-specific config
case "$task" in
    asr)
        CONFIG="./downstream/asr/config.yaml"
        OVERRIDE="-o config.downstream_expert.datarc.libri_root=${LIBRISPEECH},,config.downstream_expert.datarc.bucket_file=./data/len_for_bucket,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32"
        BEST="dev-clean-best.ckpt"
        EVAL_ARGS="-t test-clean"
        ;;
    genre_gtzan)
        CONFIG="./downstream/genre_gtzan/config_nscc.yaml"
        OVERRIDE="-o config.downstream_expert.datarc.file_path=${GTZAN},,config.downstream_expert.datarc.meta_data=${GTZAN}"
        BEST="valid-best.ckpt"
        EVAL_ARGS=""
        ;;
    vocalset_singer_id)
        CONFIG="./downstream/vocalset_singer_id/config_nscc.yaml"
        OVERRIDE="-o config.downstream_expert.datarc.file_path=${VOCALSET}"
        BEST="dev-best.ckpt"
        EVAL_ARGS=""
        ;;
    vocalset_technique_id)
        CONFIG="./downstream/vocalset_technique_id/config_nscc.yaml"
        OVERRIDE="-o config.downstream_expert.datarc.file_path=${VOCALSET}"
        BEST="dev-best.ckpt"
        EVAL_ARGS=""
        ;;
    *)
        echo "Unknown task: $task"
        exit 1
        ;;
esac

enroot start \
    --mount /data/projects/12004380:/data/projects/12004380 \
    --mount /app/apps/cuda/12.2.2:/usr/local/cuda \
    --mount /usr/lib/x86_64-linux-gnu/nvidia:/usr/lib/x86_64-linux-gnu/nvidia \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env http_proxy=$PROXY --env https_proxy=$PROXY \
    --env HTTP_PROXY=$PROXY --env HTTPS_PROXY=$PROXY \
    fabianritterg_torch2_nocuda \
    bash -c "
source /opt/conda/bin/activate s3prl_old_cuda
cd ${S3PRL}/s3prl
export PYTHONPATH=${S3PRL}/s3prl:${S3PRL}:\$PYTHONPATH
pip install soundfile modelscope fvcore addict 2>&1 | tail -1

echo '=== Train: ${method}/${task} ==='
python run_downstream.py -m train \
    -u hf_hubert_custom -k ${HF_MODEL} \
    -d ${task} -c ${CONFIG} -p ${RESULT_DIR} \
    -s hidden_states --verbose ${OVERRIDE}

if [ \$? -eq 0 ] && [ -f ${RESULT_DIR}/${BEST} ]; then
    echo '=== Eval: ${method}/${task} ==='
    python run_downstream.py -m evaluate --verbose \
        -e ${RESULT_DIR}/${BEST} \
        -u hf_hubert_custom -k ${HF_MODEL} \
        -d ${task} -c ${CONFIG} -s hidden_states \
        ${EVAL_ARGS} ${OVERRIDE}
fi
echo '=== Done: ${method}/${task} $(date) ==='
"
