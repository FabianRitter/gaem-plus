#!/bin/bash
#PBS -N gaem_batch_v2
#PBS -P 12004380
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /data/projects/12004380/fabian/generalized_model_merging/logfiles/gaem_batch_v2.log
#PBS -l select=1:ncpus=14:mem=253gb:ngpus=1:container_engine=enroot
#PBS -l container_image=~/images/fabianritterg_torch2_nocuda.sqsh
#PBS -l container_name=fabianritterg_torch2_nocuda
#PBS -l enroot_env_file=~/sample_jobs/container_env.conf

# ============================================================================
# GAEM+ Batch V2: Full evaluation with ASR + VocalSet + more weight ratios
# ============================================================================

PROJECT="/data/projects/12004380/fabian/generalized_model_merging"
S3PRL="${PROJECT}/ssl-phase1"

LIBRISPEECH="/data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech"
GTZAN="/data/projects/12004380/datasets/superb/superb/GTZAN"
VOCALSET_AUDIO="/data/projects/12004380/fabian/task-arithmetic-speech-audio/s3prl/data/vocalset_processed/audio"

PROXY="http://10.104.4.124:10104"

# Methods to evaluate (HF model dirs)
METHODS="procrustes_avg_05 procrustes_avg_07_03 procrustes_avg_03_07"

# We also need the 0.9/0.1 merge - create it first
CREATE_EXTRA_MERGES=true

echo "============================================"
echo "GAEM+ Batch V2 Downstream Evaluation"
echo "Start: $(date)"
echo "============================================"

run_task() {
    local METHOD=$1
    local TASK=$2
    local CONFIG=$3
    local OVERRIDE=$4
    local BEST_CKPT=$5
    local EVAL_ARGS=$6

    local HF_MODEL="${PROJECT}/results/exp1_merge/hf_models/${METHOD}"
    local RESULT_DIR="${PROJECT}/results/exp1_downstream/${METHOD}/${TASK}"
    local LOG="${PROJECT}/logfiles/exp1_downstream/${METHOD}/${TASK}.log"
    mkdir -p "$(dirname $LOG)"

    if [ ! -d "$HF_MODEL" ]; then
        echo "SKIP: $METHOD (model not found at $HF_MODEL)"
        return
    fi

    # Skip if already completed
    if [ -f "${RESULT_DIR}/${BEST_CKPT}" ] && [ -f "${RESULT_DIR}/test_predict.txt" ]; then
        echo "SKIP: $METHOD / $TASK (already completed)"
        return
    fi

    echo ""
    echo "========== $METHOD / $TASK =========="
    echo "Start: $(date)"

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
        ${EVAL_ARGS} \
        ${OVERRIDE}
fi
echo '=== Done: $METHOD / $TASK ==='
" 2>&1 | tee -a "$LOG"

    echo "End: $(date)"
}

# ============================================================================
# Step 0: Create additional merged models (0.9/0.1 ratios)
# ============================================================================
if [ "$CREATE_EXTRA_MERGES" = true ]; then
    echo "=== Creating additional merged models ==="
    enroot start \
        --mount /data/projects/12004380:/data/projects/12004380 \
        --mount /app/apps/cuda/12.2.2:/usr/local/cuda \
        --mount /usr/lib/x86_64-linux-gnu/nvidia:/usr/lib/x86_64-linux-gnu/nvidia \
        --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        --env http_proxy=$PROXY \
        --env https_proxy=$PROXY \
        fabianritterg_torch2_nocuda \
        bash -c "
source /opt/conda/bin/activate s3prl_old_cuda
cd ${PROJECT}
export PYTHONPATH=${PROJECT}:\$PYTHONPATH
python -c '
import torch, numpy as np, sys, os
from pathlib import Path
from transformers import HubertModel

PROJECT = Path(\"${PROJECT}\")
EXP0 = PROJECT / \"results\" / \"exp0_analysis\"
MERGE = PROJECT / \"results\" / \"exp1_merge\"

# Load alignment matrix
O = torch.from_numpy(np.load(str(EXP0 / \"procrustes_O.npy\"))).float()

# Load models
print(\"Loading HuBERT...\")
hubert = HubertModel.from_pretrained(\"facebook/hubert-base-ls960\")
print(\"Loading MERT...\")
mert = HubertModel.from_pretrained(\"m-a-p/MERT-v0-public\", trust_remote_code=True)

hubert_sd = {k: v.cpu() for k, v in hubert.state_dict().items()}
mert_sd = {k: v.cpu() for k, v in mert.state_dict().items()}

sys.path.insert(0, str(PROJECT))
from gaem.alignment.procrustes import align_state_dict_orthogonal

mert_aligned = align_state_dict_orthogonal(hubert_sd, mert_sd, O)

# Create 0.9/0.1 (speech-heavy) and 0.1/0.9 (music-heavy)
for alpha_h, alpha_m, name in [(0.9, 0.1, \"procrustes_avg_09_01\"), (0.1, 0.9, \"procrustes_avg_01_09\")]:
    out_dir = MERGE / \"hf_models\" / name
    if out_dir.exists():
        print(f\"  {name}: already exists, skipping\")
        continue
    merged = {}
    for k in hubert_sd:
        if k in mert_aligned:
            merged[k] = alpha_h * hubert_sd[k] + alpha_m * mert_aligned[k]
        else:
            merged[k] = hubert_sd[k].clone()
    model = HubertModel.from_pretrained(\"facebook/hubert-base-ls960\")
    model.load_state_dict(merged)
    model.save_pretrained(str(out_dir))
    print(f\"  Saved {name} to {out_dir}\")

print(\"Done creating extra merges\")
'
" 2>&1
    METHODS="$METHODS procrustes_avg_09_01 procrustes_avg_01_09"
fi

# ============================================================================
# Step 1: Run downstream tasks
# ============================================================================

# VocalSet singer_id configs
SINGID_CONFIG="./downstream/vocalset_singer_id/config.yaml"
SINGID_OVERRIDE="-o config.downstream_expert.datarc.file_path=${VOCALSET_AUDIO},,config.downstream_expert.datarc.meta_data=./downstream/vocalset_singer_id/data,,config.downstream_expert.datarc.num_workers=8"
SINGID_BEST="dev-best.ckpt"

# VocalSet technique_id configs
VOCID_CONFIG="./downstream/vocalset_technique_id/config.yaml"
VOCID_OVERRIDE="-o config.downstream_expert.datarc.file_path=${VOCALSET_AUDIO},,config.downstream_expert.datarc.meta_data=./downstream/vocalset_technique_id/data,,config.downstream_expert.datarc.num_workers=8"
VOCID_BEST="dev-best.ckpt"

# ASR configs
ASR_CONFIG="./downstream/asr/config.yaml"
ASR_OVERRIDE="-o config.downstream_expert.datarc.libri_root=${LIBRISPEECH},,config.downstream_expert.datarc.bucket_file=./data/len_for_bucket,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32"
ASR_BEST="dev-clean-best.ckpt"
ASR_EVAL_ARGS="-t test-clean"

for METHOD in $METHODS; do
    # Singer ID
    run_task "$METHOD" "vocalset_singer_id" "$SINGID_CONFIG" "$SINGID_OVERRIDE" "$SINGID_BEST" ""

    # Technique ID
    run_task "$METHOD" "vocalset_technique_id" "$VOCID_CONFIG" "$VOCID_OVERRIDE" "$VOCID_BEST" ""

    # ASR (longest — run last per method)
    run_task "$METHOD" "asr" "$ASR_CONFIG" "$ASR_OVERRIDE" "$ASR_BEST" "$ASR_EVAL_ARGS"
done

echo ""
echo "============================================"
echo "GAEM+ Batch V2 Complete: $(date)"
echo "============================================"
