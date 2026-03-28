#!/bin/bash
#PBS -N gaem_create_merges
#PBS -P 12004380
#PBS -q normal
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /data/projects/12004380/fabian/generalized_model_merging/logfiles/create_all_merges.log
#PBS -l select=1:ncpus=14:mem=253gb:ngpus=1:container_engine=enroot
#PBS -l container_image=~/images/fabianritterg_torch2_nocuda.sqsh
#PBS -l container_name=fabianritterg_torch2_nocuda
#PBS -l enroot_env_file=~/sample_jobs/container_env.conf

# Creates ALL merged models (global + per-layer Procrustes) in one short job.
# After this completes, submit the per-task eval jobs.

PROJECT="/data/projects/12004380/fabian/generalized_model_merging"
PROXY="http://10.104.4.124:10104"

echo "=== Creating all merged models: $(date) ==="

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
cd ${PROJECT}
export PYTHONPATH=${PROJECT}:\$PYTHONPATH
pip install soundfile -q 2>/dev/null

# Step 1: Create global Procrustes 0.9/0.1 and 0.1/0.9 (if not exist)
python -c '
import torch, numpy as np, sys
from pathlib import Path
from transformers import HubertModel

PROJECT = Path(\"${PROJECT}\")
sys.path.insert(0, str(PROJECT))
from gaem.alignment.procrustes import align_state_dict_orthogonal

O = torch.from_numpy(np.load(str(PROJECT / \"results/exp0_analysis/procrustes_O.npy\"))).float()
print(\"Loading models...\")
hubert = HubertModel.from_pretrained(\"facebook/hubert-base-ls960\")
mert = HubertModel.from_pretrained(\"m-a-p/MERT-v0-public\", trust_remote_code=True)
hubert_sd = {k: v.cpu() for k, v in hubert.state_dict().items()}
mert_sd = {k: v.cpu() for k, v in mert.state_dict().items()}
mert_aligned = align_state_dict_orthogonal(hubert_sd, mert_sd, O)

for ah, am, name in [(0.9, 0.1, \"procrustes_avg_09_01\"), (0.1, 0.9, \"procrustes_avg_01_09\")]:
    out = PROJECT / \"results/exp1_merge/hf_models\" / name
    if out.exists():
        print(f\"  {name}: exists, skip\")
        continue
    merged = {k: ah*hubert_sd[k] + am*mert_aligned.get(k, hubert_sd[k]) for k in hubert_sd}
    m = HubertModel.from_pretrained(\"facebook/hubert-base-ls960\")
    m.load_state_dict(merged)
    m.save_pretrained(str(out))
    print(f\"  Created {name}\")
print(\"Global merges done.\")
'

# Step 2: Create per-layer Procrustes models
python experiments/exp1b_perlayer_procrustes/run_exp1b.py

echo '=== All merges created: $(date) ==='
"
