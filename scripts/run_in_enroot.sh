#!/bin/bash
# Run a Python script inside the enroot container with GPU access.
# Usage: bash scripts/run_in_enroot.sh <python_script> [args...]
# Example: bash scripts/run_in_enroot.sh experiments/exp0_analysis/run_exp0.py

SCRIPT="$1"
shift
EXTRA_ARGS="$@"

PROXY="http://10.104.4.124:10104"

if [ -z "$SCRIPT" ]; then
    echo "Usage: $0 <python_script> [args...]"
    exit 1
fi

echo "Running: $SCRIPT $EXTRA_ARGS"
echo "Container: fabianritterg_torch2_nocuda (s3prl_old_cuda)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

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
cd /data/projects/12004380/fabian/generalized_model_merging
pip install soundfile -q 2>/dev/null
python $SCRIPT $EXTRA_ARGS
"
