#!/bin/bash
# Quick GPU test inside enroot container
PROXY="http://10.104.4.124:10104"

enroot start \
    --mount /data/projects/12004380:/data/projects/12004380 \
    --mount /app/apps/cuda/12.2.2:/usr/local/cuda \
    --mount /usr/lib/x86_64-linux-gnu/nvidia:/usr/lib/x86_64-linux-gnu/nvidia \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env CUDA_VISIBLE_DEVICES=1 \
    --env http_proxy=$PROXY \
    --env https_proxy=$PROXY \
    --env HTTP_PROXY=$PROXY \
    --env HTTPS_PROXY=$PROXY \
    fabianritterg_torch2_nocuda \
    bash -c '
source /opt/conda/bin/activate s3prl_old_cuda
echo "=== Python ==="
python --version
echo "=== PyTorch ==="
python -c "import torch; print(f\"torch {torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"
echo "=== Transformers ==="
python -c "import transformers; print(f\"transformers {transformers.__version__}\")"
echo "=== Test Complete ==="
'
