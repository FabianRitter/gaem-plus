#!/bin/bash
#PBS -N hold_node_gaem
#PBS -P 12004380
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /data/projects/12004380/fabian/generalized_model_merging/logfiles/hold_node_gaem.log
#PBS -l select=1:ncpus=14:mem=253gb:ngpus=1:container_engine=enroot
#PBS -l container_image=~/images/fabianritterg_torch2_nocuda.sqsh
#PBS -l container_name=fabianritterg_torch2_nocuda
#PBS -l enroot_env_file=~/sample_jobs/container_env.conf

# ============================================================================
# GAEM+ Hold Node — reserves a GPU for interactive debugging
#
# Usage:
#   qsub scripts/hold_node_gaem.sh
#
# Then SSH into the node (see GPU_DEBUG_GUIDE.md for full instructions):
#   export PBS_JOBID=<JOBID>
#   qstat -f $PBS_JOBID | grep exec_host
#   ssh <node-name>
# ============================================================================

python3 -c "
import torch, time
if torch.cuda.is_available():
    dev = torch.device('cuda:0')
    x = torch.randn(256, 256, device=dev)
    print(f'GAEM+ Hold Node: GPU {torch.cuda.get_device_name(0)} acquired', flush=True)
    while True:
        for _ in range(20):
            _ = x @ x
        time.sleep(5)
else:
    print('No GPU found, sleeping anyway', flush=True)
    import signal
    signal.pause()
"
