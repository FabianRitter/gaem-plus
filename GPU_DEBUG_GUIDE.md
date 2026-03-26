# GPU Node Debugging Guide (NSCC Cluster)

## hold_node Job

The `hold_node.sh` script reserves a GPU node for interactive debugging. It runs a minimal GPU heartbeat (~5-10% utilization) to keep the scheduler from killing the job while you SSH in and work.

**Source script:** `/data/projects/12004380/fabian/task-arithmetic-speech-audio/s3prl/hold_node.sh`
**Local copy:** `ssl-phase1/hold_node.sh` (once repo is cloned)

## Workflow

### 1. Submit the hold job
```bash
qsub hold_node.sh
# Or from this directory:
qsub /data/projects/12004380/fabian/task-arithmetic-speech-audio/s3prl/hold_node.sh
```

### 2. Check job status
```bash
qstat -u $USER
# Look for job named "hold_node" — note the JOBID (e.g., 149915.pbs111)
```

### 3. Find the assigned node
```bash
export PBS_JOBID=149915.pbs111   # replace with your actual JOBID
qstat -f $PBS_JOBID | grep exec_host
# Output: exec_host = dgx2q-a-015/0  (example)
```

### 4. SSH into the node
```bash
ssh dgx2q-a-015   # replace with actual node name from step 3
```

### 5. Enter the enroot container
```bash
# List running containers
enroot list

# Start an interactive shell in the container
enroot start --mount /data/projects/12004380:/data/projects/12004380 \
    --mount /app/apps/cuda/12.2.2:/usr/local/cuda \
    --mount /usr/lib/x86_64-linux-gnu/nvidia:/usr/lib/x86_64-linux-gnu/nvidia \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    fabianritterg_torch2_nocuda \
    bash
```

### 6. Inside the container — debug your code
```bash
source /opt/conda/bin/activate s3prl_old_cuda
cd /data/projects/12004380/fabian/generalized_model_merging/
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Now you can run/debug any code interactively
```

## Resource Allocation
- **Queue:** normal
- **GPUs:** 1
- **CPUs:** 14
- **RAM:** 253 GB
- **Walltime:** 24 hours
- **Container:** `~/images/fabianritterg_torch2_nocuda.sqsh`

## Notes
- The hold_node job is currently **in queue** — it will become usable once it starts running.
- You can submit jobs and debug simultaneously using the same node if the hold job is running.
- The heartbeat script uses ~5-10% GPU, leaving ~90% for your debugging work.
- Proxy for internet inside container: `http://10.104.4.124:10104`
