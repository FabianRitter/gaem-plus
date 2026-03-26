# Overnight Experiment Plan (2026-03-26 → 2026-03-27)

## Execution Environment

- **Node**: a2ap-dgx039 (H100 80GB)
- **Container**: enroot `fabianritterg_torch2_nocuda` with conda env `s3prl_old_cuda`
- **Packages**: torch 2.4.1+cu121, transformers 4.45.0, scipy 1.10.1
- **Claude runs directly on the node** — no SSH needed, use enroot to get full environment
- **All scripts run via**: `enroot start --mount ... bash -c "source activate && python script.py"`

## Goal

Run Experiment 0 (interference analysis) and if successful, proceed to Experiment 1 (alignment ablation + merging).

## What I Will Do

### Step 1: Write `experiments/exp0_analysis/run_exp0.py`

A self-contained Python script that:
1. Loads HuBERT (`hubert_base`) and MERT (`mert_v0_public`) via HuggingFace transformers (not s3prl — simpler, fewer dependencies)
2. Reads the calibration CSV (`data/calibration_10k.csv`)
3. For each model, extracts hidden states from all 12 transformer layers on the calibration data
4. Computes:
   - **Layerwise STI** between HuBERT and MERT weight matrices (no data needed, just state dicts)
   - **Domain interference** (cosine similarity, sign agreement between weight differences)
   - **Procrustes alignment matrix** O from last-layer features
   - **Permutation alignment matrix** P from last-layer features (for comparison)
   - **Alignment error** before/after Procrustes vs permutation
5. Saves all results to `results/exp0_analysis/` as JSON + numpy files

### Step 2: Write `scripts/run_exp0_enroot.sh`

A PBS job script that:
- Requests 1 GPU on the hold_node's queue
- Enters the enroot container (`fabianritterg_torch2_nocuda`)
- Mounts the project directory and datasets
- Installs any missing packages (scipy, soundfile)
- Runs `run_exp0.py`
- Logs everything to `logfiles/exp0/`

### Step 3: Submit and Monitor

1. `qsub scripts/run_exp0_enroot.sh`
2. Monitor via `qstat` and reading log files
3. If it fails: read the log, fix the code, resubmit

### Step 4: Debug Cycle

For each failure:
1. Read `logfiles/exp0/run_exp0.log` to find the error
2. Fix the code in `experiments/exp0_analysis/run_exp0.py`
3. Resubmit via `qsub`

Common issues I expect:
- **Import errors**: Container may not have all packages → add `pip install` in the PBS script
- **CUDA OOM**: H100 has 80GB, models are ~400MB each, should be fine. But batch size for feature extraction may need tuning.
- **Audio loading**: May need soundfile/librosa inside container → pip install in script
- **Path issues**: Container mounts vs host paths → use absolute paths consistently

### Step 5: Document Results

When exp0 succeeds, I will:
1. Save results to `results/exp0_analysis/`
2. Write `results/exp0_analysis/REPORT.md` summarizing findings
3. Commit everything to git and push
4. Sync Excel to OneDrive if any updates needed

## What I Will NOT Do

- **Not modify existing code** in `ssl-phase1/` or the original s3prl repo
- **Not start Experiment 1** (merging) — that depends on Exp 0 results
- **Not submit jobs to the `normal` queue** — only use the existing hold_node or `aiq1` queue
- **Not delete or overwrite** any existing data or checkpoints
- **Not push to any branch** other than `main` on `gaem-plus`

## How I Will Debug

```
Login node (where Claude runs)
    │
    ├── Write/fix Python code
    ├── Write PBS job scripts
    ├── qsub → submit job
    ├── qstat → check if running
    ├── Read logfiles/ → find errors
    └── Fix and resubmit
         │
Compute node (a2ap-dgx039, via PBS)
    │
    └── enroot container
        ├── pip install missing packages
        ├── Run run_exp0.py
        ├── GPU computation (feature extraction)
        └── Save results to shared filesystem
```

I cannot SSH interactively into the node. All debugging is:
1. Read logs from the shared filesystem
2. Fix code on login node
3. Resubmit PBS job

## Where Things Go

| Output | Location |
|--------|----------|
| Experiment code | `experiments/exp0_analysis/run_exp0.py` |
| PBS script | `scripts/run_exp0_enroot.sh` |
| Job logs | `logfiles/exp0/run_exp0.log` |
| Results (JSON) | `results/exp0_analysis/` |
| Summary report | `results/exp0_analysis/REPORT.md` |
| Git commits | `main` branch on `gaem-plus` repo |

## Expected Timeline

| Time | Action |
|------|--------|
| Now | Write run_exp0.py and PBS script |
| +10 min | First qsub submission |
| +15 min | Check logs, likely first failure (package issues) |
| +20 min | Fix and resubmit |
| +30-60 min | Feature extraction running (~10k audio files × 2 models) |
| +60-90 min | Results saved, report written |
| Morning | Fabian reviews results/exp0_analysis/REPORT.md |

## Success Criteria

Experiment 0 is done when I have:
- [ ] STI heatmap data (layerwise interference for all 12 layers)
- [ ] Domain interference metrics (cosine, sign agreement)
- [ ] Procrustes alignment matrix O (768×768)
- [ ] Alignment error comparison: no alignment vs permutation vs Procrustes
- [ ] Results committed to git
- [ ] REPORT.md written with findings
