# Project Structure

## Repository Layout

This project has **two git repositories**:

### 1. `generalized_model_merging/` (this repo — NEW)

The main GAEM+ research project. Contains the core merging library, experiment configs, scripts, and documentation. **This is where new code lives.**

```
generalized_model_merging/          ← git repo (this one)
├── CLAUDE.md                       # Agent instructions for Excel management & reading notes
├── IMPLEMENTATION_PLAN.md          # What we're building, why, phased timeline
├── PROJECT_STRUCTURE.md            # This file
├── GPU_DEBUG_GUIDE.md              # How to use hold_node for interactive GPU debugging
├── orchestration-draft-idea.md     # Original multi-agent orchestration idea (historical)
│
├── GAEM_Research_Matrices.xlsx     # 78 papers + 10 LLM frameworks (.gitignored, binary)
│
├── gaem/                           # ★ GAEM+ core library (standalone, reusable)
│   ├── __init__.py                 #   Package root (v0.1.0)
│   ├── setup.py                    #   pip install -e .
│   ├── alignment/                  #   Procrustes, permutation, semi-permutation
│   │   ├── procrustes.py           #     Orthogonal alignment (GLMC)
│   │   ├── permutation.py          #     Correlation-based permutation (Paper 1)
│   │   └── semi_permutation.py     #     Sinkhorn soft head alignment
│   ├── decomposition/              #   Task vector decomposition
│   │   └── lors.py                 #     Low-rank + sparse (LoRS-Merging)
│   ├── merging/                    #   Merge strategies
│   │   ├── task_arithmetic.py      #     Baselines: Task Arithmetic, TIES, DARE
│   │   └── gaem_plus.py            #     ★ Full GAEM+ pipeline + ablation runner
│   ├── evaluation/                 #   Analysis & metrics
│   │   ├── barriers.py             #     Interpolation barrier analysis
│   │   ├── interference.py         #     Domain interference (cosine, sign agreement)
│   │   └── sti.py                  #     STI metric + TSV-Merge (CVPR 2025)
│   ├── utils/                      #   I/O & feature extraction
│   │   ├── features.py             #     Extract features from audio encoders
│   │   └── checkpoint.py           #     Load/save (s3prl + HuggingFace formats)
│   └── tests/                      #   Unit tests (all passing)
│       └── test_core.py
│
├── experiments/                    # Experiment configs and entry-point scripts
│   ├── exp1_alignment_ablation/    #   Perm vs Orthogonal vs None (main result)
│   │   ├── config.yaml
│   │   └── run_exp1.py
│   ├── exp2_lors_integration/      #   LoRS + TSV ablation
│   │   └── config.yaml
│   ├── exp3_multi_encoder/         #   3-way merge (HuBERT+MERT+BEaTs)
│   └── exp4_llm_integration/       #   Phase 1: merged encoder → Speech LLM
│
├── scripts/                        # Cluster job scripts & utilities
│   ├── hold_node_gaem.sh           #   PBS job to reserve GPU for debugging
│   ├── monitor_jobs.sh             #   Cron-compatible job status monitor
│   ├── create_calibration_csv.py   #   Build 5k music + 5k speech calibration set
│   └── wait_and_create_calibration.sh  # Background: wait for download then create CSV
│
├── data/                           # Generated data files (.gitignored)
│   └── calibration_10k.csv         #   (will be created after music4all download)
│
├── paper-merging-idea/             # Research proposals & notes (historical)
│   ├── Revised_Research_Analysis 2.md
│   ├── Technical_Supplement_GAEM 2.md
│   ├── PhD_Research_Plan_Fabian_Ritter 2.docx
│   └── GAEM_Plus_Research_Proposal-updated.pptx
│
├── paper-readings/                 # Deep reading notes (linked to Excel IDs)
│   ├── _template.md
│   ├── model-merging/
│   ├── speech-audio-llms/
│   │   └── 73_speechmapper.md
│   ├── audio-encoders/
│   │   ├── analysis_orthogonal_subspaces_ssl_speech.md
│   │   ├── BEST_RQ_Analysis.md
│   │   └── MERT_Paper_Analysis.md
│   └── theory-foundations/
│
├── results/                        # Experiment outputs (.gitignored)
└── logfiles/                       # Job & monitoring logs (.gitignored)
```

### 2. `ssl-phase1/` (cloned s3prl — EXISTING repo, new branch)

A clone of `github.com/FabianRitter/task-arithmetic-speech-audio.git` on branch **`gaem-plus`**. Contains the s3prl framework with distillation, downstream evaluation, and the merging_utils from the `permutation-merging` branch.

**This is NOT where new GAEM+ code goes.** It's the evaluation infrastructure — we load HuBERT/MERT via its upstream system and run downstream tasks through its `run_downstream.py`.

```
ssl-phase1/                         ← git repo (separate, branch: gaem-plus)
├── s3prl/
│   ├── upstream/
│   │   ├── hubert/                 # HuBERT loader (hubert_base → 768d, 12L)
│   │   ├── hf_mert/               # MERT loader (mert_v0_public → 768d, 12L, 16kHz)
│   │   └── beats/                  # BEaTs loader
│   ├── downstream/                 # 11 eval tasks (5 SUPERB + 5 MARBLE + ESC-50)
│   ├── merging_utils/              # Cherry-picked from permutation-merging branch
│   │   ├── model_merger.py         # Existing permutation-based merger
│   │   ├── ties_merger.py          # TIES implementation
│   │   └── ...
│   ├── matching_functions.py       # Permutation alignment functions
│   ├── run_downstream.py           # Downstream eval entry point
│   ├── run_pretrain.py             # Distillation entry point
│   └── CLAUDE.md                   # s3prl-specific documentation
```

**Remote**: `origin → git@github.com:FabianRitter/task-arithmetic-speech-audio.git`
**Branches**: `main`, `permutation-merging` (Fabian's Paper 1), `gaem-plus` (this work)

## How the Two Repos Connect

```
gaem/  (core algorithms)
  │
  │  import gaem.alignment, gaem.merging, ...
  │
  ▼
experiments/run_exp1.py
  │
  │  1. Load HuBERT & MERT via s3prl upstream
  │  2. Extract features using gaem/utils/features.py
  │  3. Compute alignment (gaem/alignment/)
  │  4. Merge (gaem/merging/gaem_plus.py)
  │  5. Save merged checkpoint
  │
  ▼
ssl-phase1/s3prl/run_downstream.py
  │
  │  Evaluate merged checkpoint on 11 tasks
  │
  ▼
results/
```

The `gaem/` library is **standalone** — it has no dependency on s3prl. This means it can be reused in Phase 1 (LLM integration) with a different evaluation framework.

## Data Locations

| Dataset | Path | Status |
|---------|------|--------|
| LibriSpeech (all splits) | `/data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech/` | On disk (28539 files in train-clean-100) |
| music4all (16kHz) | `/data/projects/12004380/datasets/music4all-all/music4all_16khz_new/` | Downloading via rclone (69230 files) |
| GTZAN | `/data/projects/12004380/datasets/superb/superb/GTZAN/` | On disk |
| VocalSet | `/data/projects/12004380/datasets/superb/superb/VocalSet/` | On disk |
| NSynth | `/data/projects/12004380/datasets/superb/superb/NSynth/` | On disk |
| ESC-50 | `/data/projects/12004380/datasets/superb/superb/esc50-v2.0.0-full/` | On disk |
| Calibration CSV | `data/calibration_10k.csv` | Will be generated after music4all download |

## What Goes Where (Guidelines)

| I want to... | Where |
|--------------|-------|
| Add a new merging algorithm | `gaem/merging/` or `gaem/alignment/` |
| Add a new evaluation metric | `gaem/evaluation/` |
| Write an experiment script | `experiments/expN_name/` |
| Write a PBS job script | `scripts/` |
| Add a reading note for a paper | `paper-readings/{subfolder}/` |
| Modify downstream evaluation | `ssl-phase1/s3prl/downstream/` (on gaem-plus branch) |
| Add a new upstream model loader | `ssl-phase1/s3prl/upstream/` (on gaem-plus branch) |
| Track a new paper | Excel workbook (via CLAUDE.md instructions) |
