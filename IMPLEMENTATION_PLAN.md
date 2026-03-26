# GAEM+ Implementation Plan

**Author**: Fabian Ritter
**Date**: 2026-03-26
**Status**: Phase 0 in progress

---

## 1. What We Are Building

**GAEM+ (Generalized Audio Encoder Merging)** is a framework for merging pretrained audio encoders — HuBERT (speech), MERT (music), and BEaTs (audio events) — into a single encoder that retains multi-domain capabilities. The merged encoder then serves as a drop-in replacement in Speech/Audio LLM systems.

### The Pipeline

```
HuBERT (speech)  ──┐
                    │   Orthogonal      LoRS           Merge        Merged
MERT (music)    ──┤──► Alignment ──► Decomposition ──► (TIES/ ──► Encoder
                    │   (Procrustes)    (SVD+sparse)    TSV/avg)
BEaTs (audio)   ──┘
```

### Components (from 3 papers + our contribution)

| Component | Source Paper | What It Does | Status |
|-----------|-------------|-------------|--------|
| **Orthogonal alignment** | GLMC (Theus, ICML 2025) | Aligns model weight spaces using Procrustes O(d) transforms, exploiting rotation symmetry in transformer attention/FFN layers | `gaem/alignment/procrustes.py` — implemented, tested |
| **Permutation alignment** | Git Re-Basin (Ainsworth 2023) / Fabian Paper 1 | Baseline: Hungarian algorithm on correlation matrix. Subset of orthogonal (our Paper 1 approach) | `gaem/alignment/permutation.py` — implemented, tested |
| **Semi-permutation** | GLMC | Soft attention head alignment via Sinkhorn (doubly stochastic matrices) | `gaem/alignment/semi_permutation.py` — implemented |
| **LoRS decomposition** | LoRS-Merging (Zhao 2025) | Decomposes task vectors into low-rank (truncated SVD) + sparse (magnitude pruning) components | `gaem/decomposition/lors.py` — implemented, tested |
| **TSV decorrelation** | TSV (Gargiulo, CVPR 2025) | Procrustes orthogonalization of per-layer singular vectors to decorrelate inter-task directions | `gaem/evaluation/sti.py` — implemented, tested |
| **STI diagnostic** | TSV | Singular Task Interference metric: measures per-layer interference between domain task vectors | `gaem/evaluation/sti.py` — implemented, tested |
| **OSRM pre-orthogonalization** | OSRM (Zhang 2025) | Constrains pre-training subspaces to be orthogonal, reducing interference before merging | Deferred to Phase 2 |
| **Baselines** | Task Arithmetic, TIES, DARE | Standard merging baselines for ablation | `gaem/merging/task_arithmetic.py` — implemented, tested |
| **Full GAEM+ pipeline** | Our contribution | Combined pipeline: alignment + decomposition + merge + evaluation | `gaem/merging/gaem_plus.py` — implemented, tested |

---

## 2. Why This Approach

### 2.1 The Research Gap

The Model Merging survey (Yang et al., ACM Computing Surveys 2026) covers 72+ papers. **Zero** address audio/speech encoder merging. The survey's taxonomy reveals three gaps GAEM+ fills:

1. **Weight alignment is permutation-dominated.** The survey catalogs Git Re-Basin, OTFusion, etc. — all permutation-based. Orthogonal alignment (rotation group O(d)) strictly generalizes permutations (the symmetric group S_n is a subgroup of O(d)). The recent Zhang et al. (ICML 2025) proves rotation symmetry captures real functional equivalences in transformer attention that permutations miss.

2. **Subspace merging lacks structured decomposition.** TIES and DARE use unstructured magnitude pruning. LoRS decomposes task vectors into low-rank (structural) + sparse (scattered) components — a principled decomposition that allows different merge strategies per component.

3. **Audio is entirely absent.** No paper merges SSL audio encoders (HuBERT, MERT, BEaTs). Our two prior papers (permutation alignment for audio, task arithmetic distillation) are the only work in this space.

### 2.2 Why HuBERT + MERT First (Phase 0)

- Both are HuBERT-architecture (12 transformer layers, 768 hidden dim) — same width, compatible for direct merging
- Already integrated as s3prl upstreams (`hubert_base`, `mert_v0_public`)
- 11 downstream evaluation tasks already set up (5 speech SUPERB + 5 music MARBLE + 1 audio ESC-50)
- Validates the merging technique before the more expensive LLM integration
- Even though these models are "older," the research question (can orthogonal alignment outperform permutation for audio encoder merging?) is novel and publishable

### 2.3 Why Orthogonal > Permutation (Our Hypothesis)

Audio transformers use LayerNorm/RMSNorm, which is invariant under orthogonal transforms — not just permutations. When two independently-trained encoders (HuBERT on speech, MERT on music) learn equivalent functions in different coordinate frames, the true functional equivalence class is O(d), not S_n. Permutation alignment only explores a tiny fraction of this equivalence class (n! vs the continuous O(d) manifold).

**Prediction**: Orthogonal alignment will produce lower interpolation barriers and better downstream performance than permutation alignment, especially for cross-domain merges (speech+music) where the representation geometries are more divergent.

### 2.4 Why LoRS + TSV Together

LoRS and TSV are complementary:

| Aspect | LoRS | TSV |
|--------|------|-----|
| Decomposition | SVD + magnitude pruning (low-rank + sparse) | SVD + Procrustes decorrelation |
| What it does | Separates structural signal from noise in each task vector | Decorrelates inter-task directions within each layer |
| When to apply | Before merging (decompose task vectors) | During merging (decorrelate concatenated singular vectors) |
| Best for | Removing noise that would interfere with alignment | Reducing directional interference between domains |

**Our plan**: Apply LoRS first (clean up task vectors), then TSV's Procrustes on the low-rank components (decorrelate speech/music directions). The sparse components get merged via TIES.

### 2.5 Why Move to LLMs (Phase 1)

SSL downstream evaluation (ASR WER, genre classification accuracy) answers: *"Does the merged encoder retain multi-domain capabilities?"* But the compelling story for GAEM+ is: *"Can one merged encoder replace the dual-encoder setup in systems like SALMONN, reducing parameters while maintaining performance?"*

This requires evaluating the merged encoder inside an actual Speech/Audio LLM pipeline. SALMONN has open-source code. SpeechMapper has the best architecture for our purpose (modular encoder interface, cheap training) but no public repo — we may implement a minimal version ourselves.

---

## 3. The Two Phases

### Phase 0: SSL Merging Validation (Current)

**Location**: `/data/projects/12004380/fabian/generalized_model_merging/`
**Codebase**: `gaem/` library + `ssl-phase1/` (cloned s3prl, `gaem-plus` branch)
**Models**: HuBERT Base (768d, 12L) + MERT v0 (768d, 12L) — loaded via s3prl upstreams
**Evaluation**: 11 downstream tasks via s3prl's `run_downstream.py`

#### Experiments

**Exp 0: Feature Extraction & Interference Analysis**
- Extract hidden states from HuBERT and MERT on LibriSpeech dev-clean
- Compute STI (Singular Task Interference) layerwise
- Compute domain interference metrics (cosine, sign agreement)
- **Purpose**: Understand the interference landscape before trying to merge. Which layers conflict most?

**Exp 1: Alignment Ablation** (main result)
| Method | Alignment | Decomposition | Merge |
|--------|-----------|---------------|-------|
| Baseline: Weight Average | None | None | Average |
| Task Arithmetic | None | None | Weighted sum |
| TIES-Merging | None | None | TIES |
| Permutation + Average | Permutation (Hungarian) | None | Average |
| **Orthogonal + Average** | **Procrustes** | None | Average |
| Orthogonal + TIES | Procrustes | None | TIES |

- Evaluate all 6 methods across 11 downstream tasks
- Compute interpolation barriers for each alignment method
- **Hypothesis**: Orthogonal > Permutation > None

**Exp 2: LoRS + TSV Integration**
| Method | Alignment | Decomposition | Merge |
|--------|-----------|---------------|-------|
| Orth + Avg (from Exp 1) | Procrustes | None | Average |
| Orth + LoRS + Avg | Procrustes | LoRS | Avg(low-rank) + Avg(sparse) |
| Orth + LoRS + TIES | Procrustes | LoRS | Avg(low-rank) + TIES(sparse) |
| Orth + TSV-Merge | Procrustes | TSV decorrelation | TSV-Merge |
| **Full GAEM+** | **Procrustes** | **LoRS** | **Avg(low-rank) + TIES(sparse) + TSV decorrelation** |

- Sweep LoRS hyperparameters: rank_ratio ∈ {0.05, 0.1, 0.2, 0.5}, sparsity ∈ {0.7, 0.8, 0.9}
- **Hypothesis**: LoRS + TSV provides additional gains over alignment alone

**Exp 3: Weight Sweep**
- For best method from Exp 2, sweep merge weights: λ_speech ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
- Analyze trade-off: speech task performance vs music task performance as weights shift
- Compare Pareto frontier of GAEM+ vs baselines

### Phase 1: LLM Integration (Future)

**Location**: New directory (TBD), separate from s3prl
**Models**: Merged encoder from Phase 0 → Speech/Audio LLM
**Evaluation**: ST, SQA, Dynamic-SUPERB, or AIR-Bench

#### The LLM Framework Decision

| Framework | Feasibility | Pros | Cons |
|-----------|-------------|------|------|
| **SALMONN** | Easy (open source) | Highest impact, dual-encoder design is perfect target | Heavy (Vicuna 13B), compute-intensive |
| **SpeechMapper** | Moderate (no public repo) | Cheapest training (V100 pretrain + 1.5h A100 adapt), best for rapid iteration | Must implement ourselves |
| **WavLLM** | Easy (open source) | Modular encoder, good baseline | Less well-known |

**Current plan**: Start with SALMONN (has code). If compute is limiting, implement a minimal SpeechMapper-style projector. Both allow clean encoder swapping.

#### Phase 1 Experiments

**Exp 4: Encoder Replacement**
- Replace SALMONN's dual encoder (Whisper + BEaTs) with single GAEM+ merged encoder
- Compare: dual-encoder baseline vs merged-encoder on same tasks
- **Key metric**: performance retention at ~50% encoder parameters

**Exp 5: Extended Merge**
- 3-way: HuBERT + MERT + BEaTs
- Test within SALMONN pipeline
- Evaluate on cross-domain tasks (speech + music + audio events)

---

## 4. Key Technical Decisions

### 4.1 Merging Full Pretrained Models (Not Distilled Students)

We merge HuBERT and MERT directly, not via distilled students + task arithmetic. This means:

- **No shared initialization** (θ_base): HuBERT and MERT are independently pretrained. Standard task arithmetic assumes a shared base.
- **Solution**: Use alignment-based merging. Align MERT's weight space to HuBERT's using Procrustes on activation features, then interpolate: θ_merged = (1-α)·θ_HuBERT + α·align(θ_MERT).
- For LoRS/TSV components that need task vectors: use HuBERT as the reference frame and define τ_MERT = align(θ_MERT) - θ_HuBERT as the "task vector" in the aligned space.

### 4.2 Calibration Dataset for Alignment

Alignment matrices (Procrustes O, Permutation P) require activation features from both models on shared data. Following the approach from Paper 1 (permutation experiments), we use a **mixed-domain calibration set** so both encoders produce meaningful features:

- **5000 speech samples** from LibriSpeech train-clean-100 (randomly sampled)
- **5000 music samples** from music4all_16khz_new (randomly sampled, 69230 total files)
- **Combined CSV**: `data/calibration_10k.csv` (columns: index, file_path, length, label, domain)
- **Domain-specific CSVs**: `data/calibration_10k_speech.csv`, `data/calibration_10k_music.csv`

This is the same calibration strategy used in the permutation-merging experiments. Both HuBERT and MERT process these at 16kHz. The mixed dataset ensures alignment captures how both models represent both domains, not just one.

**Features extracted**: Last hidden state, time-averaged across frames → [N, 768]

**Data locations**:
- Speech: `/data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech/train-clean-100` (28539 files)
- Music: `/data/projects/12004380/datasets/music4all-all/music4all_16khz_new` (69230 files)
- Reference format: `ssl-phase1/s3prl/data/len_for_bucket/train-clean-100-5000-samples.csv`

Note: MERT v0 uses 16kHz (same as HuBERT). MERT v1 uses 24kHz — stick with v0 to avoid resampling.

### 4.3 Per-Layer vs Global Alignment

GLMC suggests per-layer alignment (different O matrix per transformer layer) because each layer has its own symmetry group. We implement both:

1. **Global alignment**: Single O computed from last-layer features, applied uniformly
2. **Per-layer alignment**: Extract features at each layer, compute O_l per layer

The ablation will reveal which matters more for audio encoders.

### 4.4 Evaluation Protocol

For Phase 0, evaluation uses s3prl's existing downstream framework:

```bash
# After merging, save the merged state dict as a checkpoint
# Then evaluate using s3prl:
python ssl-phase1/s3prl/run_downstream.py -m train \
    -u <merged_upstream> -d <task> -p <output_path>
```

We need to create a custom upstream that loads the merged checkpoint. This is a thin wrapper similar to `hubert_local`.

---

## 5. What's Ready Now

### Code

| Component | Path | Status |
|-----------|------|--------|
| Procrustes orthogonal alignment | `gaem/alignment/procrustes.py` | Tested |
| Permutation alignment | `gaem/alignment/permutation.py` | Tested |
| Semi-permutation (Sinkhorn) | `gaem/alignment/semi_permutation.py` | Implemented |
| LoRS decomposition | `gaem/decomposition/lors.py` | Tested |
| Task Arithmetic / TIES / DARE | `gaem/merging/task_arithmetic.py` | Tested |
| Full GAEM+ pipeline | `gaem/merging/gaem_plus.py` | Tested |
| Interpolation barriers | `gaem/evaluation/barriers.py` | Tested |
| Domain interference metrics | `gaem/evaluation/interference.py` | Tested |
| STI metric + TSV-Merge | `gaem/evaluation/sti.py` | Tested |
| Feature extraction utils | `gaem/utils/features.py` | Implemented |
| Checkpoint I/O | `gaem/utils/checkpoint.py` | Implemented |

### Infrastructure

| Item | Path | Status |
|------|------|--------|
| s3prl clone (gaem-plus branch) | `ssl-phase1/` | Ready |
| Merging utilities from Paper 1 | `ssl-phase1/s3prl/merging_utils/` | Imported |
| HuBERT upstream | `ssl-phase1/s3prl/upstream/hubert/` | Available |
| MERT upstream | `ssl-phase1/s3prl/upstream/hf_mert/` | Available |
| BEaTs upstream | `ssl-phase1/s3prl/upstream/beats/` | Available |
| Hold node script | `scripts/hold_node_gaem.sh` | Ready |
| Experiment configs | `experiments/exp1_alignment_ablation/` | Config ready |
| Datasets (LibriSpeech, GTZAN, VocalSet, NSynth, ESC-50) | `/data/projects/12004380/datasets/superb/superb/` | On disk |

### Literature

| Item | Status |
|------|--------|
| 78 papers tracked in Excel | Up to date |
| Core papers deep-read (GLMC, LoRS, OSRM, SpeechMapper, TSV) | Done |
| GLMC code reviewed (github.com/alexandertheus/...) | Identified, to review |
| LoRS code reviewed (github.com/qmgzhao/...) | Identified, minimal |
| OSRM code reviewed (github.com/illidanlab/OSRM) | Identified, complete |
| TSV code available (github.com/AntoAndGar/task_singular_vectors) | Identified, to review |
| Survey positioning analysis | Done |

---

## 6. What Needs GPU (Next Steps)

### Immediate (once hold_node is running)

1. **Load HuBERT and MERT** — verify both models load correctly inside the enroot container
2. **Extract features** — run both models on LibriSpeech dev-clean, save activations
3. **Compute alignment matrices** — Procrustes and permutation on real features
4. **Run STI analysis** — layerwise interference diagnosis on real model weights
5. **Merge and evaluate** — create merged checkpoints, run downstream eval

### Script Needed

A `run_exp0_analysis.py` script that:
1. Loads HuBERT and MERT via s3prl upstream interface
2. Extracts layerwise features on LibriSpeech data
3. Computes STI, cosine interference, sign agreement
4. Computes Procrustes O and permutation P alignment matrices
5. Creates merged checkpoints for all ablation variants
6. Saves everything to `results/exp0_analysis/`

A custom s3prl upstream for loading merged checkpoints (thin wrapper).

---

## 7. Public Code Repositories Referenced

| Paper | Repo | Use |
|-------|------|-----|
| GLMC | [alexandertheus/Generalized-LMC-for-Transformers](https://github.com/alexandertheus/Generalized-LMC-for-Transformers) | Study `weight_matching.py` for orthogonal alignment on GPT-2. Uses `POT` library. |
| LoRS-Merging | [qmgzhao/LoRS-Merging](https://github.com/qmgzhao/LoRS-Merging) | Reference `merge_fft-lors-merge.py` (75 lines). Minimal but shows core algorithm. |
| OSRM | [illidanlab/OSRM](https://github.com/illidanlab/OSRM) | Full training pipeline. Study `train_glue.py --init analytical` for orthogonal subspace initialization. For Phase 2. |
| TSV | [AntoAndGar/task_singular_vectors](https://github.com/AntoAndGar/task_singular_vectors) | Study SVD-based merging and STI metric implementation. |
| Yang Survey | [EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications](https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications) | Comprehensive paper list for positioning. |

---

## 8. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Orthogonal doesn't beat permutation for audio | Low (theory strongly supports it) | Still have LoRS + TSV contributions |
| HuBERT+MERT merged encoder loses too much on both domains | Medium | Weight sweep; fall back to anchor-weighted merge |
| MERT v0 features are incompatible with HuBERT features for alignment | Low (same architecture, 768d) | Use MERT v1-95M as backup (also 768d, newer) |
| No public SpeechLLM repo works for Phase 1 | Medium | SALMONN has code; or implement minimal projector ourselves |
| Compute constraints on NSCC | Medium | Phase 0 is cheap (single GPU, no training). Phase 1 needs planning. |
| Reviewer asks "why not just multi-task train?" | High | Prepare extensibility + efficiency arguments (CLAUDE.md Section: Critical Framing) |

---

## 9. Target Timeline

| Week | Milestone | Key Output |
|------|-----------|-----------|
| 1 | Exp 0: Feature extraction + interference analysis | STI heatmap, interference report |
| 1-2 | Exp 1: Alignment ablation (6 methods × 11 tasks) | Alignment comparison table |
| 3-4 | Exp 2: LoRS + TSV integration | Full GAEM+ results |
| 5 | Exp 3: Weight sweep + Pareto analysis | Optimal merge configuration |
| 6 | Phase 0 paper draft (method + SSL results) | Draft sections 1-4 |
| 7-8 | Phase 1: LLM integration setup | SALMONN/SpeechMapper running |
| 9-10 | Phase 1: LLM evaluation | Merged encoder in LLM results |
| 11-12 | Paper completion + submission | NeurIPS or ACL ARR |
