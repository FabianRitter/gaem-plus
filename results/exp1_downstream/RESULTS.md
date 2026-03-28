# Experiment 1 Results: Downstream Evaluation of Merged Encoders

**Date**: 2026-03-27 (batch job 150067)
**GPU**: NVIDIA H100 80GB (node a2ap-dgx037)
**Task**: Genre classification (GTZAN, 10 genres, ~1000 audio clips)
**Metric**: Test accuracy (higher = better)

## Genre Classification (GTZAN) Results

| Method | Alignment | Weights (H:M) | Valid-Best Acc | Test Acc |
|--------|-----------|---------------|----------------|----------|
| **HuBERT Base** (no merging) | — | 1.0 : 0.0 | — | **70.04%** |
| Naive average (no alignment) | None | 0.5 : 0.5 | — | 41.28% * |
| Simple average (our run) | None | 0.5 : 0.5 | 54.82% | **44.48%** |
| Procrustes + avg | Orthogonal | 0.5 : 0.5 | 52.28% | **48.62%** |
| **Procrustes + avg** | **Orthogonal** | **0.7 : 0.3** | **57.36%** | **51.38%** |

\* From Fabian's prior experiments (different downstream hyperparameters possible)

## Key Observations

### 1. Procrustes alignment helps, but gains are moderate

- **Simple avg → Procrustes 0.5/0.5**: 44.5% → 48.6% (+4.1 pp, +9.2% relative)
- **Simple avg → Procrustes 0.7/0.3**: 44.5% → 51.4% (+6.9 pp, +15.5% relative)
- Procrustes with speech-heavy weighting (0.7 HuBERT, 0.3 MERT) performs best

### 2. Large gap remains vs unmerged HuBERT

- HuBERT alone: 70.04% on GTZAN
- Best merge: 51.38% — still **18.7 pp below** the unmerged model
- This suggests the global Procrustes alignment is insufficient to fully recover merged model quality on this task

### 3. Weighting matters more than expected

- 0.7/0.3 (speech-heavy) outperforms 0.5/0.5 by 2.8 pp on a **music** task
- This seems counterintuitive — MERT-heavy should favor music
- Possible explanation: HuBERT's representation space is more "regular" and serves as a better anchor; pulling MERT toward HuBERT's frame preserves more structure than equal weighting

### 4. VocalSet singer_id failed (data issue)

All 3 methods failed on VocalSet: missing `train_s.txt` metadata file. The VocalSet downstream task expects metadata splits at `{data_dir}/train_s.txt`, `{data_dir}/valid_s.txt`, `{data_dir}/test_s.txt` — these need to be generated from the VocalSet dataset.

## Comparison with Prior Permutation Results

**Important caveat**: The permutation results from Paper 1 used **2-layer distilled students** (not full 12-layer pretrained models), so direct comparison is not apples-to-apples:

| Setting | GenreID Acc |
|---------|------------|
| HuBERT Base 12L (no merge) | 70.04% |
| Permutation merge, 2L distilled, λ=0.9/0.1 | 56.27% |
| **Procrustes merge, 12L pretrained, λ=0.7/0.3** | **51.38%** |
| Simple average, 12L pretrained, λ=0.5/0.5 | 44.48% |
| Naive average, 12L (prior experiment) | 41.28% |

The permutation merge on 2L distilled students (56.27%) currently outperforms our Procrustes merge on full 12L models (51.38%). However:
- Different model depths (2L vs 12L)
- Different base models (distilled students share initialization; HuBERT+MERT don't)
- Different weight ratios (0.9/0.1 vs 0.7/0.3)

## Diagnosis: Why Is the Gap Large?

1. **Global Procrustes is too coarse.** A single 768×768 orthogonal matrix cannot capture the complex layer-specific, head-specific symmetries between HuBERT and MERT. The per-head permutation approach from Paper 1 may be more appropriate for attention layers.

2. **No shared initialization.** Task arithmetic merging works best when models share a pretrained base (θ_base). HuBERT and MERT are independently pretrained — the "task vector" framing doesn't naturally apply. The Procrustes alignment helps but cannot overcome fundamental representation differences.

3. **The 12L models are harder to merge than 2L students.** Deeper models have more layers where interference can compound. The distilled 2L students in Paper 1 were specifically trained to share a HuBERT-like structure, making them more mergeable by design.

4. **LoRS decomposition and TSV decorrelation haven't been applied yet.** The current results are alignment-only. The full GAEM+ pipeline (alignment + LoRS + TSV) should further improve results.

## Files

| File | Contents |
|------|----------|
| `procrustes_avg_05/genre_gtzan/log.log` | Full training log |
| `procrustes_avg_07_03/genre_gtzan/log.log` | Full training log |
| `simple_avg/genre_gtzan/log.log` | Full training log |
| `*/genre_gtzan/valid-best.ckpt` | Best downstream checkpoint |
| `*/genre_gtzan/test_predict.txt` | Test predictions |
