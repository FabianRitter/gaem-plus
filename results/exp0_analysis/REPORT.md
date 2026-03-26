# Experiment 0 Report: Interference & Alignment Analysis

**Date**: 2026-03-26
**Models**: HuBERT Base (94.4M params) vs MERT v0 Public (94.4M params)
**Calibration data**: 1000 samples (500 speech + 500 music) from calibration_10k.csv
**GPU**: NVIDIA H100 80GB HBM3

## Key Findings

### 1. Weight-Space Analysis

| Metric | Value |
|--------|-------|
| Weight cosine similarity | **0.020** (near-orthogonal in weight space) |
| Relative L2 difference | **1.284** (MERT weights differ from HuBERT by 128% of HuBERT's norm) |

**Implication**: The models are in almost completely different coordinate frames. Direct weight averaging without alignment is essentially noise — alignment is critical.

### 2. Singular Task Interference (STI) — Layerwise

**Highest interference** (where merging will conflict most):
- Attention Q/K projections in early layers (layers 0-2): STI_norm ≈ 7.3-7.8
- Pattern: K-projections > Q-projections > V-projections consistently

**Lowest interference** (safest to merge):
- FFN output_dense across all layers: STI_norm ≈ 2.8-2.9
- FFN intermediate_dense in later layers: STI_norm ≈ 2.9

**Pattern**: Attention layers have ~2.5x more interference than FFN layers. Early attention layers are worst.

**Insight for GAEM+**: Consider layer-specific merging strategies — more aggressive alignment for attention Q/K, simpler averaging for FFN output layers.

### 3. Feature-Space Alignment

| Method | Alignment Error | Improvement |
|--------|----------------|-------------|
| No alignment | 1.1106 | — |
| Permutation (Hungarian) | 1.0358 | **6.7%** |
| **Procrustes (orthogonal)** | **0.7144** | **35.7%** |

**Procrustes outperforms permutation by 5.3x** in relative improvement (35.7% vs 6.7%).

### 4. Per-Layer Procrustes Alignment

| Layer | Before | After Procrustes | Improvement |
|-------|--------|-----------------|-------------|
| 0 (post-CNN) | 1.4954 | 0.6613 | 55.8% |
| 1 | 1.5613 | 0.7079 | 54.7% |
| 2 | 1.5933 | 0.6433 | 59.6% |
| 3 | 1.5753 | 0.6222 | 60.5% |
| 4 | 1.6162 | 0.6281 | 61.1% |
| **5** | **1.5665** | **0.5976** | **61.8%** |
| 6 | 1.5460 | 0.6192 | 59.9% |
| 7 | 1.5744 | 0.6350 | 59.7% |
| 8 | 1.5593 | 0.6330 | 59.4% |
| 9 | 1.4686 | 0.5862 | 60.1% |
| 10 | 1.4308 | 0.5807 | 59.4% |
| 11 | 1.5413 | 0.7087 | 54.0% |
| 12 (last) | 1.1105 | 0.7144 | 35.7% |

**Middle layers (3-10) benefit most** from Procrustes alignment (~59-62%). First and last layers benefit less.

### 5. Merged Checkpoints Created (Exp 1)

| Method | Checkpoint | Weights |
|--------|-----------|---------|
| Simple average | `merged_simple_avg.pt` | 0.5/0.5 |
| Permutation + avg | `merged_perm_avg_05.pt` | 0.5/0.5 |
| Procrustes + avg | `merged_procrustes_avg_05.pt` | 0.5/0.5 |
| Procrustes + avg | `merged_procrustes_avg_07_03.pt` | 0.7/0.3 (speech-heavy) |
| Procrustes + avg | `merged_procrustes_avg_03_07.pt` | 0.3/0.7 (music-heavy) |

## Next Steps

1. Create custom s3prl upstream to load merged checkpoints
2. Run downstream evaluation on key tasks (ASR, genre_gtzan, singer_id)
3. Compare Procrustes merge results against Fabian's existing permutation results
4. If Procrustes wins: proceed to LoRS + TSV decomposition experiments

## Files

| File | Contents |
|------|----------|
| `weight_interference.json` | Global cosine similarity, norm ratios |
| `layerwise_sti.json` | Per-parameter STI values |
| `alignment_results.json` | Per-layer alignment errors for all methods |
| `procrustes_O.npy` | 768×768 orthogonal alignment matrix |
| `permutation_P.npy` | 768×768 permutation matrix |
| `hubert_features_layer{0,6,12}.npy` | Saved features for further analysis |
| `mert_features_layer{0,6,12}.npy` | Saved features for further analysis |
