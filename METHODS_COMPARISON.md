# Methods Comparison: Alignment Approaches for HuBERT+MERT Merging

## Overview

This document compares alignment methods for merging HuBERT (speech) and MERT (music) encoders. Both are 12-layer transformer encoders with 768-dim hidden size, 12 attention heads (64 dims/head), and 3072-dim FFN intermediate.

---

## Method 1: Correlation-Permutation (Paper 1, Fabian Ritter)

**Source**: `matching_functions.py:match_tensors_permute()` + graph-based `model_merger.py`

### How It Works

1. **Per-node activation collection**: Forward both models on calibration data. At each graph node (between layers), collect activation statistics (covariance/correlation matrices).

2. **Per-node permutation**: At each PREFIX/POSTFIX node in the graph, compute a permutation matrix P using the Hungarian algorithm on the cross-correlation between HuBERT and MERT activations:
   ```
   P = Hungarian(|Corr(HuBERT_features, MERT_features)|)
   ```

3. **Graph-aware application**: The merge/unmerge matrices are applied consistently through the DAG:
   - `merge_node()`: P transforms output dimension of a layer (W → P @ W)
   - `unmerge_node()`: P^T transforms input dimension of next layer (W → W @ P^T)

4. **Multi-head attention handling** (`match_tensors_permute_MHA`):
   - Per-head permutation: For each of the 12 heads, a 64×64 permutation is computed independently
   - Head reordering: Optionally, the heads themselves can be permuted
   - This respects the Q_i·K_i^T structure within each head

5. **Interpolation**: After alignment, merge via weighted average:
   ```
   θ_merged = α·θ_HuBERT + (1-α)·P(θ_MERT)
   ```

### Key Properties
- **Per-layer, per-head**: Different permutation at each layer and each attention head
- **Activation-based**: Uses actual forward-pass statistics, not just weight similarities
- **Discrete**: Permutations are hard 0/1 assignments (from Hungarian algorithm)
- **Architecture-aware**: Graph structure ensures consistency across residual connections

### Alignment Space
The permutation group S_768 has 768! elements — enormous but still discrete. For per-head alignment, each head uses S_64 (64! permutations).

---

## Method 2: Global Procrustes (Current GAEM+ implementation)

**Source**: `gaem/alignment/procrustes.py:procrustes_orthogonal()`

### How It Works

1. **Feature extraction**: Forward both models on calibration data, extract last-layer hidden states (time-averaged): [N, 768].

2. **Single Procrustes solve**: Compute ONE orthogonal matrix O (768×768) minimizing:
   ```
   O = argmin_O ||X_HuBERT - X_MERT @ O||_F   s.t. O^T O = I
   ```
   Closed-form: O = U @ V^T where U Σ V^T = SVD(X_MERT^T @ X_HuBERT)

3. **Global application**: Apply O to all MERT parameters using heuristic layer-type inference:
   - Linear weights with matching dims: W → O^T @ W @ O (or W @ O, O^T @ W)
   - 1D biases: b → O^T @ b
   - LayerNorm: unchanged (invariant)
   - Non-768-dim params: unchanged

4. **Interpolation**: Same as Method 1.

### Key Properties
- **Global**: ONE matrix for the entire model
- **Feature-based**: Uses last-layer features only (not per-layer activations)
- **Continuous**: O is a continuous rotation/reflection in R^768 (much richer than permutations)
- **Architecture-agnostic**: Applied via name-based heuristics, not a graph

### Alignment Space
The orthogonal group O(768) is a continuous manifold of dimension 768×767/2 = 294,528. This is strictly richer than S_768 (permutations are a subgroup of O(768)).

---

## Method 3: Per-Layer Procrustes (Proposed next step)

**Not yet implemented.**

### How It Would Work

1. **Per-layer feature extraction**: At each of the 13 layers (CNN output + 12 transformer layers), extract hidden states from both models.

2. **Per-layer Procrustes**: Compute 13 different O_l matrices (768×768 each), one per layer:
   ```
   O_l = Procrustes(X_HuBERT_layer_l, X_MERT_layer_l)
   ```

3. **Layer-specific application**: At each transformer layer:
   - Self-attention Q/K/V inputs: apply O_l from the residual stream
   - Self-attention output: apply O_l^{-1} back to residual stream
   - FFN intermediate: apply O_l (or a separate O_l_ffn)

4. **Consistency constraint**: Must ensure O_l transforms are consistent across residual connections:
   ```
   residual + O_l(attn_output) must be compatible
   → need O_l to transform the residual stream at layer l
   ```

### Key Properties
- **Per-layer**: Different rotation at each depth
- **Continuous**: Full O(768) at each layer
- **Feature-based**: Uses layer-specific activations
- **Consistency challenge**: Residual connections mean transformations must compose correctly

### Why This Should Help
Exp 0 showed per-layer Procrustes achieves 55-62% feature alignment improvement at middle layers vs 35.7% for last-layer-only global Procrustes. The middle layers are where most of the model's capacity lies.

---

## Method 4: Per-Layer Permutation + Per-Head (Paper 1 approach on full 12L)

**Already implemented** in `model_merger.py`, results reported in Paper 1.

This is essentially Method 1 applied to full 12L HuBERT and MERT (not just 2L distilled students).

---

## Comparison Table

| Property | Corr-Perm (Paper 1) | Global Procrustes | Per-Layer Procrustes | Per-Layer Perm + Head |
|----------|---------------------|-------------------|---------------------|----------------------|
| **Alignment granularity** | Per-layer, per-head | Global (1 matrix) | Per-layer | Per-layer, per-head |
| **Transform type** | Permutation (discrete) | Orthogonal (continuous) | Orthogonal (continuous) | Permutation (discrete) |
| **Alignment space** | S_64 per head × 12 heads × 12 layers | O(768) | O(768) × 13 layers | S_64 per head × 12L |
| **Data needed** | Per-node activations on calibration set | Last-layer features | Per-layer features | Per-node activations |
| **Respects heads** | Yes (per-head P) | No (global O mixes heads) | Partially (O_l at residual stream) | Yes (per-head P) |
| **Residual consistency** | Enforced by graph | Not enforced | Must be enforced | Enforced by graph |
| **Feature space** | 64-dim per head | 768-dim | 768-dim per layer | 64-dim per head |
| **Compute cost** | O(K × 64^3) per head | O(768^3) once | O(768^3) × 13 | O(K × 64^3) per head |

---

## Why Global Procrustes Underperforms Per-Layer Permutation

### Exp 0/1 evidence:
- Global Procrustes: 35.7% feature alignment improvement → 51.4% genre accuracy (vs 70% HuBERT)
- Per-layer permutation (Paper 1, 12L): reported results in paper (TBD — awaiting paper numbers)

### Root causes:

1. **Head mixing**: A global 768×768 O matrix freely mixes information across the 12 attention heads. This is problematic because each head's Q_i·K_i^T attention computation expects head-internal structure. A permutation within each 64-dim head preserves this structure.

2. **Depth-uniformity**: A single O applied everywhere ignores that early layers (feature extraction) and late layers (task-specific) have different alignment needs. Exp 0 STI analysis showed early attention Q/K have 2.5× more interference than late FFN layers.

3. **Residual stream corruption**: When O is applied to all layers uniformly, the residual connections accumulate alignment errors. In the graph-based approach, merge/unmerge matrices cancel at residual connection points.

---

## Proposed GAEM+ Pipeline (combining best of both)

```
Step 1: Per-layer feature extraction (like Paper 1)
    ↓
Step 2: Per-layer Procrustes O_l (continuous, richer than permutation)
    ↓
Step 3: Per-head refinement: within each head, apply 64×64 Procrustes
    ↓
Step 4: Graph-aware application (ensure residual consistency)
    ↓
Step 5: LoRS decomposition on aligned task vectors
    ↓
Step 6: TSV decorrelation of low-rank components
    ↓
Step 7: TIES/DARE merge of sparse components
    ↓
Step 8: Reconstruct merged model
```

This combines:
- **Continuous alignment** (Procrustes) > discrete (permutation)
- **Per-layer specificity** (Paper 1's granularity)
- **Per-head structure preservation** (Paper 1's attention handling)
- **Structured decomposition** (LoRS + TSV, new to GAEM+)

---

## Next Experiments Needed

1. **Complete the weight sweep**: Evaluate all methods at λ = {0.1, 0.3, 0.5, 0.7, 0.9} on ASR + GenreID + SingerID + TechniqueID
2. **Per-layer Procrustes**: Implement and compare against global Procrustes and per-layer permutation
3. **LoRS + TSV on top of best alignment**: Apply decomposition to the aligned task vectors
4. **Compare against Paper 1 results**: Use the exact same downstream evaluation protocol for fair comparison
