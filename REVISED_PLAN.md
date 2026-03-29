# Revised Plan: GAEM+ Experiments

**Date**: 2026-03-29
**Reason for revision**: The global/per-layer Procrustes approach was structurally wrong — it applies O(768) across all attention heads, mixing head-internal representations. The correct approach follows GLMC's component-wise symmetry decomposition.

## What Went Wrong

We implemented Procrustes alignment as a single 768×768 orthogonal matrix per layer. This ignores that attention heads are independent 64-dim subspaces. The result: per-layer Procrustes underperforms Paper 1's per-head permutation by ~5 pp across all tasks, because it corrupts attention computations by mixing heads.

## What We Should Build

### The Correct GAEM+ Pipeline

**Replace permutations with orthogonal transforms at each node in Paper 1's graph**, preserving the component-wise structure:

```
For each transformer layer l:

1. Residual stream alignment:
   O_res ∈ O(768)  via Procrustes on layer input features
   (replaces: no explicit residual alignment in Paper 1)

2. Attention head alignment:
   For each head h = 1..12:
     O_h ∈ O(64)  via Procrustes on per-head output features
   (replaces: P_h ∈ S_64 in Paper 1's match_tensors_permute_MHA)

3. Head ordering:
   Semi-permutation via Sinkhorn on head-level similarity
   (replaces: hard head permutation in Paper 1)

4. FFN alignment:
   P_ffn ∈ S_3072 on FFN intermediate activations
   (same as Paper 1 — permutation is appropriate for ReLU/GELU neurons)

5. Apply transforms via graph to ensure residual consistency
   (same infrastructure as Paper 1's model_merger.py)
```

### Why This Should Beat Paper 1

At each node, O(n) strictly contains S_n. The Procrustes solution finds the optimal rotation, which is at least as good as the optimal permutation (and usually better since the true functional equivalence is continuous, not discrete).

- **Per-head O(64)**: 64×63/2 = 2016 continuous degrees of freedom vs 64! discrete permutations. The optimal rotation can capture soft feature mixing that permutation cannot.
- **Residual stream O(768)**: Captures rotation symmetries in LayerNorm that permutations miss entirely.
- **Semi-permutation for heads**: Allows soft head-to-head mapping (weighted combinations) vs hard 1-to-1 assignment.

### What We Keep From Paper 1

- The **graph infrastructure** (model_merger.py, HuBERTGraph, PREFIX/POSTFIX nodes)
- The **per-node activation collection** via hooks on calibration data
- The **evaluation protocol** (same 11 tasks, same configs, same train/eval splits)
- The **0.9/0.1 weight ratio** (proven to work best)

### Implementation Strategy

**Option A (recommended): Extend Paper 1's codebase**
- Fork model_merger.py → model_merger_procrustes.py
- Replace `match_tensors_permute_MHA` with `match_tensors_procrustes_MHA`
- Replace `match_tensors_permute` with `match_tensors_procrustes` for FFN/residual nodes
- Everything else stays the same (graph, hooks, evaluation)
- This gives the fairest comparison: same code, same calibration data, only the matching function changes

**Option B: Rewrite in gaem/ library**
- Implement the graph-based alignment from scratch in gaem/
- Cleaner code but harder to ensure identical evaluation protocol
- Risk of subtle differences that confound the comparison

**Recommendation**: Option A. Work in the ssl-phase1 repo on the gaem-plus branch. The comparison must be apples-to-apples.

## Experiment Plan (Revised)

### Exp A: Per-Head Procrustes (the core GAEM+ contribution)
- Same graph structure as Paper 1
- Replace permutation matching with Procrustes at each node
- Evaluate at λ = 0.9/0.1 on all 11 tasks
- **Expected**: Matches or beats Paper 1 (ID 5: avg score 957.65)

### Exp B: Ablation — Procrustes vs Permutation at Each Component
| Setting | Residual | Heads | FFN |
|---------|----------|-------|-----|
| Paper 1 (baseline) | — | S_64 perm | S_768 perm |
| Heads only | — | **O(64) Procrustes** | S_768 perm |
| All Procrustes | O(768) | **O(64) Procrustes** | **O(3072) Procrustes** |
| Procrustes + semi-perm heads | O(768) | **Sinkhorn semi-perm** | S_768 perm |

### Exp C: LoRS + TSV on Top of Best Alignment
- Take the best alignment from Exp A/B
- Apply LoRS decomposition to aligned task vectors
- Apply TSV decorrelation
- **This is the full GAEM+ pipeline — the paper's main result**

### Exp D: Weight Sweep at Best Configuration
- λ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
- Plot Pareto frontier: speech score vs music score

## Results We Have (Keep)

### Exp 0: Interference Analysis (Valid, keep)
- STI layerwise: attention Q/K highest interference (7.8), FFN lowest (2.8)
- Procrustes feature alignment: 35-62% improvement per layer
- Weight cosine similarity: 0.02 (models in different frames)

### Downstream Results (Useful as negative result / baseline)

| Method | λ | GenreID | SingerID | TechniqueID | ASR |
|--------|---|---------|----------|-------------|-----|
| HuBERT (Paper 1) | 1:0 | 63.45 | 70.25 | 61.32 | 7.84 |
| Corr-Perm (Paper 1) | 0.9:0.1 | 69.62 | 72.74 | 63.98 | 8.88 |
| No-align (Paper 1) | 0.9:0.1 | 69.56 | 71.33 | 61.75 | 9.47 |
| Per-layer Procrustes | 0.9:0.1 | 64.14 | 69.56 | 61.91 | *running* |
| Global Procrustes | 0.9:0.1 | 63.45 | 67.17 | 60.78 | *running* |
| Per-layer Procrustes | 0.7:0.3 | 53.79 | 63.54 | 56.68 | *running* |
| Global Procrustes | 0.7:0.3 | 51.38 | 64.84 | 56.33 | *running* |
| Simple avg | 0.5:0.5 | 44.48 | 47.48 | 40.64 | *running* |

These are useful to show in the paper as: "naive Procrustes without respecting component structure is insufficient — the graph-based approach matters."

## What the New Agent Should Do

1. **Read** Paper 1's `model_merger.py` and `matching_functions.py` thoroughly
2. **Implement** `match_tensors_procrustes` and `match_tensors_procrustes_MHA` following the exact same interface as the permutation versions
3. **Test** on the ssl-phase1 codebase using the same calibration data and evaluation protocol
4. **Run** Exp A: per-head Procrustes at 0.9/0.1 on all 11 tasks
5. **Compare** against Paper 1's Table I results
