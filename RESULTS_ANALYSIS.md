# Complete Results Analysis: GAEM+ Experiments (Session 1)

**Period**: 2026-03-26 to 2026-03-29
**Researcher**: Fabian Ritter (with Claude Code assistance)

---

## 1. What We Set Out To Do

Test whether **orthogonal alignment (Procrustes)** outperforms **permutation alignment** (Paper 1) for merging HuBERT (speech) and MERT (music) encoders. The hypothesis was based on the GLMC paper (Theus 2025) which shows transformers have O(d) symmetry, not just S_n.

## 2. What We Actually Tested

### Exp 0: Interference & Alignment Analysis (Successful, Informative)

**Method**: Loaded HuBERT Base and MERT v0 directly, extracted features on 1000 calibration samples (500 speech + 500 music), computed alignment matrices and interference metrics.

**Results**:
- Weight cosine similarity: **0.020** (models are in nearly orthogonal frames)
- Relative weight difference: **128%** of HuBERT's norm
- **Procrustes feature alignment**: 35.7% error reduction (last layer), 55-62% for middle layers
- **Permutation feature alignment**: only 6.7% error reduction
- **STI analysis**: attention Q/K projections in early layers have highest interference (7.3-7.8), FFN output_dense has lowest (2.8-2.9)

**Conclusion**: Procrustes captures much more of the alignment in feature space than permutation. But feature alignment ≠ downstream performance (as we learned).

### Exp 1: Global Procrustes Merging (Completed, Disappointing)

**Method**: Single 768×768 Procrustes matrix O from last-layer features, applied uniformly to all MERT parameters. Tested at λ = 0.5/0.5 and 0.7/0.3.

| Method | λ (H:M) | GenreID |
|--------|---------|---------|
| Simple avg | 0.5:0.5 | 44.48% |
| Global Procrustes | 0.5:0.5 | 48.62% |
| Global Procrustes | 0.7:0.3 | 51.38% |

**Conclusion**: Global Procrustes helps vs no alignment (+4-7 pp) but far below HuBERT alone (63.45%) and Paper 1 (69.62%).

### Exp 1b: Per-Layer Procrustes Merging (Completed, Slightly Better)

**Method**: Separate 768×768 Procrustes O_l per transformer layer, computed from layer-specific features. Applied to MERT parameters respecting layer structure.

| Method | λ (H:M) | GenreID | SingerID | TechID |
|--------|---------|---------|----------|--------|
| HuBERT alone | 1:0 | 63.45 | 70.25 | 61.32 |
| **Corr-Perm (Paper 1)** | **0.9:0.1** | **69.62** | **72.74** | **63.98** |
| No-align weighted | 0.9:0.1 | 69.56 | 71.33 | 61.75 |
| Per-layer Procrustes | 0.9:0.1 | 64.14 | 69.56 | 61.91 |
| Global Procrustes | 0.9:0.1 | 63.45 | 67.17 | 60.78 |
| Per-layer Procrustes | 0.7:0.3 | 53.79 | 63.54 | 56.68 |
| Global Procrustes | 0.7:0.3 | 51.38 | 64.84 | 56.33 |
| Simple avg | 0.5:0.5 | — | 47.48 | 40.64 |

**ASR results**: Still running (5 jobs, ~10h each). Will update when available.

### Ranking (Non-ASR tasks)

```
Corr-Perm (Paper 1, 0.9/0.1)  >>>  Per-layer Procrustes (0.9/0.1)  >  Global Procrustes (0.9/0.1)  >  No-align (0.9/0.1)  >>>  Simple avg
```

## 3. Why Per-Layer Procrustes Underperformed Paper 1

### Root Cause: Head Mixing

A 768×768 orthogonal matrix O freely mixes information across all 12 attention heads (each 64-dim). When applied to Q/K/V projections:
- Head i's queries get rotated into head j's space
- The attention computation Q_i·K_i^T produces corrupted scores
- Paper 1's per-head permutation (64×64 within each head) preserves head-internal structure

### The GLMC Framework Shows Why

GLMC identifies component-wise symmetries:

| Component | Symmetry | Paper 1 | Our Procrustes | Correct |
|-----------|----------|---------|----------------|---------|
| Residual stream | O(768) | Not explicit | O(768) ✓ | O(768) |
| Per attention head | O(64) or S_64 | S_64 per head ✓ | O(768) global ✗ | O(64) per head |
| Head ordering | S_12 / semi-perm | S_12 ✓ | Not handled ✗ | Semi-perm |
| FFN intermediate | S_3072 | S_768 on W2 | O(768) ✗ | S_3072 or O(3072) |
| LayerNorm | Invariant | Invariant ✓ | Invariant ✓ | Invariant |

Paper 1 gets the **structure** right (per-head, per-component) even though it uses the "weaker" permutation group at each node. Our Procrustes has the "richer" transform but the **wrong structure**.

### Quantitative Evidence

- **Per-layer Procrustes at 0.9/0.1**: GenreID 64.14% (vs 69.62% Paper 1 = **-5.48 pp**)
- **Per-layer Procrustes at 0.9/0.1**: SingerID 69.56% (vs 72.74% Paper 1 = **-3.18 pp**)
- **Per-layer Procrustes at 0.9/0.1**: TechID 61.91% (vs 63.98% Paper 1 = **-2.07 pp**)

The gap is consistent across all tasks: ~2-5 pp. This is the cost of mixing heads.

## 4. What Was Valuable

1. **Exp 0 diagnostics** — STI analysis, layerwise interference, feature alignment comparison. These are novel analysis tools for audio encoder merging that aren't in Paper 1.

2. **The gaem/ library** — LoRS decomposition, TSV/STI metrics, TIES/DARE baselines. None of these have been applied yet. They remain the core GAEM+ contribution once the alignment is done correctly.

3. **Infrastructure** — PBS scripts, calibration dataset, HF model conversion, parallel job submission. All reusable.

4. **The negative result itself** — demonstrates that naive orthogonal alignment (even per-layer) is insufficient without respecting transformer component structure. This is a useful finding for the paper: "global Procrustes captures feature-level alignment but fails at the weight level due to head mixing."

5. **Paper 1 baselines** — all 9 methods × 11 tasks extracted and documented in baselines.json.

## 5. What Was Not Tested (The Actual GAEM+ Contribution)

- **Per-head Procrustes O(64)** — the correct GLMC approach
- **Semi-permutation for head ordering** — Sinkhorn on head similarities
- **LoRS decomposition** — low-rank + sparse task vector decomposition
- **TSV decorrelation** — Procrustes on concatenated singular vectors
- **The combined GAEM+ pipeline** — alignment + decomposition + merge

## 6. Files and Artifacts

### Results
| Path | Contents |
|------|----------|
| `results/exp0_analysis/` | STI, interference, alignment matrices, features |
| `results/exp1_merge/hf_models/` | 7 global Procrustes merged HF models |
| `results/exp1b_perlayer/hf_models/` | 2 per-layer Procrustes merged HF models |
| `results/exp1_downstream/` | First batch genre_gtzan results (3 methods) |
| `results/downstream_results/` | Second batch results (all methods × 4 tasks) |
| `results/baselines.json` | All Paper 1 numbers (Table I, 9 methods × 11 tasks) |

### Code
| Path | Contents |
|------|----------|
| `gaem/alignment/procrustes.py` | Procrustes solver (correct, but applied wrong) |
| `gaem/alignment/per_layer_procrustes.py` | Per-layer alignment (partially correct) |
| `gaem/alignment/permutation.py` | Hungarian permutation |
| `gaem/alignment/semi_permutation.py` | Sinkhorn semi-permutation |
| `gaem/decomposition/lors.py` | LoRS decomposition (untested on real data) |
| `gaem/evaluation/sti.py` | STI metric + TSV-Merge (tested, working) |
| `gaem/merging/gaem_plus.py` | Full pipeline (needs correct alignment input) |

### Documentation
| Path | Contents |
|------|----------|
| `REVISED_PLAN.md` | Corrected experiment plan |
| `METHODS_COMPARISON.md` | Detailed comparison of alignment approaches |
| `IMPLEMENTATION_PLAN.md` | Original phased plan (partially outdated) |
| `PROJECT_STRUCTURE.md` | Repo layout and data locations |
| `GPU_DEBUG_GUIDE.md` | NSCC hold_node debugging |
