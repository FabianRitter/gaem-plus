# Agent Handover: GAEM+ Implementation

**From**: Session 1 agent (2026-03-26 to 2026-03-29)
**To**: Next agent(s)
**Date**: 2026-03-29

---

## TL;DR — What You Need to Know

We're merging HuBERT (speech) and MERT (music) audio encoders using orthogonal alignment. The first attempt (global/per-layer Procrustes) failed because it doesn't respect transformer component structure. **The next step is to implement GLMC-style per-component alignment** — per-head O(64) Procrustes for attention, with the graph-based framework from Paper 1.

---

## 1. The Research Goal

**GAEM+** merges HuBERT + MERT into a single encoder that handles both speech and music. The paper contribution combines:
1. **Orthogonal alignment** (richer than permutation) — from GLMC
2. **LoRS decomposition** (low-rank + sparse task vectors) — from LoRS-Merging
3. **TSV decorrelation** (reduce interference via SVD) — from TSV paper

The baseline to beat is **Paper 1** (Fabian's published work): per-head permutation alignment achieving avg score 957.65/1000.

## 2. What's Been Done

### Completed and Working
- `gaem/` library with alignment, decomposition, merging, evaluation modules (all tested)
- Exp 0: interference analysis (STI layerwise, feature alignment comparison)
- Calibration dataset: 10k samples (5k music4all + 5k LibriSpeech)
- Paper 1 baselines extracted (all 9 methods × 11 tasks in `results/baselines.json`)
- Infrastructure: PBS scripts, enroot configs, HF model conversion, parallel job submission

### Completed but Wrong Approach
- Global Procrustes merging (single O for entire model) → ~5 pp below Paper 1
- Per-layer Procrustes (O_l per layer) → still ~3-5 pp below Paper 1
- Root cause: O(768) mixes attention heads. Need O(64) per head.

### Running (will finish in ~2-4 hours)
- 5 ASR evaluation jobs (global Procrustes 0.9/0.1, 0.7/0.3, per-layer 0.9/0.1, 0.7/0.3, simple avg)
- Results will appear in `results/downstream_results/*/asr/log.log`

### Not Yet Done (The Actual Contribution)
- **Per-head Procrustes alignment** following GLMC symmetry structure
- **LoRS decomposition** on aligned task vectors
- **TSV decorrelation**
- **Full GAEM+ pipeline evaluation**

## 3. HuBERT/MERT Architecture (Critical for Implementation)

Both models share identical architecture (HuBERT Base):

```
Input waveform (16kHz)
  │
  ├── Feature Extractor (7 CNN layers)
  │     Conv1D: [1→512], then 6× [512→512]
  │     GroupNorm on first layer only
  │     Symmetry: permutation S_512 per CNN layer
  │
  ├── Feature Projection: Linear [512→768] + LayerNorm
  │     Symmetry: maps CNN space → residual stream space
  │
  ├── Positional Conv Encoding
  │
  └── 12 Transformer Layers (pre-LN)
        Each layer:
        │
        ├── self_attn_layer_norm (LayerNorm)  → invariant under O(768)
        │
        ├── Self-Attention (12 heads, d_head=64)
        │   ├── q_proj: [768→768] (reshaped as [12×64, 768])
        │   ├── k_proj: [768→768] (reshaped as [12×64, 768])
        │   ├── v_proj: [768→768] (reshaped as [12×64, 768])
        │   ├── Attention: softmax(Q_h K_h^T / √64) V_h  per head
        │   └── out_proj: [768→768] (reshaped as [768, 12×64])
        │   Symmetry: O(64) per head on Q/K/V, S_12 for head ordering
        │
        ├── Residual add + final_layer_norm (LayerNorm)
        │
        ├── FFN
        │   ├── fc1: [768→3072] + GELU
        │   └── fc2: [3072→768]
        │   Symmetry: S_3072 on intermediate neurons (GELU breaks O but preserves S)
        │
        └── Residual add
```

### State Dict Key Patterns
```
feature_extractor.conv_layers.{0-6}.conv.weight
feature_extractor.conv_layers.0.layer_norm.{weight,bias}
feature_projection.projection.{weight,bias}
feature_projection.layer_norm.{weight,bias}
encoder.pos_conv.0.{weight,bias}
encoder.layers.{0-11}.self_attn.{q,k,v}_proj.{weight,bias}
encoder.layers.{0-11}.self_attn.out_proj.{weight,bias}
encoder.layers.{0-11}.self_attn_layer_norm.{weight,bias}
encoder.layers.{0-11}.fc1.{weight,bias}
encoder.layers.{0-11}.fc2.{weight,bias}
encoder.layers.{0-11}.final_layer_norm.{weight,bias}
```

**MERT uses identical keys** — no prefix difference when loaded via HuggingFace `AutoModel.from_pretrained("m-a-p/MERT-v0-public")`.

## 4. The Correct GLMC-Style Implementation

### Per-Component Alignment Transforms

For each transformer layer l, compute and apply these transforms:

| Parameter | Shape | Input Transform | Output Transform | How to Compute |
|-----------|-------|----------------|-----------------|----------------|
| `self_attn_layer_norm.weight` | [768] | — | — | Invariant (don't touch) |
| `self_attn_layer_norm.bias` | [768] | — | — | Invariant (don't touch) |
| `q_proj.weight` | [768,768] | Residual O_res on cols | Per-head O_h on rows (reshaped) | O_h = Procrustes(Q_h_hubert, Q_h_mert) per head |
| `q_proj.bias` | [768] | — | Per-head O_h (reshaped) | Same O_h |
| `k_proj.weight` | [768,768] | Residual O_res on cols | Per-head O_h on rows (reshaped) | Same O_h as Q (heads must match) |
| `k_proj.bias` | [768] | — | Per-head O_h (reshaped) | Same O_h |
| `v_proj.weight` | [768,768] | Residual O_res on cols | Per-head O_v_h on rows (reshaped) | O_v_h from V head activations |
| `v_proj.bias` | [768] | — | Per-head O_v_h (reshaped) | Same O_v_h |
| `out_proj.weight` | [768,768] | Per-head O_v_h on cols (reshaped) | Residual O_res on rows | Must invert the V transform |
| `out_proj.bias` | [768] | — | Residual O_res | — |
| `final_layer_norm.weight` | [768] | — | — | Invariant |
| `final_layer_norm.bias` | [768] | — | — | Invariant |
| `fc1.weight` | [3072,768] | Residual O_res on cols | P_ffn on rows | P_ffn = Hungarian on FFN activations |
| `fc1.bias` | [3072] | — | P_ffn | — |
| `fc2.weight` | [768,3072] | P_ffn^T on cols | Residual O_res on rows | — |
| `fc2.bias` | [768] | — | Residual O_res | — |

### Key Constraint: Residual Consistency

The residual connection means: `output = residual + attn_output` and `output = residual + ffn_output`. If the residual stream is in frame O_res, then both `attn_output` and `ffn_output` must also be in frame O_res. This is automatically satisfied if:
- `out_proj` maps from head space back to O_res frame
- `fc2` maps from FFN intermediate back to O_res frame

### Implementation Approach (Option A — Recommended)

**Extend Paper 1's codebase** (at `ssl-phase1/s3prl/`):

1. In `matching_functions.py`, add:
   ```python
   def match_tensors_procrustes(correlation_matrix=None, features_A=None, features_B=None, **kwargs):
       """Replace Hungarian with Procrustes solver."""
       O = procrustes_orthogonal(features_A, features_B)
       # Return merge/unmerge in same format as match_tensors_permute
       ...

   def match_tensors_procrustes_MHA(n_heads, features_A=None, features_B=None, **kwargs):
       """Per-head Procrustes instead of per-head permutation."""
       for h in range(n_heads):
           O_h = procrustes_orthogonal(features_A_head_h, features_B_head_h)
       ...
   ```

2. In `model_merger.py`, the graph traversal and transform application logic stays the same — only the matching function changes.

3. In `graphs/hubert_graph.py`, the graph structure stays the same — PREFIX nodes mark where alignment is applied.

## 5. Paper 1's Code (What to Reuse)

### Key Files in `ssl-phase1/s3prl/`

| File | What It Does | Reuse? |
|------|-------------|--------|
| `matching_functions.py` | All matching algorithms (permutation, symmetric, MHA, ZipIt, Sinkhorn) | **Extend** — add Procrustes variants |
| `merging_utils/model_merger.py` | Graph traversal, activation collection, transform application | **Reuse directly** |
| `merging_utils/model_merger_new.py` | Newer version of merger | **Study** — may have improvements |
| `graphs/hubert_graph.py` | HuBERT DAG with PREFIX/POSTFIX nodes | **Reuse directly** |
| `graphs/base_graph.py` | Base graph class with hooks, node types | **Reuse directly** |
| `metric_calculators.py` | Covariance/correlation computation on activations | **Reuse directly** |
| `run_downstream.py` | Downstream evaluation entry point | **Reuse directly** |

### How Paper 1's Merger Works (Flow)

1. `ModelMerge.__init__()`: Creates graphs for both models, adds activation hooks
2. `ModelMerge.compute_metrics()`: Forward pass on calibration data, collects covariance matrices at each PREFIX/POSTFIX node
3. `ModelMerge.compute_transformations()`: At each node, calls the matching function (e.g., `match_tensors_permute_MHA`) on the correlation matrix to get merge/unmerge matrices
4. `ModelMerge.apply_transformations_custom()`: Walks the graph, applies merge/unmerge to weight matrices
5. `ModelMerge.merge_models()`: Weighted interpolation of aligned weights

### Important: The Matching Function Interface

```python
def match_tensors_permute_MHA(
    n_heads,              # 12
    permute_heads=False,  # whether to reorder heads
    head_assignments=[],  # pre-specified head ordering
    r=0.5,               # reduction ratio (0.5 for 2 models)
    correlation_matrix=None,  # [2*Om, 2*Om] cross-correlation
    **kwargs
):
    # Returns: merge, unmerge, extra, cost
```

The Procrustes version needs the same interface but may also need raw features (not just correlation matrix) since Procrustes operates on features directly, not correlations.

## 6. Calibration Data

- **Location**: `data/calibration_10k.csv` (5000 music4all + 5000 LibriSpeech)
- **Music audio**: `/data/projects/12004380/datasets/music4all-all/music4all_16khz_new/` (35340 files, 16kHz)
- **Speech audio**: `/data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech/train-clean-100/`
- **This is the same calibration strategy as Paper 1** (5k+5k)

## 7. Evaluation Protocol

### Downstream Tasks (Use NSCC configs from Paper 1)
| Task | Config | total_steps | Dataset |
|------|--------|-------------|---------|
| ASR | `downstream/asr/config.yaml` | — (uses epochs) | LibriSpeech |
| genre_gtzan | `downstream/genre_gtzan/config_nscc.yaml` | 10000 | GTZAN |
| vocalset_singer_id | `downstream/vocalset_singer_id/config_nscc.yaml` | 80000 | VocalSet |
| vocalset_technique_id | `downstream/vocalset_technique_id/config_nscc.yaml` | 75000 | VocalSet |

### How to Load Merged Models
Save as HuggingFace format, then use `hf_hubert_custom` upstream:
```bash
python run_downstream.py -m train \
    -u hf_hubert_custom \
    -k /path/to/hf_model_dir \
    -d genre_gtzan \
    -c ./downstream/genre_gtzan/config_nscc.yaml \
    -s hidden_states \
    -o config.downstream_expert.datarc.file_path=/data/projects/12004380/datasets/superb/superb/GTZAN
```

Note: `hf_hubert/expert.py` was patched to add `self.sample_rate = SAMPLE_RATE`.

### PBS Job Submission
Use `scripts/eval_single_task.sh` with `-v method=X,task=Y,model_base=Z`:
```bash
qsub -N "job_name" -o "log_path" \
    -v "method=my_method,task=genre_gtzan,model_base=results/my_merge/hf_models" \
    scripts/eval_single_task.sh
```

## 8. The GLMC Codebase

**Repo**: https://github.com/alexandertheus/Generalized-LMC-for-Transformers

Key files to study:
- `weight_matching.py` — orthogonal alignment implementation, `ortho_residual` function
- `merger.py` — how transforms are applied to GPT-2 layers (547 lines)
- `enums.py` — `AlignmentType.PERM`, `SOFT_PERM`, `ORTHO`

**Caveat**: GLMC is implemented for GPT-2 (decoder-only). HuBERT is an encoder-only architecture with different attention structure (no causal mask, different norm placement). The alignment logic transfers but the graph structure needs adaptation.

A subagent is currently studying this codebase and mapping it to HuBERT/MERT — results will be in the agent output or a follow-up document.

## 9. What the Next Agent Should Do

### Priority 1: Per-Head Procrustes (The Core Experiment)
1. Study `matching_functions.py:match_tensors_permute_MHA()` — understand the per-head correlation extraction
2. Implement `match_tensors_procrustes_MHA()` — same interface, Procrustes instead of Hungarian per head
3. Study `model_merger.py:compute_transformations()` — understand where matching functions are called
4. Wire in the Procrustes matching function
5. Run on all 11 tasks at λ = 0.9/0.1
6. Compare against Paper 1 Table I

### Priority 2: LoRS + TSV (If Per-Head Procrustes ≥ Paper 1)
1. After alignment, compute task vectors: τ = aligned_MERT - HuBERT
2. Apply `gaem/decomposition/lors.py` to decompose τ into low-rank + sparse
3. Apply `gaem/evaluation/sti.py:tsv_merge()` to decorrelate the low-rank component
4. Merge: base + merged_lowrank + TIES(sparse)
5. Evaluate — this is the full GAEM+ pipeline

### Priority 3: Ablation Table
- Procrustes vs Permutation at each component (heads, FFN, residual)
- With/without LoRS decomposition
- With/without TSV decorrelation

## 10. Memory Notes (for persistent context)

Check `/home/users/ntu/s220064/.claude/projects/-data-projects-12004380-fabian-generalized-model-merging/memory/MEMORY.md` for:
- User profile (Fabian's background, preferences)
- Implementation approach decisions
- Existing results from Paper 1
- Attention head alignment insight
- Cluster debugging (hold_node)
