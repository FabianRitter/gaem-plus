# Technical Supplement: Generalized Audio Encoder Merging (GAEM)

## 1. Core Insight: Why Orthogonal Transformations Matter for Audio

Your current correlation-permutation approach finds a permutation matrix P that maximizes:
```
max_P  trace(P^T * Corr(features_A, features_B))
```

The GLMC paper shows this is limiting because:
1. **Transformers have richer symmetries**: RMSNorm layers are invariant under orthogonal transformations, not just permutations
2. **Attention heads have soft alignment**: Semi-permutation allows weighted mixing, not just reordering
3. **Different widths can be aligned**: Orthogonal transformations O ∈ R^{M×N} with M ≥ N can map between different dimensions

## 2. Proposed Algorithm: GAEM (Generalized Audio Encoder Merging)

### 2.1 Symmetry Classes for Audio Transformers

For a typical audio encoder (HuBERT, MERT, BEATs), the symmetry structure is:

| Component | Symmetry Type | Transformation |
|-----------|--------------|----------------|
| Residual Stream (with RMSNorm) | Orthogonal O(d) | Global O matrix |
| FFN hidden neurons | Permutation S_n | Per-layer P_FF |
| Attention heads | Semi-permutation | Per-layer P̃_H |
| QK/OV circuits | Invertible GL(d_h) | Per-head (can be canonical) |

### 2.2 Algorithm Pseudocode

```python
def GAEM_align(encoder_A, encoder_B, data_loader):
    """
    Align encoder_B to encoder_A using generalized symmetries.
    Returns: aligned_encoder_B
    """
    
    # Step 1: Compute global orthogonal alignment for residual stream
    # (This exploits RMSNorm invariance)
    features_A = extract_residual_features(encoder_A, data_loader)
    features_B = extract_residual_features(encoder_B, data_loader)
    O_global = procrustes_orthogonal(features_A, features_B)
    
    # Step 2: Per-layer FFN permutation alignment (your existing approach)
    P_FF = []
    for layer in range(num_layers):
        ff_A = extract_ffn_activations(encoder_A, layer, data_loader)
        ff_B = extract_ffn_activations(encoder_B, layer, data_loader)
        P_FF.append(correlation_permutation(ff_A, ff_B))
    
    # Step 3: Per-layer semi-permutation for attention heads
    P_H = []
    for layer in range(num_layers):
        head_A = extract_head_outputs(encoder_A, layer, data_loader)
        head_B = extract_head_outputs(encoder_B, layer, data_loader)
        P_H.append(semi_permutation_alignment(head_A, head_B))
    
    # Step 4: Apply transformations to encoder_B
    aligned_B = apply_transformations(encoder_B, O_global, P_FF, P_H)
    
    return aligned_B

def procrustes_orthogonal(X, Y):
    """
    Find orthogonal O minimizing ||X - Y @ O||_F
    Closed-form solution via SVD
    """
    U, _, Vt = svd(Y.T @ X)
    O = U @ Vt
    return O

def semi_permutation_alignment(heads_A, heads_B, softness=0.1):
    """
    Find semi-permutation matrix for soft head alignment.
    Allows weighted combinations, not just hard assignments.
    """
    # Compute head-wise similarity matrix
    S = compute_head_similarity(heads_A, heads_B)
    
    # Soft assignment using Sinkhorn iteration
    P_soft = sinkhorn(S, temperature=softness)
    
    return P_soft
```

### 2.3 Merging After Alignment

```python
def merge_encoders(encoders, weights, alignment_method='orthogonal'):
    """
    Merge multiple aligned encoders.
    
    Args:
        encoders: List of [encoder_1, encoder_2, ..., encoder_K]
        weights: Interpolation weights [w_1, w_2, ..., w_K] summing to 1
        alignment_method: 'permutation', 'orthogonal', or 'semi_permutation'
    """
    # Choose anchor (e.g., HuBERT for speech-primary, MERT for music-primary)
    anchor = encoders[0]
    
    # Align all others to anchor
    aligned = [anchor]
    for enc in encoders[1:]:
        if alignment_method == 'orthogonal':
            aligned.append(GAEM_align(anchor, enc))
        else:
            aligned.append(correlation_permutation_align(anchor, enc))
    
    # Interpolate weights
    merged_weights = {}
    for param_name in anchor.state_dict():
        merged_weights[param_name] = sum(
            w * aligned[i].state_dict()[param_name]
            for i, w in enumerate(weights)
        )
    
    merged_encoder = copy.deepcopy(anchor)
    merged_encoder.load_state_dict(merged_weights)
    
    return merged_encoder
```

## 3. Multi-Encoder Merging Strategy

### 3.1 The Problem

Merging 3+ encoders (e.g., HuBERT + MERT + BEATs) is non-trivial because:
1. Pairwise alignment doesn't guarantee global consistency
2. Different domain encoders may have conflicting representations
3. Weight allocation across domains affects task performance

### 3.2 Proposed Approach: Anchor-Based Multi-Alignment

```python
def multi_encoder_merge(encoder_dict, anchor_name='hubert'):
    """
    encoder_dict: {'hubert': model_h, 'mert': model_m, 'beats': model_b}
    
    Strategy:
    1. Use anchor encoder as reference
    2. Align all others to anchor
    3. Merge with task-specific or learned weights
    """
    anchor = encoder_dict[anchor_name]
    
    aligned_encoders = {'anchor': anchor}
    for name, enc in encoder_dict.items():
        if name != anchor_name:
            aligned_encoders[name] = GAEM_align(anchor, enc)
    
    # Option 1: Uniform weights
    uniform_weights = {k: 1/len(encoder_dict) for k in encoder_dict}
    
    # Option 2: Task-specific weights (learned or heuristic)
    # e.g., for ASR: heavier on HuBERT
    # for Music tagging: heavier on MERT
    
    # Option 3: Learned weights via small probe
    learned_weights = learn_merge_weights(aligned_encoders, validation_data)
    
    return merge_with_weights(aligned_encoders, learned_weights)
```

### 3.3 Heterogeneous Width Merging

This is the exciting novel direction from GLMC:

```python
def heterogeneous_merge(encoder_small, encoder_large, data_loader):
    """
    Merge encoders of different widths using orthogonal transformations.
    
    Example: HuBERT-base (768 dim) + MERT (1024 dim)
    
    Key insight: Orthogonal transformations O ∈ R^{M×N} with M ≥ N
    can project from smaller to larger dimension while preserving
    functional equivalence (up to the larger model's extra capacity).
    """
    d_small = encoder_small.config.hidden_size  # e.g., 768
    d_large = encoder_large.config.hidden_size  # e.g., 1024
    
    # Extract features
    feat_small = extract_features(encoder_small, data_loader)  # N x d_small
    feat_large = extract_features(encoder_large, data_loader)  # N x d_large
    
    # Find orthogonal projection O: R^d_small -> R^d_large
    # This embeds the smaller space into the larger one
    O = extended_procrustes(feat_small, feat_large)  # d_large x d_small
    
    # Apply O to small encoder's weights to "expand" it
    expanded_small = expand_encoder(encoder_small, O)
    
    # Now both are in d_large space - can merge normally
    merged = interpolate(expanded_small, encoder_large, alpha=0.5)
    
    return merged
```

## 4. Audio LLM Integration

### 4.1 SALMONN Architecture Analysis

SALMONN currently uses:
- Whisper encoder (speech)
- BEATs encoder (audio events)
- Q-Former bridge
- Vicuna LLM

**Proposed experiment**: Replace dual encoders with merged single encoder

```python
class ModifiedSALMONN(nn.Module):
    def __init__(self, merged_encoder, llm_backbone):
        super().__init__()
        # Replace: self.whisper + self.beats
        # With: self.merged_encoder (HuBERT + MERT + BEATs merged)
        self.audio_encoder = merged_encoder
        self.qformer = QFormer(...)
        self.llm = llm_backbone  # Frozen
    
    def forward(self, audio, text_prompt):
        # Single encoder path (vs. dual in original)
        audio_features = self.audio_encoder(audio)
        bridged = self.qformer(audio_features)
        output = self.llm(text_prompt, bridged)
        return output
```

### 4.2 Evaluation Protocol

```python
benchmarks = {
    'speech': ['librispeech_asr', 'common_voice', 'superb_sid'],
    'music': ['musiccaps', 'gtzan_genre', 'nsynth_pitch'],
    'audio_events': ['audioset', 'esc50', 'urbansound8k'],
    'audio_llm': ['dynamic_superb', 'air_bench_chat']
}

metrics = {
    'interpolation_barrier': measure_loss_along_path,
    'task_performance': downstream_eval,
    'efficiency': {
        'params': count_params,
        'inference_time': benchmark_speed,
        'memory': profile_memory
    }
}
```

## 5. Key Experiments to Run

### Experiment 1: Permutation vs Orthogonal Alignment
- **Setup**: HuBERT-base + MERT-95M (same width: 768)
- **Metric**: Interpolation barrier, downstream accuracy
- **Hypothesis**: Orthogonal achieves lower barrier

### Experiment 2: Multi-Encoder (3 models)
- **Setup**: HuBERT + MERT + BEATs
- **Variations**: Different anchor choices, different weight schemes
- **Metric**: Performance across all three domains

### Experiment 3: Heterogeneous Width
- **Setup**: HuBERT-base (768) + MERT-330M (1024)
- **Metric**: Can we successfully merge different widths?
- **Hypothesis**: GLMC orthogonal projection enables this

### Experiment 4: Audio LLM Integration
- **Setup**: Merged encoder → SALMONN pipeline
- **Baseline**: Original dual-encoder SALMONN
- **Metric**: Dynamic-SUPERB scores, efficiency gains

## 6. Potential Paper Structure

1. **Introduction**: Motivation for unified audio understanding, limitation of permutation-only methods
2. **Related Work**: Git Re-Basin, Task Arithmetic, GLMC, Audio SSL models, Audio LLMs
3. **Method**: GAEM framework with orthogonal + semi-permutation alignment
4. **Experiments**:
   - 4.1 Orthogonal vs Permutation (ablation)
   - 4.2 Multi-Encoder Merging (3+ models)
   - 4.3 Heterogeneous Width Merging
   - 4.4 Audio LLM Integration
5. **Analysis**: Interpolation barriers, domain interference, efficiency
6. **Conclusion**: Unified audio encoder for all domains

## 7. Potential Extensions (Future Work / if time permits)

1. **Continuous symmetry optimization**: Instead of discrete permutation, optimize soft semi-permutation end-to-end
2. **Domain-specific gating**: Learn when to use which encoder's representation
3. **Cross-modal merging**: Extend to audio-visual encoders
4. **Compression**: Can merged encoder be pruned more than individual ones?

---

## Quick Start Implementation Plan

**Week 1-2**:
1. Fork GLMC codebase (if available) or implement Procrustes solver
2. Implement orthogonal alignment for your HuBERT+MERT setup
3. Run initial comparison: correlation-permutation vs orthogonal

**Week 3-4**:
1. Implement semi-permutation for attention heads
2. Measure interpolation barriers for different alignment strategies
3. Run downstream evaluations on speech + music benchmarks

**Week 5-6**:
1. Add BEATs encoder to create 3-way merge
2. Experiment with different anchor choices and weight schemes

**Week 7-8**:
1. Heterogeneous width experiments (if promising)
2. Begin Audio LLM integration experiments

**Week 9-12**:
1. Complete Audio LLM evaluation
2. Ablation studies and analysis
3. Paper writing

---

Good luck with your thesis, Fabian! This is an exciting direction that bridges your existing work with cutting-edge theory from the GLMC paper.
