"""
Per-layer Procrustes alignment for transformer encoders.

Instead of a single global O matrix, computes a separate orthogonal
transform O_l for each transformer layer using layer-specific features.

Key insight: different layers have different alignment needs. Middle
layers (3-10) showed 55-62% improvement in Exp 0, vs 35.7% for the
last layer alone.

The challenge is residual connections: if layer l's output is transformed
by O_l, the residual stream must be compatible. We handle this by
transforming at the residual stream level — the same O_l applies to
the attention output (before residual add) and the FFN output.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from .procrustes import procrustes_orthogonal


def compute_per_layer_alignment(
    features_a: Dict[int, torch.Tensor],
    features_b: Dict[int, torch.Tensor],
    num_layers: int = 12,
) -> Dict[int, torch.Tensor]:
    """
    Compute per-layer Procrustes alignment matrices.

    Args:
        features_a: Dict of layer_idx -> [N, d] features from anchor model
        features_b: Dict of layer_idx -> [N, d] features from model to align
        num_layers: Number of transformer layers (default 12)

    Returns:
        Dict of layer_idx -> O_l orthogonal matrix [d, d]
    """
    alignments = {}

    for layer_idx in range(num_layers + 1):  # 0 = post-CNN, 1-12 = transformer
        if layer_idx not in features_a or layer_idx not in features_b:
            continue

        fa = features_a[layer_idx]
        fb = features_b[layer_idx]

        N = min(fa.shape[0], fb.shape[0])
        fa = fa[:N]
        fb = fb[:N]

        O_l = procrustes_orthogonal(fa, fb)
        alignments[layer_idx] = O_l

    return alignments


def align_state_dict_per_layer(
    anchor_sd: Dict[str, torch.Tensor],
    to_align_sd: Dict[str, torch.Tensor],
    layer_alignments: Dict[int, torch.Tensor],
    encoder_prefix: str = "encoder.layers",
    num_heads: int = 12,
) -> Dict[str, torch.Tensor]:
    """
    Apply per-layer orthogonal alignment to a model state dict.

    For each transformer layer l:
    - The residual stream is transformed by O_l
    - This means:
      - Attention Q/K/V inputs come from the residual stream transformed by O_{l-1}
        (the previous layer's output alignment)
      - Attention output goes through out_proj then adds to residual,
        so out_proj output must be in O_l's frame
      - FFN similarly: fc1 input is in O_{l}'s frame (post-attn-norm),
        fc2 output adds to residual, must be in O_l's frame

    Simplified approach (works when all layers have same alignment):
    For each layer l, we use O_l to transform the entire layer's weights,
    mapping from the anchor's coordinate frame to the aligned frame.

    The key transforms for layer l with O_prev (incoming) and O_curr (outgoing):
    - LayerNorm: invariant (unchanged)
    - Q_proj: W_q -> W_q @ O_prev (input from residual in prev frame)
    - K_proj: W_k -> W_k @ O_prev
    - V_proj: W_v -> W_v @ O_prev
    - out_proj: W_o -> O_curr^T @ W_o (output to residual in curr frame)
    - fc1: W_1 -> W_1 @ O_curr (input from post-attn residual)
    - fc2: W_2 -> O_curr^T @ W_2 (output back to residual)

    Note: since the residual adds pre-transform + post-transform,
    we need O_prev == O_curr for consistency. We approximate this by
    using the average of adjacent layers' alignments at residual points.

    Args:
        anchor_sd: Reference state dict (HuBERT)
        to_align_sd: State dict to align (MERT)
        layer_alignments: Dict of layer_idx -> O_l [d, d]
        encoder_prefix: Prefix for encoder layers in state dict
        num_heads: Number of attention heads

    Returns:
        Aligned state dict
    """
    d = list(layer_alignments.values())[0].shape[0]  # hidden dim (768)
    aligned = {}

    for name, param in to_align_sd.items():
        # Parse layer index from parameter name
        layer_idx = _parse_layer_idx(name, encoder_prefix)

        if layer_idx is None:
            # Non-encoder params (CNN, projection, etc.) — copy unchanged
            # Could optionally align feature_projection with O_0
            if "feature_projection" in name and param.dim() == 2:
                O_0 = layer_alignments.get(0, torch.eye(d))
                if param.shape[0] == d and param.shape[1] == d:
                    aligned[name] = O_0.T @ param @ O_0
                elif param.shape[1] == d:
                    aligned[name] = param @ O_0
                elif param.shape[0] == d:
                    aligned[name] = O_0.T @ param
                else:
                    aligned[name] = param.clone()
            else:
                aligned[name] = param.clone()
            continue

        # Get alignment for this layer
        O_l = layer_alignments.get(layer_idx, torch.eye(d))
        # For residual consistency, use O_l for both input and output of layer l
        # This is an approximation — true consistency would require
        # O_{l-1} for inputs and O_l for outputs

        name_lower = name.lower()

        # Skip norm parameters (LayerNorm is O-invariant)
        if "layer_norm" in name_lower or "layernorm" in name_lower:
            aligned[name] = param.clone()
            continue

        # Attention projections
        if any(k in name_lower for k in ["q_proj", "k_proj", "v_proj", "query", "key", "value"]):
            if param.dim() == 2 and param.shape[1] == d:
                # W_qkv [d_out, d_in] where d_in = d (residual dim)
                aligned[name] = param @ O_l
            elif param.dim() == 1 and param.shape[0] == d:
                aligned[name] = param.clone()  # bias — leave unchanged for now
            else:
                aligned[name] = param.clone()
            continue

        if any(k in name_lower for k in ["out_proj", "o_proj"]):
            if param.dim() == 2 and param.shape[0] == d:
                # W_o [d, d_heads] — output to residual
                aligned[name] = O_l.T @ param
            elif param.dim() == 1 and param.shape[0] == d:
                aligned[name] = O_l.T @ param
            else:
                aligned[name] = param.clone()
            continue

        # FFN
        if any(k in name_lower for k in ["fc1", "intermediate_dense", "up_proj", "gate_proj"]):
            if param.dim() == 2 and param.shape[1] == d:
                # fc1: [intermediate, d] — input from residual
                aligned[name] = param @ O_l
            elif param.dim() == 1:
                aligned[name] = param.clone()
            else:
                aligned[name] = param.clone()
            continue

        if any(k in name_lower for k in ["fc2", "output_dense", "down_proj"]):
            if param.dim() == 2 and param.shape[0] == d:
                # fc2: [d, intermediate] — output to residual
                aligned[name] = O_l.T @ param
            elif param.dim() == 1 and param.shape[0] == d:
                aligned[name] = O_l.T @ param
            else:
                aligned[name] = param.clone()
            continue

        # Default: clone
        aligned[name] = param.clone()

    return aligned


def _parse_layer_idx(name: str, prefix: str) -> Optional[int]:
    """Extract transformer layer index from parameter name."""
    import re
    pattern = re.escape(prefix) + r"\.(\d+)\."
    match = re.search(pattern, name)
    if match:
        return int(match.group(1))
    return None
