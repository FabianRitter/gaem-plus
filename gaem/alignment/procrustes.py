"""
Orthogonal alignment via Procrustes solution.

Based on GLMC (Theus 2025): Generalized Linear Mode Connectivity.
Exploits the fact that RMSNorm/LayerNorm layers in transformers are
invariant under orthogonal transformations, not just permutations.

The Procrustes problem: Find orthogonal O minimizing ||X - Y @ O||_F
Closed-form solution via SVD: O = V @ U^T where U S V^T = SVD(Y^T X)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


def procrustes_orthogonal(
    X: torch.Tensor,
    Y: torch.Tensor,
    allow_reflection: bool = True,
) -> torch.Tensor:
    """
    Find orthogonal matrix O that minimizes ||X - Y @ O||_F.

    This is the core alignment operation for same-width encoders.

    Args:
        X: Reference features [N, d] (from anchor encoder)
        Y: Features to align [N, d] (from encoder to be aligned)
        allow_reflection: If False, constrain to SO(d) (proper rotations only).
                         If True, allow O(d) (rotations + reflections).

    Returns:
        O: Orthogonal matrix [d, d] such that Y @ O ≈ X
    """
    assert X.shape == Y.shape, f"Shape mismatch: X={X.shape}, Y={Y.shape}"

    # Compute cross-covariance matrix
    M = Y.T @ X  # [d, d]

    # SVD
    U, S, Vt = torch.linalg.svd(M)

    if not allow_reflection:
        # Ensure det(O) = +1 (proper rotation)
        # Flip the sign of the last column of U if det is negative
        det = torch.det(U @ Vt)
        if det < 0:
            U[:, -1] *= -1

    O = U @ Vt  # [d, d]

    return O


def extended_procrustes(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:
    """
    Find orthogonal projection O: R^d_small -> R^d_large for merging
    encoders of different hidden dimensions.

    Based on GLMC Section 4.3: heterogeneous width alignment.
    Uses padding + standard Procrustes.

    Args:
        X: Features from smaller encoder [N, d_small]
        Y: Features from larger encoder [N, d_large]

    Returns:
        O: Projection matrix [d_small, d_large] mapping small -> large space
    """
    d_small = X.shape[1]
    d_large = Y.shape[1]

    if d_small == d_large:
        return procrustes_orthogonal(Y, X)  # Standard case

    assert d_small < d_large, (
        f"X ({d_small}) should be smaller than Y ({d_large}). "
        "Swap inputs or use procrustes_orthogonal for same-width."
    )

    # Pad the smaller features with zeros to match larger dimension
    X_padded = F.pad(X, (0, d_large - d_small))  # [N, d_large]

    # Standard Procrustes on padded features
    O_full = procrustes_orthogonal(Y, X_padded)  # [d_large, d_large]

    # Extract the d_small x d_large submatrix
    O = O_full[:d_small, :]  # [d_small, d_large]

    return O


def align_state_dict_orthogonal(
    state_dict_anchor: Dict[str, torch.Tensor],
    state_dict_to_align: Dict[str, torch.Tensor],
    O: torch.Tensor,
    layer_types: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Apply orthogonal transformation O to a model's state dict.

    For a transformer encoder, the transformation rules are:
    - Linear weights W -> O^T @ W @ O  (or O^T @ W for output projections)
    - LayerNorm weights -> unchanged (invariant under orthogonal transforms)
    - Bias terms b -> O^T @ b

    Args:
        state_dict_anchor: Reference model state dict (for structure)
        state_dict_to_align: State dict to transform
        O: Orthogonal matrix [d, d]
        layer_types: Optional dict mapping param names to types
                    ('linear_in', 'linear_out', 'norm', 'bias', 'skip')

    Returns:
        Transformed state dict
    """
    aligned = {}

    for name, param in state_dict_to_align.items():
        if layer_types and name in layer_types:
            ltype = layer_types[name]
        else:
            # Infer type from parameter name and shape
            ltype = _infer_layer_type(name, param)

        d = O.shape[0]  # alignment dimension (e.g., 768)

        # Skip params whose dimensions don't match O (e.g., CNN layers with dim 512)
        def _dim_compatible(p, d):
            if p.dim() == 2:
                return p.shape[0] == d or p.shape[1] == d
            elif p.dim() == 1:
                return p.shape[0] == d
            return False

        if ltype == "skip" or param.dim() == 0 or not _dim_compatible(param, d):
            aligned[name] = param.clone()
        elif ltype == "norm":
            aligned[name] = param.clone()
        elif ltype == "linear_in" and param.dim() == 2:
            # Input-side linear: W -> W @ O (requires W.shape[1] == d)
            if param.shape[1] == d:
                aligned[name] = param @ O
            else:
                aligned[name] = param.clone()
        elif ltype == "linear_out" and param.dim() == 2:
            # Output-side linear: W -> O^T @ W (requires W.shape[0] == d)
            if param.shape[0] == d:
                aligned[name] = O.T @ param
            else:
                aligned[name] = param.clone()
        elif ltype == "linear_both" and param.dim() == 2:
            # Both sides: W -> O^T @ W @ O (requires both dims == d)
            if param.shape[0] == d and param.shape[1] == d:
                aligned[name] = O.T @ param @ O
            elif param.shape[0] == d:
                aligned[name] = O.T @ param
            elif param.shape[1] == d:
                aligned[name] = param @ O
            else:
                aligned[name] = param.clone()
        elif ltype == "bias" and param.dim() == 1:
            if param.shape[0] == d:
                aligned[name] = O.T @ param
            else:
                aligned[name] = param.clone()
        elif param.dim() == 1:
            if "bias" in name and param.shape[0] == d:
                aligned[name] = O.T @ param
            else:
                aligned[name] = param.clone()
        else:
            aligned[name] = param.clone()

    return aligned


def _infer_layer_type(name: str, param: torch.Tensor) -> str:
    """Heuristic layer type inference from parameter name."""
    name_lower = name.lower()

    if any(k in name_lower for k in ["layernorm", "layer_norm", "rmsnorm", "norm"]):
        if "weight" in name_lower or "gamma" in name_lower:
            return "norm"
        if "bias" in name_lower or "beta" in name_lower:
            return "norm"

    if "bias" in name_lower and param.dim() == 1:
        return "bias"

    if param.dim() == 2:
        # For attention: q, k, v projections are "linear_in"
        # Output projection is "linear_out"
        if any(k in name_lower for k in ["q_proj", "k_proj", "v_proj", "query", "key", "value"]):
            return "linear_in"
        if any(k in name_lower for k in ["out_proj", "o_proj", "output"]):
            return "linear_out"
        # FFN: fc1/up is "linear_in", fc2/down is "linear_out"
        if any(k in name_lower for k in ["fc1", "up_proj", "gate_proj", "intermediate"]):
            return "linear_in"
        if any(k in name_lower for k in ["fc2", "down_proj", "output"]):
            return "linear_out"
        # Default for 2D: both sides
        return "linear_both"

    return "skip"


def compute_alignment_error(
    X: torch.Tensor, Y: torch.Tensor, O: torch.Tensor
) -> float:
    """Compute ||X - Y @ O||_F / ||X||_F (relative Frobenius error)."""
    residual = X - Y @ O
    return (torch.norm(residual) / torch.norm(X)).item()
