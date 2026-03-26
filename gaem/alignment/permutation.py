"""
Permutation alignment (Git Re-Basin / Fabian's Paper 1).

This wraps the existing permutation matching logic from s3prl's
matching_functions.py in a clean interface consistent with the
GAEM+ pipeline. The actual implementation delegates to the
ssl-phase1 codebase when available.
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, Optional, Tuple


def correlation_permutation(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Find permutation matrix P that maximizes trace(P^T @ Corr(X, Y)).

    Uses the Hungarian algorithm on the correlation matrix.

    Args:
        X: Reference features [N, d]
        Y: Features to align [N, d]
        eps: Numerical stability for correlation computation

    Returns:
        P: Permutation matrix [d, d]
    """
    assert X.shape == Y.shape, f"Shape mismatch: X={X.shape}, Y={Y.shape}"
    d = X.shape[1]

    # Compute correlation matrix
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    std_X = X_centered.std(dim=0).clamp(min=eps)
    std_Y = Y_centered.std(dim=0).clamp(min=eps)

    corr = (X_centered / std_X).T @ (Y_centered / std_Y) / X.shape[0]  # [d, d]

    # Hungarian algorithm (maximize correlation = minimize negative correlation)
    cost_matrix = -corr.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build permutation matrix
    P = torch.zeros(d, d, dtype=X.dtype, device=X.device)
    P[col_ind, row_ind] = 1.0

    return P


def permutation_align(
    state_dict: Dict[str, torch.Tensor],
    P: torch.Tensor,
    layer_name_prefix: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Apply permutation P to a model's state dict (or a subset).

    For a given layer:
    - Weight matrices: W -> P @ W @ P^T (or P @ W for output-only)
    - Bias vectors: b -> P @ b

    Args:
        state_dict: State dict to permute
        P: Permutation matrix [d, d]
        layer_name_prefix: If set, only permute params matching this prefix

    Returns:
        Permuted state dict
    """
    aligned = {}
    for name, param in state_dict.items():
        if layer_name_prefix and not name.startswith(layer_name_prefix):
            aligned[name] = param.clone()
            continue

        if param.dim() == 2:
            # Apply permutation to both sides of weight matrix
            aligned[name] = P @ param @ P.T
        elif param.dim() == 1 and "bias" in name:
            aligned[name] = P @ param
        else:
            aligned[name] = param.clone()

    return aligned


def compute_permutation_cost(
    X: torch.Tensor, Y: torch.Tensor, P: torch.Tensor
) -> float:
    """Compute alignment cost: ||X - Y @ P||_F / ||X||_F."""
    residual = X - Y @ P
    return (torch.norm(residual) / torch.norm(X)).item()
