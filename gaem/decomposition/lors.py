"""
LoRS (Low-Rank + Sparse) task vector decomposition.

Based on LoRS-Merging (Zhao Q. 2025): decomposes task vectors into
a low-rank component (via truncated SVD) capturing compact structure,
and a sparse component (via magnitude pruning) capturing scattered details.

Key insight: Merging operates on the structured (low-rank) and scattered
(sparse) components separately, allowing different strategies for each.
"""

import torch
from typing import Dict, Tuple, Optional


def lors_decompose(
    task_vector: torch.Tensor,
    rank_ratio: float = 0.1,
    sparsity: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a 2D task vector into low-rank + sparse components.

    τ ≈ τ_lowrank + τ_sparse

    where τ_lowrank = U_k @ diag(S_k) @ V_k^T (truncated SVD)
    and τ_sparse = threshold(τ - τ_lowrank) (magnitude pruning)

    Args:
        task_vector: Weight matrix difference [m, n] (finetuned - pretrained)
        rank_ratio: Fraction of singular values to keep (0.0 to 1.0)
        sparsity: Fraction of residual entries to prune (0.0 to 1.0)

    Returns:
        tau_lowrank: Low-rank approximation [m, n]
        tau_sparse: Sparse residual [m, n]
    """
    assert task_vector.dim() == 2, (
        f"LoRS decomposition requires 2D tensor, got {task_vector.dim()}D"
    )

    m, n = task_vector.shape
    k = max(1, int(min(m, n) * rank_ratio))

    # Truncated SVD
    U, S, Vt = torch.linalg.svd(task_vector, full_matrices=False)
    tau_lowrank = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]

    # Sparse residual with magnitude pruning
    residual = task_vector - tau_lowrank
    abs_residual = residual.abs()
    threshold = torch.quantile(abs_residual.flatten(), sparsity)
    tau_sparse = torch.where(abs_residual > threshold, residual, torch.zeros_like(residual))

    return tau_lowrank, tau_sparse


def lors_decompose_state_dict(
    task_vector_dict: Dict[str, torch.Tensor],
    rank_ratio: float = 0.1,
    sparsity: float = 0.9,
    min_dim: int = 64,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Apply LoRS decomposition to all 2D parameters in a task vector state dict.

    Parameters smaller than min_dim in both dimensions are kept as-is
    in the sparse component (too small for meaningful SVD).

    Args:
        task_vector_dict: Dict of parameter name -> task vector tensor
        rank_ratio: Fraction of singular values to keep
        sparsity: Fraction of residual entries to prune
        min_dim: Minimum dimension for SVD decomposition

    Returns:
        lowrank_dict: Low-rank components
        sparse_dict: Sparse components
    """
    lowrank_dict = {}
    sparse_dict = {}

    for name, tv in task_vector_dict.items():
        if tv.dim() == 2 and min(tv.shape) >= min_dim:
            lr, sp = lors_decompose(tv, rank_ratio=rank_ratio, sparsity=sparsity)
            lowrank_dict[name] = lr
            sparse_dict[name] = sp
        else:
            # 1D params (biases, norms) or small matrices: keep in sparse
            lowrank_dict[name] = torch.zeros_like(tv)
            sparse_dict[name] = tv.clone()

    return lowrank_dict, sparse_dict


def compute_lors_stats(
    task_vector: torch.Tensor,
    rank_ratio: float = 0.1,
    sparsity: float = 0.9,
) -> Dict[str, float]:
    """
    Compute statistics about the LoRS decomposition for analysis.

    Returns energy captured by low-rank component, sparsity level,
    and reconstruction error.
    """
    tau_lr, tau_sp = lors_decompose(task_vector, rank_ratio, sparsity)

    tv_norm = torch.norm(task_vector).item()
    lr_norm = torch.norm(tau_lr).item()
    sp_norm = torch.norm(tau_sp).item()
    recon = torch.norm(task_vector - tau_lr - tau_sp).item()

    # SVD spectrum analysis
    S = torch.linalg.svdvals(task_vector)
    k = max(1, int(min(task_vector.shape) * rank_ratio))
    energy_captured = (S[:k] ** 2).sum().item() / (S ** 2).sum().item()

    return {
        "tv_frobenius_norm": tv_norm,
        "lowrank_norm": lr_norm,
        "sparse_norm": sp_norm,
        "reconstruction_error": recon,
        "relative_error": recon / tv_norm if tv_norm > 0 else 0.0,
        "energy_captured_by_topk": energy_captured,
        "rank_used": k,
        "total_rank": min(task_vector.shape),
        "sparse_nnz_ratio": (tau_sp != 0).float().mean().item(),
    }
