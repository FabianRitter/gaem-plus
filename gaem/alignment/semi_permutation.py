"""
Semi-permutation alignment for attention heads.

Based on GLMC (Theus 2025): attention heads can be soft-aligned
using doubly stochastic matrices (semi-permutations) via Sinkhorn
iteration. This allows weighted mixing of attention heads rather
than hard assignment.
"""

import torch
from typing import Optional


def sinkhorn(
    log_alpha: torch.Tensor,
    n_iters: int = 20,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for computing a doubly stochastic matrix.

    Args:
        log_alpha: Log-domain similarity matrix [n, n]
        n_iters: Number of Sinkhorn iterations
        temperature: Temperature for softmax (lower = closer to permutation)

    Returns:
        Doubly stochastic matrix [n, n]
    """
    log_alpha = log_alpha / temperature

    for _ in range(n_iters):
        # Row normalization (log-domain)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        # Column normalization (log-domain)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)

    return torch.exp(log_alpha)


def compute_head_similarity(
    heads_A: torch.Tensor,
    heads_B: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise similarity between attention heads.

    Args:
        heads_A: Head outputs from anchor [N, n_heads, d_head]
        heads_B: Head outputs from model to align [N, n_heads, d_head]

    Returns:
        Similarity matrix [n_heads, n_heads]
    """
    n_heads = heads_A.shape[1]
    sim = torch.zeros(n_heads, n_heads, device=heads_A.device)

    for i in range(n_heads):
        for j in range(n_heads):
            # Cosine similarity averaged over samples
            cos_sim = torch.nn.functional.cosine_similarity(
                heads_A[:, i, :], heads_B[:, j, :], dim=-1
            )
            sim[i, j] = cos_sim.mean()

    return sim


def semi_permutation_align(
    heads_A: torch.Tensor,
    heads_B: torch.Tensor,
    temperature: float = 0.1,
    n_sinkhorn_iters: int = 20,
) -> torch.Tensor:
    """
    Find semi-permutation matrix for soft attention head alignment.

    Args:
        heads_A: Head outputs from anchor [N, n_heads, d_head]
        heads_B: Head outputs from model to align [N, n_heads, d_head]
        temperature: Sinkhorn temperature (lower = harder assignment)
        n_sinkhorn_iters: Number of Sinkhorn iterations

    Returns:
        P_soft: Semi-permutation (doubly stochastic) matrix [n_heads, n_heads]
    """
    # Compute head-wise similarity
    S = compute_head_similarity(heads_A, heads_B)

    # Convert to doubly stochastic matrix via Sinkhorn
    P_soft = sinkhorn(S, n_iters=n_sinkhorn_iters, temperature=temperature)

    return P_soft
