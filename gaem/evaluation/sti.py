"""
Singular Task Interference (STI) metric.

Based on TSV (Gargiulo et al., CVPR 2025): "Task Singular Vectors:
Reducing Task Interference in Model Merging"

STI quantifies interference between task vectors at the singular-vector
level within each layer. This is more fine-grained than cosine similarity
because it captures directional interference weighted by importance
(singular values).

STI({Δᵢ}) = || (UᵀU - I) Σ (VᵀV - I) ||₁

where U = [U₁|U₂|...|Uₜ] are concatenated left singular vectors,
V = [V₁|V₂|...|Vₜ] are concatenated right singular vectors,
Σ is a block-diagonal matrix of singular values.

Lower STI = less interference = better mergeability.
"""

import torch
from typing import Dict, List, Optional, Tuple


def compute_sti(
    task_matrices: List[torch.Tensor],
    rank_per_task: Optional[int] = None,
) -> float:
    """
    Compute Singular Task Interference for a single layer.

    Args:
        task_matrices: List of T task matrices, each [m, n]
                      (weight differences: θ_finetuned - θ_base)
        rank_per_task: Number of singular components to keep per task.
                      Default: min(m,n) // T

    Returns:
        STI scalar value (lower = less interference)
    """
    T = len(task_matrices)
    m, n = task_matrices[0].shape
    min_dim = min(m, n)

    if rank_per_task is None:
        rank_per_task = max(1, min_dim // T)

    # SVD each task matrix and keep top-k components
    Us, Ss, Vs = [], [], []
    for delta in task_matrices:
        U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
        k = min(rank_per_task, len(S))
        Us.append(U[:, :k])     # [m, k]
        Ss.append(S[:k])         # [k]
        Vs.append(Vt[:k, :].T)  # [n, k]

    # Concatenate across tasks
    U_cat = torch.cat(Us, dim=1)    # [m, T*k]
    V_cat = torch.cat(Vs, dim=1)    # [n, T*k]

    # Build block-diagonal Sigma
    all_S = torch.cat(Ss)  # [T*k]
    Sigma = torch.diag(all_S)  # [T*k, T*k]

    # Gram matrices
    total_k = U_cat.shape[1]
    I = torch.eye(total_k, device=U_cat.device)

    U_gram = U_cat.T @ U_cat  # [T*k, T*k]
    V_gram = V_cat.T @ V_cat  # [T*k, T*k]

    # STI = || (U^T U - I) @ Sigma @ (V^T V - I) ||_1
    interference = (U_gram - I) @ Sigma @ (V_gram - I)
    sti = torch.norm(interference, p=1).item()

    return sti


def compute_sti_normalized(
    task_matrices: List[torch.Tensor],
    rank_per_task: Optional[int] = None,
) -> float:
    """
    Compute normalized STI (divided by total energy for cross-layer comparison).

    Returns STI / (sum of all singular values), making it comparable
    across layers of different sizes.
    """
    T = len(task_matrices)
    m, n = task_matrices[0].shape
    min_dim = min(m, n)

    if rank_per_task is None:
        rank_per_task = max(1, min_dim // T)

    sti = compute_sti(task_matrices, rank_per_task)

    # Total energy: sum of all singular values across tasks
    total_energy = 0.0
    for delta in task_matrices:
        S = torch.linalg.svdvals(delta)
        total_energy += S.sum().item()

    return sti / total_energy if total_energy > 0 else 0.0


def layerwise_sti(
    task_vectors: List[Dict[str, torch.Tensor]],
    rank_per_task: Optional[int] = None,
    domain_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute STI for each layer in the model.

    This is the main diagnostic function: it identifies which layers
    have the most inter-domain interference.

    Args:
        task_vectors: List of T task vector state dicts
        rank_per_task: Singular components per task (None = auto)
        domain_names: Names for reporting (e.g., ["speech", "music"])

    Returns:
        Dict of layer_name -> {"sti": value, "sti_normalized": value,
                               "rank_used": k, "layer_shape": (m,n)}
    """
    T = len(task_vectors)
    if domain_names is None:
        domain_names = [f"task_{i}" for i in range(T)]

    # Find common 2D parameters
    common_names = set(task_vectors[0].keys())
    for tv in task_vectors[1:]:
        common_names &= set(tv.keys())

    results = {}
    for name in sorted(common_names):
        # Only compute for 2D weight matrices
        if task_vectors[0][name].dim() != 2:
            continue

        matrices = [tv[name] for tv in task_vectors]
        m, n = matrices[0].shape

        # Skip very small layers
        if min(m, n) < T * 2:
            continue

        k = rank_per_task if rank_per_task else max(1, min(m, n) // T)

        sti = compute_sti(matrices, k)
        sti_norm = compute_sti_normalized(matrices, k)

        results[name] = {
            "sti": sti,
            "sti_normalized": sti_norm,
            "rank_used": k,
            "layer_shape": (m, n),
        }

    return results


def tsv_merge(
    task_matrices: List[torch.Tensor],
    rank_per_task: Optional[int] = None,
) -> torch.Tensor:
    """
    TSV-Merge: Procrustes decorrelation of task singular vectors.

    After SVD decomposition, applies Procrustes orthogonalization
    to the concatenated singular vectors to decorrelate inter-task
    directions, then reconstructs.

    This can be used as a post-alignment interference reduction step
    in the GAEM+ pipeline.

    Args:
        task_matrices: List of T task matrices [m, n]
        rank_per_task: Components per task (None = auto)

    Returns:
        Merged matrix [m, n]
    """
    T = len(task_matrices)
    m, n = task_matrices[0].shape
    min_dim = min(m, n)

    if rank_per_task is None:
        rank_per_task = max(1, min_dim // T)

    # SVD and truncate
    Us, Ss, Vts = [], [], []
    for delta in task_matrices:
        U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
        k = min(rank_per_task, len(S))
        Us.append(U[:, :k])
        Ss.append(S[:k])
        Vts.append(Vt[:k, :])

    # Concatenate
    U_cat = torch.cat(Us, dim=1)       # [m, T*k]
    V_cat = torch.cat(Vts, dim=0).T    # [n, T*k]
    S_cat = torch.cat(Ss)              # [T*k]

    # Procrustes orthogonalization on U: find nearest orthogonal matrix
    # U_perp = P_U @ Q_U^T where P_U S_U Q_U^T = SVD(U_cat)
    P_U, _, Q_Ut = torch.linalg.svd(U_cat, full_matrices=False)
    U_perp = P_U @ Q_Ut  # [m, T*k]

    # Same for V
    P_V, _, Q_Vt = torch.linalg.svd(V_cat, full_matrices=False)
    V_perp = P_V @ Q_Vt  # [n, T*k]

    # Reconstruct: M = U_perp @ diag(S) @ V_perp^T
    merged = U_perp @ torch.diag(S_cat) @ V_perp.T

    return merged


def tsv_merge_state_dict(
    task_vectors: List[Dict[str, torch.Tensor]],
    rank_per_task: Optional[int] = None,
    min_dim: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    Apply TSV-Merge to all 2D parameters in task vector state dicts.

    1D parameters (biases, norms) are averaged.

    Args:
        task_vectors: List of T task vector state dicts
        rank_per_task: Components per task (None = auto)
        min_dim: Minimum layer dimension for TSV (smaller layers averaged)

    Returns:
        Merged task vector state dict
    """
    T = len(task_vectors)
    merged = {}

    common_names = set(task_vectors[0].keys())
    for tv in task_vectors[1:]:
        common_names &= set(tv.keys())

    for name in sorted(common_names):
        matrices = [tv[name] for tv in task_vectors]

        if matrices[0].dim() == 2 and min(matrices[0].shape) >= max(min_dim, T * 2):
            merged[name] = tsv_merge(matrices, rank_per_task)
        else:
            # Simple average for 1D params or small matrices
            merged[name] = sum(matrices) / T

    return merged
