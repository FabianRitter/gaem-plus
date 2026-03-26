"""
GAEM+ (Generalized Audio Encoder Merging) — full pipeline.

Combines:
1. Task vector extraction
2. LoRS decomposition (low-rank + sparse)
3. Orthogonal alignment on low-rank components (Procrustes)
4. Sparse merging (TIES/DARE) on sparse components
5. Reconstruction of merged model

This is the main contribution: the combined pipeline that neither
GLMC, LoRS, nor OSRM implements alone.
"""

import torch
from typing import Dict, List, Optional, Tuple, Literal

from gaem.alignment.procrustes import procrustes_orthogonal, align_state_dict_orthogonal
from gaem.alignment.permutation import correlation_permutation
from gaem.decomposition.lors import lors_decompose_state_dict
from gaem.merging.task_arithmetic import (
    compute_task_vector,
    task_arithmetic_merge,
    ties_merge,
    dare_merge,
)


def gaem_plus_merge(
    base_sd: Dict[str, torch.Tensor],
    finetuned_sds: List[Dict[str, torch.Tensor]],
    weights: List[float],
    features_per_model: Optional[List[torch.Tensor]] = None,
    alignment: Literal["orthogonal", "permutation", "none"] = "orthogonal",
    decompose: bool = True,
    rank_ratio: float = 0.1,
    sparsity: float = 0.9,
    sparse_method: Literal["ties", "dare", "average"] = "ties",
    ties_k: float = 0.2,
    dare_drop_rate: float = 0.9,
    anchor_idx: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Full GAEM+ merge pipeline.

    Steps:
    1. Compute task vectors (finetuned - base)
    2. Optionally align task vectors using features
    3. Optionally decompose into low-rank + sparse (LoRS)
    4. Merge low-rank components via weighted average
    5. Merge sparse components via TIES/DARE
    6. Reconstruct: base + merged_lowrank + merged_sparse

    Args:
        base_sd: Base (pretrained) model state dict
        finetuned_sds: List of finetuned model state dicts
        weights: Merging weights for each model
        features_per_model: List of feature tensors [N, d] for alignment.
                           Required if alignment != "none".
        alignment: Alignment method ("orthogonal", "permutation", "none")
        decompose: Whether to apply LoRS decomposition
        rank_ratio: LoRS rank ratio
        sparsity: LoRS sparsity level
        sparse_method: How to merge sparse components
        ties_k: Top-k for TIES merging
        dare_drop_rate: Drop rate for DARE
        anchor_idx: Index of the anchor model for alignment

    Returns:
        Merged model state dict
    """
    n_models = len(finetuned_sds)
    assert len(weights) == n_models

    # Step 1: Compute task vectors
    task_vectors = [compute_task_vector(sd, base_sd) for sd in finetuned_sds]

    # Step 2: Align task vectors
    if alignment != "none" and features_per_model is not None:
        anchor_features = features_per_model[anchor_idx]
        aligned_tvs = [task_vectors[anchor_idx]]  # anchor stays as-is

        for i in range(n_models):
            if i == anchor_idx:
                continue

            if alignment == "orthogonal":
                O = procrustes_orthogonal(anchor_features, features_per_model[i])
                aligned_tv = align_state_dict_orthogonal(
                    base_sd, task_vectors[i], O
                )
            elif alignment == "permutation":
                P = correlation_permutation(anchor_features, features_per_model[i])
                # For permutation, we apply P to the task vector
                aligned_tv = {}
                for name, tv in task_vectors[i].items():
                    if tv.dim() == 2:
                        aligned_tv[name] = P @ tv @ P.T
                    elif tv.dim() == 1 and "bias" in name:
                        aligned_tv[name] = P @ tv
                    else:
                        aligned_tv[name] = tv.clone()
            else:
                aligned_tv = task_vectors[i]

            aligned_tvs.append(aligned_tv)

        task_vectors = aligned_tvs

    # Step 3 & 4: Decompose and merge
    if decompose:
        # LoRS decomposition on each task vector
        lowrank_components = []
        sparse_components = []
        for tv in task_vectors:
            lr, sp = lors_decompose_state_dict(tv, rank_ratio=rank_ratio, sparsity=sparsity)
            lowrank_components.append(lr)
            sparse_components.append(sp)

        # Merge low-rank components: simple weighted average
        # (low-rank captures structural info — averaging preserves this)
        merged_lowrank = {}
        for name in lowrank_components[0]:
            merged_lowrank[name] = sum(
                w * lr[name] for w, lr in zip(weights, lowrank_components)
            )

        # Merge sparse components: use TIES, DARE, or average
        if sparse_method == "ties":
            merged_sparse = ties_merge(sparse_components, weights, k=ties_k)
        elif sparse_method == "dare":
            merged_sparse = dare_merge(sparse_components, weights, drop_rate=dare_drop_rate)
        else:
            merged_sparse = {}
            for name in sparse_components[0]:
                merged_sparse[name] = sum(
                    w * sp[name] for w, sp in zip(weights, sparse_components)
                )

        # Step 5: Reconstruct
        merged_sd = {}
        for name in base_sd:
            merged_sd[name] = (
                base_sd[name]
                + merged_lowrank.get(name, torch.zeros_like(base_sd[name]))
                + merged_sparse.get(name, torch.zeros_like(base_sd[name]))
            )
    else:
        # No decomposition: direct task arithmetic merge
        merged_sd = task_arithmetic_merge(base_sd, task_vectors, weights)

    return merged_sd


def gaem_plus_merge_ablation(
    base_sd: Dict[str, torch.Tensor],
    finetuned_sds: List[Dict[str, torch.Tensor]],
    weights: List[float],
    features_per_model: Optional[List[torch.Tensor]] = None,
    anchor_idx: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Run all ablation variants and return results for comparison.

    Returns a dict of method_name -> merged_state_dict for:
    - "task_arithmetic": plain weighted average
    - "ties": TIES-Merging only
    - "permutation_only": permutation alignment + average
    - "orthogonal_only": orthogonal alignment + average
    - "lors_only": LoRS decomposition + average (no alignment)
    - "gaem_plus": full pipeline (orthogonal + LoRS + TIES)

    This is for Experiment 1 in the research plan.
    """
    results = {}

    # Baseline: Task Arithmetic
    tvs = [compute_task_vector(sd, base_sd) for sd in finetuned_sds]
    results["task_arithmetic"] = task_arithmetic_merge(base_sd, tvs, weights)

    # TIES only
    merged_ties_tv = ties_merge(tvs, weights)
    results["ties"] = {}
    for name in base_sd:
        results["ties"][name] = base_sd[name] + merged_ties_tv.get(
            name, torch.zeros_like(base_sd[name])
        )

    # Permutation + average
    if features_per_model is not None:
        results["permutation_only"] = gaem_plus_merge(
            base_sd, finetuned_sds, weights, features_per_model,
            alignment="permutation", decompose=False, anchor_idx=anchor_idx,
        )

        # Orthogonal + average
        results["orthogonal_only"] = gaem_plus_merge(
            base_sd, finetuned_sds, weights, features_per_model,
            alignment="orthogonal", decompose=False, anchor_idx=anchor_idx,
        )

    # LoRS + average (no alignment)
    results["lors_only"] = gaem_plus_merge(
        base_sd, finetuned_sds, weights, features_per_model=None,
        alignment="none", decompose=True,
    )

    # Full GAEM+
    if features_per_model is not None:
        results["gaem_plus"] = gaem_plus_merge(
            base_sd, finetuned_sds, weights, features_per_model,
            alignment="orthogonal", decompose=True,
            sparse_method="ties", anchor_idx=anchor_idx,
        )

    return results
