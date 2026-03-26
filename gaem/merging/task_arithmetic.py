"""
Task Arithmetic baselines for model merging.

Implements standard task arithmetic (Ilharco 2023), TIES-Merging (Yadav 2023),
and DARE (Yu 2024) as baselines for GAEM+ comparison.
"""

import torch
from typing import Dict, List, Optional, Tuple


def compute_task_vector(
    finetuned_sd: Dict[str, torch.Tensor],
    base_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute task vector: τ = θ_finetuned - θ_base.

    Args:
        finetuned_sd: State dict of finetuned model
        base_sd: State dict of base (pretrained) model

    Returns:
        Task vector as a state dict
    """
    tv = {}
    for name in base_sd:
        if name in finetuned_sd:
            tv[name] = finetuned_sd[name] - base_sd[name]
    return tv


def task_arithmetic_merge(
    base_sd: Dict[str, torch.Tensor],
    task_vectors: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """
    Standard task arithmetic: θ_merged = θ_base + Σ(λ_i * τ_i).

    Args:
        base_sd: Base model state dict
        task_vectors: List of task vector state dicts
        weights: Scaling coefficients for each task vector

    Returns:
        Merged model state dict
    """
    assert len(task_vectors) == len(weights)

    merged = {}
    for name in base_sd:
        merged[name] = base_sd[name].clone()
        for tv, w in zip(task_vectors, weights):
            if name in tv:
                merged[name] = merged[name] + w * tv[name]

    return merged


def ties_merge(
    task_vectors: List[Dict[str, torch.Tensor]],
    weights: List[float],
    k: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    TIES-Merging: Trim, Elect Sign, Disjoint Merge.

    1. Trim: Keep only top-k% magnitude values per task vector
    2. Elect Sign: For each parameter, take the majority sign across tasks
    3. Disjoint Merge: Average only values that agree with the elected sign

    Args:
        task_vectors: List of task vector state dicts
        weights: Scaling coefficients
        k: Top-k percentage to keep (0.0 to 1.0)

    Returns:
        Merged task vector (to be added to base model)
    """
    merged_tv = {}

    for name in task_vectors[0]:
        tensors = [tv[name] for tv in task_vectors if name in tv]
        ws = weights[:len(tensors)]

        if tensors[0].dim() == 0:
            # Scalar parameters: simple weighted average
            merged_tv[name] = sum(w * t for w, t in zip(ws, tensors))
            continue

        # Step 1: Trim - keep top-k% by magnitude
        trimmed = []
        for t in tensors:
            flat = t.abs().flatten()
            if k < 1.0:
                threshold = torch.quantile(flat, 1.0 - k)
                mask = t.abs() >= threshold
                trimmed.append(t * mask.float())
            else:
                trimmed.append(t)

        # Step 2: Elect sign - majority vote across tasks
        signs = torch.stack([torch.sign(t) for t in trimmed])
        sign_sum = signs.sum(dim=0)
        elected_sign = torch.sign(sign_sum)
        # Where sign_sum is 0, use the sign from the largest magnitude task
        zero_mask = elected_sign == 0
        if zero_mask.any():
            magnitudes = torch.stack([t.abs() for t in trimmed])
            max_idx = magnitudes.argmax(dim=0)
            for i in range(len(trimmed)):
                elected_sign[zero_mask & (max_idx == i)] = torch.sign(
                    trimmed[i][zero_mask & (max_idx == i)]
                )

        # Step 3: Disjoint merge - average values agreeing with elected sign
        accumulated = torch.zeros_like(tensors[0])
        count = torch.zeros_like(tensors[0])
        for t, w in zip(trimmed, ws):
            agree = torch.sign(t) == elected_sign
            accumulated += w * t * agree.float()
            count += agree.float()

        count = count.clamp(min=1)
        merged_tv[name] = accumulated / count

    return merged_tv


def dare_merge(
    task_vectors: List[Dict[str, torch.Tensor]],
    weights: List[float],
    drop_rate: float = 0.9,
    rescale: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    DARE: Drop And REscale merging.

    Randomly drops a fraction of task vector entries and rescales the rest.

    Args:
        task_vectors: List of task vector state dicts
        weights: Scaling coefficients
        drop_rate: Fraction of entries to drop (0.0 to 1.0)
        rescale: Whether to rescale remaining entries by 1/(1-drop_rate)

    Returns:
        Merged task vector
    """
    merged_tv = {}
    scale = 1.0 / (1.0 - drop_rate) if rescale and drop_rate < 1.0 else 1.0

    for name in task_vectors[0]:
        tensors = [tv[name] for tv in task_vectors if name in tv]
        ws = weights[:len(tensors)]

        accumulated = torch.zeros_like(tensors[0])
        for t, w in zip(tensors, ws):
            mask = torch.bernoulli(
                torch.full_like(t, 1.0 - drop_rate)
            )
            accumulated += w * t * mask * scale

        merged_tv[name] = accumulated

    return merged_tv
