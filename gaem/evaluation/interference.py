"""
Domain interference analysis.

Metrics for quantifying how much merging models from different audio
domains (speech, music, audio events) causes interference/degradation.

Based on metrics from LoRS-Merging and TSV papers.
"""

import torch
from typing import Dict, List, Optional, Tuple


def compute_domain_interference(
    base_sd: Dict[str, torch.Tensor],
    task_vectors: List[Dict[str, torch.Tensor]],
    domain_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute pairwise interference between task vectors.

    Interference is measured as the cosine similarity between flattened
    task vectors. High positive similarity = cooperative; high negative
    similarity = interfering.

    Args:
        base_sd: Base model state dict (for reference dimensions)
        task_vectors: List of task vector state dicts
        domain_names: Names for each domain (e.g., ["speech", "music", "audio"])

    Returns:
        Dict of interference metrics:
          - "cosine_{i}_{j}": pairwise cosine similarity
          - "magnitude_ratio_{i}_{j}": ratio of task vector magnitudes
          - "sign_agreement_{i}_{j}": fraction of params with same sign
    """
    n = len(task_vectors)
    if domain_names is None:
        domain_names = [f"model_{i}" for i in range(n)]

    # Flatten task vectors
    flat_tvs = []
    for tv in task_vectors:
        flat = torch.cat([tv[name].flatten() for name in sorted(tv.keys())])
        flat_tvs.append(flat)

    results = {}

    for i in range(n):
        for j in range(i + 1, n):
            name_pair = f"{domain_names[i]}_vs_{domain_names[j]}"

            # Cosine similarity
            cos = torch.nn.functional.cosine_similarity(
                flat_tvs[i].unsqueeze(0), flat_tvs[j].unsqueeze(0)
            ).item()
            results[f"cosine_{name_pair}"] = cos

            # Magnitude ratio
            mag_i = torch.norm(flat_tvs[i]).item()
            mag_j = torch.norm(flat_tvs[j]).item()
            results[f"magnitude_ratio_{name_pair}"] = min(mag_i, mag_j) / max(mag_i, mag_j)

            # Sign agreement
            sign_agree = (
                (torch.sign(flat_tvs[i]) == torch.sign(flat_tvs[j])).float().mean().item()
            )
            results[f"sign_agreement_{name_pair}"] = sign_agree

    return results


def layerwise_interference(
    task_vectors: List[Dict[str, torch.Tensor]],
    domain_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute interference at each layer for visualization.

    Returns per-layer cosine similarity between pairs of task vectors.
    Useful for identifying which layers have the most domain conflict.

    Args:
        task_vectors: List of task vector state dicts
        domain_names: Names for each domain

    Returns:
        Dict of layer_name -> {pair_name: cosine_similarity}
    """
    n = len(task_vectors)
    if domain_names is None:
        domain_names = [f"model_{i}" for i in range(n)]

    # Get common parameter names
    common_names = set(task_vectors[0].keys())
    for tv in task_vectors[1:]:
        common_names &= set(tv.keys())

    results = {}
    for name in sorted(common_names):
        if task_vectors[0][name].dim() < 2:
            continue  # Skip 1D params

        layer_results = {}
        for i in range(n):
            for j in range(i + 1, n):
                pair = f"{domain_names[i]}_vs_{domain_names[j]}"
                flat_i = task_vectors[i][name].flatten()
                flat_j = task_vectors[j][name].flatten()
                cos = torch.nn.functional.cosine_similarity(
                    flat_i.unsqueeze(0), flat_j.unsqueeze(0)
                ).item()
                layer_results[pair] = cos

        results[name] = layer_results

    return results
