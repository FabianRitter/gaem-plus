"""
Interpolation barrier analysis for merged models.

The interpolation barrier is the maximum loss increase along the
linear path between two models in weight space. A lower barrier
indicates better alignment (more connected loss basins).
"""

import torch
from typing import Dict, List, Callable, Optional, Tuple


def linear_interpolation_path(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    n_points: int = 11,
) -> List[Dict[str, torch.Tensor]]:
    """
    Generate n_points state dicts along the linear path from sd_a to sd_b.

    θ(α) = (1-α) * θ_a + α * θ_b, for α in [0, 1]

    Args:
        sd_a: Starting state dict
        sd_b: Ending state dict
        n_points: Number of interpolation points (including endpoints)

    Returns:
        List of state dicts along the interpolation path
    """
    alphas = torch.linspace(0.0, 1.0, n_points)
    path = []

    for alpha in alphas:
        interpolated = {}
        for name in sd_a:
            if name in sd_b:
                interpolated[name] = (1 - alpha) * sd_a[name] + alpha * sd_b[name]
            else:
                interpolated[name] = sd_a[name].clone()
        path.append(interpolated)

    return path


def interpolation_barrier(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    eval_fn: Callable[[Dict[str, torch.Tensor]], float],
    n_points: int = 11,
) -> Dict[str, float]:
    """
    Compute the interpolation barrier between two models.

    Args:
        sd_a: State dict of model A
        sd_b: State dict of model B
        eval_fn: Function that takes a state dict and returns a loss value.
                 Lower is better (e.g., cross-entropy loss, negative accuracy).
        n_points: Number of interpolation points

    Returns:
        Dict with:
          - "barrier": max loss along path minus average of endpoint losses
          - "loss_a": loss at model A
          - "loss_b": loss at model B
          - "max_loss": maximum loss along the path
          - "max_loss_alpha": alpha at which max loss occurs
          - "losses": all losses along the path
          - "alphas": all alpha values
    """
    path = linear_interpolation_path(sd_a, sd_b, n_points)
    alphas = torch.linspace(0.0, 1.0, n_points).tolist()

    losses = []
    for sd in path:
        loss = eval_fn(sd)
        losses.append(loss)

    loss_a = losses[0]
    loss_b = losses[-1]
    max_loss = max(losses)
    max_idx = losses.index(max_loss)
    avg_endpoint = (loss_a + loss_b) / 2

    return {
        "barrier": max_loss - avg_endpoint,
        "loss_a": loss_a,
        "loss_b": loss_b,
        "max_loss": max_loss,
        "max_loss_alpha": alphas[max_idx],
        "losses": losses,
        "alphas": alphas,
    }
