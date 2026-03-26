"""
Feature extraction utilities for alignment computation.

Extracts intermediate features from audio encoders (HuBERT, MERT, BEATs)
for computing alignment matrices (Procrustes, permutation, etc.).
"""

import torch
from typing import Optional, List, Callable
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features_from_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_idx: int = -1,
    max_samples: int = 1000,
    feature_hook: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Extract features from a specific layer of an encoder.

    Args:
        model: Audio encoder model (HuBERT, MERT, BEATs, or distilled student)
        dataloader: DataLoader yielding audio tensors
        layer_idx: Which layer to extract from (-1 = last hidden state)
        max_samples: Maximum number of samples to process
        feature_hook: Optional custom feature extraction function.
                     Signature: hook(model, batch) -> features [B, T, d]

    Returns:
        Features tensor [N, d] (time-averaged across frames)
    """
    model.eval()
    device = next(model.parameters()).device

    all_features = []
    n_samples = 0

    for batch in dataloader:
        if n_samples >= max_samples:
            break

        if isinstance(batch, (list, tuple)):
            audio = batch[0]
        else:
            audio = batch

        audio = audio.to(device)

        if feature_hook is not None:
            features = feature_hook(model, audio)
        else:
            # Default: use forward pass and extract hidden states
            output = model(audio)

            if isinstance(output, dict):
                # HuggingFace-style output
                if "hidden_states" in output and output["hidden_states"] is not None:
                    features = output["hidden_states"][layer_idx]
                elif "last_hidden_state" in output:
                    features = output["last_hidden_state"]
                else:
                    raise ValueError(f"Cannot find features in output keys: {output.keys()}")
            elif isinstance(output, (list, tuple)):
                features = output[layer_idx] if layer_idx != -1 else output[-1]
            else:
                features = output

        # Time-average: [B, T, d] -> [B, d]
        if features.dim() == 3:
            features = features.mean(dim=1)

        all_features.append(features.cpu())
        n_samples += features.shape[0]

    return torch.cat(all_features, dim=0)[:max_samples]


@torch.no_grad()
def extract_head_outputs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_idx: int = 0,
    max_samples: int = 500,
) -> torch.Tensor:
    """
    Extract per-head attention outputs for semi-permutation alignment.

    Returns tensor [N, n_heads, d_head].
    Requires hooking into the attention module.
    """
    model.eval()
    device = next(model.parameters()).device
    head_outputs = []

    # Register hook on the attention layer
    hook_storage = {"output": None}

    def _hook(module, input, output):
        # output is typically (attn_output, attn_weights)
        if isinstance(output, tuple):
            hook_storage["output"] = output[0]
        else:
            hook_storage["output"] = output

    # Find the attention module at the given layer
    # This is encoder-specific — provide a default for HuBERT-like architectures
    attn_module = _find_attention_module(model, layer_idx)
    if attn_module is None:
        raise ValueError(
            f"Could not find attention module at layer {layer_idx}. "
            "Provide a custom feature_hook instead."
        )

    handle = attn_module.register_forward_hook(_hook)

    try:
        n_samples = 0
        for batch in dataloader:
            if n_samples >= max_samples:
                break

            audio = batch[0] if isinstance(batch, (list, tuple)) else batch
            audio = audio.to(device)
            _ = model(audio)

            if hook_storage["output"] is not None:
                attn_out = hook_storage["output"]
                # Reshape to [B, n_heads, d_head] if needed
                if attn_out.dim() == 3:
                    # [B, T, d] -> time-average -> [B, d] -> reshape
                    attn_out = attn_out.mean(dim=1)
                head_outputs.append(attn_out.cpu())
                n_samples += attn_out.shape[0]
    finally:
        handle.remove()

    return torch.cat(head_outputs, dim=0)[:max_samples]


def _find_attention_module(model, layer_idx):
    """Try to find the self-attention module at a given layer index."""
    # Try common patterns for transformer encoders
    for pattern in [
        f"encoder.layers.{layer_idx}.self_attn",
        f"encoder.layers.{layer_idx}.attention",
        f"transformer.layers.{layer_idx}.self_attn",
        f"layers.{layer_idx}.attention.self_attn",
    ]:
        parts = pattern.split(".")
        mod = model
        try:
            for part in parts:
                mod = getattr(mod, part)
            return mod
        except AttributeError:
            continue
    return None
