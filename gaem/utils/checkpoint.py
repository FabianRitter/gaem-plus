"""
Checkpoint loading/saving utilities compatible with both
s3prl (SSL Phase 0) and HuggingFace (LLM Phase 1) formats.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Any


def load_checkpoint(
    path: str,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint, handling both s3prl and HuggingFace formats.

    Args:
        path: Path to checkpoint file (.ckpt or .bin or .pt)
        map_location: Device to load to

    Returns:
        Dict containing at minimum 'state_dict' key
    """
    path = Path(path)
    assert path.exists(), f"Checkpoint not found: {path}"

    ckpt = torch.load(str(path), map_location=map_location)

    # Normalize to always have 'state_dict' key
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt
        elif "model_state_dict" in ckpt:
            ckpt["state_dict"] = ckpt.pop("model_state_dict")
            return ckpt
        elif "Upstream" in ckpt:
            # s3prl format: upstream model state is under 'Upstream'
            return {"state_dict": ckpt["Upstream"], "_raw": ckpt}
        else:
            # Assume the dict IS the state dict
            return {"state_dict": ckpt}
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")


def save_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save a merged model checkpoint.

    Args:
        state_dict: Model state dict to save
        path: Output path
        metadata: Optional metadata (merge config, weights, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {"state_dict": state_dict}
    if metadata:
        ckpt["gaem_metadata"] = metadata

    torch.save(ckpt, str(path))
