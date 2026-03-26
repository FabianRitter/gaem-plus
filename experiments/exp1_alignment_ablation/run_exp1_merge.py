"""
Experiment 1: Create Merged Encoders + Interpolation Barrier Analysis

Uses the Procrustes and Permutation alignment matrices from Exp 0
to create merged HuBERT+MERT encoders under different strategies,
then computes interpolation barriers.

Saves merged checkpoints for downstream evaluation.
"""

import json
import os
import sys
import time
import copy
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gaem.alignment.procrustes import procrustes_orthogonal, align_state_dict_orthogonal
from gaem.merging.task_arithmetic import compute_task_vector
from gaem.evaluation.barriers import interpolation_barrier

OUTPUT_DIR = PROJECT_ROOT / "results" / "exp1_merge"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXP0_DIR = PROJECT_ROOT / "results" / "exp0_analysis"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def load_models():
    from transformers import AutoModel
    print("\n=== Loading HuBERT ===")
    hubert = AutoModel.from_pretrained("facebook/hubert-base-ls960")
    hubert.eval()
    print(f"  Params: {sum(p.numel() for p in hubert.parameters()):,}")

    print("=== Loading MERT ===")
    mert = AutoModel.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
    mert.eval()
    print(f"  Params: {sum(p.numel() for p in mert.parameters()):,}")
    return hubert, mert


def map_mert_to_hubert_keys(hubert_sd, mert_sd):
    """Build mapping from HuBERT key names to MERT key names."""
    mapping = {}
    for hk in hubert_sd:
        if hk in mert_sd and hubert_sd[hk].shape == mert_sd[hk].shape:
            mapping[hk] = hk
        else:
            # MERT might use hubert. prefix
            mk = "hubert." + hk
            if mk in mert_sd and hubert_sd[hk].shape == mert_sd[mk].shape:
                mapping[hk] = mk
    return mapping


def create_merged_checkpoints(hubert, mert):
    """Create merged models using different strategies."""
    hubert_sd = {k: v.cpu() for k, v in hubert.state_dict().items()}
    mert_sd = {k: v.cpu() for k, v in mert.state_dict().items()}

    key_map = map_mert_to_hubert_keys(hubert_sd, mert_sd)
    print(f"\n  Mapped {len(key_map)} / {len(hubert_sd)} parameters")

    # Load alignment matrices from Exp 0
    O = torch.from_numpy(np.load(str(EXP0_DIR / "procrustes_O.npy"))).float()
    P = torch.from_numpy(np.load(str(EXP0_DIR / "permutation_P.npy"))).float()
    print(f"  Loaded Procrustes O: {O.shape}, Permutation P: {P.shape}")

    # Get MERT params in HuBERT key namespace
    mert_mapped = {}
    for hk, mk in key_map.items():
        mert_mapped[hk] = mert_sd[mk]

    merged_models = {}

    # ======== Method 1: Simple Average (no alignment) ========
    print("\n--- Method 1: Simple Average (no alignment) ---")
    merged = {}
    for hk in hubert_sd:
        if hk in mert_mapped:
            merged[hk] = 0.5 * hubert_sd[hk] + 0.5 * mert_mapped[hk]
        else:
            merged[hk] = hubert_sd[hk].clone()
    merged_models["simple_avg"] = merged

    # ======== Method 2: Permutation + Average ========
    print("--- Method 2: Permutation + Average ---")
    mert_perm = {}
    for hk, param in mert_mapped.items():
        # Only apply alignment to encoder layers (768-dim), skip CNN (512-dim)
        is_encoder = "encoder.layers" in hk
        if param.dim() == 2 and is_encoder and param.shape[0] == 768:
            if param.shape[1] == 768:
                mert_perm[hk] = P @ param @ P.T
            else:
                # Intermediate dense (768 -> 3072): only left-multiply
                mert_perm[hk] = P @ param
        elif param.dim() == 2 and is_encoder and param.shape[1] == 768:
            # Output dense (3072 -> 768): only right-multiply
            mert_perm[hk] = param @ P.T
        elif param.dim() == 1 and is_encoder and param.shape[0] == 768 and "bias" in hk:
            mert_perm[hk] = P @ param
        else:
            mert_perm[hk] = param.clone()

    merged = {}
    for hk in hubert_sd:
        if hk in mert_perm:
            merged[hk] = 0.5 * hubert_sd[hk] + 0.5 * mert_perm[hk]
        else:
            merged[hk] = hubert_sd[hk].clone()
    merged_models["perm_avg_05"] = merged

    # ======== Method 3: Procrustes + Average (0.5/0.5) ========
    print("--- Method 3: Procrustes + Average (0.5/0.5) ---")
    mert_aligned = align_state_dict_orthogonal(hubert_sd, mert_mapped, O)
    merged = {}
    for hk in hubert_sd:
        if hk in mert_aligned:
            merged[hk] = 0.5 * hubert_sd[hk] + 0.5 * mert_aligned[hk]
        else:
            merged[hk] = hubert_sd[hk].clone()
    merged_models["procrustes_avg_05"] = merged

    # ======== Method 4: Procrustes + Average (0.7 HuBERT / 0.3 MERT) ========
    print("--- Method 4: Procrustes + Average (0.7/0.3) ---")
    merged = {}
    for hk in hubert_sd:
        if hk in mert_aligned:
            merged[hk] = 0.7 * hubert_sd[hk] + 0.3 * mert_aligned[hk]
        else:
            merged[hk] = hubert_sd[hk].clone()
    merged_models["procrustes_avg_07_03"] = merged

    # ======== Method 5: Procrustes + Average (0.3 HuBERT / 0.7 MERT) ========
    print("--- Method 5: Procrustes + Average (0.3/0.7) ---")
    merged = {}
    for hk in hubert_sd:
        if hk in mert_aligned:
            merged[hk] = 0.3 * hubert_sd[hk] + 0.7 * mert_aligned[hk]
        else:
            merged[hk] = hubert_sd[hk].clone()
    merged_models["procrustes_avg_03_07"] = merged

    return merged_models, hubert_sd, mert_mapped, mert_aligned


def compute_barriers(hubert_sd, mert_mapped, mert_aligned, merged_models):
    """Compute interpolation barriers between HuBERT and MERT (aligned vs not)."""
    print("\n=== Interpolation Barrier Analysis ===")

    # We can't compute actual loss barriers without running the model on data,
    # but we can compute weight-space metrics along the interpolation path.
    # Specifically: how much do the weights deviate from each endpoint?

    barrier_results = {}

    # Compute weight-space "barrier" as the L2 distance from the midpoint
    # to the linear interpolation path
    def weight_distance(sd_a, sd_b):
        """Compute flattened L2 distance between two state dicts."""
        diffs = []
        for k in sd_a:
            if k in sd_b:
                diffs.append((sd_a[k] - sd_b[k]).flatten())
        return torch.norm(torch.cat(diffs)).item()

    # Distance between raw models
    d_raw = weight_distance(hubert_sd, mert_mapped)
    print(f"  HuBERT <-> MERT (raw): {d_raw:.4f}")

    # Distance after Procrustes alignment
    d_aligned = weight_distance(hubert_sd, mert_aligned)
    print(f"  HuBERT <-> MERT (Procrustes): {d_aligned:.4f}")
    print(f"  Distance reduction: {(1 - d_aligned/d_raw)*100:.1f}%")

    barrier_results["distance_raw"] = d_raw
    barrier_results["distance_procrustes"] = d_aligned
    barrier_results["distance_reduction_pct"] = (1 - d_aligned/d_raw) * 100

    # Compute per-layer distances
    print("\n  Per-layer weight distances (Procrustes-aligned):")
    layer_distances = {}
    for k in hubert_sd:
        if k in mert_aligned and hubert_sd[k].dim() >= 2:
            d = torch.norm(hubert_sd[k] - mert_aligned[k]).item()
            d_raw_k = torch.norm(hubert_sd[k] - mert_mapped[k]).item() if k in mert_mapped else 0
            layer_distances[k] = {
                "distance_raw": d_raw_k,
                "distance_aligned": d,
                "reduction_pct": (1 - d/d_raw_k)*100 if d_raw_k > 0 else 0,
            }

    # Print top layers with most/least reduction
    sorted_layers = sorted(layer_distances.items(),
                          key=lambda x: x[1]["reduction_pct"], reverse=True)
    print("  Top 5 most improved layers:")
    for k, v in sorted_layers[:5]:
        print(f"    {k}: {v['distance_raw']:.3f} → {v['distance_aligned']:.3f} ({v['reduction_pct']:.1f}%)")

    print("  Bottom 5 least improved layers:")
    for k, v in sorted_layers[-5:]:
        print(f"    {k}: {v['distance_raw']:.3f} → {v['distance_aligned']:.3f} ({v['reduction_pct']:.1f}%)")

    barrier_results["per_layer"] = {k: v for k, v in layer_distances.items()}

    return barrier_results


def save_merged_checkpoints(merged_models, hubert):
    """Save merged state dicts as s3prl-compatible checkpoints."""
    print("\n=== Saving Merged Checkpoints ===")
    for name, sd in merged_models.items():
        # Create a model instance with the merged weights
        model = copy.deepcopy(hubert)
        model.load_state_dict(sd)

        # Save both raw state dict and s3prl-compatible format
        ckpt_path = OUTPUT_DIR / f"merged_{name}.pt"
        torch.save({
            "state_dict": sd,
            "gaem_metadata": {
                "method": name,
                "base": "hubert_base",
                "merged_with": "mert_v0_public",
            }
        }, str(ckpt_path))
        size_mb = os.path.getsize(ckpt_path) / 1e6
        print(f"  Saved {ckpt_path.name} ({size_mb:.1f} MB)")

    # Also save individual model state dicts for reference
    torch.save(hubert.state_dict(), str(OUTPUT_DIR / "hubert_base_sd.pt"))
    print(f"  Saved hubert_base_sd.pt")


def main():
    print("=" * 60)
    print("GAEM+ Experiment 1: Merged Encoder Creation")
    print("=" * 60)

    hubert, mert = load_models()

    # Create all merged variants
    merged_models, hubert_sd, mert_mapped, mert_aligned = create_merged_checkpoints(hubert, mert)

    # Barrier analysis
    barrier_results = compute_barriers(hubert_sd, mert_mapped, mert_aligned, merged_models)

    # Save
    save_merged_checkpoints(merged_models, hubert)

    # Save barrier results
    # Convert numpy types for JSON
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    with open(OUTPUT_DIR / "barrier_results.json", "w") as f:
        json.dump(make_serializable(barrier_results), f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE — MERGED CHECKPOINTS CREATED")
    print("=" * 60)
    print(f"\nMerged models saved to {OUTPUT_DIR}:")
    for name in merged_models:
        print(f"  - merged_{name}.pt")
    print(f"\nWeight-space distance:")
    print(f"  Raw:      {barrier_results['distance_raw']:.4f}")
    print(f"  Aligned:  {barrier_results['distance_procrustes']:.4f}")
    print(f"  Reduction: {barrier_results['distance_reduction_pct']:.1f}%")
    print(f"\nNext: Evaluate on downstream tasks (ASR, genre_gtzan)")


if __name__ == "__main__":
    main()
