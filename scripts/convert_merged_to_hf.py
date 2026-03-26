"""
Convert merged GAEM+ checkpoints to HuggingFace format.

The merged checkpoints are raw state dicts. This script loads them into
a HuBERT model and saves as HuggingFace format so they can be loaded
by s3prl's hf_hubert upstream.

Usage:
    python scripts/convert_merged_to_hf.py

Creates directories under results/exp1_merge/hf_models/<method>/
that can be loaded with: HubertModel.from_pretrained(path)
"""

import sys
import torch
from pathlib import Path
from transformers import HubertModel, HubertConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGE_DIR = PROJECT_ROOT / "results" / "exp1_merge"
HF_DIR = MERGE_DIR / "hf_models"
HF_DIR.mkdir(parents=True, exist_ok=True)

# Methods to convert
METHODS = [
    "simple_avg",
    "perm_avg_05",
    "procrustes_avg_05",
    "procrustes_avg_07_03",
    "procrustes_avg_03_07",
]


def main():
    # Load reference model to get config
    print("Loading HuBERT config...")
    ref_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    config = ref_model.config

    for method in METHODS:
        ckpt_path = MERGE_DIR / f"merged_{method}.pt"
        if not ckpt_path.exists():
            print(f"  Skipping {method}: {ckpt_path} not found")
            continue

        print(f"\n=== Converting {method} ===")
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        sd = ckpt["state_dict"]

        # Create a fresh HuBERT model and load merged weights
        model = HubertModel(config)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"    {k}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"    {k}")

        # Save in HuggingFace format
        output_dir = HF_DIR / method
        model.save_pretrained(str(output_dir))
        print(f"  Saved to {output_dir}")

        # Verify it loads
        loaded = HubertModel.from_pretrained(str(output_dir))
        print(f"  Verified: loads successfully ({sum(p.numel() for p in loaded.parameters()):,} params)")

    print(f"\n=== All conversions complete ===")
    print(f"Models saved to: {HF_DIR}")
    print(f"\nTo evaluate with s3prl:")
    print(f"  python run_downstream.py -u hf_hubert -k <path_to_hf_model> -d <task> ...")


if __name__ == "__main__":
    main()
