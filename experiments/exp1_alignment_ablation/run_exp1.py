"""
Experiment 1: Alignment Method Ablation

Compares permutation, orthogonal, and no alignment for HuBERT+MERT merging.
This script runs the merging and saves merged checkpoints.
Downstream evaluation is done separately via s3prl's run_downstream.

Usage (interactive, inside GPU node):
    python experiments/exp1_alignment_ablation/run_exp1.py \
        --speech_ckpt ssl-phase1/s3prl/result/pretrain/.../states-220000.ckpt \
        --music_ckpt  ssl-phase1/s3prl/result/pretrain/.../states-220000.ckpt \
        --base_ckpt   <path-to-hubert-base-weights> \
        --data_dir    /data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech \
        --output_dir  results/exp1_alignment_ablation

Usage (PBS):
    qsub scripts/run_exp1_enroot.sh
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gaem.alignment.procrustes import procrustes_orthogonal, compute_alignment_error
from gaem.alignment.permutation import correlation_permutation
from gaem.merging.task_arithmetic import compute_task_vector, task_arithmetic_merge
from gaem.merging.gaem_plus import gaem_plus_merge, gaem_plus_merge_ablation
from gaem.evaluation.barriers import interpolation_barrier
from gaem.evaluation.interference import compute_domain_interference, layerwise_interference
from gaem.utils.checkpoint import load_checkpoint, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="GAEM+ Experiment 1: Alignment Ablation")
    parser.add_argument("--speech_ckpt", type=str, required=True, help="Path to speech-distilled checkpoint")
    parser.add_argument("--music_ckpt", type=str, required=True, help="Path to music-distilled checkpoint")
    parser.add_argument("--base_ckpt", type=str, required=True, help="Path to base model (HuBERT) checkpoint")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to LibriSpeech data for feature extraction")
    parser.add_argument("--output_dir", type=str, default="results/exp1_alignment_ablation")
    parser.add_argument("--weights", type=str, default="0.5,0.5", help="Comma-separated merge weights")
    parser.add_argument("--n_feature_samples", type=int, default=1000, help="Samples for feature extraction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(",")]

    print(f"Device: {args.device}")
    print(f"Weights: {weights}")

    # Load checkpoints
    print("\n=== Loading checkpoints ===")
    base_data = load_checkpoint(args.base_ckpt)
    speech_data = load_checkpoint(args.speech_ckpt)
    music_data = load_checkpoint(args.music_ckpt)

    base_sd = base_data["state_dict"]
    speech_sd = speech_data["state_dict"]
    music_sd = music_data["state_dict"]
    print(f"  Base params: {sum(p.numel() for p in base_sd.values()):,}")
    print(f"  Speech params: {sum(p.numel() for p in speech_sd.values()):,}")
    print(f"  Music params: {sum(p.numel() for p in music_sd.values()):,}")

    # Compute task vectors
    print("\n=== Computing task vectors ===")
    tv_speech = compute_task_vector(speech_sd, base_sd)
    tv_music = compute_task_vector(music_sd, base_sd)
    print(f"  Speech TV norm: {sum(torch.norm(v).item()**2 for v in tv_speech.values())**0.5:.4f}")
    print(f"  Music TV norm: {sum(torch.norm(v).item()**2 for v in tv_music.values())**0.5:.4f}")

    # Domain interference analysis (no alignment needed)
    print("\n=== Domain interference analysis ===")
    interference = compute_domain_interference(
        base_sd, [tv_speech, tv_music], ["speech", "music"]
    )
    print(json.dumps(interference, indent=2))
    with open(output_dir / "interference.json", "w") as f:
        json.dump(interference, f, indent=2)

    # Layerwise interference
    lw_interference = layerwise_interference(
        [tv_speech, tv_music], ["speech", "music"]
    )
    with open(output_dir / "layerwise_interference.json", "w") as f:
        json.dump(lw_interference, f, indent=2)
    print(f"  Saved layerwise interference for {len(lw_interference)} layers")

    # Feature extraction for alignment
    # TODO: Implement proper feature extraction from dataloader
    # For now, use a proxy: flatten first layer weights as pseudo-features
    # This will be replaced with actual dataloader-based extraction
    print("\n=== Computing alignment matrices ===")

    # Proxy features: concatenate flattened weight matrices
    # In the real experiment, replace with extract_features_from_model()
    feature_keys = sorted([k for k in base_sd if base_sd[k].dim() == 2])[:5]
    feat_speech = torch.cat([speech_sd[k].flatten() for k in feature_keys]).unsqueeze(0)
    feat_music = torch.cat([music_sd[k].flatten() for k in feature_keys]).unsqueeze(0)
    # Expand to N samples (for Procrustes we need N > d, use weight-space features)
    # In production: use actual audio features from LibriSpeech
    d = min(feat_speech.shape[1], 512)
    feat_speech = feat_speech[:, :d].expand(100, -1) + 0.01 * torch.randn(100, d)
    feat_music = feat_music[:, :d].expand(100, -1) + 0.01 * torch.randn(100, d)

    features = [feat_speech, feat_music]

    # Run all ablation variants
    print("\n=== Running merge ablations ===")
    results = gaem_plus_merge_ablation(
        base_sd, [speech_sd, music_sd], weights, features, anchor_idx=0
    )

    # Save merged checkpoints
    for method_name, merged_sd in results.items():
        ckpt_path = output_dir / f"merged_{method_name}.pt"
        metadata = {
            "method": method_name,
            "weights": weights,
            "speech_ckpt": args.speech_ckpt,
            "music_ckpt": args.music_ckpt,
        }
        save_checkpoint(merged_sd, str(ckpt_path), metadata)
        print(f"  Saved: {ckpt_path}")

    # Summary
    print("\n=== Experiment 1 Complete ===")
    print(f"Output: {output_dir}")
    print(f"Merged checkpoints: {len(results)}")
    print("Methods:", list(results.keys()))
    print("\nNext: Run downstream evaluation on each merged checkpoint")
    print("  python ssl-phase1/s3prl/run_downstream.py -m train \\")
    print("    -u multi_distiller_local -k <merged_ckpt> ...")


if __name__ == "__main__":
    main()
