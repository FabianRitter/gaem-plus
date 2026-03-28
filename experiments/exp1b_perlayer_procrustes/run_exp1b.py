"""
Experiment 1b: Per-Layer Procrustes Alignment + Merged Encoder Creation

Computes separate O_l for each of the 13 layers (post-CNN + 12 transformer)
using layer-specific features on the calibration dataset, then creates
merged HuBERT+MERT checkpoints at 0.9/0.1 and 0.7/0.3 weights.

Run inside enroot container.
"""

import json
import os
import sys
import time
import copy
import csv
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from transformers import AutoModel, HubertModel, Wav2Vec2FeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gaem.alignment.procrustes import procrustes_orthogonal, compute_alignment_error
from gaem.alignment.per_layer_procrustes import (
    compute_per_layer_alignment,
    align_state_dict_per_layer,
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "exp1b_perlayer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def load_audio(filepath, target_sr=16000, max_len_sec=10.0):
    try:
        wav, sr = sf.read(filepath)
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        max_samples = int(max_len_sec * target_sr)
        wav = wav[:max_samples]
        return wav.astype(np.float32)
    except Exception as e:
        return None


@torch.no_grad()
def extract_all_layer_features(model, processor, audio_files, batch_size=16, max_samples=1000):
    """Extract time-averaged features from all hidden state layers."""
    model.eval().to(DEVICE)
    layer_features = {}
    n_processed = 0

    for i in range(0, min(len(audio_files), max_samples), batch_size):
        batch_files = audio_files[i:i+batch_size]
        wavs = [load_audio(f) for f in batch_files]
        wavs = [w for w in wavs if w is not None and len(w) > 1600]
        if not wavs:
            continue

        inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True).to(DEVICE)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        for layer_idx, hs in enumerate(outputs.hidden_states):
            avg = hs.mean(dim=1).cpu()
            if layer_idx not in layer_features:
                layer_features[layer_idx] = []
            layer_features[layer_idx].append(avg)

        n_processed += len(wavs)
        if n_processed % 200 == 0:
            print(f"    {n_processed} / {min(len(audio_files), max_samples)} samples")

    for k in layer_features:
        layer_features[k] = torch.cat(layer_features[k], dim=0)[:max_samples]

    print(f"  Extracted {n_processed} samples across {len(layer_features)} layers")
    return layer_features


def main():
    print("=" * 60)
    print("Exp 1b: Per-Layer Procrustes Alignment")
    print("=" * 60)

    # Load calibration data
    cal_csv = PROJECT_ROOT / "data" / "calibration_10k.csv"
    print(f"\nLoading calibration data from {cal_csv}")
    all_files = []
    with open(cal_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_files.append(row["file_path"])
    # Use 1000 samples (500 speech + 500 music interleaved from the shuffled CSV)
    all_files = all_files[:1000]
    print(f"  Using {len(all_files)} files")

    # Load models
    print("\n=== Loading HuBERT ===")
    hubert = AutoModel.from_pretrained("facebook/hubert-base-ls960")
    hubert_proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

    print("=== Loading MERT ===")
    mert = AutoModel.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
    mert_proc = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)

    # Extract per-layer features
    print("\n=== Extracting HuBERT per-layer features ===")
    hubert_features = extract_all_layer_features(hubert, hubert_proc, all_files, max_samples=1000)

    print("\n=== Extracting MERT per-layer features ===")
    mert_features = extract_all_layer_features(mert, mert_proc, all_files, max_samples=1000)

    # Compute per-layer alignment
    print("\n=== Computing per-layer Procrustes alignment ===")
    layer_alignments = compute_per_layer_alignment(hubert_features, mert_features, num_layers=12)

    # Report per-layer alignment quality
    print("\n  Per-layer alignment errors:")
    alignment_report = {}
    for layer_idx in sorted(layer_alignments.keys()):
        O_l = layer_alignments[layer_idx]
        N = min(hubert_features[layer_idx].shape[0], mert_features[layer_idx].shape[0])
        fh = hubert_features[layer_idx][:N]
        fm = mert_features[layer_idx][:N]
        e_before = compute_alignment_error(fh, fm, torch.eye(768))
        e_after = compute_alignment_error(fh, fm, O_l)
        improvement = (1 - e_after / e_before) * 100
        alignment_report[layer_idx] = {
            "error_before": e_before,
            "error_after": e_after,
            "improvement_pct": improvement,
        }
        print(f"    Layer {layer_idx:2d}: {e_before:.4f} → {e_after:.4f} ({improvement:.1f}% improvement)")

    # Save alignment matrices
    alignments_dir = OUTPUT_DIR / "alignments"
    alignments_dir.mkdir(exist_ok=True)
    for layer_idx, O_l in layer_alignments.items():
        np.save(str(alignments_dir / f"O_layer{layer_idx}.npy"), O_l.numpy())
    with open(OUTPUT_DIR / "alignment_report.json", "w") as f:
        json.dump({str(k): v for k, v in alignment_report.items()}, f, indent=2)
    print(f"\n  Saved {len(layer_alignments)} alignment matrices to {alignments_dir}")

    # Create merged models
    print("\n=== Creating per-layer Procrustes merged models ===")
    hubert_sd = {k: v.cpu() for k, v in hubert.state_dict().items()}
    mert_sd = {k: v.cpu() for k, v in mert.state_dict().items()}

    mert_aligned = align_state_dict_per_layer(
        hubert_sd, mert_sd, layer_alignments,
        encoder_prefix="encoder.layers",
        num_heads=12,
    )

    hf_dir = OUTPUT_DIR / "hf_models"

    for alpha_h, alpha_m, name in [
        (0.9, 0.1, "perlayer_procrustes_09_01"),
        (0.7, 0.3, "perlayer_procrustes_07_03"),
    ]:
        print(f"\n  Creating {name} (HuBERT={alpha_h}, MERT={alpha_m})")
        merged = {}
        for k in hubert_sd:
            if k in mert_aligned:
                merged[k] = alpha_h * hubert_sd[k] + alpha_m * mert_aligned[k]
            else:
                merged[k] = hubert_sd[k].clone()

        # Save as HuggingFace model
        out_dir = hf_dir / name
        model_copy = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        missing, unexpected = model_copy.load_state_dict(merged, strict=False)
        if missing:
            print(f"    Missing: {len(missing)} keys")
        model_copy.save_pretrained(str(out_dir))
        print(f"    Saved to {out_dir}")

        # Verify
        loaded = HubertModel.from_pretrained(str(out_dir))
        print(f"    Verified: {sum(p.numel() for p in loaded.parameters()):,} params")

    # Compare alignment quality: global vs per-layer
    print("\n=== Global vs Per-Layer Comparison ===")
    global_O = torch.from_numpy(
        np.load(str(PROJECT_ROOT / "results" / "exp0_analysis" / "procrustes_O.npy"))
    ).float()

    for layer_idx in [0, 3, 6, 9, 12]:
        if layer_idx not in layer_alignments:
            continue
        N = min(hubert_features[layer_idx].shape[0], mert_features[layer_idx].shape[0])
        fh = hubert_features[layer_idx][:N]
        fm = mert_features[layer_idx][:N]
        e_global = compute_alignment_error(fh, fm, global_O)
        e_perlayer = compute_alignment_error(fh, fm, layer_alignments[layer_idx])
        print(f"  Layer {layer_idx:2d}: global={e_global:.4f}, per-layer={e_perlayer:.4f}, delta={e_global-e_perlayer:.4f}")

    print("\n" + "=" * 60)
    print("EXP 1B COMPLETE")
    print("=" * 60)
    print(f"Models saved to: {hf_dir}")
    print("Next: submit downstream evaluation")


if __name__ == "__main__":
    main()
