"""
Experiment 0: Feature Extraction + Interference Analysis

Loads HuBERT and MERT, extracts features on calibration data,
computes STI, interference metrics, and alignment matrices.

Run inside enroot container with s3prl_old_cuda conda env.
"""

import json
import os
import sys
import time
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gaem.alignment.procrustes import procrustes_orthogonal, compute_alignment_error
from gaem.alignment.permutation import correlation_permutation, compute_permutation_cost
from gaem.evaluation.interference import compute_domain_interference, layerwise_interference
from gaem.evaluation.sti import layerwise_sti, compute_sti_normalized

OUTPUT_DIR = PROJECT_ROOT / "results" / "exp0_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# Step 1: Load models
# ============================================================================
def load_models():
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    print("\n=== Loading HuBERT ===")
    t0 = time.time()
    hubert_model = AutoModel.from_pretrained("facebook/hubert-base-ls960")
    hubert_model.eval().to(DEVICE)
    print(f"  Loaded in {time.time()-t0:.1f}s, params: {sum(p.numel() for p in hubert_model.parameters()):,}")

    print("\n=== Loading MERT ===")
    t0 = time.time()
    mert_model = AutoModel.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
    mert_model.eval().to(DEVICE)
    print(f"  Loaded in {time.time()-t0:.1f}s, params: {sum(p.numel() for p in mert_model.parameters()):,}")

    # Feature extractor (same for both at 16kHz)
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)

    return hubert_model, mert_model, hubert_processor, mert_processor


# ============================================================================
# Step 2: Extract features from calibration data
# ============================================================================
def load_audio(filepath, target_sr=16000, max_len_sec=10.0):
    """Load audio file, truncate to max_len_sec."""
    try:
        wav, sr = sf.read(filepath)
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)  # mono
        if sr != target_sr:
            # Simple resampling (for safety, though data should be 16kHz)
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        max_samples = int(max_len_sec * target_sr)
        wav = wav[:max_samples]
        return wav.astype(np.float32)
    except Exception as e:
        print(f"  Warning: Failed to load {filepath}: {e}")
        return None


@torch.no_grad()
def extract_features(model, processor, audio_files, batch_size=8, max_samples=1000):
    """Extract per-layer hidden states, return time-averaged features per layer."""
    layer_features = {i: [] for i in range(13)}  # 0=CNN output, 1-12=transformer layers
    n_processed = 0

    for i in range(0, min(len(audio_files), max_samples), batch_size):
        batch_files = audio_files[i:i+batch_size]
        wavs = []
        for f in batch_files:
            w = load_audio(f)
            if w is not None and len(w) > 1600:  # at least 100ms
                wavs.append(w)

        if not wavs:
            continue

        # Process
        inputs = processor(
            wavs, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(DEVICE)

        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        # hidden_states: tuple of (batch, time, 768) for each layer
        for layer_idx, hs in enumerate(outputs.hidden_states):
            # Time-average: [B, T, 768] -> [B, 768]
            avg = hs.mean(dim=1).cpu()
            layer_features[layer_idx].append(avg)

        n_processed += len(wavs)
        if n_processed % 100 == 0:
            print(f"    Processed {n_processed} / {min(len(audio_files), max_samples)}")

    # Concatenate
    for k in layer_features:
        if layer_features[k]:
            layer_features[k] = torch.cat(layer_features[k], dim=0)
        else:
            layer_features[k] = torch.zeros(0, 768)

    print(f"  Total: {n_processed} samples, {len(layer_features)} layers")
    return layer_features


# ============================================================================
# Step 3: Weight-space analysis (no data needed)
# ============================================================================
def weight_analysis(hubert_model, mert_model):
    """Compute STI and interference directly from model weights."""
    print("\n=== Weight-Space Analysis ===")

    hubert_sd = {k: v.cpu() for k, v in hubert_model.state_dict().items()}
    mert_sd = {k: v.cpu() for k, v in mert_model.state_dict().items()}

    # Find matching parameter names (both are HuBERT-architecture)
    # HuBERT uses encoder.layers.X.*, MERT may use similar or hubert.encoder.layers.X.*
    print("\n  HuBERT param names (first 10):")
    for i, k in enumerate(hubert_sd):
        if i < 10:
            print(f"    {k}: {hubert_sd[k].shape}")

    print("\n  MERT param names (first 10):")
    for i, k in enumerate(mert_sd):
        if i < 10:
            print(f"    {k}: {mert_sd[k].shape}")

    # Build task vectors relative to each other
    # Since they don't share a base, we compute: delta = MERT_param - HuBERT_param
    # for matching layers. We need to map MERT names to HuBERT names.
    common_params = {}
    for hk in hubert_sd:
        # Try direct match
        if hk in mert_sd and hubert_sd[hk].shape == mert_sd[hk].shape:
            common_params[hk] = (hubert_sd[hk], mert_sd[hk])
            continue
        # Try with hubert. prefix for MERT
        mk = "hubert." + hk
        if mk in mert_sd and hubert_sd[hk].shape == mert_sd[mk].shape:
            common_params[hk] = (hubert_sd[hk], mert_sd[mk])

    print(f"\n  Common parameters: {len(common_params)}")
    print(f"  2D weight matrices: {sum(1 for k,(h,m) in common_params.items() if h.dim()==2)}")

    # Compute task vector (MERT - HuBERT)
    tv_mert = {k: m - h for k, (h, m) in common_params.items()}

    # Domain interference (self-analysis: how different are the weights?)
    interference = {}
    flat_hubert = torch.cat([h.flatten() for k, (h, m) in common_params.items()])
    flat_mert = torch.cat([m.flatten() for k, (h, m) in common_params.items()])
    flat_diff = flat_mert - flat_hubert

    interference["weight_cosine"] = torch.nn.functional.cosine_similarity(
        flat_hubert.unsqueeze(0), flat_mert.unsqueeze(0)
    ).item()
    interference["diff_norm"] = torch.norm(flat_diff).item()
    interference["hubert_norm"] = torch.norm(flat_hubert).item()
    interference["mert_norm"] = torch.norm(flat_mert).item()
    interference["relative_diff"] = interference["diff_norm"] / interference["hubert_norm"]

    print(f"\n  Weight cosine similarity: {interference['weight_cosine']:.6f}")
    print(f"  Relative difference: {interference['relative_diff']:.4f}")

    # Layerwise STI
    # We need pairs of weight matrices for the same layer.
    # For STI, we treat HuBERT weights as "task 1" and MERT weights as "task 2"
    print("\n  Computing layerwise STI...")
    sti_results = {}
    for k, (h_param, m_param) in common_params.items():
        if h_param.dim() == 2 and min(h_param.shape) >= 4:
            try:
                sti = compute_sti_normalized([h_param, m_param])
                sti_results[k] = sti
            except Exception as e:
                pass

    # Sort by STI (highest interference first)
    sorted_sti = sorted(sti_results.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 highest-interference layers:")
    for k, v in sorted_sti[:10]:
        print(f"    {k}: STI_norm = {v:.6f}")

    print(f"\n  Top 10 lowest-interference layers:")
    for k, v in sorted_sti[-10:]:
        print(f"    {k}: STI_norm = {v:.6f}")

    return interference, sti_results, common_params


# ============================================================================
# Step 4: Feature-space alignment
# ============================================================================
def alignment_analysis(hubert_features, mert_features):
    """Compute Procrustes and permutation alignment from last-layer features."""
    print("\n=== Alignment Analysis ===")

    results = {}
    # Use last transformer layer (layer 12)
    feat_h = hubert_features[12]  # [N, 768]
    feat_m = mert_features[12]    # [N, 768]
    N = min(feat_h.shape[0], feat_m.shape[0])
    feat_h = feat_h[:N]
    feat_m = feat_m[:N]
    print(f"  Features shape: {feat_h.shape}")

    # Baseline: no alignment
    error_none = compute_alignment_error(feat_h, feat_m, torch.eye(768))
    results["error_no_alignment"] = error_none
    print(f"\n  No alignment error: {error_none:.6f}")

    # Procrustes
    print("  Computing Procrustes alignment...")
    t0 = time.time()
    O = procrustes_orthogonal(feat_h, feat_m)
    error_procrustes = compute_alignment_error(feat_h, feat_m, O)
    results["error_procrustes"] = error_procrustes
    results["procrustes_time"] = time.time() - t0
    print(f"  Procrustes error: {error_procrustes:.6f} (time: {results['procrustes_time']:.2f}s)")
    print(f"  Procrustes improvement: {(1 - error_procrustes/error_none)*100:.1f}%")

    # Verify O is orthogonal
    orth_check = torch.norm(O @ O.T - torch.eye(768)).item()
    results["procrustes_orthogonality_error"] = orth_check
    print(f"  Orthogonality check ||OO^T - I||: {orth_check:.2e}")

    # Permutation alignment
    print("  Computing permutation alignment...")
    t0 = time.time()
    P = correlation_permutation(feat_h, feat_m)
    error_perm = compute_permutation_cost(feat_h, feat_m, P)
    results["error_permutation"] = error_perm
    results["permutation_time"] = time.time() - t0
    print(f"  Permutation error: {error_perm:.6f} (time: {results['permutation_time']:.2f}s)")
    print(f"  Permutation improvement: {(1 - error_perm/error_none)*100:.1f}%")

    # Per-layer alignment analysis
    print("\n  Per-layer alignment errors:")
    layer_alignment = {}
    for layer_idx in range(13):
        fh = hubert_features[layer_idx][:N]
        fm = mert_features[layer_idx][:N]
        e_none = compute_alignment_error(fh, fm, torch.eye(fh.shape[1]))
        O_l = procrustes_orthogonal(fh, fm)
        e_proc = compute_alignment_error(fh, fm, O_l)
        layer_alignment[layer_idx] = {
            "error_none": e_none,
            "error_procrustes": e_proc,
            "improvement_pct": (1 - e_proc / e_none) * 100 if e_none > 0 else 0,
        }
        print(f"    Layer {layer_idx:2d}: none={e_none:.4f} → procrustes={e_proc:.4f} ({layer_alignment[layer_idx]['improvement_pct']:.1f}% improvement)")

    results["per_layer"] = layer_alignment

    # Save alignment matrices
    np.save(str(OUTPUT_DIR / "procrustes_O.npy"), O.numpy())
    np.save(str(OUTPUT_DIR / "permutation_P.npy"), P.numpy())
    print(f"\n  Saved alignment matrices to {OUTPUT_DIR}")

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("GAEM+ Experiment 0: Interference & Alignment Analysis")
    print("=" * 60)

    # Load models
    hubert_model, mert_model, hubert_proc, mert_proc = load_models()

    # Weight-space analysis (no data needed)
    interference, sti_results, common_params = weight_analysis(hubert_model, mert_model)

    # Save weight analysis results
    with open(OUTPUT_DIR / "weight_interference.json", "w") as f:
        json.dump(interference, f, indent=2)
    with open(OUTPUT_DIR / "layerwise_sti.json", "w") as f:
        json.dump({k: float(v) for k, v in sti_results.items()}, f, indent=2)
    print(f"\nSaved weight analysis to {OUTPUT_DIR}")

    # Load calibration data
    import csv
    cal_csv = PROJECT_ROOT / "data" / "calibration_10k.csv"
    print(f"\n=== Loading calibration data from {cal_csv} ===")
    speech_files, music_files = [], []
    with open(cal_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["domain"] == "speech":
                speech_files.append(row["file_path"])
            else:
                music_files.append(row["file_path"])
    print(f"  Speech files: {len(speech_files)}")
    print(f"  Music files: {len(music_files)}")

    # Use a subset for feature extraction (speed)
    all_files = speech_files[:500] + music_files[:500]  # 1000 total
    print(f"  Using {len(all_files)} files for feature extraction")

    # Extract features
    print("\n=== Extracting HuBERT features ===")
    hubert_features = extract_features(hubert_model, hubert_proc, all_files, batch_size=16, max_samples=1000)

    print("\n=== Extracting MERT features ===")
    mert_features = extract_features(mert_model, mert_proc, all_files, batch_size=16, max_samples=1000)

    # Feature-space alignment analysis
    alignment_results = alignment_analysis(hubert_features, mert_features)

    # Save alignment results
    # Convert per_layer dict for JSON serialization
    alignment_json = {k: v for k, v in alignment_results.items() if k != "per_layer"}
    alignment_json["per_layer"] = {
        str(k): v for k, v in alignment_results["per_layer"].items()
    }
    with open(OUTPUT_DIR / "alignment_results.json", "w") as f:
        json.dump(alignment_json, f, indent=2)

    # Save per-layer features for later use
    for layer_idx in [0, 6, 12]:  # Save selected layers
        np.save(str(OUTPUT_DIR / f"hubert_features_layer{layer_idx}.npy"),
                hubert_features[layer_idx].numpy())
        np.save(str(OUTPUT_DIR / f"mert_features_layer{layer_idx}.npy"),
                mert_features[layer_idx].numpy())

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 0 COMPLETE")
    print("=" * 60)
    print(f"\nWeight-space:")
    print(f"  Cosine similarity: {interference['weight_cosine']:.6f}")
    print(f"  Relative difference: {interference['relative_diff']:.4f}")
    print(f"\nAlignment (last layer):")
    print(f"  No alignment error:  {alignment_results['error_no_alignment']:.6f}")
    print(f"  Permutation error:   {alignment_results['error_permutation']:.6f}")
    print(f"  Procrustes error:    {alignment_results['error_procrustes']:.6f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Files: {list(OUTPUT_DIR.glob('*'))}")


if __name__ == "__main__":
    main()
