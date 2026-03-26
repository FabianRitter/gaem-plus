"""
Create calibration dataset CSV for GAEM+ alignment feature extraction.

Combines:
- 5000 random audio files from music4all_16khz_new (music domain)
- 5000 random audio files from LibriSpeech train-clean-100 (speech domain)

Output: A CSV with columns (index, file_path, length, label, domain)
matching the format used in s3prl's len_for_bucket CSVs.

Usage:
    python scripts/create_calibration_csv.py \
        --music_dir /data/projects/12004380/datasets/music4all-all/music4all_16khz_new \
        --speech_dir /data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech/train-clean-100 \
        --output data/calibration_10k.csv \
        --n_music 5000 \
        --n_speech 5000 \
        --seed 42
"""

import argparse
import csv
import os
import random
import subprocess
from pathlib import Path


def find_audio_files(directory, extensions=(".flac", ".wav", ".mp3")):
    """Recursively find all audio files in a directory."""
    files = []
    for ext in extensions:
        for f in Path(directory).rglob(f"*{ext}"):
            files.append(str(f))
    return sorted(files)


def get_audio_length_samples(filepath, sr=16000):
    """
    Get audio length in samples. Uses soxi if available, falls back to
    reading file headers, or estimates from file size.
    """
    try:
        # Try soundfile first (fast, no full read)
        import soundfile as sf
        info = sf.info(filepath)
        return int(info.frames)
    except Exception:
        pass

    try:
        # Fallback: soxi
        result = subprocess.run(
            ["soxi", "-s", filepath],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass

    # Last resort: estimate from file size (very rough for compressed formats)
    # For 16kHz 16-bit mono: ~32000 bytes/sec
    size = os.path.getsize(filepath)
    return int(size / 2)  # rough estimate for 16-bit audio


def main():
    parser = argparse.ArgumentParser(description="Create calibration CSV for GAEM+")
    parser.add_argument("--music_dir", type=str,
                        default="/data/projects/12004380/datasets/music4all-all/music4all_16khz_new",
                        help="Path to music4all_16khz_new directory")
    parser.add_argument("--speech_dir", type=str,
                        default="/data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech/train-clean-100",
                        help="Path to LibriSpeech train-clean-100")
    parser.add_argument("--output", type=str,
                        default="data/calibration_10k.csv",
                        help="Output CSV path")
    parser.add_argument("--n_music", type=int, default=5000,
                        help="Number of music samples")
    parser.add_argument("--n_speech", type=int, default=5000,
                        help="Number of speech samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--compute_lengths", action="store_true",
                        help="Compute audio lengths (slower but accurate)")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Find music files
    print(f"Scanning music directory: {args.music_dir}")
    music_files = find_audio_files(args.music_dir)
    print(f"  Found {len(music_files)} music files")

    if len(music_files) < args.n_music:
        print(f"  WARNING: Only {len(music_files)} files available, using all")
        music_sample = music_files
    else:
        music_sample = random.sample(music_files, args.n_music)
    print(f"  Selected {len(music_sample)} music samples")

    # Find speech files
    print(f"Scanning speech directory: {args.speech_dir}")
    speech_files = find_audio_files(args.speech_dir)
    print(f"  Found {len(speech_files)} speech files")

    if len(speech_files) < args.n_speech:
        print(f"  WARNING: Only {len(speech_files)} files available, using all")
        speech_sample = speech_files
    else:
        speech_sample = random.sample(speech_files, args.n_speech)
    print(f"  Selected {len(speech_sample)} speech samples")

    # Build rows
    rows = []
    idx = 0

    for filepath in music_sample:
        length = get_audio_length_samples(filepath) if args.compute_lengths else 0
        rows.append({
            "index": idx,
            "file_path": filepath,
            "length": length,
            "label": "",
            "domain": "music",
        })
        idx += 1

    for filepath in speech_sample:
        length = get_audio_length_samples(filepath) if args.compute_lengths else 0
        rows.append({
            "index": idx,
            "file_path": filepath,
            "length": length,
            "label": "",
            "domain": "speech",
        })
        idx += 1

    # Shuffle so speech and music are interleaved
    random.shuffle(rows)

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "file_path", "length", "label", "domain"])
        writer.writeheader()
        for i, row in enumerate(rows):
            row["index"] = i
            writer.writerow(row)

    print(f"\nWritten {len(rows)} rows to {args.output}")
    n_music_written = sum(1 for r in rows if r["domain"] == "music")
    n_speech_written = sum(1 for r in rows if r["domain"] == "speech")
    print(f"  Music: {n_music_written}")
    print(f"  Speech: {n_speech_written}")

    # Also create domain-specific CSVs for convenience
    music_output = args.output.replace(".csv", "_music.csv")
    speech_output = args.output.replace(".csv", "_speech.csv")

    with open(music_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "file_path", "length", "label", "domain"])
        writer.writeheader()
        for i, row in enumerate(r for r in rows if r["domain"] == "music"):
            row["index"] = i
            writer.writerow(row)
    print(f"  Music-only CSV: {music_output}")

    with open(speech_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "file_path", "length", "label", "domain"])
        writer.writeheader()
        for i, row in enumerate(r for r in rows if r["domain"] == "speech"):
            row["index"] = i
            writer.writerow(row)
    print(f"  Speech-only CSV: {speech_output}")


if __name__ == "__main__":
    main()
