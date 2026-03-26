#!/bin/bash
# Wait for music4all download to complete, then create calibration CSV.
# Run this in background: nohup bash scripts/wait_and_create_calibration.sh &

MUSIC_DIR="/data/projects/12004380/datasets/music4all-all/music4all_16khz_new"
TARGET_COUNT=69230
LOG="/data/projects/12004380/fabian/generalized_model_merging/logfiles/calibration_setup.log"
PROJECT="/data/projects/12004380/fabian/generalized_model_merging"

mkdir -p "$(dirname "$LOG")"

echo "[$(date)] Waiting for music4all download to reach $TARGET_COUNT files..." | tee -a "$LOG"

while true; do
    COUNT=$(find "$MUSIC_DIR" -type f 2>/dev/null | wc -l)
    echo "[$(date)] music4all files: $COUNT / $TARGET_COUNT" >> "$LOG"

    if [ "$COUNT" -ge "$TARGET_COUNT" ]; then
        echo "[$(date)] Download complete! $COUNT files found." | tee -a "$LOG"
        break
    fi

    # Check if rclone is still running
    if ! pgrep -f "rclone.*music4all" > /dev/null 2>&1; then
        echo "[$(date)] rclone process not found. Current count: $COUNT" | tee -a "$LOG"
        if [ "$COUNT" -gt 60000 ]; then
            echo "[$(date)] Close enough ($COUNT/$TARGET_COUNT), proceeding..." | tee -a "$LOG"
            break
        else
            echo "[$(date)] ERROR: Download seems incomplete. Exiting." | tee -a "$LOG"
            exit 1
        fi
    fi

    sleep 60
done

# Create calibration CSV
echo "[$(date)] Creating calibration CSV..." | tee -a "$LOG"
cd "$PROJECT"
python scripts/create_calibration_csv.py \
    --music_dir "$MUSIC_DIR" \
    --speech_dir /data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech/train-clean-100 \
    --output data/calibration_10k.csv \
    --n_music 5000 \
    --n_speech 5000 \
    --seed 42 2>&1 | tee -a "$LOG"

echo "[$(date)] Done!" | tee -a "$LOG"
