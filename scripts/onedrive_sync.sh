#!/bin/bash
# Sync GAEM+ research files between local and OneDrive.
#
# Usage:
#   bash scripts/onedrive_sync.sh push          # Push local → OneDrive
#   bash scripts/onedrive_sync.sh pull          # Pull OneDrive → local
#   bash scripts/onedrive_sync.sh push-excel    # Push only the Excel file
#   bash scripts/onedrive_sync.sh pull-readings # Pull only paper-readings/
#   bash scripts/onedrive_sync.sh status        # Show what differs

LOCAL="/data/projects/12004380/fabian/generalized_model_merging"
REMOTE="onedrive:Proposal-PhD/MERGING SPEECH ENCODERS RESEARCH/Unified Model Merging Last PhD Work 2026"

ACTION="${1:-status}"

case "$ACTION" in
    push)
        echo "=== Pushing Excel ==="
        rclone copy "$LOCAL/GAEM_Research_Matrices.xlsx" "$REMOTE/" -v
        echo "=== Pushing paper-readings/ ==="
        rclone sync "$LOCAL/paper-readings/" "$REMOTE/paper-readings/" -v
        echo "=== Pushing CLAUDE.md ==="
        rclone copy "$LOCAL/CLAUDE.md" "$REMOTE/" -v
        echo "=== Done ==="
        ;;

    pull)
        echo "=== Pulling Excel ==="
        rclone copy "$REMOTE/GAEM_Research_Matrices.xlsx" "$LOCAL/" -v
        echo "=== Pulling paper-readings/ ==="
        rclone sync "$REMOTE/paper-readings/" "$LOCAL/paper-readings/" -v
        echo "=== Done ==="
        ;;

    push-excel)
        echo "=== Pushing Excel ==="
        rclone copy "$LOCAL/GAEM_Research_Matrices.xlsx" "$REMOTE/" -v
        echo "=== Done ==="
        ;;

    pull-readings)
        echo "=== Pulling paper-readings/ ==="
        rclone sync "$REMOTE/paper-readings/" "$LOCAL/paper-readings/" -v
        echo "=== Done ==="
        ;;

    push-readings)
        echo "=== Pushing paper-readings/ ==="
        rclone sync "$LOCAL/paper-readings/" "$REMOTE/paper-readings/" -v
        echo "=== Done ==="
        ;;

    status)
        echo "=== Checking differences ==="
        echo ""
        echo "--- Excel ---"
        rclone check "$LOCAL/GAEM_Research_Matrices.xlsx" "$REMOTE/GAEM_Research_Matrices.xlsx" --one-way 2>&1 | tail -3
        echo ""
        echo "--- paper-readings/ ---"
        rclone check "$LOCAL/paper-readings/" "$REMOTE/paper-readings/" 2>&1 | tail -5
        echo ""
        echo "Local paper-readings:"
        find "$LOCAL/paper-readings/" -name "*.md" -not -name "_template.md" | wc -l
        echo "Remote paper-readings:"
        rclone ls "$REMOTE/paper-readings/" 2>/dev/null | grep ".md" | wc -l
        ;;

    *)
        echo "Usage: $0 {push|pull|push-excel|pull-readings|push-readings|status}"
        exit 1
        ;;
esac
