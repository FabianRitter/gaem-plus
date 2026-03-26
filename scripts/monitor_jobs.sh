#!/bin/bash
# Monitor PBS job status and log transitions
# Used as a cron-checked script to detect when hold_node starts running.

LOG="/data/projects/12004380/fabian/generalized_model_merging/logfiles/job_monitor.log"
STATE_FILE="/data/projects/12004380/fabian/generalized_model_merging/logfiles/.job_state"

mkdir -p "$(dirname "$LOG")"
touch "$STATE_FILE"

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Get hold_node job status
HOLD_STATUS=$(qstat -u "$USER" 2>/dev/null | grep "hold_node" | awk '{print $10}')
HOLD_JOBID=$(qstat -u "$USER" 2>/dev/null | grep "hold_node" | awk '{print $1}')

# Get previous state
PREV_STATUS=$(cat "$STATE_FILE" 2>/dev/null)

if [ -z "$HOLD_STATUS" ]; then
    echo "[$TIMESTAMP] hold_node: no job found" >> "$LOG"
elif [ "$HOLD_STATUS" != "$PREV_STATUS" ]; then
    echo "[$TIMESTAMP] hold_node ($HOLD_JOBID): $PREV_STATUS -> $HOLD_STATUS" >> "$LOG"
    echo "$HOLD_STATUS" > "$STATE_FILE"

    if [ "$HOLD_STATUS" = "R" ]; then
        NODE=$(qstat -f "$HOLD_JOBID" 2>/dev/null | grep exec_host | awk -F= '{print $2}' | awk -F/ '{print $1}' | tr -d ' ')
        echo "[$TIMESTAMP] *** RUNNING on node: $NODE ***" >> "$LOG"
        echo "[$TIMESTAMP] SSH command: ssh $NODE" >> "$LOG"
        echo ""
        echo "============================================"
        echo "hold_node is RUNNING on node: $NODE"
        echo "  Job ID: $HOLD_JOBID"
        echo "  SSH:    ssh $NODE"
        echo "============================================"
    fi
else
    # Same state — log only every 10 minutes to keep log clean
    MINUTE=$(date "+%M")
    if [ $((MINUTE % 10)) -eq 0 ]; then
        echo "[$TIMESTAMP] hold_node ($HOLD_JOBID): still $HOLD_STATUS" >> "$LOG"
    fi
fi

# Also check music4all download progress
MUSIC_COUNT=$(ls /data/projects/12004380/datasets/music4all-all/music4all_16khz_new/ 2>/dev/null | wc -l)
if [ "$MUSIC_COUNT" -gt 0 ] && [ "$MUSIC_COUNT" -lt 69230 ]; then
    echo "[$TIMESTAMP] music4all download: $MUSIC_COUNT / 69230 files" >> "$LOG"
elif [ "$MUSIC_COUNT" -ge 69230 ]; then
    echo "[$TIMESTAMP] music4all download: COMPLETE ($MUSIC_COUNT files)" >> "$LOG"
fi
