#!/usr/bin/env bash
# =============================================================
# Batch Poster Loop (Runs on Cron Schedule)
# =============================================================

set -euo pipefail

# Load config
if [ -f "config.conf" ]; then
    source "config.conf"
elif [ -f "config.conf.example" ]; then
    source "config.conf.example"
else
    echo "[ERROR] No config.conf found!" >&2
    exit 1
fi

source "scripts/helpers.sh"

log_info "=== Starting ERZFIO Scheduled Run ==="

# Step 1: Run pre-flight checks
bash checkermain.sh

# Allow workflow dispatch to override frames per run if requested
frames_to_post="${FPH_OVERRIDE:-$fph}"
sleep_seconds=$((mins * 60))

# Speed up delay during dry-run testing
if [ "${DRY_RUN:-0}" = "1" ]; then
    sleep_seconds=1
fi

log_info "Batch target: $frames_to_post frame(s) with ${mins}m delay between posts."

# Step 2: Loop through batch
posted_count=0
for ((i=1; i<=frames_to_post; i++)); do
    current_frame=$(get_state "current_frame")

    if [ "$current_frame" -gt "$total_frame" ]; then
        log_info "Reached end of episode at frame $current_frame! Ending batch."
        break
    fi

    log_info "Batch progress: posting item $i of $frames_to_post..."

    # Run single-frame poster
    if bash main.sh; then
        posted_count=$((posted_count + 1))
    else
        log_error "Batch aborted due to error on Frame #$current_frame"
        exit 1
    fi

    # Sleep between posts (unless it is the final frame in the batch)
    if [ "$i" -lt "$frames_to_post" ]; then
        log_info "Waiting $sleep_seconds second(s) before next frame..."
        sleep "$sleep_seconds"
    fi
done

log_info "=== Batch finished! Posted $posted_count frame(s) in this run. ==="
