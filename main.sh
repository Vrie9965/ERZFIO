#!/usr/bin/env bash
# =============================================================
# Single Frame Poster
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

# Load modules
source "scripts/helpers.sh"
source "scripts/process.sh"
source "scripts/post.sh"

# Step 1: Read current state
current_frame=$(get_state "current_frame")
total_posted=$(get_state "total_posted")

# Step 2: Check if episode is complete
if [ "$current_frame" -gt "$total_frame" ]; then
    log_info "Episode finished! Frame $current_frame is beyond total_frame $total_frame."
    exit 0
fi

# Step 3: Compute timestamp and caption
timestamp=$(calculate_timestamp "$current_frame" "$img_fps")
caption=$(render_caption "$caption_template" "$current_frame" "$timestamp")
frame_file="${frames_dir}/frame_${current_frame}.jpg"

log_info "Preparing Frame #$current_frame [Timestamp: $timestamp]"

# Step 4: Post the frame
if post_frame "$current_frame" "$frame_file" "$caption"; then
    # Step 5: Advance state on success
    next_frame=$((current_frame + 1))
    new_total_posted=$((total_posted + 1))
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    set_state "current_frame" "$next_frame"
    set_state "total_posted" "$new_total_posted"
    set_state "last_run_timestamp" "$now"

    log_info "State updated: Next frame is #$next_frame"
    exit 0
else
    log_error "Post failed for Frame #$current_frame. State was not updated."
    exit 1
fi
