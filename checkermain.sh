#!/usr/bin/env bash
# =============================================================
# Pre-Run Health and Configuration Checker
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

# Load helper functions
source "scripts/helpers.sh"

log_info "Starting pre-flight sanity checks..."

# Check 1: Tool dependencies
check_dependencies

# Check 2: State file presence
if [ ! -f "fb/state.json" ]; then
    log_info "fb/state.json not found. Initializing new state file..."
    mkdir -p fb
    echo '{"current_frame": 1, "total_posted": 0, "last_run_timestamp": ""}' > fb/state.json
fi

# Check 3: Current frame status
current_frame=$(get_state "current_frame")
total_posted=$(get_state "total_posted")

log_info "Current frame: $current_frame | Total frames: $total_frame | Total posted so far: $total_posted"

if [ "$current_frame" -gt "$total_frame" ]; then
    log_info "All frames for S${season}E${episode} have already been posted! Nothing to do."
    exit 0
fi

# Check 4: Check if the next frame image exists
frame_file="${frames_dir}/frame_${current_frame}.jpg"
if [ ! -f "$frame_file" ]; then
    log_error "Frame image file missing: $frame_file"
    exit 1
fi
log_info "Found frame file: $frame_file"

# Check 5: Token check (in live mode or if token is present during dry run)
if [ "${DRY_RUN:-0}" = "1" ]; then
    log_info "[DRY RUN MODE ENABLED] Live Facebook posting is disabled."
    if [ -n "${FB_PAGE_TOKEN:-}" ]; then
        check_token
    else
        log_info "FB_PAGE_TOKEN not provided, skipping token verification in dry-run."
    fi
else
    check_token
fi

log_info "Pre-flight checks completed successfully!"

# Output clean summary if running in GitHub Actions
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
        echo "## ERZFIO Health Check Report"
        echo "- **Status:** All checks PASSED"
        echo "- **Current Frame:** #$current_frame of $total_frame"
        echo "- **Frame File:** \`$frame_file\` (Ready to post)"
        echo "- **Facebook Token:** Verified and connected"
    } >> "$GITHUB_STEP_SUMMARY"
fi
