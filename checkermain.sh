#!/usr/bin/env bash
# =============================================================
# Pre-Run Health and Configuration Checker
# =============================================================

set -uo pipefail

CHECK_RESULTS=()
CHECK_FAILURES=0

log_info() { echo "[INFO] $*"; }
log_warn() { echo "[WARN] $*" >&2; }
log_error() { echo "[ERROR] $*" >&2; }

record_check() {
    local name="$1"
    local status="$2"
    local detail="$3"

    CHECK_RESULTS+=("| $name | $status | $detail |")
    if [ "$status" = "PASS" ]; then
        log_info "$name: $detail"
    else
        log_error "$name: $detail"
        CHECK_FAILURES=$((CHECK_FAILURES + 1))
    fi
}

is_positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && (( 10#$1 > 0 ))
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && (( 10#$1 >= 0 ))
}

config_file=""
if [ -f "config.conf" ]; then
    config_file="config.conf"
elif [ -f "config.conf.example" ]; then
    config_file="config.conf.example"
fi

if [ -z "$config_file" ]; then
    record_check "Config file" "FAIL" "Neither config.conf nor config.conf.example was found."
else
    if source "$config_file"; then
        record_check "Config file" "PASS" "Loaded $config_file."
    else
        record_check "Config file" "FAIL" "Could not load $config_file."
    fi
fi

if [ -f "scripts/helpers.sh" ]; then
    source "scripts/helpers.sh"
    set +e
else
    record_check "Helper scripts" "FAIL" "scripts/helpers.sh was not found."
fi

log_info "Starting repository integrity checks..."

missing_dependencies=()
for cmd in curl jq bc; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        missing_dependencies+=("$cmd")
    fi
done
if [ "${#missing_dependencies[@]}" -eq 0 ]; then
    record_check "Dependencies" "PASS" "curl, jq, and bc are available."
else
    record_check "Dependencies" "FAIL" "Missing: ${missing_dependencies[*]}."
fi

if [ -n "${anime_name:-}" ]; then
    record_check "anime_name" "PASS" "A name is configured."
else
    record_check "anime_name" "FAIL" "Must not be empty."
fi

if is_positive_integer "${season:-}"; then
    record_check "season" "PASS" "Configured as a positive integer."
else
    record_check "season" "FAIL" "Must be a positive integer."
fi

if is_positive_integer "${episode:-}"; then
    record_check "episode" "PASS" "Configured as a positive integer."
else
    record_check "episode" "FAIL" "Must be a positive integer."
fi

if is_positive_integer "${total_frame:-}"; then
    record_check "total_frame" "PASS" "Configured as a positive integer."
else
    record_check "total_frame" "FAIL" "Must be a positive integer."
fi

if [[ "${img_fps:-}" =~ ^[0-9]+([.][0-9]+)?$ ]] &&
    [ "$(echo "${img_fps} > 0" | bc -l 2>/dev/null)" = "1" ]; then
    record_check "img_fps" "PASS" "Configured as a positive number."
else
    record_check "img_fps" "FAIL" "Must be a positive number."
fi

if is_positive_integer "${fph:-}"; then
    record_check "fph" "PASS" "Configured as a positive integer."
else
    record_check "fph" "FAIL" "Must be a positive integer."
fi

if is_nonnegative_integer "${mins:-}"; then
    record_check "mins" "PASS" "Configured as a non-negative integer."
else
    record_check "mins" "FAIL" "Must be a non-negative integer."
fi

if [ -z "${album_id:-}" ]; then
    record_check "album_id" "PASS" "No album configured; album copy is disabled."
elif is_positive_integer "$album_id"; then
    record_check "album_id" "PASS" "Configured as a positive numeric album ID."
else
    record_check "album_id" "FAIL" "Must be blank or a positive numeric album ID."
fi

if [ -n "${caption_template:-}" ]; then
    record_check "caption_template" "PASS" "A caption template is configured."
else
    record_check "caption_template" "FAIL" "Must not be empty."
fi

if [ -n "${frames_dir:-}" ]; then
    record_check "frames_dir" "PASS" "A frame directory is configured."
else
    record_check "frames_dir" "FAIL" "Must not be empty."
fi

if [ -n "${commentsubs_dir:-}" ]; then
    record_check "commentsubs_dir" "PASS" "A comments directory is configured."
else
    record_check "commentsubs_dir" "FAIL" "Must not be empty."
fi

if [ "${enable_commentsubs:-}" = "true" ] || [ "${enable_commentsubs:-}" = "false" ]; then
    record_check "enable_commentsubs" "PASS" "Configured as true or false."
else
    record_check "enable_commentsubs" "FAIL" "Must be true or false."
fi

if [ "${DRY_RUN:-}" = "0" ] || [ "${DRY_RUN:-}" = "1" ]; then
    record_check "DRY_RUN" "PASS" "Configured as 0 or 1."
else
    record_check "DRY_RUN" "FAIL" "Must be 0 or 1."
fi

if [ "${enable_commentsubs:-false}" = "true" ]; then
    if [ -d "${commentsubs_dir:-}" ]; then
        record_check "commentsubs directory" "PASS" "Configured directory exists."
    else
        record_check "commentsubs directory" "FAIL" "Configured directory '${commentsubs_dir:-}' does not exist."
    fi
fi

if [ -n "${frames_dir:-}" ] && [ -d "$frames_dir" ]; then
    first_frame=$(find "$frames_dir" -maxdepth 1 -type f -name 'frame_*.jpg' -print -quit)
    if [ -n "$first_frame" ]; then
        record_check "Frame inventory" "PASS" "Numbered JPG frame files were found."
    else
        record_check "Frame inventory" "FAIL" "No numbered JPG frame files found in '$frames_dir'."
    fi
else
    record_check "Frame inventory" "FAIL" "Configured frames directory '${frames_dir:-}' does not exist."
fi

if [ ! -f "fb/state.json" ]; then
    log_info "fb/state.json not found. Initializing new state file..."
    mkdir -p fb
    printf '%s\n' '{"current_frame": 1, "total_posted": 0, "last_run_timestamp": ""}' > fb/state.json
fi

current_frame=""
if command -v jq >/dev/null 2>&1 && [ -f "fb/state.json" ]; then
    current_frame=$(jq -er '.current_frame | select(type == "number" and . == floor)' fb/state.json 2>/dev/null)
fi

if is_positive_integer "$current_frame"; then
    record_check "frameiterator" "PASS" "fb/state.json has a positive current_frame."
    if is_positive_integer "${total_frame:-}" && (( 10#$current_frame > 10#$total_frame )); then
        record_check "Next frame" "PASS" "Episode is complete; no next frame is expected."
    else
        frame_file="${frames_dir:-}/frame_${current_frame}.jpg"
        if [ -f "$frame_file" ]; then
            record_check "Next frame" "PASS" "Frame $current_frame exists."
        else
            record_check "Next frame" "FAIL" "Frame file '$frame_file' is missing."
        fi
    fi
else
    record_check "frameiterator" "FAIL" "fb/state.json must contain a positive integer current_frame."
fi

if command -v curl >/dev/null 2>&1 && command -v jq >/dev/null 2>&1 &&
    declare -F check_token >/dev/null; then
    if check_token; then
        record_check "FB_PAGE_TOKEN" "PASS" "Token was validated."
        if [ -z "${page_name:-}" ]; then
            record_check "page_name" "PASS" "Optional Page name check is disabled."
        elif [ "$page_name" = "$VALIDATED_PAGE_NAME" ]; then
            record_check "page_name" "PASS" "Matches the Facebook Page name exactly."
        else
            record_check "page_name" "FAIL" "Configured '$page_name', but Facebook returned '$VALIDATED_PAGE_NAME'."
        fi
    else
        record_check "FB_PAGE_TOKEN" "FAIL" "Token validation failed."
        record_check "page_name" "FAIL" "Could not verify the Page name because token validation failed."
    fi
else
    record_check "FB_PAGE_TOKEN" "FAIL" "Cannot validate token because curl, jq, or the token checker is unavailable."
    record_check "page_name" "FAIL" "Could not verify the Page name because validation tools or the token checker are unavailable."
fi

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
        echo "## RepoIntegrity Report"
        if [ "$CHECK_FAILURES" -eq 0 ]; then
            echo "- **Overall status:** PASSED"
        else
            echo "- **Overall status:** FAILED ($CHECK_FAILURES check(s) failed)"
        fi
        echo
        echo "| Check | Status | Details |"
        echo "| --- | --- | --- |"
        printf '%s\n' "${CHECK_RESULTS[@]}"
    } >> "$GITHUB_STEP_SUMMARY"
fi

if [ "$CHECK_FAILURES" -gt 0 ]; then
    log_error "Repository integrity checks completed with $CHECK_FAILURES failure(s)."
    exit 1
fi

log_info "All repository integrity checks passed."
