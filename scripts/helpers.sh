#!/usr/bin/env bash
# =============================================================
# Helper Functions for ERZFIO
# =============================================================

set -euo pipefail

# Print messages with timestamp
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Verify required command line tools are installed
check_dependencies() {
    local missing=0
    for cmd in curl jq bc; do
        if ! command -v "$cmd" &>/dev/null; then
            log_error "Missing required tool: $cmd"
            missing=1
        fi
    done
    if [ "$missing" -eq 1 ]; then
        exit 1
    fi
}

# Verify Facebook Page Access Token
check_token() {
    if [ -z "${FB_PAGE_TOKEN:-}" ]; then
        log_error "FB_PAGE_TOKEN secret is not set! Please add it in your repository secrets."
        return 1
    fi

    # Check token with Facebook Graph API using Bearer header
    local response
    if ! response=$(curl -sS -X GET "https://graph.facebook.com/v19.0/me?fields=id,name" \
        -H "Authorization: Bearer ${FB_PAGE_TOKEN}"); then
        log_error "Facebook Token validation request failed."
        return 1
    fi

    local error_msg
    error_msg=$(echo "$response" | jq -r '.error.message // empty')

    if [ -n "$error_msg" ]; then
        log_error "Facebook Token validation failed: $error_msg"
        return 1
    fi

    local current_page_id
    local current_page_name
    current_page_id=$(echo "$response" | jq -r '.id')
    current_page_name=$(echo "$response" | jq -r '.name')
    if [ -z "$current_page_id" ] || [ "$current_page_id" = "null" ] ||
        [ -z "$current_page_name" ] || [ "$current_page_name" = "null" ]; then
        log_error "Facebook Token validation returned an unexpected response."
        return 1
    fi

    log_info "Connected to Facebook Page: $current_page_name (ID: $current_page_id)"

    # Optional verification against expected page name
    if [ -n "${page_name:-}" ] && [ "$page_name" != "$current_page_name" ]; then
        log_error "Page name mismatch: Config expects '$page_name', but token is for '$current_page_name'"
        return 1
    fi

    return 0
}

# Read value from state.json
get_state() {
    local key="$1"
    local state_file="fb/state.json"
    if [ ! -f "$state_file" ]; then
        echo "0"
        return
    fi
    jq -r ".${key} // 0" "$state_file"
}

# Update value in state.json
set_state() {
    local key="$1"
    local value="$2"
    local state_file="fb/state.json"
    local temp_file="fb/state.json.tmp"

    if [ ! -f "$state_file" ]; then
        echo "{}" > "$state_file"
    fi

    # Update json property safely
    jq --arg k "$key" --arg v "$value" '.[$k] = ($v | tonumber? // $v)' "$state_file" > "$temp_file"
    mv "$temp_file" "$state_file"
}
