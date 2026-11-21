#!/usr/bin/env bash
# =============================================================
# Facebook Posting Script (Graph API via Bearer Token)
# =============================================================

set -euo pipefail

# Sourced helpers should be available, but source if needed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers.sh"

post_frame() {
    local frame_number="$1"
    local frame_file="$2"
    local caption="$3"
    local log_file="fb/log.txt"
    local album_status="skipped"

    # Check if frame file exists
    if [ ! -f "$frame_file" ]; then
        local err="Frame file not found: $frame_file"
        log_error "$err"
        echo "[!] $(date '+%Y-%m-%d %H:%M:%S') - Frame $frame_number - $err" >> "$log_file"
        return 1
    fi

    if [ -n "${album_id:-}" ] && [[ ! "$album_id" =~ ^[0-9]+$ ]]; then
        log_error "Facebook album ID must be numeric."
        return 1
    fi

    # Dry run mode: simulate post without touching Facebook
    if [ "${DRY_RUN:-0}" = "1" ]; then
        log_info "[DRY RUN] Would post frame: $frame_file"
        log_info "[DRY RUN] Caption: $caption"
        if [ -n "${album_id:-}" ]; then
            log_info "[DRY RUN] Would also add frame to Facebook album $album_id."
        fi
        echo "[√] $(date '+%Y-%m-%d %H:%M:%S') - Frame $frame_number - DRY RUN SUCCESS" >> "$log_file"
        return 0
    fi

    log_info "Posting Frame #$frame_number to Facebook..."

    # Upload photo to Facebook Page using Bearer authorization header
    local response
    response=$(curl -s -X POST "https://graph.facebook.com/v19.0/me/photos" \
        -H "Authorization: Bearer ${FB_PAGE_TOKEN}" \
        -F "source=@${frame_file}" \
        -F "message=${caption}" \
        -F "published=true")

    local error_msg
    error_msg=$(echo "$response" | jq -r '.error.message // empty')

    if [ -n "$error_msg" ]; then
        log_error "Failed to post Frame #$frame_number: $error_msg"
        echo "[!] $(date '+%Y-%m-%d %H:%M:%S') - Frame $frame_number - Error: $error_msg" >> "$log_file"
        return 1
    fi

    # Retrieve post ID and photo ID from response
    local photo_id
    local post_id
    photo_id=$(echo "$response" | jq -r '.id // empty')
    post_id=$(echo "$response" | jq -r '.post_id // .id // empty')

    log_info "Frame #$frame_number posted successfully! (Post ID: $post_id)"

    if [ -n "${album_id:-}" ]; then
        local album_response
        local album_error
        local album_photo_id

        log_info "Adding Frame #$frame_number to Facebook album $album_id..."
        if album_response=$(curl -sS -X POST "https://graph.facebook.com/v19.0/${album_id}/photos" \
            -H "Authorization: Bearer ${FB_PAGE_TOKEN}" \
            -F "source=@${frame_file}" \
            -F "message=${caption}" \
            -F "published=true"); then
            album_error=$(echo "$album_response" | jq -r '.error.message // empty')
            album_photo_id=$(echo "$album_response" | jq -r '.id // empty')

            if [ -n "$album_error" ]; then
                log_warn "Album upload failed for Frame #$frame_number: $album_error"
                album_status="failed"
            elif [ -z "$album_photo_id" ] || [ "$album_photo_id" = "null" ]; then
                log_warn "Album upload returned an unexpected response for Frame #$frame_number."
                album_status="failed"
            else
                log_info "Frame #$frame_number added to album $album_id (Photo ID: $album_photo_id)."
                album_status="posted"
            fi
        else
            log_warn "Album upload request failed for Frame #$frame_number."
            album_status="failed"
        fi
    fi

    # Optional subtitle comment posting
    local sub_status="ok"
    if [ "${enable_commentsubs:-false}" = "true" ]; then
        local sub_file="${commentsubs_dir}/frame_${frame_number}.jpg"
        if [ ! -f "$sub_file" ]; then
            sub_file="${commentsubs_dir}/frame_${frame_number}.png"
        fi

        if [ -f "$sub_file" ]; then
            log_info "Posting subtitle comment for Frame #$frame_number..."
            local comment_response
            local retry=0
            local max_retries=2
            local comment_ok=0

            while [ "$retry" -le "$max_retries" ]; do
                comment_response=$(curl -s -X POST "https://graph.facebook.com/v19.0/${post_id}/comments" \
                    -H "Authorization: Bearer ${FB_PAGE_TOKEN}" \
                    -F "source=@${sub_file}" \
                    -F "message=Subtitle")

                local comment_err
                comment_err=$(echo "$comment_response" | jq -r '.error.message // empty')

                if [ -z "$comment_err" ]; then
                    comment_ok=1
                    break
                fi

                retry=$((retry + 1))
                log_warn "Subtitle comment failed (attempt $retry/$max_retries). Retrying in 2s..."
                sleep 2
            done

            if [ "$comment_ok" -eq 0 ]; then
                log_warn "Subtitle comment failed permanently for Frame #$frame_number"
                sub_status="failed"
            fi
        fi
    fi

    # Append status marker to log: [√] for full success, [~] for partial success
    if [ "$sub_status" = "failed" ] || [ "$album_status" = "failed" ]; then
        echo "[~] $(date '+%Y-%m-%d %H:%M:%S') - Frame $frame_number - Photo posted (ID: $post_id), Album: $album_status, Subtitle: $sub_status" >> "$log_file"
    else
        echo "[√] $(date '+%Y-%m-%d %H:%M:%S') - Frame $frame_number - Post ID: $post_id" >> "$log_file"
    fi

    return 0
}
