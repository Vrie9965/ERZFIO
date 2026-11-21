#!/usr/bin/env python3
r"""
efioex6.py — Every Frame In Order Extractor (Version 6)
Streaming extraction with smart motion detection, pan thinning, Facebook safety,
cyclical talking/idle loop thinning, strict fade throttling, and High-Speed 1024 MB Async RAM Buffering.

Staged Pipeline:
  Stage 1: Streaming Chopping & Deduplication
           • Drops pitch-black frames.
           • Drops frozen stills (protects blinks & mouth flaps).
           • Thins long camera pans (keeps Start 0%, Mid 50%, End 100%).
           • Throttles fading transitions & text fades (caps at 5 frames max).
           • Asynchronous background disk writer buffered up to 1024 MB for maximum FPS.
  Stage 2: Cyclical & Talking Loop Filter (thins repeated mouth-flaps/idle loops across speech timeline).
  Stage 3: Safety & Censorship (runs ONLY on surviving frames: accurate blood & lewd checks).
  Stage 4: Subtitle Burning (burns subtitles into commentsubs/ matching survivors 1-to-1).
"""

import os
import sys
import gc
import json
import time
import queue
import shutil
import platform
import threading
import subprocess
from collections import deque

# --- Dependency Check and Auto-Install ---
REQUIRED_PACKAGES = {
    "PIL": "Pillow",
    "imagehash": "imagehash",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "av": "av",
}

def ensure_packages() -> None:
    import importlib
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        return

    print(f"[INFO] Missing packages: {', '.join(missing)}")
    print("[INFO] Installing packages automatically...")
    result = subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
    if result.returncode != 0:
        print("[ERROR] Automatic install failed. Please install manually:")
        print(f"pip install {' '.join(missing)}")
        sys.exit(1)
    print("[INFO] Packages installed successfully.\n")

ensure_packages()

import av
import cv2
import numpy as np
from PIL import Image
import imagehash


# =============================================================
# Configuration Defaults
# =============================================================

CONFIG_FILE = "config.txt"

CONFIG_DEFAULTS = {
    "ACCELERATION": "auto",           # auto = use GPU when available, cpu = force CPU
    "MAX_RAM_BUFFER_MB": 1024,        # Maximum RAM buffer in MB for async disk writing
    "VIDEO_FILE": "video.mkv",
    "OUTPUT_WIDTH": 1280,
    "QUALITY_CLEAN": 15,
    "QUALITY_SUBS": 30,
    "SUBTITLE_STREAM": -1,            # -1 = auto-detect English track
    "STILL_DIFF_THRESHOLD": 3.0,       # Maximum cell difference for true stills
    "PAN_MIN_SECONDS": 2.0,            # Minimum duration for 3-point pan thinning
    "BLACK_PIXEL_PERCENT": 98.0,       # % of near-black pixels to qualify as black frame
    "BLACK_PIXEL_MAX_LUMA": 8,         # Max luma value for black pixels
    "FADE_MAX_FRAMES": 5,              # Maximum frames to keep during any fading transition (black/text)
    "ENABLE_LOOP_FILTER": True,        # Detect & thin repeating mouth flaps / idle loops
    "LOOP_WINDOW_SIZE": 60,            # Rolling window to detect loops
    "ENABLE_BLOOD_CENSOR": True,       # Grayscale blood & gore frames
    "BLOOD_THRESHOLD_PERCENT": 5.0,    # % of deep crimson blood pixels needed to trigger
    "ENABLE_LEWD_CENSOR": True,        # Gaussian blur sensitive areas
    "LEWD_THRESHOLD_PERCENT": 25.0,    # % of exposed body skin area needed to trigger
    "SAVE_CENSORED_REVIEW": True,      # Keep originals in censored_review/
}

# Active runtime configuration
CONFIG = dict(CONFIG_DEFAULTS)


# =============================================================
# Hardware & GPU Detection
# =============================================================

def detect_gpu_capabilities() -> dict:
    """Detects GPU and hardware acceleration support (NVIDIA CUDA, NVDEC, OpenCL)."""
    info = {
        "has_nvidia": False,
        "gpu_name": "",
        "opencl_ready": False,
    }

    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if res.returncode == 0 and res.stdout.strip():
            info["has_nvidia"] = True
            info["gpu_name"] = res.stdout.strip().splitlines()[0]
    except Exception:
        pass

    try:
        cv2.ocl.setUseOpenCL(True)
        info["opencl_ready"] = cv2.ocl.useOpenCL()
    except Exception:
        pass

    return info


# =============================================================
# Asynchronous Background Disk Writer (Buffered up to 1024 MB)
# =============================================================

class AsyncFrameWriter:
    """
    Decoupled background worker that saves survivor JPEGs to disk in parallel.
    Buffers up to MAX_RAM_BUFFER_MB (default 1024 MB) in RAM so video decoding
    never waits for disk I/O, maximizing decoding FPS while strictly preventing OOM.
    """
    def __init__(self, max_buffer_frames: int = 250, quality: int = 15):
        self.queue = queue.Queue(maxsize=max(20, max_buffer_frames))
        self.quality = quality
        self.stop_signal = object()
        self.worker = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker.start()

    def submit(self, path: str, bgr: np.ndarray) -> None:
        """Adds a frame to the async queue. Blocks only if 1024 MB buffer is completely full."""
        self.queue.put((path, bgr), block=True)

    def _writer_loop(self) -> None:
        while True:
            item = self.queue.get()
            if item is self.stop_signal:
                self.queue.task_done()
                break
            path, bgr = item
            try:
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.quality)])
                if ok:
                    buf.tofile(path)
            except Exception as e:
                print(f"  [WARN] Failed writing {path}: {e}")
            finally:
                del bgr
                self.queue.task_done()

    def wait_completion(self) -> None:
        """Flushes all queued frames to disk before proceeding to next stage."""
        self.queue.join()
        self.queue.put(self.stop_signal)
        self.worker.join()


# =============================================================
# Helper Utilities
# =============================================================

def get_script_dir() -> str:
    """Returns the directory where this script lives."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def sanitize_path(text: str) -> str:
    """Removes personal username from paths for clean logging."""
    username = os.environ.get("USER", os.environ.get("USERNAME", ""))
    if username:
        text = text.replace(f"\\Users\\{username}", "\\Users\\User")
        text = text.replace(f"/Users/{username}", "/Users/User")
        text = text.replace(f"/home/{username}", "/home/User")
    return text

def write_jpeg(path: str, bgr: np.ndarray, quality: int) -> bool:
    """Unicode-safe JPEG saving using cv2.imencode."""
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if ok:
        buf.tofile(path)
    return bool(ok)

def format_timestamp(seconds: float) -> str:
    """Converts seconds into HH:MM:SS string."""
    s = max(0.0, float(seconds))
    hrs = int(s // 3600)
    mins = int((s % 3600) // 60)
    secs = int(s % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


# =============================================================
# Configuration File Management
# =============================================================

def load_config(base_dir: str) -> dict:
    """Loads configuration from config.txt."""
    cfg = dict(CONFIG_DEFAULTS)
    path = os.path.join(base_dir, CONFIG_FILE)
    if not os.path.exists(path):
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if key in CONFIG_DEFAULTS:
                default_val = CONFIG_DEFAULTS[key]
                if isinstance(default_val, bool):
                    cfg[key] = val.lower() in ("true", "1", "yes")
                elif isinstance(default_val, int):
                    try:
                        cfg[key] = int(val)
                    except ValueError:
                        pass
                elif isinstance(default_val, float):
                    try:
                        cfg[key] = float(val)
                    except ValueError:
                        pass
                else:
                    cfg[key] = val
    return cfg

def save_config(base_dir: str, cfg: dict) -> None:
    """Saves configuration back to config.txt."""
    path = os.path.join(base_dir, CONFIG_FILE)
    lines = [
        "# efioex6 configuration file",
        f"ACCELERATION = {cfg['ACCELERATION']}",
        f"MAX_RAM_BUFFER_MB = {cfg['MAX_RAM_BUFFER_MB']}",
        f"VIDEO_FILE = {cfg['VIDEO_FILE']}",
        f"OUTPUT_WIDTH = {cfg['OUTPUT_WIDTH']}",
        f"QUALITY_CLEAN = {cfg['QUALITY_CLEAN']}",
        f"QUALITY_SUBS = {cfg['QUALITY_SUBS']}",
        f"SUBTITLE_STREAM = {cfg['SUBTITLE_STREAM']}",
        f"STILL_DIFF_THRESHOLD = {cfg['STILL_DIFF_THRESHOLD']}",
        f"PAN_MIN_SECONDS = {cfg['PAN_MIN_SECONDS']}",
        f"BLACK_PIXEL_PERCENT = {cfg['BLACK_PIXEL_PERCENT']}",
        f"BLACK_PIXEL_MAX_LUMA = {cfg['BLACK_PIXEL_MAX_LUMA']}",
        f"FADE_MAX_FRAMES = {cfg['FADE_MAX_FRAMES']}",
        f"ENABLE_LOOP_FILTER = {'true' if cfg['ENABLE_LOOP_FILTER'] else 'false'}",
        f"LOOP_WINDOW_SIZE = {cfg['LOOP_WINDOW_SIZE']}",
        f"ENABLE_BLOOD_CENSOR = {'true' if cfg['ENABLE_BLOOD_CENSOR'] else 'false'}",
        f"BLOOD_THRESHOLD_PERCENT = {cfg['BLOOD_THRESHOLD_PERCENT']}",
        f"ENABLE_LEWD_CENSOR = {'true' if cfg['ENABLE_LEWD_CENSOR'] else 'false'}",
        f"LEWD_THRESHOLD_PERCENT = {cfg['LEWD_THRESHOLD_PERCENT']}",
        f"SAVE_CENSORED_REVIEW = {'true' if cfg['SAVE_CENSORED_REVIEW'] else 'false'}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================
# Stage 3: Precision Safety & Censorship Engine
# =============================================================

def is_colorful(bgr: np.ndarray) -> bool:
    """Checks if the frame has color (returns False if already black & white)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_saturation = float(hsv[:, :, 1].mean())
    return mean_saturation >= 15.0

def detect_and_censor_blood(bgr: np.ndarray, threshold_pct: float) -> tuple[bool, np.ndarray]:
    """
    Checks for heavy blood and gore. If found, converts the frame to grayscale.
    Uses strict deep-crimson color matching so regular clothes, orange jackets,
    red ribbons, blush, and normal characters NEVER trigger it.
    """
    if not CONFIG["ENABLE_BLOOD_CENSOR"]:
        return False, bgr

    if not is_colorful(bgr):
        return False, bgr

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    crimson1 = (h <= 5) & (s >= 115) & (v >= 25) & (v <= 140)
    crimson2 = (h >= 175) & (s >= 115) & (v >= 25) & (v <= 140)
    blood_mask = crimson1 | crimson2

    total_pixels = bgr.shape[0] * bgr.shape[1]
    blood_pixels = int(np.count_nonzero(blood_mask))
    blood_ratio = (blood_pixels / total_pixels) * 100.0

    if blood_ratio >= threshold_pct:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grey_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return True, grey_bgr

    return False, bgr

def detect_and_censor_lewd(bgr: np.ndarray, threshold_pct: float) -> tuple[bool, np.ndarray]:
    """
    Checks for large exposed body skin regions and applies Gaussian blur to the sensitive area.
    """
    if not CONFIG["ENABLE_LEWD_CENSOR"]:
        return False, bgr

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    skin_mask = (cr >= 135) & (cr <= 170) & (cb >= 85) & (cb <= 125)
    skin_u8 = (skin_mask.astype(np.uint8)) * 255

    contours, _ = cv2.findContours(skin_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = bgr.shape[0] * bgr.shape[1]
    modified = False
    result_bgr = bgr.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (total_area * (threshold_pct / 100.0)):
            x, y, w, h = cv2.boundingRect(cnt)
            roi = result_bgr[y:y+h, x:x+w]
            kernel_w = (w // 10) | 1
            kernel_h = (h // 10) | 1
            ksize = (max(25, kernel_w), max(25, kernel_h))
            result_bgr[y:y+h, x:x+w] = cv2.GaussianBlur(roi, ksize, 0)
            modified = True

    return modified, result_bgr


# =============================================================
# Stage 1: Motion, Pan, and Fade Detection
# =============================================================

def is_truly_black(gray: np.ndarray) -> bool:
    """Strict black check: ensures 98%+ of pixels are under 8 luma."""
    threshold = CONFIG["BLACK_PIXEL_MAX_LUMA"]
    req_pct = CONFIG["BLACK_PIXEL_PERCENT"]
    black_pixels = np.count_nonzero(gray <= threshold)
    pct = (black_pixels / gray.size) * 100.0
    return pct >= req_pct

def analyze_grid_difference(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[float, float]:
    """
    Splits frames into a 16x16 grid of cells to detect local micro-animations
    (like eye blinks and mouth movements) vs pure still images.
    Returns: (max_cell_change, overall_mean_change)
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    h, w = diff.shape
    rows, cols = 16, 16
    cell_h, cell_w = h // rows, w // cols

    max_cell = 0.0
    for r in range(rows):
        for c in range(cols):
            cell = diff[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            m = float(cell.mean())
            if m > max_cell:
                max_cell = m

    overall = float(diff.mean())
    return max_cell, overall

def is_fading_transition(prev_gray: np.ndarray, curr_gray: np.ndarray) -> bool:
    """
    Detects if the difference between two frames is a fade (brightness change)
    rather than an actual animated motion.
    Covers black fades, white fades, and text fades on static backgrounds.
    """
    luma_prev = float(prev_gray.mean())
    luma_curr = float(curr_gray.mean())
    luma_shift = abs(luma_curr - luma_prev)

    if luma_shift < 1.0:
        return False

    norm_prev = prev_gray.astype(float) / max(3.0, luma_prev)
    norm_curr = curr_gray.astype(float) / max(3.0, luma_curr)
    struct_diff = float(np.abs(norm_prev - norm_curr).mean())

    return struct_diff <= 0.22

def estimate_pan_transform(prev_gray: np.ndarray, curr_gray: np.ndarray):
    """Calculates camera motion using ORB feature matching."""
    orb = cv2.ORB_create(nfeatures=400)
    kp_a, desc_a = orb.detectAndCompute(prev_gray, None)
    kp_b, desc_b = orb.detectAndCompute(curr_gray, None)

    if desc_a is None or desc_b is None or len(kp_a) < 15 or len(kp_b) < 15:
        return None, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_a, desc_b)
    if len(matches) < 15:
        return None, 0.0

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])

    transform, inlier_mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 4.0)
    if transform is None:
        return None, 0.0

    inlier_ratio = float(inlier_mask.sum()) / len(matches)
    return transform, inlier_ratio

def compute_pan_residual(prev_gray: np.ndarray, curr_gray: np.ndarray, transform: np.ndarray) -> float:
    """Verifies that the transformed frame matches the current frame (pure camera move)."""
    h, w = curr_gray.shape
    aligned = cv2.warpPerspective(prev_gray, transform, (w, h))
    coverage = cv2.warpPerspective(np.full_like(prev_gray, 255), transform, (w, h))
    valid = coverage > 0

    diff = cv2.absdiff(aligned, curr_gray)
    changed = (diff > 25) & valid
    valid_count = int(valid.sum())
    if valid_count == 0:
        return 100.0
    return (float(changed.sum()) / valid_count) * 100.0


# =============================================================
# Buffers: Pan Buffer & Fade Buffer
# =============================================================

class FadeBuffer:
    """Buffers consecutive fading frames and strictly limits them to max_frames (5 max)."""
    def __init__(self, max_frames: int = 5):
        self.max_frames = max_frames
        self.buffer = []

    def add(self, frame_data: dict) -> None:
        self.buffer.append(frame_data)

    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    def flush(self) -> list[dict]:
        """Limits the fade transition to at most max_frames distributed from 0% to 100%."""
        if not self.buffer:
            return []

        N = len(self.buffer)
        if N <= self.max_frames:
            kept = list(self.buffer)
        else:
            kept = []
            for k in range(self.max_frames):
                target_idx = int(round((k / (self.max_frames - 1)) * (N - 1)))
                kept.append(self.buffer[target_idx])

        self.buffer = []
        return kept


class PanBuffer:
    """Buffers consecutive panning frames and keeps only 3 points if duration >= 2s."""
    def __init__(self, min_seconds: float):
        self.min_seconds = min_seconds
        self.buffer = []

    def add(self, frame_data: dict) -> None:
        self.buffer.append(frame_data)

    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    def flush(self) -> list[dict]:
        """Processes buffered frames when a pan ends. Returns list of frames to keep."""
        if not self.buffer:
            return []

        if len(self.buffer) == 1:
            kept = [self.buffer[0]]
            self.buffer = []
            return kept

        t_start = self.buffer[0]["time_s"] or 0.0
        t_end = self.buffer[-1]["time_s"] or 0.0
        duration = t_end - t_start

        if duration >= self.min_seconds and len(self.buffer) >= 3:
            first = self.buffer[0]
            mid = self.buffer[len(self.buffer) // 2]
            last = self.buffer[-1]
            first["pan_point"] = "start"
            mid["pan_point"] = "mid"
            last["pan_point"] = "end"
            kept = [first, mid, last]
        else:
            kept = list(self.buffer)

        self.buffer = []
        return kept


# =============================================================
# Stage 2: Cyclical & Talking Loop Filter
# =============================================================

def filter_cyclical_loops(base_dir: str, survivors: list[dict], frames_dir: str, window_size: int = 60) -> list[dict]:
    """
    Scans survivor frames for repeating animation loops (like talking mouth flaps or idle swaying).
    Keeps P unique frames distributed across 0% to 100% of the speech timeline, dropping redundant cycles.
    """
    total_survivors = len(survivors)
    if total_survivors < 10:
        return survivors

    print("\n>>> STAGE 2: Detecting and Thinning Cyclical / Talking Animation Loops...")
    print(f"    • Rolling buffer: {window_size} frames")
    print("    • Spreads unique loop frames across 0% to 100% of each speech timeline")

    thumbnails = []
    for s in survivors:
        fnum = s["final_number"]
        path = os.path.join(frames_dir, f"frame_{fnum}.jpg")
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            thumb = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        else:
            thumb = np.zeros((16, 16), dtype=np.uint8)
        thumbnails.append(thumb)

    keep_flags = [True] * total_survivors
    i = 0
    loops_found = 0
    frames_thinned = 0

    while i < total_survivors:
        best_period = None
        best_length = 0

        for period in range(3, 11):
            if i + period * 2 > total_survivors:
                continue

            match_len = 0
            k = i + period
            while k < total_survivors:
                diff = float(np.abs(thumbnails[k].astype(int) - thumbnails[k - period].astype(int)).mean())
                if diff <= 4.0:
                    match_len += 1
                    k += 1
                else:
                    break

            if match_len >= period and match_len > best_length:
                best_period = period
                best_length = match_len

        if best_period is not None and best_length >= best_period:
            run_start = i
            run_end = i + best_period + best_length - 1
            total_run_frames = run_end - run_start + 1
            P = best_period

            loops_found += 1

            for idx in range(run_start, run_end + 1):
                keep_flags[idx] = False

            for state in range(P):
                progress = state / max(1, P - 1)
                target_rel_idx = int(round(progress * (total_run_frames - 1)))
                best_choice = None
                best_dist = 999999
                for rel in range(total_run_frames):
                    if (rel % P) == state:
                        dist = abs(rel - target_rel_idx)
                        if dist < best_dist:
                            best_dist = dist
                            best_choice = run_start + rel

                if best_choice is not None:
                    keep_flags[best_choice] = True

            frames_thinned += (total_run_frames - P)
            i = run_end + 1
        else:
            i += 1

    print(f"    • Detected {loops_found} talking/idle loop sequences.")
    print(f"    • Thinned out {frames_thinned} redundant mouth-flap frames.")

    temp_frames_dir = os.path.join(base_dir, "frames_temp")
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir)
    os.makedirs(temp_frames_dir, exist_ok=True)

    new_survivors = []
    new_timestamps = {}
    new_idx = 1

    for idx, (surv, keep) in enumerate(zip(survivors, keep_flags)):
        old_fnum = surv["final_number"]
        old_path = os.path.join(frames_dir, f"frame_{old_fnum}.jpg")

        if keep and os.path.exists(old_path):
            new_path = os.path.join(temp_frames_dir, f"frame_{new_idx}.jpg")
            shutil.move(old_path, new_path)
            new_survivors.append({
                "final_number": new_idx,
                "time_s": surv["time_s"]
            })
            new_timestamps[str(new_idx)] = format_timestamp(surv["time_s"] or 0.0)
            new_idx += 1
        elif os.path.exists(old_path):
            os.remove(old_path)

    shutil.rmtree(frames_dir)
    shutil.move(temp_frames_dir, frames_dir)

    ts_path = os.path.join(base_dir, "timestamps.json")
    with open(ts_path, "w", encoding="utf-8") as f:
        json.dump(new_timestamps, f, indent=2)

    print(f"[INFO] Stage 2 Complete! Clean survivors remaining: {len(new_survivors)} frames.")
    return new_survivors


# =============================================================
# Subtitle Auto-Detection & Burning
# =============================================================

def find_english_sub_stream(video_path: str) -> int:
    """Finds the English subtitle stream index in the MKV."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "s",
            "-show_entries", "stream=index:stream_tags=language,title:stream=codec_name",
            "-of", "json", video_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(res.stdout)
        streams = data.get("streams", [])

        if not streams:
            return -1

        for i, s in enumerate(streams):
            tags = s.get("tags", {})
            lang = tags.get("language", "").lower()
            title = tags.get("title", "").lower()
            codec = s.get("codec_name", "").lower()

            if codec in ("hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"):
                continue

            if lang in ("eng", "en", "english") or "english" in title or "eng" in title:
                print(f"[INFO] Auto-detected English subtitle stream: #{i} ({codec})")
                return i

        for i, s in enumerate(streams):
            codec = s.get("codec_name", "").lower()
            if codec not in ("hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"):
                print(f"[INFO] Using first text subtitle stream: #{i} ({codec})")
                return i

        return -1
    except Exception as e:
        print(f"[WARN] Subtitle probe failed: {e}")
        return -1

def escape_sub_path(path: str) -> str:
    """Escapes paths for FFmpeg subtitle filter."""
    if platform.system() == "Windows":
        path = path.replace("\\", "/").replace(":", "\\:")
    return path.replace("'", "'\\''")

def burn_subtitles(base_dir: str, video_path: str, survivors: list[dict], sub_stream: int, use_gpu: bool) -> None:
    """Burns subtitles into commentsubs/ matching frames/ 1-to-1."""
    if sub_stream < 0 or not survivors:
        print("[INFO] No subtitle stream found or no survivors. Skipping commentsubs.")
        return

    ass_path = os.path.join(base_dir, "subs_extracted.ass")
    timing_path = os.path.join(base_dir, "subs_timing_list.txt")
    subs_dir = os.path.join(base_dir, "commentsubs")

    print("[INFO] Extracting subtitle track...")
    cmd_extract = [
        "ffmpeg", "-hide_banner", "-y", "-i", video_path,
        "-map", f"0:s:{sub_stream}", "-c:s", "ass", ass_path
    ]
    res = subprocess.run(cmd_extract, capture_output=True)
    if res.returncode != 0:
        print("[WARN] Could not extract subtitle stream. Skipping commentsubs.")
        return

    print(f"[INFO] Building timing map for {len(survivors)} survivor frames...")
    lines = ["ffconcat version 1.0"]
    for i in range(len(survivors) - 1):
        num_a = survivors[i]["final_number"]
        t_a = survivors[i]["time_s"] or 0.0
        t_b = survivors[i+1]["time_s"] or (t_a + 0.04)
        duration = max(0.001, t_b - t_a)
        lines.append(f"file 'frames/frame_{num_a}.jpg'")
        lines.append(f"duration {duration:.6f}")

    last_num = survivors[-1]["final_number"]
    lines.append(f"file 'frames/frame_{last_num}.jpg'")
    lines.append("duration 0.041667")

    with open(timing_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    os.makedirs(subs_dir, exist_ok=True)
    start_t = survivors[0]["time_s"] or 0.0
    vf = f"setpts=PTS+{start_t:.6f}/TB,subtitles='{escape_sub_path(ass_path)}'"

    print(f"[INFO] Burning subtitles into {len(survivors)} frames in commentsubs/...")
    cmd_burn = ["ffmpeg", "-hide_banner", "-y"]
    if use_gpu:
        cmd_burn.extend(["-hwaccel", "auto"])
    cmd_burn.extend([
        "-f", "concat", "-safe", "0", "-i", timing_path,
        "-vf", vf,
        "-vsync", "passthrough",
        "-q:v", str(CONFIG["QUALITY_SUBS"]),
        os.path.join(subs_dir, "frame_%d.jpg")
    ])
    subprocess.run(cmd_burn, capture_output=True)

    for p in (ass_path, timing_path):
        if os.path.exists(p):
            os.remove(p)

    print(f"[INFO] Subtitles burned successfully to commentsubs/ ({len(survivors)} frames).")


# =============================================================
# Video File Discovery & Selection
# =============================================================

def find_video_files(base_dir: str) -> list[str]:
    """Scans the directory for all MKV (and common video) files."""
    valid_exts = (".mkv", ".mp4", ".webm")
    try:
        files = [
            f for f in os.listdir(base_dir)
            if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(base_dir, f))
        ]
        files.sort()
        return files
    except Exception:
        return []

def select_video_file(base_dir: str) -> str | None:
    """Scans the folder for MKV files and lets the user choose one."""
    video_files = find_video_files(base_dir)

    if not video_files:
        print("\n  [WARN] No .mkv video files found in this folder!")
        print("  Place your anime .mkv episode in this folder or type the path below.")
        manual = input("  Video path (or Enter to cancel): ").strip()
        if manual and os.path.exists(os.path.join(base_dir, manual)):
            return manual
        elif manual and os.path.exists(manual):
            return manual
        return None

    if len(video_files) == 1:
        print(f"\n  [INFO] Auto-selected single video file: {video_files[0]}")
        return video_files[0]

    print("\n  ── Available Video Files in Folder ──────────────────")
    for idx, fname in enumerate(video_files, 1):
        print(f"    [{idx}] {fname}")
    print("    [M] Enter manual filename")
    print("    [0] Cancel / Back to menu\n")

    while True:
        choice = input(f"  Select video (1-{len(video_files)}): ").strip().lower()
        if choice in ("0", "cancel", "c"):
            return None
        elif choice == "m":
            manual = input("  Enter filename (or Enter to cancel): ").strip()
            if manual:
                return manual
            return None
        try:
            val = int(choice)
            if 1 <= val <= len(video_files):
                return video_files[val - 1]
        except ValueError:
            pass
        print(f"  Invalid selection. Please enter 1 to {len(video_files)}.")


# =============================================================
# Staged Extraction Pipeline
# =============================================================

def run_extraction(base_dir: str, video_file: str) -> None:
    """
    Executes the extraction pipeline in clean, RAM-safe stages:
      Stage 1: Streaming Chopping (drops black, stills, thins pans, caps fades at 5 max, async 1024MB write queue)
      Stage 2: Cyclical & Talking Loop Filter (thins repeated mouth-flaps across 0-100% speech timeline)
      Stage 3: Safety & Censorship (runs ONLY on surviving frames)
      Stage 4: Subtitle Burning (burns subtitles into commentsubs/ matching survivors)
    """
    video_path = os.path.join(base_dir, video_file)
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {sanitize_path(video_path)}")
        return

    frames_dir = os.path.join(base_dir, "frames")
    subs_dir = os.path.join(base_dir, "commentsubs")
    review_dir = os.path.join(base_dir, "censored_review")

    if os.path.exists(frames_dir) and os.listdir(frames_dir):
        ans = input("  frames/ already contains files. Overwrite? (y/n): ").strip().lower()
        if ans != "y":
            print("  Extraction cancelled.")
            return

    for folder in (frames_dir, subs_dir, review_dir):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    gpu_info = detect_gpu_capabilities()
    use_gpu = False
    accel_mode = CONFIG.get("ACCELERATION", "auto").lower()
    if accel_mode in ("auto", "cuda", "gpu"):
        if gpu_info["has_nvidia"] or gpu_info["opencl_ready"]:
            use_gpu = True

    print("\n" + "="*60)
    print("  efioex6 — Smart Streaming Frame Extractor")
    print("="*60)
    print(f"  Input Video:   {sanitize_path(video_file)}")
    print(f"  Resolution:    {CONFIG['OUTPUT_WIDTH']}px width")
    print(f"  RAM Buffer:    Up to {CONFIG['MAX_RAM_BUFFER_MB']} MB (Asynchronous Parallel Disk Writer)")
    if use_gpu:
        dev_desc = gpu_info["gpu_name"] if gpu_info["gpu_name"] else "GPU (OpenCL)"
        print(f"  Acceleration:  ENABLED ({dev_desc})")
        if gpu_info["opencl_ready"]:
            cv2.ocl.setUseOpenCL(True)
    else:
        print("  Acceleration:  CPU Multithreaded")
    print("-"*60)

    # ═════════════════════════════════════════════════════════════
    # STAGE 1: STREAMING CHOPPING & DEDUPLICATION (BUFFERED UP TO 1024 MB)
    # ═════════════════════════════════════════════════════════════
    print("\n>>> STAGE 1: Chopping & Deduplicating Frames (Streaming Pass)...")
    print("    • Drops pitch-black frames")
    print("    • Drops frozen stills (protects blinks & mouth flaps)")
    print("    • Thins long camera pans (keeps Start 0%, Mid 50%, End 100%)")
    print(f"    • Caps black & text fade transitions (strictly {CONFIG['FADE_MAX_FRAMES']} frames max)")

    # Calculate queue size: each 1280x720 frame is ~2.76 MB in RAM.
    # We buffer up to 70% of MAX_RAM_BUFFER_MB in queue, strictly capped under 1024 MB.
    frame_size_mb = (CONFIG["OUTPUT_WIDTH"] * int(CONFIG["OUTPUT_WIDTH"] * 9 / 16) * 3) / (1024 * 1024)
    max_queue_frames = max(20, int((CONFIG["MAX_RAM_BUFFER_MB"] * 0.70) / max(0.5, frame_size_mb)))

    async_writer = AsyncFrameWriter(max_buffer_frames=max_queue_frames, quality=CONFIG["QUALITY_CLEAN"])
    pan_buffer = PanBuffer(min_seconds=CONFIG["PAN_MIN_SECONDS"])
    fade_buffer = FadeBuffer(max_frames=CONFIG["FADE_MAX_FRAMES"])
    survivors = []
    timestamps = {}

    prev_small = None
    last_kept_small = None
    processed = 0
    t0 = time.perf_counter()

    def submit_survivor(item: dict) -> None:
        """Submits survivor frame to the high-speed 1024 MB async write buffer."""
        final_idx = len(survivors) + 1
        item["final_number"] = final_idx
        out_path = os.path.join(frames_dir, f"frame_{final_idx}.jpg")

        # Submit uncompressed frame to background writer
        async_writer.submit(out_path, item["bgr"])

        survivors.append({
            "final_number": final_idx,
            "time_s": item["time_s"]
        })
        timestamps[str(final_idx)] = format_timestamp(item["time_s"] or 0.0)

        # Drop references from main thread immediately
        item["bgr"] = None
        item["small"] = None

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        decoder_info = "CPU (Multithreaded)"

        if use_gpu and gpu_info["has_nvidia"]:
            try:
                orig_codec = stream.codec_context.name
                cuvid_name = None
                if "h264" in orig_codec:
                    cuvid_name = "h264_cuvid"
                elif "hevc" in orig_codec or "h265" in orig_codec:
                    cuvid_name = "hevc_cuvid"

                if cuvid_name:
                    stream.codec_context.codec = av.Codec(cuvid_name, "r")
                    decoder_info = f"GPU NVDEC ({cuvid_name})"
            except Exception:
                stream.thread_type = "AUTO"
                stream.thread_count = os.cpu_count() or 4
        else:
            stream.thread_type = "AUTO"
            stream.thread_count = os.cpu_count() or 4

        print(f"    Active Decoder: {decoder_info}\n")

        for frame in container.decode(video=0):
            processed += 1
            bgr = frame.to_ndarray(format="bgr24")
            h, w = bgr.shape[:2]
            time_s = frame.time or 0.0

            out_w = CONFIG["OUTPUT_WIDTH"]
            out_h = int(round(h * out_w / w / 2)) * 2
            if w != out_w:
                bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (480, int(out_h * 480 / out_w)), interpolation=cv2.INTER_AREA)

            # Check 1: Pitch Black frame check
            if is_truly_black(small):
                del bgr, gray, small, frame
                continue

            frame_data = {
                "orig_idx": processed,
                "bgr": bgr,
                "small": small,
                "time_s": time_s,
            }

            if last_kept_small is None:
                submit_survivor(frame_data)
                last_kept_small = small
                prev_small = small
                del bgr, gray, frame
                continue

            # Check 2: Check for Still Image vs Micro-Animation
            max_cell, overall_diff = analyze_grid_difference(prev_small, small)

            if max_cell < CONFIG["STILL_DIFF_THRESHOLD"]:
                prev_small = small
                del bgr, gray, frame
                continue

            is_micro_anim = (max_cell >= 12.0) and (overall_diff < 4.0)

            # Check 2.5: Fading Transition Check (black fade, text fade)
            is_fade = False
            if not is_micro_anim:
                is_fade = is_fading_transition(prev_small, small)

            if is_fade:
                fade_buffer.add(frame_data)
                prev_small = small
                del bgr, gray, frame
                continue
            else:
                if not fade_buffer.is_empty():
                    for ff in fade_buffer.flush():
                        submit_survivor(ff)

            # Check 3: Camera Pan Detection
            transform, inlier_ratio = estimate_pan_transform(prev_small, small)
            is_pan = False

            if not is_micro_anim and transform is not None and inlier_ratio >= 0.65:
                residual = compute_pan_residual(prev_small, small, transform)
                if residual <= 1.5:
                    is_pan = True

            if is_pan:
                pan_buffer.add(frame_data)
            else:
                if not pan_buffer.is_empty():
                    for pf in pan_buffer.flush():
                        submit_survivor(pf)
                submit_survivor(frame_data)
                last_kept_small = small

            prev_small = small

            del bgr, gray, frame

            if processed % 1000 == 0:
                gc.collect()

            if processed % 500 == 0:
                elapsed = time.perf_counter() - t0
                fps = processed / elapsed if elapsed > 0 else 0
                q_len = async_writer.queue.qsize()
                print(f"    Chopping: {processed} frames decoded | Survivors: {len(survivors)} | Queue: {q_len}/{max_queue_frames} | {fps:.1f} fps")

        if not fade_buffer.is_empty():
            for ff in fade_buffer.flush():
                submit_survivor(ff)

        if not pan_buffer.is_empty():
            for pf in pan_buffer.flush():
                submit_survivor(pf)

    # Wait for all buffered frames in the 1024 MB queue to finish writing to disk
    print("    • Flushing buffered frames to disk...")
    async_writer.wait_completion()

    ts_path = os.path.join(base_dir, "timestamps.json")
    with open(ts_path, "w", encoding="utf-8") as f:
        json.dump(timestamps, f, indent=2)

    stage1_time = time.perf_counter() - t0
    print(f"\n[INFO] Stage 1 Complete in {stage1_time/60:.1f} min! Processed: {processed} frames -> Initial Survivors: {len(survivors)}")

    # ═════════════════════════════════════════════════════════════
    # STAGE 2: CYCLICAL & TALKING LOOP FILTER
    # ═════════════════════════════════════════════════════════════
    if CONFIG["ENABLE_LOOP_FILTER"]:
        survivors = filter_cyclical_loops(base_dir, survivors, frames_dir, CONFIG["LOOP_WINDOW_SIZE"])

    # ═════════════════════════════════════════════════════════════
    # STAGE 3: SAFETY & CENSORSHIP (RUNS ONLY ON REMAINING SURVIVORS)
    # ═════════════════════════════════════════════════════════════
    blood_count = 0
    lewd_count = 0

    if CONFIG["ENABLE_BLOOD_CENSOR"] or CONFIG["ENABLE_LEWD_CENSOR"]:
        print("\n>>> STAGE 3: Applying Safety Filters on Survivors...")
        print(f"    • Blood threshold: >={CONFIG['BLOOD_THRESHOLD_PERCENT']}% dark crimson (greyscale)")
        print(f"    • Lewd threshold:  >={CONFIG['LEWD_THRESHOLD_PERCENT']}% body skin (blur)")

        t_filter = time.perf_counter()
        for idx in range(1, len(survivors) + 1):
            frame_path = os.path.join(frames_dir, f"frame_{idx}.jpg")
            if not os.path.exists(frame_path):
                continue

            frame_bgr = cv2.imread(frame_path)
            if frame_bgr is None:
                continue

            was_censored = False
            original_bgr = frame_bgr.copy()

            is_blood, frame_bgr = detect_and_censor_blood(frame_bgr, CONFIG["BLOOD_THRESHOLD_PERCENT"])
            if is_blood:
                was_censored = True
                blood_count += 1

            is_lewd, frame_bgr = detect_and_censor_lewd(frame_bgr, CONFIG["LEWD_THRESHOLD_PERCENT"])
            if is_lewd:
                was_censored = True
                lewd_count += 1

            if was_censored:
                if CONFIG["SAVE_CENSORED_REVIEW"]:
                    rev_path = os.path.join(review_dir, f"frame_{idx}_original.jpg")
                    write_jpeg(rev_path, original_bgr, CONFIG["QUALITY_CLEAN"])
                write_jpeg(frame_path, frame_bgr, CONFIG["QUALITY_CLEAN"])

            if idx % 200 == 0 or idx == len(survivors):
                print(f"    Safety checked: {idx}/{len(survivors)} frames | Blood greyed: {blood_count} | Lewd blurred: {lewd_count}")

        filter_elapsed = time.perf_counter() - t_filter
        print(f"[INFO] Stage 3 Complete in {filter_elapsed:.1f}s! Censored: {blood_count} blood, {lewd_count} lewd.")
    else:
        print("\n[INFO] Stage 3 Skipped (Safety filters disabled in config).")

    # ═════════════════════════════════════════════════════════════
    # STAGE 4: SUBTITLE BURNING (COMMENTSUBS)
    # ═════════════════════════════════════════════════════════════
    print("\n>>> STAGE 4: Subtitle Generation (commentsubs/)...")
    sub_stream = CONFIG["SUBTITLE_STREAM"]
    if sub_stream < 0:
        sub_stream = find_english_sub_stream(video_path)

    burn_subtitles(base_dir, video_path, survivors, sub_stream, use_gpu)

    total_time = time.perf_counter() - t0
    print("\n" + "="*60)
    print(f"  All Stages Finished in {total_time/60:.1f} minutes!")
    print(f"  Total Video Frames:  {processed}")
    print(f"  Final Survivors:     {len(survivors)}")
    print(f"  Clean Frames:        {sanitize_path(frames_dir)}/")
    print(f"  Comment Subtitles:   {sanitize_path(subs_dir)}/")
    if blood_count > 0 or lewd_count > 0:
        print(f"  Censored Originals:  {sanitize_path(review_dir)}/ ({blood_count} blood, {lewd_count} lewd to review)")
    print("="*60 + "\n")


# =============================================================
# Settings Menu
# =============================================================

def mode_settings(base_dir: str) -> None:
    """Interactive settings editor."""
    while True:
        cfg = load_config(base_dir)
        gpu = detect_gpu_capabilities()
        gpu_label = gpu["gpu_name"] if gpu["has_nvidia"] else "None (CPU)"

        print("\n  ── Settings ────────────────────────────────────────")
        print(f"    [1] Acceleration Mode:          {cfg['ACCELERATION']} (Detected: {gpu_label})")
        print(f"    [2] Select Video File:          {cfg['VIDEO_FILE']}")
        print(f"    [3] Output Width:               {cfg['OUTPUT_WIDTH']}px")
        print(f"    [4] Clean Quality (frames/):    {cfg['QUALITY_CLEAN']} (1-30)")
        print(f"    [5] Subtitle Quality (subs/):   {cfg['QUALITY_SUBS']} (1-30)")
        print(f"    [6] Subtitle Stream Index:      {cfg['SUBTITLE_STREAM']} (-1 = auto)")
        print(f"    [7] Pan Min Duration:           {cfg['PAN_MIN_SECONDS']} seconds")
        print(f"    [8] Fade Max Frames:            {cfg['FADE_MAX_FRAMES']} frames max")
        print(f"    [9] Max RAM Buffer:             {cfg['MAX_RAM_BUFFER_MB']} MB")
        print(f"    [A] Talking Loop Filter:        {cfg['ENABLE_LOOP_FILTER']}")
        print(f"    [B] Blood Censorship:           {cfg['ENABLE_BLOOD_CENSOR']} (Threshold: {cfg['BLOOD_THRESHOLD_PERCENT']}%)")
        print(f"    [C] Lewd Censorship:            {cfg['ENABLE_LEWD_CENSOR']} (Threshold: {cfg['LEWD_THRESHOLD_PERCENT']}%)")
        print("    [R] Reset to Defaults")
        print("    [0] Back to Main Menu\n")

        choice = input("  Enter option: ").strip().lower()
        if choice == "0":
            break
        elif choice == "r":
            save_config(base_dir, dict(CONFIG_DEFAULTS))
            print("  Settings reset to default values.")
        elif choice == "1":
            new_accel = "cpu" if cfg["ACCELERATION"] == "auto" else "auto"
            cfg["ACCELERATION"] = new_accel
            save_config(base_dir, cfg)
            print(f"  Acceleration set to: {new_accel}")
        elif choice == "2":
            selected = select_video_file(base_dir)
            if selected:
                cfg["VIDEO_FILE"] = selected
                save_config(base_dir, cfg)
                print(f"  Default video set to: {selected}")
        elif choice in ("3", "4", "5", "6", "7", "8", "9"):
            key_map = {"3": "OUTPUT_WIDTH", "4": "QUALITY_CLEAN", "5": "QUALITY_SUBS",
                       "6": "SUBTITLE_STREAM", "7": "PAN_MIN_SECONDS", "8": "FADE_MAX_FRAMES",
                       "9": "MAX_RAM_BUFFER_MB"}
            key = key_map[choice]
            val = input(f"  Enter new value for {key}: ").strip()
            if val:
                try:
                    cfg[key] = float(val) if "." in val else int(val)
                    save_config(base_dir, cfg)
                except ValueError:
                    print("  Invalid number.")
        elif choice == "a":
            cfg["ENABLE_LOOP_FILTER"] = not cfg["ENABLE_LOOP_FILTER"]
            save_config(base_dir, cfg)
            print(f"  Talking Loop Filter set to: {cfg['ENABLE_LOOP_FILTER']}")
        elif choice == "b":
            sub_opt = input("  Toggle blood censorship [T] or change threshold % [C]? ").strip().lower()
            if sub_opt == "c":
                v = input(f"  Enter blood % threshold (current: {cfg['BLOOD_THRESHOLD_PERCENT']}): ").strip()
                if v:
                    try:
                        cfg["BLOOD_THRESHOLD_PERCENT"] = float(v)
                        save_config(base_dir, cfg)
                    except ValueError:
                        print("  Invalid number.")
            else:
                cfg["ENABLE_BLOOD_CENSOR"] = not cfg["ENABLE_BLOOD_CENSOR"]
                save_config(base_dir, cfg)
        elif choice == "c":
            sub_opt = input("  Toggle lewd censorship [T] or change threshold % [C]? ").strip().lower()
            if sub_opt == "c":
                v = input(f"  Enter lewd % threshold (current: {cfg['LEWD_THRESHOLD_PERCENT']}): ").strip()
                if v:
                    try:
                        cfg["LEWD_THRESHOLD_PERCENT"] = float(v)
                        save_config(base_dir, cfg)
                    except ValueError:
                        print("  Invalid number.")
            else:
                cfg["ENABLE_LEWD_CENSOR"] = not cfg["ENABLE_LEWD_CENSOR"]
                save_config(base_dir, cfg)


# =============================================================
# Main Entry Point
# =============================================================

def main() -> None:
    base_dir = get_script_dir()
    global CONFIG
    CONFIG = load_config(base_dir)

    cli_video = None
    if len(sys.argv) > 1 and sys.argv[1].endswith((".mkv", ".mp4", ".webm", ".avi")):
        cli_video = sys.argv[1]

    while True:
        gpu = detect_gpu_capabilities()
        gpu_status = f"GPU: {gpu['gpu_name']}" if gpu["has_nvidia"] else "CPU Mode"
        found_videos = find_video_files(base_dir)
        vid_count_str = f"{len(found_videos)} video(s) found in folder" if found_videos else "No MKVs found in folder"

        print("\n  ╔══════════════════════════════════════════════════════╗")
        print("  ║         efioex6 — Smart Anime Frame Extractor        ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        print(f"   Status: {vid_count_str}  |  {gpu_status}\n")
        print("    [1] Start Streaming Extraction")
        print("    [2] Settings")
        print("    [3] Exit\n")

        choice = input("  Enter 1, 2, or 3: ").strip()
        if choice == "1":
            target = cli_video if cli_video else select_video_file(base_dir)
            if target:
                run_extraction(base_dir, target)
        elif choice == "2":
            mode_settings(base_dir)
            CONFIG = load_config(base_dir)
        elif choice == "3":
            print("  Bye!")
            sys.exit(0)
        else:
            print("  Invalid choice.")

if __name__ == "__main__":
    main()
