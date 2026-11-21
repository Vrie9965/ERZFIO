#!/usr/bin/env python3
"""Extracts anime frames through motion-aware filtering and staged processing.
Applies censorship checks, timestamp tracking, and optional subtitle burning."""
from __future__ import annotations

import os
import csv
import re
import tempfile
import sys
import gc
import json
import time
import queue
import shutil
import platform
import threading
import ctypes
from contextlib import contextmanager
import subprocess
import urllib.request
from collections import deque
from itertools import chain
from fractions import Fraction

# Dependency setup checks required modules and installs missing packages.
# Imports that depend on those packages are loaded after this check.
class ProcessPriorityController:
    """Changes this process priority on Windows.
    Uses normal, above-normal, and high scheduling classes."""
    PRIORITIES = {
        "normal": 0x00000020,
        "above_normal": 0x00008000,
        "high": 0x00000080,
    }

    def __init__(self) -> None:
        self.current_mode = None
        self.enabled = os.name == "nt"
        self._kernel32 = None
        self._warned = False
        if self.enabled:
            try:
                self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                self._kernel32.GetCurrentProcess.restype = ctypes.c_void_p
                self._kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                self._kernel32.SetPriorityClass.restype = ctypes.c_int
            except Exception:
                self._kernel32 = None

    def set(self, mode: str) -> bool:
        """Applies a process scheduling class.
        Leaves the program running if Windows rejects the change."""
        if mode not in self.PRIORITIES:
            raise ValueError(f"Unknown process priority mode: {mode}")
        if not self.enabled:
            self.current_mode = mode
            return True
        if self._kernel32 is None:
            self._warn_once("Windows priority APIs could not be initialized.")
            return False
        if self.current_mode == mode:
            return True

        handle = self._kernel32.GetCurrentProcess()
        success = self._kernel32.SetPriorityClass(handle, self.PRIORITIES[mode])
        if success:
            self.current_mode = mode
            return True

        error_code = ctypes.get_last_error()
        self._warn_once(f"SetPriorityClass failed (Windows error {error_code}).")
        return False

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            print(f"[WARN] Dynamic process priority unavailable: {message}")
            self._warned = True


PROCESS_PRIORITY = ProcessPriorityController()

OPENCL_GPU_ACTIVE = False
_OPENCL_GPU_WARNED = False


def preprocess_stage1_frame(bgr: np.ndarray, output_width: int, output_height: int) -> tuple[np.ndarray, np.ndarray]:
    """Resizes and downsamples frames using OpenCL when a suitable GPU is active.
    Returns the saved BGR image and the small grayscale analysis image."""
    global OPENCL_GPU_ACTIVE, _OPENCL_GPU_WARNED
    height, width = bgr.shape[:2]
    small_size = (480, int(output_height * 480 / output_width))

    if OPENCL_GPU_ACTIVE:
        try:
            with temporary_process_priority("above_normal"):
                gpu_frame = cv2.UMat(bgr)
                if width != output_width:
                    gpu_frame = cv2.resize(gpu_frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
                    saved_bgr = gpu_frame.get()
                else:
                    saved_bgr = bgr
                gray_gpu = cv2.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                small_gpu = cv2.resize(gray_gpu, small_size, interpolation=cv2.INTER_AREA)
                small = small_gpu.get()
            return saved_bgr, small
        except Exception as exc:
            OPENCL_GPU_ACTIVE = False
            if not _OPENCL_GPU_WARNED:
                print(f"\n[WARN] OpenCL image processing failed; continuing on CPU ({exc}).")
                _OPENCL_GPU_WARNED = True

    if width != output_width:
        bgr = cv2.resize(bgr, (output_width, output_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA)
    return bgr, small


def set_process_priority(mode: str) -> bool:
    """Sets the current Windows process priority.
    Non-Windows systems keep running without priority changes."""
    return PROCESS_PRIORITY.set(mode)


@contextmanager
def temporary_process_priority(mode: str):
    """Temporarily changes priority for one operation.
    Restores the previous class when the operation finishes."""
    previous_mode = PROCESS_PRIORITY.current_mode or "normal"
    set_process_priority(mode)
    try:
        yield
    finally:
        set_process_priority(previous_mode)


def current_subprocess_priority_flags() -> int:
    """Returns matching creation flags for heavy child processes.
    FFmpeg can then use the selected Windows priority class too."""
    if os.name != "nt":
        return 0
    return PROCESS_PRIORITY.PRIORITIES.get(PROCESS_PRIORITY.current_mode, PROCESS_PRIORITY.PRIORITIES["normal"])


def run_with_process_priority(mode: str, function, *args, **kwargs):
    """Runs a selected operation at the requested priority.
    Always restores normal priority before returning to the menu."""
    set_process_priority(mode)
    try:
        return function(*args, **kwargs)
    finally:
        set_process_priority("normal")


class ProgressLine:
    """Refreshes one terminal line at a fixed interval.
    Keeps progress output compact while long processing steps are running."""

    def __init__(self, label: str, interval: float = 0.5, indent: str = "    ") -> None:
        self.label = label
        self.interval = max(0.1, float(interval))
        self.indent = indent
        self.last_update = 0.0
        self.last_length = 0
        self.rendered = False

    def update(
        self,
        completed: int | None = None,
        total: int | None = None,
        detail: str = "",
        force: bool = False,
    ) -> None:
        now = time.perf_counter()
        if not force and now - self.last_update < self.interval:
            return

        status_parts = []
        if completed is not None and total is not None and total > 0:
            pct = min(100.0, max(0.0, completed / total * 100.0))
            status_parts.append(f"{completed:,}/{total:,} ({pct:5.1f}%)")
        elif completed is not None:
            status_parts.append(f"{completed:,}")
        if detail:
            status_parts.append(detail)
        line = f"{self.indent}{self.label}: " + " | ".join(status_parts)
        padding = " " * max(0, self.last_length - len(line))
        sys.stdout.write(f"\r{line}{padding}")
        sys.stdout.flush()
        self.last_length = max(self.last_length, len(line))
        self.last_update = now
        self.rendered = True

    def finish(self, detail: str | None = None) -> None:
        if detail is not None:
            self.update(detail=detail, force=True)
        if self.rendered:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.rendered = False
            self.last_length = 0


def _ffmpeg_time_seconds(value: str) -> float | None:
    """Converts an FFmpeg timestamp into seconds.
    Returns None when the progress value is unavailable or malformed."""
    try:
        hours, minutes, seconds = value.strip().split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (AttributeError, TypeError, ValueError):
        return None


def run_ffmpeg_with_progress(
    command: list[str],
    total_duration: float | None,
    label: str,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """Runs FFmpeg while displaying its encoded-time progress.
    Drains stderr separately so process output cannot deadlock."""
    progress_command = [command[0], "-progress", "pipe:1", "-nostats", *command[1:]]
    stderr_lines: deque[str] = deque(maxlen=80)

    process = subprocess.Popen(
        progress_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=cwd,
        creationflags=current_subprocess_priority_flags(),
    )

    def drain_stderr() -> None:
        if process.stderr is None:
            return
        for line in process.stderr:
            stderr_lines.append(line)

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    reporter = ProgressLine(label, indent="    ")
    current_time = 0.0
    current_speed = ""
    stdout_lines: list[str] = []

    try:
        if process.stdout is not None:
            for raw_line in process.stdout:
                raw_line = raw_line.strip()
                if not raw_line or "=" not in raw_line:
                    continue
                stdout_lines.append(raw_line)
                key, value = raw_line.split("=", 1)
                if key in ("out_time_us", "out_time_ms"):
                    try:
                        current_time = max(0.0, float(value) / 1_000_000.0)
                    except ValueError:
                        pass
                elif key == "out_time":
                    parsed = _ffmpeg_time_seconds(value)
                    if parsed is not None:
                        current_time = max(current_time, parsed)
                elif key == "speed":
                    current_speed = value
                elif key == "progress":
                    if total_duration and total_duration > 0:
                        pct = min(100.0, current_time / total_duration * 100.0)
                        detail = f"{current_time:.1f}/{total_duration:.1f}s ({pct:.1f}%)"
                    else:
                        detail = f"{current_time:.1f}s elapsed"
                    if current_speed and current_speed != "N/A":
                        detail += f" | {current_speed}"
                    if value == "end":
                        detail = "100.0% | finishing"
                    reporter.update(detail=detail, force=(value == "end"))
    finally:
        return_code = process.wait()
        stderr_thread.join(timeout=2.0)
        reporter.finish("complete" if return_code == 0 else "failed")

    return subprocess.CompletedProcess(
        progress_command,
        return_code,
        stdout="\n".join(stdout_lines),
        stderr="".join(stderr_lines),
    )


REQUIRED_PACKAGES = {
    "PIL": "Pillow",
    "imagehash": "imagehash",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "av": "av",
}

def ensure_packages() -> None:
    """Checks required imports and installs any missing packages.
    Stops startup when a required dependency cannot be installed."""
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

# Configuration defaults define the extractor settings.
# CONFIG holds the active values loaded from the settings file.
CONFIG_FILE = "config.txt"

CONFIG_DEFAULTS = {
    "ACCELERATION": "auto",
    "OPENCL_PREPROCESSING": True,
    "WD_TAGGER_DEVICE": "cuda",
    "PREVIEW_ENCODER": "auto",
    "MAX_RAM_BUFFER_MB": 1024,
    "VIDEO_FILE": "video.mkv",
    "OUTPUT_WIDTH": 1280,
    "QUALITY_CLEAN": 15,
    "QUALITY_SUBS": 30,
    "SUBTITLE_STREAM": -1,
    "STILL_DIFF_THRESHOLD": 3.0,
    "PAN_MIN_SECONDS": 2.0,
    "BLACK_PIXEL_PERCENT": 98.0,
    "BLACK_PIXEL_MAX_LUMA": 8,
    "FADE_MAX_FRAMES": 5,
    "ENABLE_LOOP_FILTER": True,
    "LOOP_WINDOW_SIZE": 60,
    "ENABLE_BLOOD_CENSOR": True,
    "BLOOD_THRESHOLD_PERCENT": 0.4,
    "BLOOD_CONTINUITY": True,

    "BLOOD_HIGH_PERCENT": 40.0,
    "BLOOD_BLOB_MIN_PERCENT": 0.02,
    "BLOOD_TAGGER_VARIANTS": False,

    "BLOOD_WD_TAGGER": True,
    "BLOOD_TAG_CONFIDENCE": 0.35,
    "BLOOD_TAGGER_MODEL": "SmilingWolf/wd-v1-4-moat-tagger-v2",
    "ENABLE_LEWD_CENSOR": True,
    "LEWD_THRESHOLD_PERCENT": 25.0,
    "SAVE_CENSORED_REVIEW": True,
}

CONFIG = dict(CONFIG_DEFAULTS)

def detect_gpu_capabilities() -> dict:
    """Detects available NVIDIA hardware and OpenCL support.
    Returns the acceleration details used to configure extraction."""
    info = {
        "has_nvidia": False,
        "gpu_name": "",
        "opencl_ready": False,
        "opencl_device": "",
        "opencl_vendor": "",
        "opencl_is_gpu": False,
        "opencl_is_nvidia": False,
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
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            info["opencl_ready"] = bool(cv2.ocl.useOpenCL())
            if info["opencl_ready"]:
                device = cv2.ocl.Device_getDefault()
                info["opencl_device"] = device.name() or "OpenCL device"
                info["opencl_vendor"] = device.vendorName() or "Unknown vendor"
                device_type = int(device.type())
                gpu_type = int(getattr(cv2.ocl, "Device_TYPE_GPU", 2))
                info["opencl_is_gpu"] = bool(device_type & gpu_type)
                info["opencl_is_nvidia"] = info["opencl_is_gpu"] and "nvidia" in info["opencl_vendor"].lower()
    except Exception:
        pass

    return info

class AsyncFrameWriter:
    """Writes selected frames to JPEG files on a background thread.
    A bounded queue separates frame processing from disk I/O."""
    def __init__(self, max_buffer_frames: int = 250, quality: int = 15):
        """Creates the bounded frame queue and starts its writer thread.
        Stores the JPEG quality used for saved frames."""
        self.queue = queue.Queue(maxsize=max(20, max_buffer_frames))
        self.quality = quality
        self.stop_signal = object()
        self.worker = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker.start()

    def submit(self, path: str, bgr: np.ndarray) -> None:
        """Adds a frame to the bounded output queue.
        Waits when the queue reaches its configured capacity."""
        self.queue.put((path, bgr), block=True)

    def _writer_loop(self) -> None:
        """Consumes queued frames and writes them as JPEG files.
        Marks each queue item complete after its write attempt."""
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

    def wait_completion(self, progress: ProgressLine | None = None) -> None:
        """Waits until all queued frames have finished saving.
        Optionally displays the background writer's completion progress."""
        total_tasks = self.queue.unfinished_tasks
        while self.queue.unfinished_tasks:
            unfinished = self.queue.unfinished_tasks
            completed = max(0, total_tasks - unfinished)
            if progress is not None:
                progress.update(completed, total_tasks, detail=f"{self.queue.qsize()} queued")
            time.sleep(0.1)
        if progress is not None:
            progress.finish(f"{total_tasks:,} frames written")
        self.queue.put(self.stop_signal)
        self.worker.join()

def get_script_dir() -> str:
    """Returns the folder containing the script or executable.
    This folder is used as the base path for inputs and outputs."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def sanitize_path(text: str) -> str:
    """Masks the current user folder in displayed file paths.
    Actual filesystem paths remain unchanged."""
    username = os.environ.get("USER", os.environ.get("USERNAME", ""))
    if username:
        text = text.replace(f"\\Users\\{username}", "\\Users\\User")
        text = text.replace(f"/Users/{username}", "/Users/User")
        text = text.replace(f"/home/{username}", "/home/User")
    return text

def write_jpeg(path: str, bgr: np.ndarray, quality: int) -> bool:
    """Encodes a frame and writes it as a JPEG file.
    Uses OpenCV buffer output to support Unicode paths."""
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if ok:
        buf.tofile(path)
    return bool(ok)

def format_timestamp(seconds: float) -> str:
    """Converts a time value into HH:MM:SS format.
    Used when creating the frame timestamp map."""
    s = max(0.0, float(seconds))
    hrs = int(s // 3600)
    mins = int((s % 3600) // 60)
    secs = int(s % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def load_config(base_dir: str) -> dict:
    """Loads recognized settings from config.txt.
    Invalid values leave their corresponding defaults unchanged."""
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
    """Writes the active settings to config.txt.
    Supported options are saved in a consistent order."""
    path = os.path.join(base_dir, CONFIG_FILE)
    lines = [
        "# efioex7 configuration file",
        f"ACCELERATION = {cfg['ACCELERATION']}",
        f"OPENCL_PREPROCESSING = {'true' if cfg['OPENCL_PREPROCESSING'] else 'false'}",
        f"WD_TAGGER_DEVICE = {cfg['WD_TAGGER_DEVICE']}",
        f"PREVIEW_ENCODER = {cfg['PREVIEW_ENCODER']}",
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
        f"BLOOD_HIGH_PERCENT = {cfg['BLOOD_HIGH_PERCENT']}",
        f"BLOOD_BLOB_MIN_PERCENT = {cfg['BLOOD_BLOB_MIN_PERCENT']}",
        f"BLOOD_CONTINUITY = {'true' if cfg['BLOOD_CONTINUITY'] else 'false'}",
        f"BLOOD_TAGGER_VARIANTS = {'true' if cfg['BLOOD_TAGGER_VARIANTS'] else 'false'}",
        f"BLOOD_WD_TAGGER = {'true' if cfg['BLOOD_WD_TAGGER'] else 'false'}",
        f"BLOOD_TAG_CONFIDENCE = {cfg['BLOOD_TAG_CONFIDENCE']}",
        f"BLOOD_TAGGER_MODEL = {cfg['BLOOD_TAGGER_MODEL']}",
        f"ENABLE_LEWD_CENSOR = {'true' if cfg['ENABLE_LEWD_CENSOR'] else 'false'}",
        f"LEWD_THRESHOLD_PERCENT = {cfg['LEWD_THRESHOLD_PERCENT']}",
        f"SAVE_CENSORED_REVIEW = {'true' if cfg['SAVE_CENSORED_REVIEW'] else 'false'}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def is_colorful(bgr: np.ndarray) -> bool:
    """Checks whether a frame contains meaningful color.
    Greyscale frames bypass the blood-detection pipeline."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_saturation = float(hsv[:, :, 1].mean())
    return mean_saturation >= 15.0

# These counters summarize the decisions made by the blood filter.
# The tagger state tracks whether its model is ready or unavailable.
BLOOD_STATS = {
    "gate_clear": 0,
    "instant": 0,
    "lighting_clear": 0,
    "tagger_flag": 0,
    "tagger_clear": 0,
    "degraded_flag": 0,
    "continuity": 0,
}

_LAST_BLOOD_FLAGGED = False

_TAGGER_STATE = {
    "ready": False,
    "failed": False,
    "session": None,
    "input_name": None,
    "blood_idx": -1,
    "provider": "CPUExecutionProvider",
    "model_path": None,
}

def _crimson_mask(hsv: np.ndarray) -> np.ndarray:
    """Builds a mask of pixels within the configured crimson range.
    The mask is used by the first blood-detection filter."""
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    crimson1 = (h <= 8) & (s >= 100) & (v >= 25) & (v <= 220)
    crimson2 = (h >= 172) & (s >= 100) & (v >= 25) & (v <= 220)
    return crimson1 | crimson2


def _disable_opencl(reason: str) -> None:
    """Disables the OpenCL path after a runtime error.
    The remaining frames continue through the CPU implementation."""
    global OPENCL_GPU_ACTIVE, _OPENCL_GPU_WARNED
    OPENCL_GPU_ACTIVE = False
    if not _OPENCL_GPU_WARNED:
        print(f"\n[WARN] OpenCL image processing failed; continuing on CPU ({reason}).")
        _OPENCL_GPU_WARNED = True


def _blood_color_features(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Builds the crimson mask and measures saturation for one frame.
    Uses OpenCL for color conversion and range filtering when available."""
    if OPENCL_GPU_ACTIVE:
        try:
            with temporary_process_priority("above_normal"):
                hsv_gpu = cv2.cvtColor(cv2.UMat(bgr), cv2.COLOR_BGR2HSV)
                mean_saturation = float(cv2.mean(hsv_gpu)[1])
                red_low = cv2.inRange(hsv_gpu, (0, 100, 25), (8, 255, 220))
                red_high = cv2.inRange(hsv_gpu, (172, 100, 25), (179, 255, 220))
                mask_gpu = cv2.bitwise_or(red_low, red_high)
                mask = mask_gpu.get()
            return mask, mean_saturation
        except Exception as exc:
            _disable_opencl(str(exc))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return _crimson_mask(hsv), float(hsv[:, :, 1].mean())

def _blob_filtered_ratio(crimson: np.ndarray, total_pixels: int, min_blob_px: int) -> float:
    """Measures the frame area covered by qualifying crimson regions.
    Small speckles are removed before the percentage is calculated."""
    if not crimson.any():
        return 0.0
    opened = cv2.morphologyEx(
        (crimson.astype(np.uint8)) * 255, cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    kept = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_px:
            kept += int(stats[i, cv2.CC_STAT_AREA])
    return (kept / total_pixels) * 100.0

def _augmented_variants(bgr: np.ndarray) -> list:
    """Creates contrast-adjusted and brightened frame variants.
    These copies support the optional lighting-normalization check."""

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch)
    contrast = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    bright = cv2.convertScaleAbs(bgr, alpha=1.2, beta=60)
    return [contrast, bright]

def _to_grey(bgr: np.ndarray) -> np.ndarray:
    """Converts a color frame to greyscale while retaining three channels.
    Returns the result in OpenCV BGR layout."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _gpu_tagging_requested() -> bool:
    """Checks whether CUDA inference is allowed by the active settings.
    Returns False when CPU-only processing was explicitly selected."""
    gpu_info = detect_gpu_capabilities()
    nvidia_available = bool(
        gpu_info.get("has_nvidia", False) or gpu_info.get("opencl_is_nvidia", False)
    )
    return (
        CONFIG.get("ACCELERATION", "auto").lower() != "cpu"
        and CONFIG.get("WD_TAGGER_DEVICE", "auto").lower() != "cpu"
        and nvidia_available
    )


def _ensure_onnxruntime() -> bool:
    """Loads ONNX Runtime, installing the GPU package when it is missing.
    Warns when a CPU-only package prevents CUDA inference."""
    gpu_requested = _gpu_tagging_requested()
    try:
        import onnxruntime as ort
        providers = set(ort.get_available_providers())
        if gpu_requested and "CUDAExecutionProvider" not in providers:
            print("[WARN] This Python environment has CPU-only ONNX Runtime installed.")
            print("       Use the main-menu GPU Runtime Setup option to install CUDA support.")
        return True
    except ImportError:
        package = 'onnxruntime-gpu[cuda,cudnn]' if gpu_requested else 'onnxruntime'
        print(f"[INFO] Installing {package} for the WD blood tagger...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package])
        if result.returncode != 0:
            print("[WARN] ONNX Runtime installation failed — Stage C will be skipped.")
            return False
        try:
            import onnxruntime as ort
            providers = set(ort.get_available_providers())
            if gpu_requested and "CUDAExecutionProvider" not in providers:
                print("[WARN] GPU package installed, but CUDAExecutionProvider is unavailable.")
            return True
        except ImportError:
            return False


def _create_wd_tagger_session(ort, model_path: str):
    """Creates the WD Tagger session with CUDA-first provider selection.
    CPU fallback is allowed in auto mode but not when CUDA is explicitly required."""
    requested_device = CONFIG.get("WD_TAGGER_DEVICE", "auto").lower()
    acceleration = CONFIG.get("ACCELERATION", "auto").lower()
    gpu_info = detect_gpu_capabilities()
    nvidia_present = bool(
        gpu_info.get("has_nvidia", False) or gpu_info.get("opencl_is_nvidia", False)
    )
    cuda_allowed = acceleration != "cpu" and requested_device != "cpu" and nvidia_present
    cuda_error = None

    if requested_device == "cuda" and not cuda_allowed:
        raise RuntimeError(
            "WD Tagger is configured for CUDA-only operation, but GPU acceleration is disabled "
            "or no NVIDIA GPU was detected. Check ACCELERATION and NVIDIA driver settings."
        )

    if cuda_allowed:
        try:
            preload = getattr(ort, "preload_dlls", None)
            if callable(preload):
                preload()
        except Exception as exc:
            print(f"[WARN] CUDA DLL preload did not complete ({exc}); testing CUDA provider directly.")

        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            try:
                session = ort.InferenceSession(
                    model_path,
                    providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
                )
                active = session.get_providers()
                if "CUDAExecutionProvider" in active:
                    return session, "CUDAExecutionProvider"
                cuda_error = "CUDA provider was listed but did not become active"
            except Exception as exc:
                cuda_error = f"CUDA session initialization failed: {exc}"
        else:
            cuda_error = "CUDAExecutionProvider is not installed in this ONNX Runtime build"

    if cuda_error:
        if requested_device == "cuda":
            raise RuntimeError(
                f"WD Tagger is configured for CUDA-only operation, but {cuda_error}. "
                "Run GPU Runtime Setup from the main menu."
            )
        print(f"[WARN] {cuda_error}; WD Tagger will use CPU in auto mode.")
        print("       Run GPU Runtime Setup from the main menu to enable CUDA inference.")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session, "CPUExecutionProvider"


def _init_blood_tagger(base_dir: str) -> bool:
    """Loads the configured WD Tagger model and its tag list.
    Prefers CUDA inference and retains CPU fallback."""
    if _TAGGER_STATE["ready"]:
        return True
    if _TAGGER_STATE["failed"]:
        return False
    if not _ensure_onnxruntime():
        _TAGGER_STATE["failed"] = True
        return False

    repo = str(CONFIG.get("BLOOD_TAGGER_MODEL", "SmilingWolf/wd-v1-4-moat-tagger-v2"))
    model_dir = os.path.join(base_dir, "models", repo.replace("/", "__"))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.onnx")
    tags_path = os.path.join(model_dir, "selected_tags.csv")

    try:
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1024:
            print(f"[INFO] Downloading WD Tagger model ({repo}) — one-time, cached in models/.")
            url_base = f"https://huggingface.co/{repo}/resolve/main"
            if not _download_model_file(f"{url_base}/model.onnx", model_path):
                _TAGGER_STATE["failed"] = True
                return False
            _download_model_file(f"{url_base}/selected_tags.csv", tags_path)

        import onnxruntime as ort
        session, active_provider = _create_wd_tagger_session(ort, model_path)

        blood_idx = -1
        n_rows = 0
        with open(tags_path, "r", encoding="utf-8") as file:
            first_char = file.read(1)
            file.seek(0)
            if first_char != "t":
                raise ValueError("selected_tags.csv is not a valid tag list (downloaded an error page?)")
            for row_idx, row in enumerate(csv.DictReader(file)):
                n_rows += 1
                if row["name"].strip() == "blood" and row["category"].strip() == "0":
                    blood_idx = row_idx
        if n_rows < 1000:
            raise ValueError(f"selected_tags.csv looks truncated ({n_rows} rows)")
        if blood_idx < 0:
            print("[WARN] 'blood' tag not found in selected_tags.csv — Stage C disabled.")
            _TAGGER_STATE["failed"] = True
            return False

        input_name = session.get_inputs()[0].name
        warmup_input = np.zeros((1, 448, 448, 3), np.float32)
        try:
            warmup_output = session.run(None, {input_name: warmup_input})
        except Exception as exc:
            if active_provider != "CUDAExecutionProvider":
                raise
            if CONFIG.get("WD_TAGGER_DEVICE", "cuda").lower() == "cuda":
                raise RuntimeError(
                    f"WD Tagger CUDA warm-up failed while CUDA-only mode is enabled: {exc}"
                ) from exc
            print(f"[WARN] WD Tagger CUDA warm-up failed ({exc}); retrying with CPU in auto mode.")
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            active_provider = "CPUExecutionProvider"
            input_name = session.get_inputs()[0].name
            warmup_output = session.run(None, {input_name: warmup_input})

        out_len = len(warmup_output[0][0])
        if blood_idx >= out_len:
            raise ValueError(f"blood index {blood_idx} out of range for model output ({out_len})")

        _TAGGER_STATE.update({
            "ready": True,
            "session": session,
            "input_name": input_name,
            "blood_idx": blood_idx,
            "provider": active_provider,
            "model_path": model_path,
        })
        print(f"[INFO] WD blood tagger ready ({repo}) | Provider: {active_provider}.")
        return True
    except Exception as exc:
        print(f"[WARN] WD tagger init failed ({exc}) — Stage C will be skipped.")
        _TAGGER_STATE["failed"] = True
        return False


def _wd_blood_confidence(bgr: np.ndarray) -> float | None:
    """Runs one frame through the WD Tagger model.
    Retries on CPU if CUDA inference fails."""
    if not _TAGGER_STATE["ready"]:
        return None

    h, w = bgr.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = bgr
    resized = cv2.resize(canvas, (448, 448), interpolation=cv2.INTER_AREA)
    inp = np.expand_dims(resized.astype(np.float32), 0)

    try:
        provider = _TAGGER_STATE.get("provider", "CPUExecutionProvider")
        if provider == "CUDAExecutionProvider":
            with temporary_process_priority("above_normal"):
                pred = _TAGGER_STATE["session"].run(None, {_TAGGER_STATE["input_name"]: inp})[0][0]
        else:
            pred = _TAGGER_STATE["session"].run(None, {_TAGGER_STATE["input_name"]: inp})[0][0]
    except Exception as exc:
        if _TAGGER_STATE.get("provider") == "CUDAExecutionProvider":
            if CONFIG.get("WD_TAGGER_DEVICE", "auto").lower() == "cuda":
                print(f"[ERROR] WD Tagger CUDA inference failed and CUDA-only mode is enabled ({exc}).")
                _TAGGER_STATE["failed"] = True
                _TAGGER_STATE["ready"] = False
                return None
            print(f"[WARN] WD Tagger CUDA inference failed ({exc}); switching to CPU in auto mode.")
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(_TAGGER_STATE["model_path"], providers=["CPUExecutionProvider"])
                _TAGGER_STATE["session"] = session
                _TAGGER_STATE["input_name"] = session.get_inputs()[0].name
                _TAGGER_STATE["provider"] = "CPUExecutionProvider"
                pred = session.run(None, {_TAGGER_STATE["input_name"]: inp})[0][0]
            except Exception as cpu_exc:
                print(f"[WARN] WD Tagger CPU fallback also failed ({cpu_exc}).")
                _TAGGER_STATE["failed"] = True
                _TAGGER_STATE["ready"] = False
                return None
        else:
            print(f"[WARN] WD tagger inference failed ({exc}) — treating as unavailable.")
            _TAGGER_STATE["failed"] = True
            _TAGGER_STATE["ready"] = False
            return None

    if pred.min() < 0.0 or pred.max() > 1.0:
        pred = 1.0 / (1.0 + np.exp(-pred))
    return float(pred[_TAGGER_STATE["blood_idx"]])

def detect_and_censor_blood(bgr: np.ndarray, threshold_pct: float, base_dir: str = ".") -> tuple[bool, np.ndarray]:
    """Checks frames for blood using color and semantic filters.
    Confirmed frames are converted to greyscale with scene continuity handling."""
    global _LAST_BLOOD_FLAGGED
    if not CONFIG["ENABLE_BLOOD_CENSOR"]:
        _LAST_BLOOD_FLAGGED = False
        return False, bgr

    crimson_mask, mean_saturation = _blood_color_features(bgr)
    if mean_saturation < 15.0:
        _LAST_BLOOD_FLAGGED = False
        return False, bgr

    total_pixels = bgr.shape[0] * bgr.shape[1]
    min_blob_px = max(30, int(total_pixels * CONFIG["BLOOD_BLOB_MIN_PERCENT"] / 100.0))

    gate_ratio = _blob_filtered_ratio(crimson_mask, total_pixels, min_blob_px)
    if CONFIG.get("BLOOD_CONTINUITY", True) and _LAST_BLOOD_FLAGGED and gate_ratio >= 0.1:
        BLOOD_STATS["continuity"] += 1
        _LAST_BLOOD_FLAGGED = True
        return True, _to_grey(bgr)
    if gate_ratio < threshold_pct:
        BLOOD_STATS["gate_clear"] += 1
        _LAST_BLOOD_FLAGGED = False
        return False, bgr

    votes = 0
    if CONFIG["BLOOD_TAGGER_VARIANTS"]:
        for variant in _augmented_variants(bgr):
            variant_mask, _ = _blood_color_features(variant)
            v_ratio = _blob_filtered_ratio(variant_mask, total_pixels, min_blob_px)
            if v_ratio >= threshold_pct:
                votes += 1
        if votes == 0:

            BLOOD_STATS["lighting_clear"] += 1
            _LAST_BLOOD_FLAGGED = False
            return False, bgr
    else:
        votes = 2

    if CONFIG["BLOOD_WD_TAGGER"] and _init_blood_tagger(base_dir):
        confidence = _wd_blood_confidence(bgr)
        if confidence is not None:
            if confidence >= CONFIG["BLOOD_TAG_CONFIDENCE"]:
                BLOOD_STATS["tagger_flag"] += 1
                _LAST_BLOOD_FLAGGED = True
                return True, _to_grey(bgr)
            BLOOD_STATS["tagger_clear"] += 1
            _LAST_BLOOD_FLAGGED = False
            return False, bgr

    if gate_ratio >= CONFIG["BLOOD_HIGH_PERCENT"]:
        BLOOD_STATS["instant"] += 1
        _LAST_BLOOD_FLAGGED = True
        return True, _to_grey(bgr)
    if votes >= 2:
        BLOOD_STATS["degraded_flag"] += 1
        _LAST_BLOOD_FLAGGED = True
        return True, _to_grey(bgr)
    BLOOD_STATS["tagger_clear"] += 1
    _LAST_BLOOD_FLAGGED = False
    return False, bgr

def detect_and_censor_lewd(bgr: np.ndarray, threshold_pct: float) -> tuple[bool, np.ndarray]:
    """Detects large skin-colored regions and blurs qualifying areas.
    Returns whether censorship was applied and the processed frame."""
    if not CONFIG["ENABLE_LEWD_CENSOR"]:
        return False, bgr

    if OPENCL_GPU_ACTIVE:
        try:
            with temporary_process_priority("above_normal"):
                ycrcb_gpu = cv2.cvtColor(cv2.UMat(bgr), cv2.COLOR_BGR2YCrCb)
                skin_mask_gpu = cv2.inRange(ycrcb_gpu, (0, 135, 85), (255, 170, 125))
                skin_u8 = skin_mask_gpu.get()
        except Exception as exc:
            _disable_opencl(str(exc))
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            cr = ycrcb[:, :, 1]
            cb = ycrcb[:, :, 2]
            skin_u8 = (((cr >= 135) & (cr <= 170) & (cb >= 85) & (cb <= 125)).astype(np.uint8)) * 255
    else:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_u8 = (((cr >= 135) & (cr <= 170) & (cb >= 85) & (cb <= 125)).astype(np.uint8)) * 255

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

def is_truly_black(gray: np.ndarray) -> bool:
    """Detects frames that are almost entirely black.
    Uses the configured luma limit and black-pixel percentage."""
    threshold = CONFIG["BLACK_PIXEL_MAX_LUMA"]
    req_pct = CONFIG["BLACK_PIXEL_PERCENT"]
    black_pixels = np.count_nonzero(gray <= threshold)
    pct = (black_pixels / gray.size) * 100.0
    return pct >= req_pct

def analyze_grid_difference(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[float, float]:
    """Compares two frames across a 16-by-16 grid.
    Returns the strongest local change and the overall average difference."""
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
    """Checks whether a frame change resembles a brightness fade.
    Compares luminance shifts against normalized structural differences."""
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
    """Estimates camera movement between frames using ORB features.
    Returns the transform and its matching inlier ratio."""
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
    """Measures visual difference after aligning a frame to its predecessor.
    A low residual indicates that the change is mostly camera movement."""
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

class FadeBuffer:
    """Collects consecutive fade-transition frames.
    Keeps up to the configured maximum across the transition."""
    def __init__(self, max_frames: int = 5):
        """Initializes the fade-frame limit and an empty buffer.
        The limit controls how many frames survive a transition."""
        self.max_frames = max_frames
        self.buffer = []

    def add(self, frame_data: dict) -> None:
        """Adds a frame to the pending fade sequence.
        Frames are selected when the buffer is flushed."""
        self.buffer.append(frame_data)

    def is_empty(self) -> bool:
        """Reports whether the fade buffer contains pending frames.
        Used before deciding whether the buffer needs flushing."""
        return len(self.buffer) == 0

    def flush(self) -> list[dict]:
        """Selects representative frames from the pending fade sequence.
        Returns no more than the configured maximum and clears the buffer."""
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
    """Collects frames while a camera pan is detected.
    Long pans are reduced to start, middle, and end samples."""
    def __init__(self, min_seconds: float):
        """Stores the minimum pan duration and creates an empty buffer.
        Short pans can retain all of their frames."""
        self.min_seconds = min_seconds
        self.buffer = []

    def add(self, frame_data: dict) -> None:
        """Adds a frame to the pending camera-pan sequence.
        The sequence is evaluated when the pan ends."""
        self.buffer.append(frame_data)

    def is_empty(self) -> bool:
        """Reports whether the pan buffer contains pending frames.
        Used before deciding whether the buffer needs flushing."""
        return len(self.buffer) == 0

    def flush(self) -> list[dict]:
        """Selects frames to represent the completed camera pan.
        Long pans keep three samples; shorter pans keep their frames."""
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

def filter_cyclical_loops(base_dir: str, survivors: list[dict], frames_dir: str, window_size: int = 60) -> list[dict]:
    """Finds repeated animation cycles among surviving frames.
    Removes redundant cycles while preserving representative poses over time."""
    total_survivors = len(survivors)
    if total_survivors < 10:
        return survivors

    print("\n>>> STAGE 2: Detecting and Thinning Cyclical / Talking Animation Loops...")
    print(f"    • Rolling buffer: {window_size} frames")
    print("    • Spreads unique loop frames across 0% to 100% of each speech timeline")

    thumbnails = []
    thumbnail_progress = ProgressLine("Loop thumbnails")
    for thumbnail_index, s in enumerate(survivors, 1):
        fnum = s["final_number"]
        path = os.path.join(frames_dir, f"frame_{fnum}.jpg")
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            thumb = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        else:
            thumb = np.zeros((16, 16), dtype=np.uint8)
        thumbnails.append(thumb)
        thumbnail_progress.update(
            thumbnail_index,
            total_survivors,
            detail="loading frame samples",
        )
    thumbnail_progress.finish(f"{total_survivors:,} samples loaded")

    keep_flags = [True] * total_survivors
    i = 0
    loops_found = 0
    frames_thinned = 0
    comparisons = 0
    loop_progress = ProgressLine("Loop detection")

    while i < total_survivors:
        loop_progress.update(
            i,
            total_survivors,
            detail=f"{loops_found:,} loops | {comparisons:,} comparisons",
        )
        best_period = None
        best_length = 0

        for period in range(3, 11):
            if i + period * 2 > total_survivors:
                continue

            match_len = 0
            k = i + period
            while k < total_survivors:
                comparisons += 1
                diff = float(np.abs(thumbnails[k].astype(int) - thumbnails[k - period].astype(int)).mean())
                if diff <= 4.0:
                    match_len += 1
                    k += 1
                else:
                    break
                loop_progress.update(
                    i,
                    total_survivors,
                    detail=f"{loops_found:,} loops | {comparisons:,} comparisons",
                )

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

    loop_progress.finish(
        f"{loops_found:,} loops found | {frames_thinned:,} redundant frames removed"
    )

    print(f"    • Detected {loops_found} talking/idle loop sequences.")
    print(f"    • Thinned out {frames_thinned} redundant mouth-flap frames.")

    temp_frames_dir = os.path.join(base_dir, "frames_temp")
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir)
    os.makedirs(temp_frames_dir, exist_ok=True)

    new_survivors = []
    new_timestamps = {}
    new_idx = 1
    rewrite_progress = ProgressLine("Rewriting loop-filtered frames")

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
        rewrite_progress.update(
            idx + 1,
            total_survivors,
            detail=f"{len(new_survivors):,} retained",
        )

    rewrite_progress.finish(f"{len(new_survivors):,} frames retained")

    shutil.rmtree(frames_dir)
    shutil.move(temp_frames_dir, frames_dir)

    ts_path = os.path.join(base_dir, "timestamps.json")
    with open(ts_path, "w", encoding="utf-8") as f:
        json.dump(new_timestamps, f, indent=2)

    print(f"[INFO] Stage 2 Complete! Clean survivors remaining: {len(new_survivors)} frames.")
    return new_survivors

def find_english_sub_stream(video_path: str) -> int:
    """Finds a usable English subtitle stream in the video.
    Falls back to the first available text subtitle stream."""
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
    """Escapes a subtitle file path for FFmpeg filter syntax.
    Applies Windows-specific path handling when required."""
    if platform.system() == "Windows":
        path = path.replace("\\", "/").replace(":", "\\:")
    return path.replace("'", "'\\''")

def burn_subtitles(
    base_dir: str,
    video_path: str,
    survivors: list[dict],
    sub_stream: int,
    use_gpu: bool,
    output_dir: str | None = None,
    work_dir: str | None = None,
) -> bool:
    """Burns the source subtitle stream into the current frame sequence.
    Can write to a staging folder so existing commentsubs remain recoverable."""
    subs_dir = output_dir or os.path.join(base_dir, "commentsubs")
    os.makedirs(subs_dir, exist_ok=True)

    if sub_stream < 0 or not survivors:
        print("[INFO] No subtitle stream found or no frames available. commentsubs will be empty.")
        return True

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[ERROR] FFmpeg was not found in PATH.")
        return False

    work_dir = work_dir or base_dir
    os.makedirs(work_dir, exist_ok=True)
    ass_path = os.path.join(work_dir, "subs_extracted.ass")
    timing_path = os.path.join(work_dir, "subs_timing_list.txt")

    try:
        for survivor in survivors:
            number = int(survivor["final_number"])
            frame_path = os.path.join(base_dir, "frames", f"frame_{number}.jpg")
            if not os.path.isfile(frame_path):
                print(f"[ERROR] Missing frame_{number}.jpg; subtitle rebuilding was stopped.")
                return False

        print("[INFO] Extracting subtitle track...")
        cmd_extract = [
            ffmpeg, "-hide_banner", "-y", "-i", video_path,
            "-map", f"0:s:{sub_stream}", "-c:s", "ass", ass_path
        ]
        try:
            extract_duration, _ = probe_video_timing(video_path)
        except Exception:
            extract_duration = None
        extract_result = run_ffmpeg_with_progress(
            cmd_extract,
            total_duration=extract_duration,
            label="Subtitle extraction",
            cwd=base_dir,
        )
        if extract_result.returncode != 0 or not os.path.isfile(ass_path):
            detail = (extract_result.stderr or "").strip().splitlines()
            print("[WARN] Could not extract the selected subtitle stream.")
            if detail:
                print(f"       {detail[-1]}")
            return False

        lines = ["ffconcat version 1.0"]
        manifest_progress = ProgressLine("Subtitle timing map")
        for index, item in enumerate(survivors[:-1]):
            number = int(item["final_number"])
            time_a = float(item["time_s"])
            time_b = float(survivors[index + 1]["time_s"])
            duration = max(0.001, time_b - time_a)
            frame_path = os.path.abspath(os.path.join(base_dir, "frames", f"frame_{number}.jpg"))
            manifest_path = os.path.relpath(frame_path, os.path.dirname(timing_path)).replace("\\", "/")
            manifest_path = manifest_path.replace("'", "'\\''")
            lines.append(f"file '{manifest_path}'")
            lines.append(f"duration {duration:.9f}")
            manifest_progress.update(index + 1, max(1, len(survivors) - 1))
        last_number = int(survivors[-1]["final_number"])
        last_frame_path = os.path.abspath(os.path.join(base_dir, "frames", f"frame_{last_number}.jpg"))
        last_manifest_path = os.path.relpath(last_frame_path, os.path.dirname(timing_path)).replace("\\", "/")
        last_manifest_path = last_manifest_path.replace("'", "'\\''")
        lines.append(f"file '{last_manifest_path}'")
        lines.append("duration 0.041667")

        with open(timing_path, "w", encoding="utf-8", newline="\n") as file:
            file.write("\n".join(lines) + "\n")
        manifest_progress.finish(f"{len(survivors):,} frame timings prepared")

        os.makedirs(subs_dir, exist_ok=True)
        start_time = float(survivors[0]["time_s"])
        subtitle_filter = f"setpts=PTS+{start_time:.9f}/TB,subtitles='{escape_sub_path(ass_path)}'"
        print(f"[INFO] Burning subtitles into {len(survivors):,} frames...")

        cmd_burn = [ffmpeg, "-hide_banner", "-y"]
        if use_gpu:
            cmd_burn.extend(["-hwaccel", "auto"])
        cmd_burn.extend([
            "-f", "concat", "-safe", "0", "-i", timing_path,
            "-vf", subtitle_filter,
            "-vsync", "passthrough",
            "-q:v", str(CONFIG["QUALITY_SUBS"]),
            os.path.join(subs_dir, "frame_%d.jpg")
        ])
        burn_duration = max(0.1, float(survivors[-1]["time_s"]) + 0.041667)
        burn_result = run_ffmpeg_with_progress(
            cmd_burn,
            total_duration=burn_duration,
            label="Subtitle rendering",
            cwd=base_dir,
        )
        if burn_result.returncode != 0:
            detail = (burn_result.stderr or "").strip().splitlines()
            print("[ERROR] FFmpeg failed while burning subtitles.")
            if detail:
                print(f"       {detail[-1]}")
            return False

        output_pattern = re.compile(r"^frame_(\d+)\.jpg$", re.IGNORECASE)
        output_numbers = sorted(
            int(match.group(1))
            for name in os.listdir(subs_dir)
            if (match := output_pattern.match(name))
        )
        expected_numbers = list(range(1, len(survivors) + 1))
        if output_numbers != expected_numbers:
            print(f"[ERROR] Subtitle output is incomplete: expected {len(expected_numbers)} images, generated {len(output_numbers)}.")
            return False

        print(f"[INFO] Subtitle images generated successfully ({len(output_numbers):,} frames).")
        return True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[ERROR] Subtitle generation failed: {exc}")
        return False
    finally:
        for temporary_path in (ass_path, timing_path):
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

def _format_missing_ranges(numbers: list[int]) -> str:
    """Groups missing frame numbers into compact ranges.
    Returns a readable summary for the repair confirmation prompt."""
    if not numbers:
        return "None"
    ranges = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = number
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def _load_repair_metadata(base_dir: str, frame_numbers: set[int]) -> tuple[dict | list | None, dict | None, set[int]]:
    """Loads timestamp sidecars and prepares their renumbered contents.
    Retains original source times while removing entries for missing images."""
    precise_path = os.path.join(base_dir, "frame_timestamps.json")
    simple_path = os.path.join(base_dir, "timestamps.json")
    precise_data = None
    simple_data = None
    expected_numbers = set(frame_numbers)

    if os.path.isfile(precise_path):
        with open(precise_path, "r", encoding="utf-8") as file:
            precise_data = json.load(file)
        entries = precise_data.get("frames", []) if isinstance(precise_data, dict) else precise_data
        if not isinstance(entries, list):
            raise ValueError("frame_timestamps.json has an unsupported format.")
        metadata_numbers = []
        for item in entries:
            number = int(item["frame"])
            timestamp = float(item["time_s"])
            if number < 1 or not np.isfinite(timestamp) or timestamp < 0:
                raise ValueError("frame_timestamps.json contains an invalid entry.")
            metadata_numbers.append(number)
        if len(metadata_numbers) != len(set(metadata_numbers)):
            raise ValueError("frame_timestamps.json contains duplicate frame numbers.")
        expected_numbers.update(metadata_numbers)

    if os.path.isfile(simple_path):
        with open(simple_path, "r", encoding="utf-8") as file:
            simple_data = json.load(file)
        if not isinstance(simple_data, dict):
            raise ValueError("timestamps.json has an unsupported format.")
        for key in simple_data:
            if str(key).isdigit():
                expected_numbers.add(int(key))

    upper_bound = max(expected_numbers, default=0)
    missing_numbers = set(range(1, upper_bound + 1)) - frame_numbers
    return precise_data, simple_data, missing_numbers


def repair_frames(base_dir: str) -> bool:
    """Compacts the surviving frames and remaps matching review images.
    Preserves original timestamps and leaves commentsubs for Finalize to rebuild."""
    frames_dir = os.path.join(base_dir, "frames")
    if not os.path.isdir(frames_dir):
        print("  [ERROR] frames/ was not found. Run extraction first.")
        return False

    frame_pattern = re.compile(r"^frame_(\d+)\.jpg$", re.IGNORECASE)
    frame_paths: dict[int, str] = {}
    frame_filenames = os.listdir(frames_dir)
    scan_progress = ProgressLine("Scanning frame folder", indent="  ")
    for scan_index, filename in enumerate(frame_filenames, 1):
        match = frame_pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number in frame_paths:
                scan_progress.finish()
                print(f"  [ERROR] Multiple files use frame number {number}.")
                print("  Rename the duplicate manually before finalizing.")
                return False
            frame_paths[number] = os.path.join(frames_dir, filename)
        scan_progress.update(scan_index, max(1, len(frame_filenames)), detail="checking filenames")
    scan_progress.finish(f"{len(frame_paths):,} frames found")

    if not frame_paths:
        print("  [ERROR] No frame_N.jpg files were found in frames/.")
        return False

    original_numbers = sorted(frame_paths)
    try:
        precise_data, simple_data, missing_map = _load_repair_metadata(base_dir, set(original_numbers))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"  [ERROR] Could not read timestamp metadata: {exc}")
        print("  No files were changed.")
        return False

    missing_numbers = sorted(missing_map)
    old_to_new = {old_number: new_number for new_number, old_number in enumerate(original_numbers, 1)}
    changed_count = sum(old != new for old, new in old_to_new.items())
    review_dir = os.path.join(base_dir, "censored_review")
    legacy_review_names = False
    if os.path.isdir(review_dir):
        legacy_review_names = any(
            re.match(r"^frame_\d+_original\.jpg$", name, re.IGNORECASE)
            for name in os.listdir(review_dir)
        )

    if not missing_numbers and changed_count == 0 and not legacy_review_names:
        print(f"\n  [INFO] frames/ is already continuous ({len(original_numbers):,} frames).")
        print("  No renumbering is necessary.")
        return True

    print("\n  ── Frame Repair Preview ─────────────────────────────")
    print(f"  Existing frames:  {len(original_numbers):,}")
    print(f"  Missing numbers:  {_format_missing_ranges(missing_numbers)}")
    print(f"  Frames after repair: {len(original_numbers):,} (renumbered from 1)")
    print("  Matching censored-review images will receive the same new numbers.")
    with temporary_process_priority("normal"):
        confirm = input("\n  Repair and renumber these frames? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Finalize cancelled before changing files.")
        return False

    plan: list[tuple[str, str, str]] = []
    staged_root = None
    moved_to_stage: list[tuple[str, str, str]] = []
    moved_to_destination: list[tuple[str, str, str]] = []
    metadata_originals: dict[str, bytes] = {}
    metadata_replacements: dict[str, object] = {}
    precise_entries_after = None

    try:
        for old_number, source_path in frame_paths.items():
            new_number = old_to_new[old_number]
            destination_path = os.path.join(frames_dir, f"frame_{new_number}.jpg")
            if os.path.normcase(os.path.abspath(source_path)) != os.path.normcase(os.path.abspath(destination_path)):
                plan.append((source_path, "frames", destination_path))

        review_pattern = re.compile(r"^frame_(\d+)(?:_original)?\.jpg$", re.IGNORECASE)
        if os.path.isdir(review_dir):
            for filename in os.listdir(review_dir):
                match = review_pattern.match(filename)
                if not match:
                    continue
                old_number = int(match.group(1))
                source_path = os.path.join(review_dir, filename)
                if old_number in old_to_new:
                    destination_name = f"frame_{old_to_new[old_number]}.jpg"
                else:
                    orphan_name = f"orphan_{filename}"
                    destination_name = orphan_name
                    suffix = 2
                    while os.path.exists(os.path.join(review_dir, destination_name)):
                        stem, extension = os.path.splitext(orphan_name)
                        destination_name = f"{stem}_{suffix}{extension}"
                        suffix += 1
                destination_path = os.path.join(review_dir, destination_name)
                if os.path.normcase(os.path.abspath(source_path)) != os.path.normcase(os.path.abspath(destination_path)):
                    plan.append((source_path, "censored_review", destination_path))

        normalized_destinations = [os.path.normcase(os.path.abspath(item[2])) for item in plan]
        if len(normalized_destinations) != len(set(normalized_destinations)):
            raise RuntimeError("Multiple review images would map to the same frame number. Resolve duplicate review images first.")

        source_paths = {os.path.normcase(os.path.abspath(item[0])) for item in plan}
        for source_path, _, destination_path in plan:
            destination_key = os.path.normcase(os.path.abspath(destination_path))
            if os.path.exists(destination_path) and destination_key not in source_paths:
                raise RuntimeError(f"Repair would overwrite an unrelated file: {destination_path}")

        if precise_data is not None:
            precise_path = os.path.join(base_dir, "frame_timestamps.json")
            old_entries = precise_data.get("frames", []) if isinstance(precise_data, dict) else precise_data
            new_entries = []
            for item in old_entries:
                old_number = int(item["frame"])
                if old_number in old_to_new:
                    updated = dict(item)
                    updated["frame"] = old_to_new[old_number]
                    new_entries.append(updated)
            new_entries.sort(key=lambda item: int(item["frame"]))
            precise_entries_after = len(new_entries)
            if isinstance(precise_data, dict):
                updated_precise = dict(precise_data)
                updated_precise["frames"] = new_entries
            else:
                updated_precise = new_entries
            metadata_replacements[precise_path] = updated_precise

        if simple_data is not None:
            simple_path = os.path.join(base_dir, "timestamps.json")
            updated_simple = {key: value for key, value in simple_data.items() if not str(key).isdigit()}
            remapped_values = {}
            for old_key, value in simple_data.items():
                if str(old_key).isdigit() and int(old_key) in old_to_new:
                    remapped_values[str(old_to_new[int(old_key)])] = value
            updated_simple.update(remapped_values)
            metadata_replacements[simple_path] = dict(sorted(
                updated_simple.items(),
                key=lambda item: (0, int(item[0])) if str(item[0]).isdigit() else (1, str(item[0]))
            ))

        staged_root = tempfile.mkdtemp(prefix=".efioex7_frame_repair_", dir=base_dir)
        stage_paths = []
        for index, (source_path, folder_name, destination_path) in enumerate(plan):
            stage_path = os.path.join(staged_root, f"file_{index:08d}.tmp")
            stage_paths.append((source_path, stage_path, destination_path))

        repair_progress = ProgressLine("Renumbering frames and review images", indent="  ")
        total_moves = max(1, len(stage_paths) * 2)
        for move_index, (source_path, stage_path, destination_path) in enumerate(stage_paths, 1):
            shutil.move(source_path, stage_path)
            moved_to_stage.append((source_path, stage_path, destination_path))
            repair_progress.update(
                move_index,
                total_moves,
                detail="staging files",
            )

        for move_index, (source_path, stage_path, destination_path) in enumerate(stage_paths, len(stage_paths) + 1):
            shutil.move(stage_path, destination_path)
            moved_to_destination.append((source_path, stage_path, destination_path))
            repair_progress.update(
                move_index,
                total_moves,
                detail="placing files",
            )
        repair_progress.finish(f"{len(stage_paths):,} file mappings applied")

        for metadata_path, replacement in metadata_replacements.items():
            with open(metadata_path, "rb") as file:
                metadata_originals[metadata_path] = file.read()
            temporary_path = os.path.join(staged_root, os.path.basename(metadata_path) + ".new")
            with open(temporary_path, "w", encoding="utf-8") as file:
                json.dump(replacement, file, indent=2)
                file.write("\n")
            os.replace(temporary_path, metadata_path)

        print("\n  [SUCCESS] Frame numbering repaired.")
        print(f"  Frames retained: {len(original_numbers):,}")
        print(f"  Missing frame numbers removed: {len(missing_numbers):,}")
        print(f"  New sequence: frame_1.jpg through frame_{len(original_numbers)}.jpg")
        review_count = sum(1 for _, folder, _ in plan if folder == "censored_review")
        orphan_count = sum(
            1 for _, folder, destination in plan
            if folder == "censored_review" and os.path.basename(destination).startswith("orphan_")
        )
        if review_count:
            print(f"  Censored-review images renamed: {review_count:,}")
        if orphan_count:
            print(f"  Unmatched review images preserved as orphans: {orphan_count:,}")
        if precise_data is None:
            print("  [WARN] frame_timestamps.json was not found.")
        elif precise_entries_after is not None and precise_entries_after < len(original_numbers):
            print(f"  [WARN] {len(original_numbers) - precise_entries_after} surviving frames have no precise timestamp entry.")
        print("  Original source timestamps were retained.")
        return True

    except Exception as exc:
        for metadata_path, original_bytes in metadata_originals.items():
            try:
                restore_path = metadata_path + ".restore"
                with open(restore_path, "wb") as file:
                    file.write(original_bytes)
                os.replace(restore_path, metadata_path)
            except OSError:
                pass

        if moved_to_destination and staged_root:
            for source_path, stage_path, destination_path in reversed(moved_to_destination):
                try:
                    if os.path.exists(destination_path):
                        shutil.move(destination_path, stage_path)
                except OSError:
                    pass
        if moved_to_stage:
            for source_path, stage_path, destination_path in reversed(moved_to_stage):
                try:
                    if os.path.exists(stage_path):
                        os.makedirs(os.path.dirname(source_path), exist_ok=True)
                        shutil.move(stage_path, source_path)
                except OSError:
                    pass
        print(f"\n  [ERROR] Frame repair failed: {exc}")
        print("  The program attempted to restore original filenames and metadata.")
        return False
    finally:
        if staged_root and os.path.isdir(staged_root):
            shutil.rmtree(staged_root, ignore_errors=True)


def _read_frame_timeline(base_dir: str) -> tuple[list[dict], dict]:
    """Validates numbered frames against their precise timestamp sidecar.
    Returns one chronological record for every image in the folder."""
    frames_dir = os.path.join(base_dir, "frames")
    timestamp_path = os.path.join(base_dir, "frame_timestamps.json")
    if not os.path.isdir(frames_dir):
        raise ValueError("frames/ was not found.")
    if not os.path.isfile(timestamp_path):
        raise ValueError("frame_timestamps.json was not found. Run extraction with this version first.")

    pattern = re.compile(r"^frame_(\d+)\.jpg$", re.IGNORECASE)
    frame_numbers = sorted(
        int(match.group(1))
        for name in os.listdir(frames_dir)
        if (match := pattern.match(name))
    )
    if not frame_numbers:
        raise ValueError("No frame_N.jpg images were found in frames/.")
    expected = list(range(1, len(frame_numbers) + 1))
    if frame_numbers != expected:
        missing = sorted(set(expected) - set(frame_numbers))
        extra = sorted(set(frame_numbers) - set(expected))
        details = []
        if missing:
            details.append(f"missing frame numbers: {_format_missing_ranges(missing)}")
        if extra:
            details.append(f"unexpected frame numbers: {extra[:12]}")
        raise ValueError("Frame numbering is not continuous (" + "; ".join(details) + "). Run Finalize first.")

    with open(timestamp_path, "r", encoding="utf-8") as file:
        timestamp_data = json.load(file)
    if isinstance(timestamp_data, dict):
        entries = timestamp_data.get("frames", [])
    elif isinstance(timestamp_data, list):
        entries = timestamp_data
        timestamp_data = {"frames": entries}
    else:
        raise ValueError("frame_timestamps.json has an unsupported format.")
    if not isinstance(entries, list) or not entries:
        raise ValueError("frame_timestamps.json contains no frame records.")

    by_number = {}
    for item in entries:
        number = int(item["frame"])
        time_s = float(item["time_s"])
        if number < 1 or not np.isfinite(time_s) or time_s < 0:
            raise ValueError("frame_timestamps.json contains an invalid frame number or time.")
        if number in by_number:
            raise ValueError(f"frame_timestamps.json contains duplicate frame_{number} entries.")
        by_number[number] = time_s

    if sorted(by_number) != expected:
        missing_times = sorted(set(expected) - set(by_number))
        missing_images = sorted(set(by_number) - set(expected))
        details = []
        if missing_times:
            details.append(f"frames without timestamps: {missing_times[:12]}")
        if missing_images:
            details.append(f"timestamps without frames: {missing_images[:12]}")
        raise ValueError("Frame files and timestamp metadata do not match (" + "; ".join(details) + "). Run Finalize first.")

    survivors = [{"final_number": number, "time_s": by_number[number]} for number in expected]
    times = [item["time_s"] for item in survivors]
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise ValueError("Frame timestamps are not strictly increasing. Run Finalize or inspect the timestamp file.")

    for number in expected:
        image_path = os.path.join(frames_dir, f"frame_{number}.jpg")
        if not os.path.isfile(image_path):
            raise ValueError(f"Missing generated image: frame_{number}.jpg")
    return survivors, timestamp_data


def finalize_manual_changes(base_dir: str) -> None:
    """Repairs user-edited frame numbering and rebuilds subtitle images.
    Keeps the reconstructed preview as a separate, optional operation."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        print("  [ERROR] FFmpeg and FFprobe must both be available in PATH.")
        return

    timestamp_path = os.path.join(base_dir, "frame_timestamps.json")
    if not os.path.isfile(timestamp_path):
        print("  [ERROR] frame_timestamps.json was not found. Run extraction with this version first.")
        return
    try:
        with open(timestamp_path, "r", encoding="utf-8") as file:
            timestamp_data = json.load(file)
        if not isinstance(timestamp_data, dict):
            raise ValueError("Unsupported timestamp metadata format.")
        saved_source = timestamp_data.get("source_video")
        if not saved_source:
            raise ValueError("The timestamp metadata does not identify its source video.")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"  [ERROR] Could not identify the source video: {exc}")
        return

    saved_path = saved_source if os.path.isabs(saved_source) else os.path.join(base_dir, saved_source)
    if os.path.isfile(saved_path):
        video_path = os.path.abspath(saved_path)
        print(f"  [INFO] Using extraction source: {os.path.basename(video_path)}")
    else:
        print(f"  [WARN] Extraction source was not found: {saved_source}")
        with temporary_process_priority("normal"):
            selected = select_video_file(base_dir)
        if not selected:
            return
        video_path = selected if os.path.isabs(selected) else os.path.join(base_dir, selected)
        video_path = os.path.abspath(video_path)

    if os.path.normcase(os.path.basename(saved_source)) != os.path.normcase(os.path.basename(video_path)):
        print(f"  [WARN] These frames were extracted from: {saved_source}")
        print(f"  Selected subtitle source: {os.path.basename(video_path)}")
        with temporary_process_priority("normal"):
            confirm_source = input("  Rebuild subtitles using this different source? (y/n): ").strip().lower()
        if confirm_source != "y":
            print("  Finalize cancelled.")
            return

    print("\n  ── Finalize Manual Changes ─────────────────────────")
    print("  1. Repair and renumber surviving frames.")
    print("  2. Keep each surviving frame's original source timestamp.")
    print("  3. Rebuild commentsubs/ from the corrected frames.")
    print("  This does not rerun extraction, censorship, or video reconstruction.")

    if not repair_frames(base_dir):
        return

    try:
        survivors, updated_timestamp_data = _read_frame_timeline(base_dir)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"\n  [ERROR] Finalize cannot safely rebuild commentsubs: {exc}")
        print("  The frame repair may have completed, but subtitle output was left untouched.")
        return

    sub_stream = int(CONFIG.get("SUBTITLE_STREAM", -1))
    if sub_stream < 0:
        sub_stream = find_english_sub_stream(video_path)

    staging_root = tempfile.mkdtemp(prefix=".efioex7_finalize_", dir=base_dir)
    staged_commentsubs = os.path.join(staging_root, "commentsubs")
    os.makedirs(staged_commentsubs, exist_ok=True)
    try:
        success = burn_subtitles(
            base_dir,
            video_path,
            survivors,
            sub_stream,
            False,
            output_dir=staged_commentsubs,
            work_dir=staging_root,
        )
        if not success:
            print("  [ERROR] commentsubs/ was not replaced. The previous folder was preserved.")
            return

        commentsubs_path = os.path.join(base_dir, "commentsubs")
        backup_path = os.path.join(base_dir, ".efioex7_commentsubs_backup")
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path, ignore_errors=True)
        had_previous = os.path.exists(commentsubs_path)
        if had_previous:
            os.replace(commentsubs_path, backup_path)
        try:
            os.replace(staged_commentsubs, commentsubs_path)
        except Exception:
            if had_previous and os.path.exists(backup_path):
                os.replace(backup_path, commentsubs_path)
            raise
        if os.path.isdir(backup_path):
            shutil.rmtree(backup_path, ignore_errors=True)

        generated_count = sum(
            1 for name in os.listdir(commentsubs_path)
            if re.match(r"^frame_\d+\.jpg$", name, re.IGNORECASE)
        )
        print("\n  [SUCCESS] Finalize completed.")
        print(f"  Frames: {len(survivors):,} | commentsubs images: {generated_count:,}")
        print("  Manual frame edits are now reflected in commentsubs/.")
    except (OSError, RuntimeError) as exc:
        print(f"  [ERROR] Could not replace commentsubs/: {exc}")
        print("  The previous commentsubs folder was preserved where possible.")
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

def find_video_files(base_dir: str) -> list[str]:
    """Finds supported video files in the script folder.
    Returns matching filenames in sorted order."""
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

def probe_video_timing(video_path: str) -> tuple[float, str]:
    """Reads the source video's duration and frame rate.
    Returns the values needed to rebuild its original playback timeline."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,duration,avg_frame_rate,r_frame_rate:format=duration",
            "-of", "json", video_path
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe could not read the selected video.")

    data = json.loads(result.stdout or "{}")
    video_stream = next(
        (stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"),
        None
    )
    if video_stream is None:
        raise RuntimeError("The selected file does not contain a video stream.")

    duration_value = video_stream.get("duration") or data.get("format", {}).get("duration")
    try:
        duration = float(duration_value)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        raise RuntimeError("Could not determine the source video's duration.")

    fps_value = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "24/1"
    try:
        fps = Fraction(fps_value)
        if fps <= 0 or float(fps) > 240:
            raise ValueError
    except (ValueError, ZeroDivisionError):
        fps = Fraction(24, 1)

    return duration, f"{fps.numerator}/{fps.denominator}"


def reconstruct_video(base_dir: str, video_file: str) -> None:
    """Rebuilds a playable video from processed frames and precise timestamps.
    Reuses the source video's frame rate, duration, and audio track."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        print("  [ERROR] FFmpeg and FFprobe must both be available in PATH.")
        return

    video_path = video_file if os.path.isabs(video_file) else os.path.join(base_dir, video_file)
    video_path = os.path.abspath(video_path)
    frames_dir = os.path.join(base_dir, "frames")
    timestamp_path = os.path.join(base_dir, "frame_timestamps.json")

    if not os.path.isfile(video_path):
        print(f"  [ERROR] Source video not found: {sanitize_path(video_path)}")
        return
    if not os.path.isdir(frames_dir):
        print("  [ERROR] frames/ was not found. Run extraction first.")
        return
    if not os.path.isfile(timestamp_path):
        print("  [ERROR] frame_timestamps.json was not found.")
        print("  Run extraction with this version first to save precise frame timing.")
        return

    try:
        survivors, timestamp_data = _read_frame_timeline(base_dir)
        frame_paths = [
            os.path.join(frames_dir, f"frame_{item['final_number']}.jpg")
            for item in survivors
        ]
        frame_times = [float(item["time_s"]) for item in survivors]
        saved_source = timestamp_data.get("source_video") if isinstance(timestamp_data, dict) else None

        if saved_source and os.path.normcase(os.path.basename(saved_source)) != os.path.normcase(os.path.basename(video_path)):
            print(f"  [WARN] These frames were extracted from: {saved_source}")
            print(f"  Selected source video: {os.path.basename(video_path)}")
            with temporary_process_priority("normal"):
                confirm = input("  Use this source anyway? (y/n): ").strip().lower()
            if confirm != "y":
                print("  Reconstruction cancelled. Returning to the main menu.")
                return

        source_duration, output_fps = probe_video_timing(video_path)
        last_time = frame_times[-1]
        if source_duration <= last_time:
            intervals = [b - a for a, b in zip(frame_times, frame_times[1:]) if b > a]
            fallback_interval = sorted(intervals)[len(intervals) // 2] if intervals else 1.0 / float(Fraction(output_fps))
            source_duration = last_time + fallback_interval

        output_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(base_dir, f"{output_stem}_efioex7_preview.mp4")
        if os.path.exists(output_path):
            overwrite = input(f"  {os.path.basename(output_path)} already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != "y":
                print("  Reconstruction cancelled.")
                return

        manifest_path = os.path.join(base_dir, "efioex7_reconstruction.ffconcat")
        black_path = os.path.join(base_dir, "efioex7_reconstruction_black.jpg")
        manifest_lines = ["ffconcat version 1.0"]

        if frame_times[0] > 0.000001:
            first_image = cv2.imread(frame_paths[0])
            if first_image is None:
                raise RuntimeError("Could not read the first generated frame.")
            if not cv2.imwrite(black_path, np.zeros_like(first_image)):
                raise RuntimeError("Could not create the opening timeline filler frame.")
            manifest_lines.extend([
                "file 'efioex7_reconstruction_black.jpg'",
                "option framerate 1000000",
                f"duration {frame_times[0]:.9f}"
            ])

        manifest_progress = ProgressLine("Preview timing map", indent="  ")
        for index, (frame_path, frame_time) in enumerate(zip(frame_paths, frame_times)):
            if index + 1 < len(frame_times):
                frame_duration = frame_times[index + 1] - frame_time
            else:
                frame_duration = source_duration - frame_time
            frame_duration = max(0.000001, frame_duration)
            frame_name = os.path.relpath(frame_path, base_dir).replace("\\", "/")
            manifest_lines.extend([
                f"file '{frame_name}'",
                "option framerate 1000000",
                f"duration {frame_duration:.9f}"
            ])
            manifest_progress.update(index + 1, len(frame_paths), detail="preparing frame timestamps")

        last_frame_name = os.path.relpath(frame_paths[-1], base_dir).replace("\\", "/")
        manifest_lines.extend([
            f"file '{last_frame_name}'",
            "option framerate 1000000"
        ])
        with open(manifest_path, "w", encoding="utf-8", newline="\n") as file:
            file.write("\n".join(manifest_lines) + "\n")
        manifest_progress.finish(f"{len(frame_paths):,} frame timings prepared")

        print("\n  Reconstructing video from processed frames...")
        print(f"  Frames: {len(frame_paths):,} | Source duration: {source_duration:.3f}s | Output FPS: {output_fps}")
        encoder_setting = CONFIG.get("PREVIEW_ENCODER", "auto").lower()
        ffmpeg_encoders = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True
        )
        nvenc_available = "h264_nvenc" in (ffmpeg_encoders.stdout or "")
        nvidia_present = detect_gpu_capabilities().get("has_nvidia", False)
        try_nvenc = encoder_setting in ("auto", "nvenc") and nvenc_available and nvidia_present
        if encoder_setting == "nvenc" and not (nvenc_available and nvidia_present):
            print("  [WARN] NVENC or NVIDIA hardware is unavailable; using libx264.")
        elif try_nvenc:
            print("  Preview encoder: NVIDIA NVENC (hardware)")
        else:
            print("  Preview encoder: libx264 (CPU)")

        encoder_attempts = (["h264_nvenc", "libx264"] if try_nvenc else ["libx264"])
        result = None
        for encoder_name in encoder_attempts:
            for audio_name in ("copy", "aac"):
                if os.path.exists(output_path):
                    os.remove(output_path)
                command = [
                    ffmpeg, "-hide_banner", "-y",
                    "-f", "concat", "-safe", "0", "-i", manifest_path,
                    "-i", video_path,
                    "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1",
                    "-r", output_fps, "-t", f"{source_duration:.9f}",
                ]
                if encoder_name == "h264_nvenc":
                    command.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", "0"])
                else:
                    command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])
                command.extend([
                    "-pix_fmt", "yuv420p", "-video_track_timescale", "90000",
                    "-c:a", audio_name,
                ])
                if audio_name == "aac":
                    command.extend(["-b:a", "192k"])
                command.extend(["-movflags", "+faststart", output_path])

                label = f"Preview rendering ({encoder_name}, audio {audio_name})"
                if encoder_name == "h264_nvenc":
                    with temporary_process_priority("above_normal"):
                        result = run_ffmpeg_with_progress(command, source_duration, label, cwd=base_dir)
                else:
                    result = run_ffmpeg_with_progress(command, source_duration, label, cwd=base_dir)

                if result.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                    break
                if os.path.exists(output_path):
                    os.remove(output_path)
                if encoder_name == "h264_nvenc" and audio_name == "copy":
                    print("  [WARN] NVENC attempt failed; retrying with AAC audio before CPU fallback.")
                elif audio_name == "copy":
                    print("  [WARN] Audio stream copy failed; retrying with AAC.")
                elif encoder_name == "h264_nvenc":
                    print("  [WARN] NVENC failed; falling back to software libx264.")
            if result and result.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                break

        if result.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            print("\n  [SUCCESS] Reconstructed video created:")
            print(f"  {sanitize_path(output_path)}")
            print("  This preview does not modify or participate in the extraction folders.")
            print("  Processed frames and original audio are synchronized on the source timeline.")
        else:
            print("  [ERROR] FFmpeg could not finish the reconstructed video.")
            if os.path.exists(output_path):
                os.remove(output_path)

    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"  [ERROR] Reconstruction failed: {exc}")
        print("  Returning to the main menu.")
    finally:
        for temporary_path in (
            os.path.join(base_dir, "efioex7_reconstruction.ffconcat"),
            os.path.join(base_dir, "efioex7_reconstruction_black.jpg")
        ):
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass


def select_video_file(base_dir: str) -> str | None:
    """Lets the user choose a video from the script folder.
    Supports manual paths and cancellation when needed."""
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

def _open_video_decoder(video_path: str, use_gpu: bool):
    """Opens the source with PyAV CUDA hardware decoding when supported.
    Verifies the first decoded frame before accepting NVDEC; otherwise uses CPU."""
    if use_gpu:
        container = None
        try:
            version_parts = tuple(int(x) for x in re.findall(r"\d+", getattr(av, "__version__", "0"))[:2])
            if version_parts < (17, 0):
                raise RuntimeError(
                    f"PyAV {getattr(av, '__version__', 'unknown')} is too old for safe CUDA decoding; "
                    "update with: py -m pip install --upgrade av"
                )

            from av.codec.hwaccel import HWAccel, hwdevices_available
            available_devices = {str(item).lower() for item in hwdevices_available()}
            if "cuda" not in available_devices:
                raise RuntimeError(
                    "the FFmpeg library bundled with PyAV was not built with CUDA hardware-device support"
                )

            hwaccel = HWAccel(device_type="cuda", allow_software_fallback=False)
            container = av.open(video_path, hwaccel=hwaccel)
            stream = container.streams.video[0]
            if not bool(getattr(stream.codec_context, "is_hwaccel", False)):
                raise RuntimeError("PyAV opened the stream, but the codec context is not hardware-accelerated")

            decoded = iter(container.decode(video=0))
            first_frame = next(decoded)
            return container, stream, chain((first_frame,), decoded), "GPU NVDEC (PyAV CUDA)"
        except StopIteration:
            if container is not None:
                container.close()
            raise RuntimeError("The source video contains no decodable video frames.")
        except Exception as exc:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            print(f"\n[WARN] NVDEC could not be activated ({exc}). Using CPU decoder for this video.")

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    stream.thread_count = os.cpu_count() or 4
    return container, stream, iter(container.decode(video=0)), "CPU (Multithreaded)"


def run_extraction(base_dir: str, video_file: str) -> None:
    """Runs the frame extraction and post-processing stages.
    Coordinates decoding, filtering, censorship, timestamps, and subtitles."""
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

    precise_timestamp_path = os.path.join(base_dir, "frame_timestamps.json")
    if os.path.exists(precise_timestamp_path):
        os.remove(precise_timestamp_path)

    global OPENCL_GPU_ACTIVE, _OPENCL_GPU_WARNED
    OPENCL_GPU_ACTIVE = False
    _OPENCL_GPU_WARNED = False
    gpu_info = detect_gpu_capabilities()
    accel_mode = CONFIG.get("ACCELERATION", "auto").lower()
    use_gpu = accel_mode in ("auto", "cuda", "gpu") and (
        gpu_info["has_nvidia"] or gpu_info["opencl_is_nvidia"]
    )
    OPENCL_GPU_ACTIVE = bool(
        use_gpu
        and CONFIG.get("OPENCL_PREPROCESSING", True)
        and gpu_info.get("opencl_is_nvidia", False)
        and gpu_info.get("opencl_ready", False)
    )

    print("\n" + "="*60)
    print("  efioex7 — Smart Streaming Frame Extractor")
    print("="*60)
    print(f"  Input Video:   {sanitize_path(video_file)}")
    print(f"  Resolution:    {CONFIG['OUTPUT_WIDTH']}px width")
    print(f"  RAM Buffer:    Up to {CONFIG['MAX_RAM_BUFFER_MB']} MB (Asynchronous Parallel Disk Writer)")
    if use_gpu:
        dev_desc = gpu_info["gpu_name"] if gpu_info["gpu_name"] else "NVIDIA OpenCL GPU"
        print(f"  Acceleration:  ENABLED ({dev_desc})")
        if OPENCL_GPU_ACTIVE:
            print(f"  OpenCL Image Ops: ENABLED ({gpu_info['opencl_device']})")
            print("  OpenCL stages:     Frame resize/downsample + blood/lewd color masks")
        else:
            print("  OpenCL Image Ops: CPU fallback (no NVIDIA OpenCL device or disabled)")
    else:
        print("  Acceleration:  CPU Multithreaded")
    print("-"*60)

    print("\n>>> STAGE 1: Chopping & Deduplicating Frames (Streaming Pass)...")
    print("    • Drops pitch-black frames")
    print("    • Drops frozen stills (protects blinks & mouth flaps)")
    print("    • Thins long camera pans (keeps Start 0%, Mid 50%, End 100%)")
    print(f"    • Caps black & text fade transitions (strictly {CONFIG['FADE_MAX_FRAMES']} frames max)")

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
    chop_progress = ProgressLine("Chopping", interval=0.5, indent="    ")

    def submit_survivor(item: dict) -> None:
        """Adds a selected frame to the asynchronous writer.
        Tracks its output number and timestamp."""
        final_idx = len(survivors) + 1
        item["final_number"] = final_idx
        out_path = os.path.join(frames_dir, f"frame_{final_idx}.jpg")

        async_writer.submit(out_path, item["bgr"])

        survivors.append({
            "final_number": final_idx,
            "time_s": item["time_s"]
        })
        timestamps[str(final_idx)] = format_timestamp(item["time_s"] or 0.0)

        item["bgr"] = None
        item["small"] = None

    container, stream, decoded_frames, decoder_info = _open_video_decoder(video_path, use_gpu)
    with container:
        print(f"    Active Decoder: {decoder_info}\n")

        nvdec_active = decoder_info.startswith("GPU NVDEC")
        while True:
            try:
                if nvdec_active:
                    with temporary_process_priority("above_normal"):
                        frame = next(decoded_frames)
                else:
                    frame = next(decoded_frames)
            except StopIteration:
                break
            processed += 1
            now = time.perf_counter()
            elapsed = now - t0
            fps = processed / elapsed if elapsed > 0 else 0
            chop_progress.update(
                processed,
                detail=(
                    f"frames decoded | Survivors: {len(survivors):,} | "
                    f"Queue: {async_writer.queue.qsize()}/{max_queue_frames} | {fps:.1f} fps"
                ),
            )

            bgr = frame.to_ndarray(format="bgr24")
            h, w = bgr.shape[:2]
            time_s = frame.time or 0.0

            out_w = CONFIG["OUTPUT_WIDTH"]
            out_h = int(round(h * out_w / w / 2)) * 2
            bgr, small = preprocess_stage1_frame(bgr, out_w, out_h)

            if is_truly_black(small):
                del bgr, small, frame
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
                del bgr, frame
                continue

            max_cell, overall_diff = analyze_grid_difference(prev_small, small)

            if max_cell < CONFIG["STILL_DIFF_THRESHOLD"]:
                prev_small = small
                del bgr, frame
                continue

            is_micro_anim = (max_cell >= 12.0) and (overall_diff < 4.0)

            is_fade = False
            if not is_micro_anim:
                is_fade = is_fading_transition(prev_small, small)

            if is_fade:
                fade_buffer.add(frame_data)
                prev_small = small
                del bgr, frame
                continue
            else:
                if not fade_buffer.is_empty():
                    for ff in fade_buffer.flush():
                        submit_survivor(ff)

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

            del bgr, frame

            if processed % 1000 == 0:
                gc.collect()

        chop_progress.finish(f"{processed:,} frames decoded | {len(survivors):,} survivors")

        if not fade_buffer.is_empty():
            for ff in fade_buffer.flush():
                submit_survivor(ff)

        if not pan_buffer.is_empty():
            for pf in pan_buffer.flush():
                submit_survivor(pf)

    print("    • Flushing buffered frames to disk...")
    writer_progress = ProgressLine("Writing frames", indent="    ")
    async_writer.wait_completion(writer_progress)

    ts_path = os.path.join(base_dir, "timestamps.json")
    with open(ts_path, "w", encoding="utf-8") as f:
        json.dump(timestamps, f, indent=2)

    stage1_time = time.perf_counter() - t0
    print(f"\n[INFO] Stage 1 Complete in {stage1_time/60:.1f} min! Processed: {processed} frames -> Initial Survivors: {len(survivors)}")

    if CONFIG["ENABLE_LOOP_FILTER"]:
        survivors = filter_cyclical_loops(base_dir, survivors, frames_dir, CONFIG["LOOP_WINDOW_SIZE"])

    precise_timestamp_data = {
        "source_video": os.path.basename(video_path),
        "frames": [
            {"frame": item["final_number"], "time_s": float(item["time_s"] or 0.0)}
            for item in survivors
        ]
    }
    with open(precise_timestamp_path, "w", encoding="utf-8") as f:
        json.dump(precise_timestamp_data, f, indent=2)

    blood_count = 0
    lewd_count = 0

    if CONFIG["ENABLE_BLOOD_CENSOR"] or CONFIG["ENABLE_LEWD_CENSOR"]:
        print("\n>>> STAGE 3: Applying Safety Filters on Survivors...")
        print(f"    • Blood cascade: gate >={CONFIG['BLOOD_THRESHOLD_PERCENT']}% crimson | instant >={CONFIG['BLOOD_HIGH_PERCENT']}% | WD 'blood' tag >={CONFIG['BLOOD_TAG_CONFIDENCE']}")
        print(f"    • Lewd threshold:  >={CONFIG['LEWD_THRESHOLD_PERCENT']}% body skin (blur)")

        t_filter = time.perf_counter()
        safety_progress = ProgressLine("Safety check", indent="    ")
        total_safety_frames = len(survivors)
        for idx in range(1, len(survivors) + 1):
            safety_progress.update(
                idx - 1,
                total_safety_frames,
                detail=f"Checking frame {idx:,} | Blood greyed: {blood_count} | Lewd blurred: {lewd_count}",
            )
            frame_path = os.path.join(frames_dir, f"frame_{idx}.jpg")
            if not os.path.exists(frame_path):
                safety_progress.update(
                    idx,
                    total_safety_frames,
                    detail=f"Blood greyed: {blood_count} | Lewd blurred: {lewd_count}",
                )
                continue

            frame_bgr = cv2.imread(frame_path)
            if frame_bgr is None:
                safety_progress.update(
                    idx,
                    total_safety_frames,
                    detail=f"Blood greyed: {blood_count} | Lewd blurred: {lewd_count}",
                )
                continue

            was_censored = False
            original_bgr = frame_bgr.copy()

            is_blood, frame_bgr = detect_and_censor_blood(frame_bgr, CONFIG["BLOOD_THRESHOLD_PERCENT"], base_dir)
            if is_blood:
                was_censored = True
                blood_count += 1

            is_lewd, frame_bgr = detect_and_censor_lewd(frame_bgr, CONFIG["LEWD_THRESHOLD_PERCENT"])
            if is_lewd:
                was_censored = True
                lewd_count += 1

            if was_censored:
                if CONFIG["SAVE_CENSORED_REVIEW"]:
                    rev_path = os.path.join(review_dir, f"frame_{idx}.jpg")
                    write_jpeg(rev_path, original_bgr, CONFIG["QUALITY_CLEAN"])
                write_jpeg(frame_path, frame_bgr, CONFIG["QUALITY_CLEAN"])

            safety_progress.update(
                idx,
                total_safety_frames,
                detail=f"Blood greyed: {blood_count} | Lewd blurred: {lewd_count}",
            )

        safety_progress.finish(
            f"{total_safety_frames:,} frames checked | Blood greyed: {blood_count} | Lewd blurred: {lewd_count}"
        )
        filter_elapsed = time.perf_counter() - t_filter
        print(f"[INFO] Stage 3 Complete in {filter_elapsed:.1f}s! Censored: {blood_count} blood, {lewd_count} lewd.")
        print(f"    Blood cascade breakdown: {BLOOD_STATS['gate_clear']} cleared at color gate | "
              f"{BLOOD_STATS['lighting_clear']} cleared by lighting vote | "
              f"{BLOOD_STATS['tagger_clear']} cleared by WD tagger | "
              f"{BLOOD_STATS['tagger_flag']} flagged by WD tagger | "
              f"{BLOOD_STATS['instant']} instant-grey | "
              f"{BLOOD_STATS['degraded_flag']} flagged (tagger offline) | "
              f"{BLOOD_STATS['continuity']} flagged by continuity")
    else:
        print("\n[INFO] Stage 3 Skipped (Safety filters disabled in config).")

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
        print(f"  Censored Review:     {sanitize_path(review_dir)}/ ({blood_count} blood, {lewd_count} lewd to review)")
    print("="*60 + "\n")

def _probe_onnx_providers() -> tuple[set[str], str]:
    """Checks ONNX Runtime providers in a clean child process.
    Avoids importing its native DLLs into the menu process before package repair."""
    probe_code = (
        "import onnxruntime as ort; "
        "getattr(ort, 'preload_dlls', lambda: None)(); "
        "print(','.join(ort.get_available_providers()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe_code],
        capture_output=True,
        text=True,
        creationflags=current_subprocess_priority_flags(),
    )
    if result.returncode != 0:
        return set(), (result.stderr or result.stdout or "ONNX Runtime import failed").strip()
    return {x.strip() for x in result.stdout.strip().split(",") if x.strip()}, ""


def _pyav_cuda_devices() -> tuple[list[str], str]:
    """Reports hardware devices compiled into PyAV's FFmpeg library.
    CUDA must be listed here before PyAV can use NVDEC."""
    try:
        from av.codec.hwaccel import hwdevices_available
        return [str(x) for x in hwdevices_available()], ""
    except Exception as exc:
        return [], str(exc)


def mode_gpu_runtime_setup() -> None:
    """Checks NVIDIA inference and PyAV decoder support.
    Offers to replace CPU-only ONNX Runtime with the official CUDA package."""
    print("\n  ── GPU Runtime Diagnostics / Setup ─────────────────")
    gpu = detect_gpu_capabilities()
    if gpu.get("has_nvidia"):
        print(f"    NVIDIA GPU: {gpu['gpu_name']}")
    else:
        print("    NVIDIA GPU: not detected by nvidia-smi")

    providers, provider_error = _probe_onnx_providers()
    if providers:
        print(f"    ONNX Runtime providers: {', '.join(sorted(providers))}")
        if "CUDAExecutionProvider" in providers:
            print("    WD Tagger CUDA: available (model session will verify it during Stage 3)")
        else:
            print("    WD Tagger CUDA: unavailable; CPU-only runtime is currently installed")
    else:
        print(f"    ONNX Runtime: unavailable ({provider_error})")

    hw_devices, hw_error = _pyav_cuda_devices()
    if "cuda" in {item.lower() for item in hw_devices}:
        print("    PyAV CUDA hardware decoding: compiled in")
    else:
        detail = f" ({hw_error})" if hw_error else f" (available devices: {', '.join(hw_devices) or 'none'})"
        print(f"    PyAV CUDA hardware decoding: not compiled in{detail}")
        print("    The current PyAV decoder therefore cannot activate NVDEC in this environment.")
        print("    This is separate from ONNX Runtime CUDA and requires a CUDA-capable decode backend.")

    if "CUDAExecutionProvider" in providers:
        print("\n[INFO] ONNX Runtime already exposes CUDAExecutionProvider.")
        cfg = load_config(get_script_dir())
        if cfg.get("ACCELERATION", "auto").lower() != "cpu":
            cfg["WD_TAGGER_DEVICE"] = "cuda"
            save_config(get_script_dir(), cfg)
            CONFIG.update(cfg)
            print("[INFO] WD Tagger device mode set to CUDA-only in config.txt.")
        input("  Press Enter to return to the main menu...")
        return

    if not gpu.get("has_nvidia"):
        print("\n[WARN] NVIDIA GPU was not detected. CUDA setup was not attempted.")
        input("  Press Enter to return to the main menu...")
        return

    print("\n  The current ONNX Runtime installation cannot expose CUDAExecutionProvider.")
    print("  This setup replaces the CPU-only onnxruntime package with the official GPU package.")
    print("  Only one of onnxruntime and onnxruntime-gpu should be installed in this Python environment.")
    answer = input("  Install/repair the CUDA-enabled runtime now? (y/n): ").strip().lower()
    if answer != "y":
        print("  No packages were changed.")
        input("  Press Enter to return to the main menu...")
        return

    print("\n[INFO] Removing conflicting ONNX Runtime packages...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"],
        check=False,
        creationflags=current_subprocess_priority_flags(),
    )
    print('\n[INFO] Installing ONNX Runtime GPU with CUDA/cuDNN dependencies...')
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "onnxruntime-gpu[cuda,cudnn]"],
        check=False,
        creationflags=current_subprocess_priority_flags(),
    )

    if install.returncode != 0:
        print("\n[ERROR] GPU runtime installation failed. Restoring the CPU runtime...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "onnxruntime"],
            check=False,
            creationflags=current_subprocess_priority_flags(),
        )
        input("  Press Enter to return to the main menu...")
        return

    providers, provider_error = _probe_onnx_providers()
    if "CUDAExecutionProvider" in providers:
        print("\n[SUCCESS] CUDAExecutionProvider is now available.")
        cfg = load_config(get_script_dir())
        if cfg.get("ACCELERATION", "auto").lower() != "cpu":
            cfg["WD_TAGGER_DEVICE"] = "cuda"
            save_config(get_script_dir(), cfg)
            CONFIG.update(cfg)
            print("[INFO] WD Tagger device mode set to CUDA-only in config.txt.")
        print("[INFO] Restart efioex7 before starting a new extraction.")
    else:
        print("\n[WARN] GPU package installed, but CUDA provider verification failed.")
        print(f"       {provider_error or ', '.join(sorted(providers)) or 'No providers returned.'}")
        print("       Check NVIDIA driver and Visual C++ runtime compatibility.")
        print("       CPU runtime was not automatically reinstalled because the GPU package installed successfully.")
    input("  Press Enter to return to the main menu...")


def mode_settings(base_dir: str) -> None:
    """Displays and updates the extractor configuration.
    Changes are saved to config.txt and reloaded by the main menu."""
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
        print(f"    [B] Blood Censorship:           {cfg['ENABLE_BLOOD_CENSOR']} (Gate: {cfg['BLOOD_THRESHOLD_PERCENT']}% | Instant: {cfg['BLOOD_HIGH_PERCENT']}% | WD: {cfg['WD_TAGGER_DEVICE']} / {cfg['BLOOD_TAG_CONFIDENCE']})")
        print(f"    [C] Lewd Censorship:            {cfg['ENABLE_LEWD_CENSOR']} (Threshold: {cfg['LEWD_THRESHOLD_PERCENT']}%)")
        print(f"    [D] OpenCL Image Operations:       {cfg['OPENCL_PREPROCESSING']}")
        print(f"    [E] Preview Encoder:            {cfg['PREVIEW_ENCODER']} (auto/nvenc/cpu)")
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
            print("  Blood cascade: [T]oggle censor | [C] gate % | [H] instant-grey %")
            print("  WD Tagger: [V] confidence | [G] toggle | [D] device (auto/cuda/cpu) | [W] lighting vote")
            sub_opt = input("  Choose: ").strip().lower()
            pct_keys = {"c": "BLOOD_THRESHOLD_PERCENT", "h": "BLOOD_HIGH_PERCENT", "v": "BLOOD_TAG_CONFIDENCE"}
            if sub_opt in pct_keys:
                key = pct_keys[sub_opt]
                v = input(f"  Enter new value for {key} (current: {cfg[key]}): ").strip()
                if v:
                    try:
                        cfg[key] = float(v)
                        save_config(base_dir, cfg)
                    except ValueError:
                        print("  Invalid number.")
            elif sub_opt == "t":
                cfg["ENABLE_BLOOD_CENSOR"] = not cfg["ENABLE_BLOOD_CENSOR"]
                save_config(base_dir, cfg)
            elif sub_opt == "g":
                cfg["BLOOD_WD_TAGGER"] = not cfg["BLOOD_WD_TAGGER"]
                save_config(base_dir, cfg)
            elif sub_opt == "w":
                cfg["BLOOD_TAGGER_VARIANTS"] = not cfg["BLOOD_TAGGER_VARIANTS"]
                save_config(base_dir, cfg)
            elif sub_opt == "d":
                current = cfg.get("WD_TAGGER_DEVICE", "auto").lower()
                options = {"auto": "cuda", "cuda": "cpu", "cpu": "auto"}
                cfg["WD_TAGGER_DEVICE"] = options.get(current, "auto")
                save_config(base_dir, cfg)
                print(f"  WD Tagger device set to: {cfg['WD_TAGGER_DEVICE']}")
        elif choice == "d":
            cfg["OPENCL_PREPROCESSING"] = not cfg["OPENCL_PREPROCESSING"]
            save_config(base_dir, cfg)
            print(f"  OpenCL preprocessing set to: {cfg['OPENCL_PREPROCESSING']}")
        elif choice == "e":
            current = cfg.get("PREVIEW_ENCODER", "auto").lower()
            options = {"auto": "nvenc", "nvenc": "cpu", "cpu": "auto"}
            cfg["PREVIEW_ENCODER"] = options.get(current, "auto")
            save_config(base_dir, cfg)
            print(f"  Preview encoder set to: {cfg['PREVIEW_ENCODER']}")
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

def main() -> None:
    """Loads settings and starts the interactive menu.
    Routes user choices to extraction, configuration, or exit."""
    base_dir = get_script_dir()
    global CONFIG
    CONFIG = load_config(base_dir)
    priority_ready = set_process_priority("normal")
    if os.name == "nt":
        if priority_ready:
            print("[INFO] Dynamic priority: Idle=Normal | CPU=High | GPU decode=Above Normal")
        else:
            print("[WARN] Dynamic process priority could not be enabled; Windows defaults will apply.")

    cli_video = None
    if len(sys.argv) > 1 and sys.argv[1].endswith((".mkv", ".mp4", ".webm", ".avi")):
        cli_video = sys.argv[1]

    while True:
        set_process_priority("normal")
        gpu = detect_gpu_capabilities()
        gpu_status = f"GPU: {gpu['gpu_name']}" if gpu["has_nvidia"] else "CPU Mode"
        found_videos = find_video_files(base_dir)
        vid_count_str = f"{len(found_videos)} video(s) found in folder" if found_videos else "No MKVs found in folder"

        print("\n  ╔══════════════════════════════════════════════════════╗")
        print("  ║         efioex7 — Smart Anime Frame Extractor        ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        print(f"   Status: {vid_count_str}  |  {gpu_status}\n")
        print("    [1] Start Streaming Extraction")
        print("    [2] Settings")
        print("    [3] Reconstruct Preview Video")
        print("    [4] Finalize Manual Frame Changes")
        print("    [5] GPU Runtime Setup / Diagnostics")
        print("    [6] Exit\n")

        choice = input("  Enter 1, 2, 3, 4, 5, or 6: ").strip()
        if choice == "1":
            target = cli_video if cli_video else select_video_file(base_dir)
            if target:
                run_with_process_priority("high", run_extraction, base_dir, target)
        elif choice == "2":
            mode_settings(base_dir)
            CONFIG = load_config(base_dir)
        elif choice == "3":
            target = cli_video if cli_video else select_video_file(base_dir)
            if target:
                run_with_process_priority("high", reconstruct_video, base_dir, target)
        elif choice == "4":
            run_with_process_priority("high", finalize_manual_changes, base_dir)
        elif choice == "5":
            mode_gpu_runtime_setup()
        elif choice == "6":
            set_process_priority("normal")
            print("  Bye!")
            sys.exit(0)
        else:
            print("  Invalid choice.")

if __name__ == "__main__":
    main()
