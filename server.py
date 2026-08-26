#!/usr/bin/env python3
"""Simple HTTP server for H3 Studio web UI.

Serves the frontend and provides an API to run h3-metal generation.

Usage:
  python3 server.py [--port 8080] [--model-dir PATH]
"""

import argparse
import cgi
import hashlib
import json
import math
import os
import queue
import re
import signal
import stat
import struct
import shutil
import subprocess
import sys
import time
import http.server
import urllib.parse
import threading
import uuid
from pathlib import Path

HERE = Path(__file__).parent
BASE_MODEL_DIR = HERE / "MiniMax-H3"
TURBO_MODEL_DIR = BASE_MODEL_DIR / "turbo-v4-step600-ema"
LORA_ROOT = Path(os.environ.get("H3_LORA_DIR", BASE_MODEL_DIR / "loras"))
LORA_BAKED = LORA_ROOT / "baked"
LORA_RUNTIME = LORA_ROOT / "runtime"
OUTPUTS = HERE / "outputs"
OUTPUTS.mkdir(exist_ok=True)
IMAGE_OUTPUTS = OUTPUTS / "images"
IMAGE_OUTPUTS.mkdir(exist_ok=True)
UPLOADS = HERE / "uploads"
UPLOADS.mkdir(exist_ok=True)
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
REF2VA_MODES = {"ref-image", "ref-video", "ref-audio", "multi-ref"}
SENSENOVA_DIR = Path(os.environ.get(
    "SENSENOVA_DIR", "/Users/lrb/models/SenseNova-U1.5-8B-MoT"
)).resolve()
SENSENOVA_PYTHON = SENSENOVA_DIR / ".venv-macos" / "bin" / "python"
SENSENOVA_VARIANTS = {
    "Q4_0": "SenseNova-U1.5-8B-MoT-Final-Q4_0.gguf",
    "Q5_0": "SenseNova-U1.5-8B-MoT-Final-Q5_0.gguf",
    "Q6_K": "SenseNova-U1.5-8B-MoT-Final-Q6_K.gguf",
    "Q8_0": "SenseNova-U1.5-8B-MoT-Final-Q8_0.gguf",
}

PROGRESS_LINE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9 ]*?)\s+(\d+)\s*/\s*(\d+)\s*$"
)
PHASE_PROGRESS = {
    "LoRA inspect": (1.0, 4.0, "校验 LoRA"),
    "tokenizer": (1.0, 3.0, "解析提示词"),
    "audio VAE encoder": (3.0, 12.0, "编码参考音频"),
    "video VAE encoder": (3.0, 12.0, "编码参考画面"),
    "Qwen vision": (8.0, 18.0, "理解参考画面"),
    "text encoder": (12.0, 24.0, "编码文本条件"),
    "refine text": (20.0, 26.0, "优化文本条件"),
    "precompute AdaLN": (24.0, 29.0, "准备扩散参数"),
    "LoRA merge": (26.0, 40.0, "合并 LoRA 权重"),
    "load transformer core": (29.0, 40.0, "加载生成模型"),
    # Metal's default sampler enqueues commands before the GPU finishes them.
    # Keep this range deliberately narrow; refresh_running_progress advances
    # the long, silent GPU portion using elapsed time and the job estimate.
    "denoise enqueue": (40.0, 46.0, "GPU 降噪"),
    "denoise": (82.0, 84.0, "GPU 降噪完成"),
    "audio VAE": (84.0, 89.0, "生成音频"),
    "preview VAE load": (89.0, 93.0, "加载预览解码器"),
    "video VAE load": (89.0, 93.0, "加载视频解码器"),
    "video VAE decode": (93.0, 97.0, "解码视频"),
    "FFmpeg": (97.0, 99.5, "封装 MP4"),
}

# Generation state shared by the request handler and worker thread.
job_lock = threading.Lock()
current_job = None
job_results = {}
lora_catalog_lock = threading.Lock()
lora_catalog_cache = {}


def resolve_media_tool(env_name, program):
    """Find Homebrew media tools even under launchd's minimal PATH."""
    override = os.environ.get(env_name)
    if override:
        return override
    discovered = shutil.which(program)
    if discovered:
        return discovered
    for directory in (Path("/opt/homebrew/bin"), Path("/usr/local/bin")):
        candidate = directory / program
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return program


def estimate_job_seconds(params, mode="text"):
    """Return a conservative M5-oriented runtime estimate for UI feedback."""
    width = max(1, int(params.get("width") or 864))
    height = max(1, int(params.get("height") or 480))
    frames = max(1, int(params.get("frames") or 56))
    steps = max(1, int(params.get("steps") or 20))
    layers = max(1, int(params.get("layers") or 45))
    reuse = max(1, int(params.get("reuse") or 2))
    evaluations = max(1, math.ceil(steps / reuse))

    pixels_frames = width * height * frames
    denoise_work = pixels_frames * evaluations * (layers / 45.0)
    # A persistent worker reuses the VideoVAE and exact-request DiT state, but
    # a different prompt can still invalidate the prepared DiT. Keep the
    # conservative cold/different-request allowance until runtime history is
    # available for a matching request.
    seconds = 78.0 + denoise_work / 1_800_000.0 + pixels_frames / 2_000_000.0
    if mode == "frame-anchor":
        seconds += 8.0
    elif mode in REF2VA_MODES:
        seconds += 18.0
    return round(max(20.0, seconds), 1)


def normalize_video_params(params, defaults=None):
    """Validate UI values before mutating the persistent worker session."""
    defaults = defaults or {
        "width": 864,
        "height": 480,
        "frames": 56,
        "steps": 20,
        "layers": 45,
        "reuse": 2,
        "seed": 42,
    }
    def integer(name, default, minimum, maximum):
        raw = params.get(name, default)
        if raw is None or raw == "":
            raw = default
        if isinstance(raw, bool):
            raise ValueError(f"{name} must be an integer")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"{name} must be an integer")
        if value < minimum or value > maximum:
            raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        return value

    def boolean(name):
        raw = params.get(name, False)
        if isinstance(raw, bool):
            return raw
        if raw in (0, 1):
            return bool(raw)
        raise ValueError(f"{name} must be a boolean")

    normalized = {
        "width": integer("width", defaults.get("width", 864), 32, 768 * 1344),
        "height": integer("height", defaults.get("height", 480), 32, 768 * 1344),
        "frames": integer("frames", defaults.get("frames", 56), 5, 362),
        "steps": integer("steps", defaults.get("steps", 20), 2, 1000),
        "layers": integer("layers", defaults.get("layers", 45), 35, 50),
        "reuse": integer("reuse", defaults.get("reuse", 2), 1, 3),
        "seed": integer("seed", defaults.get("seed", 42), 0, (1 << 64) - 1),
        "ssd_streaming": boolean("ssd_streaming"),
        "int8_row_fc2": boolean("int8_row_fc2"),
        "match_input_aspect": boolean("match_input_aspect"),
    }
    width, height = normalized["width"], normalized["height"]
    if width % 32 or height % 32:
        raise ValueError("width and height must be multiples of 32")
    if width * height > 768 * 1344:
        raise ValueError("canvas exceeds the released 768*1344 pixel limit")
    frames = normalized["frames"]
    aligned_frames = frames if (frames - 5) % 17 == 0 else frames + 17 - (frames - 5) % 17
    if aligned_frames > 362:
        raise ValueError("frames must align within the released 5..362 range")
    if normalized["ssd_streaming"] and normalized["int8_row_fc2"]:
        raise ValueError("SSD streaming cannot be combined with int8 row FC2")
    return normalized


def legal_canvas_for_aspect(source_width, source_height, target_pixels,
                            max_pixels=768 * 1344, max_edge=1344):
    """Choose a legal 32-pixel canvas closest to the source aspect and area."""
    if source_width < 1 or source_height < 1 or target_pixels < 32 * 32:
        raise ValueError("invalid source dimensions for aspect matching")
    source_ratio = source_width / source_height
    target_pixels = min(max_pixels, target_pixels)
    best = None
    for candidate_width in range(32, max_edge + 1, 32):
        for candidate_height in range(32, max_edge + 1, 32):
            pixels = candidate_width * candidate_height
            if pixels > max_pixels:
                continue
            ratio_error = abs(math.log(
                (candidate_width / candidate_height) / source_ratio
            ))
            area_error = abs(math.log(pixels / target_pixels))
            score = ratio_error * 100.0 + area_error
            candidate = (score, area_error, -pixels,
                         candidate_width, candidate_height)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("cannot find a legal output canvas")
    return best[3], best[4]


def primary_image_reference(mode, files):
    """Return the image that controls output aspect for an image workflow."""
    keys = {
        "frame-anchor": ("first_frame", "last_frame"),
        "ref-image": ("ref_image",),
        "ref-audio": ("ref_image_audio",),
        "multi-ref": ("ref_image",),
    }.get(mode, ())
    for key in keys:
        values = files.get(key, [])
        if values:
            return values[0]
    return None


def probe_display_size(path, ffprobe):
    """Read display-oriented dimensions, including common rotation metadata."""
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError("cannot inspect input image dimensions")
    try:
        stream = json.loads(result.stdout)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("input image has invalid dimensions") from exc
    rotation = stream.get("tags", {}).get("rotate", 0)
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = side_data["rotation"]
            break
    try:
        quarter_turn = round(float(rotation) / 90.0)
    except (TypeError, ValueError):
        quarter_turn = 0
    if quarter_turn % 2:
        width, height = height, width
    if width < 1 or height < 1:
        raise ValueError("input image has invalid dimensions")
    return width, height


def worker_hex(value):
    """Encode arbitrary UTF-8 text as one whitespace-safe protocol token."""
    return str(value).encode("utf-8").hex()


def build_worker_commands(request, job_id, output_file):
    """Build one atomic request for the long-lived native h3 worker."""
    params = request["params"]
    commands = [
        "!worker-reset",
        f"!size {params['width']}x{params['height']}",
        f"!frames {params['frames']}",
        f"!steps {params['steps']}",
        f"!layers {params['layers']}",
        f"!reuse {params['reuse']}",
        "!core-reuse 1",
        "!token-reduction off",
        "!reference-rope off",
        "!ssd-streaming off",
        "!int8-row-fc2 off",
        f"!seed {params['seed']}",
    ]
    if params["ssd_streaming"]:
        commands.append("!ssd-streaming on")
    if params["int8_row_fc2"]:
        commands.append("!int8-row-fc2 on")
    for lora in request.get("loras", []):
        commands.append(
            f"!worker-lora {lora['scope']} {lora['scale']:.9g} "
            f"{worker_hex(lora['path'])}"
        )

    mode = request["mode"]
    files = request["files"]
    if mode == "frame-anchor":
        for path in files.get("first_frame", []):
            commands.append(f"!worker-anchor first {worker_hex(path)}")
        for path in files.get("last_frame", []):
            commands.append(f"!worker-anchor last {worker_hex(path)}")
    else:
        image_key = "ref_image_audio" if mode == "ref-audio" else "ref_image"
        for path in files.get(image_key, []):
            commands.append(f"!worker-ref image {worker_hex(path)}")
        for path in files.get("ref_video", []):
            commands.append(f"!worker-ref video {worker_hex(path)}")
        for path in files.get("ref_audio", []):
            commands.append(f"!worker-ref audio {worker_hex(path)}")

    commands.append(
        f"!worker-run {job_id} {worker_hex(output_file)} "
        f"{worker_hex(request['prompt'])}"
    )
    return commands


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_lora_header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        header_size = struct.unpack("<Q", prefix)[0]
        payload_limit = path.stat().st_size - 8
        if header_size < 2 or header_size > payload_limit or header_size > 64 * 1024 * 1024:
            raise ValueError("invalid safetensors header size")
        header = json.loads(stream.read(header_size))
    header.pop("__metadata__", None)
    return header_size, header


def _checkpoint_tensor_map(transformer):
    from tools.fold_turbo_lora import checkpoint_tensors, read_header
    shards = sorted(transformer.glob("model-*.safetensors"))
    return checkpoint_tensors([(path, read_header(path)[1]) for path in shards])


def lora_catalog(model_dir):
    """Inspect regular runtime LoRA files and cache immutable public metadata."""
    LORA_RUNTIME.mkdir(parents=True, exist_ok=True)
    files = []
    with os.scandir(LORA_RUNTIME) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                continue
            if not entry.name.lower().endswith(".safetensors"):
                continue
            status = entry.stat(follow_symlinks=False)
            files.append((entry.name, status.st_ino, status.st_size,
                          status.st_mtime_ns))
    files.sort()
    variant_dirs = {
        "fl2va": Path(model_dir) / "FL2VA/transformer",
        "ref2va": Path(model_dir) / "Ref2VA/transformer",
    }
    variant_key = tuple(
        (name, directory.is_dir(),
         tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns)
               for path in sorted(directory.glob("model-*.safetensors"))))
        for name, directory in variant_dirs.items()
    )
    cache_key = (tuple(files), variant_key)
    with lora_catalog_lock:
        if lora_catalog_cache.get("key") == cache_key:
            return lora_catalog_cache["items"]

    from tools.fold_turbo_lora import resolve_pairs
    checkpoints = {
        name: _checkpoint_tensor_map(directory)
        for name, directory in variant_dirs.items() if directory.is_dir()
    }
    catalog = []
    for name, inode, size, mtime_ns in files:
        path = LORA_RUNTIME / name
        item = {
            "id": _sha256_file(path),
            "name": name,
            "size": size,
            "format": "unknown",
            "rank": None,
            "scopes": [],
            "supported": False,
            "_path": str(path.resolve()),
            "_identity": (inode, size, mtime_ns),
        }
        try:
            header_size, header = _read_lora_header(path)
            base = 8 + header_size
            formats = set()
            ranks = set()
            for scope, checkpoint in checkpoints.items():
                try:
                    pairs, detected = resolve_pairs(path, header, base, checkpoint)
                except SystemExit:
                    continue
                item["scopes"].append(scope)
                formats.update(detected)
                ranks.update(int(header[a_key]["shape"][0])
                             for a_key, _, _ in pairs.values())
            item["format"] = "+".join(sorted(formats)) or "unknown"
            item["rank"] = min(ranks) if len(ranks) == 1 else (
                f"{min(ranks)}–{max(ranks)}" if ranks else None
            )
            item["supported"] = bool(item["scopes"])
            if set(item["scopes"]) == {"fl2va", "ref2va"}:
                item["scopes"].append("both")
        except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError,
                json.JSONDecodeError, struct.error) as exc:
            item["error"] = str(exc)
        catalog.append(item)
    with lora_catalog_lock:
        lora_catalog_cache.clear()
        lora_catalog_cache.update(key=cache_key, items=catalog)
    return catalog


def public_lora_catalog(model_dir):
    hidden = {"_path", "_identity"}
    return [{key: value for key, value in item.items() if key not in hidden}
            for item in lora_catalog(model_dir)]


def resolve_lora_selection(requested, model_dir, mode):
    if requested in (None, []):
        return [], []
    if not isinstance(requested, list) or len(requested) > 8:
        raise ValueError("loras must contain at most 8 entries")
    catalog = {item["id"]: item for item in lora_catalog(model_dir)}
    active_scope = "ref2va" if mode in REF2VA_MODES else "fl2va"
    internal, public = [], []
    for index, value in enumerate(requested):
        if not isinstance(value, dict):
            raise ValueError(f"LoRA {index + 1} must be an object")
        identifier = str(value.get("id", "")).lower()
        item = catalog.get(identifier)
        if not item or not item.get("supported"):
            raise ValueError(f"LoRA {index + 1} is not available")
        try:
            scale = float(value.get("scale", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"LoRA {index + 1} scale is invalid") from exc
        if not math.isfinite(scale):
            raise ValueError(f"LoRA {index + 1} scale must be finite")
        scope = str(value.get("scope", "fl2va")).lower()
        if scope not in {"fl2va", "ref2va", "both"} or scope not in item["scopes"]:
            raise ValueError(f"LoRA {index + 1} does not support scope {scope}")
        if scope != "both" and scope != active_scope:
            raise ValueError(f"LoRA {index + 1} scope does not match {active_scope}")
        path = Path(item["_path"])
        try:
            identity = path.lstat()
            path.relative_to(LORA_RUNTIME.resolve())
        except (OSError, ValueError) as exc:
            raise ValueError(f"LoRA {index + 1} left the runtime directory") from exc
        if (not stat.S_ISREG(identity.st_mode) or
            (identity.st_ino, identity.st_size, identity.st_mtime_ns) !=
                item["_identity"]):
            raise ValueError(f"LoRA {index + 1} changed after catalog inspection")
        internal.append({
            "id": identifier,
            "name": item["name"],
            "path": item["_path"],
            "scale": scale,
            "scope": scope,
        })
        public.append({
            "id": identifier,
            "name": item["name"],
            "scale": scale,
            "scope": scope,
        })
    return internal, public


def image_capabilities():
    """Report the local image runtime and only the quantizations on disk."""
    scripts = {
        "t2i": SENSENOVA_DIR / "run_sensenova_macos.py",
        "edit": SENSENOVA_DIR / "edit_sensenova_macos.py",
        "vqa": SENSENOVA_DIR / "vqa_sensenova_macos.py",
    }
    variants = {
        name: {
            "ready": (SENSENOVA_DIR / filename).is_file(),
            "filename": filename,
        }
        for name, filename in SENSENOVA_VARIANTS.items()
    }
    return {
        "runtime": SENSENOVA_PYTHON.is_file(),
        "tasks": {name: path.is_file() for name, path in scripts.items()},
        "variants": variants,
        "model_dir": str(SENSENOVA_DIR),
    }


def estimate_image_seconds(params, task):
    """Conservative M5 Pro estimate used only for progress feedback."""
    if task == "vqa":
        max_tokens = max(1, min(1024, int(params.get("max_new_tokens") or 128)))
        # GGUFLinear dequantizes on every autoregressive token on MPS. Keep the
        # estimate conservative so 128-token answers do not appear stuck at 96%.
        return round(20.0 + max_tokens * 7.0, 1)
    steps = max(1, int(params.get("steps") or 16))
    pixels = max(1, int(params.get("width") or 2048) * int(params.get("height") or 2048))
    branch_factor = 2.0 if float(params.get("cfg_scale") or 4.0) > 1 else 1.0
    seconds = 5.0 + steps * 6.5 * branch_factor * (pixels / (2048 * 2048))
    if task == "edit":
        seconds *= 1.45
    return round(max(15.0, seconds), 1)


def refresh_running_progress(job, now=None):
    """Advance elapsed time, estimated GPU progress, and ETA for a job."""
    if not job or job.get("status") != "running":
        return
    now = time.time() if now is None else now
    started = job.get("started_timestamp", now)
    elapsed = max(0.0, now - started)
    estimate = max(1.0, float(job.get("estimated_total_seconds") or 1.0))
    progress = float(job.get("progress_percent") or 1.0)

    if job.get("kind") == "image":
        progress = max(progress, min(96.0, (elapsed / estimate) * 100.0))
    elif job.get("phase") in ("denoise enqueue", "denoise"):
        # The GPU sampler is mostly silent after command enqueue. Time-based
        # interpolation keeps the UI useful without claiming exact telemetry.
        progress = max(progress, min(81.5, (elapsed / estimate) * 100.0))

    job["progress_percent"] = round(min(99.5, progress), 1)
    job["elapsed_seconds"] = round(elapsed, 1)
    job["eta_seconds"] = (
        round(estimate - elapsed, 1) if elapsed < estimate else None
    )


def record_process_line(job, line):
    """Append one live CLI record and translate known progress phases."""
    clean = line.strip()
    for lora in (job.get("worker_request") or {}).get("loras", []):
        path = lora.get("path")
        if path:
            clean = clean.replace(path, f"<LoRA:{lora.get('name', 'adapter')}>")
    if not clean:
        return
    with job_lock:
        log = (job.get("log", "") + clean + "\n")[-20000:]
        job["log"] = log
        if job.get("kind") == "chain":
            match = re.match(r"^SHOT (\d+)/(\d+) (rendering|done)", clean)
            if match:
                shot, total, state = (int(match.group(1)),
                                      max(1, int(match.group(2))),
                                      match.group(3))
                completed = shot if state == "done" else shot - 1
                job.update(
                    phase="chain",
                    phase_label=f"镜头 {shot}/{total}" + (
                        " 完成" if state == "done" else " 渲染中"),
                    phase_completed=completed,
                    phase_total=total,
                    progress_percent=round(
                        min(97.0, 2.0 + completed / total * 93.0), 1),
                    progress_is_estimate=True,
                )
            elif clean.startswith("CHAIN done"):
                job.update(phase="saving", phase_label="拼接完成",
                           progress_percent=99.0)
            refresh_running_progress(job)
            return
        if job.get("kind") == "image":
            if clean.startswith("[gguf] loading"):
                job.update(phase="loading", phase_label="加载 SenseNova", progress_percent=5.0)
            elif clean.startswith("[gguf] parsed"):
                job.update(phase="loading", phase_label="解析量化权重", progress_percent=18.0)
            elif clean.startswith("[loaded]"):
                job.update(phase="inference", phase_label=(
                    "视觉问答" if job.get("task") == "vqa" else
                    "编辑图片" if job.get("task") == "edit" else "生成图片"
                ), progress_percent=28.0)
            elif clean.startswith("[answer]") or clean.startswith("[saved]"):
                job.update(phase="saving", phase_label="保存结果", progress_percent=98.0)
            refresh_running_progress(job)
            return
        match = PROGRESS_LINE.match(clean)
        if not match:
            refresh_running_progress(job)
            return
        phase, completed_text, total_text = match.groups()
        phase_spec = PHASE_PROGRESS.get(phase)
        if not phase_spec:
            return
        completed = int(completed_text)
        total = max(1, int(total_text))
        start, end, label = phase_spec
        fraction = min(1.0, max(0.0, completed / total))
        mapped = start + (end - start) * fraction
        job.update({
            "phase": phase,
            "phase_label": label,
            "phase_completed": completed,
            "phase_total": total,
            "progress_percent": round(max(
                float(job.get("progress_percent") or 0.0), mapped
            ), 1),
            "progress_is_estimate": True,
        })
        refresh_running_progress(job)


def stream_process_output(job, stream):
    """Read carriage-return CLI progress records without waiting for exit."""
    pending = []
    while True:
        char = stream.read(1)
        if char == "":
            if pending:
                record_process_line(job, "".join(pending))
            return
        if char in "\r\n":
            if pending:
                record_process_line(job, "".join(pending))
                pending.clear()
        else:
            pending.append(char)


class H3PersistentWorker:
    """Own one native h3 process and route its streamed output per job."""

    def __init__(self, executable, model_dir, process_env, startup_timeout=15):
        self.executable = Path(executable)
        self.model_dir = Path(model_dir)
        self.process_env = process_env
        self.startup_timeout = startup_timeout
        self.process = None
        self.reader = None
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.ready_ok = False
        self.done = threading.Event()
        self.active_job = None
        self.active_job_id = None
        self.result = None
        self.startup_log = ""
        self.starts = 0

    @staticmethod
    def _close_streams(process):
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                stream.close()

    def _record_line(self, line):
        clean = line.strip()
        if not clean:
            return
        if clean == "H3_WORKER_READY":
            with self.lock:
                self.ready_ok = True
            self.ready.set()
            return
        marker = re.fullmatch(
            r"H3_WORKER_(START|DONE|ERROR) ([A-Za-z0-9_-]{1,64})", clean
        )
        with self.lock:
            job = self.active_job
            active_job_id = self.active_job_id
            if marker and marker.group(2) == active_job_id:
                event = marker.group(1)
                if event in {"DONE", "ERROR"}:
                    self.result = event.lower()
                    self.done.set()
                return
            if not job:
                self.startup_log = (self.startup_log + clean + "\n")[-20000:]
                return
        record_process_line(job, clean)

    def _read_output(self, process):
        pending = []
        while True:
            char = process.stdout.read(1)
            if char == "":
                if pending:
                    self._record_line("".join(pending))
                break
            if char in "\r\n":
                if pending:
                    self._record_line("".join(pending))
                    pending.clear()
            else:
                pending.append(char)
        with self.lock:
            same_process = self.process is process
            if same_process and self.active_job is not None:
                self.result = "exited"
                self.done.set()
        if same_process:
            self.ready.set()

    def _start(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return
            if self.process is not None:
                self._close_streams(self.process)
            self.ready.clear()
            self.ready_ok = False
            self.startup_log = ""
            command = [
                str(self.executable), "--profile", "--worker",
                "-d", str(self.model_dir), "--seed", "42",
            ]
            process = subprocess.Popen(
                command,
                cwd=str(HERE),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=self.process_env,
                start_new_session=True,
            )
            self.process = process
            self.starts += 1
            self.reader = threading.Thread(
                target=self._read_output, args=(process,), daemon=True
            )
            self.reader.start()
        if not self.ready.wait(self.startup_timeout):
            self.stop()
            raise RuntimeError("h3 worker did not become ready")
        with self.lock:
            ready_ok = self.ready_ok
        if not ready_ok or process.poll() is not None:
            detail = self.startup_log.strip() or f"exit code {process.returncode}"
            self.stop()
            raise RuntimeError(f"h3 worker failed to start: {detail}")

    def run(self, job, request, output_file, timeout):
        self._start()
        with self.lock:
            if self.active_job is not None:
                raise RuntimeError("h3 worker is already running a job")
            process = self.process
            if process is None or process.poll() is not None:
                raise RuntimeError("h3 worker is not running")
            self.active_job = job
            self.active_job_id = job["job_id"]
            self.result = None
            self.done.clear()
        commands = build_worker_commands(
            request, job["job_id"], str(output_file)
        )
        try:
            process.stdin.write("\n".join(commands) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.stop()
            raise RuntimeError(f"h3 worker input failed: {exc}") from exc

        if not self.done.wait(timeout):
            self.stop()
            raise subprocess.TimeoutExpired(process.args, timeout)
        with self.lock:
            result = self.result
            self.active_job = None
            self.active_job_id = None
            self.result = None
        if result == "done":
            return 0
        if result == "error":
            return 1
        self.stop()
        detail = (job.get("log", "").strip().splitlines() or
                  [self.startup_log.strip() or "native worker exited unexpectedly"])[-1]
        raise RuntimeError(detail)

    def stop(self):
        with self.lock:
            process = self.process
            reader = self.reader
            self.process = None
            self.reader = None
            self.active_job = None
            self.active_job_id = None
            self.ready.clear()
            self.ready_ok = False
            self.done.set()
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2)
        self._close_streams(process)

    def status(self):
        with self.lock:
            process = self.process
            alive = process is not None and process.poll() is None
            return {
                "state": "running" if alive and self.active_job else
                         "ready" if alive else "stopped",
                "pid": process.pid if alive else None,
                "starts": self.starts,
            }


def checkpoint_ready(model_dir, variant):
    """Require the index and every referenced shard before exposing a model."""
    transformer = Path(model_dir) / variant / "transformer"
    index_path = transformer / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
        shards = set(index.get("weight_map", {}).values())
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return bool(shards) and all((transformer / shard).is_file() for shard in shards)


def model_capabilities(model_dir):
    """Report only complete checkpoints so partial downloads never look ready."""
    return {
        "fl2va": checkpoint_ready(model_dir, "FL2VA"),
        "ref2va": checkpoint_ready(model_dir, "Ref2VA"),
    }


def generation_profile(model_dir):
    """Describe safe UI defaults for the selected model tree."""
    resolved = Path(model_dir).resolve()
    turbo = (resolved == TURBO_MODEL_DIR.resolve() or
             resolved.name == "turbo-v4-step600-ema")
    try:
        model_arg = "./" + resolved.relative_to(HERE.resolve()).as_posix()
    except ValueError:
        model_arg = str(resolved)
    return {
        "name": "turbo-v4-step600-ema" if turbo else "base",
        "model_dir": str(resolved),
        "model_arg": model_arg,
        "turbo": turbo,
        "int8_sidecar": (
            resolved / "FL2VA/transformer/.h3-int8/source.tsv"
        ).is_file(),
        "text_defaults": {
            "steps": 8 if turbo else 20,
            "layers": 45,
            "reuse": 1 if turbo else 2,
        },
        "reference_defaults": ({"steps": 8, "layers": 45, "reuse": 1}
                               if turbo else
                               {"steps": 20, "layers": 45, "reuse": 2}),
    }


def public_job(job, include_log=True):
    """Return a JSON-safe snapshot without exposing the command arguments."""
    if job is None:
        return None
    refresh_running_progress(job)
    hidden = {"cmd", "worker_request", "upload_paths", "started_timestamp"}
    if not include_log:
        hidden.add("log")
    return {key: value for key, value in job.items() if key not in hidden}


def resolve_output_path(value, strip_outputs_prefix=True, create_parent=True):
    """Resolve a UI output name below OUTPUTS without duplicating outputs/."""
    raw = str(value or "").strip() or f"generation_{int(time.time())}.mp4"
    requested = Path(raw)
    if requested.is_absolute():
        raise ValueError("output path must be relative")
    parts = requested.parts
    if strip_outputs_prefix and parts and parts[0] == "outputs":
        requested = Path(*parts[1:])
    if not requested.parts:
        raise ValueError("output filename is required")
    destination = (OUTPUTS / requested).resolve()
    try:
        relative = destination.relative_to(OUTPUTS.resolve())
    except ValueError as exc:
        raise ValueError("output path must stay inside outputs") from exc
    if create_parent:
        destination.parent.mkdir(parents=True, exist_ok=True)
    return destination, relative.as_posix()


def resolve_upload_token(value):
    """Resolve an opaque browser-upload token below UPLOADS."""
    token = str(value or "").strip()
    if not token or Path(token).is_absolute():
        raise ValueError("invalid uploaded file token")
    upload_path = (UPLOADS / token).resolve()
    try:
        upload_path.relative_to(UPLOADS.resolve())
    except ValueError as exc:
        raise ValueError("uploaded file path must stay inside uploads") from exc
    if not upload_path.is_file():
        raise ValueError(f"uploaded file is missing: {Path(token).name}")
    return upload_path


def resolve_reference_files(mode, user_files):
    """Validate every UI reference and replace its token with a real path."""
    layouts = {
        "text": {},
        "frame-anchor": {"first_frame": 1, "last_frame": 1},
        "ref-image": {"ref_image": 9},
        "ref-video": {"ref_video": 1},
        "ref-audio": {"ref_image_audio": 1, "ref_audio": 3},
        "multi-ref": {"ref_image": 9, "ref_video": 3, "ref_audio": 3},
    }
    if mode not in layouts:
        raise ValueError("invalid generation mode")
    if not isinstance(user_files, dict):
        raise ValueError("files must be an object")

    resolved = {}
    for key, maximum in layouts[mode].items():
        tokens = user_files.get(key, [])
        if not isinstance(tokens, list) or len(tokens) > maximum:
            raise ValueError(f"invalid file count for {key}")
        resolved[key] = [str(resolve_upload_token(token)) for token in tokens]

    count = sum(len(items) for items in resolved.values())
    if mode == "frame-anchor" and count == 0:
        raise ValueError("first frame or last frame is required")
    if mode in ("ref-image", "ref-video") and count == 0:
        raise ValueError("a reference file is required")
    if mode == "ref-audio":
        if len(resolved.get("ref_image_audio", [])) != 1:
            raise ValueError("reference audio mode requires one image")
        if not resolved.get("ref_audio"):
            raise ValueError("reference audio mode requires audio")
    if mode == "multi-ref" and (count == 0 or count > 12):
        raise ValueError("mixed references require 1 to 12 files")
    return resolved


def cleanup_upload_paths(paths):
    """Remove per-submission upload directories after a job finishes."""
    upload_root = UPLOADS.resolve()
    for value in paths:
        path = Path(value).resolve()
        try:
            path.relative_to(upload_root)
        except ValueError:
            continue
        parent = path.parent
        if parent.parent == upload_root:
            shutil.rmtree(parent, ignore_errors=True)


def discover_existing_outputs():
    """Make completed video and image files visible after a restart."""
    for output_file in sorted(OUTPUTS.rglob("*.mp4"), key=lambda path: path.stat().st_mtime):
        relative = output_file.relative_to(OUTPUTS).as_posix()
        job_id = "existing-" + uuid.uuid5(uuid.NAMESPACE_URL, relative).hex
        job_results[job_id] = {
            "job_id": job_id,
            "status": "succeeded",
            "success": True,
            "prompt": output_file.stem,
            "mode": "text",
            "params": {},
            "output": relative,
            "video": f"/outputs/{urllib.parse.quote(relative)}",
            "duration": "--",
            "created_at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(output_file.stat().st_mtime)
            ),
            "finished_at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(output_file.stat().st_mtime)
            ),
        }
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".txt"}
    for output_file in sorted(IMAGE_OUTPUTS.iterdir(), key=lambda path: path.stat().st_mtime):
        if not output_file.is_file() or output_file.suffix.lower() not in image_suffixes:
            continue
        relative = output_file.relative_to(OUTPUTS).as_posix()
        stem_task = output_file.stem.split("-", 1)[0]
        task = stem_task if stem_task in {"t2i", "edit", "vqa"} else "t2i"
        job_id = "existing-image-" + uuid.uuid5(uuid.NAMESPACE_URL, relative).hex
        job = {
            "job_id": job_id,
            "kind": "image",
            "task": task,
            "status": "succeeded",
            "success": True,
            "prompt": output_file.stem,
            "mode": task,
            "quant": "",
            "params": {},
            "output": relative,
            "duration": "--",
            "created_at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(output_file.stat().st_mtime)
            ),
            "finished_at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(output_file.stat().st_mtime)
            ),
        }
        if output_file.suffix.lower() == ".txt":
            job["task"] = "vqa"
            job["mode"] = "vqa"
            job["answer"] = output_file.read_text(encoding="utf-8", errors="replace").strip()
        else:
            job["image"] = f"/outputs/{urllib.parse.quote(relative)}"
        job_results[job_id] = job


discover_existing_outputs()

class H3APIHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the H3 Studio API."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_file(HERE / "h3-ui.html", "text/html; charset=utf-8")
        elif path == "/api/status":
            with job_lock:
                active = job_results.get(current_job)
                jobs = [public_job(job, include_log=False) for job in job_results.values()]
                queued = [job for job in jobs if job.get("status") == "queued"]
                for position, job in enumerate(queued, 1):
                    job["queue_position"] = position
            self._json_response({
                "status": "running" if active else ("queued" if queued else "idle"),
                "current_job": public_job(active),
                "queue_length": len(queued),
                "outputs": jobs,
                "capabilities": model_capabilities(self.server.model_dir),
                "generation_profile": generation_profile(self.server.model_dir),
                "lora_catalog": public_lora_catalog(self.server.model_dir),
                "image_capabilities": image_capabilities(),
                "video_worker": self.server.video_worker.status(),
            })
        elif path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/"):]
            with job_lock:
                job = public_job(job_results.get(job_id))
            if job is None:
                self._json_response({"error": "job not found"}, 404)
            else:
                self._json_response(job)
        elif path.startswith("/outputs/"):
            try:
                output_name = urllib.parse.unquote(path[len("/outputs/"):])
                file_path, _ = resolve_output_path(
                    output_name, strip_outputs_prefix=False, create_parent=False
                )
            except ValueError as exc:
                self._json_response({"error": str(exc)}, 400)
                return
            content_types = {
                ".mp4": "video/mp4",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".txt": "text/plain; charset=utf-8",
            }
            self._serve_file(file_path, content_types.get(file_path.suffix.lower(), "application/octet-stream"))
        else:
            self._json_response({"error": "not found"}, 404)

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        image_prefix = "/api/image/outputs/"
        video_prefix = "/api/outputs/"
        if parsed.path.startswith(image_prefix):
            prefix = image_prefix
            output_kind = "image"
        elif parsed.path.startswith(video_prefix):
            prefix = video_prefix
            output_kind = "video"
        else:
            self._json_response({"error": "not found"}, 404)
            return

        output_name = urllib.parse.unquote(parsed.path[len(prefix):])
        if not output_name:
            self._json_response({"error": "output filename is required"}, 400)
            return
        try:
            output_file, relative = resolve_output_path(
                output_name, strip_outputs_prefix=False, create_parent=False
            )
        except ValueError as exc:
            self._json_response({"error": str(exc)}, 400)
            return
        if output_kind == "video":
            if output_file.suffix.lower() != ".mp4":
                self._json_response({"error": "only MP4 outputs can be deleted here"}, 400)
                return
        else:
            if not relative.startswith("images/"):
                self._json_response({"error": "image outputs must stay inside outputs/images"}, 400)
                return
            if output_file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".txt"}:
                self._json_response({"error": "unsupported image output type"}, 400)
                return

        with job_lock:
            blocking = next((
                job for job in job_results.values()
                if job.get("output") == relative and job.get("status") in ("queued", "running")
            ), None)
            if blocking:
                self._json_response({
                    "error": f"cannot delete {output_kind} output while its task is active",
                    "current_job": public_job(blocking),
                }, 409)
                return

            matching_ids = [
                job_id for job_id, job in job_results.items()
                if job.get("output") == relative
            ]
            existed = output_file.is_file()
            if not existed and not matching_ids:
                self._json_response({"error": "output not found"}, 404)
                return
            if existed:
                try:
                    output_file.unlink()
                except OSError as exc:
                    self._json_response({"error": f"could not delete output: {exc}"}, 500)
                    return
            for job_id in matching_ids:
                job_results.pop(job_id, None)

        self._json_response({
            "status": "deleted",
            "output": relative,
            "file_deleted": existed,
        })

    def do_POST(self):
        global current_job
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/uploads":
            self._handle_upload()
            return
        if parsed.path == "/api/image/generate":
            self._handle_image_generate()
            return
        if parsed.path != "/api/generate":
            self._json_response({"error": "not found"}, 404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b"{}"
        try:
            params = json.loads(body)
        except json.JSONDecodeError:
            self._json_response({"error": "invalid JSON"}, 400)
            return

        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            self._json_response({"error": "prompt is required"}, 400)
            return

        mode = params.get("mode", "text")
        if mode == "chain":
            self._handle_chain_generate(params, prompt)
            return
        if mode in REF2VA_MODES and not model_capabilities(
                self.server.model_dir)["ref2va"]:
            self._json_response({
                "error": ("参考模式需要 Ref2VA 检查点；当前模型目录只安装了 "
                          "FL2VA，参考图片、视频、音频和混合引用暂不可用"),
                "code": "ref2va_checkpoint_missing",
            }, 503)
            return

        try:
            profile = generation_profile(self.server.model_dir)
            defaults = (profile["reference_defaults"] if mode in REF2VA_MODES
                        else profile["text_defaults"])
            video_params = normalize_video_params(params, defaults)
        except ValueError as exc:
            self._json_response({"error": str(exc)}, 400)
            return

        try:
            selected_loras, public_loras = resolve_lora_selection(
                params.get("loras"), self.server.model_dir, mode
            )
        except ValueError as exc:
            self._json_response({"error": str(exc)}, 400)
            return
        if selected_loras and video_params["ssd_streaming"]:
            self._json_response({
                "error": "runtime LoRA cannot be combined with SSD streaming"
            }, 400)
            return

        try:
            output_file, output_name = resolve_output_path(params.get("output"))
        except ValueError as exc:
            self._json_response({"error": str(exc)}, 400)
            return

        try:
            user_files = resolve_reference_files(mode, params.get("files", {}))
        except ValueError as exc:
            self._json_response({"error": str(exc)}, 400)
            return

        if video_params["match_input_aspect"]:
            source_image = primary_image_reference(mode, user_files)
            if source_image:
                try:
                    source_width, source_height = probe_display_size(
                        source_image, self.server.process_env["H3_FFPROBE"]
                    )
                    matched_width, matched_height = legal_canvas_for_aspect(
                        source_width, source_height,
                        video_params["width"] * video_params["height"],
                    )
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, 400)
                    return
                video_params["width"] = matched_width
                video_params["height"] = matched_height
                video_params["source_aspect"] = f"{source_width}:{source_height}"

        job_public_params = {key: video_params[key] for key in
                             ["width", "height", "frames", "steps", "layers", "reuse"]}
        if video_params.get("source_aspect"):
            job_public_params["source_aspect"] = video_params["source_aspect"]
            job_public_params["match_input_aspect"] = True

        job_id = uuid.uuid4().hex
        with job_lock:
            # Never let queued jobs or existing files overwrite one another.
            requested = output_file
            suffix = requested.suffix or ".mp4"
            stem = requested.stem
            attempt = 1
            reserved = {job.get("output") for job in job_results.values()}
            while output_file.exists() or output_name in reserved:
                attempt += 1
                output_file = requested.with_name(f"{stem}-{attempt}{suffix}")
                output_name = output_file.relative_to(OUTPUTS.resolve()).as_posix()
            job = {
                "job_id": job_id,
                "kind": "video",
                "status": "queued",
                "success": None,
                "prompt": prompt,
                "mode": mode,
                "loras": public_loras,
                "params": job_public_params,
                "output": output_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "estimated_total_seconds": estimate_job_seconds(video_params, mode),
                "progress_percent": 0.0,
                "progress_is_estimate": True,
                "phase": "queued",
                "phase_label": "等待执行",
                "phase_completed": 0,
                "phase_total": 1,
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
                "worker_request": {
                    "prompt": prompt,
                    "mode": mode,
                    "params": video_params,
                    "files": user_files,
                    "loras": selected_loras,
                },
                "upload_paths": sorted({
                    path for paths in user_files.values() for path in paths
                }),
            }
            job_results[job_id] = job
        self.server.enqueue(job_id)

        self._json_response(public_job(job), 202)

    def _handle_chain_generate(self, params, prompt):
        """Enqueue a multi-shot chained long video via tools/shot_chain.py."""
        shots = [line.strip() for line in prompt.splitlines()
                 if line.strip() and not line.strip().startswith("#")]
        if not shots:
            self._json_response({"error": "镜头链模式需要每行一个镜头描述"}, 400)
            return
        try:
            profile = generation_profile(self.server.model_dir)
            video_params = normalize_video_params(
                params, profile["text_defaults"])
        except ValueError as exc:
            self._json_response({"error": str(exc)}, 400)
            return
        try:
            output_file, output_name = resolve_output_path(
                params.get("output") or "chain.mp4")
        except ValueError as exc:
            self._json_response({"error": str(exc)}, 400)
            return
        shot_seconds = max(1.0, round(video_params["frames"] / 24.0, 2))
        job_id = uuid.uuid4().hex
        shots_dir = OUTPUTS / ".chains"
        shots_dir.mkdir(parents=True, exist_ok=True)
        shots_file = shots_dir / f"{job_id}.txt"
        shots_file.write_text(prompt, encoding="utf-8")
        with job_lock:
            requested = output_file
            suffix = requested.suffix or ".mp4"
            stem = requested.stem
            attempt = 1
            reserved = {job.get("output") for job in job_results.values()}
            while output_file.exists() or output_name in reserved:
                attempt += 1
                output_file = requested.with_name(f"{stem}-{attempt}{suffix}")
                output_name = output_file.relative_to(
                    OUTPUTS.resolve()).as_posix()
            cmd = [
                sys.executable, "-u", str(HERE / "tools" / "shot_chain.py"),
                "-d", str(self.server.model_dir),
                "--shots-file", str(shots_file),
                "--h3", str(HERE / "h3"),
                "--width", str(video_params["width"]),
                "--height", str(video_params["height"]),
                "--seconds-per-shot", str(shot_seconds),
                "--steps", str(video_params["steps"]),
                "--layers", str(video_params["layers"]),
                "--seed", str(video_params["seed"]),
                "--out", str(output_file),
            ]
            per_shot = estimate_job_seconds(video_params, "text")
            job = {
                "job_id": job_id,
                "kind": "chain",
                "status": "queued",
                "success": None,
                "prompt": prompt,
                "mode": "chain",
                "loras": [],
                "params": {key: video_params[key] for key in
                           ["width", "height", "frames", "steps", "layers",
                            "reuse"]},
                "shots": len(shots),
                "output": output_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "estimated_total_seconds": per_shot * len(shots),
                "progress_percent": 0.0,
                "progress_is_estimate": True,
                "phase": "queued",
                "phase_label": "等待执行",
                "phase_completed": 0,
                "phase_total": len(shots),
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
                "cmd": cmd,
                "upload_paths": [],
            }
            job_results[job_id] = job
        self.server.enqueue(job_id)
        self._json_response(public_job(job), 202)

    def _handle_image_generate(self):
        """Validate and enqueue a SenseNova image or VQA task."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b"{}"
        try:
            params = json.loads(body)
        except json.JSONDecodeError:
            self._json_response({"error": "invalid JSON"}, 400)
            return

        task = str(params.get("task", "t2i"))
        if task not in {"t2i", "edit", "vqa"}:
            self._json_response({"error": "task must be t2i, edit, or vqa"}, 400)
            return
        text = str(params.get("question" if task == "vqa" else "prompt", "")).strip()
        if not text:
            self._json_response({"error": "question is required" if task == "vqa" else "prompt is required"}, 400)
            return

        quant = str(params.get("quant", "Q4_0")).upper()
        filename = SENSENOVA_VARIANTS.get(quant)
        gguf_path = SENSENOVA_DIR / filename if filename else None
        if not filename or not gguf_path.is_file():
            self._json_response({
                "error": f"SenseNova {quant} is not installed",
                "capabilities": image_capabilities(),
            }, 503)
            return

        try:
            width = int(params.get("width") or 2048)
            height = int(params.get("height") or 2048)
            steps = int(params.get("steps") or 16)
            seed = int(params.get("seed") if params.get("seed") is not None else 42)
            cfg_scale = float(params.get("cfg_scale") or 4.0)
        except (TypeError, ValueError):
            self._json_response({"error": "invalid numeric image parameters"}, 400)
            return
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            self._json_response({"error": "width and height must be positive multiples of 32"}, 400)
            return
        if not 1 <= steps <= 100:
            self._json_response({"error": "steps must be between 1 and 100"}, 400)
            return

        image_tokens = params.get("images", [])
        if not isinstance(image_tokens, list):
            self._json_response({"error": "images must be an array"}, 400)
            return
        try:
            image_paths = [resolve_upload_token(token) for token in image_tokens]
        except ValueError as exc:
            self._json_response({"error": str(exc)}, 400)
            return
        if task == "edit" and not 1 <= len(image_paths) <= 4:
            self._json_response({"error": "image editing requires 1 to 4 images"}, 400)
            return
        if task == "vqa" and len(image_paths) != 1:
            self._json_response({"error": "VQA requires exactly one image"}, 400)
            return
        if task == "t2i" and image_paths:
            self._json_response({"error": "text-to-image does not accept input images"}, 400)
            return

        suffix = ".txt" if task == "vqa" else ".png"
        output_value = params.get("output") or f"images/{task}-{int(time.time())}{suffix}"
        try:
            output_file, output_name = resolve_output_path(output_value)
        except ValueError as exc:
            self._json_response({"error": str(exc)}, 400)
            return
        if output_file.suffix.lower() != suffix:
            output_file = output_file.with_suffix(suffix)
            output_name = output_file.relative_to(OUTPUTS.resolve()).as_posix()

        script = {
            "t2i": SENSENOVA_DIR / "run_sensenova_macos.py",
            "edit": SENSENOVA_DIR / "edit_sensenova_macos.py",
            "vqa": SENSENOVA_DIR / "vqa_sensenova_macos.py",
        }[task]
        if not SENSENOVA_PYTHON.is_file() or not script.is_file():
            self._json_response({"error": "SenseNova macOS runtime is incomplete"}, 503)
            return

        cmd = [str(SENSENOVA_PYTHON), str(script), "--gguf", str(gguf_path)]
        if task == "t2i":
            cmd.extend([
                "--prompt", text, "--width", str(width), "--height", str(height),
                "--steps", str(steps), "--seed", str(seed), "--cfg-scale", str(cfg_scale),
                "--output", str(output_file),
            ])
        elif task == "edit":
            cmd.extend(["--image", *map(str, image_paths), "--prompt", text,
                        "--width", str(width), "--height", str(height),
                        "--steps", str(steps), "--seed", str(seed),
                        "--cfg-scale", str(cfg_scale), "--output", str(output_file)])
        else:
            max_tokens = max(1, min(1024, int(params.get("max_new_tokens") or 128)))
            cmd.extend(["--image", str(image_paths[0]), "--question", text,
                        "--max-new-tokens", str(max_tokens), "--output", str(output_file)])

        job_id = uuid.uuid4().hex
        with job_lock:
            requested = output_file
            stem = requested.stem
            attempt = 1
            reserved = {job.get("output") for job in job_results.values()}
            while output_file.exists() or output_name in reserved:
                attempt += 1
                output_file = requested.with_name(f"{stem}-{attempt}{suffix}")
                output_name = output_file.relative_to(OUTPUTS.resolve()).as_posix()
            # The output path is the final argument for each local wrapper.
            cmd[-1] = str(output_file)
            job = {
                "job_id": job_id,
                "kind": "image",
                "task": task,
                "status": "queued",
                "success": None,
                "prompt": text,
                "mode": task,
                "quant": quant,
                "params": {"width": width, "height": height, "steps": steps,
                           "seed": seed, "cfg_scale": cfg_scale},
                "output": output_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "estimated_total_seconds": estimate_image_seconds(params, task),
                "progress_percent": 0.0,
                "progress_is_estimate": True,
                "phase": "queued",
                "phase_label": "等待图像任务",
                "phase_completed": 0,
                "phase_total": 1,
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
                "cmd": cmd,
                "upload_paths": [str(path) for path in image_paths],
            }
            job_results[job_id] = job
        self.server.enqueue(job_id)
        self._json_response(public_job(job), 202)

    def _handle_upload(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json_response({"error": "invalid Content-Length"}, 400)
            return
        if content_length <= 0:
            self._json_response({"error": "empty upload"}, 400)
            return
        if content_length > MAX_UPLOAD_BYTES:
            self._json_response({"error": "upload exceeds 2 GiB limit"}, 413)
            return
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self._json_response({"error": "multipart/form-data is required"}, 415)
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": str(content_length),
                },
                keep_blank_values=True,
            )
            field = form["file"] if "file" in form else None
            if isinstance(field, list):
                field = field[0] if len(field) == 1 else None
            if field is None or not getattr(field, "filename", ""):
                raise ValueError("one file is required")

            original_name = Path(field.filename).name
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", original_name).strip("._")
            if not safe_name:
                safe_name = "reference.bin"
            upload_id = uuid.uuid4().hex
            upload_dir = UPLOADS / upload_id
            upload_dir.mkdir(parents=True, exist_ok=False)
            destination = upload_dir / safe_name
            written = 0
            with destination.open("wb") as output:
                while True:
                    chunk = field.file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise ValueError("upload exceeds 2 GiB limit")
                    output.write(chunk)
            if written == 0:
                raise ValueError("uploaded file is empty")
        except (KeyError, OSError, ValueError) as exc:
            if "upload_dir" in locals():
                shutil.rmtree(upload_dir, ignore_errors=True)
            self._json_response({"error": str(exc)}, 400)
            return

        self._json_response({
            "token": f"{upload_id}/{safe_name}",
            "name": original_name,
            "size": written,
        }, 201)

    def _serve_file(self, path, content_type):
        path = Path(path)
        if not path.exists() or not path.is_file():
            self._json_response({"error": "file not found"}, 404)
            return
        size = path.stat().st_size
        start, end = 0, max(0, size - 1)
        partial = False
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes=") and size:
            try:
                value = range_header[6:].split(",", 1)[0]
                first, last = value.split("-", 1)
                if first:
                    start = int(first)
                    end = int(last) if last else size - 1
                elif last:
                    start = max(0, size - int(last))
                    end = size - 1
                if start < 0 or end < start or start >= size:
                    raise ValueError
                end = min(end, size - 1)
                partial = True
            except (ValueError, TypeError):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return

        length = end - start + 1 if size else 0
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with path.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining:
                    chunk = source.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[H3 Server] {args[0]} {args[1]} {args[2]}")


class H3Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, model_dir, job_timeout=3600):
        self.model_dir = Path(model_dir).resolve()
        self.job_timeout = job_timeout
        self.process_env = os.environ.copy()
        # Windowed video attention is the validated quality/speed default:
        # +/-2 temporal latent frames plus global tokens (3.6x denoise on the
        # 10s 768-class benchmark, artifact-free in probes). Set it to 0 in
        # the shell environment to restore dense attention.
        self.process_env.setdefault("H3_ATTN_WINDOW", "2")
        if self.process_env.get("H3_ATTN_WINDOW") in {"0", ""}:
            self.process_env.pop("H3_ATTN_WINDOW", None)
        self.process_env["H3_FFMPEG"] = resolve_media_tool(
            "H3_FFMPEG", "ffmpeg"
        )
        self.process_env["H3_FFPROBE"] = resolve_media_tool(
            "H3_FFPROBE", "ffprobe"
        )
        self.video_worker = H3PersistentWorker(
            HERE / "h3", self.model_dir, self.process_env
        )
        self.job_queue = queue.Queue()
        super().__init__(addr, handler)
        self.worker = threading.Thread(target=self._run_jobs, daemon=True)
        self.worker.start()

    def enqueue(self, job_id):
        self.job_queue.put(job_id)

    def _run_jobs(self):
        """Run queued generation jobs serially (maximum concurrency: one)."""
        global current_job
        while True:
            job_id = self.job_queue.get()
            try:
                with job_lock:
                    job = job_results.get(job_id)
                    if job is None or job.get("status") != "queued":
                        continue
                    current_job = job_id
                    job["status"] = "running"
                    job["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    job["started_timestamp"] = time.time()
                    job["phase"] = "starting"
                    job["phase_label"] = (
                        "启动 SenseNova" if job.get("kind") == "image" else "启动生成器"
                    )
                    job["progress_percent"] = 1.0
                    job["elapsed_seconds"] = 0.0
                    job["eta_seconds"] = job.get("estimated_total_seconds")
                    job["log"] = ""
                    cmd = job.get("cmd")
                    worker_request = job.get("worker_request")
                    output_name = job["output"]
                output_file, _ = resolve_output_path(
                    output_name, strip_outputs_prefix=False
                )
                started = time.time()
                try:
                    if job.get("kind") == "video":
                        return_code = self.video_worker.run(
                            job, worker_request, output_file, self.job_timeout
                        )
                    else:
                        process = subprocess.Popen(
                            cmd,
                            cwd=str(HERE),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            env=self.process_env,
                        )
                        reader = threading.Thread(
                            target=stream_process_output,
                            args=(job, process.stdout),
                            daemon=True,
                        )
                        reader.start()
                        try:
                            return_code = process.wait(timeout=self.job_timeout)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            raise
                        finally:
                            reader.join(timeout=5)
                    elapsed = time.time() - started
                    log = job.get("log", "")
                    success = (return_code == 0 and output_file.is_file()
                               and output_file.stat().st_size > 0)
                    with job_lock:
                        job.update({
                            "status": "succeeded" if success else "failed",
                            "success": success,
                            "duration": f"{elapsed:.1f}s",
                            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "return_code": return_code,
                            "log": log[-20000:],
                            "progress_percent": 100.0 if success else job.get("progress_percent", 0.0),
                            "phase": "complete" if success else job.get("phase", "failed"),
                            "phase_label": (
                                "图像任务完成" if success and job.get("kind") == "image"
                                else "生成完成" if success else job.get("phase_label", "生成失败")
                            ),
                            "elapsed_seconds": round(elapsed, 1),
                            "eta_seconds": 0.0 if success else None,
                        })
                        if success:
                            output_url = f"/outputs/{urllib.parse.quote(output_name)}"
                            if job.get("kind") == "image":
                                if job.get("task") == "vqa":
                                    job["answer"] = output_file.read_text(encoding="utf-8").strip()
                                else:
                                    job["image"] = output_url
                            else:
                                job["video"] = output_url
                        else:
                            job["error"] = (log.strip().splitlines()[-1]
                                            if log.strip() else "generation failed")
                except subprocess.TimeoutExpired:
                    elapsed = time.time() - started
                    with job_lock:
                        job.update({
                            "status": "failed",
                            "success": False,
                            "error": f"generation timed out after {self.job_timeout} seconds",
                            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "duration": f"{elapsed:.1f}s",
                            "elapsed_seconds": round(elapsed, 1),
                            "eta_seconds": None,
                            "phase_label": "生成超时",
                        })
                except Exception as exc:
                    with job_lock:
                        job.update({
                            "status": "failed",
                            "success": False,
                            "error": str(exc),
                            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        })
                finally:
                    cleanup_upload_paths(job.get("upload_paths", []))
                    with job_lock:
                        if current_job == job_id:
                            current_job = None
            finally:
                self.job_queue.task_done()

    def server_close(self):
        self.video_worker.stop()
        super().server_close()


def main():
    parser = argparse.ArgumentParser(description="H3 Studio Web UI Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--model-dir", default=None,
                        help="Path to MiniMax-H3 model directory (default: local Turbo when available)")
    parser.add_argument("--job-timeout", type=int, default=3600,
                        help="Maximum seconds for one queued generation")
    args = parser.parse_args()

    preferred = TURBO_MODEL_DIR if checkpoint_ready(TURBO_MODEL_DIR, "FL2VA") else BASE_MODEL_DIR
    model_dir = Path(args.model_dir).resolve() if args.model_dir else preferred.resolve()
    if not (model_dir / "FL2VA").exists():
        print(f"Warning: {model_dir}/FL2VA not found. Model may not load correctly.")

    server = H3Server(("", args.port), H3APIHandler, model_dir, args.job_timeout)
    print(f"\n  H3 Studio → http://localhost:{args.port}")
    print(f"  Model:     {model_dir}")
    print(f"  Outputs:   {OUTPUTS}")
    print(f"  Work dir:  {HERE}")
    print(f"  FFmpeg:    {server.process_env['H3_FFMPEG']}")
    print(f"  FFprobe:   {server.process_env['H3_FFPROBE']}")
    print(f"  Timeout:   {args.job_timeout}s per task")
    print(f"  Press Ctrl+C to stop\n")

    def stop_on_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
