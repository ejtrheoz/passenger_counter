"""Detect, track, and count ksiva presentations in high-activity video intervals.

GPU inference is performed only for activity clips. Raw detections are preserved
so that the spatial/temporal post-processing can later be repeated on CPU with
``--reprocess-csv`` without running the detector again.

Examples:
    python aggregate_ksiva.py --video mp4_out/1.mp4 --devices 0 --workers 1
    python aggregate_ksiva.py --reprocess-csv run/ksiva_detections_raw.csv.gz \
        --output-dir run_reprocessed --fps 25
"""

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_VIDEO = WORKSPACE / "mp4_out" / "1.mp4"
DEFAULT_OUTPUT_DIR = WORKSPACE / "mp4_out" / "ksiva_activity"
DETECTOR_SCRIPT = WORKSPACE / "detect_ksiva_video.py"
DETECTION_COLUMNS = ["frame", "time_seconds", "class", "confidence", "x1", "y1", "x2", "y2"]
CLEAN_COLUMNS = [*DETECTION_COLUMNS, "ksiva_id"]
SUMMARY_COLUMNS = [
    "ksiva_id",
    "frame_count",
    "detection_count",
    "duration_seconds",
    "observed_duration_seconds",
    "observed_frame_coverage",
    "start_time",
    "end_time",
    "start_frame",
    "end_frame",
    "avg_confidence",
    "median_confidence",
    "edge_touch_fraction",
    "top_touch_fraction",
    "left_touch_fraction",
    "min_y1",
    "median_y1",
    "center_span_px",
    "size_span_px",
    "review_flags",
    "source_kind",
    "source_ids",
    "source_id_count",
    "source_track_ids",
    "merged_fragment_count",
]
REJECTED_COLUMNS = ["track_id", *SUMMARY_COLUMNS[1:], "rejection_reason"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and spatially track ksiva presentations.",
        allow_abbrev=False,
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Source video.")
    parser.add_argument(
        "--reprocess-csv",
        type=Path,
        help="CPU-only mode: post-process an existing raw or cleaned detection CSV instead of running YOLO.",
    )
    parser.add_argument("--fps", type=float, help="FPS override for --reprocess-csv (normally inferred from frame/time columns).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--detector-script", type=Path, default=DETECTOR_SCRIPT, help="Path to detect_ksiva_video.py.")
    parser.add_argument("--devices", default="0", help="Comma-separated CUDA device IDs, such as 0 or 0,1.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum persistent detector processes; capped to one process per unique device.",
    )
    parser.add_argument(
        "--write-annotated",
        action="store_true",
        help="Draw and encode segment MP4 files. Off by default because encoding does not affect CSV detections.",
    )
    parser.add_argument(
        "--performance-log",
        type=Path,
        help="Aggregate timing/resource JSON (default: OUTPUT_DIR/performance.metrics.json).",
    )
    parser.add_argument("--diff-seconds", type=float, default=5.0, help="Compare each frame to this many seconds earlier.")
    parser.add_argument("--resize-width", type=int, default=320, help="Width used for activity analysis.")
    parser.add_argument("--resize-height", type=int, default=240, help="Height used for activity analysis.")
    parser.add_argument("--pixel-threshold", type=int, default=25, help="Per-pixel difference threshold.")
    parser.add_argument("--peak-height", type=float, default=15000, help="Minimum changed-pixel count for an activity peak.")
    parser.add_argument("--noise-threshold", type=float, default=5000, help="Changed-pixel count where an activity interval ends.")
    parser.add_argument("--peak-distance-frames", type=int, default=25, help="Minimum frames between activity peaks.")
    parser.add_argument("--merge-gap-seconds", type=float, default=2.0, help="Merge nearby activity intervals.")
    parser.add_argument("--context-seconds", type=float, default=3.0, help="Extra video around activity intervals.")
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=0.4,
        help="Maximum detection gap within one spatial track; normalized by FPS.",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        help="Legacy explicit frame-gap override. Prefer --max-gap-seconds.",
    )
    parser.add_argument("--min-frames", type=int, default=10, help="Minimum unique frames in a valid presentation.")
    parser.add_argument("--min-avg-confidence", type=float, default=0.60, help="Minimum mean confidence.")
    parser.add_argument("--dedup-iou", type=float, default=0.70, help="Same-frame IoU used to remove duplicate boxes.")
    parser.add_argument("--match-iou", type=float, default=0.10, help="Minimum IoU for raw-detection spatial tracking.")
    parser.add_argument(
        "--distance-only-max-gap-frames",
        type=int,
        default=2,
        help=(
            "Maximum frame delta for raw association based only on center distance (default: 2). "
            "Strong IoU matches retain --max-gap-seconds/--max-frame-gap; limiting distance-only jumps "
            "reduces dense-scene undercount at the risk of fragmenting fast or intermittently detected objects."
        ),
    )
    parser.add_argument(
        "--seed-merge-iou",
        type=float,
        default=0.90,
        help="Strict IoU for merging adjacent legacy ksiva_id groups from an already-cleaned CSV.",
    )
    parser.add_argument(
        "--max-center-distance",
        type=float,
        default=1.5,
        help="Maximum center displacement in multiples of the larger box diagonal.",
    )
    parser.add_argument("--max-area-ratio", type=float, default=4.0, help="Maximum area ratio for track matching and second-pass merging.")
    parser.add_argument(
        "--second-pass-merge-iou",
        type=float,
        default=0.80,
        help=(
            "Minimum endpoint IoU for conservatively joining non-overlapping first-pass fragments "
            "(default: 0.80, covering the documented run 4 pair at about 0.83). Strict center-distance, "
            "area-ratio, non-overlap, and maximum-gap gates still apply."
        ),
    )
    parser.add_argument(
        "--second-pass-max-center-distance",
        type=float,
        default=0.15,
        help="Maximum normalized endpoint-center distance for conservative second-pass fragment merging.",
    )
    parser.add_argument(
        "--second-pass-max-gap-seconds",
        type=float,
        default=3.0,
        help=(
            "Maximum temporal gap for conservative second-pass merging (default: 3 seconds); "
            "longer joins are deliberately avoided because they can merge different presentations."
        ),
    )
    parser.add_argument("--edge-margin-pixels", type=float, default=2.0, help="Top/left margin treated as touching the image edge.")
    parser.add_argument(
        "--max-edge-touch-fraction",
        type=float,
        default=0.90,
        help="Flag tracks touching the top/left edge in at least this fraction of frames.",
    )
    parser.add_argument(
        "--top-border-min-touch-fraction",
        type=float,
        default=0.80,
        help="Minimum top-border touch fraction for a persistent limited-motion candidate (default: 0.80).",
    )
    parser.add_argument(
        "--top-border-min-observed-seconds",
        type=float,
        default=1.0,
        help="Minimum observed duration for a persistent limited-motion top-border candidate (default: 1.0 second).",
    )
    parser.add_argument(
        "--top-border-max-center-span-pixels",
        type=float,
        default=24.0,
        help="Maximum center span for a persistent limited-motion top-border candidate (default: 24 px).",
    )
    parser.add_argument(
        "--top-border-max-size-span-pixels",
        type=float,
        default=37.0,
        help="Maximum width/height span for a persistent limited-motion top-border candidate (default: 37 px).",
    )
    parser.add_argument(
        "--short-top-border-min-touch-fraction",
        type=float,
        default=0.98,
        help="Minimum top-border touch fraction for a short static candidate (default: 0.98).",
    )
    parser.add_argument(
        "--short-top-border-min-observed-seconds",
        type=float,
        default=0.5,
        help="Minimum observed duration for a short static top-border candidate (default: 0.5 seconds).",
    )
    parser.add_argument(
        "--short-top-border-max-center-span-pixels",
        type=float,
        default=4.0,
        help="Maximum center span for a short static top-border candidate (default: 4 px).",
    )
    parser.add_argument(
        "--short-top-border-max-size-span-pixels",
        type=float,
        default=6.0,
        help="Maximum width/height span for a short static top-border candidate (default: 6 px).",
    )
    parser.add_argument(
        "--static-min-seconds",
        type=float,
        default=1.5,
        help=(
            "Minimum observed duration (unique observed frames / FPS) for a static-artifact flag "
            "(default: 1.5 seconds); elapsed gaps do not count as static coverage."
        ),
    )
    parser.add_argument(
        "--static-max-center-span-pixels",
        type=float,
        default=6.0,
        help="Maximum center span for a static-artifact flag (default: 6 px).",
    )
    parser.add_argument(
        "--static-max-size-span-pixels",
        type=float,
        default=10.0,
        help="Maximum width/height span for a static-artifact flag (default: 10 px).",
    )
    parser.add_argument(
        "--static-max-span-pixels",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility override that sets both static center and size spans. "
            "Prefer the two separate --static-max-*-span-pixels options."
        ),
    )
    parser.add_argument(
        "--reject-edge-artifacts",
        action="store_true",
        help="Exclude edge-flagged tracks. Off by default because valid examples also touch frame edges.",
    )
    top_border_group = parser.add_mutually_exclusive_group()
    top_border_group.add_argument(
        "--keep-top-border-artifacts",
        dest="reject_top_border_artifacts",
        action="store_false",
        help=(
            "Keep tracks matching the compound top-border limited-motion criteria. This opts out of "
            "their default rejection."
        ),
    )
    top_border_group.add_argument(
        "--reject-top-border-artifacts",
        dest="reject_top_border_artifacts",
        action="store_true",
        help="Explicitly select the default rejection of compound top-border artifact candidates.",
    )
    parser.set_defaults(reject_top_border_artifacts=True)
    static_group = parser.add_mutually_exclusive_group()
    static_group.add_argument(
        "--keep-static-artifacts",
        dest="reject_static_artifacts",
        action="store_false",
        help=(
            "Keep tracks matching the static criterion. This opts out of the default rejection and can "
            "retain real held-still documents, but also preserves logged static background false positives."
        ),
    )
    static_group.add_argument(
        "--reject-static-artifacts",
        dest="reject_static_artifacts",
        action="store_true",
        help="Compatibility flag explicitly selecting the default exclusion of static-artifact candidates.",
    )
    parser.set_defaults(reject_static_artifacts=True)
    parser.add_argument("--discard-raw", action="store_true", help="Do not preserve the combined raw detection CSV.")
    parser.add_argument("--keep-clips", action="store_true", help="Keep temporary source clips after detection.")
    parser.add_argument(
        "--keep-temporary-on-error",
        action="store_true",
        help="Keep manifests and segment CSVs when a run fails for debugging.",
    )
    parser.add_argument("detector_args", nargs=argparse.REMAINDER, help="Arguments forwarded to detect_ksiva_video.py; add them after --.")
    return parser.parse_args()


def _canonical_devices(value):
    devices = []
    for raw_device in value.split(","):
        device = raw_device.strip().lower()
        if not device:
            continue
        if device == "cpu":
            canonical = "cpu"
        else:
            if device.startswith("cuda:"):
                device = device.split(":", 1)[1]
            if not device.isdigit():
                raise ValueError("--devices entries must be non-negative CUDA indices or 'cpu'")
            canonical = str(int(device))
        devices.append(canonical)
    if not devices:
        raise ValueError("--devices must contain at least one device ID")
    if len(devices) != len(set(devices)):
        raise ValueError("--devices aliases resolve to the same GPU; use each physical device only once")
    if "cpu" in devices and len(devices) > 1:
        raise ValueError("--devices cannot mix 'cpu' with CUDA device IDs")
    return devices


def validate(args):
    if args.reprocess_csv:
        if not args.reprocess_csv.is_file():
            raise FileNotFoundError(f"Detection CSV not found: {args.reprocess_csv}")
        if args.output_dir.resolve() == args.reprocess_csv.resolve().parent:
            raise ValueError("Use a separate --output-dir for reprocessing so the source run is not overwritten")
    else:
        for path in (args.video, args.detector_script):
            if not path.is_file():
                raise FileNotFoundError(f"Required path not found: {path}")
        _canonical_devices(args.devices)
    if args.workers < 1 or args.diff_seconds <= 0 or args.resize_width < 1 or args.resize_height < 1:
        raise ValueError("--workers, --diff-seconds, and resize dimensions must be positive")
    if args.peak_distance_frames < 1 or args.merge_gap_seconds < 0 or args.context_seconds < 0:
        raise ValueError("Activity interval thresholds must be valid")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    if (
        args.max_gap_seconds < 0
        or (args.max_frame_gap is not None and args.max_frame_gap < 0)
        or args.distance_only_max_gap_frames < 0
        or args.second_pass_max_gap_seconds < 0
    ):
        raise ValueError("Ksiva gap thresholds cannot be negative")
    if args.min_frames < 1 or not 0 <= args.min_avg_confidence <= 1:
        raise ValueError("Ksiva filtering thresholds must be valid")
    for name in (
        "dedup_iou",
        "match_iou",
        "seed_merge_iou",
        "second_pass_merge_iou",
        "max_edge_touch_fraction",
        "top_border_min_touch_fraction",
        "short_top_border_min_touch_fraction",
    ):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if (
        args.max_center_distance <= 0
        or args.second_pass_max_center_distance < 0
        or args.max_area_ratio < 1
    ):
        raise ValueError("Spatial matching thresholds must be positive (second-pass distance may be zero)")
    static_spans = (
        args.static_max_center_span_pixels,
        args.static_max_size_span_pixels,
        args.static_max_span_pixels,
    )
    top_border_thresholds = (
        args.top_border_min_observed_seconds,
        args.top_border_max_center_span_pixels,
        args.top_border_max_size_span_pixels,
        args.short_top_border_min_observed_seconds,
        args.short_top_border_max_center_span_pixels,
        args.short_top_border_max_size_span_pixels,
    )
    if (
        args.edge_margin_pixels < 0
        or args.static_min_seconds < 0
        or any(value is not None and value < 0 for value in static_spans)
        or any(not math.isfinite(value) or value < 0 for value in top_border_thresholds)
    ):
        raise ValueError("Artifact thresholds must be finite and non-negative")


def _atomic_json_write(path, payload):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _process_memory():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    snapshot = {
        "user_cpu_seconds": round(usage.ru_utime, 6),
        "system_cpu_seconds": round(usage.ru_stime, 6),
        "peak_rss_mib": round(usage.ru_maxrss / 1024, 3),
        "rss_mib": None,
    }
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                snapshot["rss_mib"] = round(float(line.split()[1]) / 1024, 3)
                break
    except (OSError, ValueError, IndexError):
        pass
    return snapshot


@contextmanager
def _phase(performance, name):
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        yield
    finally:
        entry = performance.setdefault("phases", {}).setdefault(
            name,
            {"wall_seconds": 0.0, "cpu_seconds": 0.0, "calls": 0},
        )
        entry["wall_seconds"] = round(entry["wall_seconds"] + time.perf_counter() - wall_start, 6)
        entry["cpu_seconds"] = round(entry["cpu_seconds"] + time.process_time() - cpu_start, 6)
        entry["calls"] += 1


def _forwarded_detector_args(detector_args):
    forwarded = list(detector_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    reserved = {"--video", "--output", "--log", "--jobs-manifest", "--metrics-log", "--device"}
    conflicts = sorted(
        token
        for token in forwarded
        if token.split("=", 1)[0] in reserved
    )
    if conflicts:
        raise ValueError(
            "Detector arguments managed by aggregate_ksiva.py cannot be overridden: "
            + ", ".join(conflicts)
        )
    return forwarded


def run_detector_worker(detector_script, manifest_path, metrics_path, worker_log_path, device, detector_args):
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_jobs = manifest_payload.get("jobs", [])
    metrics_path.unlink(missing_ok=True)
    for job in expected_jobs:
        Path(job["log"]).unlink(missing_ok=True)
        if job.get("output"):
            Path(job["output"]).unlink(missing_ok=True)
    command = [
        sys.executable,
        str(detector_script),
        "--jobs-manifest", str(manifest_path),
        "--metrics-log", str(metrics_path),
        "--device", device,
        *detector_args,
    ]
    wall_start = time.perf_counter()
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    with worker_log_path.open("w", encoding="utf-8") as worker_log:
        result = subprocess.run(command, text=True, stdout=worker_log, stderr=subprocess.STDOUT)
    observed_wall = time.perf_counter() - wall_start
    if result.returncode:
        try:
            tail = "\n".join(worker_log_path.read_text(encoding="utf-8").splitlines()[-40:])
        except OSError:
            tail = ""
        raise RuntimeError(
            f"Detector worker on device {device} failed with exit code {result.returncode}. "
            f"Log: {worker_log_path}\n{tail}"
        )
    if not metrics_path.is_file():
        raise RuntimeError(f"Detector worker exited successfully but wrote no metrics: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(f"Detector worker metrics report status={payload.get('status')!r}: {metrics_path}")
    recorded_manifest = payload.get("manifest")
    if recorded_manifest is None or Path(recorded_manifest).resolve() != manifest_path.resolve():
        raise RuntimeError(f"Detector worker processed an unexpected manifest: {recorded_manifest!r}")
    if str(payload.get("device")) != str(device):
        raise RuntimeError(f"Detector worker used device {payload.get('device')!r}, expected {device!r}")
    actual_jobs = payload.get("jobs", [])
    if [job.get("index") for job in actual_jobs] != [job.get("index") for job in expected_jobs]:
        raise RuntimeError("Detector worker metrics do not match the scheduled job list")
    missing_logs = [job["log"] for job in expected_jobs if not Path(job["log"]).is_file()]
    if missing_logs:
        raise RuntimeError(f"Detector worker did not create expected segment CSVs: {missing_logs}")
    payload["parent_observed_wall_seconds"] = round(observed_wall, 6)
    child_wall = payload.get("run", {}).get("wall_seconds")
    payload["subprocess_startup_and_import_overhead_seconds"] = (
        round(max(0.0, observed_wall - float(child_wall)), 6) if child_wall is not None else None
    )
    payload["worker_log"] = worker_log_path.name
    payload["metrics_file"] = metrics_path.name
    return payload


def _device_queues(jobs, devices, worker_limit):
    worker_count = min(worker_limit, len(devices), len(jobs))
    queues = [
        {"device": device, "jobs": [], "estimated_video_seconds": 0.0}
        for device in devices[:worker_count]
    ]
    for job in sorted(jobs, key=lambda item: item["end_seconds"] - item["start_seconds"], reverse=True):
        queue = min(queues, key=lambda item: item["estimated_video_seconds"])
        queue["jobs"].append(job)
        queue["estimated_video_seconds"] += job["end_seconds"] - job["start_seconds"]
    for queue in queues:
        queue["jobs"].sort(key=lambda item: item["index"])
        queue["estimated_video_seconds"] = round(queue["estimated_video_seconds"], 3)
    return queues


def _detector_summary(workers, detector_wall_seconds):
    phase_totals = {}
    yolo_totals_ms = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    total_frames = 0
    total_video_seconds = 0.0
    cumulative_job_wall = 0.0
    model_load_seconds = 0.0
    startup_overhead_seconds = 0.0
    peak_allocated_mib = None
    for worker in workers:
        model_load_seconds += float(worker.get("model_load", {}).get("wall_seconds", 0.0))
        startup_overhead_seconds += float(worker.get("subprocess_startup_and_import_overhead_seconds") or 0.0)
        for job in worker.get("jobs", []):
            if job.get("status") == "failed":
                continue
            total_frames += int(job.get("frames", 0))
            total_video_seconds += float(job.get("video_seconds", 0.0))
            cumulative_job_wall += float(job.get("wall_seconds", 0.0))
            for name, value in job.get("phases", {}).items():
                phase_totals[name] = phase_totals.get(name, 0.0) + float(value)
            for name, values in job.get("yolo_speed", {}).items():
                if values and name in yolo_totals_ms:
                    yolo_totals_ms[name] += float(values.get("total_ms", 0.0))
            gpu = job.get("gpu_memory_end") or {}
            peak = gpu.get("peak_allocated_mib")
            if peak is not None:
                peak_allocated_mib = max(peak_allocated_mib or 0.0, float(peak))
    return {
        "frames": total_frames,
        "video_seconds": round(total_video_seconds, 6),
        "detector_wall_seconds": round(detector_wall_seconds, 6),
        "effective_fps": round(total_frames / detector_wall_seconds, 3) if detector_wall_seconds else None,
        "effective_realtime_factor": round(total_video_seconds / detector_wall_seconds, 3) if detector_wall_seconds else None,
        "cumulative_job_wall_seconds": round(cumulative_job_wall, 6),
        "cumulative_model_load_seconds": round(model_load_seconds, 6),
        "cumulative_subprocess_startup_import_seconds": round(startup_overhead_seconds, 6),
        "phase_totals_seconds": {name: round(value, 6) for name, value in sorted(phase_totals.items())},
        "yolo_totals_ms": {name: round(value, 3) for name, value in yolo_totals_ms.items()},
        "max_process_peak_allocated_vram_mib": round(peak_allocated_mib, 3) if peak_allocated_mib is not None else None,
    }


def _print_performance_summary(performance):
    print("Performance phase summary:")
    phases = sorted(
        performance.get("phases", {}).items(),
        key=lambda item: item[1]["wall_seconds"],
        reverse=True,
    )
    for name, values in phases:
        share = performance.get("phase_share_percent", {}).get(name)
        print(f"  {name}: {values['wall_seconds']:.3f}s ({share:.1f}% of run)")
    detector = performance.get("detector_summary")
    if detector:
        print(
            f"Detector: {detector['frames']} frames, {detector['effective_fps']} FPS, "
            f"{detector['effective_realtime_factor']}x realtime; "
            f"peak allocated VRAM={detector['max_process_peak_allocated_vram_mib']} MiB"
        )
        for name, value in detector["phase_totals_seconds"].items():
            print(f"  detector.{name}: {value:.3f}s cumulative across jobs")


def _box_values(row):
    return tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2"))


def _box_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _box_distance_and_area_ratio(first, second):
    first_width, first_height = first[2] - first[0], first[3] - first[1]
    second_width, second_height = second[2] - second[0], second[3] - second[1]
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    center_distance = math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])
    diagonal = max(math.hypot(first_width, first_height), math.hypot(second_width, second_height), 1.0)
    first_area = max(first_width * first_height, 1.0)
    second_area = max(second_width * second_height, 1.0)
    return center_distance / diagonal, max(first_area, second_area) / min(first_area, second_area)


def _prepare_detections(detections, dedup_iou):
    missing = [column for column in DETECTION_COLUMNS if column not in detections.columns]
    if missing:
        raise ValueError(f"Detection CSV is missing columns: {', '.join(missing)}")
    source_id_columns = ["ksiva_id"] if "ksiva_id" in detections.columns else []
    prepared = detections[[*DETECTION_COLUMNS, *source_id_columns]].copy()
    if source_id_columns:
        prepared = prepared.rename(columns={"ksiva_id": "source_ksiva_id"})
        source_ids = pd.to_numeric(prepared["source_ksiva_id"], errors="coerce")
        valid_source_ids = (
            source_ids.notna()
            & source_ids.map(math.isfinite)
            & (source_ids % 1 == 0)
            & (source_ids >= 1)
        )
        if not valid_source_ids.all():
            invalid_rows = prepared.index[~valid_source_ids].tolist()
            preview = ", ".join(str(index) for index in invalid_rows[:10])
            suffix = "..." if len(invalid_rows) > 10 else ""
            raise ValueError(
                "Seeded detection CSV contains null, non-integer, or non-positive ksiva_id "
                f"at row index(es): {preview}{suffix}"
            )
        prepared["source_ksiva_id"] = source_ids.astype(int)
    prepared = prepared[prepared["class"].astype(str).str.casefold() == "ksiva"]
    numeric_columns = ["frame", "time_seconds", "confidence", "x1", "y1", "x2", "y2"]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    finite_rows = prepared[numeric_columns].apply(lambda column: column.map(math.isfinite)).all(axis=1)
    prepared = prepared[finite_rows]
    prepared = prepared[prepared["frame"] % 1 == 0]
    prepared["frame"] = prepared["frame"].astype(int)
    prepared = prepared[
        (prepared["confidence"] >= 0)
        & (prepared["confidence"] <= 1)
        & (prepared["x2"] > prepared["x1"])
        & (prepared["y2"] > prepared["y1"])
    ]

    kept_indices = []
    for _, frame_rows in prepared.groupby("frame", sort=True):
        frame_kept = []
        for index, row in frame_rows.sort_values("confidence", ascending=False).iterrows():
            box = _box_values(row)
            if all(_box_iou(box, kept_box) < dedup_iou for kept_box in frame_kept):
                kept_indices.append(index)
                frame_kept.append(box)
    return prepared.loc[kept_indices].sort_values(["frame", "confidence"], ascending=[True, False]).reset_index(drop=True)


def _assign_tracks(
    detections,
    max_gap_frames,
    match_iou,
    max_center_distance,
    max_area_ratio,
    distance_only_max_gap_frames,
):
    """Build raw tracks while allowing weak distance-only links only across short gaps.

    IoU evidence remains valid for the configured track horizon. Restricting the
    less-specific distance fallback reduces long jumps between nearby objects in
    dense scenes, at the explicit risk of fragmenting a rapidly moving object.
    """
    tracked = detections.copy()
    tracked["track_id"] = -1
    active_tracks = {}
    next_track_id = 1

    for frame, frame_rows in tracked.groupby("frame", sort=True):
        frame = int(frame)
        active_tracks = {
            track_id: state
            for track_id, state in active_tracks.items()
            if frame - state["last_frame"] <= max_gap_frames
        }
        candidates = []
        for row_index, row in frame_rows.iterrows():
            box = _box_values(row)
            for track_id, state in active_tracks.items():
                frame_delta = frame - state["last_frame"]
                iou = _box_iou(box, state["box"])
                distance, area_ratio = _box_distance_and_area_ratio(box, state["box"])
                iou_match = iou >= match_iou
                distance_match = (
                    frame_delta <= distance_only_max_gap_frames
                    and distance <= max_center_distance
                )
                if area_ratio <= max_area_ratio and (iou_match or distance_match):
                    candidates.append((iou - 0.15 * distance, iou, -distance, track_id, row_index, box))

        matched_tracks = set()
        matched_rows = set()
        for _, _, _, track_id, row_index, box in sorted(candidates, reverse=True):
            if track_id in matched_tracks or row_index in matched_rows:
                continue
            tracked.at[row_index, "track_id"] = track_id
            active_tracks[track_id] = {"last_frame": frame, "box": box}
            matched_tracks.add(track_id)
            matched_rows.add(row_index)

        for row_index, row in frame_rows.iterrows():
            if row_index in matched_rows:
                continue
            box = _box_values(row)
            track_id = next_track_id
            next_track_id += 1
            tracked.at[row_index, "track_id"] = track_id
            active_tracks[track_id] = {"last_frame": frame, "box": box}
    tracked["track_id"] = tracked["track_id"].astype(int)
    return tracked


def _merge_seeded_groups(detections, max_gap_frames, seed_merge_iou, max_area_ratio):
    """Merge legacy groups and retain their factual source_ksiva_id lineage."""
    tracked = detections.copy()
    tracked["track_id"] = -1
    groups = []
    for _, group in tracked.groupby("source_ksiva_id", sort=False):
        ordered = group.sort_values(["frame", "confidence"], ascending=[True, False])
        start_frame = int(ordered["frame"].min())
        end_frame = int(ordered["frame"].max())
        first_box = _box_values(ordered[ordered["frame"] == start_frame].iloc[0])
        last_box = _box_values(ordered[ordered["frame"] == end_frame].iloc[0])
        groups.append((start_frame, end_frame, list(group.index), first_box, last_box))

    active_tracks = {}
    next_track_id = 1
    for start_frame, end_frame, row_indices, first_box, last_box in sorted(groups):
        active_tracks = {
            track_id: state
            for track_id, state in active_tracks.items()
            if start_frame - state["last_frame"] <= max_gap_frames
        }
        candidates = []
        for track_id, state in active_tracks.items():
            iou = _box_iou(first_box, state["box"])
            _, area_ratio = _box_distance_and_area_ratio(first_box, state["box"])
            if area_ratio <= max_area_ratio and iou >= seed_merge_iou:
                candidates.append((iou, track_id))
        if candidates:
            track_id = max(candidates)[1]
        else:
            track_id = next_track_id
            next_track_id += 1
        tracked.loc[row_indices, "track_id"] = track_id
        active_tracks[track_id] = {"last_frame": end_frame, "box": last_box}

    tracked["track_id"] = tracked["track_id"].astype(int)
    lineage_by_track = {}
    for track_id, group in tracked.groupby("track_id", sort=True):
        source_ids = []
        for source_id in group["source_ksiva_id"].drop_duplicates().tolist():
            if hasattr(source_id, "item"):
                source_id = source_id.item()
            if isinstance(source_id, float) and source_id.is_integer():
                source_id = int(source_id)
            source_ids.append(source_id)
        lineage_by_track[int(track_id)] = sorted(source_ids, key=lambda value: (str(type(value)), str(value)))
    tracked["source_kind"] = "legacy_ksiva_id"
    tracked["source_ids"] = tracked["track_id"].map(
        lambda track_id: json.dumps(lineage_by_track[int(track_id)], separators=(",", ":"), ensure_ascii=False)
    )
    tracked["source_id_count"] = tracked["track_id"].map(
        lambda track_id: len(lineage_by_track[int(track_id)])
    ).astype(int)
    # Compatibility columns retain factual lineage; source_kind disambiguates
    # legacy ksiva IDs from raw first-pass track IDs.
    tracked["source_track_ids"] = tracked["source_ids"]
    tracked["merged_fragment_count"] = tracked["source_id_count"]
    return tracked


def _merge_track_fragments(
    tracked,
    fps,
    endpoint_iou_threshold,
    max_center_distance,
    max_area_ratio,
    max_gap_seconds,
):
    """Conservatively join deterministic chains of non-overlapping first-pass tracks.

    The endpoint gates intentionally require strong agreement on IoU, normalized
    center distance, area, and time. This repairs obvious fragmentation without
    optimizing toward a target count; looser thresholds risk merging distinct
    nearby presentations. A fragment may have at most one predecessor and one
    successor, and component time ranges are checked before every transitive join.
    """
    merged = tracked.copy()
    descriptors = {}
    for track_id, group in merged.groupby("track_id", sort=True):
        track_id = int(track_id)
        start_frame = int(group["frame"].min())
        end_frame = int(group["frame"].max())
        endpoint_sort = ["confidence", "x1", "y1", "x2", "y2"]
        first_row = group[group["frame"] == start_frame].sort_values(
            endpoint_sort,
            ascending=[False, True, True, True, True],
            kind="mergesort",
        ).iloc[0]
        last_row = group[group["frame"] == end_frame].sort_values(
            endpoint_sort,
            ascending=[False, True, True, True, True],
            kind="mergesort",
        ).iloc[0]
        descriptors[track_id] = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "first_box": _box_values(first_row),
            "last_box": _box_values(last_row),
        }

    track_ids = sorted(descriptors)
    parent = {track_id: track_id for track_id in track_ids}
    members = {track_id: {track_id} for track_id in track_ids}
    component_start = {track_id: descriptors[track_id]["start_frame"] for track_id in track_ids}
    component_end = {track_id: descriptors[track_id]["end_frame"] for track_id in track_ids}

    def find(track_id):
        root = track_id
        while parent[root] != root:
            root = parent[root]
        while parent[track_id] != track_id:
            next_track_id = parent[track_id]
            parent[track_id] = root
            track_id = next_track_id
        return root

    candidates = []
    for left_id in track_ids:
        left = descriptors[left_id]
        for right_id in track_ids:
            right = descriptors[right_id]
            if left["end_frame"] >= right["start_frame"]:
                continue
            gap_seconds = (right["start_frame"] - left["end_frame"]) / fps
            if gap_seconds > max_gap_seconds:
                continue
            iou = _box_iou(left["last_box"], right["first_box"])
            distance, area_ratio = _box_distance_and_area_ratio(left["last_box"], right["first_box"])
            if (
                iou >= endpoint_iou_threshold
                and distance <= max_center_distance
                and area_ratio <= max_area_ratio
            ):
                candidates.append((gap_seconds, -iou, distance, area_ratio, left_id, right_id))

    predecessor = set()
    successor = set()
    for _, _, _, _, left_id, right_id in sorted(candidates):
        if left_id in successor or right_id in predecessor:
            continue
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root == right_root or component_end[left_root] >= component_start[right_root]:
            continue
        new_root = min(left_root, right_root)
        old_root = max(left_root, right_root)
        parent[old_root] = new_root
        members[new_root] |= members.pop(old_root)
        component_start[new_root] = min(component_start[new_root], component_start.pop(old_root))
        component_end[new_root] = max(component_end[new_root], component_end.pop(old_root))
        successor.add(left_id)
        predecessor.add(right_id)

    components = sorted(
        ((component_start[root], min(source_ids), root, source_ids) for root, source_ids in members.items()),
        key=lambda item: (item[0], item[1]),
    )
    source_to_final = {}
    lineage_by_final = {}
    for final_track_id, (_, _, _, source_ids) in enumerate(components, start=1):
        sorted_source_ids = sorted(source_ids)
        lineage_by_final[final_track_id] = sorted_source_ids
        for source_track_id in sorted_source_ids:
            source_to_final[source_track_id] = final_track_id

    original_track_ids = merged["track_id"].astype(int)
    merged["track_id"] = original_track_ids.map(source_to_final).astype(int)
    merged["source_kind"] = "first_pass_track_id"
    merged["source_ids"] = merged["track_id"].map(
        lambda track_id: json.dumps(lineage_by_final[int(track_id)], separators=(",", ":"))
    )
    merged["source_id_count"] = merged["track_id"].map(
        lambda track_id: len(lineage_by_final[int(track_id)])
    ).astype(int)
    # Retained for compatibility with existing raw-run consumers.
    merged["source_track_ids"] = merged["source_ids"]
    merged["merged_fragment_count"] = merged["source_id_count"]
    return merged


def _track_statistics(tracked, edge_margin_pixels, fps):
    rows = []
    for track_id, group in tracked.groupby("track_id", sort=False):
        center_x = (group["x1"] + group["x2"]) / 2
        center_y = (group["y1"] + group["y2"]) / 2
        widths = group["x2"] - group["x1"]
        heights = group["y2"] - group["y1"]
        center_span = math.hypot(float(center_x.max() - center_x.min()), float(center_y.max() - center_y.min()))
        size_span = math.hypot(float(widths.max() - widths.min()), float(heights.max() - heights.min()))
        start_time = float(group["time_seconds"].min())
        end_time = float(group["time_seconds"].max())
        frame_count = int(group["frame"].nunique())
        frame_span = int(group["frame"].max() - group["frame"].min()) + 1
        rows.append(
            {
                "track_id": int(track_id),
                "frame_count": frame_count,
                "detection_count": int(len(group)),
                "duration_seconds": round(end_time - start_time, 3),
                "observed_duration_seconds": round(frame_count / fps, 3),
                "observed_frame_coverage": round(frame_count / frame_span, 6),
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": int(group["frame"].min()),
                "end_frame": int(group["frame"].max()),
                "avg_confidence": float(group["confidence"].mean()),
                "median_confidence": float(group["confidence"].median()),
                "edge_touch_fraction": float(((group["x1"] <= edge_margin_pixels) | (group["y1"] <= edge_margin_pixels)).mean()),
                "top_touch_fraction": float((group["y1"] <= edge_margin_pixels).mean()),
                "left_touch_fraction": float((group["x1"] <= edge_margin_pixels).mean()),
                "min_y1": float(group["y1"].min()),
                "median_y1": float(group["y1"].median()),
                "center_span_px": round(center_span, 3),
                "size_span_px": round(size_span, 3),
                "source_kind": str(group["source_kind"].iloc[0]),
                "source_ids": str(group["source_ids"].iloc[0]),
                "source_id_count": int(group["source_id_count"].iloc[0]),
                "source_track_ids": str(group["source_track_ids"].iloc[0]),
                "merged_fragment_count": int(group["merged_fragment_count"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def postprocess_detections(
    detections,
    fps,
    max_gap_seconds=0.4,
    max_frame_gap=None,
    min_frames=10,
    min_avg_confidence=0.60,
    dedup_iou=0.70,
    match_iou=0.10,
    seed_merge_iou=0.90,
    max_center_distance=1.5,
    max_area_ratio=4.0,
    edge_margin_pixels=2.0,
    max_edge_touch_fraction=0.90,
    static_min_seconds=1.5,
    static_max_span_pixels=None,
    reject_edge_artifacts=False,
    reject_static_artifacts=True,
    distance_only_max_gap_frames=2,
    second_pass_merge_iou=0.80,
    second_pass_max_center_distance=0.15,
    second_pass_max_gap_seconds=3.0,
    static_max_center_span_pixels=6.0,
    static_max_size_span_pixels=10.0,
    top_border_min_touch_fraction=0.80,
    top_border_min_observed_seconds=1.0,
    top_border_max_center_span_pixels=24.0,
    top_border_max_size_span_pixels=37.0,
    short_top_border_min_touch_fraction=0.98,
    short_top_border_min_observed_seconds=0.5,
    short_top_border_max_center_span_pixels=4.0,
    short_top_border_max_size_span_pixels=6.0,
    reject_top_border_artifacts=True,
):
    """Deduplicate, associate, conservatively repair fragments, and filter events.

    Distance-only raw links are short-range because long spatial jumps caused
    dense-run undercount. The second pass defaults to 0.80 endpoint IoU, while
    retaining strict center-distance, area, non-overlap, and three-second gap
    gates, and records factual source lineage. Static minimum duration uses
    observed unique-frame coverage (unique frames / FPS), not elapsed endpoint
    span, so empty gaps cannot turn short fragments into static candidates.
    ``static_max_span_pixels`` remains a compatibility override for both newer
    static span thresholds. The general top/left edge flag is audit-only by
    default. Compound top-border candidates are rejected by default and require
    top-edge persistence, observed duration, and limited center and size motion
    simultaneously, so real documents quickly crossing the top of the frame are
    not rejected solely for touching it.
    """
    if static_max_span_pixels is not None:
        static_max_center_span_pixels = static_max_span_pixels
        static_max_size_span_pixels = static_max_span_pixels
    if detections.empty:
        return (
            pd.DataFrame(columns=CLEAN_COLUMNS),
            pd.DataFrame(columns=SUMMARY_COLUMNS),
            pd.DataFrame(columns=REJECTED_COLUMNS),
        )
    gap_frames = max_frame_gap if max_frame_gap is not None else round(max_gap_seconds * fps)
    prepared = _prepare_detections(detections, dedup_iou)
    if prepared.empty:
        return (
            pd.DataFrame(columns=CLEAN_COLUMNS),
            pd.DataFrame(columns=SUMMARY_COLUMNS),
            pd.DataFrame(columns=REJECTED_COLUMNS),
        )
    seeded_input = "source_ksiva_id" in prepared.columns
    if seeded_input:
        tracked = _merge_seeded_groups(
            prepared,
            gap_frames,
            seed_merge_iou,
            max_area_ratio,
        )
    else:
        tracked = _assign_tracks(
            prepared,
            gap_frames,
            match_iou,
            max_center_distance,
            max_area_ratio,
            distance_only_max_gap_frames,
        )
        tracked = _merge_track_fragments(
            tracked,
            fps,
            second_pass_merge_iou,
            second_pass_max_center_distance,
            max_area_ratio,
            second_pass_max_gap_seconds,
        )
    stats = _track_statistics(tracked, edge_margin_pixels, fps)

    def artifact_flags(row):
        flags = []
        if seeded_input and (
            row["frame_count"] < min_frames or row["avg_confidence"] < min_avg_confidence
        ):
            flags.append("legacy_group_below_current_threshold")
        if row["edge_touch_fraction"] >= max_edge_touch_fraction:
            flags.append("persistent_top_or_left_edge")
        if (
            row["observed_duration_seconds"] >= static_min_seconds
            and row["center_span_px"] <= static_max_center_span_pixels
            and row["size_span_px"] <= static_max_size_span_pixels
        ):
            flags.append("static_background_candidate")
        if (
            row["top_touch_fraction"] >= top_border_min_touch_fraction
            and row["observed_duration_seconds"] >= top_border_min_observed_seconds
            and row["center_span_px"] <= top_border_max_center_span_pixels
            and row["size_span_px"] <= top_border_max_size_span_pixels
        ):
            flags.append("persistent_limited_motion_top_border_candidate")
        if (
            row["top_touch_fraction"] >= short_top_border_min_touch_fraction
            and row["observed_duration_seconds"] >= short_top_border_min_observed_seconds
            and row["center_span_px"] <= short_top_border_max_center_span_pixels
            and row["size_span_px"] <= short_top_border_max_size_span_pixels
        ):
            flags.append("short_static_top_border_candidate")
        return ";".join(flags)

    def rejection_reason(row):
        reasons = []
        if not seeded_input and row["frame_count"] < min_frames:
            reasons.append("too_few_unique_frames")
        if not seeded_input and row["avg_confidence"] < min_avg_confidence:
            reasons.append("low_average_confidence")
        flags = set(row["review_flags"].split(";")) if row["review_flags"] else set()
        if reject_edge_artifacts and "persistent_top_or_left_edge" in flags:
            reasons.append("persistent_top_or_left_edge")
        if reject_static_artifacts and "static_background_candidate" in flags:
            reasons.append("static_background_candidate")
        if reject_top_border_artifacts and "persistent_limited_motion_top_border_candidate" in flags:
            reasons.append("persistent_limited_motion_top_border_candidate")
        if reject_top_border_artifacts and "short_static_top_border_candidate" in flags:
            reasons.append("short_static_top_border_candidate")
        return ";".join(reasons)

    stats["review_flags"] = stats.apply(artifact_flags, axis=1)
    stats["rejection_reason"] = stats.apply(rejection_reason, axis=1)
    valid_stats = stats[stats["rejection_reason"] == ""].sort_values(["start_frame", "track_id"]).copy()
    valid_stats["ksiva_id"] = range(1, len(valid_stats) + 1)
    track_to_ksiva = dict(zip(valid_stats["track_id"], valid_stats["ksiva_id"]))

    clean = tracked[tracked["track_id"].isin(track_to_ksiva)].copy()
    clean["ksiva_id"] = clean["track_id"].map(track_to_ksiva)
    clean = clean.sort_values(["frame", "ksiva_id"])[CLEAN_COLUMNS].reset_index(drop=True)
    summary = valid_stats[SUMMARY_COLUMNS].reset_index(drop=True)
    rejected = stats[stats["rejection_reason"] != ""][REJECTED_COLUMNS].sort_values("start_frame").reset_index(drop=True)
    return clean, summary, rejected


def filter_detections(detections, max_frame_gap, min_frames, min_avg_confidence):
    """Compatibility adapter that applies the corrected spatial filter."""
    if detections.empty:
        return pd.DataFrame(columns=CLEAN_COLUMNS), pd.DataFrame(columns=SUMMARY_COLUMNS)
    fps = infer_fps(detections)
    clean, summary, _ = postprocess_detections(
        detections,
        fps=fps,
        max_frame_gap=max_frame_gap,
        min_frames=min_frames,
        min_avg_confidence=min_avg_confidence,
    )
    return clean, summary


def infer_fps(detections):
    if detections.empty:
        raise ValueError("Cannot infer FPS from an empty detection CSV; pass --fps")
    usable = detections[pd.to_numeric(detections["time_seconds"], errors="coerce") > 0].copy()
    if usable.empty:
        raise ValueError("Cannot infer FPS from zero timestamps; pass --fps")
    estimates = pd.to_numeric(usable["frame"], errors="coerce") / pd.to_numeric(usable["time_seconds"], errors="coerce")
    fps = float(estimates.replace([math.inf, -math.inf], pd.NA).dropna().median())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Could not infer a valid FPS; pass --fps")
    return fps


def _filter_settings(args):
    static_center_span = (
        args.static_max_span_pixels
        if args.static_max_span_pixels is not None
        else args.static_max_center_span_pixels
    )
    static_size_span = (
        args.static_max_span_pixels
        if args.static_max_span_pixels is not None
        else args.static_max_size_span_pixels
    )
    return {
        "max_gap_seconds": args.max_gap_seconds,
        "max_frame_gap": args.max_frame_gap,
        "min_frames": args.min_frames,
        "min_avg_confidence": args.min_avg_confidence,
        "dedup_iou": args.dedup_iou,
        "match_iou": args.match_iou,
        "distance_only_max_gap_frames": args.distance_only_max_gap_frames,
        "seed_merge_iou": args.seed_merge_iou,
        "max_center_distance": args.max_center_distance,
        "max_area_ratio": args.max_area_ratio,
        "second_pass_merge_iou": args.second_pass_merge_iou,
        "second_pass_max_center_distance": args.second_pass_max_center_distance,
        "second_pass_max_gap_seconds": args.second_pass_max_gap_seconds,
        "edge_margin_pixels": args.edge_margin_pixels,
        "max_edge_touch_fraction": args.max_edge_touch_fraction,
        "top_border_min_touch_fraction": args.top_border_min_touch_fraction,
        "top_border_min_observed_seconds": args.top_border_min_observed_seconds,
        "top_border_max_center_span_pixels": args.top_border_max_center_span_pixels,
        "top_border_max_size_span_pixels": args.top_border_max_size_span_pixels,
        "short_top_border_min_touch_fraction": args.short_top_border_min_touch_fraction,
        "short_top_border_min_observed_seconds": args.short_top_border_min_observed_seconds,
        "short_top_border_max_center_span_pixels": args.short_top_border_max_center_span_pixels,
        "short_top_border_max_size_span_pixels": args.short_top_border_max_size_span_pixels,
        "static_min_seconds": args.static_min_seconds,
        "static_max_span_pixels": args.static_max_span_pixels,
        "static_max_center_span_pixels": static_center_span,
        "static_max_size_span_pixels": static_size_span,
        "reject_edge_artifacts": args.reject_edge_artifacts,
        "reject_static_artifacts": args.reject_static_artifacts,
        "reject_top_border_artifacts": args.reject_top_border_artifacts,
    }


def _write_outputs(args, combined, fps, intervals, source, source_stage, performance):
    settings = _filter_settings(args)
    with _phase(performance, "pandas_postprocess"):
        clean, summary, rejected = postprocess_detections(combined, fps=fps, **settings)
    raw_path = args.output_dir / "ksiva_detections_raw.csv.gz"
    clean_path = args.output_dir / "ksiva_detections_cleaned.csv"
    summary_path = args.output_dir / "ksiva_unique_objects.csv"
    rejected_path = args.output_dir / "ksiva_rejected_objects.csv"
    history_path = args.output_dir / "aggregate.history.json"

    preserve_raw = source_stage == "raw" and not args.discard_raw
    if preserve_raw:
        with _phase(performance, "raw_csv_gzip_write"):
            combined.to_csv(raw_path, index=False, compression="gzip")
    with _phase(performance, "result_csv_write"):
        clean.to_csv(clean_path, index=False)
        summary.to_csv(summary_path, index=False)
        rejected.to_csv(rejected_path, index=False)

    parent_history = None
    if args.reprocess_csv:
        parent_history_path = args.reprocess_csv.resolve().parent / "aggregate.history.json"
        if parent_history_path.is_file():
            try:
                parent_history = json.loads(parent_history_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                parent_history = {"unreadable_history": str(parent_history_path)}
    source_warning = None
    if source_stage == "cleaned":
        source_warning = (
            "Input was already filtered. Previously rejected detections cannot be recovered; "
            "the output is a retracking of surviving detections, not a full raw reprocess."
        )
    try:
        performance_reference = str(Path(performance["metrics_path"]).resolve().relative_to(args.output_dir.resolve()))
    except ValueError:
        performance_reference = str(Path(performance["metrics_path"]).resolve())

    audit_columns = [
        "source_kind", "source_ids", "source_id_count", "source_track_ids", "merged_fragment_count"
    ]
    audit_tracks = pd.concat(
        [
            summary[["ksiva_id", *audit_columns]].assign(disposition="accepted"),
            rejected[["track_id", *audit_columns]]
            .rename(columns={"track_id": "ksiva_id"})
            .assign(disposition="rejected"),
        ],
        ignore_index=True,
    )
    merged_components = audit_tracks[audit_tracks["source_id_count"] > 1]
    merge_audit = {
        "source_stage": source_stage,
        "source_kind": (
            str(audit_tracks["source_kind"].iloc[0]) if not audit_tracks.empty else
            ("legacy_ksiva_id" if source_stage == "cleaned" else "first_pass_track_id")
        ),
        "preliminary_track_count": int(audit_tracks["source_id_count"].sum()),
        "final_track_count": int(len(audit_tracks)),
        "merged_component_count": int(len(merged_components)),
        "absorbed_fragment_count": int((audit_tracks["source_id_count"] - 1).sum()),
        "components": [
            {
                "disposition": row["disposition"],
                "result_id": int(row["ksiva_id"]),
                "source_kind": row["source_kind"],
                "source_ids": json.loads(row["source_ids"]),
                "source_id_count": int(row["source_id_count"]),
                "source_track_ids": json.loads(row["source_track_ids"]),
                "merged_fragment_count": int(row["merged_fragment_count"]),
            }
            for _, row in merged_components.sort_values(["disposition", "ksiva_id"]).iterrows()
        ],
    }
    history = {
        "schema_version": 5,
        "source": str(source),
        "source_stage": source_stage,
        "source_warning": source_warning,
        "mode": "cpu_reprocess" if args.reprocess_csv else "activity_gpu_detection",
        "fps": fps,
        "counts": {
            "source_detections": len(combined),
            "filtered_detections": len(clean),
            "unique_ksiva": len(summary),
            "rejected_tracks": len(rejected),
            "flagged_for_review": int(summary["review_flags"].ne("").sum()) if not summary.empty else 0,
            "preliminary_tracks": merge_audit["preliminary_track_count"],
            "second_pass_merged_components": merge_audit["merged_component_count"],
            "second_pass_absorbed_fragments": merge_audit["absorbed_fragment_count"],
        },
        "second_pass_merge_audit": merge_audit,
        "intervals": (
            parent_history.get("intervals", intervals)
            if isinstance(parent_history, dict)
            else [{"start_seconds": start, "end_seconds": end} for start, end in intervals]
        ),
        "filtering": settings,
        "outputs": {
            "raw_detections": raw_path.name if preserve_raw else None,
            "filtered_detections": clean_path.name,
            "unique_objects": summary_path.name,
            "rejected_objects": rejected_path.name,
        },
        "detector_args": args.detector_args if not args.reprocess_csv else None,
        "performance_metrics": performance_reference,
        "parent_history": parent_history,
    }
    with _phase(performance, "history_json_write"):
        _atomic_json_write(history_path, history)
    return clean_path, summary_path, rejected_path, history_path, len(summary)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.performance_log is None:
        performance_path = args.output_dir / "performance.metrics.json"
    elif args.performance_log.is_absolute():
        performance_path = args.performance_log
    else:
        performance_path = args.output_dir / args.performance_log
    run_wall_start = time.perf_counter()
    run_cpu_start = time.process_time()
    performance = {
        "schema_version": 1,
        "status": "running",
        "pid": os.getpid(),
        "mode": "cpu_reprocess" if args.reprocess_csv else "activity_gpu_detection",
        "source": str((args.reprocess_csv or args.video).expanduser().resolve()),
        "metrics_path": str(performance_path.resolve()),
        "configuration": {
            "requested_workers": args.workers,
            "devices": args.devices,
            "write_annotated": args.write_annotated,
            "keep_clips": args.keep_clips,
            "keep_temporary_on_error": args.keep_temporary_on_error,
            "discard_raw": args.discard_raw,
        },
        "process_memory_start": _process_memory(),
        "phases": {},
        "clips": [],
        "workers": [],
        "warnings": [],
    }
    jobs = []
    manifest_paths = []
    clips_dir = None

    try:
        validate(args)
        if args.reprocess_csv:
            with _phase(performance, "source_csv_read"):
                combined = pd.read_csv(args.reprocess_csv)
            fps = args.fps or infer_fps(combined)
            source_stage = "cleaned" if "ksiva_id" in combined.columns else "raw"
            outputs = _write_outputs(
                args,
                combined,
                fps,
                [],
                args.reprocess_csv.resolve(),
                source_stage,
                performance,
            )
            print(f"Done. CPU-only reprocessing; unique filtered ksiva: {outputs[-1]}")
            print(f"Filtered detections: {outputs[0]}")
            print(f"Unique-object summary: {outputs[1]}")
            print(f"Rejected-track audit: {outputs[2]}")
            print(f"Aggregate history: {outputs[3]}")
            performance["status"] = "completed"
            return outputs

        # Imported lazily so CSV reprocessing works without the video/GPU environment.
        with _phase(performance, "activity_module_import"):
            from aggregate_door_flow import activity_intervals, activity_scores, write_clip

        devices = _canonical_devices(args.devices)
        detector_args = _forwarded_detector_args(args.detector_args)
        clips_dir = args.output_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        with _phase(performance, "activity_scan"):
            fps, times, scores, duration = activity_scores(
                args.video,
                args.diff_seconds,
                (args.resize_width, args.resize_height),
                args.pixel_threshold,
            )
            if not math.isfinite(float(fps)) or float(fps) <= 0:
                raise ValueError(f"activity_scores returned invalid FPS: {fps!r}")
        with _phase(performance, "activity_interval_selection"):
            intervals = activity_intervals(
                times,
                scores,
                args.peak_height,
                args.noise_threshold,
                args.peak_distance_frames,
                args.merge_gap_seconds,
                args.context_seconds,
                duration,
            )

        jobs = []
        with _phase(performance, "clip_extraction_total"):
            for index, (start, end) in enumerate(intervals, start=1):
                stem = f"segment_{index:03d}_{start:.2f}_{end:.2f}"
                clip_path = clips_dir / f"{stem}.mp4"
                job = {
                    "index": index,
                    "start_seconds": start,
                    "end_seconds": end,
                    "start_frame": round(start * fps),
                    "clip": clip_path,
                    "video": args.output_dir / f"{stem}.annotated.mp4",
                    "log": args.output_dir / f"{stem}.raw.csv",
                }
                jobs.append(job)
                clip_start = time.perf_counter()
                write_clip(args.video, clip_path, start, end)
                clip_seconds = time.perf_counter() - clip_start
                try:
                    clip_bytes = clip_path.stat().st_size
                except OSError:
                    clip_bytes = None
                performance["clips"].append(
                    {
                        "index": index,
                        "start_seconds": start,
                        "end_seconds": end,
                        "video_seconds": round(end - start, 3),
                        "write_wall_seconds": round(clip_seconds, 6),
                        "bytes": clip_bytes,
                    }
                )

        manifest_paths = []
        if jobs:
            queues = _device_queues(jobs, devices, args.workers)
            performance["scheduler"] = {
                "requested_workers": args.workers,
                "actual_workers": len(queues),
                "device_queues": [
                    {
                        "device": queue["device"],
                        "job_indices": [job["index"] for job in queue["jobs"]],
                        "estimated_video_seconds": queue["estimated_video_seconds"],
                    }
                    for queue in queues
                ],
            }
            print(
                f"Activity intervals: {len(jobs)}; persistent workers: {len(queues)} "
                f"on device(s): {', '.join(queue['device'] for queue in queues)}"
            )
            futures = {}
            with _phase(performance, "detector_workers_total"):
                with ThreadPoolExecutor(max_workers=len(queues)) as executor:
                    for worker_index, queue in enumerate(queues, start=1):
                        safe_device = "".join(character if character.isalnum() else "_" for character in queue["device"])
                        prefix = f"worker_{worker_index:02d}_{safe_device}"
                        manifest_path = args.output_dir / f"{prefix}.jobs.json"
                        metrics_path = args.output_dir / f"{prefix}.metrics.json"
                        worker_log_path = args.output_dir / f"{prefix}.log"
                        manifest = {
                            "schema_version": 1,
                            "device": queue["device"],
                            "jobs": [
                                {
                                    "index": job["index"],
                                    "video": str(job["clip"]),
                                    "log": str(job["log"]),
                                    "output": str(job["video"]) if args.write_annotated else None,
                                }
                                for job in queue["jobs"]
                            ],
                        }
                        _atomic_json_write(manifest_path, manifest)
                        manifest_paths.append(manifest_path)
                        future = executor.submit(
                            run_detector_worker,
                            args.detector_script,
                            manifest_path,
                            metrics_path,
                            worker_log_path,
                            queue["device"],
                            detector_args,
                        )
                        futures[future] = queue
                    for future in as_completed(futures):
                        queue = futures[future]
                        worker_metrics = future.result()
                        performance["workers"].append(worker_metrics)
                        print(
                            f"Finished persistent worker on device {queue['device']}: "
                            f"{len(queue['jobs'])} segment(s)"
                        )
            for manifest_path in manifest_paths:
                manifest_path.unlink(missing_ok=True)
        else:
            performance["scheduler"] = {
                "requested_workers": args.workers,
                "actual_workers": 0,
                "device_queues": [],
            }
            print("No activity intervals found; detector model was not loaded.")

        detections = []
        with _phase(performance, "segment_csv_read_and_remap"):
            for job in jobs:
                segment_log = pd.read_csv(job["log"])
                if segment_log.empty:
                    continue
                segment_log["segment_id"] = job["index"]
                segment_log["segment_start_frame"] = job["start_frame"]
                segment_log["frame"] += job["start_frame"]
                segment_log["time_seconds"] = segment_log["frame"] / fps
                detections.append(segment_log)
            combined = pd.concat(detections, ignore_index=True) if detections else pd.DataFrame(columns=DETECTION_COLUMNS)
        performance["counts"] = {
            "activity_intervals": len(intervals),
            "source_detection_rows": len(combined),
        }
        outputs = _write_outputs(
            args,
            combined,
            fps,
            intervals,
            args.video.resolve(),
            "raw",
            performance,
        )

        with _phase(performance, "temporary_file_cleanup"):
            for job in jobs:
                job["log"].unlink(missing_ok=True)
                if not args.keep_clips:
                    job["clip"].unlink(missing_ok=True)
            if not args.keep_clips and clips_dir.exists():
                try:
                    clips_dir.rmdir()
                except OSError:
                    performance["warnings"].append(f"Temporary clips directory is not empty: {clips_dir}")
        print(f"Done. Unique filtered ksiva: {outputs[-1]}")
        print(f"Filtered detections: {outputs[0]}")
        print(f"Unique-object summary: {outputs[1]}")
        print(f"Rejected-track audit: {outputs[2]}")
        print(f"Aggregate history: {outputs[3]}")
        performance["status"] = "completed"
        return outputs
    except BaseException as error:
        performance["status"] = "failed"
        performance["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if performance["status"] == "failed" and not args.keep_temporary_on_error:
            with _phase(performance, "failure_temporary_cleanup"):
                for manifest_path in manifest_paths:
                    manifest_path.unlink(missing_ok=True)
                for job in jobs:
                    job["log"].unlink(missing_ok=True)
                    if not args.keep_clips:
                        job["clip"].unlink(missing_ok=True)
                if clips_dir is not None and not args.keep_clips and clips_dir.exists():
                    try:
                        clips_dir.rmdir()
                    except OSError:
                        performance["warnings"].append(f"Temporary clips directory is not empty: {clips_dir}")
        run_wall_seconds = time.perf_counter() - run_wall_start
        performance["run"] = {
            "wall_seconds": round(run_wall_seconds, 6),
            "cpu_seconds": round(time.process_time() - run_cpu_start, 6),
        }
        performance["phase_share_percent"] = {
            name: round(values["wall_seconds"] * 100 / run_wall_seconds, 2) if run_wall_seconds else None
            for name, values in performance["phases"].items()
        }
        if performance["workers"]:
            detector_wall = performance["phases"].get("detector_workers_total", {}).get("wall_seconds", 0.0)
            performance["detector_summary"] = _detector_summary(performance["workers"], detector_wall)
        performance["process_memory_end"] = _process_memory()
        _atomic_json_write(performance_path, performance)
        _print_performance_summary(performance)
        print(f"Performance metrics: {performance_path.resolve()}")


if __name__ == "__main__":
    run(parse_args())
