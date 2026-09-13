"""Detect ksiva objects in one video or a persistent-worker job manifest.

Single-video mode keeps the historical annotated-MP4 behavior. Aggregate mode
uses ``--jobs-manifest`` to load YOLO once per device and normally writes only
CSV logs, avoiding repeated model startup and unnecessary MP4 encoding.
"""

import argparse
import csv
import json
import math
import os
import resource
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = WORKSPACE / "runs" / "ksiva_yolo11_final" / "train" / "weights" / "best.pt"
DEFAULT_VIDEO = WORKSPACE / "mp4_out" / "1.mp4"
DEFAULT_OUTPUT = WORKSPACE / "ksiva_detections.mp4"
CLASS_NAME = "ksiva"
BOX_COLOR = (0, 220, 255)
CSV_FIELDS = ["frame", "time_seconds", "class", "confidence", "x1", "y1", "x2", "y2"]
MIB = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect ksiva objects and record detailed performance metrics.",
        allow_abbrev=False,
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Input video in single-video mode.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Annotated MP4 in single-video mode.")
    parser.add_argument("--log", type=Path, help="CSV detection log; defaults to output with a .csv extension.")
    parser.add_argument(
        "--jobs-manifest",
        type=Path,
        help="JSON manifest of video/log/output jobs. YOLO is loaded once and reused for all jobs.",
    )
    parser.add_argument("--metrics-log", type=Path, help="JSON file for timing, memory, VRAM, and throughput metrics.")
    parser.add_argument(
        "--no-annotated-video",
        action="store_true",
        help="Do not draw or encode annotated MP4; detection CSV is unchanged.",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Trained YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size.")
    parser.add_argument("--batch-size", type=int, default=1, help="Frames per predict call; increase after checking VRAM metrics.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on supported CUDA devices.")
    parser.add_argument("--device", default=None, help="Inference device, for example 0 or cpu (auto by default).")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames per job; 0 means the whole video.")
    return parser.parse_args()


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
        "high_water_rss_mib": None,
    }
    try:
        status = {}
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                status[key] = value.strip()
        for source, target in (("VmRSS", "rss_mib"), ("VmHWM", "high_water_rss_mib")):
            if source in status:
                snapshot[target] = round(float(status[source].split()[0]) / 1024, 3)
    except (OSError, ValueError, IndexError):
        pass
    return snapshot


def _cuda_device_index(device, torch):
    if not torch.cuda.is_available() or str(device).lower() == "cpu":
        return None
    if device is None or str(device).lower() in {"", "none", "cuda"}:
        return torch.cuda.current_device()
    value = str(device).split(",", 1)[0]
    if value.startswith("cuda:"):
        value = value.split(":", 1)[1]
    try:
        return int(value)
    except ValueError:
        return torch.cuda.current_device()


def _gpu_memory(device, reset_peak=False):
    try:
        import torch

        index = _cuda_device_index(device, torch)
        if index is None:
            return None
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        return {
            "device_index": index,
            "device_name": torch.cuda.get_device_name(index),
            "allocated_mib": round(torch.cuda.memory_allocated(index) / MIB, 3),
            "reserved_mib": round(torch.cuda.memory_reserved(index) / MIB, 3),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(index) / MIB, 3),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(index) / MIB, 3),
            "global_free_mib": round(free_bytes / MIB, 3),
            "global_total_mib": round(total_bytes / MIB, 3),
        }
    except (ImportError, RuntimeError, ValueError):
        return None


def _timed_snapshot(callback):
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = callback()
    return result, {
        "wall_seconds": round(time.perf_counter() - wall_start, 6),
        "cpu_seconds": round(time.process_time() - cpu_start, 6),
    }


def extract_detections(result):
    names = result.names
    boxes = result.boxes
    detections = []
    if boxes is None:
        return detections

    for box, confidence, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
        class_id = int(class_id.item())
        class_name = names[class_id] if isinstance(names, dict) else names[class_id]
        if class_name != CLASS_NAME:
            continue
        x1, y1, x2, y2 = (int(value) for value in box.tolist())
        detections.append(
            {
                "confidence": float(confidence.item()),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return detections


def draw_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = (detection[name] for name in ("x1", "y1", "x2", "y2"))
        score = detection["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{CLASS_NAME} {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (x1 + text_width + 8, label_top + text_height + baseline + 6),
            BOX_COLOR,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def _speed_summary(values):
    summary = {}
    for name, samples in values.items():
        finite = sorted(float(value) for value in samples if value is not None and math.isfinite(float(value)))
        if not finite:
            summary[name] = None
            continue
        p95_index = min(len(finite) - 1, math.ceil(len(finite) * 0.95) - 1)
        summary[name] = {
            "samples": len(finite),
            "total_ms": round(sum(finite), 3),
            "mean_ms_per_frame": round(sum(finite) / len(finite), 3),
            "p95_ms_per_frame": round(finite[p95_index], 3),
            "max_ms_per_frame": round(max(finite), 3),
        }
    return summary


def _job_from_args(args):
    output_path = args.output.expanduser().resolve()
    return {
        "index": 1,
        "video": str(args.video.expanduser().resolve()),
        "log": str((args.log or output_path.with_suffix(".csv")).expanduser().resolve()),
        "output": None if args.no_annotated_video else str(output_path),
    }


def _load_manifest(path):
    manifest_path = path.expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid jobs manifest {manifest_path}: {error}") from error
    jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit("Jobs manifest must contain a non-empty 'jobs' list.")
    return jobs, manifest_path


def _validate_args(args):
    weights_path = args.weights.expanduser().resolve()
    if not weights_path.is_file():
        raise SystemExit(f"Weights not found: {weights_path}")
    if not 0 <= args.conf <= 1:
        raise SystemExit("--conf must be in the range [0, 1].")
    if not 0 < args.iou <= 1:
        raise SystemExit("--iou must be in the range (0, 1].")
    if args.imgsz < 1 or args.batch_size < 1 or args.max_frames < 0:
        raise SystemExit("--imgsz and --batch-size must be positive; --max-frames cannot be negative.")
    if args.half and str(args.device).lower() == "cpu":
        raise SystemExit("--half requires a CUDA device, not CPU.")
    return weights_path


def _process_job(job, args, model):
    video_path = Path(job["video"]).expanduser().resolve()
    log_path = Path(job["log"]).expanduser().resolve()
    output_value = job.get("output")
    output_path = Path(output_value).expanduser().resolve() if output_value and not args.no_annotated_video else None
    if not video_path.is_file():
        raise RuntimeError(f"Video not found: {video_path}")

    job_wall_start = time.perf_counter()
    job_cpu_start = time.process_time()
    phases = {name: 0.0 for name in ("decode", "predict_wall", "extract", "draw", "encode", "csv_write")}
    speed_values = {"preprocess": [], "inference": [], "postprocess": []}
    capture = None
    writer = None
    frame_index = 0
    predict_calls = 0
    total_detections = 0
    first_predict_seconds = None
    memory_start = _process_memory()
    gpu_start = _gpu_memory(args.device, reset_peak=True)

    try:
        open_start = time.perf_counter()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps_value = float(capture.get(cv2.CAP_PROP_FPS))
        width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps_value) or fps_value < 0:
            raise RuntimeError(f"Invalid FPS metadata {fps_value!r}: {video_path}")
        if not math.isfinite(width_value) or not math.isfinite(height_value) or width_value <= 0 or height_value <= 0:
            raise RuntimeError(f"Invalid video dimensions {width_value!r}x{height_value!r}: {video_path}")
        fps = fps_value or 25.0
        width = int(width_value)
        height = int(height_value)
        capture_open_seconds = time.perf_counter() - open_start

        log_path.parent.mkdir(parents=True, exist_ok=True)
        writer_open_seconds = 0.0
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer_start = time.perf_counter()
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            writer_open_seconds = time.perf_counter() - writer_start
            if not writer.isOpened():
                raise RuntimeError(f"Could not open output video for writing: {output_path}")

        print(f"Job {job.get('index', '?')}: {video_path.name} ({width}x{height}, {fps:.2f} FPS)")
        print(f"  CSV: {log_path}")
        print(f"  Annotated MP4: {output_path if output_path else 'disabled'}")
        progress_interval = max(1, int(round(fps * 5)))
        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            log_writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
            log_writer.writeheader()
            end_of_video = False
            next_progress_frame = progress_interval
            while not end_of_video and (not args.max_frames or frame_index < args.max_frames):
                frames = []
                frame_numbers = []
                while len(frames) < args.batch_size and (not args.max_frames or frame_index + len(frames) < args.max_frames):
                    phase_start = time.perf_counter()
                    ok, frame = capture.read()
                    phases["decode"] += time.perf_counter() - phase_start
                    if not ok:
                        end_of_video = True
                        break
                    frames.append(frame)
                    frame_numbers.append(frame_index + len(frames) - 1)
                if not frames:
                    break

                phase_start = time.perf_counter()
                predictions = model.predict(
                    source=frames[0] if len(frames) == 1 else frames,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    batch=args.batch_size,
                    half=args.half,
                    device=args.device,
                    verbose=False,
                )
                predict_seconds = time.perf_counter() - phase_start
                predict_calls += 1
                phases["predict_wall"] += predict_seconds
                if first_predict_seconds is None:
                    first_predict_seconds = predict_seconds
                if len(predictions) != len(frames):
                    raise RuntimeError(
                        f"YOLO returned {len(predictions)} result(s) for {len(frames)} input frame(s)"
                    )

                for current_frame, current_number, prediction in zip(frames, frame_numbers, predictions):
                    speed = getattr(prediction, "speed", None) or {}
                    for name in speed_values:
                        value = speed.get(name)
                        if value is not None:
                            speed_values[name].append(value)

                    phase_start = time.perf_counter()
                    detections = extract_detections(prediction)
                    phases["extract"] += time.perf_counter() - phase_start

                    if writer is not None:
                        phase_start = time.perf_counter()
                        draw_detections(current_frame, detections)
                        phases["draw"] += time.perf_counter() - phase_start
                        phase_start = time.perf_counter()
                        writer.write(current_frame)
                        phases["encode"] += time.perf_counter() - phase_start

                    phase_start = time.perf_counter()
                    for detection in detections:
                        log_writer.writerow(
                            {
                                "frame": current_number,
                                "time_seconds": f"{current_number / fps:.3f}",
                                "class": CLASS_NAME,
                                **detection,
                            }
                        )
                    phases["csv_write"] += time.perf_counter() - phase_start
                    total_detections += len(detections)
                    frame_index = current_number + 1
                    if frame_index >= next_progress_frame:
                        elapsed = time.perf_counter() - job_wall_start
                        print(
                            f"  {frame_index / fps:.1f}s video | {frame_index / elapsed:.2f} FPS end-to-end "
                            f"| current detections: {len(detections)}"
                        )
                        while next_progress_frame <= frame_index:
                            next_progress_frame += progress_interval
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()

    wall_seconds = time.perf_counter() - job_wall_start
    cpu_seconds = time.process_time() - job_cpu_start
    video_seconds = frame_index / fps if frame_index else 0.0
    metrics = {
        "index": job.get("index"),
        "video": str(video_path),
        "log": str(log_path),
        "output": str(output_path) if output_path else None,
        "frames": frame_index,
        "predict_calls": predict_calls,
        "configured_batch_size": args.batch_size,
        "video_seconds": round(video_seconds, 6),
        "detections": total_detections,
        "wall_seconds": round(wall_seconds, 6),
        "cpu_seconds": round(cpu_seconds, 6),
        "end_to_end_fps": round(frame_index / wall_seconds, 3) if wall_seconds else None,
        "realtime_factor": round(video_seconds / wall_seconds, 3) if wall_seconds else None,
        "capture_open_seconds": round(capture_open_seconds, 6),
        "writer_open_seconds": round(writer_open_seconds, 6),
        "first_predict_seconds": round(first_predict_seconds, 6) if first_predict_seconds is not None else None,
        "phases": {name: round(value, 6) for name, value in phases.items()},
        "yolo_speed": _speed_summary(speed_values),
        "process_memory_start": memory_start,
        "process_memory_end": _process_memory(),
        "gpu_memory_start": gpu_start,
        "gpu_memory_end": _gpu_memory(args.device),
    }
    accounted_seconds = sum(phases.values()) + capture_open_seconds + writer_open_seconds
    metrics["phase_share_percent"] = {
        name: round(value * 100 / wall_seconds, 2) if wall_seconds else None
        for name, value in phases.items()
    }
    metrics["unattributed_wall_seconds"] = round(max(0.0, wall_seconds - accounted_seconds), 6)
    print(
        f"Finished job {job.get('index', '?')}: frames={frame_index}, detections={total_detections}, "
        f"wall={wall_seconds:.2f}s, FPS={metrics['end_to_end_fps']}"
    )
    return metrics


def run(args):
    manifest_path = args.jobs_manifest.expanduser().resolve() if args.jobs_manifest else None
    if args.metrics_log is not None:
        metrics_path = args.metrics_log
    elif manifest_path is not None:
        metrics_path = manifest_path.with_suffix(".metrics.json")
    else:
        metrics_path = Path(_job_from_args(args)["log"]).with_suffix(".metrics.json")

    run_wall_start = time.perf_counter()
    run_cpu_start = time.process_time()
    jobs = []
    weights_path = args.weights.expanduser().resolve()
    metrics = {
        "schema_version": 1,
        "status": "running",
        "pid": os.getpid(),
        "device": args.device,
        "weights": str(weights_path),
        "manifest": str(manifest_path) if manifest_path else None,
        "configuration": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "batch_size": args.batch_size,
            "half": args.half,
            "max_frames": args.max_frames,
            "no_annotated_video": args.no_annotated_video,
        },
        "process_memory_start": _process_memory(),
        "gpu_memory_before_model": _gpu_memory(args.device),
        "jobs": [],
    }
    try:
        weights_path = _validate_args(args)
        if manifest_path is not None:
            jobs, loaded_manifest_path = _load_manifest(manifest_path)
            manifest_path = loaded_manifest_path
        else:
            jobs = [_job_from_args(args)]
        metrics["weights"] = str(weights_path)
        metrics["manifest"] = str(manifest_path) if manifest_path else None
        print(f"Weights: {weights_path}")
        print(f"Device: {args.device if args.device is not None else 'auto'}")
        model, load_timing = _timed_snapshot(lambda: YOLO(str(weights_path)))
        metrics["model_load"] = load_timing
        metrics["gpu_memory_after_model"] = _gpu_memory(args.device)
        for job in jobs:
            job_wall_start = time.perf_counter()
            try:
                job_metrics = _process_job(job, args, model)
                job_metrics["status"] = "completed"
                metrics["jobs"].append(job_metrics)
            except BaseException as job_error:
                metrics["jobs"].append(
                    {
                        "index": job.get("index"),
                        "video": job.get("video"),
                        "log": job.get("log"),
                        "output": job.get("output"),
                        "status": "failed",
                        "error": f"{type(job_error).__name__}: {job_error}",
                        "wall_seconds": round(time.perf_counter() - job_wall_start, 6),
                        "process_memory_at_failure": _process_memory(),
                        "gpu_memory_at_failure": _gpu_memory(args.device),
                    }
                )
                raise
        metrics["status"] = "completed"
    except BaseException as error:
        metrics["status"] = "failed"
        metrics["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run_wall_seconds = time.perf_counter() - run_wall_start
        metrics["run"] = {
            "wall_seconds": round(run_wall_seconds, 6),
            "cpu_seconds": round(time.process_time() - run_cpu_start, 6),
        }
        metrics["breakdown"] = {
            "model_load_percent": round(metrics.get("model_load", {}).get("wall_seconds", 0) * 100 / run_wall_seconds, 2)
            if run_wall_seconds
            else None,
            "jobs_percent": round(sum(job["wall_seconds"] for job in metrics["jobs"]) * 100 / run_wall_seconds, 2)
            if run_wall_seconds
            else None,
        }
        metrics["telemetry_note"] = (
            "allocated/reserved/peak VRAM are process-local; global_free/global_total include all GPU processes. "
            "GPU utilization and power require external NVML/nvidia-smi sampling and are not reported here."
        )
        metrics["process_memory_end"] = _process_memory()
        metrics["gpu_memory_end"] = _gpu_memory(args.device)
        _atomic_json_write(metrics_path, metrics)
        print(f"Performance metrics: {Path(metrics_path).expanduser().resolve()}")
    return metrics


def process_video(args):
    """Compatibility entry point for existing callers."""
    return run(args)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
