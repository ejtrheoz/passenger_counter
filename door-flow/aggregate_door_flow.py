"""Run door-flow counting only on activity intervals from a video.

The script detects large scene changes in a small grayscale stream, expands and
merges those intervals, writes independent clips, and runs ``door_flow_counter``
on the clips concurrently. A single countable crossing cannot be shared by two
clips because the intervals are merged after context expansion.

Example:
    python aggregate_door_flow.py --video dataset/first_6.mp4 --devices 0 --workers 1
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_VIDEO = WORKSPACE / "dataset" / "first_6.mp4"
DEFAULT_POLYGON_DIR = WORKSPACE / "dataset" / "door_polygons"
DEFAULT_OUTPUT_DIR = WORKSPACE / "mp4_out" / "door_flow_activity"
COUNTER_SCRIPT = WORKSPACE / "door_flow_counter.py"
VRAM_SAFETY_RESERVE_MB = 512


def parse_args():
    parser = argparse.ArgumentParser(description="Detect video activity and count door crossings in those clips.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Source video.")
    parser.add_argument("--door-polygon", type=Path, help="Door polygon; defaults from the source video name.")
    parser.add_argument("--polygon-dir", type=Path, default=DEFAULT_POLYGON_DIR, help="Directory containing door polygons.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for clips and the aggregate JSON.")
    parser.add_argument("--counter-script", type=Path, default=COUNTER_SCRIPT, help="Path to door_flow_counter.py.")
    parser.add_argument("--devices", default="0", help="Comma-separated CUDA device IDs, such as 0 or 0,1.")
    parser.add_argument("--workers", type=int, default=1, help="Maximum simultaneous counter processes; raise only if GPU VRAM permits.")
    parser.add_argument("--batch-size", type=int, default=4, help="Frames processed by each BPJDet forward pass in a worker.")
    parser.add_argument("--diff-seconds", type=float, default=5.0, help="Compare each frame to this many seconds earlier.")
    parser.add_argument("--resize-width", type=int, default=320, help="Width used for activity analysis.")
    parser.add_argument("--resize-height", type=int, default=240, help="Height used for activity analysis.")
    parser.add_argument("--pixel-threshold", type=int, default=25, help="Per-pixel difference threshold.")
    parser.add_argument("--peak-height", type=float, default=15000, help="Minimum changed-pixel count for an activity peak.")
    parser.add_argument("--noise-threshold", type=float, default=5000, help="Changed-pixel count where an activity interval ends.")
    parser.add_argument("--peak-distance-frames", type=int, default=25, help="Minimum frames between activity peaks.")
    parser.add_argument("--merge-gap-seconds", type=float, default=2.0, help="Merge activity intervals separated by at most this gap.")
    parser.add_argument("--context-seconds", type=float, default=3.0, help="Extra video before and after activity for complete tracks.")
    parser.add_argument("counter_args", nargs=argparse.REMAINDER, help="Arguments forwarded to door_flow_counter.py; add them after --.")
    return parser.parse_args()


def validate(args, polygon_path):
    for path in (args.video, polygon_path, args.counter_script):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")
    if args.workers < 1 or args.batch_size < 1 or args.diff_seconds <= 0 or args.resize_width < 1 or args.resize_height < 1:
        raise ValueError("--workers, --batch-size, --diff-seconds, and resize dimensions must be positive")
    if args.peak_distance_frames < 1 or args.merge_gap_seconds < 0 or args.context_seconds < 0:
        raise ValueError("Activity interval thresholds must be valid")
    if not [device.strip() for device in args.devices.split(",") if device.strip()]:
        raise ValueError("--devices must contain at least one device ID")


def activity_scores(video_path, diff_seconds, resize_size, pixel_threshold):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        diff_frames = max(1, round(fps * diff_seconds))
        buffer = []
        scores, times = [], []
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, resize_size), cv2.COLOR_BGR2GRAY)
            buffer.append(gray)
            if len(buffer) > diff_frames:
                past = buffer.pop(0)
                difference = cv2.absdiff(gray, past)
                _, changed = cv2.threshold(difference, pixel_threshold, 255, cv2.THRESH_BINARY)
                scores.append(float(np.count_nonzero(changed)))
                times.append(frame_index / fps)
            frame_index += 1
        return fps, times, np.asarray(scores, dtype=np.float32), frame_index / fps
    finally:
        capture.release()


def activity_intervals(times, scores, peak_height, noise_threshold, peak_distance_frames, merge_gap_seconds, context_seconds, duration):
    if not len(scores):
        return []
    peaks, _ = find_peaks(scores, height=peak_height, distance=peak_distance_frames)
    intervals = []
    for peak in peaks:
        left = right = int(peak)
        while left > 0 and scores[left] > noise_threshold:
            left -= 1
        while right < len(scores) - 1 and scores[right] > noise_threshold:
            right += 1
        intervals.append([float(times[left]), float(times[right])])
    if not intervals:
        return []

    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + merge_gap_seconds:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    expanded = [[max(0.0, start - context_seconds), min(duration, end + context_seconds)] for start, end in merged]
    final = [expanded[0]]
    for start, end in expanded[1:]:
        if start <= final[-1][1]:
            final[-1][1] = max(final[-1][1], end)
        else:
            final.append([start, end])
    return final


def write_clip(video_path, destination, start_seconds, end_seconds):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create clip: {destination}")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(start_seconds * fps))
            final_frame = round(end_seconds * fps)
            while int(capture.get(cv2.CAP_PROP_POS_FRAMES)) < final_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
        finally:
            writer.release()
    finally:
        capture.release()


def gpu_memory(device):
    if str(device).lower() == "cpu":
        return None
    device_id = str(device).removeprefix("cuda:")
    command = [
        "nvidia-smi",
        f"--id={device_id}",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        index, name, total, used, free = (value.strip() for value in result.stdout.strip().split(","))
        return {"index": int(index), "name": name, "total_mb": int(total), "used_mb": int(used), "free_mb": int(free)}
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def gpu_process_memory(process_id):
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        for line in result.stdout.splitlines():
            pid, used = (value.strip() for value in line.split(","))
            if int(pid) == process_id:
                return int(used)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None
    return None


def run_counter_jobs(counter_script, jobs, polygon_path, device, batch_size, counter_args, baseline_used_mb):
    forwarded_args = list(counter_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args.pop(0)
    manifest = [
        {"video": str(clip), "output": str(output), "history_output": str(history)}
        for _, _, _, clip, output, history in jobs
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as manifest_file:
        json.dump(manifest, manifest_file)
        manifest_path = Path(manifest_file.name)
    command = [
        sys.executable,
        str(counter_script),
        "--jobs-file", str(manifest_path),
        "--door-polygon", str(polygon_path),
        "--device", device,
        "--batch-size", str(batch_size),
        *forwarded_args,
    ]
    peak_used_mb = baseline_used_mb
    peak_process_mb = 0
    started_ns = time.time_ns()
    reported_histories = set()

    def report_completed_jobs():
        for index, _, _, _, _, history in jobs:
            if history in reported_histories or not history.exists() or history.stat().st_mtime_ns < started_ns:
                continue
            try:
                result = json.loads(history.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            reported_histories.add(history)
            print(
                f"Finished segment {index}: entered={result['counts']['enter']}; "
                f"exited={result['counts']['exit']}",
                flush=True,
            )

    try:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout, tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, text=True, stdout=stdout, stderr=stderr)
            while process.poll() is None:
                memory = gpu_memory(device)
                if memory:
                    peak_used_mb = max(peak_used_mb, memory["used_mb"])
                process_memory = gpu_process_memory(process.pid)
                if process_memory is not None:
                    peak_process_mb = max(peak_process_mb, process_memory)
                report_completed_jobs()
                time.sleep(0.5)
            if process.returncode:
                stderr.seek(0)
                error = stderr.read()
                stdout.seek(0)
                raise RuntimeError(f"Counter worker failed:\n{error or stdout.read()}")
        memory = gpu_memory(device)
        if memory:
            peak_used_mb = max(peak_used_mb, memory["used_mb"])
        report_completed_jobs()
    finally:
        manifest_path.unlink(missing_ok=True)
    worker_delta_mb = peak_process_mb or max(0, peak_used_mb - baseline_used_mb)
    vram = {"device": str(device), "peak_used_mb": peak_used_mb, "peak_process_mb": peak_process_mb, "worker_delta_mb": worker_delta_mb}
    results = [json.loads(history.read_text(encoding="utf-8")) for _, _, _, _, _, history in jobs]
    return results, vram


def run(args):
    polygon_path = args.door_polygon or args.polygon_dir / f"{args.video.stem}.json"
    validate(args, polygon_path)
    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    gpu_baselines = {device: gpu_memory(device) for device in devices}
    for device, memory in gpu_baselines.items():
        if memory:
            print(f"GPU {device}: {memory['name']}; VRAM {memory['used_mb']}/{memory['total_mb']} MiB used, {memory['free_mb']} MiB free")
        else:
            print(f"GPU {device}: VRAM monitoring unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = args.output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    fps, times, scores, duration = activity_scores(
        args.video,
        args.diff_seconds,
        (args.resize_width, args.resize_height),
        args.pixel_threshold,
    )
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
    if not intervals:
        summary = {"video": str(args.video.resolve()), "fps": fps, "counts": {"enter": 0, "exit": 0}, "intervals": [], "segments": []}
        (args.output_dir / "aggregate.history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("No activity intervals found.")
        return

    jobs = []
    for index, (start, end) in enumerate(intervals, start=1):
        stem = f"segment_{index:03d}_{start:.2f}_{end:.2f}"
        clip_path = clips_dir / f"{stem}.mp4"
        write_clip(args.video, clip_path, start, end)
        jobs.append((index, start, end, clip_path, args.output_dir / f"{stem}.annotated.mp4", args.output_dir / f"{stem}.history.json"))
    print(f"Activity intervals: {len(jobs)}; processing with {min(args.workers, len(jobs))} worker(s) on device(s): {', '.join(devices)}")

    worker_count = min(args.workers, len(jobs))
    worker_jobs = [[] for _ in range(worker_count)]
    worker_durations = [0.0] * worker_count
    for job in sorted(jobs, key=lambda item: item[2] - item[1], reverse=True):
        worker_index = min(range(worker_count), key=worker_durations.__getitem__)
        worker_jobs[worker_index].append(job)
        worker_durations[worker_index] += job[2] - job[1]
    for assigned_jobs in worker_jobs:
        assigned_jobs.sort(key=lambda item: item[0])

    segments = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(run_counter_jobs, args.counter_script, assigned_jobs, polygon_path, devices[worker_index % len(devices)], args.batch_size, args.counter_args, (gpu_baselines[devices[worker_index % len(devices)]] or {}).get("used_mb", 0)): assigned_jobs
            for worker_index, assigned_jobs in enumerate(worker_jobs)
        }
        for future in as_completed(futures):
            assigned_jobs = futures[future]
            results, vram = future.result()
            for (index, start, end, clip, output, history), result in zip(assigned_jobs, results):
                segments.append({"index": index, "start_seconds": start, "end_seconds": end, "clip": str(clip.resolve()), "video": str(output.resolve()), "history": str(history.resolve()), "counts": result["counts"], "events": result["events"], "vram": vram})
            print(f"Worker finished: segments={len(assigned_jobs)}; worker VRAM={vram['peak_process_mb']} MiB; GPU peak={vram['peak_used_mb']} MiB", flush=True)

    segments.sort(key=lambda segment: segment["index"])
    counts = {"enter": sum(segment["counts"]["enter"] for segment in segments), "exit": sum(segment["counts"]["exit"] for segment in segments)}
    vram_summary = {}
    for device, baseline in gpu_baselines.items():
        if not baseline:
            continue
        worker_delta_mb = max((segment["vram"]["worker_delta_mb"] for segment in segments if segment["vram"]["device"] == device), default=0)
        estimated_two_workers_mb = baseline["used_mb"] + 2 * worker_delta_mb + VRAM_SAFETY_RESERVE_MB
        vram_summary[device] = {**baseline, "max_worker_delta_mb": worker_delta_mb, "safety_reserve_mb": VRAM_SAFETY_RESERVE_MB, "estimated_two_workers_mb": estimated_two_workers_mb, "two_workers_fit": estimated_two_workers_mb <= baseline["total_mb"]}
        verdict = "YES" if estimated_two_workers_mb <= baseline["total_mb"] else "NO"
        print(f"GPU {device}: estimated VRAM for 2 workers={estimated_two_workers_mb}/{baseline['total_mb']} MiB (includes {VRAM_SAFETY_RESERVE_MB} MiB reserve); fit={verdict}")
    summary = {"video": str(args.video.resolve()), "door_polygon": str(polygon_path.resolve()), "fps": fps, "counts": counts, "vram": vram_summary, "intervals": [{"start_seconds": start, "end_seconds": end} for start, end in intervals], "segments": segments}
    summary_path = args.output_dir / "aggregate.history.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. Entered: {counts['enter']}; exited: {counts['exit']}")
    print(f"Aggregate history: {summary_path}")


if __name__ == "__main__":
    run(parse_args())