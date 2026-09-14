"""Thin wrappers that run the aggregate scripts as subprocesses and parse their JSON output."""

import collections
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import settings


class PipelineError(RuntimeError):
    pass


def log(label, message):
    print(f"[{label}] {message}", flush=True)


_INTERVALS_RE = re.compile(r"^Activity intervals: (\d+)")
_SEGMENT_RE = re.compile(r"^Finished segment (\d+): entered=(\d+); exited=(\d+)")
_NO_ACTIVITY_RE = re.compile(r"^No activity intervals found")


class _Progress:
    """Turns the aggregate scripts' stdout into per-segment progress lines."""

    def __init__(self, label, stage):
        self.label = label
        self.stage = stage
        self.total = None
        self.done = 0
        self.entered = 0
        self.exited = 0

    def feed(self, line):
        match = _INTERVALS_RE.match(line)
        if match:
            self.total = int(match.group(1))
            log(self.label, f"{self.stage}: {self.total} activity interval(s) found, starting detection")
            return
        if _NO_ACTIVITY_RE.match(line):
            log(self.label, f"{self.stage}: no activity intervals found, nothing to process")
            return
        match = _SEGMENT_RE.match(line)
        if match:
            index, entered, exited = (int(value) for value in match.groups())
            self.done += 1
            self.entered += entered
            self.exited += exited
            total = self.total or "?"
            log(
                self.label,
                f"{self.stage}: interval {self.done}/{total} done (segment {index}): "
                f"entered={entered} exited={exited}; running total entered={self.entered} exited={self.exited}",
            )


def _run(command, cwd, label, stage, extra_env=None):
    env = {**os.environ, **(extra_env or {}), "PYTHONUNBUFFERED": "1"}
    tail = collections.deque(maxlen=40)
    progress = _Progress(label, stage)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def pump():
        for line in process.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            progress.feed(line)
            if settings.VERBOSE_PIPELINE_LOGS:
                log(label, f"{stage} | {line}")

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=settings.JOB_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait()
        log(label, f"{stage}: timed out after {settings.JOB_TIMEOUT_SECONDS}s")
        raise PipelineError(f"Pipeline timed out after {settings.JOB_TIMEOUT_SECONDS}s") from error
    finally:
        reader.join(timeout=5)
    if returncode:
        log(label, f"{stage}: FAILED with exit code {returncode}")
        raise PipelineError(f"Pipeline exited with code {returncode}:\n" + "\n".join(tail))


def _load_history(path):
    if not path.is_file():
        raise PipelineError(f"Pipeline finished but produced no {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _single_device() -> str:
    """settings.DEVICE may list several GPUs ("0,1") for the multi-worker aggregate scripts;
    single-segment analysis only ever uses one worker, so pick the first device."""
    return (settings.DEVICE.split(",")[0].strip() or "cpu")


def analyze_people_segment(video_path: Path, polygon_path: Path, output_dir: Path, label: str) -> dict:
    """Run door_flow_counter.py directly on one already-cut segment clip (no interval re-detection)."""
    started = time.perf_counter()
    device = _single_device()
    log(label, f"people-segment: analyzing (polygon={polygon_path.name}, device={device})")
    history_path = output_dir / "segment.history.json"
    command = [
        sys.executable, str(settings.DOOR_FLOW_COUNTER_SCRIPT),
        "--video", str(video_path),
        "--door-polygon", str(polygon_path),
        "--output", str(output_dir / "segment.annotated.mp4"),
        "--history-output", str(history_path),
        "--repo-dir", str(settings.BPJDET_REPO),
        "--weights", str(settings.BPJDET_WEIGHTS),
        "--device", device,
        "--batch-size", str(settings.BATCH_SIZE),
    ]
    _run(command, cwd=settings.DOOR_FLOW_DIR, label=label, stage="people-segment")
    history = _load_history(history_path)
    counts = history["counts"]
    elapsed = round(time.perf_counter() - started, 3)
    log(label, f"people-segment: done in {elapsed}s; entered={counts['enter']} exited={counts['exit']}")
    return {
        "entered": counts["enter"],
        "exited": counts["exit"],
        "door_polygon": polygon_path.name,
        "processing_seconds": elapsed,
    }


def analyze_ksiva_segment(video_path: Path, output_dir: Path, label: str) -> dict:
    """Run detect_ksiva_video.py + post-processing on one already-cut segment clip (no interval re-detection)."""
    started = time.perf_counter()
    device = _single_device()
    log(label, f"ksiva-segment: analyzing (device={device})")
    command = [
        sys.executable, str(settings.KSIVA_SEGMENT_SCRIPT),
        "--video", str(video_path),
        "--output-dir", str(output_dir),
        "--weights", str(settings.KSIVA_WEIGHTS),
        "--device", device,
        "--batch-size", str(settings.BATCH_SIZE),
    ]
    _run(command, cwd=settings.KSIVA_DIR, label=label, stage="ksiva-segment")
    history = _load_history(output_dir / "aggregate.history.json")
    counts = history["counts"]
    elapsed = round(time.perf_counter() - started, 3)
    log(label, f"ksiva-segment: done in {elapsed}s; unique_ksiva={counts['unique_ksiva']}")
    return {
        "unique_ksiva": counts["unique_ksiva"],
        "source_detections": counts.get("source_detections"),
        "filtered_detections": counts.get("filtered_detections"),
        "rejected_tracks": counts.get("rejected_tracks"),
        "flagged_for_review": counts.get("flagged_for_review"),
        "processing_seconds": elapsed,
    }

