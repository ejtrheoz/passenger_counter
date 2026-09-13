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


def count_people(video_path: Path, polygon_path: Path, output_dir: Path, label: str) -> dict:
    started = time.perf_counter()
    log(label, f"door-flow: scanning for activity (polygon={polygon_path.name}, device={settings.DEVICE})")
    command = [
        sys.executable, str(settings.DOOR_FLOW_SCRIPT),
        "--video", str(video_path),
        "--door-polygon", str(polygon_path),
        "--output-dir", str(output_dir),
        "--devices", settings.DEVICE,
        "--workers", "1",
        "--batch-size", str(settings.BATCH_SIZE),
        "--",
        "--repo-dir", str(settings.BPJDET_REPO),
        "--weights", str(settings.BPJDET_WEIGHTS),
    ]
    _run(command, cwd=settings.DOOR_FLOW_DIR, label=label, stage="door-flow")
    history = _load_history(output_dir / "aggregate.history.json")
    elapsed = round(time.perf_counter() - started, 3)
    log(label, f"door-flow: done in {elapsed}s; entered={history['counts']['enter']} exited={history['counts']['exit']}")
    return {
        "entered": history["counts"]["enter"],
        "exited": history["counts"]["exit"],
        "door_polygon": polygon_path.name,
        "fps": history.get("fps"),
        "intervals": history.get("intervals", []),
        "segments": [
            {
                "index": segment["index"],
                "start_seconds": segment["start_seconds"],
                "end_seconds": segment["end_seconds"],
                "entered": segment["counts"]["enter"],
                "exited": segment["counts"]["exit"],
            }
            for segment in history.get("segments", [])
        ],
        "processing_seconds": elapsed,
    }


def count_ksiva(video_path: Path, output_dir: Path, label: str) -> dict:
    started = time.perf_counter()
    log(label, f"ksiva: scanning for activity (device={settings.DEVICE})")
    command = [
        sys.executable, str(settings.KSIVA_SCRIPT),
        "--video", str(video_path),
        "--output-dir", str(output_dir),
        "--devices", settings.DEVICE,
        "--workers", "1",
        "--",
        "--weights", str(settings.KSIVA_WEIGHTS),
        "--batch-size", str(settings.BATCH_SIZE),
    ]
    # aggregate_ksiva.py imports activity helpers from aggregate_door_flow.py
    _run(command, cwd=settings.KSIVA_DIR, label=label, stage="ksiva", extra_env={"PYTHONPATH": str(settings.DOOR_FLOW_DIR)})
    history = _load_history(output_dir / "aggregate.history.json")
    counts = history["counts"]
    elapsed = round(time.perf_counter() - started, 3)
    log(label, f"ksiva: done in {elapsed}s; unique_ksiva={counts['unique_ksiva']}")
    return {
        "unique_ksiva": counts["unique_ksiva"],
        "source_detections": counts.get("source_detections"),
        "filtered_detections": counts.get("filtered_detections"),
        "rejected_tracks": counts.get("rejected_tracks"),
        "flagged_for_review": counts.get("flagged_for_review"),
        "fps": history.get("fps"),
        "intervals": history.get("intervals", []),
        "processing_seconds": elapsed,
    }
