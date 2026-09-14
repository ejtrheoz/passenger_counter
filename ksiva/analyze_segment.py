"""Detect and count ksiva presentations in a single already-trimmed segment clip.

Companion to ``aggregate_ksiva.py``: that script re-detects activity intervals
inside the video it is given. This script assumes its input is *already* one
continuous activity interval (for example a clip produced by
``passenger_aggregator``'s activity-segmentation step) and skips straight to
YOLO detection plus the spatial tracking/filtering post-processing, so a single
segment can be analyzed in one small, bounded-time request.

Example:
    python analyze_segment.py --video mp4_out/segments/segment_001.mp4 \
        --output-dir mp4_out/segments_out/001 --weights models/ksiva/best.pt
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

from aggregate_ksiva import DETECTION_COLUMNS, postprocess_detections

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_DETECTOR_SCRIPT = WORKSPACE / "detect_ksiva_video.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Count unique ksiva presentations in one segment clip.")
    parser.add_argument("--video", type=Path, required=True, help="Segment clip to analyze.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the raw CSV and aggregate.history.json.")
    parser.add_argument("--detector-script", type=Path, default=DEFAULT_DETECTOR_SCRIPT, help="Path to detect_ksiva_video.py.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained YOLO weights.")
    parser.add_argument("--device", default="0", help="cuda index or 'cpu'.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size.")
    parser.add_argument("--batch-size", type=int, default=4, help="Frames per predict call.")
    return parser.parse_args()


def _video_fps(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        return float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    finally:
        capture.release()


def run(args):
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    raw_csv = args.output_dir / "segment.raw.csv"
    command = [
        sys.executable, str(args.detector_script),
        "--video", str(args.video),
        "--log", str(raw_csv),
        "--no-annotated-video",
        "--weights", str(args.weights),
        "--device", args.device,
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--imgsz", str(args.imgsz),
        "--batch-size", str(args.batch_size),
    ]
    subprocess.run(command, check=True)

    if raw_csv.is_file() and raw_csv.stat().st_size:
        detections = pd.read_csv(raw_csv)
    else:
        detections = pd.DataFrame(columns=DETECTION_COLUMNS)

    fps = _video_fps(args.video)
    clean, summary, rejected = postprocess_detections(detections, fps=fps)

    history = {
        "schema_version": 1,
        "mode": "single_segment",
        "source": str(args.video.resolve()),
        "fps": fps,
        "counts": {
            "source_detections": int(len(detections)),
            "filtered_detections": int(len(clean)),
            "unique_ksiva": int(len(summary)),
            "rejected_tracks": int(len(rejected)),
            "flagged_for_review": int(summary["review_flags"].ne("").sum()) if not summary.empty else 0,
        },
        "processing_seconds": round(time.perf_counter() - started, 3),
    }
    (args.output_dir / "aggregate.history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Done. unique_ksiva={history['counts']['unique_ksiva']}")


if __name__ == "__main__":
    run(parse_args())
