"""Count bus-door entries and exits from BPJDet detections and a door polygon.

The polygon is read from ``dataset/door_polygons/<video-stem>.json`` by default.
Its point order is top-left, top-right, bottom-right, bottom-left: ``top`` is the
outside side of the door and ``bottom`` is the salon side.

Example:
    python door_flow_counter.py --video dataset/first_6.mp4 --device 0
"""

import argparse
import copy
import csv
import gzip
import json
import os
import resource
import sys
import time
from collections import deque
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_VIDEO = WORKSPACE / "dataset" / "first_6.mp4"
DEFAULT_POLYGON_DIR = WORKSPACE / "dataset" / "door_polygons"
DEFAULT_OUTPUT = WORKSPACE / "mp4_out" / "door_flow.mp4"
DEFAULT_HISTORY = WORKSPACE / "mp4_out" / "door_flow.history.json"
DEFAULT_BPJDET_REPOSITORY = WORKSPACE / "third_party" / "BPJDet"
DEFAULT_BPJDET_WEIGHTS = WORKSPACE / "models" / "bpjdet" / "ch_face_s_1536_e150_best_mMR.pt"
DEFAULT_TRANSREID_REPOSITORY = WORKSPACE / "third_party" / "TransReID"
DEFAULT_TRANSREID_CONFIG = DEFAULT_TRANSREID_REPOSITORY / "configs" / "OCC_Duke" / "vit_transreid_stride.yml"
DEFAULT_REID_WEIGHTS = WORKSPACE / "models" / "transreid_occ_duke_vit_stride.pth"
SCHEMA_VERSION = "door-flow-observability/v3"
DETECTION_FIELDS = [
    "frame", "time_seconds", "detection_index", "score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "center_x", "center_y", "bottom_x", "bottom_y", "head_x", "head_y", "center_progress", "center_lateral", "center_region",
    "bottom_progress", "bottom_lateral", "bottom_region", "inside_door", "crop_status", "embedding_status",
    "assigned_track_id", "canonical_track_id", "assignment_method", "appearance_distance", "center_distance",
    "match_iou", "bottom_distance", "previous_progress_delta", "new_progress_delta", "missing_before", "reason",
]
TRACK_EVENT_FIELDS = [
    "event", "frame", "time_seconds", "track_id", "other_track_id", "canonical_track_id", "detection_index",
    "assignment_method", "appearance_distance", "center_distance", "match_iou", "bottom_distance",
    "progress_distance", "appearance_pair_distance", "previous_progress_delta", "new_progress_delta",
    "missing_before", "missing_frames", "streak", "decision", "count_delta", "reason", "metrics_json",
]
TRACK_FIELDS = [
    "track_id", "canonical_track_id", "absorbed_by", "alias_chain", "created_frame", "first_frame", "last_frame",
    "expired_frame", "lifecycle", "termination", "observation_count", "start_region", "end_region",
    "start_progress", "end_progress", "net_progress", "bottom_start_progress", "bottom_end_progress",
    "bottom_net_progress", "was_inside_door", "was_in_lane", "finished_near_salon_edge",
    "finished_at_outer_edge", "disappeared_at_door", "missing_frames_at_end", "online_decision",
    "online_decision_frame", "final_decision", "decision_changed", "accepted", "reason_codes",
    "enter_predicates_json", "exit_predicates_json", "predicate_metrics_json",
]


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path, fieldnames, rows, compressed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    opener = gzip.open if compressed else open
    try:
        with opener(temporary, "wt", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def derived_artifact_paths(history_path):
    history_path = Path(history_path)
    name = history_path.name
    base_name = name[:-len(".history.json")] if name.endswith(".history.json") else history_path.stem
    base = history_path.with_name(base_name)
    return {
        "detections": base.with_name(base.name + ".detections.csv.gz"),
        "track_events": base.with_name(base.name + ".track_events.csv.gz"),
        "tracks": base.with_name(base.name + ".tracks.csv"),
        "metrics": base.with_name(base.name + ".metrics.json"),
    }


def resolve_artifact_paths(args):
    defaults = derived_artifact_paths(args.history_output)
    return {
        "detections": Path(args.detections_output) if args.detections_output else defaults["detections"],
        "track_events": Path(args.track_events_output) if args.track_events_output else defaults["track_events"],
        "tracks": Path(args.tracks_output) if args.tracks_output else defaults["tracks"],
        "metrics": Path(args.metrics_output) if args.metrics_output else defaults["metrics"],
    }


def normalized_config(args):
    omitted = {"jobs_file", "detections_output", "track_events_output", "tracks_output", "metrics_output"}
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key not in omitted
    }
    config.setdefault("salon_start_min_bottom_center_ratio", 2.0)
    config.setdefault("lane_fallback_progress_multiplier", 2.5)
    return config


def process_snapshot(device=None):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    snapshot = {"timestamp_unix": time.time(), "pid": os.getpid(), "max_rss_kb": usage.ru_maxrss}
    if device is not None:
        snapshot["device"] = str(device)
    if torch.cuda.is_available():
        try:
            cuda_device = device if device is not None and getattr(device, "type", None) == "cuda" else None
            snapshot["cuda_allocated_mb"] = round(torch.cuda.memory_allocated(cuda_device) / (1024 * 1024), 3)
            snapshot["cuda_reserved_mb"] = round(torch.cuda.memory_reserved(cuda_device) / (1024 * 1024), 3)
            snapshot["cuda_max_allocated_mb"] = round(torch.cuda.max_memory_allocated(cuda_device) / (1024 * 1024), 3)
        except (RuntimeError, ValueError):
            snapshot["cuda_snapshot_error"] = "unavailable"
    return snapshot


def parse_args():
    parser = argparse.ArgumentParser(description="Count people entering and exiting through a bus door.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Input video.")
    parser.add_argument("--jobs-file", type=Path, help="JSON list of video/output/history jobs processed with one model load.")
    parser.add_argument("--door-polygon", type=Path, help="Door JSON; defaults from the video name.")
    parser.add_argument("--polygon-dir", type=Path, default=DEFAULT_POLYGON_DIR, help="Directory of door JSON files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Annotated MP4 output.")
    parser.add_argument("--history-output", type=Path, default=DEFAULT_HISTORY, help="JSON event and track output.")
    parser.add_argument("--detections-output", type=Path, help="Compressed detection diagnostics CSV; derived from history by default.")
    parser.add_argument("--track-events-output", type=Path, help="Compressed tracker event CSV; derived from history by default.")
    parser.add_argument("--tracks-output", type=Path, help="Track evaluation CSV; derived from history by default.")
    parser.add_argument("--metrics-output", type=Path, help="Atomic metrics JSON; derived from history by default.")
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_BPJDET_REPOSITORY, help="BPJDet source directory.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_BPJDET_WEIGHTS, help="BPJDet body-face checkpoint.")
    parser.add_argument("--device", default=None, help="cuda:0, 0, or cpu (automatic by default).")
    parser.add_argument("--img-size", type=int, default=640, help="BPJDet inference size.")
    parser.add_argument("--batch-size", type=int, default=1, help="Frames processed by each BPJDet forward pass.")
    parser.add_argument("--confidence", type=float, default=0.20, help="Minimum person confidence.")
    parser.add_argument("--iou", type=float, default=0.75, help="NMS IoU threshold.")
    parser.add_argument("--match-iou", type=float, default=0.6, help="Body-face association IoU.")
    parser.add_argument("--tracker", choices=("bytetrack", "appearance"), default="bytetrack", help="bytetrack: motion-only ByteTrack (no ReID, no duplicate merging); appearance: legacy TransReID tracker.")
    parser.add_argument("--decision-rule", choices=("zone", "progress"), default="zone", help="zone: count a crossing when a track is seen in the outside zone and later in the inside zone (hysteresis); progress: legacy start-to-end net progress predicates.")
    parser.add_argument("--zone-outside-progress", type=float, default=None, help="Door-axis progress (px) at or below which a track is 'outside' for the zone rule. Default: 'zone.outside_progress' from the door polygon JSON, else 110.")
    parser.add_argument("--zone-inside-progress", type=float, default=None, help="Door-axis progress (px) at or above which a track is 'inside'. Default: polygon JSON 'zone.inside_progress', else 128.")
    parser.add_argument("--head-outside-x", type=float, default=None, help="Zone rule, second path: BPJDet head point at x <= this (and y <= --head-outside-max-y) marks 'outside'. Default: polygon JSON 'zone.head_outside_x', else 0 (path disabled).")
    parser.add_argument("--head-inside-x", type=float, default=None, help="Head point at x >= this after an 'outside' observation completes an enter. Default: polygon JSON 'zone.head_inside_x'.")
    parser.add_argument("--head-outside-max-y", type=float, default=None, help="Head 'outside' observations must also have y <= this. Default: polygon JSON 'zone.head_outside_max_y'.")
    parser.add_argument("--bytetrack-track-thresh", type=float, default=0.3, help="ByteTrack high-confidence threshold for starting/matching tracks.")
    parser.add_argument("--bytetrack-match-thresh", type=float, default=0.8, help="ByteTrack IoU-distance matching threshold.")
    parser.add_argument("--bytetrack-track-buffer", type=int, default=25, help="Frames a lost ByteTrack track is kept before removal.")
    parser.add_argument("--transreid-repo-dir", type=Path, default=DEFAULT_TRANSREID_REPOSITORY)
    parser.add_argument("--transreid-config", type=Path, default=DEFAULT_TRANSREID_CONFIG)
    parser.add_argument("--reid-weights", type=Path, default=DEFAULT_REID_WEIGHTS)
    parser.add_argument("--reid-distance", type=float, default=0.40, help="Maximum cosine distance for one ID.")
    parser.add_argument("--match-center-distance", type=float, default=40.0, help="Maximum center jump in pixels for an appearance match.")
    parser.add_argument("--appearance-max-missing", type=int, default=15, help="Maximum missing frames for appearance-only ID matching.")
    parser.add_argument("--appearance-sample-count", type=int, default=5, help="Number of recent embeddings kept per track for appearance matching; a larger window anchors matches against older samples and resists gradual identity drift onto a different person.")
    parser.add_argument("--tracker-max-missing", type=int, default=45, help="Frames retained after a missed detection.")
    parser.add_argument("--reconnect-iou", type=float, default=0.25, help="Minimum IoU for reconnecting a recently lost door track.")
    parser.add_argument("--reconnect-bottom-distance", type=float, default=100.0, help="Maximum bottom-center distance in pixels for reconnecting a door track.")
    parser.add_argument("--duplicate-merge-iou", type=float, default=0.35, help="Minimum IoU for merging simultaneous duplicate door tracks.")
    parser.add_argument("--duplicate-merge-bottom-distance", type=float, default=60.0, help="Maximum bottom-center distance in pixels for merging simultaneous duplicate door tracks.")
    parser.add_argument("--duplicate-merge-progress-distance", type=float, default=35.0, help="Maximum door-axis distance in pixels for merging simultaneous duplicate door tracks.")
    parser.add_argument("--duplicate-merge-min-frames", type=int, default=2, help="Consecutive matching frames required before merging duplicate door tracks.")
    parser.add_argument("--duplicate-merge-reid-distance", type=float, default=0.20, help="Maximum ReID cosine distance for merging simultaneous duplicate door tracks.")
    parser.add_argument("--min-observations", type=int, default=3, help="Minimum detections to count an event.")
    parser.add_argument("--min-progress", type=float, default=20.0, help="Minimum directed travel along door axis in pixels.")
    parser.add_argument("--entry-boundary-distance", type=float, default=50.0, help="Maximum distance in pixels from the salon door edge when an entering track is lost inside the polygon.")
    parser.add_argument(
        "--salon-start-min-bottom-center-ratio",
        type=float,
        default=2.0,
        help=(
            "For an entering track first seen in door_salon, require bottom bbox progress "
            "to be at least this multiple of center progress; 0 disables this precision filter."
        ),
    )
    parser.add_argument(
        "--lane-fallback-progress-multiplier",
        type=float,
        default=2.5,
        help=(
            "For a track never seen strictly inside the door polygon, require net and bottom "
            "progress to be at least this multiple of --min-progress before counting an enter; "
            "0 disables the lane fallback so only strict polygon containment qualifies."
        ),
    )
    parser.add_argument("--lane-margin", type=float, default=35.0, help="Extra corridor width on either side of door.")
    parser.add_argument("--completion-margin", type=float, default=18.0, help="Distance beyond a door edge required to complete an event.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the complete video.")
    parser.add_argument("--profile", action="store_true", help="Print synchronized pipeline stage timings.")
    return parser.parse_args()


class TransReIDExtractor:
    def __init__(self, repository, config_path, weights_path, device):
        sys.path.insert(0, str(repository))
        from config import cfg
        from model import make_model

        config = cfg.clone()
        config.merge_from_file(str(config_path))
        config.MODEL.PRETRAIN_CHOICE = "self"
        self.device = torch.device(device)
        self.model = make_model(config, num_class=702, camera_num=8, view_num=1)
        self.model.load_param(str(weights_path))
        self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(config.INPUT.SIZE_TEST),
                transforms.ToTensor(),
                transforms.Normalize(config.INPUT.PIXEL_MEAN, config.INPUT.PIXEL_STD),
            ]
        )

    def __call__(self, crops):
        images = torch.stack([self.transform(Image.fromarray(crop)) for crop in crops]).to(self.device)
        labels = torch.zeros(len(crops), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            return self.model(images, cam_label=labels, view_label=labels)


class AppearanceTracker:
    """Match people by appearance, then reconnect recently lost door tracks by motion."""

    def __init__(self, extractor, max_distance, match_center_distance, appearance_max_missing, max_missing_frames, door, reconnect_iou, reconnect_bottom_distance, duplicate_merge_iou, duplicate_merge_bottom_distance, duplicate_merge_progress_distance, duplicate_merge_min_frames, duplicate_merge_reid_distance, appearance_sample_count=5):
        self.extractor = extractor
        self.max_distance = max_distance
        self.match_center_distance = match_center_distance
        self.appearance_max_missing = appearance_max_missing
        self.max_missing_frames = max_missing_frames
        self.door = door
        self.reconnect_iou = reconnect_iou
        self.reconnect_bottom_distance = reconnect_bottom_distance
        self.duplicate_merge_iou = duplicate_merge_iou
        self.duplicate_merge_bottom_distance = duplicate_merge_bottom_distance
        self.duplicate_merge_progress_distance = duplicate_merge_progress_distance
        self.duplicate_merge_min_frames = duplicate_merge_min_frames
        self.duplicate_merge_reid_distance = duplicate_merge_reid_distance
        self.appearance_sample_count = appearance_sample_count
        self.next_id = 1
        self.tracks = {}
        self.created_track_ids = set()
        self.duplicate_streaks = {}
        self.track_records = {}
        self.last_update_diagnostics = {"detections": [], "events": []}

    @staticmethod
    def crop(frame, box):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(value) for value in box)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

    @staticmethod
    def bottom_center(box):
        x1, _, x2, y2 = box
        return np.array(((float(x1) + float(x2)) / 2.0, float(y2)), dtype=np.float32)

    @staticmethod
    def iou(first_box, second_box):
        first_x1, first_y1, first_x2, first_y2 = first_box
        second_x1, second_y1, second_x2, second_y2 = second_box
        intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
        intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
        intersection = intersection_width * intersection_height
        union = (first_x2 - first_x1) * (first_y2 - first_y1) + (second_x2 - second_x1) * (second_y2 - second_y1) - intersection
        return float(intersection / union) if union > 0 else 0.0

    @staticmethod
    def box_area(box):
        return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))

    @staticmethod
    def appearance_distance(first_track, second_track):
        return 1.0 - float(np.dot(first_track["last_embedding"], second_track["last_embedding"]))

    def assign(self, track_id, detection_index, detections, embeddings, result, method, metrics, frame_index):
        track = self.tracks[track_id]
        box = detections[detection_index]
        bottom = self.bottom_center(box)
        track["samples"].append(embeddings[detection_index])
        track["last_embedding"] = embeddings[detection_index]
        track["previous_bottom"] = track.get("last_bottom")
        track["last_bottom"] = bottom
        track["last_box"] = box
        track["missing"] = 0
        result[detection_index] = track_id
        diagnostic = self.last_update_diagnostics["detections"][detection_index]
        diagnostic.update({"assigned_track_id": track_id, "assignment_method": method, "reason": "accepted", **metrics})
        record = self.track_records.setdefault(track_id, {})
        record["last_matched_frame"] = frame_index
        self.last_update_diagnostics["events"].append({
            "event": "matched" if method == "appearance" else "reconnected",
            "frame": frame_index,
            "track_id": track_id,
            "detection_index": detection_index,
            "assignment_method": method,
            "reason": "accepted",
            **metrics,
        })

    def geometric_reconnect_evaluation(self, track, box):
        metrics = {
            "missing_before": track["missing"],
            "match_iou": None,
            "bottom_distance": None,
            "previous_progress_delta": None,
            "new_progress_delta": None,
        }
        previous_bottom = track.get("previous_bottom")
        last_bottom = track.get("last_bottom")
        if track["missing"] < 2:
            return False, metrics, "missing_below_reconnect_minimum"
        if previous_bottom is None or last_bottom is None:
            return False, metrics, "insufficient_motion_history"
        new_bottom = self.bottom_center(box)
        metrics["match_iou"] = self.iou(track["last_box"], box)
        metrics["bottom_distance"] = float(np.linalg.norm(new_bottom - last_bottom))
        if not self.door.in_lane(last_bottom):
            return False, metrics, "last_position_outside_lane"
        if self.door.region(new_bottom) == "other":
            return False, metrics, "new_position_outside_lane"
        if metrics["match_iou"] < self.reconnect_iou:
            return False, metrics, "reconnect_iou_below_threshold"
        if metrics["bottom_distance"] > self.reconnect_bottom_distance:
            return False, metrics, "reconnect_bottom_distance_above_threshold"
        metrics["previous_progress_delta"] = self.door.coordinates(last_bottom)[0] - self.door.coordinates(previous_bottom)[0]
        metrics["new_progress_delta"] = self.door.coordinates(new_bottom)[0] - self.door.coordinates(last_bottom)[0]
        direction_ok = (
            abs(metrics["previous_progress_delta"]) < 2.0
            or abs(metrics["new_progress_delta"]) < 2.0
            or metrics["previous_progress_delta"] * metrics["new_progress_delta"] > 0
        )
        return direction_ok, metrics, "accepted" if direction_ok else "reconnect_direction_reversal"

    def is_geometric_reconnect(self, track, box):
        return self.geometric_reconnect_evaluation(track, box)[0]

    def merge_active_duplicates(self, detections, embeddings, result, frame_index):
        candidates = []
        matching_pairs = set()
        candidate_metrics = {}
        active_ids = [track_id for track_id, track in self.tracks.items() if track["missing"] == 0]
        for index, first_id in enumerate(active_ids):
            first_track = self.tracks[first_id]
            first_bottom = first_track["last_bottom"]
            first_center = box_center(first_track["last_box"])
            if not self.door.in_lane(first_center):
                continue
            for second_id in active_ids[index + 1:]:
                second_track = self.tracks[second_id]
                second_bottom = second_track["last_bottom"]
                second_center = box_center(second_track["last_box"])
                if not self.door.in_lane(second_center):
                    continue
                pair = tuple(sorted((first_id, second_id)))
                metrics = {
                    "match_iou": self.iou(first_track["last_box"], second_track["last_box"]),
                    "bottom_distance": float(np.linalg.norm(first_bottom - second_bottom)),
                    "progress_distance": abs(self.door.coordinates(first_center)[0] - self.door.coordinates(second_center)[0]),
                    "appearance_pair_distance": self.appearance_distance(first_track, second_track),
                }
                predicates = {
                    "iou_ok": bool(metrics["match_iou"] >= self.duplicate_merge_iou),
                    "bottom_distance_ok": bool(metrics["bottom_distance"] <= self.duplicate_merge_bottom_distance),
                    "progress_distance_ok": bool(metrics["progress_distance"] <= self.duplicate_merge_progress_distance),
                    "appearance_distance_ok": bool(metrics["appearance_pair_distance"] <= self.duplicate_merge_reid_distance),
                }
                accepted = all(predicates.values())
                if accepted:
                    matching_pairs.add(pair)
                    self.duplicate_streaks[pair] = self.duplicate_streaks.get(pair, 0) + 1
                streak = self.duplicate_streaks.get(pair, 0)
                reason = "accepted" if accepted else next(name for name, value in predicates.items() if not value)
                candidate_metrics[pair] = {**metrics, "streak": streak, "reason": reason}
                self.last_update_diagnostics["events"].append({
                    "event": "merge_candidate", "frame": frame_index, "track_id": pair[0], "other_track_id": pair[1],
                    "streak": streak, "reason": reason, "metrics_json": json.dumps(predicates, sort_keys=True), **metrics,
                })
                if accepted and streak >= self.duplicate_merge_min_frames:
                    candidates.append((-metrics["match_iou"], metrics["bottom_distance"], first_id, second_id))

        self.duplicate_streaks = {pair: streak for pair, streak in self.duplicate_streaks.items() if pair in matching_pairs}
        merged = []
        for _, _, first_id, second_id in sorted(candidates):
            if first_id not in self.tracks or second_id not in self.tracks:
                continue
            survivor_id, absorbed_id = sorted((first_id, second_id))
            current_indices = [index for index, track_id in enumerate(result) if track_id in (survivor_id, absorbed_id)]
            if not current_indices:
                continue
            winner_index = max(current_indices, key=lambda index: self.box_area(detections[index]))
            survivor = self.tracks[survivor_id]
            absorbed = self.tracks[absorbed_id]
            survivor["samples"] = deque(list(survivor["samples"]) + list(absorbed["samples"]), maxlen=self.appearance_sample_count)
            if embeddings[winner_index] is not None:
                survivor["samples"].append(embeddings[winner_index])
            survivor["previous_bottom"] = survivor.get("last_bottom")
            survivor["last_bottom"] = self.bottom_center(detections[winner_index])
            survivor["last_box"] = detections[winner_index]
            survivor["missing"] = 0
            for detection_index in current_indices:
                original_id = result[detection_index]
                result[detection_index] = survivor_id if detection_index == winner_index else None
                diagnostic = self.last_update_diagnostics["detections"][detection_index]
                if detection_index != winner_index:
                    diagnostic.update({
                        "assigned_track_id": original_id,
                        "assignment_method": f"{diagnostic.get('assignment_method', 'unknown')}|merged_duplicate_suppressed",
                        "reason": f"absorbed_by_{survivor_id}",
                    })
            del self.tracks[absorbed_id]
            self.track_records.setdefault(absorbed_id, {}).update({"absorbed_by": survivor_id, "termination": "merged", "ended_frame": frame_index})
            self.duplicate_streaks = {pair: streak for pair, streak in self.duplicate_streaks.items() if absorbed_id not in pair}
            metrics = candidate_metrics.get((survivor_id, absorbed_id), {})
            self.last_update_diagnostics["events"].append({
                "event": "merged", "frame": frame_index, "track_id": survivor_id, "other_track_id": absorbed_id,
                "reason": "duplicate_thresholds_and_streak_met", **metrics,
            })
            merged.append((survivor_id, absorbed_id))
        return merged

    def extract_embeddings_batch(self, frame_detections):
        embeddings_by_frame = [[None] * len(detections) for _, detections in frame_detections]
        crops = []
        positions = []
        for frame_index, (frame, detections) in enumerate(frame_detections):
            for detection_index, box in enumerate(detections):
                crop = self.crop(frame, box)
                if crop is not None:
                    crops.append(crop)
                    positions.append((frame_index, detection_index))
        if crops:
            batch = self.extractor(crops).detach().cpu().numpy()
            batch /= np.maximum(np.linalg.norm(batch, axis=1, keepdims=True), 1e-12)
            for (frame_index, detection_index), embedding in zip(positions, batch):
                embeddings_by_frame[frame_index][detection_index] = embedding
        return embeddings_by_frame

    def update_with_embeddings(self, boxes, embeddings, frame_index=None):
        detections = [np.asarray(box, dtype=np.float32) for box in boxes]
        frame_index = -1 if frame_index is None else frame_index
        self.last_update_diagnostics = {
            "detections": [
                {
                    "detection_index": index,
                    "crop_status": "valid" if embedding is not None else "invalid_or_empty",
                    "embedding_status": "computed" if embedding is not None else "not_computed",
                    "assigned_track_id": None,
                    "assignment_method": "unassigned",
                    "reason": "no_eligible_match",
                }
                for index, embedding in enumerate(embeddings)
            ],
            "events": [],
        }
        previously_active = set(self.tracks)
        for track in self.tracks.values():
            track["missing"] += 1

        candidates = []
        for track_id, track in self.tracks.items():
            if track["missing"] > self.appearance_max_missing:
                continue
            for detection_index, embedding in enumerate(embeddings):
                if embedding is None:
                    continue
                diagnostic = self.last_update_diagnostics["detections"][detection_index]
                center_distance = float(np.linalg.norm(np.asarray(box_center(track["last_box"])) - np.asarray(box_center(detections[detection_index]))))
                if center_distance > self.match_center_distance * min(max(1, track["missing"]), 3):
                    if diagnostic.get("center_distance") is None or center_distance < diagnostic["center_distance"]:
                        diagnostic.update({
                            "center_distance": center_distance,
                            "missing_before": track["missing"],
                            "reason": "appearance_center_distance_above_threshold",
                        })
                    continue
                distance = min(1.0 - float(np.dot(sample, embedding)) for sample in track["samples"])
                if diagnostic.get("appearance_distance") is None or distance < diagnostic["appearance_distance"]:
                    diagnostic.update({
                        "appearance_distance": distance,
                        "center_distance": center_distance,
                        "missing_before": track["missing"],
                        "reason": "appearance_candidate" if distance <= self.max_distance else "appearance_distance_above_threshold",
                    })
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, detection_index, center_distance, track["missing"]))

        assigned_tracks, assigned_detections = set(), set()
        result = [None] * len(detections)
        for distance, track_id, detection_index, center_distance, missing_before in sorted(candidates):
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            self.assign(track_id, detection_index, detections, embeddings, result, "appearance", {
                "appearance_distance": distance, "center_distance": center_distance, "missing_before": missing_before,
            }, frame_index)
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)

        reconnect_candidates = []
        for track_id, track in self.tracks.items():
            if track_id in assigned_tracks:
                continue
            for detection_index, box in enumerate(detections):
                if detection_index in assigned_detections:
                    continue
                accepted, metrics, reason = self.geometric_reconnect_evaluation(track, box)
                if accepted:
                    reconnect_candidates.append((-metrics["match_iou"], metrics["bottom_distance"], track_id, detection_index, metrics))
                elif self.last_update_diagnostics["detections"][detection_index]["reason"] == "no_eligible_match":
                    self.last_update_diagnostics["detections"][detection_index]["reason"] = reason
        for _, _, track_id, detection_index, metrics in sorted(reconnect_candidates, key=lambda item: item[:4]):
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            self.assign(track_id, detection_index, detections, embeddings, result, "geometric_reconnect", metrics, frame_index)
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)

        for detection_index, embedding in enumerate(embeddings):
            if detection_index in assigned_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            if embedding is None:
                self.last_update_diagnostics["detections"][detection_index].update({
                    "assignment_method": "unassigned_invalid_crop", "reason": "embedding_unavailable_track_not_created",
                })
                continue
            bottom = self.bottom_center(detections[detection_index])
            self.tracks[track_id] = {
                "samples": deque([embedding], maxlen=self.appearance_sample_count),
                "last_embedding": embedding,
                "last_box": detections[detection_index],
                "last_bottom": bottom,
                "previous_bottom": None,
                "missing": 0,
            }
            self.created_track_ids.add(track_id)
            self.track_records[track_id] = {"created_frame": frame_index, "last_matched_frame": frame_index, "termination": "active"}
            result[detection_index] = track_id
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
            self.last_update_diagnostics["detections"][detection_index].update({
                "assigned_track_id": track_id, "assignment_method": "created", "reason": "new_track_from_unmatched_embedding",
            })
            self.last_update_diagnostics["events"].append({
                "event": "created", "frame": frame_index, "track_id": track_id, "detection_index": detection_index,
                "assignment_method": "created", "count_delta": 0, "reason": "new_track_from_unmatched_embedding",
            })

        merged = self.merge_active_duplicates(detections, embeddings, result, frame_index)
        expired_ids = [track_id for track_id, track in self.tracks.items() if track["missing"] > self.max_missing_frames]
        for track_id in sorted(previously_active):
            if track_id in self.tracks and track_id not in assigned_tracks and track_id not in expired_ids:
                self.last_update_diagnostics["events"].append({
                    "event": "missed", "frame": frame_index, "track_id": track_id,
                    "missing_frames": self.tracks[track_id]["missing"], "count_delta": 0, "reason": "no_detection_assigned",
                })
        for track_id in expired_ids:
            missing = self.tracks[track_id]["missing"]
            self.track_records.setdefault(track_id, {}).update({"expired_frame": frame_index, "termination": "expired", "ended_frame": frame_index})
            self.last_update_diagnostics["events"].append({
                "event": "expired", "frame": frame_index, "track_id": track_id,
                "missing_frames": missing, "count_delta": 0, "reason": "tracker_max_missing_exceeded",
            })
        self.tracks = {track_id: track for track_id, track in self.tracks.items() if track_id not in expired_ids}
        return result, merged

    def update(self, frame, boxes):
        detections = [np.asarray(box, dtype=np.float32) for box in boxes]
        embeddings = self.extract_embeddings_batch([(frame, detections)])[0]
        return self.update_with_embeddings(detections, embeddings)


class MotionTracker:
    """ByteTrack (motion/IoU only) behind the AppearanceTracker interface used by the pipeline.

    No ReID and no duplicate merging: one track per continuously visible body, so a person's
    whole door crossing lands in one ID instead of many fragments that each get evaluated.
    """

    def __init__(self, fps, frame_size, track_thresh, match_thresh, track_buffer, min_conf):
        from boxmot.trackers.bytetrack.bytetrack import ByteTrack

        self.tracker = ByteTrack(min_conf=min_conf, track_thresh=track_thresh, match_thresh=match_thresh, track_buffer=track_buffer, frame_rate=int(round(fps)))
        width, height = frame_size
        self.blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.max_missing_frames = track_buffer
        self.tracks = {}
        self.created_track_ids = set()
        self.track_records = {}
        self.smoothed_boxes = []
        self.last_update_diagnostics = {"detections": [], "events": []}

    @staticmethod
    def extract_embeddings_batch(frame_detections):
        return [[None] * len(detections) for _, detections in frame_detections]

    def update_with_embeddings(self, boxes, embeddings, frame_index=None, scores=None):
        detections = [np.asarray(box, dtype=np.float32) for box in boxes]
        scores = [1.0] * len(detections) if scores is None else list(scores)
        frame_index = -1 if frame_index is None else frame_index
        self.last_update_diagnostics = {
            "detections": [
                {"detection_index": index, "crop_status": "not_used", "embedding_status": "not_computed",
                 "assigned_track_id": None, "assignment_method": "unassigned", "reason": "no_eligible_match"}
                for index in range(len(detections))
            ],
            "events": [],
        }
        for track in self.tracks.values():
            track["missing"] += 1
        if detections:
            array = np.array([[*box, float(score), 0] for box, score in zip(detections, scores)], dtype=np.float32)
        else:
            array = np.empty((0, 6), dtype=np.float32)
        outputs = np.asarray(self.tracker.update(array, self.blank_frame))
        result = [None] * len(detections)
        # Kalman-smoothed boxes; observations use these (jittery raw boxes flip the zone state and were
        # measured to double false enters against the interval GT).
        self.smoothed_boxes = [None] * len(detections)
        for row in outputs:
            track_id, detection_index = int(row[4]), int(row[7])
            if not 0 <= detection_index < len(detections) or result[detection_index] is not None:
                continue
            box = np.asarray(row[:4], dtype=np.float32)
            self.smoothed_boxes[detection_index] = box
            created = track_id not in self.tracks and track_id not in self.created_track_ids
            self.tracks[track_id] = {"last_box": box, "last_bottom": AppearanceTracker.bottom_center(box), "missing": 0}
            result[detection_index] = track_id
            if created:
                self.created_track_ids.add(track_id)
                self.track_records[track_id] = {"created_frame": frame_index, "last_matched_frame": frame_index, "termination": "active"}
                method, reason = "created", "new_bytetrack_id"
            else:
                self.track_records.setdefault(track_id, {})["last_matched_frame"] = frame_index
                method, reason = "motion", "accepted"
            self.last_update_diagnostics["detections"][detection_index].update({"assigned_track_id": track_id, "assignment_method": method, "reason": reason})
            self.last_update_diagnostics["events"].append({
                "event": method if method == "created" else "matched", "frame": frame_index, "track_id": track_id,
                "detection_index": detection_index, "assignment_method": method, "count_delta": 0, "reason": reason,
            })
        for detection_index, diagnostic in enumerate(self.last_update_diagnostics["detections"]):
            if diagnostic["assigned_track_id"] is None:
                diagnostic["reason"] = "not_confirmed_by_tracker"
        expired_ids = [track_id for track_id, track in self.tracks.items() if track["missing"] > self.max_missing_frames]
        for track_id, track in self.tracks.items():
            if track["missing"] > 0 and track_id not in expired_ids:
                self.last_update_diagnostics["events"].append({
                    "event": "missed", "frame": frame_index, "track_id": track_id,
                    "missing_frames": track["missing"], "count_delta": 0, "reason": "no_detection_assigned",
                })
        for track_id in expired_ids:
            self.track_records.setdefault(track_id, {}).update({"expired_frame": frame_index, "termination": "expired", "ended_frame": frame_index})
            self.last_update_diagnostics["events"].append({
                "event": "expired", "frame": frame_index, "track_id": track_id,
                "missing_frames": self.tracks[track_id]["missing"], "count_delta": 0, "reason": "track_buffer_exceeded",
            })
            del self.tracks[track_id]
        return result, []


class DoorGeometry:
    """Door-local coordinates: progress grows from outside/top toward salon/bottom."""

    def __init__(self, polygon_path, lane_margin, completion_margin):
        polygon_path = Path(polygon_path)
        payload = json.loads(polygon_path.read_text(encoding="utf-8"))
        self.polygon = np.asarray(payload.get("points"), dtype=np.float32)
        # Optional per-camera counting calibration (zone/head thresholds are pixel quantities of one layout).
        self.zone_calibration = dict(payload.get("zone") or {})
        if self.polygon.shape != (4, 2):
            raise ValueError(f"Expected four polygon points in {polygon_path}, got {self.polygon.shape}")
        top = self.polygon[[0, 1]].mean(axis=0)
        bottom = self.polygon[[2, 3]].mean(axis=0)
        axis = bottom - top
        self.length = float(np.linalg.norm(axis))
        if self.length <= 1e-6:
            raise ValueError("Door polygon has no top-to-bottom axis")
        self.top = top
        self.axis = axis / self.length
        self.normal = np.array((-self.axis[1], self.axis[0]), dtype=np.float32)
        width = max(float(np.linalg.norm(self.polygon[1] - self.polygon[0])), float(np.linalg.norm(self.polygon[2] - self.polygon[3])))
        self.half_lane = width / 2.0 + lane_margin
        self.completion_margin = completion_margin

    def coordinates(self, point):
        relative = np.asarray(point, dtype=np.float32) - self.top
        return float(np.dot(relative, self.axis)), float(np.dot(relative, self.normal))

    def in_lane(self, point):
        progress, lateral = self.coordinates(point)
        return -self.completion_margin <= progress <= self.length + self.completion_margin and abs(lateral) <= self.half_lane

    def contains(self, point):
        return cv2.pointPolygonTest(self.polygon.reshape(-1, 1, 2), tuple(point), False) >= 0

    def region(self, point):
        progress, lateral = self.coordinates(point)
        if abs(lateral) > self.half_lane:
            return "other"
        if progress < -self.completion_margin:
            return "outside"
        if progress > self.length + self.completion_margin:
            return "salon"
        if progress <= self.length / 2.0:
            return "door_outside"
        return "door_salon"


def box_center(box):
    x1, y1, x2, y2 = box
    return [(float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0]


def has_duplicate_entry_track(first_observations, second_observations, min_iou=0.35, min_shared_frames=3, max_start_gap=20):
    if not first_observations or not second_observations:
        return False
    if abs(first_observations[0]["frame"] - second_observations[0]["frame"]) > max_start_gap:
        return False
    second_boxes = {observation["frame"]: observation["bbox"] for observation in second_observations}
    matching_frames = 0
    for observation in first_observations:
        other_box = second_boxes.get(observation["frame"])
        if other_box is not None and AppearanceTracker.iou(observation["bbox"], other_box) >= min_iou:
            matching_frames += 1
            if matching_frames >= min_shared_frames:
                return True
    return False


def evaluate_track(
    observations,
    door,
    min_observations,
    min_progress,
    entry_boundary_distance,
    missing_frames,
    max_missing_frames,
    salon_start_min_bottom_center_ratio=2.0,
    lane_fallback_progress_multiplier=2.5,
):
    """Return the crossing decision plus all predicates used to reach it."""
    metrics = {
        "observation_count": len(observations),
        "required_observations": min_observations,
        "required_exit_observations": max(min_observations, 4),
        "min_progress": min_progress,
        "entry_boundary_distance": entry_boundary_distance,
        "missing_frames": missing_frames,
        "max_missing_frames": max_missing_frames,
        "salon_start_min_bottom_center_ratio": salon_start_min_bottom_center_ratio,
        "lane_fallback_progress_multiplier": lane_fallback_progress_multiplier,
        "lane_fallback_progress_required": min_progress * lane_fallback_progress_multiplier,
        "bottom_to_center_progress_ratio": None,
        "salon_start_depth_evidence_required": False,
        "start_region": None,
        "end_region": None,
        "start_progress": None,
        "end_progress": None,
        "net_progress": None,
        "bottom_start_progress": None,
        "bottom_end_progress": None,
        "bottom_net_progress": None,
        "was_inside_door": False,
        "was_in_lane": False,
        "finished_near_salon_edge": False,
        "finished_at_outer_edge": False,
        "disappeared_at_door": False,
    }
    enter_predicates = {
        "minimum_observations": len(observations) >= min_observations,
        "was_inside_door_or_lane": False,
        "start_not_salon": False,
        "finished_beyond_or_near_salon_edge": False,
        "center_progress_sufficient": False,
        "bottom_progress_sufficient": False,
        "salon_start_has_strong_depth_evidence": False,
    }
    exit_predicates = {
        "minimum_exit_observations": len(observations) >= max(min_observations, 4),
        "started_in_salon": False,
        "was_in_lane": False,
        "reverse_progress_sufficient": False,
        "finished_at_outer_edge_or_disappeared": False,
    }
    if not observations:
        return {"decision": None, "reason_codes": ["no_observations"], "metrics": metrics, "enter_predicates": enter_predicates, "exit_predicates": exit_predicates}

    centers = [item["center"] for item in observations]
    coordinates = [door.coordinates(center) for center in centers]
    progress = [coordinate[0] for coordinate in coordinates]
    regions = [door.region(center) for center in centers]
    bottom_centers = [AppearanceTracker.bottom_center(item["bbox"]) for item in observations]
    bottom_progress = [door.coordinates(center)[0] for center in bottom_centers]
    net_progress = progress[-1] - progress[0]
    bottom_net_progress = bottom_progress[-1] - bottom_progress[0]
    bottom_to_center_progress_ratio = (
        bottom_net_progress / net_progress if net_progress > 0.0 else None
    )
    salon_start_depth_evidence_required = (
        salon_start_min_bottom_center_ratio > 0.0 and regions[0] == "door_salon"
    )
    salon_start_has_strong_depth_evidence = (
        not salon_start_depth_evidence_required
        or bottom_net_progress >= salon_start_min_bottom_center_ratio * net_progress
    )
    metrics.update({
        "start_region": regions[0],
        "end_region": regions[-1],
        "start_progress": progress[0],
        "end_progress": progress[-1],
        "net_progress": net_progress,
        "bottom_start_progress": bottom_progress[0],
        "bottom_end_progress": bottom_progress[-1],
        "bottom_net_progress": bottom_net_progress,
        "bottom_to_center_progress_ratio": bottom_to_center_progress_ratio,
        "salon_start_depth_evidence_required": salon_start_depth_evidence_required,
        "was_inside_door": any(door.contains(center) for center in centers),
        "was_in_lane": any(door.in_lane(center) for center in centers),
    })
    metrics["finished_near_salon_edge"] = door.in_lane(centers[-1]) and progress[-1] >= door.length - entry_boundary_distance
    metrics["finished_at_outer_edge"] = progress[-1] <= door.completion_margin
    metrics["disappeared_at_door"] = missing_frames >= max_missing_frames and door.in_lane(centers[-1])
    lane_fallback_progress_met = (
        lane_fallback_progress_multiplier > 0.0
        and metrics["net_progress"] >= metrics["lane_fallback_progress_required"]
        and metrics["bottom_net_progress"] >= metrics["lane_fallback_progress_required"]
    )
    enter_predicates.update({
        # Occluded/fragmented tracks near a crowded door often never register a center
        # strictly inside the tight polygon; require stronger progress evidence when
        # falling back to the broader lane so distant bystanders are not counted.
        "was_inside_door_or_lane": (
            metrics["was_inside_door"]
            or (metrics["was_in_lane"] and lane_fallback_progress_met)
        ),
        "start_not_salon": metrics["start_region"] != "salon",
        "finished_beyond_or_near_salon_edge": not door.contains(centers[-1]) or metrics["finished_near_salon_edge"],
        "center_progress_sufficient": metrics["net_progress"] >= min_progress,
        "bottom_progress_sufficient": metrics["bottom_net_progress"] >= min_progress,
        "salon_start_has_strong_depth_evidence": salon_start_has_strong_depth_evidence,
    })
    exit_predicates.update({
        "started_in_salon": metrics["start_region"] == "salon",
        "was_in_lane": metrics["was_in_lane"],
        "reverse_progress_sufficient": -metrics["net_progress"] >= min_progress,
        "finished_at_outer_edge_or_disappeared": metrics["finished_at_outer_edge"] or metrics["disappeared_at_door"],
    })

    if all(enter_predicates.values()):
        decision = "enter"
        reason_codes = ["accepted_enter"]
    elif all(exit_predicates.values()):
        decision = "exit"
        reason_codes = ["accepted_exit"]
    else:
        decision = None
        failed_enter = [f"enter_{name}_failed" for name, value in enter_predicates.items() if not value]
        failed_exit = [f"exit_{name}_failed" for name, value in exit_predicates.items() if not value]
        reason_codes = failed_enter + failed_exit
    return {
        "decision": decision,
        "reason_codes": reason_codes,
        "metrics": metrics,
        "enter_predicates": enter_predicates,
        "exit_predicates": exit_predicates,
    }


def evaluate_track_zone(observations, door, min_observations, outside_progress, inside_progress, missing_frames, max_missing_frames, head_outside_x=0.0, head_inside_x=0.0, head_outside_max_y=0.0):
    """Zone-crossing rule with hysteresis: outside zone (progress <= outside) then inside zone (>= inside) = enter, reverse = exit.

    Second enter path (head_outside_x > 0): the BPJDet head point moves from the door (x <= head_outside_x,
    y <= head_outside_max_y) to the salon side (x >= head_inside_x). People close to the camera fill the
    whole frame height, so the body centre is pinned at mid-frame and never registers as 'outside'; the
    regressed head point is not clamped and still shows the door->salon travel.

    The inward crossing wins whenever the track contains one, even if an outward crossing came first:
    an early in->out pattern is almost always the ID drifting from a person deep inside to the next
    person at the top of the door (validated offline: first-crossing/net-crossing variants lost real
    entries and produced dozens of phantom exits). Unlike the progress rule this does not depend on
    where the track starts/ends, so it works for whole, unfragmented tracks whose box shrinks past the
    camera. Metrics reuse evaluate_track so the track CSV/JSON schema stays identical.
    """
    evaluation = evaluate_track(observations, door, min_observations, 1.0, 0.0, missing_frames, max_missing_frames)
    metrics = evaluation["metrics"]
    metrics.update({"zone_outside_progress": outside_progress, "zone_inside_progress": inside_progress, "zone_first_crossing": None, "zone_crossing_frame": None, "zone_crossing_path": None})
    state = None
    crossing = None
    first_exit_frame = None
    for item in observations:
        progress, lateral = door.coordinates(item["center"])
        if abs(lateral) > door.half_lane:
            continue
        if progress <= outside_progress:
            if state == "in" and first_exit_frame is None:
                first_exit_frame = item["frame"]
            state = "out"
        elif progress >= inside_progress:
            if state == "out":
                crossing = "enter"
                metrics["zone_first_crossing"], metrics["zone_crossing_frame"], metrics["zone_crossing_path"] = crossing, item["frame"], "center"
                break
            state = "in"
    if crossing is None and head_outside_x > 0:
        head_state = None
        for item in observations:
            head = item.get("head")
            if not head:
                continue
            if head[0] <= head_outside_x and head[1] <= head_outside_max_y:
                head_state = "out"
            elif head[0] >= head_inside_x and head_state == "out":
                crossing = "enter"
                metrics["zone_first_crossing"], metrics["zone_crossing_frame"], metrics["zone_crossing_path"] = crossing, item["frame"], "head"
                break
    if crossing is None and first_exit_frame is not None:
        crossing = "exit"
        metrics["zone_first_crossing"], metrics["zone_crossing_frame"], metrics["zone_crossing_path"] = crossing, first_exit_frame, "center"
    enough = len(observations) >= min_observations
    enter_predicates = {"minimum_observations": enough, "zone_crossed_inward": crossing == "enter"}
    exit_predicates = {"minimum_exit_observations": enough, "zone_crossed_outward": crossing == "exit"}
    if all(enter_predicates.values()):
        decision, reason_codes = "enter", ["accepted_enter"]
    elif all(exit_predicates.values()):
        decision, reason_codes = "exit", ["accepted_exit"]
    else:
        decision = None
        reason_codes = [f"enter_{name}_failed" for name, value in enter_predicates.items() if not value]
        reason_codes += [f"exit_{name}_failed" for name, value in exit_predicates.items() if not value]
    return {"decision": decision, "reason_codes": reason_codes, "metrics": metrics, "enter_predicates": enter_predicates, "exit_predicates": exit_predicates}


ZONE_DEFAULTS = {"zone_outside_progress": 110.0, "zone_inside_progress": 128.0, "head_outside_x": 0.0, "head_inside_x": 0.0, "head_outside_max_y": 0.0}
ZONE_POLYGON_KEYS = {"zone_outside_progress": "outside_progress", "zone_inside_progress": "inside_progress", "head_outside_x": "head_outside_x", "head_inside_x": "head_inside_x", "head_outside_max_y": "head_outside_max_y"}


def resolve_zone_parameters(args, door):
    """Explicit CLI value > polygon JSON 'zone' block > built-in default; written back onto args."""
    resolved = {}
    for name, key in ZONE_POLYGON_KEYS.items():
        value = getattr(args, name, None)
        if value is None:
            value = door.zone_calibration.get(key, ZONE_DEFAULTS[name])
        resolved[name] = float(value)
        setattr(args, name, resolved[name])
    if not resolved["zone_outside_progress"] < resolved["zone_inside_progress"]:
        raise ValueError("zone outside progress must be smaller than inside progress")
    if resolved["head_outside_x"] > 0 and not resolved["head_outside_x"] < resolved["head_inside_x"]:
        raise ValueError("head outside x must be smaller than head inside x")
    return resolved


def make_track_evaluator(args, door):
    """Bind the configured decision rule so call sites only pass (observations, missing_frames)."""
    if getattr(args, "decision_rule", "progress") == "zone":
        zone = resolve_zone_parameters(args, door)
        return lambda observations, missing_frames: evaluate_track_zone(
            observations, door, args.min_observations, zone["zone_outside_progress"], zone["zone_inside_progress"],
            missing_frames, args.tracker_max_missing,
            head_outside_x=zone["head_outside_x"], head_inside_x=zone["head_inside_x"], head_outside_max_y=zone["head_outside_max_y"],
        )
    return lambda observations, missing_frames: evaluate_track(
        observations, door, args.min_observations, args.min_progress, args.entry_boundary_distance,
        missing_frames, args.tracker_max_missing,
        salon_start_min_bottom_center_ratio=getattr(args, "salon_start_min_bottom_center_ratio", 2.0),
        lane_fallback_progress_multiplier=getattr(args, "lane_fallback_progress_multiplier", 2.5),
    )


def classify_track(
    observations,
    door,
    min_observations,
    min_progress,
    entry_boundary_distance,
    missing_frames,
    max_missing_frames,
    salon_start_min_bottom_center_ratio=2.0,
    lane_fallback_progress_multiplier=2.5,
):
    """Classify a door crossing; evaluate_track exposes the same decision predicates."""
    return evaluate_track(
        observations,
        door,
        min_observations,
        min_progress,
        entry_boundary_distance,
        missing_frames,
        max_missing_frames,
        salon_start_min_bottom_center_ratio=salon_start_min_bottom_center_ratio,
        lane_fallback_progress_multiplier=lane_fallback_progress_multiplier,
    )["decision"]


def draw(frame, door, visible_tracks, counts):
    cv2.polylines(frame, [door.polygon.astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Entered: {counts['enter']}  Exited: {counts['exit']}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    for track_id, box, score, event in visible_tracks:
        x1, y1, x2, y2 = (int(value) for value in box)
        color = (0, 220, 255) if event is None else ((0, 220, 0) if event == "enter" else (0, 80, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id} {event or 'tracking'} {score:.2f}"
        cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return frame


def validate(args, polygon_path):
    paths = [args.video, polygon_path, args.repo_dir, args.weights]
    if getattr(args, "tracker", "appearance") == "appearance":
        paths += [args.transreid_repo_dir, args.transreid_config, args.reid_weights]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")
    if getattr(args, "decision_rule", "progress") == "zone" and None not in (args.zone_outside_progress, args.zone_inside_progress) and not args.zone_outside_progress < args.zone_inside_progress:
        raise ValueError("--zone-outside-progress must be smaller than --zone-inside-progress")
    if getattr(args, "head_outside_x", None) and args.head_inside_x is not None and not args.head_outside_x < args.head_inside_x:
        raise ValueError("--head-outside-x must be smaller than --head-inside-x")
    if getattr(args, "tracker", "appearance") == "bytetrack" and (not 0.0 < args.bytetrack_track_thresh <= 1.0 or not 0.0 < args.bytetrack_match_thresh <= 1.0 or args.bytetrack_track_buffer < 1):
        raise ValueError("ByteTrack thresholds must be valid")
    if args.max_frames < 0 or args.tracker_max_missing < 0 or args.appearance_max_missing < 0 or args.batch_size < 1:
        raise ValueError("Frame limits cannot be negative")
    if args.appearance_sample_count < 1:
        raise ValueError("--appearance-sample-count must be at least 1")
    if not 0.0 <= args.reid_distance <= 2.0:
        raise ValueError("--reid-distance must be in [0, 2]")
    if args.min_observations < 2 or args.min_progress <= 0 or args.entry_boundary_distance < 0 or args.lane_margin < 0 or args.completion_margin < 0:
        raise ValueError("Event thresholds must be non-negative and --min-observations must be at least 2")
    salon_start_min_bottom_center_ratio = getattr(
        args, "salon_start_min_bottom_center_ratio", 2.0
    )
    if (
        not np.isfinite(salon_start_min_bottom_center_ratio)
        or salon_start_min_bottom_center_ratio < 0.0
    ):
        raise ValueError(
            "--salon-start-min-bottom-center-ratio must be finite and non-negative"
        )
    lane_fallback_progress_multiplier = getattr(
        args, "lane_fallback_progress_multiplier", 2.5
    )
    if (
        not np.isfinite(lane_fallback_progress_multiplier)
        or lane_fallback_progress_multiplier < 0.0
    ):
        raise ValueError(
            "--lane-fallback-progress-multiplier must be finite and non-negative"
        )
    if not 0.0 <= args.reconnect_iou <= 1.0 or args.reconnect_bottom_distance <= 0:
        raise ValueError("Reconnect thresholds must be valid")
    if not 0.0 <= args.duplicate_merge_iou <= 1.0 or args.duplicate_merge_bottom_distance <= 0 or args.duplicate_merge_progress_distance <= 0 or args.duplicate_merge_min_frames < 2 or not 0.0 <= args.duplicate_merge_reid_distance <= 2.0:
        raise ValueError("Duplicate merge thresholds must be valid")


def allow_trusted_legacy_checkpoint():
    """BPJDet checkpoints predate PyTorch's ``weights_only=True`` default."""
    original_load = torch.load

    def load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = load


class InferenceRuntime:
    def __init__(self, args):
        allow_trusted_legacy_checkpoint()
        sys.path.insert(0, str(args.repo_dir))
        from models.experimental import attempt_load
        from utils.datasets import LoadImages
        from utils.general import check_img_size, non_max_suppression
        from utils.torch_utils import select_device
        from val import post_process_batch

        self.device = select_device(args.device or ("0" if torch.cuda.is_available() else "cpu"), batch_size=args.batch_size)
        self.model = attempt_load(str(args.weights), map_location=self.device).eval()
        self.image_size = check_img_size(args.img_size, s=int(self.model.stride.max()))
        if self.device.type != "cpu":
            self.model(torch.zeros(1, 3, self.image_size, self.image_size, device=self.device).type_as(next(self.model.parameters())))
        self.extractor = None
        if getattr(args, "tracker", "appearance") == "appearance":
            self.extractor = TransReIDExtractor(args.transreid_repo_dir, args.transreid_config, args.reid_weights, str(self.device))
        self.load_images = LoadImages
        self.non_max_suppression = non_max_suppression
        self.post_process_batch = post_process_batch


def canonical_track_id(track_id, aliases):
    current = track_id
    seen = []
    while current in aliases and current not in seen:
        seen.append(current)
        current = aliases[current]
    return current, seen + ([current] if seen else [])


def canonical_deduplicated_counts(created_track_ids, aliases, final_decisions):
    """Collapse only conflicting alias groups to avoid deduplicating a real queue.

    A canonical group whose absorbed alias fragments AGREE on one event type
    (all "enter", say) most plausibly represents several distinct, similarly
    dressed people who queued close together and got merged by mistake (each
    fragment independently satisfied the full crossing criteria) -- those
    legacy events are kept as-is. A group with CONFLICTING evidence (a mix of
    "enter" and "exit") instead points to one entity oscillating near the door
    (e.g. a driver standing by it for a long time); that is collapsed to at
    most one enter and one exit so it cannot inflate the count.

    Known trade-off (see /memories/repo/door-flow-counter-notes.md): this
    under-collapses genuinely single-entity groups that happen to agree on
    one event type (e.g. video 2 seg2 canon=6, visually confirmed one person,
    regressed from 39 to 59 vs truth 41 when this variant was last tested).
    """
    events_by_canonical = {}
    for track_id in created_track_ids:
        event = final_decisions.get(track_id)
        if event is None:
            continue
        canonical_id = canonical_track_id(track_id, aliases)[0]
        events_by_canonical.setdefault(canonical_id, []).append(event)
    counts = {"enter": 0, "exit": 0}
    for events in events_by_canonical.values():
        if len(set(events)) > 1:
            for event in set(events):
                counts[event] += 1
        else:
            for event in events:
                counts[event] += 1
    return counts


def canonical_group_reconciliation(created_track_ids, aliases, final_decisions, legacy_counts):
    grouped = {}
    for track_id in sorted(created_track_ids):
        canonical_id = canonical_track_id(track_id, aliases)[0]
        grouped.setdefault(canonical_id, []).append(track_id)

    canonical_counts = {"enter": 0, "exit": 0}
    groups = []
    duplicate_groups = 0
    conflict_groups = 0
    duplicate_legacy_units = 0
    for canonical_id, member_ids in sorted(grouped.items()):
        contributions = [
            {"track_id": track_id, "event": final_decisions[track_id]}
            for track_id in member_ids
            if track_id in final_decisions
        ]
        events = sorted({item["event"] for item in contributions})
        if len(events) > 1:
            status = "alias_conflict"
            canonical_event = None
            canonical_included_track_id = None
            conflict_groups += 1
        elif len(contributions) > 1:
            status = "legacy_alias_duplicates"
            canonical_event = events[0]
            canonical_included_track_id = canonical_id if canonical_id in final_decisions else contributions[0]["track_id"]
            canonical_counts[canonical_event] += 1
            duplicate_groups += 1
            duplicate_legacy_units += len(contributions) - 1
        elif contributions:
            status = "match"
            canonical_event = events[0]
            canonical_included_track_id = contributions[0]["track_id"]
            canonical_counts[canonical_event] += 1
        else:
            status = "match"
            canonical_event = None
            canonical_included_track_id = None
        groups.append({
            "canonical_track_id": canonical_id,
            "member_track_ids": member_ids,
            "legacy_contributions": contributions,
            "legacy_unit_count": len(contributions),
            "canonical_event": canonical_event,
            "canonical_included_track_id": canonical_included_track_id,
            "canonical_count_delta": 1 if canonical_event else 0,
            "status": status,
        })

    if conflict_groups:
        status = "alias_conflict"
    elif duplicate_groups:
        status = "legacy_alias_duplicates"
    else:
        status = "match"
    return {
        "status": status,
        "legacy_counts": dict(legacy_counts),
        "canonical_counts": canonical_counts,
        "canonical_minus_legacy": {
            key: canonical_counts[key] - legacy_counts[key] for key in legacy_counts
        },
        "legacy_minus_canonical": {
            key: legacy_counts[key] - canonical_counts[key] for key in legacy_counts
        },
        "canonical_group_count": len(groups),
        "duplicate_group_count": duplicate_groups,
        "duplicate_legacy_unit_count": duplicate_legacy_units,
        "conflict_group_count": conflict_groups,
        "groups": groups,
    }


def _partial_track_rows(state):
    tracker = state.get("tracker")
    observations = state.get("observations", {})
    aliases = state.get("aliases", {})
    online_decisions = state.get("decisions", {})
    online_frames = state.get("online_decision_frames", {})
    rows = []
    if tracker is None:
        return rows
    for track_id in sorted(tracker.created_track_ids):
        track_observations = observations.get(track_id, [])
        canonical_id, alias_chain = canonical_track_id(track_id, aliases)
        record = tracker.track_records.get(track_id, {})
        rows.append({
            "track_id": track_id,
            "canonical_track_id": canonical_id,
            "absorbed_by": aliases.get(track_id),
            "alias_chain": ">".join(str(value) for value in alias_chain),
            "created_frame": record.get("created_frame"),
            "first_frame": track_observations[0]["frame"] if track_observations else None,
            "last_frame": track_observations[-1]["frame"] if track_observations else None,
            "expired_frame": record.get("expired_frame"),
            "lifecycle": "partial_error",
            "termination": record.get("termination", "processing_error"),
            "observation_count": len(track_observations),
            "online_decision": online_decisions.get(track_id),
            "online_decision_frame": online_frames.get(track_id),
            "final_decision": None,
            "decision_changed": None,
            "accepted": False,
            "reason_codes": "processing_error_before_final_evaluation",
        })
    return rows


def persist_partial_artifacts(args, artifacts, state, error):
    """Best-effort partial audit; every write failure is reported without replacing error."""
    if not state.get("processing_started"):
        return {"attempted": False, "persisted": [], "write_errors": []}
    detection_rows = state.get("detection_rows", [])
    track_event_rows = state.get("track_event_rows", [])
    observations = state.get("observations", {})
    aliases = state.get("aliases", {})
    tracker = state.get("tracker")
    frame_index = state.get("frame_index", 0)
    fps = state.get("fps")
    error_payload = {"type": type(error).__name__, "message": str(error)}
    track_event_rows.append({
        "event": "run_error",
        "frame": frame_index,
        "time_seconds": frame_index / fps if fps else None,
        "count_delta": 0,
        "reason": "processing_or_finalization_error",
        "metrics_json": json.dumps(error_payload, sort_keys=True),
    })
    for row in detection_rows:
        assigned_id = row.get("assigned_track_id")
        row["canonical_track_id"] = canonical_track_id(assigned_id, aliases)[0] if assigned_id is not None else None
    for row in track_event_rows:
        track_id = row.get("track_id")
        row["canonical_track_id"] = canonical_track_id(track_id, aliases)[0] if track_id is not None else None
    track_rows = _partial_track_rows(state)
    output_references = {
        "annotated_video": str(args.output.resolve()),
        "history": str(args.history_output.resolve()),
        **{key: str(path.resolve()) for key, path in artifacts.items()},
    }
    history_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "complete": False,
        "video": str(args.video.resolve()),
        "door_polygon": str(state["polygon_path"].resolve()) if state.get("polygon_path") else None,
        "config": normalized_config(args),
        "video_metadata": {
            "fps": fps,
            "width": state.get("width"),
            "height": state.get("height"),
            "source_frame_count": state.get("source_frame_count"),
            "processed_frame_count": frame_index,
            "processed_duration_seconds": frame_index / fps if fps else 0.0,
        },
        "error": error_payload,
        "counts": None,
        "events": {},
        "aliases": {str(track_id): survivor_id for track_id, survivor_id in sorted(aliases.items())},
        "tracks": {
            str(track_id): observations.get(track_id, [])
            for track_id in sorted(tracker.created_track_ids if tracker is not None else observations)
        },
        "online_counts": dict(state.get("counts", {"enter": 0, "exit": 0})),
        "online_events": {
            str(track_id): event for track_id, event in sorted(state.get("decisions", {}).items())
        },
        "outputs": output_references,
        "record_counts": {
            "detections": len(detection_rows),
            "track_events": len(track_event_rows),
            "tracks": len(track_rows),
            "aliases": len(aliases),
        },
        "reconciliation": {
            "status": "incomplete",
            "note": "No completed legacy or canonical count is asserted for partial artifacts.",
        },
    }
    writes = [
        ("detections", lambda: write_csv(artifacts["detections"], DETECTION_FIELDS, detection_rows, compressed=True)),
        ("track_events", lambda: write_csv(artifacts["track_events"], TRACK_EVENT_FIELDS, track_event_rows, compressed=True)),
        ("tracks", lambda: write_csv(artifacts["tracks"], TRACK_FIELDS, track_rows)),
        ("history", lambda: atomic_write_json(args.history_output, history_payload)),
    ]
    persisted = []
    write_errors = []
    for name, writer in writes:
        try:
            writer()
            persisted.append(name)
        except Exception as write_error:
            write_errors.append({"artifact": name, "type": type(write_error).__name__, "message": str(write_error)})
    return {"attempted": True, "persisted": persisted, "write_errors": write_errors}


def run(args, runtime=None):
    artifacts = resolve_artifact_paths(args)
    configured_outputs = [Path(args.output), Path(args.history_output), *artifacts.values()]
    if len({str(path.resolve()) for path in configured_outputs}) != len(configured_outputs):
        raise ValueError("Output, history, and sidecar paths must be distinct")
    started = time.perf_counter()
    metrics_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "complete": False,
        "error": None,
        "video": str(Path(args.video).resolve()),
        "config": normalized_config(args),
        "outputs": {
            "annotated_video": str(Path(args.output).resolve()),
            "history": str(Path(args.history_output).resolve()),
            **{key: str(path.resolve()) for key, path in artifacts.items()},
        },
        "snapshots": {"start": process_snapshot()},
    }
    atomic_write_json(artifacts["metrics"], metrics_payload)
    partial_state = {}
    try:
        return _run(args, runtime, artifacts, metrics_payload, partial_state)
    except Exception as error:
        try:
            partial_result = persist_partial_artifacts(args, artifacts, partial_state, error)
        except Exception as persistence_error:
            partial_result = {
                "attempted": bool(partial_state.get("processing_started")),
                "persisted": [],
                "write_errors": [{
                    "artifact": "partial_artifact_preparation",
                    "type": type(persistence_error).__name__,
                    "message": str(persistence_error),
                }],
            }
        if partial_state.get("cleanup_errors"):
            partial_result["cleanup_errors"] = partial_state["cleanup_errors"]
        metrics_payload.update({
            "status": "error",
            "complete": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "processed_frame_count": partial_state.get("frame_index", 0),
            "partial_artifacts": partial_result,
        })
        metrics_payload.setdefault("snapshots", {})["error"] = process_snapshot()
        try:
            atomic_write_json(artifacts["metrics"], metrics_payload)
        except Exception as metrics_error:
            metrics_payload.setdefault("partial_artifacts", {}).setdefault("write_errors", []).append({
                "artifact": "metrics",
                "type": type(metrics_error).__name__,
                "message": str(metrics_error),
            })
        raise


def _run(args, runtime, artifacts, metrics_payload, partial_state):
    polygon_path = args.door_polygon or args.polygon_dir / f"{args.video.stem}.json"
    partial_state.update({
        "processing_started": False,
        "polygon_path": polygon_path,
        "frame_index": 0,
        "detection_rows": [],
        "track_event_rows": [],
        "observations": {},
        "decisions": {},
        "aliases": {},
        "counts": {"enter": 0, "exit": 0},
        "online_decision_frames": {},
        "tracker": None,
        "fps": None,
    })
    validate(args, polygon_path)
    door = DoorGeometry(polygon_path, args.lane_margin, args.completion_margin)
    runtime = runtime or InferenceRuntime(args)
    device = runtime.device
    model = runtime.model
    non_max_suppression = runtime.non_max_suppression
    post_process_batch = runtime.post_process_batch
    metrics_payload["snapshots"]["runtime_ready"] = process_snapshot(device)

    dataset = runtime.load_images(str(args.video), img_size=runtime.image_size, stride=int(model.stride.max()), auto=True)
    capture = dataset.cap
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    partial_state.update({
        "fps": fps,
        "width": width,
        "height": height,
        "source_frame_count": source_frame_count,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output: {args.output}")

    if getattr(args, "tracker", "appearance") == "bytetrack":
        tracker = MotionTracker(fps, (width, height), args.bytetrack_track_thresh, args.bytetrack_match_thresh, args.bytetrack_track_buffer, args.confidence)
    else:
        tracker = AppearanceTracker(
            runtime.extractor,
            args.reid_distance,
            args.match_center_distance,
            args.appearance_max_missing,
            args.tracker_max_missing,
            door,
            args.reconnect_iou,
            args.reconnect_bottom_distance,
            args.duplicate_merge_iou,
            args.duplicate_merge_bottom_distance,
            args.duplicate_merge_progress_distance,
            args.duplicate_merge_min_frames,
            args.duplicate_merge_reid_distance,
            appearance_sample_count=args.appearance_sample_count,
        )
    evaluate = make_track_evaluator(args, door)
    metrics_payload["config"] = normalized_config(args)
    observations, decisions, aliases, counts = {}, {}, {}, {"enter": 0, "exit": 0}
    online_decision_frames = {}
    detection_rows, track_event_rows = [], []
    partial_state.update({
        "tracker": tracker,
        "observations": observations,
        "decisions": decisions,
        "aliases": aliases,
        "counts": counts,
        "online_decision_frames": online_decision_frames,
        "detection_rows": detection_rows,
        "track_event_rows": track_event_rows,
    })
    timings = {"decode_preprocess": 0.0, "transfer": 0.0, "bpjdet": 0.0, "nms": 0.0, "postprocess": 0.0, "reid": 0.0, "tracker": 0.0, "count_draw_write": 0.0}

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def record_online_decision(track_id, event, frame, reason):
        decisions[track_id] = event
        online_decision_frames[track_id] = frame
        counts[event] += 1
        track_event_rows.append({
            "event": "online_decision", "frame": frame, "time_seconds": frame / fps, "track_id": track_id,
            "decision": event, "count_delta": 1, "reason": reason,
        })

    frame_index = 0
    partial_state["processing_started"] = True
    processing_started = time.perf_counter()
    try:
        dataset_iterator = iter(dataset)
        while not args.max_frames or frame_index < args.max_frames:
            batch_limit = args.batch_size
            if args.max_frames:
                batch_limit = min(batch_limit, args.max_frames - frame_index)
            stage_started = time.perf_counter()
            batch_items = list(islice(dataset_iterator, batch_limit))
            if not batch_items:
                break
            images = np.stack([item[1] for item in batch_items])
            timings["decode_preprocess"] += time.perf_counter() - stage_started

            synchronize()
            stage_started = time.perf_counter()
            tensor = torch.from_numpy(images).to(device).float() / 255.0
            synchronize()
            timings["transfer"] += time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            with torch.inference_mode():
                predictions = model(tensor, augment=False, scales=[1])[0]
            synchronize()
            timings["bpjdet"] += time.perf_counter() - stage_started

            data = {"dataset": "CrowdHuman", "num_offsets": 2, "part_type": "face", "match_iou_thres": args.match_iou}
            stage_started = time.perf_counter()
            batch_bodies = non_max_suppression(predictions, args.confidence, args.iou, classes=[0], num_offsets=2)
            batch_faces = non_max_suppression(predictions, args.confidence, args.iou, classes=[1], num_offsets=2)
            synchronize()
            timings["nms"] += time.perf_counter() - stage_started

            processed_frames = []
            for batch_index, (_, _, frame, _) in enumerate(batch_items):
                stage_started = time.perf_counter()
                boxes, points, scores, _, _, _ = post_process_batch(
                    data,
                    tensor[batch_index:batch_index + 1],
                    [],
                    [[frame.shape[:2]]],
                    [batch_bodies[batch_index]],
                    [batch_faces[batch_index]],
                )
                boxes, scores = boxes or [], scores or []
                # BPJDet regresses a head/face centre for every body; unlike the body box it is not
                # clamped to the frame for people close to the camera.
                heads = [[float(point[0][0]), float(point[0][1])] for point in (points or [])] if len(points or []) == len(boxes) else [None] * len(boxes)
                timings["postprocess"] += time.perf_counter() - stage_started
                processed_frames.append((frame, boxes, scores, heads))

            synchronize()
            stage_started = time.perf_counter()
            batch_embeddings = tracker.extract_embeddings_batch([(frame, boxes) for frame, boxes, _, _ in processed_frames])
            synchronize()
            timings["reid"] += time.perf_counter() - stage_started

            for (frame, boxes, scores, heads), embeddings in zip(processed_frames, batch_embeddings):
                stage_started = time.perf_counter()
                if isinstance(tracker, MotionTracker):
                    track_ids, merged_tracks = tracker.update_with_embeddings(boxes, embeddings, frame_index=frame_index, scores=scores)
                    track_boxes = [smoothed if smoothed is not None else box for box, smoothed in zip(boxes, tracker.smoothed_boxes)]
                else:
                    track_ids, merged_tracks = tracker.update_with_embeddings(boxes, embeddings, frame_index=frame_index)
                    track_boxes = boxes
                timings["tracker"] += time.perf_counter() - stage_started
                diagnostics = tracker.last_update_diagnostics
                for event_row in diagnostics["events"]:
                    row = dict(event_row)
                    row.setdefault("time_seconds", frame_index / fps)
                    track_event_rows.append(row)
                for detection_index, (box, score, head, diagnostic) in enumerate(zip(boxes, scores, heads, diagnostics["detections"])):
                    center = box_center(box)
                    bottom = AppearanceTracker.bottom_center(box)
                    center_progress, center_lateral = door.coordinates(center)
                    bottom_progress, bottom_lateral = door.coordinates(bottom)
                    detection_rows.append({
                        "frame": frame_index,
                        "time_seconds": frame_index / fps,
                        "detection_index": detection_index,
                        "score": float(score),
                        "bbox_x1": float(box[0]), "bbox_y1": float(box[1]), "bbox_x2": float(box[2]), "bbox_y2": float(box[3]),
                        "center_x": float(center[0]), "center_y": float(center[1]),
                        "bottom_x": float(bottom[0]), "bottom_y": float(bottom[1]),
                        "head_x": None if head is None else head[0], "head_y": None if head is None else head[1],
                        "center_progress": center_progress, "center_lateral": center_lateral, "center_region": door.region(center),
                        "bottom_progress": bottom_progress, "bottom_lateral": bottom_lateral, "bottom_region": door.region(bottom),
                        "inside_door": door.contains(center),
                        **diagnostic,
                    })

                stage_started = time.perf_counter()
                for track_id in tracker.created_track_ids:
                    observations.setdefault(track_id, [])
                for survivor_id, absorbed_id in merged_tracks:
                    aliases[absorbed_id] = survivor_id
                    observations.setdefault(survivor_id, []).extend(observations.get(absorbed_id, []))
                    best_observation_by_frame = {}
                    for observation in observations[survivor_id]:
                        observation_frame = observation["frame"]
                        previous = best_observation_by_frame.get(observation_frame)
                        if previous is None or AppearanceTracker.box_area(observation["bbox"]) > AppearanceTracker.box_area(previous["bbox"]):
                            best_observation_by_frame[observation_frame] = observation
                    observations[survivor_id] = [best_observation_by_frame[item_frame] for item_frame in sorted(best_observation_by_frame)]
                    absorbed_event = decisions.pop(absorbed_id, None)
                    if absorbed_event:
                        if survivor_id in decisions:
                            counts[absorbed_event] -= 1
                            track_event_rows.append({
                                "event": "online_decision_reconciled", "frame": frame_index, "time_seconds": frame_index / fps,
                                "track_id": survivor_id, "other_track_id": absorbed_id, "decision": absorbed_event,
                                "count_delta": -1, "reason": "absorbed_alias_had_duplicate_online_decision",
                            })
                        else:
                            decisions[survivor_id] = absorbed_event
                            online_decision_frames.setdefault(survivor_id, online_decision_frames.get(absorbed_id, frame_index))
                            track_event_rows.append({
                                "event": "online_decision_reassigned", "frame": frame_index, "time_seconds": frame_index / fps,
                                "track_id": survivor_id, "other_track_id": absorbed_id, "decision": absorbed_event,
                                "count_delta": 0, "reason": "absorbed_alias_decision_transferred_to_survivor",
                            })

                visible = []
                for box, score, head, track_id in zip(track_boxes, scores, heads, track_ids):
                    if track_id is None:
                        continue
                    track_history = observations.setdefault(track_id, [])
                    track_history.append({"frame": frame_index, "bbox": [float(value) for value in box], "center": box_center(box), "head": head})
                    event = decisions.get(track_id)
                    if event is None:
                        evaluation = evaluate(track_history, 0)
                        if evaluation["decision"]:
                            record_online_decision(track_id, evaluation["decision"], frame_index, evaluation["reason_codes"][0])
                    visible.append((track_id, box, float(score), decisions.get(track_id)))

                for track_id, track in tracker.tracks.items():
                    if track_id not in observations or track_id in decisions:
                        continue
                    evaluation = evaluate(observations[track_id], track["missing"])
                    if evaluation["decision"]:
                        record_online_decision(track_id, evaluation["decision"], frame_index, evaluation["reason_codes"][0])

                writer.write(draw(frame, door, visible, counts))
                timings["count_draw_write"] += time.perf_counter() - stage_started
                frame_index += 1
                partial_state["frame_index"] = frame_index
                if frame_index % 250 == 0:
                    print(f"Processed frames: {frame_index}; entered={counts['enter']}; exited={counts['exit']}", flush=True)
    finally:
        cleanup_errors = []
        for resource_name, release in (("capture", capture.release), ("writer", writer.release)):
            try:
                release()
            except Exception as cleanup_error:
                cleanup_errors.append({
                    "resource": resource_name,
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                })
        if cleanup_errors:
            partial_state["cleanup_errors"] = cleanup_errors

    processing_seconds = time.perf_counter() - processing_started
    online_counts = dict(counts)
    online_decisions = dict(decisions)
    final_decisions, final_evaluations = {}, {}
    for track_id in sorted(tracker.created_track_ids):
        evaluation = evaluate(observations.get(track_id, []), args.tracker_max_missing)
        final_evaluations[track_id] = evaluation
        if evaluation["decision"]:
            final_decisions[track_id] = evaluation["decision"]
        last_frame = observations.get(track_id, [{}])[-1].get("frame", frame_index - 1) if observations.get(track_id) else frame_index - 1
        track_event_rows.append({
            "event": "final_decision", "frame": last_frame, "time_seconds": last_frame / fps if last_frame >= 0 else None,
            "track_id": track_id, "decision": evaluation["decision"] or "", "count_delta": 1 if evaluation["decision"] else 0,
            "reason": "|".join(evaluation["reason_codes"]), "metrics_json": json.dumps(evaluation["metrics"], sort_keys=True),
        })
    legacy_counts = {
        "enter": sum(event == "enter" for event in final_decisions.values()),
        "exit": sum(event == "exit" for event in final_decisions.values()),
    }
    counts = canonical_deduplicated_counts(tracker.created_track_ids, aliases, final_decisions)
    canonical_reconciliation = canonical_group_reconciliation(
        tracker.created_track_ids,
        aliases,
        final_decisions,
        legacy_counts,
    )

    for row in detection_rows:
        assigned_id = row.get("assigned_track_id")
        row["canonical_track_id"] = canonical_track_id(assigned_id, aliases)[0] if assigned_id is not None else None
    for row in track_event_rows:
        track_id = row.get("track_id")
        row["canonical_track_id"] = canonical_track_id(track_id, aliases)[0] if track_id is not None else None

    track_rows = []
    evaluation_history = {}
    changed_tracks = []
    for track_id in sorted(tracker.created_track_ids):
        track_observations = observations.get(track_id, [])
        evaluation = final_evaluations[track_id]
        metrics = evaluation["metrics"]
        canonical_id, alias_chain = canonical_track_id(track_id, aliases)
        record = tracker.track_records.get(track_id, {})
        if record.get("termination") == "merged":
            lifecycle = "absorbed"
        elif record.get("termination") == "expired":
            lifecycle = "expired"
        else:
            lifecycle = "active_at_end" if track_id in tracker.tracks else "ended"
        actual_missing = tracker.tracks.get(track_id, {}).get("missing", record.get("expired_frame") is not None and args.tracker_max_missing + 1 or 0)
        online_decision = online_decisions.get(track_id)
        final_decision = final_decisions.get(track_id)
        if online_decision != final_decision:
            changed_tracks.append({"track_id": track_id, "canonical_track_id": canonical_id, "online": online_decision, "final": final_decision})
        row = {
            "track_id": track_id,
            "canonical_track_id": canonical_id,
            "absorbed_by": aliases.get(track_id),
            "alias_chain": ">".join(str(value) for value in alias_chain),
            "created_frame": record.get("created_frame"),
            "first_frame": track_observations[0]["frame"] if track_observations else None,
            "last_frame": track_observations[-1]["frame"] if track_observations else None,
            "expired_frame": record.get("expired_frame"),
            "lifecycle": lifecycle,
            "termination": record.get("termination", "end_of_video"),
            "observation_count": len(track_observations),
            "start_region": metrics["start_region"], "end_region": metrics["end_region"],
            "start_progress": metrics["start_progress"], "end_progress": metrics["end_progress"], "net_progress": metrics["net_progress"],
            "bottom_start_progress": metrics["bottom_start_progress"], "bottom_end_progress": metrics["bottom_end_progress"],
            "bottom_net_progress": metrics["bottom_net_progress"], "was_inside_door": metrics["was_inside_door"],
            "was_in_lane": metrics["was_in_lane"], "finished_near_salon_edge": metrics["finished_near_salon_edge"],
            "finished_at_outer_edge": metrics["finished_at_outer_edge"], "disappeared_at_door": metrics["disappeared_at_door"],
            "missing_frames_at_end": actual_missing,
            "online_decision": online_decision, "online_decision_frame": online_decision_frames.get(track_id),
            "final_decision": final_decision, "decision_changed": online_decision != final_decision,
            "accepted": final_decision is not None, "reason_codes": "|".join(evaluation["reason_codes"]),
            "enter_predicates_json": json.dumps(evaluation["enter_predicates"], sort_keys=True),
            "exit_predicates_json": json.dumps(evaluation["exit_predicates"], sort_keys=True),
            "predicate_metrics_json": json.dumps(metrics, sort_keys=True),
        }
        track_rows.append(row)
        evaluation_history[str(track_id)] = {
            "canonical_track_id": canonical_id,
            "absorbed_by": aliases.get(track_id),
            "online_decision": online_decision,
            "online_decision_frame": online_decision_frames.get(track_id),
            "final_decision": final_decision,
            "reason_codes": evaluation["reason_codes"],
            "metrics": metrics,
            "enter_predicates": evaluation["enter_predicates"],
            "exit_predicates": evaluation["exit_predicates"],
        }

    output_references = {
        "annotated_video": str(args.output.resolve()),
        "history": str(args.history_output.resolve()),
        **{key: str(path.resolve()) for key, path in artifacts.items()},
    }
    write_csv(artifacts["detections"], DETECTION_FIELDS, detection_rows, compressed=True)
    write_csv(artifacts["track_events"], TRACK_EVENT_FIELDS, track_event_rows, compressed=True)
    write_csv(artifacts["tracks"], TRACK_FIELDS, track_rows)

    history = {str(track_id): observations.get(track_id, []) for track_id in sorted(tracker.created_track_ids)}
    history_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "complete": True,
        "video": str(args.video.resolve()),
        "door_polygon": str(polygon_path.resolve()),
        "config": normalized_config(args),
        "video_metadata": {
            "fps": fps, "width": width, "height": height, "source_frame_count": source_frame_count,
            "processed_frame_count": frame_index, "processed_duration_seconds": frame_index / fps if fps else 0.0,
        },
        "counts": counts,
        "events": {str(track_id): event for track_id, event in sorted(final_decisions.items())},
        "aliases": {str(track_id): survivor_id for track_id, survivor_id in sorted(aliases.items())},
        "tracks": history,
        "online_counts": online_counts,
        "online_events": {str(track_id): event for track_id, event in sorted(online_decisions.items())},
        "track_evaluations": evaluation_history,
        "alias_details": [
            {"absorbed_track_id": track_id, "survivor_track_id": survivor_id, "canonical_track_id": canonical_track_id(track_id, aliases)[0]}
            for track_id, survivor_id in sorted(aliases.items())
        ],
        "outputs": output_references,
        "reconciliation": {
            "status": canonical_reconciliation["status"],
            "online_final_status": "match" if online_counts == counts else "different",
            "online_counts": online_counts,
            "final_counts": counts,
            "legacy_counts": canonical_reconciliation["legacy_counts"],
            "count_delta": {key: counts[key] - online_counts[key] for key in counts},
            "changed_tracks": changed_tracks,
            "canonical_counts": canonical_reconciliation["canonical_counts"],
            "canonical_minus_legacy": canonical_reconciliation["canonical_minus_legacy"],
            "legacy_minus_canonical": canonical_reconciliation["legacy_minus_canonical"],
            "canonical_group_count": canonical_reconciliation["canonical_group_count"],
            "duplicate_group_count": canonical_reconciliation["duplicate_group_count"],
            "duplicate_legacy_unit_count": canonical_reconciliation["duplicate_legacy_unit_count"],
            "conflict_group_count": canonical_reconciliation["conflict_group_count"],
            "canonical_groups": canonical_reconciliation["groups"],
            "note": (
                "final_counts collapse only CONFLICTING canonical alias groups (mixed enter/exit "
                "evidence, e.g. one entity oscillating near the door) to at most one enter and one "
                "exit. Groups where every absorbed alias fragment agrees on one event type are left "
                "as-is, since that pattern typically reflects several distinct, similarly dressed "
                "people who queued close together and were merged by mistake, each independently "
                "satisfying the full crossing criteria. legacy_counts is the pre-deduplication sum "
                "over every created ID, kept for audit. canonical_counts is a stricter diagnostic that "
                "additionally excludes alias groups with conflicting evidence entirely."
            ),
        },
    }
    atomic_write_json(args.history_output, history_payload)

    measured_seconds = sum(timings.values())
    metrics_payload.update({
        "status": "success",
        "complete": True,
        "error": None,
        "video_metadata": history_payload["video_metadata"],
        "counts": {
            "online": online_counts,
            "final": counts,
            "legacy_final": canonical_reconciliation["legacy_counts"],
            "canonical_diagnostic": canonical_reconciliation["canonical_counts"],
        },
        "reconciliation": {
            key: canonical_reconciliation[key]
            for key in (
                "status", "canonical_counts", "canonical_minus_legacy", "legacy_minus_canonical",
                "canonical_group_count", "duplicate_group_count", "duplicate_legacy_unit_count",
                "conflict_group_count",
            )
        },
        "record_counts": {"detections": len(detection_rows), "track_events": len(track_event_rows), "tracks": len(track_rows), "aliases": len(aliases)},
        "timings": {
            "processing_seconds": round(processing_seconds, 6),
            "stages": {name: {"seconds": round(seconds, 6), "percent": round(100.0 * seconds / measured_seconds, 3) if measured_seconds else 0.0} for name, seconds in timings.items()},
        },
        "throughput": {"frames_per_second": frame_index / processing_seconds if processing_seconds else 0.0, "frames": frame_index, "batch_size": args.batch_size},
        "outputs": output_references,
    })
    metrics_payload["snapshots"]["end"] = process_snapshot(device)
    atomic_write_json(artifacts["metrics"], metrics_payload)

    if args.profile:
        print("PROFILE " + json.dumps(metrics_payload["timings"], sort_keys=True))
    print(f"Done. Entered: {counts['enter']}; exited: {counts['exit']}")
    print(f"Video: {args.output}")
    print(f"History: {args.history_output}")
    return runtime, history_payload


def run_jobs(args):
    jobs = json.loads(args.jobs_file.read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("--jobs-file must contain a non-empty JSON list")
    runtime = None
    artifact_keys = {
        "detections_output": "detections_output",
        "track_events_output": "track_events_output",
        "tracks_output": "tracks_output",
        "metrics_output": "metrics_output",
    }
    for job_index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise ValueError(f"Job {job_index} must be a JSON object")
        missing = [key for key in ("video", "output", "history_output") if key not in job]
        if missing:
            raise ValueError(f"Job {job_index} is missing required paths: {', '.join(missing)}")
        job_args = copy.copy(args)
        job_args.video = Path(job["video"])
        job_args.output = Path(job["output"])
        job_args.history_output = Path(job["history_output"])
        for manifest_key, argument_name in artifact_keys.items():
            value = job.get(manifest_key)
            setattr(job_args, argument_name, Path(value) if value else None)
        runtime, _ = run(job_args, runtime)


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.jobs_file:
        run_jobs(parsed_args)
    else:
        run(parsed_args)