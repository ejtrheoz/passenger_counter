import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", REPO_ROOT / "assets")).resolve()

DOOR_FLOW_DIR = REPO_ROOT / "door-flow"
KSIVA_DIR = REPO_ROOT / "ksiva"
DOOR_FLOW_COUNTER_SCRIPT = DOOR_FLOW_DIR / "door_flow_counter.py"
KSIVA_SEGMENT_SCRIPT = KSIVA_DIR / "analyze_segment.py"

BPJDET_REPO = Path(os.environ.get("BPJDET_REPO", REPO_ROOT / "third_party" / "BPJDet"))
BPJDET_WEIGHTS = Path(os.environ.get("BPJDET_WEIGHTS", ASSETS_DIR / "models" / "bpjdet" / "ch_face_s_1536_e150_best_mMR.pt"))
KSIVA_WEIGHTS = Path(os.environ.get("KSIVA_WEIGHTS", ASSETS_DIR / "models" / "ksiva" / "best.pt"))
DOOR_POLYGON_DIR = Path(os.environ.get("DOOR_POLYGON_DIR", ASSETS_DIR / "door_polygons"))
DEFAULT_DOOR_POLYGON = os.environ.get("DEFAULT_DOOR_POLYGON", "2.json")

# "0", "0,1" or "cpu"; single-clip analysis always uses the first listed device.
DEVICE = os.environ.get("DEVICE", "0")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/passenger_counter"))
KEEP_ARTIFACTS = os.environ.get("KEEP_ARTIFACTS", "0") == "1"
JOB_TIMEOUT_SECONDS = int(os.environ.get("JOB_TIMEOUT_SECONDS", "3600"))
# Also echo the raw stdout of the aggregate scripts into the server log.
VERBOSE_PIPELINE_LOGS = os.environ.get("VERBOSE_PIPELINE_LOGS", "0") == "1"
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "2048"))
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".h264"}
