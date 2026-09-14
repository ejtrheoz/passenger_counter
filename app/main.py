import asyncio
import json
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import pipelines, settings

app = FastAPI(
    title="Passenger Counter",
    version="0.3.0",
    description=(
        "GPU-only detection service: analyzes one already-cut activity segment clip per request "
        "(door-flow entered/exited counting via `/analyze/people`, or ksiva-presentation counting via "
        "`/analyze/ksiva`). Splitting a source video into activity-interval segments and aggregating "
        "per-segment results across a whole video is owned by passenger_aggregator, which calls this "
        "service once per segment."
    ),
)

# The pipelines spawn GPU subprocesses; only one clip is processed at a time.
_job_lock = asyncio.Lock()


class PeopleResult(BaseModel):
    entered: int = Field(..., description="People counted entering in this clip.")
    exited: int = Field(..., description="People counted exiting in this clip.")
    door_polygon: str
    processing_seconds: float


class KsivaResult(BaseModel):
    unique_ksiva: int = Field(..., description="Unique ksiva presentations counted in this clip.")
    source_detections: int | None = None
    filtered_detections: int | None = None
    rejected_tracks: int | None = None
    flagged_for_review: int | None = None
    processing_seconds: float


def _resolve_polygon(name: str | None, inline_json: str | None, job_dir: Path) -> Path:
    """Either a pre-registered polygon asset (by name) or an inline JSON polygon supplied by the caller."""
    if inline_json is not None:
        try:
            payload = json.loads(inline_json)
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=400, detail=f"door_polygon_json is not valid JSON: {error}") from error
        points = payload.get("points") if isinstance(payload, dict) else None
        if not isinstance(points, list) or len(points) < 3:
            raise HTTPException(status_code=400, detail="door_polygon_json must be an object with a 'points' list of at least 3 [x, y] pairs")
        path = job_dir / "door_polygon.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path
    name = name or settings.DEFAULT_DOOR_POLYGON
    if Path(name).name != name or not name.endswith(".json"):
        raise HTTPException(status_code=400, detail="door_polygon must be a plain file name ending in .json")
    path = settings.DOOR_POLYGON_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Door polygon not found: {name}")
    return path


async def _save_upload(video: UploadFile, job_dir: Path) -> Path:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported video type {suffix!r}; allowed: {sorted(settings.ALLOWED_VIDEO_SUFFIXES)}")
    destination = job_dir / f"input{suffix}"
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with destination.open("wb") as output:
        while chunk := await video.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                raise HTTPException(status_code=413, detail=f"Video exceeds {settings.MAX_UPLOAD_MB} MB")
            output.write(chunk)
    if written == 0:
        raise HTTPException(status_code=400, detail="Empty upload")
    pipelines.log(video.filename, f"received {written / (1024 * 1024):.1f} MB")
    return destination


async def _process(video: UploadFile, run):
    settings.WORK_DIR.mkdir(parents=True, exist_ok=True)
    job_dir = settings.WORK_DIR / uuid.uuid4().hex
    job_dir.mkdir()
    label = video.filename or job_dir.name
    try:
        video_path = await _save_upload(video, job_dir)
        if _job_lock.locked():
            pipelines.log(label, "waiting for the previous clip to finish")
        async with _job_lock:
            result = await asyncio.to_thread(run, video_path, job_dir, label)
        return result
    except pipelines.PipelineError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        if not settings.KEEP_ARTIFACTS:
            shutil.rmtree(job_dir, ignore_errors=True)


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/health")
def health():
    checks = {
        "door_flow_counter_script": settings.DOOR_FLOW_COUNTER_SCRIPT.is_file(),
        "ksiva_segment_script": settings.KSIVA_SEGMENT_SCRIPT.is_file(),
        "bpjdet_repo": (settings.BPJDET_REPO / "models" / "experimental.py").is_file(),
        "bpjdet_weights": settings.BPJDET_WEIGHTS.is_file(),
        "ksiva_weights": settings.KSIVA_WEIGHTS.is_file(),
        "default_door_polygon": (settings.DOOR_POLYGON_DIR / settings.DEFAULT_DOOR_POLYGON).is_file(),
    }
    gpu = None
    if settings.DEVICE != "cpu" and shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, timeout=10, check=True).stdout.strip().splitlines()
        except (subprocess.SubprocessError, OSError):
            gpu = None
    return {
        "status": "ok" if all(checks.values()) else "degraded",
        "checks": checks,
        "device": settings.DEVICE,
        "gpu": gpu,
        "door_polygons": sorted(path.name for path in settings.DOOR_POLYGON_DIR.glob("*.json")) if settings.DOOR_POLYGON_DIR.is_dir() else [],
    }


@app.post(
    "/analyze/people",
    response_model=PeopleResult,
    summary="Count entered/exited people in one already-cut activity segment clip",
    description="Runs BPJDet + tracking on the uploaded clip only (bounded time/GPU work per call). Provide "
    "either `door_polygon` (name of a pre-registered polygon asset) or `door_polygon_json` (an inline "
    "`{\"points\": [[x, y], ...]}` polygon, e.g. one drawn by the caller on the video's first frame); "
    "`door_polygon_json` takes precedence if both are given. The caller (passenger_aggregator) is responsible "
    "for splitting the source video into activity-interval clips and summing results across clips.",
)
async def analyze_people(
    video: UploadFile = File(...),
    door_polygon: str | None = Form(None),
    door_polygon_json: str | None = Form(None),
):
    def run(path, job_dir, label):
        polygon_path = _resolve_polygon(door_polygon, door_polygon_json, job_dir)
        return pipelines.analyze_people_segment(path, polygon_path, job_dir / "result", label)

    result = await _process(video, run)
    return PeopleResult(**result)


@app.post(
    "/analyze/ksiva",
    response_model=KsivaResult,
    summary="Count unique ksiva presentations in one already-cut activity segment clip",
    description="Runs YOLO detection + spatial tracking/filtering on the uploaded clip only (bounded time/GPU "
    "work per call). The caller (passenger_aggregator) is responsible for splitting the source video into "
    "activity-interval clips and summing results across clips.",
)
async def analyze_ksiva(video: UploadFile = File(...)):
    result = await _process(video, lambda path, job_dir, label: pipelines.analyze_ksiva_segment(path, job_dir / "result", label))
    return KsivaResult(**result)
