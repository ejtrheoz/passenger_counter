import asyncio
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import pipelines, settings

app = FastAPI(title="Passenger Counter", version="0.1.0")

# The pipelines spawn GPU subprocesses; only one video is processed at a time.
_job_lock = asyncio.Lock()


def _resolve_polygon(name: str | None) -> Path:
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
            pipelines.log(label, "waiting for the previous video to finish")
        async with _job_lock:
            result = await asyncio.to_thread(run, video_path, job_dir, label)
        return {"video": video.filename, **result}
    except pipelines.PipelineError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        if not settings.KEEP_ARTIFACTS:
            shutil.rmtree(job_dir, ignore_errors=True)


@app.get("/health")
def health():
    checks = {
        "door_flow_script": settings.DOOR_FLOW_SCRIPT.is_file(),
        "ksiva_script": settings.KSIVA_SCRIPT.is_file(),
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


@app.post("/count/people")
async def count_people(video: UploadFile = File(...), door_polygon: str | None = Form(None)):
    polygon = _resolve_polygon(door_polygon)
    return await _process(video, lambda path, job_dir, label: pipelines.count_people(path, polygon, job_dir / "door_flow", label))


@app.post("/count/ksiva")
async def count_ksiva(video: UploadFile = File(...)):
    return await _process(video, lambda path, job_dir, label: pipelines.count_ksiva(path, job_dir / "ksiva", label))


@app.post("/count/all")
async def count_all(video: UploadFile = File(...), door_polygon: str | None = Form(None)):
    polygon = _resolve_polygon(door_polygon)

    def run(path, job_dir, label):
        return {
            "people": pipelines.count_people(path, polygon, job_dir / "door_flow", label),
            "ksiva": pipelines.count_ksiva(path, job_dir / "ksiva", label),
        }

    return await _process(video, run)
