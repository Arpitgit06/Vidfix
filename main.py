"""
main.py – AV-SynthRestore 3D
FastAPI backend: REST API for job management + WebSocket for live telemetry.

Endpoints
---------
GET  /                        → service info
GET  /health                  → liveness probe
POST /api/upload              → upload video file, returns job_id
POST /api/process/{job_id}    → start pipeline (async background task)
GET  /api/jobs/{job_id}       → job status
GET  /api/download/{job_id}   → download restored output
GET  /api/config              → read config.json
PUT  /api/config              → write config.json
WS   /ws/{job_id}             → real-time telemetry stream
"""

from __future__ import annotations

import os
import sys

# Ensure project root directory is in PATH so subprocesses can find local ffmpeg/ffprobe binaries
_root_dir = os.path.abspath(os.path.dirname(__file__))
if _root_dir not in os.environ["PATH"]:
    os.environ["PATH"] = _root_dir + os.pathsep + os.environ.get("PATH", "")

import asyncio
import json
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import ProcessingPipeline

class ProcessConfig(BaseModel):
    audio_mode: str = "gap_fill"
    script_text: str = ""


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

_SUPPORTED_EXTS   = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
_UPLOAD_CHUNK_SZ  = 1024 * 1024   # 1 MB per read chunk
_WS_KEEPALIVE_S   = 25            # seconds between server-side pings


# ── WebSocket Connection Manager ───────────────────────────────────────────────

class ConnectionManager:
    """
    Thread-safe registry of active WebSocket connections keyed by job_id.

    The `_lock` ensures that concurrent asyncio tasks (pipeline telemetry +
    keepalive loop) don't race on the connection dict.
    """

    def __init__(self) -> None:
        self._conns: Dict[str, WebSocket] = {}
        self._lock  = asyncio.Lock()

    async def connect(self, job_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._conns[job_id] = ws
        logger.info("WS connected  : job=%s", job_id)

    async def disconnect(self, job_id: str) -> None:
        async with self._lock:
            self._conns.pop(job_id, None)
        logger.info("WS disconnected: job=%s", job_id)

    async def send_telemetry(self, job_id: str, data: dict) -> None:
        async with self._lock:
            ws = self._conns.get(job_id)
        if ws is None:
            return
        try:
            await ws.send_json(data)
        except Exception as exc:
            logger.warning("Telemetry send failed for %s: %s", job_id, exc)
            await self.disconnect(job_id)

    def is_connected(self, job_id: str) -> bool:
        return job_id in self._conns

    def active_count(self) -> int:
        return len(self._conns)


# ── App Bootstrap ──────────────────────────────────────────────────────────────

manager = ConnectionManager()

# In-memory job registry {job_id: {"status": str, "filename": str, ...}}
_job_registry: Dict[str, dict] = {}


async def _cleanup_old_jobs() -> None:
    """Background task to delete jobs and outputs older than job_cleanup_hours."""
    while True:
        try:
            root_dir = Path(__file__).parent
            cfg_path = root_dir / "config.json"
            cleanup_hours = 24
            if cfg_path.exists():
                try:
                    with open(cfg_path, encoding="utf-8") as fh:
                        cfg = json.load(fh)
                        cleanup_hours = cfg.get("system", {}).get("job_cleanup_hours", 24)
                except Exception:
                    pass

            jobs_dir = root_dir / "jobs"
            output_dir = root_dir / "restored_output"
            now = time.time()
            max_age_sec = cleanup_hours * 3600

            # Clean jobs
            if jobs_dir.exists():
                for item in jobs_dir.iterdir():
                    if item.is_dir():
                        mtime = item.stat().st_mtime
                        if (now - mtime) > max_age_sec:
                            logger.info("Cleaning up old job directory: %s", item.name)
                            shutil.rmtree(item, ignore_errors=True)
                            _job_registry.pop(item.name, None)

            # Clean outputs
            if output_dir.exists():
                for item in output_dir.iterdir():
                    if item.is_file():
                        mtime = item.stat().st_mtime
                        if (now - mtime) > max_age_sec:
                            logger.info("Cleaning up old output file: %s", item.name)
                            item.unlink(missing_ok=True)

        except Exception as exc:
            logger.warning("Error in cleanup task: %s", exc)

        # Run every hour
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure directories exist before serving
    root_dir = Path(__file__).parent
    for d in (root_dir / "jobs", root_dir / "restored_output"):
        d.mkdir(parents=True, exist_ok=True)
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(_cleanup_old_jobs())
    logger.info("AV-SynthRestore 3D backend ready with job cleanup task.")
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("AV-SynthRestore 3D backend shutting down.")


app = FastAPI(
    title       = "AV-SynthRestore 3D",
    version     = "1.0.0",
    description = "AI-powered audio/video restoration pipeline with real-time telemetry",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
    expose_headers = ["Content-Disposition"],
    allow_credentials = False,
)

# Serve compiled frontend from root folder
_frontend_dir = Path(__file__).parent
if _frontend_dir.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
    logger.info("Frontend mounted at /ui from %s", _frontend_dir)


# ── REST Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
async def root():
    return {
        "service":    "AV-SynthRestore 3D",
        "version":    "1.0.0",
        "active_ws":  manager.active_count(),
        "jobs":       len(_job_registry),
    }


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


@app.post("/api/upload", tags=["jobs"])
async def upload_file(
    file: UploadFile = File(...),
    ref_audio: Optional[UploadFile] = File(None)
):
    """
    Accept a video upload.

    - Validates extension.
    - Writes to ``./jobs/<job_id>/<filename>`` in 1 MB chunks.
    - Returns job_id for subsequent API calls.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _SUPPORTED_EXTS:
        raise HTTPException(
            status_code = 400,
            detail      = (
                f"Unsupported file type '{ext}'. "
                f"Accepted: {', '.join(sorted(_SUPPORTED_EXTS))}"
            ),
        )

    job_id  = str(uuid.uuid4())
    root_dir = Path(__file__).parent
    job_dir = root_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    dest = job_dir / (file.filename or f"input{ext}")
    try:
        with open(dest, "wb") as fh:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_SZ)
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Write failed: {exc}") from exc

    if ref_audio:
        ref_ext = Path(ref_audio.filename or "").suffix.lower()
        ref_dest = job_dir / f"ref_audio{ref_ext}"
        try:
            with open(ref_dest, "wb") as fh:
                while True:
                    chunk = await ref_audio.read(_UPLOAD_CHUNK_SZ)
                    if not chunk:
                        break
                    fh.write(chunk)
        except Exception as exc:
            logger.warning("Failed to save ref_audio: %s", exc)

    size_mb = dest.stat().st_size / (1024 ** 2)
    logger.info("Uploaded: %s (%.1f MB) → job=%s", file.filename, size_mb, job_id)

    _job_registry[job_id] = {
        "status":   "uploaded",
        "filename": file.filename,
        "size_mb":  round(size_mb, 2),
    }

    return {
        "job_id":   job_id,
        "filename": file.filename,
        "size_mb":  round(size_mb, 2),
        "status":   "uploaded",
    }


@app.post("/api/process/{job_id}", tags=["jobs"])
async def start_processing(job_id: str, background_tasks: BackgroundTasks, config: ProcessConfig):
    """
    Launch the restoration pipeline for a previously uploaded job.
    Processing runs in a FastAPI background task so the response is immediate.
    Progress is streamed via WebSocket ``/ws/{job_id}``.
    """
    root_dir = Path(__file__).parent
    job_dir = root_dir / "jobs" / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    entry = _job_registry.setdefault(job_id, {})
    if entry.get("status") == "processing":
        raise HTTPException(status_code=409, detail="Job is already processing.")

    entry["status"] = "processing"
    entry["config"] = config.dict()
    background_tasks.add_task(_pipeline_task, job_id)

    return {"job_id": job_id, "status": "processing_started"}


@app.get("/api/jobs/{job_id}", tags=["jobs"])
async def get_job(job_id: str):
    """Return registry metadata for a job."""
    root_dir = Path(__file__).parent
    job_dir = root_dir / "jobs" / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return {
        "job_id":       job_id,
        "ws_connected": manager.is_connected(job_id),
        **_job_registry.get(job_id, {}),
    }


@app.get("/api/jobs", tags=["jobs"])
async def list_jobs():
    return {"jobs": list(_job_registry.items())}


@app.get("/api/download/{job_id}", tags=["jobs"])
async def download_result(job_id: str):
    """
    Stream the restored MP4 back to the client.

    The output filename is ``restored_<original_stem>.mp4`` placed in
    ``./restored_output/``.
    """
    root_dir = Path(__file__).parent
    job_dir = root_dir / "jobs" / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found.")

    # Recover original stem from the job dir
    src = next(
        (f for f in job_dir.iterdir()
         if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTS),
        None,
    )
    if src is None:
        raise HTTPException(status_code=404, detail="Original source file not found.")

    out_path = root_dir / "restored_output" / f"restored_{src.stem}.mp4"
    if not out_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Output not ready yet. Poll /api/jobs/{job_id} for status.",
        )

    return FileResponse(
        path        = str(out_path),
        media_type  = "video/mp4",
        filename    = out_path.name,
    )


# ── Config Endpoints ───────────────────────────────────────────────────────────

@app.get("/api/config", tags=["config"])
async def get_config():
    cfg_path = Path(__file__).parent / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as fh:
        return json.load(fh)


@app.put("/api/config", tags=["config"])
async def update_config(request: Request):
    body = await request.json()
    cfg_path = Path(__file__).parent / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=2)
    return {"status": "saved"}


# ── WebSocket Endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    Real-time telemetry stream for a job.

    - Accepts a connection and registers it with the ConnectionManager.
    - Sends a server-side keepalive ping every 25 s so proxies don't time out.
    - Pipeline tasks write telemetry via ``manager.send_telemetry(job_id, …)``.
    - Handles graceful disconnect (client close or network error).
    """
    await manager.connect(job_id, websocket)

    try:
        # Send a connection-established ack
        await websocket.send_json({
            "type":   "connected",
            "job_id": job_id,
        })

        while True:
            try:
                # Wait for a client message (pings, cancel requests) with timeout
                msg = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=_WS_KEEPALIVE_S,
                )
                # Handle simple protocol messages
                if msg.strip() == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg.strip() == "cancel":
                    logger.info("Cancel requested for job %s", job_id)
                    _job_registry.setdefault(job_id, {})["status"] = "cancelled"
                    await websocket.send_json({"type": "cancel_acknowledged"})
                    break

            except asyncio.TimeoutError:
                # Send keepalive to prevent proxy / client timeouts
                try:
                    await websocket.send_json({"type": "keepalive"})
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info("WS client disconnected: job=%s", job_id)
    except Exception as exc:
        logger.warning("WS error for job=%s: %s", job_id, exc)
    finally:
        await manager.disconnect(job_id)


# ── Background Task ────────────────────────────────────────────────────────────

async def _pipeline_task(job_id: str) -> None:
    """Runs the full processing pipeline and updates job registry on completion."""
    try:
        pipeline = ProcessingPipeline(job_id, manager)
        await pipeline.run()
        _job_registry.setdefault(job_id, {})["status"] = "complete"
    except Exception as exc:
        logger.exception("Pipeline task failed for job %s: %s", job_id, exc)
        _job_registry.setdefault(job_id, {})["status"] = "error"
        await manager.send_telemetry(job_id, {
            "job_id": job_id,
            "stage":  "error",
            "error":  str(exc),
            "overall_progress": -1,
        })


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host      = "0.0.0.0",
        port      = 8765,
        reload    = False,
        log_level = "info",
        ws_ping_interval = 20,
        ws_ping_timeout  = 40,
    )
