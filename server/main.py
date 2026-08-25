"""FastAPI tracking server for browser-based standee collectors."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import cv2
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np

from audience import AudienceSessionTracker
from audience.contracts import PersonObservation
from preprocess import preprocess
from providers.factory import create_provider
from server.database import TrackingRepository, database_url_from_environment


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = database_url_from_environment(ROOT)
repository = TrackingRepository(DATABASE_URL)


def rescale_bbox_to_source(
    observation: PersonObservation,
    processed_frame: np.ndarray,
    source_frame: np.ndarray,
) -> PersonObservation:
    """Map an observation's bbox from processed-frame space back to source-frame space.

    `preprocess` may downscale the frame the client sent (CFG.process_long_side)
    before detection runs, so providers report boxes in that internal space. That
    is an implementation detail clients cannot know about: leaking it makes every
    overlay draw undersized, offset boxes. Wire coordinates are always in the
    space of the frame the client actually sent.
    """
    processed_height, processed_width = processed_frame.shape[:2]
    source_height, source_width = source_frame.shape[:2]
    if (processed_height, processed_width) == (source_height, source_width):
        return observation

    scale_x = source_width / processed_width
    scale_y = source_height / processed_height
    x1, y1, x2, y2 = observation.bbox
    return replace(
        observation,
        bbox=(
            round(x1 * scale_x),
            round(y1 * scale_y),
            round(x2 * scale_x),
            round(y2 * scale_y),
        ),
    )


app = FastAPI(title="Tracking AI API", version="0.1.0")
cors_origins = os.getenv("TRACKING_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.on_event("startup")
def initialize_database() -> None:
    repository.initialize()


@app.get("/health")
def health() -> dict:
    repository.healthcheck()
    return {
        "status": "ok",
        "service": "tracking-ai",
        "database_backend": repository.backend,
        "database_url": repository.display_url,
    }


@app.get("/api/v1/creatives")
def list_creatives(only_active: bool = False) -> list[dict]:
    return repository.list_creatives(only_active=only_active)


@app.post("/api/v1/creatives/upload")
async def upload_creative(
    file: UploadFile = File(...),
    name: str = Form(None),
    duration_seconds: float = Form(10.0),
    campaign_id: str = Form("CMP-LOCAL"),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    ext = Path(file.filename).suffix.lower()
    creative_id = f"cr_{uuid.uuid4().hex[:8]}"
    clean_name = name.strip() if name and name.strip() else Path(file.filename).stem
    stored_filename = f"{creative_id}{ext}"
    target_path = UPLOAD_DIR / stored_filename

    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    media_type = "video" if ext in {".mp4", ".webm", ".mov", ".mkv"} or (file.content_type and file.content_type.startswith("video/")) else "image"

    data = {
        "id": creative_id,
        "name": clean_name,
        "file_name": file.filename,
        "file_url": f"/static/uploads/{stored_filename}",
        "media_type": media_type,
        "duration_seconds": max(1.0, duration_seconds),
        "campaign_id": campaign_id,
        "is_active": True,
        "sort_order": len(repository.list_creatives()),
    }
    return repository.create_creative(data)


@app.delete("/api/v1/creatives/{creative_id}")
def delete_creative(creative_id: str) -> dict:
    creatives = {c["id"]: c for c in repository.list_creatives()}
    if creative_id in creatives:
        file_url = creatives[creative_id].get("file_url", "")
        if file_url.startswith("/static/uploads/"):
            filename = file_url.removeprefix("/static/uploads/")
            target_path = UPLOAD_DIR / filename
            if target_path.is_file():
                try:
                    target_path.unlink()
                except OSError:
                    pass
    deleted = repository.delete_creative(creative_id)
    return {"status": "ok", "deleted": deleted}


@app.patch("/api/v1/creatives/{creative_id}")
def update_creative(creative_id: str, payload: dict = Body(...)) -> dict:
    allowed = {"name", "duration_seconds", "is_active", "sort_order", "campaign_id"}
    data = {k: v for k, v in payload.items() if k in allowed}
    if not data:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    updated = repository.update_creative(creative_id, data)
    return {"status": "ok", "updated": updated}


@app.post("/api/v1/creatives/reorder")
def reorder_creatives(payload: dict = Body(...)) -> dict:
    ordered_ids = payload.get("ordered_ids")
    if not isinstance(ordered_ids, list):
        raise HTTPException(status_code=400, detail="ordered_ids must be a list of strings")
    success = repository.reorder_creatives(ordered_ids)
    return {"status": "ok", "reordered": success}


@app.get("/api/v1/reports/overview")
def report_overview(
    standee_id: str | None = None,
    campaign_id: str | None = None,
    creative_id: str | None = None,
    date_from: float | None = None,
    date_to: float | None = None,
) -> dict:
    return repository.overview(standee_id, campaign_id, creative_id, date_from, date_to)


@app.get("/api/v1/reports/creatives")
def report_creatives(
    standee_id: str | None = None,
    campaign_id: str | None = None,
) -> list[dict]:
    return repository.creatives_report(standee_id=standee_id, campaign_id=campaign_id)


@app.get("/api/v1/reports/sessions")
def report_sessions(standee_id: str | None = None, limit: int = Query(100, ge=1, le=500)) -> list[dict]:
    return repository.recent_sessions(standee_id, limit)


@app.get("/api/v1/reports/timeline")
def report_timeline(standee_id: str | None = None, hours: int = Query(24, ge=1, le=720)) -> list[dict]:
    return repository.timeline(standee_id, hours)


@app.websocket("/ws/tracking/{standee_id}")
async def tracking_socket(
    websocket: WebSocket,
    standee_id: str,
    campaign_id: str = "CMP-LOCAL",
    creative_id: str = "AD-LOCAL",
    provider_name: str = "uniface",
    enable_attributes: bool = True,
) -> None:
    await websocket.accept()
    provider = create_provider(provider_name, enable_attributes=enable_attributes)
    tracker = AudienceSessionTracker(standee_id, campaign_id, creative_id)
    frame_index = 0
    try:
        await websocket.send_json({"type": "initializing", "standee_id": standee_id, "provider": provider_name})
        await asyncio.to_thread(provider.warmup)
        await websocket.send_json({"type": "ready", "standee_id": standee_id, "provider": provider_name})
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                payload = json.loads(message["text"])
                msg_type = payload.get("type")
                if msg_type == "telemetry":
                    await asyncio.to_thread(repository.save_telemetry, standee_id, time.time(), payload)
                    await websocket.send_json({"type": "telemetry_ack"})
                elif msg_type in ("set_creative", "switch_creative"):
                    new_creative_id = payload.get("creative_id", "AD-LOCAL")
                    new_campaign_id = payload.get("campaign_id", campaign_id)
                    closed = tracker.switch_creative(new_creative_id, new_campaign_id)
                    if closed:
                        await asyncio.to_thread(repository.save_sessions, closed)
                    await websocket.send_json({
                        "type": "creative_switched",
                        "creative_id": new_creative_id,
                        "campaign_id": new_campaign_id,
                        "closed_sessions": [s.as_dict() for s in closed],
                    })
                continue

            frame_bytes = message.get("bytes")
            if not frame_bytes:
                continue
            received_at = time.time()
            encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
            source_frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if source_frame is None:
                await websocket.send_json({"type": "frame_error", "message": "invalid JPEG frame"})
                continue

            frame = preprocess(source_frame)
            started = time.perf_counter()
            observations = await asyncio.to_thread(provider.observe, frame, received_at, source_frame)
            observations = [
                rescale_bbox_to_source(observation, frame, source_frame)
                for observation in observations
            ]
            closed = tracker.update(observations, received_at)
            if closed:
                await asyncio.to_thread(repository.save_sessions, closed)
            processing_ms = (time.perf_counter() - started) * 1000
            frame_index += 1
            await websocket.send_json(
                {
                    "type": "frame_result",
                    "frame_index": frame_index,
                    "server_timestamp": received_at,
                    # Coordinate space of observations[].bbox — the frame the
                    # client sent, not the internally downscaled one.
                    "frame_width": source_frame.shape[1],
                    "frame_height": source_frame.shape[0],
                    "processed_width": frame.shape[1],
                    "processed_height": frame.shape[0],
                    "processing_ms": round(processing_ms, 2),
                    "observations": [asdict(observation) for observation in observations],
                    "closed_sessions": [session.as_dict() for session in closed],
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        closed = tracker.flush()
        if closed:
            await asyncio.to_thread(repository.save_sessions, closed)
        provider.reset()

