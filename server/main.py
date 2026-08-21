"""FastAPI tracking server for browser-based standee collectors."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import cv2
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from audience import AudienceSessionTracker
from preprocess import preprocess
from providers.factory import create_provider
from server.database import TrackingRepository, database_url_from_environment


ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = database_url_from_environment(ROOT)
repository = TrackingRepository(DATABASE_URL)

app = FastAPI(title="Tracking AI API", version="0.1.0")
cors_origins = os.getenv("TRACKING_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/api/v1/reports/overview")
def report_overview(
    standee_id: str | None = None,
    campaign_id: str | None = None,
    creative_id: str | None = None,
    date_from: float | None = None,
    date_to: float | None = None,
) -> dict:
    return repository.overview(standee_id, campaign_id, creative_id, date_from, date_to)


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
) -> None:
    await websocket.accept()
    provider = create_provider(provider_name, enable_attributes=False)
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
                if payload.get("type") == "telemetry":
                    await asyncio.to_thread(repository.save_telemetry, standee_id, time.time(), payload)
                    await websocket.send_json({"type": "telemetry_ack"})
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
                    "frame_width": frame.shape[1],
                    "frame_height": frame.shape[0],
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
