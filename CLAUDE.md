# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **computer-vision audience analytics system** for digital signage and standees: `frame in → anonymous audience session analytics out` (impression, viewer, attention duration, dwell time, look-aways, and optional demographic attributes).

**Hard constraints:**
- **No model training**: Uses pretrained models (UniFace SCRFD, ByteTrack, MobileNetV3 HeadPose).
- **Privacy by design**: Anonymous measurement only — no face recognition, no embedding storage, no facial crops saved in production.

## Commands

Environment is managed by **uv** (Python pinned to 3.12 in `.python-version`).

```bash
uv sync                                                      # recreate .venv from uv.lock
uv run python run_experiment.py --provider uniface --webcam 0 --display  # live webcam demo
uv run python run_experiment.py --provider uniface --video data/test.mp4 --events-out outputs/events.jsonl  # batch video
uv run python -m unittest discover -s tests -v               # run unit tests
uv run uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload  # start FastAPI backend
```

## Architecture

- **`audience/`**:
  - `contracts.py`: Standard contracts (`PersonObservation`, `AudienceSessionEvent`).
  - `session_tracker.py`: State machine managing session lifecycles, dwell time, attention smoothing, and look-away counting.
  - `demographics.py`: Age and gender grouping normalization.
- **`providers/`**:
  - `base.py`: `VisionProvider` Protocol (`observe`, `warmup`, `reset`).
  - `factory.py`: Instantiates providers (currently `uniface`).
  - `uniface_provider.py`: SCRFD detection + ByteTrack + MobileNetV3 HeadPose + optional Age/Gender estimation.
- **`server/`**:
  - FastAPI server receiving frames over WebSocket and persisting session telemetry to SQLite/PostgreSQL.
- **`web/`**:
  - Next.js application for standee client display and real-time dashboard.
- **`configs.py`**:
  - Centralized thresholds (yaw/pitch angles, confidence thresholds, image processing resolutions).
