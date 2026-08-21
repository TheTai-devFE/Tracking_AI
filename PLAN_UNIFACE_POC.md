# UniFace Audience Analytics PoC

## 1. Objective

Build an anonymous audience-measurement pipeline for advertising displays and
compare UniFace against the current YOLO/ByteTrack/MiVOLO/MediaPipe baseline.

For each advertisement playback window, the system must estimate:

- how many people were present (`impressions`);
- how many people looked at the display (`viewers`);
- attention time, presence time, and look-away events;
- head direction (yaw, pitch, roll);
- estimated age group and gender when confidence is sufficient;
- results grouped by standee, campaign, creative, and time window.

This branch is an experiment. It must not replace the current pipeline until the
same input videos show a measurable improvement.

## 2. Scope and non-goals

### In scope

- Offline video and local webcam inference.
- A provider abstraction supporting `baseline` and `uniface` engines.
- Session-level analytics rather than one CSV row per face per frame.
- JSONL/CSV event output suitable for a future FastAPI ingestion endpoint.
- Reproducible accuracy, stability, latency, and throughput comparisons.
- Privacy-first processing: random session ids and no stored face crops by default.

### Out of scope for the first PoC

- Face recognition or persistent identity across visits.
- Claiming an exact daily unique-person count.
- Training or fine-tuning models.
- Production deployment, remote camera streaming, CMS, or APK implementation.
- Automatic ad selection before measurement quality is validated.

## 3. Measurement definitions

The names below must remain consistent across CV output, API, database, and CMS.

| Metric | Definition |
|---|---|
| `impression` | A tracked face remains visible for at least `min_presence_seconds`. |
| `viewer` | An impression accumulates at least `min_attention_seconds`. |
| `engaged_viewer` | A viewer accumulates at least `engaged_seconds`. |
| `presence_seconds` | Time between the first and final observation, excluding a gap beyond the session timeout. |
| `attention_seconds` | Accumulated intervals classified as attentive after temporal smoothing. |
| `attention_ratio` | `attention_seconds / presence_seconds`, bounded to `[0, 1]`. |
| `look_away_count` | Number of transitions from attentive to non-attentive after hysteresis. |
| `estimated_viewers` | Count of qualifying sessions. It is not a biometric unique-person count. |

Initial thresholds are configuration values, not hard-coded business rules:

- `min_presence_seconds`: 1.0 s
- `min_attention_seconds`: 0.7 s
- `engaged_seconds`: 5.0 s
- `track_gap_tolerance_seconds`: 0.5 s
- `session_expiry_seconds`: 3.0 s
- age/gender confidence below the calibrated threshold becomes `unknown`

## 4. Target architecture

```text
Video/Webcam
    |
    v
Frame preprocessing
    |
    +--> Baseline provider: Ultralytics + MediaPipe + MiVOLO
    |
    +--> UniFace provider: detector + ByteTrack + pose/gaze + attributes
             |
             v
Normalized PersonObservation contract
             |
             v
AudienceSessionTracker (business state machine)
             |
             v
AudienceSession event (JSONL/CSV)
             |
             v
Future FastAPI ingestion -> PostgreSQL -> Next.js CMS
```

Detection/model code must not own the viewer definitions. Both providers feed the
same session tracker so the benchmark compares models, not different business
logic.

## 5. Data contracts

### Per-frame observation

```python
@dataclass
class PersonObservation:
    provider: str
    track_id: int
    timestamp: float
    bbox: tuple[int, int, int, int]
    detection_confidence: float
    yaw: float | None
    pitch: float | None
    roll: float | None
    attentive: bool | None
    attention_confidence: float | None
    age: float | None
    age_group: str | None
    age_confidence: float | None
    gender: str | None
    gender_confidence: float | None
```

### Closed audience session

```json
{
  "schema_version": 1,
  "session_id": "random-uuid",
  "standee_id": "ST-001",
  "campaign_id": "CMP-001",
  "creative_id": "AD-001",
  "provider": "uniface",
  "started_at": "2026-08-14T03:30:00Z",
  "ended_at": "2026-08-14T03:30:08Z",
  "presence_seconds": 8.0,
  "attention_seconds": 5.4,
  "attention_ratio": 0.675,
  "look_away_count": 1,
  "is_viewer": true,
  "is_engaged": true,
  "age_group": "25-34",
  "age_confidence": 0.72,
  "gender": "unknown",
  "gender_confidence": 0.48
}
```

No raw frame, face crop, embedding, name, or persistent biometric identifier is
part of this event.

## 6. Proposed repository layout

Keep the current flat modules operational while introducing the experiment in a
separate package:

```text
audience/
  contracts.py          # normalized observations and closed events
  session_tracker.py    # impression/viewer/dwell/look-away state machine
  aggregation.py        # hourly/daily/campaign summaries
providers/
  base.py               # provider protocol
  baseline.py           # adapter around the existing Pipeline stages
  uniface_provider.py   # UniFace detector/tracker/pose/gaze/attributes
eval/
  run_comparison.py     # same video and timestamps through both providers
  metrics.py            # ID switches, count error, attention error, FPS
  annotations/          # local, git-ignored ground-truth files
run_experiment.py       # video/webcam runner and JSONL/overlay output
```

## 7. Implementation phases

### Phase A - Contracts and deterministic session logic

1. Add `PersonObservation`, `AudienceSession`, and configuration dataclasses.
2. Implement state transitions: detected, present, attentive, viewing, closed.
3. Handle short tracking gaps without immediately closing a session.
4. Aggregate repeated age/gender observations using robust votes and confidence.
5. Unit test dwell time, look-away transitions, expiry, recycled track ids, and
   `unknown` demographic results using explicit timestamps.

Definition of done:

- Session tests are deterministic and do not load any ML model.
- One synthetic track produces exactly one closed event with expected durations.

### Phase B - UniFace provider

1. Inspect and select the lightest viable UniFace detector for the target CPU.
   Start with SCRFD and compare YOLOv8Face only if necessary.
2. Adapt UniFace ByteTrack output to `PersonObservation`.
3. Add head pose first; evaluate gaze separately because gaze can require larger,
   sharper face crops than pose.
4. Run age/gender periodically, collect several samples per session, and never run
   it on every frame.
5. Make model weights/cache paths explicit under `models/uniface/`.
6. Add lazy initialization and clear errors for missing/download-failed weights.

Definition of done:

- A local video produces an annotated output video plus closed-session JSONL.
- No face image is written unless a debug flag is explicitly enabled.
- Provider can be selected by CLI without editing code.

### Phase C - Baseline adapter and A/B evaluation

1. Adapt the existing pipeline to the same observation contract.
2. Prepare 5-10 representative videos covering:
   - one person looking and looking away;
   - multiple people crossing;
   - partial occlusion and re-entry;
   - side profiles;
   - backlight and low light;
   - near and far faces.
3. Label ground truth at session level: entries, viewer/not-viewer, approximate
   attention interval, and visible track continuity.
4. Run both providers with identical input frames and timestamps.
5. Produce a comparison report.

Required metrics:

- impression and viewer count absolute error;
- precision/recall for viewer classification;
- attention-duration MAE;
- track fragmentation and observable ID switches;
- age-group accuracy and `unknown` rate on consented labelled samples;
- gender accuracy and `unknown` rate on consented labelled samples;
- average FPS, p95 frame latency, CPU and memory use;
- cold-start time and model storage size.

Initial acceptance targets for a controlled PoC:

- viewer count error <= 10% on labelled test clips;
- viewer precision and recall >= 0.85;
- median attention-duration error <= 1.0 s;
- no avoidable ID switch in the single-person clips;
- sustained >= 15 FPS on the intended standee hardware, or a documented sampling
  configuration that preserves metric accuracy.

These are experiment gates and should be revised after the first labelled set.

### Phase D - Event ingestion API

Proceed only after Phase C selects a provider.

1. Create a FastAPI service with versioned schemas.
2. Add endpoints for device registration/configuration, creative schedules, event
   batch ingestion, and aggregated analytics queries.
3. Make ingestion idempotent using `session_id` plus `standee_id`.
4. Store UTC timestamps and retain the device timezone as metadata.
5. Store session events in PostgreSQL; build hourly/daily aggregate tables or
   materialized views after query patterns are known.
6. Add authentication per standee and retryable batch uploads.

The FastAPI process should not receive a permanent full-frame camera stream in
production. The standee should upload compact session events whenever possible.

### Phase E - Next.js CMS

1. Campaign and creative CRUD.
2. Targeting metadata such as allowed age groups and schedule constraints.
3. Standee registration, health, current creative, and last synchronization.
4. Dashboard funnels: impressions -> viewers -> engaged viewers.
5. Charts by day/hour, standee, campaign, creative, age group, and gender.
6. Display confidence/unknown rates so estimates are not presented as facts.

### Phase F - Expo standee application

1. Play downloaded creatives offline and report the exact playback interval.
2. Associate every local audience event with the active `creative_id`.
3. Cache configuration, creatives, and unsent event batches.
4. Add device health reporting and remote configuration refresh.
5. Prototype camera frame access using an Expo development build/native module.
6. Benchmark on-device inference before choosing between native ONNX inference and
   a local/remote inference service.

Do not continuously upload raw camera video as the default design. It creates high
bandwidth, latency, availability, and privacy costs.

### Phase G - Recommendation experiment

Recommendation comes after trustworthy measurement.

1. Begin with deterministic eligibility rules: schedule, campaign status, allowed
   age groups, content safety, and frequency caps.
2. Rank eligible creatives using aggregate audience segments for that standee and
   time window, not an identity profile.
3. Use an explicit exploration percentage so new creatives receive exposure.
4. Log the reason and model/rule version for every selection.
5. Evaluate uplift using campaign-level A/B tests rather than online face identity.

## 8. Configuration and CLI

Expected experiment interface:

```powershell
uv run python run_experiment.py `
  --provider uniface `
  --video data/test.mp4 `
  --standee-id ST-001 `
  --campaign-id CMP-001 `
  --creative-id AD-001 `
  --events-out outputs/uniface-events.jsonl `
  --render-out outputs/uniface-overlay.mp4
```

All thresholds, model names, providers, sampling intervals, and privacy/debug
switches belong in configuration. They must not require stage-code edits.

## 9. Risks and controls

| Risk | Control |
|---|---|
| Double counting after occlusion or re-entry | Track-gap tolerance, session expiry, report estimated sessions rather than exact unique people. |
| Demographic bias or low-resolution guesses | Confidence threshold, `unknown`, grouped reporting, validation on target-like consented data. |
| False attention from head pose alone | Calibrate for camera/display geometry; compare pose with gaze only where crop quality permits. |
| Low FPS from running every model every frame | Throttle attributes, batch where supported, use lightweight detector, measure p95 latency. |
| OpenCV package overlap | Resolve `opencv-python` versus `opencv-contrib-python` before locking the production environment. |
| Model download failure | Pre-provision verified weights and use an explicit cache directory. |
| Privacy exposure | Process locally, discard frames, random session ids, short retention, no embeddings or recognition. |
| Metric drift between devices | Version provider/configuration in every event and maintain a calibration procedure. |

## 10. Decision gate

Adopt UniFace only if Phase C shows that it improves at least one important metric
(viewer accuracy, attention accuracy, track stability, or deployment simplicity)
without unacceptable FPS, cold-start, dependency, or licensing costs.

Possible outcomes:

- **Adopt:** UniFace provider wins and becomes the default.
- **Hybrid:** use UniFace for selected stages such as SCRFD or gaze while retaining
  current MiVOLO and geometric head pose.
- **Reject:** retain the baseline and remove UniFace from production dependencies.

Record the decision and benchmark artifacts before starting application-layer
integration.
