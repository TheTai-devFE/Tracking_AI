"""Stable data contracts shared by vision providers and analytics consumers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PersonObservation:
    provider: str
    track_id: int
    timestamp: float
    bbox: tuple[int, int, int, int]
    detection_confidence: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    attentive: bool | None = None
    attention_confidence: float | None = None
    age: float | None = None
    age_group: str | None = None
    age_confidence: float | None = None
    attribute_sample_count: int = 0
    gender: str | None = None
    gender_confidence: float | None = None
    emotion: str | None = None
    emotion_confidence: float | None = None
    distance_m: float | None = None
    gaze_direction: str | None = None
    gaze_yaw: float | None = None
    gaze_pitch: float | None = None


@dataclass(frozen=True, slots=True)
class AudienceSession:
    schema_version: int
    session_id: str
    standee_id: str
    campaign_id: str
    creative_id: str
    provider: str
    provider_track_id: int
    started_at: float
    ended_at: float
    presence_seconds: float
    attention_seconds: float
    attention_ratio: float
    look_away_count: int
    is_impression: bool
    is_viewer: bool
    is_engaged: bool
    age_group: str
    age_confidence: float | None
    gender: str
    gender_confidence: float | None = None
    estimated_age: float | None = None
    dominant_emotion: str | None = None
    average_distance_m: float | None = None
    gaze_direction: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SessionConfig:
    min_presence_seconds: float = 1.0
    min_attention_seconds: float = 0.7
    engaged_seconds: float = 5.0
    track_gap_tolerance_seconds: float = 0.5
    session_expiry_seconds: float = 2.0
    attention_on_frames: int = 3
    attention_off_frames: int = 3
    age_confidence_threshold: float = 0.55
    gender_confidence_threshold: float = 0.65

    def __post_init__(self) -> None:
        positive = (
            self.min_presence_seconds,
            self.min_attention_seconds,
            self.engaged_seconds,
            self.track_gap_tolerance_seconds,
            self.session_expiry_seconds,
        )
        if any(value < 0 for value in positive):
            raise ValueError("Session durations must be non-negative")
        if self.attention_on_frames < 1 or self.attention_off_frames < 1:
            raise ValueError("Attention frame thresholds must be at least 1")
