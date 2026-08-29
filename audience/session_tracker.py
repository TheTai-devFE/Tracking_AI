"""Convert per-frame observations into anonymous audience sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any
from uuid import uuid4

from .contracts import AudienceSession, PersonObservation, SessionConfig
from .demographics import map_age_group, normalize_gender


@dataclass
class _ActiveSession:
    session_id: str
    provider: str
    track_id: int
    first_seen: float
    last_seen: float
    presence_seconds: float = 0.0
    attention_seconds: float = 0.0
    attentive: bool = False
    attentive_streak: int = 0
    inattentive_streak: int = 0
    look_away_count: int = 0
    age_samples: list[tuple[float, str | None, float | None]] = field(default_factory=list)
    gender_samples: list[tuple[str, float | None]] = field(default_factory=list)
    emotion_samples: list[str] = field(default_factory=list)
    distance_samples: list[float] = field(default_factory=list)
    gaze_samples: list[str] = field(default_factory=list)
    creative_presence: dict[str, float] = field(default_factory=dict)
    creative_attention: dict[str, float] = field(default_factory=dict)


class AudienceSessionTracker:
    """Owns viewer definitions while model providers only supply observations."""

    def __init__(
        self,
        standee_id: str,
        campaign_id: str,
        creative_id: str,
        config: SessionConfig | None = None,
    ) -> None:
        self.standee_id = standee_id
        self.campaign_id = campaign_id
        self.creative_id = creative_id
        self.config = config or SessionConfig()
        self._active: dict[tuple[str, int], _ActiveSession] = {}

    def update(self, observations: list[PersonObservation], now: float) -> list[AudienceSession]:
        """Consume one frame and return sessions that expired on this frame."""
        if any(abs(observation.timestamp - now) > 1e-6 for observation in observations):
            raise ValueError("All observation timestamps must match now")

        closed = self._close_expired(now)
        for observation in observations:
            key = (observation.provider, observation.track_id)
            active = self._active.get(key)
            if active is None:
                active = _ActiveSession(
                    session_id=str(uuid4()),
                    provider=observation.provider,
                    track_id=observation.track_id,
                    first_seen=now,
                    last_seen=now,
                )
                self._active[key] = active
            self._apply(active, observation)
        return closed

    def flush(self) -> list[AudienceSession]:
        """Close every live session at its last observed timestamp."""
        sessions = [self._to_event(active) for active in self._active.values()]
        self._active.clear()
        return sessions

    def active_sessions_summary(self, now: float | None = None) -> list[dict[str, Any]]:
        """Return real-time state of all currently active live viewers."""
        summary = []
        for active in self._active.values():
            if now is not None and (now - active.last_seen > 1.0):
                # Skip viewers that are no longer actively detected in the current camera frame
                continue
            age_group, _, gender, _, estimated_age = self._demographics(active)
            presence = active.presence_seconds
            attention = min(active.attention_seconds, presence)
            ratio = round(attention / presence, 2) if presence > 0 else 0.0
            dominant_emotion = max(set(active.emotion_samples), key=active.emotion_samples.count) if active.emotion_samples else None
            average_distance = round(float(median(active.distance_samples)), 1) if active.distance_samples else None
            gaze = active.gaze_samples[-1] if active.gaze_samples else None

            summary.append(
                {
                    "session_id": active.session_id,
                    "track_id": active.track_id,
                    "presence_seconds": round(presence, 1),
                    "attention_seconds": round(attention, 1),
                    "attention_ratio": ratio,
                    "attentive": active.attentive,
                    "look_away_count": active.look_away_count,
                    "gender": gender,
                    "age_group": age_group,
                    "estimated_age": estimated_age,
                    "dominant_emotion": dominant_emotion,
                    "average_distance_m": average_distance,
                    "gaze_direction": gaze,
                }
            )
        return summary

    def switch_creative(self, creative_id: str, campaign_id: str | None = None) -> list[AudienceSession]:
        """Point new observation time at another creative, keeping live sessions open.

        A session measures one person's visit, so a playlist rotation must not end
        it: closing here would split a viewer who never moved into one session per
        creative, inflating headcount and truncating dwell time. Time keeps
        accruing per creative instead, and `_to_event` attributes the finished
        session to whichever creative held the viewer's attention longest.

        Returns an empty list; the signature stays list-shaped for callers that
        forward closed sessions to storage.
        """
        self.creative_id = creative_id
        if campaign_id:
            self.campaign_id = campaign_id
        return []

    def _close_expired(self, now: float) -> list[AudienceSession]:
        expired = [
            key
            for key, active in self._active.items()
            if now - active.last_seen > self.config.session_expiry_seconds
        ]
        events = [self._to_event(self._active[key]) for key in expired]
        for key in expired:
            del self._active[key]
        return events

    def _apply(self, active: _ActiveSession, observation: PersonObservation) -> None:
        now = observation.timestamp
        delta = max(0.0, now - active.last_seen)
        if delta <= self.config.track_gap_tolerance_seconds:
            active.presence_seconds += delta
            creative = self.creative_id
            active.creative_presence[creative] = active.creative_presence.get(creative, 0.0) + delta
            if active.attentive:
                active.attention_seconds += delta
                active.creative_attention[creative] = active.creative_attention.get(creative, 0.0) + delta

        attentive_now = observation.attentive is True
        if attentive_now:
            active.attentive_streak += 1
            active.inattentive_streak = 0
            if not active.attentive and active.attentive_streak >= self.config.attention_on_frames:
                active.attentive = True
        else:
            active.inattentive_streak += 1
            active.attentive_streak = 0
            if active.attentive and active.inattentive_streak >= self.config.attention_off_frames:
                active.attentive = False
                active.look_away_count += 1

        if observation.age is not None or observation.age_group is not None:
            active.age_samples.append((observation.age or 0.0, observation.age_group, observation.age_confidence))
        if observation.gender:
            gender = normalize_gender(observation.gender)
            if gender:
                active.gender_samples.append((gender, observation.gender_confidence))
        if observation.emotion:
            active.emotion_samples.append(observation.emotion)
        if observation.distance_m is not None:
            active.distance_samples.append(observation.distance_m)
        if observation.gaze_direction:
            active.gaze_samples.append(observation.gaze_direction)
        active.last_seen = now

    @staticmethod
    def _vote(samples: list[tuple[str, float | None]]) -> tuple[str, float | None]:
        if not samples:
            return "unknown", None
        weights: dict[str, float] = {}
        for label, confidence in samples:
            weights[label] = weights.get(label, 0.0) + (confidence if confidence is not None else 1.0)
        winner = max(weights, key=weights.get)
        agreement = weights[winner] / sum(weights.values())
        return winner, agreement

    def _demographics(self, active: _ActiveSession) -> tuple[str, float | None, str, float | None, float | None]:
        valid_ages = [sample[0] for sample in active.age_samples if sample[0] > 0]
        estimated_age = round(float(median(valid_ages)), 1) if valid_ages else None

        age_labels = [(group, confidence) for _, group, confidence in active.age_samples if group]
        if age_labels:
            age_group, age_confidence = self._vote(age_labels)
        elif estimated_age is not None:
            age_group = map_age_group(estimated_age)
            age_confidence = 1.0
        else:
            age_group, age_confidence = "unknown", None

        gender, gender_confidence = self._vote(active.gender_samples)
        if age_confidence is not None and age_confidence < self.config.age_confidence_threshold:
            age_group = "unknown"
        if gender_confidence is not None and gender_confidence < self.config.gender_confidence_threshold:
            gender = "unknown"
        return age_group, age_confidence, gender, gender_confidence, estimated_age

    def _dominant_creative(self, active: _ActiveSession) -> str:
        """Attribute a visit spanning several creatives to the one it watched most."""
        for tally in (active.creative_attention, active.creative_presence):
            if tally:
                return max(tally, key=tally.get)
        return self.creative_id

    def _to_event(self, active: _ActiveSession) -> AudienceSession:
        age_group, age_confidence, gender, gender_confidence, estimated_age = self._demographics(active)
        presence = active.presence_seconds
        attention = min(active.attention_seconds, presence)
        dominant_emotion = max(set(active.emotion_samples), key=active.emotion_samples.count) if active.emotion_samples else None
        average_distance_m = round(float(median(active.distance_samples)), 1) if active.distance_samples else None
        gaze_direction = max(set(active.gaze_samples), key=active.gaze_samples.count) if active.gaze_samples else None

        return AudienceSession(
            schema_version=1,
            session_id=active.session_id,
            standee_id=self.standee_id,
            campaign_id=self.campaign_id,
            creative_id=self._dominant_creative(active),
            provider=active.provider,
            provider_track_id=active.track_id,
            started_at=active.first_seen,
            ended_at=active.last_seen,
            presence_seconds=round(presence, 3),
            attention_seconds=round(attention, 3),
            attention_ratio=round(attention / presence, 4) if presence else 0.0,
            look_away_count=active.look_away_count,
            is_impression=presence >= self.config.min_presence_seconds,
            is_viewer=attention >= self.config.min_attention_seconds,
            is_engaged=attention >= self.config.engaged_seconds,
            age_group=age_group,
            age_confidence=age_confidence,
            gender=gender,
            gender_confidence=gender_confidence,
            estimated_age=estimated_age,
            dominant_emotion=dominant_emotion,
            average_distance_m=average_distance_m,
            gaze_direction=gaze_direction,
        )
