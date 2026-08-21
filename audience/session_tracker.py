"""Convert per-frame observations into anonymous audience sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
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
            if active.attentive:
                active.attention_seconds += delta

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

    def _demographics(self, active: _ActiveSession) -> tuple[str, float | None, str, float | None]:
        age_labels = [(group, confidence) for _, group, confidence in active.age_samples if group]
        if age_labels:
            age_group, age_confidence = self._vote(age_labels)
        elif active.age_samples:
            age = median(sample[0] for sample in active.age_samples)
            age_group = map_age_group(age)
            age_confidence = 1.0
        else:
            age_group, age_confidence = "unknown", None

        gender, gender_confidence = self._vote(active.gender_samples)
        if age_confidence is not None and age_confidence < self.config.age_confidence_threshold:
            age_group = "unknown"
        if gender_confidence is not None and gender_confidence < self.config.gender_confidence_threshold:
            gender = "unknown"
        return age_group, age_confidence, gender, gender_confidence

    def _to_event(self, active: _ActiveSession) -> AudienceSession:
        age_group, age_confidence, gender, gender_confidence = self._demographics(active)
        presence = active.presence_seconds
        attention = min(active.attention_seconds, presence)
        return AudienceSession(
            schema_version=1,
            session_id=active.session_id,
            standee_id=self.standee_id,
            campaign_id=self.campaign_id,
            creative_id=self.creative_id,
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
        )
