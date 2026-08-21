"""Adapter from the existing pipeline to the normalized observation contract."""

from __future__ import annotations

import numpy as np

from audience.contracts import PersonObservation
from audience.demographics import normalize_gender
from pipeline import Pipeline


class BaselineProvider:
    name = "baseline"

    def __init__(self, enable_attributes: bool = False) -> None:
        self.pipeline = Pipeline(enable_attributes=enable_attributes)

    def warmup(self) -> None:
        self.pipeline.tracker._ensure_model()

    def observe(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        source_frame: np.ndarray | None = None,
    ) -> list[PersonObservation]:
        metas = self.pipeline.process(frame_bgr, now=timestamp, source_frame=source_frame)
        return [
            PersonObservation(
                provider=self.name,
                track_id=meta.track_id,
                timestamp=timestamp,
                bbox=meta.bbox,
                yaw=meta.yaw,
                pitch=meta.pitch,
                roll=meta.roll,
                attentive=bool(meta.attention),
                age_group=meta.age_group,
                gender=normalize_gender(meta.gender),
            )
            for meta in metas
        ]

    def reset(self) -> None:
        self.pipeline.tracker.reset()
