"""Provider protocol used by runners and benchmarks."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from audience.contracts import PersonObservation


class VisionProvider(Protocol):
    name: str

    def warmup(self) -> None: ...

    def observe(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        source_frame: np.ndarray | None = None,
    ) -> list[PersonObservation]: ...

    def reset(self) -> None: ...
