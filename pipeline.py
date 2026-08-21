"""Main pipeline: glue every stage into `process(frame) -> list[PersonMeta]`.

This is the single seam the app layer will call later (NestJS / on-device). The app
never needs to know what's inside.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np

from configs import CFG
from preprocess import preprocess, to_rgb
from crop import crop_face
from tracker import FaceTracker
from head_pose import HeadPoseEstimator
from age_gender import AgeGender, AgeGenderEstimator, map_age_group
from attention import DwellTracker, is_attentive


@dataclass
class PersonMeta:
    track_id: int
    bbox: tuple[int, int, int, int]
    age_group: str | None = None
    gender: str | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    attention: int = 0
    dwell_time: float = 0.0

    def as_row(self) -> dict:
        return asdict(self)


class Pipeline:
    """Assembles the real-time branch. Stages are enabled as phases land."""

    def __init__(self, cfg=CFG, enable_attributes: bool = True):
        self.cfg = cfg
        self.enable_attributes = enable_attributes
        self.tracker = FaceTracker()                 # Phase 2
        self.head_pose = HeadPoseEstimator()         # Phase 3
        self.age_gender = AgeGenderEstimator()       # Phase 4
        self.dwell = DwellTracker()                  # Phase 5
        self._frame_idx = 0
        self._ag_samples: dict[int, list[AgeGender]] = {}   # track_id -> raw votes
        self._age_cache: dict[int, tuple[str, str]] = {}    # track_id -> (age_group, gender)
        self._last_seen: dict[int, float] = {}

    def _reduce_votes(self, samples: list[tuple[AgeGender, int]]) -> tuple[str | None, str]:
        """Collapse several noisy per-frame estimates into one answer.

        Gender by confidence-weighted vote over every sample; age by median over
        only the samples whose crop was big enough to be worth trusting. The nets
        are jumpy frame to frame — especially on the partial, motion-blurred face
        you get the instant someone walks into shot — so a single estimate is a
        coin flip and the first estimate is the worst one to keep.
        """
        scores: dict[str, float] = {}
        for ag, _ in samples:
            scores[ag.gender] = scores.get(ag.gender, 0.0) + ag.gender_conf
        gender = max(scores, key=scores.__getitem__)

        if not self.cfg.age_enabled:
            return None, gender
        usable = [ag.age for ag, px in samples if px >= self.cfg.min_face_px_for_age]
        age_group = map_age_group(float(np.median(usable))) if usable else None
        return age_group, gender

    def _attribute_crop(self, frame_bgr: np.ndarray, source_frame: np.ndarray | None,
                        bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Crop the face for age/gender from the sharpest frame we have.

        Detection runs on the CFG.process_long_side-downscaled frame for speed, but
        cropping the attribute face from that same frame throws away resolution the
        age model badly needs. When the caller hands us the original frame, scale
        the bbox back up and crop from there instead.
        """
        if source_frame is None or source_frame.shape == frame_bgr.shape:
            return crop_face(frame_bgr, bbox)
        sy = source_frame.shape[0] / frame_bgr.shape[0]
        sx = source_frame.shape[1] / frame_bgr.shape[1]
        x1, y1, x2, y2 = bbox
        return crop_face(source_frame, (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)))

    def _age_gender_voted(self, track_id: int, face_bgr: np.ndarray) -> tuple[str | None, str | None]:
        """Sample age/gender periodically until enough votes, then hold the result."""
        samples = self._ag_samples.setdefault(track_id, [])
        due = (
            not samples  # always get something on the frame a track first appears
            or (len(samples) < self.cfg.age_gender_samples
                and self._frame_idx % self.cfg.age_gender_every_n == 0)
        )
        if due:
            ag = self.age_gender.estimate(face_bgr)
            if ag is not None:
                samples.append((ag, int(face_bgr.shape[0])))
                self._age_cache[track_id] = self._reduce_votes(samples)
        return self._age_cache.get(track_id, (None, None))

    def _retire(self, live_ids: set[int], now: float) -> None:
        """Close gaze sessions for vanished tracks and expire their cached state."""
        for track_id in list(self._last_seen):
            if track_id in live_ids:
                self._last_seen[track_id] = now
                continue
            self.dwell.close(track_id, now)
            if now - self._last_seen[track_id] > self.cfg.track_expiry_seconds:
                self._last_seen.pop(track_id)
                self._ag_samples.pop(track_id, None)
                self._age_cache.pop(track_id, None)
                self.dwell.forget(track_id)

    def process(self, frame_bgr: np.ndarray, now: float | None = None,
                source_frame: np.ndarray | None = None) -> list[PersonMeta]:
        """One frame -> per-person metadata using all active pipeline stages.

        `frame_bgr` is the preprocessed (downscaled) frame everything is detected
        and reported in — all bboxes come back in its coordinates.

        `now` is the frame's timestamp in seconds and drives dwell accounting.
        It defaults to the wall clock, which is right for a live camera. Offline
        video must pass frame_idx / fps instead — batch processing runs faster
        than real time, so the wall clock would under-report every dwell.

        `source_frame` is the original full-resolution frame, used only to cut a
        sharper face crop for age/gender. Pass it whenever you have it.
        """
        self._frame_idx += 1
        tracks = self.tracker.update(frame_bgr)

        now = time.time() if now is None else now
        self._retire({t.track_id for t in tracks}, now)

        metas = []
        for t in tracks:
            self._last_seen[t.track_id] = now
            face_bgr = crop_face(frame_bgr, t.bbox)
            face_rgb = to_rgb(face_bgr) if face_bgr.size > 0 else None

            pose = self.head_pose.estimate(face_rgb) if face_rgb is not None else None
            if self.enable_attributes:
                age_group, gender = self._age_gender_voted(
                    t.track_id, self._attribute_crop(frame_bgr, source_frame, t.bbox)
                )
            else:
                age_group, gender = None, None

            attentive_now = is_attentive(pose) if pose is not None else False
            smoothed_att, dwell_time = self.dwell.update(t.track_id, attentive_now, now)

            metas.append(
                PersonMeta(
                    track_id=t.track_id,
                    bbox=t.bbox,
                    age_group=age_group,
                    gender=gender,
                    yaw=pose.yaw if pose else None,
                    pitch=pose.pitch if pose else None,
                    roll=pose.roll if pose else None,
                    attention=int(smoothed_att),
                    dwell_time=dwell_time,
                )
            )
        return metas



