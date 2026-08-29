"""Privacy-preserving UniFace provider without recognition or embeddings."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from audience.contracts import PersonObservation
from audience.demographics import map_age_group, normalize_gender
from configs import CFG, MODELS_DIR


class UniFaceProvider:
    name = "uniface"

    def __init__(self, attribute_every_n: int = 10, enable_attributes: bool = False) -> None:
        self.attribute_every_n = max(1, attribute_every_n)
        self.enable_attributes = enable_attributes
        self._frame_index = 0
        self._detector = None
        self._tracker = None
        self._head_pose = None
        self._age_gender = None
        self._gaze = None
        self._emotion = None
        self._attribute_samples: dict[int, deque[tuple[float, str]]] = defaultdict(
            lambda: deque(maxlen=15)
        )
        self._emotion_samples: dict[int, deque[tuple[str, float]]] = defaultdict(
            lambda: deque(maxlen=15)
        )
        self._distance_samples: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=15)
        )

    def _ensure_models(self) -> None:
        if self._detector is not None:
            return
        from uniface import AgeGender, BYTETracker, HeadPose, MobileGaze, SCRFD, set_cache_dir
        from uniface.constants import HeadPoseWeights, SCRFDWeights

        set_cache_dir(str(MODELS_DIR / "uniface"))
        providers = ["CPUExecutionProvider"]
        self._detector = SCRFD(
            model_name=SCRFDWeights.SCRFD_500M_KPS,
            confidence_threshold=CFG.conf_threshold,
            input_size=(CFG.process_long_side, CFG.process_long_side),
            providers=providers,
        )
        # low_thresh stays at the library default (0.1), below the detector's own
        # floor, so every box SCRFD returns lands in one of ByteTrack's two bands.
        self._tracker = BYTETracker(
            track_thresh=CFG.track_high_threshold,
            track_buffer=CFG.track_buffer_frames,
        )
        self._head_pose = HeadPose(model_name=HeadPoseWeights.MOBILENET_V3_SMALL, providers=providers)
        if self.enable_attributes:
            self._age_gender = AgeGender(providers=providers)
            try:
                self._gaze = MobileGaze(providers=providers)
            except Exception:
                self._gaze = None
            try:
                from uniface import Emotion
                self._emotion = Emotion()
            except Exception:
                self._emotion = None

    def warmup(self) -> None:
        self._ensure_models()

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1, y1 = np.maximum(box_a[:2], box_b[:2])
        x2, y2 = np.minimum(box_a[2:], box_b[2:])
        intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
        area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _crop(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        return frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]

    @staticmethod
    def _face_for_source(face, processed_frame: np.ndarray, source_frame: np.ndarray):
        """Map a detected Face back to the original frame for sharper attributes."""
        if processed_frame.shape[:2] == source_frame.shape[:2]:
            return face

        from uniface.types import Face

        scale_x = source_frame.shape[1] / processed_frame.shape[1]
        scale_y = source_frame.shape[0] / processed_frame.shape[0]
        bbox = face.bbox.astype(np.float32).copy()
        bbox[[0, 2]] *= scale_x
        bbox[[1, 3]] *= scale_y
        landmarks = face.landmarks.astype(np.float32).copy()
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y
        return Face(bbox=bbox, confidence=face.confidence, landmarks=landmarks)

    def observe(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        source_frame: np.ndarray | None = None,
    ) -> list[PersonObservation]:
        self._ensure_models()
        self._frame_index += 1
        faces = self._detector.detect(frame_bgr)
        detections = np.asarray(
            [[*face.bbox.astype(float), float(face.confidence)] for face in faces],
            dtype=np.float32,
        ).reshape((-1, 5))
        tracks = self._tracker.update(detections)

        observations = []
        for tracked in tracks:
            bbox = tracked[:4]
            track_id = int(tracked[4])
            matched_face = max(faces, key=lambda face: self._iou(bbox, face.bbox), default=None)
            if matched_face is not None and self._iou(bbox, matched_face.bbox) < 0.1:
                matched_face = None
            confidence = float(matched_face.confidence) if matched_face is not None else None
            face_crop = self._crop(frame_bgr, bbox)
            if face_crop.size == 0:
                continue

            pose = self._head_pose.estimate(face_crop)
            gaze_yaw = float(pose.yaw) if pose.yaw is not None else 0.0
            gaze_pitch = float(pose.pitch) if pose.pitch is not None else 0.0
            if self._gaze is not None:
                try:
                    g_res = self._gaze.estimate(face_crop)
                    gaze_yaw = float(np.degrees(g_res.yaw))
                    gaze_pitch = float(np.degrees(g_res.pitch))
                except Exception:
                    pass

            if abs(gaze_yaw) <= 18.0 and abs(gaze_pitch) <= 18.0:
                gaze_direction = "direct"
            elif gaze_yaw < -18.0:
                gaze_direction = "left"
            elif gaze_yaw > 18.0:
                gaze_direction = "right"
            elif gaze_pitch > 18.0:
                gaze_direction = "down"
            else:
                gaze_direction = "up"

            attentive = (gaze_direction == "direct") or (
                abs(pose.yaw) < CFG.yaw_threshold_deg and abs(pose.pitch) < CFG.pitch_threshold_deg
            )

            # Estimate standing distance in meters based on face width
            fw = frame_bgr.shape[1]
            bw = max(1.0, float(bbox[2] - bbox[0]))
            distance_m = round(float(np.clip((0.85 * fw * 0.16) / bw, 0.4, 6.0)), 1)

            if matched_face is not None and (
                not self._attribute_samples[track_id] or self._frame_index % self.attribute_every_n == 0
            ):
                attribute_frame = source_frame if source_frame is not None else frame_bgr
                attribute_face = (
                    self._face_for_source(matched_face, frame_bgr, source_frame)
                    if source_frame is not None
                    else matched_face
                )
                if self._age_gender is not None:
                    try:
                        attribute = self._age_gender.predict(attribute_frame, attribute_face)
                        self._attribute_samples[track_id].append((float(attribute.age), attribute.sex))
                    except Exception:
                        pass
                if self._emotion is not None:
                    try:
                        emo = self._emotion.predict(attribute_frame, attribute_face)
                        self._emotion_samples[track_id].append((emo.emotion.lower(), float(emo.confidence)))
                    except Exception:
                        pass

            samples = self._attribute_samples.get(track_id, [])
            age = float(np.median([sample[0] for sample in samples])) if samples else None
            gender = None
            if samples:
                labels = [sample[1] for sample in samples]
                gender = normalize_gender(max(set(labels), key=labels.count))

            emo_samples = self._emotion_samples.get(track_id, [])
            emotion = None
            emotion_confidence = None
            if emo_samples:
                emo_labels = [s[0] for s in emo_samples]
                emotion = max(set(emo_labels), key=emo_labels.count)
                emotion_confidence = max([s[1] for s in emo_samples if s[0] == emotion], default=1.0)

            observations.append(
                PersonObservation(
                    provider=self.name,
                    track_id=track_id,
                    timestamp=timestamp,
                    bbox=tuple(int(value) for value in bbox),
                    detection_confidence=confidence,
                    yaw=pose.yaw,
                    pitch=pose.pitch,
                    roll=pose.roll,
                    attentive=attentive,
                    age=age,
                    age_group=map_age_group(age) if age is not None else None,
                    attribute_sample_count=len(samples),
                    gender=gender,
                    emotion=emotion,
                    emotion_confidence=emotion_confidence,
                    distance_m=distance_m,
                    gaze_direction=gaze_direction,
                    gaze_yaw=gaze_yaw,
                    gaze_pitch=gaze_pitch,
                )
            )
        return observations

    def reset(self) -> None:
        if self._tracker is not None:
            try:
                from uniface.tracking.bytetrack.basetrack import BaseTrack
                saved_count = BaseTrack._count
                self._tracker.reset()
                BaseTrack._count = saved_count
            except Exception:
                self._tracker.reset()
        self._frame_index = 0
        self._attribute_samples.clear()
        self._emotion_samples.clear()
        self._distance_samples.clear()
