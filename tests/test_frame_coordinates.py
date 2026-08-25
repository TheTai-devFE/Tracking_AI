from __future__ import annotations

import unittest

import numpy as np

from audience import PersonObservation
from preprocess import preprocess
from server.main import rescale_bbox_to_source


def observation(bbox: tuple[int, int, int, int]) -> PersonObservation:
    return PersonObservation(provider="test", track_id=1, timestamp=0.0, bbox=bbox)


def frame(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


class RescaleBboxToSourceTest(unittest.TestCase):
    def test_identity_when_preprocess_did_not_resize(self) -> None:
        source = frame(640, 480)
        processed = preprocess(source)
        self.assertEqual(processed.shape[:2], source.shape[:2])

        original = observation((100, 80, 200, 220))
        self.assertEqual(rescale_bbox_to_source(original, processed, source).bbox, (100, 80, 200, 220))

    def test_scales_back_up_when_preprocess_downscaled(self) -> None:
        """A 1280x720 frame is downscaled to 640x360, so boxes must double."""
        source = frame(1280, 720)
        processed = preprocess(source)
        self.assertEqual((processed.shape[1], processed.shape[0]), (640, 360))

        # A face centred in the processed frame must stay centred in the source.
        rescaled = rescale_bbox_to_source(observation((260, 120, 380, 300)), processed, source)
        self.assertEqual(rescaled.bbox, (520, 240, 760, 600))

    def test_full_frame_box_maps_to_full_source_frame(self) -> None:
        source = frame(1920, 1080)
        processed = preprocess(source)
        rescaled = rescale_bbox_to_source(
            observation((0, 0, processed.shape[1], processed.shape[0])), processed, source
        )
        self.assertEqual(rescaled.bbox, (0, 0, 1920, 1080))

    def test_preserves_every_other_field(self) -> None:
        source = frame(1280, 720)
        processed = preprocess(source)
        original = PersonObservation(
            provider="uniface",
            track_id=7,
            timestamp=123.5,
            bbox=(10, 20, 30, 40),
            yaw=5.0,
            attentive=True,
            age_group="25-34",
            gender="male",
        )
        rescaled = rescale_bbox_to_source(original, processed, source)
        self.assertEqual(rescaled.track_id, 7)
        self.assertEqual(rescaled.timestamp, 123.5)
        self.assertEqual(rescaled.yaw, 5.0)
        self.assertTrue(rescaled.attentive)
        self.assertEqual(rescaled.age_group, "25-34")
        self.assertEqual(rescaled.gender, "male")


if __name__ == "__main__":
    unittest.main()
