from __future__ import annotations

import unittest

from audience import AudienceSessionTracker, PersonObservation, SessionConfig


def observation(
    timestamp: float,
    *,
    attentive: bool,
    track_id: int = 1,
    age_group: str | None = None,
    gender: str | None = None,
) -> PersonObservation:
    return PersonObservation(
        provider="test",
        track_id=track_id,
        timestamp=timestamp,
        bbox=(0, 0, 100, 100),
        attentive=attentive,
        age_group=age_group,
        gender=gender,
    )


class AudienceSessionTrackerTest(unittest.TestCase):
    def make_tracker(self, **overrides) -> AudienceSessionTracker:
        values = {
            "min_presence_seconds": 1.0,
            "min_attention_seconds": 1.0,
            "engaged_seconds": 3.0,
            "track_gap_tolerance_seconds": 1.0,
            "session_expiry_seconds": 2.0,
            "attention_on_frames": 2,
            "attention_off_frames": 2,
        }
        values.update(overrides)
        config = SessionConfig(**values)
        return AudienceSessionTracker("standee", "campaign", "creative", config)

    def test_builds_viewer_session_and_counts_look_away(self) -> None:
        tracker = self.make_tracker()
        for timestamp, attentive in [(0, True), (1, True), (2, True), (3, False), (4, False)]:
            tracker.update([observation(timestamp, attentive=attentive)], timestamp)

        event = tracker.flush()[0]
        self.assertEqual(event.presence_seconds, 4.0)
        self.assertEqual(event.attention_seconds, 3.0)
        self.assertEqual(event.look_away_count, 1)
        self.assertTrue(event.is_impression)
        self.assertTrue(event.is_viewer)
        self.assertTrue(event.is_engaged)
        self.assertEqual(event.attention_ratio, 0.75)

    def test_creative_rotation_keeps_one_session_for_a_standing_viewer(self) -> None:
        tracker = self.make_tracker(attention_on_frames=1, attention_off_frames=1)
        for timestamp in (0.0, 0.5, 1.0):
            tracker.update([observation(timestamp, attentive=True)], timestamp)

        self.assertEqual(tracker.switch_creative("creative-b"), [])

        for timestamp in (1.5, 2.0, 2.5, 3.0, 3.5):
            tracker.update([observation(timestamp, attentive=True)], timestamp)

        events = tracker.flush()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].presence_seconds, 3.5)
        # 1.0 s on the first creative against 2.5 s on the second.
        self.assertEqual(events[0].creative_id, "creative-b")

    def test_expires_missing_track_without_adding_gap_time(self) -> None:
        tracker = self.make_tracker(attention_on_frames=1, attention_off_frames=1)
        tracker.update([observation(0, attentive=True)], 0)
        tracker.update([observation(0.5, attentive=True)], 0.5)
        events = tracker.update([], 3.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].presence_seconds, 0.5)
        self.assertEqual(events[0].attention_seconds, 0.5)

    def test_keeps_provider_track_ids_separate(self) -> None:
        tracker = self.make_tracker()
        first = observation(0, attentive=False)
        second = PersonObservation(provider="other", track_id=1, timestamp=0, bbox=(0, 0, 10, 10))
        tracker.update([first, second], 0)
        self.assertEqual(len(tracker.flush()), 2)

    def test_demographics_use_temporal_agreement(self) -> None:
        tracker = self.make_tracker()
        samples = [
            ("25-34", "F"),
            ("25-34", "Female"),
            ("35-44", "Male"),
        ]
        for timestamp, (age_group, gender) in enumerate(samples):
            tracker.update(
                [observation(timestamp, attentive=False, age_group=age_group, gender=gender)],
                timestamp,
            )
        event = tracker.flush()[0]
        self.assertEqual(event.age_group, "25-34")
        self.assertAlmostEqual(event.age_confidence, 2 / 3)
        self.assertEqual(event.gender, "female")
        self.assertAlmostEqual(event.gender_confidence, 2 / 3)

    def test_rejects_mismatched_frame_timestamp(self) -> None:
        tracker = self.make_tracker()
        with self.assertRaises(ValueError):
            tracker.update([observation(1, attentive=False)], 2)


if __name__ == "__main__":
    unittest.main()
