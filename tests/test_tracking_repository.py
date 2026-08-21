from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audience.contracts import AudienceSession
from server.database import TrackingRepository, database_url_from_environment


def session(session_id: str, *, viewer: bool = True) -> AudienceSession:
    return AudienceSession(
        schema_version=1,
        session_id=session_id,
        standee_id="ST-001",
        campaign_id="CMP-001",
        creative_id="AD-001",
        provider="uniface",
        provider_track_id=1,
        started_at=100.0,
        ended_at=108.0,
        presence_seconds=8.0,
        attention_seconds=4.0 if viewer else 0.0,
        attention_ratio=0.5 if viewer else 0.0,
        look_away_count=1,
        is_impression=True,
        is_viewer=viewer,
        is_engaged=False,
        age_group="unknown",
        age_confidence=None,
        gender="unknown",
        gender_confidence=None,
    )


class TrackingRepositoryTest(unittest.TestCase):
    def test_saves_idempotently_and_builds_overview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = TrackingRepository(Path(directory) / "tracking.db")
            repository.initialize()
            self.assertEqual(repository.save_sessions([session("one"), session("two", viewer=False)]), 2)
            self.assertEqual(repository.save_sessions([session("one")]), 0)

            overview = repository.overview(standee_id="ST-001")
            self.assertEqual(overview["sessions"], 2)
            self.assertEqual(overview["impressions"], 2)
            self.assertEqual(overview["viewers"], 1)
            self.assertEqual(overview["attention_rate"], 0.25)

    def test_filters_recent_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = TrackingRepository(Path(directory) / "tracking.db")
            repository.initialize()
            repository.save_sessions([session("one")])
            self.assertEqual(repository.recent_sessions("ST-001")[0]["session_id"], "one")
            self.assertEqual(repository.recent_sessions("ST-OTHER"), [])

    def test_timeline_groups_sessions_by_utc_hour(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = TrackingRepository(Path(directory) / "tracking.db")
            repository.initialize()
            current = session("one")
            current = AudienceSession(**{**current.as_dict(), "started_at": 4_102_444_800.0})
            repository.save_sessions([current])

            timeline = repository.timeline("ST-001", hours=1_000_000)
            self.assertEqual(timeline[0]["bucket"], "2100-01-01T00:00:00Z")
            self.assertEqual(timeline[0]["viewers"], 1)

    def test_exposes_database_backend_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = TrackingRepository(Path(directory) / "tracking.db")
            self.assertEqual(repository.backend, "sqlite")
            self.assertIn("tracking.db", repository.display_url)

    def test_normalizes_hosted_postgresql_url_and_masks_password(self) -> None:
        with patch.dict("os.environ", {"DATABASE_URL": "postgres://user:secret@db.example/tracking"}):
            database_url = database_url_from_environment(Path("."))
        repository = TrackingRepository(database_url)

        self.assertTrue(database_url.startswith("postgresql+psycopg://"))
        self.assertEqual(repository.backend, "postgresql")
        self.assertNotIn("secret", repository.display_url)
        self.assertIn("***", repository.display_url)


if __name__ == "__main__":
    unittest.main()
