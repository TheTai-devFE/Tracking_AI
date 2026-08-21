from __future__ import annotations

import unittest

from eval.metrics import summarize_sessions


class MetricsTest(unittest.TestCase):
    def test_summarizes_session_funnel(self) -> None:
        summary = summarize_sessions(
            [
                {
                    "is_impression": True,
                    "is_viewer": True,
                    "is_engaged": False,
                    "presence_seconds": 4,
                    "attention_seconds": 2,
                },
                {
                    "is_impression": True,
                    "is_viewer": False,
                    "is_engaged": False,
                    "presence_seconds": 2,
                    "attention_seconds": 0,
                },
            ]
        )
        self.assertEqual(summary["sessions"], 2)
        self.assertEqual(summary["viewers"], 1)
        self.assertEqual(summary["attention_rate"], 0.3333)


if __name__ == "__main__":
    unittest.main()
