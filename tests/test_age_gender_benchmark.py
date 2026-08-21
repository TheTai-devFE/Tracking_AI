from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from eval.eval_age_gender import compute_summary, load_manifest


class AgeGenderBenchmarkTest(unittest.TestCase):
    def test_loads_relative_manifest_paths_and_normalizes_gender(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["image", "age", "gender"])
                writer.writeheader()
                writer.writerow({"image": "face.jpg", "age": "27", "gender": "F"})

            sample = load_manifest(manifest)[0]
            self.assertEqual(sample.image_path, (root / "face.jpg").resolve())
            self.assertEqual(sample.age, 27)
            self.assertEqual(sample.gender, "female")

    def test_computes_accuracy_and_latency_metrics(self) -> None:
        rows = [
            {
                "absolute_age_error": 2.0,
                "age_group_correct": True,
                "gender_correct": True,
                "latency_ms": 10.0,
                "error": "",
            },
            {
                "absolute_age_error": 8.0,
                "age_group_correct": False,
                "gender_correct": True,
                "latency_ms": 20.0,
                "error": "",
            },
        ]
        summary = compute_summary(rows, total_samples=2)
        self.assertEqual(summary["age_mae"], 5.0)
        self.assertEqual(summary["age_cs_at_5"], 0.5)
        self.assertEqual(summary["age_group_accuracy"], 0.5)
        self.assertEqual(summary["gender_accuracy"], 1.0)
        self.assertEqual(summary["mean_latency_ms"], 15.0)


if __name__ == "__main__":
    unittest.main()
