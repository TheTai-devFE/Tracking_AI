"""Benchmark UniFace and MiVOLO age/gender estimators on identical face crops."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Protocol

import cv2
import numpy as np

from audience.demographics import map_age_group, normalize_gender
from configs import CFG, MODELS_DIR


@dataclass(frozen=True, slots=True)
class GroundTruthSample:
    image_path: Path
    age: float
    gender: str


@dataclass(frozen=True, slots=True)
class Prediction:
    age: float
    gender: str
    gender_confidence: float | None


class AttributeEstimator(Protocol):
    name: str

    def predict(self, face_bgr: np.ndarray) -> Prediction: ...


class UniFaceEstimator:
    name = "uniface"

    def __init__(self) -> None:
        from uniface import AgeGender, set_cache_dir

        set_cache_dir(str(MODELS_DIR / "uniface"))
        self._model = AgeGender(providers=["CPUExecutionProvider"])

    def predict(self, face_bgr: np.ndarray) -> Prediction:
        from uniface.types import Face

        height, width = face_bgr.shape[:2]
        face = Face(
            bbox=np.asarray([0, 0, width, height], dtype=np.float32),
            confidence=1.0,
            landmarks=np.empty((0, 2), dtype=np.float32),
        )
        result = self._model.predict(face_bgr, face)
        return Prediction(
            age=float(result.age),
            gender=normalize_gender(result.sex) or "unknown",
            gender_confidence=None,
        )


def load_manifest(path: Path, limit: int | None = None) -> list[GroundTruthSample]:
    samples: list[GroundTruthSample] = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"image", "age", "gender"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Manifest must contain columns: image,age,gender")
        for line_number, row in enumerate(reader, start=2):
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            gender = normalize_gender(row["gender"])
            if gender is None:
                raise ValueError(f"Unsupported gender at manifest line {line_number}: {row['gender']!r}")
            try:
                age = float(row["age"])
            except ValueError as error:
                raise ValueError(f"Invalid age at manifest line {line_number}: {row['age']!r}") from error
            samples.append(GroundTruthSample(image_path.resolve(), age, gender))
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError("Manifest contains no samples")
    return samples


def compute_summary(rows: list[dict], total_samples: int) -> dict[str, float | int]:
    valid = [row for row in rows if not row.get("error")]
    failures = total_samples - len(valid)
    if not valid:
        return {
            "samples": total_samples,
            "successful": 0,
            "failures": failures,
            "failure_rate": 1.0,
        }

    absolute_errors = [float(row["absolute_age_error"]) for row in valid]
    latencies = [float(row["latency_ms"]) for row in valid]
    return {
        "samples": total_samples,
        "successful": len(valid),
        "failures": failures,
        "failure_rate": round(failures / total_samples, 4),
        "age_mae": round(sum(absolute_errors) / len(absolute_errors), 4),
        "age_median_absolute_error": round(median(absolute_errors), 4),
        "age_cs_at_5": round(sum(error <= 5 for error in absolute_errors) / len(valid), 4),
        "age_group_accuracy": round(sum(row["age_group_correct"] for row in valid) / len(valid), 4),
        "gender_accuracy": round(sum(row["gender_correct"] for row in valid) / len(valid), 4),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 4),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 4),
    }


def benchmark(
    estimator: AttributeEstimator,
    samples: list[GroundTruthSample],
) -> tuple[list[dict], dict[str, float | int]]:
    rows: list[dict] = []
    for sample in samples:
        image = cv2.imread(str(sample.image_path))
        if image is None:
            rows.append(
                {
                    "model": estimator.name,
                    "image": str(sample.image_path),
                    "true_age": sample.age,
                    "true_gender": sample.gender,
                    "error": "cannot_read_image",
                }
            )
            continue

        started = time.perf_counter()
        try:
            prediction = estimator.predict(image)
            latency_ms = (time.perf_counter() - started) * 1000
            true_group = map_age_group(sample.age)
            predicted_group = map_age_group(prediction.age)
            rows.append(
                {
                    "model": estimator.name,
                    "image": str(sample.image_path),
                    "true_age": sample.age,
                    "predicted_age": round(prediction.age, 3),
                    "absolute_age_error": round(abs(prediction.age - sample.age), 3),
                    "true_age_group": true_group,
                    "predicted_age_group": predicted_group,
                    "age_group_correct": predicted_group == true_group,
                    "true_gender": sample.gender,
                    "predicted_gender": prediction.gender,
                    "gender_confidence": prediction.gender_confidence,
                    "gender_correct": prediction.gender == sample.gender,
                    "latency_ms": round(latency_ms, 3),
                    "error": "",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "model": estimator.name,
                    "image": str(sample.image_path),
                    "true_age": sample.age,
                    "true_gender": sample.gender,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return rows, compute_summary(rows, len(samples))


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "image",
        "true_age",
        "predicted_age",
        "absolute_age_error",
        "true_age_group",
        "predicted_age_group",
        "age_group_correct",
        "true_gender",
        "predicted_gender",
        "gender_confidence",
        "gender_correct",
        "latency_ms",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("uniface",), default=["uniface"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/age-gender-benchmark"))
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_manifest(args.manifest, args.limit)
    estimators: list[AttributeEstimator] = []
    if "uniface" in args.models:
        estimators.append(UniFaceEstimator())

    all_rows: list[dict] = []
    summaries: dict[str, dict[str, float | int]] = {}
    for estimator in estimators:
        rows, summary = benchmark(estimator, samples)
        all_rows.extend(rows)
        summaries[estimator.name] = summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.output_dir / "predictions.csv", all_rows)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print(f"Predictions: {args.output_dir / 'predictions.csv'}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
