"""Summarize baseline and UniFace JSONL outputs from the same source video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import summarize_sessions


def load_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--uniface", required=True)
    args = parser.parse_args()

    summaries = {
        "baseline": summarize_sessions(load_jsonl(args.baseline)),
        "uniface": summarize_sessions(load_jsonl(args.uniface)),
    }
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
