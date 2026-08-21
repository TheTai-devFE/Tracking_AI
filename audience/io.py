"""Serialization helpers for audience events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

from .contracts import AudienceSession


class JsonlEventWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: TextIO | None = None

    def __enter__(self) -> JsonlEventWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="\n")
        return self

    def write(self, events: list[AudienceSession]) -> None:
        if self._file is None:
            raise RuntimeError("JsonlEventWriter must be used as a context manager")
        for event in events:
            self._file.write(json.dumps(event.as_dict(), ensure_ascii=True) + "\n")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
