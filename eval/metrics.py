"""Basic aggregate metrics for comparing experiment event files."""

from __future__ import annotations

from collections.abc import Iterable


def summarize_sessions(events: Iterable[dict]) -> dict[str, float | int]:
    rows = list(events)
    impressions = sum(bool(row["is_impression"]) for row in rows)
    viewers = sum(bool(row["is_viewer"]) for row in rows)
    engaged = sum(bool(row["is_engaged"]) for row in rows)
    attention = sum(float(row["attention_seconds"]) for row in rows)
    presence = sum(float(row["presence_seconds"]) for row in rows)
    return {
        "sessions": len(rows),
        "impressions": impressions,
        "viewers": viewers,
        "engaged_viewers": engaged,
        "total_presence_seconds": round(presence, 3),
        "total_attention_seconds": round(attention, 3),
        "attention_rate": round(attention / presence, 4) if presence else 0.0,
    }
