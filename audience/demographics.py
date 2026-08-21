"""Canonical demographic labels used by every provider and API event."""

from __future__ import annotations


AGE_BINS: tuple[tuple[int, str], ...] = (
    (18, "0-17"),
    (25, "18-24"),
    (35, "25-34"),
    (45, "35-44"),
    (55, "45-54"),
    (200, "55+"),
)


def map_age_group(age: float) -> str:
    for upper, label in AGE_BINS:
        if age < upper:
            return label
    return AGE_BINS[-1][1]


def normalize_gender(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"f", "female", "woman"}:
        return "female"
    if normalized in {"m", "male", "man"}:
        return "male"
    return None
