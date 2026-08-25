"""Construct a provider selected by the experiment CLI."""

from __future__ import annotations

from .base import VisionProvider


def create_provider(name: str = "uniface", enable_attributes: bool = False) -> VisionProvider:
    if name == "uniface":
        from .uniface_provider import UniFaceProvider

        return UniFaceProvider(enable_attributes=enable_attributes)
    raise ValueError(f"Unsupported provider: {name}. Only 'uniface' is supported.")
