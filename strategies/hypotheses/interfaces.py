"""Hypothesis strategy interfaces."""

from typing import Protocol, Sequence

from src.types import Bar, Signal


class Hypothesis(Protocol):
    """Contract for a research hypothesis strategy."""

    name: str
    version: str

    def required_features(self) -> Sequence[str]:
        """List required feature names."""

    def generate_signals(self, bars: Sequence[Bar], features: dict) -> Sequence[Signal]:
        """Generate strategy signals for the current window."""
