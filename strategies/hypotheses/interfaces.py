"""Hypothesis strategy interfaces."""

from typing import Any, Protocol, Sequence


class Hypothesis(Protocol):
    """Contract for a research hypothesis strategy."""

    name: str
    version: str

    def required_features(self) -> Sequence[str]:
        """List required feature names."""

    def generate_signals(self, bars: Any, features: Any) -> Any:
        """Generate strategy signals for the current window."""
