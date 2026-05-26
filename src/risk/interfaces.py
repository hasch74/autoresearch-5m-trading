"""Risk management interfaces."""

from typing import Any, Protocol


class RiskManager(Protocol):
    """Applies risk limits to candidate orders."""

    def validate_order(self, order: Any) -> bool:
        """Return whether an order respects risk rules."""
