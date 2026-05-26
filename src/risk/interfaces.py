"""Risk management interfaces."""

from typing import Protocol

from src.types import Order


class RiskManager(Protocol):
    """Applies risk limits to candidate orders."""

    def validate_order(self, order: Order) -> bool:
        """Return whether an order respects risk rules."""
