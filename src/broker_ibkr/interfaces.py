"""IBKR broker adapter interfaces."""

from typing import Protocol

from src.types import Order


class BrokerGateway(Protocol):
    """Broker connectivity boundary."""

    def submit(self, order: Order) -> str:
        """Submit an order and return broker order id."""
