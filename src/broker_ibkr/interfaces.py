"""IBKR broker adapter interfaces."""

from typing import Any, Protocol


class BrokerGateway(Protocol):
    """Broker connectivity boundary."""

    def submit(self, order: Any) -> str:
        """Submit an order and return broker order id."""
