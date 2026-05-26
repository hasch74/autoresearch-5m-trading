"""Paper trader interfaces."""

from typing import Any, Protocol


class PaperTrader(Protocol):
    """Executes simulated orders and tracks paper positions."""

    def process_bar(self, bar: Any) -> None:
        """Process one new market bar."""
