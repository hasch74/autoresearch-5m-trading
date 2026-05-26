"""Paper trader interfaces."""

from typing import Protocol

from src.types import Bar


class PaperTrader(Protocol):
    """Executes simulated orders and tracks paper positions."""

    def process_bar(self, bar: Bar) -> None:
        """Process one new market bar."""
