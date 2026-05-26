"""Backtester interfaces."""

from typing import Any, Protocol


class Backtester(Protocol):
    """Runs historical strategy evaluations."""

    def run(self, hypothesis_id: str) -> Any:
        """Execute a backtest for one hypothesis."""
