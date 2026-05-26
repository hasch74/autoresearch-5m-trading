"""Backtester interfaces."""

from typing import Protocol

from src.types import EvalResult


class Backtester(Protocol):
    """Runs historical strategy evaluations."""

    def run(self, hypothesis_id: str) -> EvalResult:
        """Execute a backtest for one hypothesis."""
