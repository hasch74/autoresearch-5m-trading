"""Evaluator interfaces."""

from typing import Protocol

from src.types import EvalResult


class Evaluator(Protocol):
    """Evaluates hypothesis results against promotion gates."""

    def score(self, run_result: EvalResult) -> float:
        """Return a composite score."""
