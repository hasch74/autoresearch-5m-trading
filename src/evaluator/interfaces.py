"""Evaluator interfaces."""

from typing import Any, Protocol


class Evaluator(Protocol):
    """Evaluates hypothesis results against promotion gates."""

    def score(self, run_result: Any) -> float:
        """Return a composite score."""
