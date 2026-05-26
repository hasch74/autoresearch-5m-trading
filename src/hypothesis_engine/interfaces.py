"""Hypothesis engine interfaces."""

from typing import Protocol, Sequence


class HypothesisEngine(Protocol):
    """Creates and mutates hypothesis candidates."""

    def propose(self) -> Sequence[str]:
        """Return one or more new hypothesis ids."""
