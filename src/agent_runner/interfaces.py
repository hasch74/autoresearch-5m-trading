"""Agent runner interfaces."""

from typing import Protocol


class AgentRunner(Protocol):
    """Coordinates autonomous research runs."""

    def run_once(self) -> None:
        """Execute one research cycle."""
