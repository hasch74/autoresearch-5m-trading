"""Data ingest interfaces."""

from datetime import date
from typing import Protocol, Sequence


class DataIngestor(Protocol):
    """Loads raw market data into normalized storage."""

    def sync(
        self,
        symbols: Sequence[str],
        start: date | None = None,
        end: date | None = None,
    ) -> None:
        """Sync 5-minute bar data for *symbols* into data/normalized/."""
