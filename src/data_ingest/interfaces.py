"""Data ingest interfaces."""

from typing import Protocol, Sequence


class DataIngestor(Protocol):
    """Loads raw market data into normalized storage."""

    def sync(self, symbols: Sequence[str]) -> None:
        """Sync data for a set of symbols."""
