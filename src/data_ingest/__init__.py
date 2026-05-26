"""src.data_ingest package."""

from .eodhd import fetch_bars, sync_universe
from .interfaces import DataIngestor

__all__ = ["DataIngestor", "fetch_bars", "sync_universe"]
