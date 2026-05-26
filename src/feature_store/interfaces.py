"""Feature store interfaces."""

from typing import Any, Mapping, Protocol


class FeatureStore(Protocol):
    """Reads and writes feature sets."""

    def write_features(self, key: str, payload: Mapping[str, Any]) -> None:
        """Persist a feature payload."""
