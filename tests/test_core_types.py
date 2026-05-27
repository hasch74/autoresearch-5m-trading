from datetime import datetime, timezone

import pytest

from src.types import Bar, Direction, Signal


def test_bar_rejects_available_time_before_event_time() -> None:
    event = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    available = datetime(2026, 1, 1, 14, 59, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        Bar(
            event_time=event,
            available_time=available,
            symbol="SPY",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000,
        )


def test_signal_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        Signal(
            hypothesis_id="h_test",
            symbol="SPY",
            bar_time=datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc),
            direction=Direction.LONG,
            confidence=1.5,
            stop_distance_atr=1.0,
            take_profit_distance_atr=2.0,
            max_hold_bars=3,
        )
