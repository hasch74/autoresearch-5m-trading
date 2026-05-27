from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.papertrader.simulator import PaperTrader
from src.types import Bar, Direction, Signal


def _bar_at(ts: datetime, close: float, *, high: float | None = None, low: float | None = None) -> Bar:
    actual_high = high if high is not None else close + 0.2
    actual_low = low if low is not None else close - 0.2
    return Bar(
        event_time=ts,
        available_time=ts + timedelta(seconds=5),
        symbol="SPY",
        open=close,
        high=actual_high,
        low=actual_low,
        close=close,
        volume=1000,
    )


class _AlwaysLongHypothesis:
    hypothesis_id = "h_test_a"

    def generate_signals(self, bars, features):
        bar = bars[-1]
        return [
            Signal(
                hypothesis_id=self.hypothesis_id,
                symbol=bar.symbol,
                bar_time=bar.event_time,
                direction=Direction.LONG,
                confidence=0.6,
                stop_distance_atr=1.0,
                take_profit_distance_atr=1.5,
                max_hold_bars=3,
            )
        ]


class _SecondHypothesis:
    hypothesis_id = "h_test_b"

    def __init__(self) -> None:
        self.calls = 0

    def generate_signals(self, bars, features):
        self.calls += 1
        bar = bars[-1]
        return [
            Signal(
                hypothesis_id=self.hypothesis_id,
                symbol=bar.symbol,
                bar_time=bar.event_time,
                direction=Direction.LONG,
                confidence=0.5,
                stop_distance_atr=1.0,
                take_profit_distance_atr=1.0,
                max_hold_bars=2,
            )
        ]


def test_paper_trader_exits_and_persists_fills(tmp_path) -> None:
    second = _SecondHypothesis()
    trader = PaperTrader(
        hypotheses={"h_test_a": _AlwaysLongHypothesis(), "h_test_b": second},
        paper_dir=tmp_path,
    )

    entry_bar = _bar_at(datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc), 100.0)
    features = {"atr_14": 1.0}

    assert trader.process_bar(entry_bar, features) == []
    assert second.calls == 0

    stop_bar = _bar_at(
        entry_bar.event_time + timedelta(minutes=5),
        99.4,
        high=99.8,
        low=98.7,
    )
    fills = trader.process_bar(stop_bar, features)

    assert len(fills) == 1
    assert fills[0].exit_reason == "stop"
    assert fills[0].hypothesis_id == "h_test_a"

    persisted = pd.read_parquet(tmp_path / "fills.parquet")
    assert len(persisted) == 1
    assert persisted.loc[0, "exit_reason"] == "stop"