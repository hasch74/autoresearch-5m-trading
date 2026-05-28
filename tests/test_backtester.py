from datetime import datetime, timedelta, timezone

import pandas as pd

from src.backtester.engine import run_backtest
from src.types import Bar, Direction, Signal


class _StopHypothesis:
    hypothesis_id = "h_stop"

    def generate_signals(self, bars, features):
        if len(bars) == 1:
            return [
                Signal(
                    hypothesis_id=self.hypothesis_id,
                    symbol=bars[-1].symbol,
                    bar_time=bars[-1].event_time,
                    direction=Direction.LONG,
                    confidence=0.8,
                    stop_distance_atr=1.0,
                    take_profit_distance_atr=4.0,
                    max_hold_bars=5,
                )
            ]
        return []


class _TakeProfitHypothesis:
    hypothesis_id = "h_tp"

    def generate_signals(self, bars, features):
        if len(bars) == 1:
            return [
                Signal(
                    hypothesis_id=self.hypothesis_id,
                    symbol=bars[-1].symbol,
                    bar_time=bars[-1].event_time,
                    direction=Direction.LONG,
                    confidence=0.8,
                    stop_distance_atr=1.0,
                    take_profit_distance_atr=4.0,
                    max_hold_bars=5,
                )
            ]
        return []


class _FlatHoldHypothesis:
    hypothesis_id = "h_flat"

    def __init__(self, entry_indices: set[int]) -> None:
        self.entry_indices = entry_indices

    def generate_signals(self, bars, features):
        if len(bars) - 1 in self.entry_indices:
            return [
                Signal(
                    hypothesis_id=self.hypothesis_id,
                    symbol=bars[-1].symbol,
                    bar_time=bars[-1].event_time,
                    direction=Direction.LONG,
                    confidence=0.8,
                    stop_distance_atr=10.0,
                    take_profit_distance_atr=10.0,
                    max_hold_bars=1,
                )
            ]
        return []


def _build_df(second_high: float, second_low: float) -> pd.DataFrame:
    t0 = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=5)
    rows = [
        {
            "event_time": t0,
            "available_time": t0 + timedelta(seconds=5),
            "symbol": "SPY",
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1000,
            "atr_14": 1.0,
        },
        {
            "event_time": t1,
            "available_time": t1 + timedelta(seconds=5),
            "symbol": "SPY",
            "open": 100.0,
            "high": second_high,
            "low": second_low,
            "close": 100.0,
            "volume": 1200,
            "atr_14": 1.0,
        },
    ]
    return pd.DataFrame(rows)


def test_backtester_known_trade_stop() -> None:
    df = _build_df(second_high=100.2, second_low=98.0)
    result = run_backtest(df, _StopHypothesis())

    assert result.total_trades == 1
    assert result.win_rate == 0.0
    assert result.net_pnl < 0.0


def test_backtester_known_trade_take_profit() -> None:
    df = _build_df(second_high=106.0, second_low=99.2)
    result = run_backtest(df, _TakeProfitHypothesis())

    assert result.total_trades == 1
    assert result.net_pnl > 0.0


def test_backtester_slippage_sensitivity_uses_execution_cost_stress() -> None:
    df = _build_df(second_high=100.1, second_low=99.9)
    result = run_backtest(df, _FlatHoldHypothesis({0}))

    assert result.total_trades == 1
    assert result.slippage_sensitivity == 0.0521


def test_backtester_aggregates_worst_day_and_intraday_drawdown_by_session() -> None:
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    day_two = start + timedelta(days=3)
    rows = []
    for ts in [
        start,
        start + timedelta(minutes=5),
        start + timedelta(minutes=10),
        start + timedelta(minutes=15),
        day_two,
        day_two + timedelta(minutes=5),
    ]:
        rows.append(
            {
                "event_time": ts,
                "available_time": ts + timedelta(seconds=5),
                "symbol": "SPY",
                "open": 100.0,
                "high": 100.2,
                "low": 99.8,
                "close": 100.0,
                "volume": 1000,
                "atr_14": 1.0,
            }
        )
    df = pd.DataFrame(rows)

    result = run_backtest(df, _FlatHoldHypothesis({0, 2, 4}))

    assert result.total_trades == 3
    assert result.worst_day == -4.22
    assert result.max_intraday_drawdown == 4.22
