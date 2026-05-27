from datetime import datetime, timedelta, timezone

from strategies.hypotheses.h_0001_vwap_pullback import VwapPullbackRvol
from strategies.hypotheses.h_0002_opening_range_breakout import OpeningRangeBreakout
from src.types import Bar


def _bar_at(ts: datetime, close: float) -> Bar:
    return Bar(
        event_time=ts,
        available_time=ts + timedelta(seconds=5),
        symbol="SPY",
        open=close,
        high=close + 0.2,
        low=close - 0.2,
        close=close,
        volume=1000,
    )


def test_vwap_pullback_does_not_trigger_without_features() -> None:
    hyp = VwapPullbackRvol()
    bars = [_bar_at(datetime(2026, 1, 2, 15, 30, tzinfo=timezone.utc), 100.0)]

    signals = hyp.generate_signals(bars, {"vwap": 100.0})
    assert signals == []


def test_opening_range_breakout_triggers_after_or_period() -> None:
    hyp = OpeningRangeBreakout()
    hyp.or_bars = 4
    hyp.rvol_min = 1.0
    hyp.min_breakout_bps = 0.0
    hyp.min_atr_pct_of_price = 0.0

    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    bars = [_bar_at(start + timedelta(minutes=5 * i), 100.0 + i * 0.1) for i in range(5)]

    features = {
        "or_high": 100.2,
        "or_low": 99.5,
        "rvol_20": 1.5,
        "atr_14": 1.0,
    }

    signals = hyp.generate_signals(bars, features)
    assert len(signals) == 1


def test_vwap_pullback_can_use_session_gap_pct_after_open() -> None:
    hyp = VwapPullbackRvol()
    hyp.rvol_min = 1.0

    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    bars = [_bar_at(start + timedelta(minutes=5 * i), 100.0 + i * 0.1) for i in range(3)]

    features = {
        "vwap": bars[-1].close,
        "rvol_20": 1.4,
        "atr_14": 1.0,
        "session_gap_pct": 0.01,
        "ret_open_to_now": 0.002,
    }

    signals = hyp.generate_signals(bars, features)
    assert len(signals) == 1


def test_opening_range_breakout_translates_or_levels_to_atr_distances() -> None:
    hyp = OpeningRangeBreakout()
    hyp.or_bars = 4
    hyp.rvol_min = 1.0
    hyp.min_breakout_bps = 0.0
    hyp.min_atr_pct_of_price = 0.0
    hyp.min_stop_atr_mult = 0.5
    hyp.take_profit_range_mult = 2.0

    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    bars = [_bar_at(start + timedelta(minutes=5 * i), 100.0 + i * 0.1) for i in range(5)]

    features = {
        "or_high": 100.2,
        "or_low": 99.4,
        "rvol_20": 1.5,
        "atr_14": 1.0,
    }

    signals = hyp.generate_signals(bars, features)
    assert len(signals) == 1

    sig = signals[0]
    expected_stop_atr = max(hyp.min_stop_atr_mult, (bars[-1].close - features["or_low"]) / features["atr_14"])
    expected_tp_atr = (hyp.take_profit_range_mult * (features["or_high"] - features["or_low"])) / features["atr_14"]

    assert sig.stop_distance_atr == expected_stop_atr
    assert sig.take_profit_distance_atr == expected_tp_atr
