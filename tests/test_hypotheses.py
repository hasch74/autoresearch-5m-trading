from datetime import datetime, timedelta, timezone

from strategies.hypotheses.h_0001_vwap_pullback import VwapPullbackRvol
from strategies.hypotheses.h_0002_opening_range_breakout import OpeningRangeBreakout
from strategies.hypotheses.h_0003_vwap_reversion import VwapReversionBaseline
from strategies.hypotheses.h_0004_prev_day_high_reclaim import PriorDayHighReclaimBaseline
from strategies.hypotheses.h_0005_vwap_reversion_selective import VwapReversionSelective
from strategies.hypotheses.h_0006_prev_day_high_reclaim_quality import PriorDayHighReclaimQuality
from strategies.hypotheses.h_0007_opening_range_continuation_confirmed import OpeningRangeContinuationConfirmed
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


def test_vwap_reversion_baseline_triggers_on_large_discount_to_vwap() -> None:
    hyp = VwapReversionBaseline()

    bar = _bar_at(datetime(2026, 1, 2, 19, 0, tzinfo=timezone.utc), 98.8)
    features = {
        "vwap": 100.0,
        "atr_14": 1.0,
        "rvol_20": 1.1,
        "minutes_to_close": 45,
    }

    signals = hyp.generate_signals([bar], features)
    assert len(signals) == 1
    assert signals[0].hypothesis_id == "h_0003"


def test_vwap_reversion_baseline_skips_late_day_or_high_rvol() -> None:
    hyp = VwapReversionBaseline()

    bar = _bar_at(datetime(2026, 1, 2, 20, 45, tzinfo=timezone.utc), 98.8)
    late_features = {
        "vwap": 100.0,
        "atr_14": 1.0,
        "rvol_20": 1.1,
        "minutes_to_close": 10,
    }
    crowded_features = {
        "vwap": 100.0,
        "atr_14": 1.0,
        "rvol_20": 1.6,
        "minutes_to_close": 45,
    }

    assert hyp.generate_signals([bar], late_features) == []
    assert hyp.generate_signals([bar], crowded_features) == []


def test_prior_day_high_reclaim_baseline_triggers_after_reclaim() -> None:
    hyp = PriorDayHighReclaimBaseline()

    start = datetime(2026, 1, 2, 14, 45, tzinfo=timezone.utc)
    bars = [
        _bar_at(start, 99.8),
        _bar_at(start + timedelta(minutes=5), 100.3),
    ]

    features = {
        "prior_day_high": 100.0,
        "atr_14": 1.0,
        "rvol_20": 1.2,
        "minutes_since_open": 20,
    }

    signals = hyp.generate_signals(bars, features)
    assert len(signals) == 1
    assert signals[0].hypothesis_id == "h_0004"


def test_prior_day_high_reclaim_baseline_requires_true_reclaim() -> None:
    hyp = PriorDayHighReclaimBaseline()

    start = datetime(2026, 1, 2, 14, 45, tzinfo=timezone.utc)
    bars = [
        _bar_at(start, 100.1),
        _bar_at(start + timedelta(minutes=5), 100.3),
    ]

    features = {
        "prior_day_high": 100.0,
        "atr_14": 1.0,
        "rvol_20": 1.2,
        "minutes_since_open": 20,
    }

    assert hyp.generate_signals(bars, features) == []


def test_vwap_reversion_selective_requires_clean_context() -> None:
    hyp = VwapReversionSelective()

    bar = _bar_at(datetime(2026, 1, 2, 18, 0, tzinfo=timezone.utc), 98.6)
    features = {
        "vwap": 100.0,
        "atr_14": 1.0,
        "rvol_20": 1.0,
        "volume_zscore_20": 0.3,
        "minutes_since_open": 60,
        "minutes_to_close": 90,
        "ret_open_to_now": -0.005,
    }

    signals = hyp.generate_signals([bar], features)
    assert len(signals) == 1
    assert signals[0].hypothesis_id == "h_0005"


def test_vwap_reversion_selective_blocks_early_or_extreme_volume() -> None:
    hyp = VwapReversionSelective()
    bar = _bar_at(datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc), 98.6)

    early = {
        "vwap": 100.0,
        "atr_14": 1.0,
        "rvol_20": 1.0,
        "volume_zscore_20": 0.3,
        "minutes_since_open": 10,
        "minutes_to_close": 120,
        "ret_open_to_now": -0.005,
    }
    extreme = {
        "vwap": 100.0,
        "atr_14": 1.0,
        "rvol_20": 1.0,
        "volume_zscore_20": 2.0,
        "minutes_since_open": 60,
        "minutes_to_close": 120,
        "ret_open_to_now": -0.005,
    }

    assert hyp.generate_signals([bar], early) == []
    assert hyp.generate_signals([bar], extreme) == []


def test_prior_day_high_reclaim_quality_triggers_when_not_overextended() -> None:
    hyp = PriorDayHighReclaimQuality()

    start = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    bars = [
        _bar_at(start, 99.9),
        _bar_at(start + timedelta(minutes=5), 100.2),
    ]
    features = {
        "prior_day_high": 100.0,
        "atr_14": 1.0,
        "rvol_20": 1.2,
        "minutes_since_open": 35,
        "session_gap_pct": 0.001,
        "ret_open_to_now": 0.002,
    }

    signals = hyp.generate_signals(bars, features)
    assert len(signals) == 1
    assert signals[0].hypothesis_id == "h_0006"


def test_prior_day_high_reclaim_quality_blocks_overextended_reclaim() -> None:
    hyp = PriorDayHighReclaimQuality()

    start = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    bars = [
        _bar_at(start, 99.9),
        _bar_at(start + timedelta(minutes=5), 101.0),
    ]
    features = {
        "prior_day_high": 100.0,
        "atr_14": 1.0,
        "rvol_20": 1.2,
        "minutes_since_open": 35,
        "session_gap_pct": 0.001,
        "ret_open_to_now": 0.002,
    }

    assert hyp.generate_signals(bars, features) == []


def test_opening_range_continuation_confirmed_triggers_with_quality_context() -> None:
    hyp = OpeningRangeContinuationConfirmed()

    start = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    bars = [
        _bar_at(start, 100.0),
        _bar_at(start + timedelta(minutes=5), 100.35),
    ]
    features = {
        "or_high": 100.2,
        "atr_14": 1.0,
        "atr_pct": 0.006,
        "rvol_20": 1.5,
        "minutes_since_open": 35,
        "session_gap_pct": 0.001,
        "ret_open_to_now": 0.003,
    }

    signals = hyp.generate_signals(bars, features)
    assert len(signals) == 1
    assert signals[0].hypothesis_id == "h_0007"


def test_opening_range_continuation_confirmed_blocks_outside_first_hour_or_overextended() -> None:
    hyp = OpeningRangeContinuationConfirmed()
    start = datetime(2026, 1, 2, 16, 30, tzinfo=timezone.utc)

    bars = [
        _bar_at(start, 100.0),
        _bar_at(start + timedelta(minutes=5), 101.3),
    ]

    outside_window = {
        "or_high": 100.2,
        "atr_14": 1.0,
        "atr_pct": 0.006,
        "rvol_20": 1.6,
        "minutes_since_open": 90,
        "session_gap_pct": 0.001,
        "ret_open_to_now": 0.004,
    }
    overextended = {
        "or_high": 100.2,
        "atr_14": 1.0,
        "atr_pct": 0.006,
        "rvol_20": 1.6,
        "minutes_since_open": 35,
        "session_gap_pct": 0.001,
        "ret_open_to_now": 0.004,
    }

    assert hyp.generate_signals(bars, outside_window) == []
    assert hyp.generate_signals(bars, overextended) == []
