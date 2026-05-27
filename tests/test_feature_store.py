from datetime import datetime, timedelta, timezone

import pandas as pd

from src.feature_store.features import compute_features


def test_compute_features_has_no_future_leakage() -> None:
    rows = []
    d1 = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    d2 = datetime(2026, 1, 3, 14, 30, tzinfo=timezone.utc)

    day1_closes = [100.0, 100.5, 100.3]
    day2_opens = [102.0, 102.2, 102.1]

    for i, c in enumerate(day1_closes):
        ts = d1 + timedelta(minutes=5 * i)
        rows.append({
            "event_time": ts,
            "available_time": ts + timedelta(seconds=5),
            "symbol": "SPY",
            "open": c,
            "high": c + 0.3,
            "low": c - 0.3,
            "close": c,
            "volume": 1000 + i,
        })

    for i, c in enumerate(day2_opens):
        ts = d2 + timedelta(minutes=5 * i)
        rows.append({
            "event_time": ts,
            "available_time": ts + timedelta(seconds=5),
            "symbol": "SPY",
            "open": c,
            "high": c + 0.3,
            "low": c - 0.3,
            "close": c,
            "volume": 1100 + i,
        })

    df = pd.DataFrame(rows)
    out = compute_features(df, save=False)

    # gap_pct remains first-bar-only behavior.
    day2 = out[out["event_time"].dt.date == d2.date()].reset_index(drop=True)
    assert day2.loc[0, "gap_pct"] != 0.0
    assert day2.loc[1, "gap_pct"] == 0.0

    # session_gap_pct is carried through the full session without future leakage.
    assert day2.loc[0, "session_gap_pct"] == day2.loc[1, "session_gap_pct"]
    assert day2.loc[1, "session_gap_pct"] == day2.loc[2, "session_gap_pct"]


def test_compute_features_adds_intraday_and_prior_day_features() -> None:
    rows = []
    d1 = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    d2 = datetime(2026, 1, 3, 14, 30, tzinfo=timezone.utc)

    day1 = [100.0, 101.0, 100.5]
    day2 = [102.0, 102.4, 102.2]

    for i, c in enumerate(day1):
        ts = d1 + timedelta(minutes=5 * i)
        rows.append({
            "event_time": ts,
            "available_time": ts + timedelta(seconds=5),
            "symbol": "SPY",
            "open": c,
            "high": c + 0.4,
            "low": c - 0.2,
            "close": c,
            "volume": 1000 + i * 20,
        })

    for i, c in enumerate(day2):
        ts = d2 + timedelta(minutes=5 * i)
        rows.append({
            "event_time": ts,
            "available_time": ts + timedelta(seconds=5),
            "symbol": "SPY",
            "open": c,
            "high": c + 0.3,
            "low": c - 0.1,
            "close": c,
            "volume": 1200 + i * 30,
        })

    out = compute_features(pd.DataFrame(rows), save=False)
    day2_out = out[out["event_time"].dt.date == d2.date()].reset_index(drop=True)

    assert day2_out.loc[0, "bar_in_session"] == 0
    assert day2_out.loc[1, "bar_in_session"] == 1
    assert day2_out.loc[2, "bar_in_session"] == 2

    assert day2_out.loc[0, "minutes_since_open"] == 0
    assert day2_out.loc[1, "minutes_since_open"] == 5
    assert day2_out.loc[2, "minutes_since_open"] == 10

    assert day2_out.loc[0, "minutes_to_close"] == 385
    assert day2_out.loc[1, "minutes_to_close"] == 380

    expected_prior_close = day1[-1]
    expected_prior_high = max(c + 0.4 for c in day1)
    expected_prior_low = min(c - 0.2 for c in day1)

    assert day2_out.loc[0, "prior_day_close"] == expected_prior_close
    assert day2_out.loc[1, "prior_day_close"] == expected_prior_close
    assert day2_out.loc[0, "prior_day_high"] == expected_prior_high
    assert day2_out.loc[0, "prior_day_low"] == expected_prior_low

    assert (day2_out["atr_pct"] > 0).all()
    assert day2_out["volume_zscore_20"].notna().any()
