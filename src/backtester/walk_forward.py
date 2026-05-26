"""Walk-forward split helper for backtester.

Produces (train_df, validate_df, test_df) slices from a feature DataFrame
based on the window sizes defined in configs/risk.yaml.

Usage:
    from src.backtester.walk_forward import walk_forward_splits
    for train, validate, test in walk_forward_splits(df):
        result = run_backtest(train, hypothesis)
        oos = run_backtest(test, hypothesis)
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import pandas as pd


def walk_forward_splits(
    df: pd.DataFrame,
    *,
    train_days: int = 60,
    validate_days: int = 20,
    test_days: int = 20,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Yield rolling (train, validate, test) DataFrames.

    Splits are by calendar trading day, rolling forward by test_days each iteration.
    Requires df to have an 'event_time' column with timezone-aware datetimes.

    Parameters
    ----------
    df:
        Feature DataFrame, sorted by event_time.
    train_days, validate_days, test_days:
        Window sizes in trading days. Default matches configs/risk.yaml.
    """
    eastern = df["event_time"].dt.tz_convert("America/New_York")
    unique_dates = sorted(eastern.dt.date.unique())
    total_window = train_days + validate_days + test_days

    if len(unique_dates) < total_window:
        return  # not enough data

    start = 0
    while start + total_window <= len(unique_dates):
        train_dates = unique_dates[start: start + train_days]
        val_dates = unique_dates[start + train_days: start + train_days + validate_days]
        test_dates = unique_dates[start + train_days + validate_days: start + total_window]

        date_col = eastern.dt.date
        train_df = df[date_col.isin(set(train_dates))].copy()
        val_df = df[date_col.isin(set(val_dates))].copy()
        test_df = df[date_col.isin(set(test_dates))].copy()

        if not train_df.empty and not test_df.empty:
            yield train_df, val_df, test_df

        start += test_days  # roll forward by one test window
