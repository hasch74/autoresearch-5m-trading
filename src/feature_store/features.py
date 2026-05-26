"""Feature computation for 5-minute bars.

Computes all features expected by hypothesis stubs:
  - vwap          Cumulative VWAP from session open (09:30)
  - rvol_20       Relative volume vs. 20-bar rolling mean
  - atr_14        14-bar Average True Range
  - gap_pct       Gap at open vs. prior close (first bar of session only; else 0)
  - ret_open_to_now  Return from session open bar to current close
  - or_high       Opening range high (first or_bars of session)
  - or_low        Opening range low  (first or_bars of session)

Input:  pandas DataFrame with columns matching data/normalized schema
        (event_time, symbol, open, high, low, close, volume)
Output: same DataFrame with additional feature columns appended.

The feature store also persists results to data/features/<symbol>_5m.parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_FEAT_DIR = Path("data/features")
_OR_BARS = 6   # 6 × 5m = 30 minutes opening range


def compute_features(
    df: pd.DataFrame,
    *,
    or_bars: int = _OR_BARS,
    feat_dir: Path = _FEAT_DIR,
    save: bool = True,
) -> pd.DataFrame:
    """Compute all features for *df* and optionally persist to Parquet.

    *df* must be sorted by event_time and contain a single symbol.
    Returns a new DataFrame with feature columns added.
    """
    if df.empty:
        return df

    symbol = df["symbol"].iloc[0]
    df = df.copy().sort_values("event_time").reset_index(drop=True)

    # Detect session boundaries (each calendar day is one session)
    df["_date"] = df["event_time"].dt.tz_convert("America/New_York").dt.date
    df["_bar_in_session"] = df.groupby("_date").cumcount()

    # --- VWAP (cumulative within session) ---
    df["_typical"] = (df["high"] + df["low"] + df["close"]) / 3
    df["_cum_tp_vol"] = df.groupby("_date").apply(
        lambda g: (g["_typical"] * g["volume"]).cumsum(), include_groups=False
    ).reset_index(level=0, drop=True)
    df["_cum_vol"] = df.groupby("_date")["volume"].cumsum()
    df["vwap"] = df["_cum_tp_vol"] / df["_cum_vol"].replace(0, np.nan)

    # --- RVOL-20 ---
    rolling_mean_vol = df["volume"].rolling(window=20, min_periods=1).mean()
    df["rvol_20"] = df["volume"] / rolling_mean_vol.replace(0, np.nan)

    # --- ATR-14 ---
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(span=14, min_periods=1, adjust=False).mean()

    # --- gap_pct (first bar of session vs. prior session last close) ---
    last_close = df.groupby("_date")["close"].last()
    prior_close = last_close.shift(1)
    first_bars_mask = df["_bar_in_session"] == 0
    df["gap_pct"] = 0.0
    for d, pc in prior_close.items():
        if pd.isna(pc):
            continue
        idx = df.index[df["_date"] == d]
        if idx.empty:
            continue
        first_idx = idx[0]
        df.loc[first_idx, "gap_pct"] = (df.loc[first_idx, "open"] - pc) / pc

    # --- ret_open_to_now (return from session open bar close to current close) ---
    session_open_close = df[df["_bar_in_session"] == 0].set_index("_date")["close"]
    df["_session_open_close"] = df["_date"].map(session_open_close)
    df["ret_open_to_now"] = (df["close"] - df["_session_open_close"]) / df["_session_open_close"].replace(0, np.nan)

    # --- Opening range high/low (first or_bars bars of each session) ---
    def _or_high(g: pd.DataFrame) -> pd.Series:
        result = pd.Series(np.nan, index=g.index)
        for i in range(len(g)):
            n = min(i + 1, or_bars)
            result.iloc[i] = g["high"].iloc[:n].max()
        return result

    def _or_low(g: pd.DataFrame) -> pd.Series:
        result = pd.Series(np.nan, index=g.index)
        for i in range(len(g)):
            n = min(i + 1, or_bars)
            result.iloc[i] = g["low"].iloc[:n].min()
        return result

    df["or_high"] = df.groupby("_date", group_keys=False).apply(_or_high)
    df["or_low"] = df.groupby("_date", group_keys=False).apply(_or_low)

    # Drop internal columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])

    if save:
        feat_dir.mkdir(parents=True, exist_ok=True)
        path = feat_dir / f"{symbol}_5m.parquet"
        df.to_parquet(path, index=False)

    return df


def load_features(symbol: str, feat_dir: Path = _FEAT_DIR) -> pd.DataFrame:
    """Load pre-computed features from Parquet."""
    path = feat_dir / f"{symbol}_5m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No feature file for {symbol}: {path}")
    return pd.read_parquet(path)
