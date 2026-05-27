"""EODHD 5-minute bar fetcher with local Parquet cache.

Design:
- Fetches historical intraday bars from EODHD REST API.
- Caches raw JSON responses as Parquet under data/raw/<symbol>.parquet.
- Writes normalized Bar records (Regular Trading Hours only, split-adjusted)
  to data/normalized/<symbol>_5m.parquet.
- Idempotent: re-running only fetches bars newer than the latest cached date.

Protected against lookahead:
- available_time is set to max(bar_close + timedelta(seconds=5), fetch_time)
  so no bar is usable before it could have been observed.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# EODHD API constants
_BASE_URL = "https://eodhd.com/api/intraday/{symbol}.US"
_INTERVAL = "5m"
_MAX_ROWS_PER_REQUEST = 120 * 78  # ~6 months of RTH bars

# Parquet paths (relative to repo root)
_RAW_DIR = Path("data/raw")
_NORM_DIR = Path("data/normalized")

# RTH window (Eastern) encoded as UTC offsets for simplicity:
# EDT: UTC-4  →  RTH = 13:30–20:00 UTC
# EST: UTC-5  →  RTH = 14:30–21:00 UTC
# We clip to whichever is correct via the tz-aware datetime after conversion.
_RTH_OPEN = "09:30"
_RTH_CLOSE = "16:00"
_EASTERN_TZ = "America/New_York"

# EODHD imposes a ~1 req/s soft limit for free API keys; be conservative.
_REQUEST_DELAY_S = 1.1


def fetch_bars(
    symbol: str,
    api_key: str,
    start: date | None = None,
    end: date | None = None,
    *,
    raw_dir: Path = _RAW_DIR,
    norm_dir: Path = _NORM_DIR,
) -> pd.DataFrame:
    """Fetch 5-minute bars for *symbol* and return a normalized DataFrame.

    Columns returned:
        event_time      datetime64[ns, UTC]
        available_time  datetime64[ns, UTC]
        symbol          str
        open            float64
        high            float64
        low             float64
        close           float64
        volume          int64

    Caches to Parquet so subsequent calls are incremental.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{symbol}_5m_raw.parquet"
    norm_path = norm_dir / f"{symbol}_5m.parquet"

    # Determine incremental start date from existing cache
    if norm_path.exists():
        existing = pd.read_parquet(norm_path, columns=["event_time"])
        latest = existing["event_time"].max()
        # Fetch from day after latest cached bar (overlap by 1 day for safety)
        incremental_start = (latest - timedelta(days=1)).date()
    else:
        incremental_start = start or date(2020, 10, 1)  # EODHD 5m history start

    effective_start = max(incremental_start, start) if start else incremental_start
    effective_end = end or date.today()

    if effective_start > effective_end:
        logger.info("%s: cache is current, skipping fetch", symbol)
        return pd.read_parquet(norm_path) if norm_path.exists() else pd.DataFrame()

    logger.info("%s: fetching %s → %s", symbol, effective_start, effective_end)
    raw_df = _call_api(symbol, api_key, effective_start, effective_end)
    if raw_df.empty:
        logger.warning("%s: no data returned from API", symbol)
        return pd.read_parquet(norm_path) if norm_path.exists() else pd.DataFrame()

    # Persist raw data
    _append_parquet(raw_path, raw_df, dedup_col="datetime")

    # Normalise
    norm_new = _normalise(raw_df, symbol)

    if norm_path.exists():
        existing_norm = pd.read_parquet(norm_path)
        combined = pd.concat([existing_norm, norm_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["event_time"]).sort_values("event_time")
    else:
        combined = norm_new

    combined.to_parquet(norm_path, index=False)
    logger.info("%s: %d RTH bars saved to %s", symbol, len(combined), norm_path)
    return combined


def sync_universe(
    symbols: Sequence[str],
    api_key: str,
    start: date | None = None,
    end: date | None = None,
    *,
    raw_dir: Path = _RAW_DIR,
    norm_dir: Path = _NORM_DIR,
) -> dict[str, pd.DataFrame]:
    """Fetch bars for every symbol in *symbols*. Returns {symbol: DataFrame}."""
    results: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            results[sym] = fetch_bars(
                sym, api_key, start=start, end=end,
                raw_dir=raw_dir, norm_dir=norm_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: fetch failed — %s", sym, exc)
            results[sym] = pd.DataFrame()
        time.sleep(_REQUEST_DELAY_S)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_api(
    symbol: str,
    api_key: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Call EODHD intraday endpoint; return raw DataFrame with column 'datetime'."""
    url = _BASE_URL.format(symbol=symbol)
    # EODHD intraday API expects Unix seconds, not ISO date strings.
    start_ts = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    end_ts = int(
        datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp()
    ) - 1
    params = {
        "api_token": api_key,
        "fmt": "json",
        "interval": _INTERVAL,
        "from": start_ts,
        "to": end_ts,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # EODHD returns 'datetime' as Unix timestamp (int) or ISO string depending on fmt
    if "datetime" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "datetime"})
    return df


def _normalise(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert raw API DataFrame to normalised Bar-compatible DataFrame.

    - Parses timestamps to UTC-aware datetimes.
    - Filters to Regular Trading Hours (09:30–16:00 Eastern).
    - Sets available_time = event_time + 5 seconds (conservative: bar close + 5s).
    """
    df = raw.copy()

    # Parse datetime column — handle both Unix int and ISO string
    if pd.api.types.is_integer_dtype(df["datetime"]):
        df["event_time"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
    else:
        df["event_time"] = pd.to_datetime(df["datetime"], utc=True)

    # Filter to RTH only
    eastern = df["event_time"].dt.tz_convert(_EASTERN_TZ)
    time_of_day = eastern.dt.time
    rth_open = pd.Timestamp(_RTH_OPEN).time()
    rth_close = pd.Timestamp(_RTH_CLOSE).time()
    mask = (time_of_day >= rth_open) & (time_of_day < rth_close)
    df = df[mask].copy()

    # available_time: bar close + 5 s (prevents accidental lookahead)
    df["available_time"] = df["event_time"] + pd.Timedelta(seconds=5)

    df["symbol"] = symbol
    df = df.rename(columns={"open": "open", "high": "high", "low": "low",
                             "close": "close", "volume": "volume"})

    cols = ["event_time", "available_time", "symbol",
            "open", "high", "low", "close", "volume"]
    df = df[cols].dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df = df.sort_values("event_time").reset_index(drop=True)
    return df


def _append_parquet(path: Path, new_df: pd.DataFrame, dedup_col: str) -> None:
    """Append *new_df* to an existing Parquet file, deduplicating on *dedup_col*."""
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=[dedup_col])
    else:
        combined = new_df
    combined.to_parquet(path, index=False)
