"""h_0017 - MACD signal-line trend-following breakout.

Long-only setup:
  - Standard MACD line = EMA(fast) - EMA(slow).
  - Enter when MACD histogram crosses from non-positive to positive.
  - Require the MACD line to be above zero to stay aligned with trend.
  - Skip the opening noise and the late-day tail.
"""

from __future__ import annotations

import math
from typing import Sequence

from src.types import Bar, Direction, Signal


class MacdSignalTrend:
    """MACD bullish signal-line crossover with simple intraday filters."""

    name: str = "macd_signal_trend"
    version: str = "0.1"
    hypothesis_id: str = "h_0017"

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    min_minutes_since_open: int = 30
    max_minutes_since_open: int = 180
    min_minutes_to_close: int = 45
    min_rvol: float = 1.0
    min_histogram_atr: float = 0.03
    stop_distance_atr: float = 1.2
    take_profit_distance_atr: float = 2.8
    max_hold_bars: int = 18

    def required_features(self) -> Sequence[str]:
        return ["atr_14", "minutes_since_open", "minutes_to_close", "rvol_20"]

    def _ema_series(self, values: Sequence[float], period: int) -> list[float]:
        if not values:
            return []
        multiplier = 2.0 / (period + 1.0)
        ema_values = [values[0]]
        for value in values[1:]:
            ema_values.append((value - ema_values[-1]) * multiplier + ema_values[-1])
        return ema_values

    def _macd_components(self, closes: Sequence[float]) -> tuple[float, float, float, float] | None:
        min_bars = self.slow_period + self.signal_period
        if len(closes) < min_bars:
            return None

        fast_ema = self._ema_series(closes, self.fast_period)
        slow_ema = self._ema_series(closes, self.slow_period)
        macd_line = [fast - slow for fast, slow in zip(fast_ema, slow_ema, strict=False)]
        signal_line = self._ema_series(macd_line, self.signal_period)

        if len(macd_line) < 2 or len(signal_line) < 2:
            return None

        prev_histogram = macd_line[-2] - signal_line[-2]
        histogram = macd_line[-1] - signal_line[-1]
        return macd_line[-1], signal_line[-1], prev_histogram, histogram

    def generate_signals(self, bars: Sequence[Bar], features: dict) -> Sequence[Signal]:
        atr: float | None = features.get("atr_14")
        minutes_since_open: float | None = features.get("minutes_since_open")
        minutes_to_close: float | None = features.get("minutes_to_close")
        rvol: float | None = features.get("rvol_20")
        if (
            atr is None
            or atr <= 0
            or minutes_since_open is None
            or minutes_to_close is None
            or rvol is None
        ):
            return []

        if minutes_since_open < self.min_minutes_since_open:
            return []
        if minutes_since_open > self.max_minutes_since_open:
            return []
        if minutes_to_close < self.min_minutes_to_close:
            return []
        if rvol < self.min_rvol:
            return []

        components = self._macd_components([bar.close for bar in bars])
        if components is None:
            return []

        macd_value, _, prev_histogram, histogram = components
        if prev_histogram > 0 or histogram <= 0:
            return []
        if macd_value <= 0:
            return []

        histogram_atr = histogram / atr
        if histogram_atr < self.min_histogram_atr:
            return []

        confidence = min(1.0, 0.5 + histogram_atr * 2.0 + max(0.0, rvol - 1.0) * 0.1)
        max_hold = min(self.max_hold_bars, max(1, int(math.ceil(minutes_to_close / 5.0))))
        bar = bars[-1]

        return [
            Signal(
                hypothesis_id=self.hypothesis_id,
                symbol=bar.symbol,
                bar_time=bar.event_time,
                direction=Direction.LONG,
                confidence=confidence,
                stop_distance_atr=self.stop_distance_atr,
                take_profit_distance_atr=self.take_profit_distance_atr,
                max_hold_bars=max_hold,
            )
        ]