"""h_0011 - Previous-day low + 0.8 * ATR(40) stop-entry, hold to end of day.

Core setup:
  - Compute entry level = prior_day_low + 0.8 * ATR(40).
  - Enter long when price crosses above that level (stop-style trigger).
  - Hold until end of day.

Within the current backtester contract, "hold to EOD" is approximated by setting
max_hold_bars from minutes_to_close and using very wide stop/target distances.
"""

from __future__ import annotations

import math
from typing import Sequence

from src.types import Bar, Direction, Signal


class PrevDayLowAtrBreakoutEod:
    """Previous-day low plus ATR breakout, long-only, end-of-day hold."""

    name: str = "prev_day_low_atr_breakout_eod"
    version: str = "0.1"
    hypothesis_id: str = "h_0011"

    atr_period: int = 40
    atr_multiplier: float = 0.8
    stop_distance_atr: float = 100.0
    take_profit_distance_atr: float = 100.0

    def required_features(self) -> Sequence[str]:
        return ["prior_day_low", "minutes_to_close"]

    def _atr(self, bars: Sequence[Bar]) -> float | None:
        true_ranges: list[float] = []
        for idx in range(1, len(bars)):
            cur = bars[idx]
            prev = bars[idx - 1]
            tr = max(
                cur.high - cur.low,
                abs(cur.high - prev.close),
                abs(cur.low - prev.close),
            )
            true_ranges.append(tr)

        if len(true_ranges) < self.atr_period:
            return None

        window = true_ranges[-self.atr_period :]
        return sum(window) / float(self.atr_period)

    def generate_signals(
        self,
        bars: Sequence[Bar],
        features: dict,
    ) -> Sequence[Signal]:
        if len(bars) < self.atr_period + 1:
            return []

        prior_day_low: float | None = features.get("prior_day_low")
        minutes_to_close: float | None = features.get("minutes_to_close")
        if prior_day_low is None or minutes_to_close is None:
            return []

        atr_40 = self._atr(bars)
        if atr_40 is None or atr_40 <= 0:
            return []

        entry_level = prior_day_low + self.atr_multiplier * atr_40

        prev_bar = bars[-2]
        bar = bars[-1]

        # Stop-style crossing: previous bar not above trigger, current bar reaches trigger.
        if prev_bar.high >= entry_level:
            return []
        if bar.high < entry_level:
            return []

        bars_to_close = max(1, int(math.ceil(minutes_to_close / 5.0)))
        confidence = min(1.0, 0.5 + max(0.0, (bar.high - entry_level) / atr_40) * 0.1)

        return [
            Signal(
                hypothesis_id=self.hypothesis_id,
                symbol=bar.symbol,
                bar_time=bar.event_time,
                direction=Direction.LONG,
                confidence=confidence,
                stop_distance_atr=self.stop_distance_atr,
                take_profit_distance_atr=self.take_profit_distance_atr,
                max_hold_bars=bars_to_close,
            )
        ]
