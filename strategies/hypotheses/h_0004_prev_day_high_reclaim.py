"""h_0004 — Prior-day high reclaim baseline.

Baseline hypothesis:
  When price reclaims the prior day's high after trading below it,
  enter long for a continuation test above that reference level.

This file may be modified by the research agent.
The Hypothesis interface contract (strategies/hypotheses/interfaces.py) must be respected.
"""

from __future__ import annotations

from typing import Sequence

from src.types import Bar, Direction, Signal


class PriorDayHighReclaimBaseline:
    """Simple prior-day-high reclaim comparator."""

    name: str = "prior_day_high_reclaim_baseline"
    version: str = "0.1"
    hypothesis_id: str = "h_0004"

    min_rvol: float = 0.9
    reclaim_buffer_atr: float = 0.10
    stop_atr_mult: float = 0.8
    take_profit_atr_mult: float = 1.6
    max_hold_bars: int = 8
    min_minutes_since_open: int = 15

    def required_features(self) -> Sequence[str]:
        return ["prior_day_high", "atr_14", "rvol_20", "minutes_since_open"]

    def generate_signals(
        self,
        bars: Sequence[Bar],
        features: dict,
    ) -> Sequence[Signal]:
        if len(bars) < 2:
            return []

        bar = bars[-1]
        prev_bar = bars[-2]
        prior_day_high: float | None = features.get("prior_day_high")
        atr: float | None = features.get("atr_14")
        rvol: float | None = features.get("rvol_20")
        minutes_since_open: float | None = features.get("minutes_since_open")

        if any(v is None for v in (prior_day_high, atr, rvol, minutes_since_open)):
            return []
        if atr <= 0:
            return []
        if minutes_since_open < self.min_minutes_since_open:
            return []
        if rvol < self.min_rvol:
            return []
        if prev_bar.close >= prior_day_high:
            return []

        reclaim_buffer = self.reclaim_buffer_atr * atr
        if bar.close <= prior_day_high + reclaim_buffer:
            return []

        confidence = min(1.0, 0.5 + ((bar.close - prior_day_high) / atr) / 2.0)
        return [
            Signal(
                hypothesis_id=self.hypothesis_id,
                symbol=bar.symbol,
                bar_time=bar.event_time,
                direction=Direction.LONG,
                confidence=confidence,
                stop_distance_atr=self.stop_atr_mult,
                take_profit_distance_atr=self.take_profit_atr_mult,
                max_hold_bars=self.max_hold_bars,
            )
        ]