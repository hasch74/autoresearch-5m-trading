"""h_0006 — Quality-filtered prior-day-high reclaim.

Variant hypothesis:
  Enter only when prior-day-high reclaim occurs with moderate volume,
  positive intraday context, and without overextension above the level.

This file may be modified by the research agent.
The Hypothesis interface contract (strategies/hypotheses/interfaces.py) must be respected.
"""

from __future__ import annotations

from typing import Sequence

from src.types import Bar, Direction, Signal


class PriorDayHighReclaimQuality:
    """Lower-chase prior-day-high reclaim variant."""

    name: str = "prior_day_high_reclaim_quality"
    version: str = "0.1"
    hypothesis_id: str = "h_0006"

    min_rvol: float = 0.9
    max_rvol: float = 1.7
    reclaim_buffer_atr: float = 0.08
    max_extension_above_level_atr: float = 0.55
    min_minutes_since_open: int = 25
    min_session_gap_pct: float = -0.003
    min_ret_open_to_now: float = 0.0
    stop_atr_mult: float = 0.7
    take_profit_atr_mult: float = 1.2
    max_hold_bars: int = 6

    def required_features(self) -> Sequence[str]:
        return [
            "prior_day_high",
            "atr_14",
            "rvol_20",
            "minutes_since_open",
            "session_gap_pct",
            "ret_open_to_now",
        ]

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
        session_gap_pct: float | None = features.get("session_gap_pct")
        ret_open_to_now: float | None = features.get("ret_open_to_now")

        if any(v is None for v in (prior_day_high, atr, rvol, minutes_since_open, session_gap_pct, ret_open_to_now)):
            return []
        if atr <= 0:
            return []
        if minutes_since_open < self.min_minutes_since_open:
            return []
        if session_gap_pct < self.min_session_gap_pct:
            return []
        if ret_open_to_now < self.min_ret_open_to_now:
            return []
        if not (self.min_rvol <= rvol <= self.max_rvol):
            return []
        if prev_bar.close >= prior_day_high:
            return []

        reclaim_buffer = self.reclaim_buffer_atr * atr
        if bar.close <= prior_day_high + reclaim_buffer:
            return []

        extension = (bar.close - prior_day_high) / atr
        if extension > self.max_extension_above_level_atr:
            return []

        confidence = min(1.0, 0.45 + extension / 2.0)
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
