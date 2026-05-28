"""h_0007 — Opening-range continuation with context confirmation.

Selective long-only continuation setup:
  - Trade only in the first hour after open (after OR is formed).
  - Require strong relative volume and positive market-context proxies.
  - Avoid chasing overextended breakouts above the opening range.

This file may be modified by the research agent.
The Hypothesis interface contract (strategies/hypotheses/interfaces.py) must be respected.
"""

from __future__ import annotations

from typing import Sequence

from src.types import Bar, Direction, Signal


class OpeningRangeContinuationConfirmed:
    """First-hour opening-range continuation with quality gates."""

    name: str = "opening_range_continuation_confirmed"
    version: str = "0.1"
    hypothesis_id: str = "h_0007"

    min_minutes_since_open: int = 30
    max_minutes_since_open: int = 60
    min_rvol: float = 1.35
    min_breakout_buffer_atr: float = 0.06
    max_extension_above_or_atr: float = 0.80
    min_session_gap_pct: float = -0.001
    min_ret_open_to_now: float = 0.001
    min_atr_pct: float = 0.002
    max_atr_pct: float = 0.015
    stop_atr_mult: float = 0.85
    take_profit_atr_mult: float = 1.55
    max_hold_bars: int = 8

    def required_features(self) -> Sequence[str]:
        return [
            "or_high",
            "atr_14",
            "atr_pct",
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

        or_high: float | None = features.get("or_high")
        atr: float | None = features.get("atr_14")
        atr_pct: float | None = features.get("atr_pct")
        rvol: float | None = features.get("rvol_20")
        minutes_since_open: float | None = features.get("minutes_since_open")
        session_gap_pct: float | None = features.get("session_gap_pct")
        ret_open_to_now: float | None = features.get("ret_open_to_now")

        if any(v is None for v in (or_high, atr, atr_pct, rvol, minutes_since_open, session_gap_pct, ret_open_to_now)):
            return []
        if atr <= 0:
            return []

        if not (self.min_minutes_since_open <= minutes_since_open <= self.max_minutes_since_open):
            return []
        if rvol < self.min_rvol:
            return []
        if not (self.min_atr_pct <= atr_pct <= self.max_atr_pct):
            return []

        # Market-context proxies: avoid weak/bearish session backdrop.
        if session_gap_pct < self.min_session_gap_pct:
            return []
        if ret_open_to_now < self.min_ret_open_to_now:
            return []

        # Require true reclaim/breakout now, not a stale move from earlier bars.
        if prev_bar.close >= or_high:
            return []

        breakout_buffer = self.min_breakout_buffer_atr * atr
        if bar.close <= or_high + breakout_buffer:
            return []

        extension_atr = (bar.close - or_high) / atr
        if extension_atr > self.max_extension_above_or_atr:
            return []

        confidence = min(1.0, 0.45 + 0.12 * extension_atr + 0.08 * max(0.0, rvol - self.min_rvol))
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
