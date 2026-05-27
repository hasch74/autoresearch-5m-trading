"""h_0003 — VWAP Reversion baseline.

Baseline hypothesis:
  When price stretches materially below session VWAP without panic-level RVOL,
  enter long for a mean-reversion move back toward VWAP.

This file may be modified by the research agent.
The Hypothesis interface contract (strategies/hypotheses/interfaces.py) must be respected.
"""

from __future__ import annotations

from typing import Sequence

from src.types import Bar, Direction, Signal


class VwapReversionBaseline:
    """Simple long-only VWAP mean-reversion baseline."""

    name: str = "vwap_reversion_baseline"
    version: str = "0.1"
    hypothesis_id: str = "h_0003"

    entry_atr_band: float = 0.75
    max_rvol: float = 1.25
    stop_atr_mult: float = 1.0
    take_profit_atr_mult: float = 1.25
    max_hold_bars: int = 6
    min_minutes_to_close: int = 20

    def required_features(self) -> Sequence[str]:
        return ["vwap", "atr_14", "rvol_20", "minutes_to_close"]

    def generate_signals(
        self,
        bars: Sequence[Bar],
        features: dict,
    ) -> Sequence[Signal]:
        if not bars:
            return []

        bar = bars[-1]
        vwap: float | None = features.get("vwap")
        atr: float | None = features.get("atr_14")
        rvol: float | None = features.get("rvol_20")
        minutes_to_close: float | None = features.get("minutes_to_close")

        if any(v is None for v in (vwap, atr, rvol, minutes_to_close)):
            return []
        if atr <= 0:
            return []
        if minutes_to_close < self.min_minutes_to_close:
            return []
        if rvol > self.max_rvol:
            return []

        deviation = (vwap - bar.close) / atr
        if deviation < self.entry_atr_band:
            return []

        confidence = min(1.0, 0.45 + deviation / 2.0)
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