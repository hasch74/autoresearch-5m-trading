"""h_0002 — Opening Range Breakout.

Edge hypothesis:
  After the first 30 minutes define the opening range [OR_low, OR_high].
  Enter long on the first bar that closes above OR_high with elevated volume.
  Stop below OR_low. Target 2x the opening range width.

This file may be modified by the research agent.
The Hypothesis interface contract (strategies/hypotheses/interfaces.py) must be respected.
"""

from __future__ import annotations

from typing import Sequence

from src.types import Bar, Direction, Signal


class OpeningRangeBreakout:
    """Opening Range Breakout — version 0.1."""

    name: str = "opening_range_breakout"
    version: str = "0.1"
    hypothesis_id: str = "h_0002"

    # --- parameters (agent may tune these) ---
    or_bars: int = 6                   # number of 5m bars = 30 minutes
    rvol_min: float = 1.2
    max_range_atr_ratio: float = 2.0
    stop_atr_mult: float = 1.0
    take_profit_range_mult: float = 2.0
    max_hold_bars: int = 6

    def required_features(self) -> Sequence[str]:
        return ["or_high", "or_low", "rvol_20", "atr_14"]

    def generate_signals(
        self,
        bars: Sequence[Bar],
        features: dict,
    ) -> Sequence[Signal]:
        """Return signals for the latest bar.

        Parameters
        ----------
        bars:
            Sequence of Bar objects ending with the most recent bar.
            Must contain at least or_bars + 1 bars for the opening range to be defined.
        features:
            Mapping of feature name -> float value for the latest bar.

        Returns
        -------
        Sequence of Signal objects (empty if no entry condition is met).
        """
        if len(bars) < self.or_bars + 1:
            return []  # opening range not yet formed

        bar = bars[-1]

        or_high: float | None = features.get("or_high")
        or_low: float | None = features.get("or_low")
        rvol: float | None = features.get("rvol_20")
        atr: float | None = features.get("atr_14")

        if any(v is None for v in (or_high, or_low, rvol, atr)):
            return []

        or_width = or_high - or_low
        if or_width <= 0:
            return []

        # Gate 1: skip if opening range is unusually wide
        if atr > 0 and (or_width / atr) > self.max_range_atr_ratio:
            return []

        # Gate 2: breakout — close must be above OR_high
        if bar.close <= or_high:
            return []

        # Gate 3: elevated relative volume
        if rvol < self.rvol_min:
            return []

        # Confidence scales with how cleanly the breakout occurred
        breakout_bps = (bar.close - or_high) / or_high * 10_000
        confidence = min(1.0, 0.5 + breakout_bps / 50.0)

        # Take-profit distance in ATR: map 2x OR_width to ATR multiples
        tp_atr = (self.take_profit_range_mult * or_width) / atr if atr > 0 else 2.0

        signal = Signal(
            hypothesis_id=self.hypothesis_id,
            symbol=bar.symbol,
            bar_time=bar.event_time,
            direction=Direction.LONG,
            confidence=confidence,
            stop_distance_atr=self.stop_atr_mult,
            take_profit_distance_atr=max(1.0, tp_atr),
            max_hold_bars=self.max_hold_bars,
        )
        return [signal]
