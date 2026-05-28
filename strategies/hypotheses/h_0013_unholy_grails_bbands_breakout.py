"""h_0013 - Unholy Grails inspired Bollinger breakout (long-only).

Original idea mapping:
  - Bands based on 100-period SMA of close.
    - Upper band = SMA + 3 * std (entry condition).
  - Lower band = SMA - 1 * std (exit condition).
  - Long entry when close > upper band.

Note on current engine contract:
  - The backtester consumes entry signals with bracket parameters.
    - Dynamic "close below lower band, exit next open" is implemented through
        should_exit_position(), which the backtester executes at next bar open.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Sequence

from src.types import Bar, Direction, Signal


class UnholyGrailsBbandsBreakout:
    """Bollinger 100/3 entry with lower-band next-open exit."""

    name: str = "unholy_grails_bbands_breakout"
    version: str = "0.1"
    hypothesis_id: str = "h_0013"

    bb_period: int = 100
    entry_std_mult: float = 3.0
    exit_std_mult: float = 1.0
    fallback_stop_atr: float = 0.25
    take_profit_distance_atr: float = 100.0
    max_hold_bars: int = 10_000

    def required_features(self) -> Sequence[str]:
        return ["atr_14"]

    def _bands(self, bars: Sequence[Bar]) -> tuple[float, float, float] | None:
        if len(bars) < self.bb_period:
            return None
        closes = [b.close for b in bars[-self.bb_period :]]
        mid = mean(closes)
        std = pstdev(closes)
        upper = mid + self.entry_std_mult * std
        lower = mid - self.exit_std_mult * std
        return mid, upper, lower

    def generate_signals(self, bars: Sequence[Bar], features: dict) -> Sequence[Signal]:
        if len(bars) < self.bb_period:
            return []

        atr: float | None = features.get("atr_14")
        if atr is None or atr <= 0:
            return []

        band_values = self._bands(bars)
        if band_values is None:
            return []
        _, upper, lower = band_values

        bar = bars[-1]

        # Entry: close above upper Bollinger band.
        if bar.close <= upper:
            return []

        # Approximate lower-band exit by anchoring initial stop to lower band.
        stop_atr = (bar.close - lower) / atr if bar.close > lower else self.fallback_stop_atr
        stop_atr = max(self.fallback_stop_atr, stop_atr)

        breakout_strength = (bar.close - upper) / atr
        confidence = min(1.0, 0.5 + max(0.0, breakout_strength) * 0.08)

        return [
            Signal(
                hypothesis_id=self.hypothesis_id,
                symbol=bar.symbol,
                bar_time=bar.event_time,
                direction=Direction.LONG,
                confidence=confidence,
                stop_distance_atr=stop_atr,
                take_profit_distance_atr=self.take_profit_distance_atr,
                max_hold_bars=self.max_hold_bars,
            )
        ]

    def should_exit_position(self, bars: Sequence[Bar], features: dict) -> bool:
        """Exit trigger at bar close: close below lower Bollinger band."""
        if len(bars) < self.bb_period:
            return False
        band_values = self._bands(bars)
        if band_values is None:
            return False
        _, _, lower = band_values
        return bars[-1].close < lower
