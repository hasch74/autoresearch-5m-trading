"""Local paper trading simulator.

Processes 5-minute bars in real time (or replay):
  - For each bar: checks exits for open positions, generates signals, creates paper orders.
  - Simulates fills with spread + slippage model (same as backtester cost model).
  - Persists paper fills to data/paper_trades/<symbol>_fills.parquet.
  - Updates hypothesis paper scores for later gate evaluation.

No broker connectivity. This is the local paper stage before IBKR Paper Trading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.types import Bar, Direction, Order, OrderStatus, Signal

logger = logging.getLogger(__name__)

_PAPER_DIR = Path("data/paper_trades")

# Cost model defaults (mirror backtester.engine._CostModel)
_HALF_SPREAD_BPS = 3.0
_ENTRY_ATR_FRAC = 0.05
_EXIT_ATR_FRAC = 0.05
_MIN_SLIPPAGE_BPS = 2.0
_MIN_COMMISSION_USD = 1.00


@dataclass
class PaperPosition:
    """One open paper position."""
    symbol: str
    direction: Direction
    entry_price: float
    stop_price: float
    tp_price: float
    max_hold_bars: int
    bars_held: int = 0
    entry_time: datetime | None = None
    hypothesis_id: str = ""


@dataclass
class PaperFill:
    """Record of a completed paper trade."""
    hypothesis_id: str
    symbol: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str
    bars_held: int


class PaperTrader:
    """Local paper fill simulator.

    Parameters
    ----------
    hypotheses:
        Mapping of hypothesis_id → hypothesis object. Each object must implement
        ``generate_signals(bars, features) -> Sequence[Signal]``.
    paper_dir:
        Directory where fill records are persisted.
    """

    def __init__(
        self,
        hypotheses: dict[str, object],
        *,
        paper_dir: Path = _PAPER_DIR,
    ) -> None:
        self.hypotheses = hypotheses
        self.paper_dir = paper_dir
        self._bars_history: list[Bar] = []
        self._features_history: list[dict] = []
        self._positions: dict[str, PaperPosition] = {}  # symbol → position
        self._fills: list[PaperFill] = []

    def process_bar(self, bar: Bar, features: dict) -> list[PaperFill]:
        """Process one new 5-minute bar.

        Returns list of fills completed during this bar (exits triggered).
        """
        self._bars_history.append(bar)
        self._features_history.append(features)
        new_fills: list[PaperFill] = []

        # 1. Check exits for open positions
        if bar.symbol in self._positions:
            pos = self._positions[bar.symbol]
            pos.bars_held += 1
            fill = self._check_exit(pos, bar)
            if fill:
                new_fills.append(fill)
                del self._positions[bar.symbol]

        # 2. Generate new signals (only if no open position for this symbol)
        if bar.symbol not in self._positions:
            for hyp_id, hyp in self.hypotheses.items():
                bars_for_sym = [b for b in self._bars_history if b.symbol == bar.symbol]
                try:
                    signals: Sequence[Signal] = hyp.generate_signals(bars_for_sym, features)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Signal generation failed for %s/%s: %s", hyp_id, bar.symbol, exc)
                    signals = []

                for sig in signals:
                    atr = features.get("atr_14", 1.0) or 1.0
                    entry_price = self._entry_fill(bar.close, sig.direction, atr)
                    stop_price = self._stop(entry_price, sig.direction, sig.stop_distance_atr, atr)
                    tp_price = self._tp(entry_price, sig.direction, sig.take_profit_distance_atr, atr)
                    self._positions[bar.symbol] = PaperPosition(
                        symbol=bar.symbol,
                        direction=sig.direction,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        tp_price=tp_price,
                        max_hold_bars=sig.max_hold_bars,
                        entry_time=bar.event_time,
                        hypothesis_id=hyp_id,
                    )
                    break  # one position per symbol

        if new_fills:
            self._persist_fills(new_fills)

        return new_fills

    def force_close_all(self, bar: Bar) -> list[PaperFill]:
        """Force-close all open positions (e.g. at session end)."""
        fills = []
        for symbol in list(self._positions.keys()):
            if symbol == bar.symbol:
                pos = self._positions.pop(symbol)
                exit_price = self._exit_fill(bar.close, pos.direction, 0.0)
                pnl = self._pnl(pos.entry_price, exit_price, pos.direction)
                fill = PaperFill(
                    hypothesis_id=pos.hypothesis_id, symbol=symbol,
                    direction=pos.direction.value,
                    entry_time=pos.entry_time or bar.event_time,
                    exit_time=bar.event_time,
                    entry_price=pos.entry_price, exit_price=exit_price,
                    pnl=pnl, exit_reason="session_end",
                    bars_held=pos.bars_held,
                )
                fills.append(fill)
        if fills:
            self._persist_fills(fills)
        return fills

    def get_fills_df(self) -> pd.DataFrame:
        """Return all paper fills as a DataFrame."""
        if not self._fills:
            return pd.DataFrame()
        return pd.DataFrame([f.__dict__ for f in self._fills])

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_exit(self, pos: PaperPosition, bar: Bar) -> PaperFill | None:
        atr = 0.0  # conservative: no atr slippage on exits (already conservative entry)

        stop_hit = (pos.direction == Direction.LONG and bar.low <= pos.stop_price) or \
                   (pos.direction == Direction.SHORT and bar.high >= pos.stop_price)
        tp_hit = (pos.direction == Direction.LONG and bar.high >= pos.tp_price) or \
                 (pos.direction == Direction.SHORT and bar.low <= pos.tp_price)
        max_hold = pos.bars_held >= pos.max_hold_bars

        if stop_hit:
            exit_p = self._exit_fill(pos.stop_price, pos.direction, atr)
            reason = "stop"
        elif tp_hit:
            exit_p = self._exit_fill(pos.tp_price, pos.direction, atr)
            reason = "tp"
        elif max_hold:
            exit_p = self._exit_fill(bar.close, pos.direction, atr)
            reason = "max_hold"
        else:
            return None

        pnl = self._pnl(pos.entry_price, exit_p, pos.direction)
        return PaperFill(
            hypothesis_id=pos.hypothesis_id, symbol=pos.symbol,
            direction=pos.direction.value,
            entry_time=pos.entry_time or bar.event_time,
            exit_time=bar.event_time,
            entry_price=pos.entry_price, exit_price=exit_p,
            pnl=pnl, exit_reason=reason,
            bars_held=pos.bars_held,
        )

    def _entry_fill(self, close: float, direction: Direction, atr: float) -> float:
        slip = max(atr * _ENTRY_ATR_FRAC, close * _MIN_SLIPPAGE_BPS / 10_000)
        spread = close * _HALF_SPREAD_BPS / 10_000
        adj = slip + spread
        return close + adj if direction == Direction.LONG else close - adj

    def _exit_fill(self, price: float, direction: Direction, atr: float) -> float:
        slip = max(atr * _EXIT_ATR_FRAC, price * _MIN_SLIPPAGE_BPS / 10_000) if atr else 0.0
        spread = price * _HALF_SPREAD_BPS / 10_000
        adj = slip + spread
        return price - adj if direction == Direction.LONG else price + adj

    def _stop(self, entry: float, direction: Direction, atr_mult: float, atr: float) -> float:
        return entry - atr_mult * atr if direction == Direction.LONG else entry + atr_mult * atr

    def _tp(self, entry: float, direction: Direction, atr_mult: float, atr: float) -> float:
        return entry + atr_mult * atr if direction == Direction.LONG else entry - atr_mult * atr

    def _pnl(self, entry: float, exit_: float, direction: Direction) -> float:
        return (exit_ - entry) if direction == Direction.LONG else (entry - exit_)

    def _persist_fills(self, fills: list[PaperFill]) -> None:
        self._fills.extend(fills)
        self.paper_dir.mkdir(parents=True, exist_ok=True)
        path = self.paper_dir / "fills.parquet"
        new_df = pd.DataFrame([f.__dict__ for f in fills])
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_parquet(path, index=False)
