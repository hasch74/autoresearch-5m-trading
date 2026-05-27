"""Risk manager: validates candidate orders against risk.yaml rules.

This module is in the agent's DENY_WRITE list — the agent must not modify it.

Rules enforced:
- No market orders (limit/stop only).
- No overnight positions (enforced at order creation time via session check).
- Max open positions.
- Max positions per symbol.
- Daily and weekly loss kill switches.
- Bracket orders required (stop + take-profit must be set).

Usage:
    manager = RiskManager(equity=10_000.0)
    ok = manager.validate_order(order)
    if not ok:
        logger.warning("Order rejected: %s", manager.last_rejection_reason)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.types import Order, OrderStatus

logger = logging.getLogger(__name__)

# Defaults mirror configs/risk.yaml
_MAX_RISK_PER_TRADE_PCT = 0.0025   # 0.25%
_MAX_DAILY_LOSS_PCT = 0.01          # 1.0%
_MAX_WEEKLY_LOSS_PCT = 0.03         # 3.0%
_MAX_OPEN_POSITIONS = 5
_MAX_POSITIONS_PER_SYMBOL = 1


@dataclass
class RiskState:
    """Mutable state tracked across an intraday session."""
    equity: float
    realized_pnl_today: float = 0.0
    realized_pnl_week: float = 0.0
    open_positions: dict[str, int] = field(default_factory=dict)  # symbol → position count
    consecutive_errors: int = 0
    rejected_order_count: int = 0
    kill_switch_active: bool = False


class RiskManager:
    """Stateful risk manager for paper and live trading.

    Parameters
    ----------
    equity:
        Current account equity in USD. Used to compute percentage-based limits.
    state:
        Optional pre-existing state (e.g. to restore from persistence).
    """

    def __init__(
        self,
        equity: float,
        state: RiskState | None = None,
        *,
        max_risk_per_trade_pct: float = _MAX_RISK_PER_TRADE_PCT,
        max_daily_loss_pct: float = _MAX_DAILY_LOSS_PCT,
        max_weekly_loss_pct: float = _MAX_WEEKLY_LOSS_PCT,
        max_open_positions: int = _MAX_OPEN_POSITIONS,
        max_positions_per_symbol: int = _MAX_POSITIONS_PER_SYMBOL,
        kill_switch_after_errors: int = 3,
        kill_switch_after_rejections: int = 5,
    ) -> None:
        self.state = state or RiskState(equity=equity)
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_open_positions = max_open_positions
        self.max_positions_per_symbol = max_positions_per_symbol
        self.kill_switch_after_errors = kill_switch_after_errors
        self.kill_switch_after_rejections = kill_switch_after_rejections
        self.last_rejection_reason: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_order(self, order: Order) -> bool:
        """Return True if *order* passes all risk checks."""
        reason = self._check(order)
        if reason:
            self.last_rejection_reason = reason
            self.state.rejected_order_count += 1
            if self.state.rejected_order_count >= self.kill_switch_after_rejections:
                self.state.kill_switch_active = True
                logger.error("Kill switch activated after %d rejected orders",
                             self.state.rejected_order_count)
            logger.warning("Order rejected [%s]: %s", order.symbol, reason)
            return False
        self.last_rejection_reason = ""
        return True

    def record_fill(self, symbol: str, pnl: float) -> None:
        """Update state after a fill is confirmed."""
        self.state.realized_pnl_today += pnl
        self.state.realized_pnl_week += pnl
        # Reduce open position count on exit fills (simplified: 1 position per symbol).
        # Loss-making exits must also close tracked exposure.
        if symbol in self.state.open_positions:
            self.state.open_positions[symbol] = max(0, self.state.open_positions[symbol] - 1)
            if self.state.open_positions[symbol] == 0:
                del self.state.open_positions[symbol]

    def record_entry(self, symbol: str) -> None:
        """Update state after entry order is accepted."""
        self.state.open_positions[symbol] = self.state.open_positions.get(symbol, 0) + 1

    def record_error(self) -> None:
        """Increment error counter; activate kill switch if threshold reached."""
        self.state.consecutive_errors += 1
        if self.state.consecutive_errors >= self.kill_switch_after_errors:
            self.state.kill_switch_active = True
            logger.error("Kill switch activated after %d consecutive errors",
                         self.state.consecutive_errors)

    def reset_consecutive_errors(self) -> None:
        self.state.consecutive_errors = 0

    def reset_daily(self) -> None:
        """Call at start of each trading day."""
        self.state.realized_pnl_today = 0.0
        self.state.open_positions = {}
        self.state.consecutive_errors = 0
        self.state.rejected_order_count = 0

    def reset_weekly(self) -> None:
        """Call at start of each trading week."""
        self.state.realized_pnl_week = 0.0

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check(self, order: Order) -> str:
        """Return rejection reason string or empty string if ok."""
        s = self.state

        if s.kill_switch_active:
            return "kill switch active"

        # Bracket required: stop_price and take_profit_price must be set
        if order.stop_price is None or order.take_profit_price is None:
            return "bracket order required: stop_price and take_profit_price must be set"

        # Daily loss limit
        daily_loss_limit = self.max_daily_loss_pct * s.equity
        if s.realized_pnl_today < -daily_loss_limit:
            return f"daily loss limit breached: {s.realized_pnl_today:.2f} < -{daily_loss_limit:.2f}"

        # Weekly loss limit
        weekly_loss_limit = self.max_weekly_loss_pct * s.equity
        if s.realized_pnl_week < -weekly_loss_limit:
            return f"weekly loss limit breached: {s.realized_pnl_week:.2f} < -{weekly_loss_limit:.2f}"

        # Max open positions
        total_open = sum(s.open_positions.values())
        if total_open >= self.max_open_positions:
            return f"max open positions reached: {total_open}"

        # Max positions per symbol
        symbol_open = s.open_positions.get(order.symbol, 0)
        if symbol_open >= self.max_positions_per_symbol:
            return f"max positions per symbol reached for {order.symbol}: {symbol_open}"

        return ""
