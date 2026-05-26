"""Shared domain dataclasses for the autoresearch trading system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class HypothesisStatus(str, Enum):
    DRAFT = "draft"
    BACKTEST_CANDIDATE = "backtest_candidate"
    VALIDATED_CANDIDATE = "validated_candidate"
    PAPER_ACTIVE = "paper_active"
    PAPER_PROMOTED = "paper_promoted"
    QUARANTINED = "quarantined"
    RETIRED = "retired"


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar for a single symbol.

    event_time     : close time of the bar (timezone-aware)
    available_time : when this bar became available to the system
    symbol         : ticker, e.g. 'SPY'
    open / high / low / close : prices
    volume         : traded volume for the bar period
    vwap           : volume-weighted average price (optional, set when available)
    """

    event_time: datetime
    available_time: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"Bar.high ({self.high}) < Bar.low ({self.low}) for {self.symbol}")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"Bar.high is inconsistent for {self.symbol}")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Bar.low is inconsistent for {self.symbol}")
        if self.volume < 0:
            raise ValueError(f"Bar.volume must be non-negative for {self.symbol}")
        if self.available_time < self.event_time:
            raise ValueError(
                f"Bar.available_time ({self.available_time}) cannot be before "
                f"Bar.event_time ({self.event_time}) for {self.symbol}"
            )


@dataclass(frozen=True)
class Signal:
    """Entry signal produced by a hypothesis strategy.

    hypothesis_id         : unique ID of the hypothesis that produced this signal
    symbol                : ticker
    bar_time              : event_time of the bar that triggered the signal
    direction             : LONG, SHORT, or FLAT (no trade)
    confidence            : [0.0, 1.0] strength of the signal
    stop_distance_atr     : stop-loss distance expressed in ATR multiples
    take_profit_distance_atr : take-profit distance expressed in ATR multiples
    max_hold_bars         : maximum number of bars to hold the trade
    """

    hypothesis_id: str
    symbol: str
    bar_time: datetime
    direction: Direction
    confidence: float
    stop_distance_atr: float
    take_profit_distance_atr: float
    max_hold_bars: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Signal.confidence must be in [0, 1], got {self.confidence}")
        if self.stop_distance_atr <= 0:
            raise ValueError(f"Signal.stop_distance_atr must be positive, got {self.stop_distance_atr}")
        if self.take_profit_distance_atr <= 0:
            raise ValueError(f"Signal.take_profit_distance_atr must be positive")
        if self.max_hold_bars < 1:
            raise ValueError(f"Signal.max_hold_bars must be >= 1, got {self.max_hold_bars}")


@dataclass
class Order:
    """Paper or live order derived from a signal.

    order_id         : system-assigned unique id
    hypothesis_id    : source hypothesis
    symbol           : ticker
    direction        : LONG or SHORT
    quantity         : number of shares / units
    limit_price      : limit price (None = market order — disabled initially)
    stop_price       : stop-loss price for bracket order child
    take_profit_price: take-profit price for bracket order child
    status           : current order lifecycle state
    created_at       : when the order was created
    filled_at        : when the fill was recorded (None until filled)
    fill_price       : actual fill price (None until filled)
    """

    order_id: str
    hypothesis_id: str
    symbol: str
    direction: Direction
    quantity: float
    stop_price: float
    take_profit_price: float
    status: OrderStatus = OrderStatus.PENDING
    limit_price: float | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: datetime | None = None
    fill_price: float | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order.quantity must be positive, got {self.quantity}")
        if self.stop_price <= 0:
            raise ValueError(f"Order.stop_price must be positive")
        if self.take_profit_price <= 0:
            raise ValueError(f"Order.take_profit_price must be positive")


@dataclass(frozen=True)
class EvalResult:
    """Evaluation result for one hypothesis run.

    Covers all required metrics from research_policy.yaml.
    """

    hypothesis_id: str
    run_id: str
    status: HypothesisStatus

    # Core PnL
    net_pnl: float
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float

    # Drawdown / risk
    max_drawdown: float
    max_intraday_drawdown: float
    worst_day: float
    longest_losing_streak: int

    # Activity
    trades_per_day: float
    exposure_time: float
    total_trades: int

    # Robustness
    slippage_sensitivity: float

    # Composite score set by evaluator
    composite_score: float

    # Optional walk-forward coverage
    oos_net_pnl: float | None = None
    oos_profit_factor: float | None = None

    @property
    def avg_win_to_avg_loss(self) -> float | None:
        """Ratio of average win to average loss magnitude. None if no losses."""
        if self.avg_loss == 0.0:
            return None
        return abs(self.avg_win / self.avg_loss)

    def __post_init__(self) -> None:
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError(f"EvalResult.win_rate must be in [0, 1], got {self.win_rate}")
        if self.profit_factor < 0:
            raise ValueError(f"EvalResult.profit_factor must be non-negative")
        if self.total_trades < 0:
            raise ValueError(f"EvalResult.total_trades must be non-negative")
        if not 0.0 <= self.exposure_time <= 1.0:
            raise ValueError(f"EvalResult.exposure_time must be in [0, 1]")
