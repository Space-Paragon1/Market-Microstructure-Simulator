"""Core value types shared by the matching engine and the simulator."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """+1 for BUY, -1 for SELL. Handy for signed quantities and PnL."""
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class TimeInForce(str, Enum):
    GTC = "GTC"  # good-till-cancel: remainder rests on the book
    IOC = "IOC"  # immediate-or-cancel: fill what you can, cancel the rest
    FOK = "FOK"  # fill-or-kill: fill the whole quantity immediately or do nothing


class OrderStatus(str, Enum):
    NEW = "NEW"  # resting, no fills on this message
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # some fills, remainder resting
    FILLED = "FILLED"  # fully executed
    CANCELED = "CANCELED"  # unfilled remainder removed (IOC/FOK/market/STP/cancel)
    REJECTED = "REJECTED"  # message refused; book unchanged


class STPMode(str, Enum):
    """Self-trade prevention: what happens when an order would trade with a
    resting order from the same ``owner``."""

    NONE = "NONE"  # allow self-trades
    CANCEL_RESTING = "CANCEL_RESTING"  # cancel the resting order, keep matching
    CANCEL_INCOMING = "CANCEL_INCOMING"  # stop matching, cancel the incoming remainder


class Fill(NamedTuple):
    """A single execution between an aggressing (taker) and a resting (maker) order.

    Immutable record (a ``NamedTuple`` rather than a frozen dataclass: it is
    created on the matching hot path, where frozen-dataclass construction costs
    ~2x more).
    """

    taker_order_id: str
    maker_order_id: str
    price: float
    qty: int
    aggressor_side: Side | None = None
    ts: int = 0


class ExecutionReport(NamedTuple):
    """Outcome of a single message (submit or modify) sent to the book."""

    order_id: str
    status: OrderStatus
    fills: tuple[Fill, ...] = ()
    resting_qty: int = 0  # quantity left on the book after this message
    canceled_qty: int = 0  # quantity removed without trading (IOC/FOK/market/STP)
    stp_canceled: tuple[str, ...] = ()  # resting order ids removed by self-trade prevention
    reason: str | None = None  # populated for REJECTED

    @property
    def accepted(self) -> bool:
        return self.status is not OrderStatus.REJECTED

    @property
    def filled_qty(self) -> int:
        return sum(f.qty for f in self.fills)


class OrderRejected(ValueError):
    """Raised by the convenience entry points (``place_limit``/``place_market``)
    when the book refuses an order. ``submit`` returns a REJECTED report instead."""

    def __init__(self, report: ExecutionReport) -> None:
        super().__init__(f"order {report.order_id!r} rejected: {report.reason}")
        self.report = report
