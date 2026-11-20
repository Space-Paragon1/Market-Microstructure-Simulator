from __future__ import annotations

import math
from dataclasses import dataclass

from .types import OrderType, Side, TimeInForce


@dataclass(slots=True)
class Order:
    """An order message.

    ``qty`` is the *open* quantity: the book decrements it as the order trades,
    so after a submit it equals the quantity left resting (0 if none).

    ``price`` is required for LIMIT orders and ignored for MARKET orders.
    ``ts`` is informational (the simulator stamps it with the arrival time);
    queue priority is determined by arrival order at the book, not by ``ts``.
    ``owner`` identifies the participant for self-trade prevention.
    """

    order_id: str
    side: Side
    price: float | None
    qty: int
    ts: int = 0
    order_type: OrderType = OrderType.LIMIT
    tif: TimeInForce = TimeInForce.GTC
    post_only: bool = False
    owner: str | None = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("qty must be positive")
        if self.order_type is OrderType.LIMIT:
            if self.price is None or not math.isfinite(self.price) or self.price <= 0:
                raise ValueError("price must be positive")
        elif self.post_only:
            raise ValueError("a market order cannot be post-only")

    @classmethod
    def limit(
        cls,
        order_id: str,
        side: Side,
        price: float,
        qty: int,
        *,
        ts: int = 0,
        tif: TimeInForce = TimeInForce.GTC,
        post_only: bool = False,
        owner: str | None = None,
    ) -> Order:
        return cls(order_id, side, price, qty, ts, OrderType.LIMIT, tif, post_only, owner)

    @classmethod
    def market(
        cls,
        order_id: str,
        side: Side,
        qty: int,
        *,
        ts: int = 0,
        owner: str | None = None,
    ) -> Order:
        return cls(order_id, side, None, qty, ts, OrderType.MARKET, TimeInForce.IOC, False, owner)

    @property
    def is_market(self) -> bool:
        return self.order_type is OrderType.MARKET
