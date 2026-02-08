from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from microbook import LimitOrderBook, Order, Side
from microbook.types import Fill
from .events import EventType
from .portfolio import Portfolio


@dataclass
class Action:
    time: int
    etype: EventType
    kwargs: Dict


class Strategy:
    """
    Base strategy. Subclasses implement:
    - on_tick: called periodically (e.g., every N timesteps)
    - on_fills: called after simulator processes fills
    """
    def __init__(self, name: str, portfolio: Optional[Portfolio] = None) -> None:
        self.name = name
        self.portfolio = portfolio or Portfolio()
        self.my_order_ids: Set[str] = set()
        self._ts_counter = 10_000_000  # ensure unique ts for our orders

    def _next_ts(self, now: int) -> int:
        self._ts_counter += 1
        return max(now, self._ts_counter)

    def on_tick(self, now: int, lob: LimitOrderBook) -> List[Action]:
        return []

    def on_fills(self, fills: List[Fill], lob: LimitOrderBook) -> None:
        """
        Update portfolio for fills that hit our orders.
        We infer whether fill involved our order via order_id membership.
        """
        for f in fills:
            if f.maker_order_id in self.my_order_ids:
                # We don't know side directly from Fill, so you should maintain an order registry.
                # For v1: encode side into order_id or keep a dict in strategy.
                pass
            if f.taker_order_id in self.my_order_ids:
                pass


class MarketMaker(Strategy):
    """
    Simple symmetric market maker:
    - Maintains one bid and one ask around mid
    - Cancels and replaces quotes every tick_interval
    - Inventory-aware skew: shifts quotes if inventory is large
    """
    def __init__(
        self,
        name: str = "mm",
        *,
        tick_size: float = 1.0,
        half_spread_ticks: int = 1,
        size: int = 5,
        tick_interval: int = 10,
        inventory_skew_ticks: int = 2,
    ) -> None:
        super().__init__(name=name, portfolio=Portfolio())
        self.tick_size = tick_size
        self.half_spread_ticks = half_spread_ticks
        self.size = size
        self.tick_interval = tick_interval
        self.inventory_skew_ticks = inventory_skew_ticks

        self._last_quote_t = -10**9
        self._bid_id = f"{name}_bid"
        self._ask_id = f"{name}_ask"
        self._id_to_side = {self._bid_id: Side.BUY, self._ask_id: Side.SELL}
        self.my_order_ids.update([self._bid_id, self._ask_id])

    def on_tick(self, now: int, lob: LimitOrderBook) -> List[Action]:
        if now - self._last_quote_t < self.tick_interval:
            return []

        mid = lob.midprice()
        if mid is None:
            return []

        # Inventory skew: if long -> quote lower to encourage sells; if short -> quote higher to encourage buys
        inv = self.portfolio.position
        skew = 0
        if inv > 0:
            skew = -self.inventory_skew_ticks
        elif inv < 0:
            skew = +self.inventory_skew_ticks

        bid_px = mid - (self.half_spread_ticks - skew) * self.tick_size
        ask_px = mid + (self.half_spread_ticks + skew) * self.tick_size

        actions: List[Action] = []

        # Cancel old quotes (idempotent cancel is fine)
        actions.append(Action(now, EventType.CANCEL, {"order_id": self._bid_id}))
        actions.append(Action(now, EventType.CANCEL, {"order_id": self._ask_id}))

        # Place new quotes with same IDs (priority resets each time)
        actions.append(
            Action(
                now,
                EventType.SUBMIT,
                {"order": Order(self._bid_id, Side.BUY, float(bid_px), self.size, ts=self._next_ts(now))},
            )
        )
        actions.append(
            Action(
                now,
                EventType.SUBMIT,
                {"order": Order(self._ask_id, Side.SELL, float(ask_px), self.size, ts=self._next_ts(now))},
            )
        )

        self._last_quote_t = now
        return actions


class TWAPExecutor(Strategy):
    """
    Simple TWAP-style taker:
    - Buys or sells a total_qty evenly over a window
    - Uses market orders in slices every tick_interval
    """
    def __init__(
        self,
        side: Side,
        total_qty: int,
        start: int,
        end: int,
        *,
        tick_interval: int = 10,
        name: str = "twap",
    ) -> None:
        super().__init__(name=name, portfolio=Portfolio())
        self.side = side
        self.total_qty = total_qty
        self.start = start
        self.end = end
        self.tick_interval = tick_interval

        self._sent = 0
        self._last_t = -10**9
        self._id_to_side: Dict[str, Side] = {}

    def on_tick(self, now: int, lob: LimitOrderBook) -> List[Action]:
        if now < self.start or now > self.end:
            return []
        if now - self._last_t < self.tick_interval:
            return []

        remaining = self.total_qty - self._sent
        if remaining <= 0:
            return []

        # number of slices left (inclusive)
        slices_left = max(1, ((self.end - now) // self.tick_interval) + 1)
        qty = max(1, remaining // slices_left)

        oid = f"{self.name}_{now}"
        o = Order(oid, self.side, 1.0, qty, ts=self._next_ts(now))
        setattr(o, "_is_market", True)

        self.my_order_ids.add(oid)
        self._id_to_side[oid] = self.side

        self._sent += qty
        self._last_t = now

        return [Action(now, EventType.SUBMIT, {"order": o})]
