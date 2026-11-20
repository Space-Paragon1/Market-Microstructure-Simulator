"""Strategy interface and the baseline strategies.

A strategy never touches the book directly. Its hooks receive a read-only view
of the book and return :class:`Action` objects; the simulator delivers order
messages after the strategy's ``latency`` and attributes the resulting fills
back to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from microbook import ExecutionReport, LimitOrderBook, Order, OrderStatus, Side
from microbook.types import Fill

from .events import EventType
from .portfolio import Portfolio


@dataclass
class Action:
    """A request from a strategy: an order message or a timer."""

    time: int
    etype: EventType
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def submit(cls, order: Order, time: int) -> Action:
        return cls(time, EventType.SUBMIT, {"order": order})

    @classmethod
    def cancel(cls, order_id: str, time: int) -> Action:
        return cls(time, EventType.CANCEL, {"order_id": order_id})

    @classmethod
    def modify(cls, order_id: str, time: int, *, new_price: float | None = None, new_qty: int | None = None) -> Action:
        mod: dict[str, Any] = {}
        if new_price is not None:
            mod["new_price"] = new_price
        if new_qty is not None:
            mod["new_qty"] = new_qty
        return cls(time, EventType.MODIFY, {"order_id": order_id, "modify": mod})

    @classmethod
    def timer(cls, time: int) -> Action:
        return cls(time, EventType.TIMER)


class Strategy:
    """Base strategy. Override any of the hooks; each returns a list of actions.

    ===================  ==========================================================
    ``on_start``         once, when the simulation starts
    ``on_tick``          on every ``SNAPSHOT`` event
    ``on_timer``         on a ``TIMER`` this strategy scheduled (``Action.timer``)
    ``on_fill``          for each fill of one of this strategy's orders
    ``on_report``        for each execution report of a message this strategy sent
    ===================  ==========================================================

    ``latency`` delays every order message (submit/cancel/modify) by that many
    time units between the decision and its arrival at the book.
    """

    def __init__(self, name: str, portfolio: Portfolio | None = None, *, latency: int = 0) -> None:
        if latency < 0:
            raise ValueError("latency must be >= 0")
        self.name = name
        self.portfolio = portfolio if portfolio is not None else Portfolio()
        self.latency = latency
        # Ids of orders scheduled *outside* the strategy (e.g. ``sim.schedule``)
        # that should still be attributed to it. Orders a strategy sends via
        # actions are attributed automatically.
        self.my_order_ids: set[str] = set()
        self._oid_seq = 0

    def next_order_id(self, tag: str = "o") -> str:
        """A fresh order id, unique within this strategy."""
        self._oid_seq += 1
        return f"{self.name}:{tag}{self._oid_seq}"

    # --- hooks -------------------------------------------------------------

    def on_start(self, now: int, lob: LimitOrderBook) -> list[Action]:
        return []

    def on_tick(self, now: int, lob: LimitOrderBook) -> list[Action]:
        return []

    def on_timer(self, now: int, lob: LimitOrderBook) -> list[Action]:
        return []

    def on_fill(self, now: int, fill: Fill, side: Side, is_maker: bool, lob: LimitOrderBook) -> list[Action]:
        return []

    def on_report(self, now: int, report: ExecutionReport) -> list[Action]:
        return []

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class TwoSidedQuoter(Strategy):
    """Base class for market makers that keep at most one bid and one ask.

    Subclasses implement :meth:`compute_quotes`. Quote management rules:

    * prices are rounded *passively* onto the book's grid (bids down, asks up);
    * quotes never cross the opposite touch: they are slid to one tick behind
      it and sent post-only, so a maker never pays to take liquidity;
    * an existing quote whose price is unchanged is left alone, preserving its
      queue priority; otherwise it is cancelled and replaced.
    """

    def __init__(
        self,
        name: str,
        *,
        tick_interval: int,
        latency: int = 0,
        portfolio: Portfolio | None = None,
    ) -> None:
        super().__init__(name, portfolio, latency=latency)
        if tick_interval <= 0:
            raise ValueError("tick_interval must be positive")
        self.tick_interval = tick_interval
        self._last_quote_t = -(10**18)
        # side -> (order_id, price, sent_at) of the latest quote sent
        self._quotes: dict[Side, tuple[str, float, int] | None] = {Side.BUY: None, Side.SELL: None}

    def compute_quotes(self, now: int, lob: LimitOrderBook) -> tuple[float | None, float | None, int, int] | None:
        """Return ``(bid_px, ask_px, bid_qty, ask_qty)``; a ``None`` price pulls that
        side. Return ``None`` to leave all quotes untouched this tick."""
        raise NotImplementedError

    def on_tick(self, now: int, lob: LimitOrderBook) -> list[Action]:
        if now - self._last_quote_t < self.tick_interval:
            return []
        quotes = self.compute_quotes(now, lob)
        if quotes is None:
            return []
        self._last_quote_t = now
        bid_px, ask_px, bid_qty, ask_qty = quotes
        return self.requote(now, lob, bid_px, ask_px, bid_qty, ask_qty)

    def on_report(self, now: int, report: ExecutionReport) -> list[Action]:
        if report.status is OrderStatus.REJECTED:
            for side, q in self._quotes.items():
                if q is not None and q[0] == report.order_id:
                    self._quotes[side] = None
        return []

    def requote(
        self,
        now: int,
        lob: LimitOrderBook,
        bid_px: float | None,
        ask_px: float | None,
        bid_qty: int,
        ask_qty: int,
    ) -> list[Action]:
        tick = lob.tick_size
        bb, ba = lob.best_bid(), lob.best_ask()
        if bid_px is not None:
            bid_px = lob.round_price(bid_px, Side.BUY)
            if ba is not None and bid_px >= ba:
                bid_px = lob.price_of(lob.to_ticks(ba) - 1)
        if ask_px is not None:
            ask_px = lob.round_price(ask_px, Side.SELL)
            if bb is not None and ask_px <= bb:
                ask_px = lob.price_of(lob.to_ticks(bb) + 1)
        if bid_px is not None and ask_px is not None and bid_px >= ask_px:
            ask_px = lob.price_of(lob.to_ticks(bid_px) + 1)
        if bid_px is not None and bid_px < tick:
            bid_px = None

        actions: list[Action] = []
        self._update_side(now, lob, Side.BUY, bid_px, bid_qty, actions)
        self._update_side(now, lob, Side.SELL, ask_px, ask_qty, actions)
        return actions

    def cancel_all(self, now: int) -> list[Action]:
        actions = []
        for side, q in self._quotes.items():
            if q is not None:
                actions.append(Action.cancel(q[0], now))
                self._quotes[side] = None
        return actions

    def _update_side(
        self, now: int, lob: LimitOrderBook, side: Side, px: float | None, qty: int, actions: list[Action]
    ) -> None:
        cur = self._quotes[side]
        if cur is not None:
            oid, cur_px, sent_at = cur
            alive = oid in lob or sent_at + self.latency >= now  # resting, or still in flight
            if alive and px is not None and cur_px == px:
                return  # unchanged: keep queue priority
            if alive:
                actions.append(Action.cancel(oid, now))
            self._quotes[side] = None
        if px is None or qty <= 0:
            return
        oid = self.next_order_id("b" if side is Side.BUY else "a")
        actions.append(Action.submit(Order.limit(oid, side, px, qty, ts=now, post_only=True), now))
        self._quotes[side] = (oid, px, now)


class MarketMaker(TwoSidedQuoter):
    """Baseline symmetric market maker.

    Quotes ``mid +/- half_spread_ticks`` and, when carrying inventory, shifts
    both quotes by ``inventory_skew_ticks`` against the position (long -> lower,
    to attract buyers; short -> higher). ``tick_size`` is the quoting unit and
    defaults to the book's tick.
    """

    def __init__(
        self,
        name: str = "mm",
        *,
        tick_size: float | None = None,
        half_spread_ticks: int = 1,
        size: int = 5,
        tick_interval: int = 10,
        inventory_skew_ticks: int = 2,
        latency: int = 0,
    ) -> None:
        super().__init__(name, tick_interval=tick_interval, latency=latency)
        self.tick_size = tick_size
        self.half_spread_ticks = half_spread_ticks
        self.size = size
        self.inventory_skew_ticks = inventory_skew_ticks

    def compute_quotes(self, now: int, lob: LimitOrderBook) -> tuple[float | None, float | None, int, int] | None:
        mid = lob.midprice()
        if mid is None:
            return None
        q = self.tick_size or lob.tick_size
        inv = self.portfolio.position
        skew = -self.inventory_skew_ticks if inv > 0 else self.inventory_skew_ticks if inv < 0 else 0
        bid = mid + (skew - self.half_spread_ticks) * q
        ask = mid + (skew + self.half_spread_ticks) * q
        return bid, ask, self.size, self.size


class TWAPExecutor(Strategy):
    """Time-weighted execution of ``total_qty`` between ``start`` and ``end``.

    Sends a market-order slice every ``tick_interval`` (driven by SNAPSHOT
    ticks). Slices are sized so the remainder is spread evenly over the slices
    left; the final slice sends everything outstanding. Quantity that a market
    order could not fill (thin book) is returned to the schedule. It waits for
    a two-sided market so every slice has a well-defined arrival price.
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
        latency: int = 0,
    ) -> None:
        super().__init__(name=name, latency=latency)
        if total_qty <= 0:
            raise ValueError("total_qty must be positive")
        if end < start:
            raise ValueError("end must be >= start")
        self.side = side
        self.total_qty = total_qty
        self.start = start
        self.end = end
        self.tick_interval = tick_interval
        self._sent = 0  # sent and not (yet) known to be unfilled
        self._last_t = -(10**18)

    @property
    def remaining(self) -> int:
        return self.total_qty - self._sent

    def on_tick(self, now: int, lob: LimitOrderBook) -> list[Action]:
        if now < self.start or now > self.end:
            return []
        if now - self._last_t < self.tick_interval:
            return []
        if self.remaining <= 0 or lob.midprice() is None:
            return []

        slices_left = max(1, (self.end - now) // self.tick_interval + 1)
        qty = self.remaining if slices_left == 1 else max(1, self.remaining // slices_left)

        self._sent += qty
        self._last_t = now
        return [Action.submit(Order.market(self.next_order_id(), self.side, qty, ts=now), now)]

    def on_report(self, now: int, report: ExecutionReport) -> list[Action]:
        self._sent -= report.canceled_qty  # unfilled market remainder goes back on the schedule
        return []
