"""Price-time priority central limit order book (CLOB).

Design notes
------------
* Prices are stored as **integer ticks**. Float prices are only used at the API
  boundary, so there is no float-key aliasing (``0.1 + 0.2`` vs ``0.3``) and
  off-grid prices are rejected rather than silently creating a new level.
* Each side keeps an ascending list of *sort keys* (``tick`` for bids,
  ``-tick`` for asks). The best price is always the **last** element, so reading
  the touch and removing an exhausted touch level are both O(1).
* Each level is an ``OrderedDict`` keyed by order id: FIFO for matching and
  O(1) removal for cancels.
* Aggregate quantity per level is maintained incrementally, so ``depth()`` is
  O(levels requested).

Complexity (L = price levels on a side):

==================  ===============================
add (new level)     O(log L) search + O(L) memmove
add (existing)      O(1)
cancel              O(1), plus O(log L) if the level empties
modify (reduce)     O(1), keeps queue priority
match               O(1) per fill / per level crossed
==================  ===============================
"""

from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import OrderedDict
from collections.abc import Iterator
from decimal import Decimal

from .order import Order
from .types import (
    ExecutionReport,
    Fill,
    OrderRejected,
    OrderStatus,
    OrderType,
    Side,
    STPMode,
    TimeInForce,
)

_NEG_INF = -math.inf

# Fast record construction for the hot path (bypasses NamedTuple.__new__'s
# Python-level argument handling; field order must match ExecutionReport).
_report = tuple.__new__
_ER = ExecutionReport
_NEW = OrderStatus.NEW
_PARTIAL = OrderStatus.PARTIALLY_FILLED
_FILLED = OrderStatus.FILLED
_CANCELED = OrderStatus.CANCELED
if ExecutionReport._fields != (
    "order_id",
    "status",
    "fills",
    "resting_qty",
    "canceled_qty",
    "stp_canceled",
    "reason",
):  # pragma: no cover - guards the positional construction above
    raise ImportError("ExecutionReport field order changed; update book._execute")


class _BookSide:
    """One side of the book, indexed by integer tick."""

    __slots__ = ("keys", "levels", "qty", "side", "sign")

    def __init__(self, side: Side) -> None:
        self.side = side
        self.sign = 1 if side is Side.BUY else -1
        self.levels: dict[int, OrderedDict[str, Order]] = {}
        self.keys: list[int] = []  # ascending sort keys; best price is keys[-1]
        self.qty: dict[int, int] = {}

    def best_tick(self) -> int | None:
        return self.sign * self.keys[-1] if self.keys else None

    def add(self, tick: int, order: Order) -> None:
        level = self.levels.get(tick)
        if level is None:
            level = self.levels[tick] = OrderedDict()
            insort(self.keys, self.sign * tick)
            self.qty[tick] = 0
        level[order.order_id] = order
        self.qty[tick] += order.qty

    def remove(self, tick: int, order: Order) -> None:
        level = self.levels[tick]
        del level[order.order_id]
        self.qty[tick] -= order.qty
        if not level:
            self.drop_level(tick)

    def drop_level(self, tick: int) -> None:
        del self.levels[tick]
        del self.qty[tick]
        key = self.sign * tick
        if self.keys[-1] == key:
            self.keys.pop()
        else:
            del self.keys[bisect_left(self.keys, key)]

    def ticks(self) -> Iterator[int]:
        """Ticks from best to worst."""
        s = self.sign
        for k in reversed(self.keys):
            yield s * k


class LimitOrderBook:
    """Price-time priority limit order book.

    Parameters
    ----------
    tick_size:
        Minimum price increment. Limit prices must lie on this grid.
    stp:
        Self-trade prevention policy, applied when an incoming order would
        trade against a resting order with the same (non-``None``) ``owner``.

    Order entry
    -----------
    ``submit(order)`` is the primary entry point and returns an
    :class:`ExecutionReport`; it never raises for business rejections.
    ``place_limit`` / ``place_market`` are conveniences that return the list of
    fills and raise :class:`OrderRejected` if the book refuses the order.
    """

    def __init__(self, tick_size: float = 0.01, *, stp: STPMode = STPMode.NONE) -> None:
        if not math.isfinite(tick_size) or tick_size <= 0:
            raise ValueError("tick_size must be a positive finite number")
        self.tick_size = float(tick_size)
        self.stp = STPMode(stp)
        exponent = Decimal(repr(self.tick_size)).normalize().as_tuple().exponent
        self._decimals = max(0, -exponent) if isinstance(exponent, int) else 0
        self._bids = _BookSide(Side.BUY)
        self._asks = _BookSide(Side.SELL)
        # order_id -> (order, tick) for every resting order
        self._orders: dict[str, tuple[Order, int]] = {}
        self._px_cache: dict[int, float] = {}

    # ------------------------------------------------------------------ #
    # Price grid                                                          #
    # ------------------------------------------------------------------ #

    def price_of(self, tick: int) -> float:
        """Float price of an integer tick, rounded to the grid's precision."""
        px = self._px_cache.get(tick)
        if px is None:
            px = self._px_cache[tick] = round(tick * self.tick_size, self._decimals)
        return px

    def to_ticks(self, price: float) -> int:
        """Integer tick of an on-grid price. Raises ``ValueError`` if off-grid."""
        tick = self._tick_or_none(price)
        if tick is None:
            raise ValueError(f"price {price!r} is not a multiple of tick size {self.tick_size}")
        return tick

    def round_price(self, price: float, side: Side) -> float:
        """Round a price onto the grid *passively*: bids down, asks up."""
        x = price / self.tick_size
        nearest = round(x)
        if abs(x - nearest) <= 1e-6:
            return self.price_of(int(nearest))
        return self.price_of(math.floor(x) if side is Side.BUY else math.ceil(x))

    def _tick_or_none(self, price: float) -> int | None:
        x = price / self.tick_size
        tick = round(x)
        if abs(x - tick) > 1e-6:
            return None
        return int(tick)

    # ------------------------------------------------------------------ #
    # Market data                                                         #
    # ------------------------------------------------------------------ #

    def best_bid(self) -> float | None:
        keys = self._bids.keys
        return self.price_of(keys[-1]) if keys else None

    def best_ask(self) -> float | None:
        keys = self._asks.keys
        return self.price_of(-keys[-1]) if keys else None

    def midprice(self) -> float | None:
        b, a = self._bids.keys, self._asks.keys
        if not b or not a:
            return None
        return (self.price_of(b[-1]) + self.price_of(-a[-1])) / 2.0

    def spread(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return round(a - b, self._decimals)

    def microprice(self) -> float | None:
        """Size-weighted mid: leans toward the side with *less* resting size,
        i.e. toward where the price is more likely to move next."""
        bt, at = self._bids.best_tick(), self._asks.best_tick()
        if bt is None or at is None:
            return None
        qb, qa = self._bids.qty[bt], self._asks.qty[at]
        b, a = self.price_of(bt), self.price_of(at)
        return (b * qa + a * qb) / (qa + qb)

    def top_of_book(self) -> dict[str, float | None]:
        return {"best_bid": self.best_bid(), "best_ask": self.best_ask(), "mid": self.midprice()}

    def depth(self, levels: int = 5) -> dict[str, list[tuple[float, int]]]:
        """Aggregated L2 depth, best price first."""
        out: dict[str, list[tuple[float, int]]] = {}
        for name, side in (("bids", self._bids), ("asks", self._asks)):
            rows: list[tuple[float, int]] = []
            for tick in side.ticks():
                if len(rows) >= levels:
                    break
                rows.append((self.price_of(tick), side.qty[tick]))
            out[name] = rows
        return out

    def volume_at(self, side: Side, price: float) -> int:
        tick = self._tick_or_none(price)
        if tick is None:
            return 0
        book = self._bids if side is Side.BUY else self._asks
        return book.qty.get(tick, 0)

    def get_order(self, order_id: str) -> Order | None:
        entry = self._orders.get(order_id)
        return entry[0] if entry is not None else None

    def queue_ahead(self, order_id: str) -> int | None:
        """Quantity resting ahead of ``order_id`` at its price level (O(queue))."""
        entry = self._orders.get(order_id)
        if entry is None:
            return None
        order, tick = entry
        book = self._bids if order.side is Side.BUY else self._asks
        ahead = 0
        for oid, o in book.levels[tick].items():
            if oid == order_id:
                return ahead
            ahead += o.qty
        raise AssertionError("order index out of sync with level")  # pragma: no cover

    def __contains__(self, order_id: object) -> bool:
        return order_id in self._orders

    def __len__(self) -> int:
        return len(self._orders)

    def __repr__(self) -> str:
        return (
            f"LimitOrderBook(tick_size={self.tick_size}, best_bid={self.best_bid()}, "
            f"best_ask={self.best_ask()}, orders={len(self)})"
        )

    # ------------------------------------------------------------------ #
    # Order entry                                                         #
    # ------------------------------------------------------------------ #

    def submit(self, order: Order) -> ExecutionReport:
        """Process an order according to its ``order_type`` / ``tif`` / ``post_only``."""
        return self._execute(order, market=order.order_type is OrderType.MARKET)

    def place_limit(self, order: Order) -> list[Fill]:
        """Match a limit order and rest any remainder (subject to ``tif``)."""
        report = self._execute(order, market=False)
        if report.status is OrderStatus.REJECTED:
            raise OrderRejected(report)
        return list(report.fills)

    def place_market(self, order: Order) -> list[Fill]:
        """Execute ``order`` as a market order (its price is ignored); never rests."""
        report = self._execute(order, market=True)
        if report.status is OrderStatus.REJECTED:
            raise OrderRejected(report)
        return list(report.fills)

    def cancel(self, order_id: str) -> bool:
        """Remove a resting order. Returns ``False`` if it is not on the book."""
        entry = self._orders.pop(order_id, None)
        if entry is None:
            return False
        order, tick = entry
        (self._bids if order.side is Side.BUY else self._asks).remove(tick, order)
        return True

    def modify(
        self,
        order_id: str,
        *,
        new_price: float | None = None,
        new_qty: int | None = None,
        ts: int | None = None,
    ) -> ExecutionReport:
        """Amend a resting order with exchange-style priority semantics.

        * same price and quantity reduced (or unchanged) -> amended in place, keeps priority
        * price change or quantity increase -> loses priority (cancel/replace);
          a new price that crosses the spread trades immediately and the fills
          are returned in the report
        * ``new_qty == 0`` -> cancel

        Invalid requests are REJECTED and leave the order untouched.
        """
        entry = self._orders.get(order_id)
        if entry is None:
            return self._reject(order_id, "unknown order id")
        order, tick = entry

        if new_qty is not None and new_qty < 0:
            return self._reject(order_id, "new_qty must be >= 0")
        new_tick = tick
        if new_price is not None:
            if not math.isfinite(new_price) or new_price <= 0:
                return self._reject(order_id, "price must be positive")
            parsed = self._tick_or_none(new_price)
            if parsed is None:
                return self._reject(order_id, "price not on tick grid")
            new_tick = parsed

        if new_qty == 0:
            remaining = order.qty
            self.cancel(order_id)
            return ExecutionReport(order_id, OrderStatus.CANCELED, canceled_qty=remaining)

        target = order.qty if new_qty is None else new_qty
        book = self._bids if order.side is Side.BUY else self._asks
        if new_tick == tick and target <= order.qty:
            book.qty[tick] -= order.qty - target
            order.qty = target
            return ExecutionReport(order_id, OrderStatus.NEW, resting_qty=target)

        if order.post_only:
            opp = self._asks if order.side is Side.BUY else self._bids
            if opp.keys and opp.keys[-1] >= opp.sign * new_tick:
                return self._reject(order_id, "post-only order would cross")

        self.cancel(order_id)
        order.price = self.price_of(new_tick)
        order.qty = target
        if ts is not None:
            order.ts = ts
        return self._execute(order, market=False)

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _reject(order_id: str, reason: str) -> ExecutionReport:
        return ExecutionReport(order_id, OrderStatus.REJECTED, reason=reason)

    def _execute(self, order: Order, *, market: bool) -> ExecutionReport:
        # Hot path. Validation, the crossing test and resting are inlined, and
        # reports are built with ``_report`` (tuple.__new__) to skip the
        # Python-level NamedTuple constructor.
        oid = order.order_id
        orders = self._orders
        if oid in orders:
            return self._reject(oid, "duplicate order id")
        if order.qty <= 0:
            return self._reject(oid, "qty must be positive")

        is_buy = order.side is Side.BUY
        opp = self._asks if is_buy else self._bids
        opp_keys = opp.keys
        tick = 0
        stop_key: float = _NEG_INF
        if not market:
            price = order.price
            if price is None:
                return self._reject(oid, "limit order requires a price")
            x = price / self.tick_size
            tick = round(x)
            if abs(x - tick) > 1e-6:
                return self._reject(oid, "price not on tick grid")
            if tick <= 0:
                return self._reject(oid, "price must be positive")
            # A level crosses iff its sort key >= stop_key (see _BookSide).
            stop_key = opp.sign * tick
            if order.post_only and opp_keys and opp_keys[-1] >= stop_key:
                return self._reject(oid, "post-only order would cross")

        tif = order.tif
        if tif is TimeInForce.FOK and not self._can_fill(order, opp, stop_key):
            killed, order.qty = order.qty, 0
            return _report(_ER, (oid, _CANCELED, (), 0, killed, (), None))

        if opp_keys and opp_keys[-1] >= stop_key:
            fill_list, stp_list, halted = self._match(order, opp, stop_key)
            fills: tuple[Fill, ...] = tuple(fill_list)
            stp_ids: tuple[str, ...] = tuple(stp_list) if stp_list else ()
        else:  # passive fast path: nothing to match against
            fills, stp_ids, halted = (), (), False

        remaining = order.qty
        if remaining == 0:
            return _report(_ER, (oid, _FILLED, fills, 0, 0, stp_ids, None))
        if market or halted or tif is not TimeInForce.GTC:
            order.qty = 0
            return _report(_ER, (oid, _CANCELED, fills, 0, remaining, stp_ids, None))

        order.price = self.price_of(tick)
        book = self._bids if is_buy else self._asks
        level = book.levels.get(tick)
        if level is None:
            book.add(tick, order)
        else:
            level[oid] = order
            book.qty[tick] += remaining
        orders[oid] = (order, tick)
        return _report(_ER, (oid, _PARTIAL if fills else _NEW, fills, remaining, 0, stp_ids, None))

    def _match(self, taker: Order, opp: _BookSide, stop_key: float) -> tuple[list[Fill], list[str], bool]:
        """Sweep ``opp`` from the touch while levels cross. Returns
        ``(fills, stp_canceled_ids, halted_by_stp)``. Hot path: keep it lean."""
        fills: list[Fill] = []
        stp_ids: list[str] = []
        keys, levels, qtys, sign = opp.keys, opp.levels, opp.qty, opp.sign
        orders = self._orders
        owner = taker.owner
        stp = self.stp
        check_stp = owner is not None and stp is not STPMode.NONE
        tid, tside, tts = taker.order_id, taker.side, taker.ts

        while taker.qty and keys and keys[-1] >= stop_key:
            tick = sign * keys[-1]
            level = levels[tick]
            price = self.price_of(tick)
            while taker.qty and level:
                maker = next(iter(level.values()))
                if check_stp and maker.owner == owner:
                    if stp is STPMode.CANCEL_INCOMING:
                        return fills, stp_ids, True
                    level.popitem(last=False)
                    qtys[tick] -= maker.qty
                    del orders[maker.order_id]
                    stp_ids.append(maker.order_id)
                    continue
                q = taker.qty if taker.qty < maker.qty else maker.qty
                fills.append(Fill(tid, maker.order_id, price, q, tside, tts))
                taker.qty -= q
                maker.qty -= q
                qtys[tick] -= q
                if not maker.qty:
                    level.popitem(last=False)
                    del orders[maker.order_id]
            if not level:
                del levels[tick]
                del qtys[tick]
                keys.pop()
        return fills, stp_ids, False

    def _can_fill(self, taker: Order, opp: _BookSide, stop_key: float) -> bool:
        """Would ``taker`` be completely filled right now? (FOK pre-check, no side effects.)"""
        need = taker.qty
        owner = taker.owner
        check_stp = owner is not None and self.stp is not STPMode.NONE
        for key in reversed(opp.keys):
            if key < stop_key:
                break
            tick = opp.sign * key
            if not check_stp:
                need -= opp.qty[tick]
                if need <= 0:
                    return True
                continue
            for maker in opp.levels[tick].values():
                if maker.owner == owner:
                    if self.stp is STPMode.CANCEL_INCOMING:
                        return False
                    continue
                need -= maker.qty
                if need <= 0:
                    return True
        return False

    # ------------------------------------------------------------------ #
    # Debugging                                                           #
    # ------------------------------------------------------------------ #

    def check_invariants(self) -> None:
        """Verify internal consistency; raises ``AssertionError`` on corruption.

        O(n) — intended for tests and debugging, not the hot path.
        """

        def require(cond: bool, msg: str) -> None:
            if not cond:
                raise AssertionError(msg)

        seen: set[str] = set()
        for book in (self._bids, self._asks):
            require(book.keys == sorted(set(book.keys)), f"{book.side} keys unsorted or duplicated")
            ticks = {book.sign * k for k in book.keys}
            require(ticks == set(book.levels) == set(book.qty), f"{book.side} level index mismatch")
            for tick, level in book.levels.items():
                require(len(level) > 0, f"{book.side} empty level at tick {tick}")
                require(
                    book.qty[tick] == sum(o.qty for o in level.values()),
                    f"{book.side} cached qty mismatch at tick {tick}",
                )
                for oid, o in level.items():
                    require(o.order_id == oid, f"level key {oid!r} != order id {o.order_id!r}")
                    require(o.qty > 0, f"non-positive qty resting: {oid}")
                    require(o.side is book.side, f"order {oid} on wrong side")
                    entry = self._orders.get(oid)
                    require(entry is not None and entry[0] is o and entry[1] == tick, f"index mismatch for {oid}")
                    seen.add(oid)
        require(seen == set(self._orders), "order index contains orders not on the book")
        bb, ba = self._bids.best_tick(), self._asks.best_tick()
        require(bb is None or ba is None or bb < ba, f"book is crossed: bid {bb} >= ask {ba}")
