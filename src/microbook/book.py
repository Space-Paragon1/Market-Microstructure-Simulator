from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from .order import Order
from .types import Fill, Side


@dataclass
class Level:
    price: float
    queue: Deque[Order]  # FIFO at a price level


class LimitOrderBook:
    """
    Price-time priority central limit order book (CLOB).

    - Bids sorted descending, asks sorted ascending
    - FIFO within each price level
    - Supports: limit, market, cancel, modify (with priority semantics)
    - Maintains O(1) per-level aggregated qty for fast depth()
    """

    def __init__(self) -> None:
        self._bids: Dict[float, Deque[Order]] = {}
        self._asks: Dict[float, Deque[Order]] = {}

        self._bid_prices: List[float] = []  # sorted desc
        self._ask_prices: List[float] = []  # sorted asc

        # price -> total qty at price (cached)
        self._bid_qty: Dict[float, int] = {}
        self._ask_qty: Dict[float, int] = {}

        # order_id -> (side, price)
        self._id_map: Dict[str, Tuple[Side, float]] = {}

    # --------- Public API ---------

    def best_bid(self) -> Optional[float]:
        return self._bid_prices[0] if self._bid_prices else None

    def best_ask(self) -> Optional[float]:
        return self._ask_prices[0] if self._ask_prices else None

    def midprice(self) -> Optional[float]:
        b = self.best_bid()
        a = self.best_ask()
        if b is None or a is None:
            return None
        return (b + a) / 2.0

    def top_of_book(self) -> dict:
        return {"best_bid": self.best_bid(), "best_ask": self.best_ask(), "mid": self.midprice()}

    def depth(self, levels: int = 5) -> dict:
        bids = [(p, self._bid_qty.get(p, 0)) for p in self._bid_prices[:levels]]
        asks = [(p, self._ask_qty.get(p, 0)) for p in self._ask_prices[:levels]]
        return {"bids": bids, "asks": asks}

    def place_limit(self, order: Order) -> List[Fill]:
        """
        Place a LIMIT order:
        - Match immediately if crossing.
        - Rest remainder (if any) on the book.
        Returns fills in execution order.
        """
        fills: List[Fill]
        if order.side == Side.BUY:
            fills = self._match_buy(order)
        else:
            fills = self._match_sell(order)

        if order.qty > 0:
            self._rest(order)

        return fills

    def place_market(self, order: Order) -> List[Fill]:
        """
        Place a MARKET order:
        - Consumes liquidity immediately
        - Never rests on the book
        """
        if order.side == Side.BUY:
            order.price = float("inf")  # crosses any ask
            return self._match_buy(order)
        else:
            order.price = 0.0  # crosses any bid
            return self._match_sell(order)

    def cancel(self, order_id: str) -> bool:
        """
        Cancel an order by id. Returns True if found & removed.
        """
        info = self._id_map.get(order_id)
        if info is None:
            return False

        side, price = info
        book = self._bids if side == Side.BUY else self._asks
        prices = self._bid_prices if side == Side.BUY else self._ask_prices
        agg = self._bid_qty if side == Side.BUY else self._ask_qty

        q = book.get(price)
        if q is None:
            # inconsistent but handle gracefully
            del self._id_map[order_id]
            return False

        new_q: Deque[Order] = deque()
        removed = False
        removed_qty = 0

        while q:
            o = q.popleft()
            if o.order_id == order_id and not removed:
                removed = True
                removed_qty = o.qty
                continue
            new_q.append(o)

        if not removed:
            # put the queue back to avoid losing orders
            book[price] = new_q
            return False

        # update aggregates
        agg[price] = agg.get(price, 0) - removed_qty

        if new_q:
            book[price] = new_q
            # keep agg entry (could be 0 only if bug; clamp not necessary but safe)
            if agg.get(price, 0) <= 0:
                agg[price] = sum(o.qty for o in new_q)
        else:
            # remove empty level
            book.pop(price, None)
            agg.pop(price, None)
            self._remove_price(prices, price)

        del self._id_map[order_id]
        return True

    def modify(
        self,
        order_id: str,
        *,
        new_price: float | None = None,
        new_qty: int | None = None,
        ts: int,
    ) -> bool:
        """
        Modify semantics (exchange-style):
        - Reduce qty only (no price change) => keeps priority
        - Increase qty or change price => loses priority (cancel + reinsert with new ts)
        """
        info = self._id_map.get(order_id)
        if info is None:
            return False

        side, price = info
        book = self._bids if side == Side.BUY else self._asks
        agg = self._bid_qty if side == Side.BUY else self._ask_qty

        q = book.get(price)
        if q is None:
            return False

        target: Optional[Order] = None
        for o in q:
            if o.order_id == order_id:
                target = o
                break
        if target is None:
            return False

        # qty reduction only (keeps priority)
        if new_price is None and new_qty is not None and 0 < new_qty < target.qty:
            delta = target.qty - new_qty
            target.qty = new_qty
            agg[price] = agg.get(price, 0) - delta
            return True

        # otherwise lose priority => cancel + reinsert
        old_price = target.price
        old_qty = target.qty

        if not self.cancel(order_id):
            return False

        price2 = new_price if new_price is not None else old_price
        qty2 = new_qty if new_qty is not None else old_qty
        if qty2 <= 0 or price2 <= 0:
            return False

        self.place_limit(Order(order_id=order_id, side=side, price=float(price2), qty=int(qty2), ts=ts))
        return True

    # --------- Internal helpers ---------

    def _rest(self, order: Order) -> None:
        book = self._bids if order.side == Side.BUY else self._asks
        prices = self._bid_prices if order.side == Side.BUY else self._ask_prices
        agg = self._bid_qty if order.side == Side.BUY else self._ask_qty

        if order.price not in book:
            book[order.price] = deque()
            self._insert_price(prices, order.price, descending=(order.side == Side.BUY))
            agg[order.price] = 0

        book[order.price].append(order)
        agg[order.price] = agg.get(order.price, 0) + order.qty
        self._id_map[order.order_id] = (order.side, order.price)

    def _match_buy(self, buy: Order) -> List[Fill]:
        fills: List[Fill] = []

        # Buy crosses if best ask <= buy.price
        while buy.qty > 0 and self._ask_prices:
            best_ask = self._ask_prices[0]
            if best_ask > buy.price:
                break

            ask_q = self._asks[best_ask]
            maker = ask_q[0]  # FIFO at this price
            traded = min(buy.qty, maker.qty)

            fills.append(
                Fill(
                    taker_order_id=buy.order_id,
                    maker_order_id=maker.order_id,
                    price=best_ask,
                    qty=traded,
                )
            )

            buy.qty -= traded
            maker.qty -= traded

            # update aggregated ask qty
            self._ask_qty[best_ask] = self._ask_qty.get(best_ask, 0) - traded

            if maker.qty == 0:
                ask_q.popleft()
                self._id_map.pop(maker.order_id, None)

                if not ask_q:
                    self._asks.pop(best_ask, None)
                    self._ask_prices.pop(0)
                    self._ask_qty.pop(best_ask, None)

        return fills

    def _match_sell(self, sell: Order) -> List[Fill]:
        fills: List[Fill] = []

        # Sell crosses if best bid >= sell.price
        while sell.qty > 0 and self._bid_prices:
            best_bid = self._bid_prices[0]
            if best_bid < sell.price:
                break

            bid_q = self._bids[best_bid]
            maker = bid_q[0]
            traded = min(sell.qty, maker.qty)

            fills.append(
                Fill(
                    taker_order_id=sell.order_id,
                    maker_order_id=maker.order_id,
                    price=best_bid,
                    qty=traded,
                )
            )

            sell.qty -= traded
            maker.qty -= traded

            # update aggregated bid qty
            self._bid_qty[best_bid] = self._bid_qty.get(best_bid, 0) - traded

            if maker.qty == 0:
                bid_q.popleft()
                self._id_map.pop(maker.order_id, None)

                if not bid_q:
                    self._bids.pop(best_bid, None)
                    self._bid_prices.pop(0)
                    self._bid_qty.pop(best_bid, None)

        return fills

    @staticmethod
    def _insert_price(prices: List[float], price: float, *, descending: bool) -> None:
        """
        Insert a new price into an already-sorted list without resorting the entire list.
        - asks: ascending
        - bids: descending
        """
        if descending:
            # maintain descending order by bisecting on negative values
            negs = [-p for p in prices]
            idx = bisect.bisect_left(negs, -price)
            prices.insert(idx, price)
        else:
            idx = bisect.bisect_left(prices, price)
            prices.insert(idx, price)

    @staticmethod
    def _remove_price(prices: List[float], price: float) -> None:
        for i, p in enumerate(prices):
            if p == price:
                prices.pop(i)
                return
