from __future__ import annotations
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
    Price-time priority CLOB.

    Implementation notes:
    - bids: prices descending
    - asks: prices ascending
    - For correctness + simplicity, we maintain:
        * price -> deque(Order)
        * sorted list of active prices
        * order_id -> (side, price) for cancel/lookup
    Later we can optimize with heaps / sortedcontainers.
    """

    def __init__(self) -> None:
        self._bids: Dict[float, Deque[Order]] = {}
        self._asks: Dict[float, Deque[Order]] = {}
        self._bid_prices: List[float] = []  # kept sorted desc
        self._ask_prices: List[float] = []  # kept sorted asc

        self._id_map: Dict[str, Tuple[Side, float]] = {}  # order_id -> (side, price)

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

    def place_limit(self, order: Order) -> List[Fill]:
        """
        Place a limit order:
        - Match immediately if crossing.
        - Rest remainder if any.
        Returns list of fills in execution order.
        """
        fills: List[Fill] = []
        if order.side == Side.BUY:
            fills = self._match_buy(order)
        else:
            fills = self._match_sell(order)

        if order.qty > 0:
            self._rest(order)
        return fills

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

        q = book.get(price)
        if q is None:
            # inconsistent but handle gracefully
            del self._id_map[order_id]
            return False

        # Remove from deque (O(n) at that level). OK for v1.
        new_q: Deque[Order] = deque()
        removed = False
        while q:
            o = q.popleft()
            if o.order_id == order_id and not removed:
                removed = True
                continue
            new_q.append(o)

        if removed:
            if new_q:
                book[price] = new_q
            else:
                del book[price]
                self._remove_price(prices, price)

            del self._id_map[order_id]
            return True

        return False

    # --------- Internal helpers ---------

    def _rest(self, order: Order) -> None:
        book = self._bids if order.side == Side.BUY else self._asks
        prices = self._bid_prices if order.side == Side.BUY else self._ask_prices

        if order.price not in book:
            book[order.price] = deque()
            self._insert_price(prices, order.price, descending=(order.side == Side.BUY))

        book[order.price].append(order)
        self._id_map[order.order_id] = (order.side, order.price)

    def _match_buy(self, buy: Order) -> List[Fill]:
        fills: List[Fill] = []
        # Buy crosses if best ask <= buy.price
        while buy.qty > 0 and self._ask_prices:
            best_ask = self._ask_prices[0]
            if best_ask > buy.price:
                break

            ask_q = self._asks[best_ask]
            maker = ask_q[0]  # FIFO
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

            if maker.qty == 0:
                ask_q.popleft()
                del self._id_map[maker.order_id]
                if not ask_q:
                    del self._asks[best_ask]
                    self._ask_prices.pop(0)

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

            if maker.qty == 0:
                bid_q.popleft()
                del self._id_map[maker.order_id]
                if not bid_q:
                    del self._bids[best_bid]
                    self._bid_prices.pop(0)

        return fills

    @staticmethod
    def _insert_price(prices: List[float], price: float, *, descending: bool) -> None:
        # Insert into sorted list (binary insert would be better; ok for v1)
        prices.append(price)
        prices.sort(reverse=descending)

    @staticmethod
    def _remove_price(prices: List[float], price: float) -> None:
        # remove first occurrence
        for i, p in enumerate(prices):
            if p == price:
                prices.pop(i)
                return
