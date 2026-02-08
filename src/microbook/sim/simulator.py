from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from microbook import LimitOrderBook, Order
from microbook.types import Fill, Side
from .events import Event, EventType
from .strategy import Strategy
from .analytics import TimeSeries


@dataclass
class SimResult:
    fills: List[Fill] = field(default_factory=list)
    snapshots: List[Dict] = field(default_factory=list)


class MarketSimulator:
    """
    Discrete-event simulator for the limit order book.

    - Uses a min-heap of Events ordered by (time, seq)
    - Deterministic given the same event stream
    """

    def __init__(self, lob: Optional[LimitOrderBook] = None, strategies: Optional[List[Strategy]] = None) -> None:
        self.lob = lob or LimitOrderBook()
        self.strategies = strategies or []

        self._pq: List[Event] = []
        self._seq = 0
        self.now = 0
        self.ts = TimeSeries()


        # order_id -> (strategy_index, Side)
        self._order_owner: Dict[str, Tuple[int, Side]] = {}

    def _register_order(self, order: Order) -> None:
        for i, s in enumerate(self.strategies):
            if order.order_id in s.my_order_ids:
                self._order_owner[order.order_id] = (i, order.side)
                return

    def schedule(self, time: int, etype: EventType, **kwargs) -> None:
        self._seq += 1
        ev = Event(time=time, seq=self._seq, etype=etype, **kwargs)
        heapq.heappush(self._pq, ev)

    def run(self, until: int) -> SimResult:
        out = SimResult()

        while self._pq and self._pq[0].time <= until:
            ev = heapq.heappop(self._pq)
            self.now = ev.time

            if ev.etype == EventType.SUBMIT:
                assert ev.order is not None

                # register ownership BEFORE placing (important for immediate fills)
                self._register_order(ev.order)

                if getattr(ev.order, "_is_market", False):
                    fills = self.lob.place_market(ev.order)
                else:
                    fills = self.lob.place_limit(ev.order)

                out.fills.extend(fills)

                # Attribute fills to strategies
                for f in fills:
                    # maker side
                    if f.maker_order_id in self._order_owner:
                        idx, side = self._order_owner[f.maker_order_id]
                        self.strategies[idx].portfolio.on_fill(f, f.maker_order_id, side)

                    # taker side
                    if f.taker_order_id in self._order_owner:
                        idx, side = self._order_owner[f.taker_order_id]
                        self.strategies[idx].portfolio.on_fill(f, f.taker_order_id, side)

            elif ev.etype == EventType.CANCEL:
                assert ev.order_id is not None
                self.lob.cancel(ev.order_id)
                self._order_owner.pop(ev.order_id, None)

            elif ev.etype == EventType.MODIFY:
                assert ev.order_id is not None and ev.modify is not None
                self.lob.modify(ev.order_id, ts=self.now, **ev.modify)

            elif ev.etype == EventType.SNAPSHOT:
                snap = {
                    "t": self.now,
                    "top": self.lob.top_of_book(),
                    "depth": self.lob.depth(levels=5),
                }
                self.ts.record(self.now, self.lob)
                out.snapshots.append(snap)

                # Let strategies act on each snapshot tick
                for strat in self.strategies:
                    actions = strat.on_tick(self.now, self.lob)
                    for a in actions:
                        self.schedule(a.time, a.etype, **a.kwargs)

        return out
