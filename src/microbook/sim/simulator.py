from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from microbook import LimitOrderBook, Order
from microbook.types import Fill
from .events import Event, EventType


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

    def __init__(self, lob: Optional[LimitOrderBook] = None) -> None:
        self.lob = lob or LimitOrderBook()
        self._pq: List[Event] = []
        self._seq = 0
        self.now = 0

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
                # route based on whether user intends market or limit
                # convention: order.price == float("inf") or 0 handled by place_market, but better is explicit:
                if getattr(ev.order, "_is_market", False):
                    fills = self.lob.place_market(ev.order)
                else:
                    fills = self.lob.place_limit(ev.order)
                out.fills.extend(fills)

            elif ev.etype == EventType.CANCEL:
                assert ev.order_id is not None
                self.lob.cancel(ev.order_id)

            elif ev.etype == EventType.MODIFY:
                assert ev.order_id is not None and ev.modify is not None
                self.lob.modify(ev.order_id, ts=self.now, **ev.modify)

            elif ev.etype == EventType.SNAPSHOT:
                snap = {
                    "t": self.now,
                    "top": self.lob.top_of_book(),
                    "depth": self.lob.depth(levels=5),
                }
                out.snapshots.append(snap)

        return out
