from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from microbook import LimitOrderBook, Order
from microbook.types import Fill, Side

from .events import Event, EventType
from .strategy import Strategy
from .analytics import TimeSeries
from .execution_metrics import ExecutionMetrics


@dataclass
class SimResult:
    fills: List[Fill] = field(default_factory=list)
    snapshots: List[Dict] = field(default_factory=list)
    # these will be initialized lazily on first SNAPSHOT:
    # pnl_series: Dict[str, List[float]]
    # pnl_t: List[int]


class MarketSimulator:
    """
    Discrete-event simulator for the limit order book.

    - Uses a min-heap of Events ordered by (time, seq)
    - Deterministic given the same event stream
    - Can host multiple strategies with portfolio + execution metrics tracking
    """

    def __init__(self, lob: Optional[LimitOrderBook] = None, strategies: Optional[List[Strategy]] = None) -> None:
        self.lob = lob or LimitOrderBook()
        self.strategies = strategies or []

        self._pq: List[Event] = []
        self._seq = 0
        self.now = 0

        # order_id -> (strategy_index, Side)
        self._order_owner: Dict[str, Tuple[int, Side]] = {}

        # book analytics time series (recorded on SNAPSHOT)
        self.ts = TimeSeries()

        # execution metrics per strategy (name-keyed)
        self.exec_metrics: Dict[str, ExecutionMetrics] = {
            s.name: ExecutionMetrics() for s in self.strategies
        }

    def _register_order(self, order: Order) -> None:
        """
        If the order_id belongs to a strategy, register ownership for fill attribution.
        """
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

                # record total market volume into each strategy's metrics (simple baseline)
                for m in self.exec_metrics.values():
                    m.record_market_volume(fills)

                # Attribute fills to strategies (portfolio + execution metrics)
                for f in fills:
                    # maker side
                    if f.maker_order_id in self._order_owner:
                        idx, side = self._order_owner[f.maker_order_id]
                        strat = self.strategies[idx]
                        strat.portfolio.on_fill(f, f.maker_order_id, side)
                        self.exec_metrics[strat.name].on_fill(f, side)

                    # taker side
                    if f.taker_order_id in self._order_owner:
                        idx, side = self._order_owner[f.taker_order_id]
                        strat = self.strategies[idx]
                        strat.portfolio.on_fill(f, f.taker_order_id, side)
                        self.exec_metrics[strat.name].on_fill(f, side)

            elif ev.etype == EventType.CANCEL:
                assert ev.order_id is not None
                self.lob.cancel(ev.order_id)
                self._order_owner.pop(ev.order_id, None)

            elif ev.etype == EventType.MODIFY:
                assert ev.order_id is not None and ev.modify is not None
                self.lob.modify(ev.order_id, ts=self.now, **ev.modify)

            elif ev.etype == EventType.SNAPSHOT:
                # analytics time series
                self.ts.record(self.now, self.lob)

                snap = {
                    "t": self.now,
                    "top": self.lob.top_of_book(),
                    "depth": self.lob.depth(levels=5),
                }
                out.snapshots.append(snap)

                # Let strategies act on each snapshot tick
                for strat in self.strategies:
                    actions = strat.on_tick(self.now, self.lob)
                    for a in actions:
                        self.schedule(a.time, a.etype, **a.kwargs)

                # Record MTM PnL over time (once per snapshot)
                if not hasattr(out, "pnl_series"):
                    out.pnl_series = {s.name: [] for s in self.strategies}
                    out.pnl_t = []

                out.pnl_t.append(self.now)
                for s in self.strategies:
                    mtm = s.portfolio.mark_to_market(self.lob)
                    out.pnl_series[s.name].append(float(mtm) if mtm is not None else float("nan"))

        return out
