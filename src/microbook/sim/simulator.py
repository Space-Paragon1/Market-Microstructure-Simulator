"""Discrete-event market simulator.

Events live in a min-heap keyed by ``(time, seq)`` where ``seq`` is a global
scheduling counter, so simultaneous events are processed FIFO and every run is
exactly reproducible. Strategies interact only through :class:`Action` objects;
order messages they send arrive at the book after the strategy's ``latency``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

from microbook import ExecutionReport, LimitOrderBook, Order, OrderStatus, Side, STPMode
from microbook.types import Fill

from .analytics import TimeSeries, mid_at
from .events import Event, EventType
from .execution_metrics import ExecutionMetrics
from .strategy import Action, Strategy


@dataclass
class SimResult:
    """Everything recorded during a run. Accumulates across successive ``run`` calls."""

    fills: list[Fill] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    pnl_t: list[int] = field(default_factory=list)
    pnl_series: dict[str, list[float]] = field(default_factory=dict)
    inventory_series: dict[str, list[int]] = field(default_factory=dict)
    # (time, mid) appended whenever the midprice changes; NaN while one-sided
    mid_path: list[tuple[int, float]] = field(default_factory=list)

    def mid_at(self, t: int) -> float | None:
        """Midprice in force at the end of time ``t`` (``None`` if undefined)."""
        return mid_at(self.mid_path, t)


class MarketSimulator:
    """Discrete-event simulator hosting a book and any number of strategies.

    Parameters
    ----------
    lob:
        The order book. Defaults to a 0.01-tick book with self-trade
        prevention (``CANCEL_RESTING``) so no participant can wash-trade.
    strategies:
        Participants. Names must be unique (they key metrics and series).
    record_snapshots:
        Keep an L2 snapshot dict for every SNAPSHOT event (disable for long runs).
    """

    def __init__(
        self,
        lob: LimitOrderBook | None = None,
        strategies: list[Strategy] | None = None,
        *,
        record_snapshots: bool = True,
    ) -> None:
        self.lob = lob if lob is not None else LimitOrderBook(stp=STPMode.CANCEL_RESTING)
        self.strategies: list[Strategy] = list(strategies or [])
        names = [s.name for s in self.strategies]
        if len(set(names)) != len(names):
            raise ValueError(f"strategy names must be unique, got {names}")
        self.record_snapshots = record_snapshots

        self._pq: list[tuple[int, int, Event]] = []
        self._seq = 0
        self.now = 0
        self.events_processed = 0
        self._started = False
        self._last_mid: float | None = None

        # live order_id -> (strategy index, side), for fill attribution
        self._order_owner: dict[str, tuple[int, Side]] = {}

        self.ts = TimeSeries()
        self.exec_metrics: dict[str, ExecutionMetrics] = {s.name: ExecutionMetrics() for s in self.strategies}
        self.result = SimResult(
            pnl_series={s.name: [] for s in self.strategies},
            inventory_series={s.name: [] for s in self.strategies},
        )

    # ------------------------------------------------------------------ #
    # Scheduling                                                          #
    # ------------------------------------------------------------------ #

    def schedule(self, time: int, etype: EventType, **kwargs: Any) -> None:
        """Schedule an event. ``kwargs`` are :class:`Event` fields (``order``,
        ``order_id``, ``modify``)."""
        if time < self.now:
            raise ValueError(f"cannot schedule at t={time} < now={self.now}")
        self._seq += 1
        heapq.heappush(self._pq, (time, self._seq, Event(time, self._seq, etype, **kwargs)))

    def schedule_snapshots(self, start: int, end: int, every: int = 1) -> None:
        """SNAPSHOT events at ``start, start+every, ..., <= end`` (these drive ``on_tick``)."""
        if every <= 0:
            raise ValueError("every must be positive")
        for t in range(start, end + 1, every):
            self.schedule(t, EventType.SNAPSHOT)

    def strategy(self, name: str) -> Strategy:
        for s in self.strategies:
            if s.name == name:
                return s
        raise KeyError(name)

    # ------------------------------------------------------------------ #
    # Main loop                                                           #
    # ------------------------------------------------------------------ #

    def run(self, until: int) -> SimResult:
        """Process every event with ``time <= until``; returns the (cumulative) result."""
        if not self._started:
            self._started = True
            for i, s in enumerate(self.strategies):
                self._dispatch(i, s.on_start(self.now, self.lob))

        pq = self._pq
        while pq and pq[0][0] <= until:
            t, _, ev = heapq.heappop(pq)
            self.now = t
            self.events_processed += 1
            et = ev.etype
            if et is EventType.SUBMIT:
                self._on_submit(ev)
            elif et is EventType.CANCEL:
                if ev.order_id is None:
                    raise ValueError("CANCEL event requires order_id")
                self.lob.cancel(ev.order_id)
                self._order_owner.pop(ev.order_id, None)
                self._record_mid()
            elif et is EventType.MODIFY:
                self._on_modify(ev)
            elif et is EventType.SNAPSHOT:
                self._on_snapshot()
            elif et is EventType.TIMER:
                if ev.owner is None:
                    raise ValueError("TIMER event requires an owner")
                self._dispatch(ev.owner, self.strategies[ev.owner].on_timer(t, self.lob))
            else:  # pragma: no cover
                raise ValueError(f"unknown event type {et}")
        self.now = max(self.now, until)
        return self.result

    # ------------------------------------------------------------------ #
    # Event handlers                                                      #
    # ------------------------------------------------------------------ #

    def _on_submit(self, ev: Event) -> None:
        order = ev.order
        if order is None:
            raise ValueError("SUBMIT event requires an order")
        idx = ev.owner if ev.owner is not None else self._lookup_owner(order.order_id)
        if idx is not None and order.owner is None:
            order.owner = self.strategies[idx].name
        order.ts = self.now

        mid_before = self.lob.midprice()
        report = self.lob.submit(order)
        if idx is not None and report.accepted:
            self._order_owner[order.order_id] = (idx, order.side)
        self._process_report(report, mid_before, idx)

    def _on_modify(self, ev: Event) -> None:
        if ev.order_id is None or ev.modify is None:
            raise ValueError("MODIFY event requires order_id and modify")
        owner = self._order_owner.get(ev.order_id)
        mid_before = self.lob.midprice()
        report = self.lob.modify(ev.order_id, ts=self.now, **ev.modify)
        idx = ev.owner if ev.owner is not None else (owner[0] if owner else None)
        self._process_report(report, mid_before, idx)

    def _on_snapshot(self) -> None:
        now, lob, res = self.now, self.lob, self.result
        self.ts.record(now, lob)
        if self.record_snapshots:
            res.snapshots.append({"t": now, "top": lob.top_of_book(), "depth": lob.depth(levels=5)})

        for i, strat in enumerate(self.strategies):
            self._dispatch(i, strat.on_tick(now, lob))

        mid = lob.midprice()
        res.pnl_t.append(now)
        for s in self.strategies:
            res.pnl_series[s.name].append(s.portfolio.total_pnl(mid) if mid is not None else math.nan)
            res.inventory_series[s.name].append(s.portfolio.position)

    def _process_report(self, report: ExecutionReport, mid_before: float | None, submitter: int | None) -> None:
        now, lob = self.now, self.lob
        owners = self._order_owner
        fills = report.fills
        if fills:
            self.result.fills.extend(fills)
            for m in self.exec_metrics.values():
                m.record_market_volume(fills)
            for f in fills:
                for oid, is_maker in ((f.maker_order_id, True), (f.taker_order_id, False)):
                    entry = owners.get(oid)
                    if entry is None:
                        continue
                    idx, side = entry
                    strat = self.strategies[idx]
                    strat.portfolio.on_fill(f, oid, side, is_maker=is_maker)
                    self.exec_metrics[strat.name].on_fill(f, side, is_maker, mid=mid_before, time=now)
                    self._dispatch(idx, strat.on_fill(now, f, side, is_maker, lob))
                    if is_maker and oid not in lob:
                        del owners[oid]

        for oid in report.stp_canceled:
            owners.pop(oid, None)

        if submitter is not None:
            if report.status is OrderStatus.REJECTED:
                self.exec_metrics[self.strategies[submitter].name].orders_rejected += 1
            self._dispatch(submitter, self.strategies[submitter].on_report(now, report))
        if report.status is not OrderStatus.REJECTED and report.order_id not in lob:
            owners.pop(report.order_id, None)
        self._record_mid()

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _dispatch(self, idx: int, actions: list[Action]) -> None:
        if not actions:
            return
        strat = self.strategies[idx]
        now = self.now
        arrival_mid: float | None = None
        have_mid = False
        for a in actions:
            et = a.etype
            if et is EventType.TIMER:
                if a.time <= now:
                    raise ValueError(f"{strat.name}: timers must be scheduled in the future (t={a.time}, now={now})")
                t = a.time
            else:
                t = max(a.time, now) + strat.latency
            if et is EventType.SUBMIT:
                order: Order = a.kwargs["order"]
                if order.owner is None:
                    order.owner = strat.name
                if not have_mid:
                    arrival_mid, have_mid = self.lob.midprice(), True
                self.exec_metrics[strat.name].record_order(order.order_id, order.side, order.qty, arrival_mid)
            self._seq += 1
            heapq.heappush(self._pq, (t, self._seq, Event(t, self._seq, et, owner=idx, **a.kwargs)))

    def _lookup_owner(self, order_id: str) -> int | None:
        """Attribution for orders scheduled directly via ``schedule``."""
        for i, s in enumerate(self.strategies):
            if order_id in s.my_order_ids:
                return i
        return None

    def _record_mid(self) -> None:
        m = self.lob.midprice()
        if m != self._last_mid:
            self._last_mid = m
            self.result.mid_path.append((self.now, m if m is not None else math.nan))
