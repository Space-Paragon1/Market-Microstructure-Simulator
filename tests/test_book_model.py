"""Differential (model-based) testing of the matching engine.

Hypothesis drives random sequences of submits / cancels / modifies through both
the optimized ``LimitOrderBook`` and ``RefBook``, a deliberately naive
implementation that re-sorts a flat list on every message. After every step
the two must agree exactly on fills, statuses and depth, and the engine's
internal invariants must hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import hypothesis.strategies as st
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from microbook import LimitOrderBook, Order, OrderStatus, Side, STPMode, TimeInForce

TICK = 0.01


@dataclass
class _Rest:
    oid: str
    side: Side
    tick: int
    qty: int
    owner: str | None
    post_only: bool
    seq: int


class RefBook:
    """Naive reference: a flat list of resting orders, priority by (price, seq)."""

    def __init__(self, stp: STPMode) -> None:
        self.stp = stp
        self.rest: list[_Rest] = []
        self.seq = 0

    def _queue(self, side: Side) -> list[_Rest]:
        """Resting orders on ``side`` in priority order (best price, then FIFO)."""
        orders = [o for o in self.rest if o.side is side]
        return sorted(orders, key=lambda o: (-o.tick if side is Side.BUY else o.tick, o.seq))

    def _find(self, oid: str) -> _Rest | None:
        return next((o for o in self.rest if o.oid == oid), None)

    def submit(
        self, oid: str, side: Side, tick: int | None, qty: int, tif: TimeInForce, post_only: bool, owner: str | None
    ) -> tuple[OrderStatus, list[tuple[str, int, int]], list[str]]:
        if self._find(oid) is not None:
            return OrderStatus.REJECTED, [], []
        opp = self._queue(side.opposite)

        def crosses(o: _Rest) -> bool:
            if tick is None:
                return True
            return o.tick <= tick if side is Side.BUY else o.tick >= tick

        if post_only and any(crosses(o) for o in opp):
            return OrderStatus.REJECTED, [], []

        def is_self(o: _Rest) -> bool:
            return self.stp is not STPMode.NONE and owner is not None and o.owner == owner

        if tif is TimeInForce.FOK:
            avail = 0
            for o in opp:
                if not crosses(o) or (is_self(o) and self.stp is STPMode.CANCEL_INCOMING):
                    break
                if not is_self(o):
                    avail += o.qty
            if avail < qty:
                return OrderStatus.CANCELED, [], []

        fills: list[tuple[str, int, int]] = []
        stp_ids: list[str] = []
        halted = False
        for o in opp:
            if qty == 0 or not crosses(o):
                break
            if is_self(o):
                if self.stp is STPMode.CANCEL_INCOMING:
                    halted = True
                    break
                self.rest.remove(o)
                stp_ids.append(o.oid)
                continue
            q = min(qty, o.qty)
            fills.append((o.oid, o.tick, q))
            qty -= q
            o.qty -= q
            if o.qty == 0:
                self.rest.remove(o)

        if qty == 0:
            return OrderStatus.FILLED, fills, stp_ids
        if tick is None or halted or tif is not TimeInForce.GTC:
            return OrderStatus.CANCELED, fills, stp_ids
        self.seq += 1
        self.rest.append(_Rest(oid, side, tick, qty, owner, post_only, self.seq))
        return (OrderStatus.PARTIALLY_FILLED if fills else OrderStatus.NEW), fills, stp_ids

    def cancel(self, oid: str) -> bool:
        o = self._find(oid)
        if o is None:
            return False
        self.rest.remove(o)
        return True

    def modify(
        self, oid: str, new_tick: int | None, new_qty: int | None
    ) -> tuple[OrderStatus, list[tuple[str, int, int]], list[str]]:
        o = self._find(oid)
        if o is None:
            return OrderStatus.REJECTED, [], []
        if new_qty == 0:
            self.rest.remove(o)
            return OrderStatus.CANCELED, [], []
        tick = o.tick if new_tick is None else new_tick
        qty = o.qty if new_qty is None else new_qty
        if tick == o.tick and qty <= o.qty:
            o.qty = qty
            return OrderStatus.NEW, [], []
        if o.post_only and any(
            (r.tick <= tick if o.side is Side.BUY else r.tick >= tick) for r in self._queue(o.side.opposite)
        ):
            return OrderStatus.REJECTED, [], []
        self.rest.remove(o)
        return self.submit(oid, o.side, tick, qty, TimeInForce.GTC, o.post_only, o.owner)

    def depth(self, side: Side) -> list[tuple[int, int]]:
        levels: dict[int, int] = {}
        for o in self._queue(side):
            levels[o.tick] = levels.get(o.tick, 0) + o.qty
        return list(levels.items())


ticks = st.integers(min_value=9_990, max_value=10_010)  # 99.90 .. 100.10
qtys = st.integers(min_value=1, max_value=12)
sides = st.sampled_from(list(Side))
owners = st.sampled_from([None, "alice", "bob"])
tifs = st.sampled_from([TimeInForce.GTC, TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK])


class BookMachine(RuleBasedStateMachine):
    stp = STPMode.NONE

    def __init__(self) -> None:
        super().__init__()
        self.lob = LimitOrderBook(TICK, stp=self.stp)
        self.ref = RefBook(self.stp)
        self.n = 0
        self.ids: list[str] = []

    def _new_id(self) -> str:
        self.n += 1
        oid = f"o{self.n}"
        self.ids.append(oid)
        return oid

    def _check(self, report, expected) -> None:
        status, fills, stp_ids = expected
        assert report.status is status, (report, expected)
        got = [(f.maker_order_id, round(f.price / TICK), f.qty) for f in report.fills]
        assert got == fills
        assert list(report.stp_canceled) == stp_ids

    @rule(side=sides, tick=ticks, qty=qtys, tif=tifs, post_only=st.booleans(), owner=owners)
    def limit(self, side, tick, qty, tif, post_only, owner):
        oid = self._new_id()
        post_only = post_only and tif is TimeInForce.GTC
        order = Order.limit(oid, side, tick * TICK, qty, tif=tif, post_only=post_only, owner=owner)
        self._check(self.lob.submit(order), self.ref.submit(oid, side, tick, qty, tif, post_only, owner))

    @rule(side=sides, qty=st.integers(1, 30), owner=owners)
    def market(self, side, qty, owner):
        oid = self._new_id()
        report = self.lob.submit(Order.market(oid, side, qty, owner=owner))
        self._check(report, self.ref.submit(oid, side, None, qty, TimeInForce.IOC, False, owner))

    @precondition(lambda self: self.ids)
    @rule(data=st.data())
    def resubmit_duplicate(self, data):
        oid = data.draw(st.sampled_from(self.ids))
        report = self.lob.submit(Order.limit(oid, Side.BUY, 99.0, 1))
        self._check(report, self.ref.submit(oid, Side.BUY, 9_900, 1, TimeInForce.GTC, False, None))

    @precondition(lambda self: self.ids)
    @rule(data=st.data())
    def cancel(self, data):
        oid = data.draw(st.sampled_from(self.ids))
        assert self.lob.cancel(oid) == self.ref.cancel(oid)

    @precondition(lambda self: self.ids)
    @rule(data=st.data(), new_tick=st.none() | ticks, new_qty=st.none() | st.integers(0, 15))
    def modify(self, data, new_tick, new_qty):
        oid = data.draw(st.sampled_from(self.ids))
        report = self.lob.modify(oid, new_price=None if new_tick is None else new_tick * TICK, new_qty=new_qty)
        self._check(report, self.ref.modify(oid, new_tick, new_qty))

    @invariant()
    def books_agree(self):
        self.lob.check_invariants()
        depth = self.lob.depth(levels=10_000)
        for side, key in ((Side.BUY, "bids"), (Side.SELL, "asks")):
            assert [(round(p / TICK), q) for p, q in depth[key]] == self.ref.depth(side)
        assert len(self.lob) == len(self.ref.rest)


class BookMachineCancelResting(BookMachine):
    stp = STPMode.CANCEL_RESTING


class BookMachineCancelIncoming(BookMachine):
    stp = STPMode.CANCEL_INCOMING


_settings = settings(max_examples=150, stateful_step_count=60, deadline=None)
TestBookNoSTP = BookMachine.TestCase
TestBookCancelResting = BookMachineCancelResting.TestCase
TestBookCancelIncoming = BookMachineCancelIncoming.TestCase
for _case in (TestBookNoSTP, TestBookCancelResting, TestBookCancelIncoming):
    _case.settings = _settings
del _case
