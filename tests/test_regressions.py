"""Regression tests: one per defect found in the v0.1 engine and simulator."""

from __future__ import annotations

import pytest

from microbook import LimitOrderBook, Order, OrderRejected, OrderStatus, Side
from microbook.sim import (
    AdaptiveMarketMaker,
    AdaptiveMMConfig,
    EventType,
    ExecutionMetrics,
    MarketSimulator,
    Portfolio,
    Strategy,
    TWAPExecutor,
)
from microbook.sim.strategy import Action
from microbook.types import Fill


def test_duplicate_live_order_id_is_rejected_not_orphaned():
    lob = LimitOrderBook()
    lob.place_limit(Order("x", Side.BUY, 99.0, 5))
    with pytest.raises(OrderRejected, match="duplicate"):
        lob.place_limit(Order("x", Side.BUY, 98.0, 5))
    assert lob.submit(Order("x", Side.BUY, 98.0, 5)).status is OrderStatus.REJECTED
    assert lob.cancel("x") is True
    assert lob.depth()["bids"] == []  # nothing orphaned
    lob.check_invariants()


def test_modify_that_crosses_returns_its_fills():
    lob = LimitOrderBook()
    lob.place_limit(Order("a", Side.SELL, 101.0, 5))
    lob.place_limit(Order("b", Side.BUY, 99.0, 5))
    report = lob.modify("b", new_price=101.0)
    assert report.status is OrderStatus.FILLED
    assert [(f.maker_order_id, f.price, f.qty) for f in report.fills] == [("a", 101.0, 5)]


def test_modify_crossing_fills_reach_portfolios_in_simulator():
    class Modifier(Strategy):
        def on_tick(self, now, lob):
            if now == 1:
                return [Action.submit(Order.limit("q", Side.BUY, 99.0, 5), now)]
            if now == 2:
                return [Action.modify("q", now, new_price=101.0)]
            return []

    strat = Modifier("s")
    sim = MarketSimulator(strategies=[strat])
    sim.schedule(0, EventType.SUBMIT, order=Order("ask", Side.SELL, 101.0, 5))
    sim.schedule(1, EventType.SNAPSHOT)
    sim.schedule(2, EventType.SNAPSHOT)
    sim.run(until=3)
    assert strat.portfolio.position == 5
    assert sim.exec_metrics["s"].filled_qty == 5


def test_invalid_modify_leaves_order_intact():
    lob = LimitOrderBook()
    lob.place_limit(Order("b", Side.BUY, 99.0, 5))
    for kwargs in ({"new_qty": -3}, {"new_price": -1.0}, {"new_price": 99.005}):
        assert lob.modify("b", **kwargs).status is OrderStatus.REJECTED
    assert lob.depth()["bids"] == [(99.0, 5)]


def test_noop_modify_keeps_priority():
    lob = LimitOrderBook()
    lob.place_limit(Order("b1", Side.BUY, 99.0, 5))
    lob.place_limit(Order("b2", Side.BUY, 99.0, 5))
    assert lob.modify("b1", new_qty=5, new_price=99.0).accepted
    fills = lob.place_limit(Order("s", Side.SELL, 99.0, 5))
    assert fills[0].maker_order_id == "b1"


def test_fees_are_included_in_pnl():
    p = Portfolio(maker_fee=0.5, taker_fee=0.5)
    p.on_fill(Fill("t", "m", 100.0, 10), "x", Side.BUY)
    p.on_fill(Fill("t", "m", 101.0, 10), "x", Side.SELL)
    assert p.cash == pytest.approx(0.0)  # +10 gross, -10 fees
    assert p.total_pnl(100.0) == pytest.approx(0.0)


def test_slippage_not_diluted_by_unregistered_fills():
    m = ExecutionMetrics()
    m.record_order("o1", Side.BUY, 5, arrival_mid=100.0)
    m.on_fill(Fill("o1", "m", 101.0, 5), Side.BUY, is_maker=False)
    m.on_fill(Fill("o2", "m", 101.0, 5), Side.BUY, is_maker=False)  # unregistered
    assert m.buy_slippage() == pytest.approx(1.0)


def test_twap_does_not_record_zero_arrival_mid():
    twap = TWAPExecutor(Side.BUY, total_qty=10, start=0, end=10, tick_interval=1)
    sim = MarketSimulator(strategies=[twap])
    sim.schedule(0, EventType.SUBMIT, order=Order("s", Side.SELL, 101.0, 50))  # one-sided book
    sim.schedule_snapshots(0, 2)
    sim.run(until=2)
    assert sim.exec_metrics["twap"].filled_qty == 0  # waits for a two-sided market
    sim.schedule(3, EventType.SUBMIT, order=Order("b", Side.BUY, 99.0, 50))
    sim.schedule_snapshots(3, 10)
    sim.run(until=10)
    m = sim.exec_metrics["twap"]
    assert m.filled_qty == 10
    assert m.buy_slippage() == pytest.approx(1.0)  # paid 101 vs arrival mid 100


@pytest.mark.parametrize("inventory", [-30, -10, 0, 10, 30])
def test_adaptive_mm_skew_shifts_quotes_instead_of_crossing(inventory):
    lob = LimitOrderBook(tick_size=1.0)
    lob.place_limit(Order("a", Side.SELL, 101.0, 50))
    lob.place_limit(Order("b", Side.BUY, 99.0, 50))
    amm = AdaptiveMarketMaker(AdaptiveMMConfig(inv_k=0.1, inv_limit=100, imb_k=0.0, vol_k=0.0))
    amm.portfolio.position = inventory
    bid, ask, _, _ = amm.compute_quotes(100, lob)
    assert bid is not None and ask is not None
    assert ask - bid == pytest.approx(2.0)  # width unchanged by skew
    center = (bid + ask) / 2
    if inventory > 0:
        assert center < 100.0  # long -> lean lower to sell
    elif inventory < 0:
        assert center > 100.0  # short -> lean higher to buy
    else:
        assert center == pytest.approx(100.0)


def test_adaptive_mm_imbalance_leans_toward_heavier_side():
    lob = LimitOrderBook(tick_size=1.0)
    lob.place_limit(Order("a", Side.SELL, 101.0, 10))
    lob.place_limit(Order("b", Side.BUY, 99.0, 90))  # bid-heavy: buying pressure
    amm = AdaptiveMarketMaker(AdaptiveMMConfig(inv_k=0.0, imb_k=2.0, vol_k=0.0))
    bid, ask, _, _ = amm.compute_quotes(100, lob)
    assert (bid + ask) / 2 > 100.0
