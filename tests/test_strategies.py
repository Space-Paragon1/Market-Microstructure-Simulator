from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given

from microbook import LimitOrderBook, Order, Side
from microbook.sim import (
    AdaptiveMarketMaker,
    AdaptiveMMConfig,
    AvellanedaStoikovConfig,
    AvellanedaStoikovMarketMaker,
    EventType,
    MarketMaker,
    MarketSimulator,
    TWAPExecutor,
)


def two_sided(bid=99.0, ask=101.0, bid_qty=50, ask_qty=50, tick=1.0) -> LimitOrderBook:
    lob = LimitOrderBook(tick_size=tick)
    lob.place_limit(Order("a", Side.SELL, ask, ask_qty))
    lob.place_limit(Order("b", Side.BUY, bid, bid_qty))
    return lob


@given(
    inv=st.integers(-200, 200),
    bid_qty=st.integers(1, 100),
    ask_qty=st.integers(1, 100),
    vol=st.floats(0, 5),
)
def test_adaptive_mm_never_crosses_itself_or_the_touch(inv, bid_qty, ask_qty, vol):
    lob = two_sided(bid_qty=bid_qty, ask_qty=ask_qty)
    amm = AdaptiveMarketMaker(AdaptiveMMConfig(inv_k=0.5, inv_limit=1_000, vol_k=vol, imb_k=3.0, tick_interval=1))
    amm.portfolio.position = inv
    actions = amm.on_tick(10, lob)
    quotes = {a.kwargs["order"].side: a.kwargs["order"] for a in actions if a.etype is EventType.SUBMIT}
    assert set(quotes) == {Side.BUY, Side.SELL}
    bid, ask = quotes[Side.BUY].price, quotes[Side.SELL].price
    assert bid < ask
    assert bid < 101.0 and ask > 99.0  # never through the opposite touch
    assert all(o.post_only for o in quotes.values())


def test_quoter_keeps_priority_when_price_unchanged():
    lob = two_sided()
    mm = MarketMaker(half_spread_ticks=1, tick_interval=1)
    sim = MarketSimulator(lob, [mm])
    sim.schedule_snapshots(0, 5)
    sim.run(until=5)
    assert sim.exec_metrics["mm"].fills == []  # nobody traded
    assert sim.exec_metrics["mm"].orders_submitted == 2  # quoted once, then left alone


def test_inventory_limit_pulls_one_side():
    lob = two_sided()
    amm = AdaptiveMarketMaker(AdaptiveMMConfig(inv_limit=10))
    amm.portfolio.position = 10
    bid, ask, _, _ = amm.compute_quotes(0, lob)
    assert bid is None and ask is not None


class TestAvellanedaStoikov:
    def make(self, **kw):
        cfg = AvellanedaStoikovConfig(gamma=0.5, k=10.0, horizon=1_000, sigma=0.1, **kw)
        return AvellanedaStoikovMarketMaker(cfg)

    def test_reservation_price_and_spread_formulas(self):
        a = self.make()
        tau = 1_000 - 200
        assert a.reservation_price(100.0, 4, 200, 0.1) == pytest.approx(100.0 - 4 * 0.5 * 0.01 * tau)
        import math

        assert a.optimal_spread(200, 0.1) == pytest.approx(0.5 * 0.01 * tau + (2 / 0.5) * math.log(1 + 0.5 / 10))

    def test_skew_fades_at_horizon_unless_rolling(self):
        a = self.make()
        assert a.reservation_price(100.0, 10, 1_000, 0.1) == 100.0
        b = self.make(rolling_tau=100)
        assert b.reservation_price(100.0, 10, 1_000, 0.1) == pytest.approx(100.0 - 10 * 0.5 * 0.01 * 100)

    def test_online_sigma_estimate(self):
        a = AvellanedaStoikovMarketMaker(AvellanedaStoikovConfig())
        assert a.sigma() is None
        for t, m in [(0, 100.0), (10, 100.1), (20, 100.0), (30, 100.1)]:
            a._samples.append((t, m))
        assert a.sigma() == pytest.approx((3 * 0.01 / 30) ** 0.5)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            AvellanedaStoikovMarketMaker(AvellanedaStoikovConfig(gamma=0))


class TestTWAP:
    def test_executes_full_quantity_on_schedule(self):
        twap = TWAPExecutor(Side.BUY, total_qty=50, start=0, end=90, tick_interval=10)
        sim = MarketSimulator(two_sided(ask_qty=1_000, tick=0.01), [twap])
        sim.schedule_snapshots(0, 100, every=10)
        sim.run(until=100)
        m = sim.exec_metrics["twap"]
        assert m.filled_qty == 50 and m.taker_qty == 50
        assert m.orders_submitted == 10  # one slice per interval

    def test_unfilled_quantity_is_rescheduled(self):
        twap = TWAPExecutor(Side.BUY, total_qty=20, start=0, end=0, tick_interval=1)
        sim = MarketSimulator(two_sided(ask_qty=5, tick=0.01), [twap])
        sim.schedule(0, EventType.SNAPSHOT)
        sim.run(until=0)
        assert twap.remaining == 15  # only 5 were available

    def test_validation(self):
        with pytest.raises(ValueError):
            TWAPExecutor(Side.BUY, 0, 0, 10)
        with pytest.raises(ValueError):
            TWAPExecutor(Side.BUY, 10, 10, 0)
