from __future__ import annotations

import pytest

from microbook import LimitOrderBook, Order, Side, STPMode
from microbook.sim import (
    Action,
    EventType,
    MarketConfig,
    MarketSimulator,
    Strategy,
    build_market,
)


def seeded(strategies=(), **kw) -> MarketSimulator:
    sim = MarketSimulator(strategies=list(strategies), **kw)
    sim.schedule(0, EventType.SUBMIT, order=Order("ask", Side.SELL, 101.0, 100))
    sim.schedule(0, EventType.SUBMIT, order=Order("bid", Side.BUY, 99.0, 100))
    return sim


class Recorder(Strategy):
    """Buys once at a given time and records every callback."""

    def __init__(self, name="rec", *, buy_at=1, latency=0):
        super().__init__(name, latency=latency)
        self.buy_at = buy_at
        self.events: list[tuple] = []

    def on_start(self, now, lob):
        return [Action.timer(self.buy_at)]

    def on_timer(self, now, lob):
        self.events.append(("timer", now))
        return [Action.submit(Order.market(self.next_order_id(), Side.BUY, 3), now)]

    def on_fill(self, now, fill, side, is_maker, lob):
        self.events.append(("fill", now, side, is_maker, fill.qty))
        return []

    def on_report(self, now, report):
        self.events.append(("report", now, report.status.value))
        return []


def test_timer_latency_and_callbacks():
    rec = Recorder(buy_at=5, latency=3)
    sim = seeded([rec])
    sim.run(until=20)
    assert rec.events == [("timer", 5), ("fill", 8, Side.BUY, False, 3), ("report", 8, "FILLED")]
    assert sim.result.fills[-1].ts == 8  # stamped with arrival time, not decision time
    assert rec.portfolio.position == 3


def test_timer_in_the_past_is_an_error():
    class Bad(Strategy):
        def on_start(self, now, lob):
            return [Action.timer(now)]

    with pytest.raises(ValueError, match="future"):
        MarketSimulator(strategies=[Bad("bad")]).run(until=1)


def test_maker_attribution_and_spread_capture():
    class Quoter(Strategy):
        def on_start(self, now, lob):
            return [Action.submit(Order.limit("q", Side.BUY, 100.0, 5), now)]

    q = Quoter("q")
    sim = seeded([q])
    sim.schedule(1, EventType.SUBMIT, order=Order.market("hit", Side.SELL, 5))
    sim.run(until=2)
    m = sim.exec_metrics["q"]
    assert q.portfolio.position == 5 and m.maker_qty == 5
    # pre-trade mid was (100 + 101) / 2 = 100.5; maker bought at 100
    assert m.spread_capture_per_share() == pytest.approx(0.5)


def test_default_simulator_prevents_self_trades():
    class Crosser(Strategy):
        def on_start(self, now, lob):
            return [
                Action.submit(Order.limit("s", Side.SELL, 100.0, 5), now),
                Action.submit(Order.limit("b", Side.BUY, 100.0, 5), now),
            ]

    c = Crosser("c")
    sim = MarketSimulator(strategies=[c])
    assert sim.lob.stp is STPMode.CANCEL_RESTING
    sim.run(until=1)
    assert sim.result.fills == [] and c.portfolio.volume == 0
    assert "s" not in sim.lob and "b" in sim.lob


def test_duplicate_strategy_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        MarketSimulator(strategies=[Strategy("x"), Strategy("x")])


def test_cannot_schedule_in_the_past():
    sim = seeded()
    sim.run(until=10)
    with pytest.raises(ValueError):
        sim.schedule(5, EventType.SNAPSHOT)


def test_result_series_exist_without_snapshots():
    res = seeded([Strategy("s")]).run(until=5)
    assert res.pnl_series == {"s": []} and res.inventory_series == {"s": []}


def test_mid_path_is_a_step_function():
    sim = seeded()
    sim.schedule(3, EventType.SUBMIT, order=Order("b2", Side.BUY, 100.0, 1))
    res = sim.run(until=10)
    assert res.mid_at(-1) is None
    assert res.mid_at(2) == 100.0
    assert res.mid_at(3) == 100.5 and res.mid_at(100) == 100.5


def test_snapshots_record_pnl_and_inventory():
    rec = Recorder(buy_at=2)
    sim = seeded([rec])
    sim.schedule_snapshots(0, 4, every=2)
    res = sim.run(until=4)
    assert res.pnl_t == [0, 2, 4]
    assert res.inventory_series["rec"] == [0, 0, 3]  # the t=2 snapshot precedes the t=2 timer
    assert res.pnl_series["rec"][-1] == pytest.approx(3 * (100.0 - 101.0))


# --------------------------------------------------------------------------- #
# Closed-loop market                                                          #
# --------------------------------------------------------------------------- #


def test_market_is_deterministic():
    def run():
        sim = build_market(MarketConfig(seed=3), horizon=2_000).run(2_000)
        return [tuple(f) for f in sim.result.fills], sim.result.mid_path

    assert run() == run()


def test_pnl_is_zero_sum_across_all_participants():
    """Every share bought was sold by someone: with no fees, total PnL is 0."""
    from microbook.sim import AdaptiveMarketMaker, TWAPExecutor

    strategies = [AdaptiveMarketMaker(), TWAPExecutor(Side.SELL, 200, 100, 1_500, tick_interval=10)]
    sim = build_market(MarketConfig(seed=5), strategies, horizon=2_000).run(2_000)
    mid = sim.lob.midprice()
    assert mid is not None and len(sim.result.fills) > 500
    assert sum(s.portfolio.total_pnl(mid) for s in sim.strategies) == pytest.approx(0.0, abs=1e-6)
    assert sum(s.portfolio.position for s in sim.strategies) == 0


@pytest.mark.parametrize("seed", [1, 11])
def test_price_discovery_tracks_fundamental(seed):
    """Time-averaged |mid - V| stays small relative to how far V itself wanders.
    (Single endpoints are a poor test: the deviation has fat tails.)"""
    import math
    import statistics

    mkt = build_market(MarketConfig(seed=seed), horizon=10_000)
    sim = mkt.run(10_000)
    pts = [(t, m) for t, m in zip(sim.ts.t, sim.ts.mid) if not math.isnan(m)]
    tracking_error = statistics.fmean(abs(m - mkt.fundamental.at(t)) for t, m in pts)
    v = [mkt.fundamental.at(t) for t, _ in pts]
    assert tracking_error < 0.15  # < 15 ticks on average
    assert tracking_error < 0.25 * (max(v) - min(v))
    sim.lob.check_invariants()


def test_fundamental_path_is_independent_of_strategies():
    from microbook.sim import MarketMaker

    a = build_market(MarketConfig(seed=2), horizon=500)
    b = build_market(MarketConfig(seed=2), [MarketMaker()], horizon=500)
    a.run(500), b.run(500)
    assert [a.fundamental.at(t) for t in range(500)] == [b.fundamental.at(t) for t in range(500)]


def test_owner_index_stays_bounded():
    sim = build_market(MarketConfig(seed=1), horizon=5_000).run(5_000)
    assert len(sim._order_owner) <= len(sim.lob)


def test_custom_book_is_respected():
    lob = LimitOrderBook(tick_size=0.5, stp=STPMode.NONE)
    assert MarketSimulator(lob).lob is lob
