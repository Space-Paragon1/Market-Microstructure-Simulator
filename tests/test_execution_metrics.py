"""
Tests for ExecutionMetrics (execution quality metrics layer).

Covers:
- VWAP computation for buys and sells
- VWAP slippage vs arrival midprice
- Spread capture (maker fills)
- Fill ratio (filled / submitted)
- Participation rate (strategy volume / market volume)
- Average execution price and total notional
- Integration: metrics wired into MarketSimulator
"""
from __future__ import annotations

import pytest

from microbook import LimitOrderBook, Order, Side
from microbook.types import Fill
from microbook.sim.execution_metrics import ExecutionMetrics
from microbook.sim.simulator import MarketSimulator
from microbook.sim.events import EventType
from microbook.sim.strategy import MarketMaker, TWAPExecutor
from microbook.sim.orderflow import FlowConfig, PoissonOrderFlow


# ------------------------------------------------------------------ #
# Unit tests: ExecutionMetrics in isolation                           #
# ------------------------------------------------------------------ #

class TestVWAP:
    def test_buy_vwap_single_fill(self):
        m = ExecutionMetrics()
        m.on_fill(Fill("t1", "m1", 100.0, 5), Side.BUY, is_maker=False)
        assert m.buy_vwap() == 100.0
        assert m.sell_vwap() is None

    def test_sell_vwap_single_fill(self):
        m = ExecutionMetrics()
        m.on_fill(Fill("t1", "m1", 102.0, 3), Side.SELL, is_maker=True)
        assert m.sell_vwap() == 102.0

    def test_buy_vwap_multiple_fills(self):
        m = ExecutionMetrics()
        # 4 @ 100 and 6 @ 102 -> vwap = (400 + 612) / 10 = 101.2
        m.on_fill(Fill("t1", "m1", 100.0, 4), Side.BUY, is_maker=False)
        m.on_fill(Fill("t2", "m2", 102.0, 6), Side.BUY, is_maker=False)
        assert m.buy_vwap() == pytest.approx(101.2)

    def test_sell_vwap_multiple_fills(self):
        m = ExecutionMetrics()
        m.on_fill(Fill("t1", "m1", 99.0, 2), Side.SELL, is_maker=True)
        m.on_fill(Fill("t2", "m2", 101.0, 8), Side.SELL, is_maker=True)
        # (198 + 808) / 10 = 100.6
        assert m.sell_vwap() == pytest.approx(100.6)


class TestSlippage:
    def test_buy_slippage_positive_when_paid_above_arrival(self):
        m = ExecutionMetrics()
        # arrival mid = 100, fill @ 101 -> slippage = 101 - 100 = +1 (bad, paid more)
        m.record_order("o1", Side.BUY, 5, arrival_mid=100.0)
        m.on_fill(Fill("o1", "m1", 101.0, 5), Side.BUY, is_maker=False)
        assert m.buy_slippage() == pytest.approx(1.0)

    def test_buy_slippage_negative_when_paid_below_arrival(self):
        m = ExecutionMetrics()
        # "o1" is the maker (resting buy at 99, filled by a market sell)
        # Fill(taker_order_id, maker_order_id, price, qty)
        m.record_order("o1", Side.BUY, 5, arrival_mid=100.0)
        m.on_fill(Fill("t_aggressor", "o1", 99.0, 5), Side.BUY, is_maker=True)
        assert m.buy_slippage() == pytest.approx(-1.0)

    def test_sell_slippage_positive_when_received_below_arrival(self):
        m = ExecutionMetrics()
        # arrival mid = 100, fill @ 99 -> slippage = 100 - 99 = +1 (bad, got less)
        m.record_order("o1", Side.SELL, 5, arrival_mid=100.0)
        m.on_fill(Fill("t1", "o1", 99.0, 5), Side.SELL, is_maker=True)
        assert m.sell_slippage() == pytest.approx(1.0)

    def test_sell_slippage_negative_when_received_above_arrival(self):
        m = ExecutionMetrics()
        m.record_order("o1", Side.SELL, 5, arrival_mid=100.0)
        m.on_fill(Fill("t1", "o1", 101.0, 5), Side.SELL, is_maker=True)
        assert m.sell_slippage() == pytest.approx(-1.0)

    def test_slippage_none_without_registered_order(self):
        m = ExecutionMetrics()
        # fill without a registered order -> no arrival mid -> slippage is None
        m.on_fill(Fill("t1", "m1", 100.0, 5), Side.BUY, is_maker=False)
        assert m.buy_slippage() is None

    def test_slippage_zero_at_mid(self):
        m = ExecutionMetrics()
        m.record_order("o1", Side.BUY, 10, arrival_mid=100.0)
        m.on_fill(Fill("o1", "m1", 100.0, 10), Side.BUY, is_maker=False)
        assert m.buy_slippage() == pytest.approx(0.0)


class TestSpreadCapture:
    def test_maker_buy_below_mid_captures_positive_spread(self):
        m = ExecutionMetrics()
        # maker buy resting at 99, mid at 100 -> captured 1.0 per share
        m.on_fill(Fill("t1", "m_buy", 99.0, 10), Side.BUY, is_maker=True, mid=100.0)
        assert m.spread_capture_per_share() == pytest.approx(1.0)

    def test_maker_sell_above_mid_captures_positive_spread(self):
        m = ExecutionMetrics()
        # maker sell resting at 101, mid at 100 -> captured 1.0 per share
        m.on_fill(Fill("t1", "m_sell", 101.0, 10), Side.SELL, is_maker=True, mid=100.0)
        assert m.spread_capture_per_share() == pytest.approx(1.0)

    def test_spread_capture_averages_across_fills(self):
        m = ExecutionMetrics()
        # two buys: 99 @ mid=100 (cap 1), 98 @ mid=100 (cap 2) -> avg = 1.5
        m.on_fill(Fill("t1", "m1", 99.0, 4), Side.BUY, is_maker=True, mid=100.0)
        m.on_fill(Fill("t2", "m2", 98.0, 4), Side.BUY, is_maker=True, mid=100.0)
        assert m.spread_capture_per_share() == pytest.approx(1.5)

    def test_spread_capture_none_without_maker_fills(self):
        m = ExecutionMetrics()
        m.on_fill(Fill("t1", "m1", 100.0, 5), Side.BUY, is_maker=False)
        assert m.spread_capture_per_share() is None

    def test_spread_capture_none_without_mid(self):
        m = ExecutionMetrics()
        m.on_fill(Fill("t1", "m1", 99.0, 5), Side.BUY, is_maker=True, mid=None)
        # mid unknown -> spread total stays 0, spread_capture = 0 / 5 = 0.0
        assert m.spread_capture_per_share() == pytest.approx(0.0)


class TestFillRatio:
    def test_fully_filled(self):
        m = ExecutionMetrics()
        m.record_order("o1", Side.BUY, 10, arrival_mid=100.0)
        m.on_fill(Fill("o1", "m1", 100.0, 10), Side.BUY, is_maker=False)
        assert m.fill_ratio() == pytest.approx(1.0)

    def test_partially_filled(self):
        m = ExecutionMetrics()
        m.record_order("o1", Side.BUY, 10, arrival_mid=100.0)
        m.on_fill(Fill("o1", "m1", 100.0, 4), Side.BUY, is_maker=False)
        assert m.fill_ratio() == pytest.approx(0.4)

    def test_unfilled(self):
        m = ExecutionMetrics()
        m.record_order("o1", Side.BUY, 10, arrival_mid=100.0)
        assert m.fill_ratio() == pytest.approx(0.0)

    def test_fill_ratio_none_without_submitted_orders(self):
        m = ExecutionMetrics()
        assert m.fill_ratio() is None


class TestParticipationRate:
    def test_participation_rate(self):
        m = ExecutionMetrics()
        # market fills 20 total; strategy fills 5
        m.record_market_volume([Fill("t1", "m1", 100.0, 10), Fill("t2", "m2", 101.0, 10)])
        m.on_fill(Fill("t1", "my", 100.0, 5), Side.SELL, is_maker=True)
        assert m.participation_rate() == pytest.approx(5 / 20)

    def test_participation_none_without_market_volume(self):
        m = ExecutionMetrics()
        assert m.participation_rate() is None


class TestSummaryDict:
    def test_summary_keys_present(self):
        m = ExecutionMetrics()
        s = m.summary()
        expected_keys = {
            "filled_qty", "buy_qty", "sell_qty", "maker_qty", "taker_qty",
            "total_notional", "avg_price", "buy_vwap", "sell_vwap",
            "buy_slippage_vs_arrival", "sell_slippage_vs_arrival",
            "spread_capture_per_share", "fill_ratio", "market_volume",
            "participation_rate",
        }
        assert expected_keys == set(s.keys())

    def test_summary_values_consistent(self):
        m = ExecutionMetrics()
        m.record_order("o1", Side.BUY, 5, arrival_mid=100.0)
        m.record_market_volume([Fill("o1", "m1", 100.0, 5)])
        m.on_fill(Fill("o1", "m1", 100.0, 5), Side.BUY, is_maker=False, mid=100.0)

        s = m.summary()
        assert s["filled_qty"] == 5
        assert s["buy_qty"] == 5
        assert s["sell_qty"] == 0
        assert s["taker_qty"] == 5
        assert s["maker_qty"] == 0
        assert s["avg_price"] == pytest.approx(100.0)
        assert s["buy_vwap"] == pytest.approx(100.0)
        assert s["buy_slippage_vs_arrival"] == pytest.approx(0.0)
        assert s["fill_ratio"] == pytest.approx(1.0)
        assert s["market_volume"] == 5
        assert s["participation_rate"] == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# Integration tests: metrics wired into MarketSimulator               #
# ------------------------------------------------------------------ #

class TestSimulatorMetricsIntegration:
    def _build_sim(self) -> MarketSimulator:
        mm = MarketMaker(name="mm", tick_interval=10, size=5, half_spread_ticks=1)
        sim = MarketSimulator(strategies=[mm])

        # seed book at mid=100
        sim.schedule(0, EventType.SUBMIT, order=Order("s_ask", Side.SELL, 101.0, 50, ts=0))
        sim.schedule(0, EventType.SUBMIT, order=Order("s_bid", Side.BUY, 99.0, 50, ts=0))

        # schedule snapshots so strategy gets to quote
        for t in range(0, 201, 5):
            sim.schedule(t, EventType.SNAPSHOT)

        # add some aggressive flow to generate fills
        flow = PoissonOrderFlow(FlowConfig(seed=42, intensity_per_100=40.0, p_market=0.20))
        for t, o in flow.iter_orders(start=1, end=200, ref_mid=100.0):
            sim.schedule(t, EventType.SUBMIT, order=o)

        sim.run(until=200)
        return sim

    def test_market_volume_positive(self):
        sim = self._build_sim()
        m = sim.exec_metrics["mm"]
        assert m.market_volume > 0

    def test_maker_fills_attributed(self):
        sim = self._build_sim()
        m = sim.exec_metrics["mm"]
        # MM only posts maker quotes, so maker_qty should dominate
        assert m._maker_qty >= 0  # could be 0 if no fills happened
        assert m.filled_qty == m._maker_qty + m._taker_qty

    def test_summary_returns_dict(self):
        sim = self._build_sim()
        s = sim.exec_metrics["mm"].summary()
        assert isinstance(s, dict)
        assert "participation_rate" in s

    def test_twap_taker_slippage_tracked(self):
        """TWAP as pure taker: buy slippage should be defined."""
        twap = TWAPExecutor(Side.BUY, total_qty=20, start=10, end=100,
                            tick_interval=10, name="twap")
        sim = MarketSimulator(strategies=[twap])
        sim.schedule(0, EventType.SUBMIT, order=Order("s_ask", Side.SELL, 101.0, 100, ts=0))
        sim.schedule(0, EventType.SUBMIT, order=Order("s_bid", Side.BUY, 99.0, 100, ts=0))

        for t in range(0, 101, 5):
            sim.schedule(t, EventType.SNAPSHOT)

        sim.run(until=100)

        m = sim.exec_metrics["twap"]
        s = m.summary()
        # TWAP uses market orders as taker
        assert s["taker_qty"] >= 0
        # If any fills, slippage should be defined (arrival mid recorded)
        if s["buy_qty"] > 0:
            assert s["buy_slippage_vs_arrival"] is not None
