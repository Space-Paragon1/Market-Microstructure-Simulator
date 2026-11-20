from __future__ import annotations

import math
import random
import statistics

import pytest

from microbook import Side
from microbook.sim import FundamentalValue, MarketConfig, build_market, format_table, markouts, pnl_stats
from microbook.sim.agents import poisson
from microbook.sim.analytics import inventory_stats, max_drawdown, mid_at, strategy_report
from microbook.sim.execution_metrics import FillRecord

PATH = [(0, 100.0), (10, 101.0), (20, math.nan), (30, 99.0)]


def test_mid_at_step_function():
    assert mid_at(PATH, -1) is None
    assert mid_at(PATH, 0) == 100.0 and mid_at(PATH, 9) == 100.0 and mid_at(PATH, 10) == 101.0
    assert mid_at(PATH, 25) is None  # one-sided book
    assert mid_at(PATH, 1_000) == 99.0


def test_markout_signs():
    buy = FillRecord(0, Side.BUY, 100.0, 2, True, 100.0)
    sell = FillRecord(0, Side.SELL, 100.0, 2, True, 100.0)
    assert markouts([buy], PATH, (5, 10))[10] == pytest.approx(1.0)  # price rose after buying
    assert markouts([sell], PATH, (10,))[10] == pytest.approx(-1.0)
    assert markouts([buy, sell], PATH, (10,))[10] == pytest.approx(0.0)


def test_markouts_skip_undefined_mid_and_horizon_past_end():
    f = FillRecord(15, Side.BUY, 100.0, 1, False, None)
    assert markouts([f], PATH, (5,))[5] is None  # mid undefined at t=20
    assert markouts([f], PATH, (100,), end_time=50)[100] is None


def test_drawdown_and_pnl_stats():
    assert max_drawdown([0, 5, 2, 8, 1, 3]) == 7
    s = pnl_stats([0.0, 1.0, 2.0, 3.0])
    assert s["final"] == 3.0 and s["max_drawdown"] == 0.0 and math.isnan(s["sharpe"])  # zero vol
    assert math.isnan(pnl_stats([math.nan])["final"])
    assert inventory_stats([0, -4, 2])["max_abs_inventory"] == 4


def test_format_table():
    out = format_table({"a": {"x": 1.23456, "y": None}, "bb": {"x": math.nan, "y": 3}}, precision=2)
    lines = out.splitlines()
    assert lines[0].split() == ["x", "y"] and "1.23" in lines[2] and "-" in lines[2] and "nan" in lines[3]


def test_strategy_report_on_a_market():
    sim = build_market(MarketConfig(seed=0), horizon=1_000).run(1_000)
    rep = strategy_report(sim, horizons=(1, 10))
    assert set(rep) == {"noise", "informed"}
    assert rep["informed"]["final"] > 0  # informed traders profit...
    assert rep["noise"]["final"] == pytest.approx(-rep["informed"]["final"])  # ...at noise traders' expense


def test_poisson_sampler_moments():
    rng = random.Random(0)
    for lam in (0.3, 4.0, 50.0):
        xs = [poisson(rng, lam) for _ in range(20_000)]
        assert statistics.fmean(xs) == pytest.approx(lam, rel=0.05)
    assert poisson(rng, 0.0) == 0


def test_fundamental_is_deterministic_and_query_order_independent():
    a = FundamentalValue(100.0, 0.01, jump_rate=0.01, jump_size=0.5, seed=4)
    b = FundamentalValue(100.0, 0.01, jump_rate=0.01, jump_size=0.5, seed=4)
    late = a.at(500)
    assert [b.at(t) for t in range(501)][-1] == late
    with pytest.raises(ValueError):
        a.at(-1)
