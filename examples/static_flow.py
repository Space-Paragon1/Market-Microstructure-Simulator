"""Adaptive market maker + TWAP buyer against open-loop Poisson order flow.

The flow is pre-generated around a fixed reference price (see
``microbook.sim.orderflow``), so this is a controlled setting for checking fill
attribution and execution metrics rather than a realistic market; see
``compare_market_makers.py`` for the closed-loop market with price discovery.

    python examples/static_flow.py
"""

from __future__ import annotations

from microbook import LimitOrderBook, Order, Side
from microbook.sim import (
    AdaptiveMarketMaker,
    AdaptiveMMConfig,
    EventType,
    MarketSimulator,
    TWAPExecutor,
    format_table,
    strategy_report,
)
from microbook.sim.orderflow import FlowConfig, PoissonOrderFlow


def main() -> None:
    amm = AdaptiveMarketMaker(AdaptiveMMConfig(tick_size=1.0, tick_interval=10, size=5, inv_limit=25), name="amm")
    twap = TWAPExecutor(Side.BUY, total_qty=40, start=50, end=250, tick_interval=20, name="twap_buy")

    sim = MarketSimulator(LimitOrderBook(tick_size=0.01), [amm, twap])
    sim.schedule(0, EventType.SUBMIT, order=Order("seed_ask", Side.SELL, 101.0, 50))
    sim.schedule(0, EventType.SUBMIT, order=Order("seed_bid", Side.BUY, 99.0, 50))
    sim.schedule_snapshots(0, 500, every=5)

    flow = PoissonOrderFlow(FlowConfig(seed=7, intensity_per_100=40.0, p_market=0.10))
    for t, order in flow.iter_orders(start=1, end=500, ref_mid=100.0):
        sim.schedule(t, EventType.SUBMIT, order=order)

    res = sim.run(until=500)

    print("=== execution metrics ===")
    print(format_table({name: m.summary() for name, m in sim.exec_metrics.items()}))
    print("\n=== PnL / risk ===")
    cols = ["final", "max_drawdown", "mean_abs_inventory", "final_inventory", "volume", "markout_1", "markout_100"]
    print(format_table(strategy_report(sim), cols))
    print(f"\nfills: {len(res.fills)}   final top of book: {sim.lob.top_of_book()}")


if __name__ == "__main__":
    main()
