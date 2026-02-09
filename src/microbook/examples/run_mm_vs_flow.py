from microbook import Order, Side
from microbook.sim.simulator import MarketSimulator
from microbook.sim.events import EventType
from microbook.sim.orderflow import FlowConfig, PoissonOrderFlow
from microbook.sim.strategy import TWAPExecutor
from microbook.sim.strategies_mm import AdaptiveMarketMaker, AdaptiveMMConfig

def seed_book(sim: MarketSimulator):
    sim.schedule(0, EventType.SUBMIT, order=Order("seed_s1", Side.SELL, 101.0, 50, ts=0))
    sim.schedule(0, EventType.SUBMIT, order=Order("seed_b1", Side.BUY,  99.0, 50, ts=0))


def main():
    mm = AdaptiveMarketMaker(AdaptiveMMConfig(
    tick_interval=10,
    size=5,
    base_half_spread_ticks=1,
    vol_window=30,
    vol_k=3.0,
    inv_limit=25,
    inv_k=0.08,
    imb_k=2.0,
))

    twap = TWAPExecutor(Side.BUY, total_qty=40, start=50, end=250, tick_interval=20, name="twap_buy")

    sim = MarketSimulator(strategies=[mm, twap])
    seed_book(sim)

    flow = PoissonOrderFlow(FlowConfig(seed=7, intensity_per_100=40.0, p_market=0.10))
    for t, o in flow.iter_orders(start=1, end=500, ref_mid=100.0):
        sim.schedule(t, EventType.SUBMIT, order=o)
        if t % 5 == 0:
            sim.schedule(t, EventType.SNAPSHOT)

    res = sim.run(until=500)
    print("=== EXECUTION METRICS ===")
    for name, m in sim.exec_metrics.items():
        print(
            name,
            "market_volume=", m.market_volume,
            "filled_qty=", m.filled_qty,
            "buy_qty=", m.buy_qty,
            "sell_qty=", m.sell_qty,
        )

    

    print("RUN COMPLETE")
    print("fills:", len(res.fills))
    print("final top:", sim.lob.top_of_book())
    for s in sim.strategies:
        mtm = s.portfolio.mark_to_market(sim.lob)
        print(f"{s.name}: pos={s.portfolio.position} cash={s.portfolio.cash:.2f} realized={s.portfolio.realized_pnl:.2f} mtm={mtm}")

    import matplotlib.pyplot as plt

    plt.figure()
    for name, series in res.pnl_series.items():
        plt.plot(res.pnl_t, series, label=name)

    plt.title("Strategy MTM PnL over time")
    plt.xlabel("time")
    plt.ylabel("PnL (MTM)")
    plt.legend()
    plt.show()


    # report
    for s in sim.strategies:
        mtm = s.portfolio.mark_to_market(sim.lob)
        print(f"{s.name}: pos={s.portfolio.position} cash={s.portfolio.cash:.2f} "
              f"realized={s.portfolio.realized_pnl:.2f} mtm={mtm if mtm is not None else 'NA'}")

    print("fills:", len(res.fills))
    print("final top:", sim.lob.top_of_book())


if __name__ == "__main__":
    main()
