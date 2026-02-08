from microbook import Order, Side
from microbook.sim.simulator import MarketSimulator
from microbook.sim.events import EventType
from microbook.sim.orderflow import FlowConfig, PoissonOrderFlow
from microbook.sim.strategy import MarketMaker, TWAPExecutor


def seed_book(sim: MarketSimulator):
    sim.schedule(0, EventType.SUBMIT, order=Order("seed_s1", Side.SELL, 101.0, 50, ts=0))
    sim.schedule(0, EventType.SUBMIT, order=Order("seed_b1", Side.BUY,  99.0, 50, ts=0))


def main():
    mm = MarketMaker(tick_interval=10, size=5, half_spread_ticks=1)
    twap = TWAPExecutor(Side.BUY, total_qty=40, start=50, end=250, tick_interval=20, name="twap_buy")

    sim = MarketSimulator(strategies=[mm, twap])
    seed_book(sim)

    flow = PoissonOrderFlow(FlowConfig(seed=7, intensity_per_100=40.0, p_market=0.10))
    for t, o in flow.iter_orders(start=1, end=500, ref_mid=100.0):
        sim.schedule(t, EventType.SUBMIT, order=o)
        if t % 5 == 0:
            sim.schedule(t, EventType.SNAPSHOT)

    res = sim.run(until=500)

    # report
    for s in sim.strategies:
        mtm = s.portfolio.mark_to_market(sim.lob)
        print(f"{s.name}: pos={s.portfolio.position} cash={s.portfolio.cash:.2f} "
              f"realized={s.portfolio.realized_pnl:.2f} mtm={mtm if mtm is not None else 'NA'}")

    print("fills:", len(res.fills))
    print("final top:", sim.lob.top_of_book())


if __name__ == "__main__":
    main()
