from microbook import LimitOrderBook, Order, Side
from microbook.sim.simulator import MarketSimulator
from microbook.sim.events import EventType
from microbook.sim.orderflow import FlowConfig, PoissonOrderFlow

def seed_book(sim: MarketSimulator):
    # seed some initial liquidity so mid exists
    sim.schedule(0, EventType.SUBMIT, order=Order("seed_s1", Side.SELL, 101.0, 20, ts=0))
    sim.schedule(0, EventType.SUBMIT, order=Order("seed_b1", Side.BUY,  99.0, 20, ts=0))

def main():
    sim = MarketSimulator()
    seed_book(sim)

    flow = PoissonOrderFlow(FlowConfig(seed=42, intensity_per_100=35.0, p_market=0.10))
    ref_mid = 100.0

    # schedule flow + snapshots
    for t, o in flow.iter_orders(start=1, end=500, ref_mid=ref_mid):
        sim.schedule(t, EventType.SUBMIT, order=o)
        if t % 50 == 0:
            sim.schedule(t, EventType.SNAPSHOT)

    res = sim.run(until=500)

    print("fills:", len(res.fills))
    if res.snapshots:
        print("last snapshot:", res.snapshots[-1])

if __name__ == "__main__":
    main()
