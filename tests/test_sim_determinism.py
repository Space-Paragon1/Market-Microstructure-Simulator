from microbook.sim.simulator import MarketSimulator
from microbook.sim.events import EventType
from microbook.sim.orderflow import FlowConfig, PoissonOrderFlow
from microbook import Order, Side

def test_sim_is_deterministic_given_seed():
    def run(seed: int):
        sim = MarketSimulator()
        sim.schedule(0, EventType.SUBMIT, order=Order("seed_s", Side.SELL, 101.0, 20, ts=0))
        sim.schedule(0, EventType.SUBMIT, order=Order("seed_b", Side.BUY,  99.0, 20, ts=0))

        flow = PoissonOrderFlow(FlowConfig(seed=seed, intensity_per_100=30.0, p_market=0.10))
        for t, o in flow.iter_orders(start=1, end=200, ref_mid=100.0):
            sim.schedule(t, EventType.SUBMIT, order=o)

        res = sim.run(until=200)
        # hashable signature: (maker, taker, price, qty)
        sig = [(f.maker_order_id, f.taker_order_id, f.price, f.qty) for f in res.fills]
        top = sim.lob.top_of_book()
        return sig, top

    sig1, top1 = run(seed=123)
    sig2, top2 = run(seed=123)
    assert sig1 == sig2
    assert top1 == top2
