from microbook.sim.portfolio import Portfolio
from microbook.types import Fill
from microbook import Side


def test_realized_pnl_long_roundtrip():
    p = Portfolio()
    # buy 10 @ 100
    p.on_fill(Fill("t", "m", 100.0, 10), my_order_id="x", my_side=Side.BUY)
    assert p.position == 10
    assert p.avg_cost == 100.0

    # sell 10 @ 101 -> realized pnl = 10
    p.on_fill(Fill("t", "m", 101.0, 10), my_order_id="x", my_side=Side.SELL)
    assert p.position == 0
    assert round(p.realized_pnl, 6) == 10.0
