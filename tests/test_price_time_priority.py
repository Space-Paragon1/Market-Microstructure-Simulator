from microbook import LimitOrderBook, Order, Side


def test_fifo_same_price_level():
    lob = LimitOrderBook()

    # Two asks at same price; earlier ts should fill first
    lob.place_limit(Order("a1", Side.SELL, 100.0, 5, ts=1))
    lob.place_limit(Order("a2", Side.SELL, 100.0, 5, ts=2))

    fills = lob.place_limit(Order("b1", Side.BUY, 100.0, 7, ts=3))

    assert [(f.maker_order_id, f.qty) for f in fills] == [("a1", 5), ("a2", 2)]
