from microbook import LimitOrderBook, Order, Side


def test_cancel_removes_order():
    lob = LimitOrderBook()
    lob.place_limit(Order("b1", Side.BUY, 99.0, 5, ts=1))
    lob.place_limit(Order("b2", Side.BUY, 99.0, 5, ts=2))

    assert lob.cancel("b1") is True

    # now b2 should be first at that price
    fills = lob.place_limit(Order("s1", Side.SELL, 99.0, 3, ts=3))
    assert [(f.maker_order_id, f.qty) for f in fills] == [("b2", 3)]
