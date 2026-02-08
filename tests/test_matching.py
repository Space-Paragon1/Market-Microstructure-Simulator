from microbook import LimitOrderBook, Order, Side


def test_crossing_generates_fills_and_rests_remainder():
    lob = LimitOrderBook()
    lob.place_limit(Order("s1", Side.SELL, 101.0, 3, ts=1))
    lob.place_limit(Order("s2", Side.SELL, 102.0, 3, ts=2))

    fills = lob.place_limit(Order("b1", Side.BUY, 102.0, 10, ts=3))

    # should fill s1@101 qty3 then s2@102 qty3, remainder rests bid at 102
    assert [(f.price, f.qty) for f in fills] == [(101.0, 3), (102.0, 3)]
    assert lob.best_bid() == 102.0
    assert lob.best_ask() is None
