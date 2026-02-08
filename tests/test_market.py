from microbook import LimitOrderBook, Order, Side

def test_market_buy_consumes_and_does_not_rest():
    lob = LimitOrderBook()
    lob.place_limit(Order("s1", Side.SELL, 101.0, 3, ts=1))
    lob.place_limit(Order("s2", Side.SELL, 102.0, 3, ts=2))

    fills = lob.place_market(Order("mb1", Side.BUY, 1.0, 10, ts=3))  # price ignored
    assert [(f.price, f.qty) for f in fills] == [(101.0, 3), (102.0, 3)]

    # no bid should rest from the market order
    assert lob.best_bid() is None
