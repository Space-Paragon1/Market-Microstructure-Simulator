from microbook import LimitOrderBook, Order, Side

def test_modify_reduce_keeps_priority():
    lob = LimitOrderBook()
    lob.place_limit(Order("b1", Side.BUY, 99.0, 10, ts=1))
    lob.place_limit(Order("b2", Side.BUY, 99.0, 10, ts=2))

    # reduce qty on b1 should NOT lose priority
    assert lob.modify("b1", new_qty=5, ts=99) is True

    fills = lob.place_limit(Order("s1", Side.SELL, 99.0, 6, ts=3))
    assert [(f.maker_order_id, f.qty) for f in fills] == [("b1", 5), ("b2", 1)]

def test_modify_increase_loses_priority():
    lob = LimitOrderBook()
    lob.place_limit(Order("b1", Side.BUY, 99.0, 5, ts=1))
    lob.place_limit(Order("b2", Side.BUY, 99.0, 5, ts=2))

    # increase b1 loses priority -> b2 should fill first
    assert lob.modify("b1", new_qty=10, ts=3) is True

    fills = lob.place_limit(Order("s1", Side.SELL, 99.0, 6, ts=4))
    assert [(f.maker_order_id, f.qty) for f in fills] == [("b2", 5), ("b1", 1)]
