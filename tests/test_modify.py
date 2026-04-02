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


def test_modify_to_zero_cancels():
    lob = LimitOrderBook()
    lob.place_limit(Order("b1", Side.BUY, 99.0, 10, ts=1))
    lob.place_limit(Order("b2", Side.BUY, 99.0, 5, ts=2))

    # modify to qty=0 should cancel b1 and return True
    assert lob.modify("b1", new_qty=0, ts=99) is True

    # only b2 remains
    fills = lob.place_limit(Order("s1", Side.SELL, 99.0, 10, ts=3))
    assert [(f.maker_order_id, f.qty) for f in fills] == [("b2", 5)]


def test_modify_twice_keeps_priority():
    lob = LimitOrderBook()
    lob.place_limit(Order("b1", Side.BUY, 99.0, 20, ts=1))
    lob.place_limit(Order("b2", Side.BUY, 99.0, 10, ts=2))

    # reduce b1 twice without losing priority
    assert lob.modify("b1", new_qty=15, ts=3) is True
    assert lob.modify("b1", new_qty=8, ts=4) is True

    fills = lob.place_limit(Order("s1", Side.SELL, 99.0, 10, ts=5))
    assert [(f.maker_order_id, f.qty) for f in fills] == [("b1", 8), ("b2", 2)]


def test_modify_then_cancel():
    lob = LimitOrderBook()
    lob.place_limit(Order("b1", Side.BUY, 99.0, 10, ts=1))

    assert lob.modify("b1", new_qty=5, ts=2) is True
    assert lob.cancel("b1") is True

    # book should be empty
    fills = lob.place_limit(Order("s1", Side.SELL, 99.0, 1, ts=3))
    assert fills == []
