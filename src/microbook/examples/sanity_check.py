from microbook import LimitOrderBook, Order, Side

lob = LimitOrderBook()
lob.place_limit(Order("a1", Side.SELL, 100.0, 5, ts=1))
lob.place_limit(Order("b1", Side.BUY, 99.0, 5, ts=2))

print(lob.top_of_book())
print(lob.depth(3))

lob.modify("b1", new_price=100.0, ts=3)   # loses priority (cancel+reinsert)
fills = lob.place_market(Order("m1", Side.BUY, 1.0, 10, ts=4))
print(fills)
