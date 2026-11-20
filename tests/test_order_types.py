"""Order types, time-in-force, self-trade prevention and the tick grid."""

from __future__ import annotations

import pytest

from microbook import LimitOrderBook, Order, OrderStatus, OrderType, Side, STPMode, TimeInForce


def book(stp: STPMode = STPMode.NONE) -> LimitOrderBook:
    lob = LimitOrderBook(stp=stp)
    lob.place_limit(Order.limit("a1", Side.SELL, 101.0, 5, owner="alice"))
    lob.place_limit(Order.limit("a2", Side.SELL, 102.0, 5, owner="bob"))
    lob.place_limit(Order.limit("b1", Side.BUY, 99.0, 5, owner="alice"))
    return lob


class TestTimeInForce:
    def test_ioc_fills_what_it_can_and_cancels_rest(self):
        lob = book()
        r = lob.submit(Order.limit("x", Side.BUY, 101.0, 8, tif=TimeInForce.IOC))
        assert r.status is OrderStatus.CANCELED
        assert r.filled_qty == 5 and r.canceled_qty == 3 and r.resting_qty == 0
        assert "x" not in lob and lob.best_bid() == 99.0

    def test_fok_fills_completely_across_levels(self):
        lob = book()
        r = lob.submit(Order.limit("x", Side.BUY, 102.0, 10, tif=TimeInForce.FOK))
        assert r.status is OrderStatus.FILLED and r.filled_qty == 10

    def test_fok_kill_has_no_side_effects(self):
        lob = book()
        before = lob.depth()
        r = lob.submit(Order.limit("x", Side.BUY, 102.0, 11, tif=TimeInForce.FOK))
        assert r.status is OrderStatus.CANCELED and r.fills == () and r.canceled_qty == 11
        assert lob.depth() == before

    def test_gtc_remainder_rests(self):
        lob = book()
        r = lob.submit(Order.limit("x", Side.BUY, 101.0, 8))
        assert r.status is OrderStatus.PARTIALLY_FILLED and r.resting_qty == 3
        assert lob.best_bid() == 101.0


class TestMarketAndPostOnly:
    def test_market_order_on_empty_book_is_canceled(self):
        r = LimitOrderBook().submit(Order.market("m", Side.BUY, 5))
        assert r.status is OrderStatus.CANCELED and r.canceled_qty == 5

    def test_market_order_sweeps_and_never_rests(self):
        lob = book()
        r = lob.submit(Order.market("m", Side.BUY, 20))
        assert [(f.price, f.qty) for f in r.fills] == [(101.0, 5), (102.0, 5)]
        assert r.canceled_qty == 10 and lob.best_ask() is None and lob.best_bid() == 99.0

    def test_market_order_requires_no_price(self):
        o = Order.market("m", Side.SELL, 3)
        assert o.price is None and o.order_type is OrderType.MARKET and o.is_market

    def test_post_only_rejected_if_it_would_cross(self):
        lob = book()
        r = lob.submit(Order.limit("p", Side.BUY, 101.0, 1, post_only=True))
        assert r.status is OrderStatus.REJECTED and "post-only" in (r.reason or "")
        assert "p" not in lob

    def test_post_only_rests_if_passive(self):
        lob = book()
        assert lob.submit(Order.limit("p", Side.BUY, 100.0, 1, post_only=True)).status is OrderStatus.NEW

    def test_post_only_modify_that_would_cross_is_rejected_and_order_kept(self):
        lob = book()
        lob.submit(Order.limit("p", Side.BUY, 100.0, 1, post_only=True))
        assert lob.modify("p", new_price=101.0).status is OrderStatus.REJECTED
        assert lob.get_order("p") is not None and lob.best_bid() == 100.0

    @pytest.mark.parametrize("bad", [0, -1])
    def test_invalid_orders_raise(self, bad):
        with pytest.raises(ValueError):
            Order("x", Side.BUY, 100.0, bad)
        with pytest.raises(ValueError):
            Order("x", Side.BUY, float(bad), 1)
        with pytest.raises(ValueError):
            Order("x", Side.BUY, None, 1)


class TestSelfTradePrevention:
    def test_none_allows_self_trade(self):
        lob = book(STPMode.NONE)
        fills = lob.place_limit(Order.limit("x", Side.BUY, 101.0, 5, owner="alice"))
        assert [f.maker_order_id for f in fills] == ["a1"]

    def test_cancel_resting_removes_own_order_and_keeps_matching(self):
        lob = book(STPMode.CANCEL_RESTING)
        r = lob.submit(Order.limit("x", Side.BUY, 102.0, 5, owner="alice"))
        assert r.stp_canceled == ("a1",)
        assert [(f.maker_order_id, f.qty) for f in r.fills] == [("a2", 5)]
        assert "a1" not in lob
        lob.check_invariants()

    def test_cancel_incoming_stops_at_own_order(self):
        lob = book(STPMode.CANCEL_INCOMING)
        r = lob.submit(Order.limit("x", Side.BUY, 102.0, 5, owner="alice"))
        assert r.status is OrderStatus.CANCELED and r.fills == () and r.canceled_qty == 5
        assert "a1" in lob

    def test_fok_respects_stp(self):
        lob = book(STPMode.CANCEL_RESTING)
        # only 5 shares are available to alice once her own a1 is excluded
        r = lob.submit(Order.limit("x", Side.BUY, 102.0, 6, owner="alice", tif=TimeInForce.FOK))
        assert r.status is OrderStatus.CANCELED and "a1" in lob

    def test_orders_without_owner_never_trigger_stp(self):
        lob = LimitOrderBook(stp=STPMode.CANCEL_INCOMING)
        lob.place_limit(Order("a", Side.SELL, 101.0, 5))
        assert len(lob.place_limit(Order("b", Side.BUY, 101.0, 5))) == 1


class TestTickGrid:
    def test_off_grid_price_rejected(self):
        lob = LimitOrderBook(tick_size=0.05)
        assert lob.submit(Order("x", Side.BUY, 100.02, 1)).status is OrderStatus.REJECTED

    def test_float_noise_lands_on_same_level(self):
        lob = LimitOrderBook(tick_size=0.1)
        lob.place_limit(Order("x", Side.BUY, 0.1 + 0.2, 1))
        lob.place_limit(Order("y", Side.BUY, 0.3, 1))
        assert lob.depth()["bids"] == [(0.3, 2)]
        assert lob.get_order("x").price == 0.3  # normalized onto the grid

    def test_round_price_is_passive(self):
        lob = LimitOrderBook(tick_size=0.01)
        assert lob.round_price(100.019, Side.BUY) == 100.01
        assert lob.round_price(100.011, Side.SELL) == 100.02
        assert lob.round_price(100.01, Side.SELL) == 100.01

    @pytest.mark.parametrize("tick", [0, -0.01, float("nan"), float("inf")])
    def test_invalid_tick_size(self, tick):
        with pytest.raises(ValueError):
            LimitOrderBook(tick_size=tick)


class TestMarketData:
    def test_queue_ahead_and_volume_at(self):
        lob = LimitOrderBook()
        lob.place_limit(Order("b1", Side.BUY, 99.0, 3))
        lob.place_limit(Order("b2", Side.BUY, 99.0, 4))
        assert lob.queue_ahead("b1") == 0 and lob.queue_ahead("b2") == 3
        assert lob.queue_ahead("nope") is None
        assert lob.volume_at(Side.BUY, 99.0) == 7 and lob.volume_at(Side.SELL, 99.0) == 0

    def test_microprice_leans_toward_thin_side(self):
        lob = LimitOrderBook()
        lob.place_limit(Order("b", Side.BUY, 99.0, 90))
        lob.place_limit(Order("a", Side.SELL, 101.0, 10))
        assert lob.microprice() == pytest.approx(100.8)
        assert lob.spread() == 2.0 and len(lob) == 2
