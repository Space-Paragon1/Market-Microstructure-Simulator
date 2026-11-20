from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given

from microbook import Side
from microbook.sim import Portfolio
from microbook.types import Fill


def fill(px: float, qty: int) -> Fill:
    return Fill("t", "m", px, qty)


def test_short_roundtrip_realizes_profit():
    p = Portfolio()
    p.on_fill(fill(101.0, 10), "x", Side.SELL)
    assert p.position == -10 and p.avg_cost == 101.0
    p.on_fill(fill(100.0, 10), "x", Side.BUY)
    assert p.position == 0 and p.realized_pnl == pytest.approx(10.0) and p.avg_cost == 0.0


def test_flip_long_to_short_resets_cost_basis():
    p = Portfolio()
    p.on_fill(fill(100.0, 5), "x", Side.BUY)
    p.on_fill(fill(102.0, 8), "x", Side.SELL)  # close 5 (+10), open short 3 @ 102
    assert p.position == -3 and p.avg_cost == 102.0 and p.realized_pnl == pytest.approx(10.0)


def test_partial_reduce_keeps_cost_basis():
    p = Portfolio()
    p.on_fill(fill(100.0, 4), "x", Side.BUY)
    p.on_fill(fill(104.0, 6), "x", Side.BUY)  # avg 102.4
    p.on_fill(fill(103.0, 5), "x", Side.SELL)
    assert p.position == 5 and p.avg_cost == pytest.approx(102.4)
    assert p.realized_pnl == pytest.approx(5 * 0.6)


def test_maker_rebate_and_taker_fee():
    p = Portfolio(maker_fee=-0.002, taker_fee=0.003)
    p.on_fill(fill(100.0, 100), "x", Side.BUY, is_maker=True)
    p.on_fill(fill(100.0, 100), "x", Side.SELL, is_maker=False)
    assert p.fees_paid == pytest.approx(0.1)
    assert p.total_pnl(100.0) == pytest.approx(-0.1)
    assert p.volume == 200


fills_strategy = st.lists(
    st.tuples(
        st.sampled_from(list(Side)),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=9_000, max_value=11_000),  # price in cents
        st.booleans(),
    ),
    max_size=60,
)


@given(fills_strategy, st.integers(min_value=9_000, max_value=11_000), st.floats(-1000, 1000))
def test_pnl_identity_holds_for_any_fill_sequence(seq, mark_cents, start_cash):
    """realized + unrealized - fees == change in cash + position * mark."""
    p = Portfolio(cash=start_cash, maker_fee=-0.001, taker_fee=0.002)
    for side, qty, cents, maker in seq:
        p.on_fill(fill(cents / 100, qty), "x", side, is_maker=maker)
    mark = mark_cents / 100
    assert p.total_pnl(mark) == pytest.approx(p.cash - start_cash + p.position * mark, abs=1e-6)
    assert p.position == sum(q if s is Side.BUY else -q for s, q, _, _ in seq)
    if p.position == 0:
        assert p.avg_cost == 0.0
