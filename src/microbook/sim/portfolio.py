from __future__ import annotations

from dataclasses import dataclass, field

from microbook import LimitOrderBook, Side
from microbook.types import Fill


@dataclass
class Portfolio:
    """Single-asset position and PnL accounting (average-cost method).

    Fees are charged per share and are *negative* for rebates, e.g. a typical
    maker/taker schedule is ``maker_fee=-0.002, taker_fee=0.003``.

    Identity (checked by the test-suite for arbitrary fill sequences)::

        total_pnl(mark) == cash - initial_cash + position * mark
    """

    cash: float = 0.0
    position: int = 0
    avg_cost: float = 0.0  # average entry price of the open position
    realized_pnl: float = 0.0  # gross of fees
    fees_paid: float = 0.0  # net of rebates
    maker_fee: float = 0.0
    taker_fee: float = 0.0
    volume: int = 0  # shares traded
    initial_cash: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.initial_cash = self.cash

    def on_fill(self, fill: Fill, my_order_id: str | None, my_side: Side, *, is_maker: bool = False) -> None:
        """Apply a fill of one of *my* orders. ``my_side`` is my order's side;
        ``is_maker`` selects the maker or taker fee."""
        qty, px = fill.qty, fill.price
        signed = qty if my_side is Side.BUY else -qty
        pos = self.position
        new_pos = pos + signed

        self.cash -= signed * px
        if pos == 0 or (pos > 0) == (signed > 0):
            # opening or adding: blend the average cost
            self.avg_cost = (self.avg_cost * abs(pos) + px * qty) / abs(new_pos)
        else:
            # reducing, closing or flipping: realize PnL on the closed quantity
            closed = min(qty, abs(pos))
            self.realized_pnl += closed * (px - self.avg_cost) * (1 if pos > 0 else -1)
            if new_pos == 0:
                self.avg_cost = 0.0
            elif (new_pos > 0) != (pos > 0):
                self.avg_cost = px  # flipped: the residual was opened at px

        self.position = new_pos
        fee = (self.maker_fee if is_maker else self.taker_fee) * qty
        self.cash -= fee
        self.fees_paid += fee
        self.volume += qty

    def unrealized_pnl(self, mark: float) -> float:
        return (mark - self.avg_cost) * self.position

    def total_pnl(self, mark: float) -> float:
        """Realized + unrealized PnL, net of fees, marking the position at ``mark``."""
        return self.realized_pnl + self.unrealized_pnl(mark) - self.fees_paid

    def mark_to_market(self, lob: LimitOrderBook) -> float | None:
        """Total PnL marked at the book's midprice (``None`` if the book is one-sided)."""
        mid = lob.midprice()
        return None if mid is None else self.total_pnl(mid)
