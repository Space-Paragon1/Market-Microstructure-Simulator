from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

from microbook.types import Fill
from microbook import Side, LimitOrderBook


@dataclass
class Portfolio:
    """
    Single-asset portfolio:
    - cash in quote currency
    - position in base units
    - tracks realized pnl via average cost
    """
    cash: float = 0.0
    position: int = 0
    avg_cost: float = 0.0     # average cost of current position
    realized_pnl: float = 0.0

    fee_per_share: float = 0.0  # optional simple fee model

    def on_fill(self, fill: Fill, my_order_id: str, my_side: Side) -> None:
        """
        Update portfolio if THIS fill involves my order.
        my_side is the side of MY order (maker or taker).
        """
        qty = fill.qty
        px = fill.price
        fee = self.fee_per_share * qty

        if my_side == Side.BUY:
            # Buying increases position, reduces cash
            self.cash -= px * qty
            self.cash -= fee

            # Update avg cost
            new_pos = self.position + qty
            if self.position == 0:
                self.avg_cost = px
            elif self.position > 0:
                # add to long
                self.avg_cost = (self.avg_cost * self.position + px * qty) / new_pos
            else:
                # covering short -> realize pnl on covered amount
                cover = min(qty, -self.position)
                self.realized_pnl += (self.avg_cost - px) * cover  # short profit if buy lower
                # if flips to long, set avg_cost for remaining
                if new_pos > 0:
                    self.avg_cost = px

            self.position = new_pos

        else:  # SELL
            self.cash += px * qty
            self.cash -= fee

            new_pos = self.position - qty
            if self.position == 0:
                self.avg_cost = px
            elif self.position < 0:
                # add to short
                self.avg_cost = (self.avg_cost * (-self.position) + px * qty) / (-new_pos)
            else:
                # selling long -> realize pnl on sold amount
                sold = min(qty, self.position)
                self.realized_pnl += (px - self.avg_cost) * sold
                if new_pos < 0:
                    self.avg_cost = px  # flipped to short

            self.position = new_pos

    def mark_to_market(self, lob: LimitOrderBook) -> Optional[float]:
        mid = lob.midprice()
        if mid is None:
            return None
        # Unrealized pnl:
        # long: (mid - avg_cost) * pos
        # short: (avg_cost - mid) * (-pos)
        if self.position > 0:
            unreal = (mid - self.avg_cost) * self.position
        elif self.position < 0:
            unreal = (self.avg_cost - mid) * (-self.position)
        else:
            unreal = 0.0
        return self.realized_pnl + unreal
