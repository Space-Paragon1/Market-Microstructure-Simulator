from __future__ import annotations

from dataclasses import dataclass
from microbook.types import Fill
from microbook import Side


@dataclass
class ExecutionMetrics:
    """
    Minimal execution metrics per strategy.
    - market_volume: total executed volume in the market (from all fills)
    - filled_qty: total qty executed by this strategy (both maker/taker fills that belong to it)
    - buy_qty / sell_qty: directional breakdown of this strategy's executed qty
    """
    market_volume: int = 0
    filled_qty: int = 0
    buy_qty: int = 0
    sell_qty: int = 0

    def record_market_volume(self, fills: list[Fill]) -> None:
        for f in fills:
            self.market_volume += f.qty

    def on_fill(self, fill: Fill, side: Side) -> None:
        self.filled_qty += fill.qty
        if side == Side.BUY:
            self.buy_qty += fill.qty
        else:
            self.sell_qty += fill.qty
