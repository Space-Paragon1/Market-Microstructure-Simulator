from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from microbook.types import Fill
from microbook import Side


class ExecutionMetrics:
    """
    Per-strategy execution quality metrics.

    Tracks:
    - Total market volume (for participation rate)
    - Maker vs taker fills
    - VWAP of buys and sells
    - VWAP slippage vs arrival (decision-time) midprice
    - Spread capture per share (maker perspective)
    - Fill ratio (filled / submitted)
    - Participation rate (strategy volume / market volume)
    - Average execution price and total notional
    """

    def __init__(self) -> None:
        self.market_volume: int = 0

        # directional fill counts
        self._buy_qty: int = 0
        self._sell_qty: int = 0

        # directional fill notional (for VWAP)
        self._buy_notional: float = 0.0
        self._sell_notional: float = 0.0

        # maker / taker breakdown
        self._maker_qty: int = 0
        self._taker_qty: int = 0

        # spread capture (maker fills only):
        #   maker buy  filled at P, mid M -> captured (M - P) per share
        #   maker sell filled at P, mid M -> captured (P - M) per share
        self._maker_spread_total: float = 0.0

        # for arrival-mid VWAP slippage:
        # order_id -> (side, submitted_qty, arrival_mid)
        self._order_registry: Dict[str, Tuple[Side, int, float]] = {}

        # qty-weighted arrival-mid accumulator per direction
        self._buy_arrival_mid_notional: float = 0.0   # sum(qty * arrival_mid) for buy fills
        self._sell_arrival_mid_notional: float = 0.0

        # fill-ratio denominator: track submitted qty from registered orders
        self._submitted_qty: int = 0

    # ------------------------------------------------------------------ #
    # Recording API                                                        #
    # ------------------------------------------------------------------ #

    def record_order(self, order_id: str, side: Side, qty: int, arrival_mid: float) -> None:
        """
        Called when a strategy decides to submit an order.
        Captures the decision-time midprice (arrival mid) for slippage measurement.
        """
        self._order_registry[order_id] = (side, qty, arrival_mid)
        self._submitted_qty += qty

    def record_market_volume(self, fills: List[Fill]) -> None:
        """Accumulate total market-wide traded volume (called for every fill event)."""
        for f in fills:
            self.market_volume += f.qty

    def on_fill(
        self,
        fill: Fill,
        my_side: Side,
        is_maker: bool,
        mid: Optional[float] = None,
    ) -> None:
        """
        Called for each fill that involves this strategy's order.

        Parameters
        ----------
        fill     : the Fill record
        my_side  : the side of THIS strategy's order (BUY or SELL)
        is_maker : True if the strategy's order was the passive (resting) side
        mid      : midprice at the time the fill was generated (for spread capture)
        """
        qty = fill.qty
        px = fill.price

        # directional
        if my_side == Side.BUY:
            self._buy_qty += qty
            self._buy_notional += px * qty
        else:
            self._sell_qty += qty
            self._sell_notional += px * qty

        # maker / taker
        if is_maker:
            self._maker_qty += qty
            # spread capture
            if mid is not None:
                if my_side == Side.BUY:
                    self._maker_spread_total += (mid - px) * qty
                else:
                    self._maker_spread_total += (px - mid) * qty
        else:
            self._taker_qty += qty

        # VWAP slippage: look up arrival mid for this order
        oid = fill.maker_order_id if is_maker else fill.taker_order_id
        rec = self._order_registry.get(oid)
        if rec is not None:
            _, _, arrival_mid = rec
            if my_side == Side.BUY:
                self._buy_arrival_mid_notional += arrival_mid * qty
            else:
                self._sell_arrival_mid_notional += arrival_mid * qty

    # ------------------------------------------------------------------ #
    # Computed metrics                                                     #
    # ------------------------------------------------------------------ #

    @property
    def filled_qty(self) -> int:
        return self._buy_qty + self._sell_qty

    @property
    def buy_qty(self) -> int:
        return self._buy_qty

    @property
    def sell_qty(self) -> int:
        return self._sell_qty

    def buy_vwap(self) -> Optional[float]:
        """Volume-weighted average price of all buy fills."""
        return self._buy_notional / self._buy_qty if self._buy_qty > 0 else None

    def sell_vwap(self) -> Optional[float]:
        """Volume-weighted average price of all sell fills."""
        return self._sell_notional / self._sell_qty if self._sell_qty > 0 else None

    def avg_price(self) -> Optional[float]:
        """Blended average execution price across all fills."""
        total = self._buy_qty + self._sell_qty
        if total == 0:
            return None
        return (self._buy_notional + self._sell_notional) / total

    def total_notional(self) -> float:
        """Total traded notional (sum of |qty * price| across all fills)."""
        return self._buy_notional + self._sell_notional

    def buy_slippage(self) -> Optional[float]:
        """
        VWAP slippage for buys vs arrival midprice.
        Positive means we paid above the decision-time mid (cost).
        Only defined when buys were attributed to registered orders.
        """
        if self._buy_qty == 0 or self._buy_arrival_mid_notional == 0.0:
            return None
        arrival_vwap = self._buy_arrival_mid_notional / self._buy_qty
        vwap = self._buy_notional / self._buy_qty
        return vwap - arrival_vwap

    def sell_slippage(self) -> Optional[float]:
        """
        VWAP slippage for sells vs arrival midprice.
        Positive means we received below the decision-time mid (cost).
        """
        if self._sell_qty == 0 or self._sell_arrival_mid_notional == 0.0:
            return None
        arrival_vwap = self._sell_arrival_mid_notional / self._sell_qty
        vwap = self._sell_notional / self._sell_qty
        return arrival_vwap - vwap

    def spread_capture_per_share(self) -> Optional[float]:
        """
        Average spread captured per share as a maker.
        Positive = captured spread (good for MM).
        """
        return self._maker_spread_total / self._maker_qty if self._maker_qty > 0 else None

    def fill_ratio(self) -> Optional[float]:
        """
        Ratio of filled qty to submitted qty.
        1.0 = completely filled, 0.0 = nothing filled.
        """
        if self._submitted_qty == 0:
            return None
        return self.filled_qty / self._submitted_qty

    def participation_rate(self) -> Optional[float]:
        """Strategy's share of total market volume."""
        if self.market_volume == 0:
            return None
        return self.filled_qty / self.market_volume

    def summary(self) -> dict:
        """Return all metrics as a flat dict."""
        return {
            "filled_qty": self.filled_qty,
            "buy_qty": self._buy_qty,
            "sell_qty": self._sell_qty,
            "maker_qty": self._maker_qty,
            "taker_qty": self._taker_qty,
            "total_notional": self.total_notional(),
            "avg_price": self.avg_price(),
            "buy_vwap": self.buy_vwap(),
            "sell_vwap": self.sell_vwap(),
            "buy_slippage_vs_arrival": self.buy_slippage(),
            "sell_slippage_vs_arrival": self.sell_slippage(),
            "spread_capture_per_share": self.spread_capture_per_share(),
            "fill_ratio": self.fill_ratio(),
            "market_volume": self.market_volume,
            "participation_rate": self.participation_rate(),
        }
