from __future__ import annotations

from typing import NamedTuple

from microbook import Side
from microbook.types import Fill


class FillRecord(NamedTuple):
    """One fill from a strategy's point of view (kept for post-trade analysis such as markouts)."""

    time: int | None
    side: Side
    price: float
    qty: int
    is_maker: bool
    mid: float | None  # midprice just before the trade


class ExecutionMetrics:
    """Per-strategy execution quality metrics.

    * VWAP of buys and sells, blended average price, total notional
    * slippage vs **arrival** (decision-time) midprice, per direction
    * spread capture per share on maker fills (vs the pre-trade mid)
    * fill ratio (filled / submitted) and participation rate (share of market volume)

    Each ratio is computed only over the fills that carry the information it
    needs (e.g. slippage only over fills of orders with a known arrival mid),
    so missing data never dilutes a metric.
    """

    def __init__(self) -> None:
        self.market_volume: int = 0
        self.fills: list[FillRecord] = []
        self.orders_submitted: int = 0
        self.orders_rejected: int = 0

        self._buy_qty = 0
        self._sell_qty = 0
        self._buy_notional = 0.0
        self._sell_notional = 0.0
        self._maker_qty = 0
        self._taker_qty = 0

        # spread capture: maker fills with a known mid
        self._maker_spread_total = 0.0
        self._maker_spread_qty = 0

        # order_id -> (side, submitted_qty, arrival_mid | None)
        self._order_registry: dict[str, tuple[Side, int, float | None]] = {}
        self._submitted_qty = 0

        # slippage: fills of orders with a known arrival mid, per direction
        self._buy_arr_qty = 0
        self._buy_arr_notional = 0.0  # sum(qty * fill price)
        self._buy_arr_mid_notional = 0.0  # sum(qty * arrival mid)
        self._sell_arr_qty = 0
        self._sell_arr_notional = 0.0
        self._sell_arr_mid_notional = 0.0

    # ------------------------------------------------------------------ #
    # Recording API                                                       #
    # ------------------------------------------------------------------ #

    def record_order(self, order_id: str, side: Side, qty: int, arrival_mid: float | None) -> None:
        """Register an order at decision time. ``arrival_mid`` may be ``None``
        (no two-sided market), in which case its fills are excluded from slippage."""
        self._order_registry[order_id] = (side, qty, arrival_mid)
        self._submitted_qty += qty
        self.orders_submitted += 1

    def record_market_volume(self, fills: list[Fill] | tuple[Fill, ...]) -> None:
        """Accumulate market-wide traded volume (called for every fill in the market)."""
        for f in fills:
            self.market_volume += f.qty

    def on_fill(
        self,
        fill: Fill,
        my_side: Side,
        is_maker: bool,
        mid: float | None = None,
        time: int | None = None,
    ) -> None:
        """Record a fill of one of this strategy's orders.

        ``mid`` is the midprice immediately before the trade (used for spread
        capture); ``time`` is the simulation time (used for markouts).
        """
        qty, px = fill.qty, fill.price
        is_buy = my_side is Side.BUY
        if is_buy:
            self._buy_qty += qty
            self._buy_notional += px * qty
        else:
            self._sell_qty += qty
            self._sell_notional += px * qty

        if is_maker:
            self._maker_qty += qty
            if mid is not None:
                self._maker_spread_total += ((mid - px) if is_buy else (px - mid)) * qty
                self._maker_spread_qty += qty
        else:
            self._taker_qty += qty

        rec = self._order_registry.get(fill.maker_order_id if is_maker else fill.taker_order_id)
        if rec is not None and rec[2] is not None:
            arrival = rec[2]
            if is_buy:
                self._buy_arr_qty += qty
                self._buy_arr_notional += px * qty
                self._buy_arr_mid_notional += arrival * qty
            else:
                self._sell_arr_qty += qty
                self._sell_arr_notional += px * qty
                self._sell_arr_mid_notional += arrival * qty

        self.fills.append(FillRecord(time, my_side, px, qty, is_maker, mid))

    # ------------------------------------------------------------------ #
    # Computed metrics                                                    #
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

    @property
    def maker_qty(self) -> int:
        return self._maker_qty

    @property
    def taker_qty(self) -> int:
        return self._taker_qty

    def buy_vwap(self) -> float | None:
        return self._buy_notional / self._buy_qty if self._buy_qty else None

    def sell_vwap(self) -> float | None:
        return self._sell_notional / self._sell_qty if self._sell_qty else None

    def avg_price(self) -> float | None:
        total = self.filled_qty
        return (self._buy_notional + self._sell_notional) / total if total else None

    def total_notional(self) -> float:
        return self._buy_notional + self._sell_notional

    def buy_slippage(self) -> float | None:
        """Buy VWAP minus qty-weighted arrival mid. Positive = paid up (cost)."""
        q = self._buy_arr_qty
        return (self._buy_arr_notional - self._buy_arr_mid_notional) / q if q else None

    def sell_slippage(self) -> float | None:
        """Qty-weighted arrival mid minus sell VWAP. Positive = sold below arrival (cost)."""
        q = self._sell_arr_qty
        return (self._sell_arr_mid_notional - self._sell_arr_notional) / q if q else None

    def spread_capture_per_share(self) -> float | None:
        """Average distance from the pre-trade mid earned per maker share (positive = earned)."""
        q = self._maker_spread_qty
        return self._maker_spread_total / q if q else None

    def fill_ratio(self) -> float | None:
        """Filled quantity / submitted quantity."""
        return self.filled_qty / self._submitted_qty if self._submitted_qty else None

    def participation_rate(self) -> float | None:
        """This strategy's share of total market volume."""
        return self.filled_qty / self.market_volume if self.market_volume else None

    def summary(self) -> dict[str, float | int | None]:
        """All metrics as a flat dict."""
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
