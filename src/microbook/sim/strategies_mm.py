from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from microbook import LimitOrderBook, Order, Side
from microbook.sim.strategy import Strategy, Action
from microbook.sim.events import EventType
from microbook.sim.metrics import imbalance


@dataclass
class AdaptiveMMConfig:
    tick_size: float = 1.0
    base_half_spread_ticks: int = 1
    size: int = 5
    tick_interval: int = 10

    # inventory control
    inv_target: int = 0
    inv_limit: int = 25
    inv_k: float = 0.08  # how aggressively we skew quotes by inventory

    # volatility control (uses recent mid changes)
    vol_window: int = 30
    vol_k: float = 3.0   # widen spread as volatility increases

    # imbalance leaning
    imb_k: float = 2.0   # shift quotes based on L2 imbalance


class AdaptiveMarketMaker(Strategy):
    """
    Adaptive market maker:
    - Dynamic spread widens with volatility
    - Inventory-aware skew (pushes toward flattening inventory)
    - Optional order-book imbalance leaning
    """

    def __init__(self, cfg: AdaptiveMMConfig, name: str = "amm") -> None:
        super().__init__(name=name)
        self.cfg = cfg

        self._last_quote_t = -10**9
        self._bid_id = f"{name}_bid"
        self._ask_id = f"{name}_ask"
        self.my_order_ids.update([self._bid_id, self._ask_id])

        self._mid_hist: List[float] = []

    def _record_mid(self, lob: LimitOrderBook) -> Optional[float]:
        m = lob.midprice()
        if m is None:
            return None
        self._mid_hist.append(float(m))
        if len(self._mid_hist) > self.cfg.vol_window:
            self._mid_hist.pop(0)
        return float(m)

    def _vol_proxy(self) -> float:
        # mean absolute mid change (simple, stable)
        if len(self._mid_hist) < 2:
            return 0.0
        diffs = [abs(self._mid_hist[i] - self._mid_hist[i-1]) for i in range(1, len(self._mid_hist))]
        return sum(diffs) / len(diffs)

    def on_tick(self, now: int, lob: LimitOrderBook) -> List[Action]:
        if now - self._last_quote_t < self.cfg.tick_interval:
            return []

        mid = self._record_mid(lob)
        if mid is None:
            return []

        # --- dynamic half spread in ticks ---
        vol = self._vol_proxy()
        half_spread = self.cfg.base_half_spread_ticks + int(self.cfg.vol_k * vol / self.cfg.tick_size)

        # --- inventory skew (in ticks) ---
        inv = self.portfolio.position
        inv_err = inv - self.cfg.inv_target
        inv_skew = int(self.cfg.inv_k * inv_err)  # positive means we’re long -> skew down

        # clamp skew so we don’t go crazy
        inv_skew = max(-self.cfg.base_half_spread_ticks - 5, min(self.cfg.base_half_spread_ticks + 5, inv_skew))

        # --- imbalance leaning ---
        im = imbalance(lob, levels=3)
        imb_skew = 0
        if im is not None:
            imb_skew = int(self.cfg.imb_k * im)

        # Combine skews:
        # long inventory => want to encourage sells: lower bid, lower ask (shift down)
        # short inventory => shift up
        total_skew = inv_skew + imb_skew

        bid_px = mid - (half_spread + total_skew) * self.cfg.tick_size
        ask_px = mid + (half_spread + total_skew) * self.cfg.tick_size

        # Inventory guard: if too long, stop quoting bid; if too short, stop quoting ask
        quote_bid = inv < self.cfg.inv_limit
        quote_ask = inv > -self.cfg.inv_limit

        actions: List[Action] = []

        # Cancel existing quotes each refresh
        actions.append(Action(now, EventType.CANCEL, {"order_id": self._bid_id}))
        actions.append(Action(now, EventType.CANCEL, {"order_id": self._ask_id}))

        if quote_bid:
            actions.append(Action(now, EventType.SUBMIT, {
                "order": Order(self._bid_id, Side.BUY, float(bid_px), self.cfg.size, ts=self._next_ts(now))
            }))

        if quote_ask:
            actions.append(Action(now, EventType.SUBMIT, {
                "order": Order(self._ask_id, Side.SELL, float(ask_px), self.cfg.size, ts=self._next_ts(now))
            }))

        self._last_quote_t = now
        return actions
