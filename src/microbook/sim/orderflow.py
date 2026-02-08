from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Iterator, Optional

from microbook import Order, Side


@dataclass
class FlowConfig:
    seed: int = 7
    # average orders per 100 time units (discrete time)
    intensity_per_100: float = 20.0
    # order sizes
    min_qty: int = 1
    max_qty: int = 10
    # price placement around a reference mid
    tick: float = 1.0
    max_ticks_away: int = 5
    # probability an order is market
    p_market: float = 0.05


class PoissonOrderFlow:
    """
    Discrete-time Poisson-like generator:
    - For each time step, with probability proportional to intensity,
      generate 0 or 1 order (simple & stable for v1).
    """

    def __init__(self, cfg: FlowConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._oid = 0

    def _next_id(self) -> str:
        self._oid += 1
        return f"o{self._oid:06d}"

    def iter_orders(self, *, start: int, end: int, ref_mid: float) -> Iterator[tuple[int, Order]]:
        p = min(1.0, self.cfg.intensity_per_100 / 100.0)
        for t in range(start, end + 1):
            if self.rng.random() > p:
                continue

            side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
            qty = self.rng.randint(self.cfg.min_qty, self.cfg.max_qty)

            is_market = self.rng.random() < self.cfg.p_market
            if is_market:
                # dummy price; simulator uses _is_market flag
                price = 1.0
            else:
                ticks = self.rng.randint(1, self.cfg.max_ticks_away)
                if side == Side.BUY:
                    price = ref_mid - ticks * self.cfg.tick
                else:
                    price = ref_mid + ticks * self.cfg.tick

            o = Order(order_id=self._next_id(), side=side, price=float(price), qty=int(qty), ts=t)
            if is_market:
                setattr(o, "_is_market", True)
            yield t, o
