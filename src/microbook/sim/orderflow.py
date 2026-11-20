"""Pre-generated (open-loop) order flow.

``PoissonOrderFlow`` produces a fixed stream of orders around a *constant*
reference price. Because it never reacts to the book, prices cannot move; it
is useful for controlled matching/attribution tests. For a market with price
discovery and adverse selection use the closed-loop agents in
:mod:`microbook.sim.agents` (see :func:`microbook.sim.scenarios.build_market`).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from microbook import Order, Side


@dataclass
class FlowConfig:
    seed: int = 7
    # average orders per 100 time units (at most one order per unit)
    intensity_per_100: float = 20.0
    # order sizes
    min_qty: int = 1
    max_qty: int = 10
    # price placement around a reference mid
    tick: float = 1.0
    max_ticks_away: int = 5
    # probability an order is a market order
    p_market: float = 0.05


class PoissonOrderFlow:
    """Bernoulli-thinned arrivals on the integer clock (a discrete-time
    approximation of a Poisson process: each unit holds at most one order)."""

    def __init__(self, cfg: FlowConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._oid = 0

    def _next_id(self) -> str:
        self._oid += 1
        return f"o{self._oid:06d}"

    def iter_orders(self, *, start: int, end: int, ref_mid: float) -> Iterator[tuple[int, Order]]:
        p = min(1.0, self.cfg.intensity_per_100 / 100.0)
        rng = self.rng
        for t in range(start, end + 1):
            if rng.random() > p:
                continue
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            qty = rng.randint(self.cfg.min_qty, self.cfg.max_qty)
            if rng.random() < self.cfg.p_market:
                yield t, Order.market(self._next_id(), side, qty, ts=t)
                continue
            ticks = rng.randint(1, self.cfg.max_ticks_away)
            offset = ticks * self.cfg.tick
            price = ref_mid - offset if side is Side.BUY else ref_mid + offset
            yield t, Order.limit(self._next_id(), side, float(price), qty, ts=t)
