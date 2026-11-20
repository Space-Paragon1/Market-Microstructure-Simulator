"""Closed-loop background market participants.

Together these produce a market with genuine price discovery:

* :class:`FundamentalValue` — a latent "true" price (Gaussian random walk with
  optional Poisson jumps). No book participant observes it except informed traders.
* :class:`NoiseTrader` — a zero-intelligence population in the spirit of
  Smith, Farmer, Gillemot & Krishnamurthy (2003): Poisson limit-order arrivals
  placed relative to the *current* opposite touch, Poisson market orders, and
  proportional random cancellation. It supplies most of the liquidity and
  random-walks the price on its own.
* :class:`InformedTrader` — trades with immediate-or-cancel orders when the
  book is mispriced relative to the fundamental by more than a threshold. It
  drags the book toward the fundamental and is the source of **adverse
  selection** for liquidity providers.

Every agent draws from private RNG streams seeded from a run seed, and arrival
streams are separated from state-dependent draws, so different strategies can
be compared against *identical* exogenous order arrivals (common random numbers).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from microbook import LimitOrderBook, Order, Side, TimeInForce
from microbook.types import Fill

from .strategy import Action, Strategy


def poisson(rng: random.Random, lam: float) -> int:
    """Poisson(lam) sample (Knuth's method; fine for the small rates used here)."""
    if lam <= 0:
        return 0
    if lam > 30:  # normal approximation keeps large rates O(1)
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    limit = math.exp(-lam)
    k, p = 0, rng.random()
    while p > limit:
        k += 1
        p *= rng.random()
    return k


class FundamentalValue:
    """Latent value process on the integer clock.

    ``V[t] = V[t-1] + N(0, vol^2) + J``, where ``J = +/-jump_size`` with
    probability ``jump_rate`` per step. The path is generated lazily but
    deterministically: the value at ``t`` does not depend on query order.
    """

    def __init__(
        self,
        start: float,
        vol_per_step: float,
        *,
        jump_rate: float = 0.0,
        jump_size: float = 0.0,
        seed: int = 0,
    ) -> None:
        if start <= 0 or vol_per_step < 0 or not 0 <= jump_rate <= 1:
            raise ValueError("invalid fundamental parameters")
        self.vol = vol_per_step
        self.jump_rate = jump_rate
        self.jump_size = jump_size
        self._rng = random.Random(f"{seed}/fundamental")
        self._path = [float(start)]

    def at(self, t: int) -> float:
        if t < 0:
            raise ValueError("t must be >= 0")
        path, rng = self._path, self._rng
        while len(path) <= t:
            v = path[-1] + rng.gauss(0.0, self.vol)
            if self.jump_rate and rng.random() < self.jump_rate:
                v += self.jump_size if rng.random() < 0.5 else -self.jump_size
            path.append(max(v, 1e-9))
        return path[t]


@dataclass
class NoiseConfig:
    limit_rate: float = 2.0  # expected limit orders per time step
    market_rate: float = 0.15  # expected market orders per time step
    cancel_prob: float = 0.02  # per-step cancellation probability of each resting order
    min_qty: int = 1
    max_qty: int = 10
    # limit price = opposite touch -/+ (1 + Geometric) ticks, with this mean distance
    mean_offset_ticks: float = 4.0
    # initial book: this many levels per side around the fundamental
    initial_levels: int = 10
    initial_qty: int = 15


class NoiseTrader(Strategy):
    """Zero-intelligence liquidity population (see module docstring).

    Its orders represent many anonymous traders, so each carries a unique
    ``owner`` and the population never triggers self-trade prevention against
    itself. ``portfolio`` holds the *aggregate* position of the population.
    """

    def __init__(
        self,
        cfg: NoiseConfig,
        fundamental: FundamentalValue,
        *,
        name: str = "noise",
        seed: int = 0,
    ) -> None:
        super().__init__(name)
        self.cfg = cfg
        self.fundamental = fundamental
        self._flow = random.Random(f"{seed}/{name}/flow")  # state-independent draws
        self._cancel = random.Random(f"{seed}/{name}/cancel")  # state-dependent draws
        self._live: list[str] = []  # resting (or in-flight) order ids
        self._live_pos: dict[str, int] = {}
        self._last_mid: float | None = None
        p = 1.0 / max(cfg.mean_offset_ticks, 1.0)  # geometric success prob, mean 1/p
        self._log_q = math.log(1.0 - p) if p < 1.0 else None

    # --- helpers ------------------------------------------------------------

    def _offset(self) -> int:
        """Ticks behind the opposite touch: 1 + Geometric, mean ~ ``mean_offset_ticks``."""
        if self._log_q is None:
            return 1
        u = 1.0 - self._flow.random()  # (0, 1]
        return 1 + int(math.log(u) / self._log_q)

    def _limit(self, oid: str, side: Side, px: float, qty: int, now: int) -> Action:
        self._live_pos[oid] = len(self._live)
        self._live.append(oid)
        return Action.submit(Order.limit(oid, side, px, qty, ts=now, owner=f"{self.name}/{oid}"), now)

    def _drop(self, oid: str) -> None:
        i = self._live_pos.pop(oid)
        last = self._live.pop()
        if last != oid:
            self._live[i] = last
            self._live_pos[last] = i

    # --- hooks ----------------------------------------------------------------

    def on_start(self, now: int, lob: LimitOrderBook) -> list[Action]:
        v = self.fundamental.at(now)
        center = lob.to_ticks(lob.round_price(v, Side.BUY))
        actions: list[Action] = []
        for k in range(1, self.cfg.initial_levels + 1):
            actions.append(
                self._limit(self.next_order_id("b"), Side.BUY, lob.price_of(center - k + 1), self.cfg.initial_qty, now)
            )
            actions.append(
                self._limit(self.next_order_id("a"), Side.SELL, lob.price_of(center + k), self.cfg.initial_qty, now)
            )
        actions.append(Action.timer(now + 1))
        return actions

    def on_timer(self, now: int, lob: LimitOrderBook) -> list[Action]:
        cfg, rng = self.cfg, self._flow
        actions: list[Action] = []

        bb, ba = lob.best_bid(), lob.best_ask()
        mid = lob.midprice()
        if mid is not None:
            self._last_mid = mid
        ref = self._last_mid if self._last_mid is not None else self.fundamental.at(now)
        ref_t = lob.to_ticks(lob.round_price(ref, Side.BUY))
        bb_t = lob.to_ticks(bb) if bb is not None else ref_t
        ba_t = lob.to_ticks(ba) if ba is not None else ref_t + 1

        for _ in range(poisson(rng, cfg.limit_rate)):
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            qty = rng.randint(cfg.min_qty, cfg.max_qty)
            off = self._offset()
            tick = ba_t - off if side is Side.BUY else bb_t + off
            if tick >= 1:
                actions.append(self._limit(self.next_order_id(), side, lob.price_of(tick), qty, now))

        for _ in range(poisson(rng, cfg.market_rate)):
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            qty = rng.randint(cfg.min_qty, cfg.max_qty)
            oid = self.next_order_id("m")
            actions.append(Action.submit(Order.market(oid, side, qty, ts=now, owner=f"{self.name}/{oid}"), now))

        # Each live order is cancelled independently with prob cancel_prob. An
        # order picked here may still be in flight; the cancel is sent anyway
        # and arrives after it (FIFO), so nothing is ever orphaned.
        n_cancel = min(len(self._live), poisson(self._cancel, cfg.cancel_prob * len(self._live)))
        for _ in range(n_cancel):
            oid = self._live[int(self._cancel.random() * len(self._live))]
            self._drop(oid)
            actions.append(Action.cancel(oid, now))

        actions.append(Action.timer(now + 1))
        return actions

    def on_fill(self, now: int, fill: Fill, side: Side, is_maker: bool, lob: LimitOrderBook) -> list[Action]:
        oid = fill.maker_order_id
        if is_maker and oid in self._live_pos and oid not in lob:
            self._drop(oid)  # fully filled
        return []


@dataclass
class InformedConfig:
    rate: float = 0.4  # expected arrivals per time step
    qty: int = 5
    threshold_ticks: float = 1.0  # minimum edge vs the fundamental before trading


class InformedTrader(Strategy):
    """Trades on the gap between the book and the fundamental value.

    On each arrival it buys up to ``qty`` with an IOC limit at
    ``V - threshold`` if the best ask is below that, or sells at
    ``V + threshold`` if the best bid is above it. It never pays more than the
    fundamental minus its required edge.
    """

    def __init__(
        self,
        cfg: InformedConfig,
        fundamental: FundamentalValue,
        *,
        name: str = "informed",
        seed: int = 0,
    ) -> None:
        super().__init__(name)
        self.cfg = cfg
        self.fundamental = fundamental
        self._rng = random.Random(f"{seed}/{name}/flow")

    def on_start(self, now: int, lob: LimitOrderBook) -> list[Action]:
        return [Action.timer(now + 1)]

    def on_timer(self, now: int, lob: LimitOrderBook) -> list[Action]:
        actions = [Action.timer(now + 1)]
        if poisson(self._rng, self.cfg.rate) == 0:
            return actions
        v = self.fundamental.at(now)
        edge = self.cfg.threshold_ticks * lob.tick_size
        bb, ba = lob.best_bid(), lob.best_ask()
        if ba is not None and ba < v - edge:
            px = lob.round_price(v - edge, Side.BUY)
            o = Order.limit(self.next_order_id(), Side.BUY, px, self.cfg.qty, ts=now, tif=TimeInForce.IOC)
            actions.append(Action.submit(o, now))
        elif bb is not None and bb > v + edge:
            px = lob.round_price(v + edge, Side.SELL)
            o = Order.limit(self.next_order_id(), Side.SELL, px, self.cfg.qty, ts=now, tif=TimeInForce.IOC)
            actions.append(Action.submit(o, now))
        return actions
