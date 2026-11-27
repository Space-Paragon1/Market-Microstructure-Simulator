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
* :class:`MetaorderTrader` — uninformed institutions splitting large orders
  (Pareto-distributed sizes) into child market orders. Order splitting is the
  mechanism behind the long memory of trade signs (Lillo, Mike & Farmer 2005)
  and produces the metaorders used to measure price impact.
* :class:`HawkesProcess` — optional self-exciting arrivals for the noise
  traders' market orders, which cluster activity and hence volatility.

Every agent draws from private RNG streams seeded from a run seed, and arrival
streams are separated from state-dependent draws, so different strategies can
be compared against *identical* exogenous order arrivals (common random numbers).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from microbook import LimitOrderBook, Order, Side, TimeInForce
from microbook.stylized import MetaorderRecord
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

    ``V[t] = V[t-1] + sigma[t] * Z + J``, where ``Z ~ N(0, 1)`` and
    ``J = +/-jump_size`` with probability ``jump_rate`` per step.

    With ``vol_of_vol > 0`` the volatility is stochastic: ``log sigma`` follows
    a mean-reverting AR(1) around ``log vol_per_step`` with per-step
    persistence ``vol_persistence``, normalized so the *average variance*
    stays ``vol_per_step ** 2``. This produces volatility clustering and fat
    tails at every horizon, as news arrival does in real markets.

    The path is generated lazily but deterministically: the value at ``t``
    does not depend on query order.
    """

    def __init__(
        self,
        start: float,
        vol_per_step: float,
        *,
        jump_rate: float = 0.0,
        jump_size: float = 0.0,
        vol_of_vol: float = 0.0,
        vol_persistence: float = 0.999,
        seed: int = 0,
    ) -> None:
        if start <= 0 or vol_per_step < 0 or not 0 <= jump_rate <= 1:
            raise ValueError("invalid fundamental parameters")
        if vol_of_vol < 0 or not 0 <= vol_persistence < 1:
            raise ValueError("invalid stochastic-volatility parameters")
        self.vol = vol_per_step
        self.jump_rate = jump_rate
        self.jump_size = jump_size
        self.vol_of_vol = vol_of_vol
        self.vol_persistence = vol_persistence
        self._rng = random.Random(f"{seed}/fundamental")
        self._path = [float(start)]
        # stationary sd of log-vol; shift the mean so E[sigma^2] = vol^2
        self._h_sd = vol_of_vol / math.sqrt(1 - vol_persistence**2) if vol_of_vol else 0.0
        self._h_mean = math.log(vol_per_step) - self._h_sd**2 if vol_per_step > 0 else 0.0
        self._h = self._h_mean + self._h_sd * self._rng.gauss(0.0, 1.0) if vol_of_vol else self._h_mean

    def at(self, t: int) -> float:
        if t < 0:
            raise ValueError("t must be >= 0")
        path, rng = self._path, self._rng
        while len(path) <= t:
            if self.vol_of_vol:
                self._h = (
                    self._h_mean
                    + self.vol_persistence * (self._h - self._h_mean)
                    + self.vol_of_vol * rng.gauss(0.0, 1.0)
                )
                sigma = math.exp(self._h)
            else:
                sigma = self.vol
            v = path[-1] + rng.gauss(0.0, sigma)
            if self.jump_rate and rng.random() < self.jump_rate:
                v += self.jump_size if rng.random() < 0.5 else -self.jump_size
            path.append(max(v, 1e-9))
        return path[t]


@dataclass(frozen=True)
class HawkesConfig:
    """Self-excitation of noise market orders.

    ``branching`` is the expected number of extra orders each order triggers
    (must be < 1). The kernel is an equal-weight mixture of exponentials with
    the given ``timescales`` (in steps), a standard approximation of the slowly
    decaying kernels estimated on real data. The *mean* arrival rate is kept
    equal to ``NoiseConfig.market_rate``, so turning excitation on changes how
    activity clusters, not how much there is.
    """

    branching: float = 0.8
    timescales: tuple[float, ...] = (10.0, 100.0, 1_000.0)


class HawkesProcess:
    """Discrete-time Hawkes process: counts per step ``N_t ~ Poisson(lambda_t)`` with
    ``lambda_t = mu + sum_k sum_{s<t} a_k * d_k^(t-s) * N_s``.

    Each kernel component ``k`` has decay ``d_k = exp(-1/tau_k)`` and weight
    ``a_k = (branching / K) * (1 - d_k) / d_k`` so that the total expected
    offspring per event equals ``branching``.
    """

    def __init__(self, mean_rate: float, cfg: HawkesConfig, rng: random.Random) -> None:
        if not 0 <= cfg.branching < 1:
            raise ValueError("branching ratio must be in [0, 1)")
        if not cfg.timescales or min(cfg.timescales) <= 0:
            raise ValueError("timescales must be positive")
        self.mu = mean_rate * (1 - cfg.branching)
        self._rng = rng
        self._d = [math.exp(-1.0 / tau) for tau in cfg.timescales]
        w = cfg.branching / len(cfg.timescales)
        self._a = [w * (1 - d) / d for d in self._d]
        self._e = [0.0] * len(self._d)

    @property
    def intensity(self) -> float:
        return self.mu + sum(self._e)

    def step(self) -> int:
        n = poisson(self._rng, self.intensity)
        self._e = [(e + a * n) * d for e, a, d in zip(self._e, self._a, self._d, strict=True)]
        return n


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
    # self-exciting market orders (None = Poisson arrivals at market_rate)
    market_hawkes: HawkesConfig | None = None


class _Pool:
    """Live order ids with O(1) add, remove and uniform random pick."""

    __slots__ = ("ids", "pos")

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.pos: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, oid: object) -> bool:
        return oid in self.pos

    def add(self, oid: str) -> None:
        self.pos[oid] = len(self.ids)
        self.ids.append(oid)

    def drop(self, oid: str) -> None:
        i = self.pos.pop(oid)
        last = self.ids.pop()
        if last != oid:
            self.ids[i] = last
            self.pos[last] = i

    def cancel_some(self, rng: random.Random, prob: float, now: int, out: list[Action]) -> None:
        """Cancel each live order independently with probability ``prob``.

        An order picked here may still be in flight; the cancel is sent anyway
        and arrives after it (FIFO), so nothing is ever orphaned.
        """
        n = min(len(self.ids), poisson(rng, prob * len(self.ids)))
        for _ in range(n):
            oid = self.ids[int(rng.random() * len(self.ids))]
            self.drop(oid)
            out.append(Action.cancel(oid, now))


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
        self._hawkes = (
            HawkesProcess(cfg.market_rate, cfg.market_hawkes, random.Random(f"{seed}/{name}/hawkes"))
            if cfg.market_hawkes is not None
            else None
        )
        self._live = _Pool()  # resting (or in-flight) order ids
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
        self._live.add(oid)
        return Action.submit(Order.limit(oid, side, px, qty, ts=now, owner=f"{self.name}/{oid}"), now)

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

        n_market = self._hawkes.step() if self._hawkes is not None else poisson(rng, cfg.market_rate)
        for _ in range(n_market):
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            qty = rng.randint(cfg.min_qty, cfg.max_qty)
            oid = self.next_order_id("m")
            actions.append(Action.submit(Order.market(oid, side, qty, ts=now, owner=f"{self.name}/{oid}"), now))

        self._live.cancel_some(self._cancel, cfg.cancel_prob, now, actions)

        actions.append(Action.timer(now + 1))
        return actions

    def on_fill(self, now: int, fill: Fill, side: Side, is_maker: bool, lob: LimitOrderBook) -> list[Action]:
        oid = fill.maker_order_id
        if is_maker and oid in self._live and oid not in lob:
            self._live.drop(oid)  # fully filled
        return []


@dataclass
class InformedConfig:
    rate: float = 0.4  # expected arrivals per time step
    qty: int = 5
    threshold_ticks: float = 1.0  # minimum edge vs the fundamental before trading
    # extra shares per tick of mispricing beyond the threshold (0 = fixed size);
    # arbitrage capital that scales with the opportunity closes large gaps fast
    qty_per_tick: float = 0.0
    max_qty: int = 1_000


class InformedTrader(Strategy):
    """Trades on the gap between the book and the fundamental value.

    On each arrival it buys with an IOC limit at ``V - threshold`` if the
    best ask is below that, or sells at ``V + threshold`` if the best bid is
    above it. It never pays more than the fundamental minus its required edge.
    The size is ``qty`` plus ``qty_per_tick`` per tick of mispricing beyond the
    threshold (capped at ``max_qty``).
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
        cfg = self.cfg
        v = self.fundamental.at(now)
        edge = cfg.threshold_ticks * lob.tick_size
        bb, ba = lob.best_bid(), lob.best_ask()
        if ba is not None and ba < v - edge:
            side, px = Side.BUY, lob.round_price(v - edge, Side.BUY)
            gap = lob.to_ticks(px) - lob.to_ticks(ba)  # ticks of edge beyond the threshold
        elif bb is not None and bb > v + edge:
            side, px = Side.SELL, lob.round_price(v + edge, Side.SELL)
            gap = lob.to_ticks(bb) - lob.to_ticks(px)
        else:
            return actions
        qty = min(cfg.max_qty, cfg.qty + int(cfg.qty_per_tick * gap))
        o = Order.limit(self.next_order_id(), side, px, qty, ts=now, tif=TimeInForce.IOC)
        actions.append(Action.submit(o, now))
        return actions


@dataclass
class MetaorderConfig:
    """Uninformed institutions splitting large orders into child market orders.

    Metaorders arrive as a Poisson process. Total sizes follow a Pareto law
    ``P(Q > q) ~ q^-size_alpha`` truncated to ``[min_size, max_size]``; with
    ``1 < size_alpha < 2`` the trade-sign autocorrelation decays as
    ``lag^-(size_alpha - 1)`` (Lillo, Mike & Farmer 2005). Each active metaorder
    sends a child market order of ``child_qty`` shares with probability
    ``child_rate`` per step until it is done.
    """

    rate: float = 0.01
    size_alpha: float = 1.5
    min_size: int = 10
    max_size: int = 10_000
    child_qty: int = 5
    child_rate: float = 0.1


@dataclass
class _Meta:
    side: Side
    qty: int
    remaining: int
    start: int
    owner: str


class MetaorderTrader(Strategy):
    """Background population executing metaorders (see :class:`MetaorderConfig`).

    Completed metaorders are logged in ``completed`` as
    :class:`~microbook.stylized.MetaorderRecord` (side, size, start, end), the
    data needed to measure price impact. All draws are state-independent, so
    the flow is identical across strategy variants (common random numbers).
    """

    def __init__(self, cfg: MetaorderConfig, *, name: str = "meta", seed: int = 0) -> None:
        super().__init__(name)
        if not (cfg.size_alpha > 0 and 0 < cfg.min_size <= cfg.max_size and cfg.child_qty > 0):
            raise ValueError("invalid metaorder parameters")
        if not 0 < cfg.child_rate <= 1:
            raise ValueError("child_rate must be in (0, 1]")
        self.cfg = cfg
        self._rng = random.Random(f"{seed}/{name}/flow")
        self.active: list[_Meta] = []
        self.completed: list[MetaorderRecord] = []
        self._n = 0

    def draw_size(self) -> int:
        """Inverse-CDF sample of the truncated Pareto size distribution."""
        c = self.cfg
        tail = (c.min_size / c.max_size) ** c.size_alpha
        u = self._rng.random()
        return min(c.max_size, int(c.min_size * (1 - u * (1 - tail)) ** (-1 / c.size_alpha)))

    def on_start(self, now: int, lob: LimitOrderBook) -> list[Action]:
        return [Action.timer(now + 1)]

    def on_timer(self, now: int, lob: LimitOrderBook) -> list[Action]:
        rng, cfg = self._rng, self.cfg
        for _ in range(poisson(rng, cfg.rate)):
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            q = self.draw_size()
            self._n += 1
            self.active.append(_Meta(side, q, q, now, f"{self.name}/{self._n}"))

        actions: list[Action] = []
        still: list[_Meta] = []
        for m in self.active:
            if m.start < now and rng.random() < cfg.child_rate:
                qty = min(m.remaining, cfg.child_qty)
                m.remaining -= qty
                oid = self.next_order_id()
                actions.append(Action.submit(Order.market(oid, m.side, qty, ts=now, owner=m.owner), now))
                if m.remaining == 0:
                    self.completed.append(MetaorderRecord(m.side.sign, m.qty, m.start, now))
                    continue
            still.append(m)
        self.active = still
        actions.append(Action.timer(now + 1))
        return actions
