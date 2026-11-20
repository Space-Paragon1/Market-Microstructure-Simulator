"""Market-making strategies beyond the symmetric baseline."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from itertools import pairwise

from microbook import LimitOrderBook
from microbook.sim.metrics import imbalance
from microbook.sim.strategy import TwoSidedQuoter

Quotes = tuple[float | None, float | None, int, int]


@dataclass
class AdaptiveMMConfig:
    tick_size: float | None = None  # quoting unit; None = the book's tick
    base_half_spread_ticks: float = 1.0
    size: int = 5
    tick_interval: int = 10
    latency: int = 0

    # inventory control
    inv_target: int = 0
    inv_limit: int = 25  # stop quoting the side that would grow |inventory| beyond this
    inv_k: float = 0.08  # ticks of quote shift per share of inventory error
    max_skew_ticks: float = 6.0

    # volatility control: widen by vol_k * (mean |mid change| per refresh)
    vol_window: int = 30
    vol_k: float = 3.0

    # order-book imbalance leaning (ticks of shift at full imbalance)
    imb_k: float = 2.0


class AdaptiveMarketMaker(TwoSidedQuoter):
    """Heuristic market maker with three controls:

    * **volatility-aware spread** — the half-spread widens with the mean
      absolute mid change over the last ``vol_window`` refreshes;
    * **inventory skew** — both quotes shift *against* the position
      (long -> lower, to sell; short -> higher, to buy), plus a hard limit;
    * **imbalance lean** — both quotes shift *toward* the heavier side of the
      top-3 book (bid-heavy -> higher), anticipating the next price move.

    Skews shift the quote *centre*; they never change the spread width, so the
    bid stays strictly below the ask.
    """

    def __init__(self, cfg: AdaptiveMMConfig | None = None, name: str = "amm") -> None:
        cfg = cfg or AdaptiveMMConfig()
        super().__init__(name, tick_interval=cfg.tick_interval, latency=cfg.latency)
        self.cfg = cfg
        self._mid_hist: deque[float] = deque(maxlen=max(2, cfg.vol_window))

    def _vol_proxy(self) -> float:
        h = self._mid_hist
        if len(h) < 2:
            return 0.0
        return sum(abs(b - a) for a, b in pairwise(h)) / (len(h) - 1)

    def compute_quotes(self, now: int, lob: LimitOrderBook) -> Quotes | None:
        mid = lob.midprice()
        if mid is None:
            return None
        cfg = self.cfg
        self._mid_hist.append(mid)
        q = cfg.tick_size or lob.tick_size

        half = (cfg.base_half_spread_ticks + cfg.vol_k * self._vol_proxy() / q) * q

        inv = self.portfolio.position
        im = imbalance(lob, levels=3) or 0.0
        shift_down = cfg.inv_k * (inv - cfg.inv_target) - cfg.imb_k * im
        shift_down = max(-cfg.max_skew_ticks, min(cfg.max_skew_ticks, shift_down))
        center = mid - shift_down * q

        bid = center - half if inv < cfg.inv_limit else None
        ask = center + half if inv > -cfg.inv_limit else None
        return bid, ask, cfg.size, cfg.size


@dataclass
class AvellanedaStoikovConfig:
    gamma: float = 0.1  # risk aversion
    k: float = 100.0  # fill-intensity decay: lambda(delta) = A * exp(-k * delta), delta in price units
    horizon: int = 10_000  # terminal time T
    # If set, use this constant time-to-horizon instead of T - t (the stationary
    # "rolling horizon" variant common in practice: the skew does not fade).
    rolling_tau: int | None = None
    sigma: float | None = None  # mid volatility per sqrt(time unit); None = estimate online
    vol_window: int = 50  # mid samples used by the online estimator
    size: int = 5
    tick_interval: int = 10
    inv_limit: int = 50
    min_half_spread_ticks: float = 0.5
    latency: int = 0


class AvellanedaStoikovMarketMaker(TwoSidedQuoter):
    """Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*.

    With inventory ``q``, mid ``s``, volatility ``sigma`` and time to horizon
    ``tau = T - t``, the optimal quotes are centred on the reservation price::

        r     = s - q * gamma * sigma^2 * tau
        delta = gamma * sigma^2 * tau + (2 / gamma) * ln(1 + gamma / k)    # total spread

    so inventory risk is managed by moving the centre (not by widening). In
    the finite-horizon form the skew fades as ``t -> T``; with
    ``rolling_tau`` set, ``tau`` is held constant instead.
    """

    def __init__(self, cfg: AvellanedaStoikovConfig | None = None, name: str = "as") -> None:
        cfg = cfg or AvellanedaStoikovConfig()
        if cfg.gamma <= 0 or cfg.k <= 0:
            raise ValueError("gamma and k must be positive")
        super().__init__(name, tick_interval=cfg.tick_interval, latency=cfg.latency)
        self.cfg = cfg
        self._samples: deque[tuple[int, float]] = deque(maxlen=max(2, cfg.vol_window))

    def sigma(self) -> float | None:
        """Configured sigma, or realized volatility per sqrt(time unit) of sampled mids."""
        if self.cfg.sigma is not None:
            return self.cfg.sigma
        s = self._samples
        if len(s) < 3:
            return None
        pts = list(s)
        sq = sum((b[1] - a[1]) ** 2 for a, b in pairwise(pts))
        dt = pts[-1][0] - pts[0][0]
        return math.sqrt(sq / dt) if dt > 0 else None

    def tau(self, now: int) -> int:
        if self.cfg.rolling_tau is not None:
            return self.cfg.rolling_tau
        return max(self.cfg.horizon - now, 0)

    def reservation_price(self, mid: float, inventory: int, now: int, sigma: float) -> float:
        return mid - inventory * self.cfg.gamma * sigma**2 * self.tau(now)

    def optimal_spread(self, now: int, sigma: float) -> float:
        g, k = self.cfg.gamma, self.cfg.k
        return g * sigma**2 * self.tau(now) + (2.0 / g) * math.log1p(g / k)

    def compute_quotes(self, now: int, lob: LimitOrderBook) -> Quotes | None:
        mid = lob.midprice()
        if mid is None:
            return None
        cfg = self.cfg
        self._samples.append((now, mid))
        sigma = self.sigma() or 0.0
        inv = self.portfolio.position

        r = self.reservation_price(mid, inv, now, sigma)
        half = max(self.optimal_spread(now, sigma) / 2.0, cfg.min_half_spread_ticks * lob.tick_size)
        bid = r - half if inv < cfg.inv_limit else None
        ask = r + half if inv > -cfg.inv_limit else None
        return bid, ask, cfg.size, cfg.size
