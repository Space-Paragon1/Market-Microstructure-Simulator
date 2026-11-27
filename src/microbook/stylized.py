"""Stylized facts of market data: the statistical signatures every realistic
market model must reproduce (Cont 2001; Bouchaud, Bonart, Donier & Gould 2018).

All functions are pure and work on plain sequences, so they apply equally to
simulator output (``SimResult.mid_path`` / ``SimResult.fills``) and to replayed
historical data.

=========================  ====================================  ==================
Fact                       Statistic here                        Typical equities
=========================  ====================================  ==================
Fat tails                  excess kurtosis, Hill tail index      kurtosis >> 0, alpha ~ 3
Aggregational Gaussianity  kurtosis falls as the horizon grows   -
No return autocorrelation  ACF(r) at lag >= 1                    ~ 0 (negative at tick scale)
Volatility clustering      ACF(|r|) at lags 1..100               > 0, slow decay
Long memory of order sign  ACF(sign) ~ lag^-gamma                gamma ~ 0.5
Square-root impact         I(Q) ~ sigma * (Q / V)^delta          delta ~ 0.5
=========================  ====================================  ==================
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from .types import Fill

__all__ = [
    "MetaorderRecord",
    "StylizedFacts",
    "acf",
    "aggressor_signs",
    "bin_impacts",
    "excess_kurtosis",
    "hill_tail_index",
    "impact_curve",
    "log_lags",
    "metaorder_impacts",
    "power_law_fit",
    "sample_returns",
    "stylized_facts",
]


def _time_of(entry: tuple[int, float]) -> int:
    return entry[0]


def _value_at(path: Sequence[tuple[int, float]], t: int) -> float | None:
    i = bisect_right(path, t, key=_time_of) - 1
    if i < 0:
        return None
    v = path[i][1]
    return None if math.isnan(v) else v


def sample_returns(mid_path: Sequence[tuple[int, float]], dt: int, *, start: int = 0, end: int) -> list[float]:
    """Log returns of a step-function mid path sampled every ``dt`` time units.

    Intervals where the mid is undefined at either end are skipped.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    out: list[float] = []
    prev = _value_at(mid_path, start)
    for t in range(start + dt, end + 1, dt):
        cur = _value_at(mid_path, t)
        if prev is not None and cur is not None and prev > 0 and cur > 0:
            out.append(math.log(cur / prev))
        prev = cur
    return out


def acf(xs: Sequence[float], lags: Iterable[int]) -> list[float]:
    """Sample autocorrelation at each lag (biased estimator, like most references)."""
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two observations")
    mu = statistics.fmean(xs)
    d = [x - mu for x in xs]
    var = math.fsum(x * x for x in d)
    if var == 0:
        return [math.nan for _ in lags]
    out = []
    for lag in lags:
        if not 0 < lag < n:
            raise ValueError(f"lag {lag} out of range for {n} observations")
        out.append(math.fsum(d[i] * d[i + lag] for i in range(n - lag)) / var)
    return out


def log_lags(max_lag: int, n: int = 20) -> list[int]:
    """About ``n`` distinct integer lags, logarithmically spaced over ``[1, max_lag]``."""
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    pts = {round(math.exp(i * math.log(max_lag) / max(n - 1, 1))) for i in range(n)}
    return sorted(p for p in pts if p >= 1)


def excess_kurtosis(xs: Sequence[float]) -> float:
    """Fourth standardized moment minus 3 (0 for a Gaussian)."""
    if len(xs) < 4:
        return math.nan
    mu = statistics.fmean(xs)
    m2 = statistics.fmean((x - mu) ** 2 for x in xs)
    if m2 == 0:
        return math.nan
    m4 = statistics.fmean((x - mu) ** 4 for x in xs)
    return m4 / (m2 * m2) - 3.0


def hill_tail_index(xs: Sequence[float], tail: float = 0.05) -> float:
    """Hill estimator of the tail exponent ``alpha`` of ``|x|``: ``P(|X| > x) ~ x^-alpha``.

    Uses the largest ``tail`` fraction of observations. Smaller ``alpha`` means
    fatter tails (a Gaussian has none: the estimate grows with sample size).
    """
    if not 0 < tail < 1:
        raise ValueError("tail must be in (0, 1)")
    a = sorted((abs(x) for x in xs if x != 0), reverse=True)
    k = int(len(a) * tail)
    if k < 2:
        return math.nan
    ref = a[k]
    return k / math.fsum(math.log(a[i] / ref) for i in range(k))


def aggressor_signs(fills: Iterable[Fill]) -> list[int]:
    """One sign per aggressive order (+1 buy, -1 sell), in time order.

    Consecutive fills of the same taker order (a sweep through several makers)
    count as one trade, as in the empirical literature.
    """
    out: list[int] = []
    last_taker: str | None = None
    for f in fills:
        if f.aggressor_side is None or f.taker_order_id == last_taker:
            continue
        last_taker = f.taker_order_id
        out.append(f.aggressor_side.sign)
    return out


def power_law_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Least-squares fit of ``y = c * x^-gamma`` on log-log axes, ignoring ``y <= 0``.

    Returns ``(gamma, r_squared)``.
    """
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys, strict=True) if x > 0 and y > 0 and not math.isnan(y)]
    if len(pts) < 3:
        return math.nan, math.nan
    lx = [p[0] for p in pts]
    ly = [p[1] for p in pts]
    mx, my = statistics.fmean(lx), statistics.fmean(ly)
    sxx = math.fsum((x - mx) ** 2 for x in lx)
    sxy = math.fsum((x - mx) * (y - my) for x, y in pts)
    slope = sxy / sxx
    ss_tot = math.fsum((y - my) ** 2 for y in ly)
    ss_res = math.fsum((y - (my + slope * (x - mx))) ** 2 for x, y in pts)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return -slope, r2


class MetaorderRecord(NamedTuple):
    """A large order executed as a sequence of child orders."""

    side: int  # +1 buy, -1 sell
    qty: int
    start: int
    end: int


def metaorder_impacts(
    metaorders: Iterable[MetaorderRecord],
    mid_path: Sequence[tuple[int, float]],
    *,
    daily_volume: float,
    sigma: float,
) -> list[tuple[float, float]]:
    """``(Q / daily_volume, impact / sigma)`` for each metaorder with a defined mid.

    The impact of one metaorder is ``side * log(mid(end) / mid(start))``: the
    price move in its direction during execution. ``sigma`` is the return
    volatility over the same reference "day" as ``daily_volume``.
    """
    pts: list[tuple[float, float]] = []
    for m in metaorders:
        a, b = _value_at(mid_path, m.start), _value_at(mid_path, m.end)
        if a is None or b is None or m.qty <= 0:
            continue
        pts.append((m.qty / daily_volume, m.side * math.log(b / a) / sigma))
    return pts


def bin_impacts(
    points: Sequence[tuple[float, float]], n_bins: int = 8
) -> tuple[list[tuple[float, float, int]], float, float]:
    """Group ``(size, impact)`` points into log-spaced size bins and fit ``impact ~ size^delta``.

    Returns ``(bins, delta, r_squared)`` with bins ``(mean size, mean impact, count)``.
    Bins whose mean impact is not positive cannot enter the log-log fit.
    """
    if len(points) < n_bins:
        return [], math.nan, math.nan
    lo, hi = min(p[0] for p in points), max(p[0] for p in points)
    if lo == hi:
        return [(lo, statistics.fmean(p[1] for p in points), len(points))], math.nan, math.nan
    edges = [math.exp(math.log(lo) + i * (math.log(hi) - math.log(lo)) / n_bins) for i in range(n_bins + 1)]
    groups: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for q, i_ in points:
        # clamp both ends: exp(log(x)) can land a hair either side of x
        groups[min(max(bisect_right(edges, q) - 1, 0), n_bins - 1)].append((q, i_))
    bins = [(statistics.fmean(p[0] for p in g), statistics.fmean(p[1] for p in g), len(g)) for g in groups if g]
    gamma, r2 = power_law_fit([b[0] for b in bins], [b[1] for b in bins])
    return bins, -gamma, r2


def impact_curve(
    metaorders: Iterable[MetaorderRecord],
    mid_path: Sequence[tuple[int, float]],
    *,
    daily_volume: float,
    sigma: float,
    n_bins: int = 8,
) -> tuple[list[tuple[float, float, int]], float, float]:
    """Average metaorder impact against size, and the fitted impact exponent.

    Combines :func:`metaorder_impacts` and :func:`bin_impacts`. The square-root
    law predicts ``impact / sigma ~ Y * (Q / V)^0.5`` with ``Y`` of order one.
    """
    pts = metaorder_impacts(metaorders, mid_path, daily_volume=daily_volume, sigma=sigma)
    return bin_impacts(pts, n_bins)


@dataclass(frozen=True)
class StylizedFacts:
    """Summary statistics of one run (or one replayed day)."""

    dt_short: int
    dt_long: int
    n_returns: int
    kurtosis_short: float
    kurtosis_long: float
    hill_alpha: float
    return_acf_1: float  # lag-1 autocorrelation of dt_short returns
    abs_return_acf: dict[int, float]  # volatility clustering, lags in units of dt_short
    n_trades: int
    sign_acf: dict[int, float]  # lags in trades
    sign_gamma: float  # fitted ACF(sign) ~ lag^-gamma
    sign_gamma_r2: float

    def rows(self) -> dict[str, float]:
        """Flat ``{statistic: value}`` dict, convenient for tables."""
        out = {
            f"kurtosis (dt={self.dt_short})": self.kurtosis_short,
            f"kurtosis (dt={self.dt_long})": self.kurtosis_long,
            "Hill tail alpha": self.hill_alpha,
            "ACF r lag 1": self.return_acf_1,
        }
        out.update({f"ACF |r| lag {k}": v for k, v in self.abs_return_acf.items()})
        out.update({f"ACF sign lag {k}": v for k, v in self.sign_acf.items()})
        out["sign ACF gamma"] = self.sign_gamma
        return out


def stylized_facts(
    mid_path: Sequence[tuple[int, float]],
    fills: Sequence[Fill],
    *,
    end: int,
    start: int = 0,
    dt_short: int = 10,
    dt_long: int = 1_000,
    vol_lags: Sequence[int] = (1, 10, 100),
    sign_lags: Sequence[int] = (1, 10, 100, 1_000),
) -> StylizedFacts:
    """Compute the standard stylized facts from a mid path and a fill stream."""
    r = sample_returns(mid_path, dt_short, start=start, end=end)
    r_long = sample_returns(mid_path, dt_long, start=start, end=end)
    absr = [abs(x) for x in r]
    max_vol_lag = len(r) - 1
    vlags = [k for k in vol_lags if k < max_vol_lag]

    signs = aggressor_signs(fills)
    slags = [k for k in sign_lags if k < len(signs) - 1]
    fit_lags = log_lags(min(max(slags, default=1), max(len(signs) // 10, 1)), 16)
    sign_vals = acf(signs, fit_lags) if len(signs) > 20 else []
    gamma, r2 = power_law_fit(fit_lags, sign_vals) if sign_vals else (math.nan, math.nan)

    return StylizedFacts(
        dt_short=dt_short,
        dt_long=dt_long,
        n_returns=len(r),
        kurtosis_short=excess_kurtosis(r),
        kurtosis_long=excess_kurtosis(r_long),
        hill_alpha=hill_tail_index(r),
        return_acf_1=acf(r, [1])[0] if len(r) > 2 else math.nan,
        abs_return_acf=dict(zip(vlags, acf(absr, vlags), strict=True)) if vlags else {},
        n_trades=len(signs),
        sign_acf=dict(zip(slags, acf(signs, slags), strict=True)) if slags else {},
        sign_gamma=gamma,
        sign_gamma_r2=r2,
    )
