"""Post-trade analytics: book time series, markouts, PnL and inventory statistics."""

from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from microbook import LimitOrderBook

from .execution_metrics import FillRecord
from .metrics import imbalance, spread

if TYPE_CHECKING:
    from .simulator import MarketSimulator


@dataclass
class TimeSeries:
    """Book state sampled on every SNAPSHOT: midprice, spread and top-3 imbalance."""

    t: list[int] = field(default_factory=list)
    mid: list[float] = field(default_factory=list)
    spr: list[float] = field(default_factory=list)
    imb: list[float] = field(default_factory=list)

    def record(self, now: int, lob: LimitOrderBook) -> None:
        self.t.append(now)
        m = lob.midprice()
        self.mid.append(m if m is not None else math.nan)
        s = spread(lob)
        self.spr.append(s if s is not None else math.nan)
        im = imbalance(lob, levels=3)
        self.imb.append(im if im is not None else math.nan)

    def as_dict(self) -> dict[str, list[Any]]:
        return {"t": self.t, "mid": self.mid, "spread": self.spr, "imbalance": self.imb}


def _time_of(entry: tuple[int, float]) -> int:
    return entry[0]


def mid_at(mid_path: Sequence[tuple[int, float]], t: int) -> float | None:
    """Value of a step-function mid path (``SimResult.mid_path``) at the end of time ``t``."""
    i = bisect_right(mid_path, t, key=_time_of) - 1
    if i < 0:
        return None
    m = mid_path[i][1]
    return None if math.isnan(m) else m


def markouts(
    fills: Iterable[FillRecord],
    mid_path: Sequence[tuple[int, float]],
    horizons: Sequence[int] = (1, 10, 100),
    *,
    end_time: int | None = None,
) -> dict[int, float | None]:
    """Volume-weighted markout per share at each horizon, from the trader's side.

    For a fill at price ``p`` and time ``t`` the markout at horizon ``h`` is
    ``side * (mid(t + h) - p)``: positive means the trade was profitable when
    marked ``h`` time units later. For a market maker, the decay of markouts
    with horizon measures **adverse selection**: spread captured at ``h = 0``
    that is given back as the price moves against the fill.

    Fills whose horizon would extend past ``end_time`` are excluded.
    """
    out: dict[int, float | None] = {}
    fills = list(fills)
    for h in horizons:
        num = 0.0
        qty = 0
        for f in fills:
            if f.time is None or (end_time is not None and f.time + h > end_time):
                continue
            m = mid_at(mid_path, f.time + h)
            if m is None:
                continue
            num += f.side.sign * (m - f.price) * f.qty
            qty += f.qty
        out[h] = num / qty if qty else None
    return out


def max_drawdown(series: Sequence[float]) -> float:
    """Largest peak-to-trough decline (a non-negative number). NaNs are skipped."""
    peak = -math.inf
    mdd = 0.0
    for x in series:
        if math.isnan(x):
            continue
        peak = max(peak, x)
        mdd = max(mdd, peak - x)
    return mdd


def pnl_stats(pnl: Sequence[float]) -> dict[str, float]:
    """Summary of a cumulative PnL path sampled at regular intervals.

    ``sharpe`` is mean/stdev of per-sample PnL increments scaled by
    sqrt(number of increments): a dimensionless signal-to-noise ratio for the
    whole run (not annualized).
    """
    xs = [x for x in pnl if not math.isnan(x)]
    if not xs:
        return {"final": math.nan, "max_drawdown": math.nan, "pnl_vol": math.nan, "sharpe": math.nan}
    incs = [b - a for a, b in pairwise(xs)]
    vol = statistics.pstdev(incs) if len(incs) > 1 else 0.0
    mean = statistics.fmean(incs) if incs else 0.0
    sharpe = mean / vol * math.sqrt(len(incs)) if vol > 0 else math.nan
    return {"final": xs[-1], "max_drawdown": max_drawdown(xs), "pnl_vol": vol, "sharpe": sharpe}


def inventory_stats(inventory: Sequence[int]) -> dict[str, float]:
    if not inventory:
        return {"mean_abs_inventory": math.nan, "max_abs_inventory": math.nan, "final_inventory": math.nan}
    return {
        "mean_abs_inventory": statistics.fmean(abs(q) for q in inventory),
        "max_abs_inventory": float(max(abs(q) for q in inventory)),
        "final_inventory": float(inventory[-1]),
    }


def strategy_report(
    sim: MarketSimulator,
    names: Iterable[str] | None = None,
    *,
    horizons: Sequence[int] = (1, 10, 100),
) -> dict[str, dict[str, Any]]:
    """One flat dict per strategy combining PnL, inventory, execution and markout metrics."""
    res = sim.result
    end = sim.now
    out: dict[str, dict[str, Any]] = {}
    for name in names if names is not None else [s.name for s in sim.strategies]:
        strat = sim.strategy(name)
        m = sim.exec_metrics[name]
        row: dict[str, Any] = {}
        row.update(pnl_stats(res.pnl_series.get(name, [])))
        row.update(inventory_stats(res.inventory_series.get(name, [])))
        row["fees"] = strat.portfolio.fees_paid
        row["volume"] = m.filled_qty
        row["maker_share"] = m.maker_qty / m.filled_qty if m.filled_qty else math.nan
        row["spread_capture"] = m.spread_capture_per_share()
        for h, v in markouts(m.fills, res.mid_path, horizons, end_time=end).items():
            row[f"markout_{h}"] = v
        row["fill_ratio"] = m.fill_ratio()
        row["participation"] = m.participation_rate()
        out[name] = row
    return out


def format_table(rows: dict[str, dict[str, Any]], columns: Sequence[str] | None = None, *, precision: int = 4) -> str:
    """Render ``{row_name: {col: value}}`` as a fixed-width text table."""
    if not rows:
        return ""
    cols = list(columns) if columns is not None else list(next(iter(rows.values())).keys())

    def fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return "nan" if math.isnan(v) else f"{v:.{precision}f}"
        return str(v)

    body = [[name, *(fmt(r.get(c)) for c in cols)] for name, r in rows.items()]
    header = ["", *cols]
    widths = [max(len(row[i]) for row in [header, *body]) for i in range(len(header))]

    def render(cells: list[str]) -> str:
        # first column (row names) left-aligned, values right-aligned
        return "  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(cells, widths, strict=True)))

    return "\n".join([render(header), "  ".join("-" * w for w in widths), *(render(row) for row in body)])
