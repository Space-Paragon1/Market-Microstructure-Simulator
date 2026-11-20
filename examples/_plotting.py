"""Figure rendering for the examples (requires matplotlib: ``pip install -e '.[plot]'``)."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (light surface); slots assigned in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": TEXT_2,
        "lines.linewidth": 1.5,
        "font.family": "sans-serif",
    }
)


def _end_label(ax: Any, x: float, y: float, text: str) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=TEXT_2,
        annotation_clip=False,
    )


def _end_labels(ax: Any, x: float, items: list[tuple[str, float]], min_gap_frac: float = 0.06) -> None:
    """Direct labels at the line ends, nudged apart vertically so they never collide."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    placed: list[float] = []
    for name, y in sorted(items, key=lambda it: it[1]):
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.annotate(
            name,
            (x, y),
            xytext=(x, y_text),
            textcoords="data",
            va="center",
            fontsize=8,
            color=TEXT_2,
            annotation_clip=False,
        )


def _mean_se(rows: list[dict[str, Any]], key: str, scale: float = 1.0) -> tuple[float, float]:
    xs = [r[key] * scale for r in rows if r[key] is not None and not math.isnan(r[key])]
    if not xs:
        return math.nan, math.nan
    se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return statistics.fmean(xs), se


def plot_experiments(
    comp: dict[str, list[dict[str, Any]]],
    lat: dict[str, list[dict[str, Any]]],
    inf: dict[str, list[dict[str, Any]]],
    horizons: tuple[int, ...],
    tick: float,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)

    # (a) markout curves: adverse selection decay
    ax = axes[0]
    for color, (name, rows) in zip(SERIES, comp.items(), strict=False):
        ys = []
        for h in horizons:
            vals = [r["markout_curve"][h] for r in rows if r["markout_curve"][h] is not None]
            ys.append(statistics.fmean(vals) / tick if vals else math.nan)
        ax.plot(horizons, ys, color=color, marker="o", markersize=4, label=name)
    ax.set_xscale("log")
    ax.set_xticks(horizons, [str(h) for h in horizons])
    ax.set_xlabel("horizon after fill (time steps, log scale)")
    ax.set_ylabel("markout (ticks per share)")
    ax.set_title("(a) Markouts decay: adverse selection")
    ax.legend(loc="upper right")

    # (b) latency sensitivity
    ax = axes[1]
    xs = [int(k) for k in lat]
    stats = [_mean_se(v, "final") for v in lat.values()]
    ax.errorbar(
        xs,
        [m for m, _ in stats],
        yerr=[1.96 * s for _, s in stats],
        color=SERIES[1],
        marker="o",
        markersize=5,
        capsize=3,
        elinewidth=1,
    )
    ax.set_xlabel("order-entry latency (time steps)")
    ax.set_ylabel("final PnL (mean, 95% CI)")
    ax.set_title("(b) Latency costs the Adaptive MM")
    _end_label(ax, xs[-1], stats[-1][0], "Adaptive MM")

    # (c) informed-flow intensity
    ax = axes[2]
    xs_f = [float(k) for k in inf]
    stats = [_mean_se(v, "final") for v in inf.values()]
    ax.errorbar(
        xs_f,
        [m for m, _ in stats],
        yerr=[1.96 * s for _, s in stats],
        color=SERIES[1],
        marker="o",
        markersize=5,
        capsize=3,
        elinewidth=1,
    )
    ax.set_xlabel("informed arrivals per time step")
    ax.set_ylabel("final PnL (mean, 95% CI)")
    ax.set_title("(c) Informed flow: PnL peaks, then adverse selection wins")
    _end_label(ax, xs_f[-1], stats[-1][0], "Adaptive MM")

    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_market_run(run: dict[str, Any], tick: float, path: Path) -> None:
    t = run["t"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(t, run["fundamental"], color=MUTED, linewidth=1.2, label="fundamental value (latent)")
    ax.plot(t, run["mid"], color=SERIES[0], linewidth=1.0, label="book midprice")
    ax.set_ylabel("price")
    ax.set_title("Price discovery: the book tracks a fundamental it cannot see")
    ax.legend(loc="upper left")

    ax = axes[1]
    name = "Adaptive MM"
    ax.step(t, run["inventory"][name], where="post", color=SERIES[1], linewidth=0.9)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylabel("inventory (shares)")
    ax.set_title(f"{name} inventory: skew keeps it mean-reverting around zero")

    ax = axes[2]
    for color, (name, pnl) in zip(SERIES, run["pnl"].items(), strict=False):
        ax.plot(t, pnl, color=color, label=name)
    ax.set_ylabel("PnL (marked to mid)")
    ax.set_xlabel("time step")
    ax.set_title("Cumulative PnL on a single seed (same market for every strategy)")
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper left")
    _end_labels(ax, t[-1] + (t[-1] - t[0]) * 0.008, [(n, p[-1]) for n, p in run["pnl"].items()])

    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
