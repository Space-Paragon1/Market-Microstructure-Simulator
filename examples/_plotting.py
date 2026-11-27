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


def plot_stylized_facts(summ: dict[str, dict[str, Any]], kurt_dts: tuple[int, ...], acf_dt: int, path: Path) -> None:
    """Six-panel stylized-facts comparison of the simple and realistic markets."""
    colors = {"realistic": SERIES[0], "simple": SERIES[1]}
    labels = {"realistic": "MarketConfig.realistic()", "simple": "MarketConfig() (simple)"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)

    def err(ax: Any, xs: list[float], ms: list[tuple[float, float]], color: str, label: str) -> None:
        ax.errorbar(
            xs,
            [m for m, _ in ms],
            yerr=[1.96 * (0 if math.isnan(s) else s) for _, s in ms],
            color=color,
            marker="o",
            markersize=3.5,
            capsize=2,
            elinewidth=0.8,
            label=label,
        )

    # (a) tails: complementary CDF of standardized |r|
    ax = axes[0, 0]
    anchor = None
    for m in ("simple", "realistic"):
        xs = sorted(x for x in summ[m]["std_abs_r100"] if x > 0)
        n = len(xs)
        if n:
            ccdf = [1 - i / n for i in range(n)]
            ax.plot(xs, ccdf, color=colors[m], linewidth=1.5, label=labels[m])
            if m == "realistic":
                anchor = next((c for x, c in zip(xs, ccdf, strict=True) if x >= 2), None)
    g = [0.5 + 0.05 * i for i in range(90)]
    ax.plot(g, [math.erfc(x / math.sqrt(2)) for x in g], color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(
        "Gaussian",
        (2.9, math.erfc(2.9 / math.sqrt(2))),
        xytext=(-52, -4),
        textcoords="offset points",
        fontsize=8,
        color=TEXT_2,
    )
    if anchor:
        ref = [2.0 * 1.1**i for i in range(25)]
        ax.plot(ref, [anchor * (x / 2) ** -3 for x in ref], color=MUTED, linestyle=":", linewidth=1)
        ax.annotate(
            "∝ x⁻³", (8.0, anchor * 4.0**-3), xytext=(4, 4), textcoords="offset points", fontsize=8, color=TEXT_2
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 30)
    ax.set_ylim(1e-4, 1.2)
    ax.set_xlabel("|100-step return| / std (log)")
    ax.set_ylabel("P(|r| > x) (log)")
    ax.set_title("(a) Fat tails: far heavier than Gaussian")

    # (b) aggregational Gaussianity
    ax = axes[0, 1]
    for m in ("simple", "realistic"):
        err(ax, list(kurt_dts), [summ[m]["kurtosis"][dt] for dt in kurt_dts], colors[m], labels[m])
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(list(kurt_dts), [str(d) for d in kurt_dts])
    ax.set_xlabel("return horizon (steps, log)")
    ax.set_ylabel("excess kurtosis (95% CI)")
    ax.set_title("(b) Tails thin out at longer horizons")

    # (c) return autocorrelation
    ax = axes[0, 2]
    for m in ("simple", "realistic"):
        c = summ[m]["acf_r"]
        err(ax, [k * acf_dt for k in c], list(c.values()), colors[m], labels[m])
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(f"lag (steps, log; {acf_dt}-step returns)")
    ax.set_ylabel("return autocorrelation")
    ax.set_title("(c) Returns: the simple market's momentum is gone")

    # (d) volatility clustering
    ax = axes[1, 0]
    for m in ("simple", "realistic"):
        c = summ[m]["acf_abs"]
        err(ax, [k * acf_dt for k in c], list(c.values()), colors[m], labels[m])
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(f"lag (steps, log; {acf_dt}-step returns)")
    ax.set_ylabel("|return| autocorrelation")
    ax.set_title("(d) Volatility clusters")

    # (e) order-sign memory
    ax = axes[1, 1]
    for m in ("simple", "realistic"):
        c = {k: v for k, v in summ[m]["acf_sign"].items() if v[0] > 0}
        ax.plot(list(c), [v[0] for v in c.values()], color=colors[m], marker="o", markersize=3.5, label=labels[m])
    lags = [1, 10, 100, 1000]
    ax.plot(lags, [0.5 * x**-0.5 for x in lags], color=MUTED, linestyle=":", linewidth=1)
    _end_label(ax, lags[-1], 0.5 * lags[-1] ** -0.5, "real: ∝ lag⁻⁰·⁵")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lag (trades, log)")
    ax.set_ylabel("sign autocorrelation (log; > 0 only)")
    ax.set_title("(e) Order signs: memory too short")

    # (f) metaorder impact
    ax = axes[1, 2]
    bins = summ["realistic"]["impact_bins"]
    if bins:
        xs = [b[0] for b in bins]
        err(ax, xs, [(b[1], b[2]) for b in bins], colors["realistic"], labels["realistic"])
        lo, hi = min(xs), max(xs)
        ref_x = [lo * (hi / lo) ** (i / 20) for i in range(21)]
        ax.plot(ref_x, [0.5 * math.sqrt(x) for x in ref_x], color=MUTED, linestyle=":", linewidth=1)
        _end_label(ax, ref_x[-1], 0.5 * math.sqrt(ref_x[-1]), "real: 0.5·√(Q/V)")
        y, _ = summ["realistic"]["impact_Y"]
        ax.set_title(f"(f) Impact: Y = {y:.2f}, below the √ law")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("metaorder size / daily volume (log)")
    ax.set_ylabel("impact / daily volatility (95% CI)")

    handles, names = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
