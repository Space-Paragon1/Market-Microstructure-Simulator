# Market Microstructure Simulator

A research-oriented **Python market microstructure simulator** implementing a
**central limit order book (CLOB)** with strict **price–time priority**, wrapped
in an **event-driven simulation and backtesting framework** for studying
execution quality, liquidity provision, and strategy behavior.

Designed to mirror the **core mechanics of real electronic markets** (matching
engines, order flow, execution metrics) with an emphasis on **correctness
first**, followed by performance, analytics, and realism.

---

## Key Features

### 1. Matching Engine (CLOB)
- Central Limit Order Book with strict **price–time priority**
- FIFO queues at each price level
- Partial fills and multi-level sweeping
- Deterministic execution order
- Supports:
  - Limit orders and market orders
  - Cancel
  - Modify with correct priority semantics:
    - Quantity reduction → retains queue priority
    - Quantity increase or price change → loses priority (re-queued)

### 2. Event-Driven Market Simulator
- Discrete-event loop driven by a priority queue `(time, sequence)`
- Deterministic replay given identical event streams
- Event types: `SUBMIT`, `CANCEL`, `MODIFY`, `SNAPSHOT`
- Synthetic order flow via seeded **Poisson-style generator** (`PoissonOrderFlow`)

### 3. Strategy Backtesting Layer
- Clean `Strategy` base class with `on_tick(now, lob)` callbacks on `SNAPSHOT` events
- Implemented strategies:
  - **`MarketMaker`** — baseline symmetric quoting
  - **`TWAPExecutor`** — taker-style time-sliced execution
  - **`AdaptiveMarketMaker`** — production-style maker with:
    - Volatility-aware spread widening (mean-abs mid change)
    - Inventory-risk quote skew + hard inventory limits
    - Order-book imbalance leaning (L2 top-3 levels)
- Per-strategy **`Portfolio`** accounting:
  - Position (inventory), cash, average cost basis
  - Realized PnL and mark-to-market (MTM)

### 4. Execution Quality Metrics
Full per-strategy execution analytics tracked from **decision-time midprice**:

| Metric | Description |
|---|---|
| `buy_vwap` / `sell_vwap` | Volume-weighted average fill price by direction |
| `buy_slippage_vs_arrival` | Fill VWAP minus arrival mid (buys; positive = cost) |
| `sell_slippage_vs_arrival` | Arrival mid minus fill VWAP (sells; positive = cost) |
| `spread_capture_per_share` | Average spread earned per share as maker |
| `fill_ratio` | Filled qty / submitted qty |
| `participation_rate` | Strategy volume / total market volume |
| `avg_price` | Blended average execution price |
| `total_notional` | Total traded notional |
| `maker_qty` / `taker_qty` | Maker vs taker fill breakdown |

`ExecutionMetrics.summary()` returns a flat `dict` of all metrics.

### 5. Analytics & Visualization
- **`TimeSeries`** time-series logging: midprice, bid–ask spread, order-book imbalance
- `SimResult` captures: fills, per-strategy PnL series, time axis
- Example plots (via matplotlib):
  - Strategy MTM PnL over time
  - Spread dynamics
  - Inventory trajectories

### 6. Performance & Profiling
- Benchmark harness for matching engine throughput (`bench_matching.py`)
- `cProfile` / `pstats` profiling harness (`profile_matching.py`)
- Optimizations:
  - Binary insertion for price levels (no full re-sort on each order)
  - Cached per-price aggregated depth → `O(1)` depth lookup

---

## Project Structure

```
src/microbook/
├── book.py                  # LimitOrderBook — CLOB matching engine
├── order.py                 # Order dataclass
├── types.py                 # Side enum, Fill dataclass
├── __init__.py              # Public exports: LimitOrderBook, Order, Side, Fill
├── sim/
│   ├── simulator.py         # MarketSimulator (discrete-event, heapq)
│   ├── events.py            # Event, EventType (SUBMIT/CANCEL/MODIFY/SNAPSHOT)
│   ├── strategy.py          # Strategy base, MarketMaker, TWAPExecutor
│   ├── strategies_mm.py     # AdaptiveMarketMaker (vol-aware, inv skew, imbalance)
│   ├── portfolio.py         # Portfolio (cash, position, realized PnL, MTM)
│   ├── execution_metrics.py # ExecutionMetrics (VWAP, slippage, spread capture, …)
│   ├── analytics.py         # TimeSeries (mid, spread, imbalance over time)
│   ├── metrics.py           # spread() / imbalance() helpers
│   ├── orderflow.py         # PoissonOrderFlow synthetic order generator
│   └── runner.py            # schedule_strategy_ticks helper
├── bench/
│   ├── bench_matching.py    # Throughput benchmark
│   └── profile_matching.py  # cProfile harness
└── examples/
    ├── run_mm_vs_flow.py    # AdaptiveMM + TWAP vs Poisson flow (with plots)
    ├── run_with_plots.py    # MM + TWAP with matplotlib charts
    ├── run_sim.py           # Minimal simulation script
    └── sanity_check.py      # Quick correctness smoke test

tests/
├── test_cancel.py
├── test_market.py
├── test_matching.py
├── test_modify.py
├── test_pnl_math.py
├── test_price_time_priority.py
├── test_sim_determinism.py
└── test_execution_metrics.py   # 27 tests for ExecutionMetrics
```

---

## Installation

```bash
git clone <repo-url>
cd Market-Microstructure-Simulator
pip install -e .
pip install pytest matplotlib
```

---

## Running Tests

```bash
python -m pytest -q
```

35 tests, all passing.

---

## Running Examples

**Adaptive market maker vs Poisson order flow (with metrics + PnL plot):**
```bash
python src/microbook/examples/run_mm_vs_flow.py
```

**Baseline MM + TWAP with plots:**
```bash
python src/microbook/examples/run_with_plots.py
```

---

## Benchmarks

```bash
# Matching engine throughput
python src/microbook/bench/bench_matching.py

# Profile hotspots
python src/microbook/bench/profile_matching.py
```

---

## Quick Example

```python
from microbook import LimitOrderBook, Order, Side
from microbook.sim.simulator import MarketSimulator
from microbook.sim.events import EventType
from microbook.sim.orderflow import PoissonOrderFlow, FlowConfig
from microbook.sim.strategies_mm import AdaptiveMarketMaker, AdaptiveMMConfig
from microbook.sim.strategy import TWAPExecutor

mm   = AdaptiveMarketMaker(AdaptiveMMConfig(tick_interval=10, size=5, inv_limit=25))
twap = TWAPExecutor(Side.BUY, total_qty=40, start=50, end=250, tick_interval=20)

sim = MarketSimulator(strategies=[mm, twap])

# seed resting liquidity
sim.schedule(0, EventType.SUBMIT, order=Order("ask0", Side.SELL, 101.0, 50, ts=0))
sim.schedule(0, EventType.SUBMIT, order=Order("bid0", Side.BUY,   99.0, 50, ts=0))

# synthetic background flow
flow = PoissonOrderFlow(FlowConfig(seed=7, intensity_per_100=40.0, p_market=0.10))
for t, o in flow.iter_orders(start=1, end=500, ref_mid=100.0):
    sim.schedule(t, EventType.SUBMIT, order=o)
    if t % 5 == 0:
        sim.schedule(t, EventType.SNAPSHOT)

res = sim.run(until=500)

# execution quality per strategy
for name, m in sim.exec_metrics.items():
    print(name, m.summary())
```

---

## Design Philosophy

**Correctness before speed** — Exchange-style semantics are enforced and tested
before any optimization is applied.

**Determinism and traceability** — Every fill can be replayed and attributed to
the originating strategy and decision.

**Research extensibility** — The system is built to support:
- Execution quality research (VWAP, slippage, spread capture)
- Market impact analysis
- Strategy comparison under controlled flow regimes
- Microstructure experiments (imbalance, inventory dynamics)

---

## Example Use Cases

- Compare maker vs taker execution quality under identical flow
- Study spread dynamics and inventory risk under aggressive order flow
- Analyze PnL volatility and drawdown for a market-making strategy
- Benchmark matching engine throughput and profile hotspots
- Prototype and backtest new execution strategies in a controlled environment

---

## Disclaimer

This project is for **educational and research purposes only**.
It does not connect to live markets and does not represent any production or
proprietary trading system.
