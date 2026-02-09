# Market Microstructure Simulator

A research-oriented **Python market microstructure simulator** implementing a
**central limit order book (CLOB)** with strict **price–time priority**, wrapped
in an **event-driven simulation and backtesting framework** for studying
execution quality, liquidity provision, and strategy behavior.

This project is designed to mirror the **core mechanics of real electronic
markets** (matching engines, order flow, execution metrics), with an emphasis on
**correctness first**, followed by **performance, analytics, and realism**.

---

## Key Features

### 1. Matching Engine (CLOB)
- Central Limit Order Book with **price–time priority**
- FIFO queues at each price level
- Partial fills and multi-level sweeping
- Deterministic execution order
- Supports:
  - Limit orders
  - Market orders
  - Cancel
  - Modify with correct priority semantics:
    - Quantity reduction → retains priority
    - Quantity increase or price change → loses priority

### 2. Event-Driven Market Simulator
- Discrete-event loop using a priority queue `(time, sequence)`
- Deterministic replay given identical event streams
- Supports:
  - Order submissions
  - Cancels / modifies
  - Periodic snapshots
- Synthetic order flow via seeded Poisson-style generator

---

### 3. Strategy Backtesting Layer
- Strategy interface with `on_tick()` callbacks
- Implemented strategies:
  - **TWAP Executor** (taker-style execution)
  - **Market Maker** (baseline)
  - **Adaptive Market Maker**:
    - Volatility-aware spread widening
    - Inventory-risk skew and hard inventory limits
    - Order-book imbalance leaning
- Per-strategy portfolio accounting:
  - Inventory
  - Cash
  - Realized PnL
  - Mark-to-market (MTM)

---

### 4. Execution Quality Metrics (Step 5C)
Tracks execution performance using **decision-time midprice**:

- VWAP slippage vs decision mid
- Spread capture (maker quality)
- Fill ratio
- Participation rate (strategy volume / total market volume)
- Average execution price
- Total traded notional

Metrics are computed from:
- Recorded order decisions
- Actual realized fills
- Total market volume observed in simulation

---

### 5. Analytics & Visualization
- Time-series logging of:
  - Midprice
  - Bid–ask spread
  - Order-book imbalance
- Snapshot-based analytics
- Example plots:
  - Spread over time
  - Strategy PnL (MTM) over time
  - Inventory trajectories

---

### 6. Performance & Profiling
- Benchmark harness for matching engine throughput
- Profiling via `cProfile` and `pstats`
- Optimizations implemented:
  - Binary insertion for price levels (no full re-sort)
  - Cached per-price aggregated depth (`O(1)` depth lookup)
- Designed to preserve correctness while improving throughput

---

## Project Structure

# Market Microstructure Simulator

A research-oriented **Python market microstructure simulator** implementing a
**central limit order book (CLOB)** with strict **price–time priority**, wrapped
in an **event-driven simulation and backtesting framework** for studying
execution quality, liquidity provision, and strategy behavior.

This project is designed to mirror the **core mechanics of real electronic
markets** (matching engines, order flow, execution metrics), with an emphasis on
**correctness first**, followed by **performance, analytics, and realism**.

---

## Key Features

### 1. Matching Engine (CLOB)
- Central Limit Order Book with **price–time priority**
- FIFO queues at each price level
- Partial fills and multi-level sweeping
- Deterministic execution order
- Supports:
  - Limit orders
  - Market orders
  - Cancel
  - Modify with correct priority semantics:
    - Quantity reduction → retains priority
    - Quantity increase or price change → loses priority

### 2. Event-Driven Market Simulator
- Discrete-event loop using a priority queue `(time, sequence)`
- Deterministic replay given identical event streams
- Supports:
  - Order submissions
  - Cancels / modifies
  - Periodic snapshots
- Synthetic order flow via seeded Poisson-style generator

---

### 3. Strategy Backtesting Layer
- Strategy interface with `on_tick()` callbacks
- Implemented strategies:
  - **TWAP Executor** (taker-style execution)
  - **Market Maker** (baseline)
  - **Adaptive Market Maker**:
    - Volatility-aware spread widening
    - Inventory-risk skew and hard inventory limits
    - Order-book imbalance leaning
- Per-strategy portfolio accounting:
  - Inventory
  - Cash
  - Realized PnL
  - Mark-to-market (MTM)

---

### 4. Execution Quality Metrics (Step 5C)
Tracks execution performance using **decision-time midprice**:

- VWAP slippage vs decision mid
- Spread capture (maker quality)
- Fill ratio
- Participation rate (strategy volume / total market volume)
- Average execution price
- Total traded notional

Metrics are computed from:
- Recorded order decisions
- Actual realized fills
- Total market volume observed in simulation

---

### 5. Analytics & Visualization
- Time-series logging of:
  - Midprice
  - Bid–ask spread
  - Order-book imbalance
- Snapshot-based analytics
- Example plots:
  - Spread over time
  - Strategy PnL (MTM) over time
  - Inventory trajectories

---

### 6. Performance & Profiling
- Benchmark harness for matching engine throughput
- Profiling via `cProfile` and `pstats`
- Optimizations implemented:
  - Binary insertion for price levels (no full re-sort)
  - Cached per-price aggregated depth (`O(1)` depth lookup)
- Designed to preserve correctness while improving throughput

---

## Project Structure

microbook/
├── book.py # Limit order book & matching engine
├── order.py # Order data model
├── types.py # Enums and fill records
├── sim/
│ ├── simulator.py # Event-driven simulator
│ ├── events.py # Event definitions
│ ├── orderflow.py # Synthetic order flow generator
│ ├── strategy.py # Strategy base + TWAP / baseline MM
│ ├── strategies_mm.py # Adaptive market maker
│ ├── portfolio.py # Inventory & PnL accounting
│ ├── metrics.py # Spread / imbalance metrics
│ ├── analytics.py # Time-series logging
│ └── execution_metrics.py# Slippage, spread capture, fill metrics
├── bench/
│ ├── bench_matching.py # Throughput benchmark
│ └── profile_matching.py # Profiling harness
├── examples/
│ ├── run_mm_vs_flow.py
│ └── run_with_plots.py
└── README.md


---

## Installation
```bash
git clone <repo-url>
cd Market-Microstructure-Simulator
pip install -U pytest matplotlib

Running Tests
pytest -q

Running a Simulation
python examples/run_mm_vs_flow.py
or with plots:
python examples/run_with_plots.py

Benchmarks

Run matching engine benchmark:
python src/microbook/bench/bench_matching.py
Profile hotspots:
python src/microbook/bench/profile_matching.py

Design Philosophy

Correctness before speed
Exchange-style semantics are enforced and tested before optimization.

Determinism and traceability
Every fill can be replayed and attributed to a strategy.

Research extensibility
The system is designed to support:
Execution research
Market impact analysis
Strategy comparison
Microstructure experiments

Example Use Cases
Compare maker vs taker execution quality
Study spread dynamics under aggressive order flow
Analyze inventory risk and PnL volatility
Benchmark matching engine performance
Prototype execution strategies in a controlled environment

Disclaimer
This project is for educational and research purposes only.
It does not connect to live markets and does not represent any
production or proprietary trading system.

