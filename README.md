# Market Microstructure Simulator

A research-oriented **Python market microstructure simulator** implementing a
**central limit order book (CLOB)** with strict **price–time priority**, designed
as a foundation for studying order flow, execution mechanics, and trading
strategy behavior.

This project is intentionally built to mirror the **core mechanics used in
real electronic exchanges**, with an emphasis on correctness, determinism, and
testability before performance optimization.

---

## Features

### Core Matching Engine
- Central Limit Order Book (CLOB)
- **Price–time priority** matching
- FIFO queues at each price level
- Partial fills and multi-level sweeping
- Deterministic execution order

### Order Types & Lifecycle
- Limit orders (buy / sell)
- Market orders (aggressive liquidity-taking)
- Order cancellation
- Order modification with **correct priority semantics**:
  - Quantity reduction → retains priority
  - Quantity increase or price change → loses priority (cancel + reinsert)

### Book State Introspection
- Top-of-book (L1): best bid, best ask, midprice
- Depth snapshots (L2): aggregated quantity by price level

### Testing & Correctness
- Pytest-based unit tests covering:
  - Price–time priority
  - FIFO behavior at identical price levels
  - Partial fills
  - Cancel and modify semantics
  - Market order behavior
- Deterministic fill traces for validation

---

## Project Structure

├── src/
│ └── microbook/
│ ├── init.py
│ ├── book.py # Limit order book & matching engine
│ ├── order.py # Order data model
│ └── types.py # Enums and fill records
├── tests/
│ ├── test_matching.py
│ ├── test_price_time_priority.py
│ ├── test_cancel.py
│ ├── test_modify.py
│ └── test_market.py
├── examples/
│ └── sanity_check.py # Usage example (manual run)
├── pyproject.toml
└── README.md


---

## Installation

```bash
git clone <your-repo-url>
cd microstructure-sim
pip install -U pytest

Running Tests
All correctness guarantees are enforced via unit tests:
pytest -q

Usage Example (Sanity Check)
A simple usage demo is provided in: src/microbook/examples/sanity_check.py
Run it with: python src/microbook/examples/sanity_check.py
This script demonstrates:
- Placing limit orders

- Inspecting top-of-book and depth

- Modifying orders and observing priority changes

- Executing market orders and consuming liquidity