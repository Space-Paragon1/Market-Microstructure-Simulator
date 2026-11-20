# Changelog

## 0.2.0

A rewrite of the matching engine and a major extension of the simulator.

### Fixed
- **Duplicate order ids** orphaned the earlier order (still matchable, no longer cancelable). They are now rejected.
- **`modify()` that crossed the spread discarded its fills**, so trades never reached portfolios or metrics. Fills are now returned in the `ExecutionReport` and attributed by the simulator.
- **Invalid `modify()` destroyed the order** (e.g. `new_qty=-3` cancelled it, then returned `False`). Requests are validated first and rejections leave the order untouched.
- A no-op `modify()` (same price and quantity) lost queue priority.
- **Fees were deducted from cash but missing from mark-to-market PnL.**
- **Slippage vs arrival was diluted** by fills without a registered arrival mid (the denominator used all fills). Spread capture had the same defect.
- `TWAPExecutor` traded into a one-sided book and recorded an arrival mid of `0.0`, producing slippage of about 100.
- **`AdaptiveMarketMaker` "skew" widened or narrowed the spread instead of shifting it.** When short it quoted a crossed market (bid above ask), and the imbalance lean had the wrong sign.
- Bid-level insertion copied the whole price list on every insert, and cancel rebuilt the whole queue (O(n)).
- `SimResult.pnl_series` did not exist unless a SNAPSHOT event had fired.
- `examples/run_mm_vs_flow.py` had a syntax error.

### Added
- Integer tick grid (`LimitOrderBook(tick_size=...)`); off-grid prices are rejected.
- Order types and time-in-force: `MARKET`, `LIMIT` × `GTC` / `IOC` / `FOK`, plus `post_only`.
- Self-trade prevention (`STPMode.CANCEL_RESTING` / `CANCEL_INCOMING`), keyed on `Order.owner`.
- `submit()` returning an `ExecutionReport` (status, fills, resting / cancelled quantity, reason).
- Book queries: `spread`, `microprice`, `queue_ahead`, `volume_at`, `get_order`, `check_invariants`.
- Simulator: order-entry **latency** per strategy, strategy **timers**, `on_start` / `on_fill` / `on_report` hooks, unique-name validation, `schedule_snapshots`, mid-price path, and inventory series.
- Closed-loop market agents: `FundamentalValue`, `NoiseTrader` (zero-intelligence), `InformedTrader`; `build_market()` scenario.
- `AvellanedaStoikovMarketMaker` (finite and rolling horizon) and the `TwoSidedQuoter` base class.
- Analytics: markouts, PnL statistics (drawdown, Sharpe), inventory statistics, `strategy_report`, `format_table`.
- Maker/taker fee schedule in `Portfolio`.
- Tests: model-based differential testing against a reference book (Hypothesis), property tests for PnL accounting, zero-sum and determinism checks (38 → 121 tests).
- Tooling: hatchling build, ruff, mypy `--strict`, coverage, and GitHub Actions CI on Python 3.10–3.13.

### Changed (breaking)
- `LimitOrderBook.modify()` returns an `ExecutionReport` instead of `bool`; use `.accepted`.
- Market orders are `Order.market(...)` / `order_type=OrderType.MARKET`. The `setattr(order, "_is_market", True)` hack no longer works (`Order` uses slots).
- `place_limit` / `place_market` raise `OrderRejected` when the book refuses an order.
- `Fill` and `ExecutionReport` are `NamedTuple`s. Construction is about 2× faster than frozen dataclasses on the matching hot path.
- `Portfolio.fee_per_share` is replaced by `maker_fee` / `taker_fee`; `mark_to_market` is now net of fees.
- `MarketMaker.tick_size` and `AdaptiveMMConfig.tick_size` default to the book's tick (was `1.0`).
- Market makers are post-only and never quote through the opposite touch; unchanged quotes keep their queue priority.
- Spread capture is measured against the **pre-trade** mid (was the post-trade mid).
- `MarketSimulator()` with no book now uses self-trade prevention (`CANCEL_RESTING`).
- `MarketSimulator.run()` accumulates into `sim.result` across calls.
- `Strategy.on_fills` (an unused stub) is replaced by `on_fill`; `_next_ts` is replaced by `next_order_id`.
- Removed `microbook.sim.runner` (use `MarketSimulator.schedule_snapshots`) and the empty `microbook.matching`.
- Examples and benchmarks moved out of the installed package into `examples/` and `benchmarks/`.

### Performance
On a realistic message mix (55% adds, 35% cancels, 10% aggressive), Apple M2, Python 3.12:
throughput 139k → 484k msg/s; cancel p50 13.8 µs → 0.5 µs.
