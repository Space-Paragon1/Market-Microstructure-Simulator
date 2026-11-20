"""Order-book state metrics."""

from __future__ import annotations

from microbook import LimitOrderBook


def spread(lob: LimitOrderBook) -> float | None:
    """Best ask minus best bid (``None`` if either side is empty)."""
    return lob.spread()


def imbalance(lob: LimitOrderBook, levels: int = 3) -> float | None:
    """Depth imbalance over the top ``levels``: (bid qty - ask qty) / total, in [-1, 1].
    Positive means more resting size on the bid (buying pressure)."""
    d = lob.depth(levels=levels)
    bids = sum(q for _, q in d["bids"])
    asks = sum(q for _, q in d["asks"])
    total = bids + asks
    return None if total == 0 else (bids - asks) / total
