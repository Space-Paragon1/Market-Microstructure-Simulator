from __future__ import annotations
from typing import Dict, Optional

from microbook import LimitOrderBook


def spread(lob: LimitOrderBook) -> Optional[float]:
    b = lob.best_bid()
    a = lob.best_ask()
    if b is None or a is None:
        return None
    return a - b


def imbalance(lob: LimitOrderBook, levels: int = 3) -> Optional[float]:
    d = lob.depth(levels=levels)
    bids = sum(q for _, q in d["bids"])
    asks = sum(q for _, q in d["asks"])
    total = bids + asks
    if total == 0:
        return None
    return (bids - asks) / total
