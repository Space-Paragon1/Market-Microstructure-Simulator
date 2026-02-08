from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from microbook import LimitOrderBook
from .metrics import spread, imbalance


@dataclass
class TimeSeries:
    t: List[int] = field(default_factory=list)
    mid: List[float] = field(default_factory=list)
    spr: List[float] = field(default_factory=list)
    imb: List[float] = field(default_factory=list)

    def record(self, now: int, lob: LimitOrderBook) -> None:
        self.t.append(now)

        m = lob.midprice()
        self.mid.append(float(m) if m is not None else float("nan"))

        s = spread(lob)
        self.spr.append(float(s) if s is not None else float("nan"))

        im = imbalance(lob, levels=3)
        self.imb.append(float(im) if im is not None else float("nan"))

    def as_dict(self) -> Dict[str, List[float]]:
        return {"t": self.t, "mid": self.mid, "spread": self.spr, "imbalance": self.imb}
