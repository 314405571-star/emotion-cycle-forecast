"""Market breadth from a sample of stock-days — pure, standard library only.

Four numbers carry the whole model:

    up_ratio      share of stocks that closed up
    limit_ratio   share that closed at the limit
    seal_rate     limit-ups / stocks that touched the limit
    big_loss      share that fell 7% or more

The notebook computes these from a random sample of 800 names. Nothing here
depends on where the sample came from, which is why the whole cycle model can
be replayed from a hand-written list of stock-days.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

__all__ = ["StockDay", "Breadth", "measure", "BIG_LOSS", "STRONG", "MID"]


BIG_LOSS = -0.07      # 7% down counts as a big loss
STRONG = 0.05         # +5% or better: the "strong" bucket
MID = 0.02            # +2%..+5%: the "mid" bucket


@dataclass(frozen=True)
class StockDay:
    """What the model needs to know about one stock on one day."""

    change: float               # close / preclose - 1
    sealed: bool = False        # closed at the limit
    touched: bool = False       # high reached the limit at some point


@dataclass(frozen=True)
class Breadth:
    total: int
    up: int
    limit_up: int
    touched: int
    big_loss: int
    strong: int
    mid: int

    @property
    def up_ratio(self) -> float:
        return self.up / self.total if self.total else 0.0

    @property
    def limit_ratio(self) -> float:
        return self.limit_up / self.total if self.total else 0.0

    @property
    def big_loss_ratio(self) -> float:
        return self.big_loss / self.total if self.total else 0.0

    @property
    def seal_rate(self) -> float | None:
        """None when no stock touched the limit — there is nothing to seal.

        The notebook returns 0.5 here, which turns a dead market into a
        perfectly neutral seal score. That is a *choice*, not a measurement,
        so it is surfaced as ``None`` and made explicit where it is used.
        """
        return self.limit_up / self.touched if self.touched else None

    @property
    def diffusion(self) -> float | None:
        """Mid-strength names per strong name, ×100. ``None`` with no strong names."""
        if self.strong == 0:
            return None
        return self.mid / self.strong * 100.0


def measure(days: Iterable[StockDay]) -> Breadth:
    total = up = limit_up = touched = big_loss = strong = mid = 0
    for day in days:
        total += 1
        if day.change > 0:
            up += 1
        if day.sealed:
            limit_up += 1
        if day.touched:
            touched += 1
        if day.change <= BIG_LOSS:
            big_loss += 1
        if day.change > STRONG:
            strong += 1
        elif MID < day.change <= STRONG:
            mid += 1
    return Breadth(total=total, up=up, limit_up=limit_up, touched=touched,
                   big_loss=big_loss, strong=strong, mid=mid)
