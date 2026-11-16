"""JoinQuant adapter: daily bars → breadth → the monitor.

JoinQuant is not importable outside its research environment, so the import is
lazy and everything that can be tested lives elsewhere. This module is
deliberately thin: one function that turns a date into a :class:`Breadth`.

Two things the notebook did that are fixed here:

* ``random.seed(date)`` reseeded the *global* RNG on every call, which silently
  changes the sample for every other piece of code in the process. A local
  ``random.Random`` is used instead.
* the sample is drawn from every listed security, including names that were
  not trading that day; those come back with ``close == 0`` and were dropped
  one by one. They are excluded before the sample is drawn here, so the sample
  size means what it says.

    python -m src.emotioncycle.jq_source --start 2026-01-05 --end 2026-06-30
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from typing import Sequence

from .breadth import Breadth, StockDay, measure
from .index import Config
from .monitor import Monitor

__all__ = ["JQSource", "breadth_for", "run_dates", "main"]


class JQSource:
    def __init__(self, sample: int = 800, min_stocks: int = 100) -> None:
        self.sample = sample
        self.min_stocks = min_stocks

    def universe(self, day: str) -> list[str]:
        from jqdata import get_all_securities  # lazy

        frame = get_all_securities(["stock"], date=day)
        return list(frame.index)

    def stock_days(self, day: str) -> list[StockDay]:
        from jqdata import get_price  # lazy

        universe = self.universe(day)
        if not universe:
            return []
        picked = random.Random(day).sample(universe, min(self.sample, len(universe)))
        prices = get_price(
            picked, end_date=day, frequency="1d",
            fields=["close", "high", "high_limit"], count=2,
            skip_paused=False, panel=False,
        )
        if prices is None or prices.empty:
            return []

        out: list[StockDay] = []
        for _, group in prices.groupby("code"):
            group = group.sort_values("time")
            if len(group) < 2:
                continue
            last = float(group["close"].iloc[-1])
            prev = float(group["close"].iloc[-2])
            high = float(group["high"].iloc[-1])
            limit = float(group["high_limit"].iloc[-1]) if "high_limit" in group else last * 1.1
            if last <= 0 or prev <= 0:
                continue                      # suspended / not yet listed
            touched = limit > 0 and high >= limit - 0.01
            out.append(StockDay(
                change=last / prev - 1.0,
                sealed=touched and last >= limit - 0.01,
                touched=touched,
            ))
        return out

    def breadth(self, day: str) -> Breadth | None:
        days = self.stock_days(day)
        if len(days) < self.min_stocks:
            return None
        return measure(days)


def breadth_for(day: str, source: JQSource) -> Breadth | None:
    return source.breadth(day)


def run_dates(days: Sequence[str], source: JQSource | None = None,
              cfg: Config = Config()) -> list:
    source = source or JQSource()
    monitor = Monitor(cfg)
    states = []
    for day in days:
        breadth = source.breadth(day)
        if breadth is None:
            continue
        states.append(monitor.update(day, breadth, confidence=0.98))
    return states


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emotion-cycle monitor (JoinQuant)")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--sample", type=int, default=800)
    args = parser.parse_args(argv)

    from jqdata import get_trade_days  # lazy

    days = [d.strftime("%Y-%m-%d")
            for d in get_trade_days(start_date=args.start, end_date=args.end)]
    states = run_dates(days, JQSource(sample=args.sample))
    for state in states[-10:]:
        print(f"{state.date}  {state.phase_cn:<4}  {state.emotion:5.1f}  "
              f"{state.position:.0%}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
