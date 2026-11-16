"""Offline demo: a synthetic market with a hidden cycle, the full pipeline, five figures.

No network, no JoinQuant. The generator draws a *hidden* phase from a Markov
chain, then emits 800 stock-days per session consistent with that phase. The
model never sees the hidden phase — it only sees the breadth it produces.

    python examples/demo_cycle.py
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.emotioncycle.breadth import Breadth, StockDay, measure  # noqa: E402
from src.emotioncycle.forecast import evaluate, predict  # noqa: E402
from src.emotioncycle.index import Config, emotion_index  # noqa: E402
from src.emotioncycle.monitor import Monitor  # noqa: E402
from src.emotioncycle.phases import (  # noqa: E402
    CYCLE,
    PHASE_CN,
    Signal,
    determine_phase,
    early_signal,
    position_for,
    turning_points,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
DAYS = 260
SAMPLE = 800
SEED = 7
START = date(2025, 1, 2)

#: Hidden-phase distributions. Each is (up_ratio, limit_ratio, seal_rate,
#: big_loss_ratio, strong_share, mid_share).
REGIMES = {
    "ice":        dict(up=0.16, limit=0.004, seal=0.34, loss=0.085, strong=0.010, mid=0.030),
    "repair":     dict(up=0.44, limit=0.012, seal=0.55, loss=0.040, strong=0.030, mid=0.075),
    "rise":       dict(up=0.62, limit=0.026, seal=0.72, loss=0.020, strong=0.055, mid=0.110),
    "climax":     dict(up=0.76, limit=0.048, seal=0.84, loss=0.014, strong=0.085, mid=0.090),
    "divergence": dict(up=0.50, limit=0.024, seal=0.54, loss=0.046, strong=0.045, mid=0.070),
    "fall":       dict(up=0.30, limit=0.009, seal=0.38, loss=0.072, strong=0.020, mid=0.045),
    "normal":     dict(up=0.50, limit=0.012, seal=0.60, loss=0.030, strong=0.030, mid=0.080),
}


@dataclass
class Day:
    date: str
    latent: str
    breadth: Breadth
    raw_emotion: float = 0.0
    emotion: float = 50.0
    phase: str = "normal"
    position: float = 0.5
    early: Signal | None = None
    turns: tuple[Signal, ...] = ()
    subs: dict = field(default_factory=dict)


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _next_phase(phase: str, rng: random.Random) -> str:
    if phase == "normal":
        return rng.choices(["normal", "repair", "ice"], [0.72, 0.20, 0.08])[0]
    index = CYCLE.index(phase)
    roll = rng.random()
    if roll < 0.84:
        return phase
    if roll < 0.955:
        return CYCLE[(index + 1) % len(CYCLE)]
    return CYCLE[(index + 2) % len(CYCLE)]


RATE_KEYS = ("up", "limit", "seal", "loss", "strong", "mid")


def _stock_days(rates: dict, rng: random.Random) -> list[StockDay]:
    """800 stock-days consistent with the realised rates, plus sampling noise."""
    n = SAMPLE
    up_rate = _clip(rng.gauss(rates["up"], 0.025), 0.03, 0.95)
    limit_rate = _clip(rng.gauss(rates["limit"], max(0.0015, rates["limit"] * 0.30)), 0.0, 0.12)
    seal = _clip(rng.gauss(rates["seal"], 0.05), 0.15, 0.97)
    loss_rate = _clip(rng.gauss(rates["loss"], max(0.002, rates["loss"] * 0.25)), 0.0, 0.25)
    strong_rate = _clip(rng.gauss(rates["strong"], 0.004), 0.0, 0.2)
    mid_rate = _clip(rng.gauss(rates["mid"], 0.008), 0.0, 0.25)

    n_limit = max(0, round(limit_rate * n))
    n_touch = max(n_limit, round(n_limit / seal)) if seal > 0 else n_limit
    n_strong = round(strong_rate * n)
    n_mid = round(mid_rate * n)
    n_loss = round(loss_rate * n)
    n_touch = min(n_touch, n - n_loss)
    n_strong = min(n_strong, n - n_touch - n_loss)
    n_mid = min(n_mid, n - n_touch - n_loss - n_strong)

    days: list[StockDay] = []
    for _ in range(n_limit):
        days.append(StockDay(0.10, sealed=True, touched=True))
    for _ in range(n_touch - n_limit):
        days.append(StockDay(rng.uniform(0.02, 0.06), sealed=False, touched=True))
    for _ in range(n_strong):
        days.append(StockDay(rng.uniform(0.051, 0.095)))
    for _ in range(n_mid):
        days.append(StockDay(rng.uniform(0.021, 0.049)))
    for _ in range(n_loss):
        days.append(StockDay(rng.uniform(-0.10, -0.07)))

    rest = n - len(days)
    target_up = round(up_rate * n) - (n_touch + n_strong + n_mid)
    target_up = _clip(target_up, 0, rest)
    for i in range(rest):
        if i < target_up:
            days.append(StockDay(rng.uniform(0.001, 0.019)))
        else:
            days.append(StockDay(rng.uniform(-0.045, -0.001)))

    rng.shuffle(days)
    return days


def simulate(days: int = DAYS, seed: int = SEED, smooth: float = 0.6) -> list[Day]:
    """A regime switch moves the tape partway each session — markets decay, they
    do not step. ``smooth`` is the share of the gap closed per day."""
    rng = random.Random(seed)
    out: list[Day] = []
    phase = "normal"
    rates = dict(REGIMES["normal"])
    day = START
    while len(out) < days:
        if day.weekday() < 5:
            phase = _next_phase(phase, rng) if out else phase
            target = REGIMES[phase]
            rates = {k: rates[k] + smooth * (target[k] - rates[k]) for k in RATE_KEYS}
            breadth = measure(_stock_days(rates, rng))
            out.append(Day(date=day.isoformat(), latent=phase, breadth=breadth))
        day += timedelta(days=1)
    return out


def run(days: list[Day], cfg: Config = Config()) -> list[Day]:
    """The model: index → blend → clamp → phase → position. Hidden phase unused."""
    monitor = Monitor(cfg)
    for item in days:
        state = monitor.update(item.date, item.breadth, confidence=0.98)
        item.subs = state.subs
        item.raw_emotion = emotion_index(item.breadth, cfg)
        item.emotion = state.emotion
        item.phase = state.phase
        item.position = state.position
        item.early = state.early
        item.turns = state.turns
    return days


def summarise(days: list[Day]) -> dict:
    agreement = sum(1 for d in days if d.phase == d.latent) / len(days)
    result = evaluate([d.emotion for d in days], [d.phase for d in days])
    return {"agreement": agreement, "evaluation": result}


def main() -> int:
    from examples import figures

    days = run(simulate())
    stats = summarise(days)
    evaluation = stats["evaluation"]

    print("Emotion-cycle demo — synthetic market, hidden phases")
    print(f"  sessions            {len(days)}")
    print(f"  inferred == latent  {stats['agreement']:.1%}")
    print("\n  phase mix")
    for phase in ("ice", "repair", "rise", "climax", "divergence", "fall", "normal"):
        latent = sum(1 for d in days if d.latent == phase)
        inferred = sum(1 for d in days if d.phase == phase)
        print(f"    {phase:<11}{PHASE_CN[phase]:<4} latent {latent:>4}   inferred {inferred:>4}")

    print("\n  next-day forecast (walk-forward)")
    print(f"    rule table   {evaluation.accuracy:.1%}  (n={evaluation.n})")
    print(f"    persistence  {evaluation.baseline:.1%}")
    print(f"    on the {evaluation.transition_n} days the phase actually changed:"
          f"  table {evaluation.transition_accuracy:.1%}  vs persistence 0.0%")
    print("\n  calibration (stated confidence → realised accuracy)")
    for conf in sorted(evaluation.by_confidence):
        n, hits = evaluation.by_confidence[conf]
        print(f"    {conf:.2f} → {hits / n:.1%}   (n={n})")

    last = days[-1]
    forecast = predict([d.emotion for d in days], [d.phase for d in days])
    print(f"\n  latest: {last.date}  {PHASE_CN[last.phase]}  "
          f"emotion {last.emotion:.1f}  position {last.position:.0%}")
    if forecast:
        print(f"  tomorrow: {PHASE_CN[forecast.main]} (alt {PHASE_CN[forecast.alt]}, "
              f"{forecast.confidence:.0%}) — {forecast.reason}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures.hero(days, FIG_DIR / "cycle_hero.png")
    figures.indicators(days, FIG_DIR / "indicators.png")
    figures.transitions(days, FIG_DIR / "transitions.png")
    figures.forecast(days, evaluation, FIG_DIR / "forecast_eval.png")
    figures.calibration(evaluation, FIG_DIR / "calibration.png")
    print(f"\n  five figures written to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
