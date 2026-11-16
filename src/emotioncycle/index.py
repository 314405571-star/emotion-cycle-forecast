"""The 0-100 emotion index — pure, standard library only.

Four sub-scores, each rescaled to 0-100 and weighted:

    limit   min(100, limit_ratio * 1000)       10% of the market at the limit = 100
    up      (up_ratio - 0.30) / 0.40 * 100      30% up = 0, 70% up = 100
    seal    seal_rate * 100
    loss    1 - (big_loss_ratio / 0.05)         5% of the market down 7% = 0

Two fixes against the notebook live here:

* ``loss`` divided by a hard-coded count of 40 stocks. That is 5% of an
  800-name sample, so the same market read twice as bad the moment the sample
  size changed. It is expressed as a *share* here.
* the notebook blends raw and yesterday's value only while confidence < 0.60
  and switches to the raw value at exactly 0.60 — a 0.41·(raw-yesterday) jump
  at the boundary. Blending is continuous here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .breadth import Breadth

__all__ = ["Config", "subscores", "emotion_index", "blend", "clamp", "pinned"]

#: The notebook's constants, kept as defaults so the two can be compared.
DEFAULTS = dict(
    w_limit=0.35, w_up=0.25, w_seal=0.25, w_loss=0.15,
    up_floor=0.30, up_span=0.40, limit_scale=1000.0,
    big_loss_span=0.05, neutral_seal=50.0, max_daily_change=15.0,
    ice=30.0, climax=75.0,
)


@dataclass(frozen=True)
class Config:
    w_limit: float = 0.35
    w_up: float = 0.25
    w_seal: float = 0.25
    w_loss: float = 0.15
    up_floor: float = 0.30
    up_span: float = 0.40
    limit_scale: float = 1000.0
    #: Share of the sample at -7% that drives the loss score to zero.
    big_loss_span: float = 0.05
    #: Seal score used when nothing touched the limit.
    neutral_seal: float = 50.0
    max_daily_change: float = 15.0
    ice: float = 30.0
    climax: float = 75.0

    def weights(self) -> tuple[float, float, float, float]:
        return self.w_limit, self.w_up, self.w_seal, self.w_loss


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def subscores(breadth: Breadth, cfg: Config = Config()) -> dict[str, float]:
    seal = breadth.seal_rate
    return {
        "limit": min(100.0, breadth.limit_ratio * cfg.limit_scale),
        "up": _clip((breadth.up_ratio - cfg.up_floor) / cfg.up_span * 100.0, 0.0, 100.0),
        # None → neutral, and the caller can see it in ``seal_defined``
        "seal": cfg.neutral_seal if seal is None else seal * 100.0,
        "loss": _clip(1.0 - breadth.big_loss_ratio / cfg.big_loss_span, 0.0, 1.0) * 100.0,
    }


def emotion_index(breadth: Breadth, cfg: Config = Config()) -> float:
    parts = subscores(breadth, cfg)
    total = (
        parts["limit"] * cfg.w_limit
        + parts["up"] * cfg.w_up
        + parts["seal"] * cfg.w_seal
        + parts["loss"] * cfg.w_loss
    )
    return _clip(total, 0.0, 100.0)


def blend(yesterday: float, raw: float, confidence: float) -> float:
    """Always blend. ``confidence=1`` gives the raw value, ``0`` gives yesterday's."""
    confidence = _clip(confidence, 0.0, 1.0)
    return yesterday * (1.0 - confidence) + raw * confidence


def clamp(previous: float, value: float, max_change: float = 15.0) -> float:
    """Limit the day-over-day move.

    The notebook applies this *after* blending, so a market that keeps moving
    more than 15 points is pinned at exactly +15 every day and the phase
    machine reads it as a permanent "rise". The clamp is kept (it is the
    notebook's rule); :func:`pinned` makes the side effect visible.
    """
    delta = value - previous
    if abs(delta) > max_change:
        return previous + max_change * (1.0 if delta > 0 else -1.0)
    return value


def pinned(previous: float, value: float, max_change: float = 15.0) -> bool:
    """True when the clamp is doing the work rather than the measurement."""
    return abs(value - previous) > max_change
