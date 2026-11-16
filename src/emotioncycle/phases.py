"""The seven-phase state machine — pure, standard library only.

    ice → repair → rise → climax → divergence → fall → ice

``normal`` is the neutral bucket outside the cycle. Each phase maps to a
position-size ceiling, and two special conditions produce early signals and
turning points.

The machine is a *hysteresis* rule: it compares today's emotion with yesterday's
(±3 points), so a flat market does not flip phase every day.
"""

from __future__ import annotations

from dataclasses import dataclass

from .breadth import Breadth
from .index import Config

__all__ = [
    "PHASES",
    "PHASE_CN",
    "CYCLE",
    "POSITIONS",
    "Signal",
    "determine_phase",
    "turning_points",
    "early_signal",
    "position_for",
    "next_in_cycle",
]

PHASES = ("ice", "repair", "rise", "climax", "divergence", "fall", "normal")

PHASE_CN = {
    "ice": "冰点", "repair": "修复", "rise": "上升",
    "climax": "高潮", "divergence": "分歧", "fall": "退潮", "normal": "正常",
}

#: The cycle in order; ``normal`` sits outside it.
CYCLE = ("ice", "repair", "rise", "climax", "divergence", "fall")

#: Position-size ceiling per phase (the notebook's table).
POSITIONS = {
    "ice": 0.20, "repair": 0.60, "rise": 0.80, "climax": 0.40,
    "divergence": 0.30, "fall": 0.10, "normal": 0.50,
}

#: Where the cycle goes next. Used to stop the machine skipping a rung.
FOLLOWS = {
    "ice": "repair", "repair": "rise", "rise": "climax",
    "climax": "divergence", "divergence": "fall", "fall": "ice",
    "normal": "repair",
}


@dataclass(frozen=True)
class Signal:
    kind: str
    desc: str
    urgency: int = 2          # 1 = act now, 2 = watch


def _is_climax(breadth: Breadth) -> bool:
    return (breadth.limit_ratio > 0.04
            and (breadth.seal_rate or 0.0) > 0.70
            and breadth.up_ratio > 0.60)


def _is_ice(breadth: Breadth) -> bool:
    # ``seal_rate`` is None when nothing touched the limit: that *is* ice.
    return breadth.up_ratio < 0.25 and breadth.big_loss_ratio > 0.035


def next_in_cycle(phase: str) -> str:
    return FOLLOWS.get(phase, phase)


def determine_phase(
    emotion: float,
    prev_emotion: float,
    prev_phase: str,
    breadth: Breadth,
    cfg: Config = Config(),
    trend3: float = 0.0,
) -> str:
    """Classify today. Order matters: the two absolute conditions win first.

    The notebook's last fallback is ``emotion < prev - 3 → fall``, checked
    before ``emotion > prev + 3 → rise``. From ``climax`` a 4-point drop is
    therefore a fall — skipping ``divergence`` entirely, although the previous
    branch only fires at -5. The cycle order is repaired here: from climax the
    machine steps to divergence, never straight to the ebb.
    """
    if _is_climax(breadth):
        return "climax"
    if _is_ice(breadth):
        return "ice"
    # Divergence is the first big drop after strength. The notebook requires
    # yesterday to be *labelled* climax, so a single day of lag in recognising
    # the peak loses the phase for the whole leg. It may also follow `rise`.
    if prev_phase in ("climax", "rise") and emotion < prev_emotion - 5:
        return "divergence"
    # A recovery can be one big day *or* three steady ones. With the notebook's
    # one-day rule a market that climbs +3 a day never leaves `fall`, because
    # +3 is not > +3; it then jumps straight to `rise`, skipping `repair`.
    recovering = (emotion > prev_emotion + 5 or trend3 > 5) and breadth.up_ratio > 0.40
    if prev_phase in ("ice", "divergence", "fall") and recovering:
        return "repair"
    if emotion < prev_emotion - 3 and emotion > 30:
        if prev_phase == "climax":
            return "divergence"          # do not skip a rung
        return "fall"
    # The cycle is fall → repair → rise. Letting any non-climax phase jump to
    # `rise` turns a slow recovery into an "uptrend" without ever repairing.
    if emotion > prev_emotion + 3 and prev_phase in ("repair", "rise"):
        return "rise"
    return prev_phase


def turning_points(
    prev_phase: str,
    phase: str,
    breadth: Breadth,
    prev_breadth: Breadth | None,
) -> tuple[Signal, ...]:
    """Two named reversals, both of which need yesterday's breadth."""
    if prev_breadth is None:
        return ()
    signals: list[Signal] = []

    if prev_phase == "divergence" and phase in ("repair", "rise", "climax"):
        prev_limit = prev_breadth.limit_ratio
        prev_seal = prev_breadth.seal_rate or 0.0
        limit = breadth.limit_ratio
        seal = breadth.seal_rate or 0.0
        if limit > prev_limit * 1.2 and seal > prev_seal * 1.1:
            signals.append(Signal("turn_strong", "分歧转一致（买入信号）", 1))

    if prev_phase == "ice" and phase == "repair" and breadth.up_ratio > 0.40:
        signals.append(Signal("turn_repair", "冰点转修复（左侧布局信号）", 1))

    return tuple(signals)


def early_signal(phase: str, diffusion: float | None) -> Signal | None:
    """Diffusion divergence, one step before the phase machine notices."""
    if diffusion is None:
        return None
    if phase == "climax" and diffusion < 35:
        return Signal("climax", "高潮期扩散度骤降（提前减仓）", 2)
    if phase == "ice" and diffusion > 55:
        return Signal("ice", "冰点期扩散度回升（提前关注）", 2)
    return None


def position_for(phase: str) -> float:
    return POSITIONS.get(phase, 0.5)
