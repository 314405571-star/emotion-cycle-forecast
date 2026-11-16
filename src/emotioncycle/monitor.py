"""The stateful monitor: breadth in, a state out.

This is the notebook's ``update`` / ``update_for_research`` / ``save_daily_state``
collapsed into one object. The only state that matters is what the next session
needs: yesterday's emotion and phase, yesterday's breadth (for turning points),
and enough history for the forecast.

Calling :meth:`update` twice for the same session is an error rather than a
silent append — the notebook's diffusion history grew once per call, so an
intraday loop recorded 48 "days" a day.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .breadth import Breadth
from .forecast import Forecast, predict
from .index import Config, blend, clamp, emotion_index, pinned, subscores
from .phases import (
    PHASE_CN,
    Signal,
    determine_phase,
    early_signal,
    position_for,
    turning_points,
)

__all__ = ["State", "Monitor"]


@dataclass(frozen=True)
class State:
    date: str
    emotion: float
    phase: str
    position: float
    early: Signal | None
    turns: tuple[Signal, ...]
    pinned: bool
    subs: dict = field(default_factory=dict)

    @property
    def phase_cn(self) -> str:
        return PHASE_CN.get(self.phase, self.phase)


class Monitor:
    def __init__(self, cfg: Config = Config()) -> None:
        self.cfg = cfg
        self.emotion = 50.0
        self.phase = "normal"
        self.position = 0.5
        self.state: State | None = None
        self._breadth: Breadth | None = None
        self._emotions: list[float] = []
        self._phases: list[str] = []
        self._dates: list[str] = []

    # -- history ---------------------------------------------------------
    @property
    def emotions(self) -> list[float]:
        return list(self._emotions)

    @property
    def phases(self) -> list[str]:
        return list(self._phases)

    def forecast(self, num_days: int = 7) -> Forecast | None:
        return predict(self._emotions, self._phases, num_days=num_days,
                       emotion=self.emotion)

    # -- the update ------------------------------------------------------
    def update(self, day: str, breadth: Breadth, confidence: float = 1.0) -> State:
        if self._dates and day <= self._dates[-1]:
            raise ValueError(f"{day} is not after {self._dates[-1]}: one update per session")

        raw = emotion_index(breadth, self.cfg)
        blended = blend(self.emotion, raw, confidence)
        value = clamp(self.emotion, blended, self.cfg.max_daily_change)
        trend3 = value - (self._emotions[-3] if len(self._emotions) >= 3 else 50.0)

        phase = determine_phase(value, self.emotion, self.phase, breadth,
                                self.cfg, trend3=trend3)
        turns = turning_points(self.phase, phase, breadth, self._breadth)
        early = early_signal(phase, breadth.diffusion)

        self.state = State(
            date=day, emotion=value, phase=phase, position=position_for(phase),
            early=early, turns=turns,
            pinned=pinned(self.emotion, blended, self.cfg.max_daily_change),
            subs=subscores(breadth, self.cfg),
        )
        self.emotion, self.phase = value, phase
        self.position = self.state.position
        self._breadth = breadth
        self._emotions.append(value)
        self._phases.append(phase)
        self._dates.append(day)
        return self.state

    # -- reporting -------------------------------------------------------
    def summary(self) -> str:
        if self.state is None:
            return "no sessions yet"
        lines = [
            f"{self.state.date}  {self.state.phase_cn:<4} "
            f"emotion {self.state.emotion:5.1f}  position {self.state.position:.0%}",
        ]
        forecast = self.forecast()
        if forecast:
            lines.append(f"  tomorrow: {PHASE_CN[forecast.main]} "
                         f"(alt {PHASE_CN[forecast.alt]}, {forecast.confidence:.0%})")
            lines.append(f"  {forecast.reason}")
        return "\n".join(lines)
