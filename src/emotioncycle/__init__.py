"""A-share emotion-cycle monitor: breadth → index → phase → position → forecast.

The core (:mod:`breadth`, :mod:`index`, :mod:`phases`, :mod:`forecast`) is pure
standard library and needs no market data to run. :mod:`jq_source` is the
JoinQuant adapter.
"""

from .breadth import Breadth, StockDay, measure
from .forecast import Evaluation, Forecast, evaluate, predict, slope_of
from .index import Config, blend, clamp, emotion_index, pinned, subscores
from .monitor import Monitor, State
from .phases import (
    CYCLE,
    PHASES,
    PHASE_CN,
    POSITIONS,
    Signal,
    determine_phase,
    early_signal,
    position_for,
    turning_points,
)

__all__ = [
    "Breadth",
    "CYCLE",
    "Config",
    "Evaluation",
    "Forecast",
    "Monitor",
    "PHASES",
    "PHASE_CN",
    "POSITIONS",
    "Signal",
    "State",
    "StockDay",
    "blend",
    "clamp",
    "determine_phase",
    "early_signal",
    "emotion_index",
    "evaluate",
    "measure",
    "pinned",
    "position_for",
    "predict",
    "slope_of",
    "subscores",
    "turning_points",
]
