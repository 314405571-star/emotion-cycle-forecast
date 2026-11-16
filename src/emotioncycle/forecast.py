"""Next-day phase forecast — the rule table, and the evaluator it never had.

The notebook predicts tomorrow's phase from (current phase, 7-day slope, last
one-day change) with a hand-written table. Every branch carries a confidence
constant — 0.50, 0.55, 0.60, 0.65, 0.70 — that was asserted, never measured.

This module keeps the table and adds :func:`evaluate`, a walk-forward check
that answers the only two questions worth asking:

* how often is the main prediction right, against "tomorrow is the same as
  today" (the persistence baseline)?
* do the stated confidences mean anything (a calibration curve)?

The table's `fall` branch is deliberately kept as written: a *positive* slope
while in the ebb is predicted to lead to ice, not repair. That is the notebook's
call, and it is a defensible one (the first bounce of a broken market is often
a dead-cat), but it is worth seeing in the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .phases import PHASES

__all__ = ["Forecast", "Evaluation", "predict", "evaluate", "slope_of"]


@dataclass(frozen=True)
class Forecast:
    main: str
    alt: str | None
    confidence: float
    reason: str
    watch: str


@dataclass
class Evaluation:
    n: int = 0
    hits: int = 0
    baseline_hits: int = 0
    #: Days on which the phase actually changed — the only days a forecaster can
    #: beat persistence on, since persistence is wrong there by construction.
    transition_n: int = 0
    transition_hits: int = 0
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    #: (predicted, actual) — the diagonal is the hit rate.
    forecast_confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    by_phase: dict[str, list[int]] = field(default_factory=dict)   # [n, hits, base_hits]
    by_confidence: dict[float, list[int]] = field(default_factory=dict)  # [n, hits]

    @property
    def accuracy(self) -> float:
        return self.hits / self.n if self.n else 0.0

    @property
    def baseline(self) -> float:
        return self.baseline_hits / self.n if self.n else 0.0

    @property
    def transition_accuracy(self) -> float:
        return self.transition_hits / self.transition_n if self.transition_n else 0.0


def slope_of(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def predict(
    emotions: Sequence[float],
    phases: Sequence[str],
    *,
    num_days: int = 7,
    emotion: float | None = None,
) -> Forecast | None:
    """Tomorrow's phase. ``None`` until three sessions of history exist."""
    if len(emotions) < 3:
        return None
    recent = list(emotions)[-num_days:]
    phase = phases[-1]
    slope = slope_of(recent)
    last_change = recent[-1] - recent[-2] if len(recent) >= 2 else 0.0
    level = recent[-1] if emotion is None else emotion

    if phase == "ice":
        if slope > 2 or last_change > 5:
            return Forecast("repair", "ice", 0.70, "冰点后情绪快速回暖，大概率进入修复期", "关注竞价高开和量能放大")
        if slope > 0:
            return Forecast("repair", "ice", 0.55, "冰点后情绪缓慢回升，可能进入弱修复", "观察开盘后能否站稳分时均线")
        return Forecast("ice", "repair", 0.60, "冰点后情绪继续低迷，可能延续冰点", "等待放量杀跌后的企稳信号")

    if phase == "repair":
        if slope > 3 or last_change > 8:
            return Forecast("rise", "repair", 0.70, "修复期情绪加速上升，有望进入上升期", "关注涨停家数和封板率")
        if slope > 0:
            return Forecast("repair", "divergence", 0.60, "修复期情绪温和回升，可能持续修复", "观察上涨比是否持续改善")
        return Forecast("divergence", "repair", 0.55, "修复期情绪出现回落，可能转为分歧", "注意前期强势股是否出现亏钱效应")

    if phase == "rise":
        if slope > 5 or last_change > 10:
            return Forecast("climax", "rise", 0.70, "上升期情绪加速，大概率进入高潮期", "高潮期需警惕，逐步止盈")
        if slope > 0:
            return Forecast("rise", "climax", 0.65, "上升期情绪平稳上升，可能持续", "持股为主，观察是否出现高潮信号")
        return Forecast("divergence", "rise", 0.60, "上升期情绪出现滞涨，可能转入分歧", "减仓观望，等待方向选择")

    if phase == "climax":
        if slope < -3 or last_change < -5:
            return Forecast("divergence", "fall", 0.70, "高潮后情绪快速回落，大概率进入分歧期", "竞价若低开且量能萎缩，则确认分歧")
        if slope < 0:
            return Forecast("divergence", "climax", 0.60, "高潮后情绪缓慢回落，可能进入分歧", "关注龙头股表现，若走弱则减仓")
        return Forecast("climax", "divergence", 0.50, "高潮期情绪仍强，可能延续，但风险加大", "高潮延续需放量，否则为强弩之末")

    if phase == "divergence":
        if slope > 2 or last_change > 5:
            return Forecast("repair", "rise", 0.70, "分歧后情绪回升，可能转为修复", "关注竞价爆量高开，确认转一致")
        if slope > 0:
            return Forecast("repair", "divergence", 0.55, "分歧后情绪小幅回暖，可能弱修复", "观察封板率是否改善")
        return Forecast("fall", "ice", 0.65, "分歧后情绪继续下行，大概率进入退潮期", "空仓回避，等待冰点")

    if phase == "fall":
        if slope > 1 or last_change > 2:
            return Forecast("ice", "repair", 0.60, "退潮期情绪下跌趋缓，可能进入冰点", "冰点后可左侧布局，但需等待明确信号")
        return Forecast("fall", "ice", 0.60, "退潮期延续，继续下跌", "严控仓位，不轻易抄底")

    # normal
    if level > 55:
        return Forecast("rise", "repair", 0.50, "正常期，情绪偏强，可能转入上升", "观察开盘后30分钟情绪走向")
    if level < 45:
        return Forecast("repair", "ice", 0.50, "正常期，情绪偏弱，可能转入修复", "观察开盘后30分钟情绪走向")
    return Forecast("normal", "repair", 0.50, "正常期，方向不明，需结合竞价判断", "观察开盘后30分钟情绪走向")


def evaluate(emotions: Sequence[float], phases: Sequence[str], **kwargs) -> Evaluation:
    """Walk-forward: predict each day from the data before it, score it."""
    result = Evaluation()
    for t in range(2, len(emotions) - 1):
        forecast = predict(emotions[: t + 1], phases[: t + 1], **kwargs)
        if forecast is None:
            continue
        actual = phases[t + 1]
        hit = forecast.main == actual
        base_hit = phases[t] == actual

        result.n += 1
        result.hits += hit
        result.baseline_hits += base_hit
        if actual != phases[t]:
            result.transition_n += 1
            result.transition_hits += hit
        result.confusion[(phases[t], actual)] = result.confusion.get((phases[t], actual), 0) + 1
        key = (forecast.main, actual)
        result.forecast_confusion[key] = result.forecast_confusion.get(key, 0) + 1

        row = result.by_phase.setdefault(phases[t], [0, 0, 0])
        row[0] += 1
        row[1] += hit
        row[2] += base_hit

        conf = result.by_confidence.setdefault(forecast.confidence, [0, 0])
        conf[0] += 1
        conf[1] += hit
    return result
