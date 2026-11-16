"""Regression tests for the next-day rule table and its evaluator.

Run with:  python tests/test_forecast.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.emotioncycle.forecast import (  # noqa: E402
    evaluate,
    predict,
    slope_of,
)


# ------------------------------------------------------------------ the table
def test_no_forecast_before_three_sessions():
    assert predict([50.0, 50.0], ["normal", "normal"]) is None


def test_slope_is_per_session():
    assert abs(slope_of([10.0, 12.0, 14.0])) - 2.0 < 1e-9
    assert slope_of([50.0]) == 0.0


def test_climax_with_a_big_drop_predicts_divergence():
    emotions = [55.0, 62.0, 70.0, 74.0, 60.0]
    phases = ["rise", "rise", "climax", "climax", "climax"]
    forecast = predict(emotions, phases)
    assert forecast.main == "divergence"
    assert forecast.alt == "fall"
    assert forecast.confidence == 0.70


def test_normal_uses_the_level():
    forecast = predict([40.0, 45.0, 60.0], ["normal", "normal", "normal"])
    assert forecast.main == "rise"
    low = predict([60.0, 55.0, 40.0], ["normal", "normal", "normal"])
    assert low.main == "repair"
    mid = predict([50.0, 50.0, 50.0], ["normal", "normal", "normal"])
    assert mid.main == "normal"


def test_ice_with_a_positive_slope_predicts_repair():
    forecast = predict([20.0, 22.0, 30.0], ["ice", "ice", "ice"])
    assert forecast.main == "repair"
    assert forecast.alt == "ice"


def test_evaluate_matches_predict_step_by_step():
    # Guards the evaluator's indexing: every scored day must be exactly the
    # forecast that ``predict`` returns for the data before it.
    emotions = [20.0, 22.0, 25.0, 30.0, 40.0, 55.0, 70.0, 72.0]
    phases = ["ice", "ice", "repair", "rise", "rise", "rise", "climax", "climax"]
    result = evaluate(emotions, phases)

    hits = n = 0
    for t in range(2, len(emotions) - 1):
        forecast = predict(emotions[: t + 1], phases[: t + 1])
        if forecast is None:
            continue
        n += 1
        hits += forecast.main == phases[t + 1]
    assert (result.n, result.hits) == (n, hits)


# ----------------------------------------------------------------- the evaluator
def test_evaluate_scores_the_table_against_persistence():
    emotions = [20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
    phases = ["ice"] * 6
    result = evaluate(emotions, phases)
    assert result.n == 3                       # t = 2, 3, 4
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.baseline <= 1.0
    assert result.transition_n == 0            # nothing changed
    assert sum(v[0] for v in result.by_confidence.values()) == 3


def test_transition_days_are_counted_separately():
    emotions = [20.0, 25.0, 30.0, 40.0, 50.0]
    phases = ["ice", "repair", "rise", "climax", "fall"]
    result = evaluate(emotions, phases)
    assert result.n == 2                       # t = 2, 3
    assert result.transition_n == 2            # both scored days changed phase
    assert result.baseline == 0.0              # persistence cannot hit a turn


def test_confusion_covers_every_scored_day():
    emotions = [20.0, 24.0, 31.0, 45.0, 60.0, 74.0, 78.0]
    phases = ["ice", "ice", "repair", "rise", "rise", "climax", "climax"]
    result = evaluate(emotions, phases)
    assert sum(result.confusion.values()) == result.n
    assert sum(result.forecast_confusion.values()) == result.n


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
