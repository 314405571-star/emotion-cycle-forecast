"""End-to-end test of the stateful monitor.

Run with:  python tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.emotioncycle.breadth import Breadth  # noqa: E402
from src.emotioncycle.monitor import Monitor  # noqa: E402


def breadth(up, limit, seal, loss, total=800) -> Breadth:
    n_limit = round(limit * total)
    n_touch = max(n_limit, round(n_limit / seal)) if seal else n_limit
    return Breadth(total=total, up=round(up * total), limit_up=n_limit,
                   touched=n_touch, big_loss=round(loss * total),
                   strong=round(0.03 * total), mid=round(0.08 * total))


def test_monitor_walks_a_cycle():
    monitor = Monitor()
    hot = breadth(0.70, 0.045, 0.80, 0.015)
    dead = breadth(0.18, 0.002, 0.35, 0.090)
    normal = breadth(0.50, 0.012, 0.60, 0.030)

    states = []
    for day, b in zip(("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"),
                      (normal, hot, hot, dead)):
        states.append(monitor.update(day, b, confidence=0.98))

    assert states[1].phase == "climax"
    assert states[3].phase == "ice"
    assert states[1].position == 0.40          # climax caps the position
    assert states[3].position == 0.20
    assert 0 <= states[3].emotion <= 100


def test_one_update_per_session():
    monitor = Monitor()
    b = breadth(0.5, 0.01, 0.6, 0.03)
    monitor.update("2026-01-05", b)
    try:
        monitor.update("2026-01-05", b)
    except ValueError as exc:
        assert "one update per session" in str(exc)
    else:
        raise AssertionError("a second update for the same session must fail")


def test_forecast_needs_history_and_then_appears():
    monitor = Monitor()
    b = breadth(0.5, 0.01, 0.6, 0.03)
    assert monitor.forecast() is None
    for i, day in enumerate(["2026-01-05", "2026-01-06", "2026-01-07"]):
        monitor.update(day, b)
    assert monitor.forecast() is not None
    assert "tomorrow" in monitor.summary()


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
