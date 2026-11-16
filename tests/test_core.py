"""Regression tests for breadth, the index and the phase machine.

Run with:  python tests/test_core.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.emotioncycle.breadth import Breadth, StockDay, measure  # noqa: E402
from src.emotioncycle.index import (  # noqa: E402
    Config,
    blend,
    clamp,
    emotion_index,
    pinned,
    subscores,
)
from src.emotioncycle.phases import (  # noqa: E402
    POSITIONS,
    determine_phase,
    early_signal,
    position_for,
    turning_points,
)


def breadth(up=0.50, limit=0.010, seal=0.60, loss=0.020, strong=0.03, mid=0.08,
            total=800) -> Breadth:
    """A Breadth with the requested ratios (counts are what the model reads)."""
    n_limit = round(limit * total)
    n_touch = max(n_limit, round(n_limit / seal)) if seal else n_limit
    return Breadth(
        total=total, up=round(up * total), limit_up=n_limit, touched=n_touch,
        big_loss=round(loss * total), strong=round(strong * total),
        mid=round(mid * total),
    )


# --------------------------------------------------------------------- breadth
def test_measure_counts_the_four_numbers():
    days = [
        StockDay(0.10, sealed=True, touched=True),
        StockDay(0.04, sealed=False, touched=True),
        StockDay(-0.08),
        StockDay(0.03),
        StockDay(-0.01),
    ]
    b = measure(days)
    assert (b.total, b.up, b.limit_up, b.touched, b.big_loss) == (5, 3, 1, 2, 1)
    assert abs(b.up_ratio - 0.6) < 1e-12
    assert abs(b.limit_ratio - 0.2) < 1e-12
    assert abs(b.seal_rate - 0.5) < 1e-12


def test_seal_rate_is_none_when_nothing_touched():
    b = measure([StockDay(0.01), StockDay(-0.02)])
    assert b.seal_rate is None
    # The notebook returns 0.5 here, which silently reads as a neutral seal
    # score on a day with no limit-up attempts at all.
    cfg = Config()
    assert subscores(b, cfg)["seal"] == cfg.neutral_seal


def test_diffusion_needs_a_strong_name():
    assert measure([StockDay(0.03), StockDay(0.04)]).diffusion is None
    assert abs(measure([StockDay(0.08), StockDay(0.03)]).diffusion - 100.0) < 1e-9


# ----------------------------------------------------------------------- index
def test_big_loss_score_scales_with_sample_size():
    # 5% of the market down 7% is the floor of the loss score, whatever the
    # sample size. The notebook divides the *count* by 40, so the same market
    # scores differently at 400 and 800 names.
    small = breadth(total=400, loss=0.05)
    big = breadth(total=800, loss=0.05)
    assert subscores(small)["loss"] == subscores(big)["loss"] == 0.0

    legacy_small = max(0.0, 1 - 20 / 40) * 100     # 20 of 400
    legacy_big = max(0.0, 1 - 40 / 40) * 100       # 40 of 800
    assert (legacy_small, legacy_big) == (50.0, 0.0)


def test_subscores_are_clipped():
    parts = subscores(breadth(up=0.95, limit=0.20, seal=0.99, loss=0.0))
    assert 0 <= parts["up"] <= 100
    assert parts["limit"] == 100.0
    assert parts["loss"] == 100.0


def test_emotion_is_bounded_and_ordered():
    calm = emotion_index(breadth(up=0.2, limit=0.002, seal=0.3, loss=0.08))
    hot = emotion_index(breadth(up=0.8, limit=0.05, seal=0.9, loss=0.005))
    assert 0 <= calm < 40 < 60 < hot <= 100


def test_blend_is_continuous_at_the_notebook_gate():
    # The notebook uses the raw value from confidence 0.60 upwards and a blend
    # below it, so the output jumps by 0.41·(raw - yesterday) at the boundary.
    yesterday, raw = 30.0, 80.0
    below = blend(yesterday, raw, 0.5999)
    above = blend(yesterday, raw, 0.6001)
    assert abs(above - below) < 0.1

    legacy_below = yesterday * (1 - 0.5999) + raw * 0.5999
    legacy_above = raw
    assert abs(legacy_above - legacy_below) > 15


def test_clamp_and_pinned():
    assert clamp(50.0, 60.0) == 60.0
    assert clamp(50.0, 90.0) == 65.0
    assert pinned(50.0, 90.0) is True
    assert pinned(50.0, 60.0) is False


# ---------------------------------------------------------------------- phases
def test_climax_and_ice_come_from_breadth():
    cfg = Config()
    hot = breadth(up=0.65, limit=0.05, seal=0.75)
    dead = breadth(up=0.18, limit=0.002, seal=0.3, loss=0.09)
    assert determine_phase(50, 50, "rise", hot, cfg) == "climax"
    assert determine_phase(50, 50, "repair", dead, cfg) == "ice"


def test_climax_never_jumps_straight_to_fall():
    # A 4-point drop after the peak: the notebook's -5 rule for divergence does
    # not fire, its -3 rule for fall does, and the cycle order is broken.
    cfg = Config()
    mild = breadth(up=0.5, limit=0.02, seal=0.6)
    assert determine_phase(70, 74, "climax", mild, cfg) == "divergence"
    # The notebook would have returned "fall" here.
    legacy = "fall" if 70 < 74 - 3 and 70 > 30 else "climax"
    assert legacy == "fall"


def test_gradual_recovery_reaches_repair():
    # +3 a day never satisfies "more than +5 in one day"; three steady days do.
    cfg = Config()
    improving = breadth(up=0.5, limit=0.015, seal=0.55, loss=0.03)
    assert determine_phase(23, 20, "fall", improving, cfg, trend3=0.0) == "fall"
    assert determine_phase(23, 20, "fall", improving, cfg, trend3=8.0) == "repair"


def test_divergence_can_follow_rise():
    cfg = Config()
    cooling = breadth(up=0.45, limit=0.02, seal=0.5, loss=0.04)
    assert determine_phase(50, 58, "rise", cooling, cfg) == "divergence"


def test_rise_needs_repair_first():
    cfg = Config()
    mild = breadth(up=0.5, limit=0.015, seal=0.55)
    # fall → rise would skip the repair rung
    assert determine_phase(40, 30, "fall", mild, cfg) != "rise"
    assert determine_phase(40, 30, "repair", mild, cfg) == "rise"


def test_turning_points_need_yesterday():
    cfg = Config()
    today = breadth(up=0.5, limit=0.03, seal=0.7)
    yesterday = breadth(up=0.42, limit=0.01, seal=0.5)
    signals = turning_points("divergence", "rise", today, yesterday)
    assert [s.kind for s in signals] == ["turn_strong"]

    ice_day = breadth(up=0.45, limit=0.005, seal=0.3, loss=0.06)
    signals = turning_points("ice", "repair", ice_day, yesterday)
    assert [s.kind for s in signals] == ["turn_repair"]
    assert turning_points("ice", "repair", ice_day, None) == ()


def test_early_signal_needs_extreme_diffusion():
    assert early_signal("climax", 30.0).kind == "climax"
    assert early_signal("ice", 60.0).kind == "ice"
    assert early_signal("climax", 60.0) is None
    assert early_signal("ice", None) is None


def test_positions_are_a_ceiling():
    assert POSITIONS["rise"] == max(POSITIONS.values())
    assert position_for("fall") == POSITIONS["fall"]
    assert 0 <= position_for("unknown") <= 1


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
