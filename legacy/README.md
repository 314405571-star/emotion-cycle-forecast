# Legacy: the original module, verbatim

`v2.4.py` is the single code cell of `notebooks/情绪因子2.ipynb`, extracted
without a single edit. Every "the notebook did X" claim in
[`../docs/算法总结.md`](../docs/算法总结.md) can be checked against it.

It is a JoinQuant module (`from jqdata import *`), so it will not import
outside that environment. The class is `EmotionCycle`; the interesting parts
are:

| Method | What it does |
|---|---|
| `_fetch_market_data` | draws 800 names, reads two daily bars, returns four ratios |
| `_calculate_raw_emotion` | the weighted 0–100 index |
| `_determine_phase` | the seven-phase machine |
| `_detect_turning_points` | 分歧转一致 / 冰点转修复 |
| `predict_next_phase` | the next-day rule table |
| `update` / `update_for_research` | intraday and daily entry points |

The file also contains the author's test harness (bottom third), which walks
seven sessions and prints the summary box.

What this repo changes, with a test for each:

- `big_loss_score = max(0, 1 - count/40)` — a count over a constant.
- `seal_rate = 0.5` when nothing touched the limit.
- blending switches off at confidence 0.60 instead of staying continuous.
- the phase machine can jump `climax → fall` and `fall → rise`.
- `repair` needs a +5 one-day jump, so a gradual recovery never leaves `fall`.
- `divergence` is only reachable from `climax`.
- `predict_next_phase` has no evaluator; its confidences are asserted.
- the diffusion history appends on every `update()` call.

None of these raise an exception. That is the point.
