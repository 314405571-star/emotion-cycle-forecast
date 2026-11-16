"""The five demo figures. Matplotlib only, fixed data, no network."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Patch  # noqa: E402

SHORT = {
    "ice": "Ice", "repair": "Rep", "rise": "Rise", "climax": "Clim",
    "divergence": "Div", "fall": "Fall", "normal": "Norm",
}

PHASE_EN = {
    "ice": "Ice", "repair": "Repair", "rise": "Rise", "climax": "Climax",
    "divergence": "Divergence", "fall": "Fall", "normal": "Normal",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

#: Ordered hot/cold ramp: ice is the cold pole, climax the hot pole.
PHASE_COLOR = {
    "ice": "#184f95",
    "repair": "#3987e5",
    "rise": "#e8a33d",
    "climax": "#c0392b",
    "divergence": "#d97757",
    "fall": "#8c4a3a",
    "normal": "#898781",
}

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECOND,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": SECOND,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
}


def _fig(w, h):
    plt.rcParams.update(STYLE)
    return plt.figure(figsize=(w, h), dpi=170)


def _clean(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)


def _header(fig, title, subtitle, meta="", x=0.06, y=0.965):
    fig.text(x, y, title, fontsize=17, fontweight="semibold", color=INK,
             ha="left", va="top")
    fig.text(x, y - 0.055, subtitle, fontsize=9.6, color=SECOND, ha="left", va="top")
    if meta:
        fig.text(0.985, y, meta, fontsize=8.2, color=MUTED, ha="right", va="top",
                 family="DejaVu Sans Mono")


def _ribbon(ax, days, key, height=1.0):
    """Coloured blocks for a phase sequence; labels where the block is wide."""
    start = 0
    for i in range(1, len(days) + 1):
        if i == len(days) or days[i].__getattribute__(key) != days[start].__getattribute__(key):
            phase = days[start].__getattribute__(key)
            width = i - start
            ax.barh(0.5, width, left=start, height=height,
                    color=PHASE_COLOR[phase], edgecolor=SURFACE, linewidth=0.8)
            if width >= 9:
                ax.text(start + width / 2, 0.5, SHORT[phase], ha="center",
                        va="center", fontsize=7.4, color="white")
            start = i
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(days))
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


# ------------------------------------------------------------------ figure 1
def hero(days, path: Path) -> None:
    n = len(days)
    x = np.arange(n)
    agreement = sum(1 for d in days if d.phase == d.latent) / n

    fig = _fig(14.0, 7.6)
    ax_lat = fig.add_axes([0.075, 0.858, 0.895, 0.026])
    ax_inf = fig.add_axes([0.075, 0.805, 0.895, 0.026])
    ax_emo = fig.add_axes([0.075, 0.40, 0.895, 0.35])
    ax_pos = fig.add_axes([0.075, 0.12, 0.895, 0.17])

    for ax, key, label in ((ax_lat, "latent", "hidden"), (ax_inf, "phase", "inferred")):
        _ribbon(ax, days, key)
        ax.set_yticks([0.5])
        ax.set_yticklabels([label], fontsize=7.4)
        ax.tick_params(axis="y", length=0, pad=2)

    ax_emo.plot(x, [d.emotion for d in days], color=INK, linewidth=1.4, zorder=4)
    ax_emo.fill_between(x, 0, [d.emotion for d in days], color=BLUE, alpha=0.10,
                        linewidth=0, zorder=2)
    for level, label in ((30, "ice ≤ 30"), (75, "climax ≥ 75")):
        ax_emo.axhline(level, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=3)
        ax_emo.text(n - 1, level + 1.6, label, fontsize=7.8, color=MUTED, ha="right")
    ax_emo.set_ylim(0, 100)
    ax_emo.set_xlim(0, n)
    ax_emo.set_ylabel("emotion index (0–100)", fontsize=8.8)
    ax_emo.set_xticks([])
    _clean(ax_emo)

    ax_pos.step(x, [d.position for d in days], where="post", color=INK, linewidth=1.4)
    ax_pos.fill_between(x, 0, [d.position for d in days], step="post", color=ORANGE,
                        alpha=0.16, linewidth=0)
    ax_pos.set_ylim(0, 0.92)
    ax_pos.set_xlim(0, n)
    ax_pos.set_ylabel("position ceiling", fontsize=8.8)
    ax_pos.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _clean(ax_pos)
    ticks = [i for i in range(0, n, 40)]
    ax_pos.set_xticks(ticks)
    ax_pos.set_xticklabels([days[i].date[5:] for i in ticks], fontsize=8.0)
    ax_pos.set_xlabel("2025 sessions", fontsize=8.6)

    handles = [Patch(facecolor=PHASE_COLOR[p], edgecolor="none", label=PHASE_EN[p])
               for p in ("ice", "repair", "rise", "climax", "divergence", "fall", "normal")]
    ax_emo.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
                  ncol=7, frameon=False, fontsize=7.8, handlelength=1.1,
                  handleheight=0.9, columnspacing=1.3)

    _header(fig, "One line, one cycle",
            "the index reads breadth; the phase machine turns it into a position ceiling",
            f"agreement {agreement:.0%} vs {1/7:.0%} chance · synthetic", y=0.975)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def indicators(days, path: Path) -> None:
    n = len(days)
    x = np.arange(n)
    series = [
        ("up ratio", [d.breadth.up_ratio for d in days], "share closing up", BLUE, 1.0),
        ("limit ratio", [d.breadth.limit_ratio for d in days], "share closing at the limit", ORANGE, 0.10),
        ("seal rate", [(d.breadth.seal_rate or 0.0) for d in days], "sealed / touched", AQUA, 1.0),
        ("big-loss ratio", [d.breadth.big_loss_ratio for d in days], "share down ≥ 7%", "#c0392b", 0.20),
    ]
    fig = _fig(13.0, 7.2)
    top = 0.765
    for i, (label, values, note, color, cap) in enumerate(series):
        ax = fig.add_axes([0.075, top - i * 0.145, 0.885, 0.105])
        ax.fill_between(x, 0, values, color=color, alpha=0.16, linewidth=0)
        ax.plot(x, values, color=color, linewidth=1.2)
        ax.set_xlim(0, n)
        ax.set_ylim(0, cap)
        ax.set_yticks([0, cap])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7.2, length=0, pad=2)
        ax.text(0.002, 0.95, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.4, color=color, fontweight="semibold")
        ax.text(0.105, 0.95, f"· {note}", transform=ax.transAxes, ha="left", va="top",
                fontsize=7.6, color=MUTED)
        _clean(ax)

    ax = fig.add_axes([0.075, 0.095, 0.885, 0.13])
    ax.fill_between(x, 0, [d.emotion for d in days], color=INK, alpha=0.10, linewidth=0)
    ax.plot(x, [d.emotion for d in days], color=INK, linewidth=1.4)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis="y", labelsize=7.2, length=0, pad=2)
    ax.text(0.002, 0.94, "emotion index", transform=ax.transAxes, ha="left", va="top",
            fontsize=8.4, color=INK, fontweight="semibold")
    ax.set_xticks([i for i in range(0, n, 40)])
    ax.set_xticklabels([days[i].date[5:] for i in ticks_of(days)], fontsize=7.8)
    _clean(ax)
    ax.set_xlabel("2025 sessions", fontsize=8.6)

    _header(fig, "Four numbers, one index",
            "each sub-score is rescaled to 0–100 and weighted 35 / 25 / 25 / 15",
            "synthetic · 260 sessions")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def ticks_of(days, step=40):
    return list(range(0, len(days), step))


# ------------------------------------------------------------------ figure 3
def transitions(days, path: Path) -> None:
    order = ["ice", "repair", "rise", "climax", "divergence", "fall", "normal"]
    counts = {a: {b: 0 for b in order} for a in order}
    latent_counts = {a: {b: 0 for b in order} for a in order}
    for a, b in zip(days[:-1], days[1:]):
        counts[a.phase][b.phase] += 1
        latent_counts[a.latent][b.latent] += 1

    fig = _fig(13.2, 6.4)
    ax = fig.add_axes([0.02, 0.03, 0.46, 0.80])
    ax2 = fig.add_axes([0.60, 0.15, 0.30, 0.62])

    angles = {p: math.radians(-90 + 360 * i / len(order)) for i, p in enumerate(order)}
    radius = 0.78
    pos = {p: (radius * math.cos(a), radius * math.sin(a)) for p, a in angles.items()}
    cmap = LinearSegmentedColormap.from_list("cycle", ["#e1e0d9", "#3987e5", "#184f95"])

    cells = [(a, b) for a in order for b in order if a != b and counts[a][b] > 0]
    peak = max([counts[a][b] for a, b in cells] + [1])
    for a, b in cells:
        value = counts[a][b]
        width = 0.5 + 4.0 * math.sqrt(value / peak)
        color = cmap(0.15 + 0.85 * min(1.0, value / peak))
        arrow = FancyArrowPatch(pos[a], pos[b], connectionstyle="arc3,rad=0.13",
                                arrowstyle="-|>", mutation_scale=10,
                                linewidth=width, color=color, alpha=0.85,
                                shrinkA=22, shrinkB=22, zorder=2)
        ax.add_patch(arrow)

    for p in order:
        x0, y0 = pos[p]
        ax.add_patch(Circle((x0, y0), 0.13, facecolor=PHASE_COLOR[p],
                            edgecolor=SURFACE, linewidth=2.0, zorder=4))
        ink = INK if p in ("rise", "divergence", "normal") else "white"
        ax.text(x0, y0, PHASE_EN[p], ha="center", va="center", fontsize=8.2,
                color=ink, zorder=5)
        stay = counts[p][p]
        if stay:
            ang = angles[p]
            lx, ly = math.cos(ang) * (radius + 0.30), math.sin(ang) * (radius + 0.30)
            ax.add_patch(Circle((lx, ly), 0.075, facecolor=SURFACE,
                                edgecolor=PHASE_COLOR[p], linewidth=1.2, zorder=4))
            ax.text(lx, ly, str(stay), ha="center", va="center", fontsize=7.0,
                    color=SECOND, zorder=5)

    ax.set_xlim(-1.16, 1.16)
    ax.set_ylim(-1.14, 1.14)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("inferred transitions (counts; rings are self-loops)", loc="left",
                 fontsize=10.2, fontweight="semibold", color=INK, x=-0.02, pad=4)

    matrix = np.array([[counts[a][b] for b in order] for a in order], dtype=float)
    latent = np.array([[latent_counts[a][b] for b in order] for a in order], dtype=float)
    for m in (matrix, latent):
        row = m.sum(axis=1, keepdims=True)
        m[:] = np.divide(m, row, out=np.zeros_like(m), where=row > 0)
    ax2.imshow(matrix, cmap=cmap, vmin=0, vmax=0.9)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels([PHASE_EN[p] for p in order], fontsize=7.6, rotation=40,
                        ha="right")
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([PHASE_EN[p] for p in order], fontsize=7.6)
    ax2.set_title("row-normalised, inferred", loc="left", fontsize=9.2,
                  fontweight="semibold", color=INK, pad=6)
    for i in range(len(order)):
        for j in range(len(order)):
            if matrix[i, j] >= 0.08:
                ax2.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                         fontsize=6.4, color="white" if matrix[i, j] > 0.45 else SECOND)
    ax2.tick_params(length=0)
    for side in ax2.spines.values():
        side.set_visible(False)

    _header(fig, "The machine has a cycle, not a random walk",
            "arrow width = how often the inferred phase moved from one to the next",
            "synthetic · 260 sessions", x=0.035)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ figure 4
def forecast(days, evaluation, path: Path) -> None:
    order = ["ice", "repair", "rise", "climax", "divergence", "fall", "normal"]
    matrix = np.zeros((len(order), len(order)))
    for (a, b), count in evaluation.forecast_confusion.items():
        if a in order and b in order:
            matrix[order.index(a), order.index(b)] += count

    fig = _fig(12.4, 5.2)
    ax = fig.add_axes([0.10, 0.18, 0.34, 0.58])
    ax2 = fig.add_axes([0.56, 0.18, 0.38, 0.58])

    cmap = LinearSegmentedColormap.from_list("conf", [SURFACE, "#9ec5f4", "#184f95"])
    ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([PHASE_EN[p] for p in order], fontsize=7.4, rotation=40, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([PHASE_EN[p] for p in order], fontsize=7.4)
    ax.set_xlabel("actual next phase", fontsize=8.4)
    ax.set_ylabel("predicted next phase", fontsize=8.4)
    peak = matrix.max() if matrix.size else 1
    for i in range(len(order)):
        for j in range(len(order)):
            if matrix[i, j]:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                        fontsize=6.8,
                        color="white" if matrix[i, j] > peak * 0.55 else SECOND)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("where the forecast lands", loc="left", fontsize=9.8,
                 fontweight="semibold", color=INK, pad=8)

    rows = [p for p in order if p in evaluation.by_phase]
    values = [evaluation.by_phase[p][1] / evaluation.by_phase[p][0] for p in rows]
    bases = [evaluation.by_phase[p][2] / evaluation.by_phase[p][0] for p in rows]
    y = np.arange(len(rows))[::-1]
    ax2.barh(y, values, height=0.55, color=BLUE, edgecolor="none", zorder=3)
    ax2.scatter(bases, y, s=34, facecolor=SURFACE, edgecolor=ORANGE, linewidth=1.4,
                zorder=5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(
        [f"{PHASE_EN[p]}   {v:.0%}   n={evaluation.by_phase[p][0]}"
         for p, v in zip(rows, values)], fontsize=7.8)
    ax2.set_xlim(0, 1.12)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_xlabel("next-day accuracy", fontsize=8.4)
    ax2.axvline(evaluation.accuracy, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax2.axvline(evaluation.baseline, color=ORANGE, linewidth=1.0, linestyle=(0, (4, 3)))
    _clean(ax2, "x")
    ax2.tick_params(axis="y", length=0)
    ax2.legend(handles=[
        Patch(facecolor=BLUE, label=f"rule table {evaluation.accuracy:.0%}"),
        Line2D([], [], marker="o", markersize=6, linestyle="none",
               markerfacecolor=SURFACE, markeredgecolor=ORANGE,
               label=f"persistence {evaluation.baseline:.0%}"),
    ], loc="lower right", frameon=False, fontsize=7.6, handlelength=1.2)

    _header(fig, "The rule table, measured",
            f"it trails “no change” on every phase, and catches "
            f"{evaluation.transition_accuracy:.0%} of the {evaluation.transition_n} turns",
            f"n = {evaluation.n} · synthetic")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ figure 5
def calibration(evaluation, path: Path) -> None:
    buckets = sorted(evaluation.by_confidence)
    stated = [b for b in buckets if evaluation.by_confidence[b][0] >= 5]
    realised = [evaluation.by_confidence[b][1] / evaluation.by_confidence[b][0] for b in stated]
    counts = [evaluation.by_confidence[b][0] for b in stated]
    base = evaluation.accuracy

    fig = _fig(8.6, 4.8)
    ax = fig.add_axes([0.10, 0.16, 0.84, 0.58])
    ax.plot([0.4, 0.8], [0.4, 0.8], color=BASELINE, linewidth=1.0,
            linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.78, 0.79, "perfect calibration", fontsize=7.8, color=MUTED,
            ha="right", va="bottom")
    ax.axhline(base, color=MUTED, linewidth=1.0, zorder=2)
    ax.text(0.755, base + 0.010, f"overall {base:.0%}", fontsize=7.8, color=SECOND,
            ha="right")

    ax.scatter(stated, realised, s=[18 + 0.25 * c for c in counts], color=BLUE,
               edgecolor=SURFACE, linewidth=1.2, zorder=4)
    for x, y, c in zip(stated, realised, counts):
        ax.annotate(f"n={c}", (x, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=7.2, color=MUTED)
    ax.set_xlim(0.44, 0.76)
    ax.set_ylim(0.0, 0.9)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("confidence the table states", fontsize=8.6)
    ax.set_ylabel("share that came true", fontsize=8.6)
    _clean(ax, "both")

    _header(fig, "Do the confidences mean anything?",
            "each dot is a branch of the rule table; below the diagonal it is overclaiming",
            f"n = {evaluation.n} · synthetic")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
