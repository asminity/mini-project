"""Fig. 1 of the manuscript: the three layers and the shared temporal protocol."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, INK2, MUTED, AXIS = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BLUE, ORANGE, LIGHTBLUE = "#2a78d6", "#eb6834", "#9ec5f4"


def draw(width: float):
    fig, ax = plt.subplots(figsize=(width, 2.75))
    ax.set(xlim=(0, 100), ylim=(0, 38.5))
    ax.axis("off")

    def box(x, y, w, h, title, lines, face="#f4f3ef", edge=AXIS):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                    fc=face, ec=edge, lw=0.7))
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=7.4,
                fontweight="bold")
        ax.text(x + w / 2, y + h - 4.3, "\n".join(lines), ha="center", va="top", fontsize=6.0,
                color=INK2, linespacing=1.28)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.8, color=INK2, shrinkA=0, shrinkB=0))

    for x, label in ((11, "Layers and data"), (35.5, "Feature construction"),
                     (63.2, "Shared temporal protocol"), (90.2, "Output")):
        ax.text(x, 38.2, label, ha="center", va="top", fontsize=7.0, color=MUTED)

    rows = [
        (25.5, "Bitcoin transactions", ["Elliptic++: 203,769 nodes,", "182 features, 49 time steps"],
         "Transaction graph", ["degrees, 2-hop reach, component,", "in/out-neighbour feature means"]),
        (14.0, "Bitcoin wallets", ["Elliptic++: 822,942 wallets,", "55 features, 49 time steps"],
         "Address graph", ["degrees, 2-hop reach, component,", "in/out-neighbour feature means"]),
        (2.5, "Network flows", ["UGR'16 sample: 5.0 M flows,", "5 days, 2.42 M IP-minutes"],
         "Window aggregation", ["per source IP and minute:", "volume, fan-out, flags, ports"]),
    ]
    for y, t, l, ft, fl in rows:
        box(0.5, y, 21, 8.6, t, l, face="#eaf2fd", edge=LIGHTBLUE)
        box(24.5, y, 22, 8.6, ft, fl)
        arrow(21.9, y + 4.3, 24.1, y + 4.3)
        arrow(46.9, y + 4.3, 49.3, 18.3 + (y - 14.0) * 0.2)

    ax.add_patch(FancyBboxPatch((49.8, 2.5), 26.8, 31.6, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc="white", ec=AXIS, lw=0.7))
    ax.text(63.2, 33.0, "time", ha="center", va="top", fontsize=6.4, color=MUTED)
    arrow(51.4, 30.4, 75.2, 30.4)
    for x, w, lab, colr in ((51.4, 12.6, "fit", BLUE), (64.2, 4.0, "val", ORANGE), (68.6, 6.6, "test", INK)):
        ax.add_patch(FancyBboxPatch((x, 26.6), w, 2.6, boxstyle="square,pad=0", fc=colr, ec="white", lw=0.8))
        ax.text(x + w / 2, 27.9, lab, ha="center", va="center", fontsize=6.4, color="white")
    steps = ["1. Fit ensemble members on fit steps",
             "2. Early stopping and thresholds",
             "    from validation steps only",
             "3. Refit on fit + validation",
             "4. Score the later test steps once",
             "5. Bootstrap 95% intervals"]
    ax.text(51.2, 24.6, "\n".join(steps), ha="left", va="top", fontsize=6.3, color=INK, linespacing=1.42)
    ax.text(63.2, 7.6, "LightGBM + XGBoost + Random Forest,\nscores averaged over models and seeds",
            ha="center", va="top", fontsize=6.0, color=INK2, linespacing=1.28)
    arrow(77.1, 18.3, 79.6, 18.3)
    box(80.0, 20.5, 19.6, 13.6, "Three-way verdict",
        [r"score $\geq \tau_d$:  positive", r"$\tau_r \leq$ score $< \tau_d$:  suspicious",
         r"score $< \tau_r$:  negative"])
    box(80.0, 2.5, 19.6, 15.2, "Reported evidence",
        ["precision, recall, F1, MCC,", "PR-AUC, ROC-AUC, 95% CI,", "per-step results, ablations,",
         "baselines, drift analyses"])
    return fig
