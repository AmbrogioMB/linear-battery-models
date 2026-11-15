"""Figure of E2: share of the benefit of a perfectly operated battery lost by each policy.

One panel per objective, one row per policy, one marker per case
(gap_share = (J - J_PF) / (J_NB - J_PF)). Values beyond the axis are drawn at
the right edge with an open marker. Reads results/summary/closed_loop.csv.

Usage: python code/plot_closed_loop.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lb import data
from plotstyle import MUTED, OBJ_LABEL, POLICY_LABEL, TYPE_STYLE, WIDTH, save, setup

ORDER = ["AB", "EV-RT", "2S-RT", "2S-RT-nofc", "2S-RH", "OR-RT", "HY-EV-DA", "HY-EV-1H", "HY-OR-1H"]
XMAX = {"aef": 16, "mef": 50, "cost": 60}  # percent


def main():
    setup()
    df = pd.read_csv(data.ROOT / "results" / "summary" / "closed_loop.csv")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.9), sharey=True)
    y = {p: i for i, p in enumerate(ORDER)}
    rng = np.random.default_rng(0)
    for ax, obj in zip(axes, ("aef", "mef", "cost"), strict=True):
        s = df[(df.objective == obj) & df.policy.isin(ORDER)]
        xmax = XMAX[obj]
        for typ, st in TYPE_STYLE.items():
            t = s[s.type == typ]
            x = 100 * t.gap_share.to_numpy()
            yy = t.policy.map(y).to_numpy() + (0.12 if typ == "apartment" else -0.12) + rng.uniform(-0.05, 0.05, len(t))
            inside = x <= xmax
            kw = dict(marker=st["marker"], s=14, linewidths=0.8)
            ax.scatter(x[inside], yy[inside], color=st["color"], label=st["label"], **kw)
            ax.scatter(np.full((~inside).sum(), xmax), yy[~inside], facecolors="white", edgecolors=st["color"], **kw)
        ax.axhline(y["AB"] + 0.5, color=MUTED, lw=0.5)
        ax.set_xlim(-0.03 * xmax, xmax * 1.03)
        ax.set_title(OBJ_LABEL[obj])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(ORDER)), [POLICY_LABEL[p] for p in ORDER])
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    xlabel = "Benefit of a perfectly operated battery lost by the policy (\\%; the scales differ across panels)"
    fig.supxlabel(xlabel, y=-0.02, fontsize=8)
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.15), handletextpad=0.1)
    save(fig, "closed_loop")


if __name__ == "__main__":
    main()
