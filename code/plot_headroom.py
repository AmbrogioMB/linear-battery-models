"""Figure of E1: relative headroom of optimization vs self-sufficiency of the rule.

One panel per objective; one point per household and PV/battery configuration
in Milan. Reads results/summary/headroom.csv (run summarize.py first).

Usage: python code/plot_headroom.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from lb import data
from plotstyle import OBJ_LABEL, TYPE_STYLE, WIDTH, save, setup


def main():
    setup()
    df = pd.read_csv(data.ROOT / "results" / "summary" / "headroom.csv")
    df = df[df.site == "milan"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3), sharex=True, sharey=True)
    for ax, obj in zip(axes, ("aef", "mef", "cost"), strict=True):
        s = df[df.objective == obj]
        for typ in ("apartment", "house"):  # houses on top: they are fewer
            st = TYPE_STYLE[typ]
            t = s[s.type == typ]
            ax.scatter(
                100 * t.ss_ab,
                100 * t.gap_share,
                s=4,
                marker=st["marker"],
                color=st["color"],
                alpha=0.35,
                linewidths=0,
                label=st["label"],
                rasterized=True,
            )
        ax.set_title(OBJ_LABEL[obj])
        ax.set_xlabel("Self-sufficiency with SC (\\%)")
        ax.set_xlim(10, 101)
        ax.set_ylim(-2, 112)
    axes[0].set_ylabel("Relative headroom (\\%)")
    h, lab = axes[0].get_legend_handles_labels()
    leg = axes[0].legend(h[::-1], lab[::-1], loc="upper right", markerscale=3, handletextpad=0.2)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    save(fig, "headroom")


if __name__ == "__main__":
    main()
