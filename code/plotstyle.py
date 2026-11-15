"""Common style of the figures of the paper (PDF with LaTeX fonts).

Colours: first three categorical slots of the reference palette of the
dataviz skill, validated for colour-vision deficiency on a white surface
(all pairs). They are always paired with a second encoding (marker shape or
panel title), since the third one has low contrast on white.
"""

from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
# inside the paper project the figures go straight into the LaTeX source; elsewhere into figures/
FIG_DIR = (
    _ROOT.parents[1] / "articolo" / "arxiv" / "figures" if _ROOT.parent.name == "esperimenti" else _ROOT / "figures"
)
WIDTH = 6.3  # inches, text width of the paper

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
TYPE_STYLE = {  # household type -> colour and marker
    "house": dict(color=BLUE, marker="o", label="Houses"),
    "apartment": dict(color=ORANGE, marker="^", label="Apartments"),
}
OBJ_LABEL = {"aef": "Average emission factors", "mef": "Marginal emission factors", "cost": "Cost"}
POLICY_LABEL = {  # code name -> paper name
    "AB": "SC",
    "EV-RT": "MPC-D",
    "2S-RT": "MPC-S",
    "2S-RT-nofc": "MPC-S0",
    "2S-RH": "MPC-SL",
    "OR-RT": "MPC-P",
    "HY-EV-DA": "HYB-24h",
    "HY-EV-1H": "HYB-1h",
    "HY-OR-1H": "HYB-P",
    "PF": "PF",
    "NB": "NB",
}


def setup():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.2,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"written {path}")
