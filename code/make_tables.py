"""LaTeX bodies of the tables of the paper, from results/summary/*.csv.

Writes results/summary/table_headroom.tex and table_closed_loop.tex (only the
rows between \\midrule and \\bottomrule), which are copied into content.tex.

Usage: python code/make_tables.py
"""

import pandas as pd

from lb import data

S = data.ROOT / "results" / "summary"
N_DAYS = 318  # simulated days
BASE = dict(site="milan", kwp=4.5, capacity=10.0)
VARIANTS = [  # (column, changes with respect to the base case)
    ("Rome", dict(site="rome")),
    ("Palermo", dict(site="palermo")),
    ("3 kWp", dict(kwp=3.0)),
    ("6 kWp", dict(kwp=6.0)),
    ("5 kWh", dict(capacity=5.0)),
    ("15 kWh", dict(capacity=15.0)),
]
ROWS = [
    ("aef", "AEF"),
    ("mef", "MEF"),
    ("cost", "Cost"),
    ("cost_w0", "Cost, $w = 0$"),
    ("cost_w10", "Cost, $w = 0.10$"),
]
TYPES = [("house", "houses"), ("apartment", "apartments")]


def pct(v):
    return f"{100 * v:.1f}"


def table_headroom():
    df = pd.read_csv(S / "headroom.csv")

    def sel(obj, typ, **chg):
        cfg = {**BASE, **chg}
        return df[
            (df.objective == obj)
            & (df.type == typ)
            & (df.site == cfg["site"])
            & (df.kwp == cfg["kwp"])
            & (df.capacity == cfg["capacity"])
        ]

    lines = []
    for obj, name in ROWS:
        for typ, tname in TYPES:
            b = sel(obj, typ)
            unit = "kg" if obj in ("aef", "mef") else "\\euro"
            cells = [
                f"{name}, {tname}",
                f"{pct(b.gap_share.median())} ({pct(b.gap_share.quantile(0.25))}--{pct(b.gap_share.quantile(0.75))})",
                f"{b.headroom_abs.median():.0f}~{unit}",
            ]
            for _, chg in VARIANTS:
                v = sel(obj, typ, **chg)
                cells.append(pct(v.gap_share.median()) if len(v) else "--")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    return "\n".join(lines[:-1]) + "\n"


GROUPS = [  # (objective, type, column)
    ("aef", "house", "AEF, houses"),
    ("mef", "house", "MEF, houses"),
    ("cost", "house", "Cost, houses"),
    ("cost", "apartment", "Cost, apartments"),
]
POLICIES = [
    ("EV-RT", "MPC-D"),
    ("2S-RT", "MPC-S"),
    ("2S-RT-nofc", "MPC-S0"),
    ("2S-RH", "MPC-SL"),
    ("OR-RT", "MPC-P"),
    ("HY-EV-DA", "HYB-24h"),
    ("HY-EV-1H", "HYB-1h"),
    ("HY-OR-1H", "HYB-P"),
]


def table_closed_loop():
    df = pd.read_csv(S / "closed_loop.csv")
    lines = []
    # headroom of each group, for reference
    cells = ["SC--PF headroom"]
    for obj, typ, _ in GROUPS:
        g = df[(df.objective == obj) & (df.type == typ) & (df.policy == "PF")]
        unit = "kg" if obj in ("aef", "mef") else "\\euro"
        cells.append(f"{g.headroom_abs.median():.0f}~{unit}")
    lines.append(" & ".join(cells) + " & & \\\\")
    lines.append("\\midrule")
    for code, name in POLICIES:
        cells = [name]
        for obj, typ, _ in GROUPS:
            g = df[(df.objective == obj) & (df.type == typ) & (df.policy == code)]
            r = 100 * g.recovered.median()
            cells.append("$<-100$" if r < -100 else f"${r:.0f}$")
        p = df[(df.policy == code) & df.objective.isin(["aef", "mef", "cost"])]
        cells.append(f"{p.n_solves.median() / N_DAYS:.0f}")
        cells.append(f"{(p.time_mean * p.n_solves / N_DAYS).median():.2g}")  # solver time per simulated day
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


VARIANT_POLICIES = [("EV-RT", "MPC-D"), ("2S-RT", "MPC-S"), ("OR-RT", "MPC-P"), ("HY-EV-1H", "HYB-1h")]
VARIANT_ROWS = [  # (objective, type, label); ranges over the three houses, single cases for the wear cost
    ("mef", "house", "MEF, base"),
    ("mef_prev", "house", "MEF, previous-year factors"),
    ("mef_net", "house", "MEF, exports credited"),
    None,
    ("cost_w0", "house", "Cost, $w = 0$, house"),
    ("cost", "house", "Cost, $w = 0.05$, house"),
    ("cost_w10", "house", "Cost, $w = 0.10$, house"),
    None,
    ("cost_w0", "apartment", "Cost, $w = 0$, apartment"),
    ("cost", "apartment", "Cost, $w = 0.05$, apartment"),
    ("cost_w10", "apartment", "Cost, $w = 0.10$, apartment"),
]


def table_variants():
    """Recovered share in the variants of the second round, on the cases where they were run."""
    df = pd.read_csv(S / "closed_loop.csv")
    homes = {o: set(g.home) for o, g in df.groupby("objective")}
    lines = []
    for row in VARIANT_ROWS:
        if row is None:
            lines.append("\\addlinespace")
            continue
        obj, typ, label = row
        # same households as the variant: the three MEF houses, the median house and apartment for cost
        keep = homes["mef_prev"] if obj.startswith("mef") else homes["cost_w0"]
        g = df[(df.objective == obj) & (df.type == typ) & df.home.isin(keep)]
        unit = "kg" if obj.startswith("mef") else "\\euro"
        h = g[g.policy == "PF"].headroom_abs
        head = f"{h.min():.0f}--{h.max():.0f}" if len(h) > 1 else f"{h.iloc[0]:.0f}"
        cells = [label, f"{head}~{unit}"]
        for code, _ in VARIANT_POLICIES:
            r = 100 * g[g.policy == code].recovered
            if r.empty:
                cells.append("--")
            elif len(r) > 1 and round(r.min()) != round(r.max()):
                cells.append(f"${r.min():.0f}$--${r.max():.0f}$")
            else:
                cells.append(f"${r.median():.0f}$")
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def main():
    tables = (
        ("table_headroom", table_headroom()),
        ("table_closed_loop", table_closed_loop()),
        ("table_variants", table_variants()),
    )
    for name, body in tables:
        (S / f"{name}.tex").write_text(body, encoding="utf-8", newline="\n")
        print(f"--- {name}\n{body}")


if __name__ == "__main__":
    main()
