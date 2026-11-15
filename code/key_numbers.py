"""Numbers quoted in the text of the paper that are not in a table.

Usage: python code/key_numbers.py > results/summary/key_numbers.txt
"""

import sys

import numpy as np
import pandas as pd

from lb import data

S = data.ROOT / "results" / "summary"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    h = pd.read_csv(S / "headroom.csv")
    print("== E1: Spearman correlation of relative headroom with SC self-sufficiency (all configurations)")
    for obj, g in h.groupby("objective"):
        print(f"{obj}: {g.gap_share.corr(g.ss_ab, method='spearman'):.3f} (n={len(g)})")
    b = h[(h.site == "milan") & (h.kwp == 4.5) & (h.capacity == 10.0)]
    print("== E1 base case: medians by objective and type")
    cols = [
        "battery_benefit_sc",
        "headroom_abs",
        "headroom_rel_sc",
        "ss_ab",
        "grid_charge_pf",
        "cycles_ab",
        "cycles_pf",
    ]
    print(b.groupby(["objective", "type"])[cols].median().round(3).to_string())
    print("== E1: share of household-configurations with relative headroom below 5% / above 20%")
    for obj, g in h.groupby(["objective", "type"]):
        print(obj, f"<5%: {(g.gap_share < 0.05).mean():.3f}", f">20%: {(g.gap_share > 0.2).mean():.3f}", f"n={len(g)}")
    print("== E1: rows where SC is worse than no battery")
    print(h[h.gap_share > 1].groupby(["objective", "type"]).size().to_string())
    print(h[h.gap_share > 1].load_kwh.describe().round(0).to_string())

    print("== Grid signals over the simulation period: mean, median daily range, median daily coefficient of variation")
    g = np.load(data.ROOT / "data" / "grid.npz")
    m = (g["dates"] >= data.SIM_FIRST) & (g["dates"] <= data.SIM_LAST)
    for k in ("alpha_avg", "alpha_marg", "price"):
        x = g[k][m]
        cv = np.median(x.std(1) / x.mean(1))
        print(f"{k}: mean {x.mean():.1f}, daily range {np.median(x.max(1) - x.min(1)):.1f}, daily CV {cv:.3f}")

    c = pd.read_csv(S / "closed_loop.csv")
    pd.set_option("display.width", 250)
    print("== E2: recovered share by case")
    print(c.pivot_table(index=["objective", "type", "quantile", "site"], columns="policy", values="recovered").round(3))
    print("== E2: gap share by case")
    print(c.pivot_table(index=["objective", "type", "quantile", "site"], columns="policy", values="gap_share").round(3))
    print("== E2: saving over SC (kg or EUR)")
    print(c.pivot_table(index=["objective", "type", "quantile", "site"], columns="policy", values="saving").round(1))
    allc = c
    c = allc[allc.objective.isin(["aef", "mef", "cost"])]  # the 20 main cases
    print("== E2: number of main cases where the policy is worse than SC, by objective")
    w = c[(c.policy != "AB") & (c.saving < 0)]
    print(w.pivot_table(index="policy", columns="objective", values="saving", aggfunc="size", fill_value=0).to_string())
    print("== E2: range of the recovered share over the main cases, by objective and type")
    r = c.groupby(["objective", "type", "policy"]).recovered.agg(["min", "median", "max"])
    print(r.round(3).to_string())
    print("== E2: range of the gap share of 2S-RH and of the other policies")
    print(c.groupby("policy").gap_share.agg(["min", "max"]).round(3).to_string())
    print("== E2: median house with AEF in Milan (157904): energy flows (kWh)")
    m = c[(c.home == 157904) & (c.objective == "aef")].set_index("policy")
    print(m[["grid_charge_kwh", "import_kwh", "export_kwh", "cycles_per_day"]].round(0).to_string())
    print("== E2 variants: recovered share and saving")
    v = allc[allc.objective.isin(["mef_prev", "mef_net", "cost_w0", "cost_w10"])]
    print(v.pivot_table(index=["objective", "type", "home"], columns="policy", values="recovered").round(4).to_string())
    print(v.pivot_table(index=["objective", "type", "home"], columns="policy", values="saving").round(1).to_string())
    print("== E2: solver time per simulated day (s), median over the main cases")
    print((c.time_mean * c.n_solves / 318).groupby(c.policy).median().round(4).to_string())
    print("== E2: grid charging, import, export (kWh), medians by objective")
    print(c.pivot_table(index="objective", columns="policy", values=["grid_charge_kwh"], aggfunc="median").round(0))


if __name__ == "__main__":
    main()
