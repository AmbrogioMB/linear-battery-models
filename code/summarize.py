"""Collect the results of E1 and E2 into tidy CSV files for the paper.

Inputs:  results/headroom/<home>.csv (E1), results/cl2/<case>/<policy>.csv (E2),
         results/cases.csv, data/homes.csv, results/sensitivity/<home>.csv (E1b).
Outputs: results/summary/headroom.csv        one row per home and configuration (E1)
         results/summary/headroom_stats.csv  quartiles by objective, type and configuration
         results/summary/closed_loop.csv     one row per case and policy (E2)
         results/summary/closed_loop_stats.csv  medians by objective, type and policy

Metrics (J = realized objective over the simulation period, lower is better):
  gap_share  = (J - J_PF) / (J_NB - J_PF): share of the benefit of a perfectly
               operated battery that is lost (0 for PF, 1 for no battery);
               for SC in E1 it is the relative headroom of the paper.
  recovered  = (J_SC - J) / (J_SC - J_PF): share of the SC-PF headroom recovered.
  saving     = J_SC - J in kgCO2e (emission objectives) or EUR (cost).

Usage: python code/summarize.py
"""

import pandas as pd

from lb import data

RES = data.ROOT / "results"
OUT = RES / "summary"
SCALE = {"aef": 1e-3, "mef": 1e-3, "mef_prev": 1e-3, "mef_net": 1e-3}  # g -> kg; cost objectives are already in EUR


def scale(obj):
    return SCALE.get(obj, 1.0)


def quartiles(s):
    return pd.Series({"q25": s.quantile(0.25), "median": s.median(), "q75": s.quantile(0.75), "n": s.size})


def headroom():
    df = pd.concat(pd.read_csv(f) for f in sorted((RES / "headroom").glob("*.csv")))
    homes = pd.read_csv(data.ROOT / "data" / "homes.csv")[["home", "type", "pilot"]]
    df = df.merge(homes, on="home", validate="many_to_one")
    k = df.objective.map(scale)
    df["gap_share"] = df.headroom_rel_gain
    df["headroom_abs"] = k * (df.ab - df.pf)  # kg or EUR over the simulation period
    df["battery_benefit_sc"] = k * (df.nb - df.ab)
    df["headroom_rel_sc"] = (df.ab - df.pf) / df.ab.abs()
    df.to_csv(OUT / "headroom.csv", index=False, lineterminator="\n")
    keys = ["objective", "type", "site", "kwp", "capacity"]
    rows = []
    for var in ("gap_share", "headroom_abs", "headroom_rel_sc", "ss_ab"):
        st = df.groupby(keys)[var].apply(quartiles).unstack().reset_index()
        st.insert(len(keys), "variable", var)
        rows.append(st)
    pd.concat(rows).to_csv(OUT / "headroom_stats.csv", index=False, lineterminator="\n")
    return df


# objective variants of the second round of E2 -> objective of the case they are derived from
BASE_OBJECTIVE = {"mef_prev": "mef", "mef_net": "mef", "cost_w0": "cost", "cost_w10": "cost"}


def closed_loop():
    df = pd.concat(pd.read_csv(f) for f in sorted((RES / "cl2").glob("*/*.csv")))
    cases = pd.read_csv(RES / "cases.csv").rename(columns={"objective": "base_objective"})
    df["base_objective"] = df.objective.replace(BASE_OBJECTIVE)
    main = df[df.objective == df.base_objective]
    assert len(main) == 11 * len(cases), "results/cl2 does not contain all the policies of all the cases"
    keys = ["home", "site", "base_objective"]
    df = df.merge(cases[keys + ["type", "quantile", "headroom_rel_gain"]], on=keys, validate="many_to_one")
    ref = df.pivot(index="case", columns="policy", values="value")
    nb, sc, pf = (df.case.map(ref[p]) for p in ("NB", "AB", "PF"))
    k = df.objective.map(scale)
    df["gap_share"] = (df.value - pf) / (nb - pf)
    df["recovered"] = (sc - df.value) / (sc - pf)
    df["saving"] = k * (sc - df.value)
    df["headroom_abs"] = k * (sc - pf)
    df.to_csv(OUT / "closed_loop.csv", index=False, lineterminator="\n")
    st = (
        df.groupby(["objective", "type", "policy"])[["gap_share", "recovered", "saving", "headroom_abs", "time_mean"]]
        .median()
        .reset_index()
    )
    st.to_csv(OUT / "closed_loop_stats.csv", index=False, lineterminator="\n")
    return df


def sensitivity():
    df = pd.concat(pd.read_csv(f) for f in sorted((RES / "sensitivity").glob("*.csv")))
    homes = pd.read_csv(data.ROOT / "data" / "homes.csv")[["home", "type"]]
    df = df.merge(homes, on="home", validate="many_to_one")
    df["gap_share"] = (df.ab - df.pf) / (df.nb - df.pf)
    df["headroom_abs"] = df.objective.map(scale) * (df.ab - df.pf)
    df["battery_benefit_sc"] = df.objective.map(scale) * (df.nb - df.ab)
    df.to_csv(OUT / "sensitivity.csv", index=False, lineterminator="\n")
    keys = ["variant", "objective", "site", "type"]
    st = df.groupby(keys)[["gap_share", "headroom_abs", "battery_benefit_sc", "grid_charge_pf", "cycles_pf"]]
    st.median().reset_index().to_csv(OUT / "sensitivity_stats.csv", index=False, lineterminator="\n")
    return df


def main():
    OUT.mkdir(exist_ok=True)
    h = headroom()
    c = closed_loop()
    print(f"E1: {h.home.nunique()} homes, {len(h)} rows; E2: {c.case.nunique()} cases, {len(c)} rows")
    if (RES / "sensitivity").exists():
        s = sensitivity()
        print(f"E1b: {s.home.nunique()} homes, {len(s)} rows")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
