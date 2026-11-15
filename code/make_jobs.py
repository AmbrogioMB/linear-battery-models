"""Write the job lists for run_jobs.py.

  python code/make_jobs.py headroom [--shards 200]  -> jobs/headroom.txt (E1, all homes)
  python code/make_jobs.py cases                    -> results/cases.csv (E2 cases, needs E1)
  python code/make_jobs.py closed-loop              -> jobs/closed_loop2.txt (E2, needs cases.csv)
  python code/make_jobs.py sensitivity              -> jobs/sensitivity.txt (E1 sensitivity, all homes)
  python code/make_jobs.py pilot --home H           -> jobs/pilot2.txt (all policies, 7 days, timing)

Cases of E2: base configuration (Milan, 4.5 kWp, 10 kWh / 5 kW); for each
objective and for houses and apartments separately, the homes at the 25th,
50th and 75th percentiles of the relative headroom (AB - PF) / (NB - PF);
for Rome and Palermo, the median house with the average emission factor.
That is 20 cases, i.e. 220 closed-loop jobs (11 policies), plus 67 jobs for the
objective variants on a subset of the cases (EXTRA below): 287 in total.
"""

import argparse

import numpy as np
import pandas as pd

from exp_closed_loop import POLICIES
from lb import data

JOBS = data.ROOT / "jobs"
BASE = dict(site="milan", kwp=4.5, capacity=10.0, power=5.0)
QUANTILES = (0.25, 0.5, 0.75)


def headroom(shards):
    lines = [f"python code/exp_headroom.py --all --shard {i}/{shards}" for i in range(shards)]
    write("headroom.txt", lines)


def pick(df, q):
    v = df["headroom_rel_gain"]
    return df.iloc[int(np.argmin(np.abs(v - v.quantile(q))))]


def cases():
    files = sorted((data.ROOT / "results" / "headroom").glob("*.csv"))
    df = pd.concat([pd.read_csv(f, dtype={"home": str}) for f in files])
    homes = pd.read_csv(data.ROOT / "data" / "homes.csv", dtype={"home": str}).set_index("home")
    df["type"] = homes.loc[df.home, "type"].values
    base = (df.kwp == BASE["kwp"]) & (df.capacity == BASE["capacity"])
    rows = []
    for obj in ("aef", "mef", "cost"):
        for htype in ("house", "apartment"):
            sub = df[base & (df.site == "milan") & (df.objective == obj) & (df.type == htype)]
            for q in QUANTILES:
                rows.append(case_row(pick(sub, q), q))
    for site in ("rome", "palermo"):
        sub = df[base & (df.site == site) & (df.objective == "aef") & (df.type == "house")]
        rows.append(case_row(pick(sub, 0.5), 0.5))
    out = pd.DataFrame(rows).drop_duplicates(subset=["home", "site", "objective"])
    out.to_csv(data.ROOT / "results" / "cases.csv", index=False, lineterminator="\n")
    print(out.to_string())


def case_row(r, q):
    return dict(
        home=r.home,
        type=r.type,
        site=r.site,
        kwp=r.kwp,
        capacity=r.capacity,
        power=r.power,
        objective=r.objective,
        quantile=q,
        headroom_rel_gain=r.headroom_rel_gain,
    )


def cmd(r, policy, days=None):
    s = (
        f"python code/exp_closed_loop.py --home {r['home']} --site {r['site']} --kwp {r['kwp']:g} "
        f"--cap {r['capacity']:g} --power {r['power']:g} --obj {r['objective']} --policy {policy}"
    )
    return s + (f" --days {days}" if days else "")


# Additional E2 runs on the median-headroom cases (second round, 2026-11-14):
# objective -> policies. mef_prev: plan with the marginal factors of the previous
# year; mef_net: exports credited; cost_w0/cost_w10: wear-cost sensitivity.
REDUCED = ("NB", "AB", "PF", "EV-RT", "2S-RT", "OR-RT", "HY-EV-1H")
EXTRA = {
    "mef_prev": ("mef", ("house",), ("NB", "AB", "PF", "EV-RT", "2S-RT", "HY-EV-1H")),
    "mef_net": ("mef", ("house",), REDUCED),
    "cost_w0": ("cost", ("house", "apartment"), REDUCED),
    "cost_w10": ("cost", ("house", "apartment"), REDUCED),
}


def extra_cases(cs):
    """Rows of cases.csv to which each extra objective is applied (mef: the three houses; cost: the medians)."""
    rows = []
    for obj, (base, types, policies) in EXTRA.items():
        sub = cs[(cs.objective == base) & cs.type.isin(types) & (cs.site == "milan")]
        if base == "cost":
            sub = sub[sub["quantile"] == 0.5]
        rows += [(dict(r, objective=obj), policies) for _, r in sub.iterrows()]
    return rows


def closed_loop():
    cs = pd.read_csv(data.ROOT / "results" / "cases.csv", dtype={"home": str})
    jobs = [(r, p) for _, r in cs.iterrows() for p in POLICIES]
    jobs += [(r, p) for r, policies in extra_cases(cs) for p in policies]
    # longest jobs first (two-stage models), cheap ones last
    jobs.sort(key=lambda j: (not j[1].startswith("2S"), j[1]))
    write("closed_loop2.txt", [cmd(r, p) for r, p in jobs])


def sensitivity(shards):
    write("sensitivity.txt", [f"python code/exp_sensitivity.py --all --shard {i}/{shards}" for i in range(shards)])


def pilot(home):
    lines = [cmd(dict(home=home, objective=o, **BASE), p, days=7) for o in ("aef", "cost") for p in POLICIES]
    lines += [cmd(dict(home=home, objective=o, **BASE), p, days=7) for o in ("mef_prev", "mef_net") for p in REDUCED]
    write("pilot2.txt", lines)


def write(name, lines):
    JOBS.mkdir(exist_ok=True)
    (JOBS / name).write_text("\n".join(lines) + "\n", newline="\n")
    print(f"jobs/{name}: {len(lines)} jobs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("headroom", "cases", "closed-loop", "pilot", "sensitivity"))
    ap.add_argument("--shards", type=int, default=200)
    ap.add_argument("--home")
    a = ap.parse_args()
    if a.what == "headroom":
        headroom(a.shards)
    elif a.what == "cases":
        cases()
    elif a.what == "closed-loop":
        closed_loop()
    elif a.what == "sensitivity":
        sensitivity(min(a.shards, 100))
    else:
        pilot(a.home or data.homes()[0])


if __name__ == "__main__":
    main()
