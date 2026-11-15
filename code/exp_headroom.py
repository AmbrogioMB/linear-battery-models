"""Experiment E1: headroom of optimization over the self-consumption rule.

For one home and every configuration (site, PV size, battery, objective),
the rule (AB) and the perfect-foresight benchmark (PF) are simulated over the
simulation period with the actual data. The relative headroom
(AB - PF) / (NB - PF) and (AB - PF) / PF bound the benefit of any
optimization-based policy with respect to the rule.

Output: results/headroom/<home>.csv (one row per configuration). Homes already
done are skipped, so the script can be relaunched.

Usage: python code/exp_headroom.py <home> [<home> ...]
       python code/exp_headroom.py --all [--shard i/n]
"""

import argparse
import time

import numpy as np
import pandas as pd

from lb import data
from lb.lp import solve_pf
from lb.rules import run_rule
from lb.system import evaluate

OUT = data.ROOT / "results" / "headroom"
SITES = ("milan", "rome", "palermo")
KWP = (3.0, 4.5, 6.0)
BATTERIES = ((5.0, 2.5), (10.0, 5.0), (15.0, 5.0))  # (kWh, kW)
OBJECTIVES = ("aef", "mef", "cost", "cost_w0", "cost_w10")


def configurations():
    for site in SITES:
        for kwp in KWP:
            for cap, pw in BATTERIES:
                yield site, kwp, cap, pw


def run_home(home):
    path = OUT / f"{home}.csv"
    if path.exists():
        return
    rows = []
    t0 = time.time()
    for site, kwp, cap, pw in configurations():
        ab_cache = None
        for obj in OBJECTIVES:
            if obj.startswith("cost") and site != "milan":
                continue
            c = data.load_case(home, site, kwp, cap, pw, obj)
            idx = c.sim_days
            ld, pv, x = c.load[idx].ravel(), c.pv[idx].ravel(), c.x[idx].ravel()
            E0 = c.battery.capacity / 2
            if ab_cache is None:  # the rule does not depend on the objective
                ab_cache = run_rule(ld, pv, c.battery, E0)
            bp, bm, E = ab_cache
            nb = evaluate(ld, pv, x, 0 * ld, 0 * ld, c.tariff, c.battery, 0, 0)
            ab = evaluate(ld, pv, x, bp, bm, c.tariff, c.battery, E0, E[-1])
            pf_sol = solve_pf(ld, pv, x, c.tariff, c.battery, E0)
            pf = evaluate(ld, pv, x, pf_sol["bp"], pf_sol["bm"], c.tariff, c.battery, E0, pf_sol["E"][-1])
            assert abs(pf["value"] - pf_sol["obj"]) <= 1e-5 * max(1, abs(pf_sol["obj"])), (pf, pf_sol["obj"])
            rows.append(
                dict(
                    home=home,
                    site=site,
                    kwp=kwp,
                    capacity=cap,
                    power=pw,
                    objective=obj,
                    nb=nb["value"],
                    ab=ab["value"],
                    pf=pf["value"],
                    load_kwh=nb["load_kwh"],
                    pv_kwh=nb["pv_kwh"],
                    import_nb=nb["import_kwh"],
                    import_ab=ab["import_kwh"],
                    import_pf=pf["import_kwh"],
                    ss_ab=ab["self_sufficiency"],
                    ss_pf=pf["self_sufficiency"],
                    cycles_ab=ab["cycles_per_day"],
                    cycles_pf=pf["cycles_per_day"],
                    grid_charge_pf=pf["grid_charge_kwh"],
                    pf_time=pf_sol["time"],
                    # deficit not covered by PV at night: energy the battery would need (kWh/day)
                    deficit_kwh_day=0.25 * np.maximum(ld - pv, 0).sum() / len(idx),
                    surplus_kwh_day=0.25 * np.maximum(pv - ld, 0).sum() / len(idx),
                )
            )
    df = pd.DataFrame(rows)
    df["headroom_rel_pf"] = (df.ab - df.pf) / df.pf.abs()
    df["headroom_rel_gain"] = (df.ab - df.pf) / (df.nb - df.pf)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".tmp"), index=False, lineterminator="\n")
    path.with_suffix(".tmp").replace(path)
    print(f"{home}: {len(df)} configurations in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("homes", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th home starting from i")
    a = ap.parse_args()
    homes = data.homes() if a.all else a.homes
    i, n = map(int, a.shard.split("/"))
    for h in homes[i::n]:
        run_home(h)


if __name__ == "__main__":
    main()
