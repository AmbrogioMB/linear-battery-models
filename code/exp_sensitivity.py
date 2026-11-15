"""Experiment E1b: sensitivity of the headroom to accounting and battery assumptions.

For one home, in the base PV and battery configuration, the rule (AB) and the
perfect-foresight benchmark (PF) are evaluated under variants of E1:
  net    exports credited at the marginal factor (objective mef_net), 3 sites
  eff90  round-trip efficiency 0.90 instead of 0.96 (aef, mef, cost; Milan)
  beta0  no emissions attributed to the battery throughput (aef, mef; Milan)
  beta66 twice the base value (aef, mef; Milan)

Output: results/sensitivity/<home>.csv (one row per variant, objective and
site). Homes already done are skipped.

Usage: python code/exp_sensitivity.py <home> [<home> ...]
       python code/exp_sensitivity.py --all [--shard i/n]
"""

import argparse
import dataclasses
import time

import numpy as np
import pandas as pd

from lb import data
from lb.lp import solve_pf
from lb.rules import run_rule
from lb.system import BETA, evaluate

OUT = data.ROOT / "results" / "sensitivity"
KWP, CAP, POWER = 4.5, 10.0, 5.0
ETA90 = float(np.sqrt(0.90))
VARIANTS = (
    [("net", "mef_net", site, {}, {}) for site in ("milan", "rome", "palermo")]
    + [("eff90", obj, "milan", dict(eta=ETA90, mu=ETA90), {}) for obj in ("aef", "mef", "cost")]
    + [("beta0", obj, "milan", {}, dict(wear=0.0)) for obj in ("aef", "mef")]
    + [("beta66", obj, "milan", {}, dict(wear=2 * BETA)) for obj in ("aef", "mef")]
)


def run_home(home):
    path = OUT / f"{home}.csv"
    if path.exists():
        return
    rows, t0 = [], time.time()
    for variant, obj, site, bat_chg, tar_chg in VARIANTS:
        c = data.load_case(home, site, KWP, CAP, POWER, obj)
        bat = dataclasses.replace(c.battery, **bat_chg)
        tar = dataclasses.replace(c.tariff, **tar_chg)
        idx = c.sim_days
        ld, pv, x = c.load[idx].ravel(), c.pv[idx].ravel(), c.x[idx].ravel()
        E0 = bat.capacity / 2
        bp, bm, E = run_rule(ld, pv, bat, E0)
        nb = evaluate(ld, pv, x, 0 * ld, 0 * ld, tar, bat, 0, 0)
        ab = evaluate(ld, pv, x, bp, bm, tar, bat, E0, E[-1])
        sol = solve_pf(ld, pv, x, tar, bat, E0)
        pf = evaluate(ld, pv, x, sol["bp"], sol["bm"], tar, bat, E0, sol["E"][-1])
        assert abs(pf["value"] - sol["obj"]) <= 1e-5 * max(1, abs(sol["obj"])), (pf, sol["obj"])
        rows.append(
            dict(
                home=home,
                variant=variant,
                objective=obj,
                site=site,
                nb=nb["value"],
                ab=ab["value"],
                pf=pf["value"],
                ss_ab=ab["self_sufficiency"],
                import_ab=ab["import_kwh"],
                import_pf=pf["import_kwh"],
                export_ab=ab["export_kwh"],
                export_pf=pf["export_kwh"],
                cycles_ab=ab["cycles_per_day"],
                cycles_pf=pf["cycles_per_day"],
                grid_charge_pf=pf["grid_charge_kwh"],
            )
        )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path.with_suffix(".tmp"), index=False, lineterminator="\n")
    path.with_suffix(".tmp").replace(path)
    print(f"{home}: {len(rows)} variants in {time.time() - t0:.0f}s", flush=True)


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
