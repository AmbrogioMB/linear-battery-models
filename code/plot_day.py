"""Figure of one day of operation of the hybrid policy, with its setpoints.

The setpoints are not stored by exp_closed_loop.py: the day is replayed from
the SOC stored at its start in results/cl2/<case>/HY-EV-1H.npz, with the same
random seeds, and the replayed flows are checked against the stored ones.

Usage: python code/plot_day.py --case 180146_milan_pv4.5_b10-5_mef --date 2024-03-12
       python code/plot_day.py --case ... --list   (days where the hybrid gains most over SC)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from exp_closed_loop import EPS_REL, M_RH, POLICIES, RESULTS_DIR, SEED, Forecaster
from lb import data
from lb.lp import RecedingLP
from lb.rules import SETPOINT_DEFAULTS, rule_step, setpoints_from_plan, step_soc
from lb.system import DT, T
from plotstyle import AQUA, BLUE, INK, MUTED, ORANGE, WIDTH, save, setup

POLICY = "HY-EV-1H"
SIGNAL_LABEL = {"aef": "AEF (gCO$_2$/kWh)", "mef": "MEF (gCO$_2$/kWh)", "cost": "Price (\\euro/kWh)"}


def load_case(label):
    home, site, pv, bat, obj = label.split("_")
    cap, power = (float(v) for v in bat[1:].split("-"))
    return data.load_case(home, site, float(pv[2:]), cap, power, obj)


def replay(c, i, E):
    """Replay day i of the simulation with the hybrid policy; return flows, SOC and setpoints."""
    spec, bat, day = POLICIES[POLICY], c.battery, c.sim_days[i]
    eps = EPS_REL * np.median(np.abs(c.tariff.buy(c.x)))
    lp, fc = RecedingLP("det", M_RH, bat, c.tariff, eps), Forecaster(c, True)
    BP, BM, EE = np.zeros(T), np.zeros(T), np.zeros(T + 1)
    SP = {k: np.zeros(T) for k in SETPOINT_DEFAULTS}
    used, EE[0] = 0.0, E
    for tau in range(T):
        if tau % spec["hybrid"] == 0:
            rng = np.random.default_rng([SEED, int(day), tau])
            L, S, X = (a.mean(0, keepdims=True) for a in fc.window(day, tau, M_RH, False, rng))
            r = lp.solve(L, S, X, E, max(bat.budget - used, 0.0), tau)
            plan = setpoints_from_plan(r["bp"], r["bm"], r["E"], (L - S).mean(0))
            n_set = min(spec["hybrid"], T - tau)
            sp = {k: np.full(T, v) for k, v in SETPOINT_DEFAULTS.items()}
            for k, v in plan.items():
                sp[k][tau : tau + n_set] = v[:n_set]
        for k in SP:
            SP[k][tau] = sp[k][tau]
        p, m = rule_step(E, c.load[day, tau] - c.pv[day, tau], used, bat, **{k: v[tau] for k, v in sp.items()})
        BP[tau], BM[tau] = p, m
        used += DT * (p / bat.mu + bat.eta * m)
        E = step_soc(E, p, m, bat)
        EE[tau + 1] = E
    return BP, BM, EE, SP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="180146_milan_pv4.5_b10-5_mef")
    ap.add_argument("--date", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    c = load_case(a.case)
    d = data.ROOT / "results" / RESULTS_DIR / a.case
    hy, sc = np.load(d / f"{POLICY}.npz"), np.load(d / "AB.npz")
    dates = list(hy["dates"])
    if a.list:
        gain = sc["daily"] - hy["daily"]
        for i in np.argsort(-gain)[:15]:
            print(dates[i], f"gain {gain[i]:.1f}", f"charged {DT * hy['bm'][i].sum():.1f} kWh")
        return
    i = dates.index(a.date)
    BP, BM, EE, SP = replay(c, i, float(hy["E"][i, 0]))
    assert np.allclose(BP, hy["bp"][i], atol=1e-6) and np.allclose(BM, hy["bm"][i], atol=1e-6), "replay differs"
    day = c.sim_days[i]

    setup()
    h = np.arange(T) * DT
    he = np.arange(T + 1) * DT
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 0.7, 1.3]))
    ax = axes[0]
    ax.plot(h, c.load[day], color=INK, lw=1, label="Load")
    ax.plot(h, c.pv[day], color=ORANGE, label="PV")
    ax.plot(h, c.pv_fc[day], color=ORANGE, ls="--", lw=1, label="PV forecast")
    ax.set_ylabel("Power (kW)")
    ax.legend(loc="upper left", ncol=3)
    ax = axes[1]
    ax.step(h, c.tariff.buy(c.x[day]) if a.case.endswith("cost") else c.x[day], where="post", color=MUTED)
    ax.set_ylabel(SIGNAL_LABEL[c.objective])
    ax = axes[2]
    ax.plot(he, sc["E"][i], color=MUTED, lw=1, label="SOC with SC")
    ax.plot(he, EE, color=BLUE, lw=1.6, label="SOC with HYB-1h")
    cap = c.battery.capacity
    styles = dict(floor=(AQUA, "-"), target=(ORANGE, "-"), ceiling=(INK, ":"), dump=(INK, "--"))
    neutral = {"floor": 0.0, "target": 0.0, "ceiling": np.inf, "dump": np.inf}
    for k, (col, ls) in styles.items():
        v = np.where(SP[k] == neutral[k], np.nan, np.clip(SP[k], 0, cap))
        if np.isfinite(v).any():
            ax.step(h, v, where="post", color=col, ls=ls, lw=1, label=k.capitalize())
    ax.set_ylim(-0.3, cap + 0.3)
    ax.set_ylabel("SOC (kWh)")
    ax.set_xlabel(f"Hour of {a.date}")
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlim(0, 24)
    ax.legend(loc="upper left", ncol=3)
    for ax in axes:
        ax.grid(axis="x", visible=False)
    save(fig, "day")


if __name__ == "__main__":
    main()
