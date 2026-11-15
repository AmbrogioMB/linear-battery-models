"""Experiment E2: closed-loop operation of one case with one policy.

The battery is operated over the simulation period with the actual data, the
state of charge being carried over from each period to the next.

Policies (information: RT = the current period is measured before deciding;
EA = only the periods up to t-1 are known):
  NB          no battery
  AB          self-consumption rule (RT)
  PF          perfect foresight over the whole period (lower bound)
  EV-RT       receding horizon, deterministic model on the point forecast (RT)
  2S-RT       receding horizon, two-stage stochastic model (RT)
  2S-RT-nofc  as 2S-RT, scenarios not conditioned on the weather forecast
  2S-RH       as 2S-RT, but the decision is taken before measuring period t (EA)
  OR-RT       receding horizon with perfect information on the 24-h window (RT)
  HY-EV-DA    hybrid: the rule executes SOC setpoints (floor, ceiling, target, dump) from a
              deterministic plan computed once per day at midnight (EA)
  HY-EV-1H    hybrid, deterministic plan recomputed every hour (EA)
  HY-OR-1H    hybrid, plan with perfect information on the window, every hour

Scenarios come from the PCA model fitted on the 28 days preceding the current
day; the current day is sampled conditionally on the observations so far and
on the part of the PV forecast available at decision time (see FC_DELAY), the
next day conditionally on the available part of its forecast.
The signal is treated as known in advance when it is (marginal factors are a
fixed table; day-ahead prices of the next day are known from 14:00). With the
objective mef_prev the policies plan with the table of the previous year and
are evaluated with the actual one.

Output: results/cl2/<case>/<policy>.npz (daily checkpoints, resumable) and
results/cl2/<case>/<policy>.csv (summary).

Usage: python code/exp_closed_loop.py --home H --site milan --kwp 4.5 --cap 10 --power 5 --obj aef --policy 2S-RT
"""

import argparse
import time

import numpy as np
import pandas as pd

from lb import data
from lb.lp import RecedingLP, solve_pf
from lb.rules import SETPOINT_DEFAULTS, project, rule_step, setpoints_from_plan, step_soc
from lb.scenarios import ScenarioModel
from lb.system import DT, TARIFFS, T, evaluate

# results/cl2: runs with the causal forecast information and the per-day cycle budget
# in the lookahead (the results reported in the paper)
RESULTS_DIR = "cl2"
M_RH = 100  # scenarios per receding-horizon decision
K = 5  # principal components
SEED = 7
EPS_REL = 1e-5  # tie-breaking reward, relative to the median buy coefficient
PRICE_KNOWN_FROM = 56  # day-ahead prices of tomorrow known from 14:00
# Availability of the weather forecast. The archive (Open-Meteo previous runs,
# *_previous_day1) gives for each valid time the value predicted 24 hours
# earlier, i.e. by a model run started at least 24 h before the valid time.
# Allowing FC_DELAY for the publication of the run and for the interpolation of
# hourly values, at period tau only the forecasts of the valid times up to
# tau + 24 h - FC_DELAY are available: the rest of the profile must not be
# used, not even to condition the scenario model.
FC_DELAY = 24  # periods (6 h)


def fc_available(tau):
    """Number of leading periods of today's and of tomorrow's forecast profile available at period tau."""
    return min(T, tau + T - FC_DELAY), max(tau - FC_DELAY, 0)


POLICIES = {
    "NB": {},
    "AB": {},
    "PF": {},
    "EV-RT": dict(lp="det", rt=True, fc=True),
    "2S-RT": dict(lp="2s", rt=True, fc=True),
    "2S-RT-nofc": dict(lp="2s", rt=True, fc=False),
    "2S-RH": dict(lp="2s", rt=False, fc=True),
    "OR-RT": dict(lp="det", rt=True, oracle=True),
    "HY-EV-DA": dict(lp="det", hybrid=96, fc=True),
    "HY-EV-1H": dict(lp="det", hybrid=4, fc=True),
    "HY-OR-1H": dict(lp="det", hybrid=4, oracle=True),
}


class Forecaster:
    """Scenarios of (load, PV, signal) for the 24-h window starting at (day, tau)."""

    def __init__(self, case, use_fc):
        self.c, self.use_fc = case, use_fc
        self.day, self.sm = None, None

    def _model(self, day):
        if day != self.day:
            c, w = self.c, slice(day - data.WINDOW_DAYS, day)
            xk = c.x if c.x_plan is None else c.x_plan
            blocks = [c.load[w], c.pv[w], xk[w]] + ([c.pv_fc[w]] if self.use_fc else [])
            self.sm = ScenarioModel(blocks, log_blocks=(2,), K=K, caps={0: (0, None), 1: (0, c.kwp)})
            self.day = day
        return self.sm

    def _x_known(self, tomorrow, tau):
        obj = self.c.objective
        return obj.startswith("mef") or (obj.startswith("cost") and (not tomorrow or tau >= PRICE_KNOWN_FROM))

    def window(self, day, tau, n, realtime, rng):
        c, sm = self.c, self._model(day)
        k = tau + 1 if realtime else tau
        n_today, n_next = fc_available(tau)
        # signal known in advance: the plan may use a different table from the realized one (mef_prev)
        xk = c.x if c.x_plan is None else c.x_plan
        obs = {0: c.load[day, :k], 1: c.pv[day, :k], 2: xk[day] if self._x_known(False, tau) else c.x[day, :k]}
        if self.use_fc:
            obs[3] = c.pv_fc[day, :n_today]
        L, S, X = sm.sample(n, rng, obs)[:3]
        if tau == 0:
            return L, S, X
        obs = {3: c.pv_fc[day + 1, :n_next]} if self.use_fc else {}
        if self._x_known(True, tau):
            obs[2] = xk[day + 1]
        Ln, Sn, Xn = sm.sample(n, rng, obs)[:3]
        return (
            np.hstack([L[:, tau:], Ln[:, :tau]]),
            np.hstack([S[:, tau:], Sn[:, :tau]]),
            np.hstack([X[:, tau:], Xn[:, :tau]]),
        )


def actual_window(c, day, tau):
    f = lambda a: np.concatenate([a[day, tau:], a[day + 1, :tau]])[None, :]  # noqa: E731
    return f(c.load), f(c.pv), f(c.x)


def run(c, policy, out_dir):
    spec = POLICIES[policy]
    npz, csv = out_dir / f"{policy}.npz", out_dir / f"{policy}.csv"
    if csv.exists():
        print(f"{c.label} {policy}: done")
        return
    days = c.sim_days
    nD, bat = len(days), c.battery
    E0 = bat.capacity / 2
    if policy == "PF":
        idx = days
        sol = solve_pf(c.load[idx].ravel(), c.pv[idx].ravel(), c.x[idx].ravel(), c.tariff, bat, E0)
        BP, BM = sol["bp"].reshape(nD, T), sol["bm"].reshape(nD, T)
        EE = np.array([sol["E"][i * T : (i + 1) * T + 1] for i in range(nD)])
        finish(c, policy, days, BP, BM, EE, [sol["time"]], npz, csv)
        return

    BP, BM, EE = np.zeros((nD, T)), np.zeros((nD, T)), np.zeros((nD, T + 1))
    times, start, E = [], 0, E0
    if npz.exists():  # resume from the last daily checkpoint
        z = np.load(npz)
        start = int(z["n"])
        BP[:start], BM[:start], EE[:start] = z["bp"][:start], z["bm"][:start], z["E"][:start]
        times, E = list(z["times"]), float(EE[start - 1, -1])
        print(f"{c.label} {policy}: resuming at day {start}")
    eps = EPS_REL * np.median(np.abs(c.tariff.buy(c.x)))
    lp = RecedingLP(spec["lp"], M_RH, bat, c.tariff, eps) if "lp" in spec else None
    fc = Forecaster(c, spec.get("fc", False)) if "lp" in spec and not spec.get("oracle") else None
    t0 = time.time()
    for i in range(start, nD):
        day = days[i]
        used, EE[i, 0] = 0.0, E
        sp = None
        for tau in range(T):
            rem = max(bat.budget - used, 0.0)
            net = c.load[day, tau] - c.pv[day, tau]
            if policy == "NB":
                p, m = 0.0, 0.0
            elif policy == "AB":
                p, m = rule_step(E, net, used, bat)
            elif "hybrid" in spec:
                if tau % spec["hybrid"] == 0:
                    rng = np.random.default_rng([SEED, int(day), tau])
                    L, S, X = (
                        actual_window(c, day, tau) if spec.get("oracle") else fc.window(day, tau, M_RH, False, rng)
                    )
                    if spec["lp"] == "det" and not spec.get("oracle"):
                        L, S, X = L.mean(0, keepdims=True), S.mean(0, keepdims=True), X.mean(0, keepdims=True)
                    r = lp.solve(L, S, X, E, rem, tau)
                    times.append(r["time"])
                    plan = setpoints_from_plan(r["bp"], r["bm"], r["E"], (L - S).mean(0))
                    n_set = min(spec["hybrid"], T - tau)
                    sp = {k: np.full(T, v) for k, v in SETPOINT_DEFAULTS.items()}
                    for k, v in plan.items():
                        sp[k][tau : tau + n_set] = v[:n_set]
                p, m = rule_step(E, net, used, bat, **{k: v[tau] for k, v in sp.items()})
            else:
                rng = np.random.default_rng([SEED, int(day), tau])
                if spec.get("oracle"):
                    L, S, X = actual_window(c, day, tau)
                else:
                    L, S, X = fc.window(day, tau, M_RH, spec["rt"], rng)
                    if spec["lp"] == "det":
                        L, S, X = L.mean(0, keepdims=True), S.mean(0, keepdims=True), X.mean(0, keepdims=True)
                r = lp.solve(L, S, X, E, rem, tau)
                times.append(r["time"])
                p, m = project(r["bp0"], r["bm0"], E, used, bat)
            BP[i, tau], BM[i, tau] = p, m
            used += DT * (p / bat.mu + bat.eta * m)
            E = step_soc(E, p, m, bat)
            EE[i, tau + 1] = E
        np.savez(npz, bp=BP, bm=BM, E=EE, times=np.array(times), n=i + 1)
        if i % 20 == 0 or i == nD - 1:
            print(f"{c.label} {policy} {c.dates[day]} E={E:.2f} {time.time() - t0:.0f}s", flush=True)
    finish(c, policy, days, BP, BM, EE, times, npz, csv)


def finish(c, policy, days, BP, BM, EE, times, npz, csv):
    np.savez(npz, bp=BP, bm=BM, E=EE, times=np.array(times), n=len(days))
    ld, pv, x = c.load[days], c.pv[days], c.x[days]
    ev = evaluate(ld, pv, x, BP, BM, c.tariff, c.battery, EE[0, 0], EE[-1, -1])
    daily = [evaluate(ld[i], pv[i], x[i], BP[i], BM[i], c.tariff, c.battery, 0, 0)["value"] for i in range(len(days))]
    np.savez(npz, bp=BP, bm=BM, E=EE, times=np.array(times), n=len(days), daily=np.array(daily), dates=c.dates[days])
    row = dict(
        case=c.label,
        home=c.home,
        site=c.site,
        kwp=c.kwp,
        capacity=c.battery.capacity,
        power=c.battery.power,
        objective=c.objective,
        policy=policy,
        **ev,
        time_mean=float(np.mean(times)) if len(times) else np.nan,
        time_max=float(np.max(times)) if len(times) else np.nan,
        n_solves=len(times),
    )
    pd.DataFrame([row]).to_csv(csv, index=False, lineterminator="\n")
    print(f"{c.label} {policy}: {ev['value']:.1f} {c.tariff.unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--site", default="milan")
    ap.add_argument("--kwp", type=float, default=4.5)
    ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--power", type=float, default=5.0)
    ap.add_argument("--obj", default="aef", choices=list(TARIFFS))
    ap.add_argument("--policy", required=True, choices=list(POLICIES))
    ap.add_argument("--days", type=int, default=None, help="simulate only the first N days (tests)")
    a = ap.parse_args()
    c = data.load_case(a.home, a.site, a.kwp, a.cap, a.power, a.obj)
    out = data.ROOT / "results" / RESULTS_DIR / c.label
    if a.days:
        c.max_days = a.days
        out = data.ROOT / "results" / (RESULTS_DIR + "_test") / c.label
    out.mkdir(parents=True, exist_ok=True)
    run(c, a.policy, out)


if __name__ == "__main__":
    main()
