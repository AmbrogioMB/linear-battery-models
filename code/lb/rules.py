"""Rule-based policies executed in real time.

rule_step implements the self-consumption rule (AB) with four optional
setpoints coming from a planner (hybrid policies):
- floor: the battery does not discharge below this state of charge to cover
  the load;
- ceiling: the battery does not charge from the PV surplus above this state
  of charge (the surplus is exported);
- target: if the state of charge is below this level, the battery charges
  (from PV and, if needed, from the grid) towards it;
- dump: if the state of charge is above this level, the battery discharges
  towards it even beyond the load (the excess is exported).
With floor = target = 0 and ceiling = dump = inf the rule is exactly AB.
A setpoint array covers the periods for which the plan is executed.
"""

import numpy as np

from .system import DT


def rule_step(E, net, used, bat, floor=0.0, target=0.0, dump=np.inf, ceiling=np.inf):
    """One period of the rule. net = load - pv (realized, real-time).

    Returns (b_plus, b_minus). `used` is the throughput DT*(sum b+/mu + eta sum b-)
    already consumed in the current calendar day.
    """
    rem = max(bat.budget - used, 0.0)
    cmax = max(min(bat.power, (bat.capacity - E) / (bat.eta * DT), rem / (bat.eta * DT)), 0.0)
    charge = min(max(-net, 0.0), cmax, max(ceiling - E, 0.0) / (bat.eta * DT))
    if E < target:
        charge = max(charge, min((target - E) / (bat.eta * DT), cmax))
    if charge > 0:
        return 0.0, charge
    dmax = min(bat.mu * bat.power, rem * bat.mu / DT)
    dis = min(max(net, 0.0), bat.mu * max(E - floor, 0.0) / DT, dmax)
    if E > dump:
        dis = max(dis, min(bat.mu * (E - dump) / DT, dmax))
    return max(dis, 0.0), 0.0


def project(rp, rm, E, used, bat):
    """Project a raw decision onto the feasible set given SOC and cycle budget."""
    rem = max(bat.budget - used, 0.0)
    p = min(max(rp, 0.0), bat.mu * bat.power, bat.mu * E / DT, rem * bat.mu / DT)
    m = min(
        max(rm, 0.0), bat.power, (bat.capacity - E) / (bat.eta * DT), max(rem - DT * p / bat.mu, 0) / (bat.eta * DT)
    )
    return max(p, 0.0), max(m, 0.0)


def step_soc(E, p, m, bat):
    return min(max(E + DT * (bat.eta * m - p / bat.mu), 0.0), bat.capacity)


def run_rule(load, pv, bat, E0, setpoints=None, T=96):
    """Apply the rule to a long trajectory (1-D arrays), with daily cycle budget.

    setpoints: optional dict of arrays (same length) with keys among floor,
    target, dump, ceiling. Returns bp, bm, E (E has one more entry than the inputs).
    """
    n = len(load)
    bp, bm, E = np.zeros(n), np.zeros(n), np.zeros(n + 1)
    E[0] = e = E0
    used = 0.0
    net = np.asarray(load) - np.asarray(pv)
    for t in range(n):
        if t % T == 0:
            used = 0.0
        sp = {k: v[t] for k, v in (setpoints or {}).items()}
        p, m = rule_step(e, net[t], used, bat, **sp)
        used += DT * (p / bat.mu + bat.eta * m)
        e = step_soc(e, p, m, bat)
        bp[t], bm[t], E[t + 1] = p, m, e
    return bp, bm, E


SETPOINT_DEFAULTS = dict(floor=0.0, target=0.0, dump=np.inf, ceiling=np.inf)


def setpoints_from_plan(bp_plan, bm_plan, E_plan, net_hat, tol=1e-3):
    """Translate a planned trajectory into rule setpoints (dict of arrays).

    With E' the planned SOC after period t and s, d the forecast PV surplus and
    deficit:
    target  = E' if the plan charges more than s (grid charging), 0 otherwise;
    ceiling = E' if the plan charges less than s (PV exported), inf otherwise;
    floor   = E' if the plan discharges less than d (energy held), 0 otherwise;
    dump    = E' if the plan discharges more than d (export), inf otherwise.
    """
    E1 = E_plan[1:]
    surplus, deficit = np.maximum(-net_hat, 0), np.maximum(net_hat, 0)
    return dict(
        target=np.where(bm_plan > surplus + tol, E1, 0.0),
        ceiling=np.where(bm_plan < surplus - tol, E1, np.inf),
        floor=np.where(bp_plan < deficit - tol, E1, 0.0),
        dump=np.where(bp_plan > deficit + tol, E1, np.inf),
    )
