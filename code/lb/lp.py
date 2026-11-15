"""Linear programs: perfect-foresight benchmark and receding-horizon models.

All models use an explicit SOC variable E_t, t = 0..H, with
E_{t+1} = E_t + DT * (eta * b-_t - b+_t / mu), and per-calendar-day cycle budgets.
Simultaneous charging and discharging is not excluded explicitly: with
nonnegative buy coefficients, sell <= buy and a positive wear term it only
wastes energy and is not optimal.
"""

import os

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .system import DT, T

_ENV = None


def env():
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.setParam("Threads", int(os.environ.get("GRB_THREADS", "1")))
        _ENV.start()
    return _ENV


def solve_pf(load, pv, x, tariff, bat, E0):
    """Perfect-foresight benchmark over a horizon of whole days (1-D inputs).

    The objective is exactly the evaluated one (see system.evaluate), including
    the terminal charge on a final SOC below E0, so the optimal value is a lower
    bound on the realized objective of every non-anticipative policy.
    """
    N = len(load)
    assert N % T == 0
    buy, sell = tariff.buy(x), tariff.sell(x)
    m = gp.Model(env=env())
    bp = m.addMVar(N, lb=0, ub=bat.mu * bat.power)
    bm = m.addMVar(N, lb=0, ub=bat.power)
    E = m.addMVar(N + 1, lb=0, ub=bat.capacity)
    yp = m.addMVar(N, lb=0)
    ym = m.addMVar(N, lb=0)
    d = m.addVar(lb=0)
    m.addConstr(E[1:] == E[:-1] + DT * (bat.eta * bm - bp / bat.mu))
    m.addConstr(E[0] == E0)
    m.addConstr(d >= E0 - E[N])
    m.addConstr(yp - ym + bp - bm == np.asarray(load) - np.asarray(pv))
    nD = N // T
    m.addConstr(DT * (bp.reshape(nD, T).sum(axis=1) / bat.mu + bat.eta * bm.reshape(nD, T).sum(axis=1)) <= bat.budget)
    m.setObjective(DT * (buy @ yp - sell @ ym) + tariff.wear * DT * bp.sum() + buy.mean() / bat.eta * d, GRB.MINIMIZE)
    m.Params.Method = 2  # barrier (with crossover): about 10x faster than simplex on these long LPs
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"PF status {m.Status}")
    out = dict(obj=m.ObjVal, bp=bp.X, bm=bm.X, E=E.X, time=m.Runtime)
    m.dispose()
    return out


class RecedingLP:
    """Lookahead LP over a window of H = T periods, built once and updated.

    kind = "det" (one trajectory: point forecast or perfect information),
           "pb"  (one schedule common to M scenarios, grid exchanges per scenario),
           "2s"  (two-stage: first-period decision common, full recourse after).
    The window starts at period tau of the current day; the cycle budget is
    imposed separately on each calendar day through throughput variables G
    cumulated within the day.
    eps: tie-breaking reward per kWh stored and per period (favours storing
    surplus energy as early as possible).
    """

    def __init__(self, kind, M, bat, tariff, eps):
        assert kind in ("det", "pb", "2s")
        if kind == "det":
            M = 1
        self.kind, self.M, self.bat, self.tariff = kind, M, bat, tariff
        H = T
        m = gp.Model(env=env())
        shape = (M, H) if kind == "2s" else (H,)
        bp = m.addMVar(shape, lb=0, ub=bat.mu * bat.power)
        bm = m.addMVar(shape, lb=0, ub=bat.power)
        eshape = (M, H + 1) if kind == "2s" else (H + 1,)
        E = m.addMVar(eshape, lb=0, ub=bat.capacity)
        m.addConstr(E[..., 1:] == E[..., :-1] + DT * (bat.eta * bm - bp / bat.mu))
        if kind == "2s":
            self.e0 = m.addConstr(E[:, 0] == np.full(M, bat.capacity / 2))
        else:
            self.e0 = m.addConstr(E[0] == bat.capacity / 2)
        # G: throughput cumulated within the calendar day. At the first period of
        # the next day the variable `reset` (free there, fixed to 0 elsewhere)
        # lets G restart from zero, so that each day has its own budget and
        # the budget not used today is not carried over to tomorrow.
        G = m.addMVar(shape, lb=0)
        reset = m.addMVar(shape, lb=0, ub=0)
        m.addConstr(G[..., 0] == DT * (bp[..., 0] / bat.mu + bat.eta * bm[..., 0]))
        m.addConstr(reset[..., 1:] <= G[..., :-1])  # only the throughput of the previous day is dropped
        m.addConstr(G[..., 1:] == G[..., :-1] + DT * (bp[..., 1:] / bat.mu + bat.eta * bm[..., 1:]) - reset[..., 1:])
        if kind == "2s":
            m.addConstr(bp[1:, 0] == bp[0, 0])
            m.addConstr(bm[1:, 0] == bm[0, 0])
        self.yp = m.addMVar((M, H), lb=0)
        self.ym = m.addMVar((M, H), lb=0)
        if kind == "2s":
            self.bal = m.addConstr(self.yp - self.ym + bp - bm == np.zeros((M, H)))
        else:
            self.bal = m.addConstr(self.yp - self.ym + bp[None, :] - bm[None, :] == np.zeros((M, H)))
        scale = 1.0 / M if kind == "2s" else 1.0
        m.setObjective(scale * (tariff.wear * DT * bp.sum() - eps * E[..., 1:].sum()), GRB.MINIMIZE)
        m.Params.Method = 1 if kind == "2s" else 0
        self.m, self.bp, self.bm, self.E, self.G, self.reset = m, bp, bm, E, G, reset

    def solve(self, load, pv, x, E0, rem_today, tau):
        """Scenarios (M, H) of load, PV and signal for the window. Returns a dict
        with the first decision and, for det/pb, the planned trajectory."""
        H = T
        load, pv, x = (np.atleast_2d(a) for a in (load, pv, x))
        self.bal.RHS = load - pv
        self.yp.Obj = (DT / self.M) * self.tariff.buy(x)
        self.ym.Obj = -(DT / self.M) * self.tariff.sell(x)
        self.e0.RHS = np.full(self.M, E0) if self.kind == "2s" else E0
        today = np.arange(H) < H - tau
        self.G.UB = np.broadcast_to(np.where(today, rem_today, self.bat.budget), self.G.shape)
        midnight = (np.arange(H) == H - tau) & (tau > 0)
        self.reset.UB = np.broadcast_to(np.where(midnight, GRB.INFINITY, 0.0), self.reset.shape)
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            self.m.reset()
            self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"receding LP status {self.m.Status}")
        bp, bm = self.bp.X, self.bm.X
        if self.kind == "2s":
            return dict(bp0=float(bp[0, 0]), bm0=float(bm[0, 0]), time=self.m.Runtime)
        return dict(bp0=float(bp[0]), bm0=float(bm[0]), bp=bp, bm=bm, E=self.E.X, time=self.m.Runtime)
