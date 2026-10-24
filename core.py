"""
core.py

Shared components for the experiments:
  - system parameters;
  - PCA-based scenario model (unconditional and conditional sampling);
  - Automatic Battery (AB) rule-based policy;
  - linear programs for the PB, OB, FB0-FB3, two-stage and deterministic models;
  - simulator that applies a battery policy to realized trajectories, with a
    projection onto the feasible set (used for out-of-sample evaluation).

All models use an explicit state-of-charge variable E_t, t = 0..H, with
E_{t+1} = E_t + dt * (eta * b^-_t - b^+_t / mu), which is equivalent to the
cumulative-sum formulation of the paper but yields sparser LPs.
"""
import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# -------------------------
# System parameters
# -------------------------
T = 96
DT = 0.25
P_B = 5.0           # battery power [kW]
B_MAX = 13.5        # battery capacity [kWh]
ETA = MU = np.sqrt(0.96)
P_MAX = 6.4         # grid exchange limit [kW]
C_CYC = 2           # equivalent cycles per day
BETA = 32.9         # gCO2e/kWh of energy drawn from the battery
P_PV = 6.4
CYCLE_BUDGET = 2 * C_CYC * B_MAX  # bound on dt*(sum b+/mu + eta*sum b-)

ENV = None
# Number of Gurobi threads per process (set GRB_THREADS in the environment)
THREADS = int(os.environ.get("GRB_THREADS", "1"))


def env(threads=None):
    global ENV
    if ENV is None:
        ENV = gp.Env(empty=True)
        ENV.setParam("OutputFlag", 0)
        ENV.setParam("Threads", threads or THREADS)
        ENV.start()
    return ENV


# =============================================================================
# Scenario model
# =============================================================================
class ScenarioModel:
    """
    Joint PCA model of the daily profiles of load, PV generation and log carbon
    intensity. Each block is centred and scaled by its overall standard
    deviation, the three blocks are stacked into a 3T-dimensional vector, and
    the first K principal components are retained. The residual variance
    (probabilistic PCA) is used as observation noise when conditioning on a
    partially observed day.
    """

    def __init__(self, load, solar, alpha, K=5):
        X = np.hstack([load, solar, np.log(alpha)])
        self.n = len(X)
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        self.scale = np.repeat([Xc[:, i * T:(i + 1) * T].std() + 1e-12 for i in range(3)], T)
        Z = Xc / self.scale
        _, sig, Vt = np.linalg.svd(Z, full_matrices=False)
        lam = sig ** 2 / max(self.n - 1, 1)
        self.K = min(K, len(lam))
        self.lam = lam[:self.K]
        self.V = Vt[:self.K].T                      # (3T, K)
        self.explained = lam[:self.K].sum() / lam.sum()
        D = 3 * T
        self.sigma2 = max(lam[self.K:].sum() / (D - self.K), 1e-8)
        self.W = self.V * np.sqrt(np.maximum(self.lam - self.sigma2, 1e-12))
        self.pv_cap = P_PV
        self.night = solar.max(axis=0) <= 1e-6   # slots with no PV in the data

    def _to_profiles(self, Zs):
        X = Zs * self.scale + self.mean
        load = np.clip(X[:, :T], 0, None)
        solar = np.clip(X[:, T:2 * T], 0, self.pv_cap)
        solar[:, self.night] = 0.0
        alpha = np.exp(X[:, 2 * T:])
        return load, solar, alpha

    def sample(self, n, rng):
        """Unconditional sampling: z_k ~ N(0, lambda_k) on the retained components."""
        z = rng.normal(0, np.sqrt(self.lam), size=(n, self.K))
        return self._to_profiles(z @ self.V.T)

    def sample_conditional(self, obs_load, obs_solar, obs_alpha, n, rng):
        """
        Sample complete daily profiles given the first tau observed values of
        each signal (tau = len(obs_load)). Observed entries are returned as observed.
        """
        tau = len(obs_load)
        if tau == 0:
            z = rng.normal(size=(n, self.K))
            out = self._to_profiles(z @ self.W.T)
            return out
        idx = np.concatenate([np.arange(tau) + i * T for i in range(3)])
        x_obs = np.concatenate([obs_load, obs_solar, np.log(obs_alpha)])
        r = (x_obs - self.mean[idx]) / self.scale[idx]
        Wo = self.W[idx]
        prec = np.eye(self.K) + Wo.T @ Wo / self.sigma2
        cov = np.linalg.inv(prec)
        mz = cov @ Wo.T @ r / self.sigma2
        Lc = np.linalg.cholesky((cov + cov.T) / 2)
        z = mz + rng.normal(size=(n, self.K)) @ Lc.T
        load, solar, alpha = self._to_profiles(z @ self.W.T)
        load[:, :tau], solar[:, :tau], alpha[:, :tau] = obs_load, obs_solar, obs_alpha
        return load, solar, alpha


# =============================================================================
# Automatic Battery (rule-based self-consumption)
# =============================================================================
def ab_step(E, net, used, budget=CYCLE_BUDGET):
    """
    One step of the self-consumption rule, vectorized over scenarios.
    net = l - s. Returns (b_plus, b_minus). `used` is the throughput
    dt*(sum b+/mu + eta*sum b-) already consumed in the current day.
    """
    rem = np.maximum(budget - used, 0)
    charge = np.minimum.reduce([np.maximum(-net, 0), (B_MAX - E) / (DT * ETA),
                                np.full_like(E, P_B), rem / (DT * ETA)])
    discharge = np.minimum.reduce([np.maximum(net, 0), MU * E / DT,
                                   np.full_like(E, MU * P_B), rem * MU / DT])
    return np.maximum(discharge, 0), np.maximum(charge, 0)


def ab_day(load, solar, E0):
    """Apply AB to a set of daily trajectories (M, T) starting from E0 (M,)."""
    M = load.shape[0]
    E = np.array(E0, dtype=float) * np.ones(M)
    used = np.zeros(M)
    bp = np.zeros((M, T)); bm = np.zeros((M, T))
    for t in range(T):
        bp[:, t], bm[:, t] = ab_step(E, load[:, t] - solar[:, t], used)
        used += DT * (bp[:, t] / MU + ETA * bm[:, t])
        E = E + DT * (ETA * bm[:, t] - bp[:, t] / MU)
    return bp, bm, E


def ab_periodic(load, solar, iters=50):
    """
    AB in cyclostationary regime: the initial state of charge of each day is
    the fixed point E0 = E_T(E0) of the daily map (same day repeated), which
    mirrors the periodicity constraint imposed on the optimization models.
    """
    E0 = np.zeros(load.shape[0])
    for _ in range(iters):
        bp, bm, ET = ab_day(load, solar, E0)
        if np.max(np.abs(ET - E0)) < 1e-6:
            break
        E0 = ET
    bp, bm, ET = ab_day(load, solar, E0)
    return bp, bm, E0, ET


# =============================================================================
# Evaluation
# =============================================================================
def emissions(load, solar, alpha, bp, bm, E0=None, ET=None):
    """
    Daily emissions [gCO2e] for each scenario: dt*sum(alpha*y+) + beta*dt*sum(b+/mu).
    If E0, ET are given, a policy that ends the day with less energy than it
    started with is charged the emissions needed to restore the deficit from the
    grid, (E0-ET)^+ * mean(alpha) / eta; no credit is given for a surplus.
    Returns (emissions, grid_limit_violation_flag).
    """
    y = load - solar - bp + bm
    yp = np.maximum(y, 0)
    viol = (np.abs(y) > P_MAX + 1e-6).any(axis=1)
    e = DT * (alpha * yp).sum(axis=1) + BETA * DT * bp.sum(axis=1) / MU
    if E0 is not None:
        e = e + np.maximum(E0 - ET, 0) * alpha.mean(axis=1) / ETA
    return e, viol


def simulate_affine(load, solar, policy, E0):
    """
    Apply a (possibly affine-feedback) policy to the trajectories (M, T),
    projecting each raw action onto the feasible set given the current state
    of charge and the residual cycle budget.

    policy(t, load, solar) -> raw (b_plus, b_minus) arrays of shape (M,),
    computed from observations up to t-1 only.
    Returns bp, bm, ET, projected (bool per scenario).
    """
    M = load.shape[0]
    E = np.full(M, float(E0)) if np.isscalar(E0) else np.array(E0, float)
    used = np.zeros(M)
    bp = np.zeros((M, T)); bm = np.zeros((M, T))
    proj = np.zeros(M, bool)
    for t in range(T):
        rp, rm = policy(t, load, solar)
        rp = rp * np.ones(M); rm = rm * np.ones(M)
        rem = np.maximum(CYCLE_BUDGET - used, 0)
        p = np.clip(rp, 0, np.minimum.reduce([np.full(M, MU * P_B), MU * E / DT, rem * MU / DT]))
        m = np.clip(rm, 0, np.minimum.reduce([np.full(M, P_B), (B_MAX - E) / (DT * ETA),
                                              np.maximum(rem - DT * p / MU, 0) / (DT * ETA)]))
        proj |= (np.abs(p - rp) > 1e-5) | (np.abs(m - rm) > 1e-5)
        bp[:, t], bm[:, t] = p, m
        used += DT * (p / MU + ETA * m)
        E = E + DT * (ETA * m - p / MU)
    return bp, bm, E, proj


def schedule_policy(bp_sched, bm_sched):
    return lambda t, l, s: (bp_sched[t], bm_sched[t])


def fb_policy(kind, coef):
    """Affine feedback policy from the coefficients returned by solve_fb."""
    lp, lm = coef["lam_p"], coef["lam_m"]
    if kind == "FB0":
        Pp, Pm, Sp, Sm = coef["phi_p"], coef["phi_m"], coef["psi_p"], coef["psi_m"]

        def pol(t, l, s):
            return (lp[t] + l[:, :t] @ Pp[t, :t] + s[:, :t] @ Sp[t, :t],
                    lm[t] + l[:, :t] @ Pm[t, :t] + s[:, :t] @ Sm[t, :t])
    elif kind == "FB1":
        Pp, Pm, Sp, Sm = coef["phi_p"], coef["phi_m"], coef["psi_p"], coef["psi_m"]

        def pol(t, l, s):
            return (lp[t] + l[:, :t] @ Pp[:t] + s[:, :t] @ Sp[:t],
                    lm[t] + l[:, :t] @ Pm[:t] + s[:, :t] @ Sm[:t])
    else:
        Pp, Pm, Sp, Sm = coef["phi_p"], coef["phi_m"], coef["psi_p"], coef["psi_m"]

        def pol(t, l, s):
            fl, fs = fb_features(kind, l, s, t)
            return lp[t] + Pp[t] * fl + Sp[t] * fs, lm[t] + Pm[t] * fl + Sm[t] * fs
    return pol


def fb_features(kind, l, s, t):
    """Features of FB2 (cumulative sums) and FB3 (last observation) at time t."""
    if t == 0:
        return np.zeros(l.shape[0]), np.zeros(l.shape[0])
    if kind == "FB2":
        return l[:, :t].sum(axis=1), s[:, :t].sum(axis=1)
    return l[:, t - 1], s[:, t - 1]


# =============================================================================
# Linear programs
# =============================================================================
def _soc_block(m, H, shape, E0, bp, bm, name):
    """SOC variables E (shape + (H+1,)) with dynamics and bounds."""
    E = m.addMVar(shape + (H + 1,), lb=0, ub=B_MAX, name=f"E{name}")
    m.addConstr(E[..., 1:] == E[..., :-1] + DT * (ETA * bm - bp / MU))
    if E0 is not None:
        m.addConstr(E[..., 0] == E0)
    return E


def solve_pb(load, solar, alpha, E0=None, cyclic=True, E_end=None, budget=CYCLE_BUDGET,
             time_limit=None, soc_reward=0.0):
    """
    Programmed Battery: scenario-independent schedule (b+, b-), scenario-dependent
    grid exchanges. E0=None means the initial SOC is a decision variable.
    cyclic: E_H = E_0. E_end: fixed terminal SOC (overrides cyclic).
    soc_reward: tie-breaking reward per kWh of stored energy and period.
    """
    M, H = load.shape
    m = gp.Model(env=env())
    bp = m.addMVar(H, lb=0, ub=MU * P_B)
    bm = m.addMVar(H, lb=0, ub=P_B)
    E = _soc_block(m, H, (), E0, bp, bm, "")
    yp = m.addMVar((M, H), lb=0, ub=P_MAX)
    ym = m.addMVar((M, H), lb=0, ub=P_MAX)
    m.addConstr(yp - ym + bp[None, :] - bm[None, :] == load - solar)
    m.addConstr(DT * (bp.sum() / MU + ETA * bm.sum()) <= budget)
    if E_end is not None:
        m.addConstr(E[H] == E_end)
    elif cyclic:
        m.addConstr(E[H] == E[0])
    m.setObjective((DT / M) * (alpha * yp).sum() + BETA * DT / MU * bp.sum() - soc_reward * E[1:].sum(),
                   GRB.MINIMIZE)
    if time_limit:
        m.Params.TimeLimit = time_limit
    m.optimize()
    out = dict(obj=m.ObjVal + soc_reward * E.X[1:].sum(), bp=bp.X, bm=bm.X, E=E.X, time=m.Runtime)
    m.dispose()
    return out


def solve_ob(load, solar, alpha, E0=None, cyclic=True, budget=CYCLE_BUDGET, chunk=500):
    """
    Omniscient Battery: scenario-dependent decisions (perfect information).
    The problem is separable by scenario and is solved in chunks.
    Returns per-scenario objective values and decisions.
    """
    M, H = load.shape
    objs = np.zeros(M); BP = np.zeros((M, H)); BM = np.zeros((M, H)); EE = np.zeros((M, H + 1))
    rt = 0.0
    for a in range(0, M, chunk):
        sl = slice(a, min(a + chunk, M)); n = sl.stop - sl.start
        m = gp.Model(env=env())
        bp = m.addMVar((n, H), lb=0, ub=MU * P_B)
        bm = m.addMVar((n, H), lb=0, ub=P_B)
        E = _soc_block(m, H, (n,), E0, bp, bm, "")
        yp = m.addMVar((n, H), lb=0, ub=P_MAX)
        ym = m.addMVar((n, H), lb=0, ub=P_MAX)
        m.addConstr(yp - ym + bp - bm == load[sl] - solar[sl])
        m.addConstr(DT * (bp.sum(axis=1) / MU + ETA * bm.sum(axis=1)) <= budget)
        if cyclic:
            m.addConstr(E[:, H] == E[:, 0])
        m.setObjective((alpha[sl] * yp).sum() * DT + BETA * DT / MU * bp.sum(), GRB.MINIMIZE)
        m.optimize()
        rt += m.Runtime
        BP[sl], BM[sl], EE[sl] = bp.X, bm.X, E.X
        objs[sl] = DT * (alpha[sl] * yp.X).sum(axis=1) + BETA * DT / MU * bp.X.sum(axis=1)
        m.dispose()
    return dict(obj=objs, bp=BP, bm=BM, E=EE, time=rt)


def solve_fb(kind, load, solar, alpha, E0=None, cyclic=True, E_end=None, budget=CYCLE_BUDGET,
             coef_bound=P_B, time_limit=None, obs_prev=None, soc_reward=0.0):
    """
    Feedback Battery models: b_{t,xi} affine in the observations up to t-1.
      FB0: full lower-triangular coefficient matrices (phi_{t,r}, psi_{t,r});
      FB1: time-invariant coefficients (phi_r, psi_r);
      FB2: coefficients on the cumulative sums of past load and PV;
      FB3: coefficients on the last observation only (l_{t-1}, s_{t-1}).
    The initial SOC is common to all scenarios (non-anticipative); E0=None
    means it is a decision variable.
    obs_prev: optional (l, s) observed just before the horizon, used as the
    t = 0 feature of FB3 in the receding-horizon setting.
    """
    M, H = load.shape
    m = gp.Model(env=env())
    lb, ub = -coef_bound, coef_bound
    lam_p = m.addMVar(H, lb=lb, ub=ub)
    lam_m = m.addMVar(H, lb=lb, ub=ub)
    bp = m.addMVar((M, H), lb=0, ub=MU * P_B)
    bm = m.addMVar((M, H), lb=0, ub=P_B)
    if kind == "FB0":
        mask = np.tril(np.ones((H, H)), -1)
        phi_p = m.addMVar((H, H), lb=lb * mask, ub=ub * mask)
        phi_m = m.addMVar((H, H), lb=lb * mask, ub=ub * mask)
        psi_p = m.addMVar((H, H), lb=lb * mask, ub=ub * mask)
        psi_m = m.addMVar((H, H), lb=lb * mask, ub=ub * mask)
        m.addConstr(bp == lam_p[None, :] + load @ phi_p.T + solar @ psi_p.T)
        m.addConstr(bm == lam_m[None, :] + load @ phi_m.T + solar @ psi_m.T)
    elif kind == "FB1":
        phi_p, phi_m, psi_p, psi_m = (m.addMVar(H, lb=lb, ub=ub) for _ in range(4))
        Cp = m.addMVar((M, H), lb=-GRB.INFINITY)
        Cm = m.addMVar((M, H), lb=-GRB.INFINITY)
        m.addConstr(Cp[:, 0] == 0); m.addConstr(Cm[:, 0] == 0)
        m.addConstr(Cp[:, 1:] == Cp[:, :-1] + load[:, :-1] * phi_p[None, :-1] + solar[:, :-1] * psi_p[None, :-1])
        m.addConstr(Cm[:, 1:] == Cm[:, :-1] + load[:, :-1] * phi_m[None, :-1] + solar[:, :-1] * psi_m[None, :-1])
        m.addConstr(bp == lam_p[None, :] + Cp)
        m.addConstr(bm == lam_m[None, :] + Cm)
    else:
        phi_p, phi_m, psi_p, psi_m = (m.addMVar(H, lb=lb, ub=ub) for _ in range(4))
        Fl, Fs = np.zeros((M, H)), np.zeros((M, H))
        if kind == "FB2":
            Fl[:, 1:] = np.cumsum(load, axis=1)[:, :-1]
            Fs[:, 1:] = np.cumsum(solar, axis=1)[:, :-1]
        else:
            Fl[:, 1:], Fs[:, 1:] = load[:, :-1], solar[:, :-1]
            if obs_prev is not None:
                Fl[:, 0], Fs[:, 0] = obs_prev
        m.addConstr(bp == lam_p[None, :] + Fl * phi_p[None, :] + Fs * psi_p[None, :])
        m.addConstr(bm == lam_m[None, :] + Fl * phi_m[None, :] + Fs * psi_m[None, :])
    B0 = m.addVar(lb=0, ub=B_MAX) if E0 is None else E0
    E = m.addMVar((M, H + 1), lb=0, ub=B_MAX)
    m.addConstr(E[:, 1:] == E[:, :-1] + DT * (ETA * bm - bp / MU))
    m.addConstr(E[:, 0] == B0)
    if E_end is not None:
        m.addConstr(E[:, H] == E_end)
    elif cyclic:
        m.addConstr(E[:, H] == B0)
    yp = m.addMVar((M, H), lb=0, ub=P_MAX)
    ym = m.addMVar((M, H), lb=0, ub=P_MAX)
    m.addConstr(yp - ym + bp - bm == load - solar)
    m.addConstr(DT * (bp.sum(axis=1) / MU + ETA * bm.sum(axis=1)) <= budget)
    m.setObjective((DT / M) * (alpha * yp).sum() + BETA * DT / (MU * M) * bp.sum()
                   - (soc_reward / M) * E[:, 1:].sum(), GRB.MINIMIZE)
    if time_limit:
        m.Params.TimeLimit = time_limit
    m.Params.Method = 2
    m.optimize()
    if m.SolCount == 0:
        m.dispose()
        return None
    coef = dict(lam_p=lam_p.X, lam_m=lam_m.X, phi_p=phi_p.X, phi_m=phi_m.X, psi_p=psi_p.X, psi_m=psi_m.X)
    out = dict(obj=m.ObjVal, coef=coef, B0=(B0.X if E0 is None else E0), time=m.Runtime,
               status=m.Status, bp0=bp.X[:, 0].mean(), bm0=bm.X[:, 0].mean())
    m.dispose()
    return out


def solve_two_stage(load, solar, alpha, E0, budget=CYCLE_BUDGET):
    """
    Two-stage stochastic model for the receding-horizon setting: the first-period
    decision is common to all scenarios (here-and-now), while all subsequent
    decisions adapt to the scenario (full recourse).
    """
    M, H = load.shape
    m = gp.Model(env=env())
    bp = m.addMVar((M, H), lb=0, ub=MU * P_B)
    bm = m.addMVar((M, H), lb=0, ub=P_B)
    E = _soc_block(m, H, (M,), E0, bp, bm, "")
    m.addConstr(bp[1:, 0] == bp[0, 0]); m.addConstr(bm[1:, 0] == bm[0, 0])
    yp = m.addMVar((M, H), lb=0, ub=P_MAX)
    ym = m.addMVar((M, H), lb=0, ub=P_MAX)
    m.addConstr(yp - ym + bp - bm == load - solar)
    m.addConstr(DT * (bp.sum(axis=1) / MU + ETA * bm.sum(axis=1)) <= budget)
    m.setObjective((DT / M) * (alpha * yp).sum() + BETA * DT / (MU * M) * bp.sum(), GRB.MINIMIZE)
    m.optimize()
    out = dict(obj=m.ObjVal, bp0=bp.X[0, 0], bm0=bm.X[0, 0], time=m.Runtime)
    m.dispose()
    return out
