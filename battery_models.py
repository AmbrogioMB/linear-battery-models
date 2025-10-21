"""
battery_models.py

Contains linear optimization models for PV-BESS emission minimization:
 - OB_model (Omniscient Battery)
 - PB      (Programmed Battery)
 - FB2_model
 - FB1_model
 - FB0_model

All models use Gurobi for optimization and return the solution arrays
in the same format as your original scripts.
"""
import numpy as np
from gurobipy import Model, GRB, quicksum

# -------------------------
# Shared constants / config
# -------------------------
T = 96         # number of time steps (15-min intervals)
dt = 0.25      # time step length [h]
c_cycles = 2   # cycle limit
beta = 32.9    # battery lifecycle carbon equivalent (gCO2e/kWh) used in objective

# -------------------------
# Utility to unpack u
# -------------------------
def _unpack_u(u):
    """
    Unpack user tuple `u` into common parameters.

    u expected as (P_b_max, B_max, eta_RT, P_max)
    where eta_RT is a round-trip efficiency parameter used as sqrt in code.
    """
    P_b_max = u[0]
    B_max = u[1]
    P_max = u[3]
    mu = np.sqrt(u[2])
    eta = np.sqrt(u[2])
    return P_b_max, B_max, P_max, mu, eta

# -------------------------
# Omniscient Battery (OB)
# -------------------------
def OB_model(u, solar, loads, alpha):
    """
    Omniscient Battery model.
    Assumes that decisions can be scenario-dependent (perfect foresight).
    This function follows your original OB implementation closely.

    Inputs:
      u     : tuple (P_b_max, B_max, eta_RT, P_max)
      solar : array-like of shape (M, T)
      loads : array-like of shape (M, T)
      alpha : array-like of shape (M, T) - carbon intensity per scenario/time

    Returns:
      Y_p, Y_m, B_p, B_m, B
      where each is a list (length M) of arrays length T (or lists).
    """
    M = len(solar)
    P_b_max, B_max, P_max, mu, eta = _unpack_u(u)

    # dictionaries for Gurobi variables
    y_p = {}
    y_m = {}
    b_p = {}
    b_m = {}
    B_0 = {}

    model = Model("OB")
    model.ModelSense = GRB.MINIMIZE

    # Create variables and constraints per scenario xi
    for xi in range(M):
        # Initial battery energy as variable (scenario dependent)
        B_0[xi] = model.addVar(lb=0, ub=B_max, obj=0, name=f"B0_{xi}")

        for t in range(T):
            # grid import/export (y_p = import, y_m = export)
            y_p[(t, xi)] = model.addVar(lb=0, ub=P_max, obj=(1.0 / M) * alpha[xi][t] * dt, name=f"y_p_{t}_{xi}")
            y_m[(t, xi)] = model.addVar(lb=0, ub=P_max, obj=0, name=f"y_m_{t}_{xi}")

            # battery charge/discharge (b_p = discharge, b_m = charge)
            b_p[(t, xi)] = model.addVar(lb=0, ub=mu * P_b_max, obj=beta * dt / (mu * M), name=f"b_p_{t}_{xi}")
            b_m[(t, xi)] = model.addVar(lb=0, ub=P_b_max, obj=0, name=f"b_m_{t}_{xi}")

            # power balance constraint (per t, xi)
            model.addConstr(
                y_p[(t, xi)] - y_m[(t, xi)]
                == loads[xi][t] - solar[xi][t] - b_p[(t, xi)] + b_m[(t, xi)],
                name=f"balance_{t}_{xi}"
            )

            # battery state constraints (per prefix up to t)
            expr_b_p = quicksum((1.0 / mu) * b_p[(r, xi)] for r in range(t + 1))
            expr_b_m = quicksum(eta * b_m[(r, xi)] for r in range(t + 1))

            model.addConstr(B_0[xi] - dt * (expr_b_p - expr_b_m) <= B_max, name=f"cap_ub_{t}_{xi}")
            model.addConstr(B_0[xi] - dt * (expr_b_p - expr_b_m) >= 0, name=f"cap_lb_{t}_{xi}")

        # energy neutrality for scenario xi
        model.addConstr(quicksum((1.0 / mu) * b_p[(t, xi)] - eta * b_m[(t, xi)] for t in range(T)) == 0, name=f"neutral_{xi}")

        # cycle constraint for scenario xi
        model.addConstr(dt * ( (1.0 / mu) * quicksum(b_p[(t, xi)] for t in range(T)) + eta * quicksum(b_m[(t, xi)] for t in range(T)) ) <= 2 * c_cycles * B_max, name=f"cycle_{xi}")

    # Solve
    model.optimize()

    # Extract solution arrays
    Y_p = [[0] * T for _ in range(M)]
    Y_m = [[0] * T for _ in range(M)]
    B_p = [[0] * T for _ in range(M)]
    B_m = [[0] * T for _ in range(M)]
    B = [[0] * T for _ in range(M)]

    for xi in range(M):
        for t in range(T):
            Y_p[xi][t] = y_p[(t, xi)].X
            Y_m[xi][t] = y_m[(t, xi)].X
            B_p[xi][t] = b_p[(t, xi)].X
            B_m[xi][t] = b_m[(t, xi)].X

        # compute B trajectory from B_0 and cumulative flows
        for t in range(T):
            cumulative_b_p = sum(b_p[(j, xi)].X for j in range(t + 1))
            cumulative_b_m = sum(b_m[(j, xi)].X for j in range(t + 1))
            B[xi][t] = B_0[xi].X - dt * ((1.0 / mu) * cumulative_b_p - eta * cumulative_b_m)

    return Y_p, Y_m, B_p, B_m, B

# -------------------------
# Programmed Battery (PB)
# -------------------------
def PB(u, solar, loads, alpha):
    """
    Programmed Battery (PB) model.
    Decisions b_p[t], b_m[t] are planned ex ante (scenario-independent),
    while grid exchanges y_p[(t,xi)], y_m[(t,xi)] remain scenario-dependent.

    Inputs:
      u, solar, loads, alpha same as OB_model.

    Returns:
      Y_p, Y_m, B_p, B_m, B
    """
    M = len(solar)
    P_b_max, B_max, P_max, mu, eta = _unpack_u(u)

    # Dictionaries for variables
    y_p = {}
    y_m = {}
    b_p = {}
    b_m = {}
    # B_0 is a single variable (not scenario-dependent in your PB code)
    model = Model("PB")
    model.ModelSense = GRB.MINIMIZE

    # Single initial battery energy variable (shared across scenarios in PB)
    B_0 = model.addVar(lb=0, ub=B_max, obj=0, name="B_0")

    # Create time-coupled battery variables (scenario-independent planned schedule)
    for t in range(T):
        b_p[t] = model.addVar(lb=0, ub=mu * P_b_max, obj=beta * dt / mu, name=f"b_p_{t}")
        b_m[t] = model.addVar(lb=0, ub=P_b_max, obj=0, name=f"b_m_{t}")

    # scenario-dependent grid variables and balance constraints
    for xi in range(M):
        for t in range(T):
            y_p[(t, xi)] = model.addVar(lb=0, ub=P_max, obj=(1.0 / M) * alpha[xi][t] * dt, name=f"y_p_{t}_{xi}")
            y_m[(t, xi)] = model.addVar(lb=0, ub=P_max, obj=0, name=f"y_m_{t}_{xi}")

            # power balance: uses planned b_p[t], b_m[t] (same for all scenarios)
            model.addConstr(
                y_p[(t, xi)] - y_m[(t, xi)]
                == loads[xi][t] - solar[xi][t] - b_p[t] + b_m[t],
                name=f"balance_{t}_{xi}"
            )

    # battery capacity bounds for each prefix (shared B_0 and shared b_p/b_m)
    for t in range(T):
        model.addConstr(B_0 - dt * quicksum((1.0 / mu) * b_p[r] - eta * b_m[r] for r in range(t + 1)) >= 0, name=f"cap_lb_{t}")
        model.addConstr(B_0 - dt * quicksum((1.0 / mu) * b_p[r] - eta * b_m[r] for r in range(t + 1)) <= B_max, name=f"cap_ub_{t}")

    # energy neutrality and cycle constraint (using planned schedule)
    model.addConstr(quicksum((1.0 / mu) * b_p[t] - eta * b_m[t] for t in range(T)) == 0, name="neutral")
    model.addConstr(dt * ((1.0 / mu) * quicksum(b_p[t] for t in range(T)) + eta * quicksum(b_m[t] for t in range(T))) <= 2 * c_cycles * B_max, name="cycle")

    # solve
    model.optimize()

    # Extract results
    M = len(solar)
    Y_p = [[0] * T for _ in range(M)]
    Y_m = [[0] * T for _ in range(M)]
    B_p = np.array([b_p[t].X for t in range(T)])
    B_m = np.array([b_m[t].X for t in range(T)])
    B = [0] * T

    for xi in range(M):
        for t in range(T):
            Y_p[xi][t] = y_p[(t, xi)].X
            Y_m[xi][t] = y_m[(t, xi)].X

    for t in range(T):
        cumulative_b_p = sum(b_p[j].X for j in range(t + 1))
        cumulative_b_m = sum(b_m[j].X for j in range(t + 1))
        B[t] = B_0.X - dt * ((1.0 / mu) * cumulative_b_p - eta * cumulative_b_m)

    return Y_p, Y_m, B_p, B_m, B

# -------------------------
# FB2: aggregated-sum feedback model
# -------------------------
def FB2_model(u, solar, loads, alpha):
    """
    FB2: feedback model where b_t depends on aggregated sums of past load/solar:
      b_{t,xi}^+ = lambda_t^+ + phi_t^+ * sum_{r=1}^{t-1} l_{r,xi} + psi_t^+ * sum_{r=1}^{t-1} s_{r,xi}
    and similarly for b^-.

    This implementation follows your original FB2 code structure (linear variables for coefficients).
    """
    M = len(solar)
    P_b_max, B_max, P_max, mu, eta = _unpack_u(u)

    # decision variables containers
    y_p = {}
    y_m = {}
    b_p = {}
    b_m = {}
    B_0 = {}
    # coefficients per time t
    alpha_p = {}
    alpha_m = {}
    beta_p = {}
    beta_m = {}
    gamma_p = {}
    gamma_m = {}
    Y_bin = {}  # binary to avoid simultaneous charge/discharge

    model = Model("FB2")
    model.ModelSense = GRB.MINIMIZE

    # coefficient variables (one scalar per t for each coefficient type)
    for t in range(T):
        alpha_p[t] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, name=f"alpha_p_{t}")
        alpha_m[t] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, name=f"alpha_m_{t}")
        beta_p[t] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, name=f"beta_p_{t}")
        beta_m[t] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, name=f"beta_m_{t}")
        gamma_p[t] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, name=f"gamma_p_{t}")
        gamma_m[t] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, name=f"gamma_m_{t}")

    # scenario-dependent variables and constraints
    for xi in range(M):
        B_0[xi] = model.addVar(lb=0, ub=B_max, obj=0, name=f"B0_{xi}")

        for t in range(T):
            y_p[(t, xi)] = model.addVar(lb=0, ub=P_max, obj=(1.0 / M) * alpha[xi][t] * dt, name=f"y_p_{t}_{xi}")
            y_m[(t, xi)] = model.addVar(lb=0, ub=P_max, obj=0, name=f"y_m_{t}_{xi}")
            b_p[(t, xi)] = model.addVar(lb=0, ub=mu * P_b_max, obj=beta * dt / (mu * M), name=f"b_p_{t}_{xi}")
            b_m[(t, xi)] = model.addVar(lb=0, ub=P_b_max, obj=0, name=f"b_m_{t}_{xi}")
            Y_bin[(t, xi)] = model.addVar(obj=0, vtype=GRB.BINARY, name=f"Ybin_{t}_{xi}")

            # power balance
            model.addConstr(y_p[(t, xi)] - y_m[(t, xi)] == loads[xi][t] - solar[xi][t] - b_p[(t, xi)] + b_m[(t, xi)], name=f"bal_{t}_{xi}")

            # build linear expressions for aggregated sums of past values
            # note: sums from r=0..t-1 (r < t)
            if t == 0:
                sum_l = 0
                sum_s = 0
            else:
                sum_l = quicksum(loads[xi][r] for r in range(t))
                sum_s = quicksum(solar[xi][r] for r in range(t))

            # define b_p and b_m as affine functions of aggregated sums
            model.addConstr(b_p[(t, xi)] == alpha_p[t] * sum_l + beta_p[t] * sum_s + gamma_p[t], name=f"b_p_def_{t}_{xi}")
            model.addConstr(b_m[(t, xi)] == alpha_m[t] * sum_l + beta_m[t] * sum_s + gamma_m[t], name=f"b_m_def_{t}_{xi}")

            # prevent simultaneous charge/discharge using binary Y
            model.addConstr(b_p[(t, xi)] <= P_b_max * Y_bin[(t, xi)], name=f"no_simul_p_{t}_{xi}")
            model.addConstr(b_m[(t, xi)] <= P_b_max * (1 - Y_bin[(t, xi)]), name=f"no_simul_m_{t}_{xi}")

            # battery bounds via cumulative expressions
            expr_b_p = quicksum((1.0 / mu) * b_p[(r, xi)] for r in range(t + 1))
            expr_b_m = quicksum(eta * b_m[(r, xi)] for r in range(t + 1))

            model.addConstr(B_0[xi] - dt * (expr_b_p - expr_b_m) <= B_max, name=f"cap_ub_{t}_{xi}")
            model.addConstr(B_0[xi] - dt * (expr_b_p - expr_b_m) >= 0, name=f"cap_lb_{t}_{xi}")

        # energy neutrality and cycle constraint
        model.addConstr(quicksum((1.0 / mu) * b_p[(t, xi)] - eta * b_m[(t, xi)] for t in range(T)) == 0, name=f"neutral_{xi}")
        model.addConstr(dt * ((1.0 / mu) * quicksum(b_p[(t, xi)] for t in range(T)) + eta * quicksum(b_m[(t, xi)] for t in range(T))) <= 2 * c_cycles * B_max, name=f"cycle_{xi}")

    # solve
    model.optimize()

    # extract results
    Y_p = [[0] * T for _ in range(M)]
    Y_m = [[0] * T for _ in range(M)]
    B_p = [[0] * T for _ in range(M)]
    B_m = [[0] * T for _ in range(M)]
    B = [[0] * T for _ in range(M)]

    for xi in range(M):
        for t in range(T):
            Y_p[xi][t] = y_p[(t, xi)].X
            Y_m[xi][t] = y_m[(t, xi)].X
            B_p[xi][t] = b_p[(t, xi)].X
            B_m[xi][t] = b_m[(t, xi)].X

        for t in range(T):
            cumulative_b_p = sum(b_p[(j, xi)].X for j in range(t + 1))
            cumulative_b_m = sum(b_m[(j, xi)].X for j in range(t + 1))
            B[xi][t] = B_0[xi].X - dt * ((1.0 / mu) * cumulative_b_p - eta * cumulative_b_m)

    # coefficient arrays
    ALPHA_p = np.array([alpha_p[t].X for t in range(T)])
    ALPHA_m = np.array([alpha_m[t].X for t in range(T)])
    BETA_p = np.array([beta_p[t].X for t in range(T)])
    BETA_m = np.array([beta_m[t].X for t in range(T)])
    GAMMA_p = np.array([gamma_p[t].X for t in range(T)])
    GAMMA_m = np.array([gamma_m[t].X for t in range(T)])

    return Y_p, Y_m, B_p, B_m, B, ALPHA_p, ALPHA_m, BETA_p, BETA_m, GAMMA_p, GAMMA_m

# -------------------------
# FB1: time-invariant-feedback model
# -------------------------
def FB1_model(u, solar, loads, alpha):
    """
    FB1: reduces parameter dimensionality by making feedback coefficients
    time-invariant (depend on lag index r but not on t).
    """
    M = len(solar)
    P_b_max, B_max, P_max, mu, eta = _unpack_u(u)

    # decision containers
    y_p = {}
    y_m = {}
    b_p = {}
    b_m = {}
    B_0 = {}
    alpha_p = {}
    alpha_m = {}
    beta_p = {}
    beta_m = {}
    gamma_p = {}
    gamma_m = {}
    Y_bin = {}

    model = Model("FB1")
    model.ModelSense = GRB.MINIMIZE

    # coefficient variables: alpha_p[r], beta_p[r], etc., for r in 0..T-1
    for r in range(T):
        alpha_p[r] = model.addVar(lb=-P_b_max, ub=P_b_max, obj=0, name=f"alpha_p_{r}")
        alpha_m[r] = model.addVar(lb=-P_b_max, ub=P_b_max, obj=0, name=f"alpha_m_{r}")
        beta_p[r] = model.addVar(lb=-P_b_max, ub=P_b_max, obj=0, name=f"beta_p_{r}")
        beta_m[r] = model.addVar(lb=-P_b_max, ub=P_b_max, obj=0, name=f"beta_m_{r}")
    for t in range(T):
        gamma_p[t] = model.addVar(lb=-P_b_max, ub=P_b_max, obj=0, name=f"gamma_p_{t}")
        gamma_m[t] = model.addVar(lb=-P_b_max, ub=P_b_max, obj=0, name=f"gamma_m_{t}")

    # scenario constraints
    for xi in range(M):
        B_0[xi] = model.addVar(lb=0, ub=B_max, obj=0, name=f"B0_{xi}")

        for t in range(T):
            y_p[(t, xi)] = model.addVar(lb=0, ub=P_max, obj=(1.0 / M) * alpha[xi][t] * dt, name=f"y_p_{t}_{xi}")
            y_m[(t, xi)] = model.addVar(lb=0, ub=P_max, obj=0, name=f"y_m_{t}_{xi}")
            b_p[(t, xi)] = model.addVar(lb=0, ub=mu * P_b_max, obj=beta * dt / (mu * M), name=f"b_p_{t}_{xi}")
            b_m[(t, xi)] = model.addVar(lb=0, ub=P_b_max, obj=0, name=f"b_m_{t}_{xi}")
            Y_bin[(t, xi)] = model.addVar(obj=0, vtype=GRB.BINARY, name=f"Ybin_{t}_{xi}")

            model.addConstr(y_p[(t, xi)] - y_m[(t, xi)] == loads[xi][t] - solar[xi][t] - b_p[(t, xi)] + b_m[(t, xi)], name=f"bal_{t}_{xi}")

            # feedback expression uses alpha_p[r], beta_p[r] for past indices r < t
            if t == 0:
                bbp = 0
                bbm = 0
            else:
                bbp = quicksum(alpha_p[r] * loads[xi][r] + beta_p[r] * solar[xi][r] for r in range(t))
                bbm = quicksum(alpha_m[r] * loads[xi][r] + beta_m[r] * solar[xi][r] for r in range(t))

            model.addConstr(b_p[(t, xi)] == bbp + gamma_p[t], name=f"b_p_def_{t}_{xi}")
            model.addConstr(b_m[(t, xi)] == bbm + gamma_m[t], name=f"b_m_def_{t}_{xi}")

            # avoid simultaneous charge/discharge
            model.addConstr(b_p[(t, xi)] <= P_b_max * Y_bin[(t, xi)], name=f"no_simul_p_{t}_{xi}")
            model.addConstr(b_m[(t, xi)] <= P_b_max * (1 - Y_bin[(t, xi)]), name=f"no_simul_m_{t}_{xi}")

            expr_b_p = quicksum((1.0 / mu) * b_p[(r, xi)] for r in range(t + 1))
            expr_b_m = quicksum(eta * b_m[(r, xi)] for r in range(t + 1))
            model.addConstr(B_0[xi] - dt * (expr_b_p - expr_b_m) <= B_max, name=f"cap_ub_{t}_{xi}")
            model.addConstr(B_0[xi] - dt * (expr_b_p - expr_b_m) >= 0, name=f"cap_lb_{t}_{xi}")

        model.addConstr(quicksum((1.0 / mu) * b_p[(t, xi)] - eta * b_m[(t, xi)] for t in range(T)) == 0, name=f"neutral_{xi}")
        model.addConstr(dt * ((1.0 / mu) * quicksum(b_p[(t, xi)] for t in range(T)) + eta * quicksum(b_m[(t, xi)] for t in range(T))) <= 2 * c_cycles * B_max, name=f"cycle_{xi}")

    model.optimize()

    # extract
    Y_p = [[0] * T for _ in range(M)]
    Y_m = [[0] * T for _ in range(M)]
    B_p = [[0] * T for _ in range(M)]
    B_m = [[0] * T for _ in range(M)]
    B = [[0] * T for _ in range(M)]

    for xi in range(M):
        for t in range(T):
            Y_p[xi][t] = y_p[(t, xi)].X
            Y_m[xi][t] = y_m[(t, xi)].X
            B_p[xi][t] = b_p[(t, xi)].X
            B_m[xi][t] = b_m[(t, xi)].X

        for t in range(T):
            cumulative_b_p = sum(b_p[(j, xi)].X for j in range(t + 1))
            cumulative_b_m = sum(b_m[(j, xi)].X for j in range(t + 1))
            B[xi][t] = B_0[xi].X - dt * ((1.0 / mu) * cumulative_b_p - eta * cumulative_b_m)

    ALPHA_p = np.array([alpha_p[r].X for r in range(T)])
    ALPHA_m = np.array([alpha_m[r].X for r in range(T)])
    BETA_p = np.array([beta_p[r].X for r in range(T)])
    BETA_m = np.array([beta_m[r].X for r in range(T)])
    GAMMA_p = np.array([gamma_p[t].X for t in range(T)])
    GAMMA_m = np.array([gamma_m[t].X for t in range(T)])

    return Y_p, Y_m, B_p, B_m, B, ALPHA_p, ALPHA_m, BETA_p, BETA_m, GAMMA_p, GAMMA_m

# -------------------------
# FB0: full matrix feedback model
# -------------------------
def FB0_model(u, solar, loads, alpha):
    """
    FB0: full matrix-form feedback, where each b_{t,xi} depends on coefficients alpha_p[(t,r)], etc.
    This function preserves the exact logic of your original FB0 implementation.
    """
    M = len(solar)
    P_b_max, B_max, P_max, mu, eta = _unpack_u(u)

    # containers
    y_p = {}
    y_m = {}
    b_p = {}
    b_m = {}
    B_0 = {}
    alpha_p = {}
    alpha_m = {}
    beta_p = {}
    beta_m = {}
    gamma_p = {}
    gamma_m = {}
    Y_bin = {}

    model = Model("FB0")
    model.ModelSense = GRB.MINIMIZE

    # coefficient variables alpha_p[(i,j)] etc.
    for i in range(T):
        for j in range(T):
            alpha_p[(i, j)] = model.addVar(lb=-P_b_max, ub=P_b_max, obj=0, name=f"alpha_p_{i}_{j}")
            alpha_m[(i, j)] = model.addVar(lb=-P_b_max, ub=P_b_max, obj=0, name=f"alpha_m_{i}_{j}")
            beta_p[(i, j)] = model.addVar(lb=-P_b_max, ub=P_b_max, obj=0, name=f"beta_p_{i}_{j}")
            beta_m[(i, j)] = model.addVar(lb=-P_b_max, ub=P_b_max, obj=0, name=f"beta_m_{i}_{j}")
        gamma_p[i] = model.addVar(lb=-P_b_max, ub=P_b_max, obj=0, name=f"gamma_p_{i}")
        gamma_m[i] = model.addVar(lb=-P_b_max, ub=P_b_max, obj=0, name=f"gamma_m_{i}")

    # scenario-dependent part
    for xi in range(M):
        B_0[xi] = model.addVar(lb=0, ub=B_max, obj=0, name=f"B0_{xi}")

        for t in range(T):
            y_p[(t, xi)] = model.addVar(lb=0, ub=P_max, obj=(1.0 / M) * alpha[xi][t] * dt, name=f"y_p_{t}_{xi}")
            y_m[(t, xi)] = model.addVar(lb=0, ub=P_max, obj=0, name=f"y_m_{t}_{xi}")
            b_p[(t, xi)] = model.addVar(lb=0, ub=mu * P_b_max, obj=beta * dt / (mu * M), name=f"b_p_{t}_{xi}")
            b_m[(t, xi)] = model.addVar(lb=0, ub=P_b_max, obj=0, name=f"b_m_{t}_{xi}")
            Y_bin[(t, xi)] = model.addVar(obj=0, vtype=GRB.BINARY, name=f"Ybin_{t}_{xi}")

            model.addConstr(y_p[(t, xi)] - y_m[(t, xi)] == loads[xi][t] - solar[xi][t] - b_p[(t, xi)] + b_m[(t, xi)], name=f"bal_{t}_{xi}")

            # compute linear combination over past r < t
            if t == 0:
                bbp = 0
                bbm = 0
            else:
                bbp = quicksum(alpha_p[(t, r)] * loads[xi][r] + beta_p[(t, r)] * solar[xi][r] for r in range(t))
                bbm = quicksum(alpha_m[(t, r)] * loads[xi][r] + beta_m[(t, r)] * solar[xi][r] for r in range(t))

            model.addConstr(b_p[(t, xi)] == bbp + gamma_p[t], name=f"b_p_def_{t}_{xi}")
            model.addConstr(b_m[(t, xi)] == bbm + gamma_m[t], name=f"b_m_def_{t}_{xi}")

            # prevent simultaneous charge/discharge
            model.addConstr(b_p[(t, xi)] <= P_b_max * Y_bin[(t, xi)], name=f"no_simul_p_{t}_{xi}")
            model.addConstr(b_m[(t, xi)] <= P_b_max * (1 - Y_bin[(t, xi)]), name=f"no_simul_m_{t}_{xi}")

            # battery bounds
            expr_b_p = quicksum((1.0 / mu) * b_p[(r, xi)] for r in range(t + 1))
            expr_b_m = quicksum(eta * b_m[(r, xi)] for r in range(t + 1))
            model.addConstr(B_0[xi] - dt * (expr_b_p - expr_b_m) <= B_max, name=f"cap_ub_{t}_{xi}")
            model.addConstr(B_0[xi] - dt * (expr_b_p - expr_b_m) >= 0, name=f"cap_lb_{t}_{xi}")

        model.addConstr(quicksum((1.0 / mu) * b_p[(t, xi)] - eta * b_m[(t, xi)] for t in range(T)) == 0, name=f"neutral_{xi}")
        model.addConstr(dt * ((1.0 / mu) * quicksum(b_p[(t, xi)] for t in range(T)) + eta * quicksum(b_m[(t, xi)] for t in range(T))) <= 2 * c_cycles * B_max, name=f"cycle_{xi}")

    model.optimize()

    # extract
    Y_p = [[0] * T for _ in range(M)]
    Y_m = [[0] * T for _ in range(M)]
    B_p = [[0] * T for _ in range(M)]
    B_m = [[0] * T for _ in range(M)]
    B = [[0] * T for _ in range(M)]

    for xi in range(M):
        for t in range(T):
            Y_p[xi][t] = y_p[(t, xi)].X
            Y_m[xi][t] = y_m[(t, xi)].X
            B_p[xi][t] = b_p[(t, xi)].X
            B_m[xi][t] = b_m[(t, xi)].X

        for t in range(T):
            cumulative_b_p = sum(b_p[(j, xi)].X for j in range(t + 1))
            cumulative_b_m = sum(b_m[(j, xi)].X for j in range(t + 1))
            B[xi][t] = B_0[xi].X - dt * ((1.0 / mu) * cumulative_b_p - eta * cumulative_b_m)

    # coefficient arrays of shape (T, T) for alpha/beta in FB0
    ALPHA_p = np.array([[alpha_p[(i, j)].X for j in range(T)] for i in range(T)])
    ALPHA_m = np.array([[alpha_m[(i, j)].X for j in range(T)] for i in range(T)])
    BETA_p = np.array([[beta_p[(i, j)].X for j in range(T)] for i in range(T)])
    BETA_m = np.array([[beta_m[(i, j)].X for j in range(T)] for i in range(T)])
    GAMMA_p = np.array([gamma_p[t].X for t in range(T)])
    GAMMA_m = np.array([gamma_m[t].X for t in range(T)])

    return Y_p, Y_m, B_p, B_m, B, ALPHA_p, ALPHA_m, BETA_p, BETA_m, GAMMA_p, GAMMA_m
