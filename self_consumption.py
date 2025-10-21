def self_cons(load, solar, b_max, P_max, P_b_max, e0=0, mu=0.9, eta=0.9, c=2):
    """
    Compute the self-consumption-based control policy for a PV-battery system.

    The algorithm operates greedily at each time step:
      - When PV generation exceeds demand, the surplus is stored in the battery 
        (if capacity and cycle constraints allow), otherwise sold to the aggregator.
      - When demand exceeds PV generation, the battery is discharged first 
        (if energy is available and within limits), otherwise the deficit is covered by imports.

    This corresponds to the Automatic Battery (AB) model, used as a 
    rule-based benchmark in the paper.

    Parameters
    ----------
    load : list or array (length = 96)
        Electrical demand at each 15-minute interval [kW].
    solar : list or array (length = 96)
        PV generation at each 15-minute interval [kW].
    b_max : float
        Maximum battery capacity [kWh].
    P_max : float
        Maximum power that can be exchanged with the aggregator (import/export) [kW].
    P_b_max : float
        Maximum charge/discharge power of the battery [kW].
    e0 : float, optional
        Initial battery state of charge [kWh]. Default is 0.
    mu : float, optional
        Discharge efficiency of the battery (0 < mu ≤ 1). Default is 0.9.
    eta : float, optional
        Charge efficiency of the battery (0 < eta ≤ 1). Default is 0.9.
    c : float, optional
        Maximum number of equivalent charge/discharge cycles allowed. Default is 2.

    Returns
    -------
    x : list
        Net power exchange with the grid (positive = import, negative = export) [kW].
    b_pl : list
        Discharging power of the battery [kW].
    b_mi : list
        Charging power of the battery [kW].
    e : list
        Battery state of charge trajectory [kWh].

    Notes
    -----
    - The time step is fixed at 15 minutes (Δt = 0.25 h).
    - If any operational limit (power, capacity, or cycle) is violated, 
      the algorithm returns False to indicate infeasibility.
    - This implementation models a single user; in multi-user systems, it can be
      applied independently or in parallel across users.
    """
    T = range(len(load))
    e = [e0]         # Battery energy (state of charge)
    b_pl = []         # Discharge power (battery to load/grid)
    b_mi = []         # Charge power (PV to battery)
    x = []            # Net exchange with grid (positive = import)
    dT = 0.25         # Time step [h] = 15 minutes

    for t in T:
        if solar[t] - load[t] >= 0:
            # PV generation exceeds load → attempt to charge the battery
            b_pl_t = 0
            b_pl.append(b_pl_t)

            # Energy that can be stored without violating capacity or cycle limits
            in_battery = min(
                solar[t] - load[t],
                (b_max - e[t]) / (dT * eta),  # Available capacity
                P_b_max,                      # Power limit
                (2 * c * b_max - dT * ((1 / mu) * sum(b_pl)) - dT * eta * sum(b_mi)) / (dT * eta)
            )

            to_sell = solar[t] - load[t] - in_battery
            if to_sell <= P_max:
                # Feasible: store as much as possible and sell the rest
                b_mi_t = in_battery
                b_mi.append(b_mi_t)
            else:
                # Violation of export limit → infeasible case
                return False

        else:
            # Load exceeds PV generation → attempt to discharge the battery
            b_mi_t = 0
            b_mi.append(b_mi_t)

            # Energy that can be discharged without violating energy or cycle limits
            from_battery = min(
                load[t] - solar[t],
                mu * e[t] / dT,              # Available stored energy
                mu * P_b_max,                # Power limit
                (2 * c * b_max - dT * ((1 / mu) * sum(b_pl)) - dT * eta * sum(b_mi)) / (dT / mu)
            )

            to_buy = load[t] - solar[t] - from_battery
            if to_buy <= P_max:
                # Feasible: discharge as much as possible and buy the rest
                b_pl_t = from_battery
                b_pl.append(b_pl_t)
            else:
                # Violation of import limit → infeasible case
                return False

        # Update battery energy and net exchange
        e_t = e[t] + dT * eta * b_mi_t - dT * (1 / mu) * b_pl_t
        e.append(e_t)
        x_t = load[t] - solar[t] - b_pl_t + b_mi_t
        x.append(x_t)

    return x, b_pl, b_mi, e
