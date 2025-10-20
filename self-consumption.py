def self_cons(load, solar, b_max, P_max, P_b_max, e0=0, mu=0.9, eta=0.9, c=2):
    """
    Algorithm to compute the solution by optimizing self-consumption
    At a given time t, if the solar exceeds the load, what's left is stored in the battery, if there is enough
    space in the battery and if doing so does not violate the cycle constraint. Else, the exceeding energy is sold to
    the aggregator.
    If the load exceeds the solar, the load is first satisfied by using the battery, if doing so does not violate the
    cycle constraint. The remaining load is satisfied by buying energy from the aggregator.
    The algorithm works for just one user, it should be applied to all users (can be done in parallel)
    load: list of length 96 with every load
    solar: list of length 96 with solar energy production
    b_max: maximum capacity of the battery
    e0: initial charge of the battery
    mu, eta: constants
    c: maximum number of cycles
    """
    T = range(len(load))
    e = [e0]
    b_pl = []
    b_mi = []
    x = []
    dT = 0.25
    for t in T:
        if solar[t] - load[t] >=0 :
            b_pl_t = 0
            b_pl.append(b_pl_t)
            in_battery = min(solar[t] - load[t], (b_max - e[t])/(dT * eta), P_b_max, (2 * c * b_max - dT * ((1 / mu) * sum(b_pl)) - dT * eta * sum(b_mi) ) / (dT * eta))
            to_sell = solar[t] - load[t] - in_battery
            if to_sell <= P_max:
                b_mi_t = in_battery
                b_mi.append(b_mi_t)
            else:
                # infeasible
                return(False)            
            
        else:
            b_mi_t = 0
            b_mi.append(b_mi_t)
            from_battery = min(load[t] - solar[t], mu * e[t] /dT, mu * P_b_max, (2 * c * b_max - dT * ((1 / mu) * sum(b_pl)) - dT * eta * sum(b_mi) ) / (dT / mu))
            to_buy = load[t] - solar[t] - from_battery
            if to_buy <= P_max:
                b_pl_t = from_battery
                b_pl.append(b_pl_t)
            else:
                # infeasible
                return(False)

        e_t = e[t] + dT * eta * b_mi_t - dT * (1 / mu) * b_pl_t
        e.append(e_t)
        x_t = load[t] - solar[t] - b_pl_t + b_mi_t
        x.append(x_t)
    return x, b_pl, b_mi, e

 