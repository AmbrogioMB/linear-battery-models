"""System parameters, objectives and evaluation of battery trajectories.

Notation (as in the paper): b+ discharge power, b- charge power [kW], E state
of charge [kWh], y+ import, y- export [kW]; DT = 0.25 h, T = 96 periods per day.

An objective ("tariff") turns the exogenous signal x_t (carbon intensity in
gCO2e/kWh, or day-ahead price in EUR/MWh) into a buy coefficient
buy_t = k_buy * x_t + c_buy and a sell coefficient sell_t = k_sell * x_t, and
adds a wear term per kWh delivered by the battery. The cost of a trajectory is

    DT * sum_t (buy_t y+_t - sell_t y-_t) + wear * DT * sum_t b+_t.
"""

from dataclasses import dataclass

import numpy as np

T = 96
DT = 0.25


@dataclass(frozen=True)
class Battery:
    capacity: float = 10.0  # kWh
    power: float = 5.0  # kW (charge; discharge is limited to mu * power)
    eta: float = float(np.sqrt(0.96))  # charging efficiency
    mu: float = float(np.sqrt(0.96))  # discharging efficiency
    cycles: float = 2.0  # equivalent full cycles per day

    @property
    def budget(self):
        """Bound on DT * sum_t (b+_t / mu + eta * b-_t) over one calendar day."""
        return 2 * self.cycles * self.capacity


@dataclass(frozen=True)
class Tariff:
    name: str
    signal: str  # key of the signal in the grid dataset
    k_buy: float = 1.0
    c_buy: float = 0.0
    k_sell: float = 0.0
    wear: float = 0.0
    unit: str = "gCO2e"

    def buy(self, x):
        return self.k_buy * np.asarray(x) + self.c_buy

    def sell(self, x):
        return self.k_sell * np.asarray(x)


# Life-cycle emissions of lithium-ion battery storage per kWh delivered [gCO2e/kWh]:
# median of the NREL life cycle assessment harmonization, NREL fact sheet
# NREL/FS-6A50-80580 (2021), Table 1 (total life cycle, lithium-ion battery).
BETA = 33.0

# Retail cost of an Italian resident domestic customer with a wholesale-indexed
# offer in 2024 [EUR/kWh]:
#   buy_t = (1 + VAT) * ((1 + LOSSES) * p_t / 1000 + VOLUMETRIC),
# with p_t the day-ahead price [EUR/MWh] (IT-North, as a proxy of the PUN),
# LOSSES the conventional loss factor of low-voltage withdrawals (10% from 2023,
# ARERA deliberazione 584/2023/R/eel, TIS Tabella 4) and VOLUMETRIC the
# per-kWh components of Q1 2024 for resident domestic customers, checked on
# the official ARERA tables on 2026-11-13:
#   dispatching PD 0.01609 (delibera 626/2023/R/eel, Tabella 2.1)
#   + network sigma3 0.01057 (630/2023/R/eel, TIT 2024-2027 Tabella 4)
#   + UC3 0.00156, UC6 0.00007, ASOS 0.025398, ARIM 0.006987 (633/2023/R/com,
#     Tabelle 7, 1 and 6, consolidated in the tables of 45/2024/R/com)
#   + excise 0.0227 (any use in dwellings, D.M. 30/12/2011; ADM table of the
#     excise rates updated to 1 January 2024). The exemption of the first
#     150 kWh/month for resident supplies up to 3 kW (D.Lgs. 504/1995,
#     art. 52, comma 3, lettera e) is neglected.
# From Q2 2024 the system charges of the typical customer rose from 3.24 to
# 3.86 cEUR/kWh (ARERA technical sheets of 28/12/2023 and 27/06/2024); this is
# neglected. Fixed (per-point and per-kW) charges do not depend on operation.
VAT, LOSSES = 0.10, 0.10
VOLUMETRIC = 0.01609 + (0.01057 + 0.00156 + 0.00007) + (0.025398 + 0.006987) + 0.0227
# Exports are paid at the hourly zonal day-ahead price (Ritiro Dedicato, ARERA
# deliberazione 280/07); the guaranteed minimum prices for small plants are ignored.
# Battery wear cost per kWh delivered [EUR/kWh]: an assumption, with
# sensitivity 0 and 0.10. For reference, installed home storage costs
# 400-1000 EUR/kWh of usable capacity with a 15-year life (Kost et al.,
# Fraunhofer ISE, Stromgestehungskosten Erneuerbare Energien, July 2024).
COST_WEAR = 0.05


def cost_tariff(name, wear):
    k = (1 + VAT) * (1 + LOSSES) * 1e-3
    return Tariff(name, "price", k_buy=k, c_buy=(1 + VAT) * VOLUMETRIC, k_sell=1e-3, wear=wear, unit="EUR")


TARIFFS = {
    # average (production-based) carbon intensity; exports are not credited
    "aef": Tariff("aef", "alpha_avg", wear=BETA),
    # marginal emission factor estimated by regression (see build_grid.py)
    "mef": Tariff("mef", "alpha_marg", wear=BETA),
    # as mef, but the lookahead policies plan with the table estimated on the previous year (see data.py)
    "mef_prev": Tariff("mef_prev", "alpha_marg", wear=BETA),
    # net-exchange accounting: exports are credited at the marginal factor (avoided grid emissions)
    "mef_net": Tariff("mef_net", "alpha_marg", k_sell=1.0, wear=BETA),
    # retail cost, exports paid at the zonal price; wear sensitivity variants
    "cost": cost_tariff("cost", COST_WEAR),
    "cost_w0": cost_tariff("cost_w0", 0.0),
    "cost_w10": cost_tariff("cost_w10", 0.10),
}


def evaluate(load, pv, x, bp, bm, tariff, bat, E0, E_end):
    """Realized objective and statistics of a trajectory (arrays of equal shape).

    A trajectory that ends with less energy than it started with is charged the
    cost of restoring the deficit from the grid at the average buy coefficient,
    (E0 - E_end)^+ * mean(buy) / eta; no credit is given for a surplus.
    """
    load, pv, x, bp, bm = (np.ravel(a) for a in (load, pv, x, bp, bm))
    y = load - pv - bp + bm
    yp, ym = np.maximum(y, 0), np.maximum(-y, 0)
    buy, sell = tariff.buy(x), tariff.sell(x)
    terminal = max(E0 - E_end, 0.0) * buy.mean() / bat.eta
    obj = DT * (buy @ yp - sell @ ym) + tariff.wear * DT * bp.sum() + terminal
    n_days = len(load) / T
    return dict(
        value=obj,
        terminal=terminal,
        import_kwh=DT * yp.sum(),
        export_kwh=DT * ym.sum(),
        load_kwh=DT * load.sum(),
        pv_kwh=DT * pv.sum(),
        self_sufficiency=1 - DT * yp.sum() / (DT * load.sum()),
        cycles_per_day=DT * (bp.sum() / bat.mu + bat.eta * bm.sum()) / (2 * bat.capacity) / n_days,
        grid_charge_kwh=DT * np.minimum(bm, np.maximum(load - pv + bm - bp, 0)).sum(),
    )
