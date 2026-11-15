"""Load the processed datasets and assemble a case.

A case is (home, site, PV size, battery, objective). All arrays are daily
(n_days, 96), in kW (load, PV) or in the unit of the objective signal.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .system import TARIFFS, Battery

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# simulation period: complete day-ahead forecasts are available from 2024-01-20, and
# the scenario model uses the 28 preceding days; the last day needs the next one
SIM_FIRST, SIM_LAST = "2024-02-17", "2024-12-30"
WINDOW_DAYS = 28


@lru_cache(maxsize=None)
def _npz(name):
    with np.load(DATA / name, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@dataclass
class Case:
    home: str
    site: str
    kwp: float
    battery: Battery
    objective: str
    dates: np.ndarray
    load: np.ndarray
    pv: np.ndarray
    pv_fc: np.ndarray
    x: np.ndarray
    max_days: int | None = None  # simulate only the first days (tests)
    x_plan: np.ndarray | None = None  # signal used for planning when it differs from the realized one

    @property
    def tariff(self):
        return TARIFFS[self.objective]

    @property
    def sim_days(self):
        idx = np.where((self.dates >= SIM_FIRST) & (self.dates <= SIM_LAST))[0]
        return idx[: self.max_days]

    @property
    def label(self):
        b = self.battery
        return f"{self.home}_{self.site}_pv{self.kwp:g}_b{b.capacity:g}-{b.power:g}_{self.objective}"


def previous_year_signal(dates, x):
    """Daily profiles of the same quarter of the previous year (NaN for the first year).

    The marginal factors are a table by year, quarter and hour, so all the days
    of a quarter share one profile: a policy that plans with the table of the
    previous year uses only information available before the evaluation year.
    """
    year = np.array([int(d[:4]) for d in dates])
    quarter = np.array([(int(d[5:7]) - 1) // 3 for d in dates])
    out = np.full(x.shape, np.nan)
    for y in np.unique(year)[1:]:
        for q in range(4):
            src, dst = (year == y - 1) & (quarter == q), (year == y) & (quarter == q)
            profile = np.median(x[src], axis=0)
            # tolerate isolated days with missing source data (1 January of the first year)
            if (~np.isclose(x[src], profile).all(axis=1)).sum() > 1:
                raise ValueError("the signal is not a table by year and quarter")
            out[dst] = profile
    return out


def homes():
    return [str(h) for h in _npz("loads.npz")["homes"]]


def load_case(home, site, kwp, capacity, power, objective, load_scale=1.0):
    g, pv, ld = _npz("grid.npz"), _npz("pv.npz"), _npz("loads.npz")
    assert (g["dates"] == pv["dates"]).all() and (g["dates"] == ld["dates"]).all()
    s = list(pv["sites"]).index(site)
    h = list(ld["homes"]).index(home)
    if objective.startswith("cost") and site != "milan":
        raise ValueError("the cost objective uses IT-North prices and is only defined for milan")
    x_plan = previous_year_signal(g["dates"], g["alpha_marg"]) if objective == "mef_prev" else None
    return Case(
        x_plan=x_plan,
        home=home,
        site=site,
        kwp=kwp,
        battery=Battery(capacity=capacity, power=power),
        objective=objective,
        dates=g["dates"],
        load=load_scale * ld["load"][h].astype(float),
        pv=kwp * pv["actual"][s],
        pv_fc=kwp * pv["forecast"][s],
        x=g[TARIFFS[objective].signal],
    )
