"""Build PV generation per kWp at 15-minute resolution for each site.

PV model (per kWp, south-facing plane tilted by 30 degrees):
    P = G/1000 * (1 + GAMMA * (Tc - 25)) * (1 - LOSS),   Tc = Ta + G * (NOCT - 20) / 800,
with G the global tilted irradiance [W/m2] and Ta the air temperature.
- actual PV: satellite irradiance (SARAH3) and ERA5 temperature;
- day-ahead forecast: ICON irradiance and temperature forecast by the run of
  the previous day (available from 2024-01-19, NaN before).
Hourly averages (preceding-hour convention) are placed at the middle of the
hour and linearly interpolated to 15 minutes, then shifted to UTC+1.
The model is validated against PVGIS 5.3 for 2023 (annual yield, hourly
correlation), reported on screen and in data/pv_validation.csv.

Output: data/pv.npz with dates, sites, actual and forecast arrays (n_sites, n_days, 96).

Usage: python code/build_pv.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "weather"
OUT = ROOT / "data"
FIRST, LAST = "2023-01-01", "2024-12-31"
TZ = "Etc/GMT-1"

# Cell temperature from the NOCT model and linear power temperature
# coefficient of crystalline silicon (Skoplaki and Palyvos, Solar Energy 83
# (2009) 614-624, doi:10.1016/j.solener.2008.10.008); NOCT = 45 C: the median
# NOCT of the 21,335 crystalline-silicon modules of the California Energy
# Commission module list is 45.4 C (interquartile range 43.8-46.3 C; NREL SAM
# library "CEC Modules.csv", version 2026.7.3); LOSS = 14% is the default system loss of PVGIS ("We have
# given a default value of 14% for the overall losses", PVGIS user manual, JRC).
GAMMA = -0.004  # 1/K
NOCT = 45.0  # degC
LOSS = 0.14


def pv_model(G, Ta):
    G = np.clip(np.nan_to_num(G, nan=0.0), 0, None)
    Tc = Ta + G * (NOCT - 20) / 800
    return np.clip(G / 1000 * (1 + GAMMA * (Tc - 25)) * (1 - LOSS), 0, 1)


def series(path, var):
    d = json.loads(path.read_text())["hourly"]
    idx = pd.to_datetime(d["time"]).tz_localize("UTC")
    return pd.Series([np.nan if v is None else v for v in d[var]], index=idx, dtype=float)


def to_days(hourly):
    """Hourly averages over the preceding hour (UTC) -> (n_days, 96) in UTC+1."""
    s = hourly.copy()
    s.index = s.index - pd.Timedelta(minutes=30)  # middle of the averaging interval
    days = pd.date_range(FIRST, LAST, freq="D", tz=TZ)
    grid = pd.date_range(days[0] + pd.Timedelta(minutes=7.5), periods=len(days) * 96, freq="15min")
    full = s.reindex(s.index.union(grid.tz_convert("UTC"))).interpolate(method="time", limit=4)
    v = full.reindex(grid.tz_convert("UTC")).values
    return v.reshape(len(days), 96), np.array([str(d.date()) for d in days])


def main():
    with open(ROOT / "code" / "sites.csv") as f:
        sites = list(csv.DictReader(f))
    act, fc, val = [], [], []
    for s in sites:
        n = s["site"]
        G = series(RAW / f"{n}_actual.json", "global_tilted_irradiance")
        Ta = series(RAW / f"{n}_temp.json", "temperature_2m").reindex(G.index).interpolate(limit=3)
        p_act = pd.Series(pv_model(G.values, Ta.values), index=G.index)
        Gf = series(RAW / f"{n}_forecast.json", "global_tilted_irradiance_previous_day1")
        Tf = series(RAW / f"{n}_forecast.json", "temperature_2m_previous_day1")
        p_fc = pd.Series(np.where(Gf.isna(), np.nan, pv_model(Gf.values, Tf.fillna(15).values)), index=Gf.index)
        a, dates = to_days(p_act)
        f_, _ = to_days(p_fc)
        a = np.nan_to_num(np.clip(a, 0, 1))
        f_ = np.clip(f_, 0, 1)
        act.append(a)
        fc.append(f_)

        # validation against PVGIS 2023 (hourly, kW per kWp, stamps at HH:10 UTC)
        pg = json.loads((RAW / f"{n}_pvgis2023.json").read_text())["outputs"]["hourly"]
        pidx = pd.to_datetime([h["time"] for h in pg], format="%Y%m%d:%H%M").floor("h").tz_localize("UTC")
        pvgis = pd.Series([h["P"] / 1000 for h in pg], index=pidx)
        ours = p_act.copy()
        ours.index = ours.index - pd.Timedelta(hours=1)  # preceding-hour average -> start of hour
        j = pd.concat([pvgis, ours], axis=1, keys=["pvgis", "ours"], sort=True).dropna()
        j = j[j.index.year == 2023]
        m = dates >= "2024-01-19"
        err = f_[m] - a[m]
        idx = np.where(m)[0]
        pers = a[idx - 1] - a[idx]  # persistence: yesterday's profile as forecast
        val.append(
            dict(
                site=n,
                yield_ours_2023=j["ours"].sum(),
                yield_pvgis_2023=j["pvgis"].sum(),
                hourly_corr_2023=j.corr().iloc[0, 1],
                yield_2024=a[dates >= "2024-01-01"].sum() / 4,
                fc_bias_kwh_per_day=np.nanmean(np.nansum(err, 1)) / 4,
                fc_mae_daily_kwh=np.nanmean(np.abs(np.nansum(err, 1))) / 4,
                fc_rmse_15min=np.sqrt(np.nanmean(err**2)),
                fc_skill_vs_persistence=1 - np.sqrt(np.nanmean(err**2)) / np.sqrt(np.mean(pers**2)),
            )
        )
    OUT.mkdir(exist_ok=True)
    np.savez(
        OUT / "pv.npz",
        dates=dates,
        sites=np.array([s["site"] for s in sites]),
        actual=np.array(act),
        forecast=np.array(fc),
    )
    v = pd.DataFrame(val)
    v.to_csv(OUT / "pv_validation.csv", index=False, lineterminator="\n")
    print(v.round(3).to_string())


if __name__ == "__main__":
    main()
