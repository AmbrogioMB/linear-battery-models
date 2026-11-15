"""Build the grid signals at 15-minute resolution (UTC+1, 96 periods per day).

Output data/grid.npz with arrays (n_days, 96) and dates:
- alpha_avg: average production-based carbon intensity of the Italian
  generation mix [gCO2e/kWh] (operational emission factors per production
  type; imports and pumped-storage output are not included, as in ISPRA);
- alpha_marg: marginal emission factor [gCO2e/kWh], estimated by regressing
  the hourly change of fossil emissions on the hourly change of residual load
  (load minus wind and solar, as provided by Energy-Charts), i.e. the
  regression approach of Hawkes (2010) and Siler-Evans et al. (2012) with the
  variable renewables taken out of the explanatory variable, so that the
  solar ramps do not bias the estimate; separately for every year,
  quarter and hour of the day (pooling the two neighbouring hours); values
  are clipped to [0, 1000];
- price: day-ahead price of the IT-North bidding zone [EUR/MWh].
Hourly values are held constant over the four quarters of the hour.
A table of the marginal emission factors is written to data/mef_table.csv.

Usage: python code/build_grid.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "grid"
OUT = ROOT / "data"
FIRST, LAST = "2023-01-01", "2024-12-31"
TZ = "Etc/GMT-1"  # UTC+1 all year (no daylight saving time)

# Operational emission factors [gCO2/kWh of electricity] per production type.
# They are consistent with the ISPRA emission factors per unit of primary energy
# of 2023 (ISPRA, Rapporti 413/2025, Tabella 1.2: natural gas 59.18, steam coal
# 93.40, fuel oil 76.43 tCO2/TJ) with net efficiencies of about 53%, 37% and 42%.
# Check: the resulting annual factor of gross production is 255.7 g/kWh in 2023
# and 214.8 g/kWh in 2024, against 256.3 and 215.9 (preliminary) in ISPRA,
# Rapporti 413/2025, Tabella 1.3 (validate() prints this comparison).
EMISSION_FACTORS = {
    "fossil brown coal": 1000,
    "fossil coal-derived gas": 400,
    "fossil gas": 400,
    "fossil hard coal": 900,
    "fossil oil": 650,
    "others": 300,
    "waste": 300,
}
ISPRA_GROSS_PRODUCTION = {2023: 256.3, 2024: 215.9}
# not domestic generation; pumped-storage output is excluded as in ISPRA (net of pumping)
EXCLUDED = ("consumption", "cross border", "load", "residual", "share", "pumped storage")


def read(kind):
    frames = []
    for f in sorted(RAW.glob(f"{kind}_*.json")):
        d = json.loads(f.read_text())
        idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
        if kind == "power":
            cols = {p["name"]: [np.nan if v is None else v for v in p["data"]] for p in d["production_types"]}
        else:
            cols = {"price": [np.nan if v is None else v for v in d["price"]]}
        frames.append(pd.DataFrame(cols, index=idx))
    df = pd.concat(frames)
    df = df[~df.index.duplicated()].sort_index().astype(float)
    df = df[
        (df.index >= pd.Timestamp(FIRST, tz="UTC") - pd.Timedelta(hours=1))
        & (df.index < pd.Timestamp(LAST, tz="UTC") + pd.Timedelta(days=1))
    ]
    return df.loc[:, df.isna().mean() < 0.5]  # drop series that appear only sporadically


def to_days(s):
    """Hourly series (UTC, start-of-interval stamps) -> (n_days, 96) in UTC+1."""
    s = s.copy()
    s.index = s.index.tz_convert(TZ)
    q = s.resample("15min").ffill()
    end = q.index[-1] + pd.Timedelta(minutes=45)
    q = q.reindex(pd.date_range(q.index[0], end, freq="15min")).ffill()
    days = pd.date_range(FIRST, LAST, freq="D", tz=TZ)
    full = q.reindex(pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23, minutes=45), freq="15min"))
    full = full.bfill().ffill()  # only the first hour of 2023 is missing (UTC+1 shift)
    return full.values.reshape(len(days), 96), np.array([str(d.date()) for d in days])


def main():
    df = read("power")
    gen_cols = [c for c in df.columns if not any(k in c.lower() for k in EXCLUDED)]
    print("missing hours per series:", df[gen_cols].isna().sum()[lambda v: v > 0].to_dict())
    gen = df[gen_cols].clip(lower=0).interpolate(limit_direction="both")
    ef = np.array([next((v for k, v in EMISSION_FACTORS.items() if k in c.lower()), 0) for c in gen_cols])
    emis = gen.values @ ef  # [MW * g/kWh] = kg/h
    alpha = pd.Series(emis / gen.sum(axis=1).values, index=df.index)
    for y, ref in ISPRA_GROSS_PRODUCTION.items():
        sel = df.index.year == y
        print(f"{y}: annual factor {emis[sel].sum() / gen[sel].values.sum():.1f} g/kWh (ISPRA {ref})")

    # marginal emission factors
    load = df["Residual load"].interpolate(limit_direction="both")
    loc = df.index.tz_convert(TZ)
    reg = pd.DataFrame(
        {
            "dE": pd.Series(emis, index=df.index).diff(),
            "dL": load.diff(),
            "year": loc.year,
            "q": loc.quarter,
            "h": loc.hour,
        }
    ).dropna()
    rows = []
    for (y, q, h), _ in reg.groupby(["year", "q", "h"]):
        # pool the neighbouring hours (h-1, h, h+1) to reduce the estimation noise
        g = reg[(reg.year == y) & (reg.q == q) & (((reg.h - h) % 24 <= 1) | ((h - reg.h) % 24 <= 1))]
        X = np.c_[g["dL"], np.ones(len(g))]
        coef, *_ = np.linalg.lstsq(X, g["dE"], rcond=None)
        r2 = 1 - ((g["dE"] - X @ coef) ** 2).sum() / max(((g["dE"] - g["dE"].mean()) ** 2).sum(), 1e-9)
        rows.append(dict(year=y, quarter=q, hour=h, mef=coef[0], r2=r2, n=len(g)))
    tab = pd.DataFrame(rows)
    tab["mef_clipped"] = tab["mef"].clip(0, 1000)
    OUT.mkdir(exist_ok=True)
    tab.to_csv(OUT / "mef_table.csv", index=False, lineterminator="\n")
    key = pd.MultiIndex.from_arrays([loc.year, loc.quarter, loc.hour])
    mef = pd.Series(tab.set_index(["year", "quarter", "hour"])["mef_clipped"].reindex(key).values, index=df.index)

    price = read("price")["price"]
    a_avg, dates = to_days(alpha)
    a_marg, _ = to_days(mef)
    p, _ = to_days(price.reindex(df.index).interpolate(limit=3))
    np.savez(OUT / "grid.npz", dates=dates, alpha_avg=a_avg, alpha_marg=a_marg, price=p)

    for name, v in (("alpha_avg", a_avg), ("alpha_marg", a_marg), ("price", p)):
        print(f"{name:10s} mean {v.mean():7.1f}  min {v.min():7.1f}  max {v.max():7.1f}  nan {np.isnan(v).sum()}")
    print(f"MEF: mean {tab.mef.mean():.0f}, median R2 {tab.r2.median():.2f}, negative bins {(tab.mef < 0).sum()}")
    print(tab.pivot_table(index="hour", columns=["year", "quarter"], values="mef").round(0).to_string())


if __name__ == "__main__":
    main()
