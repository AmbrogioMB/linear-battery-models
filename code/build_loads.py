"""Select the households and build their load profiles (15 min, UTC+1).

Source: Swiss residential smart-meter dataset (Zenodo 10.5281/zenodo.21398768,
CC BY 4.0), energy withdrawn per 15-minute interval [kWh], 2023-2024.

Selection (households without electrified heating, whose consumption is the
household load):
- installation type: single-family house (main sample) or apartment
  (extended sample, to span household sizes);
- no PV (no injection), no heat pump of any kind, no electric water heater,
  no storage or direct electric heating, no EV charging station, no recorded
  change of the device configuration;
- participants in the tariff pilot are kept and flagged (column `pilot`):
  without controllable devices, automated load control does not apply;
- at least MIN_COVERAGE of the 15-minute values of 2023-2024 available;
- average winter (Dec-Feb) consumption at most MAX_WINTER_SUMMER times the
  summer (Jun-Aug) one, to exclude unrecorded electric heating;
- consumption in 2024 of at least MIN_ANNUAL_KWH, to exclude unoccupied
  dwellings (added after E1: it removes three apartments with 2, 25 and 51 kWh,
  the next lowest consumes 223 kWh; the E2 cases, chosen before, are within
  0.33 percentile points of their target quantiles without these three).
Gaps of up to one hour are filled by linear interpolation, longer gaps with
the value of the same period one week earlier (or later).

Output: data/loads.npz (homes, types, dates, load [kW] with shape (n_homes, n_days, 96))
and data/homes.csv (selection statistics).

Usage: python code/build_loads.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "zurich" / "extracted"
OUT = ROOT / "data"
FIRST, LAST = "2023-01-01", "2024-12-31"
TZ = "Etc/GMT-1"
MIN_COVERAGE = 0.98
# homes using much more electricity in winter than in summer probably heat
# electrically even if not flagged (the device records may be incomplete)
MAX_WINTER_SUMMER = 1.6
MIN_ANNUAL_KWH = 100.0
EXCLUDE_FLAGS = ["1_ewh", "1_hp", "1_hp-add", "1_hp-wh", "1_ev", "1_storage_heating", "1_direct_heating", "1_change"]


def find(name):
    hits = sorted(RAW.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {RAW}")
    return hits[0]


def truthy(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def main():
    meta = pd.read_csv(find("metadata.csv"), sep=";", encoding="utf-8-sig")
    meter_dir = find("smart_meter_data")
    print("installation types:", meta["0_installation_type"].value_counts().to_dict())
    htype = meta["0_installation_type"].map({"Single-family house": "house", "Apartment": "apartment"})
    flagged = meta[EXCLUDE_FLAGS].astype(str).eq("True").any(axis=1)
    pilot = meta["2_energy_tariff"].notna() | meta["2_grid_tariff"].notna()
    no_pv = meta["0_max_net_injection"].isna() | (meta["0_max_net_injection"] <= 0)
    cand = meta[htype.notna() & ~flagged & no_pv].assign(type=htype, pilot=pilot)
    print(f"houses and apartments without flagged loads and PV: {cand.type.value_counts().to_dict()}")

    days = pd.date_range(FIRST, LAST, freq="D", tz=TZ)
    grid = pd.date_range(days[0], periods=len(days) * 96, freq="15min").tz_convert("UTC")
    homes, loads, stats = [], [], []
    for _, r in cand.iterrows():
        mid = str(r["0_meter_id"])
        df = pd.read_csv(meter_dir / f"{mid}.csv", sep=";")
        idx = pd.to_datetime(df["timestamp_utc"], utc=True)
        if df["kWh_to_grid"].fillna(0).sum() > 0:
            continue
        s = pd.Series(df["kWh_to_installation"].values * 4.0, index=idx)  # kWh per 15 min -> kW
        s = s[~s.index.duplicated()].reindex(grid)
        cov = s.notna().mean()
        if cov < MIN_COVERAGE:
            stats.append(dict(home=mid, type=r["type"], pilot=r["pilot"], coverage=cov, selected=False))
            continue
        s = s.interpolate(limit=4, limit_area="inside")
        for lag in (96 * 7, -96 * 7, 96 * 14, -96 * 14):
            s = s.fillna(s.shift(lag))
        s = s.ffill().bfill()
        mon = s.index.month
        ratio = s[mon.isin([12, 1, 2])].mean() / s[mon.isin([6, 7, 8])].mean()
        if ratio > MAX_WINTER_SUMMER:
            stats.append(
                dict(home=mid, type=r["type"], pilot=r["pilot"], coverage=cov, winter_summer=ratio, selected=False)
            )
            continue
        annual = s[grid.year == 2024].sum() / 4
        if annual < MIN_ANNUAL_KWH:
            stats.append(
                dict(
                    home=mid,
                    type=r["type"],
                    pilot=r["pilot"],
                    coverage=cov,
                    winter_summer=ratio,
                    selected=False,
                    annual_kwh_2024=annual,
                )
            )
            continue
        homes.append(mid)
        loads.append(s.values.reshape(len(days), 96))
        stats.append(
            dict(
                home=mid,
                type=r["type"],
                pilot=r["pilot"],
                coverage=cov,
                winter_summer=ratio,
                selected=True,
                annual_kwh_2024=annual,
                peak_kw=s.max(),
            )
        )
    L = np.array(loads, dtype=np.float32)
    OUT.mkdir(exist_ok=True)
    st = pd.DataFrame(stats)
    types = st.set_index("home").loc[homes, "type"].values
    np.savez_compressed(
        OUT / "loads.npz",
        homes=np.array(homes),
        types=np.array(types, dtype=str),
        dates=np.array([str(d.date()) for d in days]),
        load=L,
    )
    st.to_csv(OUT / "homes.csv", index=False, lineterminator="\n")
    for t, sel in st[st.selected].groupby("type"):
        a = sel.annual_kwh_2024
        print(
            f"{t}: {len(sel)} selected ({sel.pilot.sum()} in the tariff pilot); annual consumption 2024 [kWh]: "
            f"median {a.median():.0f}, 10-90% {a.quantile(0.1):.0f}-{a.quantile(0.9):.0f}"
        )


if __name__ == "__main__":
    main()
