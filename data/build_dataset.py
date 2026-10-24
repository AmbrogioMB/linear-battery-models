"""
build_dataset.py

Build the household-level dataset used in all experiments from Italian data
of the Energy-Charts platform (Fraunhofer ISE), which redistributes data of the
ENTSO-E Transparency Platform. Missing monthly raw files are downloaded into
raw/ first.

For every day we obtain three 96-dimensional profiles (15-minute resolution):
  - l : household load [kW], obtained by scaling the national load;
  - s : PV generation [kW], obtained by scaling the national PV output;
  - a : grid carbon intensity [gCO2e/kWh], computed from the generation mix.

Time is expressed in Central European Time (UTC+1) throughout the year, so
that every day has exactly 96 intervals (no daylight-saving discontinuities).

Output: dataset.npz with arrays dates (n_days,), load, solar, alpha (n_days, 96).
Usage: python build_dataset.py
"""
import glob
import json
import os
import time
import urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
URL = "https://api.energy-charts.info/public_power?country=it&start={}&end={}"
FIRST_MONTH, LAST_DOWNLOAD = "2025-10-01", "2026-10-21"  # raw data needed by this script
OUT = os.path.join(HERE, "dataset.npz")

T = 96
FIRST_DAY = "2026-01-01"
LAST_DAY = "2026-10-19"

# Household scaling (same convention as in the original study)
P_PV = 6.4                 # PV nominal power [kWp]
DOMESTIC_SHARE = 0.22      # share of national load due to households
N_HOUSEHOLDS = 25_690_057  # number of domestic consumers
DELTA = 2.65               # ratio between the considered user and the average household
SPECIFIC_YIELD = 1350.0    # target annual PV yield [kWh/kWp] (typical value for Italy)

# Operational emission factors [gCO2/kWh] per production type
EMISSION_FACTORS = {
    "fossil brown coal": 1000,
    "fossil coal-derived gas": 400,
    "fossil gas": 400,
    "fossil hard coal": 900,
    "fossil oil": 650,
    "fossil peat": 380,
    "others": 300,
    "waste": 300,
}
# Series that are not domestic generation
EXCLUDED = ("consumption", "cross border", "load", "residual", "share")


def download():
    """Download the monthly raw files that are not yet in raw/."""
    os.makedirs(RAW, exist_ok=True)
    for m in pd.date_range(FIRST_MONTH, LAST_DOWNLOAD, freq="MS"):
        path = os.path.join(RAW, f"it_{m:%Y_%m}.json")
        if os.path.isfile(path):
            continue
        end = min(m + pd.offsets.MonthBegin(1), pd.Timestamp(LAST_DOWNLOAD))
        url = URL.format(f"{m:%Y-%m-%d}T00:00Z", f"{end:%Y-%m-%d}T00:00Z")
        for k in range(6):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    data = r.read().decode()
                break
            except Exception as e:  # rate limiting or transient errors
                print(f"  {m:%Y-%m}: retry {k + 1} ({e})")
                time.sleep(10 * (k + 1))
        else:
            raise RuntimeError(f"download failed: {url}")
        with open(path, "w") as f:
            f.write(data)
        print(f"downloaded {m:%Y-%m}")
        time.sleep(3)


def read_raw():
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, "it_*.json"))):
        d = json.load(open(f))
        idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
        cols = {p["name"]: [np.nan if x is None else x for x in p["data"]] for p in d["production_types"]}
        frames.append(pd.DataFrame(cols, index=idx))
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def carbon_intensity(df):
    gen_cols = [c for c in df.columns if not any(k in c.lower() for k in EXCLUDED)]
    gen = df[gen_cols].clip(lower=0).fillna(0)
    ef = np.array([next((v for k, v in EMISSION_FACTORS.items() if k in c.lower()), 0) for c in gen_cols])
    return (gen.values @ ef) / gen.sum(axis=1).values


def to_days(series):
    s = series.copy()
    s.index = s.index.tz_convert("Etc/GMT-1")  # CET without DST (UTC+1)
    s = s.resample("15min").mean().interpolate(limit=8)
    df = s.to_frame("v")
    df["date"] = df.index.date
    df["slot"] = df.index.hour * 4 + df.index.minute // 15
    piv = df.pivot(index="date", columns="slot", values="v")
    return piv


def main():
    download()
    df = read_raw()
    df = df[df.index >= pd.Timestamp("2025-12-31T23:00Z")]
    raw_solar = df["Solar"].astype(float)
    raw_load = df["Load"].astype(float)
    alpha = pd.Series(carbon_intensity(df), index=df.index)

    piv_s, piv_l, piv_a = to_days(raw_solar), to_days(raw_load), to_days(alpha)
    days = pd.date_range(FIRST_DAY, LAST_DAY, freq="D").date
    piv_s, piv_l, piv_a = piv_s.reindex(days), piv_l.reindex(days), piv_a.reindex(days)
    ok = ~(piv_s.isna().any(axis=1) | piv_l.isna().any(axis=1) | piv_a.isna().any(axis=1))
    print(f"complete days: {ok.sum()} / {len(days)}; missing: {list(np.array(days)[~ok.values])}")
    # fill isolated incomplete days by interpolation along the day axis
    piv_s, piv_l, piv_a = (p.interpolate(axis=0, limit_direction="both") for p in (piv_s, piv_l, piv_a))

    # PV: national output normalized by an effective capacity giving SPECIFIC_YIELD over 12 months
    df12 = read_raw()
    df12 = df12[(df12.index >= pd.Timestamp("2025-10-20T00:00Z")) & (df12.index < pd.Timestamp("2026-10-20T00:00Z"))]
    s12 = df12["Solar"].astype(float).resample("1h").mean()
    c_eff = s12.sum() / SPECIFIC_YIELD  # [MW] such that sum(S/c_eff) [h] = yield
    solar = P_PV * piv_s.values / c_eff
    solar = np.clip(solar, 0, P_PV)

    load = DELTA * DOMESTIC_SHARE * 1000 * piv_l.values / N_HOUSEHOLDS
    a = piv_a.values

    np.savez(OUT, dates=np.array([str(d) for d in days]), load=load, solar=solar, alpha=a, c_eff=c_eff)
    print(f"effective PV capacity: {c_eff:.0f} MW; PV peak {solar.max():.2f} kW; "
          f"daily PV {solar.sum(1).mean() * 0.25:.1f} kWh; daily load {load.sum(1).mean() * 0.25:.1f} kWh; "
          f"alpha mean {a.mean():.0f} [{a.min():.0f}, {a.max():.0f}] g/kWh")
    m = pd.Series(range(len(days)), index=pd.to_datetime(days)).groupby(lambda x: x.month)
    for mo, ix in m.groups.items():
        ii = [days.tolist().index(d.date()) for d in ix]
        print(f"  month {mo:2d}: PV {solar[ii].sum(1).mean() / 4:5.1f} kWh/d, load {load[ii].sum(1).mean() / 4:5.1f} kWh/d, "
              f"alpha {a[ii].mean():5.0f}")


if __name__ == "__main__":
    main()
