"""Download irradiance and temperature for the PV sites from Open-Meteo.

For each site (sites.csv) and a south-facing plane tilted by TILT degrees:
- actual: satellite-derived global tilted irradiance (EUMETSAT CM SAF SARAH3,
  Open-Meteo satellite radiation API), hourly, 2023-2024;
- temperature: ERA5 2 m air temperature (Open-Meteo historical weather API);
- forecast: day-ahead forecast of tilted irradiance and temperature
  (DWD ICON, Open-Meteo previous runs API, variable *_previous_day1, i.e. the
  value forecast by the model run of the previous day). Available from
  2024-01-19;
- PVGIS 5.3 hourly PV output for 2023 (same plane, 14% losses), only to
  validate the PV model of build_pv.py.

Open-Meteo data are distributed under CC BY 4.0. Hourly values of radiation
are averages over the preceding hour (UTC timestamps).
Raw JSON responses are stored in data_raw/weather/.

Usage: python code/fetch_weather.py
"""

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_raw" / "weather"
TILT, AZIMUTH = 30, 0  # Open-Meteo convention: azimuth 0 = south
START, END = "2023-01-01", "2024-12-31"
FC_START = "2024-01-19"
PVGIS = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"

REQUESTS = {
    "actual": (
        "https://satellite-api.open-meteo.com/v1/archive",
        dict(hourly="global_tilted_irradiance,shortwave_radiation", start_date=START, end_date=END),
    ),
    "temp": (
        "https://archive-api.open-meteo.com/v1/archive",
        dict(hourly="temperature_2m", start_date=START, end_date=END, models="era5"),
    ),
    "forecast": (
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        dict(
            hourly="global_tilted_irradiance_previous_day1,temperature_2m_previous_day1",
            start_date=FC_START,
            end_date=END,
            models="icon_seamless",
        ),
    ),
}


def sites():
    with open(ROOT / "code" / "sites.csv") as f:
        return list(csv.DictReader(f))


def fetch(url, retries=6):
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return json.load(r)
        except Exception as e:
            print(f"  retry {k + 1}: {e}", flush=True)
            time.sleep(15 * (k + 1))
    raise RuntimeError(f"failed: {url}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in sites():
        for kind, (base, params) in REQUESTS.items():
            path = OUT / f"{s['site']}_{kind}.json"
            if path.exists():
                continue
            q = dict(latitude=s["lat"], longitude=s["lon"], tilt=TILT, azimuth=AZIMUTH, timezone="GMT", **params)
            print(f"{s['site']} {kind}", flush=True)
            d = fetch(base + "?" + urllib.parse.urlencode(q))
            if "hourly" not in d:
                raise RuntimeError(f"{s['site']} {kind}: {d}")
            path.write_text(json.dumps(d))
            time.sleep(2)
        # PVGIS 5.3 hourly PV output for 2023 (SARAH3), used only to validate the PV model
        path = OUT / f"{s['site']}_pvgis2023.json"
        if not path.exists():
            q = dict(
                lat=s["lat"],
                lon=s["lon"],
                startyear=2023,
                endyear=2023,
                pvcalculation=1,
                peakpower=1,
                loss=14,
                angle=TILT,
                aspect=AZIMUTH,
                outputformat="json",
            )
            print(f"{s['site']} pvgis", flush=True)
            path.write_text(json.dumps(fetch(PVGIS + "?" + urllib.parse.urlencode(q))))


if __name__ == "__main__":
    main()
