"""Download Italian grid data for 2023-2024 from the Energy-Charts API.

- public_power (country=it): load and generation per production type
  (Energy-Charts redistributes ENTSO-E Transparency Platform data);
- price (bzn=IT-North): day-ahead price of the Northern Italy bidding zone
  (licence reported by the API: CC BY 4.0, Bundesnetzagentur | SMARD.de).
  Prices of the other Italian zones are served for "private and internal use
  only" and are therefore not used.

Raw JSON responses are stored month by month in data_raw/grid/.

Usage: python code/fetch_grid.py [first] [last]   (default 2023-01-01 2025-01-01)
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_raw" / "grid"
API = "https://api.energy-charts.info"
QUERIES = {
    "power": API + "/public_power?country=it&start={start}&end={end}",
    "price": API + "/price?bzn=IT-North&start={start}&end={end}",
}


def fetch(url, retries=6):
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # rate limiting or transient errors
            print(f"  retry {k + 1}: {e}", flush=True)
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"failed: {url}")


def main(first="2023-01-01", last="2025-01-01"):
    OUT.mkdir(parents=True, exist_ok=True)
    for m in pd.date_range(first, last, freq="MS", inclusive="left"):
        end = m + pd.offsets.MonthBegin(1)
        for kind, url in QUERIES.items():
            path = OUT / f"{kind}_{m:%Y_%m}.json"
            if path.exists():
                continue
            print(f"{kind} {m:%Y-%m}", flush=True)
            d = fetch(url.format(start=f"{m:%Y-%m-%d}T00:00Z", end=f"{end:%Y-%m-%d}T00:00Z"))
            path.write_text(json.dumps(d))
            time.sleep(2)


if __name__ == "__main__":
    main(*sys.argv[1:])
