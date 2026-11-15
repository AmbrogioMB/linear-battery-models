"""Download the Swiss residential smart-meter dataset (15-min, 2023-2024).

Source: "Dataset on residential electricity load profiles in Switzerland",
Zenodo record 21398769 (DOI 10.5281/zenodo.21398768), CC BY 4.0,
described in arXiv:2607.19294.

The archive (about 900 MB) is saved in data_raw/zurich/ and extracted there.

Usage: python code/fetch_loads.py
"""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_raw" / "zurich"
RECORD = "https://zenodo.org/api/records/21398769"


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD, timeout=60) as r:
        rec = json.load(r)
    (OUT / "zenodo_record.json").write_text(json.dumps(rec, indent=1))
    for f in rec["files"]:
        path = OUT / f["key"]
        expected = f["checksum"].split(":")[-1]
        if path.exists() and md5(path) == expected:
            print(f"{f['key']}: already downloaded")
        else:
            print(f"{f['key']}: downloading {f['size'] / 1e6:.0f} MB ...", flush=True)
            urllib.request.urlretrieve(f["links"]["self"], path)
            if md5(path) != expected:
                raise RuntimeError(f"checksum mismatch for {path}")
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                z.extractall(OUT / "extracted")
            print(f"extracted to {OUT / 'extracted'}")


if __name__ == "__main__":
    main()
