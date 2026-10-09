"""Generate event-study request files for multiple event windows.

For each grouping (sector / industry / country) and each symmetric window
W in {1, 5, 10, 15, 20}, writes data/01_RequestFile_<grouping>_w{W:02d}.csv
with win_start=-W, win_end=+W. Estimation window unchanged (ends day -11,
120 days), so the estimation gap stays identical across windows.

The w10 files are exact copies of the original request files (which used
(-10, +10)) and are included for uniformity.
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"
WINDOWS = (1, 5, 10, 15, 20)

FILES = {
    "sector": "01_RequestFile_sector_gics.csv",
    "industry": "01_RequestFile_industry_gics.csv",
    "country": "01_RequestFile_country.csv",
}


def main():
    for name, path in FILES.items():
        df = pd.read_csv(DATA / path, sep=";", header=None)
        stem = Path(path).stem  # keep original stem incl. "_gics" suffix
        for w in WINDOWS:
            out = df.copy()
            out[5] = -w  # win_start
            out[6] = w   # win_end
            outfile = DATA / f"{stem}_w{w:02d}.csv"
            out.to_csv(outfile, sep=";", header=False, index=False)
            print(f"  {outfile.name}: window (-{w}, +{w}), est_end day {out.iloc[0, 7]} len {out.iloc[0, 8]}")

    print(f"Generated {len(FILES) * len(WINDOWS)} request files in {DATA}")


if __name__ == "__main__":
    main()