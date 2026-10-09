"""
Run an ARC event study through the eventstudytools.com API.

Usage:
    python run_event_study.py sector [window]    # e.g. sector 10 -> 01_RequestFile_sector_w10.csv
    python run_event_study.py industry [window]
    python run_event_study.py country [window]

Window: one of 1, 5, 10, 15, 20 (symmetric around the event). Uses the
corresponding *_wNN.csv request file and writes to <name>_analysis_wNN/results/.
The original w10 request files map to the pre-existing <name>_analysis dirs.

Requires the EST_API_KEY environment variable.
"""

import os
import sys
from pathlib import Path

from eventstudytools import ARCInput, EventStudyAPI

BASE = Path(__file__).parent
WINDOWS = (1, 5, 10, 15, 20)

ANALYSES = {
    "sector": "01_RequestFile_sector_gics",
    "industry": "01_RequestFile_industry_gics",
    "country": "01_RequestFile_country",
}


def run_analysis(name: str, window: int) -> None:
    if name not in ANALYSES:
        sys.exit(f"Unknown analysis '{name}'. Choose from: {', '.join(ANALYSES)}")
    if window not in WINDOWS:
        sys.exit(f"Unknown window '{window}'. Choose from: {WINDOWS}")
    if not os.environ.get("EST_API_KEY"):
        sys.exit("EST_API_KEY environment variable is not set.")

    request_file = f"data/{ANALYSES[name]}_w{window:02d}.csv"
    dest_root = f"{name}_analysis" if window == 10 else f"{name}_analysis_w{window:02d}"
    dest = BASE / dest_root / "results"
    dest.mkdir(parents=True, exist_ok=True)

    api = EventStudyAPI()
    results = api.run(
        ARCInput(benchmark_model="mm", return_type="log"),
        files={
            "request_file": request_file,
            "firm_data": "data/02_FirmData.csv",
            "market_data": "data/03_MarketData.csv",
        },
        dest_dir=str(dest),
    )
    for rf in results:
        print(rf.name, "->", rf.local_path)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    w = int(sys.argv[2]) if len(sys.argv) == 3 else 10
    run_analysis(sys.argv[1], w)
