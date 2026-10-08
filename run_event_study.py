"""
Run an ARC event study through the eventstudytools.com API.

Usage:
    python run_event_study.py sector     # 01_RequestFile_sector_gics.csv
    python run_event_study.py industry   # 01_RequestFile_industry_gics.csv
    python run_event_study.py country    # 01_RequestFile_country.csv

Requires the EST_API_KEY environment variable.
Outputs land in <analysis>_analysis/results/.
"""

import os
import sys
from pathlib import Path

from eventstudytools import ARCInput, EventStudyAPI

BASE = Path(__file__).parent

ANALYSES = {
    "sector": "data/01_RequestFile_sector_gics.csv",
    "industry": "data/01_RequestFile_industry_gics.csv",
    "country": "data/01_RequestFile_country.csv",
}


def run_analysis(name: str) -> None:
    if name not in ANALYSES:
        sys.exit(f"Unknown analysis '{name}'. Choose from: {', '.join(ANALYSES)}")
    if not os.environ.get("EST_API_KEY"):
        sys.exit("EST_API_KEY environment variable is not set.")

    request_file = ANALYSES[name]
    dest = BASE / f"{name}_analysis" / "results"
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
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_analysis(sys.argv[1])
