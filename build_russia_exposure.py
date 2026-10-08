"""
Build the firm-level Russia-exposure dummy.

A firm is Russia-exposed (russia_exposure = 1) if it has at least one supplier
or customer located in / linked to Russia, based on the SPLC relationship files:

    data/splc_firms_suppliers.xlsx, data/splc_firms_customers.xlsx

Russian counterparties are identified by their Bloomberg id:
  - exchange suffix " RU Equity" (MICEX) or " RM Equity" (MOEX), or
  - known Russian corporates listed abroad (RUSAL 486 HK, Evraz, Petropavlovsk,
    Polyus Gold, Highland Gold, En+, O'Key).

Output: data/firm_russia_exposure.csv (one row per firm in the request files):
  firm_id;russia_exposure;ru_supplier;ru_customer;n_ru_suppliers;n_ru_customers
"""

import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"

RU_ID_RE = re.compile(r"\b(RU|RM) Equity$")
KNOWN_RU_ABROAD = {
    "ENPL LI Equity",     # En+ Group
    "EVR LN Equity",      # Evraz
    "HGM LN Equity",      # Highland Gold
    "OKEY LI Equity",     # O'Key Group
    "POG LN Equity",      # Petropavlovsk
    "PGIL LN Equity",     # Polyus Gold International
    "486 HK Equity",      # RUSAL (HK listing)
}


def is_russian(counterparty_id: str) -> bool:
    return bool(RU_ID_RE.search(counterparty_id)) or counterparty_id in KNOWN_RU_ABROAD


def request_firms() -> pd.Series:
    firm_series = []
    for name in ("sector_gics", "industry_gics", "country"):
        df = pd.read_csv(DATA / f"01_RequestFile_{name}.csv", sep=";", header=None)
        firm_series.append(df[1])
    firms = pd.concat(firm_series).dropna().drop_duplicates()
    return firms.sort_values().reset_index(drop=True)


def main():
    suppliers = pd.read_excel(DATA / "splc_firms_suppliers.xlsx")
    customers = pd.read_excel(DATA / "splc_firms_customers.xlsx")

    rows = []
    for firm in request_firms():
        sup_ids = suppliers.loc[suppliers["firm"] == firm, "id"].astype(str)
        cus_ids = customers.loc[customers["firm"] == firm, "id"].astype(str)
        ru_sup = {x for x in sup_ids if is_russian(x)}
        ru_cus = {x for x in cus_ids if is_russian(x)}
        rows.append(
            {
                "firm_id": firm,
                "russia_exposure": int(len(ru_sup) > 0 or len(ru_cus) > 0),
                "ru_supplier": int(len(ru_sup) > 0),
                "ru_customer": int(len(ru_cus) > 0),
                "n_ru_suppliers": len(ru_sup),
                "n_ru_customers": len(ru_cus),
            }
        )

    out = pd.DataFrame(rows).sort_values(["russia_exposure", "firm_id"], ascending=[False, True])
    out.to_csv(DATA / "firm_russia_exposure.csv", sep=";", index=False)

    n_exposed = int(out["russia_exposure"].sum())
    print(f"Firms: {len(out)}, Russia-exposed: {n_exposed} ({n_exposed / len(out):.1%})")
    print(f"  via supplier only : {int(((out['ru_supplier'] == 1) & (out['ru_customer'] == 0)).sum())}")
    print(f"  via customer only : {int(((out['ru_supplier'] == 0) & (out['ru_customer'] == 1)).sum())}")
    print(f"  via both          : {int(((out['ru_supplier'] == 1) & (out['ru_customer'] == 1)).sum())}")
    print(f"Saved: {DATA / 'firm_russia_exposure.csv'}")


if __name__ == "__main__":
    main()