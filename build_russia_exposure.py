"""
Build the firm-level Russia-exposure dummy.

Every firm appearing in the SPLC relationship files is Russia-exposed by
construction (all counterparties are Russian):

    data/splc_firms_suppliers.xlsx, data/splc_firms_customers.xlsx

A firm counts as exposed if it appears in the `firm` column of either file.

Output: data/firm_russia_exposure.csv (one row per firm in the request files):
  firm_id;russia_exposure;ru_supplier;ru_customer;n_ru_suppliers;n_ru_customers
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"


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
        rows.append(
            {
                "firm_id": firm,
                "russia_exposure": int(len(sup_ids) > 0 or len(cus_ids) > 0),
                "ru_supplier": int(len(sup_ids) > 0),
                "ru_customer": int(len(cus_ids) > 0),
                "n_ru_suppliers": len(sup_ids),
                "n_ru_customers": len(cus_ids),
            }
        )

    out = pd.DataFrame(rows).sort_values(["russia_exposure", "firm_id"], ascending=[False, True])
    out.to_csv(DATA / "firm_russia_exposure.csv", sep=";", index=False)

    n_exposed = int(out["russia_exposure"].sum())
    print(f"Firms: {len(out)}, Russia-exposed: {n_exposed} ({n_exposed / len(out):.1%})")
    print(f"  supplier file only  : {int(((out['ru_supplier'] == 1) & (out['ru_customer'] == 0)).sum())}")
    print(f"  customer file only  : {int(((out['ru_supplier'] == 0) & (out['ru_customer'] == 1)).sum())}")
    print(f"  both files          : {int(((out['ru_supplier'] == 1) & (out['ru_customer'] == 1)).sum())}")
    print(f"Saved: {DATA / 'firm_russia_exposure.csv'}")


if __name__ == "__main__":
    main()