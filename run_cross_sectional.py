"""
Cross-sectional analysis of firm-level abnormal returns.

Builds a firm-level dataset (CAR/BHAR from the sector ARC run + Russia-exposure
dummies + GICS sector / industry group / country from the request files + optional
firm characteristics) and estimates cross-sectional OLS regressions of abnormal
returns on Russia exposure and controls.

Specifications (hypothesis ladder):
    M1  CAR ~ russia_exposure
    M2  CAR ~ ru_supplier + ru_customer
    M3  M1 + C(gics_sector)
    M4  M1 + C(country_iso)
    M5  M1 + C(gics_sector) + C(country_iso)
    M6  M1 + C(gics_sector) + russia_exposure:C(financial)            (H3 moderator)
    M7  M1 + C(gics_sector) + ln_size + pb + pe                        (controls, if available)

Robustness: BHAR as dependent variable; winsorized CAR; HC1 standard errors.

Outputs -> cross_sectional/{results,figures}/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = Path(__file__).parent
DATA = BASE / "data"
RESULTS = BASE / "cross_sectional" / "results"
FIGURES = BASE / "cross_sectional" / "figures"
DAYS = list(range(-10, 11))
CAR_COLS = [f"AR({d})" for d in DAYS]

SPECS = {
    "M1 baseline": "car ~ russia_exposure",
    "M2 supplier vs customer": "car ~ ru_supplier + ru_customer",
    "M3 + sector FE": "car ~ russia_exposure + C(gics_sector)",
    "M4 + country FE": "car ~ russia_exposure + C(country_iso)",
    "M5 + sector + country FE": "car ~ russia_exposure + C(gics_sector) + C(country_iso)",
    "M6 financial interaction": "car ~ russia_exposure + C(gics_sector) + russia_exposure:C(financial)",
    "M7 + firm controls (sector FE)": "car ~ russia_exposure + ln_size + pb + pe + C(gics_sector)",
    "M8 + firm controls (sector + country FE)": "car ~ russia_exposure + ln_size + pb + pe + C(gics_sector) + C(country_iso)",
}

ROBUSTNESS_DV = {"car_bh": "bhar ~ russia_exposure + C(gics_sector)", "car_win": "car_win ~ russia_exposure + C(gics_sector)"}


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def load_request(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", header=None, names=[
        "event_id", "firm_id", "market_id", "event_date", "group", "ws", "we", "ee", "len"])
    return df[["event_id", "firm_id", "group"]]


def build_dataset() -> pd.DataFrame:
    req_sector = load_request(DATA / "01_RequestFile_sector_gics.csv").rename(columns={"group": "gics_sector"})
    req_industry = load_request(DATA / "01_RequestFile_industry_gics.csv").rename(columns={"group": "gics_industry_group"})
    req_country = load_request(DATA / "01_RequestFile_country.csv").rename(columns={"group": "country_iso"})

    firms = req_sector.merge(req_industry, on=["event_id", "firm_id"]).merge(
        req_country, on=["event_id", "firm_id"]
    )

    ar = pd.read_csv(BASE / "sector_analysis" / "results" / "ar_results.csv", sep=";")
    ar[CAR_COLS] = ar[CAR_COLS].apply(pd.to_numeric, errors="coerce")
    car = ar[["Event ID"] + CAR_COLS].set_index("Event ID").sum(axis=1).rename("car")
    df = firms.merge(car, left_on="event_id", right_index=True, how="left")

    car_res = pd.read_csv(BASE / "sector_analysis" / "results" / "car_results.csv", sep=";")
    bhar = car_res[["Event ID", "BHAR value"]].rename(
        columns={"Event ID": "event_id", "BHAR value": "bhar"}
    )
    bhar["bhar"] = pd.to_numeric(bhar["bhar"], errors="coerce")
    df = df.merge(bhar, on="event_id", how="left")

    exp = pd.read_csv(DATA / "firm_russia_exposure.csv", sep=";")
    df = df.merge(exp, on="firm_id", how="left")

    df["financial"] = (df["gics_sector"] == "Financials").astype(int)
    df["car_win"] = df["car"].clip(df["car"].quantile(0.01), df["car"].quantile(0.99))

    controls = [DATA / "firm_controls.xlsx", DATA / "firm_characteristics.csv"]
    chars = next((p for p in controls if p.exists()), None)
    if chars is not None and chars.suffix.lower() == ".xlsx":
        c = pd.read_excel(chars)
    elif chars is not None:
        c = pd.read_csv(chars, sep=";")
    else:
        c = None
    if c is not None and "firm_id" not in c.columns and "firm" in c.columns:
        c = c.rename(columns={"firm": "firm_id"})
    if c is not None and "firm_id" in c.columns and "date" in c.columns:
        c["date"] = pd.to_datetime(c["date"])
        c = c[c["date"] <= pd.Timestamp("2022-01-31")].sort_values("date").groupby("firm_id").last().reset_index()
    if c is not None:
        df = df.merge(c.drop(columns=["date"], errors="ignore"), on="firm_id", how="left")
        if "size_eur" in df.columns:
            df["ln_size"] = np.log(df["size_eur"].replace(0, np.nan))
        for col in ("pb", "pe"):
            if col in df.columns:
                df[col + "_w"] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))
        df["pb"] = df["pb_w"]
        df["pe"] = df["pe_w"]

    return df.sort_values("firm_id").reset_index(drop=True)


def run_model(df: pd.DataFrame, formula: str) -> dict:
    m = smf.ols(formula=formula, data=df).fit(cov_type="HC1")
    res = []
    for term in m.params.index:
        if term.startswith("C(") or term in ("Intercept",):
            continue
        res.append({
            "term": term,
            "coef": m.params[term],
            "t": m.tvalues[term],
            "p": m.pvalues[term],
            "se": m.bse[term],
        })
    return {
        "n": int(m.nobs),
        "r2": float(m.rsquared),
        "r2_adj": float(m.rsquared_adj),
        "fes": [k for k in m.params.index if k.startswith("C(")],
        "rows": res,
    }


def fit_all(df: pd.DataFrame, used_car: str = "car") -> pd.DataFrame:
    model_dfs = []
    for name, formula in SPECS.items():
        res = run_model(df, formula)
        rows = pd.DataFrame(res["rows"])
        rows["model"] = name
        rows["n"] = res["n"]
        rows["r2"] = res["r2"]
        rows["r2_adj"] = res["r2_adj"]
        rows["fes"] = ", ".join(res["fes"]) if res["fes"] else ""
        model_dfs.append(rows)
    return pd.concat(model_dfs, ignore_index=True)


def tex_regression_table(df: pd.DataFrame, path: Path):
    models = list(dict.fromkeys(df["model"]))
    terms = list(dict.fromkeys(df["term"]))
    lookup = {(r["model"], r["term"]): r for _, r in df.iterrows()}
    lines = ["\\begin{table}[ht]", "\\centering", "\\small",
             "\\begin{tabular}{l" + "c" * len(models) + "}",
             "\\toprule", " & " + " & ".join(str(m) for m in models) + " \\\\",
             "\\midrule"]
    for term in terms:
        cells = []
        for model in models:
            r = lookup.get((model, term))
            if r is None or pd.isna(r["coef"]):
                cells.append("")
            else:
                cells.append(f"{r['coef']:.3f}{stars(r['p'])}\n({r['t']:.2f})")
        lines.append(" & ".join(["\\textit{" + str(term) + "}"] + cells) + " \\\\")
    lines.append("\\midrule")
    for name, col, fmt_ in [("N", "n", "{:.0f}"), ("R2", "r2", "{:.3f}")]:
        stats = [fmt_.format(df[df["model"] == m][col].iloc[0]) for m in models]
        lines.append(" & ".join([name] + stats) + " \\\\")
    fe_vals = [df[df["model"] == m]["fes"].iloc[0] for m in models]
    lines.append(" & ".join(["Sector FE"] + ["Y" if "gics_sector" in v else "" for v in fe_vals]) + " \\\\")
    lines.append(" & ".join(["Country FE"] + ["Y" if "country_iso" in v else "" for v in fe_vals]) + " \\\\")
    lines.append("\\addlinespace")
    lines.append("\\multicolumn{" + str(len(models) + 1) + "}{l}{\\footnotesize "
                 "\\textit{Notes:} HC1 robust t-statistics in parentheses. "
                 "* p$<$0.10, ** p$<$0.05, *** p$<$0.01. CAR = cumulative abnormal return "
                 "over (-10, +10).} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Cross-sectional regressions of CAR(-10,+10) on Russia exposure}",
              "\\label{tab:xs_regressions}", "\\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def descriptives(df: pd.DataFrame, path: Path):
    rows = []
    for grp, sub in df.groupby("russia_exposure"):
        rows.append({
            "group": "Exposed" if grp == 1 else "Not exposed",
            "n": len(sub),
            "car_mean": sub["car"].mean(),
            "car_median": sub["car"].median(),
            "bhar_mean": sub["bhar"].mean(),
        })
    desc = pd.DataFrame(rows)
    desc.to_csv(path.with_suffix(".csv"), sep=";", index=False)
    desc.to_latex(path, index=False, float_format="%.4f", caption="CAR(-10,+10) by Russia exposure")
    return desc


def fig_car_by_exposure(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    df["car_win_pct"] = df["car_win"] * 100
    groups = [df.loc[df["russia_exposure"] == 0, "car_win_pct"], df.loc[df["russia_exposure"] == 1, "car_win_pct"]]
    bp = ax.boxplot(groups, tick_labels=["Not exposed (0)", "Exposed (1)"], showfliers=False, widths=0.5)
    for box, color in zip(bp["boxes"], ["#1f77b4", "#d62728"]):
        box.set(color=color, linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.8)
    for i, g in enumerate(groups, start=1):
        ax.text(i, g.mean(), f"mean {g.mean():+.2f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("CAR(-10, +10) winsorized (%)")
    ax.set_title("Firm-level CARs by Russia exposure", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "car_by_exposure.png", dpi=300)
    plt.close(fig)


def fig_coef_plot(df: pd.DataFrame, dvs: list):
    models = []
    for dv in dvs:
        res = run_model(df, ROBUSTNESS_DV[dv] if dv in ROBUSTNESS_DV else SPECS["M3 + sector FE"])
        if res["rows"]:
            r = res["rows"][0] if res["rows"][0]["term"] == "russia_exposure" else next(
                (x for x in res["rows"] if x["term"] == "russia_exposure"), None)
            if r:
                models.append((dv, r["coef"], r["se"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, coef, se) in enumerate(models):
        ax.errorbar([coef], [i], xerr=[1.96 * se], fmt="o", capsize=4, label=name)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m[0] for m in models])
    ax.set_xlabel("Russia exposure coefficient")
    ax.set_title("Russia exposure coefficient across specifications\n(95% CI, sector FE)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "coef_plot.png", dpi=300)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    df.to_csv(RESULTS / "xs_dataset.csv", sep=";", index=False)
    print(f"Dataset: {len(df)} firms, CAR obs {df['car'].notna().sum()}")

    descriptives(df, RESULTS / "descriptives.tex")

    res = fit_all(df)
    res.to_csv(RESULTS / "regressions.csv", sep=";", index=False)
    tex_regression_table(res, RESULTS / "regressions_table.tex")

    rob = []
    for dv, formula in ROBUSTNESS_DV.items():
        dep_col = formula.split("~")[0].strip()
        sub = df.dropna(subset=[dep_col])
        r = run_model(sub, formula)
        for x in r["rows"]:
            x["dv"] = dv
        rob += r["rows"]
    rob_df = pd.DataFrame(rob)
    if not rob_df.empty:
        rob_df.to_csv(RESULTS / "robustness.csv", sep=";", index=False)

    fig_car_by_exposure(df)
    fig_coef_plot(df, dvs=["car", "car_win", "car_bh"])

    print("\nRegression results (russia_exposure):")
    for _, row in res[res["term"] == "russia_exposure"].iterrows():
        print(f"  {row['model']:<28} coef {row['coef']:+.4f} (t={row['t']: .2f}, p={row['p']:.3f})")
    print(f"\nOutputs -> {RESULTS}")


if __name__ == "__main__":
    main()