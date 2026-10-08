"""
Figures and LaTeX tables for the two ARC event-study analyses
(sector grouping vs. industry-group grouping).

Figures: PNG, 300 dpi (Overleaf-compatible) -> <analysis>_analysis/figures/
Tables : CSV (API output) + .tex (Overleaf \\input) -> <analysis>_analysis/results/

Usage:
    python visualize_results.py sector
    python visualize_results.py industry
    python visualize_results.py all
"""

import re
import shutil
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"
DAYS = list(range(-10, 11))
EVENT_DAY = 0
GROUP_LABEL = {"sector": "GICS sector", "industry": "GICS industry group"}

SECTOR_COLORS = {
    "Industrials": "#1f77b4",
    "Materials": "#8c564b",
    "Financials": "#d62728",
    "Utilities": "#7f7f7f",
    "Consumer Staples": "#2ca02c",
    "Consumer Discretionary": "#e377c2",
    "Communication Services": "#bcbd22",
    "Real Estate": "#9467bd",
    "Health Care": "#17becf",
    "Energy": "#ff7f0e",
    "Information Technology": "#2f4b7c",
}


def analysis_paths(name: str):
    root = BASE / f"{name}_analysis"
    return (
        root / "results",
        root / "figures",
        DATA / f"01_RequestFile_{name}_gics.csv",
    )


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


def load_ar(results: Path) -> pd.DataFrame:
    df = pd.read_csv(results / "ar_results.csv", sep=";")
    ar_cols = [f"AR({d})" for d in DAYS]
    df = df[["Event ID"] + ar_cols].dropna(subset=["Event ID"])
    df = df.rename(columns={c: int(c[3:-1]) for c in ar_cols})
    return df.set_index("Event ID")


def load_request(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=[
            "event_id",
            "firm_id",
            "market_id",
            "event_date",
            "group",
            "win_start",
            "win_end",
            "est_end",
            "est_len",
        ],
    )
    return df.set_index("event_id")


def load_caar(results: Path) -> pd.DataFrame:
    return pd.read_csv(results / "caar_results.csv", sep=";")


def group_colors(groups) -> dict:
    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    return {
        g: SECTOR_COLORS.get(g, tab20[i % 20]) for i, g in enumerate(sorted(groups))
    }


def fig_aar_timeseries(ar: pd.DataFrame, figs: Path, label: str):
    aar = ar[DAYS].mean()
    n = len(ar)
    se = ar[DAYS].std() / np.sqrt(n)
    x = np.array(aar.index, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, aar.to_numpy(dtype=float) * 100, color="#1f77b4", alpha=0.8, label="AAR")
    ax.fill_between(
        x,
        (aar - 1.96 * se).to_numpy(dtype=float) * 100,
        (aar + 1.96 * se).to_numpy(dtype=float) * 100,
        color="#1f77b4",
        alpha=0.15,
        label="95% CI",
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(EVENT_DAY, color="red", linestyle="--", linewidth=1, label="Event (24.02.2022)")
    ax.set_xlabel("Day relative to event")
    ax.set_ylabel("Average abnormal return (%)")
    ax.set_title(f"Average Abnormal Return around the event (N={n} firms)", fontweight="bold")
    ax.set_xticks(DAYS)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "aar_timeseries.png", dpi=300)
    plt.close(fig)


def fig_caar_paths(ar: pd.DataFrame, req: pd.DataFrame, caar: pd.DataFrame, figs: Path, label: str):
    groups = sorted(req["group"].dropna().unique())
    colors = group_colors(groups)

    if len(groups) > 15:
        ranked = caar.sort_values("CAAR Value")
        keep = list(ranked["Grouping Variable"].head(5)) + list(ranked["Grouping Variable"].tail(5))
        title_add = " (top/bottom 5 by CAAR)"
    else:
        keep = groups
        title_add = ""

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    overall = ar[DAYS].mean().cumsum()
    ax.plot(overall.index, overall.values * 100, color="black", linewidth=2.5, label="All firms")

    for grp in keep:
        ids = req.index[req["group"] == grp]
        sub = ar.loc[ar.index.isin(ids), DAYS]
        if len(sub) < 5:
            continue
        path = sub.mean().cumsum()
        ax.plot(
            path.index,
            path.values * 100,
            color=colors[grp],
            linewidth=1.3,
            alpha=0.85,
            label=f"{grp} (n={len(sub)})",
        )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(EVENT_DAY, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Day relative to event")
    ax.set_ylabel("Cumulative abnormal return (%)")
    ax.set_title(f"CAAR paths by {label}, window (-10, +10){title_add}", fontweight="bold")
    ax.set_xticks(DAYS)
    ax.legend(loc="best", fontsize=7, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(figs / "caar_paths.png", dpi=300)
    plt.close(fig)


def fig_caar_bars(caar: pd.DataFrame, figs: Path, label: str):
    df = caar.sort_values("CAAR Value")
    n = len(df)

    fig, ax = plt.subplots(figsize=(9, max(5.5, 0.3 * n + 2.5)))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in df["CAAR Value"]]
    bars = ax.barh(df["Grouping Variable"], df["CAAR Value"] * 100, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(f"CAAR (-10, +10) (%)")
    ax.set_title(f"CAAR by {label} with Patell Z significance", fontweight="bold")

    for bar, (_, row) in zip(bars, df.iterrows()):
        lbl = f'{row["CAAR Value"] * 100:+.2f}{stars(row["Patell Z P-Value"])}'
        x = bar.get_width()
        ax.annotate(
            lbl,
            xy=(x, bar.get_y() + bar.get_height() / 2),
            xytext=(3 if x >= 0 else -3, 0),
            textcoords="offset points",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=7,
        )

    ax.text(
        0.99,
        0.01,
        "*** p<0.01  ** p<0.05  * p<0.10",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(figs / "caar_by_group.png", dpi=300)
    plt.close(fig)


def fig_car_distribution(ar: pd.DataFrame, figs: Path, label: str):
    car = ar.sum(axis=1)
    car_win = car.clip(car.quantile(0.01), car.quantile(0.99))
    mean, med = car.mean(), car.median()
    t_stat = mean / (car.std() / np.sqrt(len(car)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(car_win * 100, bins=50, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.axvline(mean * 100, color="red", linewidth=1.5, label=f"Mean = {mean * 100:.2f}%")
    ax.axvline(med * 100, color="black", linewidth=1.5, linestyle="--", label=f"Median = {med * 100:.2f}%")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("CAR (-10, +10) (%)")
    ax.set_ylabel("Number of firms")
    ax.set_title(f"Distribution of firm-level CARs (N={len(car)}, t={t_stat:.2f})", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "car_distribution.png", dpi=300)
    plt.close(fig)
    return mean, med, t_stat


def fig_betas(results: Path, figs: Path, label: str):
    df = pd.read_csv(results / "analysis_report.csv", sep=";")
    df = df[pd.to_numeric(df["Event ID"], errors="coerce").notna()]
    beta = pd.to_numeric(df["Beta"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(beta, bins=40, color="#9467bd", alpha=0.85, edgecolor="white")
    ax.axvline(beta.mean(), color="red", linewidth=1.5, label=f"Mean = {beta.mean():.2f}")
    ax.axvline(1, color="black", linewidth=1, linestyle="--", label="Beta = 1")
    ax.set_xlabel("Market model beta (estimation window, 120 days)")
    ax.set_ylabel("Number of firms")
    ax.set_title(f"Distribution of estimated betas (N={len(beta)})", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "beta_distribution.png", dpi=300)
    plt.close(fig)
    return beta


def fig_aar_heatmap(ar: pd.DataFrame, req: pd.DataFrame, figs: Path, label: str):
    rows = {}
    for grp in sorted(req["group"].dropna().unique()):
        ids = req.index[req["group"] == grp]
        sub = ar.loc[ar.index.isin(ids), DAYS]
        if len(sub) >= 5:
            rows[grp] = sub.mean()
    mat = pd.DataFrame(rows).T
    n = len(mat)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.22 * n + 3)))
    vmax = np.abs(mat.values).max()
    im = ax.imshow(mat.values * 100, cmap="RdBu_r", vmin=-vmax * 100, vmax=vmax * 100, aspect="auto")
    ax.set_xticks(range(len(DAYS)))
    ax.set_xticklabels(DAYS)
    ax.set_yticks(range(n))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_xlabel("Day relative to event")
    ax.set_title(f"Average abnormal return by {label} and day (%)", fontweight="bold")
    fig.colorbar(im, ax=ax, label="AAR (%)")
    ax.axvline(10, color="black", linewidth=1.5)
    fig.tight_layout()
    fig.savefig(figs / "aar_heatmap.png", dpi=300)
    plt.close(fig)


def fig_event_timeline(req: pd.DataFrame, caar: pd.DataFrame, figs: Path, label: str):
    dec1, apr1 = pd.Timestamp("2021-12-01"), pd.Timestamp("2022-04-01")
    event = pd.Timestamp("2022-02-24")

    mkt = pd.read_csv(DATA / "03_MarketData.csv", sep=";", header=None, names=["id", "date", "px"])
    mkt["date"] = pd.to_datetime(mkt["date"], dayfirst=True)
    mkt = mkt.sort_values("date").set_index("date")
    cal = mkt.index
    pos = cal.get_loc(event)
    win_start, win_end = cal[pos - 10], cal[pos + 10]
    mkt_ret = mkt["px"].pct_change()

    firms = pd.read_csv(DATA / "02_FirmData.csv", sep=";", header=None, names=["firm", "date", "px"])
    firms["date"] = pd.to_datetime(firms["date"], dayfirst=True)
    firms = firms[firms["date"].between(dec1, apr1)]
    firms["ret"] = firms.groupby("firm")["px"].pct_change()
    firms = firms.join(req.set_index("firm_id")["group"], on="firm")
    grp_ret = (
        firms.dropna(subset=["ret"]).groupby(["date", "group"])["ret"].mean().unstack("group")
    )

    top3 = list(caar.sort_values("CAAR Value")["Grouping Variable"].head(3))
    plot_df = pd.DataFrame({"Market (SXXP)": mkt_ret})
    for g in top3:
        plot_df[g] = grp_ret[g]
    plot_df = plot_df.loc[(plot_df.index >= dec1) & (plot_df.index <= apr1)] * 100
    cum = (1 + plot_df / 100).cumprod() * 100 - 100

    palette = ["#d62728", "#ff7f0e", "#2ca02c"]
    colors = {"Market (SXXP)": "black", **dict(zip(top3, palette))}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col in cum.columns:
        ax.plot(
            cum.index,
            cum[col],
            label=col,
            color=colors[col],
            linewidth=2 if col == "Market (SXXP)" else 1.6,
        )

    ax.axvline(event, color="red", linewidth=1.8, label="Event (24.02.2022)")
    ax.axvspan(win_start, win_end, color="red", alpha=0.08, label="Event window (-10, +10)")
    ax.axvline(win_start, color="red", linewidth=1, linestyle="--")
    ax.axvline(win_end, color="red", linewidth=1, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.7)

    ax.set_ylabel("Cumulative return since 01.12.2021 (%)")
    ax.set_title(
        f"Market and 3 most affected {label.lower()}s around the event\n"
        "(equal-weighted portfolios, market model benchmark)",
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xlim(dec1, apr1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figs / "event_timeline_dec_apr.png", dpi=300)
    plt.close(fig)
    print(f"  Event window: {win_start.date()} to {win_end.date()}")


LATEX_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def latex_escape(value) -> str:
    return re.sub(
        r"[\\&%$#_{}~^]",
        lambda m: LATEX_ESCAPES[m.group()],
        str(value),
    )


def tex_tables(results: Path, caar: pd.DataFrame, ar: pd.DataFrame, req: pd.DataFrame, name: str, label: str):
    caar_cols = [
        "Grouping Variable",
        "CAAR Value",
        "Precision Weighted CAAR Value",
        "Number of CARs considered",
        "Patell Z",
        "Patell Z P-Value",
        "StdCSect T",
        "StdCSect T P-Value",
    ]
    caar_tex = caar[caar_cols].copy()
    caar_tex["Grouping Variable"] = caar_tex["Grouping Variable"].map(latex_escape)
    caar_tex.to_latex(
        results / "caar_table.tex",
        index=False,
        float_format="%.4f",
        caption=f"CAAR (-10, +10) by {label} ({name} analysis)",
        label=f"tab:{name}_caar",
    )

    aar = pd.read_csv(results / "aar_results.csv", sep=";")
    aar = aar[aar.iloc[:, 1] == "Average Abnormal Return (AAR)"].set_index(aar.columns[0])
    aar = aar.iloc[:, 1:]
    aar.columns = [int(c) for c in aar.columns]
    aar = aar[DAYS].astype(float)
    aar.index = [latex_escape(i) for i in aar.index]
    aar.index.name = label
    aar.to_latex(
        results / "aar_table.tex",
        float_format="%.4f",
        caption=f"AAR by day and {label} ({name} analysis)",
        label=f"tab:{name}_aar",
    )


def sync_to_thesis(name: str, figs: Path):
    target = BASE.parent / "data" / "3_output" / "figures" / f"{name}_analysis"
    if target.parent.exists():
        target.mkdir(exist_ok=True)
        for png in figs.glob("*.png"):
            shutil.copy2(png, target / png.name)
        print(f"  Synced to {target}")


def analyze(name: str):
    results, figs, request_path = analysis_paths(name)
    if not results.exists():
        sys.exit(f"{results} not found - run 'python run_event_study.py {name}' first.")
    figs.mkdir(exist_ok=True)
    label = GROUP_LABEL[name]

    ar = load_ar(results)
    req = load_request(request_path)
    caar = load_caar(results)

    fig_aar_timeseries(ar, figs, label)
    fig_caar_paths(ar, req, caar, figs, label)
    fig_caar_bars(caar, figs, label)
    mean, med, t_stat = fig_car_distribution(ar, figs, label)
    beta = fig_betas(results, figs, label)
    fig_aar_heatmap(ar, req, figs, label)
    fig_event_timeline(req, caar, figs, label)
    tex_tables(results, caar, ar, req, name, label)
    sync_to_thesis(name, figs)

    print(f"[{name}] groups={caar.shape[0]}, mean CAR={mean * 100:.2f}%, "
          f"median={med * 100:.2f}%, t={t_stat:.2f}, beta mean={beta.mean():.2f}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("sector", "industry", "all"):
        sys.exit(__doc__)
    names = ["sector", "industry"] if sys.argv[1] == "all" else [sys.argv[1]]
    for name in names:
        analyze(name)


if __name__ == "__main__":
    main()
