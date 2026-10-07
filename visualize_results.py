"""
Visualisations for the eventstudytools.com API run (24.02.2022, window -10..+10).

Inputs : results/ar_results.csv, results/aar_results.csv, results/caar_results.csv,
         results/analysis_report.csv, data/01_RequestFile.csv
Outputs: results/figures/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures"
DAYS = list(range(-10, 11))
EVENT_DAY = 0

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


def load_ar() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "ar_results.csv", sep=";")
    ar_cols = [f"AR({d})" for d in DAYS]
    df = df[["Event ID"] + ar_cols].dropna(subset=["Event ID"])
    df = df.rename(columns={c: int(c[3:-1]) for c in ar_cols})
    return df.set_index("Event ID")


def load_request() -> pd.DataFrame:
    df = pd.read_csv(
        BASE / "data" / "01_RequestFile.csv",
        sep=";",
        header=None,
        names=[
            "event_id",
            "firm_id",
            "market_id",
            "event_date",
            "sector",
            "win_start",
            "win_end",
            "est_end",
            "est_len",
        ],
    )
    return df.set_index("event_id")


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


def fig_aar_timeseries(ar: pd.DataFrame):
    aar = ar[DAYS].mean()
    n = len(ar)
    se = ar[DAYS].std() / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(aar.index, aar.values * 100, color="#1f77b4", alpha=0.8, label="AAR")
    ax.fill_between(
        np.array(aar.index, dtype=float),
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
    fig.savefig(FIGURES / "aar_timeseries.png", dpi=300)
    plt.close(fig)


def fig_caar_paths(ar: pd.DataFrame, req: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))

    overall = ar[DAYS].mean().cumsum()
    ax.plot(overall.index, overall.values * 100, color="black", linewidth=2.5, label="All firms")

    sectors = req["sector"].dropna().unique()
    for sec in sorted(sectors):
        ids = req.index[req["sector"] == sec]
        sub = ar.loc[ar.index.isin(ids), DAYS]
        if len(sub) < 5:
            continue
        path = sub.mean().cumsum()
        ax.plot(
            path.index,
            path.values * 100,
            color=SECTOR_COLORS.get(sec, "gray"),
            linewidth=1.2,
            alpha=0.85,
            label=f"{sec} (n={len(sub)})",
        )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(EVENT_DAY, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Day relative to event")
    ax.set_ylabel("Cumulative abnormal return (%)")
    ax.set_title("CAAR paths by GICS sector, window (-10, +10)", fontweight="bold")
    ax.set_xticks(DAYS)
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "caar_paths_by_sector.png", dpi=300)
    plt.close(fig)


def fig_caar_bars():
    df = pd.read_csv(RESULTS / "caar_results.csv", sep=";")
    df = df.sort_values("CAAR Value")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in df["CAAR Value"]]
    bars = ax.barh(df["Grouping Variable"], df["CAAR Value"] * 100, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("CAAR (-10, +10) (%)")
    ax.set_title("CAAR by GICS sector with Patell Z significance", fontweight="bold")

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
            fontsize=8,
        )

    ax.text(
        0.99,
        0.02,
        "*** p<0.01  ** p<0.05  * p<0.10",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "caar_by_sector.png", dpi=300)
    plt.close(fig)


def fig_car_distribution(ar: pd.DataFrame):
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
    ax.set_title(
        f"Distribution of firm-level CARs (N={len(car)}, t={t_stat:.2f})",
        fontweight="bold",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "car_distribution.png", dpi=300)
    plt.close(fig)
    return mean, med, t_stat


def fig_betas():
    df = pd.read_csv(RESULTS / "analysis_report.csv", sep=";")
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
    fig.savefig(FIGURES / "beta_distribution.png", dpi=300)
    plt.close(fig)
    return beta


def fig_aar_heatmap(ar: pd.DataFrame, req: pd.DataFrame):
    sectors = req["sector"].dropna().unique()
    rows = {}
    for sec in sorted(sectors):
        ids = req.index[req["sector"] == sec]
        sub = ar.loc[ar.index.isin(ids), DAYS]
        if len(sub) >= 5:
            rows[sec] = sub.mean()
    mat = pd.DataFrame(rows).T

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = np.abs(mat.values).max()
    im = ax.imshow(mat.values * 100, cmap="RdBu_r", vmin=-vmax * 100, vmax=vmax * 100, aspect="auto")
    ax.set_xticks(range(len(DAYS)))
    ax.set_xticklabels(DAYS)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_xlabel("Day relative to event")
    ax.set_title("Average abnormal return by sector and day (%)", fontweight="bold")
    fig.colorbar(im, ax=ax, label="AAR (%)")
    ax.axvline(10, color="black", linewidth=1.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "aar_heatmap_by_sector.png", dpi=300)
    plt.close(fig)


def fig_event_timeline(req: pd.DataFrame):
    fig_dir = FIGURES
    dec1, apr1 = pd.Timestamp("2021-12-01"), pd.Timestamp("2022-04-01")
    event = pd.Timestamp("2022-02-24")

    mkt = pd.read_csv(BASE / "data" / "03_MarketData.csv", sep=";", header=None, names=["id", "date", "px"])
    mkt["date"] = pd.to_datetime(mkt["date"], dayfirst=True)
    mkt = mkt.sort_values("date").set_index("date")
    cal = mkt.index  # trading calendar
    pos = cal.get_loc(event)
    win_start, win_end = cal[pos - 10], cal[pos + 10]

    mkt_ret = mkt["px"].pct_change()

    firms = pd.read_csv(BASE / "data" / "02_FirmData.csv", sep=";", header=None, names=["firm", "date", "px"])
    firms["date"] = pd.to_datetime(firms["date"], dayfirst=True)
    firms = firms[firms["date"].between(dec1, apr1)]
    firms["ret"] = firms.groupby("firm")["px"].pct_change()

    sector_of = req.set_index("firm_id")["sector"].dropna()
    firms = firms.join(sector_of.rename("sector"), on="firm")
    sector_ret = (
        firms.dropna(subset=["ret"]).groupby(["date", "sector"])["ret"].mean().unstack("sector")
    )

    top3 = ["Financials", "Consumer Discretionary", "Consumer Staples"]
    plot_df = pd.DataFrame({"Market (SXXP)": mkt_ret})
    for s in top3:
        plot_df[s] = sector_ret[s]
    plot_df = plot_df.loc[(plot_df.index >= dec1) & (plot_df.index <= apr1)] * 100
    cum = (1 + plot_df / 100).cumprod() * 100 - 100

    colors = {"Market (SXXP)": "black", **dict(zip(top3, ["#d62728", "#ff7f0e", "#2ca02c"]))}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col in cum.columns:
        style = dict(color=colors[col], linewidth=2 if col == "Market (SXXP)" else 1.6)
        ax.plot(cum.index, cum[col], label=col, **style)

    ax.axvline(event, color="red", linewidth=1.8, linestyle="-", label="Event (24.02.2022)")
    ax.axvspan(win_start, win_end, color="red", alpha=0.08, label="Event window (-10, +10)")
    ax.axvline(win_start, color="red", linewidth=1, linestyle="--")
    ax.axvline(win_end, color="red", linewidth=1, linestyle="--")

    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_ylabel("Cumulative return since 01.12.2021 (%)")
    ax.set_title(
        "Market and most-affected sectors around the event\n"
        "(equal-weighted sector portfolios, market model benchmark)",
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xlim(dec1, apr1)
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "event_timeline_dec_apr.png", dpi=300)
    plt.close(fig)
    print(f"Event window: {win_start.date()} to {win_end.date()}")


def main():
    FIGURES.mkdir(exist_ok=True)
    ar = load_ar()
    req = load_request()

    fig_aar_timeseries(ar)
    fig_caar_paths(ar, req)
    fig_caar_bars()
    mean, med, t_stat = fig_car_distribution(ar)
    beta = fig_betas()
    fig_aar_heatmap(ar, req)
    fig_event_timeline(req)

    import shutil

    thesis_figs = BASE.parent / "data" / "3_output" / "figures"
    if thesis_figs.exists():
        for png in FIGURES.glob("*.png"):
            shutil.copy2(png, thesis_figs / png.name)

    print(f"Figures saved to {FIGURES}")
    print(f"Mean CAR = {mean * 100:.2f}%, median = {med * 100:.2f}%, t = {t_stat:.2f}")
    print(f"Beta: mean = {beta.mean():.2f}, median = {beta.median():.2f}")


if __name__ == "__main__":
    main()
