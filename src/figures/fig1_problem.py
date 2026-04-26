"""
Fig 1 — The coverage gap: biodiversity credit markets select against ecological rarity.

Panels:
  a) OTG transaction count distribution histogram (functional coverage)
  b) BCT market share vs like-for-like precision (IPART data)

All empirical values computed live from raw CSV data.
External IPART statistics hardcoded with explicit source citations.

Source data:
  - data/raw/nsw/nsw_credit_transactions_register.csv  (2,244 rows)
  - data/raw/nsw/nsw_credit_supply_register.csv         (3,777 rows)
  - IPART Annual Report 2022-23, Discussion Paper 2024-25
"""

import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.figures.style import (
    set_nature_style,
    COLORS,
    panel_label,
    DOUBLE_COL,
    PCT_FMT,
)

OUT = str(PROJECT_ROOT / "output" / "figures" / "main" / "fig1_problem")
DATA = PROJECT_ROOT / "data" / "raw" / "nsw" / "nsw_credit_transactions_register.csv"
SUPPLY = PROJECT_ROOT / "data" / "raw" / "nsw" / "nsw_credit_supply_register.csv"


# =============================================================================
# Data loading — all values computed from raw CSV
# =============================================================================


def load_histogram_data():
    """
    Load OTG transaction counts using the fig1 union approach (same as
    verify_claims.py): count unique OTG names in priced ecosystem
    transfers, then compute how many of the 252 supply-register OTGs are
    represented.

    This approach counts OTG names from the TRANSACTION register as the traded
    universe (140 unique names), against the 252-OTG supply register universe.
    The 112 never-traded count reflects supply-register OTGs that have no
    transaction register entry with that exact name.

    Returns
    -------
    all_counts : ndarray of ints
        Transaction counts for all 252 OTGs (zeros for never-traded).
    total_otg : int
    n_never : int
    n_functional : int
    """
    # --- Supply register: universe of 252 ecosystem OTGs ---
    supply = pd.read_csv(SUPPLY, encoding="utf-8-sig")
    supply.columns = supply.columns.str.strip()
    supply_eco = supply[supply["Ecosystem or Species"] == "Ecosystem"]
    all_supply_otgs = set(supply_eco["Offset Trading Group"].dropna().str.strip().unique())
    total_otg = len(all_supply_otgs)

    # --- Transaction register: priced ecosystem transactions ---
    txn = pd.read_csv(DATA, encoding="utf-8-sig", low_memory=False)
    txn.columns = txn.columns.str.strip()
    txn["price"] = pd.to_numeric(txn["Price Per Credit (Ex-GST)"], errors="coerce")
    transfers = txn[txn["Transaction Type"] == "Transfer"].copy()
    priced = transfers[(transfers["price"] >= 100) & transfers["price"].notna()].copy()

    # Ecosystem = Scientific Name is empty
    priced["credit_type"] = np.where(
        priced["Scientific Name"].fillna("").str.strip() != "", "Species", "Ecosystem"
    )
    eco = priced[priced["credit_type"] == "Ecosystem"].copy()
    eco["otg_clean"] = eco["Offset Trading Group"].fillna("").str.strip()

    # Fig1 union approach: count per unique OTG name in transaction register
    # (same method as verify_claims.py)
    tg_counts = eco.groupby("otg_clean").size().reset_index(name="n_txn")
    n_traded = len(tg_counts)
    n_never = total_otg - n_traded
    n_functional = int((tg_counts["n_txn"] >= 5).sum())

    traded_counts = tg_counts["n_txn"].values
    all_counts = np.concatenate([traded_counts, np.zeros(n_never, dtype=int)])

    return all_counts, total_otg, n_never, n_functional


# =============================================================================
# Panel drawing
# =============================================================================


def draw_otg_histogram(ax, all_counts, total_otg, n_never, n_functional):
    """Panel a: OTG transaction count histogram."""

    BIN_CAP = 30
    bins = list(range(0, BIN_CAP + 2))
    clipped = np.clip(all_counts, 0, BIN_CAP)

    ax.hist(
        clipped,
        bins=bins,
        color=COLORS["compliance"],
        edgecolor="white",
        linewidth=0.3,
        align="left",
        rwidth=0.85,
    )

    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.15)  # extra headroom for annotations
    ymax = ax.get_ylim()[1]

    # EC2 threshold line
    ax.axvline(x=5, color="#E31A1C", linestyle="--", linewidth=0.8, zorder=5)
    ax.text(
        5.5,
        ymax * 0.55,
        "Functional\ncoverage\nthreshold",
        fontsize=4.5,
        color="#E31A1C",
        va="top",
    )

    # Annotate never-traded group
    ax.annotate(
        f"Never traded: {n_never} OTGs ({n_never / total_otg:.0%})",
        xy=(0.5, n_never),
        xytext=(12, ymax * 0.88),
        fontsize=5,
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#999999", lw=0.5),
        va="bottom",
        ha="left",
    )

    # Annotate functional market group
    ax.annotate(
        f"Functional markets: {n_functional} OTGs ({n_functional / total_otg:.0%})",
        xy=(15, 2),
        xytext=(14, ymax * 0.42),
        fontsize=5,
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#999999", lw=0.5),
        va="bottom",
        ha="left",
    )

    ax.set_xlabel("Lifetime transaction count")
    ax.set_ylabel("Number of OTGs")

    xticks = [0, 5, 10, 15, 20, 25, 30]
    ax.set_xticks(xticks)
    xticklabels = [str(t) for t in xticks]
    xticklabels[-1] = f"{BIN_CAP}+"
    ax.set_xticklabels(xticklabels)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_bct_share(ax):
    """Panel b: BCT market share vs like-for-like precision.

    All values are from IPART reports (external data, not in CSV):
      - BCT credit share: IPART Annual Report 2022-23 p.31; Discussion Paper 2024-25 p.7
      - Like-for-like rate: Discussion Paper 2024-25 p.10
      - 2022-23 like-for-like assumed ~90% (pre-Discussion Paper era)
    """
    # Source: nsw_data.py lines 90-98, IPART Annual Report 2022-23 p.31,
    #         IPART Discussion Paper 2024-25 p.7 and p.10
    years = [2023, 2024, 2025]
    bct_credits = [7, 5, 16]           # % of total credits purchased
    lfl_pct = [90, 90, 65]             # % like-for-like rate
    x_labels = ["2022\u201323", "2023\u201324", "2024\u201325"]

    ax2 = ax.twinx()

    # BCT credit share bars (left axis, %)
    bars = ax.bar(
        years,
        bct_credits,
        color=COLORS["bct"],
        alpha=0.70,
        width=0.5,
        label="BCT credit share",
        zorder=3,
    )

    # Like-for-like line (right axis, %)
    (l2,) = ax2.plot(
        years,
        lfl_pct,
        color="#888888",
        marker="o",
        ms=4,
        lw=1.5,
        ls="--",
        label="Like-for-like rate",
        zorder=5,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("BCT share of credits (%)", color=COLORS["bct"])
    ax2.set_ylabel("Like-for-like rate (%)", color="#555555")
    ax.tick_params(axis="y", labelcolor=COLORS["bct"])
    ax2.tick_params(axis="y", labelcolor="#555555")

    ax.set_xlim(2022.5, 2025.5)
    ax.set_xticks(years)
    ax.set_xticklabels(x_labels, fontsize=5.5)
    ax.set_ylim(0, 25)
    ax2.set_ylim(50, 105)

    lines = [bars, l2]
    labels = ["BCT credit share", "Like-for-like rate"]
    ax.legend(lines, labels, fontsize=4.5, loc="upper left")

    ax.spines["top"].set_visible(False)


# =============================================================================
# Main
# =============================================================================


def main():
    set_nature_style()

    print("Loading histogram data from raw CSV...")
    all_counts, total_otg, n_never, n_functional = load_histogram_data()
    print(
        f"  {total_otg} OTGs, {n_never} never traded ({n_never/total_otg:.0%}), "
        f"{n_functional} functional ({n_functional/total_otg:.0%})"
    )

    # --- Figure layout: 1 row, 2 panels (a, b) ---
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL, 2.8),
        gridspec_kw={"wspace": 0.55},
    )

    print("Drawing panels...")
    draw_otg_histogram(ax_a, all_counts, total_otg, n_never, n_functional)
    draw_bct_share(ax_b)

    panel_label(ax_a, "a")
    panel_label(ax_b, "b")

    # Ensure output directory exists
    out_dir = Path(OUT).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(OUT + ".pdf", dpi=300)
    fig.savefig(OUT + ".png", dpi=300)
    plt.close(fig)
    print(f"Saved {OUT}.pdf")
    print(f"Saved {OUT}.png")


if __name__ == "__main__":
    main()
