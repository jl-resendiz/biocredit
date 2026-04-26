"""
Figure 2: Model validation and policy outcomes.

Panels:
  a) Observed vs predicted transaction distribution by vegetation formation.
  b) Policy-scenario functional coverage with confidence intervals.

OBSERVED values in panel (a) computed live from raw CSV data.
PREDICTED and scenario results read from output/results/abm_results.json.

Source:
  - data/raw/nsw/nsw_credit_transactions_register.csv  (2,244 rows)
  - data/raw/nsw/nsw_credit_supply_register.csv         (3,777 rows)
  - scripts/run_abm.py output (50 MC seeds, 72 timesteps)
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.figures.style import (
    DOUBLE_COL,
    set_nature_style,
    panel_label,
    COLORS,
)

DATA = PROJECT_ROOT / "data" / "raw" / "nsw" / "nsw_credit_transactions_register.csv"
SUPPLY = PROJECT_ROOT / "data" / "raw" / "nsw" / "nsw_credit_supply_register.csv"
OUT_DIR = PROJECT_ROOT / "output" / "figures" / "main"
RESULTS_DIR = PROJECT_ROOT / "output" / "results"

FORMATION_NORMALIZE = {
    "Grassy woodlands": "Grassy Woodlands",
}


# =============================================================================
# Fallback values (used when abm_results.json not found)
# Source: run_abm.py baseline output (50 MC seeds, 72 timesteps)
# =============================================================================

FALLBACK_PREDICTED_SHARES = {
    "Grassy Woodlands": 39.1,
    "Dry Sclerophyll Forests (Shrubby sub-formation)": 19.7,
    "Dry Sclerophyll Forests (Shrub/grass sub-formation)": 12.2,
    "Forested Wetlands": 7.7,
    "Semi-arid Woodlands (Grassy sub-formation)": 5.6,
    "Wet Sclerophyll Forests (Shrubby sub-formation)": 3.6,
    "Freshwater Wetlands": 3.1,
    "Semi-arid Woodlands (Shrubby sub-formation)": 2.5,
    "Wet Sclerophyll Forests (Grassy sub-formation)": 1.9,
    "Rainforests": 1.9,
    "Grasslands": 1.7,
    "Arid Shrublands (Chenopod sub-formation)": 0.6,
    "Heathlands": 0.2,
    "Saline Wetlands": 0.3,
    "Arid Shrublands (Acacia sub-formation)": 0.0,
}

FALLBACK_FUND_ROUTING = {
    "model_compliance_pct": 65.8,
    "model_bct_pct": 34.2,
}

# Fallback values from abm_results.json (2026-03-20)
FALLBACK_SCENARIOS = {
    "Baseline": {"ec2_median": 16.3, "ec2_p10": 14.6, "ec2_p90": 17.9},
    "Procurement Flex (20%)": {"ec2_median": 17.1, "ec2_p10": 15.4, "ec2_p90": 18.7},
    "Price Floor (AUD 3,000)": {"ec2_median": 16.3, "ec2_p10": 14.7, "ec2_p90": 17.9},
    "Combined": {"ec2_median": 17.1, "ec2_p10": 15.5, "ec2_p90": 18.7},
}


# =============================================================================
# Load results from JSON or fallback
# =============================================================================


def load_abm_results():
    """Load ABM results from JSON, falling back to hardcoded values."""
    json_path = RESULTS_DIR / "abm_results.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        print(f"Loaded ABM results from {json_path}")
        print(f"  Timestamp: {data.get('timestamp', 'unknown')}")
        return data
    print("WARNING: output/results/abm_results.json not found. Using fallback values.")
    print("Run 'python scripts/run_abm.py' to generate fresh results.")
    return None


# =============================================================================
# Data loading -- OBSERVED shares computed live from CSV
# =============================================================================


def compute_observed_formation_shares():
    """Compute observed transaction share per formation from raw CSV.

    Returns dict: {formation_name: share_pct} where share_pct sums to ~100%.
    """
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
    eco["formation"] = (
        eco["Vegetation Formation"]
        .fillna("")
        .str.strip()
        .replace(FORMATION_NORMALIZE)
    )
    eco = eco[eco["formation"] != ""]

    form_counts = eco.groupby("formation").size().reset_index(name="n_txn")
    total = form_counts["n_txn"].sum()
    form_counts["share_pct"] = form_counts["n_txn"] / total * 100.0

    return dict(zip(form_counts["formation"], form_counts["share_pct"]))


# =============================================================================
# Short names for formation labels
# =============================================================================

def shorten_formation(name):
    """Shorten long formation names for readability on x-axis."""
    replacements = [
        ("Dry Sclerophyll Forests (Shrubby sub-formation)", "Dry Scl. (Shrubby)"),
        ("Dry Sclerophyll Forests (Shrub/grass sub-formation)", "Dry Scl. (Shrub/grass)"),
        ("Wet Sclerophyll Forests (Grassy sub-formation)", "Wet Scl. (Grassy)"),
        ("Wet Sclerophyll Forests (Shrubby sub-formation)", "Wet Scl. (Shrubby)"),
        ("Semi-arid Woodlands (Grassy sub-formation)", "Semi-arid Wdl (Grassy)"),
        ("Semi-arid Woodlands (Shrubby sub-formation)", "Semi-arid Wdl (Shrubby)"),
        ("Arid Shrublands (Chenopod sub-formation)", "Arid Shrub (Chenopod)"),
        ("Arid Shrublands (Acacia sub-formation)", "Arid Shrub (Acacia)"),
    ]
    for long, short in replacements:
        if name == long:
            return short
    return name


# =============================================================================
# Build figure
# =============================================================================


def make_fig2():
    set_nature_style()

    # --- Load ABM results (JSON or fallback) ---
    abm = load_abm_results()

    if abm is not None:
        predicted_shares = {
            fname: fdata["predicted_pct"]
            for fname, fdata in abm["formation_shares"].items()
        }
        fund_routing = abm["fund_routing"]
        scenarios_raw = abm["scenarios"]
        # Display the six Table 1 scenarios in the same order:
        # Bypass > Procurement Flex > Rarity > Baseline > Price Floor > BCT Precision.
        scenario_order = [
            ("Bypass Reduction", "Bypass\nReduction"),
            ("Procurement Flex (20%)", "Procurement\nFlex (20%)"),
            ("Rarity Multiplier (2x)", "Rarity\nMultiplier"),
            ("Baseline", "Baseline"),
            ("Price Floor (AUD 3,000)", "Price Floor\n(AUD 3,000)"),
            ("BCT Precision Mandate", "BCT Precision\nMandate"),
        ]
        scenario_display = {}
        for json_key, display in scenario_order:
            if json_key in scenarios_raw:
                vals = scenarios_raw[json_key]
                scenario_display[display] = {
                    "ec2": vals["ec2_median"],
                    "ci_lo": vals["ec2_p10"],
                    "ci_hi": vals["ec2_p90"],
                }
    else:
        predicted_shares = FALLBACK_PREDICTED_SHARES
        fund_routing = FALLBACK_FUND_ROUTING
        scenario_display = {
            "Baseline": {
                "ec2": FALLBACK_SCENARIOS["Baseline"]["ec2_median"],
                "ci_lo": FALLBACK_SCENARIOS["Baseline"]["ec2_p10"],
                "ci_hi": FALLBACK_SCENARIOS["Baseline"]["ec2_p90"],
            },
            "Procurement\nFlex (20%)": {
                "ec2": FALLBACK_SCENARIOS["Procurement Flex (20%)"]["ec2_median"],
                "ci_lo": FALLBACK_SCENARIOS["Procurement Flex (20%)"]["ec2_p10"],
                "ci_hi": FALLBACK_SCENARIOS["Procurement Flex (20%)"]["ec2_p90"],
            },
            "Price Floor\n(AUD 3,000)": {
                "ec2": FALLBACK_SCENARIOS["Price Floor (AUD 3,000)"]["ec2_median"],
                "ci_lo": FALLBACK_SCENARIOS["Price Floor (AUD 3,000)"]["ec2_p10"],
                "ci_hi": FALLBACK_SCENARIOS["Price Floor (AUD 3,000)"]["ec2_p90"],
            },
            "Combined": {
                "ec2": FALLBACK_SCENARIOS["Combined"]["ec2_median"],
                "ci_lo": FALLBACK_SCENARIOS["Combined"]["ec2_p10"],
                "ci_hi": FALLBACK_SCENARIOS["Combined"]["ec2_p90"],
            },
        }

    # --- Compute observed shares from raw data ---
    print("Computing observed formation shares from raw CSV...")
    observed_shares = compute_observed_formation_shares()

    # Observed Fund routing (always from CODEBOOK.md, not model)
    observed_compliance_pct = 64.6  # Source: CODEBOOK.md: 709/1097 = 64.6%
    observed_bct_pct = 35.4         # Source: CODEBOOK.md: 388/1097 = 35.4%
    observed_ec2 = 17.9             # Source: CODEBOOK.md: 45/252 = 17.9%

    # --- Merge observed + predicted into sorted list ---
    all_formations = sorted(observed_shares.keys(), key=lambda f: -observed_shares[f])

    names = [shorten_formation(f) for f in all_formations]
    obs_vals = [observed_shares.get(f, 0.0) for f in all_formations]
    pred_vals = [predicted_shares.get(f, 0.0) for f in all_formations]

    print(f"  {len(all_formations)} formations, top 3 observed: "
          f"{all_formations[0]} ({obs_vals[0]:.1f}%), "
          f"{all_formations[1]} ({obs_vals[1]:.1f}%), "
          f"{all_formations[2]} ({obs_vals[2]:.1f}%)")

    # --- Figure layout ---
    fig = plt.figure(figsize=(DOUBLE_COL, 5.5))
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[1, 0.85],
        hspace=0.55,
        left=0.10, right=0.97, top=0.96, bottom=0.10,
    )

    ax_a = fig.add_subplot(gs[0, 0])   # Panel a top: validation
    ax_b = fig.add_subplot(gs[1, 0])   # Panel b bottom: scenarios

    # -- Panel a: Observed vs predicted transaction shares ---------
    COLOR_OBS = COLORS["compliance"]  # blue
    COLOR_PRED = COLORS["bct"]        # orange

    x = np.arange(len(names))
    w = 0.38
    ax_a.bar(x - w / 2, obs_vals, w, color=COLOR_OBS, label="Observed", zorder=3)
    ax_a.bar(x + w / 2, pred_vals, w, color=COLOR_PRED, label="Predicted", zorder=3)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(names, rotation=55, ha="right", fontsize=5.5)
    ax_a.set_ylabel("% of total transactions")
    ax_a.legend(frameon=False, fontsize=6)
    ax_a.set_xlim(-0.6, len(names) - 0.4)
    ax_a.set_ylim(0, max(max(obs_vals), max(pred_vals)) * 1.15)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    panel_label(ax_a, "a", x=-0.06, y=1.08)

    # -- Panel b: Policy scenario FC with confidence intervals -----
    scenario_names = list(scenario_display.keys())
    ec2_vals = [scenario_display[s]["ec2"] for s in scenario_names]
    ci_lo = [scenario_display[s]["ci_lo"] for s in scenario_names]
    ci_hi = [scenario_display[s]["ci_hi"] for s in scenario_names]

    err_lo = [v - l for v, l in zip(ec2_vals, ci_lo)]
    err_hi = [h - v for v, h in zip(ec2_vals, ci_hi)]

    # Colors aligned with Table 1 order:
    # Bypass (effective, dark teal), Procurement (effective, lighter teal),
    # Rarity (neutral, grey), Baseline (reference, grey),
    # Price Floor (counterproductive, light orange),
    # BCT Precision (counterproductive, dark orange).
    SCENARIO_COLORS = [
        "#1D91C0",  # Bypass — strongest effect
        "#41B6C4",  # Procurement Flex
        "#A1D99B",  # Rarity Multiplier
        "#BDBDBD",  # Baseline (reference)
        "#FDAE6B",  # Price Floor
        "#E6550D",  # BCT Precision (most damaging)
    ]

    x_b = np.arange(len(scenario_names))
    bars = ax_b.bar(
        x_b, ec2_vals, color=SCENARIO_COLORS, width=0.55,
        edgecolor="black", linewidth=0.4, zorder=3,
    )
    ax_b.errorbar(
        x_b, ec2_vals, yerr=[err_lo, err_hi],
        fmt="none", ecolor="black", elinewidth=0.7, capsize=3, zorder=4,
    )
    ax_b.axhline(
        observed_ec2, color="#E31A1C", ls="--", lw=0.8, zorder=2,
        label=f"Observed = {observed_ec2}%",
    )
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(scenario_names, fontsize=7)
    ax_b.set_ylabel("Functional coverage (%)")
    ax_b.set_ylim(0, 24)
    ax_b.legend(frameon=False, fontsize=6, loc="upper left")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    panel_label(ax_b, "b", x=-0.08, y=1.06)

    # -- Save -------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig2_mechanism.pdf")
    fig.savefig(OUT_DIR / "fig2_mechanism.png")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'fig2_mechanism.pdf'}")
    print(f"Saved: {OUT_DIR / 'fig2_mechanism.png'}")


if __name__ == "__main__":
    make_fig2()
