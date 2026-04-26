"""
Figure 3: Robustness -- global Sobol sensitivity analysis and demand-function
specification test.

Panels:
  a) Sobol tornado plot: total Sobol index S_Ti (and first-order S_i) per
     parameter, ranked by total contribution to FC variance. Bootstrap 95%
     CI whiskers. Source: output/results/sobol_indices.json.
  b) Demand-function specification test: ad-hoc vs micro-founded formation-
     level transaction correlation scatter with Spearman rho annotation.
     Source: output/results/demand_robustness_results.json.

Source scripts:
  - scripts/sensitivity_sobol.py        (Saltelli sampling, SALib)
  - scripts/robustness_demand_functions.py
"""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.figures.style import (
    DOUBLE_COL,
    set_nature_style,
    panel_label,
    COLORS,
)

OUT_DIR = PROJECT_ROOT / "output" / "figures" / "main"
RESULTS_DIR = PROJECT_ROOT / "output" / "results"


# =============================================================================
# Fallback values (used when JSON not found)
# Source: robustness_sensitivity.py and robustness_demand_functions.py output
# =============================================================================

FALLBACK_SENSITIVITY = [
    {
        "param": "Production cost",
        "range_label": "$399\u2013$4,047",
        "ec2_lo": 16.3,
        "ec2_hi": 16.9,
    },
    {
        "param": "n_compliance",
        "range_label": "10\u201360",
        "ec2_lo": 16.3,
        "ec2_hi": 17.1,
    },
    {
        "param": "n_intermediary",
        "range_label": "3\u201320",
        "ec2_lo": 16.3,
        "ec2_hi": 17.1,
    },
    {
        "param": "n_habitat_bank",
        "range_label": "10\u201360",
        "ec2_lo": 16.5,
        "ec2_hi": 17.3,
    },
    {
        "param": "P_variation",
        "range_label": "0.3\u20130.7",
        "ec2_lo": 15.5,
        "ec2_hi": 17.5,
    },
    {
        "param": "n_bct",
        "range_label": "4\u201324",
        "ec2_lo": 13.3,
        "ec2_hi": 18.7,
    },
]

FALLBACK_BASELINE_EC2 = 16.3

FALLBACK_DEMAND_FUNC_DATA = {
    "Grassy Woodlands":                                     (310, 295),
    "Dry Scl. (Shrubby)":                                   (156, 148),
    "Dry Scl. (Shrub/grass)":                               (97, 92),
    "Forested Wetlands":                                     (61, 58),
    "Semi-arid Wdl (Grassy)":                                (44, 42),
    "Wet Scl. (Shrubby)":                                    (29, 27),
    "Freshwater Wetlands":                                    (25, 24),
    "Semi-arid Wdl (Shrubby)":                               (20, 19),
    "Wet Scl. (Grassy)":                                      (15, 14),
    "Rainforests":                                             (15, 14),
    "Grasslands":                                              (13, 12),
    "Arid Shrub (Chenopod)":                                   (5, 5),
    "Heathlands":                                               (2, 2),
    "Saline Wetlands":                                          (2, 2),
    "Arid Shrub (Acacia)":                                      (0, 0),
}

FALLBACK_SPEARMAN_RHO = 0.97
FALLBACK_QUALITATIVE_CLAIMS = "5/5"


# =============================================================================
# Load results from JSON or fallback
# =============================================================================


def load_sensitivity_results():
    """Load sensitivity results from JSON, falling back to hardcoded values."""
    json_path = RESULTS_DIR / "sensitivity_results.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        print(f"Loaded sensitivity results from {json_path}")
        print(f"  Timestamp: {data.get('timestamp', 'unknown')}")
        return data
    print("WARNING: output/results/sensitivity_results.json not found. Using fallback values.")
    print("Run 'python scripts/robustness_sensitivity.py' to generate fresh results.")
    return None


def load_demand_robustness_results():
    """Load demand robustness results from JSON, falling back to hardcoded values."""
    json_path = RESULTS_DIR / "demand_robustness_results.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        print(f"Loaded demand robustness results from {json_path}")
        print(f"  Timestamp: {data.get('timestamp', 'unknown')}")
        return data
    print("WARNING: output/results/demand_robustness_results.json not found. Using fallback values.")
    print("Run 'python scripts/robustness_demand_functions.py' to generate fresh results.")
    return None


# =============================================================================
# Build figure
# =============================================================================


def make_fig3():
    set_nature_style()

    # --- Load sensitivity results ---
    sens_data = load_sensitivity_results()
    if sens_data is not None:
        sensitivity = []
        baseline_ec2 = sens_data.get("baseline_ec2", FALLBACK_BASELINE_EC2)
        for param_key, sweep_info in sens_data.get("sweeps", {}).items():
            sensitivity.append({
                "param": sweep_info.get("param", param_key),
                "range_label": sweep_info.get("range", ""),
                "ec2_lo": sweep_info["ec2_lo"],
                "ec2_hi": sweep_info["ec2_hi"],
            })
        # Sort by span ascending (least sensitive first)
        sensitivity.sort(key=lambda s: s["ec2_hi"] - s["ec2_lo"])
    else:
        sensitivity = FALLBACK_SENSITIVITY
        baseline_ec2 = FALLBACK_BASELINE_EC2

    # --- Load demand robustness results ---
    demand_data = load_demand_robustness_results()
    if demand_data is not None:
        demand_func_data = {}
        for fname, counts in demand_data.get("formation_counts", {}).items():
            demand_func_data[fname] = (counts["adhoc"], counts["microfounded"])
        spearman_rho = demand_data.get("spearman_rho", FALLBACK_SPEARMAN_RHO)
        qualitative_claims = demand_data.get(
            "qualitative_claims_passed", FALLBACK_QUALITATIVE_CLAIMS
        )
    else:
        demand_func_data = FALLBACK_DEMAND_FUNC_DATA
        spearman_rho = FALLBACK_SPEARMAN_RHO
        qualitative_claims = FALLBACK_QUALITATIVE_CLAIMS

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL, 3.2),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.40},
    )

    # -- Panel a: Sobol tornado --------------------------------
    sobol_path = RESULTS_DIR / "sobol_indices.json"
    if sobol_path.exists():
        with open(sobol_path, "r", encoding="utf-8") as fh:
            sobol = json.load(fh)
        names = sobol["problem"]["names"]
        s1 = np.array(sobol["sobol"]["S1"])
        s1c = np.array(sobol["sobol"]["S1_conf"])
        st = np.array(sobol["sobol"]["ST"])
        stc = np.array(sobol["sobol"]["ST_conf"])
        n_design = sobol.get("design", {}).get("N", "?")
        # Sort by total Sobol descending
        order = np.argsort(st)
        names_s = [names[i] for i in order]
        s1_s = s1[order]
        s1c_s = s1c[order]
        st_s = st[order]
        stc_s = stc[order]

        DISPLAY = {
            "P_variation": r"$P_\mathrm{variation}$",
            "n_BCT": r"$n_\mathrm{BCT}$",
            "monthly_obligations_mult": "monthly obligations",
            "production_cost_mu": r"$\mu_{\log c}$",
            "power_law_alpha": r"power-law $\alpha$",
            "liquidity_weight_max": "liquidity weight max",
        }
        labels = [DISPLAY.get(n, n) for n in names_s]

        y = np.arange(len(names_s))
        # Total Sobol bars (filled)
        ax_a.barh(
            y, st_s, height=0.55, color=COLORS["compliance"], alpha=0.85,
            edgecolor="white", linewidth=0.4, zorder=3,
            label=r"Total Sobol $S_{T_i}$",
        )
        # First-order Sobol overlay (smaller)
        ax_a.barh(
            y, s1_s, height=0.30, color=COLORS["bct"], alpha=0.95,
            edgecolor="white", linewidth=0.4, zorder=4,
            label=r"First-order $S_i$",
        )
        # Bootstrap whiskers on Total
        ax_a.errorbar(
            st_s, y, xerr=stc_s, fmt="none",
            ecolor="black", elinewidth=0.6, capsize=2.5, zorder=5,
        )
        ax_a.set_yticks(y)
        ax_a.set_yticklabels(labels, fontsize=6.5)
        ax_a.set_xlabel("Sobol index for functional coverage")
        ax_a.set_xlim(0, max(0.05, max(st_s) * 1.25))
        ax_a.axvline(0, color="#888888", lw=0.4, zorder=1)
        ax_a.legend(frameon=False, fontsize=5.5, loc="lower right")
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)
        panel_label(ax_a, "a", x=-0.32, y=1.06)
        ax_a.text(
            0.97, 0.04, f"$N={n_design}$ (Saltelli)",
            transform=ax_a.transAxes, fontsize=5.5, ha="right", va="bottom",
            color="#666666",
        )
    else:
        # Fallback: no Sobol JSON yet; render a placeholder note.
        ax_a.text(
            0.5, 0.5,
            "Sobol indices not yet computed.\n"
            "Run `python scripts/sensitivity_sobol.py --N 128 --inner-seeds 3 --k 3 --workers 8`\n"
            "to populate output/results/sobol_indices.json and re-render.",
            transform=ax_a.transAxes,
            fontsize=7, ha="center", va="center",
            color="#888888",
        )
        ax_a.set_xticks([])
        ax_a.set_yticks([])
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)
        ax_a.spines["left"].set_visible(False)
        ax_a.spines["bottom"].set_visible(False)
        panel_label(ax_a, "a", x=-0.32, y=1.06)

    # -- Panel b: Demand function scatter (ad-hoc vs micro-founded)
    formations = list(demand_func_data.keys())
    adhoc_txn = np.array([demand_func_data[f][0] for f in formations], dtype=float)
    mf_txn = np.array([demand_func_data[f][1] for f in formations], dtype=float)

    # Scatter plot
    ax_b.scatter(
        adhoc_txn, mf_txn,
        s=25, color=COLORS["compliance"], edgecolor="white",
        linewidth=0.3, zorder=3, alpha=0.9,
    )

    # 1:1 reference line
    max_val = max(adhoc_txn.max(), mf_txn.max()) * 1.1
    ax_b.plot(
        [0, max_val], [0, max_val],
        color="#AAAAAA", ls="--", lw=0.6, zorder=1,
    )

    # Label top formations
    label_threshold = 50
    for i, f in enumerate(formations):
        if adhoc_txn[i] >= label_threshold:
            ax_b.annotate(
                f, (adhoc_txn[i], mf_txn[i]),
                textcoords="offset points",
                xytext=(5, -3),
                fontsize=4.5,
                color="#555555",
            )

    # Spearman rho annotation
    ax_b.text(
        0.05, 0.92,
        f"Spearman $\\rho$ = {spearman_rho:.2f}\n{qualitative_claims} claims preserved",
        transform=ax_b.transAxes,
        fontsize=6, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
    )

    ax_b.set_xlabel("Ad-hoc demand (txn count)")
    ax_b.set_ylabel("Micro-founded demand (txn count)")
    ax_b.set_xlim(-10, max_val)
    ax_b.set_ylim(-10, max_val)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    panel_label(ax_b, "b", x=-0.16, y=1.06)

    # -- Save -------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig3_policy.pdf")
    fig.savefig(OUT_DIR / "fig3_policy.png")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'fig3_policy.pdf'}")
    print(f"Saved: {OUT_DIR / 'fig3_policy.png'}")


if __name__ == "__main__":
    make_fig3()
