"""
Trend analysis: do sub-threshold OTGs show upward trend in transaction frequency?

Sub-threshold = OTGs with < 5 lifetime priced ecosystem transactions.
Tests whether monthly transaction counts for these OTGs trend upward over time.

Methods:
  - Mann-Kendall test (non-parametric, no normality assumption) on monthly
    transaction counts aggregated across all sub-threshold OTGs.
  - OLS linear regression slope as effect-size estimate.
  - Also tests never-traded OTGs separately.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TXN_FILE = PROJECT_ROOT / "data" / "raw" / "nsw" / "nsw_credit_transactions_register.csv"
SUPPLY_FILE = PROJECT_ROOT / "data" / "raw" / "nsw" / "nsw_credit_supply_register.csv"

PRICE_FLOOR = 100
FUNCTIONAL_THRESHOLD = 5

SEP = "=" * 72
SUB = "-" * 72

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
txn = pd.read_csv(TXN_FILE, dtype=str, encoding="utf-8-sig")
supply = pd.read_csv(SUPPLY_FILE, dtype=str, encoding="utf-8-sig")

txn["date"] = pd.to_datetime(txn["Transaction Date"], errors="coerce")
txn["price"] = pd.to_numeric(txn["Price Per Credit (Ex-GST)"], errors="coerce")

# Ecosystem priced transfers only (same filter as verify_claims.py)
eco = txn[
    (txn["Transaction Type"] == "Transfer")
    & (txn["Transaction Status"] == "Completed")
    & (txn["price"] >= PRICE_FLOOR)
    & txn["Offset Trading Group"].notna()
    & txn["Vegetation Formation"].notna()
].copy()

# All 252 supply OTGs
supply_otgs = set(supply["Offset Trading Group"].dropna().str.strip().unique())

# Lifetime transaction count per OTG
otg_counts = eco.groupby("Offset Trading Group").size().reset_index(name="lifetime_txns")

# Classify OTGs
all_otg_df = pd.DataFrame({"Offset Trading Group": list(supply_otgs)})
all_otg_df = all_otg_df.merge(otg_counts, on="Offset Trading Group", how="left")
all_otg_df["lifetime_txns"] = all_otg_df["lifetime_txns"].fillna(0).astype(int)
all_otg_df["sub_threshold"] = all_otg_df["lifetime_txns"] < FUNCTIONAL_THRESHOLD
all_otg_df["never_traded"] = all_otg_df["lifetime_txns"] == 0

sub_threshold_otgs = set(all_otg_df[all_otg_df["sub_threshold"]]["Offset Trading Group"])
never_traded_otgs = set(all_otg_df[all_otg_df["never_traded"]]["Offset Trading Group"])

print(SEP)
print("TREND ANALYSIS: SUB-THRESHOLD OTG TRANSACTION FREQUENCY")
print(SEP)
print(f"Total supply OTGs:           {len(supply_otgs)}")
print(f"Sub-threshold (<5 txns):     {len(sub_threshold_otgs)}")
print(f"  of which never traded:     {len(never_traded_otgs)}")
print(f"  of which 1-4 txns:         {len(sub_threshold_otgs) - len(never_traded_otgs)}")
print()

# ---------------------------------------------------------------------------
# Monthly transaction counts for sub-threshold OTGs
# ---------------------------------------------------------------------------
eco_sub = eco[eco["Offset Trading Group"].isin(sub_threshold_otgs)].copy()
eco_sub["year_month"] = eco_sub["date"].dt.to_period("M")

# Full date range of the register
date_min = eco["date"].min().to_period("M")
date_max = eco["date"].max().to_period("M")
all_months = pd.period_range(date_min, date_max, freq="M")

monthly_sub = (
    eco_sub.groupby("year_month").size()
    .reindex(all_months, fill_value=0)
    .reset_index()
)
monthly_sub.columns = ["year_month", "txn_count"]
monthly_sub["month_idx"] = range(len(monthly_sub))

print(f"Observation window: {date_min} to {date_max} ({len(all_months)} months)")
print()

# ---------------------------------------------------------------------------
# Mann-Kendall test (non-parametric trend)
# ---------------------------------------------------------------------------
def mann_kendall(x):
    """Returns (tau, p-value, S, variance_S)."""
    n = len(x)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    # Variance under H0 (no ties adjustment for simplicity)
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    return tau, p, s, var_s

counts = monthly_sub["txn_count"].values
tau, p_mk, s_mk, _ = mann_kendall(counts)

# OLS slope
slope, intercept, r, p_ols, se = sp_stats.linregress(monthly_sub["month_idx"], counts)

print(SUB)
print("ALL SUB-THRESHOLD OTGs (0-4 lifetime transactions)")
print(SUB)
print(f"Monthly txn counts: mean={counts.mean():.2f}, median={np.median(counts):.1f}, "
      f"max={counts.max()}")
print()
print("Mann-Kendall test (non-parametric trend):")
print(f"  S statistic:  {s_mk}")
print(f"  Kendall tau:  {tau:.4f}")
print(f"  p-value:      {p_mk:.4f}")
trend_dir = "UPWARD" if tau > 0 else "DOWNWARD"
sig = "SIGNIFICANT" if p_mk < 0.05 else "NOT SIGNIFICANT"
print(f"  Result:       {trend_dir} trend — {sig} (alpha=0.05)")
print()
print("OLS linear regression (effect size):")
print(f"  Slope:        {slope:.4f} txns/month")
print(f"  R²:           {r**2:.4f}")
print(f"  p-value:      {p_ols:.4f}")
print()

# ---------------------------------------------------------------------------
# Same analysis for OTGs with 1-4 txns (excluding never-traded)
# ---------------------------------------------------------------------------
otgs_1_to_4 = sub_threshold_otgs - never_traded_otgs
eco_1to4 = eco[eco["Offset Trading Group"].isin(otgs_1_to_4)].copy()
eco_1to4["year_month"] = eco_1to4["date"].dt.to_period("M")

monthly_1to4 = (
    eco_1to4.groupby("year_month").size()
    .reindex(all_months, fill_value=0)
    .reset_index()
)
monthly_1to4.columns = ["year_month", "txn_count"]
monthly_1to4["month_idx"] = range(len(monthly_1to4))

counts_1to4 = monthly_1to4["txn_count"].values
tau2, p_mk2, s_mk2, _ = mann_kendall(counts_1to4)
slope2, _, r2, p_ols2, _ = sp_stats.linregress(monthly_1to4["month_idx"], counts_1to4)

print(SUB)
print("OTGs WITH 1-4 LIFETIME TRANSACTIONS (excluding never-traded)")
print(SUB)
print(f"Monthly txn counts: mean={counts_1to4.mean():.2f}, "
      f"median={np.median(counts_1to4):.1f}, max={counts_1to4.max()}")
print()
print("Mann-Kendall test:")
print(f"  S statistic:  {s_mk2}")
print(f"  Kendall tau:  {tau2:.4f}")
print(f"  p-value:      {p_mk2:.4f}")
trend_dir2 = "UPWARD" if tau2 > 0 else "DOWNWARD"
sig2 = "SIGNIFICANT" if p_mk2 < 0.05 else "NOT SIGNIFICANT"
print(f"  Result:       {trend_dir2} trend — {sig2} (alpha=0.05)")
print()
print("OLS linear regression (effect size):")
print(f"  Slope:        {slope2:.4f} txns/month")
print(f"  R²:           {r2**2:.4f}")
print(f"  p-value:      {p_ols2:.4f}")
print()

# ---------------------------------------------------------------------------
# Per-OTG trend: how many individual sub-threshold OTGs show upward trends?
# ---------------------------------------------------------------------------
print(SUB)
print("PER-OTG ANALYSIS (OTGs with >=2 transactions to allow trend test)")
print(SUB)

results = []
for otg, grp in eco[eco["Offset Trading Group"].isin(sub_threshold_otgs - never_traded_otgs)].groupby("Offset Trading Group"):
    grp = grp.copy()
    grp["year_month"] = grp["date"].dt.to_period("M")
    monthly = (
        grp.groupby("year_month").size()
        .reindex(all_months, fill_value=0)
        .reset_index()
    )
    monthly.columns = ["year_month", "txn_count"]
    monthly["month_idx"] = range(len(monthly))
    c = monthly["txn_count"].values
    if c.sum() < 2:
        continue
    tau_i, p_i, _, _ = mann_kendall(c)
    results.append({"otg": otg, "lifetime_txns": int(otg_counts[otg_counts["Offset Trading Group"]==otg]["lifetime_txns"].values[0]),
                    "tau": tau_i, "p": p_i,
                    "upward": tau_i > 0, "sig": p_i < 0.05})

df_res = pd.DataFrame(results)
if len(df_res) > 0:
    n_upward = df_res["upward"].sum()
    n_sig_upward = (df_res["upward"] & df_res["sig"]).sum()
    n_downward = (~df_res["upward"]).sum()
    n_sig_downward = (~df_res["upward"] & df_res["sig"]).sum()
    print(f"OTGs tested (>=2 txns):  {len(df_res)}")
    print(f"  Upward trend:          {n_upward} ({n_upward/len(df_res):.0%}), "
          f"of which significant: {n_sig_upward}")
    print(f"  Downward/flat trend:   {n_downward} ({n_downward/len(df_res):.0%}), "
          f"of which significant: {n_sig_downward}")
    print()
    if n_sig_upward > 0:
        print("OTGs with SIGNIFICANT upward trends:")
        for _, row in df_res[df_res["upward"] & df_res["sig"]].iterrows():
            print(f"  {row['otg'][:60]:<60} tau={row['tau']:.3f} p={row['p']:.3f} "
                  f"(lifetime txns: {row['lifetime_txns']})")

print()
print(SEP)
print("SUMMARY FOR MANUSCRIPT CLAIM")
print(SEP)
print(f"Claim: 'six years of data show no upward trend in transaction frequency'")
print()
agg_result = "SUPPORTED" if (p_mk >= 0.05 or tau <= 0) else "NOT SUPPORTED"
print(f"Aggregate Mann-Kendall (all sub-threshold): tau={tau:.4f}, p={p_mk:.4f} -> {agg_result}")
agg_result2 = "SUPPORTED" if (p_mk2 >= 0.05 or tau2 <= 0) else "NOT SUPPORTED"
print(f"Aggregate Mann-Kendall (1-4 txns only):     tau={tau2:.4f}, p={p_mk2:.4f} -> {agg_result2}")
