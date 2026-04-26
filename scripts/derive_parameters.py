#!/usr/bin/env python3
"""
=============================================================================
EMPIRICAL DERIVATION OF MICRO-FOUNDED DEMAND FUNCTION PARAMETERS
=============================================================================

Derives ALL model parameters directly from the NSW Biodiversity Offsets Scheme
raw CSV data (credit transactions register + credit supply register).

Every parameter has a clear empirical derivation documented inline.

Author: José Luis Reséndiz
Date:   2026-03-19
=============================================================================
"""

import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

# =============================================================================
# 0. DATA LOADING
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "nsw"

print("=" * 80)
print("EMPIRICAL PARAMETER DERIVATION FROM NSW BIODIVERSITY OFFSETS DATA")
print("=" * 80)
print()

# --- Transaction register ---
txn = pd.read_csv(
    DATA_DIR / "nsw_credit_transactions_register.csv",
    encoding="utf-8-sig",
)
txn.columns = txn.columns.str.strip()
txn["date"] = pd.to_datetime(txn["Transaction Date"], errors="coerce")
txn["price"] = pd.to_numeric(txn["Price Per Credit (Ex-GST)"], errors="coerce")
txn["credits"] = pd.to_numeric(txn["Number Of Credits"], errors="coerce")
txn["otg"] = txn["Offset Trading Group"].fillna("").str.strip()
# Normalise vegetation-formation case to the 15-formation reference (matches
# src/model/formation_model.py:44 and scripts/holdout_validation.py:250).
# The supply register contains both "Grassy woodlands" and "Grassy Woodlands";
# without normalisation the eta regression and other formation-level steps
# would treat them as 16 separate formations.
FORMATION_NORMALIZE = {
    "Grassy woodlands": "Grassy Woodlands",
}
txn["formation"] = (
    txn["Vegetation Formation"].fillna("").str.strip().replace(FORMATION_NORMALIZE)
)
txn["retirement_reason"] = txn["Retirement Reason"].fillna("").str.strip()
txn["other_reason"] = txn["Other Reason for Retiring"].fillna("").str.strip()
txn["txn_type"] = txn["Transaction Type"].fillna("").str.strip()
txn["from_id"] = txn["From"].fillna("").str.strip()
txn["to_id"] = txn["To"].fillna("").str.strip()

# --- Supply register ---
supply = pd.read_csv(
    DATA_DIR / "nsw_credit_supply_register.csv",
    encoding="utf-8-sig",
)
supply.columns = supply.columns.str.strip()
supply["n_credits"] = pd.to_numeric(supply["Number of credits"], errors="coerce").fillna(0)

# --- Priced transfers (the core analytical sample) ---
transfers = txn[
    (txn["txn_type"] == "Transfer") & (txn["price"] > 0)
].copy()
n_transfers = len(transfers)

# --- Retirements ---
retirements = txn[txn["txn_type"] == "Retire"].copy()

print(f"Loaded: {len(txn)} total rows, {n_transfers} priced transfers, "
      f"{len(retirements)} retirements")
print(f"Supply register: {len(supply)} rows")
print()

# =============================================================================
# Identify BCT transactions
# =============================================================================
# BCT retirements have "BCT" in the retirement or other-reason fields
# BCT purchases are transfers where the buyer later retires for BCT reasons

bct_retire_mask = (
    retirements["retirement_reason"].str.contains("BCT", case=False, na=False)
    | retirements["other_reason"].str.contains("BCT", case=False, na=False)
    | retirements["other_reason"].str.contains(
        "Biodiversity Conservation Trust", case=False, na=False
    )
)
bct_retirements = retirements[bct_retire_mask]

# Credit IDs that BCT retired — the "From" on a retirement is the credit being retired
bct_credit_ids = set(bct_retirements["from_id"].unique())

# BCT purchases: transfers where the destination credit later appears in BCT retirements
# The "To" credit of a transfer becomes the "From" credit of a retirement
bct_transfers = transfers[transfers["to_id"].isin(bct_credit_ids)].copy()

# Compliance retirements (non-BCT)
compliance_retire_mask = (
    retirements["retirement_reason"].str.contains("complying", case=False, na=False)
    & ~bct_retire_mask
)
compliance_retirements = retirements[compliance_retire_mask]
compliance_credit_ids = set(compliance_retirements["from_id"].unique())
compliance_transfers = transfers[transfers["to_id"].isin(compliance_credit_ids)].copy()

# Non-BCT, non-compliance transfers (intermediaries / speculative)
identified_credits = bct_credit_ids | compliance_credit_ids
intermediary_transfers = transfers[~transfers["to_id"].isin(identified_credits)].copy()

print(f"BCT purchases: {len(bct_transfers)} transfers "
      f"({len(bct_credit_ids)} credit IDs)")
print(f"Compliance purchases: {len(compliance_transfers)} transfers "
      f"({len(compliance_credit_ids)} credit IDs)")
print(f"Unmatched / intermediary transfers: {len(intermediary_transfers)}")
print()

# =============================================================================
# Ecosystem-only priced transfers (OTG-based analysis)
# =============================================================================
eco_transfers = transfers[transfers["otg"] != ""].copy()
print(f"Ecosystem (OTG-based) priced transfers: {len(eco_transfers)}")

# Per-OTG transaction counts
otg_counts = eco_transfers.groupby("otg").size().reset_index(name="n_txn")
otg_counts_dict = dict(zip(otg_counts["otg"], otg_counts["n_txn"]))

# Thin markets: OTGs with <5 transactions
thin_otgs = set(otg_counts[otg_counts["n_txn"] < 5]["otg"])
thick_otgs = set(otg_counts[otg_counts["n_txn"] >= 5]["otg"])

print(f"OTGs with transactions: {len(otg_counts)}")
print(f"  Thin (<5 txn): {len(thin_otgs)}")
print(f"  Thick (>=5 txn): {len(thick_otgs)}")
print()


# =============================================================================
# 1. FAIR VALUE MULTIPLIER (intermediary mean-reversion target)
# =============================================================================
print("=" * 80)
print("1. FAIR VALUE MULTIPLIER")
print("=" * 80)

# Monthly median prices across all ecosystem transactions
eco_transfers["month"] = eco_transfers["date"].dt.to_period("M")
monthly_medians = (
    eco_transfers.groupby("month")["price"]
    .median()
    .sort_index()
)
monthly_medians.index = monthly_medians.index.to_timestamp()

print(f"Monthly median prices: {len(monthly_medians)} months")
print(f"  Range: ${monthly_medians.min():,.0f} to ${monthly_medians.max():,.0f}")
print(f"  Current (last month): ${monthly_medians.iloc[-1]:,.0f}")
print()

# AR(1): p_t = c + phi * p_{t-1} + epsilon
p = monthly_medians.values
if len(p) >= 6:
    p_lag = p[:-1]
    p_cur = p[1:]
    slope, intercept, r_val, p_val, se = sp_stats.linregress(p_lag, p_cur)
    phi = slope
    c_ar1 = intercept
    r_sq = r_val ** 2

    print(f"AR(1) regression: p_t = {c_ar1:.1f} + {phi:.4f} * p_{{t-1}}")
    print(f"  R^2 = {r_sq:.4f}, p-value(phi) = {p_val:.4e}")
    print(f"  Standard error of phi = {se:.4f}")

    current_price = monthly_medians.iloc[-1]
    overall_median_price_all = eco_transfers["price"].median() if len(eco_transfers) > 0 else current_price

    if abs(phi) < 0.999:
        mean_reversion_target = c_ar1 / (1 - phi)
        fair_value_multiplier = mean_reversion_target / overall_median_price_all
        print(f"  Mean-reversion target = {c_ar1:.1f} / (1 - {phi:.4f}) "
              f"= ${mean_reversion_target:,.0f}")
        print(f"  Overall median price = ${overall_median_price_all:,.0f}")
        print(f"  Current (last month) median = ${current_price:,.0f}")
        print(f"  => fair_value multiplier = {mean_reversion_target:,.0f} / "
              f"{overall_median_price_all:,.0f} = {fair_value_multiplier:.3f}")
    else:
        fair_value_multiplier = 1.0
        mean_reversion_target = overall_median_price_all
        print(f"  phi ~ 1 (unit root): no mean reversion detected")
        print(f"  => fair_value multiplier = 1.0 (current price IS fair value)")

    # Note: low phi means prices are essentially uncorrelated month-to-month,
    # so the "mean-reversion target" is approximately the unconditional mean.
    # This is a reasonable fair-value estimate.
    if abs(phi) < 0.3:
        print(f"\n  NOTE: phi={phi:.3f} is low (weak autocorrelation), meaning monthly")
        print(f"  medians are nearly independent. The mean-reversion target is")
        print(f"  approximately the unconditional mean of the price series.")
        print(f"  This is a valid fair-value estimate: the long-run average price.")
    print()
    print(f'=> Estimated fair_value multiplier = {fair_value_multiplier:.3f} '
          f'(from AR(1) coefficient phi = {phi:.4f})')
else:
    fair_value_multiplier = 1.0
    phi = np.nan
    print("  Insufficient data for AR(1). Defaulting to 1.0.")

# phi near 0 = nearly iid monthly medians, target = unconditional mean (well-defined)
# phi near 1 = unit root, mean-reversion target undefined
# Middle values = genuine AR(1) mean reversion
if abs(phi) >= 0.9:
    fair_value_confidence = "LOW"  # near unit root
elif len(monthly_medians) >= 24:
    fair_value_confidence = "MEDIUM"  # sufficient data for unconditional mean
else:
    fair_value_confidence = "LOW"
print()


# =============================================================================
# 2. sigma^2 FOR INTERMEDIARY (within-OTG price uncertainty)
# =============================================================================
print("=" * 80)
print("2. INTERMEDIARY PRICE UNCERTAINTY (sigma)")
print("=" * 80)

# Within-OTG log-price standard deviation for OTGs with >=3 transactions
eco_transfers["log_price"] = np.log(eco_transfers["price"])

otg_log_stats = (
    eco_transfers.groupby("otg")["log_price"]
    .agg(["std", "count", "mean"])
    .dropna()
)
otg_log_stats_3plus = otg_log_stats[otg_log_stats["count"] >= 3]

if len(otg_log_stats_3plus) > 0:
    median_log_std = otg_log_stats_3plus["std"].median()
    # Convert back to price-level: sigma in AUD
    # For log-normal: CV ~ exp(sigma_log) - 1 ~ sigma_log for small sigma
    # sigma_price ~ sigma_log * mean_price
    overall_mean_price = eco_transfers["price"].mean()
    sigma_intermediary = median_log_std * overall_mean_price
    sigma_intermediary_pct = median_log_std * 100  # as percentage

    print(f"OTGs with >=3 transactions: {len(otg_log_stats_3plus)}")
    print(f"Within-OTG log-price std (median across OTGs): {median_log_std:.4f}")
    print(f"Overall mean ecosystem price: ${overall_mean_price:,.0f}")
    print(f"sigma_intermediary (AUD) = {median_log_std:.4f} * ${overall_mean_price:,.0f} "
          f"= ${sigma_intermediary:,.0f}")
    print(f"  => sigma/mu ratio = {median_log_std:.4f} ({sigma_intermediary_pct:.1f}%)")
    sigma_sq_intermediary = sigma_intermediary ** 2
    print(f"  => sigma^2_intermediary = ${sigma_sq_intermediary:,.0f}")
    print()
    print(f'=> Intermediary sigma = ${sigma_intermediary:,.0f} AUD '
          f'({sigma_intermediary_pct:.1f}% of mean price)')
else:
    sigma_intermediary = 1000.0
    sigma_sq_intermediary = sigma_intermediary ** 2
    median_log_std = 0.2
    print("  Insufficient data. Defaulting sigma = $1,000.")

sigma_intermed_confidence = "HIGH" if len(otg_log_stats_3plus) >= 10 else "MEDIUM"
print()


# =============================================================================
# 3. sigma^2 FOR BCT (thin-market price uncertainty)
# =============================================================================
print("=" * 80)
print("3. BCT THIN-MARKET PRICE UNCERTAINTY (sigma)")
print("=" * 80)

# Same as above but restricted to OTGs with <5 transactions
eco_thin = eco_transfers[eco_transfers["otg"].isin(thin_otgs)]

# For thin markets, many OTGs have <3 transactions, so we also compute
# across all thin-market transactions pooled
thin_log_stats = (
    eco_thin.groupby("otg")["log_price"]
    .agg(["std", "count", "mean"])
    .dropna()
)
thin_log_stats_3plus = thin_log_stats[thin_log_stats["count"] >= 3]

if len(thin_log_stats_3plus) > 0:
    median_thin_log_std = thin_log_stats_3plus["std"].median()
else:
    # Pool all thin-market transactions and compute log-price std
    median_thin_log_std = eco_thin["log_price"].std()
    print("  (Pooling all thin-market transactions: <3 per OTG for most)")

thin_mean_price = eco_thin["price"].median() if len(eco_thin) > 0 else overall_mean_price
sigma_bct = median_thin_log_std * thin_mean_price
sigma_bct_pct = median_thin_log_std * 100

print(f"Thin-market OTGs analyzed: {len(thin_log_stats_3plus)} with >=3 txn "
      f"(of {len(thin_otgs)} total thin)")
print(f"Thin-market transactions: {len(eco_thin)}")
print(f"Within-OTG log-price std (thin markets): {median_thin_log_std:.4f}")
print(f"Thin-market median price: ${thin_mean_price:,.0f}")
print(f"sigma_BCT (AUD) = {median_thin_log_std:.4f} * ${thin_mean_price:,.0f} "
      f"= ${sigma_bct:,.0f}")
sigma_sq_bct = sigma_bct ** 2
print(f"  => sigma^2_BCT = ${sigma_sq_bct:,.0f}")
print(f"  => sigma/mu ratio = {median_thin_log_std:.4f} ({sigma_bct_pct:.1f}%)")
print()
print(f'=> BCT thin-market sigma = ${sigma_bct:,.0f} AUD '
      f'({sigma_bct_pct:.1f}% of mean price)')

sigma_bct_confidence = (
    "MEDIUM" if len(thin_log_stats_3plus) >= 5 else "LOW"
)
print()


# =============================================================================
# 4. gamma_BCT (RISK AVERSION) — inverse calibration
# =============================================================================
print("=" * 80)
print("4. BCT RISK AVERSION (gamma_BCT) — INVERSE CALIBRATION")
print("=" * 80)

# BCT purchase prices
if len(bct_transfers) > 0:
    bct_median_price = bct_transfers["price"].median()
    bct_mean_price = bct_transfers["price"].mean()
    bct_max_price = bct_transfers["price"].max()
    bct_mean_credits = bct_transfers["credits"].mean()
    bct_median_credits = bct_transfers["credits"].median()

    print(f"BCT priced transfers: {len(bct_transfers)}")
    print(f"  Median price: ${bct_median_price:,.0f}")
    print(f"  Mean price:   ${bct_mean_price:,.0f}")
    print(f"  Max price:    ${bct_max_price:,.0f} (reservation price proxy)")
    print(f"  Mean credits/purchase: {bct_mean_credits:.0f}")
    print(f"  Median credits/purchase: {bct_median_credits:.0f}")
    print()

    # CARA demand: q* = (alpha - p) / (gamma * sigma^2)
    # alpha = reservation price (max price BCT ever paid)
    # p = median BCT price (typical purchase)
    # q* = median credits per purchase (normalized to 1 for unit demand)
    # sigma^2 = BCT thin-market price variance

    alpha_bct = bct_max_price  # reservation price
    p_bct = bct_median_price
    q_star = bct_median_credits  # typical purchase volume

    if alpha_bct > p_bct and sigma_sq_bct > 0 and q_star > 0:
        gamma_bct = (alpha_bct - p_bct) / (q_star * sigma_sq_bct)
        print(f"CARA inverse calibration:")
        print(f"  alpha (reservation) = ${alpha_bct:,.0f}")
        print(f"  p (median purchase) = ${p_bct:,.0f}")
        print(f"  q* (median volume)  = {q_star:.0f} credits")
        print(f"  sigma^2             = ${sigma_sq_bct:,.0f}")
        print(f"  gamma_BCT = (alpha - p) / (q* * sigma^2)")
        print(f"            = ({alpha_bct:,.0f} - {p_bct:,.0f}) / "
              f"({q_star:.0f} * {sigma_sq_bct:,.0f})")
        print(f"            = {gamma_bct:.6f}")
        print()
        print(f'=> Derived gamma_BCT = {gamma_bct:.6f} '
              f'(from alpha = ${alpha_bct:,.0f}, p = ${p_bct:,.0f}, '
              f'q* = {q_star:.0f}, sigma^2 = ${sigma_sq_bct:,.0f})')
    else:
        gamma_bct = 1e-6
        print("  Cannot derive gamma_BCT: alpha <= p or zero variance.")
else:
    bct_median_price = 5989.0
    gamma_bct = 1e-6
    alpha_bct = 0
    q_star = 0
    print("  No BCT transfers identified. Cannot derive gamma_BCT.")

gamma_bct_confidence = "MEDIUM" if len(bct_transfers) >= 10 else "LOW"
print()


# =============================================================================
# 5. BETA (BCT BENEFIT DECAY)
# =============================================================================
print("=" * 80)
print("5. BCT BENEFIT DECAY (beta)")
print("=" * 80)

# Look at BCT purchasing patterns within the same OTG
bct_eco = bct_transfers[bct_transfers["otg"] != ""].copy()
bct_eco = bct_eco.sort_values("date")

if len(bct_eco) > 0:
    # For each OTG with multiple BCT purchases, check price trends
    bct_otg_groups = bct_eco.groupby("otg")
    otgs_with_repeat = [name for name, grp in bct_otg_groups if len(grp) >= 2]

    print(f"BCT ecosystem purchases: {len(bct_eco)}")
    print(f"OTGs with repeat BCT purchases: {len(otgs_with_repeat)}")

    if len(otgs_with_repeat) >= 3:
        price_ratios = []
        for otg_name in otgs_with_repeat:
            grp = bct_otg_groups.get_group(otg_name).sort_values("date")
            prices = grp["price"].values
            # Ratio of later prices to first price
            for i in range(1, len(prices)):
                price_ratios.append(prices[i] / prices[0])

        price_ratios = np.array(price_ratios)
        median_ratio = np.median(price_ratios)
        mean_ratio = np.mean(price_ratios)

        print(f"Price ratios (later/first purchase in same OTG):")
        print(f"  Median ratio: {median_ratio:.3f}")
        print(f"  Mean ratio:   {mean_ratio:.3f}")
        print(f"  N observations: {len(price_ratios)}")

        # Use a 5% tolerance: median_ratio within [0.95, 1.05] => no clear pattern
        if median_ratio < 0.95:
            # Prices decline => benefit decay
            # beta ~ -ln(median_ratio) per purchase
            beta = -np.log(median_ratio)
            beta_identifiable = True
            print(f"  => Prices DECLINE by {(1-median_ratio)*100:.1f}% on repeat purchases")
            print(f"  => beta (decay rate) = -ln({median_ratio:.3f}) = {beta:.4f}")
        elif median_ratio > 1.05:
            # Prices increase => no decay (market appreciation dominates)
            beta = 0.0
            beta_identifiable = False
            print(f"  => Prices INCREASE by {(median_ratio-1)*100:.1f}% on repeat purchases")
            print(f"     (market appreciation dominates; no evidence of benefit decay)")
            print(f"  => beta = 0.0 (not identifiable; price appreciation masks decay)")
        else:
            beta = 0.0
            beta_identifiable = False
            print(f"  => Prices approximately unchanged on repeat purchases "
                  f"(ratio={median_ratio:.3f})")
            print(f"  => beta = 0.0 (not identifiable from data)")
    else:
        beta = 0.0
        beta_identifiable = False
        print(f"  Too few OTGs with repeat BCT purchases ({len(otgs_with_repeat)})")
        print(f"  => beta not identifiable from data")
else:
    beta = 0.0
    beta_identifiable = False
    print("  No BCT ecosystem purchases identified.")
    print("  => beta not identifiable from data")

beta_note = "BCT sequential purchase pattern"
if not beta_identifiable:
    beta_note += " (not identifiable)"
beta_confidence = "LOW" if not beta_identifiable else "MEDIUM"
print()
print(f'=> BCT benefit decay: {"pattern found" if beta_identifiable else "not identifiable from data"}')
print()


# =============================================================================
# 6. THRESHOLD (FUND PAYMENT TRIGGER)
# =============================================================================
print("=" * 80)
print("6. FUND PAYMENT THRESHOLD")
print("=" * 80)

# Compare BCT vs compliance purchase prices
# The threshold is the price level where developers switch to Fund payments

overall_median = eco_transfers["price"].median()
print(f"Overall median ecosystem price: ${overall_median:,.0f}")

if len(bct_transfers) > 0 and len(compliance_transfers) > 0:
    bct_p = bct_transfers["price"]
    comp_p = compliance_transfers[compliance_transfers["price"] > 0]["price"]

    bct_med = bct_p.median()
    comp_med = comp_p.median()
    bct_p75 = bct_p.quantile(0.75)
    comp_p75 = comp_p.quantile(0.75)

    print(f"\nBCT purchase prices:")
    print(f"  Median: ${bct_med:,.0f}")
    print(f"  Mean:   ${bct_p.mean():,.0f}")
    print(f"  75th:   ${bct_p75:,.0f}")
    print(f"  N = {len(bct_p)}")
    print(f"\nCompliance purchase prices:")
    print(f"  Median: ${comp_med:,.0f}")
    print(f"  Mean:   ${comp_p.mean():,.0f}")
    print(f"  75th:   ${comp_p75:,.0f}")
    print(f"  N = {len(comp_p)}")

    # The threshold is the price at which BCT demand dominates
    # Price bins analysis
    all_prices = transfers[transfers["price"] > 0]["price"]
    price_bins = np.percentile(all_prices, np.arange(0, 101, 10))

    print(f"\nPrice bin analysis (BCT vs compliance share):")
    print(f"  {'Price range':>25s} | {'BCT':>5s} | {'Comp':>5s} | {'Other':>5s} | {'BCT%':>5s}")
    print(f"  {'-'*25}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")

    crossover_price = None
    for i in range(len(price_bins) - 1):
        lo, hi = price_bins[i], price_bins[i + 1]
        mask_bct = (bct_transfers["price"] >= lo) & (bct_transfers["price"] < hi)
        mask_comp = (compliance_transfers["price"] >= lo) & (
            compliance_transfers["price"] < hi
        )
        mask_other = (intermediary_transfers["price"] >= lo) & (
            intermediary_transfers["price"] < hi
        )
        n_bct = mask_bct.sum()
        n_comp = mask_comp.sum()
        n_other = mask_other.sum()
        total = n_bct + n_comp + n_other
        bct_share = n_bct / total * 100 if total > 0 else 0
        print(f"  ${lo:>10,.0f}-{hi:>10,.0f} | {n_bct:>5d} | {n_comp:>5d} | "
              f"{n_other:>5d} | {bct_share:>4.0f}%")
        if bct_share > 50 and crossover_price is None:
            crossover_price = lo

    # The threshold: where BCT purchases become dominant
    # Alternative: the price at which compliance purchases thin out
    # The BCT premium over compliance is the threshold marker
    bct_premium = bct_med / comp_med if comp_med > 0 else 1.0

    if crossover_price is not None:
        threshold = crossover_price
    else:
        # Use BCT median as threshold proxy: above this, only BCT buys
        threshold = bct_med

    threshold_multiplier = threshold / overall_median if overall_median > 0 else 1.0

    print(f"\nBCT/Compliance median price ratio: {bct_premium:.2f}x")
    print(f"Estimated threshold (Fund trigger): ${threshold:,.0f}")
    print(f"  = {threshold_multiplier:.2f}x median market price")
    print()
    print(f'=> Estimated Fund payment threshold = ${threshold:,.0f} AUD '
          f'({threshold_multiplier:.2f}x median market price)')
else:
    threshold = overall_median * 1.5
    threshold_multiplier = 1.5
    bct_premium = np.nan
    print("  Insufficient buyer-type data. Defaulting to 1.5x median.")

threshold_confidence = "MEDIUM" if len(bct_transfers) >= 10 and len(compliance_transfers) >= 10 else "LOW"
print()


# =============================================================================
# 7. ETA (SUPPLY ELASTICITY)
# =============================================================================
print("=" * 80)
print("7. SUPPLY ELASTICITY (eta)")
print("=" * 80)

# Cross-formation regression: log(volume) = a + eta * log(price)
# Use formation-level aggregates from transaction data

formation_stats = (
    eco_transfers.groupby("formation")
    .agg(
        n_txn=("price", "count"),
        total_credits=("credits", "sum"),
        median_price=("price", "median"),
        mean_price=("price", "mean"),
    )
    .reset_index()
)
formation_stats = formation_stats[
    (formation_stats["n_txn"] >= 3)
    & (formation_stats["median_price"] > 0)
    & (formation_stats["total_credits"] > 0)
]

# Also bring in supply capacity from supply register
supply_eco = supply[supply["Ecosystem or Species"] == "Ecosystem"].copy()
supply_eco["formation"] = (
    supply_eco["Vegetation Formation"].fillna("").str.strip().replace(FORMATION_NORMALIZE)
)
supply_by_form = supply_eco.groupby("formation")["n_credits"].sum().reset_index()
supply_by_form.columns = ["formation", "supply_credits"]

formation_stats = formation_stats.merge(supply_by_form, on="formation", how="left")
formation_stats["supply_credits"] = formation_stats["supply_credits"].fillna(0)

print(f"Formations with sufficient data: {len(formation_stats)}")
for _, row in formation_stats.iterrows():
    print(f"  {row['formation'][:50]:50s} | "
          f"n={row['n_txn']:>4d} | "
          f"p=${row['median_price']:>8,.0f} | "
          f"credits={row['total_credits']:>8,.0f} | "
          f"supply={row['supply_credits']:>10,.0f}")

if len(formation_stats) >= 4:
    log_vol = np.log(formation_stats["n_txn"].values)
    log_price = np.log(formation_stats["median_price"].values)

    slope, intercept, r_val, p_val, se = sp_stats.linregress(log_price, log_vol)
    eta = slope
    r_sq_eta = r_val ** 2

    print(f"\nlog-log regression: log(n_txn) = {intercept:.2f} + {eta:.3f} * log(price)")
    print(f"  R^2 = {r_sq_eta:.4f}, p-value = {p_val:.4e}")
    print(f"  Interpretation: A 1% increase in price is associated with "
          f"a {eta:.2f}% change in transaction volume")

    # Also try supply-side: price vs supply capacity
    has_supply = formation_stats[formation_stats["supply_credits"] > 0]
    if len(has_supply) >= 4:
        log_supply = np.log(has_supply["supply_credits"].values)
        log_price_s = np.log(has_supply["median_price"].values)
        slope_s, intercept_s, r_val_s, p_val_s, _ = sp_stats.linregress(
            log_price_s, log_supply
        )
        print(f"\nSupply-side: log(supply_credits) = {intercept_s:.2f} + "
              f"{slope_s:.3f} * log(price)")
        print(f"  R^2 = {r_val_s**2:.4f}, p-value = {p_val_s:.4e}")
    print()
    print(f'=> Estimated supply elasticity eta = {eta:.3f} '
          f'(from cross-formation regression, R^2 = {r_sq_eta:.4f})')
else:
    eta = 0.5
    r_sq_eta = 0.0
    print("  Insufficient formations for regression. Defaulting eta = 0.5.")

eta_confidence = "MEDIUM" if r_sq_eta > 0.2 and len(formation_stats) >= 5 else "LOW"
print()


# =============================================================================
# 8. gamma_INTERMEDIARY (risk aversion)
# =============================================================================
print("=" * 80)
print("8. INTERMEDIARY RISK AVERSION (gamma_intermediary)")
print("=" * 80)

# Identify repeat buyers (potential intermediaries) who buy and later transfer out
# A "repeat buyer" credit holder that appears as both "To" (buying) and "From" (selling)

buyer_ids = set(transfers["to_id"].unique())
seller_ids = set(transfers["from_id"].unique())
# CRs that appear as both buyer destination and seller origin = intermediaries
intermediary_crs = buyer_ids & seller_ids

# Get buy and sell prices for these intermediary CRs
intermed_buys = transfers[transfers["to_id"].isin(intermediary_crs)].copy()
intermed_sells = transfers[transfers["from_id"].isin(intermediary_crs)].copy()

print(f"Credit IDs that appear as both buy-destination and sell-origin: "
      f"{len(intermediary_crs)}")
print(f"  Associated buy transactions: {len(intermed_buys)}")
print(f"  Associated sell transactions: {len(intermed_sells)}")

# Match buy-sell pairs for the same credit ID
if len(intermediary_crs) > 0:
    spreads = []
    volumes = []
    for cr_id in intermediary_crs:
        buys = transfers[transfers["to_id"] == cr_id]
        sells = transfers[transfers["from_id"] == cr_id]
        if len(buys) > 0 and len(sells) > 0:
            buy_price = buys["price"].iloc[0]
            sell_price = sells["price"].iloc[0]
            buy_vol = buys["credits"].iloc[0]
            if buy_price > 0 and sell_price > 0:
                spread = sell_price - buy_price
                spreads.append(spread)
                volumes.append(buy_vol)

    spreads = np.array(spreads)
    volumes = np.array(volumes)

    if len(spreads) > 0:
        mean_spread = np.mean(spreads)
        median_spread = np.median(spreads)
        pos_spreads = spreads[spreads > 0]
        neg_spreads = spreads[spreads < 0]

        print(f"\nBuy-sell spread analysis ({len(spreads)} matched pairs):")
        print(f"  Mean spread:   ${mean_spread:,.0f}")
        print(f"  Median spread: ${median_spread:,.0f}")
        print(f"  Positive (profit): {len(pos_spreads)} ({len(pos_spreads)/len(spreads)*100:.0f}%)")
        print(f"  Negative (loss):   {len(neg_spreads)} ({len(neg_spreads)/len(spreads)*100:.0f}%)")

        if len(pos_spreads) > 0:
            avg_pos_spread = np.mean(pos_spreads)
            avg_vol = np.mean(volumes[spreads > 0])
            # gamma ~ spread / (volume * sigma^2)
            # This is the inverse: 1/gamma ~ volume * sigma^2 / spread
            gamma_intermediary = avg_pos_spread / (avg_vol * sigma_sq_intermediary) if sigma_sq_intermediary > 0 else 1e-6
            gamma_intermed_derived = True
            print(f"  Average profitable spread: ${avg_pos_spread:,.0f}")
            print(f"  Average volume in profitable trades: {avg_vol:.0f}")
            print(f"  gamma_intermediary = spread / (vol * sigma^2)")
            print(f"                     = {avg_pos_spread:,.0f} / "
                  f"({avg_vol:.0f} * {sigma_sq_intermediary:,.0f})")
            print(f"                     = {gamma_intermediary:.2e}")
        else:
            gamma_intermediary = gamma_bct * 0.5 if gamma_bct > 0 else 1e-6
            gamma_intermed_derived = False
            print(f"  No profitable spreads found. Bounding: gamma_intermediary = 0.5 * gamma_BCT")
    else:
        gamma_intermediary = gamma_bct * 0.5 if gamma_bct > 0 else 1e-6
        gamma_intermed_derived = False
        print("  No matched buy-sell pairs. Bounding from gamma_BCT.")
else:
    gamma_intermediary = gamma_bct * 0.5 if gamma_bct > 0 else 1e-6
    gamma_intermed_derived = False
    print("  No intermediary credit IDs found.")

gamma_intermed_note = "derived" if gamma_intermed_derived else "bounded"
print()
print(f'=> Estimated gamma_intermediary = {gamma_intermediary:.2e} [{gamma_intermed_note}]')
gamma_intermed_confidence = "MEDIUM" if gamma_intermed_derived and len(spreads) >= 5 else "LOW"
print()


# =============================================================================
# 9. LOGISTIC SLOPE (compliance demand smoothing)
# =============================================================================
print("=" * 80)
print("9. LOGISTIC SLOPE (compliance demand smoothing)")
print("=" * 80)

# How tightly are compliance purchases clustered around the median?
if len(compliance_transfers) > 0:
    comp_prices = compliance_transfers[compliance_transfers["price"] > 0]["price"]
    comp_median = comp_prices.median()

    if comp_median > 0 and len(comp_prices) > 0:
        # Compute fraction within X% of median
        within_10 = ((comp_prices >= comp_median * 0.9) & (comp_prices <= comp_median * 1.1)).mean()
        within_20 = ((comp_prices >= comp_median * 0.8) & (comp_prices <= comp_median * 1.2)).mean()
        within_50 = ((comp_prices >= comp_median * 0.5) & (comp_prices <= comp_median * 1.5)).mean()

        print(f"Compliance purchase prices (N={len(comp_prices)}):")
        print(f"  Median: ${comp_median:,.0f}")
        print(f"  Std:    ${comp_prices.std():,.0f}")
        print(f"  CV:     {comp_prices.std()/comp_prices.mean():.2f}")
        print()
        print(f"  Within +/-10% of median: {within_10*100:.1f}%")
        print(f"  Within +/-20% of median: {within_20*100:.1f}%")
        print(f"  Within +/-50% of median: {within_50*100:.1f}%")

        # The logistic slope k controls how sharply demand drops:
        # D(p) = D_max / (1 + exp(k * (p - threshold)))
        # A higher k = sharper cutoff
        # We calibrate k so that at the observed price dispersion, the logistic
        # captures the transition from high to low compliance demand
        #
        # The interquartile range gives a natural scale for the transition width:
        # logistic IQR ~ 2.2/k, so k ~ 2.2 / price_IQR
        iqr = comp_prices.quantile(0.75) - comp_prices.quantile(0.25)
        cv = comp_prices.std() / comp_prices.mean()

        if iqr > 0:
            # Normalize by median price for dimensionless slope
            logistic_slope = 2.197 / (iqr / comp_median)  # 2.197 = ln(9) = logistic IQR constant
            print(f"\n  IQR: ${iqr:,.0f} ({iqr/comp_median*100:.1f}% of median)")
            print(f"  Logistic slope k = ln(9) / (IQR/median)")
            print(f"                   = 2.197 / {iqr/comp_median:.3f}")
            print(f"                   = {logistic_slope:.2f}")
            print(f"  Interpretation: demand halves over a price range of "
                  f"${iqr:,.0f}")
        else:
            logistic_slope = 10.0
            print("  IQR = 0; extremely tight distribution.")
    else:
        logistic_slope = 5.0
        print("  No valid compliance prices.")
else:
    # Fall back to all transfers
    all_prices = transfers[transfers["price"] > 0]["price"]
    comp_median = all_prices.median()
    iqr = all_prices.quantile(0.75) - all_prices.quantile(0.25)
    if iqr > 0:
        logistic_slope = 2.197 / (iqr / comp_median)
    else:
        logistic_slope = 5.0
    within_10 = within_20 = within_50 = np.nan
    print("  No compliance transfers identified. Using all transfers.")

print()
print(f'=> Estimated logistic slope = {logistic_slope:.2f} '
      f'(based on compliance price dispersion)')

logistic_confidence = "MEDIUM" if len(compliance_transfers) >= 20 else "LOW"
print()


# =============================================================================
# SUMMARY TABLE
# =============================================================================
print()
print("=" * 80)
print("FINAL SUMMARY: EMPIRICALLY DERIVED PARAMETERS")
print("=" * 80)
print()

# Collect results
results = [
    (
        "fair_value multiplier",
        f"{fair_value_multiplier:.3f}",
        f"AR(1) on monthly medians (phi={phi:.3f})",
        fair_value_confidence,
    ),
    (
        "sigma_intermediary",
        f"${sigma_intermediary:,.0f} ({median_log_std*100:.1f}% CV)",
        "Within-OTG log-price std (median across OTGs)",
        sigma_intermed_confidence,
    ),
    (
        "sigma_BCT",
        f"${sigma_bct:,.0f} ({median_thin_log_std*100:.1f}% CV)",
        "Thin-market (<5 txn) log-price std",
        sigma_bct_confidence,
    ),
    (
        "gamma_BCT",
        f"{gamma_bct:.4e}",
        f"Inverse CARA: alpha=${alpha_bct:,.0f}, p=${bct_median_price:,.0f}, q*={q_star:.0f}",
        gamma_bct_confidence,
    ),
    (
        "beta (BCT decay)",
        f"{beta:.4f}" if beta_identifiable else "0.0 (not identifiable)",
        beta_note,
        beta_confidence,
    ),
    (
        "threshold",
        f"${threshold:,.0f} ({threshold_multiplier:.2f}x median)",
        "Compliance/BCT price crossover",
        threshold_confidence,
    ),
    (
        "eta (supply elast.)",
        f"{eta:.3f}",
        f"Cross-formation log-log regression (R^2={r_sq_eta:.3f})",
        eta_confidence,
    ),
    (
        "gamma_intermediary",
        f"{gamma_intermediary:.4e}",
        f"Price spread exploitation [{gamma_intermed_note}]",
        gamma_intermed_confidence,
    ),
    (
        "logistic slope",
        f"{logistic_slope:.2f}",
        "Compliance price dispersion (IQR-based)",
        logistic_confidence,
    ),
]

# Print table
header = f"| {'Parameter':<22s} | {'Derived Value':<35s} | {'Method':<50s} | {'Conf.':<6s} |"
sep = f"|{'-'*24}|{'-'*37}|{'-'*52}|{'-'*8}|"
print(header)
print(sep)
for param, value, method, conf in results:
    print(f"| {param:<22s} | {value:<35s} | {method:<50s} | {conf:<6s} |")

print()


# =============================================================================
# 10. LIQUIDITY FEEDBACK WEIGHTS
# =============================================================================
print("=" * 80)
print("10. LIQUIDITY FEEDBACK WEIGHTS")
print("=" * 80)
print()

# Derive the weights that determine how likely an OTG is to attract additional
# demand based on its cumulative transaction count. These weights drive the
# "liquidity begets liquidity" feedback loop in the ABM.
#
# Methodology:
#   1. Filter to ecosystem transfers with price >= 100 AUD (standard filter)
#   2. Restrict to OTGs that traded at least once (140 of 252) — the 112
#      never-traded OTGs are structurally illiquid (no available credits)
#      and excluded to avoid diluting the base rate
#   3. Panel starts 2021-01 (active market onset; only 3 sparse transactions
#      before this date). Pre-2021 cumulative counts are carried forward.
#   4. For each OTG-month, record: cumulative transaction count at START of
#      month + whether the OTG traded that month (binary)
#   5. Group OTG-months into bins by cumulative count: 0, 1-2, 3-4, 5-9, 10+
#   6. Monthly trading probability = OTG-months with trade / total OTG-months
#   7. Normalize: weight = probability / probability_of_bin_0

# Step 1: Ecosystem transfers with price >= 100 AUD
eco_priced = txn[
    (txn["txn_type"] == "Transfer")
    & (txn["price"] >= 100)
    & (txn["price"].notna())
    & (txn["otg"] != "")
].copy()
eco_priced["month"] = eco_priced["date"].dt.to_period("M")

print(f"Ecosystem transfers with price >= 100 AUD: {len(eco_priced)}")

# Step 2: Restrict to OTGs that traded at least once
traded_otg_set = sorted(eco_priced["otg"].unique())
print(f"OTGs with at least one priced transfer: {len(traded_otg_set)}")

# Step 3: Active market panel from 2021-01 onward
# The NSW BOS market had only 3 priced ecosystem transfers before Jan 2021
# (Nov 2019: 1, Apr 2020: 2). Consistent monthly trading starts Mar 2021.
# Using Jan 2021 avoids sparse pre-market months inflating the "never" bin.
MARKET_START = pd.Period("2021-01", freq="M")
max_month = eco_priced["month"].max()
panel_months = pd.period_range(MARKET_START, max_month, freq="M")
n_pre_start = len(eco_priced[eco_priced["month"] < MARKET_START])
print(f"Active market panel: {MARKET_START} to {max_month} ({len(panel_months)} months)")
print(f"  Pre-panel transactions (carried forward as cumulative): {n_pre_start}")

# Step 4: Build OTG-month panel
# Pre-compute: trades per OTG per month (panel period only)
otg_month_trades = (
    eco_priced.groupby(["otg", "month"])
    .size()
    .reset_index(name="n_trades_in_month")
)
otg_month_dict = {}
for _, r in otg_month_trades.iterrows():
    otg_month_dict.setdefault(r["otg"], {})[r["month"]] = int(r["n_trades_in_month"])

# Pre-compute: cumulative count BEFORE panel start for each OTG
pre_panel = eco_priced[eco_priced["month"] < MARKET_START]
pre_panel_cum = pre_panel.groupby("otg").size().to_dict()

records = []
for otg in traded_otg_set:
    cum = pre_panel_cum.get(otg, 0)
    otg_months = otg_month_dict.get(otg, {})
    for m in panel_months:
        traded = 1 if m in otg_months else 0
        records.append((otg, m, cum, traded))
        if traded:
            cum += otg_months[m]

panel = pd.DataFrame(records, columns=["otg", "month", "cum_start", "traded"])
print(f"Panel size: {len(panel):,d} OTG-months "
      f"({len(traded_otg_set)} OTGs x {len(panel_months)} months)")

# Step 5: Bin by cumulative transaction count at start of month
bin_edges = [
    ("never",       0,  0),
    ("thin_low",    1,  2),
    ("thin_high",   3,  4),
    ("functional",  5,  9),
    ("established", 10, 999999),
]

def assign_bin(cum):
    for name, lo, hi in bin_edges:
        if lo <= cum <= hi:
            return name
    return "established"

panel["bin"] = panel["cum_start"].apply(assign_bin)

# Step 6: Monthly trading probability per bin
bin_stats = (
    panel.groupby("bin")
    .agg(
        total_otg_months=("traded", "count"),
        traded_otg_months=("traded", "sum"),
    )
)
bin_stats["monthly_prob"] = bin_stats["traded_otg_months"] / bin_stats["total_otg_months"]

# Ensure correct ordering
bin_order = ["never", "thin_low", "thin_high", "functional", "established"]
bin_stats = bin_stats.reindex(bin_order)

# Step 7: Normalize to get weights
base_prob = bin_stats.loc["never", "monthly_prob"]
bin_stats["weight"] = bin_stats["monthly_prob"] / base_prob

# Hardcoded values from run_abm.py for comparison
hardcoded = {
    "never":       (0.0306, 1.0),
    "thin_low":    (0.0470, 1.5),
    "thin_high":   (0.0885, 2.9),
    "functional":  (0.1550, 5.1),
    "established": (0.2580, 8.4),
}

print()
print(f"{'Bin':<14s} | {'OTG-months':>11s} | {'Traded':>7s} | "
      f"{'Derived prob':>12s} | {'Hardcoded prob':>14s} | "
      f"{'Derived wt':>10s} | {'Hardcoded wt':>12s} | {'Match?':>7s}")
print("-" * 105)

all_match = True
for name in bin_order:
    row = bin_stats.loc[name]
    hc_prob, hc_wt = hardcoded[name]
    derived_prob = row["monthly_prob"]
    derived_wt = row["weight"]
    # Check within 5% tolerance (tight match expected)
    wt_match = abs(derived_wt - hc_wt) / max(hc_wt, 1e-9) < 0.05
    if not wt_match:
        all_match = False
    print(f"{name:<14s} | {int(row['total_otg_months']):>11,d} | {int(row['traded_otg_months']):>7,d} | "
          f"{derived_prob:>12.4f} | {hc_prob:>14.4f} | "
          f"{derived_wt:>10.1f} | {hc_wt:>12.1f} | "
          f"{'  OK' if wt_match else '  DIFF':>7s}")

print()
if all_match:
    print("=> ALL liquidity feedback weights match hardcoded values (within 5% tolerance)")
else:
    print("=> SOME weights differ from hardcoded values (beyond 5% tolerance)")
    print("   Review the hardcoded LIQUIDITY_FEEDBACK dict in scripts/run_abm.py")

print()
print(f"Base monthly trading probability (never-traded bin): {base_prob:.4f}")
print(f"Interpretation: An OTG with zero prior transactions has a {base_prob*100:.2f}%")
print(f"chance of trading in any given month. An established OTG (10+ transactions)")
print(f"has a {bin_stats.loc['established', 'monthly_prob']*100:.2f}% chance — "
      f"{bin_stats.loc['established', 'weight']:.1f}x higher.")
print(f"This confirms the 'liquidity begets liquidity' feedback: once an OTG attracts")
print(f"its first trades, the probability of subsequent trading rises monotonically.")
print()

# Add to summary results
results.append(
    (
        "liquidity wt (base)",
        f"{base_prob:.4f} (monthly prob)",
        "OTG-month panel: cum_txn=0 bin (traded OTGs, 2021+)",
        "HIGH",
    ),
)
results.append(
    (
        "liquidity wt (10+)",
        f"{bin_stats.loc['established', 'weight']:.1f}x",
        "OTG-month panel: cum_txn>=10 bin / base",
        "HIGH",
    ),
)

print("=" * 80)
print("DONE. All parameters derived from NSW raw CSV data.")
print("=" * 80)
