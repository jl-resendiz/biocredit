#!/usr/bin/env python3
"""
Formation-level agent-based model of the NSW Biodiversity Offsets Scheme.

Implements the production ABM:
  - 252 ecosystem credit types grouped by 15 vegetation formations across 40
    IBRA subregions
  - Four behavioural-strategy species: cost-minimising compliance,
    mean-reversion intermediation, thin-market procurement, cost-floor supply
  - Liquidity-feedback kernel from empirical transition probabilities
  - Single calibration target: P_variation matched to observed 35.4%
    Fund-routing rate

Key parameter derivations (from data/CODEBOOK.md and IPART 2024-25):
  - BCT budget per agent per month: lognormal(log(326,389), 0.5)
    Source: 45% of approximately $105M annual market value, allocated across
    12 BCT-type agents and 12 monthly procurement events.
  - Habitat-bank production cost: lognormal(7.62, 0.5)
    Source: log of Q1 (25th percentile) of 764 priced ecosystem credit
    prices = log(2,033) = 7.62.
  - Liquidity-feedback weights {1.0, 1.5, 2.9, 5.1, 8.4} for cumulative
    transaction buckets {0, 1-2, 3-4, 5-9, 10+}.

Run: 6 scenarios (baseline + 5 interventions) x 50 MC seeds x 72 timesteps.

Author: Jose Luis Resendiz
"""

import sys
import warnings
import time
import json
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats as sp_stats

# =============================================================================
# 0. DERIVE PARAMETERS FROM DATA
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "nsw"

print("=" * 110)
print("Formation-level ABM: NSW Biodiversity Offsets Scheme")
print("  Formation-level matching + partial like-for-like + liquidity feedback")
print("  P_variation calibrated to observed Fund-routing rate (35.4%)")
print("=" * 110)
print()


def derive_all_parameters():
    """Derive all micro-founded parameters from raw NSW data."""
    print("[0/10] Deriving parameters from NSW transaction data...")

    txn = pd.read_csv(
        DATA_DIR / "nsw_credit_transactions_register.csv", encoding="utf-8-sig"
    )
    txn.columns = txn.columns.str.strip()
    txn["date"] = pd.to_datetime(txn["Transaction Date"], errors="coerce")
    txn["price"] = pd.to_numeric(txn["Price Per Credit (Ex-GST)"], errors="coerce")
    txn["credits"] = pd.to_numeric(txn["Number Of Credits"], errors="coerce")
    txn["otg"] = txn["Offset Trading Group"].fillna("").str.strip()
    txn["formation"] = txn["Vegetation Formation"].fillna("").str.strip()
    txn["retirement_reason"] = txn["Retirement Reason"].fillna("").str.strip()
    txn["other_reason"] = txn["Other Reason for Retiring"].fillna("").str.strip()
    txn["txn_type"] = txn["Transaction Type"].fillna("").str.strip()
    txn["from_id"] = txn["From"].fillna("").str.strip()
    txn["to_id"] = txn["To"].fillna("").str.strip()

    # Standardised price filter (per data/CODEBOOK.md): priced transactions
    # require Transfer + price >= 100 AUD. The 100-AUD floor excludes ~10
    # near-zero rows that are administrative artefacts (cf. CODEBOOK.md
    # "Price Per Credit" row).
    transfers = txn[(txn["txn_type"] == "Transfer") & (txn["price"] >= 100)].copy()
    retirements = txn[txn["txn_type"] == "Retire"].copy()

    # Identify BCT transactions
    bct_retire_mask = (
        retirements["retirement_reason"].str.contains("BCT", case=False, na=False)
        | retirements["other_reason"].str.contains("BCT", case=False, na=False)
        | retirements["other_reason"].str.contains(
            "Biodiversity Conservation Trust", case=False, na=False
        )
    )
    bct_retirements = retirements[bct_retire_mask]
    bct_credit_ids = set(bct_retirements["from_id"].unique())
    bct_transfers = transfers[transfers["to_id"].isin(bct_credit_ids)].copy()

    compliance_retire_mask = (
        retirements["retirement_reason"].str.contains("complying", case=False, na=False)
        & ~bct_retire_mask
    )
    compliance_retirements = retirements[compliance_retire_mask]
    compliance_credit_ids = set(compliance_retirements["from_id"].unique())
    compliance_transfers = transfers[transfers["to_id"].isin(compliance_credit_ids)].copy()

    # Ecosystem transfers
    eco_transfers = transfers[transfers["otg"] != ""].copy()
    eco_transfers["log_price"] = np.log(eco_transfers["price"])

    otg_counts = eco_transfers.groupby("otg").size().reset_index(name="n_txn")
    thin_otgs = set(otg_counts[otg_counts["n_txn"] < 5]["otg"])

    overall_median = eco_transfers["price"].median()
    overall_mean = eco_transfers["price"].mean()

    # 1. Fair value multiplier (AR(1) on monthly medians)
    eco_transfers["month"] = eco_transfers["date"].dt.to_period("M")
    monthly_medians = eco_transfers.groupby("month")["price"].median().sort_index()
    monthly_medians.index = monthly_medians.index.to_timestamp()

    p = monthly_medians.values
    if len(p) >= 6:
        p_lag, p_cur = p[:-1], p[1:]
        slope, intercept, r_val, p_val, se = sp_stats.linregress(p_lag, p_cur)
        phi = slope
        if abs(phi) < 0.999:
            mean_rev_target = intercept / (1 - phi)
            fair_value_multiplier = mean_rev_target / overall_median
        else:
            fair_value_multiplier = 1.0
    else:
        fair_value_multiplier = 1.0

    # 2. Sigma intermediary
    otg_log_stats = eco_transfers.groupby("otg")["log_price"].agg(["std", "count"]).dropna()
    otg_log_stats_3plus = otg_log_stats[otg_log_stats["count"] >= 3]
    if len(otg_log_stats_3plus) > 0:
        median_log_std = otg_log_stats_3plus["std"].median()
        sigma_intermediary = median_log_std * overall_mean
    else:
        sigma_intermediary = 1000.0

    # 3. Sigma BCT (thin markets)
    eco_thin = eco_transfers[eco_transfers["otg"].isin(thin_otgs)]
    thin_log_stats = eco_thin.groupby("otg")["log_price"].agg(["std", "count"]).dropna()
    thin_log_stats_3plus = thin_log_stats[thin_log_stats["count"] >= 3]
    if len(thin_log_stats_3plus) > 0:
        median_thin_log_std = thin_log_stats_3plus["std"].median()
    else:
        median_thin_log_std = eco_thin["log_price"].std() if len(eco_thin) > 0 else 0.3
    thin_mean_price = eco_thin["price"].median() if len(eco_thin) > 0 else overall_mean
    sigma_bct = median_thin_log_std * thin_mean_price

    # 4. Gamma intermediary (buy-sell spread analysis)
    buyer_ids = set(transfers["to_id"].unique())
    seller_ids = set(transfers["from_id"].unique())
    intermediary_crs = buyer_ids & seller_ids

    gamma_intermediary = 2.27e-7  # fallback
    if len(intermediary_crs) > 0:
        spreads, volumes = [], []
        for cr_id in intermediary_crs:
            buys = transfers[transfers["to_id"] == cr_id]
            sells = transfers[transfers["from_id"] == cr_id]
            if len(buys) > 0 and len(sells) > 0:
                buy_p = buys["price"].iloc[0]
                sell_p = sells["price"].iloc[0]
                buy_vol = buys["credits"].iloc[0]
                if buy_p > 0 and sell_p > 0:
                    spreads.append(sell_p - buy_p)
                    volumes.append(buy_vol)
        spreads = np.array(spreads)
        volumes = np.array(volumes)
        if len(spreads) > 0:
            pos_mask = spreads > 0
            if pos_mask.sum() > 0:
                avg_pos_spread = np.mean(spreads[pos_mask])
                avg_vol = np.mean(volumes[pos_mask])
                sigma_sq = sigma_intermediary**2
                if sigma_sq > 0 and avg_vol > 0:
                    gamma_intermediary = avg_pos_spread / (avg_vol * sigma_sq)

    # 5. Threshold (BCT/compliance crossover)
    if len(bct_transfers) > 0 and len(compliance_transfers) > 0:
        bct_med = bct_transfers["price"].median()
        comp_med = compliance_transfers[compliance_transfers["price"] > 0]["price"].median()
        threshold_multiplier = bct_med / overall_median if overall_median > 0 else 1.5
    else:
        threshold_multiplier = 1.5

    # 6. Logistic slope (compliance price dispersion)
    logistic_slope = 1.27  # fallback
    if len(compliance_transfers) > 0:
        comp_prices = compliance_transfers[compliance_transfers["price"] > 0]["price"]
        comp_median_price = comp_prices.median()
        if comp_median_price > 0 and len(comp_prices) > 5:
            iqr = comp_prices.quantile(0.75) - comp_prices.quantile(0.25)
            if iqr > 0:
                logistic_slope = 2.197 / (iqr / comp_median_price)

    # 7. Supply elasticity
    formation_stats = (
        eco_transfers.groupby("formation")
        .agg(n_txn=("price", "count"), median_price=("price", "median"))
        .reset_index()
    )
    formation_stats = formation_stats[
        (formation_stats["n_txn"] >= 3) & (formation_stats["median_price"] > 0)
    ]
    eta = 0.39  # fallback
    if len(formation_stats) >= 4:
        log_vol = np.log(formation_stats["n_txn"].values)
        log_price = np.log(formation_stats["median_price"].values)
        slope_e, _, _, _, _ = sp_stats.linregress(log_price, log_vol)
        eta = slope_e

    # --- Observed Fund routing rate ---
    n_bct_retire = len(bct_retirements)
    n_total_retire = len(retirements)
    observed_fund_rate = n_bct_retire / n_total_retire if n_total_retire > 0 else 0.35

    params = {
        "fair_value_multiplier": fair_value_multiplier,
        "sigma_intermediary": sigma_intermediary,
        "sigma_bct": sigma_bct,
        "gamma_intermediary": gamma_intermediary,
        "threshold_multiplier": threshold_multiplier,
        "logistic_slope": logistic_slope,
        "eta": eta,
        "observed_fund_rate": observed_fund_rate,
        "n_bct_retirements": n_bct_retire,
        "n_total_retirements": n_total_retire,
    }

    print("  Derived parameters:")
    for k, v in params.items():
        if isinstance(v, float) and abs(v) < 0.001:
            print(f"    {k:<30s} = {v:.4e}")
        elif isinstance(v, int):
            print(f"    {k:<30s} = {v}")
        else:
            print(f"    {k:<30s} = {v:.4f}")
    print()

    return params


# =============================================================================
# 1. LIQUIDITY FEEDBACK TABLE (empirical transition probabilities)
# =============================================================================

LIQUIDITY_FEEDBACK = {
    "never":       (0, 0, 0.0306, 1.0),
    "thin_low":    (1, 2, 0.0470, 1.5),
    "thin_high":   (3, 4, 0.0885, 2.9),
    "functional":  (5, 9, 0.1550, 5.1),
    "established": (10, 999999, 0.2580, 8.4),
}


def get_liquidity_weight(cumulative_txn: int) -> float:
    """Return the liquidity feedback weight for an OTG given its cumulative transaction count."""
    if cumulative_txn <= 0:
        return 1.0
    elif cumulative_txn <= 2:
        return 1.5
    elif cumulative_txn <= 4:
        return 2.9
    elif cumulative_txn <= 9:
        return 5.1
    else:
        return 8.4


# =============================================================================
# 2. DATA LOADING WITH FORMATION-LEVEL MATCHING
# =============================================================================

FORMATION_NORMALIZE = {
    "Grassy woodlands": "Grassy Woodlands",
}


@dataclass
class OTG:
    name: str
    formation: str
    is_tec: bool
    observed_txn_count: int = 0
    observed_median_price: float = 0.0
    observed_credits: float = 0.0
    supply_capacity: float = 0.0
    subregions: List[str] = field(default_factory=list)
    direct_match_txn: float = 0.0
    formation_match_txn: float = 0.0


@dataclass
class Formation:
    name: str
    otgs: List[OTG]

    @property
    def n_otgs(self) -> int:
        return len(self.otgs)

    @property
    def n_tec(self) -> int:
        return sum(1 for o in self.otgs if o.is_tec)

    @property
    def n_non_tec(self) -> int:
        return sum(1 for o in self.otgs if not o.is_tec)

    @property
    def total_txn(self) -> int:
        return sum(o.observed_txn_count for o in self.otgs)

    @property
    def median_price(self) -> float:
        prices = [o.observed_median_price for o in self.otgs if o.observed_median_price > 0]
        return float(np.median(prices)) if prices else 3000.0


@dataclass
class SubregionDemand:
    name: str
    n_txn: int
    formations: Dict[str, int]
    otg_txn: Dict[str, int]


def load_otg_data_formation_matching() -> Tuple[
    List["Formation"], List[OTG], List[SubregionDemand], Dict
]:
    """Load OTG data with FORMATION-LEVEL MATCHING."""

    # ---- Load supply register ----
    supply = pd.read_csv(DATA_DIR / "nsw_credit_supply_register.csv", encoding="utf-8-sig")
    supply.columns = supply.columns.str.strip()
    supply_eco = supply[supply["Ecosystem or Species"] == "Ecosystem"].copy()
    supply_eco["Vegetation Formation"] = supply_eco["Vegetation Formation"].replace(
        FORMATION_NORMALIZE
    )
    supply_eco["n_credits"] = pd.to_numeric(
        supply_eco["Number of credits"], errors="coerce"
    ).fillna(0)

    tec_col = "Threatened Ecological Community (NSW)"
    non_tec_labels = {"Not a TEC", "No Associated TEC", "No TEC Associated", ""}

    otg_info = (
        supply_eco.groupby("Offset Trading Group")
        .agg(
            formation=("Vegetation Formation", "first"),
            supply_capacity=("n_credits", "sum"),
            tec_vals=(tec_col, lambda x: list(x.dropna().unique())),
            subregions=("IBRA Subregion", lambda x: list(x.dropna().unique())),
        )
        .reset_index()
    )
    otg_info["is_tec"] = otg_info["tec_vals"].apply(
        lambda vals: any(
            str(v).strip() not in non_tec_labels for v in vals if isinstance(v, str)
        )
    )

    supply_otg_names = set(otg_info["Offset Trading Group"])

    otg_formation_map: Dict[str, str] = {}
    otg_capacity_map: Dict[str, float] = {}
    formation_otgs_map: Dict[str, List[str]] = {}
    for _, row in otg_info.iterrows():
        otg_name = row["Offset Trading Group"]
        form_name = row["formation"]
        cap = float(row["supply_capacity"])
        otg_formation_map[otg_name] = form_name
        otg_capacity_map[otg_name] = cap
        formation_otgs_map.setdefault(form_name, []).append(otg_name)

    otg_subregion_map: Dict[str, set] = {}
    for _, row in supply_eco.iterrows():
        otg_name = row["Offset Trading Group"]
        sub = row.get("IBRA Subregion", "")
        if pd.notna(sub) and str(sub).strip():
            otg_subregion_map.setdefault(otg_name, set()).add(str(sub).strip())

    sub_form_supply: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    for _, row in otg_info.iterrows():
        otg_name = row["Offset Trading Group"]
        form_name = row["formation"]
        cap = float(row["supply_capacity"])
        for sub in otg_subregion_map.get(otg_name, []):
            key = (sub, form_name)
            sub_form_supply.setdefault(key, []).append((otg_name, max(1.0, cap)))

    # ---- Load transaction register ----
    txn = pd.read_csv(
        DATA_DIR / "nsw_credit_transactions_register.csv", encoding="utf-8-sig"
    )
    txn.columns = txn.columns.str.strip()
    txn["price"] = pd.to_numeric(txn["Price Per Credit (Ex-GST)"], errors="coerce")
    txn["credits"] = pd.to_numeric(txn["Number Of Credits"], errors="coerce")
    transfers = txn[txn["Transaction Type"] == "Transfer"].copy()
    priced = transfers[(transfers["price"] >= 100) & transfers["price"].notna()].copy()
    priced["credit_type"] = np.where(
        priced["Scientific Name"].fillna("").str.strip() != "", "Species", "Ecosystem"
    )
    eco = priced[priced["credit_type"] == "Ecosystem"].copy()
    eco["Vegetation Formation"] = eco["Vegetation Formation"].replace(FORMATION_NORMALIZE)

    # =========================================================================
    # FORMATION-LEVEL MATCHING
    # =========================================================================

    otg_txn_assigned: Dict[str, float] = {}
    otg_prices: Dict[str, List[float]] = {}
    otg_credits_assigned: Dict[str, float] = {}
    otg_direct_count: Dict[str, float] = {}
    otg_formation_count: Dict[str, float] = {}

    n_direct = 0
    n_formation_matched = 0
    n_unmatched = 0

    formation_direct_txn: Dict[str, int] = {}
    formation_level_txn: Dict[str, int] = {}
    unmatched_details: List[str] = []

    for _, row in eco.iterrows():
        txn_otg = str(row.get("Offset Trading Group", "")).strip()
        txn_form = str(row.get("Vegetation Formation", "")).strip()
        price = row["price"]
        credits = row["credits"] if pd.notna(row["credits"]) else 0

        if txn_otg in supply_otg_names:
            otg_txn_assigned[txn_otg] = otg_txn_assigned.get(txn_otg, 0) + 1
            otg_prices.setdefault(txn_otg, []).append(price)
            otg_credits_assigned[txn_otg] = otg_credits_assigned.get(txn_otg, 0) + credits
            otg_direct_count[txn_otg] = otg_direct_count.get(txn_otg, 0) + 1
            n_direct += 1
            formation_direct_txn[txn_form] = formation_direct_txn.get(txn_form, 0) + 1
        else:
            if txn_form and txn_form in formation_otgs_map:
                candidate_otgs = formation_otgs_map[txn_form]
                capacities = np.array(
                    [max(1.0, otg_capacity_map.get(o, 1.0)) for o in candidate_otgs],
                    dtype=float,
                )
                total_cap = capacities.sum()
                for i, cand_otg in enumerate(candidate_otgs):
                    share = capacities[i] / total_cap
                    otg_txn_assigned[cand_otg] = otg_txn_assigned.get(cand_otg, 0) + share
                    otg_prices.setdefault(cand_otg, []).append(price)
                    otg_credits_assigned[cand_otg] = (
                        otg_credits_assigned.get(cand_otg, 0) + credits * share
                    )
                    otg_formation_count[cand_otg] = (
                        otg_formation_count.get(cand_otg, 0) + share
                    )
                n_formation_matched += 1
                formation_level_txn[txn_form] = formation_level_txn.get(txn_form, 0) + 1
            else:
                n_unmatched += 1
                unmatched_details.append(f"OTG={txn_otg}, Form={txn_form}")

    # ---- Build subregion demand data ----
    subregion_data: Dict[str, dict] = {}
    for _, row in eco.iterrows():
        sub = row.get("Sub Region", "")
        if pd.isna(sub) or str(sub).strip() == "":
            continue
        sub = str(sub).strip()
        form = str(row.get("Vegetation Formation", "")).strip()

        if sub not in subregion_data:
            subregion_data[sub] = {"n_txn": 0, "formations": {}, "otg_txn": {}}
        subregion_data[sub]["n_txn"] += 1
        subregion_data[sub]["formations"][form] = (
            subregion_data[sub]["formations"].get(form, 0) + 1
        )

        txn_otg = str(row.get("Offset Trading Group", "")).strip()
        if txn_otg in supply_otg_names:
            subregion_data[sub]["otg_txn"][txn_otg] = (
                subregion_data[sub]["otg_txn"].get(txn_otg, 0) + 1
            )
        else:
            txn_form = str(row.get("Vegetation Formation", "")).strip()
            if txn_form and txn_form in formation_otgs_map:
                candidate_otgs = formation_otgs_map[txn_form]
                sub_candidates = [
                    o for o in candidate_otgs if sub in otg_subregion_map.get(o, set())
                ]
                if not sub_candidates:
                    sub_candidates = candidate_otgs
                capacities = np.array(
                    [max(1.0, otg_capacity_map.get(o, 1.0)) for o in sub_candidates],
                    dtype=float,
                )
                total_cap = capacities.sum()
                for i, cand_otg in enumerate(sub_candidates):
                    share = capacities[i] / total_cap
                    subregion_data[sub]["otg_txn"][cand_otg] = (
                        subregion_data[sub]["otg_txn"].get(cand_otg, 0) + share
                    )

    subregion_demands = [
        SubregionDemand(
            name=sub,
            n_txn=data["n_txn"],
            formations=data["formations"],
            otg_txn=data["otg_txn"],
        )
        for sub, data in subregion_data.items()
    ]

    # ---- Formation median prices ----
    form_txn = eco.groupby("Vegetation Formation").agg(
        median_price=("price", "median")
    ).reset_index()
    form_price_dict = dict(zip(form_txn["Vegetation Formation"], form_txn["median_price"]))

    # ---- Build OTG objects ----
    all_otgs: List[OTG] = []
    for _, row in otg_info.iterrows():
        otg_name = row["Offset Trading Group"]
        form_name = row["formation"]
        assigned_txn = otg_txn_assigned.get(otg_name, 0)
        prices = otg_prices.get(otg_name, [])
        med_price = float(np.median(prices)) if prices else 0.0
        assigned_credits = otg_credits_assigned.get(otg_name, 0)
        if med_price <= 0:
            med_price = form_price_dict.get(form_name, 0.0)
        subs = list(otg_subregion_map.get(otg_name, []))
        all_otgs.append(
            OTG(
                name=otg_name,
                formation=form_name,
                is_tec=row["is_tec"],
                observed_txn_count=int(round(assigned_txn)),
                observed_median_price=float(med_price) if med_price > 0 else 0.0,
                observed_credits=float(assigned_credits),
                supply_capacity=float(row["supply_capacity"]),
                subregions=subs,
                direct_match_txn=otg_direct_count.get(otg_name, 0),
                formation_match_txn=otg_formation_count.get(otg_name, 0),
            )
        )

    # ---- Build formation objects ----
    formation_dict: Dict[str, List[OTG]] = {}
    for otg in all_otgs:
        formation_dict.setdefault(otg.formation, []).append(otg)
    formations = [
        Formation(name=name, otgs=otgs)
        for name, otgs in sorted(formation_dict.items(), key=lambda x: -len(x[1]))
    ]

    # ---- Matching report ----
    formation_demand_weights: Dict[str, float] = {}
    for form_name in formation_otgs_map:
        direct = formation_direct_txn.get(form_name, 0)
        from_form = formation_level_txn.get(form_name, 0)
        formation_demand_weights[form_name] = direct + from_form

    matching_report = {
        "n_total": n_direct + n_formation_matched + n_unmatched,
        "n_direct": n_direct,
        "n_formation_matched": n_formation_matched,
        "n_unmatched": n_unmatched,
        "formation_direct_txn": formation_direct_txn,
        "formation_level_txn": formation_level_txn,
        "formation_demand_weights": formation_demand_weights,
        "unmatched_details": unmatched_details,
    }

    return formations, all_otgs, subregion_demands, matching_report


# =============================================================================
# 3. GEOGRAPHIC DEMAND ENGINE (with liquidity feedback hooks)
# =============================================================================


def _power_law_weights(counts: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    n = len(counts)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    order = np.argsort(-counts)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    weights = ranks ** (-alpha)
    weights /= weights.sum()
    return weights


class GeographicDemandEngine:
    """
    Generates obligations using the empirical geographic distribution of
    development pressure across IBRA subregions.

    Enhanced with liquidity feedback: when generating standard (non-flex)
    obligations, the OTG selection weights within a subregion x formation
    cell are multiplied by the empirical liquidity feedback weights.
    """

    def __init__(self, formations, all_otgs, subregion_demands):
        self.formations = formations
        self.all_otgs = all_otgs
        self.formation_lookup = {f.name: f for f in formations}
        self.otg_lookup = {o.name: o for o in all_otgs}

        self.subregions = [s for s in subregion_demands if s.n_txn > 0]
        sub_txn = np.array([s.n_txn for s in self.subregions], dtype=float)
        self.subregion_weights = sub_txn / sub_txn.sum() if sub_txn.sum() > 0 else None

        self.sub_form_otgs: Dict[str, Dict[str, List[OTG]]] = {}
        for otg in all_otgs:
            for sub in otg.subregions:
                if sub not in self.sub_form_otgs:
                    self.sub_form_otgs[sub] = {}
                form = otg.formation
                if form not in self.sub_form_otgs[sub]:
                    self.sub_form_otgs[sub][form] = []
                self.sub_form_otgs[sub][form].append(otg)

        self.sub_form_weights: Dict[str, Tuple[List[str], np.ndarray]] = {}
        for sd in self.subregions:
            forms, weights = [], []
            for form_name, count in sd.formations.items():
                if form_name in self.sub_form_otgs.get(sd.name, {}):
                    forms.append(form_name)
                    weights.append(count)
            if forms:
                w = np.array(weights, dtype=float)
                w = w**2
                w /= w.sum()
                self.sub_form_weights[sd.name] = (forms, w)

        self.sub_otg_txn: Dict[str, Dict[str, int]] = {}
        for sd in self.subregions:
            self.sub_otg_txn[sd.name] = sd.otg_txn

    def generate_obligation(
        self, rng, cumulative_otg_txn: Dict[str, int] = None, apply_feedback: bool = True
    ):
        if self.subregion_weights is None or len(self.subregions) == 0:
            return None

        si = rng.choice(len(self.subregions), p=self.subregion_weights)
        sub = self.subregions[si]

        if sub.name not in self.sub_form_weights:
            return None
        forms, form_w = self.sub_form_weights[sub.name]
        fi = rng.choice(len(forms), p=form_w)
        form_name = forms[fi]

        otgs_in_cell = self.sub_form_otgs.get(sub.name, {}).get(form_name, [])
        if not otgs_in_cell:
            return None

        sub_otg_txn = self.sub_otg_txn.get(sub.name, {})
        otg_counts = np.array(
            [sub_otg_txn.get(o.name, 0) + o.observed_txn_count * 0.1 for o in otgs_in_cell],
            dtype=float,
        )

        otg_weights = _power_law_weights(otg_counts, alpha=4.0)

        if apply_feedback and cumulative_otg_txn is not None:
            feedback_weights = np.array(
                [get_liquidity_weight(cumulative_otg_txn.get(o.name, 0)) for o in otgs_in_cell],
                dtype=float,
            )
            otg_weights = otg_weights * feedback_weights
            total = otg_weights.sum()
            if total > 0:
                otg_weights /= total
            else:
                otg_weights = np.ones(len(otgs_in_cell)) / len(otgs_in_cell)

        oi = rng.choice(len(otgs_in_cell), p=otg_weights)
        otg = otgs_in_cell[oi]
        form = self.formation_lookup.get(otg.formation)
        if form is None:
            return None
        return (otg, form)


# =============================================================================
# 4. CONFIGURATION
# =============================================================================


@dataclass
class FormationConfig:
    T: int = 72
    n_compliance: int = 30
    n_intermediary: int = 10
    n_bct: int = 12
    n_habitat_bank: int = 30
    monthly_obligations: float = 12.0
    seed: int = 42
    procurement_flex_share: float = 0.0
    price_floor: float = 0.0
    p_variation: float = 0.5  # Calibrated parameter
    bct_exact_match_only: bool = False  # Scenario B: restrict BCT to exact OTG
    rarity_multiplier: float = 1.0  # Scenario C: credit multiplier for rare OTGs


# =============================================================================
# 5. DATA-DERIVED AGENTS WITH PARTIAL LIKE-FOR-LIKE
# =============================================================================

PARAMS: Dict[str, float] = {}


class FormationAgent:
    _counter = 0

    def __init__(self, agent_type: str, cash: float, rng: np.random.RandomState):
        FormationAgent._counter += 1
        self.id = FormationAgent._counter
        self.agent_type = agent_type
        self.cash = cash
        self.rng = rng
        self.transactions: List[Tuple[str, float, float]] = []


class ComplianceAgentPL4L(FormationAgent):
    """
    Data-derived compliance buyer with PARTIAL like-for-like.
    Logistic willingness (slope=1.27, threshold=1.11*price).

    When obligation for OTG X:
      1. Try to buy from OTG X directly
      2. If no supply: with prob P_variation, try substitution within formation
      3. Otherwise: pay into Fund (BCT takes over)
    """

    def __init__(self, cash: float, rng: np.random.RandomState):
        super().__init__("Compliance", cash, rng)

    def bid_l4l(
        self,
        obligation_otg,
        formation,
        supply_available,
        price_floor,
        p_variation: float,
        cumulative_otg_txn: Dict[str, int] = None,
        rarity_multiplier: float = 1.0,
    ):
        """
        Partial like-for-like bid.

        Returns: (otg_name, price, qty, route_type) or None
        route_type: "direct" | "variation" | None (Fund)
        """
        base_price = formation.median_price if formation.median_price > 0 else 3000.0
        threshold_mult = PARAMS.get("threshold_multiplier", 1.11)
        slope = PARAMS.get("logistic_slope", 1.27)

        # Rarity multiplier: require more credits for rare OTGs
        qty_mult = 1.0
        if rarity_multiplier > 1.0 and cumulative_otg_txn is not None:
            if cumulative_otg_txn.get(obligation_otg.name, 0) < 5:
                qty_mult = rarity_multiplier

        def _try_buy(otg):
            avail = supply_available.get(otg.name, 0.0)
            if avail <= 0:
                return None
            ask = otg.observed_median_price if otg.observed_median_price > 0 else base_price
            ask = max(ask, price_floor) * self.rng.lognormal(0, 0.15)

            price_ratio = ask / (base_price + 1e-10)
            willingness = 1.0 / (1.0 + np.exp(slope * (price_ratio - threshold_mult)))
            affordable = min(1.0, self.cash * 0.8 / (ask + 1e-10))

            if willingness * affordable < 0.5:
                return None

            qty = min(self.rng.lognormal(2.5, 0.8) * qty_mult, avail, self.cash / (ask + 1))
            return (otg.name, ask, qty) if qty >= 1 else None

        # Step 1: Try to buy from EXACT obligation OTG
        result = _try_buy(obligation_otg)
        if result:
            return (*result, "direct")

        # Step 2: With probability P_variation, try variation rules (same formation)
        if self.rng.random() < p_variation:
            # Search for alternative OTG within SAME formation that has supply
            candidates = []
            for otg in formation.otgs:
                if otg.name == obligation_otg.name:
                    continue  # Already tried
                avail = supply_available.get(otg.name, 0.0)
                if avail <= 0:
                    continue
                ask = otg.observed_median_price if otg.observed_median_price > 0 else base_price
                ask = max(ask, price_floor) * self.rng.lognormal(0, 0.15)

                price_ratio = ask / (base_price + 1e-10)
                willingness = 1.0 / (1.0 + np.exp(slope * (price_ratio - threshold_mult)))
                affordable = min(1.0, self.cash * 0.8 / (ask + 1e-10))

                if willingness * affordable >= 0.5:
                    liq_w = 1.0
                    if cumulative_otg_txn is not None:
                        liq_w = get_liquidity_weight(cumulative_otg_txn.get(otg.name, 0))
                    candidates.append((otg.name, ask, avail, liq_w))

            if candidates:
                # Pick cheapest available (weighted by liquidity)
                if cumulative_otg_txn is not None and len(candidates) > 1:
                    cand_weights = np.array([c[3] for c in candidates], dtype=float)
                    cand_prices = np.array([c[1] for c in candidates], dtype=float)
                    price_rank = np.argsort(np.argsort(cand_prices)) + 1
                    price_w = 1.0 / price_rank
                    combined_w = cand_weights * price_w
                    combined_w /= combined_w.sum()
                    idx = self.rng.choice(len(candidates), p=combined_w)
                else:
                    candidates.sort(key=lambda x: x[1])
                    idx = 0

                otg_name, price, avail, _ = candidates[idx]
                qty = min(self.rng.lognormal(2.5, 0.8) * qty_mult, avail, self.cash / (price + 1))
                if qty >= 1:
                    return (otg_name, price, qty, "variation")

        # Step 3: Failed -> pay into Fund (BCT takes over)
        return None


class IntermediaryAgentPL4L(FormationAgent):
    """
    Data-derived intermediary WITH liquidity feedback.
    CARA (gamma=2.27e-7, sigma=2314, fair_value=1.43*price).
    NOT constrained by like-for-like.
    """

    def __init__(self, cash: float, rng: np.random.RandomState):
        super().__init__("Intermediary", cash, rng)

    def bid(
        self,
        formations,
        supply_available,
        price_floor,
        cumulative_otg_txn: Dict[str, int] = None,
    ):
        gamma = PARAMS.get("gamma_intermediary", 2.27e-7)
        sigma = PARAMS.get("sigma_intermediary", 2314.0)
        fv_mult = PARAMS.get("fair_value_multiplier", 1.43)
        sigma_sq = sigma**2

        weights = []
        candidate_otgs = []
        for f in formations:
            for otg in f.otgs:
                avail = supply_available.get(otg.name, 0.0)
                if avail <= 0:
                    continue
                liq_weight = max(1, otg.observed_txn_count) ** 3.0
                if cumulative_otg_txn is not None:
                    liq_weight *= get_liquidity_weight(
                        cumulative_otg_txn.get(otg.name, 0)
                    )
                weights.append(liq_weight)
                candidate_otgs.append((otg, f))

        if not candidate_otgs:
            return None

        weights = np.array(weights)
        weights /= weights.sum()
        idx = self.rng.choice(len(candidate_otgs), p=weights)
        otg, form = candidate_otgs[idx]

        base_price = (
            otg.observed_median_price if otg.observed_median_price > 0 else form.median_price
        )
        fair_value = base_price * fv_mult
        fair_value *= self.rng.lognormal(0, 0.1)
        ask_price = base_price * self.rng.lognormal(0, 0.1)
        ask_price = max(ask_price, price_floor)

        q_star = (fair_value - ask_price) / (gamma * sigma_sq + 1e-10)
        if q_star <= 0:
            return None

        avail = supply_available.get(otg.name, 0.0)
        qty = min(q_star, avail, self.cash / (ask_price + 1) * 0.3)
        if qty < 1:
            return None
        return (otg.name, ask_price, qty)


class BCTAgentPL4L(FormationAgent):
    """
    BCT agent: handles Fund obligations + proactive thin-market purchases.
    Risk-neutral budget-constrained (Arrow-Lind), beta=0, bid cap=$5,989.
    EXEMPT from liquidity feedback.
    """

    def __init__(self, cash: float, rng: np.random.RandomState):
        super().__init__("BCT", cash, rng)
        self.annual_budget = rng.lognormal(np.log(326389), 0.5)
        self.budget_remaining = self.annual_budget
        self.thin_market_preference = rng.uniform(2.0, 5.0)
        self.fund_obligations: List[Tuple] = []

    def reset_budget(self):
        self.budget_remaining = self.annual_budget

    NON_TEC_PURCHASE_PROB = 0.05

    def receive_fund_obligation(self, otg, formation):
        """Receive an obligation from the Fund (compliance couldn't buy)."""
        self.fund_obligations.append((otg, formation))

    def bid_fund_obligation(self, supply_available, price_floor,
                            bct_exact_match_only=False, rarity_multiplier=1.0,
                            cumulative_otg_txn=None):
        """
        Process a Fund obligation: BCT has discretion to buy from the
        obligation OTG or a related OTG within the same formation.
        If bct_exact_match_only=True, skip the formation fallback.
        """
        if self.budget_remaining <= 0 or not self.fund_obligations:
            return None

        otg, formation = self.fund_obligations.pop(0)

        # Rarity multiplier: require more credits for rare OTGs
        qty_mult = 1.0
        if rarity_multiplier > 1.0 and cumulative_otg_txn is not None:
            if cumulative_otg_txn.get(otg.name, 0) < 5:
                qty_mult = rarity_multiplier

        # Try the exact obligation OTG first
        avail = supply_available.get(otg.name, 0.0)
        if avail > 0:
            base_price = (
                otg.observed_median_price
                if otg.observed_median_price > 0
                else formation.median_price
            )
            if base_price <= 0:
                base_price = 3000.0
            bid_price = base_price * self.rng.lognormal(0, 0.15)
            bid_price = max(bid_price, price_floor)
            bid_price = min(bid_price, 5989.0 * self.rng.lognormal(0, 0.2))

            target_qty = self.rng.lognormal(1.5, 0.8) * qty_mult
            max_affordable = self.budget_remaining / (bid_price + 1e-10)
            max_from_cash = self.cash / (bid_price + 1) * 0.4
            qty = min(target_qty, avail, max_affordable, max_from_cash)
            if qty >= 1:
                return (otg.name, bid_price, qty, "BCT-Fund")

        # If exact match mandated, no fallback
        if bct_exact_match_only:
            return None

        # Fallback: buy from another OTG within the same formation
        candidates = []
        for other_otg in formation.otgs:
            other_avail = supply_available.get(other_otg.name, 0.0)
            if other_avail <= 0:
                continue
            base_p = (
                other_otg.observed_median_price
                if other_otg.observed_median_price > 0
                else formation.median_price
            )
            if base_p <= 0:
                base_p = 3000.0
            candidates.append((other_otg, other_avail, base_p))

        if not candidates:
            return None

        # Prefer thin-market OTGs within the formation
        cand_weights = []
        for c_otg, c_avail, c_price in candidates:
            w = 1.0
            if c_otg.observed_txn_count < 5:
                w = self.thin_market_preference * max(1, 5 - c_otg.observed_txn_count)
            cand_weights.append(w)
        cand_weights = np.array(cand_weights)
        cand_weights /= cand_weights.sum()
        idx = self.rng.choice(len(candidates), p=cand_weights)
        chosen_otg, chosen_avail, chosen_base = candidates[idx]

        bid_price = chosen_base * self.rng.lognormal(0, 0.15)
        bid_price = max(bid_price, price_floor)
        bid_price = min(bid_price, 5989.0 * self.rng.lognormal(0, 0.2))

        target_qty = self.rng.lognormal(1.5, 0.8) * qty_mult
        max_affordable = self.budget_remaining / (bid_price + 1e-10)
        max_from_cash = self.cash / (bid_price + 1) * 0.4
        qty = min(target_qty, chosen_avail, max_affordable, max_from_cash)
        if qty >= 1:
            return (chosen_otg.name, bid_price, qty, "BCT-Fund-Flex")
        return None

    def bid_proactive(self, formations, supply_available, price_floor,
                      bct_exact_match_only=False):
        """BCT proactive bid -- NO liquidity feedback. Thin-market preference."""
        if self.budget_remaining <= 0 or bct_exact_match_only:
            return None

        allow_non_tec = self.rng.random() < self.NON_TEC_PURCHASE_PROB

        candidate_otgs = []
        weights = []
        for f in formations:
            for otg in f.otgs:
                if not otg.is_tec and not allow_non_tec:
                    continue
                avail = supply_available.get(otg.name, 0.0)
                if avail <= 0:
                    continue
                if otg.observed_txn_count < 5:
                    w = self.thin_market_preference * max(1, 5 - otg.observed_txn_count)
                else:
                    w = 1.0
                if not otg.is_tec:
                    w *= 0.1
                weights.append(w)
                candidate_otgs.append((otg, f))

        if not candidate_otgs:
            return None

        weights = np.array(weights)
        weights /= weights.sum()
        idx = self.rng.choice(len(candidate_otgs), p=weights)
        otg, form = candidate_otgs[idx]

        base_price = (
            otg.observed_median_price if otg.observed_median_price > 0 else form.median_price
        )
        if base_price <= 0:
            base_price = 3000.0

        bid_price = base_price * self.rng.lognormal(0, 0.15)
        bid_price = max(bid_price, price_floor)
        bid_price = min(bid_price, 5989.0 * self.rng.lognormal(0, 0.2))

        target_qty = self.rng.lognormal(1.5, 0.8)
        max_affordable = self.budget_remaining / (bid_price + 1e-10)
        max_from_cash = self.cash / (bid_price + 1) * 0.4

        avail = supply_available.get(otg.name, 0.0)
        qty = min(target_qty, avail, max_affordable, max_from_cash)
        if qty < 1:
            return None
        return (otg.name, bid_price, qty)


class HabitatBankAgentPL4L(FormationAgent):
    """
    Data-derived habitat bank.
    Supply elasticity eta = 0.39, cost = lognormal(7.62, 0.5).
    Source: Q1 of ecosystem credit prices = $2,033; log(2033) = 7.62.
    """

    def __init__(self, cash, rng, formations):
        super().__init__("HabitatBank", cash, rng)
        n_specialise = rng.randint(1, min(4, len(formations) + 1))
        form_weights = np.array([max(1, f.total_txn) for f in formations], dtype=float)
        form_weights /= form_weights.sum()
        self.specialisation_indices = rng.choice(
            len(formations), size=n_specialise, replace=False, p=form_weights
        )
        self.production_cost = rng.lognormal(7.62, 0.5)
        self.capacity_per_month = rng.lognormal(2.5, 0.8)

    def produce(self, formations, supply_available):
        eta = PARAMS.get("eta", 0.39)

        new_credits = max(1, int(self.rng.poisson(self.capacity_per_month * 0.4)))
        cost = new_credits * self.production_cost * 0.15
        if self.cash < cost:
            return
        self.cash -= cost

        for fi in self.specialisation_indices:
            if fi >= len(formations):
                continue
            form = formations[fi]

            otgs_with_cap = [o for o in form.otgs if o.supply_capacity > 0]
            if not otgs_with_cap:
                otgs_with_cap = form.otgs[:1]

            cap = np.array([o.supply_capacity for o in otgs_with_cap], dtype=float)
            txn_arr = np.array(
                [max(1, o.observed_txn_count) for o in otgs_with_cap], dtype=float
            )

            price_ratios = []
            for o in otgs_with_cap:
                p = o.observed_median_price if o.observed_median_price > 0 else form.median_price
                cost_ratio = self.production_cost / (p + 1e-10)
                margin = max(0.0, 1.0 - cost_ratio)
                supply_weight = margin**eta if margin > 0 else 0.0
                price_ratios.append(supply_weight)
            price_ratios = np.array(price_ratios)

            combined = cap * txn_arr * (price_ratios + 0.01)
            combined = (
                combined / combined.sum()
                if combined.sum() > 0
                else np.ones(len(cap)) / len(cap)
            )

            credits_to_form = max(1, new_credits // len(self.specialisation_indices))
            for _ in range(credits_to_form):
                otg_idx = self.rng.choice(len(otgs_with_cap), p=combined)
                otg = otgs_with_cap[otg_idx]
                supply_available[otg.name] = supply_available.get(otg.name, 0.0) + 1


# =============================================================================
# 6. OBLIGATION GENERATION (with liquidity feedback)
# =============================================================================


def generate_obligations(
    demand_engine,
    formations,
    all_otgs,
    rng,
    monthly_obligations=12.0,
    procurement_flex_share=0.0,
    cumulative_otg_txn: Dict[str, int] = None,
):
    """
    Generate OTG-specific obligations.

    Standard: OTG drawn from geographic demand engine + liquidity feedback.
    Flex (procurement_flex_share): can be fulfilled by ANY OTG in ANY formation.
    """
    n_obligations = max(1, rng.poisson(monthly_obligations))
    obligations = []  # (otg, formation, is_flex)
    formation_lookup = {f.name: f for f in formations}
    n_standard = int(n_obligations * (1.0 - procurement_flex_share))
    n_flex = n_obligations - n_standard

    # Standard obligations: WITH liquidity feedback, OTG-specific
    for _ in range(n_standard):
        result = demand_engine.generate_obligation(
            rng, cumulative_otg_txn=cumulative_otg_txn, apply_feedback=True
        )
        if result is not None:
            otg, form = result
            obligations.append((otg, form, False))  # is_flex=False

    # Flex obligations: can be fulfilled by ANY OTG in the same formation
    if n_flex > 0:
        for _ in range(n_flex):
            result = demand_engine.generate_obligation(
                rng, cumulative_otg_txn=cumulative_otg_txn, apply_feedback=True
            )
            if result is not None:
                otg, form = result
                obligations.append((otg, form, True))  # is_flex=True

    return obligations


# =============================================================================
# 7. MARKET CLEARING (with partial like-for-like)
# =============================================================================


def _compliance_flex_bid(agent, obligation_otg, formation, supply_available, price_floor):
    """Flex bid: try obligation OTG first, then cheapest in formation."""
    base_price = formation.median_price if formation.median_price > 0 else 3000.0
    threshold_mult = PARAMS.get("threshold_multiplier", 1.11)
    slope = PARAMS.get("logistic_slope", 1.27)

    def _try_buy(otg):
        avail = supply_available.get(otg.name, 0.0)
        if avail <= 0:
            return None
        ask = otg.observed_median_price if otg.observed_median_price > 0 else base_price
        ask = max(ask, price_floor) * agent.rng.lognormal(0, 0.15)
        price_ratio = ask / (base_price + 1e-10)
        willingness = 1.0 / (1.0 + np.exp(slope * (price_ratio - threshold_mult)))
        affordable = min(1.0, agent.cash * 0.8 / (ask + 1e-10))
        if willingness * affordable < 0.5:
            return None
        qty = min(agent.rng.lognormal(2.5, 0.8), avail, agent.cash / (ask + 1))
        return (otg.name, ask, qty) if qty >= 1 else None

    result = _try_buy(obligation_otg)
    if result:
        return result

    candidates = []
    for otg in formation.otgs:
        avail = supply_available.get(otg.name, 0.0)
        if avail <= 0:
            continue
        ask = otg.observed_median_price if otg.observed_median_price > 0 else base_price
        ask = max(ask, price_floor) * agent.rng.lognormal(0, 0.15)
        price_ratio = ask / (base_price + 1e-10)
        willingness = 1.0 / (1.0 + np.exp(slope * (price_ratio - threshold_mult)))
        affordable = min(1.0, agent.cash * 0.8 / (ask + 1e-10))
        if willingness * affordable >= 0.5:
            candidates.append((otg.name, ask, avail))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1])
    otg_name, price, avail = candidates[0]
    qty = min(agent.rng.lognormal(2.5, 0.8), avail, agent.cash / (price + 1))
    if qty < 1:
        return None
    return (otg_name, price, qty)


def clear_market_step(
    formations,
    all_otgs,
    obligations,
    compliance_agents,
    intermediary_agents,
    bct_agents,
    habitat_banks,
    supply_available,
    rng,
    price_floor=0.0,
    p_variation=0.5,
    cumulative_otg_txn: Dict[str, int] = None,
    bct_exact_match_only: bool = False,
    rarity_multiplier: float = 1.0,
):
    """
    Single timestep market clearing with PARTIAL like-for-like.

    Compliance agents try exact OTG first.
    If fail: with prob P_variation, try variation rules (same formation).
    If still fail: obligation goes to BCT via Fund.
    """
    transactions = []
    fund_payments = 0
    direct_purchases = 0
    variation_purchases = 0

    # 1. Habitat banks produce
    for bank in habitat_banks:
        bank.produce(formations, supply_available)

    # 2. Compliance agents bid on their obligations (PARTIAL L4L)
    rng.shuffle(obligations)
    for i, (obl_otg, form, is_flex) in enumerate(obligations):
        agent_idx = i % len(compliance_agents)
        agent = compliance_agents[agent_idx]

        if is_flex:
            # FLEX obligation: compliance can buy ANY OTG in the formation
            result = _compliance_flex_bid(
                agent, obl_otg, form, supply_available, price_floor
            )
            if result:
                otg_name, price, qty = result
                supply_available[otg_name] = max(0, supply_available.get(otg_name, 0) - qty)
                agent.cash -= price * qty
                agent.transactions.append((otg_name, price, qty))
                transactions.append((otg_name, price, qty, "Compliance"))
                direct_purchases += 1
            else:
                # Even flex failed -> Fund
                fund_payments += 1
                bct_idx = rng.randint(len(bct_agents))
                bct_agents[bct_idx].receive_fund_obligation(obl_otg, form)
        else:
            # PARTIAL L4L obligation
            result = agent.bid_l4l(
                obl_otg, form, supply_available, price_floor,
                p_variation=p_variation,
                cumulative_otg_txn=cumulative_otg_txn,
                rarity_multiplier=rarity_multiplier,
            )
            if result is not None:
                otg_name, price, qty, route_type = result
                supply_available[otg_name] = max(0, supply_available.get(otg_name, 0) - qty)
                agent.cash -= price * qty
                agent.transactions.append((otg_name, price, qty))
                transactions.append((otg_name, price, qty, "Compliance"))
                if route_type == "direct":
                    direct_purchases += 1
                else:
                    variation_purchases += 1
            else:
                # Failed -> Fund payment -> BCT takes over
                fund_payments += 1
                bct_idx = rng.randint(len(bct_agents))
                bct_agents[bct_idx].receive_fund_obligation(obl_otg, form)

    # 3. BCT agents process Fund obligations
    for agent in bct_agents:
        while agent.fund_obligations and agent.budget_remaining > 0:
            result = agent.bid_fund_obligation(
                supply_available, price_floor,
                bct_exact_match_only=bct_exact_match_only,
                rarity_multiplier=rarity_multiplier,
                cumulative_otg_txn=cumulative_otg_txn,
            )
            if result:
                otg_name, bid_price, qty, btype = result
                supply_available[otg_name] = max(
                    0, supply_available.get(otg_name, 0) - qty
                )
                agent.cash -= bid_price * qty
                agent.budget_remaining -= bid_price * qty
                agent.transactions.append((otg_name, bid_price, qty))
                transactions.append((otg_name, bid_price, qty, btype))
            else:
                break
        agent.fund_obligations.clear()

    # 4. BCT agents proactive purchases (15% active, NO liquidity feedback)
    for agent in bct_agents:
        if rng.random() > 0.15:
            continue
        result = agent.bid_proactive(formations, supply_available, price_floor,
                                     bct_exact_match_only=bct_exact_match_only)
        if result:
            otg_name, price, qty = result
            supply_available[otg_name] = max(0, supply_available.get(otg_name, 0) - qty)
            agent.cash -= price * qty
            agent.budget_remaining -= price * qty
            agent.transactions.append((otg_name, price, qty))
            transactions.append((otg_name, price, qty, "BCT"))

    # 5. Intermediaries (20% active, WITH liquidity feedback)
    for agent in intermediary_agents:
        if rng.random() > 0.20:
            continue
        result = agent.bid(
            formations, supply_available, price_floor,
            cumulative_otg_txn=cumulative_otg_txn,
        )
        if result:
            otg_name, price, qty = result
            supply_available[otg_name] = max(0, supply_available.get(otg_name, 0) - qty)
            agent.cash -= price * qty
            agent.transactions.append((otg_name, price, qty))
            transactions.append((otg_name, price, qty, "Intermediary"))

    return transactions, fund_payments, direct_purchases, variation_purchases


# =============================================================================
# 8. SIMULATION RUNNER
# =============================================================================


def run_single(formations, all_otgs, subregion_demands, config, seed=None):
    rng = np.random.RandomState(seed if seed is not None else config.seed)
    FormationAgent._counter = 0

    demand_engine = GeographicDemandEngine(formations, all_otgs, subregion_demands)

    compliance_agents = [
        ComplianceAgentPL4L(rng.lognormal(np.log(200_000), 0.5), rng)
        for _ in range(config.n_compliance)
    ]
    intermediary_agents = [
        IntermediaryAgentPL4L(rng.lognormal(np.log(100_000), 0.7), rng)
        for _ in range(config.n_intermediary)
    ]
    bct_agents = [
        BCTAgentPL4L(rng.lognormal(np.log(60_000), 0.4), rng)
        for _ in range(config.n_bct)
    ]
    habitat_banks = [
        HabitatBankAgentPL4L(rng.lognormal(np.log(250_000), 0.5), rng, formations)
        for _ in range(config.n_habitat_bank)
    ]

    supply_available: Dict[str, float] = {}
    total_supply = sum(o.supply_capacity for o in all_otgs)
    for otg in all_otgs:
        if otg.supply_capacity > 0:
            monthly_share = otg.supply_capacity / max(1, total_supply) * 7700
            supply_available[otg.name] = max(1, monthly_share * 2)
        else:
            supply_available[otg.name] = rng.exponential(0.2)

    cumulative_otg_txn: Dict[str, int] = {o.name: 0 for o in all_otgs}

    all_transactions = []
    total_fund_payments = 0
    total_direct_purchases = 0
    total_variation_purchases = 0
    midpoint_otg_txn: Dict[str, int] = {}

    for t in range(config.T):
        if t % 12 == 0:
            for agent in bct_agents:
                agent.reset_budget()

        for agent in compliance_agents:
            agent.cash += rng.lognormal(np.log(15_000), 0.3)
        for agent in intermediary_agents:
            agent.cash += rng.lognormal(np.log(8_000), 0.3)
        for agent in bct_agents:
            agent.cash += rng.lognormal(np.log(3_000), 0.3)
        for agent in habitat_banks:
            agent.cash += rng.lognormal(np.log(12_000), 0.3)

        obligations = generate_obligations(
            demand_engine,
            formations,
            all_otgs,
            rng,
            monthly_obligations=config.monthly_obligations,
            procurement_flex_share=config.procurement_flex_share,
            cumulative_otg_txn=cumulative_otg_txn,
        )

        step_txns, fund_pay, direct_buy, variation_buy = clear_market_step(
            formations,
            all_otgs,
            obligations,
            compliance_agents,
            intermediary_agents,
            bct_agents,
            habitat_banks,
            supply_available,
            rng,
            price_floor=config.price_floor,
            p_variation=config.p_variation,
            cumulative_otg_txn=cumulative_otg_txn,
            bct_exact_match_only=config.bct_exact_match_only,
            rarity_multiplier=config.rarity_multiplier,
        )

        total_fund_payments += fund_pay
        total_direct_purchases += direct_buy
        total_variation_purchases += variation_buy

        # Update cumulative transaction counts (THE FEEDBACK STEP)
        for otg_name, price, qty, buyer_type in step_txns:
            cumulative_otg_txn[otg_name] = cumulative_otg_txn.get(otg_name, 0) + 1

        all_transactions.extend(step_txns)

        # Snapshot at midpoint
        if t == config.T // 2 - 1:
            midpoint_otg_txn = dict(cumulative_otg_txn)

    return {
        "transactions": all_transactions,
        "supply_final": dict(supply_available),
        "cumulative_otg_txn": dict(cumulative_otg_txn),
        "midpoint_otg_txn": midpoint_otg_txn,
        "total_fund_payments": total_fund_payments,
        "total_direct_purchases": total_direct_purchases,
        "total_variation_purchases": total_variation_purchases,
    }


# =============================================================================
# 9. METRICS
# =============================================================================


def compute_metrics(result, formations, all_otgs):
    all_transactions = result["transactions"]
    total_otgs = len(all_otgs)
    otg_txn_count: Dict[str, int] = {}
    otg_buyer_types: Dict[str, Dict[str, int]] = {}
    for otg_name, price, qty, buyer_type in all_transactions:
        otg_txn_count[otg_name] = otg_txn_count.get(otg_name, 0) + 1
        if otg_name not in otg_buyer_types:
            otg_buyer_types[otg_name] = {}
        otg_buyer_types[otg_name][buyer_type] = (
            otg_buyer_types[otg_name].get(buyer_type, 0) + 1
        )

    traded = sum(1 for o in all_otgs if otg_txn_count.get(o.name, 0) >= 1)
    ec1 = traded / total_otgs if total_otgs > 0 else 0
    functional = sum(1 for o in all_otgs if otg_txn_count.get(o.name, 0) >= 5)
    ec2 = functional / total_otgs if total_otgs > 0 else 0

    tec_otgs = [o for o in all_otgs if o.is_tec]
    non_tec_otgs = [o for o in all_otgs if not o.is_tec]
    tec_functional = sum(1 for o in tec_otgs if otg_txn_count.get(o.name, 0) >= 5)
    non_tec_functional = sum(1 for o in non_tec_otgs if otg_txn_count.get(o.name, 0) >= 5)
    ec2_tec = tec_functional / len(tec_otgs) if tec_otgs else 0
    ec2_non_tec = non_tec_functional / len(non_tec_otgs) if non_tec_otgs else 0

    formation_metrics = {}
    formations_with_functional = 0
    for f in formations:
        f_traded = sum(1 for o in f.otgs if otg_txn_count.get(o.name, 0) >= 1)
        f_functional = sum(1 for o in f.otgs if otg_txn_count.get(o.name, 0) >= 5)
        f_total_txn = sum(otg_txn_count.get(o.name, 0) for o in f.otgs)
        f_ec2 = f_functional / len(f.otgs) if f.otgs else 0
        if f_functional > 0:
            formations_with_functional += 1
        formation_metrics[f.name] = {
            "n_otgs": len(f.otgs),
            "n_tec": f.n_tec,
            "traded": f_traded,
            "functional": f_functional,
            "total_txn": f_total_txn,
            "ec2": f_ec2,
        }

    # Buyer-type breakdown
    n_by_type = {}
    for otg_name, price, qty, buyer_type in all_transactions:
        n_by_type[buyer_type] = n_by_type.get(buyer_type, 0) + 1

    # Fund routing stats
    total_obligations = (
        result["total_fund_payments"]
        + result["total_direct_purchases"]
        + result["total_variation_purchases"]
    )
    fund_rate = (
        result["total_fund_payments"] / total_obligations
        if total_obligations > 0
        else 0
    )

    # Cost metrics: total expenditure and average price by buyer type
    total_cost = sum(price * qty for _, price, qty, _ in all_transactions)
    total_qty = sum(qty for _, _, qty, _ in all_transactions)
    avg_price = total_cost / total_qty if total_qty > 0 else 0

    cost_by_type: Dict[str, float] = {}
    qty_by_type: Dict[str, float] = {}
    for _, price, qty, buyer_type in all_transactions:
        cost_by_type[buyer_type] = cost_by_type.get(buyer_type, 0) + price * qty
        qty_by_type[buyer_type] = qty_by_type.get(buyer_type, 0) + qty
    avg_price_by_type = {
        bt: cost_by_type[bt] / qty_by_type[bt] if qty_by_type[bt] > 0 else 0
        for bt in cost_by_type
    }

    # Compliance-only cost (most relevant for policy cost assessment)
    compliance_cost = cost_by_type.get("Compliance", 0)
    compliance_qty = qty_by_type.get("Compliance", 0)
    compliance_avg_price = compliance_cost / compliance_qty if compliance_qty > 0 else 0

    return {
        "ec1": ec1,
        "ec2": ec2,
        "ec2_tec": ec2_tec,
        "ec2_non_tec": ec2_non_tec,
        "n_tec_functional": tec_functional,
        "n_non_tec_functional": non_tec_functional,
        "formations_with_functional": formations_with_functional,
        "total_formations": len(formations),
        "formation_metrics": formation_metrics,
        "otg_txn_count": otg_txn_count,
        "total_transactions": len(all_transactions),
        "n_by_type": n_by_type,
        "fund_rate": fund_rate,
        "total_fund_payments": result["total_fund_payments"],
        "total_direct_purchases": result["total_direct_purchases"],
        "total_variation_purchases": result["total_variation_purchases"],
        "total_obligations": total_obligations,
        "total_cost": total_cost,
        "avg_price": avg_price,
        "compliance_cost": compliance_cost,
        "compliance_avg_price": compliance_avg_price,
        "cost_by_type": cost_by_type,
        "avg_price_by_type": avg_price_by_type,
    }


# =============================================================================
# 10. MONTE CARLO
# =============================================================================


def monte_carlo(formations, all_otgs, subregion_demands, config, n_runs=50):
    all_metrics = []
    all_cumulative = []
    all_midpoint = []

    for i in range(n_runs):
        result = run_single(formations, all_otgs, subregion_demands, config, seed=i)
        metrics = compute_metrics(result, formations, all_otgs)
        all_metrics.append(metrics)
        all_cumulative.append(result["cumulative_otg_txn"])
        all_midpoint.append(result["midpoint_otg_txn"])
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{n_runs} complete")

    ec2_vals = [m["ec2"] for m in all_metrics]
    ec2_tec_vals = [m["ec2_tec"] for m in all_metrics]
    ec2_non_tec_vals = [m["ec2_non_tec"] for m in all_metrics]
    n_func_formations = [m["formations_with_functional"] for m in all_metrics]
    n_tec_func = [m["n_tec_functional"] for m in all_metrics]
    n_non_tec_func = [m["n_non_tec_functional"] for m in all_metrics]
    fund_rates = [m["fund_rate"] for m in all_metrics]
    total_costs = [m["total_cost"] for m in all_metrics]
    avg_prices = [m["avg_price"] for m in all_metrics]
    compliance_costs = [m["compliance_cost"] for m in all_metrics]
    compliance_avg_prices = [m["compliance_avg_price"] for m in all_metrics]

    formation_ec2 = {}
    for fname in all_metrics[0]["formation_metrics"]:
        vals = [m["formation_metrics"][fname]["ec2"] for m in all_metrics]
        txn_vals = [m["formation_metrics"][fname]["total_txn"] for m in all_metrics]
        func_vals = [m["formation_metrics"][fname]["functional"] for m in all_metrics]
        formation_ec2[fname] = {
            "ec2_median": float(np.median(vals)),
            "ec2_p10": float(np.percentile(vals, 10)),
            "ec2_p90": float(np.percentile(vals, 90)),
            "txn_median": float(np.median(txn_vals)),
            "func_median": float(np.median(func_vals)),
            "n_otgs": all_metrics[0]["formation_metrics"][fname]["n_otgs"],
            "n_tec": all_metrics[0]["formation_metrics"][fname]["n_tec"],
        }

    return {
        "ec2": {
            "median": float(np.median(ec2_vals)),
            "p10": float(np.percentile(ec2_vals, 10)),
            "p90": float(np.percentile(ec2_vals, 90)),
        },
        "ec2_tec": {
            "median": float(np.median(ec2_tec_vals)),
            "p10": float(np.percentile(ec2_tec_vals, 10)),
            "p90": float(np.percentile(ec2_tec_vals, 90)),
        },
        "ec2_non_tec": {
            "median": float(np.median(ec2_non_tec_vals)),
            "p10": float(np.percentile(ec2_non_tec_vals, 10)),
            "p90": float(np.percentile(ec2_non_tec_vals, 90)),
        },
        "formations_with_functional": {
            "median": float(np.median(n_func_formations)),
            "p10": float(np.percentile(n_func_formations, 10)),
            "p90": float(np.percentile(n_func_formations, 90)),
        },
        "n_tec_functional": {"median": float(np.median(n_tec_func))},
        "n_non_tec_functional": {"median": float(np.median(n_non_tec_func))},
        "fund_rate": {
            "median": float(np.median(fund_rates)),
            "p10": float(np.percentile(fund_rates, 10)),
            "p90": float(np.percentile(fund_rates, 90)),
        },
        "formation_ec2": formation_ec2,
        "total_cost": {
            "median": float(np.median(total_costs)),
            "p10": float(np.percentile(total_costs, 10)),
            "p90": float(np.percentile(total_costs, 90)),
        },
        "avg_price": {
            "median": float(np.median(avg_prices)),
            "p10": float(np.percentile(avg_prices, 10)),
            "p90": float(np.percentile(avg_prices, 90)),
        },
        "compliance_cost": {
            "median": float(np.median(compliance_costs)),
            "p10": float(np.percentile(compliance_costs, 10)),
            "p90": float(np.percentile(compliance_costs, 90)),
        },
        "compliance_avg_price": {
            "median": float(np.median(compliance_avg_prices)),
            "p10": float(np.percentile(compliance_avg_prices, 10)),
            "p90": float(np.percentile(compliance_avg_prices, 90)),
        },
        "n_runs": n_runs,
        "all_metrics": all_metrics,
        "all_cumulative": all_cumulative,
        "all_midpoint": all_midpoint,
    }


# =============================================================================
# 11. MAIN
# =============================================================================


def main():
    global PARAMS
    t0 = time.time()

    # --- Derive parameters ---
    PARAMS = derive_all_parameters()
    observed_fund_rate = PARAMS.get("observed_fund_rate", 0.354)

    # --- Load data with formation-level matching ---
    print("[1/10] Loading OTG data with FORMATION-LEVEL MATCHING...")
    formations, all_otgs, subregion_demands, matching_report = (
        load_otg_data_formation_matching()
    )
    print(f"  Loaded {len(all_otgs)} OTGs in {len(formations)} formations")
    tec_count = sum(1 for o in all_otgs if o.is_tec)
    print(f"  TEC OTGs: {tec_count}, non-TEC: {len(all_otgs) - tec_count}")
    print(f"  Subregions with transactions: {len(subregion_demands)}")

    # =========================================================================
    # A) PARAMETER CHANGE LOG
    # =========================================================================
    print("\n")
    print("=" * 110)
    print("A) PARAMETER CHANGE LOG (old experiment_partial_l4l.py -> FINAL)")
    print("=" * 110)
    print()
    print(f"  {'Parameter':<35s} {'OLD value':<25s} {'NEW value':<25s} {'Source'}")
    print(f"  {'-'*35} {'-'*25} {'-'*25} {'-'*40}")
    print(f"  {'BCT annual budget (per agent/mo)':<35s} {'lognormal(log(40000),0.5)':<25s} "
          f"{'lognormal(log(326389),0.5)':<25s} IPART 2024-25 Discussion Paper p.7")
    print(f"  {'Production cost (log-scale)':<35s} {'lognormal(7.8, 0.5)':<25s} "
          f"{'lognormal(7.62, 0.5)':<25s} Q1 of 764 ecosystem credit prices = $2,033")
    print()
    print("  Budget derivation:")
    print("    BCT value share = 45% of total market (IPART 2024-25)")
    print("    Total market value 2022-23 = $105.1M (IPART Annual Report 2023-24)")
    print("    BCT annual spend = 0.45 * $105.1M = $47.3M")
    print("    Per agent per month = $47.3M / 12 agents / 12 months = $326,389")
    print()
    print("  Production cost derivation:")
    print("    Q1 (25th percentile) of ecosystem credit prices = $2,033")
    print("    log(2033) = 7.62")
    print("    Rationale: price below which 25% of transactions occur approximates")
    print("    the floor set by production costs (banks will not sell below cost)")
    print()

    # =========================================================================
    # MATCHING REPORT
    # =========================================================================
    print("=" * 110)
    print("FORMATION-LEVEL MATCHING REPORT")
    print("=" * 110)
    mr = matching_report
    total_txn_matched = mr["n_total"]
    print(f"\n  Total ecosystem transactions processed: {total_txn_matched}")
    print(
        f"  Direct OTG matches:      {mr['n_direct']:>4d}  "
        f"({mr['n_direct'] / total_txn_matched * 100:.1f}%)"
    )
    print(
        f"  Formation-level matches: {mr['n_formation_matched']:>4d}  "
        f"({mr['n_formation_matched'] / total_txn_matched * 100:.1f}%)"
    )
    print(
        f"  Truly unmatched:         {mr['n_unmatched']:>4d}  "
        f"({mr['n_unmatched'] / total_txn_matched * 100:.1f}%)"
    )

    # --- Observed targets ---
    print("\n[2/10] Calibration targets (from data):")
    obs_func = sum(1 for o in all_otgs if o.observed_txn_count >= 5)
    obs_tec_func = sum(1 for o in all_otgs if o.is_tec and o.observed_txn_count >= 5)
    n_non_tec = len(all_otgs) - tec_count
    obs_ntec_func = sum(1 for o in all_otgs if not o.is_tec and o.observed_txn_count >= 5)
    print(f"  EC2 overall: {obs_func}/{len(all_otgs)} = {obs_func / len(all_otgs):.1%}")
    print(f"  EC2-TEC:     {obs_tec_func}/{tec_count} = {obs_tec_func / tec_count:.1%}")
    print(f"  EC2-nonTEC:  {obs_ntec_func}/{n_non_tec} = {obs_ntec_func / n_non_tec:.1%}")
    print(f"  Fund routing rate (observed): {observed_fund_rate:.1%}")
    print(f"    ({PARAMS.get('n_bct_retirements', 388)} BCT / "
          f"{PARAMS.get('n_total_retirements', 1097)} total retirements)")

    obs_extinct = 0
    obs_func_formations = 0
    obs_extinct_forms = []
    for f in formations:
        has_func = any(o.observed_txn_count >= 5 for o in f.otgs)
        if has_func:
            obs_func_formations += 1
        has_any_txn = any(o.observed_txn_count >= 1 for o in f.otgs)
        if not has_any_txn:
            obs_extinct += 1
            obs_extinct_forms.append(f.name)
    print(f"  Functional formations: {obs_func_formations}/{len(formations)}")
    print(f"  Extinct formations (0 txn): {obs_extinct}")
    print(f"  Extinct: {obs_extinct_forms}")

    # =========================================================================
    # CALIBRATION PHASE: Sweep P_variation
    # =========================================================================
    print("\n")
    print("=" * 110)
    print("B) CALIBRATION RE-SWEEP: Verify P_variation after budget fix")
    print(f"  Target Fund routing rate: {observed_fund_rate:.1%}")
    print(f"  Sweeping P_variation in [0.3, 0.4, 0.5, 0.6, 0.7] with 20 seeds each")
    print("=" * 110)

    p_variation_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    calibration_results = []
    n_cal_runs = 20

    for pv in p_variation_values:
        print(f"\n  [CAL] P_variation = {pv:.1f} ({n_cal_runs} MC seeds)...")
        config_cal = FormationConfig(p_variation=pv)
        stats = monte_carlo(
            formations, all_otgs, subregion_demands, config_cal, n_runs=n_cal_runs
        )
        fund_rate_med = stats["fund_rate"]["median"]
        ec2_med = stats["ec2"]["median"]
        calibration_results.append({
            "p_variation": pv,
            "fund_rate": fund_rate_med,
            "fund_rate_p10": stats["fund_rate"]["p10"],
            "fund_rate_p90": stats["fund_rate"]["p90"],
            "ec2": ec2_med,
            "ec2_p10": stats["ec2"]["p10"],
            "ec2_p90": stats["ec2"]["p90"],
        })
        print(
            f"    Fund rate = {fund_rate_med:.1%} "
            f"[{stats['fund_rate']['p10']:.1%} - {stats['fund_rate']['p90']:.1%}], "
            f"EC2 = {ec2_med:.1%}"
        )

    # Print calibration table
    print("\n")
    print("=" * 110)
    print("CALIBRATION CURVE")
    print("=" * 110)
    print(
        f"  {'P_variation':>12s} | {'Fund rate':>12s} | {'Fund CI':>20s} | "
        f"{'EC2':>8s} | {'EC2 CI':>20s} | {'Selected':>8s}"
    )
    print(f"  {'-' * 12} | {'-' * 12} | {'-' * 20} | {'-' * 8} | {'-' * 20} | {'-' * 8}")

    # Find best P_variation
    best_pv = None
    best_dist = float("inf")
    for cr in calibration_results:
        dist = abs(cr["fund_rate"] - observed_fund_rate)
        if dist < best_dist:
            best_dist = dist
            best_pv = cr["p_variation"]

    for cr in calibration_results:
        selected = "<--" if cr["p_variation"] == best_pv else ""
        fund_ci = f"[{cr['fund_rate_p10']:.1%} - {cr['fund_rate_p90']:.1%}]"
        ec2_ci = f"[{cr['ec2_p10']:.1%} - {cr['ec2_p90']:.1%}]"
        print(
            f"  {cr['p_variation']:>12.1f} | {cr['fund_rate']:>11.1%} | {fund_ci:>20s} | "
            f"{cr['ec2']:>7.1%} | {ec2_ci:>20s} | {selected:>8s}"
        )

    print(f"\n  SELECTED: P_variation = {best_pv:.1f}")
    print(f"  Fund rate at selected: {[cr for cr in calibration_results if cr['p_variation'] == best_pv][0]['fund_rate']:.1%}")
    print(f"  Observed Fund rate:    {observed_fund_rate:.1%}")
    print(f"  Distance:              {best_dist:.1%}")

    # =========================================================================
    # FULL RUN: 4 scenarios x 50 MC seeds with calibrated P_variation
    # =========================================================================
    print("\n\n")
    print("=" * 110)
    print(f"FULL RUN: 4 scenarios x 50 MC seeds x 72 timesteps")
    print(f"  Calibrated P_variation = {best_pv:.1f}")
    print("=" * 110)

    scenarios = {
        "Baseline": FormationConfig(p_variation=best_pv),
        "Procurement Flex (20%)": FormationConfig(
            p_variation=best_pv, procurement_flex_share=0.20
        ),
        "Price Floor (AUD 3,000)": FormationConfig(
            p_variation=best_pv, price_floor=3000.0
        ),
        "Combined": FormationConfig(
            p_variation=best_pv, procurement_flex_share=0.20, price_floor=3000.0
        ),
        "Bypass Reduction": FormationConfig(
            p_variation=0.8  # Higher P_variation = more substitution before Fund routing
        ),
        "BCT Precision Mandate": FormationConfig(
            p_variation=best_pv, bct_exact_match_only=True
        ),
        "Rarity Multiplier (2x)": FormationConfig(
            p_variation=best_pv, rarity_multiplier=2.0
        ),
    }

    n_runs = 50
    scenario_results = {}

    for i, (name, config) in enumerate(scenarios.items()):
        print(f"\n[{i + 5}/10] Running {name} ({n_runs} MC seeds, T={config.T})...")
        stats = monte_carlo(formations, all_otgs, subregion_demands, config, n_runs=n_runs)
        scenario_results[name] = stats
        ec2 = stats["ec2"]
        fr = stats["fund_rate"]
        print(f"  EC2 = {ec2['median']:.1%} [{ec2['p10']:.1%} - {ec2['p90']:.1%}]")
        print(
            f"  EC2-TEC = {stats['ec2_tec']['median']:.1%}, "
            f"EC2-nonTEC = {stats['ec2_non_tec']['median']:.1%}"
        )
        print(f"  Fund rate = {fr['median']:.1%} [{fr['p10']:.1%} - {fr['p90']:.1%}]")
        print(
            f"  Formations with functional markets: "
            f"{stats['formations_with_functional']['median']:.0f}/{len(formations)}"
        )

    elapsed = time.time() - t0

    # =========================================================================
    # COMPREHENSIVE RESULTS
    # =========================================================================

    print("\n\n")
    print("=" * 110)
    print("COMPREHENSIVE RESULTS: FINAL DEFINITIVE ABM (all parameters traceable)")
    print(f"  P_variation = {best_pv:.1f} (calibrated to observed Fund rate = {observed_fund_rate:.1%})")
    print("=" * 110)

    bl = scenario_results["Baseline"]
    bl_ec2 = bl["ec2"]["median"]
    bl_ec2_tec = bl["ec2_tec"]["median"]

    bl_extinct = sum(
        1 for fname, fstats in bl["formation_ec2"].items() if fstats["txn_median"] == 0
    )

    bl_total_txn = np.median([m["total_transactions"] for m in bl["all_metrics"]])
    bl_monthly_txn = bl_total_txn / 72

    gw_txn = bl["formation_ec2"].get("Grassy Woodlands", {}).get("txn_median", 0)
    total_txn_all = sum(fstats["txn_median"] for fstats in bl["formation_ec2"].values())
    bl_gw_share = gw_txn / total_txn_all if total_txn_all > 0 else 0

    # --- C) Main comparison table ---
    print("\nC) MAIN COMPARISON TABLE")
    print("-" * 130)
    header = (
        f"  {'Metric':<25s} {'PL4L old':>12s} "
        f"{'FINAL (this)':>14s} {'Observed':>10s}"
    )
    print(header)
    print(
        f"  {'-' * 25} {'-' * 12} "
        f"{'-' * 14} {'-' * 10}"
    )
    print(
        f"  {'EC2 overall':<25s} {'14.3%':>12s} "
        f"{bl_ec2:>13.1%} {'17.9%':>10s}"
    )
    print(
        f"  {'EC2-TEC':<25s} {'20.3%':>12s} "
        f"{bl_ec2_tec:>13.1%} {'22.8%':>10s}"
    )
    print(
        f"  {'Formations extinct':<25s} {'--':>12s} "
        f"{bl_extinct:>14d} {'5':>10s}"
    )
    print(
        f"  {'Txn/month':<25s} {'--':>12s} "
        f"{bl_monthly_txn:>14.1f} {'12.3':>10s}"
    )
    print(
        f"  {'GW share':<25s} {'38%':>12s} "
        f"{bl_gw_share:>13.0%} {'32%':>10s}"
    )
    bl_fund_rate = bl["fund_rate"]["median"]
    print(
        f"  {'Fund routing rate':<25s} {'34.4%':>12s} "
        f"{bl_fund_rate:>13.1%} {'35%':>10s}"
    )
    print("-" * 130)

    # --- D) Scenario comparison with CIs ---
    print("\nD) SCENARIO COMPARISON TABLE (with 10th-90th CI)")
    print("-" * 110)
    print(
        f"  {'Scenario':<30s} {'EC2 [10th-90th]':>25s} "
        f"{'EC2-TEC':>10s} {'EC2-nonTEC':>12s} {'Form.Ext':>10s} "
        f"{'Fund rate':>10s} {'Txn/mo':>8s}"
    )
    print(
        f"  {'-' * 30} {'-' * 25} {'-' * 10} {'-' * 12} {'-' * 10} "
        f"{'-' * 10} {'-' * 8}"
    )

    for name, stats in scenario_results.items():
        ec2_s = stats["ec2"]
        ec2_str = f"{ec2_s['median']:.1%} [{ec2_s['p10']:.1%} - {ec2_s['p90']:.1%}]"
        ec2_tec_str = f"{stats['ec2_tec']['median']:.1%}"
        ec2_ntec_str = f"{stats['ec2_non_tec']['median']:.1%}"
        extinct = sum(
            1
            for fname, fstats in stats["formation_ec2"].items()
            if fstats["txn_median"] == 0
        )
        s_total_txn = np.median([m["total_transactions"] for m in stats["all_metrics"]])
        s_monthly = s_total_txn / 72
        fr_str = f"{stats['fund_rate']['median']:.1%}"
        print(
            f"  {name:<30s} {ec2_str:>25s} "
            f"{ec2_tec_str:>10s} {ec2_ntec_str:>12s} {extinct:>10d} "
            f"{fr_str:>10s} {s_monthly:>8.1f}"
        )
    print("-" * 110)

    # --- E) Formation-level breakdown (baseline) ---
    print("\nE) FORMATION-LEVEL BREAKDOWN (Baseline with Partial L4L)")
    print("-" * 110)
    print(
        f"  {'Formation':<55s} {'OTGs':>5s} {'TEC':>4s} "
        f"{'EC2':>7s} {'Func':>5s} {'Txn':>7s} {'% Total':>8s}"
    )
    print("-" * 110)

    sorted_forms_bl = sorted(
        bl["formation_ec2"].items(), key=lambda x: -x[1]["txn_median"]
    )
    for fname, fstats in sorted_forms_bl:
        ec2_str = f"{fstats['ec2_median']:.0%}"
        share = fstats["txn_median"] / total_txn_all * 100 if total_txn_all > 0 else 0
        print(
            f"  {fname:<55s} {fstats['n_otgs']:>5d} "
            f"{fstats['n_tec']:>4d} {ec2_str:>7s} "
            f"{fstats['func_median']:>5.0f} {fstats['txn_median']:>7.0f} "
            f"{share:>7.1f}%"
        )
    print("-" * 110)

    # Verify extinct formations
    bl_extinct_forms = [
        fname for fname, fstats in bl["formation_ec2"].items() if fstats["txn_median"] == 0
    ]
    print(f"\n  Model-extinct formations ({len(bl_extinct_forms)}): {bl_extinct_forms}")
    print(f"  Observed-extinct formations ({len(obs_extinct_forms)}): {obs_extinct_forms}")

    # Check specific formations
    wss = bl["formation_ec2"].get("Wet Sclerophyll Forests (Shrubby sub-formation)", {})
    if not wss:
        wss = bl["formation_ec2"].get("Wet Sclerophyll Forests (Shrubby Sub-formation)", {})
    wss_txn = wss.get("txn_median", 0) if wss else 0
    print(f"  Wet Sclerophyll Shrubby: txn_median = {wss_txn:.0f} (should be > 0)")

    target_extinct = ["Heathlands", "Saline Wetlands", "Arid Shrublands (Acacia sub-formation)",
                      "Arid Shrublands (Chenopod sub-formation)"]
    for te in target_extinct:
        te_stats = bl["formation_ec2"].get(te, {})
        te_txn = te_stats.get("txn_median", -1)
        status = "EXTINCT" if te_txn == 0 else f"ALIVE (txn={te_txn:.0f})"
        print(f"  {te}: {status}")

    # --- F) Qualitative tests (7) ---
    print("\nF) QUALITATIVE TESTS (all 7 must pass, extinct target = 4)")
    print("-" * 110)

    # Test 1: EC2 << 50%
    test1 = bl_ec2 < 0.50
    print(
        f"  1. EC2 << 50% (rarity selection)?               "
        f"{'PASS' if test1 else 'FAIL'}  (EC2 = {bl_ec2:.1%})"
    )

    # Test 2: Same formations extinct?
    overlap = set(bl_extinct_forms) & set(obs_extinct_forms)
    test2 = len(overlap) >= len(obs_extinct_forms) * 0.6 if obs_extinct_forms else True
    print(
        f"  2. Same formations extinct as observed?          "
        f"{'PASS' if test2 else 'FAIL'}  "
        f"(Model: {bl_extinct_forms})"
    )
    print(f"     Observed extinct: {obs_extinct_forms}")
    print(f"     Overlap: {list(overlap)}")

    # Test 3: Combined is best?
    ec2s_by_scenario = {name: stats["ec2"]["median"] for name, stats in scenario_results.items()}
    ranking = sorted(ec2s_by_scenario.keys(), key=lambda k: -ec2s_by_scenario[k])
    test3 = ranking[0] == "Combined"
    print(
        f"  3. Combined is best scenario?                    "
        f"{'PASS' if test3 else 'FAIL'}  (Best: {ranking[0]})"
    )

    # Test 4: Baseline is worst?
    test4 = ranking[-1] == "Baseline"
    print(
        f"  4. Baseline is worst scenario?                   "
        f"{'PASS' if test4 else 'FAIL'}  (Worst: {ranking[-1]})"
    )

    # Test 5: Grassy Woodlands dominates?
    test5 = bl_gw_share > 0.20
    print(
        f"  5. Grassy Woodlands dominates transactions?       "
        f"{'PASS' if test5 else 'FAIL'}  (Share = {bl_gw_share:.0%})"
    )

    # Test 6: EC2-TEC > EC2-nonTEC?
    test6 = bl["ec2_tec"]["median"] > bl["ec2_non_tec"]["median"]
    print(
        f"  6. EC2-TEC > EC2-nonTEC?                         "
        f"{'PASS' if test6 else 'FAIL'}  "
        f"({bl['ec2_tec']['median']:.1%} vs {bl['ec2_non_tec']['median']:.1%})"
    )

    # Test 7: Monthly transactions ~ 12?
    test7 = 5.0 <= bl_monthly_txn <= 25.0
    print(
        f"  7. Monthly transactions ~ 12?                    "
        f"{'PASS' if test7 else 'FAIL'}  ({bl_monthly_txn:.1f}/mo)"
    )

    all_tests = [test1, test2, test3, test4, test5, test6, test7]
    n_pass = sum(all_tests)
    print(f"\n  OVERALL: {n_pass}/7 tests passed")
    print("-" * 110)

    # --- H) Spearman correlation with observed ---
    print("\nH) SPEARMAN RANK CORRELATION (FINAL model vs observed)")
    print("-" * 110)

    formation_names = list(bl["formation_ec2"].keys())
    dd_txn_per_form = np.array(
        [bl["formation_ec2"][f]["txn_median"] for f in formation_names]
    )
    obs_txn_per_form = np.array(
        [next((fm.total_txn for fm in formations if fm.name == f), 0) for f in formation_names]
    )

    if len(formation_names) >= 3:
        rho, pval = sp_stats.spearmanr(obs_txn_per_form, dd_txn_per_form)
        print(f"  Spearman rho (observed vs model txn counts): {rho:.4f}")
        print(f"  p-value: {pval:.4e}")
        print(
            f"  Interpretation: "
            f"{'Strong' if rho > 0.7 else 'Moderate' if rho > 0.4 else 'Weak'} "
            f"rank agreement"
        )
    else:
        rho = np.nan

    print(f"\n  {'Formation':<55s} {'Obs.Txn':>8s} {'Model.Txn':>10s} {'Ratio':>8s}")
    print(f"  {'-' * 55} {'-' * 8} {'-' * 10} {'-' * 8}")
    sort_idx = np.argsort(-obs_txn_per_form)
    for i in sort_idx:
        f = formation_names[i]
        obs_v = obs_txn_per_form[i]
        mod_v = dd_txn_per_form[i]
        ratio_str = f"{mod_v / obs_v:.2f}" if obs_v > 0 else "--"
        print(f"  {f:<55s} {obs_v:>8.0f} {mod_v:>10.0f} {ratio_str:>8s}")
    print("-" * 110)

    # --- G) Fund routing analysis ---
    print("\nG) FUND ROUTING ANALYSIS (verify compliance/BCT split matches 65%/35%)")
    print("-" * 110)
    print(
        f"  {'Scenario':<30s} {'Direct':>10s} {'Variation':>10s} "
        f"{'Fund':>10s} {'Fund rate':>10s} {'Obs. rate':>10s}"
    )
    print(
        f"  {'-' * 30} {'-' * 10} {'-' * 10} "
        f"{'-' * 10} {'-' * 10} {'-' * 10}"
    )

    for name, stats in scenario_results.items():
        med_direct = np.median([m["total_direct_purchases"] for m in stats["all_metrics"]])
        med_variation = np.median([m["total_variation_purchases"] for m in stats["all_metrics"]])
        med_fund = np.median([m["total_fund_payments"] for m in stats["all_metrics"]])
        med_fr = stats["fund_rate"]["median"]
        print(
            f"  {name:<30s} {med_direct:>10.0f} {med_variation:>10.0f} "
            f"{med_fund:>10.0f} {med_fr:>9.1%} {observed_fund_rate:>9.1%}"
        )
    print("-" * 110)

    # Detailed breakdown for Baseline
    bl_direct = np.median([m["total_direct_purchases"] for m in bl["all_metrics"]])
    bl_variation = np.median([m["total_variation_purchases"] for m in bl["all_metrics"]])
    bl_fund = np.median([m["total_fund_payments"] for m in bl["all_metrics"]])
    bl_total_obl = bl_direct + bl_variation + bl_fund
    print(f"\n  Baseline obligation routing (median across {n_runs} seeds):")
    print(f"    Direct OTG match:      {bl_direct:>6.0f}  "
          f"({bl_direct / bl_total_obl * 100:.1f}%)")
    print(f"    Variation rules:       {bl_variation:>6.0f}  "
          f"({bl_variation / bl_total_obl * 100:.1f}%)")
    print(f"    Fund (BCT takes over): {bl_fund:>6.0f}  "
          f"({bl_fund / bl_total_obl * 100:.1f}%)")
    print(f"    Total obligations:     {bl_total_obl:>6.0f}")
    print()
    print(f"    Compliance share (direct + variation): "
          f"{(bl_direct + bl_variation) / bl_total_obl * 100:.1f}%  "
          f"(Observed: 65%)")
    print(f"    BCT/Fund share:                        "
          f"{bl_fund / bl_total_obl * 100:.1f}%  "
          f"(Observed: 35%)")

    # --- Like-for-like analysis ---
    print("\n  LIKE-FOR-LIKE ANALYSIS")
    print("-" * 110)

    # Buyer type breakdown
    print(f"\n  Transaction buyer type breakdown (Baseline):")
    bl_by_type = {}
    for m in bl["all_metrics"]:
        for btype, count in m["n_by_type"].items():
            bl_by_type[btype] = bl_by_type.get(btype, 0) + count
    n_seeds = len(bl["all_metrics"])
    total_bl_txn = sum(bl_by_type.values())
    print(f"  {'Buyer type':<25s} {'Total':>10s} {'Per seed':>10s} {'Share':>8s}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 8}")
    for btype in sorted(bl_by_type.keys()):
        count = bl_by_type[btype]
        per_seed = count / n_seeds
        share = count / total_bl_txn if total_bl_txn > 0 else 0
        print(f"  {btype:<25s} {count:>10d} {per_seed:>10.1f} {share:>7.1%}")

    # BCT-only OTGs analysis
    print(f"\n  BCT-only demand OTGs (in baseline):")
    bct_only_otgs = []
    for m in bl["all_metrics"]:
        for otg_name in m["otg_txn_count"]:
            pass  # just counting

    # Collect per-OTG buyer types across seeds
    otg_btype_all: Dict[str, Dict[str, int]] = {}
    for m in bl["all_metrics"]:
        for otg_name, price, qty, btype in m.get("_raw_txns", []):
            pass  # We don't have raw txns in metrics, use n_by_type at OTG level

    # Instead, count OTGs where BCT is the only buyer across seeds
    otg_has_compliance = set()
    otg_has_bct = set()
    for m in bl["all_metrics"]:
        for btype_key, btype_dict in []:
            pass
    # We can't easily get per-OTG buyer type from current metrics structure
    # Use cumulative data instead
    print(f"  (Per-OTG buyer type breakdown requires raw transaction logs; skipping)")

    # --- Liquidity trap analysis ---
    print("\n  LIQUIDITY TRAP ANALYSIS")
    print("  How many OTGs that were zero at t=36 became non-zero by t=72?")
    print("-" * 110)
    print(
        f"  {'Scenario':<30s} {'Zero@t36':>10s} {'Rescued by t72':>16s} "
        f"{'Rescue rate':>12s} {'Still trapped':>14s}"
    )
    print(
        f"  {'-' * 30} {'-' * 10} {'-' * 16} {'-' * 12} {'-' * 14}"
    )

    for name, stats in scenario_results.items():
        rescue_counts = []
        zero_at_mid_counts = []
        still_trapped_counts = []
        for run_idx in range(len(stats["all_midpoint"])):
            mid = stats["all_midpoint"][run_idx]
            end = stats["all_cumulative"][run_idx]
            zero_at_mid = sum(1 for o in all_otgs if mid.get(o.name, 0) == 0)
            rescued = sum(
                1 for o in all_otgs if mid.get(o.name, 0) == 0 and end.get(o.name, 0) > 0
            )
            still_trapped = zero_at_mid - rescued
            rescue_counts.append(rescued)
            zero_at_mid_counts.append(zero_at_mid)
            still_trapped_counts.append(still_trapped)

        med_zero = np.median(zero_at_mid_counts)
        med_rescued = np.median(rescue_counts)
        med_trapped = np.median(still_trapped_counts)
        rescue_rate = med_rescued / med_zero if med_zero > 0 else 0
        print(
            f"  {name:<30s} {med_zero:>10.0f} {med_rescued:>16.0f} "
            f"{rescue_rate:>11.1%} {med_trapped:>14.0f}"
        )
    print("-" * 110)

    # --- Cumulative txn distribution by formation ---
    print("\n  Average cumulative transaction count per formation at t=72 (Baseline)")
    print("-" * 110)

    avg_cum_txn = {}
    for otg in all_otgs:
        vals = [cum.get(otg.name, 0) for cum in bl["all_cumulative"]]
        avg_cum_txn[otg.name] = np.mean(vals)

    formation_cum = {}
    for f in formations:
        otg_avgs = [avg_cum_txn.get(o.name, 0) for o in f.otgs]
        total_cum = sum(otg_avgs)
        mean_cum = np.mean(otg_avgs) if otg_avgs else 0
        max_cum = max(otg_avgs) if otg_avgs else 0
        n_zero = sum(1 for v in otg_avgs if v < 0.5)
        formation_cum[f.name] = {
            "total": total_cum,
            "mean": mean_cum,
            "max": max_cum,
            "n_zero": n_zero,
            "n_otgs": len(f.otgs),
        }

    print(
        f"  {'Formation':<55s} {'Avg/OTG':>8s} {'Max OTG':>8s} "
        f"{'Total':>7s} {'Zero':>5s} {'OTGs':>5s}"
    )
    print(f"  {'-' * 55} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 5} {'-' * 5}")

    for fname, fc in sorted(formation_cum.items(), key=lambda x: -x[1]["total"]):
        print(
            f"  {fname:<55s} {fc['mean']:>8.1f} {fc['max']:>8.1f} "
            f"{fc['total']:>7.0f} {fc['n_zero']:>5d} {fc['n_otgs']:>5d}"
        )
    print("-" * 110)

    # --- VERDICT ---
    print("\n")
    print("=" * 110)
    if n_pass == 7:
        verdict = "ALL 7 QUALITATIVE TESTS PASSED"
    else:
        failed = [
            i + 1 for i, t in enumerate(all_tests) if not t
        ]
        verdict = f"{n_pass}/7 TESTS PASSED (failed: {failed})"
    print(f"VERDICT: {verdict}")
    print()
    print(f"  Calibrated P_variation = {best_pv:.1f}")
    print(f"  Fund routing rate:  Model = {bl_fund_rate:.1%}, Observed = {observed_fund_rate:.1%}")
    print()
    print(f"  Scenario ranking: {' > '.join(ranking)}")
    print(f"  EC2 with partial L4L:        {bl_ec2:.1%}")
    print(f"  EC2 ad-hoc:                  14.7%")
    print(f"  EC2 DD+FB (subregion match): 15.1%")
    print(f"  EC2 DD+FB+Strict L4L:        14.9%")
    print(f"  EC2 DD+FB+FormMatch:         17.1%")
    print(f"  EC2 DD+FB+L4L (strict):      7.1%")
    print(f"  EC2 observed:                17.9%")
    print()

    delta_vs_adhoc = bl_ec2 - 0.147
    delta_vs_ddfb = bl_ec2 - 0.151
    delta_vs_fm = bl_ec2 - 0.171
    delta_vs_observed = bl_ec2 - 0.179

    print(f"  Delta vs Ad-hoc:             {delta_vs_adhoc:+.1%}")
    print(f"  Delta vs DD+FB:              {delta_vs_ddfb:+.1%}")
    print(f"  Delta vs DD+FB+FormMatch:    {delta_vs_fm:+.1%}")
    print(f"  Delta vs Observed:           {delta_vs_observed:+.1%}")
    print()

    if abs(delta_vs_observed) < 0.03:
        print(
            "  INTERPRETATION: Partial like-for-like with calibrated variation rules\n"
            "  brings the model CLOSEST to observed EC2. The P_variation parameter\n"
            "  captures the realistic balance between strict like-for-like constraints\n"
            "  and NSW variation rules that allow some substitution within formations.\n"
            "  This is calibrated to ONE observable datum (35% Fund routing rate),\n"
            "  NOT ad-hoc."
        )
    elif delta_vs_observed < -0.03:
        print(
            "  INTERPRETATION: Partial like-for-like produces LOWER EC2 than observed.\n"
            "  The like-for-like constraint concentrates demand on OTGs where supply\n"
            "  exists, while obligations for OTGs without supply are routed through\n"
            "  the Fund, reinforcing rarity selection."
        )
    else:
        print(
            "  INTERPRETATION: Partial like-for-like produces HIGHER EC2 than observed.\n"
            "  The Fund routing mechanism (BCT taking over failed obligations) provides\n"
            "  an alternative pathway that partially compensates for supply gaps."
        )

    print("=" * 110)

    # =========================================================================
    # I) FULL PARAMETER TABLE — Supplementary Table 5
    # =========================================================================
    print("\n")
    print("=" * 110)
    print("I) FULL PARAMETER TABLE (Supplementary Table 5)")
    print("   Every parameter with value and EXACT source reference")
    print("=" * 110)
    print()
    print(f"  {'Parameter':<40s} {'Symbol':<12s} {'Value':<25s} {'Source'}")
    print(f"  {'-'*40} {'-'*12} {'-'*25} {'-'*50}")

    # Market structure
    print(f"  {'--- MARKET STRUCTURE ---'}")
    print(f"  {'Compliance agents':<40s} {'N_c':<12s} {'30':<25s} "
          f"{'Calibrated to mean 12.3 txn/month (NSW register)'}")
    print(f"  {'Intermediary agents':<40s} {'N_i':<12s} {'10':<25s} "
          f"{'Buy-sell pairs in transaction register'}")
    print(f"  {'BCT agents':<40s} {'N_bct':<12s} {'12':<25s} "
          f"{'BCT Annual Report 2023-24: 12 regional offices'}")
    print(f"  {'Habitat bank agents':<40s} {'N_hb':<12s} {'30':<25s} "
          f"{'NSW supply register: ~30 active banks'}")
    print(f"  {'Monthly obligations':<40s} {'lambda':<12s} {'Poisson(12)':<25s} "
          f"{'764 eco txn / 62 months = 12.3/month (register)'}")
    print(f"  {'Timesteps':<40s} {'T':<12s} {'72':<25s} "
          f"{'6 years (2019-2025 market history)'}")
    print(f"  {'Monte Carlo seeds':<40s} {'S':<12s} {'50':<25s} "
          f"{'Convergence verified at S=50'}")

    # Pre-compute derived values for table
    _theta = PARAMS.get("threshold_multiplier", 1.11)
    _slope = PARAMS.get("logistic_slope", 1.27)
    _gamma = PARAMS.get("gamma_intermediary", 2.27e-7)
    _sigma = PARAMS.get("sigma_intermediary", 2314)
    _fv = PARAMS.get("fair_value_multiplier", 1.43)
    _eta = PARAMS.get("eta", 0.39)
    _theta_str = f"{_theta:.2f}x median"
    _slope_str = f"{_slope:.2f}"
    _gamma_str = f"{_gamma:.2e}"
    _sigma_str = f"{_sigma:.0f}"
    _fv_str = f"{_fv:.2f}"
    _eta_str = f"{_eta:.2f}"
    _pv_str = f"{best_pv:.1f}"

    # Agent parameters
    print(f"\n  --- COMPLIANCE BUYER ---")
    print(f"  {'Cash endowment':<40s} {'':<12s} {'LN(log(200000), 0.5)':<25s} "
          f"Median dev. application cost (NSW Planning Portal)")
    print(f"  {'Monthly income':<40s} {'':<12s} {'LN(log(15000), 0.3)':<25s} "
          f"Offset obligation amortization schedule")
    print(f"  {'Willingness threshold':<40s} {'theta':<12s} {_theta_str:<25s} "
          f"BCT/compliance price crossover (register)")
    print(f"  {'Logistic slope':<40s} {'k':<12s} {_slope_str:<25s} "
          f"IQR of compliance transfer prices (register)")
    print(f"  {'P_variation':<40s} {'p_v':<12s} {_pv_str:<25s} "
          f"Calibrated to 35.4% Fund routing rate")

    # Intermediary parameters
    print(f"\n  --- INTERMEDIARY/SPECULATOR ---")
    print(f"  {'Cash endowment':<40s} {'':<12s} {'LN(log(100000), 0.7)':<25s} "
          f"Estimated from buy-sell volume analysis")
    print(f"  {'Monthly income':<40s} {'':<12s} {'LN(log(8000), 0.3)':<25s} "
          f"Proportional to trading volume")
    print(f"  {'CARA risk aversion':<40s} {'gamma':<12s} {_gamma_str:<25s} "
          f"Derived from buy-sell spread/volume (register)")
    print(f"  {'Price volatility (sigma)':<40s} {'sigma':<12s} {_sigma_str:<25s} "
          f"Median within-OTG log-price std * mean (register)")
    print(f"  {'Fair value multiplier':<40s} {'fv':<12s} {_fv_str:<25s} "
          f"AR(1) mean-reversion target / median price (register)")
    print(f"  {'Activity rate':<40s} {'':<12s} {'0.20':<25s} "
          f"20% of intermediaries active per timestep")

    # BCT parameters
    print(f"\n  --- BCT (PROCUREMENT-FLEXIBLE) ---")
    print(f"  {'Cash endowment':<40s} {'':<12s} {'LN(log(60000), 0.4)':<25s} "
          f"BCT operational budget per office")
    print(f"  {'Monthly income':<40s} {'':<12s} {'LN(log(3000), 0.3)':<25s} "
          f"Monthly cash flow allocation")
    print(f"  {'Annual budget (per agent/month)':<40s} {'B_bct':<12s} {'LN(log(326389), 0.5)':<25s} "
          f"IPART 2024-25: 45% of $105.1M / 12 / 12")
    print(f"  {'Bid cap':<40s} {'':<12s} {'$5,989':<25s} "
          f"95th pctl of BCT transfer prices (register)")
    print(f"  {'Thin-market preference':<40s} {'':<12s} {'U(2.0, 5.0)':<25s} "
          f"BCT mandate: priority for underserved OTGs")
    print(f"  {'Non-TEC purchase probability':<40s} {'':<12s} {'0.05':<25s} "
          f"BCT retirements: 95% TEC-associated")
    print(f"  {'Proactive activity rate':<40s} {'':<12s} {'0.15':<25s} "
          f"15% of BCT agents make proactive bids per step")

    # Habitat bank parameters
    print(f"\n  --- HABITAT BANK ---")
    print(f"  {'Cash endowment':<40s} {'':<12s} {'LN(log(250000), 0.5)':<25s} "
          f"Est. bank establishment cost")
    print(f"  {'Monthly income':<40s} {'':<12s} {'LN(log(12000), 0.3)':<25s} "
          f"Monthly operational revenue")
    print(f"  {'Production cost (log-scale)':<40s} {'c':<12s} {'LN(7.62, 0.5)':<25s} "
          f"Q1 of eco credit prices = $2,033; log(2033)=7.62")
    print(f"  {'Capacity per month':<40s} {'':<12s} {'LN(2.5, 0.8)':<25s} "
          f"Mean ~12 credits/month per bank (supply register)")
    print(f"  {'Supply elasticity':<40s} {'eta':<12s} {_eta_str:<25s} "
          f"Log-log regression: formation volume vs price")

    # Market mechanism parameters
    print(f"\n  --- MARKET MECHANISM ---")
    print(f"  {'Price noise (compliance)':<40s} {'':<12s} {'LN(0, 0.15)':<25s} "
          f"Bid price stochasticity around reference")
    print(f"  {'Price noise (intermediary)':<40s} {'':<12s} {'LN(0, 0.10)':<25s} "
          f"Narrower noise for informed traders")
    print(f"  {'Order quantity (compliance)':<40s} {'':<12s} {'LN(2.5, 0.8)':<25s} "
          f"Median ~12 credits per order (register)")
    print(f"  {'Liquidity feedback weights':<40s} {'':<12s} {'1.0/1.5/2.9/5.1/8.4':<25s} "
          f"Empirical transition probs (Table 1, register)")
    print(f"  {'Power-law concentration alpha':<40s} {'':<12s} {'4.0':<25s} "
          f"Zipf exponent for subregion OTG selection")

    # Scenario parameters
    print(f"\n  --- SCENARIOS ---")
    print(f"  {'Procurement flex share':<40s} {'':<12s} {'0.20':<25s} "
          f"Policy scenario: 20% of obligations flexible")
    print(f"  {'Price floor':<40s} {'':<12s} {'$3,000':<25s} "
          f"Policy scenario: minimum credit price")

    print()
    print("=" * 110)

    # =========================================================================
    # SAVE RESULTS JSON
    # =========================================================================
    save_results_json(
        scenario_results=scenario_results,
        formations=formations,
        all_otgs=all_otgs,
        best_pv=best_pv,
        observed_fund_rate=observed_fund_rate,
        n_runs=n_runs,
    )

    print(f"\nTotal elapsed time: {elapsed:.1f}s")
    print("DONE.")


def save_results_json(
    scenario_results,
    formations,
    all_otgs,
    best_pv,
    observed_fund_rate,
    n_runs,
):
    """Save all model results to JSON for downstream figure scripts."""
    out_dir = Path(__file__).resolve().parent.parent / "output" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    bl = scenario_results["Baseline"]

    # --- Formation shares (predicted) ---
    total_txn_all = sum(
        fstats["txn_median"] for fstats in bl["formation_ec2"].values()
    )
    formation_shares = {}
    for fname, fstats in bl["formation_ec2"].items():
        predicted_pct = (
            fstats["txn_median"] / total_txn_all * 100.0
            if total_txn_all > 0
            else 0.0
        )
        # Observed share: compute from formation objects
        obs_txn = next((f.total_txn for f in formations if f.name == fname), 0)
        obs_total = sum(f.total_txn for f in formations)
        observed_pct = obs_txn / obs_total * 100.0 if obs_total > 0 else 0.0
        formation_shares[fname] = {
            "observed_pct": round(observed_pct, 1),
            "predicted_pct": round(predicted_pct, 1),
        }

    # --- Fund routing ---
    bl_direct = float(np.median([m["total_direct_purchases"] for m in bl["all_metrics"]]))
    bl_variation = float(np.median([m["total_variation_purchases"] for m in bl["all_metrics"]]))
    bl_fund = float(np.median([m["total_fund_payments"] for m in bl["all_metrics"]]))
    bl_total_obl = bl_direct + bl_variation + bl_fund
    compliance_share = (bl_direct + bl_variation) / bl_total_obl * 100.0 if bl_total_obl > 0 else 0
    bct_share = bl_fund / bl_total_obl * 100.0 if bl_total_obl > 0 else 0

    # --- Spearman rho ---
    formation_names = list(bl["formation_ec2"].keys())
    dd_txn = np.array([bl["formation_ec2"][f]["txn_median"] for f in formation_names])
    obs_txn_arr = np.array(
        [next((fm.total_txn for fm in formations if fm.name == f), 0) for f in formation_names]
    )
    if len(formation_names) >= 3:
        rho_val, p_val = sp_stats.spearmanr(obs_txn_arr, dd_txn)
    else:
        rho_val, p_val = float("nan"), float("nan")

    # --- Baseline summary ---
    bl_total_txn = float(np.median([m["total_transactions"] for m in bl["all_metrics"]]))
    bl_monthly_txn = bl_total_txn / 72
    gw_txn = bl["formation_ec2"].get("Grassy Woodlands", {}).get("txn_median", 0)
    bl_gw_share = gw_txn / total_txn_all if total_txn_all > 0 else 0
    bl_extinct = sum(
        1 for fstats in bl["formation_ec2"].values() if fstats["txn_median"] == 0
    )

    # --- Build scenarios dict ---
    scenarios_out = {}
    for name, stats in scenario_results.items():
        ec2_s = stats["ec2"]
        s_total_txn = float(np.median([m["total_transactions"] for m in stats["all_metrics"]]))
        scenarios_out[name] = {
            "ec2_median": round(ec2_s["median"] * 100, 1),
            "ec2_p10": round(ec2_s["p10"] * 100, 1),
            "ec2_p90": round(ec2_s["p90"] * 100, 1),
            "ec2_tec": round(stats["ec2_tec"]["median"] * 100, 1),
            "ec2_ntec": round(stats["ec2_non_tec"]["median"] * 100, 1),
            "fund_rate": round(stats["fund_rate"]["median"] * 100, 1),
            "txn_per_month": round(s_total_txn / 72, 1),
            "total_cost_median": round(stats["total_cost"]["median"], 0),
            "avg_price_median": round(stats["avg_price"]["median"], 0),
            "compliance_cost_median": round(stats["compliance_cost"]["median"], 0),
            "compliance_avg_price_median": round(stats["compliance_avg_price"]["median"], 0),
        }

    results = {
        "model": "formation_abm",
        "script": "scripts/run_abm.py",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "bct_budget": 326389,
            "production_cost_mu": 7.62,
            "p_variation": best_pv,
            "mc_seeds": n_runs,
            "timesteps": 72,
        },
        "baseline": {
            "ec2_median": round(bl["ec2"]["median"] * 100, 1),
            "ec2_p10": round(bl["ec2"]["p10"] * 100, 1),
            "ec2_p90": round(bl["ec2"]["p90"] * 100, 1),
            "ec2_tec": round(bl["ec2_tec"]["median"] * 100, 1),
            "ec2_ntec": round(bl["ec2_non_tec"]["median"] * 100, 1),
            "fund_rate": round(bl["fund_rate"]["median"] * 100, 1),
            "compliance_share": round(compliance_share, 1),
            "bct_share": round(bct_share, 1),
            "txn_per_month": round(bl_monthly_txn, 1),
            "spearman_rho": round(float(rho_val), 4) if not np.isnan(rho_val) else None,
            "spearman_p": float(f"{p_val:.2e}") if not np.isnan(p_val) else None,
            "formations_extinct": bl_extinct,
            "grassy_woodlands_share": round(bl_gw_share, 2),
        },
        "scenarios": scenarios_out,
        "formation_shares": formation_shares,
        "fund_routing": {
            "model_compliance_pct": round(compliance_share, 1),
            "model_bct_pct": round(bct_share, 1),
        },
        "cost_analysis": {
            name: {
                "total_cost_median_aud": scenarios_out[name]["total_cost_median"],
                "compliance_cost_median_aud": scenarios_out[name]["compliance_cost_median"],
                "avg_price_median_aud": scenarios_out[name]["avg_price_median"],
                "compliance_avg_price_median_aud": scenarios_out[name]["compliance_avg_price_median"],
                "cost_change_vs_baseline_pct": round(
                    (scenarios_out[name]["total_cost_median"] - scenarios_out["Baseline"]["total_cost_median"])
                    / scenarios_out["Baseline"]["total_cost_median"]
                    * 100,
                    1,
                )
                if scenarios_out["Baseline"]["total_cost_median"] > 0
                else 0,
            }
            for name in scenarios_out
        },
    }

    json_path = out_dir / "abm_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
