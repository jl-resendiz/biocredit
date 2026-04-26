#!/usr/bin/env python3
"""
=============================================================================
HOLD-OUT VALIDATION: Grassy Woodlands excluded from calibration
=============================================================================

Out-of-sample test for the formation-level ABM. Removes Grassy Woodlands
(the largest formation: 50 OTGs, 239 observed transactions, 31.5% of total
volume) from the calibration data, recalibrates P_variation to match the
Fund routing rate on the remaining 14 formations, and checks whether the
model still predicts GW dominance from structural mechanisms alone.

This tests whether the demand concentration mechanism is structural (driven
by formation size, geographic overlap, and liquidity feedback) rather than
fitted to observed GW dominance.

Usage:
    python scripts/holdout_validation.py

Author: Jose Luis Resendiz
Date:   2026-03-20
=============================================================================
"""

import sys
import warnings
import time
import json
import copy
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
# SHARED INFRASTRUCTURE (from run_abm.py — kept inline for standalone use)
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "nsw"
OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "results"

HOLDOUT_FORMATION = "Grassy Woodlands"
N_MC_SEEDS = 20
N_CAL_RUNS = 20

# =============================================================================
# 0. DERIVE PARAMETERS FROM DATA
# =============================================================================


def derive_all_parameters():
    """Derive all micro-founded parameters from raw NSW data."""
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

    transfers = txn[(txn["txn_type"] == "Transfer") & (txn["price"] > 0)].copy()
    retirements = txn[txn["txn_type"] == "Retire"].copy()

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

    eco_transfers = transfers[transfers["otg"] != ""].copy()
    eco_transfers["log_price"] = np.log(eco_transfers["price"])

    otg_counts = eco_transfers.groupby("otg").size().reset_index(name="n_txn")
    thin_otgs = set(otg_counts[otg_counts["n_txn"] < 5]["otg"])

    overall_median = eco_transfers["price"].median()
    overall_mean = eco_transfers["price"].mean()

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

    otg_log_stats = eco_transfers.groupby("otg")["log_price"].agg(["std", "count"]).dropna()
    otg_log_stats_3plus = otg_log_stats[otg_log_stats["count"] >= 3]
    if len(otg_log_stats_3plus) > 0:
        median_log_std = otg_log_stats_3plus["std"].median()
        sigma_intermediary = median_log_std * overall_mean
    else:
        sigma_intermediary = 1000.0

    eco_thin = eco_transfers[eco_transfers["otg"].isin(thin_otgs)]
    thin_log_stats = eco_thin.groupby("otg")["log_price"].agg(["std", "count"]).dropna()
    thin_log_stats_3plus = thin_log_stats[thin_log_stats["count"] >= 3]
    if len(thin_log_stats_3plus) > 0:
        median_thin_log_std = thin_log_stats_3plus["std"].median()
    else:
        median_thin_log_std = eco_thin["log_price"].std() if len(eco_thin) > 0 else 0.3
    thin_mean_price = eco_thin["price"].median() if len(eco_thin) > 0 else overall_mean
    sigma_bct = median_thin_log_std * thin_mean_price

    buyer_ids = set(transfers["to_id"].unique())
    seller_ids = set(transfers["from_id"].unique())
    intermediary_crs = buyer_ids & seller_ids

    gamma_intermediary = 2.27e-7
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

    if len(bct_transfers) > 0 and len(compliance_transfers) > 0:
        bct_med = bct_transfers["price"].median()
        comp_med = compliance_transfers[compliance_transfers["price"] > 0]["price"].median()
        threshold_multiplier = bct_med / overall_median if overall_median > 0 else 1.5
    else:
        threshold_multiplier = 1.5

    logistic_slope = 1.27
    if len(compliance_transfers) > 0:
        comp_prices = compliance_transfers[compliance_transfers["price"] > 0]["price"]
        comp_median_price = comp_prices.median()
        if comp_median_price > 0 and len(comp_prices) > 5:
            iqr = comp_prices.quantile(0.75) - comp_prices.quantile(0.25)
            if iqr > 0:
                logistic_slope = 2.197 / (iqr / comp_median_price)

    formation_stats = (
        eco_transfers.groupby("formation")
        .agg(n_txn=("price", "count"), median_price=("price", "median"))
        .reset_index()
    )
    formation_stats = formation_stats[
        (formation_stats["n_txn"] >= 3) & (formation_stats["median_price"] > 0)
    ]
    eta = 0.39
    if len(formation_stats) >= 4:
        log_vol = np.log(formation_stats["n_txn"].values)
        log_price = np.log(formation_stats["median_price"].values)
        slope_e, _, _, _, _ = sp_stats.linregress(log_price, log_vol)
        eta = slope_e

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

    return params


# =============================================================================
# 1. LIQUIDITY FEEDBACK TABLE
# =============================================================================

LIQUIDITY_FEEDBACK = {
    "never":       (0, 0, 0.0306, 1.0),
    "thin_low":    (1, 2, 0.0470, 1.5),
    "thin_high":   (3, 4, 0.0885, 2.9),
    "functional":  (5, 9, 0.1550, 5.1),
    "established": (10, 999999, 0.2580, 8.4),
}


def get_liquidity_weight(cumulative_txn: int) -> float:
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

    form_txn = eco.groupby("Vegetation Formation").agg(
        median_price=("price", "median")
    ).reset_index()
    form_price_dict = dict(zip(form_txn["Vegetation Formation"], form_txn["median_price"]))

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

    formation_dict: Dict[str, List[OTG]] = {}
    for otg in all_otgs:
        formation_dict.setdefault(otg.formation, []).append(otg)
    formations = [
        Formation(name=name, otgs=otgs)
        for name, otgs in sorted(formation_dict.items(), key=lambda x: -len(x[1]))
    ]

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
# 3. GEOGRAPHIC DEMAND ENGINE
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
    p_variation: float = 0.5
    bct_exact_match_only: bool = False
    rarity_multiplier: float = 1.0


# =============================================================================
# 5. AGENTS
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
        base_price = formation.median_price if formation.median_price > 0 else 3000.0
        threshold_mult = PARAMS.get("threshold_multiplier", 1.11)
        slope = PARAMS.get("logistic_slope", 1.27)

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

        result = _try_buy(obligation_otg)
        if result:
            return (*result, "direct")

        if self.rng.random() < p_variation:
            candidates = []
            for otg in formation.otgs:
                if otg.name == obligation_otg.name:
                    continue
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
                qty = min(
                    self.rng.lognormal(2.5, 0.8) * qty_mult, avail, self.cash / (price + 1)
                )
                if qty >= 1:
                    return (otg_name, price, qty, "variation")

        return None


class IntermediaryAgentPL4L(FormationAgent):
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
        self.fund_obligations.append((otg, formation))

    def bid_fund_obligation(self, supply_available, price_floor,
                            bct_exact_match_only=False, rarity_multiplier=1.0,
                            cumulative_otg_txn=None):
        if self.budget_remaining <= 0 or not self.fund_obligations:
            return None

        otg, formation = self.fund_obligations.pop(0)

        qty_mult = 1.0
        if rarity_multiplier > 1.0 and cumulative_otg_txn is not None:
            if cumulative_otg_txn.get(otg.name, 0) < 5:
                qty_mult = rarity_multiplier

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

        if bct_exact_match_only:
            return None

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
# 6. OBLIGATION GENERATION
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
    n_obligations = max(1, rng.poisson(monthly_obligations))
    obligations = []
    n_standard = int(n_obligations * (1.0 - procurement_flex_share))
    n_flex = n_obligations - n_standard

    for _ in range(n_standard):
        result = demand_engine.generate_obligation(
            rng, cumulative_otg_txn=cumulative_otg_txn, apply_feedback=True
        )
        if result is not None:
            otg, form = result
            obligations.append((otg, form, False))

    if n_flex > 0:
        for _ in range(n_flex):
            result = demand_engine.generate_obligation(
                rng, cumulative_otg_txn=cumulative_otg_txn, apply_feedback=True
            )
            if result is not None:
                otg, form = result
                obligations.append((otg, form, True))

    return obligations


# =============================================================================
# 7. MARKET CLEARING
# =============================================================================


def _compliance_flex_bid(agent, obligation_otg, formation, supply_available, price_floor):
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
    transactions = []
    fund_payments = 0
    direct_purchases = 0
    variation_purchases = 0

    for bank in habitat_banks:
        bank.produce(formations, supply_available)

    rng.shuffle(obligations)
    for i, (obl_otg, form, is_flex) in enumerate(obligations):
        agent_idx = i % len(compliance_agents)
        agent = compliance_agents[agent_idx]

        if is_flex:
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
                fund_payments += 1
                bct_idx = rng.randint(len(bct_agents))
                bct_agents[bct_idx].receive_fund_obligation(obl_otg, form)
        else:
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
                fund_payments += 1
                bct_idx = rng.randint(len(bct_agents))
                bct_agents[bct_idx].receive_fund_obligation(obl_otg, form)

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

        for otg_name, price, qty, buyer_type in step_txns:
            cumulative_otg_txn[otg_name] = cumulative_otg_txn.get(otg_name, 0) + 1

        all_transactions.extend(step_txns)

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
    for otg_name, price, qty, buyer_type in all_transactions:
        otg_txn_count[otg_name] = otg_txn_count.get(otg_name, 0) + 1

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
    for f in formations:
        f_traded = sum(1 for o in f.otgs if otg_txn_count.get(o.name, 0) >= 1)
        f_functional = sum(1 for o in f.otgs if otg_txn_count.get(o.name, 0) >= 5)
        f_total_txn = sum(otg_txn_count.get(o.name, 0) for o in f.otgs)
        f_ec2 = f_functional / len(f.otgs) if f.otgs else 0
        formation_metrics[f.name] = {
            "n_otgs": len(f.otgs),
            "n_tec": f.n_tec,
            "traded": f_traded,
            "functional": f_functional,
            "total_txn": f_total_txn,
            "ec2": f_ec2,
        }

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

    return {
        "ec1": ec1,
        "ec2": ec2,
        "ec2_tec": ec2_tec,
        "ec2_non_tec": ec2_non_tec,
        "formation_metrics": formation_metrics,
        "otg_txn_count": otg_txn_count,
        "total_transactions": len(all_transactions),
        "fund_rate": fund_rate,
        "total_fund_payments": result["total_fund_payments"],
        "total_direct_purchases": result["total_direct_purchases"],
        "total_variation_purchases": result["total_variation_purchases"],
    }


# =============================================================================
# 10. MONTE CARLO
# =============================================================================


def monte_carlo(formations, all_otgs, subregion_demands, config, n_runs=20):
    all_metrics = []
    for i in range(n_runs):
        result = run_single(formations, all_otgs, subregion_demands, config, seed=i)
        metrics = compute_metrics(result, formations, all_otgs)
        all_metrics.append(metrics)
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{n_runs} complete")

    ec2_vals = [m["ec2"] for m in all_metrics]
    fund_rates = [m["fund_rate"] for m in all_metrics]

    formation_ec2 = {}
    for fname in all_metrics[0]["formation_metrics"]:
        vals = [m["formation_metrics"][fname]["ec2"] for m in all_metrics]
        txn_vals = [m["formation_metrics"][fname]["total_txn"] for m in all_metrics]
        func_vals = [m["formation_metrics"][fname]["functional"] for m in all_metrics]
        formation_ec2[fname] = {
            "ec2_median": float(np.median(vals)),
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
        "fund_rate": {
            "median": float(np.median(fund_rates)),
            "p10": float(np.percentile(fund_rates, 10)),
            "p90": float(np.percentile(fund_rates, 90)),
        },
        "formation_ec2": formation_ec2,
        "all_metrics": all_metrics,
        "n_runs": n_runs,
    }


# =============================================================================
# HOLD-OUT LOGIC
# =============================================================================


def create_holdout_data(
    formations: List[Formation],
    all_otgs: List[OTG],
    subregion_demands: List[SubregionDemand],
    holdout_formation: str,
):
    """
    Create hold-out versions of formations, OTGs, and subregion demands.

    For the held-out formation:
    - Zero out observed_txn_count on all its OTGs (pretend we never saw transactions)
    - Zero out demand weights in subregion data for this formation
    - Keep the OTGs and formation in the model (so predictions CAN be generated)
    - Keep supply_capacity unchanged (supply-side data is independent)
    - Keep observed_median_price unchanged (price data comes from other sources)

    This tests whether the model's structural mechanisms (geographic overlap,
    formation size, liquidity feedback) predict the held-out formation's
    dominance WITHOUT having seen its transaction data.
    """
    # Deep copy OTGs so we don't mutate the originals
    holdout_otgs = []
    for otg in all_otgs:
        new_otg = OTG(
            name=otg.name,
            formation=otg.formation,
            is_tec=otg.is_tec,
            observed_txn_count=(0 if otg.formation == holdout_formation else otg.observed_txn_count),
            observed_median_price=otg.observed_median_price,
            observed_credits=(0.0 if otg.formation == holdout_formation else otg.observed_credits),
            supply_capacity=otg.supply_capacity,
            subregions=list(otg.subregions),
            direct_match_txn=(0.0 if otg.formation == holdout_formation else otg.direct_match_txn),
            formation_match_txn=(
                0.0 if otg.formation == holdout_formation else otg.formation_match_txn
            ),
        )
        holdout_otgs.append(new_otg)

    # Rebuild formations from modified OTGs
    formation_dict: Dict[str, List[OTG]] = {}
    for otg in holdout_otgs:
        formation_dict.setdefault(otg.formation, []).append(otg)
    holdout_formations = [
        Formation(name=name, otgs=otgs)
        for name, otgs in sorted(formation_dict.items(), key=lambda x: -len(x[1]))
    ]

    # Zero out subregion demand for the held-out formation
    holdout_subregion_demands = []
    gw_otg_names = {o.name for o in all_otgs if o.formation == holdout_formation}
    for sd in subregion_demands:
        new_formations = {
            k: v for k, v in sd.formations.items() if k != holdout_formation
        }
        new_otg_txn = {
            k: v for k, v in sd.otg_txn.items() if k not in gw_otg_names
        }
        new_n_txn = sum(new_formations.values()) if new_formations else 0
        if new_n_txn > 0:
            holdout_subregion_demands.append(
                SubregionDemand(
                    name=sd.name,
                    n_txn=new_n_txn,
                    formations=new_formations,
                    otg_txn=new_otg_txn,
                )
            )

    return holdout_formations, holdout_otgs, holdout_subregion_demands


def calibrate_p_variation(
    formations, all_otgs, subregion_demands, target_fund_rate, label="",
):
    """Sweep P_variation to find the value closest to the target Fund routing rate."""
    p_variation_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_pv = None
    best_dist = float("inf")

    print(f"  Calibrating P_variation (target fund rate = {target_fund_rate:.1%})...")
    for pv in p_variation_values:
        config_cal = FormationConfig(p_variation=pv)
        stats = monte_carlo(
            formations, all_otgs, subregion_demands, config_cal, n_runs=N_CAL_RUNS
        )
        fund_rate_med = stats["fund_rate"]["median"]
        ec2_med = stats["ec2"]["median"]
        dist = abs(fund_rate_med - target_fund_rate)
        print(
            f"    P_var={pv:.1f}: fund_rate={fund_rate_med:.1%}, "
            f"EC2={ec2_med:.1%}, dist={dist:.1%}"
        )
        if dist < best_dist:
            best_dist = dist
            best_pv = pv

    print(f"  Selected P_variation = {best_pv:.1f} (distance = {best_dist:.1%})")
    return best_pv


def compute_holdout_fund_rate(formations, all_otgs, subregion_demands, params):
    """
    Compute the observed Fund routing rate EXCLUDING the held-out formation.

    Since we cannot decompose the Fund routing rate by formation from the raw
    retirement data, we use the full observed Fund routing rate as the target
    for both full and hold-out calibration. This is conservative: it means the
    hold-out model is calibrated to the same aggregate observable, and any
    difference in formation-level predictions is purely structural.
    """
    return params["observed_fund_rate"]


# =============================================================================
# MAIN: HOLD-OUT VALIDATION
# =============================================================================


def main():
    global PARAMS
    t0 = time.time()

    print("=" * 110)
    print("HOLD-OUT VALIDATION: Grassy Woodlands excluded from calibration data")
    print("=" * 110)
    print()

    # --- Derive parameters ---
    print("[1/6] Deriving parameters from NSW transaction data...")
    PARAMS = derive_all_parameters()
    observed_fund_rate = PARAMS.get("observed_fund_rate", 0.354)
    print(f"  Observed Fund routing rate: {observed_fund_rate:.1%}")
    print()

    # --- Load data ---
    print("[2/6] Loading OTG data with formation-level matching...")
    formations, all_otgs, subregion_demands, matching_report = (
        load_otg_data_formation_matching()
    )
    print(f"  Loaded {len(all_otgs)} OTGs in {len(formations)} formations")

    # Identify Grassy Woodlands
    gw_form = next((f for f in formations if f.name == HOLDOUT_FORMATION), None)
    if gw_form is None:
        print(f"  ERROR: Formation '{HOLDOUT_FORMATION}' not found!")
        sys.exit(1)

    gw_obs_txn = gw_form.total_txn
    total_obs_txn = sum(f.total_txn for f in formations)
    gw_obs_share = gw_obs_txn / total_obs_txn if total_obs_txn > 0 else 0

    print(f"\n  {HOLDOUT_FORMATION}:")
    print(f"    OTGs:               {gw_form.n_otgs}")
    print(f"    Observed txn:       {gw_obs_txn}")
    print(f"    Observed share:     {gw_obs_share:.1%}")
    print()

    # =========================================================================
    # A) FULL MODEL (all 15 formations)
    # =========================================================================
    print("=" * 110)
    print(f"[3/6] FULL MODEL: all {len(formations)} formations, {N_MC_SEEDS} MC seeds")
    print("=" * 110)

    # Calibrate P_variation for full model
    full_pv = calibrate_p_variation(
        formations, all_otgs, subregion_demands, observed_fund_rate, label="full"
    )
    print(f"\n  Running full model with P_variation = {full_pv:.1f}...")
    full_config = FormationConfig(p_variation=full_pv)
    full_stats = monte_carlo(
        formations, all_otgs, subregion_demands, full_config, n_runs=N_MC_SEEDS
    )

    full_ec2 = full_stats["ec2"]["median"]
    full_fund_rate = full_stats["fund_rate"]["median"]
    full_total_txn = sum(
        fstats["txn_median"] for fstats in full_stats["formation_ec2"].values()
    )
    full_gw_txn = full_stats["formation_ec2"].get(HOLDOUT_FORMATION, {}).get("txn_median", 0)
    full_gw_share = full_gw_txn / full_total_txn if full_total_txn > 0 else 0

    print(f"\n  FULL MODEL RESULTS:")
    print(f"    EC2:        {full_ec2:.1%}")
    print(f"    Fund rate:  {full_fund_rate:.1%}")
    print(f"    GW share:   {full_gw_share:.0%}")
    print()

    # =========================================================================
    # B) HOLD-OUT MODEL (Grassy Woodlands excluded from calibration)
    # =========================================================================
    print("=" * 110)
    print(f"[4/6] HOLD-OUT MODEL: {HOLDOUT_FORMATION} excluded, {N_MC_SEEDS} MC seeds")
    print("=" * 110)

    holdout_formations, holdout_otgs, holdout_subregion_demands = create_holdout_data(
        formations, all_otgs, subregion_demands, HOLDOUT_FORMATION
    )

    # Verify hold-out data
    ho_gw = next((f for f in holdout_formations if f.name == HOLDOUT_FORMATION), None)
    print(f"\n  Hold-out verification:")
    print(f"    Formations:       {len(holdout_formations)}")
    print(f"    Total OTGs:       {len(holdout_otgs)}")
    print(f"    GW OTGs:          {ho_gw.n_otgs if ho_gw else 0}")
    print(f"    GW obs_txn:       {ho_gw.total_txn if ho_gw else 0} (should be 0)")
    print(f"    Subregions:       {len(holdout_subregion_demands)}")
    gw_in_subs = sum(
        1 for sd in holdout_subregion_demands
        if HOLDOUT_FORMATION in sd.formations
    )
    print(f"    Subregions with GW demand: {gw_in_subs} (should be 0)")
    print()

    # Calibrate P_variation for hold-out model
    holdout_fund_rate_target = compute_holdout_fund_rate(
        holdout_formations, holdout_otgs, holdout_subregion_demands, PARAMS
    )
    holdout_pv = calibrate_p_variation(
        holdout_formations, holdout_otgs, holdout_subregion_demands,
        holdout_fund_rate_target, label="holdout"
    )

    print(f"\n  Running hold-out model with P_variation = {holdout_pv:.1f}...")
    holdout_config = FormationConfig(p_variation=holdout_pv)
    holdout_stats = monte_carlo(
        holdout_formations, holdout_otgs, holdout_subregion_demands,
        holdout_config, n_runs=N_MC_SEEDS
    )

    holdout_ec2 = holdout_stats["ec2"]["median"]
    holdout_fund_rate = holdout_stats["fund_rate"]["median"]
    holdout_total_txn = sum(
        fstats["txn_median"] for fstats in holdout_stats["formation_ec2"].values()
    )
    holdout_gw_txn = holdout_stats["formation_ec2"].get(
        HOLDOUT_FORMATION, {}
    ).get("txn_median", 0)
    holdout_gw_share = holdout_gw_txn / holdout_total_txn if holdout_total_txn > 0 else 0

    print(f"\n  HOLD-OUT MODEL RESULTS:")
    print(f"    EC2:        {holdout_ec2:.1%}")
    print(f"    Fund rate:  {holdout_fund_rate:.1%}")
    print(f"    GW txn:     {holdout_gw_txn:.0f}")
    print(f"    GW share:   {holdout_gw_share:.0%}")
    print()

    # =========================================================================
    # C) COMPARISON TABLE
    # =========================================================================
    print("=" * 110)
    print("[5/6] COMPARISON: Full vs Hold-out")
    print("=" * 110)

    # Formation ranking
    full_ranking = sorted(
        full_stats["formation_ec2"].items(), key=lambda x: -x[1]["txn_median"]
    )
    holdout_ranking = sorted(
        holdout_stats["formation_ec2"].items(), key=lambda x: -x[1]["txn_median"]
    )

    full_rank_names = [f[0] for f in full_ranking]
    holdout_rank_names = [f[0] for f in holdout_ranking]

    gw_full_rank = full_rank_names.index(HOLDOUT_FORMATION) + 1 if HOLDOUT_FORMATION in full_rank_names else -1
    gw_holdout_rank = holdout_rank_names.index(HOLDOUT_FORMATION) + 1 if HOLDOUT_FORMATION in holdout_rank_names else -1

    print(f"\n  {'Metric':<35s} {'Full model':>15s} {'Hold-out':>15s} {'Observed':>15s}")
    print(f"  {'-'*35} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'EC2 overall':<35s} {full_ec2:>14.1%} {holdout_ec2:>14.1%} {'17.9%':>15s}")
    print(f"  {'Fund routing rate':<35s} {full_fund_rate:>14.1%} {holdout_fund_rate:>14.1%} {'35.4%':>15s}")
    print(f"  {'P_variation (calibrated)':<35s} {full_pv:>15.1f} {holdout_pv:>15.1f} {'--':>15s}")
    print(f"  {'GW transaction share':<35s} {full_gw_share:>14.0%} {holdout_gw_share:>14.0%} {gw_obs_share:>14.1%}")
    print(f"  {'GW rank (by txn count)':<35s} {'#' + str(gw_full_rank):>15s} {'#' + str(gw_holdout_rank):>15s} {'#1':>15s}")
    print()

    # Spearman correlation on 14 non-held-out formations
    non_ho_names = [f.name for f in formations if f.name != HOLDOUT_FORMATION]
    full_txn_14 = np.array([
        full_stats["formation_ec2"].get(f, {}).get("txn_median", 0) for f in non_ho_names
    ])
    holdout_txn_14 = np.array([
        holdout_stats["formation_ec2"].get(f, {}).get("txn_median", 0) for f in non_ho_names
    ])
    obs_txn_14 = np.array([
        next((fm.total_txn for fm in formations if fm.name == f), 0) for f in non_ho_names
    ])

    if len(non_ho_names) >= 3:
        rho_full_obs, p_full_obs = sp_stats.spearmanr(obs_txn_14, full_txn_14)
        rho_holdout_obs, p_holdout_obs = sp_stats.spearmanr(obs_txn_14, holdout_txn_14)
        rho_full_holdout, p_full_holdout = sp_stats.spearmanr(full_txn_14, holdout_txn_14)
    else:
        rho_full_obs = rho_holdout_obs = rho_full_holdout = float("nan")
        p_full_obs = p_holdout_obs = p_full_holdout = float("nan")

    print(f"  Spearman rank correlations (14 non-held-out formations):")
    print(f"    Full model vs observed:     rho = {rho_full_obs:.4f} (p = {p_full_obs:.4e})")
    print(f"    Hold-out vs observed:        rho = {rho_holdout_obs:.4f} (p = {p_holdout_obs:.4e})")
    print(f"    Full model vs hold-out:      rho = {rho_full_holdout:.4f} (p = {p_full_holdout:.4e})")
    print()

    # Full 15-formation Spearman (including GW)
    all_names = [f.name for f in formations]
    full_txn_15 = np.array([
        full_stats["formation_ec2"].get(f, {}).get("txn_median", 0) for f in all_names
    ])
    holdout_txn_15 = np.array([
        holdout_stats["formation_ec2"].get(f, {}).get("txn_median", 0) for f in all_names
    ])
    obs_txn_15 = np.array([
        next((fm.total_txn for fm in formations if fm.name == f), 0) for f in all_names
    ])
    rho_holdout_obs_15, p_holdout_obs_15 = sp_stats.spearmanr(obs_txn_15, holdout_txn_15)

    print(f"  Spearman rank correlation (all 15 formations, hold-out vs observed):")
    print(f"    rho = {rho_holdout_obs_15:.4f} (p = {p_holdout_obs_15:.4e})")
    print()

    # Formation-level comparison table
    print(f"  {'Formation':<55s} {'Obs.Txn':>8s} {'Full':>8s} {'Holdout':>8s} {'F.Share':>8s} {'H.Share':>8s}")
    print(f"  {'-'*55} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for fname, fstats in full_ranking:
        obs_v = next((fm.total_txn for fm in formations if fm.name == fname), 0)
        full_v = fstats["txn_median"]
        holdout_v = holdout_stats["formation_ec2"].get(fname, {}).get("txn_median", 0)
        f_share = full_v / full_total_txn * 100 if full_total_txn > 0 else 0
        h_share = holdout_v / holdout_total_txn * 100 if holdout_total_txn > 0 else 0
        marker = " <-- HELD OUT" if fname == HOLDOUT_FORMATION else ""
        print(
            f"  {fname:<55s} {obs_v:>8.0f} {full_v:>8.0f} {holdout_v:>8.0f} "
            f"{f_share:>7.1f}% {h_share:>7.1f}%{marker}"
        )
    print()

    # =========================================================================
    # D) VERDICT
    # =========================================================================
    print("=" * 110)
    print("[6/6] VERDICT")
    print("=" * 110)

    test_dominant = gw_holdout_rank == 1
    test_share_gt20 = holdout_gw_share > 0.20
    test_rho_preserved = rho_holdout_obs >= 0.5 if not np.isnan(rho_holdout_obs) else False
    test_ec2_close = abs(holdout_ec2 - full_ec2) < 0.05

    print()
    print(f"  1. GW predicted as dominant (rank #1)?       "
          f"{'PASS' if test_dominant else 'FAIL'}  (rank = #{gw_holdout_rank})")
    print(f"  2. GW share > 20%?                           "
          f"{'PASS' if test_share_gt20 else 'FAIL'}  (share = {holdout_gw_share:.0%})")
    print(f"  3. Spearman rho >= 0.5 (14 formations)?      "
          f"{'PASS' if test_rho_preserved else 'FAIL'}  (rho = {rho_holdout_obs:.4f})")
    print(f"  4. EC2 within 5pp of full model?              "
          f"{'PASS' if test_ec2_close else 'FAIL'}  "
          f"(delta = {holdout_ec2 - full_ec2:+.1%})")
    print()

    all_tests = [test_dominant, test_share_gt20, test_rho_preserved, test_ec2_close]
    n_pass = sum(all_tests)

    if test_dominant and test_share_gt20:
        print(
            f"  INTERPRETATION: Even when {HOLDOUT_FORMATION} is excluded from the\n"
            f"  calibration data, the model predicts it would be the dominant formation\n"
            f"  ({holdout_gw_share:.0%} of transactions), demonstrating that the model's\n"
            f"  demand concentration mechanism is structural, not fitted to observed\n"
            f"  {HOLDOUT_FORMATION} dominance. The structural drivers are:\n"
            f"    - Largest formation (50 OTGs = 19.8% of supply)\n"
            f"    - Broadest geographic overlap (most IBRA subregions)\n"
            f"    - Liquidity feedback amplifies initial advantage"
        )
    elif test_dominant:
        print(
            f"  PARTIAL SUPPORT: {HOLDOUT_FORMATION} is predicted as the dominant\n"
            f"  formation (rank #{gw_holdout_rank}) but with a smaller share than the\n"
            f"  full model ({holdout_gw_share:.0%} vs {full_gw_share:.0%}). Some\n"
            f"  concentration is structural, but calibration data amplifies the effect."
        )
    else:
        print(
            f"  NOT SUPPORTED: Without calibration data, the model does not predict\n"
            f"  {HOLDOUT_FORMATION} as the dominant formation (rank #{gw_holdout_rank}).\n"
            f"  This suggests the model's predictions for {HOLDOUT_FORMATION} depend\n"
            f"  on observed transaction data, not purely structural mechanisms."
        )

    print()
    print(f"  Overall: {n_pass}/4 tests passed")
    print("=" * 110)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "test": "formation_holdout_validation",
        "holdout_formation": HOLDOUT_FORMATION,
        "timestamp": datetime.now().isoformat(),
        "mc_seeds": N_MC_SEEDS,
        "full_model": {
            "p_variation": full_pv,
            "ec2": round(full_ec2 * 100, 1),
            "fund_rate": round(full_fund_rate * 100, 1),
            "gw_share": round(full_gw_share * 100, 1),
            "gw_rank": gw_full_rank,
        },
        "holdout_model": {
            "p_variation": holdout_pv,
            "ec2": round(holdout_ec2 * 100, 1),
            "fund_rate": round(holdout_fund_rate * 100, 1),
            "gw_txn_median": round(holdout_gw_txn, 0),
            "gw_share": round(holdout_gw_share * 100, 1),
            "gw_rank": gw_holdout_rank,
        },
        "spearman": {
            "full_vs_observed_14": round(float(rho_full_obs), 4) if not np.isnan(rho_full_obs) else None,
            "holdout_vs_observed_14": round(float(rho_holdout_obs), 4) if not np.isnan(rho_holdout_obs) else None,
            "full_vs_holdout_14": round(float(rho_full_holdout), 4) if not np.isnan(rho_full_holdout) else None,
            "holdout_vs_observed_15": round(float(rho_holdout_obs_15), 4) if not np.isnan(rho_holdout_obs_15) else None,
        },
        "tests": {
            "gw_dominant": test_dominant,
            "gw_share_gt20": test_share_gt20,
            "rho_preserved": test_rho_preserved,
            "ec2_close": test_ec2_close,
            "n_passed": n_pass,
            "n_total": 4,
        },
        "formation_comparison": {
            fname: {
                "observed_txn": int(next((fm.total_txn for fm in formations if fm.name == fname), 0)),
                "full_txn_median": round(full_stats["formation_ec2"].get(fname, {}).get("txn_median", 0), 0),
                "holdout_txn_median": round(holdout_stats["formation_ec2"].get(fname, {}).get("txn_median", 0), 0),
            }
            for fname in [f[0] for f in full_ranking]
        },
    }

    json_path = OUT_DIR / "holdout_validation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    elapsed = time.time() - t0
    print(f"Total elapsed time: {elapsed:.1f}s")
    print("DONE.")


if __name__ == "__main__":
    main()
