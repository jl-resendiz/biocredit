#!/usr/bin/env python3
"""
=============================================================================
SENSITIVITY SWEEPS: Agent counts, production cost, P_variation
=============================================================================

Tests robustness of the final formation ABM to its two weakest parameters
(agent counts and production cost) plus a re-verification of P_variation.

Sweep 1: Agent counts (n_compliance, n_intermediary, n_bct, n_habitat_bank)
Sweep 2: Production cost (P5, P10, Q1=current, Median)
Sweep 3: P_variation (re-verification)

20 MC seeds per configuration, 72 timesteps, Baseline scenario only.
Total: 38 configurations x 20 seeds = 760 runs.

Author: Jose Luis Resendiz
Date:   2026-03-19
=============================================================================
"""

import sys
import warnings
import time
import json
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Import the entire simulation engine from run_abm.py
# We add the scripts directory to the path and import the module.
# ---------------------------------------------------------------------------
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

# We cannot directly import run_abm because it calls main() at module
# level (if __name__ == "__main__"), so we import the pieces we need by exec-ing
# the file up to the main() call. Instead, let's just re-import the needed
# components by running the module's code directly.

# Actually, run_abm.py has a `if __name__ == "__main__"` guard at the
# bottom, so we can import it safely as a module.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "run_abm", SCRIPTS_DIR / "run_abm.py"
)
ef = importlib.util.module_from_spec(spec)

# We need to suppress the auto-run. Let's check if main is guarded.
# Reading the file showed: the file ends with main() call presumably in
# if __name__ == "__main__" block. Let's verify quickly.

# Actually from what I read, the file defines main() but I didn't see the
# if __name__ guard. Let me just copy the needed pieces directly to be safe.
# This avoids running the full experiment when we import.

# ---------------------------------------------------------------------------
# We'll copy the essential simulation machinery inline to avoid import issues.
# This is the cleanest approach given the script structure.
# ---------------------------------------------------------------------------

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "nsw"

# =============================================================================
# COPY OF ESSENTIAL CODE FROM run_abm.py
# (Parameters, data loading, agents, simulation, metrics)
# =============================================================================

# --- Liquidity feedback ---
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


FORMATION_NORMALIZE = {"Grassy woodlands": "Grassy Woodlands"}


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
    def n_otgs(self):
        return len(self.otgs)

    @property
    def n_tec(self):
        return sum(1 for o in self.otgs if o.is_tec)

    @property
    def n_non_tec(self):
        return sum(1 for o in self.otgs if not o.is_tec)

    @property
    def total_txn(self):
        return sum(o.observed_txn_count for o in self.otgs)

    @property
    def median_price(self):
        prices = [o.observed_median_price for o in self.otgs if o.observed_median_price > 0]
        return float(np.median(prices)) if prices else 3000.0


@dataclass
class SubregionDemand:
    name: str
    n_txn: int
    formations: Dict[str, int]
    otg_txn: Dict[str, int]


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
    production_cost_log_mean: float = 7.62  # NEW: allow override
    production_cost_log_std: float = 0.5
    bct_budget_log_mean: float = np.log(326389)  # NEW: allow override


PARAMS: Dict[str, float] = {}


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


# --- Data loading (formation matching) ---
def load_otg_data_formation_matching():
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
    otg_formation_map = {}
    otg_capacity_map = {}
    formation_otgs_map = {}
    for _, row in otg_info.iterrows():
        otg_name = row["Offset Trading Group"]
        form_name = row["formation"]
        cap = float(row["supply_capacity"])
        otg_formation_map[otg_name] = form_name
        otg_capacity_map[otg_name] = cap
        formation_otgs_map.setdefault(form_name, []).append(otg_name)

    otg_subregion_map = {}
    for _, row in supply_eco.iterrows():
        otg_name = row["Offset Trading Group"]
        sub = row.get("IBRA Subregion", "")
        if pd.notna(sub) and str(sub).strip():
            otg_subregion_map.setdefault(otg_name, set()).add(str(sub).strip())

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

    otg_txn_assigned = {}
    otg_prices = {}
    otg_credits_assigned = {}
    otg_direct_count = {}
    otg_formation_count = {}

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

    # Subregion demand data
    subregion_data = {}
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

    all_otgs = []
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

    formation_dict = {}
    for otg in all_otgs:
        formation_dict.setdefault(otg.formation, []).append(otg)
    formations = [
        Formation(name=name, otgs=otgs)
        for name, otgs in sorted(formation_dict.items(), key=lambda x: -len(x[1]))
    ]

    return formations, all_otgs, subregion_demands


# --- Geographic demand engine ---
def _power_law_weights(counts, alpha=2.0):
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

        self.sub_form_otgs = {}
        for otg in all_otgs:
            for sub in otg.subregions:
                if sub not in self.sub_form_otgs:
                    self.sub_form_otgs[sub] = {}
                form = otg.formation
                if form not in self.sub_form_otgs[sub]:
                    self.sub_form_otgs[sub][form] = []
                self.sub_form_otgs[sub][form].append(otg)

        self.sub_form_weights = {}
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

        self.sub_otg_txn = {}
        for sd in self.subregions:
            self.sub_otg_txn[sd.name] = sd.otg_txn

    def generate_obligation(self, rng, cumulative_otg_txn=None, apply_feedback=True):
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


# --- Agents ---
class FormationAgent:
    _counter = 0

    def __init__(self, agent_type, cash, rng):
        FormationAgent._counter += 1
        self.id = FormationAgent._counter
        self.agent_type = agent_type
        self.cash = cash
        self.rng = rng
        self.transactions = []


class ComplianceAgentPL4L(FormationAgent):
    def __init__(self, cash, rng):
        super().__init__("Compliance", cash, rng)

    def bid_l4l(self, obligation_otg, formation, supply_available, price_floor,
                p_variation, cumulative_otg_txn=None):
        base_price = formation.median_price if formation.median_price > 0 else 3000.0
        threshold_mult = PARAMS.get("threshold_multiplier", 1.11)
        slope = PARAMS.get("logistic_slope", 1.27)

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
            qty = min(self.rng.lognormal(2.5, 0.8), avail, self.cash / (ask + 1))
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
                qty = min(self.rng.lognormal(2.5, 0.8), avail, self.cash / (price + 1))
                if qty >= 1:
                    return (otg_name, price, qty, "variation")
        return None


class IntermediaryAgentPL4L(FormationAgent):
    def __init__(self, cash, rng):
        super().__init__("Intermediary", cash, rng)

    def bid(self, formations, supply_available, price_floor, cumulative_otg_txn=None):
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
        fair_value = base_price * fv_mult * self.rng.lognormal(0, 0.1)
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
    NON_TEC_PURCHASE_PROB = 0.05

    def __init__(self, cash, rng, budget_log_mean=np.log(326389)):
        super().__init__("BCT", cash, rng)
        self.annual_budget = rng.lognormal(budget_log_mean, 0.5)
        self.budget_remaining = self.annual_budget
        self.thin_market_preference = rng.uniform(2.0, 5.0)
        self.fund_obligations = []

    def reset_budget(self):
        self.budget_remaining = self.annual_budget

    def receive_fund_obligation(self, otg, formation):
        self.fund_obligations.append((otg, formation))

    def bid_fund_obligation(self, supply_available, price_floor):
        if self.budget_remaining <= 0 or not self.fund_obligations:
            return None
        otg, formation = self.fund_obligations.pop(0)
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
            target_qty = self.rng.lognormal(1.5, 0.8)
            max_affordable = self.budget_remaining / (bid_price + 1e-10)
            max_from_cash = self.cash / (bid_price + 1) * 0.4
            qty = min(target_qty, avail, max_affordable, max_from_cash)
            if qty >= 1:
                return (otg.name, bid_price, qty, "BCT-Fund")

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
        target_qty = self.rng.lognormal(1.5, 0.8)
        max_affordable = self.budget_remaining / (bid_price + 1e-10)
        max_from_cash = self.cash / (bid_price + 1) * 0.4
        qty = min(target_qty, chosen_avail, max_affordable, max_from_cash)
        if qty >= 1:
            return (chosen_otg.name, bid_price, qty, "BCT-Fund-Flex")
        return None

    def bid_proactive(self, formations, supply_available, price_floor):
        if self.budget_remaining <= 0:
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
    def __init__(self, cash, rng, formations, prod_cost_log_mean=7.62, prod_cost_log_std=0.5):
        super().__init__("HabitatBank", cash, rng)
        n_specialise = rng.randint(1, min(4, len(formations) + 1))
        form_weights = np.array([max(1, f.total_txn) for f in formations], dtype=float)
        form_weights /= form_weights.sum()
        self.specialisation_indices = rng.choice(
            len(formations), size=n_specialise, replace=False, p=form_weights
        )
        self.production_cost = rng.lognormal(prod_cost_log_mean, prod_cost_log_std)
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


# --- Obligation generation ---
def generate_obligations(demand_engine, formations, all_otgs, rng,
                         monthly_obligations=12.0, procurement_flex_share=0.0,
                         cumulative_otg_txn=None):
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


# --- Compliance flex bid ---
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


# --- Market clearing ---
def clear_market_step(formations, all_otgs, obligations, compliance_agents,
                      intermediary_agents, bct_agents, habitat_banks,
                      supply_available, rng, price_floor=0.0, p_variation=0.5,
                      cumulative_otg_txn=None):
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
            result = _compliance_flex_bid(agent, obl_otg, form, supply_available, price_floor)
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
                p_variation=p_variation, cumulative_otg_txn=cumulative_otg_txn,
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
            result = agent.bid_fund_obligation(supply_available, price_floor)
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
        result = agent.bid_proactive(formations, supply_available, price_floor)
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


# --- Single run (with overridable production cost and BCT budget) ---
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
        BCTAgentPL4L(
            rng.lognormal(np.log(60_000), 0.4), rng,
            budget_log_mean=config.bct_budget_log_mean,
        )
        for _ in range(config.n_bct)
    ]
    habitat_banks = [
        HabitatBankAgentPL4L(
            rng.lognormal(np.log(250_000), 0.5), rng, formations,
            prod_cost_log_mean=config.production_cost_log_mean,
            prod_cost_log_std=config.production_cost_log_std,
        )
        for _ in range(config.n_habitat_bank)
    ]

    supply_available = {}
    total_supply = sum(o.supply_capacity for o in all_otgs)
    for otg in all_otgs:
        if otg.supply_capacity > 0:
            monthly_share = otg.supply_capacity / max(1, total_supply) * 7700
            supply_available[otg.name] = max(1, monthly_share * 2)
        else:
            supply_available[otg.name] = rng.exponential(0.2)

    cumulative_otg_txn = {o.name: 0 for o in all_otgs}
    all_transactions = []
    total_fund_payments = 0
    total_direct_purchases = 0
    total_variation_purchases = 0

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
            demand_engine, formations, all_otgs, rng,
            monthly_obligations=config.monthly_obligations,
            procurement_flex_share=config.procurement_flex_share,
            cumulative_otg_txn=cumulative_otg_txn,
        )

        step_txns, fund_pay, direct_buy, variation_buy = clear_market_step(
            formations, all_otgs, obligations, compliance_agents,
            intermediary_agents, bct_agents, habitat_banks,
            supply_available, rng, price_floor=config.price_floor,
            p_variation=config.p_variation,
            cumulative_otg_txn=cumulative_otg_txn,
        )

        total_fund_payments += fund_pay
        total_direct_purchases += direct_buy
        total_variation_purchases += variation_buy

        for otg_name, price, qty, buyer_type in step_txns:
            cumulative_otg_txn[otg_name] = cumulative_otg_txn.get(otg_name, 0) + 1
        all_transactions.extend(step_txns)

    return {
        "transactions": all_transactions,
        "supply_final": dict(supply_available),
        "cumulative_otg_txn": dict(cumulative_otg_txn),
        "total_fund_payments": total_fund_payments,
        "total_direct_purchases": total_direct_purchases,
        "total_variation_purchases": total_variation_purchases,
    }


# --- Metrics ---
def compute_metrics(result, formations, all_otgs):
    all_transactions = result["transactions"]
    total_otgs = len(all_otgs)
    otg_txn_count = {}
    for otg_name, price, qty, buyer_type in all_transactions:
        otg_txn_count[otg_name] = otg_txn_count.get(otg_name, 0) + 1

    traded = sum(1 for o in all_otgs if otg_txn_count.get(o.name, 0) >= 1)
    ec1 = traded / total_otgs if total_otgs > 0 else 0
    functional = sum(1 for o in all_otgs if otg_txn_count.get(o.name, 0) >= 5)
    ec2 = functional / total_otgs if total_otgs > 0 else 0

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

    # Formation-level transaction counts for Spearman rho
    formation_txn = {}
    for f in formations:
        f_total_txn = sum(otg_txn_count.get(o.name, 0) for o in f.otgs)
        formation_txn[f.name] = f_total_txn

    return {
        "ec1": ec1,
        "ec2": ec2,
        "fund_rate": fund_rate,
        "formation_txn": formation_txn,
        "otg_txn_count": otg_txn_count,
        "total_transactions": len(all_transactions),
    }


# --- Sweep runner ---
def run_sweep_point(formations, all_otgs, subregion_demands, config, n_runs=20):
    """Run n_runs MC seeds for a single configuration. Return summary stats."""
    ec2_vals = []
    fund_rates = []
    formation_txn_sums = {}

    for i in range(n_runs):
        result = run_single(formations, all_otgs, subregion_demands, config, seed=i)
        metrics = compute_metrics(result, formations, all_otgs)
        ec2_vals.append(metrics["ec2"])
        fund_rates.append(metrics["fund_rate"])
        for fname, ftxn in metrics["formation_txn"].items():
            if fname not in formation_txn_sums:
                formation_txn_sums[fname] = []
            formation_txn_sums[fname].append(ftxn)

    # Compute Spearman rho vs observed
    formation_names = list(formation_txn_sums.keys())
    model_txn = np.array([np.median(formation_txn_sums[f]) for f in formation_names])
    obs_txn = np.array(
        [next((fm.total_txn for fm in formations if fm.name == f), 0) for f in formation_names]
    )
    if len(formation_names) >= 3 and np.std(model_txn) > 0 and np.std(obs_txn) > 0:
        rho, _ = sp_stats.spearmanr(obs_txn, model_txn)
    else:
        rho = np.nan

    return {
        "ec2_median": float(np.median(ec2_vals)),
        "ec2_p10": float(np.percentile(ec2_vals, 10)),
        "ec2_p90": float(np.percentile(ec2_vals, 90)),
        "fund_rate_median": float(np.median(fund_rates)),
        "fund_rate_p10": float(np.percentile(fund_rates, 10)),
        "fund_rate_p90": float(np.percentile(fund_rates, 90)),
        "rho": float(rho),
    }


# =============================================================================
# MAIN: SENSITIVITY SWEEPS
# =============================================================================


def main():
    global PARAMS
    t0 = time.time()

    print("=" * 110)
    print("SENSITIVITY SWEEPS: Agent counts, production cost, P_variation")
    print("  20 MC seeds per config, 72 timesteps, Baseline scenario only")
    print("=" * 110)
    print()

    # --- Derive parameters ---
    print("[0] Deriving parameters from NSW data...")
    PARAMS = derive_all_parameters()
    observed_fund_rate = PARAMS.get("observed_fund_rate", 0.354)
    print(f"  Observed Fund rate = {observed_fund_rate:.1%}")
    print()

    # --- Load data ---
    print("[1] Loading OTG data with formation-level matching...")
    formations, all_otgs, subregion_demands = load_otg_data_formation_matching()
    print(f"  Loaded {len(all_otgs)} OTGs in {len(formations)} formations")
    print()

    # =========================================================================
    # BASELINE REFERENCE (to determine stability thresholds)
    # =========================================================================
    print("[2] Running BASELINE reference (20 MC seeds)...")
    baseline_config = FormationConfig(p_variation=0.5)
    baseline_stats = run_sweep_point(
        formations, all_otgs, subregion_demands, baseline_config, n_runs=20
    )
    bl_ec2 = baseline_stats["ec2_median"]
    bl_fr = baseline_stats["fund_rate_median"]
    bl_rho = baseline_stats["rho"]
    print(f"  Baseline EC2 = {bl_ec2:.1%}, Fund rate = {bl_fr:.1%}, rho = {bl_rho:.3f}")
    print(f"  Stability band: EC2 in [{bl_ec2 - 0.03:.1%}, {bl_ec2 + 0.03:.1%}]")
    print()

    def is_stable(ec2, rho):
        return abs(ec2 - bl_ec2) <= 0.03 and rho > 0.90

    # Store all sweep results for summary
    sweep_summary = []

    # =========================================================================
    # SWEEP 1a: n_compliance
    # =========================================================================
    print("=" * 110)
    print("SWEEP 1a: Compliance agent count")
    print("  Hold: n_intermediary=10, n_bct=12, n_habitat_bank=30")
    print("=" * 110)
    compliance_values = [10, 15, 20, 25, 30, 40, 50, 60]
    print(f"\n  {'n_compliance':>14s} | {'EC2':>8s} | {'Fund rate':>10s} | "
          f"{'Spearman rho':>13s} | {'Stable?':>8s}")
    print(f"  {'-'*14} | {'-'*8} | {'-'*10} | {'-'*13} | {'-'*8}")

    sweep1a_results = []
    for nc in compliance_values:
        config = FormationConfig(n_compliance=nc, p_variation=0.5)
        stats = run_sweep_point(formations, all_otgs, subregion_demands, config, n_runs=20)
        stable = is_stable(stats["ec2_median"], stats["rho"])
        sweep1a_results.append((nc, stats))
        print(f"  {nc:>14d} | {stats['ec2_median']:>7.1%} | {stats['fund_rate_median']:>9.1%} | "
              f"{stats['rho']:>13.3f} | {'YES' if stable else 'NO':>8s}")
    print()

    ec2_vals_1a = [s["ec2_median"] for _, s in sweep1a_results]
    fr_vals_1a = [s["fund_rate_median"] for _, s in sweep1a_results]
    all_stable_1a = all(is_stable(s["ec2_median"], s["rho"]) for _, s in sweep1a_results)
    sweep_summary.append({
        "param": "n_compliance",
        "range": f"{compliance_values[0]}--{compliance_values[-1]}",
        "ec2_range": f"{min(ec2_vals_1a):.1%}--{max(ec2_vals_1a):.1%}",
        "fr_range": f"{min(fr_vals_1a):.1%}--{max(fr_vals_1a):.1%}",
        "ranking_stable": all_stable_1a,
    })

    # =========================================================================
    # SWEEP 1b: n_intermediary
    # =========================================================================
    print("=" * 110)
    print("SWEEP 1b: Intermediary agent count")
    print("  Hold: n_compliance=30, n_bct=12, n_habitat_bank=30")
    print("=" * 110)
    intermediary_values = [3, 5, 7, 10, 15, 20]
    print(f"\n  {'n_intermediary':>14s} | {'EC2':>8s} | {'Fund rate':>10s} | "
          f"{'Spearman rho':>13s} | {'Stable?':>8s}")
    print(f"  {'-'*14} | {'-'*8} | {'-'*10} | {'-'*13} | {'-'*8}")

    sweep1b_results = []
    for ni in intermediary_values:
        config = FormationConfig(n_intermediary=ni, p_variation=0.5)
        stats = run_sweep_point(formations, all_otgs, subregion_demands, config, n_runs=20)
        stable = is_stable(stats["ec2_median"], stats["rho"])
        sweep1b_results.append((ni, stats))
        print(f"  {ni:>14d} | {stats['ec2_median']:>7.1%} | {stats['fund_rate_median']:>9.1%} | "
              f"{stats['rho']:>13.3f} | {'YES' if stable else 'NO':>8s}")
    print()

    ec2_vals_1b = [s["ec2_median"] for _, s in sweep1b_results]
    fr_vals_1b = [s["fund_rate_median"] for _, s in sweep1b_results]
    all_stable_1b = all(is_stable(s["ec2_median"], s["rho"]) for _, s in sweep1b_results)
    sweep_summary.append({
        "param": "n_intermediary",
        "range": f"{intermediary_values[0]}--{intermediary_values[-1]}",
        "ec2_range": f"{min(ec2_vals_1b):.1%}--{max(ec2_vals_1b):.1%}",
        "fr_range": f"{min(fr_vals_1b):.1%}--{max(fr_vals_1b):.1%}",
        "ranking_stable": all_stable_1b,
    })

    # =========================================================================
    # SWEEP 1c: n_bct (with budget adjustment)
    # =========================================================================
    print("=" * 110)
    print("SWEEP 1c: BCT agent count (total BCT spend = $47.3M held constant)")
    print("  Hold: n_compliance=30, n_intermediary=10, n_habitat_bank=30")
    print("  Per-agent budget = $47.3M / n_bct / 12 months")
    print("=" * 110)
    bct_values = [4, 6, 8, 12, 16, 20, 24]
    TOTAL_BCT_SPEND = 47.3e6
    print(f"\n  {'n_bct':>8s} | {'Budget/agent/mo':>16s} | {'EC2':>8s} | "
          f"{'Fund rate':>10s} | {'Spearman rho':>13s} | {'Stable?':>8s}")
    print(f"  {'-'*8} | {'-'*16} | {'-'*8} | {'-'*10} | {'-'*13} | {'-'*8}")

    sweep1c_results = []
    for nb in bct_values:
        per_agent_mo = TOTAL_BCT_SPEND / nb / 12
        config = FormationConfig(
            n_bct=nb, p_variation=0.5,
            bct_budget_log_mean=np.log(per_agent_mo),
        )
        stats = run_sweep_point(formations, all_otgs, subregion_demands, config, n_runs=20)
        stable = is_stable(stats["ec2_median"], stats["rho"])
        sweep1c_results.append((nb, stats))
        print(f"  {nb:>8d} | ${per_agent_mo:>14,.0f} | {stats['ec2_median']:>7.1%} | "
              f"{stats['fund_rate_median']:>9.1%} | {stats['rho']:>13.3f} | "
              f"{'YES' if stable else 'NO':>8s}")
    print()

    ec2_vals_1c = [s["ec2_median"] for _, s in sweep1c_results]
    fr_vals_1c = [s["fund_rate_median"] for _, s in sweep1c_results]
    all_stable_1c = all(is_stable(s["ec2_median"], s["rho"]) for _, s in sweep1c_results)
    sweep_summary.append({
        "param": "n_bct",
        "range": f"{bct_values[0]}--{bct_values[-1]}",
        "ec2_range": f"{min(ec2_vals_1c):.1%}--{max(ec2_vals_1c):.1%}",
        "fr_range": f"{min(fr_vals_1c):.1%}--{max(fr_vals_1c):.1%}",
        "ranking_stable": all_stable_1c,
    })

    # =========================================================================
    # SWEEP 1d: n_habitat_bank
    # =========================================================================
    print("=" * 110)
    print("SWEEP 1d: Habitat bank count")
    print("  Hold: n_compliance=30, n_intermediary=10, n_bct=12")
    print("=" * 110)
    habitat_bank_values = [10, 15, 20, 25, 30, 40, 50, 60]
    print(f"\n  {'n_habitat_bank':>14s} | {'EC2':>8s} | {'Fund rate':>10s} | "
          f"{'Spearman rho':>13s} | {'Stable?':>8s}")
    print(f"  {'-'*14} | {'-'*8} | {'-'*10} | {'-'*13} | {'-'*8}")

    sweep1d_results = []
    for nh in habitat_bank_values:
        config = FormationConfig(n_habitat_bank=nh, p_variation=0.5)
        stats = run_sweep_point(formations, all_otgs, subregion_demands, config, n_runs=20)
        stable = is_stable(stats["ec2_median"], stats["rho"])
        sweep1d_results.append((nh, stats))
        print(f"  {nh:>14d} | {stats['ec2_median']:>7.1%} | {stats['fund_rate_median']:>9.1%} | "
              f"{stats['rho']:>13.3f} | {'YES' if stable else 'NO':>8s}")
    print()

    ec2_vals_1d = [s["ec2_median"] for _, s in sweep1d_results]
    fr_vals_1d = [s["fund_rate_median"] for _, s in sweep1d_results]
    all_stable_1d = all(is_stable(s["ec2_median"], s["rho"]) for _, s in sweep1d_results)
    sweep_summary.append({
        "param": "n_habitat_bank",
        "range": f"{habitat_bank_values[0]}--{habitat_bank_values[-1]}",
        "ec2_range": f"{min(ec2_vals_1d):.1%}--{max(ec2_vals_1d):.1%}",
        "fr_range": f"{min(fr_vals_1d):.1%}--{max(fr_vals_1d):.1%}",
        "ranking_stable": all_stable_1d,
    })

    # =========================================================================
    # SWEEP 2: Production cost
    # =========================================================================
    print("=" * 110)
    print("SWEEP 2: Production cost (percentiles of observed ecosystem prices)")
    print("  P5=$399, P10=$679, Q1=$2033 (CURRENT), Median=$4047")
    print("=" * 110)
    cost_sweep = [
        ("P5", 399, np.log(399)),
        ("P10", 679, np.log(679)),
        ("Q1 (current)", 2033, np.log(2033)),
        ("Median", 4047, np.log(4047)),
    ]
    print(f"\n  {'Percentile':>14s} | {'Cost AUD':>10s} | {'EC2':>8s} | "
          f"{'Fund rate':>10s} | {'Spearman rho':>13s} | {'Stable?':>8s}")
    print(f"  {'-'*14} | {'-'*10} | {'-'*8} | {'-'*10} | {'-'*13} | {'-'*8}")

    sweep2_results = []
    for label, cost_aud, log_cost in cost_sweep:
        config = FormationConfig(
            p_variation=0.5,
            production_cost_log_mean=log_cost,
        )
        stats = run_sweep_point(formations, all_otgs, subregion_demands, config, n_runs=20)
        stable = is_stable(stats["ec2_median"], stats["rho"])
        sweep2_results.append((label, cost_aud, stats))
        print(f"  {label:>14s} | ${cost_aud:>9,d} | {stats['ec2_median']:>7.1%} | "
              f"{stats['fund_rate_median']:>9.1%} | {stats['rho']:>13.3f} | "
              f"{'YES' if stable else 'NO':>8s}")
    print()

    ec2_vals_2 = [s["ec2_median"] for _, _, s in sweep2_results]
    fr_vals_2 = [s["fund_rate_median"] for _, _, s in sweep2_results]
    all_stable_2 = all(is_stable(s["ec2_median"], s["rho"]) for _, _, s in sweep2_results)
    sweep_summary.append({
        "param": "production_cost",
        "range": f"$399--$4,047",
        "ec2_range": f"{min(ec2_vals_2):.1%}--{max(ec2_vals_2):.1%}",
        "fr_range": f"{min(fr_vals_2):.1%}--{max(fr_vals_2):.1%}",
        "ranking_stable": all_stable_2,
    })

    # =========================================================================
    # SWEEP 3: P_variation (re-verification)
    # =========================================================================
    print("=" * 110)
    print("SWEEP 3: P_variation (calibration re-verification, 20 seeds)")
    print("=" * 110)
    pvar_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n  {'P_variation':>12s} | {'EC2':>8s} | {'Fund rate':>10s} | "
          f"{'Spearman rho':>13s} | {'Stable?':>8s}")
    print(f"  {'-'*12} | {'-'*8} | {'-'*10} | {'-'*13} | {'-'*8}")

    sweep3_results = []
    for pv in pvar_values:
        config = FormationConfig(p_variation=pv)
        stats = run_sweep_point(formations, all_otgs, subregion_demands, config, n_runs=20)
        stable = is_stable(stats["ec2_median"], stats["rho"])
        sweep3_results.append((pv, stats))
        print(f"  {pv:>12.1f} | {stats['ec2_median']:>7.1%} | {stats['fund_rate_median']:>9.1%} | "
              f"{stats['rho']:>13.3f} | {'YES' if stable else 'NO':>8s}")
    print()

    ec2_vals_3 = [s["ec2_median"] for _, s in sweep3_results]
    fr_vals_3 = [s["fund_rate_median"] for _, s in sweep3_results]
    all_stable_3 = all(is_stable(s["ec2_median"], s["rho"]) for _, s in sweep3_results)
    sweep_summary.append({
        "param": "P_variation",
        "range": f"0.3--0.7",
        "ec2_range": f"{min(ec2_vals_3):.1%}--{max(ec2_vals_3):.1%}",
        "fr_range": f"{min(fr_vals_3):.1%}--{max(fr_vals_3):.1%}",
        "ranking_stable": all_stable_3,
    })

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    elapsed = time.time() - t0
    print("\n")
    print("=" * 110)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print(f"  Baseline: EC2={bl_ec2:.1%}, Fund rate={bl_fr:.1%}, rho={bl_rho:.3f}")
    print(f"  Stability criterion: EC2 within +/-3pp of baseline AND rho > 0.90")
    print(f"  Total runtime: {elapsed / 60:.1f} minutes")
    print("=" * 110)
    print()
    print(f"  {'Parameter':<18s} | {'Swept range':<18s} | {'EC2 range':<18s} | "
          f"{'Fund rate range':<18s} | {'Ranking stable?':<16s} | {'Conclusion'}")
    print(f"  {'-'*18} | {'-'*18} | {'-'*18} | {'-'*18} | {'-'*16} | {'-'*30}")

    # Determine sensitivity for conclusions
    for ss in sweep_summary:
        ec2_parts = ss["ec2_range"].replace("%", "").split("--")
        ec2_lo = float(ec2_parts[0]) / 100
        ec2_hi = float(ec2_parts[1]) / 100
        ec2_span = ec2_hi - ec2_lo

        if ss["ranking_stable"]:
            if ec2_span <= 0.03:
                conclusion = "INSENSITIVE (robust)"
            elif ec2_span <= 0.06:
                conclusion = "LOW sensitivity"
            else:
                conclusion = "MODERATE sensitivity (ranking OK)"
        else:
            if ec2_span <= 0.06:
                conclusion = "MODERATE (ranking shift)"
            else:
                conclusion = "SENSITIVE (ranking shift)"

        ss["conclusion"] = conclusion
        print(f"  {ss['param']:<18s} | {ss['range']:<18s} | {ss['ec2_range']:<18s} | "
              f"{ss['fr_range']:<18s} | {'YES' if ss['ranking_stable'] else 'NO':<16s} | "
              f"{conclusion}")
    print()

    # =========================================================================
    # VERDICT
    # =========================================================================
    print("=" * 110)
    print("VERDICT")
    print("=" * 110)
    print()

    # Check qualitative robustness
    all_ec2_below_50 = True
    all_fund_near_35 = True
    all_rho_high = True

    all_sweep_results = (
        [(nc, s) for nc, s in sweep1a_results]
        + [(ni, s) for ni, s in sweep1b_results]
        + [(nb, s) for nb, s in sweep1c_results]
        + [(nh, s) for nh, s in sweep1d_results]
        + [(ca, s) for _, ca, s in sweep2_results]
        + [(pv, s) for pv, s in sweep3_results]
    )

    for val, stats in all_sweep_results:
        if stats["ec2_median"] >= 0.50:
            all_ec2_below_50 = False
        if abs(stats["fund_rate_median"] - 0.35) > 0.15:
            all_fund_near_35 = False
        if stats["rho"] < 0.70:
            all_rho_high = False

    print(f"  1. EC2 << 50% across ALL {len(all_sweep_results)} configs?  "
          f"{'YES' if all_ec2_below_50 else 'NO'}")
    print(f"  2. Fund rate near 35% (+/-15pp)?                   "
          f"{'YES' if all_fund_near_35 else 'NO'}")
    print(f"  3. Formation rank correlation > 0.70?              "
          f"{'YES' if all_rho_high else 'NO'}")
    print()

    # Most/least sensitive
    sensitivities = {}
    for ss in sweep_summary:
        ec2_parts = ss["ec2_range"].replace("%", "").split("--")
        ec2_span = float(ec2_parts[1]) / 100 - float(ec2_parts[0]) / 100
        sensitivities[ss["param"]] = ec2_span

    most_sensitive = max(sensitivities, key=sensitivities.get)
    least_sensitive = min(sensitivities, key=sensitivities.get)

    print(f"  MOST sensitive parameter:  {most_sensitive} "
          f"(EC2 span = {sensitivities[most_sensitive]:.1%})")
    print(f"  LEAST sensitive parameter: {least_sensitive} "
          f"(EC2 span = {sensitivities[least_sensitive]:.1%})")
    print()

    if all_ec2_below_50 and all_rho_high:
        print("  OVERALL: ALL qualitative conclusions are ROBUST across all sensitivity sweeps.")
        print("  The central finding -- EC2 << 50%, markets select against rarity -- holds")
        print("  regardless of agent count or production cost assumptions.")
    else:
        issues = []
        if not all_ec2_below_50:
            issues.append("EC2 exceeds 50% in some configs")
        if not all_rho_high:
            issues.append("Formation ranking unstable in some configs")
        if not all_fund_near_35:
            issues.append("Fund rate deviates >15pp from 35%")
        print(f"  CAUTION: Some qualitative conclusions may not be fully robust.")
        print(f"  Issues: {'; '.join(issues)}")

    print()
    print("=" * 110)
    print(f"  Total elapsed time: {elapsed / 60:.1f} minutes ({elapsed:.0f} seconds)")
    print("=" * 110)

    # =========================================================================
    # SAVE RESULTS JSON
    # =========================================================================
    save_sensitivity_results_json(sweep_summary, bl_ec2)


def save_sensitivity_results_json(sweep_summary, baseline_ec2):
    """Save sensitivity results to JSON for downstream figure scripts."""
    out_dir = Path(__file__).resolve().parent.parent / "output" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    sweeps = {}
    for ss in sweep_summary:
        ec2_parts = ss["ec2_range"].replace("%", "").split("--")
        ec2_lo = float(ec2_parts[0])
        ec2_hi = float(ec2_parts[1])
        span_pp = ec2_hi - ec2_lo

        sweeps[ss["param"]] = {
            "param": ss["param"],
            "range": ss["range"],
            "ec2_lo": round(ec2_lo, 1),
            "ec2_hi": round(ec2_hi, 1),
            "span_pp": round(span_pp, 1),
            "ranking_stable": ss["ranking_stable"],
        }

    results = {
        "script": "scripts/robustness_sensitivity.py",
        "timestamp": datetime.now().isoformat(),
        "baseline_ec2": round(baseline_ec2 * 100, 1),
        "sweeps": sweeps,
    }

    json_path = out_dir / "sensitivity_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
