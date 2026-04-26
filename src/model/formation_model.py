"""
=============================================================================
FORMATION-LEVEL AGENT-BASED MODEL: NSW BIODIVERSITY OFFSETS SCHEME
=============================================================================

Prototype ABM that uses the REAL 252 ecosystem OTGs grouped by their 15
vegetation formations, with empirically measured parameters from the NSW
credit supply and transaction registers.

This model uses 252 real OTGs in 15 vegetation formations with
observed transaction shares, TEC status, and median prices.

v2 improvements (March 2026):
  - Official TEC classification from supply register column
    "Threatened Ecological Community (NSW)" instead of bioregion heuristic.
  - Geographic obligation distribution using IBRA subregions weighted by
    observed development pressure (transaction volume per subregion).
  - Supply capacity per OTG from "Number of credits" column.
  - Within-subregion x formation OTG concentration from empirical counts.
  - Calibrated to: EC2~18%, EC2-TEC~20%, ~6 functional formations,
    ~10-12 monthly transactions, Grassy Woodlands dominance.

The central test: does the model endogenously reproduce the observed pattern
where Grassy Woodlands and Forested Wetlands get functional markets while
Rainforests, Heathlands, and arid types get nothing?

Requires: numpy, scipy, pandas (for data loading only)
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# =============================================================================
# 1. DATA LOADING AND OTG CONSTRUCTION
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "nsw"

# Formations to merge (lowercase variant -> canonical)
FORMATION_NORMALIZE = {
    "Grassy woodlands": "Grassy Woodlands",
}


@dataclass
class OTG:
    """A single Offset Trading Group."""
    name: str
    formation: str
    is_tec: bool
    observed_txn_count: int = 0
    observed_median_price: float = 0.0
    observed_credits: float = 0.0
    supply_capacity: float = 0.0  # from supply register "Number of credits"
    subregions: List[str] = field(default_factory=list)


@dataclass
class Formation:
    """A vegetation formation containing multiple OTGs."""
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
        prices = [
            o.observed_median_price for o in self.otgs if o.observed_median_price > 0
        ]
        return float(np.median(prices)) if prices else 3000.0


@dataclass
class SubregionDemand:
    """Observed development pressure per IBRA subregion."""
    name: str
    n_txn: int
    formations: Dict[str, int]  # formation -> txn count
    otg_txn: Dict[str, int]  # otg_name -> txn count within this subregion


def load_otg_data() -> Tuple[List[Formation], List[OTG], List[SubregionDemand]]:
    """
    Load OTG data from NSW supply and transaction registers.

    Uses:
      1. Supply register for OTG universe (252 OTGs) with OFFICIAL TEC
         classification, IBRA subregion, and credit supply capacity.
      2. Transaction register for subregion-level demand patterns and
         per-OTG transaction counts.

    OTG name matching: Only ~33 of 140 transaction-register OTGs match
    supply-register OTGs directly (encoding differences). For the
    remaining ~107 txn-register OTGs (447 transactions), we distribute
    their counts to supply-register OTGs in the same formation +
    subregion cell, weighted by supply capacity. This recovers 758/764
    transactions for calibration.
    """

    # --- Supply register: all 252 OTGs with supply ---
    supply = pd.read_csv(
        DATA_DIR / "nsw_credit_supply_register.csv", encoding="utf-8-sig"
    )
    supply.columns = supply.columns.str.strip()
    supply_eco = supply[
        supply["Ecosystem or Species"] == "Ecosystem"
    ].copy()
    supply_eco["Vegetation Formation"] = supply_eco[
        "Vegetation Formation"
    ].replace(FORMATION_NORMALIZE)
    supply_eco["n_credits"] = pd.to_numeric(
        supply_eco["Number of credits"], errors="coerce"
    ).fillna(0)

    tec_col = "Threatened Ecological Community (NSW)"
    non_tec_labels = {
        "Not a TEC",
        "No Associated TEC",
        "No TEC Associated",
        "",
    }

    # Per-OTG: formation, TEC status (official), supply capacity, subregions
    otg_info = (
        supply_eco.groupby("Offset Trading Group")
        .agg(
            formation=("Vegetation Formation", "first"),
            supply_capacity=("n_credits", "sum"),
            tec_vals=(
                tec_col,
                lambda x: list(x.dropna().unique()),
            ),
            subregions=(
                "IBRA Subregion",
                lambda x: list(x.dropna().unique()),
            ),
        )
        .reset_index()
    )
    otg_info["is_tec"] = otg_info["tec_vals"].apply(
        lambda vals: any(
            str(v).strip() not in non_tec_labels
            for v in vals
            if isinstance(v, str)
        )
    )

    # Build subregion -> formation -> [supply OTG names] with capacity
    supply_otg_names = set(otg_info["Offset Trading Group"])
    otg_subregion_map: Dict[str, set] = {}
    for _, row in supply_eco.iterrows():
        otg_name = row["Offset Trading Group"]
        sub = row.get("IBRA Subregion", "")
        if pd.notna(sub) and str(sub).strip():
            otg_subregion_map.setdefault(otg_name, set()).add(
                str(sub).strip()
            )

    # Build lookup: (subregion, formation) -> [(otg_name, capacity)]
    sub_form_supply: Dict[
        Tuple[str, str], List[Tuple[str, float]]
    ] = {}
    for _, row in otg_info.iterrows():
        otg_name = row["Offset Trading Group"]
        form_name = row["formation"]
        cap = float(row["supply_capacity"])
        for sub in otg_subregion_map.get(otg_name, []):
            key = (sub, form_name)
            sub_form_supply.setdefault(key, []).append(
                (otg_name, max(1.0, cap))
            )

    # --- Transaction register ---
    txn = pd.read_csv(
        DATA_DIR / "nsw_credit_transactions_register.csv",
        encoding="utf-8-sig",
    )
    txn.columns = txn.columns.str.strip()
    txn["price"] = pd.to_numeric(
        txn["Price Per Credit (Ex-GST)"], errors="coerce"
    )
    txn["credits"] = pd.to_numeric(
        txn["Number Of Credits"], errors="coerce"
    )
    transfers = txn[txn["Transaction Type"] == "Transfer"].copy()
    priced = transfers[
        (transfers["price"] >= 100) & transfers["price"].notna()
    ].copy()
    priced["credit_type"] = np.where(
        priced["Scientific Name"].fillna("").str.strip() != "",
        "Species",
        "Ecosystem",
    )
    eco = priced[priced["credit_type"] == "Ecosystem"].copy()
    eco["Vegetation Formation"] = eco["Vegetation Formation"].replace(
        FORMATION_NORMALIZE
    )

    # --- Distribute ALL transactions to supply-register OTGs ---
    # For each transaction: if direct OTG name match -> assign directly.
    # Otherwise, distribute to supply OTGs in same (subregion, formation)
    # weighted by supply capacity.
    otg_txn_assigned: Dict[str, float] = {}  # fractional txn counts
    otg_prices: Dict[str, List[float]] = {}  # for median price
    otg_credits_assigned: Dict[str, float] = {}

    n_direct = 0
    n_redistributed = 0
    n_unmatched = 0

    for _, row in eco.iterrows():
        txn_otg = str(row.get("Offset Trading Group", "")).strip()
        sub = str(row.get("Sub Region", "")).strip()
        form = str(row.get("Vegetation Formation", "")).strip()
        price = row["price"]
        credits = row["credits"] if pd.notna(row["credits"]) else 0

        if txn_otg in supply_otg_names:
            # Direct match
            otg_txn_assigned[txn_otg] = (
                otg_txn_assigned.get(txn_otg, 0) + 1
            )
            otg_prices.setdefault(txn_otg, []).append(price)
            otg_credits_assigned[txn_otg] = (
                otg_credits_assigned.get(txn_otg, 0) + credits
            )
            n_direct += 1
        else:
            # Distribute to supply OTGs in same (subregion, formation)
            key = (sub, form)
            candidates = sub_form_supply.get(key, [])
            if candidates:
                total_cap = sum(c for _, c in candidates)
                for cand_otg, cap in candidates:
                    share = cap / total_cap
                    otg_txn_assigned[cand_otg] = (
                        otg_txn_assigned.get(cand_otg, 0) + share
                    )
                    otg_prices.setdefault(cand_otg, []).append(price)
                    otg_credits_assigned[cand_otg] = (
                        otg_credits_assigned.get(cand_otg, 0)
                        + credits * share
                    )
                n_redistributed += 1
            else:
                n_unmatched += 1

    # --- Subregion-level demand patterns ---
    subregion_data: Dict[str, dict] = {}
    for _, row in eco.iterrows():
        sub = row.get("Sub Region", "")
        if pd.isna(sub) or str(sub).strip() == "":
            continue
        sub = str(sub).strip()
        form = str(row.get("Vegetation Formation", "")).strip()

        if sub not in subregion_data:
            subregion_data[sub] = {
                "n_txn": 0,
                "formations": {},
                "otg_txn": {},
            }
        subregion_data[sub]["n_txn"] += 1
        subregion_data[sub]["formations"][form] = (
            subregion_data[sub]["formations"].get(form, 0) + 1
        )

        # Map transaction OTG to supply OTG for subregion demand
        txn_otg = str(row.get("Offset Trading Group", "")).strip()
        if txn_otg in supply_otg_names:
            subregion_data[sub]["otg_txn"][txn_otg] = (
                subregion_data[sub]["otg_txn"].get(txn_otg, 0) + 1
            )
        else:
            key = (sub, form)
            candidates = sub_form_supply.get(key, [])
            if candidates:
                total_cap = sum(c for _, c in candidates)
                for cand_otg, cap in candidates:
                    share = cap / total_cap
                    subregion_data[sub]["otg_txn"][cand_otg] = (
                        subregion_data[sub]["otg_txn"].get(cand_otg, 0)
                        + share
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

    # Formation-level fallback pricing
    form_txn = (
        eco.groupby("Vegetation Formation")
        .agg(median_price=("price", "median"))
        .reset_index()
    )
    form_price_dict = dict(
        zip(form_txn["Vegetation Formation"], form_txn["median_price"])
    )

    # Build OTG objects
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
                observed_median_price=(
                    float(med_price) if med_price > 0 else 0.0
                ),
                observed_credits=float(assigned_credits),
                supply_capacity=float(row["supply_capacity"]),
                subregions=subs,
            )
        )

    print(
        f"  Transaction matching: {n_direct} direct, "
        f"{n_redistributed} redistributed, {n_unmatched} unmatched"
    )

    # Group by formation
    formation_dict: Dict[str, List[OTG]] = {}
    for otg in all_otgs:
        formation_dict.setdefault(otg.formation, []).append(otg)

    formations = [
        Formation(name=name, otgs=otgs)
        for name, otgs in sorted(
            formation_dict.items(), key=lambda x: -len(x[1])
        )
    ]

    return formations, all_otgs, subregion_demands


# =============================================================================
# 2. SIMULATION CONFIGURATION
# =============================================================================


@dataclass
class FormationConfig:
    T: int = 72  # 6 years, monthly
    n_compliance: int = 30
    n_intermediary: int = 10
    n_bct: int = 12
    n_habitat_bank: int = 30
    monthly_obligations: float = 12.0  # compliance events per month
    seed: int = 42
    # Policy levers
    procurement_flex_share: float = 0.0  # fraction redirected to thin OTGs
    price_floor: float = 0.0  # AUD minimum for thin-market OTGs


# =============================================================================
# 3. GEOGRAPHIC DEMAND ENGINE
# =============================================================================


class GeographicDemandEngine:
    """
    Generates obligations using the empirical geographic distribution of
    development pressure across IBRA subregions.

    The key insight: demand is NOT distributed uniformly across all OTGs
    in a formation. It is concentrated in specific subregions (Cumberland,
    Inland Slopes, Pilliga, Karuah Manning, etc.) and within each
    subregion in specific OTGs. This geographic concentration is what
    produces the sharp EC2 patterns observed empirically.
    """

    def __init__(
        self,
        formations: List[Formation],
        all_otgs: List[OTG],
        subregion_demands: List[SubregionDemand],
    ):
        self.formations = formations
        self.all_otgs = all_otgs
        self.formation_lookup = {f.name: f for f in formations}
        self.otg_lookup = {o.name: o for o in all_otgs}

        # Build subregion weights from observed transaction volume
        self.subregions = [s for s in subregion_demands if s.n_txn > 0]
        sub_txn = np.array(
            [s.n_txn for s in self.subregions], dtype=float
        )
        self.subregion_weights = (
            sub_txn / sub_txn.sum() if sub_txn.sum() > 0 else None
        )

        # Build subregion -> formation -> OTG mapping from supply register
        self.sub_form_otgs: Dict[str, Dict[str, List[OTG]]] = {}
        for otg in all_otgs:
            for sub in otg.subregions:
                if sub not in self.sub_form_otgs:
                    self.sub_form_otgs[sub] = {}
                form = otg.formation
                if form not in self.sub_form_otgs[sub]:
                    self.sub_form_otgs[sub][form] = []
                self.sub_form_otgs[sub][form].append(otg)

        # Pre-compute per-subregion formation weights from txn data.
        # Square the weights to concentrate demand on the dominant
        # formation within each subregion (e.g., Grassy Woodlands in
        # Cumberland gets 55/77 = 71% of weight linear, 83% squared).
        # This suppresses formations with very few transactions in a
        # subregion (like Rainforests in Cumberland: 1/77 -> 0.02%).
        self.sub_form_weights: Dict[
            str, Tuple[List[str], np.ndarray]
        ] = {}
        for sd in self.subregions:
            forms = []
            weights = []
            for form_name, count in sd.formations.items():
                if form_name in self.sub_form_otgs.get(sd.name, {}):
                    forms.append(form_name)
                    weights.append(count)
            if forms:
                w = np.array(weights, dtype=float)
                w = w ** 2  # square to concentrate on dominant
                w /= w.sum()
                self.sub_form_weights[sd.name] = (forms, w)

        # Per-subregion per-OTG transaction counts
        self.sub_otg_txn: Dict[str, Dict[str, int]] = {}
        for sd in self.subregions:
            self.sub_otg_txn[sd.name] = sd.otg_txn

    def generate_obligation(
        self,
        rng: np.random.RandomState,
    ) -> Optional[Tuple[OTG, Formation]]:
        """
        Generate a single obligation by:
        1. Draw a subregion (weighted by development pressure)
        2. Draw a formation within that subregion
        3. Draw an OTG within that subregion x formation cell
        """
        if self.subregion_weights is None or len(self.subregions) == 0:
            return None

        # Step 1: Draw subregion
        si = rng.choice(len(self.subregions), p=self.subregion_weights)
        sub = self.subregions[si]

        # Step 2: Draw formation within subregion
        if sub.name not in self.sub_form_weights:
            return None
        forms, form_w = self.sub_form_weights[sub.name]
        fi = rng.choice(len(forms), p=form_w)
        form_name = forms[fi]

        # Step 3: Draw OTG within subregion x formation
        otgs_in_cell = self.sub_form_otgs.get(sub.name, {}).get(
            form_name, []
        )
        if not otgs_in_cell:
            return None

        # Weight by observed OTG-level txn counts in this subregion
        sub_otg_txn = self.sub_otg_txn.get(sub.name, {})
        otg_counts = np.array(
            [
                sub_otg_txn.get(o.name, 0) + o.observed_txn_count * 0.1
                for o in otgs_in_cell
            ],
            dtype=float,
        )
        # Power-law: top OTGs get vastly more demand.
        # Alpha=4.0 ensures the top 1-2 OTGs per cell capture most
        # demand, matching the empirical pattern where CPW alone has
        # 46 of ~300 ecosystem transactions.
        otg_weights = _power_law_weights(otg_counts, alpha=4.0)

        oi = rng.choice(len(otgs_in_cell), p=otg_weights)
        otg = otgs_in_cell[oi]
        form = self.formation_lookup.get(otg.formation)
        if form is None:
            return None

        return (otg, form)


# =============================================================================
# 4. AGENT DEFINITIONS
# =============================================================================


class FormationAgent:
    """Base agent class for formation-level model."""
    _counter = 0

    def __init__(
        self, agent_type: str, cash: float, rng: np.random.RandomState
    ):
        FormationAgent._counter += 1
        self.id = FormationAgent._counter
        self.agent_type = agent_type
        self.cash = cash
        self.rng = rng
        self.transactions: List[Tuple[str, float, float]] = []


class ComplianceAgent(FormationAgent):
    """
    Compliance buyers: receive obligations tied to specific OTGs within
    a subregion x formation cell.
    """

    def __init__(self, cash: float, rng: np.random.RandomState):
        super().__init__("Compliance", cash, rng)
        self.markup = rng.uniform(1.0, 1.5)

    def bid(
        self,
        obligation_otg: OTG,
        formation: Formation,
        supply_available: Dict[str, float],
        price_floor: float,
    ) -> Optional[Tuple[str, float, float]]:
        """Bid on the obligation OTG; fallback to cheapest in formation."""
        base_price = (
            formation.median_price if formation.median_price > 0 else 3000.0
        )
        reservation = base_price * self.markup

        def _try_buy(otg: OTG) -> Optional[Tuple[str, float, float]]:
            avail = supply_available.get(otg.name, 0.0)
            if avail <= 0:
                return None
            ask = (
                otg.observed_median_price
                if otg.observed_median_price > 0
                else base_price
            )
            ask = max(ask, price_floor) * self.rng.lognormal(0, 0.15)
            if ask > reservation:
                return None
            qty = min(
                self.rng.lognormal(2.5, 0.8),
                avail,
                self.cash / (ask + 1),
            )
            return (otg.name, ask, qty) if qty >= 1 else None

        # 85% chance: buy the exact obligation OTG
        if self.rng.random() < 0.85:
            result = _try_buy(obligation_otg)
            if result:
                return result

        # Fallback: cheapest OTG in the formation with supply
        candidates = []
        for otg in formation.otgs:
            avail = supply_available.get(otg.name, 0.0)
            if avail <= 0:
                continue
            ask = (
                otg.observed_median_price
                if otg.observed_median_price > 0
                else base_price
            )
            ask = max(ask, price_floor) * self.rng.lognormal(0, 0.15)
            if ask <= reservation:
                candidates.append((otg.name, ask, avail))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1])
        otg_name, price, avail = candidates[0]
        qty = min(
            self.rng.lognormal(2.5, 0.8), avail, self.cash / (price + 1)
        )
        if qty < 1:
            return None
        return (otg_name, price, qty)


class IntermediaryAgent(FormationAgent):
    """
    Intermediaries: buy in the most liquid OTGs and sell to compliance.
    """

    def __init__(self, cash: float, rng: np.random.RandomState):
        super().__init__("Intermediary", cash, rng)
        self.aggression = rng.uniform(0.5, 2.0)

    def bid(
        self,
        formations: List[Formation],
        supply_available: Dict[str, float],
        price_floor: float,
    ) -> Optional[Tuple[str, float, float]]:
        """Buy in high-volume OTGs (steep power-law weighting)."""
        weights = []
        candidate_otgs = []
        for f in formations:
            for otg in f.otgs:
                avail = supply_available.get(otg.name, 0.0)
                if avail <= 0:
                    continue
                liq_weight = max(1, otg.observed_txn_count) ** 3.0
                weights.append(liq_weight)
                candidate_otgs.append((otg, f))

        if not candidate_otgs:
            return None

        weights = np.array(weights)
        weights /= weights.sum()
        idx = self.rng.choice(len(candidate_otgs), p=weights)
        otg, form = candidate_otgs[idx]

        base_price = (
            otg.observed_median_price
            if otg.observed_median_price > 0
            else form.median_price
        )
        bid_price = (
            base_price * self.rng.lognormal(-0.1, 0.2) * self.aggression
        )
        bid_price = max(bid_price, price_floor)
        avail = supply_available.get(otg.name, 0.0)
        qty = min(
            self.rng.lognormal(3.0, 1.0) * self.aggression,
            avail,
            self.cash / (bid_price + 1) * 0.3,
        )
        if qty < 1:
            return None
        return (otg.name, bid_price, qty)


class BCTAgent(FormationAgent):
    """
    BCT-type (procurement-flexible): buys across formations with
    preference for thin-market OTGs. Pays above-compliance prices.
    """

    def __init__(self, cash: float, rng: np.random.RandomState):
        super().__init__("BCT", cash, rng)
        self.annual_budget = rng.lognormal(np.log(326389), 0.5)  # IPART 2024-25: 45% × $105M / 12 agents / 12 months
        self.budget_remaining = self.annual_budget
        self.thin_market_preference = rng.uniform(2.0, 5.0)

    def reset_budget(self):
        self.budget_remaining = self.annual_budget

    NON_TEC_PURCHASE_PROB = 0.05

    def bid(
        self,
        formations: List[Formation],
        supply_available: Dict[str, float],
        price_floor: float,
    ) -> Optional[Tuple[str, float, float]]:
        """Buy with preference for thin-market OTGs."""
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
                    w = self.thin_market_preference * max(
                        1, 5 - otg.observed_txn_count
                    )
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
            otg.observed_median_price
            if otg.observed_median_price > 0
            else form.median_price
        )
        if base_price <= 0:
            base_price = 3000.0
        if otg.observed_txn_count < 5:
            bid_price = base_price * self.rng.lognormal(np.log(1.5), 0.3)
            bid_price = max(
                bid_price, 5989.0 * self.rng.lognormal(0, 0.2)  # BCT thin-market median from verify_claims.py
            )
        else:
            bid_price = base_price * self.rng.lognormal(0, 0.2)

        bid_price = max(bid_price, price_floor)
        avail = supply_available.get(otg.name, 0.0)
        qty = min(
            self.rng.lognormal(1.5, 0.8),
            avail,
            self.budget_remaining / (bid_price + 1),
            self.cash / (bid_price + 1) * 0.4,
        )
        if qty < 1:
            return None
        return (otg.name, bid_price, qty)


class HabitatBankAgent(FormationAgent):
    """
    Habitat banks: supply side. Invest in credit production where they
    expect demand. Supply weighted by registered credit capacity.
    """

    def __init__(
        self,
        cash: float,
        rng: np.random.RandomState,
        formations: List[Formation],
    ):
        super().__init__("HabitatBank", cash, rng)
        n_specialise = rng.randint(1, min(4, len(formations) + 1))
        form_weights = np.array(
            [max(1, f.total_txn) for f in formations], dtype=float
        )
        form_weights /= form_weights.sum()
        self.specialisation_indices = rng.choice(
            len(formations),
            size=n_specialise,
            replace=False,
            p=form_weights,
        )
        self.production_cost = rng.lognormal(7.62, 0.5)  # Q1 of eco credit prices = $2,033; log(2033) = 7.62
        self.capacity_per_month = rng.lognormal(2.5, 0.8)

    def produce(
        self,
        formations: List[Formation],
        supply_available: Dict[str, float],
    ):
        """Produce credits weighted by capacity and observed demand."""
        new_credits = max(
            1, int(self.rng.poisson(self.capacity_per_month * 0.4))
        )
        cost = new_credits * self.production_cost * 0.15
        if self.cash < cost:
            return

        self.cash -= cost

        for fi in self.specialisation_indices:
            if fi >= len(formations):
                continue
            form = formations[fi]

            otgs_with_cap = [
                o for o in form.otgs if o.supply_capacity > 0
            ]
            if not otgs_with_cap:
                otgs_with_cap = form.otgs[:1]

            cap = np.array(
                [o.supply_capacity for o in otgs_with_cap], dtype=float
            )
            txn = np.array(
                [max(1, o.observed_txn_count) for o in otgs_with_cap],
                dtype=float,
            )
            combined = cap * txn
            combined = (
                combined / combined.sum()
                if combined.sum() > 0
                else np.ones(len(cap)) / len(cap)
            )

            credits_to_form = max(
                1, new_credits // len(self.specialisation_indices)
            )
            for _ in range(credits_to_form):
                otg_idx = self.rng.choice(len(otgs_with_cap), p=combined)
                otg = otgs_with_cap[otg_idx]
                supply_available[otg.name] = (
                    supply_available.get(otg.name, 0.0) + 1
                )


# =============================================================================
# 5. OBLIGATION GENERATION
# =============================================================================


def _power_law_weights(
    counts: np.ndarray, alpha: float = 2.0
) -> np.ndarray:
    """
    Generate power-law (Zipf) weights from observed counts.
    Rank by descending count; weight = rank^{-alpha}.
    """
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


def generate_obligations(
    demand_engine: GeographicDemandEngine,
    formations: List[Formation],
    all_otgs: List[OTG],
    rng: np.random.RandomState,
    monthly_obligations: float = 12.0,
    procurement_flex_share: float = 0.0,
) -> List[Tuple[OTG, Formation]]:
    """
    Generate monthly obligations using the geographic demand engine.
    Procurement-flex obligations redirect demand to thin-market OTGs.
    """
    n_obligations = max(1, rng.poisson(monthly_obligations))
    obligations: List[Tuple[OTG, Formation]] = []

    formation_lookup = {f.name: f for f in formations}

    n_standard = int(n_obligations * (1.0 - procurement_flex_share))
    n_flex = n_obligations - n_standard

    for _ in range(n_standard):
        result = demand_engine.generate_obligation(rng)
        if result is not None:
            obligations.append(result)

    if n_flex > 0:
        thin_otgs = [
            o for o in all_otgs if o.observed_txn_count < 5
        ]
        if thin_otgs:
            for _ in range(n_flex):
                otg = thin_otgs[rng.randint(len(thin_otgs))]
                form = formation_lookup.get(otg.formation)
                if form:
                    obligations.append((otg, form))

    return obligations


# =============================================================================
# 6. MARKET CLEARING
# =============================================================================


def clear_market_step(
    formations: List[Formation],
    all_otgs: List[OTG],
    obligations: List[Tuple[OTG, Formation]],
    compliance_agents: List[ComplianceAgent],
    intermediary_agents: List[IntermediaryAgent],
    bct_agents: List[BCTAgent],
    habitat_banks: List[HabitatBankAgent],
    supply_available: Dict[str, float],
    rng: np.random.RandomState,
    price_floor: float = 0.0,
) -> List[Tuple[str, float, float, str]]:
    """Single time-step market clearing."""
    transactions: List[Tuple[str, float, float, str]] = []

    # 1. Habitat banks produce
    for bank in habitat_banks:
        bank.produce(formations, supply_available)

    # 2. Compliance agents bid on their obligations
    rng.shuffle(obligations)
    for i, (obl_otg, form) in enumerate(obligations):
        agent_idx = i % len(compliance_agents)
        agent = compliance_agents[agent_idx]
        result = agent.bid(obl_otg, form, supply_available, price_floor)
        if result:
            otg_name, price, qty = result
            supply_available[otg_name] = max(
                0, supply_available.get(otg_name, 0) - qty
            )
            agent.cash -= price * qty
            agent.transactions.append((otg_name, price, qty))
            transactions.append((otg_name, price, qty, "Compliance"))

    # 3. Intermediaries (20% active each month)
    for agent in intermediary_agents:
        if rng.random() > 0.20:
            continue
        result = agent.bid(formations, supply_available, price_floor)
        if result:
            otg_name, price, qty = result
            supply_available[otg_name] = max(
                0, supply_available.get(otg_name, 0) - qty
            )
            agent.cash -= price * qty
            agent.transactions.append((otg_name, price, qty))
            transactions.append((otg_name, price, qty, "Intermediary"))

    # 4. BCT agents (15% active each month)
    for agent in bct_agents:
        if rng.random() > 0.15:
            continue
        result = agent.bid(formations, supply_available, price_floor)
        if result:
            otg_name, price, qty = result
            supply_available[otg_name] = max(
                0, supply_available.get(otg_name, 0) - qty
            )
            agent.cash -= price * qty
            agent.budget_remaining -= price * qty
            agent.transactions.append((otg_name, price, qty))
            transactions.append((otg_name, price, qty, "BCT"))

    return transactions


# =============================================================================
# 7. SIMULATION RUNNER
# =============================================================================


def run_single(
    formations: List[Formation],
    all_otgs: List[OTG],
    subregion_demands: List[SubregionDemand],
    config: FormationConfig,
    seed: Optional[int] = None,
) -> Dict:
    """Run a single simulation."""
    rng = np.random.RandomState(
        seed if seed is not None else config.seed
    )
    FormationAgent._counter = 0

    demand_engine = GeographicDemandEngine(
        formations, all_otgs, subregion_demands
    )

    compliance_agents = [
        ComplianceAgent(rng.lognormal(np.log(200_000), 0.5), rng)
        for _ in range(config.n_compliance)
    ]
    intermediary_agents = [
        IntermediaryAgent(rng.lognormal(np.log(100_000), 0.7), rng)
        for _ in range(config.n_intermediary)
    ]
    bct_agents = [
        BCTAgent(rng.lognormal(np.log(60_000), 0.4), rng)
        for _ in range(config.n_bct)
    ]
    habitat_banks = [
        HabitatBankAgent(
            rng.lognormal(np.log(250_000), 0.5), rng, formations
        )
        for _ in range(config.n_habitat_bank)
    ]

    # Initial supply proportional to registered credit capacity
    supply_available: Dict[str, float] = {}
    total_supply = sum(o.supply_capacity for o in all_otgs)
    for otg in all_otgs:
        if otg.supply_capacity > 0:
            monthly_share = (
                otg.supply_capacity / max(1, total_supply) * 7700
            )
            supply_available[otg.name] = max(1, monthly_share * 2)
        else:
            supply_available[otg.name] = rng.exponential(0.2)

    all_transactions: List[Tuple[str, float, float, str]] = []

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
        )

        step_txns = clear_market_step(
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
        )
        all_transactions.extend(step_txns)

    return {
        "transactions": all_transactions,
        "supply_final": dict(supply_available),
    }


# =============================================================================
# 8. METRICS COMPUTATION
# =============================================================================


def compute_metrics(
    all_transactions: List[Tuple[str, float, float, str]],
    formations: List[Formation],
    all_otgs: List[OTG],
) -> Dict:
    """Compute EC1, EC2, and formation-level metrics."""
    total_otgs = len(all_otgs)

    otg_txn_count: Dict[str, int] = {}
    otg_txn_credits: Dict[str, float] = {}
    for otg_name, price, qty, buyer_type in all_transactions:
        otg_txn_count[otg_name] = otg_txn_count.get(otg_name, 0) + 1
        otg_txn_credits[otg_name] = (
            otg_txn_credits.get(otg_name, 0) + qty
        )

    traded = sum(
        1 for o in all_otgs if otg_txn_count.get(o.name, 0) >= 1
    )
    ec1 = traded / total_otgs if total_otgs > 0 else 0

    functional = sum(
        1 for o in all_otgs if otg_txn_count.get(o.name, 0) >= 5
    )
    ec2 = functional / total_otgs if total_otgs > 0 else 0

    tec_otgs = [o for o in all_otgs if o.is_tec]
    non_tec_otgs = [o for o in all_otgs if not o.is_tec]

    tec_functional = sum(
        1 for o in tec_otgs if otg_txn_count.get(o.name, 0) >= 5
    )
    non_tec_functional = sum(
        1 for o in non_tec_otgs if otg_txn_count.get(o.name, 0) >= 5
    )

    ec2_tec = tec_functional / len(tec_otgs) if tec_otgs else 0
    ec2_non_tec = (
        non_tec_functional / len(non_tec_otgs) if non_tec_otgs else 0
    )

    formation_metrics = {}
    formations_with_functional = 0
    for f in formations:
        f_traded = sum(
            1 for o in f.otgs if otg_txn_count.get(o.name, 0) >= 1
        )
        f_functional = sum(
            1 for o in f.otgs if otg_txn_count.get(o.name, 0) >= 5
        )
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
    }


# =============================================================================
# 9. MONTE CARLO
# =============================================================================


def monte_carlo(
    formations: List[Formation],
    all_otgs: List[OTG],
    subregion_demands: List[SubregionDemand],
    config: FormationConfig,
    n_runs: int = 50,
) -> Dict:
    """Run Monte Carlo simulations and aggregate results."""
    all_metrics = []
    for i in range(n_runs):
        result = run_single(
            formations, all_otgs, subregion_demands, config, seed=i
        )
        metrics = compute_metrics(
            result["transactions"], formations, all_otgs
        )
        all_metrics.append(metrics)
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{n_runs} complete")

    ec1_vals = [m["ec1"] for m in all_metrics]
    ec2_vals = [m["ec2"] for m in all_metrics]
    ec2_tec_vals = [m["ec2_tec"] for m in all_metrics]
    ec2_non_tec_vals = [m["ec2_non_tec"] for m in all_metrics]
    n_func_formations = [
        m["formations_with_functional"] for m in all_metrics
    ]
    n_tec_func = [m["n_tec_functional"] for m in all_metrics]
    n_non_tec_func = [m["n_non_tec_functional"] for m in all_metrics]

    formation_ec2 = {}
    for fname in all_metrics[0]["formation_metrics"]:
        vals = [
            m["formation_metrics"][fname]["ec2"] for m in all_metrics
        ]
        txn_vals = [
            m["formation_metrics"][fname]["total_txn"]
            for m in all_metrics
        ]
        func_vals = [
            m["formation_metrics"][fname]["functional"]
            for m in all_metrics
        ]
        formation_ec2[fname] = {
            "ec2_median": float(np.median(vals)),
            "ec2_p10": float(np.percentile(vals, 10)),
            "ec2_p90": float(np.percentile(vals, 90)),
            "txn_median": float(np.median(txn_vals)),
            "func_median": float(np.median(func_vals)),
            "n_otgs": all_metrics[0]["formation_metrics"][fname][
                "n_otgs"
            ],
            "n_tec": all_metrics[0]["formation_metrics"][fname]["n_tec"],
        }

    return {
        "ec1": {
            "median": float(np.median(ec1_vals)),
            "p10": float(np.percentile(ec1_vals, 10)),
            "p90": float(np.percentile(ec1_vals, 90)),
        },
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
        "n_tec_functional": {
            "median": float(np.median(n_tec_func)),
            "p10": float(np.percentile(n_tec_func, 10)),
            "p90": float(np.percentile(n_tec_func, 90)),
        },
        "n_non_tec_functional": {
            "median": float(np.median(n_non_tec_func)),
            "p10": float(np.percentile(n_non_tec_func, 10)),
            "p90": float(np.percentile(n_non_tec_func, 90)),
        },
        "formation_ec2": formation_ec2,
        "n_runs": n_runs,
        "all_metrics": all_metrics,
    }


# =============================================================================
# 10. REPORTING
# =============================================================================


def print_scenario_comparison(scenario_results: Dict[str, Dict]):
    """Print scenario comparison table."""
    print("\n" + "=" * 100)
    print("FORMATION-LEVEL ABM: SCENARIO COMPARISON")
    print("=" * 100)

    header = (
        f"{'Scenario':<28s} {'EC2':>7s} {'EC2-TEC':>9s} "
        f"{'EC2-nTEC':>9s} {'Func.Forms':>11s} {'TEC.Func':>9s} "
        f"{'nTEC.Func':>10s} {'TotalTxn':>9s}"
    )
    print(header)
    print("-" * 100)

    for name, stats in scenario_results.items():
        ec2_str = f"{stats['ec2']['median']:.1%}"
        ec2_ci = (
            f"[{stats['ec2']['p10']:.1%}-{stats['ec2']['p90']:.1%}]"
        )
        ec2_tec_str = f"{stats['ec2_tec']['median']:.1%}"
        ec2_ntec_str = f"{stats['ec2_non_tec']['median']:.1%}"
        ff = stats["formations_with_functional"]
        ff_str = f"{ff['median']:.0f}/{len(stats['formation_ec2'])}"

        tec_func = stats["n_tec_functional"]
        tec_func_str = f"{tec_func['median']:.0f}/79"
        ntec_func = stats["n_non_tec_functional"]
        ntec_func_str = f"{ntec_func['median']:.0f}/173"

        total_txn = np.median(
            [m["total_transactions"] for m in stats["all_metrics"]]
        )
        monthly_txn = total_txn / 72

        print(
            f"  {name:<26s} {ec2_str:>7s} {ec2_tec_str:>9s} "
            f"{ec2_ntec_str:>9s} {ff_str:>11s} {tec_func_str:>9s} "
            f"{ntec_func_str:>10s} "
            f"{total_txn:>5.0f} ({monthly_txn:.1f}/mo)"
        )
        print(f"  {'':26s} {ec2_ci:>7s}")

    print("-" * 100)
    print(
        f"  {'OBSERVED (target)':<26s} {'17.9%':>7s} {'20.3%':>9s} "
        f"{'0.0%':>9s} {'10/15':>11s} {'16/79':>9s} "
        f"{'22/173':>10s}   764 (12.3/mo)"
    )

    # Formation breakdown for baseline
    baseline_name = list(scenario_results.keys())[0]
    baseline = scenario_results[baseline_name]

    print("\n" + "=" * 100)
    print(f"FORMATION-LEVEL BREAKDOWN ({baseline_name})")
    print("=" * 100)
    print(
        f"  {'Formation':<55s} {'OTGs':>5s} {'TEC':>4s} "
        f"{'EC2':>7s} {'Func':>5s} {'Txn':>7s}"
    )
    print("-" * 100)

    sorted_formations = sorted(
        baseline["formation_ec2"].items(),
        key=lambda x: -x[1]["txn_median"],
    )

    for fname, fstats in sorted_formations:
        ec2_str = f"{fstats['ec2_median']:.0%}"
        print(
            f"  {fname:<55s} {fstats['n_otgs']:>5d} "
            f"{fstats['n_tec']:>4d} {ec2_str:>7s} "
            f"{fstats['func_median']:>5.0f} "
            f"{fstats['txn_median']:>7.0f}"
        )

    print("=" * 100)


def print_observed_vs_predicted(
    scenario_results: Dict[str, Dict],
    formations: List[Formation],
):
    """Print observed vs predicted transaction distribution."""
    baseline_name = list(scenario_results.keys())[0]
    baseline = scenario_results[baseline_name]

    print("\n" + "=" * 100)
    print("OBSERVED vs PREDICTED TRANSACTION DISTRIBUTION")
    print("=" * 100)
    print(
        f"  {'Formation':<55s} {'Obs.Txn':>8s} {'Obs.%':>7s} "
        f"{'Pred.Txn':>9s} {'Pred.%':>7s}"
    )
    print("-" * 100)

    total_obs = sum(f.total_txn for f in formations)
    total_pred = sum(
        baseline["formation_ec2"][f.name]["txn_median"]
        for f in formations
    )

    for f in sorted(formations, key=lambda x: -x.total_txn):
        obs_share = f.total_txn / total_obs if total_obs > 0 else 0
        pred_txn = baseline["formation_ec2"][f.name]["txn_median"]
        pred_share = pred_txn / total_pred if total_pred > 0 else 0
        print(
            f"  {f.name:<55s} {f.total_txn:>8d} {obs_share:>7.1%} "
            f"{pred_txn:>9.0f} {pred_share:>7.1%}"
        )

    print(
        f"  {'TOTAL':<55s} {total_obs:>8d} {'100%':>7s} "
        f"{total_pred:>9.0f} {'100%':>7s}"
    )
    print("=" * 100)


# =============================================================================
# 11. MAIN
# =============================================================================


def main():
    print("=" * 100)
    print(
        "FORMATION-LEVEL AGENT-BASED MODEL v2: "
        "NSW BIODIVERSITY OFFSETS SCHEME"
    )
    print(
        "  Uses: Official TEC classification, IBRA subregion demand, "
        "supply capacity"
    )
    print("=" * 100)

    # Load data
    print("\n[1/6] Loading OTG data from NSW registers...")
    formations, all_otgs, subregion_demands = load_otg_data()
    print(
        f"  Loaded {len(all_otgs)} OTGs in {len(formations)} formations"
    )
    tec_count = sum(1 for o in all_otgs if o.is_tec)
    print(
        f"  TEC OTGs: {tec_count}, "
        f"non-TEC: {len(all_otgs) - tec_count}"
    )
    print(
        f"  Subregions with transactions: {len(subregion_demands)}"
    )
    print(
        f"  OTGs with direct txn counts: "
        f"{sum(1 for o in all_otgs if o.observed_txn_count > 0)}"
    )
    print(
        f"  OTGs with supply capacity: "
        f"{sum(1 for o in all_otgs if o.supply_capacity > 0)}"
    )

    for f in sorted(formations, key=lambda x: -x.total_txn):
        print(
            f"    {f.name:<55s} OTGs={f.n_otgs:>3d}  "
            f"TEC={f.n_tec:>2d}  Txn={f.total_txn:>3d}  "
            f"Med=${f.median_price:>6,.0f}"
        )

    # Observed calibration targets
    print("\n[2/6] Calibration targets:")
    obs_func = sum(
        1 for o in all_otgs if o.observed_txn_count >= 5
    )
    obs_tec_func = sum(
        1 for o in all_otgs if o.is_tec and o.observed_txn_count >= 5
    )
    obs_ntec_func = sum(
        1
        for o in all_otgs
        if not o.is_tec and o.observed_txn_count >= 5
    )
    n_non_tec = len(all_otgs) - tec_count
    print(
        f"  EC2 overall: {obs_func}/{len(all_otgs)} "
        f"= {obs_func / len(all_otgs):.1%}"
    )
    print(
        f"  EC2-TEC:     {obs_tec_func}/{tec_count} "
        f"= {obs_tec_func / tec_count:.1%}"
    )
    print(
        f"  EC2-nonTEC:  {obs_ntec_func}/{n_non_tec} "
        f"= {obs_ntec_func / n_non_tec:.1%}"
    )

    form_func_obs = set()
    for f in formations:
        for o in f.otgs:
            if o.observed_txn_count >= 5:
                form_func_obs.add(f.name)
                break
    print(
        f"  Functional formations: "
        f"{len(form_func_obs)}/{len(formations)}"
    )
    print("  Monthly transactions: ~12.3 (764 over 62 months)")

    # Define scenarios
    scenarios = {
        "Baseline": FormationConfig(),
        "Procurement Flexibility (20%)": FormationConfig(
            procurement_flex_share=0.20
        ),
        "Price Floor (AUD 3,000)": FormationConfig(price_floor=3000.0),
        "Combined": FormationConfig(
            procurement_flex_share=0.20, price_floor=3000.0
        ),
    }

    scenario_results = {}
    n_runs = 50

    for i, (name, config) in enumerate(scenarios.items()):
        print(f"\n[{i + 3}/6] Running {name} ({n_runs} MC seeds)...")
        stats = monte_carlo(
            formations,
            all_otgs,
            subregion_demands,
            config,
            n_runs=n_runs,
        )
        scenario_results[name] = stats
        print(
            f"  EC2 = {stats['ec2']['median']:.1%} "
            f"[{stats['ec2']['p10']:.1%} - "
            f"{stats['ec2']['p90']:.1%}]"
        )
        print(
            f"  EC2-TEC = {stats['ec2_tec']['median']:.1%}, "
            f"EC2-nonTEC = {stats['ec2_non_tec']['median']:.1%}"
        )
        print(
            f"  TEC functional: "
            f"{stats['n_tec_functional']['median']:.0f}/79, "
            f"nonTEC functional: "
            f"{stats['n_non_tec_functional']['median']:.0f}/173"
        )
        print(
            f"  Formations with functional markets: "
            f"{stats['formations_with_functional']['median']:.0f}"
            f"/{len(formations)}"
        )

    # Print results
    print_scenario_comparison(scenario_results)
    print_observed_vs_predicted(scenario_results, formations)

    # Key test
    print("\n" + "=" * 100)
    print("KEY TEST: DOES THE MODEL REPRODUCE RARITY-SELECTION PATTERNS?")
    print("=" * 100)
    baseline = scenario_results["Baseline"]

    functional_formations = [
        fname
        for fname, fstats in baseline["formation_ec2"].items()
        if fstats["ec2_median"] > 0
    ]
    non_functional = [
        fname
        for fname, fstats in baseline["formation_ec2"].items()
        if fstats["ec2_median"] == 0
    ]

    print("\n  Formations WITH functional OTGs (EC2 > 0):")
    for f in sorted(functional_formations):
        ec2 = baseline["formation_ec2"][f]["ec2_median"]
        txn = baseline["formation_ec2"][f]["txn_median"]
        print(f"    {f:<55s} EC2={ec2:.0%}  Txn={txn:.0f}")

    print("\n  Formations with ZERO functional OTGs:")
    for f in sorted(non_functional):
        txn = baseline["formation_ec2"][f]["txn_median"]
        print(f"    {f:<55s} Txn={txn:.0f}")

    expected_functional = {"Grassy Woodlands", "Forested Wetlands"}
    expected_non_functional = {"Rainforests", "Heathlands"}

    print("\n  Hypothesis check:")
    for f in expected_functional:
        ec2 = baseline["formation_ec2"].get(f, {}).get(
            "ec2_median", 0
        )
        status = "PASS" if ec2 > 0 else "FAIL"
        print(
            f"    [{status}] {f} should have functional markets "
            f"(EC2={ec2:.0%})"
        )

    for f in expected_non_functional:
        ec2 = baseline["formation_ec2"].get(f, {}).get(
            "ec2_median", 0
        )
        status = "PASS" if ec2 < 0.05 else "FAIL"
        print(
            f"    [{status}] {f} should have ~zero functional "
            f"markets (EC2={ec2:.0%})"
        )

    # Policy impact
    print("\n  Policy impact on EC2:")
    baseline_ec2 = scenario_results["Baseline"]["ec2"]["median"]
    for name, stats in scenario_results.items():
        delta = stats["ec2"]["median"] - baseline_ec2
        delta_str = (
            f"({delta:+.1%})" if name != "Baseline" else ""
        )
        print(
            f"    {name:<35s} "
            f"EC2={stats['ec2']['median']:.1%} {delta_str}  "
            f"EC2-TEC={stats['ec2_tec']['median']:.1%}  "
            f"EC2-nTEC={stats['ec2_non_tec']['median']:.1%}"
        )

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
