"""
Species definitions for the biodiversity credit market ecology model.

Each market participant type is modelled as a biological species with:
- r: intrinsic growth rate (how fast capital grows under ideal conditions)
- K: carrying capacity (maximum sustainable capital in the market)
- N0: initial population (starting capital)

Species classification follows functional ecology principles:
participants are grouped by their behavioural strategy, not their
institutional identity.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class Species:
    """A single species (market participant type)."""
    name: str
    short: str          # abbreviated label for plots
    r: float            # intrinsic growth rate
    K: float            # carrying capacity
    N0: float           # initial population
    color: str          # plot color
    description: str = ""


@dataclass
class SpeciesPool:
    """Complete set of species for a market simulation."""
    species: List[Species]

    @property
    def n(self) -> int:
        return len(self.species)

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.species]

    @property
    def colors(self) -> List[str]:
        return [s.color for s in self.species]

    @property
    def r(self) -> np.ndarray:
        return np.array([s.r for s in self.species])

    @property
    def K(self) -> np.ndarray:
        return np.array([s.K for s in self.species])

    @property
    def N0(self) -> np.ndarray:
        return np.array([s.N0 for s in self.species])


# ── Default species pool (Phase 1 baseline) ─────────────────

DEFAULT_SPECIES = SpeciesPool([
    Species(
        name="Compliance Buyers",
        short="Compliance",
        r=0.03, K=500, N0=100,
        color="#2166AC",
        description=(
            "Developers and infrastructure proponents legally required to "
            "offset biodiversity impacts. Price-insensitive within regulatory "
            "limits. Dominant species in mature compliance markets (e.g. NSW)."
        ),
    ),
    Species(
        name="Intermediaries",
        short="Intermediary",
        r=0.08, K=200, N0=30,
        color="#E31A1C",
        description=(
            "Participants who buy credits to resell at profit, or intermediaries "
            "who warehouse credits ahead of demand. Fast growth, momentum-driven. "
            "Includes government intermediaries like NSW Credits Supply Fund."
        ),
    ),
    Species(
        name="BCT-type (Procurement Flexible)",
        short="BCT-type",
        r=0.01, K=150, N0=50,
        color="#FF7F00",
        description=(
            "Compliance intermediaries with procurement flexibility: willing to pay "
            "above-compliance rates in thin markets, providing the only viable demand "
            "signal for rare OTGs. Calibrated to BCT empirical behaviour: median "
            "AUD 5,989 in thin markets vs AUD 1,125 for direct compliance (5.3x ratio). "
            "BCT retires credits on behalf of developer obligations via the Biodiversity "
            "Conservation Fund. Procurement flexibility — not least-cost selection — "
            "is the key functional difference from direct compliance buyers."
        ),
    ),
    Species(
        name="Habitat Banks",
        short="HabitatBank",
        r=0.04, K=600, N0=120,
        color="#6A3D9A",
        description=(
            "Landholders who create biodiversity stewardship agreements and generate "
            "credits. Supply side of the market. Growth responds to demand signals "
            "and price incentives. Mutualistic with compliance buyers."
        ),
    ),
])


# ── NSW-calibrated species pool (Phase 2) ────────────────────

NSW_SPECIES = SpeciesPool([
    Species("Compliance Buyers", "Compliance",
            r=0.045, K=35.0, N0=0.5, color="#2166AC"),
    Species("CSF / Intermediary", "CSF",
            r=0.015, K=5.0, N0=0.05, color="#E31A1C"),
    Species("BCT (compliance intermediary)", "BCT",
            r=0.02, K=8.0, N0=0.1, color="#FF7F00"),
    Species("Habitat Banks", "HabitatBank",
            r=0.04, K=40.0, N0=0.3, color="#6A3D9A"),
])
