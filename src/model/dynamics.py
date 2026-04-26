"""
Generalised Lotka-Volterra simulation engine.

Implements the core dynamics from Scholl, Calinescu & Farmer (2021),
adapted for biodiversity credit markets:

    dN_i/dt = r_i * N_i * (1 - (Σ_j α_ij * N_j) / K_i) + noise

Where:
    N_i   = capital (population) of species i
    r_i   = intrinsic growth rate
    K_i   = carrying capacity
    α_ij  = interaction coefficient (species j's effect on species i)
    noise = stochastic perturbation (Brownian)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

from src.model.species import SpeciesPool


@dataclass
class SimConfig:
    """Simulation configuration."""
    T: float = 120.0        # total time (months)
    dt: float = 0.1         # time step
    sigma: float = 0.05     # noise intensity
    seed: int = 42          # random seed
    price_fundamental: float = 100.0   # fundamental price
    price_sensitivity: float = 0.5     # price response to demand/supply


@dataclass
class SimResult:
    """Simulation output."""
    time: np.ndarray        # (n_steps,)
    N: np.ndarray           # (n_steps, n_species)
    price: np.ndarray       # (n_steps,)
    config: SimConfig = field(repr=False)
    species_names: List[str] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.time)

    @property
    def n_species(self) -> int:
        return self.N.shape[1]

    def total_demand(self, demand_indices: Optional[List[int]] = None) -> np.ndarray:
        """Sum of demand-side species populations over time."""
        if demand_indices is None:
            demand_indices = list(range(self.n_species - 1))  # all except last (supply)
        return self.N[:, demand_indices].sum(axis=1)

    def market_shares(self) -> np.ndarray:
        """Species shares of total population (n_steps, n_species)."""
        total = self.N.sum(axis=1, keepdims=True) + 1e-10
        return self.N / total

    def ec2_proxy(self, compliance_idx: int = 0,
                  bct_idx: int = 3) -> np.ndarray:
        """
        Proxy for Ecological Coverage 2 (EC2): fraction of demand-side capital
        held by agents with OTG-flexible procurement (compliance + BCT-type).
        Note: true EC2 requires OTG-level transaction tracking; this is a
        population-level approximation used in the GLV mean-field context.
        """
        demand = self.N[:, :self.n_species - 1].sum(axis=1) + 1e-10
        flexible = self.N[:, compliance_idx] + self.N[:, bct_idx]
        return flexible / demand



def compute_price(N: np.ndarray, N0: np.ndarray, config: SimConfig,
                  demand_indices: List[int] = None,
                  supply_index: int = -1) -> float:
    """Compute market clearing price from demand/supply ratio."""
    if demand_indices is None:
        demand_indices = list(range(len(N) - 1))

    demand = sum(N[i] for i in demand_indices) + 0.5
    supply = N[supply_index] + 0.5

    demand_0 = sum(N0[i] for i in demand_indices) + 0.5
    supply_0 = N0[supply_index] + 0.5

    ratio = (demand / supply) / (demand_0 / supply_0 + 1e-10)
    price = config.price_fundamental * ratio ** config.price_sensitivity
    return np.clip(price, config.price_fundamental * 0.01,
                   config.price_fundamental * 100)


def simulate(pool: SpeciesPool, alpha: np.ndarray,
             config: Optional[SimConfig] = None,
             shocks: Optional[List[Tuple[float, int, float]]] = None,
             ) -> SimResult:
    """
    Run a generalised Lotka-Volterra market simulation.

    Parameters
    ----------
    pool : SpeciesPool
        Species definitions (r, K, N0)
    alpha : np.ndarray
        Interaction matrix (n_species × n_species)
    config : SimConfig, optional
        Simulation parameters
    shocks : list of (time, species_index, magnitude), optional
        Exogenous shocks to growth rates at specified times

    Returns
    -------
    SimResult
    """
    if config is None:
        config = SimConfig()

    rng = np.random.RandomState(config.seed)
    n = pool.n
    n_steps = int(config.T / config.dt)

    N_hist = np.zeros((n_steps, n))
    price_hist = np.zeros(n_steps)
    t_hist = np.zeros(n_steps)

    N = pool.N0.copy()
    r = pool.r.copy()
    K = pool.K.copy()

    for step in range(n_steps):
        t = step * config.dt
        t_hist[step] = t
        N_hist[step] = N
        price_hist[step] = compute_price(N, pool.N0, config)

        # Effective growth rates (with price feedback)
        r_eff = r.copy()
        p_dev = (price_hist[step] - config.price_fundamental) / config.price_fundamental

        # Apply shocks
        if shocks:
            for shock_t, shock_species, shock_mag in shocks:
                if abs(t - shock_t) < config.dt:
                    r_eff[shock_species] += shock_mag

        # GLV dynamics
        interaction = alpha @ N
        logistic = 1.0 - interaction / K
        dN = r_eff * N * logistic * config.dt

        # Stochastic noise
        noise = config.sigma * N * rng.normal(0, 1, n) * np.sqrt(config.dt)

        # Update
        N = np.maximum(N + dN + noise, 1e-4)

    return SimResult(
        time=t_hist,
        N=N_hist,
        price=price_hist,
        config=config,
        species_names=pool.names,
    )
