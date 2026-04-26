# Mapping to the Scholl, Calinescu & Farmer (2021) Framework

**Reference:** "How market ecology explains market malfunction", *PNAS* 118(26): e2015574118.
**Reference implementation:** https://github.com/INET-Complexity/market-ecology

This document maps each element of the market-ecology framework to its corresponding implementation in this repository.

## Framework elements

| Element | Scholl et al. 2021 | This implementation |
|---|---|---|
| Domain | Financial asset market (single stock) | Biodiversity credit market (15 vegetation formations, 252 OTGs) |
| Implementation language | C++ core + Python wrapper | Python (NumPy / SciPy / Pandas) |
| Species (behavioural strategies) | Value (`dividend_discount`), Noise (`mean_reverting_noise`), Trend (`trend_follower`) | Cost-minimising compliance, mean-reversion intermediation, thin-market procurement, cost-floor supply |
| Population measure | Net asset value (NAV) per fund | Strategy population fixed by regulatory structure |
| Price formation | Walrasian tâtonnement | Formation-level matching with like-for-like rules |
| Interaction matrix | Computed numerically from ABM via finite-difference perturbation | 4×4 species-level computed by ABM perturbation; 15×15 formation-level derived from the GLV Jacobian (Jaccard subregion overlap + price competition) |
| Calibration | Standard market microstructure tests | NSW transaction register (1,124 priced transactions, Nov 2019 – Mar 2026); single calibration target ($P_{\text{variation}}$ matched to 35.4% Fund routing rate) |
| Primary outcome | Market efficiency, price dynamics | Functional coverage (FC): fraction of ecosystem credit types sustaining ≥5 transactions |

## Models in this repository

The paper uses two complementary models at the formation level.

1. **Formation-level ABM** (primary). Driver: `scripts/run_abm.py`. Implements the liquidity-feedback kernel, $P_{\text{variation}}$ calibration, and Fund-routing logic across 50 Monte Carlo seeds. `src/model/formation_model.py` provides the entity classes (OTG, Formation, SubregionDemand) and matching engine. Produces FC = 16.3% (observed: 17.9%).

2. **GLV mean-field model** (analytical). Driver: `src/model/glv_formation.py`. Provides numerically computed competitive-exclusion thresholds, irreversibility boundaries, and parameter sensitivity. Used for the community-matrix cross-check.

Supporting: `src/model/dynamics.py` (abstract mean-field GLV dynamics for stylised illustration).

## Community matrix

Following Scholl et al.'s methodology, the 4×4 species-level community matrix is computed numerically from the ABM via finite-difference perturbation of each species' population (20% perturbation, 20 Monte Carlo seeds). The Spearman rank correlation between off-diagonal elements of the ABM-derived matrix and the GLV $\alpha$ is $\rho = 0.689$ ($p < 10^{-30}$), confirming that the GLV captures the species-level competitive structure.

The 15×15 formation-level interaction matrix is derived from the GLV's analytical Jacobian, with off-diagonal entries combining geographic Jaccard overlap of IBRA subregions and price-ratio competition between formations.
