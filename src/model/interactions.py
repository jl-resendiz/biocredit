"""
Interaction matrices for biodiversity credit market ecology.

The interaction matrix alpha[i,j] encodes how species j affects species i:
  alpha[i,j] > 0  →  j COMPETES with i (reduces i's effective carrying capacity)
  alpha[i,j] < 0  →  j is MUTUALISTIC with i (increases i's effective space)
  alpha[i,j] = 0  →  no interaction
  alpha[i,i] = 1  →  intraspecific competition (self-limiting, normalised)

Key ecological relationships in biodiversity credit markets:
  Compliance ↔ HabitatBank: mutualism (reliable buyer ↔ reliable seller)
  Compliance → BCT-type: competition (same credits, unequal resources)
  Intermediary → BCT-type: strong competition (crowds out on price)
"""

import numpy as np


def default_alpha() -> np.ndarray:
    """
    Baseline interaction matrix.

    Species order: [Compliance, Intermediary, BCT-type, HabitatBank]
    """
    return np.array([
        #  Comp   Interm BCT    HBank
        [  1.00,  0.30, -0.10, -0.30],  # Compliance
        [ -0.20,  1.00,  0.05, -0.10],  # Intermediary
        [ -0.10,  0.60,  1.00, -0.30],  # BCT-type
        [ -0.25, -0.05, -0.20,  1.00],  # HabitatBank
    ])


def nsw_alpha() -> np.ndarray:
    """
    NSW-calibrated interaction matrix.

    Key empirical constraints:
    - Extreme buyer concentration (IPART buyer HHI 2966 in 2024-25, scale 0-10000)
    - BCT competes directly with compliance for same credits
    - CSF acts as mutualistic intermediary
    - Seller growth (+37% YoY) responds to compliance demand

    Species order: [Compliance, CSF, BCT, HabitatBank]
    """
    return np.array([
        #  Comp   CSF    BCT    HBank
        [  1.00, -0.15,  0.25, -0.30],  # Compliance
        [ -0.25,  1.00, -0.10, -0.20],  # CSF
        [  0.35, -0.10,  1.00, -0.25],  # BCT
        [ -0.30, -0.10, -0.15,  1.00],  # HabitatBank
    ])


def scenario_alpha(base: str = "default", **overrides) -> np.ndarray:
    """
    Generate scenario-specific interaction matrices.

    Parameters
    ----------
    base : str
        Starting matrix: 'default' or 'nsw'
    **overrides : dict
        Key-value pairs like spec_cons=0.8 to override specific interactions.
        Keys use format '{row_short}_{col_short}' with species abbreviations:
        comp, spec, bct, hbank (or csf, bct for NSW).

    Returns
    -------
    np.ndarray : Modified interaction matrix
    """
    alpha = default_alpha() if base == "default" else nsw_alpha()

    idx_map_default = {"comp": 0, "spec": 1, "bct": 2, "hbank": 3}
    idx_map_nsw = {"comp": 0, "csf": 1, "bct": 2, "hbank": 3}
    idx_map = idx_map_default if base == "default" else idx_map_nsw

    for key, value in overrides.items():
        parts = key.split("_")
        if len(parts) == 2 and parts[0] in idx_map and parts[1] in idx_map:
            i, j = idx_map[parts[0]], idx_map[parts[1]]
            alpha[i, j] = value

    return alpha
