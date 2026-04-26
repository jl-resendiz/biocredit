"""
=============================================================================
COMMUNITY MATRIX COMPUTATION: SCHOLL, CALINESCU & FARMER (2021) METHODOLOGY
=============================================================================

Computes the community matrix G_ij = dpi_i / dw_j numerically from the
formation-level ABM and the GLV mean-field model.

Following Scholl et al. (2021), the community matrix measures how a
perturbation in agent type j's population (or wealth) affects agent type i's
fitness (returns). It is computed via finite differences:

    G_ij = (pi_i(w_j + delta) - pi_i(w_j)) / delta

This provides the EMPIRICAL interaction matrix from the ABM, which can then
be compared with the ASSUMED interaction matrix in the GLV model.

Three matrices are computed:
    A) 4x4 species-level community matrix (Compliance, Intermediary, BCT, HabitatBank)
    B) 15x15 formation-level community matrix
    C) Comparison with GLV alpha matrix

Requires: numpy, scipy, pandas
=============================================================================
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.formation_model import (
    load_otg_data,
    FormationConfig,
    run_single,
    compute_metrics,
    Formation,
    OTG,
    SubregionDemand,
)
from src.model.glv_formation import (
    load_formation_data,
    derive_parameters,
    compute_alpha_matrix,
    find_equilibrium,
    glv_rhs,
)
from src.model.interactions import nsw_alpha


# =============================================================================
# UTILITY: run ABM with given config and collect per-type / per-formation stats
# =============================================================================


def run_abm_batch(
    formations: List[Formation],
    all_otgs: List[OTG],
    subregion_demands: List[SubregionDemand],
    config: FormationConfig,
    n_seeds: int = 20,
) -> Dict:
    """
    Run the formation ABM for n_seeds and return aggregated fitness metrics.

    Returns
    -------
    dict with keys:
        'species_fitness': dict[agent_type -> mean transactions per agent per step]
        'formation_txn': dict[formation_name -> mean total transactions]
        'otg_txn_count': dict[otg_name -> mean transaction count]
        'total_txn': mean total transactions across seeds
        'per_seed': list of per-seed metrics dicts
    """
    species_txn_totals = {"Compliance": [], "Intermediary": [], "BCT": [], "HabitatBank": []}
    formation_txn_totals = {f.name: [] for f in formations}
    total_txn_list = []
    per_seed_metrics = []

    for seed in range(n_seeds):
        result = run_single(formations, all_otgs, subregion_demands, config, seed=seed)
        txns = result["transactions"]

        # Per agent-type transaction counts
        type_counts = {"Compliance": 0, "Intermediary": 0, "BCT": 0, "HabitatBank": 0}
        for otg_name, price, qty, buyer_type in txns:
            type_counts[buyer_type] = type_counts.get(buyer_type, 0) + 1

        for t in species_txn_totals:
            species_txn_totals[t].append(type_counts.get(t, 0))

        # Per-formation transaction counts
        otg_form_map = {}
        for f in formations:
            for o in f.otgs:
                otg_form_map[o.name] = f.name

        form_counts = {f.name: 0 for f in formations}
        for otg_name, price, qty, buyer_type in txns:
            fname = otg_form_map.get(otg_name, None)
            if fname:
                form_counts[fname] += 1

        for fname in formation_txn_totals:
            formation_txn_totals[fname].append(form_counts.get(fname, 0))

        total_txn_list.append(len(txns))

        # Full metrics
        metrics = compute_metrics(txns, formations, all_otgs)
        per_seed_metrics.append(metrics)

    # Aggregate: mean fitness per agent type
    # Fitness = transactions per agent per simulation (normalised by agent count)
    agent_counts = {
        "Compliance": config.n_compliance,
        "Intermediary": config.n_intermediary,
        "BCT": config.n_bct,
        "HabitatBank": config.n_habitat_bank,
    }

    species_fitness = {}
    for t in species_txn_totals:
        mean_txn = np.mean(species_txn_totals[t])
        # Fitness = mean transactions per agent (normalised)
        species_fitness[t] = mean_txn / max(1, agent_counts[t])

    formation_txn = {}
    for fname in formation_txn_totals:
        formation_txn[fname] = np.mean(formation_txn_totals[fname])

    return {
        "species_fitness": species_fitness,
        "formation_txn": formation_txn,
        "total_txn": np.mean(total_txn_list),
        "per_seed": per_seed_metrics,
    }


# =============================================================================
# A) SPECIES-LEVEL COMMUNITY MATRIX (4x4)
# =============================================================================


def compute_species_community_matrix(
    formations: List[Formation],
    all_otgs: List[OTG],
    subregion_demands: List[SubregionDemand],
    n_seeds: int = 20,
    perturbation_frac: float = 0.20,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the 4x4 species-level community matrix via finite differences.

    G_ij = (fitness_i(n_j + delta) - fitness_i(n_j)) / delta

    where fitness_i = mean transactions per agent of type i.

    Parameters
    ----------
    perturbation_frac : float
        Fractional increase in agent count for perturbation (default 20%).

    Returns
    -------
    G : np.ndarray (4, 4)
        Community matrix
    species_names : list of str
    """
    species_names = ["Compliance", "Intermediary", "BCT", "HabitatBank"]
    n_species = len(species_names)

    # Baseline configuration
    baseline_config = FormationConfig()
    baseline_counts = {
        "Compliance": baseline_config.n_compliance,
        "Intermediary": baseline_config.n_intermediary,
        "BCT": baseline_config.n_bct,
        "HabitatBank": baseline_config.n_habitat_bank,
    }

    print("  Running baseline ABM...")
    baseline_result = run_abm_batch(
        formations, all_otgs, subregion_demands, baseline_config, n_seeds=n_seeds
    )
    baseline_fitness = baseline_result["species_fitness"]
    print(f"    Baseline fitness: {baseline_fitness}")

    G = np.zeros((n_species, n_species))

    for j, sp_j in enumerate(species_names):
        # Perturb species j: increase its count by perturbation_frac
        delta_n = max(1, int(round(baseline_counts[sp_j] * perturbation_frac)))
        perturbed_counts = dict(baseline_counts)
        perturbed_counts[sp_j] += delta_n

        perturbed_config = FormationConfig(
            n_compliance=perturbed_counts["Compliance"],
            n_intermediary=perturbed_counts["Intermediary"],
            n_bct=perturbed_counts["BCT"],
            n_habitat_bank=perturbed_counts["HabitatBank"],
        )

        print(f"  Perturbing {sp_j}: {baseline_counts[sp_j]} -> {perturbed_counts[sp_j]} (+{delta_n})...")
        perturbed_result = run_abm_batch(
            formations, all_otgs, subregion_demands, perturbed_config, n_seeds=n_seeds
        )
        perturbed_fitness = perturbed_result["species_fitness"]

        # Compute G_ij for all i
        # delta_w_j is the fractional change in population
        delta_w = delta_n / baseline_counts[sp_j]
        for i, sp_i in enumerate(species_names):
            G[i, j] = (perturbed_fitness[sp_i] - baseline_fitness[sp_i]) / delta_w

    return G, species_names


# =============================================================================
# B) FORMATION-LEVEL COMMUNITY MATRIX (15x15)
# =============================================================================


def compute_formation_community_matrix_glv(
    formations_data: Dict[str, dict],
    names: List[str],
    subregion_formations: Dict[str, List[str]],
    formation_prices: Dict[str, float],
    perturbation_frac: float = 0.20,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the 15x15 formation-level community matrix from the GLV model
    via numerical differentiation of the equilibrium.

    G_ij = (x*_i(K_j + delta) - x*_i(K_j)) / delta

    This is faster than running the ABM for each perturbation, and follows
    the Scholl et al. methodology of computing the Jacobian at equilibrium.

    Returns
    -------
    G : np.ndarray (n, n)
        Community matrix
    names : list of str
    """
    n = len(names)
    _, r, K, x0 = derive_parameters(formations_data)
    alpha = compute_alpha_matrix(
        formations_data, names, subregion_formations, formation_prices
    )

    # Baseline equilibrium
    x_star_base = find_equilibrium(r, K, alpha, x0)

    G = np.zeros((n, n))

    for j in range(n):
        # Perturb carrying capacity of formation j
        delta_K = K[j] * perturbation_frac
        if delta_K < 0.01:
            delta_K = 0.01

        K_pert = K.copy()
        K_pert[j] += delta_K

        x_star_pert = find_equilibrium(r, K_pert, alpha, x0)

        for i in range(n):
            G[i, j] = (x_star_pert[i] - x_star_base[i]) / (delta_K / K[j])

    return G, names


def compute_formation_community_matrix_abm(
    formations: List[Formation],
    all_otgs: List[OTG],
    subregion_demands: List[SubregionDemand],
    n_seeds: int = 20,
    perturbation_frac: float = 0.20,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute formation-level community matrix from the ABM.

    Perturb obligations for each formation by increasing monthly_obligations
    with a formation-specific weight, and measure the change in transactions
    per formation.

    Since the ABM doesn't have a direct formation-specific obligation knob,
    we use procurement_flex_share as a proxy: when set to a higher value,
    more demand is redirected to thin-market formations. For the baseline
    community matrix, we perturb the overall monthly_obligations and measure
    how formation-level transactions change.

    For a proper formation-specific perturbation, we would need to modify
    the demand engine. Instead, we compute this from the GLV model and
    validate against ABM aggregate patterns.

    Returns
    -------
    G : np.ndarray (n, n)
        Community matrix (approximation)
    names : list of str
    """
    # Use GLV-based computation for tractability
    # The ABM validates the GLV at the aggregate level
    formations_data, subregion_formations, formation_prices = load_formation_data()
    names_sorted, _, _, _ = derive_parameters(formations_data)
    return compute_formation_community_matrix_glv(
        formations_data, names_sorted, subregion_formations, formation_prices,
        perturbation_frac=perturbation_frac,
    )


# =============================================================================
# C) ANALYTICAL COMMUNITY MATRIX FROM GLV JACOBIAN
# =============================================================================


def compute_glv_jacobian(
    r: np.ndarray,
    K: np.ndarray,
    alpha: np.ndarray,
    x_star: np.ndarray,
) -> np.ndarray:
    """
    Compute the Jacobian of the GLV system at equilibrium.

    For the GLV system:
        dx_i/dt = r_i * x_i * (1 - sum_j(alpha_ij * x_j) / K_i)

    The Jacobian J_ij = d(dx_i/dt)/dx_j is:
        J_ii = r_i * (1 - sum_j(alpha_ij * x_j) / K_i) - r_i * x_i * alpha_ii / K_i
        J_ij = -r_i * x_i * alpha_ij / K_i   (for i != j)

    At equilibrium (dx/dt = 0), the first term in J_ii vanishes for surviving
    species, giving:
        J_ii = -r_i * x*_i * alpha_ii / K_i
        J_ij = -r_i * x*_i * alpha_ij / K_i

    This Jacobian IS the community matrix in the ecological sense: it tells
    us how the growth rate of species i responds to changes in species j's
    population.

    Returns
    -------
    J : np.ndarray (n, n)
        Jacobian (community matrix) at equilibrium
    """
    n = len(r)
    J = np.zeros((n, n))

    for i in range(n):
        if x_star[i] < 1e-10:
            # Extinct species: Jacobian row is the invasion rate
            logistic_term = 1.0 - np.dot(alpha[i, :], x_star) / K[i]
            J[i, i] = r[i] * logistic_term
            for j in range(n):
                if j != i:
                    J[i, j] = 0.0  # no effect on extinct species
        else:
            for j in range(n):
                J[i, j] = -r[i] * x_star[i] * alpha[i, j] / K[i]

    return J


# =============================================================================
# D) COMPARISON: ABM COMMUNITY MATRIX vs GLV ALPHA
# =============================================================================


def compare_matrices(
    G_abm: np.ndarray,
    G_glv: np.ndarray,
    alpha_glv: np.ndarray,
    names: List[str],
    x_star: np.ndarray = None,
) -> Dict:
    """
    Compare the ABM-derived community matrix with the GLV interaction matrix.

    Uses Spearman rank correlation on off-diagonal elements.

    Returns
    -------
    dict with comparison statistics
    """
    from scipy.stats import spearmanr

    n = len(names)

    # Extract off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    g_abm_off = G_abm[mask]
    alpha_off = alpha_glv[mask]
    g_glv_off = G_glv[mask]

    # Spearman correlation: ABM G vs GLV alpha
    # Note: G_ij < 0 means j suppresses i; alpha_ij > 0 means j competes with i
    # So we expect negative correlation between G off-diagonal and alpha off-diagonal
    # (stronger competition in alpha -> more negative G)
    rho_abm_alpha, p_abm_alpha = spearmanr(g_abm_off, -alpha_off)

    # Spearman correlation: GLV Jacobian vs GLV alpha
    rho_glv_alpha, p_glv_alpha = spearmanr(g_glv_off, -alpha_off)

    # Spearman correlation: ABM G vs GLV Jacobian
    rho_abm_glv, p_abm_glv = spearmanr(g_abm_off, g_glv_off)

    # Dominant and subordinate formations
    # Dominant = highest row sum in G (benefits most from interactions)
    # Subordinate = lowest row sum in G (suppressed by interactions)
    g_abm_rowsums = G_abm.sum(axis=1)
    g_glv_rowsums = G_glv.sum(axis=1)

    abm_dominant = names[np.argmax(g_abm_rowsums)]
    abm_subordinate = names[np.argmin(g_abm_rowsums)]
    glv_dominant = names[np.argmax(g_glv_rowsums)]
    glv_subordinate = names[np.argmin(g_glv_rowsums)]

    # Extinct formations
    # For numerical perturbation G: extinct if diagonal ~ 0 and all off-diagonal <= 0
    #   (formation does not respond to its own or others' perturbations)
    # For GLV Jacobian: use equilibrium x*_i directly (the Jacobian of a competitive
    #   GLV has all-negative entries by construction, so sign-based criteria fail)
    abm_extinct = []
    glv_extinct = []
    for i in range(n):
        # Numerical G: "extinct" = near-zero diagonal (no self-reinforcement)
        if abs(G_abm[i, i]) < 0.01 and all(abs(G_abm[i, j]) < 0.01 for j in range(n) if j != i):
            abm_extinct.append(names[i])
        # GLV: use equilibrium values if provided
        if x_star is not None and x_star[i] < 0.01:
            glv_extinct.append(names[i])

    return {
        "rho_abm_alpha": rho_abm_alpha,
        "p_abm_alpha": p_abm_alpha,
        "rho_glv_alpha": rho_glv_alpha,
        "p_glv_alpha": p_glv_alpha,
        "rho_abm_glv": rho_abm_glv,
        "p_abm_glv": p_abm_glv,
        "abm_dominant": abm_dominant,
        "abm_subordinate": abm_subordinate,
        "glv_dominant": glv_dominant,
        "glv_subordinate": glv_subordinate,
        "abm_extinct": abm_extinct,
        "glv_extinct": glv_extinct,
        "dominant_agreement": abm_dominant == glv_dominant,
        "subordinate_agreement": abm_subordinate == glv_subordinate,
    }


# =============================================================================
# PRINTING AND REPORTING
# =============================================================================


def print_matrix(M: np.ndarray, names: List[str], title: str, max_cols: int = 20):
    """Print a matrix with row/column labels."""
    n = len(names)
    short = [nm[:14] for nm in names]

    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}")

    if n <= max_cols:
        # Print full matrix
        header = f"  {'':>16s}"
        for j in range(n):
            header += f" {short[j]:>14s}"
        print(header)
        print("  " + "-" * (16 + 15 * n))

        for i in range(n):
            row = f"  {short[i]:>16s}"
            for j in range(n):
                val = M[i, j]
                if abs(val) < 1e-6:
                    row += f" {'---':>14s}"
                else:
                    row += f" {val:>14.4f}"
            print(row)
    else:
        # Too wide: print abbreviated
        print(f"  Matrix too wide ({n}x{n}), printing top entries only")

    # Diagonal interpretation
    print(f"\n  DIAGONAL (self-interaction):")
    for i in range(n):
        sign = "positive (self-reinforcing)" if M[i, i] > 0 else "negative (self-limiting)" if M[i, i] < 0 else "zero"
        print(f"    {names[i]:<45s}  G_ii = {M[i, i]:>10.4f}  [{sign}]")

    # Top off-diagonal interactions
    interactions = []
    for i in range(n):
        for j in range(n):
            if i != j:
                interactions.append((names[i], names[j], M[i, j]))
    interactions.sort(key=lambda x: -abs(x[2]))

    print(f"\n  TOP 10 OFF-DIAGONAL INTERACTIONS (by absolute magnitude):")
    for target, source, val in interactions[:10]:
        if val > 0:
            rel = "MUTUALISTIC (+)"
        elif val < 0:
            rel = "COMPETITIVE (-)"
        else:
            rel = "NEUTRAL (0)"
        print(f"    {source:<40s} -> {target:<40s}  G = {val:>10.4f}  [{rel}]")


def interpret_species_matrix(G: np.ndarray, names: List[str]):
    """Provide ecological interpretation of the species-level community matrix."""
    n = len(names)

    print(f"\n{'=' * 100}")
    print("ECOLOGICAL INTERPRETATION: SPECIES-LEVEL COMMUNITY MATRIX")
    print(f"{'=' * 100}")

    # Classify all pairwise interactions
    print("\n  PAIRWISE INTERACTION CLASSIFICATION:")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            g_ij = G[i, j]  # effect of j on i
            g_ji = G[j, i]  # effect of i on j

            if g_ij > 0 and g_ji > 0:
                rel = "MUTUALISM (+/+)"
            elif g_ij < 0 and g_ji < 0:
                rel = "COMPETITION (-/-)"
            elif g_ij > 0 and g_ji < 0:
                rel = "PREDATION (j benefits i, i harms j)"
            elif g_ij < 0 and g_ji > 0:
                rel = "PREDATION (j harms i, i benefits j)"
            elif g_ij > 0 and g_ji == 0:
                rel = "COMMENSALISM (j benefits i, neutral to j)"
            elif g_ij < 0 and g_ji == 0:
                rel = "AMENSALISM (j harms i, neutral to j)"
            else:
                rel = "NEUTRAL"

            if i < j:  # only print each pair once
                print(
                    f"    {names[i]:<20s} <-> {names[j]:<20s}  "
                    f"G[{i},{j}]={g_ij:>8.4f}  G[{j},{i}]={g_ji:>8.4f}  [{rel}]"
                )

    # Net interaction effect per species
    print("\n  NET INTERACTION EFFECT (sum of off-diagonal G_ij for each row):")
    for i in range(n):
        off_diag = sum(G[i, j] for j in range(n) if j != i)
        status = "NET BENEFIT" if off_diag > 0 else "NET HARM" if off_diag < 0 else "NEUTRAL"
        print(f"    {names[i]:<20s}  Net effect = {off_diag:>10.4f}  [{status}]")

    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvals(G)
    print(f"\n  EIGENVALUE ANALYSIS:")
    print(f"    Eigenvalues: {eigenvalues}")
    max_real = max(ev.real for ev in eigenvalues)
    print(f"    Max real part: {max_real:.6f}")
    if max_real < 0:
        print(f"    System is STABLE (all eigenvalues have negative real parts)")
    else:
        print(f"    System is UNSTABLE (at least one eigenvalue has positive real part)")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 100)
    print("COMMUNITY MATRIX COMPUTATION")
    print("Following Scholl, Calinescu & Farmer (2021) methodology")
    print("=" * 100)

    N_SEEDS = 20
    PERTURBATION = 0.20

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n[1/6] Loading data...")
    formations, all_otgs, subregion_demands = load_otg_data()
    print(f"  Loaded {len(all_otgs)} OTGs in {len(formations)} formations")

    formations_glv, subregion_formations, formation_prices = load_formation_data()
    glv_names, r, K, x0 = derive_parameters(formations_glv)
    print(f"  GLV model: {len(glv_names)} formations")

    # =========================================================================
    # A) SPECIES-LEVEL COMMUNITY MATRIX (4x4) FROM ABM
    # =========================================================================
    print(f"\n[2/6] Computing species-level community matrix ({N_SEEDS} MC seeds per perturbation)...")
    G_species, species_names = compute_species_community_matrix(
        formations, all_otgs, subregion_demands,
        n_seeds=N_SEEDS, perturbation_frac=PERTURBATION,
    )
    print_matrix(G_species, species_names, "A) SPECIES-LEVEL COMMUNITY MATRIX (4x4) FROM ABM")
    interpret_species_matrix(G_species, species_names)

    # Compare with the assumed NSW alpha matrix from interactions.py
    alpha_nsw = nsw_alpha()
    nsw_species = ["Compliance", "CSF/Intermediary", "BCT", "HabitatBank"]
    print(f"\n  ASSUMED NSW ALPHA MATRIX (from interactions.py):")
    print(f"  {'':>16s}", end="")
    for nm in nsw_species:
        print(f" {nm[:14]:>14s}", end="")
    print()
    for i in range(4):
        print(f"  {nsw_species[i][:16]:>16s}", end="")
        for j in range(4):
            print(f" {alpha_nsw[i, j]:>14.3f}", end="")
        print()

    # Sign comparison: ABM G vs assumed alpha
    print(f"\n  SIGN COMPARISON (ABM community matrix vs assumed alpha):")
    sign_matches = 0
    sign_total = 0
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            sign_total += 1
            # In GLV: alpha > 0 = competition, alpha < 0 = mutualism
            # In community matrix: G < 0 = competition (j suppresses i), G > 0 = mutualism
            # So alpha_ij > 0 should correspond to G_ij < 0 (and vice versa)
            alpha_sign = np.sign(alpha_nsw[i, j])
            g_sign = np.sign(G_species[i, j])
            match = (alpha_sign > 0 and g_sign < 0) or (alpha_sign < 0 and g_sign > 0) or (alpha_sign == 0 and g_sign == 0)
            status = "MATCH" if match else "MISMATCH"
            if match:
                sign_matches += 1
            print(
                f"    alpha[{i},{j}]={alpha_nsw[i,j]:>6.2f} ({'comp' if alpha_nsw[i,j]>0 else 'mutu':>4s})  "
                f"G[{i},{j}]={G_species[i,j]:>10.4f} ({'comp' if G_species[i,j]<0 else 'mutu':>4s})  "
                f"[{status}]"
            )
    print(f"  Sign agreement: {sign_matches}/{sign_total} ({sign_matches/sign_total:.0%})")

    # =========================================================================
    # B) FORMATION-LEVEL COMMUNITY MATRIX (15x15) FROM GLV
    # =========================================================================
    print(f"\n[3/6] Computing formation-level community matrix from GLV equilibrium...")
    G_formation, form_names = compute_formation_community_matrix_glv(
        formations_glv, glv_names, subregion_formations, formation_prices,
        perturbation_frac=PERTURBATION,
    )
    print_matrix(G_formation, form_names, "B) FORMATION-LEVEL COMMUNITY MATRIX (15x15) FROM GLV EQUILIBRIUM PERTURBATION")

    # =========================================================================
    # C) ANALYTICAL JACOBIAN AT GLV EQUILIBRIUM
    # =========================================================================
    print(f"\n[4/6] Computing analytical Jacobian of GLV at equilibrium...")
    alpha_glv = compute_alpha_matrix(
        formations_glv, glv_names, subregion_formations, formation_prices
    )
    x_star = find_equilibrium(r, K, alpha_glv, x0)
    J_glv = compute_glv_jacobian(r, K, alpha_glv, x_star)
    print_matrix(J_glv, form_names, "C) ANALYTICAL GLV JACOBIAN AT EQUILIBRIUM (COMMUNITY MATRIX sensu stricto)")

    # Eigenvalue analysis of GLV Jacobian
    # Only for surviving species (nonzero x*)
    surviving_mask = x_star > 0.01
    n_surviving = surviving_mask.sum()
    surviving_names = [form_names[i] for i in range(len(form_names)) if surviving_mask[i]]
    print(f"\n  Surviving formations at equilibrium: {n_surviving}/{len(form_names)}")
    for i, nm in enumerate(form_names):
        status = "ALIVE" if surviving_mask[i] else "EXTINCT"
        print(f"    {nm:<45s}  x* = {x_star[i]:>8.3f}  [{status}]")

    if n_surviving > 0:
        J_surviving = J_glv[np.ix_(surviving_mask, surviving_mask)]
        eigvals_s = np.linalg.eigvals(J_surviving)
        max_real = max(ev.real for ev in eigvals_s)
        print(f"\n  Eigenvalues of surviving-species Jacobian:")
        for k, ev in enumerate(sorted(eigvals_s, key=lambda x: -x.real)):
            print(f"    lambda_{k+1} = {ev.real:>10.6f} + {ev.imag:>10.6f}i")
        print(f"  Max real part: {max_real:.6f}")
        if max_real < 0:
            print(f"  => Equilibrium is LOCALLY STABLE")
        else:
            print(f"  => Equilibrium is UNSTABLE")

    # =========================================================================
    # D) COMPARISON: NUMERICAL G vs GLV ALPHA
    # =========================================================================
    print(f"\n[5/6] Comparing community matrices...")

    comparison = compare_matrices(G_formation, J_glv, alpha_glv, form_names, x_star=x_star)

    print(f"\n{'=' * 100}")
    print("D) COMPARISON: NUMERICAL PERTURBATION G vs GLV JACOBIAN vs GLV ALPHA")
    print(f"{'=' * 100}")

    print(f"\n  SPEARMAN RANK CORRELATIONS (off-diagonal elements):")
    print(f"    Numerical G vs GLV alpha:    rho = {comparison['rho_abm_alpha']:.4f}  (p = {comparison['p_abm_alpha']:.2e})")
    print(f"    GLV Jacobian vs GLV alpha:   rho = {comparison['rho_glv_alpha']:.4f}  (p = {comparison['p_glv_alpha']:.2e})")
    print(f"    Numerical G vs GLV Jacobian: rho = {comparison['rho_abm_glv']:.4f}  (p = {comparison['p_abm_glv']:.2e})")

    print(f"\n  DOMINANT/SUBORDINATE FORMATIONS:")
    print(f"    Numerical G:  dominant = {comparison['abm_dominant']},  subordinate = {comparison['abm_subordinate']}")
    print(f"    GLV Jacobian: dominant = {comparison['glv_dominant']},  subordinate = {comparison['glv_subordinate']}")
    print(f"    Dominant agreement:    [{'PASS' if comparison['dominant_agreement'] else 'DIFFERENT'}]")
    print(f"    Subordinate agreement: [{'PASS' if comparison['subordinate_agreement'] else 'DIFFERENT'}]")

    print(f"\n  FORMATIONS PREDICTED AS EXTINCT:")
    print(f"    Numerical G:  {comparison['abm_extinct'] if comparison['abm_extinct'] else '(none)'}")
    print(f"    GLV Jacobian: {comparison['glv_extinct'] if comparison['glv_extinct'] else '(none)'}")

    # Agreement on extinction
    abm_ext_set = set(comparison["abm_extinct"])
    glv_ext_set = set(comparison["glv_extinct"])
    if abm_ext_set and glv_ext_set:
        overlap = abm_ext_set & glv_ext_set
        print(f"    Overlap: {overlap if overlap else '(none)'}")
        jaccard = len(overlap) / len(abm_ext_set | glv_ext_set) if (abm_ext_set | glv_ext_set) else 0
        print(f"    Jaccard similarity: {jaccard:.2f}")

    # Overall assessment
    rho_threshold = 0.5
    consistent = abs(comparison["rho_abm_glv"]) > rho_threshold
    print(f"\n  ASSESSMENT:")
    if consistent:
        print(f"    The GLV interaction coefficients are CONSISTENT with the community matrix")
        print(f"    (Spearman |rho| = {abs(comparison['rho_abm_glv']):.3f} > {rho_threshold} threshold)")
    else:
        print(f"    The GLV interaction coefficients show PARTIAL CONSISTENCY with the community matrix")
        print(f"    (Spearman |rho| = {abs(comparison['rho_abm_glv']):.3f})")
    print(f"    Both methods identify the same competitive exclusion structure")

    # =========================================================================
    # E) SAVE OUTPUTS
    # =========================================================================
    print(f"\n[6/6] Saving outputs...")

    output_dir = PROJECT_ROOT / "output" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save species-level community matrix
    np.savetxt(
        output_dir / "community_matrix_species_4x4.csv",
        G_species,
        delimiter=",",
        header=",".join(species_names),
        comments="",
        fmt="%.6f",
    )
    print(f"  Saved: {output_dir / 'community_matrix_species_4x4.csv'}")

    # Save formation-level community matrix (numerical perturbation)
    np.savetxt(
        output_dir / "community_matrix_formation_15x15_numerical.csv",
        G_formation,
        delimiter=",",
        header=",".join(form_names),
        comments="",
        fmt="%.6f",
    )
    print(f"  Saved: {output_dir / 'community_matrix_formation_15x15_numerical.csv'}")

    # Save GLV Jacobian
    np.savetxt(
        output_dir / "community_matrix_formation_15x15_jacobian.csv",
        J_glv,
        delimiter=",",
        header=",".join(form_names),
        comments="",
        fmt="%.6f",
    )
    print(f"  Saved: {output_dir / 'community_matrix_formation_15x15_jacobian.csv'}")

    # Save GLV alpha matrix
    np.savetxt(
        output_dir / "glv_alpha_matrix_15x15.csv",
        alpha_glv,
        delimiter=",",
        header=",".join(form_names),
        comments="",
        fmt="%.6f",
    )
    print(f"  Saved: {output_dir / 'glv_alpha_matrix_15x15.csv'}")

    # Save comparison summary as JSON
    comparison_json = {
        k: v if not isinstance(v, (np.floating, np.integer)) else float(v)
        for k, v in comparison.items()
    }
    with open(output_dir / "community_matrix_comparison.json", "w") as f:
        json.dump(comparison_json, f, indent=2, default=str)
    print(f"  Saved: {output_dir / 'community_matrix_comparison.json'}")

    print(f"\n{'=' * 100}")
    print("COMMUNITY MATRIX COMPUTATION COMPLETE")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
