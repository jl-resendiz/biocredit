#!/usr/bin/env python3
"""
Global sensitivity analysis (Sobol indices) for the formation-level ABM.

Replaces the OFAT sweeps in `scripts/robustness_sensitivity.py` with a
variance-based, Saltelli-sampled global analysis as required by
Saltelli et al. (2008) and Harenberg et al. (2019, Quantitative Economics).

Usage:
    python scripts/sensitivity_sobol.py [--N 512] [--inner-seeds 5] [--workers 8]

Default design:
    k = 6 sensitivity-relevant parameters (uniform priors documented below)
    N = 512  (power of 2 for Sobol' sequence quality)
    Total Saltelli samples = N * (k + 2) = 4,096
    Inner Monte Carlo seeds per Saltelli row = 5 (to denoise stochastic FC)
    Total ABM runs = 4,096 * 5 = 20,480

Wall time:
    ~30 s per ABM run on a single core
    -> single-core: ~5 days
    -> 8 cores: ~17 hours (recommended)

Output:
    output/results/sobol_indices.json
        - problem dict (parameters, bounds, distributions)
        - per-sample raw FC values (n=4,096)
        - first-order Sobol S_i and total Sobol S_Ti per parameter
        - bootstrap 95% confidence intervals (B=1,000)
        - convergence diagnostics

Pre-registered seed: 42 (set in problem dict).

Author: Jose Luis Resendiz (assisted)
Date:   2026-04-26 (initial scaffold)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Make the repo root importable so `from scripts.run_abm import ...` works
# regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np

try:
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import saltelli as saltelli_sample
except ImportError:
    print(
        "ERROR: SALib is not installed. Install with `pip install SALib>=1.4`.\n"
        "It is also pinned in requirements.txt.",
        file=sys.stderr,
    )
    sys.exit(2)


# =============================================================================
# PROBLEM DICT — pre-registered parameter ranges
# =============================================================================
#
# These six parameters are the sensitivity-relevant subset of the ABM. Bounds
# are documented per parameter:
#
#   P_variation:   bounds [0.3, 0.7] match the manuscript's calibration grid
#                  (Supp. Table 6).
#   n_BCT:         bounds [8, 16] cover the institutional design range; the
#                  manuscript reports 5.4pp FC sensitivity within this band.
#   production_cost_mu (mu_log_c):
#                  bounds [7.4, 7.85] are roughly +/-25% around the calibrated
#                  log-mean 7.62.
#   power_law_alpha:
#                  bounds [2.0, 6.0] match the within-formation concentration
#                  sweep range in Supp. Table 5.
#   liquidity_weight_max:
#                  bounds [5.0, 12.0] bracket the empirical 8.4x maximum
#                  liquidity-feedback weight by ~+/-40%.
#   monthly_obligations_mult:
#                  bounds [0.7, 1.3] perturb the calibrated obligation rate
#                  by +/-30%.
#
# All distributions are uniform on the bounds (least-informative); future
# work could substitute Bayesian priors.
# =============================================================================

PROBLEM_K6: Dict = {
    "num_vars": 6,
    "names": [
        "P_variation",
        "n_BCT",
        "production_cost_mu",
        "power_law_alpha",
        "liquidity_weight_max",
        "monthly_obligations_mult",
    ],
    "bounds": [
        [0.3, 0.7],
        [8.0, 16.0],
        [7.40, 7.85],
        [2.0, 6.0],
        [5.0, 12.0],
        [0.7, 1.3],
    ],
    "dists": ["unif"] * 6,
}

# Reduced problem dict containing only parameters that flow through the
# existing FormationConfig fields without additional code in run_abm.py.
# These three are the ones a runnable Sobol scan can use TODAY:
#
#   P_variation:                FormationConfig.p_variation
#   n_BCT:                      FormationConfig.n_bct
#   monthly_obligations_mult:   scales FormationConfig.monthly_obligations
#
# The remaining three (production_cost_mu, power_law_alpha,
# liquidity_weight_max) require small additional plumbing in run_abm.py
# to consume from the PARAMS dict; they are kept in PROBLEM_K6 for
# completeness once that plumbing is added.
PROBLEM_K3: Dict = {
    "num_vars": 3,
    "names": [
        "P_variation",
        "n_BCT",
        "monthly_obligations_mult",
    ],
    "bounds": [
        [0.3, 0.7],
        [8.0, 16.0],
        [0.7, 1.3],
    ],
    "dists": ["unif"] * 3,
}

QOI = "functional_coverage"  # Quantity of interest
PRE_REGISTERED_SEED = 42


# =============================================================================
# Single-evaluation wrapper around the ABM
# =============================================================================


def evaluate_one_sample(
    sample_idx: int,
    params: np.ndarray,
    param_names: List[str],
    inner_seeds: int,
) -> Dict:
    """Run the ABM at one Saltelli sample point and return mean FC over inner seeds.

    `params` is the parameter vector in problem-dict order; `param_names`
    must match the active problem dict (PROBLEM_K3 or PROBLEM_K6).
    """
    from scripts.run_abm import (
        FormationConfig,
        load_otg_data_formation_matching,
        monte_carlo,
    )
    import scripts.run_abm as run_abm_mod

    # Build a name->value map for the current sample
    pmap = {name: float(val) for name, val in zip(param_names, params)}

    # Defaults for any k=3 / k=6 dimension not in the current sample
    p_var = pmap.get("P_variation", 0.5)
    n_bct = int(round(pmap.get("n_BCT", 12.0)))
    oblig_mult = pmap.get("monthly_obligations_mult", 1.0)

    # k=6 extensions (only consumed if the corresponding code paths in
    # run_abm.py read from PARAMS; otherwise these are passed but ignored).
    # The smoke-test confirms the k=3 path; k=6 wiring is described in
    # docs/SOBOL_INTEGRATION_NOTES.md.
    if "production_cost_mu" in pmap:
        run_abm_mod.PARAMS["production_cost_mu_override"] = pmap["production_cost_mu"]
    if "power_law_alpha" in pmap:
        run_abm_mod.PARAMS["power_law_alpha_override"] = pmap["power_law_alpha"]
    if "liquidity_weight_max" in pmap:
        run_abm_mod.PARAMS["liquidity_weight_max_override"] = pmap["liquidity_weight_max"]

    # Each call loads data fresh (cheap, ~1s) and overrides parameters.
    # load_otg_data_formation_matching returns
    # (formations, all_otgs, subregion_demands, matching_report); the report
    # is unused here.
    formations, all_otgs, subregion_demands, _matching_report = (
        load_otg_data_formation_matching()
    )

    # Aggregate FC across `inner_seeds` runs by calling monte_carlo with
    # n_runs=inner_seeds and reading its median.
    config = FormationConfig(
        T=72,
        n_compliance=30,
        n_intermediary=10,
        n_bct=n_bct,
        n_habitat_bank=30,
        monthly_obligations=12.0 * oblig_mult,
        p_variation=p_var,
        seed=PRE_REGISTERED_SEED + sample_idx * 1000,
    )
    result = monte_carlo(
        formations, all_otgs, subregion_demands, config, n_runs=inner_seeds
    )
    fc_median = float(result["ec2"]["median"])
    fc_p10 = float(result["ec2"]["p10"])
    fc_p90 = float(result["ec2"]["p90"])

    return {
        "sample_idx": sample_idx,
        "params": params.tolist(),
        "fc_median": fc_median,
        "fc_p10": fc_p10,
        "fc_p90": fc_p90,
        "n_inner_seeds": inner_seeds,
    }


# =============================================================================
# Driver
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--N",
        type=int,
        default=512,
        help="Saltelli base sample size; total samples = N*(k+2). Default: 512.",
    )
    parser.add_argument(
        "--inner-seeds",
        type=int,
        default=5,
        help="Inner Monte Carlo seeds per Saltelli row (denoising). Default: 5.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes. Default: 1 (serial). 8 recommended on a workstation.",
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=[3, 6],
        default=3,
        help=(
            "Problem dimensionality. k=3 (default): P_variation, n_BCT, "
            "monthly_obligations_mult — all wired through FormationConfig today. "
            "k=6: adds production_cost_mu, power_law_alpha, liquidity_weight_max — "
            "requires additional run_abm.py wiring (see docs/SOBOL_INTEGRATION_NOTES.md)."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Validation mode: N=8, inner_seeds=1, k=3 — confirms the pipeline runs "
            "end-to-end before committing to a full sweep (~10 minutes)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output" / "results" / "sobol_indices.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        args.N = 8
        args.inner_seeds = 1
        args.k = 3
        print("=== SMOKE-TEST mode: N=8, inner_seeds=1, k=3 ===")

    problem = PROBLEM_K3 if args.k == 3 else PROBLEM_K6

    print(f"Sobol GSA over {problem['num_vars']} parameters: {problem['names']}")
    print(f"  N = {args.N}; total samples = N*(k+2) = {args.N * (problem['num_vars'] + 2)}")
    print(f"  inner_seeds = {args.inner_seeds}")
    print(f"  workers = {args.workers}")
    print(f"  output = {args.out}")
    print()

    # ---- Saltelli sample ---------------------------------------------------
    np.random.seed(PRE_REGISTERED_SEED)
    samples = saltelli_sample.sample(problem, args.N, calc_second_order=False)
    n_samples = samples.shape[0]
    print(f"Generated {n_samples} Saltelli samples.")

    # ---- Evaluate ABM at every sample --------------------------------------
    t0 = time.time()
    results: List[Dict] = []

    if args.workers <= 1:
        for i, p in enumerate(samples):
            res = evaluate_one_sample(i, p, problem["names"], args.inner_seeds)
            results.append(res)
            if (i + 1) % max(1, n_samples // 20) == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-9)
                eta = (n_samples - i - 1) / max(rate, 1e-9)
                print(
                    f"  [{i+1:5d}/{n_samples}] FC={res['fc_median']:.4f} "
                    f"({rate*60:.1f} runs/min, ETA {eta/3600:.1f} h)"
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    evaluate_one_sample, i, p, problem["names"], args.inner_seeds
                ): i
                for i, p in enumerate(samples)
            }
            done = 0
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                done += 1
                if done % max(1, n_samples // 20) == 0:
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-9)
                    eta = (n_samples - done) / max(rate, 1e-9)
                    print(
                        f"  [{done:5d}/{n_samples}] FC={res['fc_median']:.4f} "
                        f"({rate*60:.1f} runs/min, ETA {eta/3600:.1f} h)"
                    )

    # Sort back into Saltelli order
    results.sort(key=lambda r: r["sample_idx"])
    Y = np.array([r["fc_median"] for r in results])
    elapsed = time.time() - t0
    print(f"\nABM evaluations complete in {elapsed/60:.1f} min.")

    # ---- Sobol analysis ----------------------------------------------------
    Si = sobol_analyze.analyze(
        problem,
        Y,
        calc_second_order=False,
        num_resamples=1000,
        conf_level=0.95,
        print_to_console=False,
    )
    print("\nSobol indices (95% bootstrap CI):")
    print(f"  {'parameter':<28} {'S_i':>10} {'CI_S_i':>15} {'S_Ti':>10} {'CI_S_Ti':>15}")
    for i, name in enumerate(problem["names"]):
        print(
            f"  {name:<28} {Si['S1'][i]:>10.4f} "
            f"({Si['S1_conf'][i]:>5.4f}) {Si['ST'][i]:>10.4f} "
            f"({Si['ST_conf'][i]:>5.4f})"
        )

    # ---- Write JSON --------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "script": "scripts/sensitivity_sobol.py",
        "timestamp": datetime.now().isoformat(),
        "qoi": QOI,
        "problem": problem,
        "design": {
            "k": args.k,
            "N": args.N,
            "calc_second_order": False,
            "total_saltelli_samples": int(n_samples),
            "inner_seeds": args.inner_seeds,
            "total_abm_runs": int(n_samples * args.inner_seeds),
            "pre_registered_seed": PRE_REGISTERED_SEED,
        },
        "elapsed_seconds": elapsed,
        "raw": [
            {
                "sample_idx": r["sample_idx"],
                "params": r["params"],
                "fc_median": r["fc_median"],
                "fc_p10": r["fc_p10"],
                "fc_p90": r["fc_p90"],
            }
            for r in results
        ],
        "sobol": {
            "S1": Si["S1"].tolist(),
            "S1_conf": Si["S1_conf"].tolist(),
            "ST": Si["ST"].tolist(),
            "ST_conf": Si["ST_conf"].tolist(),
            "interaction_share_1_minus_sumS1": float(1.0 - np.sum(Si["S1"])),
        },
        "convergence_checks": {
            "all_S1_in_unit_interval": bool(
                np.all(Si["S1"] >= -0.05) and np.all(Si["S1"] <= 1.05)
            ),
            "all_ST_in_unit_interval": bool(
                np.all(Si["ST"] >= -0.05) and np.all(Si["ST"] <= 1.05)
            ),
            "S_T_geq_S_for_all_params": bool(
                np.all(Si["ST"] + 0.05 >= Si["S1"])
            ),
        },
    }
    args.out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
