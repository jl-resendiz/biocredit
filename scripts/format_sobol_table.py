#!/usr/bin/env python3
"""
Format the Sobol GSA results from `output/results/sobol_indices.json`
into a LaTeX table block, ready to paste into the placeholder slot in
`manuscript/supplementary.tex` (Supp. Note 2).

Usage:
    python scripts/format_sobol_table.py            # prints LaTeX to stdout
    python scripts/format_sobol_table.py --check    # reports convergence/sanity
    python scripts/format_sobol_table.py --insert   # write into supp.tex in-place

The --insert mode replaces the placeholder block in supp.tex flagged by
the markers
    % BEGIN_SOBOL_TABLE_PLACEHOLDER
    ...
    % END_SOBOL_TABLE_PLACEHOLDER
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


REPO = Path(__file__).resolve().parent.parent
JSON_PATH = REPO / "output" / "results" / "sobol_indices.json"
SUPP_PATH = REPO / "manuscript" / "supplementary.tex"

# Pretty-print map: program-name -> LaTeX-friendly display
DISPLAY_NAME = {
    "P_variation": r"$P_{\text{variation}}$",
    "n_BCT": r"$n_{\text{BCT}}$",
    "monthly_obligations_mult": r"monthly obligations $\times$",
    "production_cost_mu": r"$\mu_{\log c}$",
    "power_law_alpha": r"power-law $\alpha$",
    "liquidity_weight_max": r"liquidity-weight max",
}


def fmt_table(payload: Dict) -> str:
    """Build a LaTeX table from the Sobol JSON payload."""
    problem = payload["problem"]
    sobol = payload["sobol"]
    design = payload["design"]
    names = problem["names"]
    s1 = sobol["S1"]
    s1c = sobol["S1_conf"]
    st = sobol["ST"]
    stc = sobol["ST_conf"]
    interaction_share = sobol["interaction_share_1_minus_sumS1"]

    rows = []
    for i, n in enumerate(names):
        display = DISPLAY_NAME.get(n, n.replace("_", r"\_"))
        gap = st[i] - s1[i]
        rows.append(
            f"  {display:<30} & "
            f"{s1[i]:6.3f} & ($\\pm${s1c[i]:.3f}) & "
            f"{st[i]:6.3f} & ($\\pm${stc[i]:.3f}) & "
            f"{gap:6.3f} \\\\"
        )

    table = (
        "\\begin{table}[h]\n"
        "\\caption{%\n"
        "  \\textbf{Supplementary Table 9 | Global sensitivity analysis (Sobol indices) for functional coverage.}\n"
        f"  Saltelli design with $N={design['N']}$ base samples and $k={design['k']}$ parameters,\n"
        f"  for a total of {design['total_saltelli_samples']:,} sample points; each point\n"
        f"  averaged over {design['inner_seeds']} inner Monte Carlo seeds. Total ABM runs:\n"
        f"  {design['total_abm_runs']:,}. First-order Sobol index $S_i$ measures the share of\n"
        "  output variance explained by parameter $i$ alone; total Sobol index $S_{T_i}$\n"
        "  includes all interactions involving $i$. The interaction term $S_{T_i} - S_i$\n"
        "  quantifies how much of $i$'s influence on FC variance is mediated through\n"
        "  joint effects with other parameters. Bootstrap 95\\% confidence intervals (B=1{,}000)\n"
        "  in parentheses. Total interaction share $1 - \\sum_i S_i = "
        f"{interaction_share:.3f}$.\n"
        "}\\label{tab:sobol_indices}\n"
        "\\begin{tabular}{@{}lrlrlr@{}}\n"
        "\\toprule\n"
        "Parameter & $S_i$ & 95\\% CI & $S_{T_i}$ & 95\\% CI & $S_{T_i} - S_i$ \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) +
        "\n\\botrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return table


def fmt_check(payload: Dict) -> str:
    """Print convergence and sanity checks."""
    sobol = payload["sobol"]
    cc = payload["convergence_checks"]
    s1 = sobol["S1"]
    st = sobol["ST"]
    s1c = sobol["S1_conf"]
    stc = sobol["ST_conf"]
    names = payload["problem"]["names"]
    out = []
    out.append("Sobol convergence and sanity checks:")
    for k, v in cc.items():
        flag = "[OK]" if v else "[FAIL]"
        out.append(f"  {flag} {k}: {v}")
    out.append("")
    out.append("Per-parameter:")
    for i, n in enumerate(names):
        out.append(
            f"  {n:<30} S1={s1[i]:+.3f} (CI {s1c[i]:.3f}) "
            f"ST={st[i]:+.3f} (CI {stc[i]:.3f})"
        )
    out.append("")
    if max(s1c) > 0.1 or max(stc) > 0.1:
        out.append(
            "WARNING: at least one CI > 0.1; "
            "consider larger N for tighter convergence."
        )
    return "\n".join(out)


def insert_into_supp(table: str) -> bool:
    """Replace the placeholder block in supplementary.tex.

    Returns True if a substitution was made, False if the placeholder
    markers were not found.
    """
    text = SUPP_PATH.read_text(encoding="utf-8")
    begin = "% BEGIN_SOBOL_TABLE_PLACEHOLDER"
    end = "% END_SOBOL_TABLE_PLACEHOLDER"
    i = text.find(begin)
    j = text.find(end)
    if i < 0 or j < 0 or j < i:
        return False
    new = text[: i] + begin + "\n" + table + end + text[j + len(end):]
    SUPP_PATH.write_text(new, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=JSON_PATH,
        help="Path to sobol_indices.json (default: output/results/sobol_indices.json).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print convergence/sanity report and exit (no LaTeX output).",
    )
    parser.add_argument(
        "--insert",
        action="store_true",
        help="Write the LaTeX table into supplementary.tex in-place.",
    )
    args = parser.parse_args()

    if not args.json.exists():
        print(f"ERROR: {args.json} does not exist. Run `make sobol` first.", file=sys.stderr)
        return 2

    payload = json.loads(args.json.read_text(encoding="utf-8"))

    if args.check:
        print(fmt_check(payload))
        return 0

    table = fmt_table(payload)

    if args.insert:
        ok = insert_into_supp(table)
        if not ok:
            print(
                "ERROR: placeholder markers not found in supplementary.tex.\n"
                "       Expected:\n"
                "         % BEGIN_SOBOL_TABLE_PLACEHOLDER\n"
                "         % END_SOBOL_TABLE_PLACEHOLDER\n",
                file=sys.stderr,
            )
            return 3
        print(f"Inserted Sobol table into {SUPP_PATH}.")
        return 0

    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
