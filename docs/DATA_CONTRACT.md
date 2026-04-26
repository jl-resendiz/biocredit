# Data Contract

Single source of truth for all pipeline interfaces. Every script that reads or
writes data must conform to the contracts defined here. Auditors should use this
file to verify traceability from raw data to manuscript.

Last updated: 2026-03-20

---

## 1. Raw Data Sources

### 1.1 Transaction Register
- **File**: `data/raw/nsw/nsw_credit_transactions_register.csv`
- **Source**: NSW Government Land Management and Biodiversity Conservation portal
- **Accessed**: March 2026
- **Rows**: 2,244
- **Row types**: Transfer (1,134), Retire (1,097), Cancelled (11)

| Column | Type | Used by | Purpose |
|--------|------|---------|---------|
| Transaction Date | date string | verify_claims, model_diagnostics | Temporal analysis |
| Transaction Type | enum: Transfer/Retire/Cancelled | ALL scripts | Row filtering |
| Offset Trading Group | string (nullable) | ALL scripts | Entity key for ecosystem credits |
| Vegetation Formation | string | run_abm, derive_parameters | Formation-level grouping |
| Sub Region | string | run_abm | Geographic demand engine |
| Price Per Credit (Ex-GST) | string with $ and commas | verify_claims, derive_params | Price analysis |
| Retirement Reason | string (nullable) | verify_claims | BCT classification |
| Number Of Credits | numeric | derive_parameters, run_abm, robustness_sensitivity, holdout_validation, model_diagnostics | Volume per transaction. Used as `q_star = bct_median_credits` in the inverse CARA calibration of `gamma_BCT` and `gamma_intermediary` (`derive_parameters.py:330-356`); also as `avg_vol` in the `gamma_intermediary` spread analysis (`run_abm.py:178-189`). |

**Known data quality issues:**
- Price column requires cleaning: remove `$`, `,`, convert to numeric
- OTG names contain HTML entities (`&amp;`, `&#39;`) that must be unescaped
- OTG names in this file do NOT exactly match supply register OTG names for ~58% of transactions (requires fuzzy/formation-level matching)
- 10 transactions have price = NULL; filtered by price >= 100 AUD threshold

### 1.2 Supply Register
- **File**: `data/raw/nsw/nsw_credit_supply_register.csv`
- **Source**: Same portal as transactions
- **Accessed**: March 2026
- **Rows**: 3,777 (1,681 ecosystem credits)

| Column | Type | Used by | Purpose |
|--------|------|---------|---------|
| Offset Trading Group | string | ALL scripts | Entity key |
| Vegetation Formation | string | run_abm | Formation grouping |
| IBRA Subregion | string | run_abm | Geographic matching |
| IBRA Region | string | Not used directly | Available |
| Ecosystem or Species | enum | ALL scripts | Filter: "Ecosystem" only |
| Threatened Ecological Community (NSW) | string (nullable) | verify_claims, run_abm | TEC classification |
| Number of credits | numeric | run_abm | Supply capacity |

**Known data quality issues:**
- TEC column uses multiple null-like values: empty, "Not a TEC", "No Associated TEC", "No TEC Associated"
- OTG names are the authoritative source (252 unique ecosystem OTGs)

### 1.3 External Data (not in CSV)

| Statistic | Value | Source | Used in |
|-----------|-------|--------|---------|
| HHI 2024-25 | 2,966 | IPART Discussion Paper 2024-25, p.6 | main.tex |
| HHI 2023-24 | 2,187 | IPART Annual Report 2023-24, p.8 | main.tex |
| BCT credit share | 7% (2022-23), 16% (2024-25) | IPART reports | main.tex |
| BCT like-for-like rate | 90% (2022-23), 65% (2024-25) | IPART reports | main.tex |
| BCT annual budget | AUD 47.3M | IPART 2024-25, p.7 (45% x $105.1M) | main.tex, run_abm.py |
| NSW infrastructure pipeline | AUD 113B | NSW Treasury Budget Paper 2022-23 | main.tex |
| Global offset market | USD 11.7B/yr | UNEP 2023 | main.tex |
| National schemes | 19 | IAPB 2024 | main.tex |

**These values cannot be reproduced from the CSV files.** They are sourced from
cited government reports and documented in `data/CODEBOOK.md` with page numbers.

---

## 2. Pipeline Stages

### Stage 1: Parameter Derivation
```
Input:  data/raw/nsw/*.csv
Script: scripts/derive_parameters.py
Output: stdout (printed report, no file output)
```
Derives: fair_value_multiplier, sigma_intermediary, gamma_intermediary,
logistic_slope, supply_elasticity (eta), liquidity feedback weights.

**Note**: run_abm.py has its own `derive_all_parameters()` function that
reads the same CSVs and derives the same parameters. The two implementations
are independent but use identical methodology. This is intentional redundancy,
not a bug.

### Stage 2: ABM Simulation
```
Input:  data/raw/nsw/*.csv
Script: scripts/run_abm.py
Output: output/results/abm_results.json
Seeds:  50 MC seeds (0, 1, 2, ..., 49) — deterministic
Time:   ~20 minutes for 7 scenarios
```

**Output schema** (`abm_results.json`):
```json
{
  "model": "formation_abm",
  "timestamp": "ISO 8601",
  "parameters": {
    "bct_budget": 326389,
    "production_cost_mu": 7.62,
    "p_variation": 0.5,
    "mc_seeds": 50,
    "timesteps": 72
  },
  "baseline": {
    "ec2_median": float,     // FC as percentage (e.g., 16.3)
    "ec2_p10": float,
    "ec2_p90": float,
    "fund_rate": float,      // as percentage
    "compliance_share": float,
    "bct_share": float,
    "spearman_rho": float,
    "spearman_p": float
  },
  "scenarios": {
    "<name>": {
      "ec2_median": float,
      "ec2_p10": float,
      "ec2_p90": float,
      "fund_rate": float,
      "total_cost_median": float,
      "avg_price_median": float,
      "compliance_cost_median": float,
      "compliance_avg_price_median": float
    }
  },
  "cost_analysis": {
    "<name>": {
      "total_cost_median_aud": float,
      "compliance_cost_median_aud": float,
      "avg_price_median_aud": float,
      "compliance_avg_price_median_aud": float,
      "cost_change_vs_baseline_pct": float
    }
  },
  "formation_shares": { "<formation>": {"observed_pct": float, "predicted_pct": float} }
}
```

### Stage 3: GLV Model
```
Input:  data/raw/nsw/*.csv
Script: python -m src.model.glv_formation
Output: output/results/glv_results.json
```

### Stage 4: Robustness
```
Input:  data/raw/nsw/*.csv + output/results/abm_results.json
Scripts: scripts/robustness_sensitivity.py, robustness_demand_functions.py, robustness_empirical.py
Output: output/results/sensitivity_results.json, demand_robustness_results.json
```

### Stage 5: Figures
```
Input:  data/raw/nsw/*.csv (fig1), output/results/*.json (fig2, fig3, suppfig1)
Scripts: src/figures/fig1_problem.py, fig2_mechanism.py, fig3_policy.py, suppfig1_robustness.py
Output: output/figures/main/fig1_problem.pdf, fig2_mechanism.pdf, fig3_policy.pdf
        output/figures/supplementary/suppfig1_robustness.pdf
```

**Fallback behavior**: fig2, fig3, and suppfig1 have hardcoded fallback values
that activate if JSON files are missing. They print `WARNING:` to stderr.
This should never happen if `make all` is used because robustness now runs
before figures.

### Stage 6: Verification
```
Input:  data/raw/nsw/*.csv + output/results/abm_results.json + manuscript claims (hardcoded dict)
Script: scripts/verify_claims.py
Output: stdout (comparison table)
```

**MANUSCRIPT dict**: Lines 51-76 of verify_claims.py contain a manually
maintained dict of expected manuscript values. This dict MUST be updated
whenever manuscript numbers change. It is NOT parsed from main.tex.

### Stage 7: Diagnostics
```
Input:  data/raw/nsw/*.csv + output/results/abm_results.json
Script: scripts/model_diagnostics.py
Output: output/results/diagnostics.json + stdout
```

---

## 3. Entity Resolution: OTG Name Matching

The single most complex data quality issue in this pipeline.

**Problem**: OTG names in the transaction register do NOT exactly match OTG
names in the supply register for approximately 58% of ecosystem transactions.

**Cause**: The transaction register and supply register are maintained
independently by the NSW government. Names differ due to HTML encoding,
whitespace, abbreviation, and version changes.

**Resolution approaches used**:

| Context | Method | Script | Result |
|---------|--------|--------|--------|
| Headline counts (140 traded, 45 functional, 112 never) | Direct name matching (no remap) | verify_claims.py Section 2 | Conservative count |
| TEC classification (61% traded, 18 functional) | Supply-register matching | verify_claims.py Section 3 | Fuzzy-remap matching used in manuscript |
| Formation-level model | Formation-level matching (group by Vegetation Formation) | run_abm.py | 100% match rate at formation level |
| Buyer classification (BCT-only, compliance) | Fuzzy remap (4 manual remaps) | verify_claims.py Section 3-4 | Allows buyer-type analysis |

**Important**: The manuscript uses fuzzy-remap matching numbers (61% of TECs
traded, 18 functional). These are derived from supply-register matching in
verify_claims.py.

**Headline count methodology (140 traded, 112 never traded)**:
The "140" is the count of unique OTG names appearing in ecosystem priced
transfers. The "252" is the count of unique OTG names in the supply register.
252 - 140 = 112. Of the 140 transaction OTG names, only 33 match supply
register names exactly (raw). The rest (107) are the same OTGs under different
encoding. This means some transaction names may map to the same supply OTG.
If so, the true "traded" count is lower and "never traded" is higher, making
the coverage gap LARGER. The current 140/112 split is therefore the
conservative (optimistic) estimate of market reach.

---

## 4. Manuscript Number Registry

Every quantitative claim in the manuscript and its pipeline source:

### From raw CSV (verified by verify_claims.py)
| Claim | Value | verify_claims key |
|-------|-------|------------------|
| Priced transactions | 1,124 | Priced transactions |
| Ecosystem credits | 764 | Eco priced |
| Total OTGs | 252 | Total OTGs (ecosystem) |
| Never traded | 112 (44%) | Never traded OTGs / pct |
| Functional | 45 (18%) | Functional OTGs / pct |
| TEC OTGs | 79 | TEC OTGs |
| Non-TEC OTGs | 173 | Non-TEC OTGs |
| TEC excluded | 47% | TEC excluded pct |
| TEC functional | 18 | TEC functional |
| TEC traded | 61% | TEC traded pct |
| BCT-only OTGs | 24 | BCT-only OTGs |
| BCT premium | 5.3x | BCT premium ratio |
| Compliance retirements | 65% | Compliance retirement pct |
| BCT retirements | 35% | BCT retirement pct |
| Voluntary | 1 | Voluntary retirements |
| Total retirements | 1,097 | Total retirements |
| Price CV | 1.56 | CV (price) |
| Formations | 15 | Vegetation formations |
| IBRA subregions | 40 | IBRA subregions |

### From ABM model (verified by verify_claims.py model section)
| Claim | Value | JSON path |
|-------|-------|-----------|
| Baseline FC | 16.3% | scenarios.Baseline.ec2_median |
| Bypass Reduction FC | 17.5% | scenarios."Bypass Reduction".ec2_median |
| Procurement Flex FC | 17.1% | scenarios."Procurement Flex (20%)".ec2_median |
| Rarity Multiplier FC | 16.5% | scenarios."Rarity Multiplier (2x)".ec2_median |
| Price Floor FC | 16.3% | scenarios."Price Floor (AUD 3,000)".ec2_median |
| BCT Precision FC | 12.7% | scenarios."BCT Precision Mandate".ec2_median |
| Spearman rho | 0.98 (0.9839) | baseline.spearman_rho |
| Compliance share | 65.8% | baseline.compliance_share |
| BCT share | 34.2% | baseline.bct_share |
| Fund rate | 34.1% | baseline.fund_rate |

### From external sources (not verifiable from CSV)
| Claim | Source | Documented in |
|-------|--------|--------------|
| HHI values | IPART reports | CODEBOOK.md external sources |
| BCT share trend | IPART reports | CODEBOOK.md |
| L4L precision decline | IPART reports | CODEBOOK.md |
| $11.7B market | UNEP 2023 | Citation |
| 19 schemes | IAPB 2024 | Citation |

---

## 5. How to Audit This Pipeline

For future auditors:

1. **Verify raw data integrity**: Check CSV checksums against documented values
2. **Run `python scripts/verify_claims.py`**: Should report ALL MATCH
3. **Run `python scripts/model_diagnostics.py`**: Should report ALL PASS
4. **Check Table 1**: Compare each cell against `output/results/abm_results.json`
5. **Check Supp Table 8**: Compare against JSON cost_analysis section
6. **Check external claims**: Verify IPART numbers against CODEBOOK.md page references
7. **Check entity resolution**: The manuscript uses fuzzy-remap matching (61% TEC traded, 18 functional). Direct name matching gives lower numbers (42%/16) but is not used in the manuscript.
