# Why Biodiversity Offset Markets Fail to Reach Most Ecosystem Types

**Reséndiz, J.L.** — Smith School of Enterprise and the Environment, University of Oxford

> Replication package for: *Why biodiversity offset markets fail to reach most ecosystem types* (manuscript under review, *Nature Ecology & Evolution*)

---

## What this package reproduces

All three main figures, one supplementary figure, and supporting analyses from raw data to publication-ready PDFs:

| Output | Script | What it shows |
|--------|--------|---------------|
| Fig. 1 | `src/figures/fig1_problem.py` | The coverage gap: OTG transaction histogram + BCT precision erosion |
| Fig. 2 | `src/figures/fig2_mechanism.py` | The model: observed vs predicted by formation + Fund routing validation + policy scenarios |
| Fig. 3 | `src/figures/fig3_policy.py` | Robustness: parameter sensitivity + demand function specification test |
| Supp. Fig. 1 | `src/figures/suppfig1_robustness.py` | GLV interaction matrix, sensitivity sweep, abstract GLV, two-model comparison |

---

## Requirements

Python 3.10 or later.

```bash
pip install numpy>=1.24 scipy>=1.10 matplotlib>=3.7 pandas>=2.0
```

---

## Reproduce everything

```bash
make all          # generates all figures + runs verification
make fig1         # single figure
make fig2
make fig3
make suppfig1
make verify       # verify all manuscript claims against raw data
make abm          # run primary ABM model
make glv          # run GLV analytical model
make clean        # remove generated outputs
```

Figures are written to `output/figures/main/` as PDFs.

---

## Data

Two raw CSV files from the NSW Government, documented in `data/CODEBOOK.md`:

| Dataset | Rows | File |
|---------|------|------|
| NSW BOS credit transactions register | 2,244 | `data/raw/nsw/nsw_credit_transactions_register.csv` |
| NSW BOS credit supply register | 3,777 | `data/raw/nsw/nsw_credit_supply_register.csv` |

**1,124 priced transactions** from the NSW Biodiversity Offsets Scheme (November 2019 – March 2026): 764 ecosystem credit transactions (68%) and 360 species credit transactions (32%).

The **supply register** provides the universe of 252 ecosystem OTGs with registered credit supply.

All external statistics (HHI, BCT like-for-like rate, market value) are from IPART Annual Reports 2023-24 and 2024-25, documented in `data/CODEBOOK.md`.

---

## Central empirical findings

Verified by `scripts/verify_claims.py` (22/22 claims match raw data):

| Metric | Value | Source |
|--------|-------|--------|
| Ecosystem OTGs with registered supply | 252 | Supply register |
| OTGs never traded (EC1 gap) | 112 (44%) | Transaction register |
| OTGs sustaining functional markets (EC2) | 45 (18%) | Transaction register (threshold: ≥5 transactions) |
| TEC OTGs | 79 | Supply register (official TEC column) |
| Non-TEC OTGs | 173 | 252 - 79 |
| TEC OTGs ever traded (strict matching) | 33 of 79 (42%) | Transaction × Supply register |
| Non-TEC OTGs ever traded (strict matching) | 0 of 173 (0%) | Fisher's exact p < 10⁻¹⁹ |
| BCT-only OTGs | 24 | Retirement records |
| BCT median price, thin markets (AUD) | 5,989 | Transaction register |
| Compliance median price, thin markets (AUD) | 1,125 | Transaction register |
| BCT / compliance price ratio | 5.3× | Mann-Whitney U = 1,414, p < 0.0001 |
| Compliance retirements | 65% (709 of 1,097) | Retirement Reason field |
| BCT retirements | 35% (388 of 1,097) | Retirement Reason field |
| Voluntary retirements | 1 | Retirement Reason field |

---

## Two models (all parameters from data)

Every parameter in every model is derived from raw data or published sources. See `docs/PARAMETER_REGISTRY.md` for the complete inventory.

| Model | File | EC2 (baseline) | Role |
|-------|------|----------------|------|
| Formation ABM | `scripts/run_abm.py` | 16.3% | **Primary model** — like-for-like with variation rules, liquidity feedback, 50 MC seeds |
| GLV mean-field | `src/model/glv_formation.py` | 7.0% | Analytical companion — exclusion thresholds, sensitivity analysis |

Both produce the same scenario ranking: **Procurement Flexibility ≥ Combined > Baseline ≥ Price Floor.**

Run the primary model:

```bash
python scripts/run_abm.py
```

---

## Policy scenarios

| Scenario | EC2 (ABM) | Fund rate | Interpretation |
|----------|-----------|-----------|----------------|
| Procurement Flexibility (20%) | 17.1% | 30.2% | Most effective single intervention |
| Combined (Flex + Floor) | 17.1% | 46.3% | No marginal benefit over flex alone |
| Baseline | 16.3% | 34.1% | Current market structure |
| Price Floor (AUD 3,000) | 16.3% | 50.5% | Counterproductive — increases Fund routing |

---

## Verification and robustness

| Script | What it checks |
|--------|----------------|
| `scripts/verify_claims.py` | Every empirical number in the manuscript vs raw CSV |
| `scripts/robustness_empirical.py` | EC2 threshold sensitivity, temporal split, OTG matching, BCT premium |
| `scripts/robustness_sensitivity.py` | Parameter perturbation: all produce <1pp EC2 variation (except n_bct: 5.4pp) |
| `scripts/robustness_demand_functions.py` | Ad-hoc vs micro-founded demand: ρ ≥ 0.97, 5/5 claims preserved |
| `scripts/community_matrix.py` | ABM community matrix vs GLV α: Spearman ρ = 0.689 |
| `scripts/derive_parameters.py` | Derives all model parameters from raw CSV |

---

## Documentation

| File | Contents |
|------|----------|
| `data/CODEBOOK.md` | **Single source of truth** — every CSV field, classification rule, derived quantity |
| `docs/PARAMETER_REGISTRY.md` | Every model parameter with symbol, value, source, and plain English meaning |
| `docs/DATA_SOURCES.md` | Full provenance for all data |
| `docs/MODEL_ASSUMPTIONS.md` | Explicit model assumptions with testability criteria |
| `docs/DECISIONS.md` | Analytical choices log |
| `docs/RELATIONSHIP_TO_ORIGINAL.md` | Maps to Scholl, Calinescu & Farmer (2021) framework |

---

## Repository structure

```
data/
├── CODEBOOK.md                              # Single source of truth for all data
└── raw/nsw/                                 # NSW transaction + supply registers
scripts/
├── run_abm.py                      # PRIMARY MODEL — definitive ABM
├── verify_claims.py              # Verifies all manuscript numbers
├── derive_parameters.py                     # Derives model params from CSV
├── robustness_empirical.py                   # 7 robustness tests
├── robustness_sensitivity.py                    # Parameter sensitivity analysis
├── robustness_demand_functions.py            # Ad-hoc vs micro-founded comparison
├── community_matrix.py                      # ABM community matrix computation
├── ec_analysis.py                           # Standalone EC1/EC2 analysis
└── tec_market_analysis.py                   # TEC vs market activity analysis
src/
├── model/
│   ├── formation_model.py                   # Formation-level ABM (used by fig1)
│   ├── glv_formation.py                     # GLV analytical model
│   ├── dynamics.py                          # Abstract GLV (suppfig1 panel c)
│   ├── species.py                           # Species definitions
│   └── interactions.py                      # Alpha matrix
├── calibration/
│   └── nsw_data.py                          # IPART statistics (Tier 2 data)
├── earlywarning/
│   └── signals.py                           # 8-indicator early warning system
└── figures/
    ├── fig1_problem.py                      # Fig 1: coverage gap
    ├── fig2_mechanism.py                    # Fig 2: model validation + policy
    ├── fig3_policy.py                       # Fig 3: robustness
    ├── suppfig1_robustness.py               # Supp Fig 1: technical detail
    └── style.py                             # Nature journal styling
manuscript/
├── main.tex                                 # Paper
├── supplementary.tex                        # Supplementary Information
└── references.bib                           # Bibliography
docs/
├── PARAMETER_REGISTRY.md                    # Every parameter documented
├── DATA_SOURCES.md                          # Data provenance
├── MODEL_ASSUMPTIONS.md                     # 8 assumptions
├── DECISIONS.md                             # Choices log
└── RELATIONSHIP_TO_ORIGINAL.md              # Scholl et al. mapping
output/
├── figures/main/                            # Generated PDFs (make all)
└── tables/                                  # Community matrix CSVs
```

---

## Built on: Scholl, Calinescu & Farmer (2021)

This package extends the **market ecology framework** of:

> Farmer, J.D. (2002). Market force, ecology and evolution. *Industrial and Corporate Change*, 11(5), 895-953.

> Scholl, M.P., Calinescu, A. & Farmer, J.D. (2021). How market ecology explains market malfunction. *PNAS*, 118(26), e2015574118.

The GLV is used as an analytically tractable complement to the ABM, not as a rigorous theoretical reduction. The community matrix (Spearman ρ = 0.689 between ABM-derived and GLV coefficients) validates that the GLV captures the competitive structure. See Supplementary Note 1 for the full theoretical discussion.

---

## License

MIT License.

Data from the NSW Government are publicly available under open government licence and are redistributed here for reproducibility purposes only.
