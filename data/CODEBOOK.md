# Data Codebook

Single source of truth for all raw data used in this paper.
Referenced by all scripts and models in the pipeline.

## Files

| File | Rows | Source | Access date |
|---|---|---|---|
| `data/raw/nsw/nsw_credit_transactions_register.csv` | 2,244 | NSW Biodiversity Credits & Conservation Management portal | March 2026 |
| `data/raw/nsw/nsw_credit_supply_register.csv` | 3,777 | NSW Biodiversity Credits & Conservation Management portal | March 2026 |

---

## File 1: nsw_credit_transactions_register.csv

Every row is a credit transaction (transfer, retirement, or cancellation).

### Key fields

| Column | Type | Non-empty | Unique | Description | Used in pipeline |
|---|---|---|---|---|---|
| Transaction Date | date (YYYY-MM-DD) | 2,241 | 476 | Date of transaction | Date range, temporal split |
| Transaction ID | string | 2,243 | 1,207 | Unique transaction identifier | Deduplication |
| Transaction Type | categorical | 2,242 | 3 | `Transfer` (1,134), `Retire` (1,097), `Cancelled` (11) | Filter: Transfer = priced transactions; Retire = retirement analysis |
| Transaction Status | categorical | 2,244 | 2 | `Completed` (overwhelming majority), other (rare). Documents whether the transaction was finalised by the regulator. | Filter (`scripts/robustness_trend.py:45` only): restrict to `Completed`. Other live scripts do not filter on this field; all scripts produce the same headline numbers because the non-Completed rows are negligible. |
| From | string (CR-XXXXX) | 2,244 | 1,614 | Seller credit ID (anonymised) | Seller count |
| To | string (CR-XXXXX) | 2,243 | 2,227 | Buyer credit ID (anonymised) | Buyer count |
| Offset Trading Group | string | 1,578 | 156 | Ecosystem OTG name. Empty for species transactions. | Functional coverage, formation matching |
| Vegetation Formation | categorical | 1,578 | 15 | Keith vegetation formation. Empty for species transactions. | **Formation-level demand weights** |
| Scientific Name | string | 666 | 77 | Species scientific name. Empty for ecosystem transactions. | Filter: empty = ecosystem |
| Common Name | string | 666 | 78 | Species common name | Not used directly |
| Sub Region | string | 2,244 | 47 | IBRA subregion of the transaction | Geographic demand engine |
| Number Of Credits | integer | 2,244 | 547 | Credits transacted | Volume analysis |
| Price Per Credit (Ex-GST) | numeric | 2,244 | 403 | Price in AUD. Values include "0" for retirements. | Price analysis (filter: >= 100 AUD) |
| Retirement Reason | categorical | 1,097 | 7 | Why credits were retired (see values below) | BCT/compliance classification |
| Other Reason for Retiring | string | 113 | 24 | Free-text detail for "Other" retirements | BCT acquittal identification |

### Transaction Type values
- `Transfer` (1,134): Credit sold from one party to another. Has price.
- `Retire` (1,097): Credit cancelled to fulfil an obligation. Price = 0.
- `Cancelled` (11): Administrative cancellation.

### Retirement Reason values
| Value | Count | Classification |
|---|---|---|
| "For the purpose of complying with a requirement to retire biodiversity credits of a planning approval or a vegetation clearing approval" | 660 | Compliance |
| "For the purpose of BCT complying with an obligation to retire Biodiversity Credits" | 388 | BCT |
| "For the purposes of complying with a requirement to retire credits of a biodiversity certification of land" | 25 | Compliance |
| "Other (please specify)" | 17 | See Other Reason field |
| "For the purposes of a planning agreement" | 4 | Compliance |
| "For the purpose of complying with an order requiring a biodiversity stewardship site owner to retire biodiversity credits" | 2 | Compliance |
| "A voluntary purpose" | 1 | Voluntary |

### Classification rules used in pipeline
- **Compliance retirements** = all non-BCT, non-voluntary = 660 + 25 + 17 + 4 + 2 = 709 (65%)
- **BCT retirements** = 388 (35%)
- **Voluntary** = 1
- **Total** = 1,097
- **Priced ecosystem transactions** = Transfer rows where Price >= 100 AUD AND Scientific Name is empty = 764 (from direct OTG matching) or 770 (including Vegetation Formation column matching)
- **Priced species transactions** = Transfer rows where Price >= 100 AUD AND Scientific Name is not empty = 360

### BCT identification (canonical rule)

The canonical rule applied across the entire pipeline (`scripts/verify_claims.py`,
`scripts/derive_parameters.py`, `scripts/run_abm.py`, `scripts/robustness_*.py`,
`scripts/holdout_validation.py`, `scripts/community_matrix.py`) is:

> A retirement is classified as **BCT** if `Retirement Reason` contains the
> substring "BCT" (case-insensitive) **OR** `Other Reason for Retiring`
> contains the substring "BCT" or the phrase "Biodiversity Conservation
> Trust" (case-insensitive).

Rationale: the official BCT retirement-reason string ("For the purpose of
BCT complying with an obligation to retire Biodiversity Credits", 388 rows)
catches the unambiguous cases. The 17 "Other (please specify)" rows
include a small number of BCT-acquittal entries written as free text in
the `Other Reason for Retiring` field; the substring fallback recovers
these so they are not silently misclassified as Compliance. The combined
rule yields the same 388 / 35% headline as the strict-Reason rule (the
Other-Reason rows that match are a handful) but ensures BCT-vs-compliance
classification is not sensitive to reporting-format choices that vary
across years.

A buyer-side classification is derived from the `From` column of BCT
retirements: any credit ID that retires under a BCT-classified row is
flagged as a BCT-acquired credit, and `Transfer` rows whose `To` matches
any such BCT credit ID are classified as BCT purchases. This buyer-side
rule defines BCT-only OTGs (n=24) and the BCT thin-market premium (5.3×).

### Ecosystem vs species identification
- **Ecosystem transaction**: `Scientific Name` field is empty; `Offset Trading Group` and `Vegetation Formation` fields are populated
- **Species transaction**: `Scientific Name` field is populated; `Offset Trading Group` and `Vegetation Formation` fields are empty

---

## File 2: nsw_credit_supply_register.csv

Every row is a credit line (a block of credits registered for potential sale).

### Key fields

| Column | Type | Non-empty | Unique | Description | Used in pipeline |
|---|---|---|---|---|---|
| Credit ID | string | 3,775 | 2,337 | Unique credit identifier | Credit tracing |
| Ecosystem or Species | categorical | 2,336 | 2 | `Ecosystem` (1,681), `Species` (655) | Filter |
| Offset Trading Group | string | 1,678 | 252 | Ecosystem OTG name | OTG universe (252 types) |
| Vegetation Formation | categorical | 1,681 | 16 | Keith vegetation formation | Formation grouping |
| Threatened Ecological Community (NSW) | string | 1,678 | 69 | Official TEC classification | **TEC/non-TEC classification** |
| IBRA Subregion | string | 2,704 | 162 | Geographic subregion | Geographic demand engine |
| IBRA Region | string | 2,704 | 121 | Broader region | Not used directly |
| Number of credits | integer | 2,677 | 910 | Credits available for sale | Supply capacity |
| Species Scientific Name | string | 655 | 138 | For species credits | Not used for ecosystem analysis |
| Vegetation Class | string | 1,678 | 76 | Finer vegetation classification | Not used directly |
| Hollow Bearing Trees | categorical | 1,211 | 3 | Yes/No/Yes (including artificial) | Not used |

### TEC classification rule
An OTG is classified as TEC if ANY of its supply register rows has a `Threatened Ecological Community (NSW)` value that is NOT one of:
- "Not a TEC"
- "No TEC Associated"
- "No Associated TEC"
- (empty)

Result: **79 TEC OTGs, 173 non-TEC OTGs** (total 252)

### 15 Vegetation Formations
| Formation | OTGs | TEC OTGs |
|---|---|---|
| Grassy Woodlands | 50 | 27 |
| Dry Sclerophyll Forests (Shrubby sub-formation) | 35 | 5 |
| Dry Sclerophyll Forests (Shrub/grass sub-formation) | 30 | 5 |
| Forested Wetlands | 19 | 10 |
| Semi-arid Woodlands (Shrubby sub-formation) | 20 | 4 |
| Freshwater Wetlands | 17 | 6 |
| Rainforests | 18 | 10 |
| Wet Sclerophyll Forests (Grassy sub-formation) | 13 | 1 |
| Grasslands | 12 | 5 |
| Semi-arid Woodlands (Grassy sub-formation) | 11 | 4 |
| Heathlands | 9 | 0 |
| Wet Sclerophyll Forests (Shrubby sub-formation) | 7 | 1 |
| Arid Shrublands (Chenopod sub-formation) | 5 | 0 |
| Saline Wetlands | 4 | 1 |
| Arid Shrublands (Acacia sub-formation) | 2 | 0 |

---

## Key derived quantities

These are computed by `scripts/verify_claims.py` from the raw data above:

| Quantity | Value | Derivation |
|---|---|---|
| Total priced transactions | 1,124 | Transfer rows, Price >= 100 AUD |
| Ecosystem priced | 764 | Priced + Scientific Name empty |
| Species priced | 360 | Priced + Scientific Name not empty |
| Total OTGs | 252 | Unique Offset Trading Group in supply register (Ecosystem only) |
| Basic reach (ever traded) | 56% (140/252) | OTGs with >= 1 priced ecosystem transaction |
| Functional coverage (FC) | 17.9% (45/252) | OTGs with >= 5 priced ecosystem transactions |
| TEC OTGs | 79 | Official TEC classification from supply register |
| Non-TEC OTGs | 173 | 252 - 79 |
| FC-TEC | 22.8% (18/79) | TEC OTGs with >= 5 transactions (supply-register matching) |
| FC-nonTEC | 0% (0/173) | Under direct name matching, zero non-TEC OTGs have transactions |
| BCT-only OTGs | 24 | OTGs with BCT retirements but zero compliance retirements |
| BCT thin-market median | AUD 5,989 | Median BCT purchase price in OTGs with < 5 transactions |
| Compliance thin-market median | AUD 1,125 | Median compliance purchase price in OTGs with < 5 transactions |
| BCT premium | 5.3x | 5,989 / 1,125 |
| Total retirements | 1,097 | Retire transaction type |
| Compliance retirements | 65% (709) | All non-BCT, non-voluntary |
| BCT retirements | 35% (388) | BCT retirement reason |
| Voluntary | 1 | "A voluntary purpose" |

---

## External data sources (not in CSV)

| Statistic | Value | Source | Page |
|---|---|---|---|
| HHI 2023-24 | 2,187 | IPART Annual Report 2023-24 | p.8 |
| HHI 2024-25 | 2,966 | IPART Discussion Paper 2024-25 | p.6 |
| BCT like-for-like rate | 65% | IPART Discussion Paper 2024-25 | p.10 |
| BCT credit share | 16% | IPART Discussion Paper 2024-25 | p.7 |
| BCT value share | 45% | IPART Discussion Paper 2024-25 | p.7 |
| Bypass rate | 18% | IPART Discussion Paper 2024-25 | p.12 |
| NSW infrastructure pipeline | AUD 113 billion | NSW Treasury Budget Paper No. 3, 2022-23 | p.1 |
| Major project obligations | 90% of total | NSW Audit Office Performance Audit, Aug 2022 | p.4 |
| Global offset market value | USD 11.7 billion/year | UNEP State of Finance for Nature, 2023 | Fig. 1 |
| Government-led credit schemes | 19 | IAPB Framework Report, Oct 2024 | p.6 |

---

## Pipeline reference

Scripts that read this data:

| Script | Reads | Purpose |
|---|---|---|
| `scripts/verify_claims.py` | Both CSVs | Computes ALL manuscript statistics |
| `scripts/robustness_empirical.py` | Both CSVs | 7 robustness tests |
| `scripts/derive_parameters.py` | Both CSVs | Derives model parameters from data |
| `scripts/community_matrix.py` | Via model imports | Community matrix validation |
| `src/model/formation_model.py` | Both CSVs | Primary ABM |
| `src/model/glv_formation.py` | Both CSVs | GLV analytical model |
| `src/figures/fig1_problem.py` | Both CSVs | Figure 1 generation |
| `scripts/run_abm.py` | Both CSVs → `output/results/abm_results.json` | ABM scenarios + cost metrics |

---

## Model-generated cost metrics (abm_results.json)

The `cost_analysis` section of `output/results/abm_results.json` contains scenario-level
cost estimates computed from simulated transactions. These are **model outputs, not
observed data** — the model was calibrated to the Fund routing rate (35.4%), not to
price levels.

| Field | Definition | Unit |
|---|---|---|
| `total_cost_median_aud` | Median total market expenditure (Σ price × qty) across 50 MC seeds | AUD |
| `compliance_cost_median_aud` | Median expenditure by compliance buyers only | AUD |
| `avg_price_median_aud` | Median average credit price (total cost / total qty) | AUD/credit |
| `compliance_avg_price_median_aud` | Median average price paid by compliance buyers | AUD/credit |
| `cost_change_vs_baseline_pct` | (scenario cost − baseline cost) / baseline cost × 100 | % |

**Caveat**: Absolute cost values depend on model price distributions (lognormal reservation
prices, production costs). Relative comparisons across scenarios (e.g., +0.6% for
Procurement Flexibility) are more robust than absolute magnitudes because the same price
generation process applies to all scenarios.
