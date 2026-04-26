# Data Sources

All data used in this project, with provenance, access dates, and notes.

## Reproducibility tiers

| Tier | Description | Reproducible? |
|------|-------------|---------------|
| **Tier 1** | Computed from NSW public register CSVs (transactions + supply) | Yes — CSV files in `data/raw/nsw/` |
| **Tier 2** | From IPART regulatory reports (HHI, proponent shares, bypass rate) | No — IPART has proponent-linked admin data; the public register uses anonymised CR-IDs |

Tier 2 statistics are cited with full URL and page references below.
The manuscript Methods section contains an explicit disclosure of this limitation.
Field-level documentation for all CSV files is in `data/CODEBOOK.md`.

## NSW Biodiversity Offsets Scheme (primary data)

| Source | URL | Date accessed | Notes |
|--------|-----|---------------|-------|
| NSW Environment: Understanding the biodiversity credits market | https://www.environment.nsw.gov.au/.../understanding-biodiversity-credits-market | Feb 2026 | Total market value 2010–2023: AUD $763.1M. FY2022-23: $105.1M. FY2021-22: $63.3M. |
| IPART Discussion Paper 2024–25 **(Tier 2)** | https://www.ipart.nsw.gov.au/sites/default/files/cm9_documents/Discussion-Paper-2024-25-Biodiversity-Market-Monitoring-Review-30-October-2025.pdf | Oct 2025 | BCT: 16% credits, 45% value (p.7); like-for-like 65% (p.10). Buyer HHI 2966, seller HHI 678 (p.6). Top-5 buyers 79%, sellers 43% (p.6). 45 buyers, 70 sellers (p.6). 118 credit types traded +40% (p.7). Fund bypass ~1/6 of market (p.8). Fund: 80K obligations, $337M cumulative (p.8). Acquittal: 23% full, 20% purchased-pending, 28% committed (p.10). Amendment Act 2024 commenced 7 Mar 2025 (p.2). |
| IPART Annual Report 2023–24 **(Tier 2)** | https://www.ipart.nsw.gov.au/sites/default/files/cm9_documents/Discussion-Paper-2023-24-Biodiversity-Market-Monitoring-Review-9-September-2024.PDF | Sep 2024 | Proponent A ~27% (p.6 graph). Proponent B = 28% (p.7). BCT: 5% credits, 28% value; like-for-like ~90%. Fund transfers 136->70 developers (p.7). Bypass 66% (p.7). Buyer HHI 2187 (first year IPART reported HHI). |
| IPART Annual Report 2022–23 **(Tier 2)** | https://www.ipart.nsw.gov.au/documents/annual-report/annual-report-2022-23-biodiversity-market-monitoring-december-2023 | Dec 2023 | 3 buyers = 85% of credits (p.31). Proponent A ~60% (~51,000 credits). BCT ~7% from graph (p.31); BCT+CSF combined ~15%. 152 Fund vs 36 market developers = 81% bypass rate. 5 key market problems. BCF phase-out recommended. **Does not report HHI.** |
| NSW Audit Office: Effectiveness of the BOS (Sep 2022) | https://www.audit.nsw.gov.au/... | Feb 2026 | Major project obligations = 90% of total. Used for compliance demand concentration parameter. |
| NSW Treasury Budget Paper No. 3, 2022-23 | — | Feb 2026 | AUD $113 billion infrastructure pipeline. Used for compliance demand projections. |
| NSW BOS Credit Transactions Register (CSV) | https://customer.lmbc.nsw.gov.au/... | Mar 2026 | 2,244 rows: 1,134 transfers, 1,097 retirements, 11 cancellations. 1,124 priced transfers after removing Price < 100 AUD. 764 ecosystem (68%), 360 species (32%). Retirements identified by Transaction Type = "Retire"; Retirement Reason field used for BCT/compliance classification. |
| NSW BOS Credit Supply Register (CSV) | https://customer.lmbc.nsw.gov.au/... | Mar 2026 | 3,777 rows. Supply-side OTG universe: 252 ecosystem OTGs with registered supply used for EC1/EC2 denominator. 79 TEC OTGs, 173 non-TEC OTGs. 15 vegetation formations. |
| UNEP: State of Finance for Nature (2023) | — | Feb 2026 | Global offset market value: USD $11.7 billion/year (Fig. 1). |
| IAPB: Framework Report (Oct 2024) | — | Feb 2026 | 19 government-led biodiversity credit schemes globally (p.6). |
| BASE Energy: State of Play of Biodiversity Credits (2025) | — | Feb 2026 | FY2023-24: 101,894 credits = AUD $285M. |

## Derived metrics (from transaction + supply registers)

| Metric | Value | Source |
|--------|-------|--------|
| EC1 (>=1 transaction) | 56% (140/252 ecosystem OTGs) | Transactions + Supply registers |
| EC2 (>=5 transactions, functional) | 17.9% (45/252 ecosystem OTGs) | Transactions + Supply registers |
| OTGs never traded | 44% (112/252) | Transactions + Supply registers |
| TEC OTGs | 79 | Supply register TEC classification |
| Non-TEC OTGs | 173 | Supply register |
| EC2-TEC (strict matching) | 20.3% (16/79) | Transactions + Supply registers |
| BCT-only OTGs | 24 OTGs (no direct compliance buyer) | Retirement reason linkage |
| BCT median price, thin markets | AUD 5,989 | Transactions + Retirement linkage |
| Compliance median price, thin markets | AUD 1,125 | Transactions + Retirement linkage |
| Price ratio (BCT/compliance, thin) | 5.3x | Derived (5,989 / 1,125) |
| Voluntary retirements | 1 (0.09% of 1,097) | Transactions register |

## UK Biodiversity Net Gain (out-of-sample validation only)

| Source | URL | Date accessed | Notes |
|--------|-----|---------------|-------|
| DEFRA: BNG statutory credits annual report 2024–25 | https://www.gov.uk/government/publications/... | Mar 2026 | Year 1 total: GBP 206,180. First sale: GBP 35,120 (Sep 2024). |
| Natural England: Statutory biodiversity credit prices | https://www.gov.uk/guidance/statutory-biodiversity-credit-prices | Mar 2026 | GBP 42,000 (low) to GBP 650,000 (very high distinctiveness) per credit. |
| CW Habitats: BNG unit pricing | https://cwhabitats.co.uk/how-much-are-bng-units-worth/ | Mar 2026 | Private market: GBP 20,000–GBP 25,000 per unit. |
| Carbon Pulse: First UK statutory credit sale | https://carbon-pulse.com/318528/ | Mar 2026 | Sep 2024, GBP 35,120. |

Data file: `data/raw/uk_bng/uk_bng_year1.csv`.
Note: UK BNG is used only for out-of-sample qualitative predictions. No calibration performed on UK data.

## External / Comparative

| Source | URL | Date accessed | Notes |
|--------|-----|---------------|-------|
| Scholl, Calinescu & Farmer (2021) | https://doi.org/10.1016/j.jedc.2021.104138 | Feb 2026 | Original market ecology framework. JEDC. |

## Author calculations (Tier 2b — derived from IPART-reported shares)

IPART does not compute or report HHI. The values below are author calculations
using IPART-reported market shares with a uniform distribution assumption for
the "smaller buyers" residual.

| Metric | Value | Method | Source data |
|--------|-------|--------|-------------|
| HHI 2022–23 | 0.41 | 0.60^2 + 0.15^2 + (0.25/10)^2 x 10 | IPART Annual Report 2022-23 p.31 shares |
| HHI 2023–24 | 0.19 | 0.27^2 + 0.28^2 + 0.13^2 + (0.32/15)^2 x 15 | IPART Discussion Paper 2023-24 p.6 shares |
| Bypass rate 2022–23 | 81% | 152 / (152 + 36) | IPART Annual Report 2022-23 (fund vs market developers) |
| Bypass rate 2023–24 | 66% | 70 / (70 + 36) | IPART Discussion Paper 2023-24 p.7 |

## Known data quality issues

1. **BCT dispute (March 2025):** BCT sent IPART a letter on 6 March 2025
   "providing feedback on perceived inaccuracies" in the Annual Report 2023-24.
   The nature of the disputed figures is not publicly disclosed.
   Source: IPART Annual Report 2023-24 landing page note.

2. **Proponent A share discrepancy:** Annual Report 2022-23 says "~60%";
   Discussion Paper 2023-24 retro-cites "63% (50,734 credits)". We use 60%
   (the primary source) and note the 3 pp discrepancy.

3. **Pre-2022 data:** IPART monitoring began in 2022-23; no IPART-sourced
   values are reported for earlier years.

4. **BCT share data coverage:** Only one verified BCT-alone data point from the
   Annual Report era (7% in 2022-23, Annual Report p.31 graph). The Discussion
   Paper 2024-25 provides three-year series: 7%, 5%, 16%.

5. **BCT market share trend reversal:** BCT credit share was ~7% (2022-23),
   fell to 5% (2023-24), then grew to 16% (2024-25). Earlier versions of this
   manuscript reported a declining trend based on incomplete data. The 2024-25
   Discussion Paper revealed the reversal. The paper now focuses on like-for-like
   quality erosion (90%->65%) rather than volume decline.

## Data gaps (priority for follow-up)

1. **UK BNG off-site unit market data** — would enable first true transaction-level out-of-sample test of rarity selection in a newly launched statutory market
2. **Victoria (Australia) Native Vegetation Register** — potential second calibration market; similar OTG structure to NSW
3. **NSW individual buyer identifiers** — aggregate register obtained; buyer IDs not public, needed to estimate within-type variance and validate BCT-type agent classification empirically
4. **BCT off-market acquisition records** — BCT shifted to reverse auctions and conservation agreements from ~2025; these transactions do not appear in the public register; FOI request would enable full BCT procurement footprint analysis
