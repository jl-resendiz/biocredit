# NSW Raw Data

## What's here

Files downloaded from the NSW Government Biodiversity Credits & Conservation Management portal (accessed March 2026):

- `nsw_credit_transactions_register.csv` — All credit transactions: transfers (1,134), retirements (1,097), and cancellations (11). Transfers with Price >= 100 AUD are the primary calibration dataset (1,124 priced transactions; 764 ecosystem, 360 species). Retirements are identified by `Transaction Type = "Retire"` and include the `Retirement Reason` field used for buyer classification.
- `nsw_credit_supply_register.csv` — All accredited credit supply by OTG. Source of the EC1/EC2 denominator (252 ecosystem OTGs with registered supply).

Field-level documentation is in [`data/CODEBOOK.md`](../../CODEBOOK.md).

## Download URLs

The two public registers are available at:
```
https://customer.lmbc.nsw.gov.au/application/BOAMCreditTransactionSaleRegisterExport
https://customer.lmbc.nsw.gov.au/application/BOAMCreditSupplyRegisterExport
```

Note: these URLs may return 403 errors from outside Australia; access via an Australian network or request from DCCEEW: bosadmin@environment.nsw.gov.au

## Key derived statistics

| Metric | Value | Source |
|--------|-------|--------|
| Total priced transactions (after filter) | 1,124 | Transactions register (Transfer, Price >= 100 AUD) |
| — Ecosystem | 764 (68%) | Transactions register |
| — Species | 360 (32%) | Transactions register |
| Total retirement records | 1,097 | Transactions register (Transaction Type = "Retire") |
| — Direct compliance | 709 (65%) | Retirement Reason field |
| — BCT-mediated compliance | 388 (35%) | Retirement Reason field |
| — Voluntary | 1 (0.09%) | Retirement Reason field |
| Ecosystem OTGs with registered supply | 252 | Supply register |
| OTGs never traded (EC1 gap) | 112 (44%) | Transactions + Supply |
| OTGs with ≥5 transactions (EC2) | 45 (17.9%) | Transactions + Supply |
| TEC OTGs | 79 | Supply register |
| Non-TEC OTGs | 173 | Supply register |
| EC2-TEC (strict matching) | 20.3% (16/79) | Transactions + Supply |
| BCT-only OTGs | 24 | Retirement reason linkage |
| BCT thin-market median price | AUD 5,989 | Transactions + Retirement linkage |
| Compliance thin-market median price | AUD 1,125 | Transactions + Retirement linkage |
| BCT premium (thin markets) | 5.3x | Derived (5,989 / 1,125) |

## Buyer classification linkage

Transfer buyers are linked to retirement reason via: `transfer["To"] == retire["From"]`.
The dominant retirement reason per entity is used to classify that entity's buyer type.
See `scripts/verify_claims.py` for the current implementation.

## Remaining data gaps

1. **Individual buyer identifiers** — the public register does not include buyer names/IDs at transaction level; aggregate buyer shares are available from IPART reports only
2. **BCT off-market acquisitions** — BCT's reverse auction and conservation agreement purchases from ~2025 onward are not in the public transaction register; these require an FOI request
