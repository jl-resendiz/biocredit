# Model Specification

Specification of the formation-level ABM and the GLV analytical complement, organised by structural assumption. Each entry states the assumption and its empirical or theoretical anchor.

## A1. Market-ecology framework

The biodiversity credit market is modelled as an ecological community in which behavioural strategies compete for finite compliance demand. The Generalised Lotka-Volterra (GLV) framework provides an analytically tractable complement to the agent-based simulation, following Scholl, Calinescu & Farmer (2021). Cross-model agreement on the species-level community matrix is established via Spearman $\rho = 0.689$ ($p < 10^{-30}$).

## A2. Fifteen vegetation formations as competing populations

The 252 ecosystem credit types are grouped into 15 Keith vegetation formations. Each formation is a population that competes for finite buyer demand. Formations with more historical liquidity attract proportionally more future demand (preferential attachment). The 15 formations are the resolution at which transaction counts support statistical calibration; they are ecologically meaningful (Keith 2004) and mirror the NSW regulatory framework. Empirical concentration is high: the top three formations account for over 60% of transactions.

## A3. Four behavioural strategies as species

Market participants are grouped by behavioural strategy:

1. **Cost-minimising compliance** — agents seeking the cheapest credit that satisfies the obligation. NSW realisation: compliance buyers under statutory offset obligations.
2. **Mean-reversion intermediation** — agents bidding against rolling fundamental value to exploit price dispersion. NSW realisation: intermediaries (brokers, price-volatility-exploiting entities including the Credits Supply Fund).
3. **Thin-market procurement** — budget-constrained agents that bid above the compliance median in low-liquidity types to ensure coverage. NSW realisation: the Biodiversity Conservation Trust.
4. **Cost-floor supply** — credit producers that sell only when market price exceeds a fraction of marginal production cost. NSW realisation: habitat banks (landholders generating and selling credits).

Two organisations are the same species iff they share the same signal function $\phi_i$. Implementation: `src/model/species.py`; interaction coefficients in `src/model/interactions.py`.

The empirical signature for thin-market procurement is the price-tier separation: the Trust pays a median of AUD\,5,989 per credit in thin markets versus AUD\,1,125 for direct compliance buyers (5.3× ratio).

## A4. Like-for-like matching with variation rules

Credits match within the same vegetation formation by default; with probability $P_{\text{variation}} = 0.5$ a buyer accepts a credit from a different formation that is admissible under regulatory variation rules. The single calibration target $P_{\text{variation}}$ matches the observed Fund-routing rate of 35.4%. NSW regulation requires like-for-like matching but allows variation under specified conditions (IPART 2024, like-for-like rate 65% in 2024–25).

## A5. Liquidity feedback as an empirical preferential-attachment kernel

OTGs that have traded before attract more demand. The transition weights (1.0, 1.5, 2.9, 5.1, 8.4 for cumulative transaction buckets {0, 1–2, 3–4, 5–9, 10+}) are empirically estimated from the NSW transaction register and applied as a fixed kernel during simulation. This kernel is the third reinforcing mechanism alongside least-cost selection and Fund routing.

## A6. BCT as risk-neutral procurement agent

BCT-type agents maximise expected credit procurement without utility-theoretic risk aversion, following Arrow-Lind: a government agency spreading risk across taxpayers is approximately risk-neutral. BCT's mandate under the Biodiversity Conservation Act 2016 is coverage, not cost minimisation; the empirically calibrated bid ceiling (AUD\,5,989) captures procurement uncertainty as a cap on willingness to pay.

## A7. Habitat-bank supply elasticity

Habitat-bank supply responds inelastically to price changes (estimated elasticity 0.39). Production cost is drawn from $\mathrm{lognormal}(\mu_{\log}=7.62, \sigma=0.5)$, anchored at the first quartile of observed ecosystem credit prices (AUD\,2,033). The elasticity is derived from NSW supply-register data (`scripts/derive_parameters.py`); biodiversity credit production requires multi-year restoration, supporting low elasticity.

## A8. Geographic demand distribution by IBRA subregion

Compliance demand is distributed across IBRA subregions proportionally to observed transaction frequencies; each subregion's demand is matched against the formations present in that subregion. The IBRA subregion field is recorded in both registers (47 subregions in transactions, 162 in supply), constraining which formations can satisfy each demand event.
