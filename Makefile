.PHONY: all results figures verify check clean robustness manifest holdout diagnostics community_matrix params abm glv fig1 fig2 fig3 suppfig1 sobol sobol_smoke

PYTHON = python
SRC = src
OUT = output

# ── Full pipeline ────────────────────────────────────────────
all: results community_matrix robustness figures verify check manifest

# ── Run models (generate results JSON) ──────────────────────
results: abm glv

abm:
	$(PYTHON) scripts/run_abm.py

glv:
	$(PYTHON) -m src.model.glv_formation

# ── Community matrix validation (ABM Jacobian vs GLV alpha) ─
community_matrix:
	$(PYTHON) scripts/community_matrix.py

# ── Parameter derivation audit (documentation; not run by `all`) ─
# Prints every model parameter with its empirical derivation. The
# runtime derivation lives inside `scripts/run_abm.py:derive_all_parameters`
# and `src/model/glv_formation.py:derive_parameters`; this script is the
# auditable single-stop record cited in docs/PARAMETER_REGISTRY.md.
params:
	$(PYTHON) scripts/derive_parameters.py

# ── Figures (read from results JSON) ────────────────────────
figures: fig1 fig2 fig3 suppfig1

fig1:
	$(PYTHON) -m src.figures.fig1_problem

fig2:
	$(PYTHON) -m src.figures.fig2_mechanism

fig3:
	$(PYTHON) -m src.figures.fig3_policy

suppfig1:
	$(PYTHON) -m src.figures.suppfig1_robustness

# ── Verification ─────────────────────────────────────────────
verify:
	$(PYTHON) scripts/verify_claims.py

# ── Consistency check (manuscript vs pipeline) ───────────────
check:
	$(PYTHON) scripts/check_consistency.py

# ── Robustness ───────────────────────────────────────────────
robustness:
	$(PYTHON) scripts/robustness_empirical.py
	$(PYTHON) scripts/robustness_sensitivity.py
	$(PYTHON) scripts/robustness_demand_functions.py
	$(PYTHON) scripts/robustness_trend.py

# ── Global Sobol sensitivity (heavy; not part of `all`) ──────
# Smoke test (~10 min on a single core, k=3, N=8, 1 inner seed):
sobol_smoke:
	$(PYTHON) scripts/sensitivity_sobol.py --smoke-test

# Full Saltelli design (k=3, N=512, 5 inner seeds = 12,800 ABM runs;
# allow ~13h on 8 cores; tune --workers for available core count):
sobol:
	$(PYTHON) scripts/sensitivity_sobol.py --N 512 --inner-seeds 5 --k 3 --workers 8

# ── Manifest (reproducibility snapshot) ──────────────────────
manifest:
	$(PYTHON) scripts/generate_manifest.py

# ── Hold-out validation ─────────────────────────────────────
holdout:
	$(PYTHON) scripts/holdout_validation.py

# ── Diagnostics ──────────────────────────────────────────────
diagnostics:
	$(PYTHON) scripts/model_diagnostics.py

# ── Clean ────────────────────────────────────────────────────
clean:
	rm -f $(OUT)/figures/main/*.png
	rm -f $(OUT)/figures/main/*.pdf
	rm -f $(OUT)/figures/supplementary/*.png
	rm -f $(OUT)/figures/supplementary/*.pdf
	rm -f $(OUT)/tables/*.csv
	rm -f $(OUT)/results/*.json
