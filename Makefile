# Reproduce the analyses, figures and numbers of the semaglutide REM heart-rate paper.
#
# Targets run in the order listed under `all`; each one consumes the outputs of the ones
# before it. Every target needs HPP data access (see README.md).

PY ?= python3

.PHONY: all transitions upstream cohort secondary figs numbers

all: transitions upstream cohort secondary figs numbers

# Visit-reported medication transition table, read by step 26.
# Writes src/visit_transition_analysis/outputs/.
transitions:
	$(PY) src/visit_transition_analysis.py

# Visit-transition reanalysis (step 26): the published-cohort comparison table and the diet
# cache that step 48 reads. Raw sleep panel (step 31): read by Figure 1.
upstream:
	$(PY) steps/26_paper_reanalysis.py
	$(PY) steps/31_raw_sleep_glp_cohort.py

# Date-anchored cohort, matching and primary estimates for every modality (semaglutide),
# plus the two alternative exposure definitions used by step 51.
cohort:
	$(PY) steps/48_unified_cohort.py --exposure semaglutide --modalities paper --skip-raw
	$(PY) steps/48_unified_cohort.py --exposure any_glp1 --modalities sleep,anthropometrics,blood_pressure --skip-raw
	$(PY) steps/48_unified_cohort.py --exposure non_semaglutide --modalities sleep,anthropometrics,blood_pressure --skip-raw

# Sleep secondary analyses, overlap-weighted alternative matching, and sleep robustness.
secondary:
	$(PY) steps/49_unified_sleep_secondary.py
	$(PY) steps/50_unified_alternative_matching.py
	$(PY) steps/51_unified_sleep_robustness.py

# Figure 1 (needs paper/.participant_crosswalk.json), then Figures 2-4 and S1-S6.
figs:
	$(PY) paper/build_fig1_unified.py
	$(PY) paper/build_unified_figures.py

# paper/NUMBERS.md and paper/tables/*.md.
numbers:
	$(PY) paper/build_numbers.py
