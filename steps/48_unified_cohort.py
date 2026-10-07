"""One cohort, one matching rule, two instruments. Replaces the steps/26 + steps/31 split.

The whole specification, which is the point of this file
--------------------------------------------------------
Initiators are participants with a semaglutide medication report carrying a parseable start
date (brand or generic; tirzepatide/Mounjaro excluded). Controls are participants with no
GLP-1 report of any kind, dated or not. For each initiator the PRE visit is the last sleep
study strictly before the start date and the POST visit the first sleep study at least
FLOOR_MONTHS after it. Controls are eligible for any pre/post visit pair an initiator
realises. Each initiator is matched to one control without replacement, exactly blocked on
the visit pair and sex, nearest on a propensity score within a 0.2 SD caliper. The same
procedure is applied twice: once to participants with WatchPAT summary outcomes (TABULAR),
and once to the subset that also has per-second channels (RAW).

That is all of it. What it replaces, and why
--------------------------------------------
`steps/26` took exposure from a visit-reported medication TRANSITION table and fixed pre/post
to the two visit LABELS that transition names. `steps/31`/`32` took exposure from medication
REPORTS and dates, chose pre/post by date, and did no matching at all -- overlap weighting
against a never-GLP reference. Two defensible designs that were never reconciled, and the
seams were load-bearing:

  * Figure 1c's 33 initiators are NOT a subset of Figure 1d's 44; 26 overlap. Of the 7
    raw-only, 6 fail because the outcome does not exist at the two visits their label window
    names -- they all already HAVE tabular features, some at three visits -- and 1 fails at
    matching.
  * The label rule's pre visit need not precede the drug. `enforce_prior_visit_dates`
    (src/aggregate_medication_stats.py:488) tolerates a start date up to 3 months earlier;
    2 of the published 44 were already on semaglutide at their own "pre" visit.
  * It carries no exposure-duration floor: 37% of confirmations sit inside titration.

Re-pairing the tabular arm by date alone moved REM from +6.97 [4.36, 9.72] to +4.32
[2.01, 6.72] on a LARGER, better-balanced cohort passing the same gates -- see
`steps/47_date_anchored_pairing.py`, which priced that change in isolation and is the direct
ancestor of this file.

Why the raw set nests by construction
-------------------------------------
The raw arm does NOT re-derive its own cohort. It inherits the tabular arm's matched
initiators and their exact pre/post windows, and asks only whether the per-second channels
also yield a usable value at those two visits. Controls are then matched inside the
raw-available pool by the same procedure. So the raw treated set is a subset of the tabular
matched set definitionally, and the two panels are two READOUTS OF ONE COHORT rather than
two cohorts that happen to overlap.

This was got wrong on the first attempt, which is worth recording. The raw arm originally
re-paired and re-matched independently, on the theory that "raw eligibility = tabular
eligibility AND has channels" makes nesting automatic. It does not, for two reasons: the
per-second QC (MIN_STATE_SECONDS on scored epochs) and the summary QC (total_rem_sleep_time,
quality_score_heart_rate) can disagree on a given night, so raw availability is not a subset
of tabular availability; and even with nested eligibility, two independent 1:1 no-reuse
matches draw from different control pools and need not retain the same treated units. The
assertion in `main()` caught it -- 1 raw-only initiator -- and is kept as a tripwire.

The mechanism is `raw_values_panel`: it emits the SAME long schema and the SAME measurement
names as `load_snapshot_panel("sleep")`, so `prepare_units` -> `unit_deltas` -> `pair_effects`
-> `summarize_primary` run over either instrument unchanged. The two arms are estimated by
identical code rather than by parallel implementations -- which is what "matched the same
way" has to mean if it is to be checkable.

Strictness
----------
Exact blocks are `pre_stage, post_stage, gender`, down from step 26's six. The four dropped
covariates (diabetic, smoking, lipid-lowering, antihypertensive) move into the propensity
model, where they are still balanced rather than fragmenting an ~8.5k control pool into
strata. Step 26's blocking is still run and reported, so the choice is visible: at a 3-month
floor the six-way blocking gave max |SMD| 0.131 on 36 matched, a balance failure caused by
thin strata rather than by confounding.

The floor is a choice, not a finding
------------------------------------
The estimate rises with FLOOR_MONTHS (measured under the old blocking: +4.32 at 0, +5.14 at
2, +6.09 at 3). It therefore must be argued from the titration schedule and NOT selected from
the results column. 3 months is post-titration for Ozempic (0.25 -> 0.5 -> 1.0 mg over ~8
weeks); Wegovy climbs to 2.4 mg over 16 weeks and is still titrating at 3 months, which the
Methods must state. `--floor-months` re-runs any other value and `--sweep` runs the set.

NOTE: this contradicts the run's existing position that under 6 months is sub-therapeutic,
which is also how Figure 1b's exemplar was chosen. If this spec is adopted the position has
to be revised in CURATION.md, in `_test_fig1_alt.py`'s exemplar selection and in the Methods,
or the run will carry two definitions of "exposed".

Scope: writes only to `outputs/unified_cohort/`. No figure, table or manuscript file is
touched, and steps 26/31/32 keep working so the published numbers stay reproducible.

Usage
-----
    PY=/net/mraid20/export/jasmine/david/anaconda3/bin/python3
    $PY steps/48_unified_cohort.py                      # tabular + raw at the 3-month floor
    $PY steps/48_unified_cohort.py --floor-months 6
    $PY steps/48_unified_cohort.py --sweep              # 0/2/3/4/6, tabular only
    $PY steps/48_unified_cohort.py --skip-raw           # fast iteration on the tabular arm
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "outputs" / "unified_cohort"
CACHE = OUT / "cache"

FLOOR_MONTHS = 3.0
MAIN_STAGES = ("00_00_visit", "02_00_visit", "04_00_visit", "06_00_visit")
# Interim visits carry real data -- 333 sleep studies at 03_01, 210 at 01_01, 114 at 05_01 --
# and nothing in a date-anchored design needs a visit to be a "main" one: the rule only asks
# whether a study sits before the start date or far enough after it. `--stages all` admits
# them. The cost is that the visit pair is an exact-blocking key, so more distinct stages
# means more strata and thinner matching; measured below rather than assumed.
INTERIM_STAGES = ("01_01_visit", "03_01_visit", "05_01_visit", "00_01_visit",
                  "02_02_visit", "04_01_visit")
STAGE_SETS = {"main": MAIN_STAGES, "all": MAIN_STAGES + INTERIM_STAGES}
SEED = 20260917
CALIPER_SD = 0.2
# Exact blocks. `relaxed` is the three that carry design meaning: the visit-wave pair and
# sex. The stage pair controls cohort follow-up wave, not exact calendar time; calendar dates
# vary between participants and are used to anchor each participant's drug exposure window.
# `plus_lipids` adds the one covariate that measurably needs it -- see below.
# `step26` is the published six, kept runnable so the comparison is an argument and not an
# assertion.
#
# Why on_lipid_lowering earns a block and diabetic/smoking do not: measured on this cohort,
# it enters matching ALREADY balanced (|SMD| 0.028) and the match perturbs it to 0.131, while
# the covariates with real imbalance are fixed by the propensity model alone (hip
# circumference 1.24 -> -0.05, visceral fat 0.67 -> 0.05, antihypertensive 0.38 -> -0.02).
# Blocking is the cheap remedy for a covariate the propensity model has no incentive to
# protect because it was never unbalanced. Adding all six back would be the expensive one.
EXACT_BLOCKS = {
    "relaxed": ["pre_stage", "post_stage", "gender"],
    "plus_lipids": ["pre_stage", "post_stage", "gender", "on_lipid_lowering"],
    "plus_age_band": ["pre_stage", "post_stage", "gender", "on_lipid_lowering", "age_band"],
    "step26": ["pre_stage", "post_stage", "gender", "diabetic", "on_lipid_lowering",
               "smoking"],
}
DEFAULT_BLOCKING = "plus_lipids"
# `plus_age_band` was TRIED AND REJECTED, recorded here so it is not retried. The raw arm's
# worst covariate is age (|SMD| 0.055 before matching, -0.155 after), and coarsened exact
# matching on 5-year bands is the textbook remedy. Measured, it made everything worse:
#
#              tabular max|SMD|   tabular pairs   raw max|SMD|   raw pairs
#   plus_lipids          0.087             110          0.155          74
#   plus_age_band        0.250              89          0.195          54
#
# Five age bands x visit pair x sex x lipid-lowering fragments the control pool faster than
# the extra block buys balance, and the loss shows up on OTHER covariates. This is the same
# strata-thinning that made step 26's six-way blocking fail at small n, reproduced. The
# residual age imbalance is reported rather than tuned away.
AGE_BAND_YEARS = 5

# Raw stage -> the tabular measurement it stands in for. Whole sleep has no raw state of its
# own and is derived in `raw_values_panel` as the duration-weighted mean over rem+light+deep,
# which is what the WatchPAT summary's `heart_rate_mean_during_sleep` also represents.
RAW_TO_MEASUREMENT = {
    "rem": "heart_rate_mean_during_rem",
    "nrem": "heart_rate_mean_during_nrem",
    "wake": "heart_rate_mean_during_wake",
    "sleep": "heart_rate_mean_during_sleep",
    "rem_minus_nrem": "heart_rate_rem_minus_nrem",
}
MEASUREMENTS = tuple(RAW_TO_MEASUREMENT.values())

# The two numbers this has to be explainable against.
PUBLISHED = {"label_rem": (6.97, 4.36, 9.72, 41), "dated_rem": (4.32, 2.01, 6.72, 48),
             "panel_c_rem": (4.22, None, None, 33)}


def as_registration_code(values, step26=None) -> pd.Series:
    """Any of the three id spellings -> the '10K_1234567890' form the analysis keys on.

    Three are live: the snapshot writes '10k_...', the exposure and covariate frames
    '10K_...', and the raw gold link table plus this run's cohort CSVs a bare integer.
    `normalize_reg` (src/glp_modality_matched_analysis.py:231) only upper-cases, which is all
    the snapshot needs, so a bare-digit id joins to nothing and the failure looks like an
    empty cohort rather than an error. `step26` is accepted and ignored for call-site
    compatibility -- the rule is self-contained on purpose, since this file should not depend
    on a sibling test script for its id handling.
    """
    text = pd.Series(values, dtype="string").str.strip()
    bare = text.str.fullmatch(r"\d+").fillna(False)
    return text.mask(bare, "10K_" + text).str.upper()


def canonical_stage(stage: str) -> str:
    """Stage names, without destroying sub-visits.

    NOT `steps/47`'s `normalise_stage`, which must not be used once interim visits are in
    play. That one exists to turn the transitions file's '02 visit' into '02_00_visit' and
    matches `^(\\d{2})[ _]`, so it rewrites an ALREADY canonical '03_01_visit' to
    '03_00_visit' (a stage that does not exist) and folds '00_01_visit' into baseline. It is
    harmless while everything is filtered to the four main visits and silently wrong after.
    Here: pass canonical names through untouched, and only rewrite the loose spellings.
    """
    text = str(stage).strip().lower()
    if text == "baseline":
        return "00_00_visit"
    if re.fullmatch(r"\d{2}_\d{2}_visit", text):
        return text
    match = re.fullmatch(r"(\d{2})[ _]visit", text)
    return f"{match.group(1)}_00_visit" if match else text


def canonical_visit_dates(step26) -> pd.Series:
    """One date per (participant, research_stage): the date of the visit itself.

    `dxa`, `microbiome_function` and `diet` reach this file keyed by (participant, stage)
    without a measurement date, yet still have to be ordered against a drug start date, so
    they take the VISIT's date. LabData's `TimelineLoader` records it from the 10K warehouse
    visit and call tables (`done_at`). 1,563 participants, mostly with 2023-2026 visits, are
    missing from those tables; their visits take the anthropometry or blood-pressure date
    instead. Both are measured at the visit, and both fall within 7 days of the timeline date
    for every visit where the two coexist.

    This calendar used to be the earliest dated measurement at the stage, pooled across the
    snapshot panels. That pool included HMO blood tests, which are health-fund lab records
    filed under the stage and drawn a median 399 days before the visit, so it dated half of
    all stages more than 6 months early and put 28 of 115 DXA "pre" scans after the drug
    start.
    """
    from LabData.DataLoaders.TimelineLoader import TimelineLoader

    timeline = TimelineLoader().get_data(study_ids=["10K"]).df.reset_index()
    table = pd.DataFrame({
        "participant_id": as_registration_code(timeline.RegistrationCode, step26).to_numpy(),
        "stage": timeline.stage.map(canonical_stage).to_numpy(),
        "date": pd.to_datetime(timeline.Date, utc=True),
    })
    timeline_dates = table.dropna(subset=["date"]).groupby(["participant_id", "stage"]).date.min()
    measured_at_visit = pd.concat(
        [visit_dates_for(step26, "anthropometrics"), visit_dates_for(step26, "blood_pressure")],
        axis=1,
    ).min(axis=1)
    return timeline_dates.combine_first(measured_at_visit)


def load_panel(step26, modality: str, exposure: pd.DataFrame) -> pd.DataFrame:
    """The outcome panel for any modality, snapshot-backed or not."""
    if modality in step26.PANELS:
        return step26.load_snapshot_panel(modality)
    if modality == "dxa":
        return step26.load_dxa_panel()
    if modality == "microbiome":
        return step26.load_microbiome_panel()
    if modality == "microbiome_function":
        return step26.load_humann_panel()
    if modality.startswith("mb_") and modality != "mb_diversity":
        return load_microbiome_level(step26, modality.removeprefix("mb_"))
    if modality == "mb_diversity":
        return load_microbiome_diversity(step26)
    if modality == "diet":
        return step26.load_diet_panel(exposure.rename(
            columns={"participant_id": "RegistrationCode"}))
    raise ValueError(f"unknown modality {modality!r}")


def dates_for(step26, modality: str, canonical: pd.Series) -> pd.Series:
    """The modality's own measurement dates when it has them, the visit calendar otherwise."""
    if modality in step26.PANELS:
        return visit_dates_for(step26, modality)
    if modality in ("dxa", "microbiome"):
        try:
            return visit_dates_for(step26, modality)
        except Exception as error:                   # noqa: BLE001
            logging.warning("%s: no own dates (%s); using the visit calendar", modality, error)
    return canonical


def visit_dates_for(step26, modality: str) -> pd.Series:
    """Earliest measurement date per (participant, stage), for the modality being paired.

    Each modality carries its own dates, and the date that matters is the one the OUTCOME
    was measured on -- pairing a blood-pressure reading against the date of a sleep study
    would put the wrong visit on either side of the drug. So this is per modality rather
    than a single shared calendar.
    """
    if modality in step26.PANELS:
        spec = step26.PANELS[modality]
        source, columns = spec["source"], list(spec["columns"])[:1]
    elif modality == "microbiome":
        source, columns = ("gut_microbiome", "gut_mb_metaphlan_genus"), None
    else:
        raise ValueError(f"{modality}: no snapshot source for dates")
    frame, meta = step26.read_snapshot(*source, columns=columns)
    stage = pd.Series(meta["research_stage"].to_numpy()).map(canonical_stage)
    if "RegistrationCode" in frame.index.names:
        participant = frame.index.get_level_values("RegistrationCode")
    else:
        participant = meta["RegistrationCode"]
    participant = as_registration_code(pd.Series(np.asarray(participant)), step26)

    when = None
    for level in frame.index.names:
        if level and "date" in str(level).lower():
            when = pd.to_datetime(frame.index.get_level_values(level), utc=True, errors="coerce")
            break
    if when is None:
        for column in ("Date", "date", "created_at", "collection_date"):
            if column in meta.columns:
                when = pd.to_datetime(meta[column], utc=True, errors="coerce")
                break
    if when is None:
        raise ValueError(f"{modality}: no measurement date in index or metadata")
    table = pd.DataFrame({"participant_id": np.asarray(participant),
                          "stage": stage.to_numpy(), "date": np.asarray(when)})
    return table.dropna(subset=["date"]).groupby(["participant_id", "stage"]).date.min()


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RUN / "steps" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# GLP-1 receptor agonists present in this cohort, by agent. Tirzepatide is a dual GIP/GLP-1
# agonist rather than a pure GLP-1, which is why step 26 carries it as a separate active
# comparator and why it is named here rather than folded into "any GLP-1" silently.
AGENTS = {
    "semaglutide": r"ozempic|wegovy|rybelsus|semaglutide|אוזמפיק|ריבלסוס|וויגובי",
    "liraglutide": r"saxenda|victoza|liraglutide|סקסנדה|ויקטוזה",
    "dulaglutide": r"trulicity|dulaglutide|טרוליסיטי",
    "tirzepatide": r"mounjaro|tirzepatide|מאונג",
}
# Conservative calendar plausibility floors for a recorded treatment start. These are not
# used to infer availability in a particular country; they only prevent plainly impossible
# dates (for example, semaglutide in 2002) from entering the exposure funnel. They precede or
# coincide with broad clinical availability of each agent.
AGENT_PLAUSIBILITY_FLOORS = {
    "semaglutide": pd.Timestamp("2018-01-01", tz="UTC"),
    "liraglutide": pd.Timestamp("2010-01-01", tz="UTC"),
    "dulaglutide": pd.Timestamp("2014-01-01", tz="UTC"),
    "tirzepatide": pd.Timestamp("2022-01-01", tz="UTC"),
}
EXPOSURE_SETS = {
    "semaglutide": ("semaglutide",),
    "any_glp1": tuple(AGENTS),
    "non_semaglutide": ("liraglutide", "dulaglutide", "tirzepatide"),
    "incretin_potent": ("semaglutide", "tirzepatide"),
}


# Taxonomic levels MetaPhlAn provides. The published analysis used genus only, and only a
# hand-picked subset of genera; every level is carried here because a compositional shift can
# be invisible at one rank and obvious at another -- a phylum-level Firmicutes/Bacteroidota
# move is a different claim from a single genus moving, and neither implies the other.
MICROBIOME_LEVELS = ("phylum", "class", "order", "family", "genus", "species")
# 25%, up from the published 10%. A taxon present in 1 participant in 10 contributes mostly
# zeros, and a CLR on mostly-zero columns is dominated by the pseudocount rather than by
# biology. At 25% the retained counts are phylum 11, genus 167, species 255 -- still ample,
# and every retained column has real variation to estimate on. It also cuts the multiplicity
# burden roughly in half at genus and species.
MB_PREVALENCE = 0.25

# The manuscript-scale family. Taxonomic-rank expansions are deliberately separate: adding
# hundreds of overlapping class/order/family/genus/species representations changes the
# multiplicity universe and should not happen merely because `--modalities all` was typed.
# CGM is not in it: the study records CGM essentially at baseline only (8 initiators had a
# usable pre and post CGM, 5 matched), so it cannot carry a before/after comparison. It
# never produced an estimate or entered the cross-modality family; it was dropped on
# 2026-10-05 so the paper does not count it as a modality.
PAPER_MODALITIES = [
    "anthropometrics", "blood_pressure", "sleep", "hmo", "dxa",
    "microbiome", "microbiome_function", "diet",
]


def _taxon_label(column: str) -> str:
    """'k__Bacteria|p__Firmicutes|...|g__Blautia' -> 'Blautia'. Hashed species ids pass through."""
    return str(column).split("|")[-1].split("__")[-1]


def load_microbiome_level(step26, level: str,
                          prevalence: float = MB_PREVALENCE) -> pd.DataFrame:
    """Prevalence-filtered CLR abundances at one taxonomic rank, as a long panel.

    CLR rather than raw relative abundance because the data are compositional: abundances
    sum to one, so a genuine rise in one taxon forces an apparent fall in every other, and
    per-taxon tests on proportions would report that artefact as many findings. The centring
    is over the retained columns only, which is what makes the prevalence threshold part of
    the transform rather than a filter applied afterwards.
    """
    frame, meta = step26.read_snapshot("gut_microbiome", f"gut_mb_metaphlan_{level}")
    work = step26._participant_and_stage(frame, meta)
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    matrix = work[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    retained = matrix.columns[matrix.gt(0).mean().ge(prevalence)].tolist()
    if not retained:
        raise ValueError(f"{level}: no taxon reaches {prevalence:.0%} prevalence")
    reference = matrix[retained]
    nonzero = reference.to_numpy()[reference.to_numpy() > 0]
    pseudocount = float(np.quantile(nonzero, 0.01) / 2) if nonzero.size else 1e-6
    log_reference = np.log(reference + pseudocount)
    clr = log_reference.sub(log_reference.mean(axis=1), axis=0)

    # One allocation, not one per taxon. Inserting 255 species columns into a frame in a
    # loop reallocates each time; on the 22,860 x 4,644 species table that is enough memory
    # churn to kill the process, which is how this first failed -- silently, with no
    # traceback, after the pandas fragmentation warnings.
    derived = clr.rename(columns={c: f"mb_{level}::{_taxon_label(c)}" for c in retained})
    derived = derived.assign(participant_id=work["participant_id"].to_numpy(),
                             research_stage=work["research_stage"].to_numpy())
    del matrix, reference, log_reference, clr
    logging.info("microbiome %s: %d/%d taxa at >=%.0f%% prevalence",
                 level, len(retained), len(numeric), prevalence * 100)
    return step26._to_long(derived, [c for c in derived.columns
                                     if c not in {"participant_id", "research_stage"}])


def load_microbiome_diversity(step26, prevalence: float = MB_PREVALENCE) -> pd.DataFrame:
    """Alpha diversity at genus and species: richness, Shannon, Simpson, evenness.

    Computed on the FULL matrix, not the prevalence-filtered one -- richness that counts only
    common taxa is not richness. The prevalence-filtered richness is carried alongside as its
    own endpoint because the published run reported that variant, so the two stay comparable.

    Evenness is Shannon / log(richness): it separates "fewer taxa" from "the same taxa,
    less evenly distributed", which richness and Shannon alone confound.
    """
    derived = None
    for level in ("genus", "species"):
        frame, meta = step26.read_snapshot("gut_microbiome", f"gut_mb_metaphlan_{level}")
        work = step26._participant_and_stage(frame, meta)
        numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        matrix = work[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        present = matrix.gt(0)
        richness = present.sum(axis=1)
        proportions = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
        safe = proportions.replace(0, np.nan)
        shannon = -(proportions * np.log(safe)).sum(axis=1)
        simpson = 1.0 - (proportions ** 2).sum(axis=1)
        retained = matrix.columns[present.mean().ge(prevalence)]
        block = pd.DataFrame({
            f"mb_diversity::{level}_richness": richness,
            f"mb_diversity::{level}_richness_prevalent": present[retained].sum(axis=1),
            f"mb_diversity::{level}_shannon": shannon,
            f"mb_diversity::{level}_simpson": simpson,
            f"mb_diversity::{level}_evenness": shannon / np.log(richness.where(richness > 1)),
        }, index=work.index)
        block["participant_id"] = work["participant_id"].to_numpy()
        block["research_stage"] = work["research_stage"].to_numpy()
        derived = block if derived is None else derived.join(
            block.drop(columns=["participant_id", "research_stage"]), how="outer")
    return step26._to_long(derived, [c for c in derived.columns
                                     if c not in {"participant_id", "research_stage"}])


def load_all_glp_reports() -> pd.DataFrame:
    """Every recognized GLP-1/GIP-GLP-1 row, including undated and generic-name rows."""
    module = importlib.import_module("LabData.DataLoaders.Medications10KLoader")
    data = getattr(module, "Medications10KLoader")().get_data()
    frame = (data.df if hasattr(data, "df") else data).reset_index()
    name = frame.medication.astype(str)
    recognized = pd.Series(False, index=frame.index)
    for pattern in AGENTS.values():
        recognized |= name.str.contains(pattern, case=False, regex=True)
    frame = frame[recognized].copy()
    frame["participant"] = frame.RegistrationCode.astype(str).str.replace(
        "10K_", "", case=False, regex=False
    )
    frame["report_date"] = pd.to_datetime(frame.Date, errors="coerce", utc=True)
    return frame


def build_exposure(step31, agents: tuple[str, ...] = ("semaglutide",)) -> pd.DataFrame:
    """Initiators: a dated semaglutide report, earliest one per participant.

    Brand OR generic, tirzepatide excluded. A participant with any row for another agent is
    excluded even when that row has no parseable date. That makes the default a clean
    single-agent semaglutide cohort rather than a mixture of initiators and switchers.

    Undated selected-agent reports are ineligible to anchor a pre/post split, but they still
    count when screening for another GLP agent. Selected-agent dates earlier than the
    conservative agent-specific plausibility floor are also excluded.
    """
    all_reports = load_all_glp_reports()
    all_names = all_reports.medication.astype(str)
    # Purity is screened on every row. `Start` is relevant to choosing an index date, not to
    # deciding whether a participant has ever reported a different agent.
    other_rows = pd.Series(False, index=all_reports.index)
    for agent, pattern in AGENTS.items():
        if agent not in agents:
            other_rows |= all_names.str.contains(pattern, case=False, regex=True)
    contaminated = set(all_reports.loc[other_rows, "participant"])

    reports = all_reports
    # `Start` is true on 54,883 of 54,902 medication rows, so this filters almost nothing
    # -- but `Date` is only documented as the self-reported start on START events
    # (src/aggregate_medication_stats.py:440), and anchoring exposure on a continuation
    # row would silently use a created_at fallback as if it were a start date.
    if "Start" in reports.columns:
        reports = reports[reports.Start.fillna(True).astype(bool)]
    name = reports.medication.astype(str)
    wanted = pd.Series(False, index=reports.index)
    plausible = pd.Series(False, index=reports.index)
    for agent in agents:
        agent_row = name.str.contains(AGENTS[agent], case=False, regex=True)
        wanted |= agent_row
        plausible |= agent_row & reports.report_date.ge(AGENT_PLAUSIBILITY_FLOORS[agent])
    # Anyone on a NON-selected GLP-1 is dropped rather than treated as unexposed to it: a
    # participant who takes liraglutide and later semaglutide is not a clean semaglutide
    # initiator, and leaving them in would date exposure from the wrong agent.
    selected = reports[
        wanted & plausible & reports.report_date.notna()
        & ~reports.participant.isin(contaminated)
    ]
    first = selected.groupby("participant").report_date.min().rename("start_date")
    return first.reset_index()


def pair_by_date(step47, exposure: pd.DataFrame, available: dict[str, set],
                 visit_date: pd.Series, floor_months: float,
                 stages: tuple[str, ...] = MAIN_STAGES,
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pre = last visit before the start; post = first visit at >= floor_months on drug.

    `available` is the set of stages at which the instrument in question has a usable value,
    so the same function pairs the tabular and raw arms -- the arms differ only in what they
    consider available, never in how a window is chosen.
    """
    rows, audit = [], []
    for record in exposure.itertuples(index=False):
        participant = record.participant_id
        start = record.start_date
        entry = {"participant_id": participant, "start_date": start}
        usable = []
        for stage in stages:
            if stage not in available.get(participant, set()):
                continue
            when = visit_date.get((participant, stage))
            if when is not None and pd.notna(when):
                usable.append((stage, when))
        entry["n_usable_visits"] = len(usable)
        if pd.isna(start) or not usable:
            audit.append({**entry, "outcome": "no usable sleep visit"})
            continue
        # Stage order is usually chronological but is not the estimand. Sort on the
        # measurement dates explicitly so an anomalous/repeated stage cannot make
        # `before[-1]` or `mature[0]` select the wrong visit.
        usable = sorted(usable, key=lambda item: item[1])
        before = [(s, w) for s, w in usable if w < start]
        after = [(s, w, (w - start).days / 30.44) for s, w in usable if w >= start]
        mature = sorted(
            [(s, w, m) for s, w, m in after if m >= floor_months],
            key=lambda item: item[1],
        )
        if not before:
            audit.append({**entry, "outcome": "no visit before the start date"})
            continue
        if not mature:
            audit.append({**entry, "outcome": (
                "no visit after the start date" if not after
                else f"post visit under {floor_months:g} months")})
            continue
        pre_stage, pre_date = before[-1]
        post_stage, post_date, months = mature[0]
        assert pre_date < start <= post_date
        rows.append({"RegistrationCode": participant, "pre_stage": pre_stage,
                     "post_stage": post_stage, "pre_date": pre_date,
                     "post_date": post_date, "start_date": start,
                     "months_on_drug": months})
        audit.append({**entry, "outcome": "paired", "pre_stage": pre_stage,
                      "post_stage": post_stage, "months_on_drug": round(months, 2)})
    return pd.DataFrame(rows), pd.DataFrame(audit)


def raw_visit_means(step31, needed: pd.DataFrame) -> pd.DataFrame:
    """Per-second stage means for each (participant, stage), cached by uuid.

    Cached because the extraction reads several per-night parquet files per visit and the
    control pool runs to thousands of visits; a re-run with a different floor must not pay
    for it twice. The cache is keyed by uuid, which is immutable, so it stays valid across
    cohort definitions.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / "raw_visit_means.parquet"
    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    known = set(cached.uuid) if len(cached) else set()
    todo = needed[~needed.uuid.isin(known)]
    logging.info("raw extraction: %d visits needed, %d cached, %d to read",
                 len(needed), len(needed) - len(todo), len(todo))

    records = []
    for index, row in enumerate(todo.itertuples(index=False), start=1):
        means = step31.visit_stage_means(row.uuid)
        if means is None:
            continue
        record = {"uuid": row.uuid}
        for state in ("rem", "nrem", "wake", "light", "deep"):
            record[state] = means.get(state, np.nan)
            record[f"{state}_seconds"] = means.get(f"{state}_seconds", 0)
        records.append(record)
        if index % 250 == 0:
            logging.info("  extracted %d/%d", index, len(todo))
    fresh = pd.DataFrame(records)
    combined = pd.concat([cached, fresh], ignore_index=True) if len(fresh) else cached
    if len(fresh):
        combined.to_parquet(cache_path, index=False)
    # Return only the requested visits. The cache also holds visits from earlier runs with
    # other floors or cohorts; returning those would widen control eligibility to whatever
    # happened to be extracted before, making the estimate depend on cache history.
    if combined.empty:
        return combined
    return combined[combined.uuid.isin(needed.uuid)].reset_index(drop=True)


def visits_to_extract(link: pd.DataFrame, participants: set[str],
                      windows: pd.DataFrame) -> pd.DataFrame:
    """The (participant, stage) visits the raw arm can actually use, and no others.

    Two filters, both of which pay for themselves at 0.58 s per visit. Only participants the
    TABULAR arm already found eligible can enter the raw arm at all (nesting), and only
    stages that are an end of some window an initiator realised can ever be read by
    `unit_deltas`. Without the second filter the extraction reads every stage every
    participant has, which on this cohort is thousands of visits that no estimate touches.
    """
    ends = set()
    for pre, post in windows[["pre_stage", "post_stage"]].drop_duplicates().to_numpy():
        ends |= {pre, post}
    keep = link[link.participant_id.isin(participants) & link.research_stage.isin(ends)]
    return keep.drop_duplicates(["participant_id", "research_stage"])


def raw_values_panel(step31, link: pd.DataFrame, participants: set[str]) -> pd.DataFrame:
    """A raw-channel outcome panel wearing the tabular panel's exact schema.

    This is the hinge of the whole file. Because the frame it returns is indistinguishable
    in shape and column names from `load_snapshot_panel("sleep")`, every downstream step --
    eligibility, covariate completeness, propensity, matching, deltas, bootstrap -- is the
    same code for both instruments. Anything that reimplemented those steps for raw data
    would reintroduce exactly the divergence this redesign removes.
    """
    means = raw_visit_means(step31, link)
    needed = link
    if means.empty:
        return pd.DataFrame(columns=["participant_id", "research_stage", "measurement", "value"])
    frame = needed.merge(means, on="uuid", how="inner")

    # Whole sleep: duration-weighted over the three sleep states, so it matches what the
    # summary field means rather than an unweighted average of three state means.
    sleep_seconds = frame[["rem_seconds", "light_seconds", "deep_seconds"]].sum(axis=1)
    weighted = (
        frame.rem.fillna(0) * frame.rem_seconds
        + frame.light.fillna(0) * frame.light_seconds
        + frame.deep.fillna(0) * frame.deep_seconds
    )
    frame["sleep"] = np.where(sleep_seconds > 0, weighted / sleep_seconds.replace(0, np.nan), np.nan)
    frame["rem_minus_nrem"] = frame["rem"] - frame["nrem"]

    long = frame.melt(
        id_vars=["participant_id", "research_stage"],
        value_vars=list(RAW_TO_MEASUREMENT),
        var_name="state", value_name="value",
    ).dropna(subset=["value"])
    long["measurement"] = long.state.map(RAW_TO_MEASUREMENT)
    return long[["participant_id", "research_stage", "measurement", "value"]]


def available_stages(values: pd.DataFrame) -> dict[str, set]:
    """Stages at which a participant has ANY usable value in this panel.

    Any, not a nominated measurement: `unit_deltas` drops a participant per measurement, so
    a window only has to be readable for something. Keying on one flagship column would
    discard participants whose other outcomes in the same panel are perfectly usable.
    """
    return values.groupby("participant_id").research_stage.apply(set).to_dict()


def estimate(step26, values: pd.DataFrame, windows: pd.DataFrame, covariates: pd.DataFrame,
             excluded: set[str], instrument: str, exact_cols: list[str]) -> dict:
    """Eligibility -> propensity -> 1:1 no-reuse match -> per-state DiD. All reused code."""
    from importlib import import_module
    matching = import_module("research_utils.analysis.models")

    units = step26.prepare_units(values, windows, covariates, excluded,
                                 audit_label=f"{instrument}:{'+'.join(exact_cols)}")
    if units.empty or units.treated.sum() == 0:
        return {}
    if "age_band" in exact_cols:
        units = units.assign(
            age_band=(units.age // AGE_BAND_YEARS * AGE_BAND_YEARS).astype("Int64"))
    pairs = matching.optimal_pair_match(
        units, covariate_cols=step26.MATCH_COVARIATES, treatment_col="treated",
        id_col="participant_id", exact_cols=exact_cols, caliper_sd=CALIPER_SD, seed=SEED,
    )
    deltas = step26.unit_deltas(values, units)
    rows = step26.pair_effects(pairs, deltas)
    result = step26.summarize_primary(rows, instrument)
    balance = step26.balance_table(units, pairs, instrument)
    eligible = set(units.loc[units.treated.eq(1), "participant_id"])
    matched = set(pairs.treated_id.astype(str)) if len(pairs) else set()
    eligible_treated = len(eligible)
    retention = len(matched) / eligible_treated if eligible_treated else 0.0
    max_abs_smd = (float(balance.smd_after.abs().max())
                   if len(balance) else np.nan)
    payload = {
        "units": units, "pairs": pairs, "rows": rows, "balance": balance,
        "result": result, "eligible": eligible, "matched": matched,
        "retention": retention, "max_abs_smd": max_abs_smd,
    }
    # No measurement filter: every panel keeps all of its own outcomes. Filtering to the four
    # sleep heart-rate columns was correct while this was a sleep-only step and silently
    # empties every other modality. `summarize_primary` returns an empty frame when a panel
    # has too few pairs for the bootstrap (MIN_N), which CGM does at 9 initiators, so the
    # column may legitimately be absent -- hence the guard rather than an assumption.
    if result.empty or "measurement" not in result.columns:
        logging.warning("%s: no estimable measurements (%d matched pairs)",
                        instrument, len(pairs))
        return payload
    # The prespecified design gate, same definition as step 26 (:1585): retention, balance
    # and caliper together. Recorded per modality rather than enforced -- a modality that
    # fails is reported as failing, which is the only way the HMO arm's 0.285 stays visible
    # instead of being quietly dropped or tuned until it passes.
    result = result.copy()
    result["design_gate_pass"] = bool(
        retention >= step26.MIN_MATCH_RETENTION and max_abs_smd < step26.MAX_ABS_SMD)
    payload["result"] = result
    return payload


def funnel_rows(instrument: str, exposure: pd.DataFrame, audit: pd.DataFrame,
                windows: pd.DataFrame, outcome: dict) -> list[dict]:
    reasons = audit.outcome.value_counts().to_dict() if len(audit) else {}
    return [
        {"instrument": instrument, "stage": "1 dated semaglutide report",
         "n": len(exposure), "note": ""},
        {"instrument": instrument, "stage": "2 pairable (pre before start, post past floor)",
         "n": len(windows),
         "note": "; ".join(f"{k}: {v}" for k, v in reasons.items() if k != "paired")},
        {"instrument": instrument, "stage": "3 outcome + covariate eligible",
         "n": len(outcome.get("eligible", ())), "note": "prepare_units"},
        {"instrument": instrument, "stage": "4 matched 1:1",
         "n": len(outcome.get("matched", ())), "note": "optimal_pair_match"},
    ]


def timing_diagnostics(instrument: str, pairs: pd.DataFrame, windows: pd.DataFrame,
                       visit_date: pd.Series) -> dict:
    """Aggregate calendar/follow-up diagnostics without writing participant identifiers."""
    treated_windows = windows.drop_duplicates("RegistrationCode").set_index(
        "RegistrationCode"
    )
    rows = []
    for pair in pairs.itertuples(index=False):
        if pair.treated_id not in treated_windows.index:
            continue
        treated = treated_windows.loc[pair.treated_id]
        control_pre = visit_date.get((pair.control_id, pair.pre_stage))
        control_post = visit_date.get((pair.control_id, pair.post_stage))
        if pd.isna(control_pre) or pd.isna(control_post):
            continue
        treated_pre = pd.Timestamp(treated.pre_date)
        treated_post = pd.Timestamp(treated.post_date)
        start = pd.Timestamp(treated.start_date)
        rows.append({
            "treated_followup_months": (treated_post - treated_pre).days / 30.44,
            "control_followup_months": (control_post - control_pre).days / 30.44,
            "pre_calendar_difference_days": (treated_pre - control_pre).days,
            "post_calendar_difference_days": (treated_post - control_post).days,
            "pre_calendar_gap_days": abs((treated_pre - control_pre).days),
            "post_calendar_gap_days": abs((treated_post - control_post).days),
            "pre_to_start_months": (start - treated_pre).days / 30.44,
            "start_to_post_months": (treated_post - start).days / 30.44,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"modality": instrument, "n_pairs_with_dates": 0}
    treated_followup = frame.treated_followup_months.to_numpy(float)
    control_followup = frame.control_followup_months.to_numpy(float)
    pooled_sd = np.sqrt(
        (np.var(treated_followup, ddof=1) + np.var(control_followup, ddof=1)) / 2
    )
    followup_smd = (
        (treated_followup.mean() - control_followup.mean()) / pooled_sd
        if np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan
    )
    return {
        "modality": instrument,
        "n_pairs_with_dates": int(len(frame)),
        "treated_followup_median_months": frame.treated_followup_months.median(),
        "control_followup_median_months": frame.control_followup_months.median(),
        "followup_duration_smd": followup_smd,
        "paired_followup_difference_mean_months": (
            frame.treated_followup_months - frame.control_followup_months
        ).mean(),
        "pre_calendar_difference_median_days": (
            frame.pre_calendar_difference_days.median()
        ),
        "post_calendar_difference_median_days": (
            frame.post_calendar_difference_days.median()
        ),
        "absolute_pre_calendar_gap_median_days": frame.pre_calendar_gap_days.median(),
        "absolute_post_calendar_gap_median_days": frame.post_calendar_gap_days.median(),
        "pre_to_start_median_months": frame.pre_to_start_months.median(),
        "start_to_post_median_months": frame.start_to_post_months.median(),
    }


def run_instrument(step26, step31, step47, instrument: str, values: pd.DataFrame,
                   exposure: pd.DataFrame, visit_date: pd.Series, covariates: pd.DataFrame,
                   excluded: set[str], floor: float, blocking: str,
                   windows: pd.DataFrame | None = None,
                   stages: tuple[str, ...] = MAIN_STAGES,
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # `windows` supplied means "use this cohort's pre/post visits verbatim" -- the raw arm
    # passes the tabular arm's, which is what makes the two readouts share a cohort instead
    # of merely overlapping. Only the tabular arm derives windows from dates.
    if windows is None:
        windows, audit = pair_by_date(
            step47, exposure, available_stages(values), visit_date, floor, stages)
    else:
        have = available_stages(values)
        keep = [
            row.pre_stage in have.get(row.RegistrationCode, set())
            and row.post_stage in have.get(row.RegistrationCode, set())
            for row in windows.itertuples(index=False)
        ]
        audit = pd.DataFrame({
            "participant_id": windows.RegistrationCode,
            "outcome": np.where(keep, "paired", "no usable value at the inherited window"),
        })
        windows = windows[keep]
    logging.info("%s: %d initiators paired", instrument, len(windows))
    if windows.empty:
        return {}, windows, audit
    outcome = estimate(step26, values, windows, covariates, excluded,
                       instrument, EXACT_BLOCKS[blocking])
    return outcome, windows, audit


def gated_hierarchical_multiplicity(step26, results: pd.DataFrame) -> pd.DataFrame:
    """Run outcome multiplicity only over modalities that passed the design gate.

    Retention and balance are outcome-blind design criteria. A failing modality stays in
    the exported result table, but it cannot be called a discovery and must not alter the
    across-family BH correction for the claim-eligible universe.
    """
    result = results.reset_index(drop=True).copy()
    result["modality_gate_p"] = np.nan
    result["modality_gate_fdr"] = np.nan
    result["hierarchical_significant"] = False
    eligible = result.design_gate_pass.fillna(False).astype(bool)
    if eligible.any():
        tested = step26.add_hierarchical_multiplicity(
            result.loc[eligible].copy(), p_column="paired_t_p"
        )
        for column in ("modality_gate_p", "modality_gate_fdr", "hierarchical_significant"):
            result.loc[tested.index, column] = tested[column]
    result["claim_eligible"] = (
        eligible & result.hierarchical_significant.fillna(False).astype(bool)
    )
    return result


def run_directory(args, modality_label: str) -> Path:
    floor = f"{args.floor_months:g}".replace(".", "p")
    raw_label = "with_raw" if not args.skip_raw else "tabular"
    key = "__".join([
        args.exposure, f"floor_{floor}m", f"stages_{args.stages}",
        f"blocks_{args.blocking}", f"modalities_{modality_label}", raw_label,
    ])
    return OUT / "runs" / key


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-months", type=float, default=FLOOR_MONTHS)
    parser.add_argument("--blocking", choices=sorted(EXACT_BLOCKS),
                        default=DEFAULT_BLOCKING)
    parser.add_argument("--skip-raw", action="store_true",
                        help="tabular only; the raw extraction is the slow part")
    parser.add_argument("--sweep", action="store_true",
                        help="re-run the tabular arm at 0/2/3/4/6 months")
    parser.add_argument("--exposure", choices=sorted(EXPOSURE_SETS),
                        default="semaglutide")
    parser.add_argument("--stages", choices=sorted(STAGE_SETS), default="main",
                        help="'all' admits interim visits (03_01, 01_01, 05_01, ...)")
    parser.add_argument(
        "--modalities", default="sleep",
        help=("comma-separated; 'paper' for the prespecified manuscript family; "
              "or 'all' for the manuscript family plus exploratory microbiome ranks"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    step26 = _load("step26", "26_paper_reanalysis.py")
    stages = STAGE_SETS[args.stages]
    # Step 26 filters every panel to its own SUPPORTED_STAGES inside `_participant_and_stage`
    # (:476), so interim visits are dropped before this file ever sees them. Setting the
    # constant is deliberate: it is the module's stage configuration, and reusing its loader
    # keeps the per-modality QC (duration floors, quality gates) byte-identical to the
    # published analysis. Re-implementing the loader to widen the stage set would fork that
    # QC, which is the failure mode this whole redesign exists to remove.
    step26.SUPPORTED_STAGES = tuple(stages)
    step31 = _load("step31", "31_raw_sleep_glp_cohort.py")
    step47 = _load("step47", "47_date_anchored_pairing.py")

    exposure = build_exposure(step31, EXPOSURE_SETS[args.exposure])
    exposure["participant_id"] = as_registration_code(
        exposure.participant, step26).to_numpy()
    covariates = step26.load_covariates()
    all_glp = load_all_glp_reports()
    excluded = set(as_registration_code(
        pd.Series(sorted(all_glp.participant.unique())), step26))
    # The sweep runs on sleep only; the per-modality loop below loads its own panels.
    sleep_values = step26.load_snapshot_panel("sleep")
    sleep_dates = visit_dates_for(step26, "sleep")
    logging.info("initiators with a dated semaglutide report: %d", len(exposure))
    logging.info("never-GLP exclusion set: %d", len(excluded))

    if args.sweep:
        output_dir = run_directory(args, "sleep_sweep")
        output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for floor in (0.0, 2.0, 3.0, 4.0, 6.0):
            outcome, windows, _ = run_instrument(
                step26, step31, step47, "sleep", sleep_values, exposure,
                sleep_dates, covariates, excluded, floor, args.blocking, stages=stages)
            rem = outcome["result"].set_index("measurement").loc[
                "heart_rate_mean_during_rem"] if outcome else None
            records.append({
                "floor_months": floor, "paired": len(windows),
                "eligible": len(outcome.get("eligible", ())),
                "matched": len(outcome.get("matched", ())),
                "rem_did": None if rem is None else round(float(rem.effect), 2),
                "rem_ci_low": None if rem is None else round(float(rem.ci_low), 2),
                "rem_ci_high": None if rem is None else round(float(rem.ci_high), 2),
                "max_abs_smd": None if not outcome else round(
                    float(outcome["balance"].smd_after.abs().max()), 3),
            })
        sweep = pd.DataFrame(records)
        sweep.to_csv(output_dir / "floor_sweep.csv", index=False)
        print("\n=== exposure-floor sweep, tabular arm, unified spec ===")
        print(sweep.to_string(index=False))
        print(f"\nwrote {output_dir}")
        return 0

    floor = args.floor_months
    all_modalities = (PAPER_MODALITIES
                      + [f"mb_{level}" for level in MICROBIOME_LEVELS]
                      + ["mb_diversity"])
    if args.modalities == "paper":
        modalities = PAPER_MODALITIES
        modality_label = "paper"
    elif args.modalities == "all":
        modalities = all_modalities
        modality_label = "all"
    else:
        modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
        readable = "-".join(modalities)
        modality_label = (readable if len(readable) <= 80 else
                          f"custom_{hashlib.sha256(readable.encode()).hexdigest()[:10]}")
    output_dir = run_directory(args, modality_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = canonical_visit_dates(step26)
    logging.info("visit calendar: %d (participant, stage) dates", len(canonical))
    published = pd.read_csv(RUN / "outputs" / "paper_reanalysis" / "primary_1to1_results.csv")

    funnel, comparisons, results, statuses, timing = [], [], {}, [], []
    for modality in modalities:
        logging.info("---- %s ----", modality)
        try:
            values = load_panel(step26, modality, exposure)
            visit_date = dates_for(step26, modality, canonical)
        except Exception as error:                   # noqa: BLE001
            logging.warning("%s: unavailable (%s)", modality, error)
            statuses.append({"modality": modality, "status": "unavailable",
                             "pairable": np.nan, "eligible_treated": np.nan,
                             "matched": np.nan, "retention": np.nan,
                             "max_abs_smd": np.nan, "design_gate_pass": False,
                             "detail": str(error)})
            continue
        outcome, windows, audit = run_instrument(
            step26, step31, step47, modality, values, exposure,
            visit_date, covariates, excluded, floor, args.blocking, stages=stages)
        if not outcome:
            logging.warning("%s: no outcome-eligible treated units", modality)
            statuses.append({"modality": modality, "status": "not_eligible",
                             "pairable": len(windows), "eligible_treated": 0,
                             "matched": 0, "retention": 0.0,
                             "max_abs_smd": np.nan, "design_gate_pass": False,
                             "detail": "no outcome-eligible treated units"})
            continue
        if outcome["result"].empty:
            logging.warning("%s: no estimable endpoint", modality)
            funnel += funnel_rows(modality, exposure, audit, windows, outcome)
            timing.append(timing_diagnostics(
                modality, outcome["pairs"], windows, visit_date))
            outcome["balance"].to_csv(
                output_dir / f"balance_{modality}.csv", index=False)
            statuses.append({
                "modality": modality, "status": "not_estimable",
                "pairable": len(windows), "eligible_treated": len(outcome["eligible"]),
                "matched": len(outcome["matched"]),
                "retention": round(outcome["retention"], 3),
                "max_abs_smd": round(outcome["max_abs_smd"], 3),
                "design_gate_pass": False,
                "detail": "fewer than the minimum pairs for endpoint estimation",
            })
            continue
        statuses.append({"modality": modality, "status": "estimated",
                         "pairable": len(windows),
                         "eligible_treated": len(outcome["eligible"]),
                         "matched": len(outcome["matched"]),
                         "retention": round(outcome["retention"], 3),
                         "max_abs_smd": round(outcome["max_abs_smd"], 3),
                         "design_gate_pass": bool(
                             outcome["result"].design_gate_pass.iloc[0]),
                         "detail": ""})
        results[modality] = outcome
        funnel += funnel_rows(modality, exposure, audit, windows, outcome)
        timing.append(timing_diagnostics(
            modality, outcome["pairs"], windows, visit_date))
        outcome["result"].to_csv(output_dir / f"did_{modality}.csv", index=False)
        outcome["balance"].to_csv(output_dir / f"balance_{modality}.csv", index=False)
        if modality == "sleep":
            tabular, tab_windows = outcome, windows
            audit.drop(columns=["participant_id"]).to_csv(
                output_dir / "pairing_audit_sleep.csv", index=False)

        old = published[published.modality.eq(modality)].set_index("measurement")
        for row in outcome["result"].itertuples(index=False):
            ref = old.loc[row.measurement] if row.measurement in old.index else None
            comparisons.append({
                "modality": modality, "measurement": row.measurement,
                "published_effect": None if ref is None else round(float(ref.effect), 3),
                "published_n": None if ref is None else int(ref.n_pairs),
                "unified_effect": round(float(row.effect), 3),
                "unified_ci_low": round(float(row.ci_low), 3),
                "unified_ci_high": round(float(row.ci_high), 3),
                "unified_n": int(row.n_pairs),
                "max_abs_smd": round(float(outcome["balance"].smd_after.abs().max()), 3),
            })

    raw = {}
    if not args.skip_raw and "sleep" in results:
        link = step31.load_sleep_visits()
        link["participant_id"] = as_registration_code(link.participant, step26).to_numpy()
        link["research_stage"] = link.research_stage.map(canonical_stage)
        wanted = set(tabular["units"].participant_id)
        inherited = tab_windows[tab_windows.RegistrationCode.isin(tabular["matched"])]
        needed = visits_to_extract(link, wanted, inherited)
        logging.info("raw extraction scope: %d participants, %d visits (~%.0f min)",
                     needed.participant_id.nunique(), len(needed), len(needed) * 0.58 / 60)
        raw_values = raw_values_panel(step31, needed, wanted)
        raw, raw_windows, raw_audit = run_instrument(
            step26, step31, step47, "sleep_raw", raw_values, exposure,
            visit_date, covariates, excluded, floor, args.blocking,
            windows=inherited, stages=stages)
        if raw:
            funnel += funnel_rows("sleep_raw", exposure, raw_audit, raw_windows, raw)
            raw["result"].to_csv(output_dir / "did_sleep_raw.csv", index=False)
            raw["balance"].to_csv(output_dir / "balance_sleep_raw.csv", index=False)

    # Cross-modality multiplicity, which only exists once every modality is in one frame:
    # Simes within each modality, then BH across modalities, gating the within-modality FDR
    # (step 26:1153). Running it per modality would not be the same test.
    combined = [outcome["result"] for outcome in results.values()]
    if raw:
        combined.append(raw["result"])
    if not combined:
        raise RuntimeError("No modality produced an estimable result")
    primary = gated_hierarchical_multiplicity(
        step26, pd.concat(combined, ignore_index=True))
    primary.to_csv(output_dir / "primary_results.csv", index=False)

    gates = pd.DataFrame([
        {"modality": name, "eligible_treated": len(o["eligible"]),
         "matched": len(o["matched"]), "retention": round(o["retention"], 3),
         "max_abs_smd": round(o["max_abs_smd"], 3),
         "design_gate_pass": bool(o["result"].design_gate_pass.iloc[0])}
        for name, o in ({**results, **({"sleep_raw": raw} if raw else {})}).items()
    ])
    gates.to_csv(output_dir / "design_gates.csv", index=False)
    print("\n=== design gates (retention >= 0.85 and max |SMD| < 0.10) ===")
    print(gates.to_string(index=False))
    survivors = primary[primary.claim_eligible]
    print(f"\n=== claim-eligible multiplicity: {len(survivors)} of {len(primary)} endpoints "
          f"pass their modality's design gate and survive hierarchical correction ===")
    if len(survivors):
        print(survivors.groupby("modality").measurement.count().to_string())
    else:
        print("none")

    pd.DataFrame(funnel).to_csv(output_dir / "funnel.csv", index=False)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output_dir / "comparison_to_published.csv", index=False)
    pd.DataFrame(statuses).to_csv(output_dir / "modality_status.csv", index=False)
    pd.DataFrame(timing).to_csv(output_dir / "timing_diagnostics.csv", index=False)

    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "script": str(Path(__file__).relative_to(RUN)),
        "script_sha256": file_sha256(Path(__file__)),
        "arguments": vars(args),
        "requested_modalities": modalities,
        "completed_modalities": list(results) + (["sleep_raw"] if raw else []),
        "modality_status": statuses,
        "dated_clean_initiators": int(len(exposure)),
        "never_glp_exclusion_set": int(len(excluded)),
        "n_endpoints": int(len(primary)),
        "n_claim_eligible": int(primary.claim_eligible.sum()),
        "design_passing_modalities": gates.loc[gates.design_gate_pass, "modality"].tolist(),
        "design_failing_modalities": gates.loc[~gates.design_gate_pass, "modality"].tolist(),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\n=== unified spec, {floor:g}-month floor, {args.stages} stages, "
          f"{args.blocking} blocking ===")
    print(pd.DataFrame(funnel)[["instrument", "stage", "n"]].to_string(index=False))
    print("\n=== published (label pairing) vs unified (date pairing), all modalities ===")
    show = comparison.copy()
    show["measurement"] = show.measurement.str.slice(0, 38)
    concise = show[~show.modality.eq("microbiome_function")]
    print(concise.to_string(index=False))
    omitted = len(show) - len(concise)
    if omitted:
        print(f"... {omitted} microbiome-function endpoints are in comparison_to_published.csv")
    if raw:
        assert raw["matched"] <= results["sleep"]["matched"], (
            f"raw treated set is not nested: "
            f"{len(raw['matched'] - results['sleep']['matched'])} raw-only")
        print(f"\nnesting holds: {len(raw['matched'])} raw initiators subset of "
              f"{len(results['sleep']['matched'])} tabular")
    print(f"\nwrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
