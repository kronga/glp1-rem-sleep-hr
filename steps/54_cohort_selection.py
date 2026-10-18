"""Cohort selection for the sleep analysis: upstream exposure funnel, control branch, and
initiators with versus without paired sleep studies.

Reuses the step-48 definitions through the ``Pipeline`` of
``steps/51_unified_sleep_robustness.py`` (the rebuilt sleep match is checked against the
versioned step-48 run), and recounts each step of the exposure funnel with the rules of
``steps/48_unified_cohort.py::build_exposure``:

1. Initiator funnel: participants with any semaglutide record -> no record of another GLP-1
   drug -> a dated semaglutide start on or after the plausibility floor (494) -> the sleep
   pairing losses by reason (step 48 pairing audit) -> 96 paired. Some medication reports
   come from participants with no dated study visit; they are lost at the pairing step.
2. Control funnel: HPP participants with a dated study visit -> no GLP-1 record of any kind
   -> sleep outcomes at both visits of an initiator's visit pair and complete matching
   covariates (unique participants) -> matched controls.
3. Baseline characteristics of the 494 initiators with and without paired sleep studies,
   at the last visit before the recorded start with the matching covariates measured.

Outputs (aggregate only, no identifiers): ``outputs/cohort_selection/``

    python steps/54_cohort_selection.py
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "outputs" / "cohort_selection"

# Step-48 pairing-audit outcome -> wording in the paper, in the order the audit tests them.
PAIRING_LOSSES = {
    "no usable sleep visit": "no usable sleep study at any visit",
    "no visit before the start date": "no sleep study before the start",
    "no visit after the start date": "no sleep study after the start",
    "post visit under 3 months": "follow-up sleep study <3 months after the start",
}
CONTINUOUS = {"age": "Age, years", "bmi": "Body-mass index, kg/m²",
              "waist_circumference": "Waist circumference, cm",
              "start_year": "Year of recorded start"}
BINARY = {"gender": "Female sex", "diabetic": "Diabetes", "smoking": "Current smoking",
          "on_antihypertensive": "Antihypertensive therapy",
          "on_lipid_lowering": "Lipid-lowering therapy"}
REQUIRED = ["age", "gender", "bmi"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initiator_funnel(pipe, sleep: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each step of the semaglutide exposure funnel, recounted with build_exposure's rules."""
    step48 = pipe.step48
    reports = step48.load_all_glp_reports()
    reports["participant_id"] = step48.as_registration_code(reports.participant).to_numpy()
    name = reports.medication.astype(str)
    is_semaglutide = name.str.contains(step48.AGENTS["semaglutide"], case=False, regex=True)
    is_other = pd.Series(False, index=reports.index)
    for agent, pattern in step48.AGENTS.items():
        if agent != "semaglutide":
            is_other |= name.str.contains(pattern, case=False, regex=True)

    any_semaglutide = set(reports.loc[is_semaglutide, "participant_id"])
    single_agent = any_semaglutide - set(reports.loc[is_other, "participant_id"])
    starts = reports[is_semaglutide & reports.participant_id.isin(single_agent)]
    if "Start" in starts.columns:
        starts = starts[starts.Start.fillna(True).astype(bool)]
    floor = step48.AGENT_PLAUSIBILITY_FLOORS["semaglutide"]
    dated = starts[starts.report_date.notna() & starts.report_date.ge(floor)]
    initiators = set(dated.participant_id)

    expected = set(pipe.exposure("semaglutide").participant_id)
    if initiators != expected:
        raise AssertionError(f"recounted initiators ({len(initiators)}) differ from step 48 "
                             f"({len(expected)})")
    audit = sleep["audit"]
    if len(audit) != len(expected):
        raise AssertionError("the pairing audit does not cover every initiator")
    paired = int(audit.outcome.eq("paired").sum())
    if paired != len(sleep["windows"]):
        raise AssertionError("pairing audit and sleep windows disagree")

    rows = [
        {"step": "any_semaglutide", "label": "participants with any semaglutide record",
         "n": len(any_semaglutide)},
        {"step": "single_agent", "label": "with no record of another GLP-1 receptor agonist",
         "n": len(single_agent), "removed_reason": "record of another GLP-1 receptor agonist"},
        {"step": "initiators", "label": "with a plausible recorded start date",
         "n": len(initiators), "removed_reason": "no start date, or one before 2018"},
        {"step": "sleep_paired",
         "label": "with a sleep study before and ≥3 months after the start",
         "n": paired, "removed_reason": "no paired sleep studies"},
    ]
    funnel = pd.DataFrame(rows)
    funnel["removed"] = (funnel.n.shift(1) - funnel.n).astype("Int64")

    losses = audit.loc[audit.outcome.ne("paired"), "outcome"].value_counts()
    unknown = set(losses.index) - set(PAIRING_LOSSES)
    if unknown:
        raise AssertionError(f"unexpected pairing-audit outcomes: {sorted(unknown)}")
    pairing_losses = pd.DataFrame([
        {"audit_outcome": outcome, "reason": label, "n": int(losses.get(outcome, 0))}
        for outcome, label in PAIRING_LOSSES.items()
    ])
    if pairing_losses.n.sum() != len(initiators) - paired:
        raise AssertionError("pairing losses do not add up")
    return funnel, pairing_losses


def control_funnel(pipe, visit_participants: set[str], sleep: dict) -> pd.DataFrame:
    never_glp1 = visit_participants - pipe.excluded
    units = sleep["units"]
    pool = set(units.loc[units.treated.eq(0), "participant_id"])
    if not pool <= never_glp1:
        raise AssertionError("the control pool holds a participant outside the never-GLP-1 set")
    matched = sleep["pairs"].control_id.nunique()
    funnel = pd.DataFrame([
        {"step": "visit", "label": "HPP participants with a dated study visit",
         "n": len(visit_participants)},
        {"step": "never_glp1", "label": "with no GLP-1 record of any kind",
         "n": len(never_glp1), "removed_reason": "any GLP-1 record"},
        {"step": "sleep_pool",
         "label": "with sleep studies at both visits of an initiator's visit pair "
                  "and complete matching covariates", "n": len(pool),
         "removed_reason": "no sleep studies at an initiator's visit pair, or incomplete covariates"},
        {"step": "matched", "label": "matched controls", "n": int(matched),
         "removed_reason": "not selected by the match"},
    ])
    funnel["removed"] = (funnel.n.shift(1) - funnel.n).astype("Int64")
    return funnel


def baseline_characteristics(pipe, sleep: dict) -> pd.DataFrame:
    """Initiators with vs without paired sleep studies, at the last pre-start visit."""
    exposure = pipe.exposure("semaglutide")[["participant_id", "start_date"]]
    covariates = pipe.covariates.dropna(subset=REQUIRED)
    dates = pipe.canonical_dates.rename("visit_date").reset_index()
    dates.columns = ["participant_id", "research_stage", "visit_date"]
    visits = covariates.merge(dates, on=["participant_id", "research_stage"], how="inner")
    visits = visits.merge(exposure, on="participant_id", how="inner")
    baseline = (visits[visits.visit_date < visits.start_date]
                .sort_values("visit_date").groupby("participant_id").tail(1))
    baseline = exposure.merge(baseline.drop(columns="start_date"), on="participant_id",
                              how="left")
    baseline["start_year"] = baseline.start_date.dt.year + baseline.start_date.dt.dayofyear / 366
    baseline["paired"] = baseline.participant_id.isin(set(sleep["windows"].RegistrationCode))

    records = [{"characteristic": "Initiators", "paired_value": int(baseline.paired.sum()),
                "unpaired_value": int((~baseline.paired).sum()), "smd": np.nan}]
    measured = baseline.dropna(subset=REQUIRED)
    groups = {True: measured[measured.paired], False: measured[~measured.paired]}
    records.append({"characteristic": "With covariates measured before the start",
                    "paired_value": len(groups[True]), "unpaired_value": len(groups[False]),
                    "smd": np.nan})
    for column, label in CONTINUOUS.items():
        a, b = groups[True][column].dropna(), groups[False][column].dropna()
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        records.append({"characteristic": label, "kind": "continuous",
                        "paired_mean": a.mean(), "paired_sd": a.std(ddof=1),
                        "unpaired_mean": b.mean(), "unpaired_sd": b.std(ddof=1),
                        "smd": (a.mean() - b.mean()) / pooled})
    for column, label in BINARY.items():
        a, b = groups[True][column].fillna(0).eq(1), groups[False][column].fillna(0).eq(1)
        pa, pb = a.mean(), b.mean()
        spread = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
        records.append({"characteristic": label, "kind": "binary",
                        "paired_n": int(a.sum()), "paired_pct": 100 * pa,
                        "unpaired_n": int(b.sum()), "unpaired_pct": 100 * pb,
                        "smd": (pa - pb) / spread if spread > 0 else 0.0})
    return pd.DataFrame(records)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    step51 = load_module("selection_step51", RUN / "steps/51_unified_sleep_robustness.py")
    pipe = step51.Pipeline()
    sleep = pipe.rebuild("sleep", "semaglutide", step51.FINAL_RUN)
    visit_participants = set(pipe.canonical_dates.index.get_level_values(0))

    funnel, pairing_losses = initiator_funnel(pipe, sleep)
    controls = control_funnel(pipe, visit_participants, sleep)
    characteristics = baseline_characteristics(pipe, sleep)
    if int(controls.set_index("step").n["matched"]) != len(sleep["pairs"]):
        raise AssertionError("matched controls differ from matched pairs")

    funnel.to_csv(OUT / "initiator_funnel.csv", index=False)
    pairing_losses.to_csv(OUT / "sleep_pairing_losses.csv", index=False)
    controls.to_csv(OUT / "control_funnel.csv", index=False)
    characteristics.to_csv(OUT / "paired_vs_unpaired_initiators.csv", index=False)
    for frame in (funnel, pairing_losses, controls):
        logging.info("\n%s", frame.to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
