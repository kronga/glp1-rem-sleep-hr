"""Date-anchored pre/post pairing for the sleep DiD, beside the published visit-label one.

Writes only to `outputs/date_anchored_pairing/`. No figure, table, or manuscript text is
touched -- this is a parallel estimate, produced so the size of the move can be read before
anyone decides whether the primary should change.

What the published analysis does, and the two leaks it has
----------------------------------------------------------
`steps/26` pairs each initiator on the two VISIT LABELS carried by their strict
visit-reported medication transition: `pre_stage = prior_absent_visit_stage`,
`post_stage = confirmation_visit_stage` (26:252). `unit_deltas` (26:860) then reads the
outcome at exactly those two labels and drops the participant if either is absent. The
labels are also the exact-blocking keys, so the control pool is built from the windows the
treated actually use (26:720) -- that is what makes study period a matched confounder.

Two consequences, both measured on the published 44 matched initiators before this was
written:

1. THE PRE VISIT NEED NOT PRECEDE THE DRUG. `enforce_prior_visit_dates`
   (src/aggregate_medication_stats.py:488) accepts a start date up to `window_months = 3`
   BEFORE the prior-absent visit; only a larger gap forces a re-pair or an exclusion.
   2 of the 44 were already taking semaglutide at their own "pre" visit, by 2 and 61 days.
   A contaminated baseline attenuates the DiD.
2. THE POST VISIT IS WHEREVER CONFIRMATION LANDED. It carries no exposure-duration floor:
   across the 139 semaglutide candidates, 37% of confirmations sit under 6 months, i.e.
   inside the 0.25 -> 0.5 -> 1.0 mg titration rather than at maintenance.

Neither is a coding error. Both follow from anchoring on a report rather than on a date,
and "prior_absent_visit_stage" should not be read as confirmed absence in any case -- a
participant does not report NOT taking a drug, the drug is simply absent from that visit's
list.

What this step does instead
---------------------------
Same machinery, one input changed. `analyse_modality` takes the pairing as its `exposure`
frame, so replacing that frame re-derives control windows, propensity, exact blocking,
1:1 no-reuse matching and the bootstrap consistently -- nothing here reimplements the
estimator, which is the only reason the two numbers are comparable at all.

    pre  = latest visit with a usable sleep outcome STRICTLY BEFORE medication_start_date
    post = earliest visit with a usable outcome at >= MIN_EXPOSURE_MONTHS on drug

Both leaks close by construction: the pre visit precedes exposure, and the post visit
clears the titration window. The cost is that pairing is now anchored on a self-reported
start date, so `MEDICATION_DATE_AUDIT.md` applies with full force -- the arm is only as
good as those dates. That is the trade this step exists to price, not to settle.

It also recovers initiators the label rule cannot reach: a participant who skipped the
exact visit their transition names is dropped by `unit_deltas` even with clean studies on
both sides of their start date. 5 of the 7 initiators that Figure 1c draws but Figure 1d
does not are exactly this case.

The 2x2 this completes
----------------------
Pairing rule x instrument. Two cells are published, two are not:

                      visit-label pairing        date pairing
    tabular           Fig 1d, +6.97, n=41        <- this step
    raw channels      <- this step, section 2    Fig 1c, +4.22, n=33

Section 2 answers the same question on the raw side: which raw-cohort participants would
survive the label rule, and does the 33 become a subset of the matched set once the
tabular arm is re-paired.

Usage
-----
    PY=/net/mraid20/export/jasmine/david/anaconda3/bin/python3
    $PY steps/47_date_anchored_pairing.py
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "outputs" / "date_anchored_pairing"
SNAPSHOT_SLEEP = Path(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/"
    "snapshots/pheno_data_snapshots/out_dir/sleep/sleep_hrv"
)
GOLD = Path("/net/mraid20/export/genie/LabData/Data/Pheno/gold/sleep")

# Semaglutide titrates 0.25 -> 0.5 -> 1.0 mg over ~3 months (Wegovy to 2.4 mg over 4-5),
# so a post visit earlier than this is sub-maintenance and dilutes the contrast. The same
# floor is used to pick Figure 1b's exemplar.
MIN_EXPOSURE_MONTHS = 6.0
SUPPORTED_STAGES = ("00_00_visit", "02_00_visit", "04_00_visit", "06_00_visit")


def load_step26():
    """Import `steps/26` as a module. Its `main()` is __main__-guarded, so nothing runs."""
    spec = importlib.util.spec_from_file_location(
        "step26", RUN / "steps" / "26_paper_reanalysis.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["step26"] = module
    spec.loader.exec_module(module)
    return module


def normalise_stage(stage: str) -> str:
    """'baseline' / '02 visit' / '02_00_visit' -> '02_00_visit'.

    Three spellings are in play: the transitions CSV writes '02 visit', the snapshot
    metadata writes '02_00_visit', and step 26 maps between them with
    STAGE_TO_RESEARCH_STAGE. Comparing the raw strings silently matches nothing, which
    reads as a finding rather than a bug -- so every stage goes through here.
    """
    text = str(stage).strip().lower()
    if text in ("baseline", "00_00_visit"):
        return "00_00_visit"
    match = re.match(r"^(\d{2})[ _]", text)
    return f"{match.group(1)}_00_visit" if match else text


def as_registration_code(values, step26) -> pd.Series:
    """Any of the three id spellings -> the '10K_1234567890' form the analysis keys on.

    `normalize_reg` only upper-cases (src/glp_modality_matched_analysis.py:231), which is
    all the snapshot needs because it already carries a '10k_' prefix. The raw gold link
    table and this run's raw-cohort CSVs carry BARE digits, so upper-casing leaves them
    unjoinable against the exposure frame -- and an empty join here reports "the raw
    cohort shares nobody with the matched set", which is a plausible-looking claim rather
    than an obvious failure. Hence prefixing first, normalising second.
    """
    text = pd.Series(values, dtype="string").str.strip()
    bare = text.str.fullmatch(r"\d+").fillna(False)
    text = text.mask(bare, "10K_" + text)
    return step26.normalize_reg(text)


def sleep_visit_dates(step26) -> pd.Series:
    """Earliest study date per (participant, stage) -- the visit date the pairing uses.

    The sleep study is the outcome being paired, so its own date is the right anchor:
    using the questionnaire visit date instead would pair on a day the participant may
    not have been studied.

    Ids go through step 26's own `normalize_reg` rather than a local regex. The snapshot
    writes '10k_1234567890', the exposure frame '10K_1234567890', and the raw gold link
    table a bare int64 -- three spellings of one id, and a lookup keyed on the wrong one
    returns nothing at all rather than failing, which reads as "no participant has a
    usable visit" instead of as a bug.
    """
    frame = pd.read_parquet(SNAPSHOT_SLEEP / "df.parquet", columns=["quality_score_heart_rate"])
    meta = pd.read_parquet(SNAPSHOT_SLEEP / "df_metadata.parquet", columns=["research_stage"])
    frame = frame.join(meta, how="left").reset_index()
    frame["participant_id"] = as_registration_code(frame.RegistrationCode, step26).to_numpy()
    frame["stage"] = frame.research_stage.map(normalise_stage)
    return frame.groupby(["participant_id", "stage"]).date.min()


def date_anchored_exposure(step26, published: pd.DataFrame, values: pd.DataFrame,
                           min_exposure_months: float = MIN_EXPOSURE_MONTHS,
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-pair every initiator on dates. Returns (exposure frame, per-participant audit).

    `min_exposure_months` is a SEPARATE choice from the pairing rule and the two must not
    be read as one thing. Date pairing on its own ADDS initiators the label rule drops for
    skipping the visit their transition names. The floor SUBTRACTS initiators the label
    rule keeps, because the label rule takes the confirmation visit whatever the exposure
    duration is. Run with 0 to see the pairing rule alone; 6 is the analysis worth
    reporting, since a post visit inside titration is not a treated observation.

    `values` is step 26's own sleep panel, so "has a usable outcome at this stage" means
    exactly what it means in the published analysis -- the REM/NREM duration floors and
    the quality gate are already applied there. Availability is not re-derived here, which
    is what keeps the two arms' QC identical and the comparison about pairing alone.
    """
    visit_date = sleep_visit_dates(step26)
    available = (
        values.groupby("participant_id").research_stage.apply(set).to_dict()
    )
    published = published.copy()
    published["start"] = pd.to_datetime(
        published["medication_start_date"], errors="coerce", utc=True
    )

    rows, audit = [], []
    for record in published.itertuples(index=False):
        participant = record.RegistrationCode
        stages = available.get(participant, set())
        entry = {
            "participant_id": participant,
            "label_pre": record.pre_stage,
            "label_post": record.post_stage,
            "start": record.start,
        }
        usable = []
        for stage in SUPPORTED_STAGES:
            if stage not in stages:
                continue
            when = visit_date.get((participant, stage))
            if when is not None and pd.notna(when):
                usable.append((stage, when))
        entry["n_usable_visits"] = len(usable)
        if pd.isna(record.start) or not usable:
            audit.append({**entry, "outcome": "no start date" if pd.isna(record.start)
                          else "no usable sleep visit"})
            continue
        before = [(s, w) for s, w in usable if w < record.start]
        after = [
            (s, w, (w - record.start).days / 30.44) for s, w in usable if w >= record.start
        ]
        mature = [(s, w, m) for s, w, m in after if m >= min_exposure_months]
        if not before:
            audit.append({**entry, "outcome": "no visit before the start date"})
            continue
        if not mature:
            reason = ("no visit after the start date" if not after
                      else f"post visit under {min_exposure_months:g} months")
            audit.append({**entry, "outcome": reason})
            continue
        pre_stage, pre_date = before[-1]
        post_stage, post_date, months = mature[0]
        rows.append(
            {"RegistrationCode": participant, "pre_stage": pre_stage,
             "post_stage": post_stage}
        )
        audit.append({
            **entry, "outcome": "paired", "date_pre": pre_stage, "date_post": post_stage,
            "months_on_drug_at_post": round(months, 2),
            "same_as_label": bool(pre_stage == record.pre_stage
                                  and post_stage == record.post_stage),
        })
    return pd.DataFrame(rows), pd.DataFrame(audit)


def sleep_rows(result: dict) -> pd.DataFrame:
    frame = result["primary_result"]
    return frame[frame.measurement.str.startswith("heart_rate_mean_during_")].copy()


def compare(published: pd.DataFrame, dated: pd.DataFrame) -> pd.DataFrame:
    keep = ["measurement", "n_pairs", "treated_delta", "control_delta",
            "effect", "ci_low", "ci_high", "paired_t_p"]
    left = published[keep].add_suffix("_label").rename(
        columns={"measurement_label": "measurement"})
    right = dated[keep].add_suffix("_dated").rename(
        columns={"measurement_dated": "measurement"})
    return left.merge(right, on="measurement", how="outer")


def raw_cohort_under_label_rule(step26, published_exposure: pd.DataFrame) -> pd.DataFrame:
    """Section 2: which raw-channel participants the visit-label rule would keep.

    Step 31 picks each raw pre/post night by DATE relative to the drug report, never by
    visit label, so the raw cohort is already the date cell of the 2x2. To ask what the
    label rule would do to it, the two nights it actually drew have to be resolved back to
    their research stages -- which the link table carries, so no per-second channel is
    re-read here.
    """
    import pyarrow.parquet as pq

    link = pq.read_table(
        GOLD / "sleep_all_hrv_with_participant.parquet",
        columns=["participant_id", "research_stage", "uuid", "study_date"],
    ).to_pandas(ignore_metadata=True)
    link = link[link.uuid.isin(set(os.listdir(GOLD / "timeseries")))].copy()
    link["participant_id"] = as_registration_code(
        link.participant_id.astype("int64").astype(str), step26
    ).to_numpy()
    link["stage"] = link.research_stage.map(normalise_stage)
    link["study_date"] = pd.to_datetime(link.study_date, errors="coerce", utc=True)

    remlock = pd.read_csv(
        RUN / "outputs" / "_test_figures" / "cohort_remlock_paper_strict.csv",
        dtype={"participant": str},
    )
    drawn = sorted(as_registration_code(remlock.participant, step26).unique())
    labels = published_exposure.set_index("RegistrationCode")[["pre_stage", "post_stage"]]

    rows = []
    for participant in drawn:
        visits = link[link.participant_id == participant]
        stages = set(visits.stage)
        label = labels.loc[participant] if participant in labels.index else None
        rows.append({
            "participant_id": participant,
            "raw_stages_available": ",".join(sorted(s[:2] for s in stages)),
            "label_pre": None if label is None else label.pre_stage,
            "label_post": None if label is None else label.post_stage,
            "raw_has_label_pre": None if label is None else label.pre_stage in stages,
            "raw_has_label_post": None if label is None else label.post_stage in stages,
        })
    frame = pd.DataFrame(rows)
    frame["survives_label_rule"] = (
        frame.raw_has_label_pre.fillna(False) & frame.raw_has_label_post.fillna(False)
    )
    return frame


def raw_did_under_label_rule(step26, survivors: set[str], drawn: set[str]) -> pd.DataFrame:
    """The raw cohort's own DiD, restricted to participants the label rule would keep.

    This is the fourth cell of the 2x2: raw channels x visit-label pairing. It is a
    RESTRICTION, not a re-estimation -- for these participants step 31's date-chosen pre
    and post nights already sit at their own label visits, so the cached per-visit state
    means are exactly what a label-paired raw analysis would read. Nothing is re-extracted.

    One caveat the number carries: the overlap weights were fitted on the full raw cohort
    and are NOT refit on the subset, so this is "panel c with the label-rule dropouts
    removed", not a fresh overlap-weighted design. Refitting would change the weights and
    make the contrast with panel c about two things at once.
    """
    states = pd.read_csv(
        RUN / "outputs" / "_test_figures" / "cohort_states_paper_strict.csv",
        dtype={"participant": str},
    )
    states["participant"] = as_registration_code(states.participant, step26).to_numpy()
    # `cohort_states` is written BEFORE `_test_fig1_alt`'s MIN_BOUTS_PER_VISIT floor, so it
    # carries one treated participant panel c does not draw. Restricting to `drawn` is what
    # makes the first row reproduce the published +4.22 rather than a near-miss at +4.36 --
    # a baseline row that does not reproduce its own panel is worse than no baseline row.
    states = states[(states.treated == 0) | states.participant.isin(drawn)]
    kept = states[(states.treated == 0) | states.participant.isin(survivors)]

    rows = []
    for label, frame in (("panel c as drawn", states), ("label-rule survivors", kept)):
        record = {"cohort": label,
                  "n_treated": int(frame[frame.treated == 1].participant.nunique())}
        for state in ("rem", "nrem"):
            arms = {}
            for arm in (1, 0):
                side = frame[frame.treated == arm]
                for visit in ("pre", "post"):
                    mean, _ = alt_weighted_mean(
                        side[f"{state}_{visit}"].to_numpy(),
                        side.overlap_weight.to_numpy(),
                    )
                    arms[(arm, visit)] = mean
            treated_delta = arms[(1, "post")] - arms[(1, "pre")]
            control_delta = arms[(0, "post")] - arms[(0, "pre")]
            record[f"{state}_treated_delta"] = round(treated_delta, 3)
            record[f"{state}_control_delta"] = round(control_delta, 3)
            record[f"{state}_did"] = round(treated_delta - control_delta, 3)
        record["rem_minus_nrem"] = round(record["rem_did"] - record["nrem_did"], 3)
        rows.append(record)
    return pd.DataFrame(rows)


def alt_weighted_mean(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Overlap-weighted mean and its Kish-effective-n SEM, as `_test_fig1_alt` computes it.

    Imported rather than reimplemented would be better, but that module builds figures on
    import; this is the same six lines it uses.
    """
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if not len(values):
        return float("nan"), float("nan")
    mean = float(np.average(values, weights=weights))
    n_eff = weights.sum() ** 2 / np.square(weights).sum()
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, float(np.sqrt(variance / n_eff))


def explain_raw_only(values: pd.DataFrame, raw: pd.DataFrame, audit: pd.DataFrame,
                     label_matched: set[str], dated_matched: set[str]) -> pd.DataFrame:
    """Per-participant account of every initiator Figure 1c draws but the tabular set omits.

    Written because "the raw cohort has 7 participants the tabular analysis does not" invites
    the obvious fix -- compute their tabular features from the raw channels -- and that fix
    is a no-op. Wherever a raw study exists, a QC-passing tabular row exists too: checked
    both ways over these participants, 0 visits have raw without a tabular row and 0 have a
    tabular row that fails QC where the raw is usable. The features are already there. What
    is missing is a PAIRING the estimator will accept, and for some of them a CONTROL.

    Columns say which, per participant, so the claim can be checked rather than believed:
      rem_hr_by_visit   every visit with a QC-passing tabular REM heart rate, and its value
      label_window      the two visits the published rule demands
      label_blocker     which end of that window has no sleep study
      dated_window      the two visits date pairing selects instead
      verdict           the single reason this participant is not in the tabular matched set

    Ids are replaced by opaque R-labels: `outputs/` is not gitignored in this run.
    """
    rem = values[values.measurement.eq("heart_rate_mean_during_rem")]
    by_visit = {
        participant: dict(zip(group.research_stage, group.value.round(1)))
        for participant, group in rem.groupby("participant_id")
    }
    audit = audit.set_index("participant_id")
    # The set the question is actually about: initiators Figure 1c draws that the PUBLISHED
    # tabular analysis omits. Enumerating those missing from the dated set instead answers a
    # different question and hides the recoveries, which is the whole point of the table.
    raw_only = sorted(set(raw.participant_id) - label_matched)

    rows = []
    for index, participant in enumerate(raw_only, start=1):
        visits = by_visit.get(participant, {})
        entry = audit.loc[participant] if participant in audit.index else None
        label_pre = None if entry is None else entry.label_pre
        label_post = None if entry is None else entry.label_post
        blocker = ", ".join(
            end for end, stage in (("pre", label_pre), ("post", label_post))
            if stage is not None and stage not in visits
        )
        if entry is not None and entry.outcome == "paired":
            dated_window = f"{str(entry.date_pre)[:2]}->{str(entry.date_post)[:2]}"
        else:
            dated_window = ""
        in_label = participant in label_matched
        in_dated = participant in dated_matched
        if in_dated:
            verdict = "RECOVERED by date pairing, now in the tabular matched set"
        elif blocker:
            verdict = (f"label window needs {blocker} visit, never attended"
                       + ("; date pairing recovers it" if dated_window else ""))
        elif entry is not None and entry.outcome != "paired":
            verdict = f"date pairing: {entry.outcome}"
        else:
            verdict = "eligible under both rules; lost at the matching step (no control)"
        rows.append({
            "label": f"R{index}",
            "rem_hr_by_visit": " ".join(f"{s[:2]}:{v:g}" for s, v in sorted(visits.items())),
            "n_visits_with_tabular_rem": len(visits),
            "label_window": f"{str(label_pre)[:2]}->{str(label_post)[:2]}",
            "label_blocker": blocker or "none",
            "dated_window": dated_window or "not paired",
            "in_label_matched": in_label,
            "in_dated_matched": in_dated,
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-exposure-months", type=float, default=MIN_EXPOSURE_MONTHS,
                        help="0 isolates the pairing rule; 6 is the reportable analysis")
    args = parser.parse_args()
    tag = f"_floor{args.min_exposure_months:g}"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    step26 = load_step26()

    published_exposure, _, _ = step26.load_exposure_cohorts()
    ever_glp = step26.ever_glp_participants(published_exposure)
    covariates = step26.load_covariates()
    values = step26.load_snapshot_panel("sleep")
    logging.info("published exposure: %d initiators",
                 published_exposure.RegistrationCode.nunique())

    # ---- section 1: the tabular arm, re-paired on dates -------------------------
    dated_exposure, audit = date_anchored_exposure(
        step26, published_exposure, values, args.min_exposure_months)
    audit.to_csv(OUT / f"pairing_audit{tag}.csv", index=False)
    logging.info("date-anchored exposure: %d initiators paired",
                 dated_exposure.RegistrationCode.nunique())
    print("\nwhy each initiator was or was not paired on dates:")
    print(audit.outcome.value_counts().to_string())
    paired = audit[audit.outcome == "paired"]
    same = int(paired.same_as_label.fillna(False).astype(bool).sum())
    print(f"\n  of the {len(paired)} paired, {same} land on the SAME two visits the "
          f"label rule uses; {len(paired) - same} differ")
    print(f"  months on drug at the post visit: median "
          f"{paired.months_on_drug_at_post.median():.1f} "
          f"(floor {args.min_exposure_months:g})")
    # The add/subtract decomposition. Reporting only the net count is what made the first
    # write-up of this step read as a contradiction -- "date pairing is stricter" and
    # "date pairing recovers 6 people" are both true, of different subsets.
    # NOT the add/subtract decomposition: the date frame is re-paired FROM the label frame,
    # so it is nested inside it by construction and "added = 0" is arithmetic, not a result.
    # Date pairing adds initiators further downstream, in `unit_deltas`, by naming visits
    # where the outcome actually exists -- so the comparison that answers "more or fewer
    # people" is between the two MATCHED sets, printed after the estimates below.
    print(f"\n  exposure frames (dated is re-paired from label, hence nested):"
          f"  label {published_exposure.RegistrationCode.nunique()}"
          f"  dated {dated_exposure.RegistrationCode.nunique()}")

    published_result = step26.analyse_modality(
        "sleep", values, published_exposure, covariates, ever_glp)
    dated_result = step26.analyse_modality(
        "sleep", values, dated_exposure, covariates, ever_glp)
    if not dated_result:
        print("\nno eligible treated units under date pairing -- nothing to compare")
        return 1

    side_by_side = compare(sleep_rows(published_result), sleep_rows(dated_result))
    side_by_side.to_csv(OUT / f"sleep_did_comparison{tag}.csv", index=False)
    # `diagnostics` is a plain dict (26:1599), not a frame -- one row per modality.
    pd.DataFrame([{"pairing": n, **r["diagnostics"]}
                  for n, r in (("label", published_result), ("dated", dated_result))
                  ]).to_csv(OUT / f"diagnostics{tag}.csv", index=False)
    for name, result in (("label", published_result), ("dated", dated_result)):
        result["balance"].to_csv(OUT / f"balance_{name}{tag}.csv", index=False)

    print("\n=== sleep heart-rate DiD, visit-label vs date pairing ===")
    for row in side_by_side.itertuples(index=False):
        state = row.measurement.replace("heart_rate_mean_during_", "")
        print(f"  {state:<6}"
              f"  label {row.effect_label:+6.2f} [{row.ci_low_label:5.2f},"
              f" {row.ci_high_label:5.2f}] n={int(row.n_pairs_label):3d}"
              f"   dated {row.effect_dated:+6.2f} [{row.ci_low_dated:5.2f},"
              f" {row.ci_high_dated:5.2f}] n={int(row.n_pairs_dated):3d}")
    for name, result in (("label", published_result), ("dated", dated_result)):
        diag = result["diagnostics"]
        print(f"  {name:<5} design gates: eligible {diag['treated_outcome_eligible']:3d}"
              f"  matched {diag['primary_matched_treated']:3d}"
              f"  retention {diag['primary_retention']:.1%}"
              f"  max |SMD| {diag['max_abs_smd_primary']:.3f}"
              f"  pass={bool(diag['design_gate_pass'])}")

    # ---- section 2: the raw cohort, and the subset questions --------------------
    raw = raw_cohort_under_label_rule(step26, published_exposure)
    raw.to_csv(OUT / "raw_cohort_under_label_rule.csv", index=False)
    drawn = set(raw.participant_id)

    def matched_ids(result: dict) -> set[str]:
        pairs = result["primary_pairs"]
        column = "treated_id" if "treated_id" in pairs.columns else "participant_id"
        return set(pairs[column].astype(str))

    label_matched = matched_ids(published_result)
    dated_matched = matched_ids(dated_result)
    print("\n=== matched initiators: does date pairing add or remove people? ===")
    print(f"  label-matched {len(label_matched)}   date-matched {len(dated_matched)}"
          f"   in both {len(label_matched & dated_matched)}"
          f"   date only {len(dated_matched - label_matched)}"
          f"   label only {len(label_matched - dated_matched)}")

    explained = explain_raw_only(values, raw, audit, label_matched, dated_matched)
    explained.to_csv(OUT / f"raw_only_explained{tag}.csv", index=False)
    print("\n=== every raw-cohort initiator absent from a tabular matched set ===")
    for row in explained.itertuples(index=False):
        print(f"  {row.label:<4} tabular REM HR at {row.n_visits_with_tabular_rem} visits "
              f"[{row.rem_hr_by_visit}]")
        print(f"       label window {row.label_window}  blocked: {row.label_blocker}"
              f"   dated window {row.dated_window}")
        print(f"       -> {row.verdict}")

    # Which STAGE each absentee drops at. "Not in the matched set" has three quite
    # different causes and they need separating: never paired (no window exists), paired
    # but not outcome/covariate eligible, or eligible but unmatched (no control in its
    # exact block). Only the third is a matching problem.
    print("\n=== where each raw-cohort absentee drops out of the funnel ===")
    funnel = []
    for name, result, exposure in (
        ("label", published_result, published_exposure),
        ("dated", dated_result, dated_exposure),
    ):
        matched = matched_ids(result)
        units = result["units"]
        eligible = set(units.loc[units.treated.eq(1), "participant_id"].astype(str))
        paired_set = set(exposure.RegistrationCode.astype(str))
        absent = sorted(drawn - matched)
        stages = {"never paired": 0, "paired, not eligible": 0, "eligible, unmatched": 0}
        for participant in absent:
            if participant not in paired_set:
                stages["never paired"] += 1
            elif participant not in eligible:
                stages["paired, not eligible"] += 1
            else:
                stages["eligible, unmatched"] += 1
        funnel.append({"pairing": name, "raw_absent": len(absent), **stages})
        print(f"  {name:<6} {len(absent):2d} of the raw 33 absent:  "
              + "   ".join(f"{k} {v}" for k, v in stages.items()))
    pd.DataFrame(funnel).to_csv(OUT / f"absentee_funnel{tag}.csv", index=False)

    print("\n=== raw cohort (Figure 1c, n=33) against the tabular matched sets ===")
    for name, ids in (("label-paired (published)", label_matched),
                      ("date-paired (this step)", dated_matched)):
        print(f"  tabular matched initiators, {name:<26}: {len(ids):3d}"
              f"   shared with the raw 33: {len(drawn & ids):2d}"
              f"   raw-only: {len(drawn - ids):2d}   is the 33 a subset? {drawn <= ids}")
    survivors = set(raw.loc[raw.survives_label_rule, "participant_id"])
    print(f"\n  raw participants whose two drawn nights sit at their OWN label visits: "
          f"{len(survivors)}/{len(raw)}")
    raw_did = raw_did_under_label_rule(step26, survivors, drawn)
    raw_did.to_csv(OUT / "raw_did_under_label_rule.csv", index=False)
    print("\n=== raw channels: as drawn vs restricted to label-rule survivors ===")
    for row in raw_did.itertuples(index=False):
        print(f"  {row.cohort:<22} n={row.n_treated:2d}"
              f"   REM DiD {row.rem_did:+.2f}"
              f"   nonREM DiD {row.nrem_did:+.2f}"
              f"   REM-nonREM {row.rem_minus_nrem:+.2f}")
    print(f"  written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
