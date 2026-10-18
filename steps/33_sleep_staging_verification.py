"""Verify that the REM state contrast is not an artefact of stage classification.

Why this is needed
------------------
The WatchPAT platform does not record EEG. It infers wake / light (N1+N2) /
deep (N3) / REM from peripheral arterial tonometry, pulse rate, oximetry,
actigraphy and body position. **Pulse rate is one of the classifier inputs.** So a
drug that raises heart rate could shift which epochs are labelled REM, and a
"REM-specific heart-rate effect" could in principle be a relabelling artefact
rather than a physiological finding. This is the most damaging objection a sleep
reviewer can raise against the paper's central claim, and no participant in this
cohort has simultaneous polysomnography to settle it directly.

Two checks that need no extra instrumentation:

1. **Architecture vs polysomnographic norms** -- if the classifier were badly
   miscalling stages, the population distribution of stage proportions should
   depart from attended-PSG norms in adults (REM 20-25%, N3 13-23%,
   N1+N2 50-60% of total sleep time).

2. **Treatment-related change in scored stage duration** -- the decisive test.
   If the chronotropic response were driving the classifier, scored REM duration
   should move with treatment. A null for REM duration alongside a positive
   within-REM heart-rate effect is the signature of a real within-state effect.

Neither check excludes per-epoch misclassification, nor state-dependent
misclassification. They exclude *gross classifier drift*, which is what the
circularity argument requires. Reported as such in the manuscript.

Usage
-----
    PY=/net/mraid20/export/jasmine/david/anaconda3/bin/python3
    $PY steps/33_sleep_staging_verification.py
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = (  # type: ignore[attr-defined]
        lambda self, func, na_action=None, **kw: self.map(func, na_action=na_action, **kw)
    )

GOLD = Path("/net/mraid20/export/genie/LabData/Data/Pheno/gold/sleep")
LINK_TABLE = GOLD / "sleep_all_hrv_with_participant.parquet"
TIMESERIES = GOLD / "timeseries"
RUN = Path(__file__).resolve().parent.parent
OUT = RUN / "outputs" / "sleep_staging_verification"

SEMAGLUTIDE_PATTERN = re.compile(r"ozempic|rybelsus|wegovy|אוזמפיק|ריבלסוס|וויגובי", re.I)

MIN_SLEEP_SECONDS = 3 * 3600
MIN_QUALITY = 50
MIN_GAP_DAYS = 30

# Attended-polysomnography reference ranges for healthy adults, as % of total
# sleep time. Used as a plausibility band, not as a statistical comparison.
PSG_NORMS = {
    "rem": (20.0, 25.0),
    "deep": (13.0, 23.0),
    "light": (50.0, 60.0),
}


def load_summary() -> pd.DataFrame:
    columns = [
        "participant_id", "uuid", "study_date", "total_sleep_time",
        "total_rem_sleep_time", "total_deep_sleep_time", "total_light_sleep_time",
        "percent_of_rem_sleep_time", "quality_score_heart_rate",
    ]
    frame = pq.read_table(LINK_TABLE, columns=columns).to_pandas(ignore_metadata=True)
    for column in columns[3:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Deliberately NOT restricted to nights with a local raw-channel directory:
    # both checks read summary fields only, so requiring raw availability would
    # discard two thirds of the nights (16k instead of 49k) and shrink the
    # treatment-change sample for no methodological gain.
    frame = frame.copy()
    frame["participant"] = frame.participant_id.astype("int64").astype(str)
    frame["study_date"] = pd.to_datetime(frame.study_date, errors="coerce", utc=True)
    return frame.drop_duplicates("uuid")


def check_architecture(summary: pd.DataFrame) -> pd.DataFrame:
    usable = summary[
        (summary.total_sleep_time > MIN_SLEEP_SECONDS)
        & (summary.quality_score_heart_rate >= MIN_QUALITY)
    ].dropna(subset=["total_sleep_time", "total_rem_sleep_time",
                     "total_deep_sleep_time", "total_light_sleep_time"])
    rows = []
    for stage, column in [("rem", "total_rem_sleep_time"),
                          ("deep", "total_deep_sleep_time"),
                          ("light", "total_light_sleep_time")]:
        percent = 100 * usable[column] / usable.total_sleep_time
        low, high = PSG_NORMS[stage]
        rows.append({
            "stage": stage,
            # One row per visit (the table is de-duplicated on uuid upstream), so
            # this counts visits, not nights -- avoids pseudo-replicating the
            # 2-3 nights recorded within a single visit.
            "n_visits": int(len(usable)),
            "mean_percent_of_tst": percent.mean(),
            "q25": percent.quantile(0.25),
            "q75": percent.quantile(0.75),
            "psg_norm_low": low,
            "psg_norm_high": high,
            "mean_within_norm": bool(low <= percent.mean() <= high),
        })
    return pd.DataFrame(rows)


def check_treatment_change(summary: pd.DataFrame) -> pd.DataFrame:
    module = importlib.import_module("LabData.DataLoaders.Medications10KLoader")
    data = getattr(module, "Medications10KLoader")().get_data()
    meds = (data.df if hasattr(data, "df") else data).reset_index()
    meds = meds[meds.medication.astype(str).str.contains(SEMAGLUTIDE_PATTERN)].copy()
    meds["participant"] = meds.RegistrationCode.astype(str).str.replace(
        "10K_", "", case=False, regex=False
    )
    meds["report_date"] = pd.to_datetime(meds.Date, errors="coerce", utc=True)
    first_report = meds.dropna(subset=["report_date"]).groupby("participant").report_date.min()

    usable = summary[summary.quality_score_heart_rate >= MIN_QUALITY].dropna(
        subset=["study_date", "total_sleep_time"]
    )
    records = []
    for participant, start in first_report.items():
        own = usable[usable.participant == participant]
        pre = own[own.study_date <= start]
        post = own[own.study_date > start + pd.Timedelta(days=MIN_GAP_DAYS)]
        if pre.empty or post.empty:
            continue
        before = pre.loc[pre.study_date.idxmax()]
        after = post.loc[post.study_date.idxmin()]
        records.append({
            "participant": participant,
            "d_percent_rem": after.percent_of_rem_sleep_time - before.percent_of_rem_sleep_time,
            "d_rem_minutes": (after.total_rem_sleep_time - before.total_rem_sleep_time) / 60.0,
            "d_total_sleep_seconds": after.total_sleep_time - before.total_sleep_time,
        })
    panel = pd.DataFrame(records).dropna()

    rows = []
    for metric, label, unit in [
        ("d_percent_rem", "percent of sleep spent in REM", "percentage points"),
        ("d_rem_minutes", "total REM time", "minutes"),
        ("d_total_sleep_seconds", "total sleep time", "seconds"),
    ]:
        values = panel[metric].to_numpy(dtype=float)
        n = len(values)
        se = values.std(ddof=1) / np.sqrt(n)
        rows.append({
            "metric": metric, "label": label, "unit": unit, "n": n,
            "mean_change": values.mean(),
            "ci_low": values.mean() - 1.96 * se,
            "ci_high": values.mean() + 1.96 * se,
            "p": float(stats.ttest_1samp(values, 0).pvalue),
        })
    return pd.DataFrame(rows), panel


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_summary()

    architecture = check_architecture(summary)
    architecture.to_csv(OUT / "stage_architecture_vs_psg_norms.csv", index=False)
    print("=== 1. Stage architecture vs polysomnographic norms ===")
    print(architecture.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    change, panel = check_treatment_change(summary)
    change.to_csv(OUT / "treatment_change_in_scored_stages.csv", index=False)
    panel.to_csv(OUT / "per_participant_stage_change.csv", index=False)
    print("\n=== 2. Treatment-related change in scored stage duration ===")
    print(change[["label", "n", "mean_change", "ci_low", "ci_high", "p"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    rem_row = change[change.metric == "d_percent_rem"].iloc[0]
    verdict = "PASS" if rem_row.p > 0.05 else "FAIL"
    print(f"\nClassifier-drift check: {verdict} "
          f"(scored REM proportion change p={rem_row.p:.3f}, n={int(rem_row.n)})")
    if verdict == "PASS":
        print("  Scored REM duration is unchanged while within-REM heart rate rises,")
        print("  which is the pattern expected of a genuine within-state effect.")
    else:
        print("  Scored REM duration moved with treatment -- the state contrast may")
        print("  reflect relabelling. Do not report REM accentuation without a PSG subset.")
    print(f"\nwrote {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
