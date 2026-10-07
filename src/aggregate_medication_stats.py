#!/usr/bin/env python
"""Aggregate descriptive 10K medication-use statistics.

This script intentionally keeps the analysis descriptive: it summarizes who
reported medication use, which medications are common, and how records split by
research stage, collection method, and ATC metadata.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = pd.DataFrame.map


DEFAULT_TOP_N = 30
DEFAULT_MIN_COUNT = 3
SUPPLEMENT_NAME_PATTERN = re.compile(
    "|".join(
        [
            "vitamin",
            "ויטמין",
            "omega",
            "אומגה",
            "probiotic",
            "פרוביוט",
            "curcumin",
            "כורכומ",
            "magnesium",
            "מגנז",
            "calcium",
            "סידן",
            "zinc",
            "אבץ",
            "iron",
            "ברזל",
            "folic",
            "folate",
            "חומצה פולית",
            "biotin",
            "ביוטין",
            "prenatal",
            "פרנטל",
            "b complex",
            "creatine",
            "קריאטין",
            "multi",
            "מולטי",
            "q10",
            "קיו10",
            "collagen",
            "קולגן",
        ]
    ),
    flags=re.IGNORECASE,
)
SUPPLEMENT_ATC_PREFIXES = ("A11", "A12", "B03A", "B03B")
SUPPLEMENT_ATC_CODES = {"C10AX06", "OTHER SUPPLEMENT"}

ATC_CATEGORY_LABELS = {
    "A": "Alimentary tract and metabolism",
    "B": "Blood and blood forming organs",
    "C": "Cardiovascular system",
    "D": "Dermatologicals",
    "G": "Genito urinary system and sex hormones",
    "H": "Systemic hormonal preparations",
    "J": "Anti-infectives for systemic use",
    "L": "Antineoplastic and immunomodulating agents",
    "M": "Musculo-skeletal system",
    "N": "Nervous system",
    "P": "Antiparasitic products",
    "R": "Respiratory system",
    "S": "Sensory organs",
    "V": "Various",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate 10K medication usage statistics from Medications10KLoader."
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        type=Path,
        help="Directory for tabular outputs, relative to this script unless absolute.",
    )
    parser.add_argument(
        "--fig-dir",
        default="figs",
        type=Path,
        help="Directory for plot outputs, relative to this script unless absolute.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        type=Path,
        help="Directory for logs, relative to this script unless absolute.",
    )
    parser.add_argument(
        "--top-n",
        default=DEFAULT_TOP_N,
        type=int,
        help="Number of top medications or ATC groups to plot.",
    )
    parser.add_argument(
        "--min-count",
        default=DEFAULT_MIN_COUNT,
        type=int,
        help="Minimum participant count for rows retained in the main summaries.",
    )
    parser.add_argument(
        "--gen-cache",
        action="store_true",
        help="Regenerate the LabData medication cache before aggregating.",
    )
    parser.add_argument(
        "--reg-ids-csv",
        type=Path,
        help="Optional CSV containing RegistrationCode values to restrict the analysis.",
    )
    parser.add_argument(
        "--prescription-only",
        action="store_true",
        help="Filter out supplement-like products before aggregating.",
    )
    return parser.parse_args()


def resolve_output_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "aggregate_medication_stats.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def load_registration_codes(reg_ids_csv: Path | None) -> list[str] | None:
    if reg_ids_csv is None:
        return None

    reg_ids = pd.read_csv(reg_ids_csv)
    if "RegistrationCode" not in reg_ids.columns:
        raise ValueError(f"{reg_ids_csv} must contain a RegistrationCode column")
    values = reg_ids["RegistrationCode"].dropna().astype(str).unique().tolist()
    logging.info("Loaded %d RegistrationCode filters from %s", len(values), reg_ids_csv)
    return values


def load_medications(gen_cache: bool, reg_ids: Iterable[str] | None):
    try:
        from LabData.DataLoaders.Medications10KLoader import Medications10KLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import LabData Medications10KLoader. Activate the LabData "
            "Python environment or install LabData/LabUtils requirements."
        ) from exc

    loader = Medications10KLoader(gen_cache=gen_cache)
    kwargs = {}
    if reg_ids is not None:
        kwargs["reg_ids"] = list(reg_ids)

    logging.info("Loading Medications10KLoader data")
    return loader.get_data(**kwargs)


def normalize_stage(stage: object) -> str:
    if pd.isna(stage):
        return "unknown"
    stage_text = str(stage)
    return "baseline" if stage_text == "baseline" else "follow_up"


def normalize_specific_stage(stage: object) -> str:
    if pd.isna(stage):
        return "unknown"
    stage_text = str(stage)
    if stage_text == "baseline":
        return "baseline"

    match = re.match(r"^(\d{2})_(\d{2})_(call|visit)$", stage_text)
    if not match:
        return stage_text

    visit_number, subvisit_number, stage_type = match.groups()
    if subvisit_number == "00":
        return f"{visit_number} {stage_type}"
    return f"{visit_number}.{subvisit_number} {stage_type}"


def stage_sort_key(stage: str) -> tuple[int, int, int, int, str]:
    if stage == "baseline":
        return (0, 0, 0, 0, stage)
    if stage == "unknown":
        return (9, 999, 999, 999, stage)

    match = re.match(r"^(\d{2})(?:\.(\d{2}))? (call|visit)$", stage)
    if match:
        visit_number, subvisit_number, stage_type = match.groups()
        stage_type_order = 0 if stage_type == "call" else 1
        return (1, int(visit_number), int(subvisit_number or 0), stage_type_order, stage)
    return (8, 999, 999, 999, stage)


def stage_column_suffix(stage: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", stage).strip("_")


def prepare_medication_events(data) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = data.df.reset_index()

    metadata = data.df_metadata.reset_index()
    metadata_key = ["RegistrationCode", "Date", "medication"]
    metadata_cols = [
        col
        for col in [
            "research_stage",
            "created_at",
            "updated_at",
            "source_updated_at",
            "source_id",
            "collection_method",
            "start_month",
            "start_year",
            "end_month",
            "end_year",
        ]
        if col in metadata.columns
    ]
    events = events.merge(metadata[metadata_key + metadata_cols], on=metadata_key, how="left")
    events["stage_group"] = events.get("research_stage", pd.Series(index=events.index)).map(normalize_stage)
    events["specific_stage"] = events.get("research_stage", pd.Series(index=events.index)).map(normalize_specific_stage)
    events["Start"] = events["Start"].astype("boolean")

    columns_metadata = data.df_columns_metadata.rename(columns={"column_name": "medication"})
    return events, columns_metadata


def add_atc_levels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    atc = df["ATC"].fillna("").astype(str).str.strip() if "ATC" in df.columns else pd.Series("", index=df.index)
    df["ATC1"] = atc.str[:1].where(atc.str.len() >= 1, "")
    df["ATC2"] = atc.str[:3].where(atc.str.len() >= 3, "")
    df["ATC3"] = atc.str[:4].where(atc.str.len() >= 4, "")
    df["ATC4"] = atc.str[:5].where(atc.str.len() >= 5, "")
    df["ATC5"] = atc.str[:7].where(atc.str.len() >= 7, "")
    return df


def derive_category(row: pd.Series) -> pd.Series:
    atc = row.get("ATC")
    if pd.isna(atc):
        atc = ""
    atc = str(atc).strip()

    if atc == "OTHER SUPPLEMENT":
        category_code = "SUPPLEMENT"
        category = "Supplements and non-ATC products"
    elif atc:
        category_code = atc[:1]
        category = ATC_CATEGORY_LABELS.get(category_code, "Other ATC category")
    else:
        category_code = "UNPARSED"
        category = "Uncategorized / unparsed"

    conditions = row.get("Conditions")
    if pd.isna(conditions):
        primary_condition = ""
    else:
        primary_condition = str(conditions).split("/")[0].strip()

    return pd.Series(
        {
            "category_code": category_code,
            "category": category,
            "primary_condition": primary_condition,
        }
    )


def derive_drug_family_or_purpose(row: pd.Series) -> str:
    generic = "" if pd.isna(row.get("Generic")) else str(row.get("Generic")).lower()
    medication = "" if pd.isna(row.get("medication")) else str(row.get("medication")).lower()
    atc = "" if pd.isna(row.get("ATC")) else str(row.get("ATC")).strip()
    text = f"{generic} {medication}"

    if "levothyroxine" in text:
        return "Thyroid hormone replacement"
    if any(term in text for term in ["escitalopram", "sertraline", "fluoxetine", "paroxetine", "citalopram", "fluvoxamine"]):
        return "SSRI antidepressant/anxiolytic"
    if any(term in text for term in ["atorvastatin", "rosuvastatin", "simvastatin", "pravastatin"]):
        if "ezetimibe" in text:
            return "Statin + cholesterol absorption inhibitor"
        return "Statin cholesterol-lowering"
    if "ezetimibe" in text:
        return "Cholesterol absorption inhibitor"
    if any(term in text for term in ["esomeprazole", "omeprazole", "pantoprazole", "lansoprazole"]):
        return "Proton pump inhibitor for acid reflux"
    if any(term in text for term in ["acetylsalicylic", "aspirin"]):
        return "Antiplatelet/salicylate"
    if "bisoprolol" in text:
        return "Beta blocker antihypertensive"
    if "ramipril" in text:
        return "ACE inhibitor antihypertensive"
    if "semaglutide" in text:
        return "GLP-1 receptor agonist for diabetes/weight"
    if "metformin" in text:
        return "Biguanide diabetes medication"
    if "valsartan" in text and "amlodipine" in text:
        return "ARB + calcium-channel blocker"
    if any(term in text for term in ["estradiol", "estrogen", "norethisterone"]):
        return "Sex hormone / HRT"
    if "amphetamine" in text:
        return "Stimulant for ADHD"
    if "valsartan" in text or "losartan" in text or "candesartan" in text:
        return "ARB antihypertensive"
    if "amlodipine" in text or "lercanidipine" in text:
        return "Calcium-channel blocker"

    if atc.startswith("C10"):
        return "Lipid-lowering medication"
    if atc.startswith("C09"):
        return "Renin-angiotensin antihypertensive"
    if atc.startswith("C07"):
        return "Beta blocker"
    if atc.startswith("C08"):
        return "Calcium-channel blocker"
    if atc.startswith("A02B"):
        return "Acid-suppressing medication"
    if atc.startswith("A10"):
        return "Diabetes medication"
    if atc.startswith("N06A"):
        return "Antidepressant/anxiolytic"
    if atc.startswith("N06B"):
        return "Stimulant/ADHD medication"
    if atc.startswith("H03"):
        return "Thyroid medication"
    if atc.startswith("G03"):
        return "Sex hormone / HRT"
    if atc.startswith("B01"):
        return "Antithrombotic medication"
    if atc.startswith("R03"):
        return "Asthma/COPD airway medication"
    if atc.startswith("M01"):
        return "Anti-inflammatory pain medication"
    if atc.startswith("N02"):
        return "Analgesic pain medication"
    if atc.startswith("N05"):
        return "Sedative/anxiolytic/antipsychotic"

    return "Other/unclear purpose"


def add_medication_categories(medication_summary: pd.DataFrame) -> pd.DataFrame:
    categorized = medication_summary.copy()
    category_cols = categorized.apply(derive_category, axis=1)
    family = categorized.apply(derive_drug_family_or_purpose, axis=1)
    insert_at = 1
    for col in ["category_code", "category", "primary_condition"]:
        if col in categorized.columns:
            categorized = categorized.drop(columns=[col])
        categorized.insert(insert_at, col, category_cols[col])
        insert_at += 1
    if "drug_family_or_purpose" in categorized.columns:
        categorized = categorized.drop(columns=["drug_family_or_purpose"])
    categorized.insert(insert_at, "drug_family_or_purpose", family)
    return categorized


def sort_medications_by_category(medication_summary: pd.DataFrame) -> pd.DataFrame:
    return medication_summary.sort_values(
        ["category", "participant_count", "start_event_count", "medication"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def supplement_exclusion_reasons(row: pd.Series) -> list[str]:
    reasons = []
    atc = "" if pd.isna(row.get("ATC")) else str(row.get("ATC")).strip()
    medication = "" if pd.isna(row.get("medication")) else str(row.get("medication"))
    generic = "" if pd.isna(row.get("Generic")) else str(row.get("Generic"))
    category = "" if pd.isna(row.get("category")) else str(row.get("category"))
    combined_text = f"{medication} {generic}"

    if atc in SUPPLEMENT_ATC_CODES:
        reasons.append(f"ATC={atc}")
    if atc.startswith(SUPPLEMENT_ATC_PREFIXES):
        reasons.append(f"ATC_prefix={atc[:4] if atc.startswith('B03') else atc[:3]}")
    if category == "Supplements and non-ATC products":
        reasons.append("category=supplement_non_atc")
    if SUPPLEMENT_NAME_PATTERN.search(combined_text):
        reasons.append("name_or_generic_pattern")
    return reasons


def filter_prescription_only(
    events: pd.DataFrame, med_metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = add_medication_categories(med_metadata)
    metadata["exclusion_reason"] = metadata.apply(
        lambda row: "; ".join(supplement_exclusion_reasons(row)), axis=1
    )
    excluded = metadata[metadata["exclusion_reason"] != ""].copy()
    excluded_meds = set(excluded["medication"])

    filtered_events = events[~events["medication"].isin(excluded_meds)].copy()
    filtered_metadata = med_metadata[~med_metadata["medication"].isin(excluded_meds)].copy()
    return filtered_events, filtered_metadata, excluded


def true_start_events(events: pd.DataFrame) -> pd.DataFrame:
    return events[events["Start"].fillna(False)].copy()


def attach_start_date_and_usage_months(
    transitions: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Add medication_start_date and months_under_usage_at_confirmation to a transitions table.

    medication_start_date: earliest Date of start events for (RegistrationCode, medication).
      Date is the self-reported start (start_year/start_month) or source_created_at fallback.

    months_under_usage_at_confirmation: months between medication_start_date and the
      confirmation visit date, estimated from the earliest created_at record for
      (RegistrationCode, confirmation_visit_stage) across any medication at that stage.
    """
    if transitions.empty:
        return transitions

    starts = true_start_events(events)

    start_dates = (
        starts.groupby(["RegistrationCode", "medication"])["Date"]
        .min()
        .rename("medication_start_date")
        .reset_index()
    )
    result = transitions.merge(start_dates, on=["RegistrationCode", "medication"], how="left")
    result["medication_start_date"] = (
        pd.to_datetime(result["medication_start_date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )

    if "created_at" not in events.columns:
        result["months_under_usage_at_confirmation"] = pd.NA
        return result

    visit_dates = (
        events.groupby(["RegistrationCode", "specific_stage"])["created_at"]
        .min()
        .reset_index()
        .rename(columns={"specific_stage": "confirmation_visit_stage", "created_at": "_confirmation_date"})
    )
    visit_dates["_confirmation_date"] = (
        pd.to_datetime(visit_dates["_confirmation_date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    result = result.merge(visit_dates, on=["RegistrationCode", "confirmation_visit_stage"], how="left")
    result["months_under_usage_at_confirmation"] = (
        (result["_confirmation_date"] - result["medication_start_date"]).dt.days / 30.44
    ).round(1)
    result = result.drop(columns=["_confirmation_date"])
    return result


def validate_and_fix_prior_visit_dates(
    transitions: pd.DataFrame,
    events: pd.DataFrame,
    window_months: float = 3.0,
) -> pd.DataFrame:
    """
    Check that medication_start_date does not predate prior_absent_visit_stage date by
    more than window_months. Where it does, push prior_absent_visit_stage back to the
    latest earlier absent visit whose date is not predated within the window.

    Adds two columns:
      prior_visit_date_fixed: True if prior_absent_visit_stage was replaced with an earlier visit
      prior_visit_date_valid: True if the final prior visit passes the date check
        (False when start predates all prior absent visits — drug started before study baseline)
    """
    if transitions.empty or "medication_start_date" not in transitions.columns:
        return transitions

    if "created_at" not in events.columns:
        result = transitions.copy()
        result["prior_visit_date_fixed"] = False
        result["prior_visit_date_valid"] = True
        return result

    window = pd.Timedelta(days=round(window_months * 30.44))

    stages = sorted(events["specific_stage"].dropna().unique(), key=stage_sort_key)
    stage_rank = {stage: i for i, stage in enumerate(stages)}

    visit_dates_df = (
        events.groupby(["RegistrationCode", "specific_stage"])["created_at"]
        .min()
        .reset_index()
    )
    visit_dates_df["visit_date"] = (
        pd.to_datetime(visit_dates_df["created_at"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    visit_date_map = (
        visit_dates_df.set_index(["RegistrationCode", "specific_stage"])["visit_date"]
        .to_dict()
    )

    starts = true_start_events(events)
    present_map = {
        (reg, med): set(grp["specific_stage"].dropna().unique())
        for (reg, med), grp in starts.groupby(["RegistrationCode", "medication"])
    }

    observed_visit_map = {}
    for reg, grp in events.groupby("RegistrationCode"):
        visit_stages = sorted(
            [
                s for s in grp["specific_stage"].dropna().unique()
                if s == "baseline" or str(s).endswith(" visit")
            ],
            key=stage_sort_key,
        )
        observed_visit_map[reg] = visit_stages

    result = transitions.copy()
    fixed_flags = []
    valid_flags = []

    for idx, row in result.iterrows():
        reg = row["RegistrationCode"]
        med = row["medication"]
        reported_order = row["reported_stage_order"]
        med_start = row["medication_start_date"]
        current_prior = row["prior_absent_visit_stage"]

        prior_date = visit_date_map.get((reg, current_prior))

        if pd.isna(med_start) or prior_date is None or pd.isna(prior_date):
            fixed_flags.append(False)
            valid_flags.append(True)
            continue

        if med_start >= prior_date - window:
            fixed_flags.append(False)
            valid_flags.append(True)
            continue

        # med_start predates prior_date by more than window — look for earlier valid visit
        present_stages = present_map.get((reg, med), set())
        observed_visits = observed_visit_map.get(reg, [])
        candidates = [
            s for s in reversed(observed_visits)
            if stage_rank.get(s, 999) < reported_order and s not in present_stages
        ]

        fixed = False
        for candidate in candidates:
            cand_date = visit_date_map.get((reg, candidate))
            if cand_date is None or pd.isna(cand_date):
                continue
            if med_start >= cand_date - window:
                result.at[idx, "prior_absent_visit_stage"] = candidate
                result.at[idx, "prior_absent_visit_stage_order"] = stage_rank.get(candidate, 0)
                fixed = True
                break

        fixed_flags.append(fixed)
        valid_flags.append(fixed)

    result["prior_visit_date_fixed"] = fixed_flags
    result["prior_visit_date_valid"] = valid_flags
    return result


def aggregate_participant_burden(events: pd.DataFrame, med_metadata: pd.DataFrame) -> pd.DataFrame:
    starts = true_start_events(events)
    starts = starts.merge(med_metadata[["medication", "ATC"]], on="medication", how="left")
    starts = add_atc_levels(starts)

    participant = starts.groupby("RegistrationCode").agg(
        unique_medications=("medication", "nunique"),
        medication_start_events=("medication", "size"),
        first_medication_date=("Date", "min"),
        last_medication_date=("Date", "max"),
        unique_atc1_groups=("ATC1", lambda x: x[x != ""].nunique()),
        unique_atc3_groups=("ATC3", lambda x: x[x != ""].nunique()),
    )

    stage_counts = (
        starts.groupby(["RegistrationCode", "stage_group"])["medication"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns=lambda col: f"unique_medications_{col}")
    )
    return participant.join(stage_counts, how="left").fillna(0).reset_index()


def aggregate_medications(
    events: pd.DataFrame, med_metadata: pd.DataFrame, min_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = true_start_events(events)

    summary = starts.groupby("medication").agg(
        participant_count=("RegistrationCode", "nunique"),
        start_event_count=("Start", "size"),
        first_reported_date=("Date", "min"),
        last_reported_date=("Date", "max"),
    )

    stop_counts = (
        events[events["Start"].fillna(False) == False]
        .groupby("medication")
        .size()
        .rename("stop_event_count")
    )
    stage_counts = (
        starts.groupby(["medication", "stage_group"])["RegistrationCode"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns=lambda col: f"participants_{col}")
    )
    collection_counts = (
        starts.groupby(["medication", "collection_method"])["RegistrationCode"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns=lambda col: f"participants_collection_{col}")
    )

    summary = summary.join(stop_counts, how="left").join(stage_counts, how="left").join(collection_counts, how="left")
    summary["stop_event_count"] = summary["stop_event_count"].fillna(0).astype(int)
    summary = summary.reset_index().merge(med_metadata, on="medication", how="left")
    summary = add_medication_categories(summary)
    summary = summary.sort_values(["participant_count", "start_event_count"], ascending=False)

    retained = summary[summary["participant_count"] >= min_count].copy()
    return retained, summary


def aggregate_specific_stages(events: pd.DataFrame, medication_summary: pd.DataFrame) -> pd.DataFrame:
    starts = true_start_events(events)
    stage_counts = (
        starts.groupby(["medication", "specific_stage"])["RegistrationCode"]
        .nunique()
        .unstack(fill_value=0)
    )

    ordered_stage_cols = sorted(stage_counts.columns, key=stage_sort_key)
    stage_counts = stage_counts[ordered_stage_cols]
    stage_counts = stage_counts.rename(columns=lambda col: f"participants_{col.replace(' ', '_')}")

    base_cols = ["medication", "participant_count", "category", "Generic", "ATC"]
    base_cols = [col for col in base_cols if col in medication_summary.columns]
    return medication_summary[base_cols].merge(stage_counts.reset_index(), on="medication", how="left").fillna(0)


def aggregate_new_reports_after_absence(
    events: pd.DataFrame, medication_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    starts = true_start_events(events)
    stages = sorted(events["specific_stage"].dropna().unique(), key=stage_sort_key)
    stage_rank = {stage: i for i, stage in enumerate(stages)}

    observed = events[["RegistrationCode", "specific_stage"]].drop_duplicates()
    observed_by_participant = {
        reg: sorted(group["specific_stage"].dropna().unique(), key=stage_sort_key)
        for reg, group in observed.groupby("RegistrationCode")
    }

    present = starts[["RegistrationCode", "medication", "specific_stage"]].drop_duplicates()
    records = []
    for (reg, medication), group in present.groupby(["RegistrationCode", "medication"]):
        observed_stages = observed_by_participant.get(reg, [])
        if not observed_stages:
            continue

        present_stages = sorted(group["specific_stage"].dropna().unique(), key=stage_sort_key)
        present_stage_set = set(present_stages)
        for reported_stage in present_stages:
            reported_rank = stage_rank[reported_stage]
            prior_absent_stages = [
                stage
                for stage in observed_stages
                if stage_rank[stage] < reported_rank and stage not in present_stage_set
            ]
            if not prior_absent_stages:
                continue

            prior_absent_stage = prior_absent_stages[-1]
            records.append(
                {
                    "RegistrationCode": reg,
                    "medication": medication,
                    "prior_absent_stage": prior_absent_stage,
                    "reported_stage": reported_stage,
                    "prior_absent_stage_order": stage_rank[prior_absent_stage],
                    "reported_stage_order": reported_rank,
                }
            )
            break

    detail = pd.DataFrame.from_records(records)
    base_cols = [
        col
        for col in [
            "medication",
            "participant_count",
            "category",
            "primary_condition",
            "drug_family_or_purpose",
            "Generic",
            "ATC",
        ]
        if col in medication_summary.columns
    ]

    if detail.empty:
        summary = medication_summary[base_cols].copy()
        summary.insert(1, "new_report_after_absence_participant_count", 0)
        transition_summary = pd.DataFrame(
            columns=[
                "prior_absent_stage",
                "reported_stage",
                "participant_count",
                "medication_count",
                "participant_medication_pairs",
            ]
        )
        return summary, detail, transition_summary

    reported_stage_counts = (
        detail.groupby(["medication", "reported_stage"])["RegistrationCode"]
        .nunique()
        .unstack(fill_value=0)
    )
    reported_stage_counts = reported_stage_counts[
        sorted(reported_stage_counts.columns, key=stage_sort_key)
    ].rename(columns=lambda col: f"new_at_{stage_column_suffix(col)}")

    prior_absent_stage_counts = (
        detail.groupby(["medication", "prior_absent_stage"])["RegistrationCode"]
        .nunique()
        .unstack(fill_value=0)
    )
    prior_absent_stage_counts = prior_absent_stage_counts[
        sorted(prior_absent_stage_counts.columns, key=stage_sort_key)
    ].rename(columns=lambda col: f"prior_absent_at_{stage_column_suffix(col)}")

    totals = detail.groupby("medication").agg(
        new_report_after_absence_participant_count=("RegistrationCode", "nunique"),
        participant_medication_pairs=("RegistrationCode", "size"),
        first_reported_stage_order=("reported_stage_order", "min"),
    )

    summary = (
        medication_summary[base_cols]
        .merge(totals.reset_index(), on="medication", how="left")
        .merge(reported_stage_counts.reset_index(), on="medication", how="left")
        .merge(prior_absent_stage_counts.reset_index(), on="medication", how="left")
        .fillna(0)
    )
    count_cols = [
        col
        for col in summary.columns
        if col.startswith("new_at_")
        or col.startswith("prior_absent_at_")
        or col
        in {
            "new_report_after_absence_participant_count",
            "participant_medication_pairs",
            "first_reported_stage_order",
        }
    ]
    summary[count_cols] = summary[count_cols].astype(int)
    summary = summary.sort_values(
        ["new_report_after_absence_participant_count", "participant_count"],
        ascending=False,
    )

    transition_summary = (
        detail.groupby(["prior_absent_stage", "reported_stage"])
        .agg(
            participant_count=("RegistrationCode", "nunique"),
            medication_count=("medication", "nunique"),
            participant_medication_pairs=("RegistrationCode", "size"),
        )
        .reset_index()
    )
    transition_summary["prior_absent_stage_order"] = transition_summary["prior_absent_stage"].map(stage_rank)
    transition_summary["reported_stage_order"] = transition_summary["reported_stage"].map(stage_rank)
    transition_summary = transition_summary.sort_values(
        ["reported_stage_order", "prior_absent_stage_order"]
    )

    detail = detail.merge(
        medication_summary[[col for col in base_cols if col != "participant_count"]],
        on="medication",
        how="left",
    )
    return summary, detail, transition_summary


def aggregate_categories(events: pd.DataFrame, med_metadata: pd.DataFrame, min_count: int) -> pd.DataFrame:
    starts = true_start_events(events)
    category_metadata = add_medication_categories(med_metadata)
    category_cols = ["medication", "category_code", "category"]
    starts = starts.merge(category_metadata[category_cols], on="medication", how="left")
    starts["category_code"] = starts["category_code"].fillna("UNPARSED")
    starts["category"] = starts["category"].fillna("Uncategorized / unparsed")

    grouped = starts.groupby(["category_code", "category"]).agg(
        participant_count=("RegistrationCode", "nunique"),
        start_event_count=("Start", "size"),
        medication_count=("medication", "nunique"),
    )
    grouped = grouped.reset_index()
    grouped = grouped[grouped["participant_count"] >= min_count]
    return grouped.sort_values("participant_count", ascending=False)


def aggregate_atc(events: pd.DataFrame, med_metadata: pd.DataFrame, min_count: int) -> pd.DataFrame:
    parsed = true_start_events(events)
    parsed = parsed.merge(med_metadata[["medication", "ATC"]], on="medication", how="left")
    parsed = add_atc_levels(parsed)
    parsed = parsed[parsed["ATC"].notna() & (parsed["ATC"].astype(str).str.strip() != "")]

    frames = []
    for level in ["ATC1", "ATC2", "ATC3", "ATC4", "ATC5"]:
        level_df = parsed[parsed[level] != ""]
        if level_df.empty:
            continue
        grouped = level_df.groupby(level).agg(
            participant_count=("RegistrationCode", "nunique"),
            start_event_count=("Start", "size"),
            medication_count=("medication", "nunique"),
        )
        grouped = grouped.reset_index().rename(columns={level: "atc_code"})
        grouped.insert(0, "atc_level", level)
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(columns=["atc_level", "atc_code", "participant_count", "start_event_count", "medication_count"])

    atc_summary = pd.concat(frames, ignore_index=True)
    atc_summary = atc_summary[atc_summary["participant_count"] >= min_count]
    return atc_summary.sort_values(["atc_level", "participant_count"], ascending=[True, False])


def aggregate_unparsed(medication_summary_all: pd.DataFrame) -> pd.DataFrame:
    if "PARSED" in medication_summary_all.columns:
        mask = medication_summary_all["PARSED"].fillna(False) == False
    else:
        mask = medication_summary_all["ATC"].isna() if "ATC" in medication_summary_all.columns else True

    cols = [
        col
        for col in [
            "medication",
            "participant_count",
            "start_event_count",
            "stop_event_count",
            "Generic",
            "ATC",
            "Expanded ATC",
            "Conditions",
            "ICD11",
            "PARSED",
        ]
        if col in medication_summary_all.columns
    ]
    return medication_summary_all.loc[mask, cols].sort_values("participant_count", ascending=False)


def plot_top_medications(medication_summary: pd.DataFrame, fig_dir: Path, top_n: int) -> None:
    top = medication_summary.head(top_n).sort_values("participant_count")
    if top.empty:
        return
    labels = top["medication"]
    if "drug_family_or_purpose" in top.columns:
        labels = top["medication"] + "\n" + top["drug_family_or_purpose"]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(top))))
    ax.barh(labels, top["participant_count"], color="#4C78A8")
    ax.set_xlabel("Unique participants")
    ax.set_ylabel("")
    ax.set_title(f"Top {len(top)} medications in 10K")
    fig.tight_layout()
    fig.savefig(fig_dir / "top_medications_by_participants.png", dpi=200)
    plt.close(fig)


def plot_medication_burden(participant_burden: pd.DataFrame, fig_dir: Path) -> None:
    if participant_burden.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(participant_burden["unique_medications"], bins=30, color="#59A14F", edgecolor="white")
    ax.set_xlabel("Unique medications per participant")
    ax.set_ylabel("Participants")
    ax.set_title("Medication burden distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "medication_burden_histogram.png", dpi=200)
    plt.close(fig)


def plot_top_atc(atc_summary: pd.DataFrame, fig_dir: Path, top_n: int) -> None:
    top = atc_summary[atc_summary["atc_level"] == "ATC3"].head(top_n)
    if top.empty:
        top = atc_summary.head(top_n)
    top = top.sort_values("participant_count")
    if top.empty:
        return
    labels = top["atc_level"] + " " + top["atc_code"]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(top))))
    ax.barh(labels, top["participant_count"], color="#F28E2B")
    ax.set_xlabel("Unique participants")
    ax.set_ylabel("")
    ax.set_title(f"Top {len(top)} ATC groups")
    fig.tight_layout()
    fig.savefig(fig_dir / "top_atc_groups_by_participants.png", dpi=200)
    plt.close(fig)


def plot_categories(category_summary: pd.DataFrame, fig_dir: Path) -> None:
    top = category_summary.sort_values("participant_count").copy()
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(5, 0.34 * len(top))))
    ax.barh(top["category"], top["participant_count"], color="#76B7B2")
    ax.set_xlabel("Summed medication participant counts")
    ax.set_ylabel("")
    ax.set_title("Medication categories")
    fig.tight_layout()
    fig.savefig(fig_dir / "medication_categories_by_participants.png", dpi=200)
    plt.close(fig)


def plot_stage_split(medication_summary: pd.DataFrame, fig_dir: Path, top_n: int) -> None:
    stage_cols = [col for col in ["participants_baseline", "participants_follow_up", "participants_unknown"] if col in medication_summary.columns]
    if not stage_cols:
        return

    top = medication_summary.head(top_n).set_index("medication")[stage_cols].fillna(0).sort_values(stage_cols[0])
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(top))))
    top.plot(kind="barh", stacked=True, ax=ax, color=["#4C78A8", "#E15759", "#BAB0AC"][: len(stage_cols)])
    ax.set_xlabel("Unique participants")
    ax.set_ylabel("")
    ax.set_title(f"Research-stage split for top {len(top)} medications")
    fig.tight_layout()
    fig.savefig(fig_dir / "top_medications_stage_split.png", dpi=200)
    plt.close(fig)


def plot_specific_stage_split(specific_stage_summary: pd.DataFrame, fig_dir: Path, top_n: int) -> None:
    stage_cols = [col for col in specific_stage_summary.columns if col.startswith("participants_")]
    stage_cols = [col for col in stage_cols if col not in {"participants_collection_web interface", "participants_collection_zoho"}]
    if not stage_cols:
        return

    def col_sort_key(col: str) -> tuple[int, int, int, int, str]:
        label = col.removeprefix("participants_").replace("_", " ")
        return stage_sort_key(label)

    stage_cols = sorted(stage_cols, key=col_sort_key)
    top = specific_stage_summary.head(top_n).set_index("medication")[stage_cols].fillna(0)
    top = top.loc[top.sum(axis=1).sort_values().index]
    if top.empty:
        return

    rename_cols = {col: col.removeprefix("participants_").replace("_", " ") for col in stage_cols}
    top = top.rename(columns=rename_cols)

    fig, ax = plt.subplots(figsize=(12, max(7, 0.3 * len(top))))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(top.columns))]
    top.plot(kind="barh", stacked=True, ax=ax, color=colors)
    ax.set_xlabel("Unique participants")
    ax.set_ylabel("")
    ax.set_title(f"Specific research-stage split for top {len(top)} medications")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "top_medications_specific_stage_split.png", dpi=200)
    plt.close(fig)


def plot_new_reports_after_absence(new_report_summary: pd.DataFrame, fig_dir: Path, top_n: int) -> None:
    stage_cols = [col for col in new_report_summary.columns if col.startswith("new_at_")]
    if not stage_cols:
        return

    def col_sort_key(col: str) -> tuple[int, int, int, int, str]:
        label = col.removeprefix("new_at_").replace("_", " ")
        return stage_sort_key(label)

    stage_cols = sorted(stage_cols, key=col_sort_key)
    top = new_report_summary[
        new_report_summary["new_report_after_absence_participant_count"] > 0
    ].head(top_n)
    if top.empty:
        return

    if "drug_family_or_purpose" in top.columns:
        top = top.copy()
        top["plot_label"] = top["medication"] + "\n" + top["drug_family_or_purpose"]
        index_col = "plot_label"
    else:
        index_col = "medication"

    top = top.set_index(index_col)[stage_cols].fillna(0)
    top = top.loc[top.sum(axis=1).sort_values().index]
    rename_cols = {col: col.removeprefix("new_at_").replace("_", " ") for col in stage_cols}
    top = top.rename(columns=rename_cols)

    fig, ax = plt.subplots(figsize=(12, max(7, 0.3 * len(top))))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(top.columns))]
    top.plot(kind="barh", stacked=True, ax=ax, color=colors)
    ax.set_xlabel("Participants newly reporting medication after earlier non-report")
    ax.set_ylabel("")
    ax.set_title(f"New medication reports after earlier non-report, top {len(top)}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "top_new_medication_reports_after_absence.png", dpi=200)
    plt.close(fig)


def write_outputs(
    out_dir: Path,
    participant_burden: pd.DataFrame,
    medication_summary: pd.DataFrame,
    medications_by_category: pd.DataFrame,
    category_summary: pd.DataFrame,
    specific_stage_summary: pd.DataFrame,
    new_report_summary: pd.DataFrame,
    new_report_detail: pd.DataFrame,
    new_report_transition_summary: pd.DataFrame,
    excluded_medications: pd.DataFrame,
    atc_summary: pd.DataFrame,
    unparsed: pd.DataFrame,
) -> None:
    participant_burden.to_csv(out_dir / "participant_medication_burden.csv", index=False)
    medication_summary.to_csv(out_dir / "medication_summary.csv", index=False)
    medications_by_category.to_csv(out_dir / "medications_by_category.csv", index=False)
    category_summary.to_csv(out_dir / "category_summary.csv", index=False)
    specific_stage_summary.to_csv(out_dir / "specific_stage_medication_summary.csv", index=False)
    new_report_summary.to_csv(out_dir / "new_reports_after_absence_summary.csv", index=False)
    new_report_detail.to_csv(out_dir / "new_reports_after_absence_detail.csv", index=False)
    new_report_transition_summary.to_csv(out_dir / "new_reports_after_absence_transitions.csv", index=False)
    excluded_medications.to_csv(out_dir / "excluded_supplement_like_medications.csv", index=False)
    atc_summary.to_csv(out_dir / "atc_summary.csv", index=False)
    unparsed.to_csv(out_dir / "unparsed_medications.csv", index=False)

    workbook = out_dir / "medication_usage_summary.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        participant_burden.to_excel(writer, sheet_name="participant_burden", index=False)
        medication_summary.to_excel(writer, sheet_name="medication_summary", index=False)
        medications_by_category.to_excel(writer, sheet_name="meds_by_category", index=False)
        category_summary.to_excel(writer, sheet_name="category_summary", index=False)
        specific_stage_summary.to_excel(writer, sheet_name="specific_stage_summary", index=False)
        new_report_summary.to_excel(writer, sheet_name="new_after_absence", index=False)
        new_report_transition_summary.to_excel(writer, sheet_name="new_after_abs_trans", index=False)
        excluded_medications.to_excel(writer, sheet_name="excluded_supplements", index=False)
        atc_summary.to_excel(writer, sheet_name="atc_summary", index=False)
        unparsed.to_excel(writer, sheet_name="unparsed_medications", index=False)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out_dir = resolve_output_path(args.out_dir, script_dir)
    fig_dir = resolve_output_path(args.fig_dir, script_dir)
    log_dir = resolve_output_path(args.log_dir, script_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir)

    reg_ids = load_registration_codes(args.reg_ids_csv)
    data = load_medications(gen_cache=args.gen_cache, reg_ids=reg_ids)
    events, med_metadata = prepare_medication_events(data)
    logging.info("Loaded %d medication event rows", len(events))
    excluded_medications = pd.DataFrame()
    if args.prescription_only:
        events, med_metadata, excluded_medications = filter_prescription_only(events, med_metadata)
        logging.info(
            "Prescription-only filter removed %d medication names; %d event rows remain",
            len(excluded_medications),
            len(events),
        )

    participant_burden = aggregate_participant_burden(events, med_metadata)
    medication_summary, medication_summary_all = aggregate_medications(events, med_metadata, args.min_count)
    medications_by_category = sort_medications_by_category(medication_summary)
    category_summary = aggregate_categories(events, med_metadata, args.min_count)
    specific_stage_summary = aggregate_specific_stages(events, medication_summary)
    (
        new_report_summary,
        new_report_detail,
        new_report_transition_summary,
    ) = aggregate_new_reports_after_absence(events, medication_summary)
    atc_summary = aggregate_atc(events, med_metadata, args.min_count)
    unparsed = aggregate_unparsed(medication_summary_all)

    write_outputs(
        out_dir,
        participant_burden,
        medication_summary,
        medications_by_category,
        category_summary,
        specific_stage_summary,
        new_report_summary,
        new_report_detail,
        new_report_transition_summary,
        excluded_medications,
        atc_summary,
        unparsed,
    )
    plot_top_medications(medication_summary, fig_dir, args.top_n)
    plot_medication_burden(participant_burden, fig_dir)
    plot_top_atc(atc_summary, fig_dir, args.top_n)
    plot_categories(category_summary, fig_dir)
    plot_stage_split(medication_summary, fig_dir, args.top_n)
    plot_specific_stage_split(specific_stage_summary, fig_dir, args.top_n)
    plot_new_reports_after_absence(new_report_summary, fig_dir, args.top_n)

    logging.info("Unique participants with medication starts: %d", len(participant_burden))
    logging.info("Medication summary rows retained: %d", len(medication_summary))
    logging.info("Medication categories retained: %d", len(category_summary))
    logging.info(
        "Participant-medication new reports after earlier non-report: %d",
        len(new_report_detail),
    )
    logging.info("ATC summary rows retained: %d", len(atc_summary))
    logging.info("Unparsed medication rows: %d", len(unparsed))
    logging.info("Wrote outputs to %s and figures to %s", out_dir, fig_dir)


if __name__ == "__main__":
    main()
