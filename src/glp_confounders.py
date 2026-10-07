#!/usr/bin/env python
"""Pre-exposure confounders for GLP-1 adjustment, beyond age/gender/BMI.

- diabetic: self-reported Diabetes at or before the unit's pre-treatment research
  stage in the updated 10K conditions tables (`baseline_conditions_all.csv` /
  `follow_up_conditions_all.csv`, `Consolidated name == 'Diabetes'`). An explicit
  `ever` sensitivity mode is available because self-reported diagnosis can lag.
  Prediabetes is excluded by default (set INCLUDE_PREDIABETES to broaden). These
  files are 10K-linked ('10K_...').
  (MedicalConditionLoader is broken here; DiagnosesLoader is a DIFFERENT cohort with
  integer IDs that do not map to the 10K codes — do not use it.)
- on_antihypertensive / on_lipid_lowering: any such drug reported at baseline
  (ATC C02/C03/C07/C08/C09 and C10 respectively).
- vat_area: DXA total_scan_vat_area at baseline (cleaner visceral-adiposity measure
  than BMI/waist); NaN when the participant has no baseline DXA, so units without it
  are dropped by the matching pipeline ("use the DXA if available").

`augment_covariates` adds these columns to the per-(RegistrationCode, research_stage)
covariate table; diabetic/med flags fill 0 when absent, vat_area stays NaN. The
default diabetes timing uses only information available by that research stage.
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from glp_modality_matched_analysis import normalize_reg

EXTRA_COVARIATES = ["diabetic", "on_antihypertensive", "on_lipid_lowering", "vat_area"]
CONDITIONS_DIR = Path("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/for_review")
CONDITION_FILES = ["baseline_conditions_all.csv", "follow_up_conditions_all.csv"]
DIABETES_LABELS = ("Diabetes",)          # lenient self-report; excludes prediabetes/gestational
INCLUDE_PREDIABETES = False
ANTIHYPERTENSIVE_ATC = ("C02", "C03", "C07", "C08", "C09")
LIPID_ATC = ("C10",)


def _regs(index) -> pd.Series:
    return normalize_reg(index.get_level_values("RegistrationCode"))


def _stage_order(values: pd.Series) -> pd.Series:
    """Map 10K research-stage labels to their leading chronological stage number."""
    text = values.astype("string").str.lower()
    order = pd.to_numeric(text.str.extract(r"^(\d+)", expand=False), errors="coerce")
    return order.mask(text.eq("baseline"), 0.0)


@lru_cache(maxsize=1)
def self_reported_diabetes_events() -> pd.DataFrame:
    """Unique participant/stage rows where Diabetes was self-reported."""
    labels = set(DIABETES_LABELS) | ({"Prediabetes"} if INCLUDE_PREDIABETES else set())
    frames = []
    for fname in CONDITION_FILES:
        df = pd.read_csv(
            CONDITIONS_DIR / fname,
            usecols=["RegistrationCode", "research_stage", "Consolidated name"],
        )
        hit = df["Consolidated name"].astype(str).isin(labels)
        events = df.loc[hit, ["RegistrationCode", "research_stage"]].copy()
        events["RegistrationCode"] = normalize_reg(events["RegistrationCode"]).to_numpy()
        events["diabetes_stage_order"] = _stage_order(events["research_stage"])
        frames.append(events[["RegistrationCode", "diabetes_stage_order"]])
    return pd.concat(frames, ignore_index=True).dropna().drop_duplicates()


def diabetes_flags(covariates: pd.DataFrame, timing: str = "pre_stage") -> pd.DataFrame:
    """Diabetes indicator per participant/stage using pre-stage or ever information."""
    if timing not in {"pre_stage", "ever"}:
        raise ValueError(f"unknown diabetes timing: {timing}")
    keys = covariates[["RegistrationCode", "research_stage"]].drop_duplicates().copy()
    events = self_reported_diabetes_events()
    if timing == "ever":
        diabetic = set(events["RegistrationCode"])
        keys["diabetic"] = keys["RegistrationCode"].isin(diabetic).astype(float)
        return keys

    keys["pre_stage_order"] = _stage_order(keys["research_stage"])
    first_report = events.groupby("RegistrationCode", as_index=False)["diabetes_stage_order"].min()
    keys = keys.merge(first_report, on="RegistrationCode", how="left")
    keys["diabetic"] = (
        keys["diabetes_stage_order"].notna()
        & keys["pre_stage_order"].notna()
        & keys["diabetes_stage_order"].le(keys["pre_stage_order"])
    ).astype(float)
    return keys[["RegistrationCode", "research_stage", "diabetic"]]


@lru_cache(maxsize=1)
def baseline_med_flags() -> pd.DataFrame:
    from aggregate_medication_stats import load_medications, prepare_medication_events

    data = load_medications(gen_cache=False, reg_ids=None)
    events, meta = prepare_medication_events(data)
    base = events[(events["specific_stage"] == "baseline") & (events["Start"].fillna(False) == True)].copy()
    base = base.merge(meta[["medication", "ATC"]].drop_duplicates("medication"), on="medication", how="left")
    base["RegistrationCode"] = normalize_reg(base["RegistrationCode"]).to_numpy()
    atc = base["ATC"].fillna("").astype(str)
    base["is_antihypertensive"] = atc.str.startswith(ANTIHYPERTENSIVE_ATC)
    base["is_lipid"] = atc.str.startswith(LIPID_ATC)
    grouped = base.groupby("RegistrationCode").agg(
        on_antihypertensive=("is_antihypertensive", "max"),
        on_lipid_lowering=("is_lipid", "max"),
    ).astype(float)
    return grouped.reset_index()


@lru_cache(maxsize=1)
def baseline_vat_area() -> pd.DataFrame:
    d = getattr(importlib.import_module("LabData.DataLoaders.DEXALoader"), "DEXALoader")().get_data()
    df = d.df
    meta = d.df_metadata
    if "total_scan_vat_area" not in df.columns:
        raise ValueError("DEXA lacks total_scan_vat_area")
    stage = pd.Series(meta["research_stage"].to_numpy(), index=range(len(df)))
    work = pd.DataFrame({
        "RegistrationCode": _regs(df.index).to_numpy(),
        "research_stage": meta["research_stage"].to_numpy(),
        "vat_area": pd.to_numeric(df["total_scan_vat_area"].to_numpy(), errors="coerce"),
    })
    work = work[work["research_stage"] == "baseline"].dropna(subset=["vat_area"])
    return work.groupby("RegistrationCode", as_index=False)["vat_area"].mean()


@lru_cache(maxsize=1)
def smoking_status() -> pd.DataFrame:
    """Per-participant current-smoker indicator from LifeStyleLoader.

    Uses `smoke_tobacco_now` (numeric coding); a participant is a current smoker if
    int(value) > 0 at any assessment (so the 1.0 code counts; 0.5/0/-1 do not,
    matching the agreed recipe). Deduped to one value per participant via max over
    visits ("current smoker at any observed visit"). Missing is treated as 0 downstream.
    """
    d = getattr(importlib.import_module("LabData.DataLoaders.LifeStyleLoader"),
                "LifeStyleLoader")().get_data(df="number")
    if "smoke_tobacco_now" not in d.df.columns:
        raise ValueError("LifeStyle lacks smoke_tobacco_now")
    work = pd.DataFrame({
        "RegistrationCode": _regs(d.df.index).to_numpy(),
        "smoke_raw": pd.to_numeric(d.df["smoke_tobacco_now"].to_numpy(), errors="coerce"),
    })
    work["smoking"] = (work["smoke_raw"].fillna(0).astype(int) > 0).astype(float)
    return work.groupby("RegistrationCode", as_index=False)["smoking"].max()


def load_baseline_confounders(snapshot_dir: Path) -> pd.DataFrame:
    meds = baseline_med_flags()
    vat = baseline_vat_area()
    logging.info("Baseline med flags: %d participants; DXA VAT area: %d participants", len(meds), len(vat))

    conf = meds.merge(vat, on="RegistrationCode", how="outer")
    conf["on_antihypertensive"] = conf["on_antihypertensive"].fillna(0.0)
    conf["on_lipid_lowering"] = conf["on_lipid_lowering"].fillna(0.0)
    # vat_area intentionally left NaN when no baseline DXA.
    return conf


def augment_covariates(
    covariates: pd.DataFrame,
    snapshot_dir: Path,
    diabetes_timing: str = "pre_stage",
) -> pd.DataFrame:
    """Add the extra confounders to a per-(RegistrationCode, research_stage) covariate table."""
    conf = load_baseline_confounders(snapshot_dir)
    merged = covariates.merge(conf, on="RegistrationCode", how="left")
    flags = diabetes_flags(covariates, timing=diabetes_timing)
    merged = merged.merge(flags, on=["RegistrationCode", "research_stage"], how="left")
    logging.info(
        "Diabetic participant-stage rows (%s): %d across %d participants",
        diabetes_timing,
        int(merged["diabetic"].fillna(0).sum()),
        merged.loc[merged["diabetic"].eq(1), "RegistrationCode"].nunique(),
    )
    for col in ["diabetic", "on_antihypertensive", "on_lipid_lowering"]:
        merged[col] = merged[col].fillna(0.0)
    return merged


def augment_for_matching(
    covariates: pd.DataFrame,
    treated_ids: set[str],
    diabetes_timing: str = "pre_stage",
) -> pd.DataFrame:
    """Add vat_area and diabetic to the matching covariate table used by
    glp_modality_matched_analysis.py.

    Unlike augment_covariates() (used by the complete-case ANCOVA-IPTW sensitivity),
    vat_area is imputed for GLP-1 treated participants who lack a baseline DXA scan,
    using the median vat_area of the rest of the GLP-1 cohort that does have one, so
    the matched-DiD pipeline does not drop treated participants for missing DXA.
    Controls without vat_area remain NaN and are excluded from the matching pool by
    the caller's existing dropna(subset=MATCH_COVARIATES).
    """
    vat = baseline_vat_area()
    treated_vat = vat.loc[vat["RegistrationCode"].isin(treated_ids), "vat_area"]
    impute_value = float(treated_vat.median())
    logging.info(
        "VAT: %d/%d GLP-1 treated participants have baseline DXA; imputing median %.1f for the rest",
        len(treated_vat), len(treated_ids), impute_value,
    )
    merged = covariates.merge(vat, on="RegistrationCode", how="left")
    needs_impute = merged["RegistrationCode"].isin(treated_ids) & merged["vat_area"].isna()
    merged.loc[needs_impute, "vat_area"] = impute_value

    flags = diabetes_flags(covariates, timing=diabetes_timing)
    merged = merged.merge(flags, on=["RegistrationCode", "research_stage"], how="left")
    merged["diabetic"] = merged["diabetic"].fillna(0.0)

    smoking = smoking_status()
    merged = merged.merge(smoking, on="RegistrationCode", how="left")
    merged["smoking"] = merged["smoking"].fillna(0.0)  # missing lifestyle -> non-smoker
    return merged
