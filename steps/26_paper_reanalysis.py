"""Paper-facing GLP-1 reanalysis with adjudicated exposure and bounded control reuse.

This is a clean reanalysis alongside (not an overwrite of) the historical
1:5-with-replacement results. It implements the decisions recorded during the
July 2026 paper-readiness review:

* primary exposure = timing-valid, visit-confirmed semaglutide initiation;
  Mounjaro/tirzepatide-labelled records are a separate sensitivity;
* controls exclude every participant ever observed on any A10BJ GLP-1 therapy;
* primary matching is optimal 1:1 without replacement, exact on sex and visit
  window, inside a 0.2-SD PS-logit caliper, with no fallback;
* sensitivities are 1:1--1:3 matching with maximum control reuse 2, overlap
  weighting, and new-user non-GLP diabetes active comparators;
* inference includes 95% CIs, equivalence tests for nulls, within-domain FDR,
  duration models that do not interpret the <3-month band, formal REM-vs-
  NREM/wake contrasts, and CGM mean-adjustment contrasts;
* "mediation" is replaced by descriptive covariate-attenuation models.

Data routing:

* requested phenotype snapshot ``out_dir`` for every available modality;
* cached LabData loaders for medication history, DXA, and diet because those
  datasets are absent from that snapshot;
* the previously used mapped 10K HUMAnN aggregate for functional pathways,
  because it is not present in the snapshot or a LabData loader. Taxonomic
  gut microbiome remains sourced from the canonical snapshot.

Run with the LabData Python environment:

    PYTHONPATH=<repo>/src:<run>/src \
      /net/mraid20/export/jasmine/david/anaconda3/bin/python3 \
      steps/26_paper_reanalysis.py
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(pd.DataFrame, "map"):
    # LabData's medication loader still calls the pandas-1.x name.
    pd.DataFrame.applymap = pd.DataFrame.map

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[2]
SRC = RUN / "src"
sys.path[:0] = [str(REPO / "src"), str(SRC)]

from glp_modality_matched_analysis import (  # noqa: E402
    DEFAULT_SNAPSHOT_DIR,
    STAGE_TO_RESEARCH_STAGE,
    normalize_reg,
)
import glp_confounders as confounders  # noqa: E402
from research_utils.analysis.models import (  # noqa: E402
    capacity_caliper_match,
    optimal_pair_match,
    propensity_score,
)
from functools import partial  # noqa: E402

from research_utils.analysis.stats import bh_fdr as _bh_fdr  # noqa: E402

# Standard step-up Benjamini-Hochberg (monotonic q), as the Methods state. The shared helper's
# default is the simpler p*m/rank form, kept there for parity with another paper; it is
# slightly conservative and is not what this analysis reports (switched 2026-10-06).
bh_fdr = partial(_bh_fdr, monotonic=True)

SNAPSHOT = Path(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/"
    "snapshots/pheno_data_snapshots/out_dir"
)
if SNAPSHOT != DEFAULT_SNAPSHOT_DIR:
    raise RuntimeError("The reanalysis must use the canonical phenotype snapshot out_dir")

OUT = RUN / "outputs" / "paper_reanalysis"
CACHE = OUT / "cache"
TRANSITIONS = (
    RUN
    / "src"
    / "visit_transition_analysis"
    / "outputs"
    / "strict_visit_reported_participant_medication_transitions.csv"
)
HUMANN_PATHWAYS = Path(
    "/net/mraid20/export/mb/MBPipeline/Analyses/HUMAnN/joined_pathways/"
    "pathways_all__aggregated_tenk.parquet"
)
SEED = 20260727
N_BOOT = 2000
CALIPER_SD = 0.2
MIN_N = 8
MIN_MATCH_RETENTION = 0.85
MAX_ABS_SMD = 0.10
MIN_HEART_RATE_STAGE_SECONDS = 300
SUPPORTED_STAGES = {"00_00_visit", "02_00_visit", "04_00_visit", "06_00_visit"}
STAGE_NORMALIZE = {"baseline": "00_00_visit", **STAGE_TO_RESEARCH_STAGE}

CONTINUOUS_COVARIATES = [
    "age",
    "bmi",
    "waist_circumference",
    "hip_circumference",
    "vat_area",
]
BINARY_COVARIATES = [
    "gender",
    "diabetic",
    "smoking",
    "on_antihypertensive",
    "on_lipid_lowering",
    "vat_missing",
]
MATCH_COVARIATES = CONTINUOUS_COVARIATES + BINARY_COVARIATES
# Filled by prepare_units(); exported so the Methods can state how many units
# the covariate-completeness requirement removes, and why.
COVARIATE_ELIGIBILITY_AUDIT: list[dict] = []
BASE_EXACT_COLS = [
    "pre_stage",
    "post_stage",
    "gender",
    "diabetic",
    "on_lipid_lowering",
]
SMOKING_EXACT_MODALITIES = {"sleep", "hmo", "diet"}
IMPUTE_GROUPS = ["pre_stage", "post_stage", "gender"]

PANELS = {
    "anthropometrics": {
        "source": ("anthropometrics", "anthropometrics"),
        "columns": ["weight", "bmi", "waist_circumference", "hip_circumference"],
    },
    "blood_pressure": {
        "source": ("blood_pressure", "blood_pressure"),
        "columns": [
            "sitting_blood_pressure_systolic",
            "sitting_blood_pressure_diastolic",
            "sitting_blood_pressure_pulse_rate",
        ],
    },
    "sleep": {
        "source": ("sleep", "sleep_hrv"),
        "columns": [
            "heart_rate_mean_during_sleep",
            "heart_rate_mean_during_rem",
            "heart_rate_mean_during_nrem",
            "heart_rate_mean_during_wake",
            "total_sleep_time",
            "total_rem_sleep_time",
            "total_wake_time",
            "quality_score_heart_rate",
            "ahi",
            "odi",
            "ahi_4_percent",
            "ahi_during_rem",
            "odi_during_rem",
            "rdi_during_rem",
            "sleep_efficiency",
            "neurokit_hrv_time_rmssd_during_rem",
            "neurokit_hrv_time_rmssd_during_nrem",
            "neurokit_hrv_time_sdnn_during_rem",
            "neurokit_hrv_time_sdnn_during_nrem",
            "neurokit_hrv_time_pnn20_during_rem",
            "neurokit_hrv_time_pnn20_during_nrem",
            "neurokit_hrv_quality_score_during_rem",
            "neurokit_hrv_quality_score_during_nrem",
        ],
    },
    "cgm": {
        "source": ("CGM", "CGM"),
        "columns": [
            "iglu_mean",
            "iglu_cv",
            "iglu_sd",
            "iglu_mage",
            "iglu_gmi",
            "iglu_in_range_70_180",
            "iglu_above_140",
        ],
    },
    "hmo": {
        "source": ("HMO_blood_tests", "HMO_blood_tests"),
        "columns": [
            "bt_glucose",
            "bt_hba1c",
            "bt_fructosamine",
            "bt_hdl_cholesterol",
            "bt_ldl_cholesterol",
            "bt_non_hdl_cholesterol",
            "bt_total_cholesterol",
            "bt_triglycerides",
        ],
    },
}

MICROBIOME_GENERA = [
    "Bifidobacterium",
    "Blautia",
    "Dorea",
    "Megasphaera",
    "Roseburia",
    "Desulfovibrio",
]


def setup_logging() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUT / "reanalysis.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def exact_columns(modality: str) -> list[str]:
    """Exact blocks chosen by the prespecified retention/balance gate."""
    return BASE_EXACT_COLS + (
        ["smoking"] if modality in SMOKING_EXACT_MODALITIES else []
    )


def _bool(values: pd.Series) -> pd.Series:
    return values.astype("string").str.lower().map({"true": True, "false": False}).fillna(False)


def load_exposure_cohorts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(TRANSITIONS)
    raw = raw[raw["approach"].eq("strict_visit_reported")].copy()
    glp = raw[
        raw["drug_family_or_purpose"].astype(str).str.contains("GLP", case=False, na=False)
    ].copy()
    glp["RegistrationCode"] = normalize_reg(glp["RegistrationCode"]).to_numpy()
    name = glp["medication"].fillna("").astype(str)
    generic = glp["Generic"].fillna("").astype(str)
    is_mounjaro = name.str.contains("mounjaro|מאונג|tirzepatide", case=False, regex=True)
    is_tirzepatide = is_mounjaro | generic.str.contains("tirzepatide", case=False)
    is_semaglutide = (
        generic.str.contains("semaglutide", case=False)
        | name.str.contains("semaglutide|ozempic|wegovy|rybelsus|אוזמפיק|וויגובי", case=False)
    ) & ~is_tirzepatide
    glp["agent_class"] = np.select(
        [is_semaglutide, is_tirzepatide],
        ["semaglutide", "tirzepatide_or_mounjaro_label"],
        default="other_glp1",
    )
    glp["pre_stage"] = glp["prior_absent_visit_stage"].map(STAGE_TO_RESEARCH_STAGE)
    glp["post_stage"] = glp["confirmation_visit_stage"].map(STAGE_TO_RESEARCH_STAGE)
    glp["supported_window"] = glp["pre_stage"].notna() & glp["post_stage"].notna()
    glp["timing_valid"] = (
        _bool(glp["prior_visit_date_valid"])
        & pd.to_numeric(glp["months_under_usage_at_confirmation"], errors="coerce").ge(0)
    )
    glp = glp.sort_values(
        ["RegistrationCode", "prior_absent_visit_stage_order", "confirmation_visit_stage_order"]
    )

    primary = (
        glp[
            glp["agent_class"].eq("semaglutide")
            & glp["supported_window"]
            & glp["timing_valid"]
        ]
        .drop_duplicates("RegistrationCode", keep="first")
        .reset_index(drop=True)
    )
    semaglutide_visit_confirmed = (
        glp[glp["agent_class"].eq("semaglutide") & glp["supported_window"]]
        .drop_duplicates("RegistrationCode", keep="first")
        .reset_index(drop=True)
    )
    tirzepatide = (
        glp[
            glp["agent_class"].eq("tirzepatide_or_mounjaro_label")
            & glp["supported_window"]
            & glp["timing_valid"]
        ]
        .drop_duplicates("RegistrationCode", keep="first")
        .reset_index(drop=True)
    )

    audit_rows = [
        {"cohort": "all_strict_glp_records", "n": glp["RegistrationCode"].nunique()},
        {
            "cohort": "semaglutide_brand_consistent",
            "n": glp.loc[glp["agent_class"].eq("semaglutide"), "RegistrationCode"].nunique(),
        },
        {
            "cohort": "semaglutide_supported_window",
            "n": semaglutide_visit_confirmed["RegistrationCode"].nunique(),
        },
        {
            "cohort": "semaglutide_primary_timing_valid",
            "n": primary["RegistrationCode"].nunique(),
        },
        {
            "cohort": "tirzepatide_or_mounjaro_timing_valid",
            "n": tirzepatide["RegistrationCode"].nunique(),
        },
    ]
    pd.DataFrame(audit_rows).to_csv(OUT / "exposure_audit.csv", index=False)
    glp[
        [
            "RegistrationCode",
            "agent_class",
            "pre_stage",
            "post_stage",
            "supported_window",
            "timing_valid",
            "months_under_usage_at_confirmation",
        ]
    ].to_csv(CACHE / "exposure_record_audit_private.csv", index=False)
    return primary, semaglutide_visit_confirmed, tirzepatide


def ever_glp_participants(primary: pd.DataFrame) -> set[str]:
    """All cached medication users whose dictionary ATC begins A10BJ."""
    module = importlib.import_module("LabData.DataLoaders.Medications10KLoader")
    loader = getattr(module, "Medications10KLoader")()
    loader._qc_dir = "/tmp/glp-paper-reanalysis-medications-qc"
    Path(loader._qc_dir).mkdir(parents=True, exist_ok=True)
    data = loader.get_data()
    dictionary = data.df_columns_metadata.copy()
    glp_names = set(
        dictionary.loc[
            dictionary["ATC"].fillna("").astype(str).str.upper().str.startswith("A10BJ"),
            "column_name",
        ].astype(str)
    )
    frame = data.df.reset_index()
    active = frame["Start"].fillna(False).astype(bool)
    users = set(
        normalize_reg(frame.loc[active & frame["medication"].astype(str).isin(glp_names), "RegistrationCode"])
    )
    # Transition records are authoritative for the index cohort and protect
    # against dictionary misses / spelling drift.
    users.update(primary["RegistrationCode"].astype(str))
    logging.info("Ever-observed A10BJ/transition GLP-1 users excluded from controls: %d", len(users))
    pd.DataFrame([{"metric": "ever_glp_excluded_controls", "n": len(users)}]).to_csv(
        OUT / "control_exclusion_audit.csv", index=False
    )
    return users


def baseline_medication_flags() -> pd.DataFrame:
    """Baseline antihypertensive/lipid flags from the cached LabData loader."""
    module = importlib.import_module("LabData.DataLoaders.Medications10KLoader")
    loader = getattr(module, "Medications10KLoader")()
    loader._qc_dir = "/tmp/glp-paper-reanalysis-baseline-medications-qc"
    Path(loader._qc_dir).mkdir(parents=True, exist_ok=True)
    data = loader.get_data()
    frame = data.df.reset_index()
    frame["research_stage"] = data.df_metadata["research_stage"].to_numpy()
    dictionary = data.df_columns_metadata[["column_name", "ATC"]].drop_duplicates(
        "column_name"
    )
    frame = frame.merge(
        dictionary,
        left_on="medication",
        right_on="column_name",
        how="left",
    )
    frame = frame[
        frame["Start"].fillna(False).astype(bool)
        & frame["research_stage"].astype(str).eq("baseline")
    ].copy()
    frame["participant_id"] = normalize_reg(frame["RegistrationCode"]).to_numpy()
    atc = frame["ATC"].fillna("").astype(str).str.upper()
    frame["on_antihypertensive"] = atc.str.startswith(
        ("C02", "C03", "C07", "C08", "C09")
    )
    frame["on_lipid_lowering"] = atc.str.startswith("C10")
    return (
        frame.groupby("participant_id", as_index=False)[
            ["on_antihypertensive", "on_lipid_lowering"]
        ]
        .max()
        .astype(
            {
                "on_antihypertensive": float,
                "on_lipid_lowering": float,
            }
        )
    )


def read_snapshot(
    dataset: str,
    table: str,
    columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_dir = SNAPSHOT / dataset / table
    frame = pd.read_parquet(table_dir / "df.parquet", columns=list(columns) if columns else None)
    metadata = pd.read_parquet(table_dir / "df_metadata.parquet")
    if len(frame) != len(metadata):
        metadata = metadata.reindex(frame.index)
        if len(frame) != len(metadata):
            raise ValueError(f"Could not align {dataset}/{table} data and metadata")
    return frame, metadata


def load_covariates() -> pd.DataFrame:
    frame, meta = read_snapshot(
        "anthropometrics",
        "anthropometrics",
        ["bmi", "waist_circumference", "hip_circumference"],
    )
    cov = pd.DataFrame(
        {
            "participant_id": normalize_reg(
                frame.index.get_level_values("RegistrationCode")
            ).to_numpy(),
            "research_stage": meta["research_stage"].replace(STAGE_NORMALIZE).to_numpy(),
            "age": pd.to_numeric(meta["age"], errors="coerce").to_numpy() / 365.25,
            "gender": (
                meta["gender"]
                .astype("string")
                .str.lower()
                .map({"female": 1.0, "f": 1.0, "male": 0.0, "m": 0.0})
                .to_numpy()
            ),
            "bmi": pd.to_numeric(frame["bmi"], errors="coerce").to_numpy(),
            "waist_circumference": pd.to_numeric(
                frame["waist_circumference"], errors="coerce"
            ).to_numpy(),
            "hip_circumference": pd.to_numeric(
                frame["hip_circumference"], errors="coerce"
            ).to_numpy(),
        }
    )
    cov = (
        cov[cov["research_stage"].isin(SUPPORTED_STAGES)]
        .groupby(["participant_id", "research_stage"], as_index=False)
        .mean(numeric_only=True)
    )
    legacy = cov.rename(columns={"participant_id": "RegistrationCode"})
    diabetes = confounders.diabetes_flags(legacy, timing="pre_stage")
    vat = confounders.baseline_vat_area()
    smoking = confounders.smoking_status()
    medications = baseline_medication_flags()
    cov = cov.merge(
        diabetes.rename(columns={"RegistrationCode": "participant_id"}),
        on=["participant_id", "research_stage"],
        how="left",
    )
    cov = cov.merge(
        vat.rename(columns={"RegistrationCode": "participant_id"}),
        on="participant_id",
        how="left",
    )
    cov = cov.merge(
        smoking.rename(columns={"RegistrationCode": "participant_id"}),
        on="participant_id",
        how="left",
    )
    cov = cov.merge(medications, on="participant_id", how="left")
    for col in ["diabetic", "smoking", "on_antihypertensive", "on_lipid_lowering"]:
        cov[col] = pd.to_numeric(cov[col], errors="coerce").fillna(0.0)
    return cov


def _participant_and_stage(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["research_stage"] = metadata["research_stage"].replace(STAGE_NORMALIZE).to_numpy()
    if "RegistrationCode" in out.index.names:
        out["participant_id"] = normalize_reg(
            out.index.get_level_values("RegistrationCode")
        ).to_numpy()
    else:
        out["participant_id"] = normalize_reg(metadata["RegistrationCode"]).to_numpy()
    return out[out["research_stage"].isin(SUPPORTED_STAGES)].copy()


def _to_long(frame: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    per_stage = (
        frame.groupby(["participant_id", "research_stage"], as_index=False)[value_cols]
        .mean(numeric_only=True)
    )
    return per_stage.melt(
        id_vars=["participant_id", "research_stage"],
        value_vars=value_cols,
        var_name="measurement",
        value_name="value",
    ).dropna(subset=["value"])


def load_snapshot_panel(modality: str) -> pd.DataFrame:
    spec = PANELS[modality]
    frame, metadata = read_snapshot(*spec["source"], columns=spec["columns"])
    work = _participant_and_stage(frame, metadata)
    columns = list(spec["columns"])
    if modality == "sleep":
        # Require at least five minutes in each state before interpreting its
        # mean heart rate. The WatchPAT summary exposes REM, pooled NREM, and
        # in-study wake; light (N1+N2) and deep (N3) heart-rate means require
        # inaccessible raw channel streams and are not imputed from durations.
        heart_rate_quality = pd.to_numeric(
            work["quality_score_heart_rate"], errors="coerce"
        )
        stage_duration = {
            "rem": pd.to_numeric(work["total_rem_sleep_time"], errors="coerce"),
            "nrem": (
                pd.to_numeric(work["total_sleep_time"], errors="coerce")
                - pd.to_numeric(work["total_rem_sleep_time"], errors="coerce")
            ),
            "wake": pd.to_numeric(work["total_wake_time"], errors="coerce"),
        }
        for segment, duration in stage_duration.items():
            work.loc[
                duration.lt(MIN_HEART_RATE_STAGE_SECONDS)
                | duration.isna()
                | heart_rate_quality.lt(50)
                | heart_rate_quality.isna(),
                f"heart_rate_mean_during_{segment}",
            ] = np.nan
        # Standard per-segment HRV gate. Heart-rate and respiratory features
        # remain available regardless of HRV quality.
        for segment in ("rem", "nrem"):
            quality = pd.to_numeric(
                work[f"neurokit_hrv_quality_score_during_{segment}"], errors="coerce"
            )
            for col in [c for c in columns if c.startswith("neurokit_hrv") and c.endswith(segment)]:
                if "quality_score" not in col:
                    work.loc[quality.lt(50) | quality.isna(), col] = np.nan
        for col in [c for c in columns if c.startswith("heart_rate_")]:
            numeric = pd.to_numeric(work[col], errors="coerce")
            work.loc[~numeric.between(30, 140), col] = np.nan
        for col in [c for c in columns if "_rmssd_" in c or "_sdnn_" in c]:
            numeric = pd.to_numeric(work[col], errors="coerce")
            work.loc[~numeric.between(0, 500), col] = np.nan
        for col in [c for c in columns if "_pnn20_" in c]:
            numeric = pd.to_numeric(work[col], errors="coerce")
            work.loc[~numeric.between(0, 100), col] = np.nan
        work["heart_rate_rem_minus_nrem"] = (
            pd.to_numeric(work["heart_rate_mean_during_rem"], errors="coerce")
            - pd.to_numeric(work["heart_rate_mean_during_nrem"], errors="coerce")
        )
        columns = [
            c
            for c in columns
            if "quality_score" not in c
            and c
            not in {
                "total_sleep_time",
                "total_rem_sleep_time",
                "total_wake_time",
            }
        ] + ["heart_rate_rem_minus_nrem"]
    return _to_long(work, columns)


def load_dxa_panel() -> pd.DataFrame:
    module = importlib.import_module("LabData.DataLoaders.DEXALoader")
    data = getattr(module, "DEXALoader")().get_data()
    columns = [
        "body_comp_total_fat_mass",
        "body_comp_total_lean_mass",
        "body_comp_android_fat_mass",
        "body_comp_gynoid_fat_mass",
        "total_scan_vat_area",
        "total_scan_vat_volume",
    ]
    frame = data.df[columns].copy()
    metadata = data.df_metadata
    work = pd.DataFrame(
        {col: pd.to_numeric(frame[col], errors="coerce").to_numpy() for col in columns}
    )
    work["participant_id"] = normalize_reg(
        pd.Series(frame.index.get_level_values("RegistrationCode"))
    ).to_numpy()
    work["research_stage"] = (
        metadata["research_stage"].replace(STAGE_NORMALIZE).to_numpy()
    )
    work = work[work["research_stage"].isin(SUPPORTED_STAGES)]
    # Known failed scans encode mass outcomes as zero.
    for col in columns:
        work.loc[work[col].eq(0), col] = np.nan
    return _to_long(work, columns)


def load_microbiome_panel() -> pd.DataFrame:
    frame, metadata = read_snapshot("gut_microbiome", "gut_mb_metaphlan_genus")
    work = _participant_and_stage(frame, metadata)
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    matrix = work[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    prevalence = matrix.gt(0).mean()
    reference_cols = prevalence[prevalence.ge(0.10)].index.tolist()
    selected: dict[str, str] = {}
    for genus in MICROBIOME_GENERA:
        hits = [col for col in reference_cols if col.endswith(f"|g__{genus}")]
        if hits:
            selected[genus] = hits[0]
    reference = matrix[reference_cols]
    nonzero = reference.to_numpy()[reference.to_numpy() > 0]
    pseudocount = float(np.quantile(nonzero, 0.01) / 2) if nonzero.size else 1e-6
    log_reference = np.log(reference + pseudocount)
    center = log_reference.mean(axis=1)
    derived = pd.DataFrame(index=work.index)
    for genus, col in selected.items():
        derived[f"microbiome_clr::{genus}"] = log_reference[col] - center
    proportions = reference.div(reference.sum(axis=1).replace(0, np.nan), axis=0)
    derived["microbiome_shannon"] = -(proportions * np.log(proportions.replace(0, np.nan))).sum(
        axis=1
    )
    derived["participant_id"] = work["participant_id"].to_numpy()
    derived["research_stage"] = work["research_stage"].to_numpy()
    logging.info(
        "Microbiome subcomposition: %d prevalence-filtered genera; selected %s; pseudocount %.3g",
        len(reference_cols),
        sorted(selected),
        pseudocount,
    )
    return _to_long(
        derived,
        [c for c in derived.columns if c not in {"participant_id", "research_stage"}],
    )


def load_humann_panel() -> pd.DataFrame:
    """Load the prior 10K-mapped HUMAnN table and apply the historical CLR recipe."""
    if not HUMANN_PATHWAYS.exists():
        raise FileNotFoundError(
            f"Previously used HUMAnN aggregate is unavailable: {HUMANN_PATHWAYS}"
        )
    raw = pd.read_parquet(HUMANN_PATHWAYS)
    required = {"RegistrationCode", "research_stage"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"HUMAnN aggregate missing identifier columns: {sorted(missing)}")
    pathway_columns = [col for col in raw.columns if col not in required]
    matrix = raw[pathway_columns].apply(pd.to_numeric, errors="coerce")
    prevalence = matrix.gt(0).mean(axis=0)
    retained = prevalence[prevalence.ge(0.10)].index.tolist()
    matrix = matrix[retained]
    values = matrix.to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    pseudocount = float(positive.min() / 2.0) if positive.size else 1e-6
    values = np.where(np.isfinite(values) & (values > 0), values, pseudocount)
    logged = np.log(values)
    clr = logged - logged.mean(axis=1, keepdims=True)
    work = pd.DataFrame(
        clr,
        columns=[f"humann_clr::{col}" for col in retained],
    )
    work["participant_id"] = normalize_reg(
        pd.Series(raw["RegistrationCode"].to_numpy(), dtype="string")
    ).to_numpy()
    work["research_stage"] = (
        raw["research_stage"].astype(str).replace(STAGE_NORMALIZE).to_numpy()
    )
    work = work[work["research_stage"].isin(SUPPORTED_STAGES)].copy()
    logging.info(
        "HUMAnN functional pathways: source=%s; retained=%d/%d at >=10%% "
        "prevalence; pseudocount=%.3g",
        HUMANN_PATHWAYS,
        len(retained),
        len(pathway_columns),
        pseudocount,
    )
    return _to_long(
        work,
        [col for col in work.columns if col.startswith("humann_clr::")],
    )


def visit_dates() -> pd.DataFrame:
    frame, metadata = read_snapshot("anthropometrics", "anthropometrics", [])
    dates = pd.DataFrame(
        {
            "participant_id": normalize_reg(
                frame.index.get_level_values("RegistrationCode")
            ).to_numpy(),
            "research_stage": metadata["research_stage"].replace(STAGE_NORMALIZE).to_numpy(),
            "visit_date": pd.to_datetime(
                frame.index.get_level_values("date"), errors="coerce", utc=True
            ).tz_convert(None),
        }
    )
    dates = dates[dates["research_stage"].isin(SUPPORTED_STAGES)].dropna(
        subset=["visit_date"]
    )
    return dates.groupby(["participant_id", "research_stage"], as_index=False)["visit_date"].min()


def load_diet_panel(windows: pd.DataFrame) -> pd.DataFrame:
    cache_path = CACHE / "diet_per_stage_labdata.parquet"
    if cache_path.exists():
        logging.info("Using cached LabData diet-stage aggregate: %s", cache_path)
        return pd.read_parquet(cache_path)

    module = importlib.import_module("LabData.DataLoaders.DietLoggingLoader")
    loader = getattr(module, "DietLoggingLoader")()
    raw = loader.get_data().df
    reset = raw.reset_index()
    reset["participant_id"] = normalize_reg(reset["RegistrationCode"]).to_numpy()
    reset["log_date"] = pd.to_datetime(reset["Date"], errors="coerce", utc=True).dt.tz_convert(
        None
    )
    reset = reset.dropna(subset=["log_date"]).copy()
    dates = visit_dates()
    assigned = pd.merge_asof(
        reset.sort_values("log_date"),
        dates.sort_values("visit_date"),
        left_on="log_date",
        right_on="visit_date",
        by="participant_id",
        direction="nearest",
        tolerance=pd.Timedelta(days=60),
    ).dropna(subset=["research_stage"])
    assigned["log_day"] = assigned["log_date"].dt.date
    day_counts = (
        assigned.groupby(["participant_id", "research_stage"])["log_day"].nunique()
    )
    valid_keys = set(day_counts[day_counts.ge(3)].index)
    window_set = set(map(tuple, windows[["pre_stage", "post_stage"]].drop_duplicates().to_numpy()))
    stages_by_participant = {
        participant: set(group["research_stage"])
        for participant, group in assigned.groupby("participant_id")
    }
    eligible = {
        participant
        for participant, stages in stages_by_participant.items()
        if any(
            (participant, pre) in valid_keys and (participant, post) in valid_keys
            for pre, post in window_set
        )
    }
    assigned = assigned[assigned["participant_id"].isin(eligible)].copy()
    original = assigned.set_index(["RegistrationCode", "Date", "food_id"])[
        ["weight", "unit_id", "meal_type", "score"]
    ]
    nutrients = loader.add_nutrients(original)
    nutrients = loader.add_macronut_kcal(nutrients).reset_index()
    keys = assigned[
        ["RegistrationCode", "Date", "food_id", "participant_id", "research_stage", "log_day"]
    ]
    nutrients = nutrients.merge(keys, on=["RegistrationCode", "Date", "food_id"], how="inner")
    for col in ["energy_kcal", "totallipid_kcal", "carbohydrate_kcal", "protein_kcal"]:
        nutrients[col] = pd.to_numeric(nutrients[col], errors="coerce")
    daily = (
        nutrients.groupby(["participant_id", "research_stage", "log_day"], as_index=False)[
            ["energy_kcal", "totallipid_kcal", "carbohydrate_kcal", "protein_kcal"]
        ]
        .sum(min_count=1)
    )
    per_stage = daily.groupby(["participant_id", "research_stage"], as_index=False).agg(
        energy_kcal_per_day=("energy_kcal", "mean"),
        fat_kcal=("totallipid_kcal", "sum"),
        carb_kcal=("carbohydrate_kcal", "sum"),
        protein_kcal=("protein_kcal", "sum"),
        energy_total=("energy_kcal", "sum"),
        n_diet_days=("log_day", "nunique"),
    )
    per_stage = per_stage[per_stage["n_diet_days"].ge(3) & per_stage["energy_total"].gt(0)]
    per_stage["pct_fat_calories"] = 100 * per_stage["fat_kcal"] / per_stage["energy_total"]
    per_stage["pct_carb_calories"] = 100 * per_stage["carb_kcal"] / per_stage["energy_total"]
    per_stage["pct_protein_calories"] = (
        100 * per_stage["protein_kcal"] / per_stage["energy_total"]
    )
    columns = [
        "energy_kcal_per_day",
        "pct_fat_calories",
        "pct_carb_calories",
        "pct_protein_calories",
    ]
    long = per_stage.melt(
        id_vars=["participant_id", "research_stage"],
        value_vars=columns,
        var_name="measurement",
        value_name="value",
    ).dropna(subset=["value"])
    long.to_parquet(cache_path, index=False)
    return long


def outcome_availability(values: pd.DataFrame) -> pd.Series:
    return values.groupby("participant_id")["research_stage"].apply(set)


def prepare_units(
    values: pd.DataFrame,
    treated: pd.DataFrame,
    covariates: pd.DataFrame,
    excluded_controls: set[str],
    audit_label: str | None = None,
) -> pd.DataFrame:
    availability = outcome_availability(values)
    treated_ids = set(treated["RegistrationCode"])
    treated_windows = set(
        map(tuple, treated[["RegistrationCode", "pre_stage", "post_stage"]].to_numpy())
    )
    rows = []
    for pre, post in treated[["pre_stage", "post_stage"]].drop_duplicates().itertuples(index=False):
        eligible = [
            participant
            for participant, stages in availability.items()
            if pre in stages and post in stages
        ]
        for participant in eligible:
            if participant in treated_ids:
                if (participant, pre, post) not in treated_windows:
                    continue
                arm = 1
            else:
                if participant in excluded_controls:
                    continue
                arm = 0
            rows.append(
                {
                    "participant_id": participant,
                    "pre_stage": pre,
                    "post_stage": post,
                    "treated": arm,
                }
            )
    units = pd.DataFrame(rows)
    pre_cov = covariates.rename(columns={"research_stage": "pre_stage"})
    units = units.merge(pre_cov, on=["participant_id", "pre_stage"], how="left")
    units["vat_missing"] = units["vat_area"].isna().astype(float)
    required = ["age", "gender", "bmi", "waist_circumference", "hip_circumference"]
    missing_sex = units["gender"].isna()
    missing_other = units[required].isna().any(axis=1) & ~missing_sex
    audit = {
        "label": audit_label or "unlabelled",
        "unit_rows_before": int(len(units)),
        "participants_before": int(units["participant_id"].nunique()),
        "unit_rows_dropped_missing_sex": int(missing_sex.sum()),
        "participants_dropped_missing_sex": int(
            units.loc[missing_sex, "participant_id"].nunique()
        ),
        "treated_rows_dropped_missing_sex": int(
            units.loc[missing_sex, "treated"].sum()
        ),
        "unit_rows_dropped_other_covariate": int(missing_other.sum()),
    }
    units = units.dropna(subset=required).copy()
    audit["unit_rows_after"] = int(len(units))
    audit["participants_after"] = int(units["participant_id"].nunique())
    COVARIATE_ELIGIBILITY_AUDIT.append(audit)
    # Symmetric imputation—same rule in both arms, within sex × window where possible.
    units["vat_area"] = units.groupby(IMPUTE_GROUPS, dropna=False)["vat_area"].transform(
        lambda x: x.fillna(x.median())
    )
    units["vat_area"] = units["vat_area"].fillna(units["vat_area"].median())
    units = units.dropna(subset=MATCH_COVARIATES).reset_index(drop=True)
    if units["treated"].nunique() < 2:
        return units
    return propensity_score(
        units,
        treatment_col="treated",
        covariate_cols=MATCH_COVARIATES,
    )


def unit_deltas(values: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    wide = values.pivot_table(
        index=["participant_id", "measurement"],
        columns="research_stage",
        values="value",
        aggfunc="mean",
    )
    records = []
    for unit in units[
        ["participant_id", "pre_stage", "post_stage"]
    ].drop_duplicates().itertuples(index=False):
        try:
            participant = wide.loc[unit.participant_id]
        except KeyError:
            continue
        for measurement, row in participant.iterrows():
            pre_value = row.get(unit.pre_stage, np.nan)
            post_value = row.get(unit.post_stage, np.nan)
            if pd.isna(pre_value) or pd.isna(post_value):
                continue
            records.append(
                {
                    "participant_id": unit.participant_id,
                    "pre_stage": unit.pre_stage,
                    "post_stage": unit.post_stage,
                    "measurement": measurement,
                    "pre_value": float(pre_value),
                    "post_value": float(post_value),
                    "delta": float(post_value - pre_value),
                }
            )
    return pd.DataFrame(records)


def pair_effects(pairs: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    keys = ["participant_id", "pre_stage", "post_stage", "measurement"]
    treated = deltas.rename(
        columns={
            "participant_id": "treated_id",
            "pre_value": "treated_pre",
            "post_value": "treated_post",
            "delta": "treated_delta",
        }
    )
    control = deltas.rename(
        columns={
            "participant_id": "control_id",
            "pre_value": "control_pre",
            "post_value": "control_post",
            "delta": "control_delta",
        }
    )
    out = pairs.merge(
        treated,
        on=["treated_id", "pre_stage", "post_stage"],
        how="inner",
    )
    out = out.merge(
        control,
        on=["control_id", "pre_stage", "post_stage", "measurement"],
        how="inner",
    )
    out["did"] = out["treated_delta"] - out["control_delta"]
    return out


def _tost(differences: np.ndarray, margin: float) -> float:
    differences = differences[np.isfinite(differences)]
    if len(differences) < 3 or not np.isfinite(margin) or margin <= 0:
        return np.nan
    se = differences.std(ddof=1) / np.sqrt(len(differences))
    if se == 0:
        return 0.0 if abs(differences.mean()) < margin else 1.0
    dof = len(differences) - 1
    p_lower = stats.t.sf((differences.mean() + margin) / se, dof)
    p_upper = stats.t.cdf((differences.mean() - margin) / se, dof)
    return float(max(p_lower, p_upper))


def fit_hc3(
    y: Iterable[float],
    x: np.ndarray,
    weights: Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least squares with an HC3 sandwich covariance.

    ``x`` must include any intercept column. Frequency/precision weights are
    applied through the standard square-root transformation before computing
    leverage and the HC3 meat.
    """
    outcome = np.asarray(y, dtype=float)
    design = np.asarray(x, dtype=float)
    weight = (
        np.ones(len(outcome), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    valid = (
        np.isfinite(outcome)
        & np.isfinite(weight)
        & (weight > 0)
        & np.isfinite(design).all(axis=1)
    )
    outcome, design, weight = outcome[valid], design[valid], weight[valid]
    if len(outcome) <= design.shape[1]:
        nan = np.full(design.shape[1], np.nan)
        return nan, nan, nan
    root_weight = np.sqrt(weight)
    xw = design * root_weight[:, None]
    yw = outcome * root_weight
    bread = np.linalg.pinv(xw.T @ xw)
    beta = bread @ xw.T @ yw
    transformed_residual = yw - xw @ beta
    leverage = np.einsum("ij,jk,ik->i", xw, bread, xw)
    adjusted = transformed_residual / np.clip(1 - leverage, 1e-8, None)
    meat = xw.T @ ((adjusted**2)[:, None] * xw)
    covariance = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    dof = max(len(outcome) - design.shape[1], 1)
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    return beta, se, p


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean, matching the contrast's method."""
    values = values[np.isfinite(values)]
    if len(values) < MIN_N:
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_primary(pair_rows: pd.DataFrame, modality: str) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    # A SEPARATE stream for the per-arm intervals. Drawing them from `rng` would shift
    # every subsequent contrast bootstrap and silently move already-published CIs.
    arm_rng = np.random.default_rng(SEED + 1)
    records = []
    for measurement, group in pair_rows.groupby("measurement"):
        differences = group["did"].to_numpy(float)
        differences = differences[np.isfinite(differences)]
        if len(differences) < MIN_N:
            continue
        bootstrap = rng.choice(
            differences,
            size=(N_BOOT, len(differences)),
            replace=True,
        ).mean(axis=1)
        treated_pre = group["treated_pre"].to_numpy(float)
        control_pre = group["control_pre"].to_numpy(float)
        baseline = np.r_[treated_pre, control_pre]
        margin = 0.2 * np.nanstd(baseline, ddof=1)
        # Pre-treatment balance on the OUTCOME itself. The covariate balance table
        # cannot carry this: an outcome is not a matching covariate, so a claim like
        # "baseline sleep heart rate is balanced to |SMD| 0.001" had no source file and
        # no figure to cite. Exported 2026-09-06 so it has both.
        pooled_sd = np.sqrt(
            (np.nanvar(treated_pre, ddof=1) + np.nanvar(control_pre, ddof=1)) / 2
        )
        baseline_smd = (
            float((np.nanmean(treated_pre) - np.nanmean(control_pre)) / pooled_sd)
            if pooled_sd > 0
            else 0.0
        )
        try:
            wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
        except ValueError:
            wilcoxon_p = np.nan
        # Each arm's own change gets the same bootstrap the contrast gets. Until
        # 2026-09-04 these were bare means with no interval, which is why the
        # arm-decomposition figure could only be drawn as a dumbbell -- it had no
        # uncertainty to plot, and a dumbbell silently implies both ends are precise.
        arm_intervals = {
            column: _bootstrap_mean_ci(group[column].to_numpy(float), arm_rng)
            for column in ("treated_delta", "control_delta")
        }
        records.append(
            {
                "modality": modality,
                "measurement": measurement,
                "n_pairs": len(differences),
                "treated_pre_mean": float(np.nanmean(treated_pre)),
                "control_pre_mean": float(np.nanmean(control_pre)),
                "baseline_smd": baseline_smd,
                "treated_delta": group["treated_delta"].mean(),
                "treated_delta_ci_low": arm_intervals["treated_delta"][0],
                "treated_delta_ci_high": arm_intervals["treated_delta"][1],
                "control_delta": group["control_delta"].mean(),
                "control_delta_ci_low": arm_intervals["control_delta"][0],
                "control_delta_ci_high": arm_intervals["control_delta"][1],
                "effect": differences.mean(),
                "ci_low": np.quantile(bootstrap, 0.025),
                "ci_high": np.quantile(bootstrap, 0.975),
                "paired_t_p": stats.ttest_1samp(differences, 0).pvalue,
                "wilcoxon_p": wilcoxon_p,
                "equivalence_margin_0p2_baseline_sd": margin,
                "equivalence_p": _tost(differences, margin),
                "equivalent_small_effect": _tost(differences, margin) < 0.05,
            }
        )
    result = pd.DataFrame(records)
    if not result.empty:
        result["fdr_within_modality"] = bh_fdr(result["paired_t_p"].to_numpy())
    return result


STAGE_YEAR = {
    "00_00_visit": 0.0,
    "02_00_visit": 2.0,
    "04_00_visit": 4.0,
    "06_00_visit": 6.0,
}


def untreated_drift(
    deltas: pd.DataFrame,
    units: pd.DataFrame,
    pairs: pd.DataFrame,
    modality: str,
) -> pd.DataFrame:
    """How much every never-GLP-1 participant changes over the same visit windows.

    The matched control arm is 44 people. Asking whether their change is "just what
    happens over two years" needs a reference the design does not already condition on,
    so this takes the WHOLE eligible control pool -- thousands of participants, same
    modality, same instrument -- and reweights it to the (pre_stage, post_stage)
    distribution the matched pairs actually used. Without that reweighting the pool
    average would mix intervals the matched pairs never spanned.

    Intervals are the nominal visit spacing encoded in the stage label (02_00_visit is
    the two-year visit), not per-participant elapsed days, which the unit table does not
    carry. `interval_years` is reported so the figure can state it rather than imply it.
    """
    if pairs.empty or deltas.empty:
        return pd.DataFrame()
    window_weight = (
        pairs.groupby(["pre_stage", "post_stage"]).size().rename("window_weight")
    )
    window_weight = window_weight / window_weight.sum()
    controls = set(units.loc[units["treated"].eq(0), "participant_id"])
    pool = deltas[deltas["participant_id"].isin(controls)].merge(
        window_weight, left_on=["pre_stage", "post_stage"], right_index=True, how="inner"
    )
    if pool.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(SEED + 2)
    records = []
    for measurement, group in pool.groupby("measurement"):
        # Every participant in a window carries that window's weight, so the estimator
        # is a weighted sum of per-window means. Resampling participants WITHIN each
        # window and recombining gives the same interval as resampling the pooled
        # sample, at a fraction of the memory -- the pool runs to five figures of rows.
        windows, means, weights = [], [], []
        for (pre, post), block in group.groupby(["pre_stage", "post_stage"]):
            values = (
                block.groupby("participant_id")["delta"].mean().to_numpy(float)
            )
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            windows.append((values, STAGE_YEAR[post] - STAGE_YEAR[pre]))
            means.append(values.mean())
            weights.append(float(block["window_weight"].iloc[0]))
        if not windows:
            continue
        n_participants = int(group["participant_id"].nunique())
        if n_participants < MIN_N:
            continue
        weights = np.array(weights) / np.sum(weights)
        drift = float(np.dot(weights, means))
        bootstrap = np.zeros(N_BOOT)
        for weight, (values, _) in zip(weights, windows, strict=True):
            draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True)
            bootstrap += weight * draws.mean(axis=1)
        years = float(np.dot(weights, [span for _, span in windows]))
        records.append(
            {
                "modality": modality,
                "measurement": measurement,
                "n_participants": n_participants,
                "interval_years": years,
                "drift": drift,
                "drift_ci_low": float(np.quantile(bootstrap, 0.025)),
                "drift_ci_high": float(np.quantile(bootstrap, 0.975)),
                "drift_per_year": drift / years if years else np.nan,
            }
        )
    return pd.DataFrame(records)


def add_hierarchical_multiplicity(
    results: pd.DataFrame,
    *,
    p_column: str,
) -> pd.DataFrame:
    """Gate endpoint discoveries by Simes modality tests and BH across modalities."""
    if results.empty:
        return results
    result = results.copy()
    gate_p: dict[str, float] = {}
    for modality, group in result.groupby("modality"):
        values = group[p_column].dropna().sort_values().to_numpy(float)
        if len(values):
            ranks = np.arange(1, len(values) + 1)
            gate_p[modality] = float(np.minimum(1, np.min(len(values) * values / ranks)))
        else:
            gate_p[modality] = np.nan
    valid_modalities = [key for key, value in gate_p.items() if np.isfinite(value)]
    gate_fdr = dict.fromkeys(gate_p, np.nan)
    if valid_modalities:
        adjusted = bh_fdr(np.array([gate_p[key] for key in valid_modalities]))
        gate_fdr.update(dict(zip(valid_modalities, adjusted, strict=True)))
    result["modality_gate_p"] = result["modality"].map(gate_p)
    result["modality_gate_fdr"] = result["modality"].map(gate_fdr)
    result["hierarchical_significant"] = (
        result["modality_gate_fdr"].lt(0.05)
        & result["fdr_within_modality"].lt(0.05)
    )
    return result


def weighted_effect(
    pairs: pd.DataFrame,
    deltas: pd.DataFrame,
    *,
    design: str,
    modality: str,
) -> pd.DataFrame:
    links = pair_effects(pairs, deltas)
    records = []
    for measurement, group in links.groupby("measurement"):
        treated = (
            group[["treated_id", "treated_delta"]]
            .drop_duplicates("treated_id")
            .rename(columns={"treated_id": "participant_id", "treated_delta": "outcome"})
        )
        treated["treated"] = 1.0
        treated["weight"] = 1.0
        group = group.copy()
        group["link_weight"] = group.get("match_weight", 1.0)
        controls = (
            group.groupby("control_id")
            .apply(
                lambda x: pd.Series(
                    {
                        "outcome": np.average(x["control_delta"], weights=x["link_weight"]),
                        "weight": x["link_weight"].sum(),
                    }
                )
            )
            .reset_index()
            .rename(columns={"control_id": "participant_id"})
        )
        controls["treated"] = 0.0
        sample = pd.concat([treated, controls], ignore_index=True).dropna()
        if treated.shape[0] < MIN_N or controls.shape[0] < MIN_N:
            continue
        design_matrix = np.column_stack([np.ones(len(sample)), sample["treated"]])
        beta, standard_error, p_value = fit_hc3(
            sample["outcome"], design_matrix, sample["weight"]
        )
        estimate = float(beta[1])
        se = float(standard_error[1])
        records.append(
            {
                "design": design,
                "modality": modality,
                "measurement": measurement,
                "n_treated": treated.shape[0],
                "n_controls": controls.shape[0],
                "effect": estimate,
                "ci_low": estimate - 1.96 * se,
                "ci_high": estimate + 1.96 * se,
                "p": float(p_value[1]),
            }
        )
    result = pd.DataFrame(records)
    if not result.empty:
        result["fdr_within_modality"] = bh_fdr(result["p"].to_numpy())
    return result


def overlap_effect(units: pd.DataFrame, deltas: pd.DataFrame, modality: str) -> pd.DataFrame:
    merged = deltas.merge(
        units[
            ["participant_id", "pre_stage", "post_stage", "treated", "propensity"]
        ].drop_duplicates(),
        on=["participant_id", "pre_stage", "post_stage"],
        how="inner",
    )
    merged["weight"] = np.where(
        merged["treated"].eq(1), 1 - merged["propensity"], merged["propensity"]
    )
    records = []
    for measurement, group in merged.groupby("measurement"):
        # Collapse repeated control windows to one independent participant.
        group = group.copy()
        group["weighted_delta"] = group["delta"] * group["weight"]
        sample = (
            group.groupby(["participant_id", "treated"], as_index=False)
            .agg(
                weighted_delta=("weighted_delta", "sum"),
                weight=("weight", "sum"),
            )
        )
        sample["delta"] = sample["weighted_delta"] / sample["weight"]
        if sample.loc[sample["treated"].eq(1), "participant_id"].nunique() < MIN_N:
            continue
        design_matrix = np.column_stack([np.ones(len(sample)), sample["treated"]])
        beta, standard_error, p_value = fit_hc3(
            sample["delta"], design_matrix, sample["weight"]
        )
        estimate = float(beta[1])
        se = float(standard_error[1])
        records.append(
            {
                "design": "overlap_weighted",
                "modality": modality,
                "measurement": measurement,
                "n_treated": sample["treated"].eq(1).sum(),
                "n_controls": sample["treated"].eq(0).sum(),
                "effect": estimate,
                "ci_low": estimate - 1.96 * se,
                "ci_high": estimate + 1.96 * se,
                "p": float(p_value[1]),
            }
        )
    result = pd.DataFrame(records)
    if not result.empty:
        result["fdr_within_modality"] = bh_fdr(result["p"].to_numpy())
    return result


def balance_table(units: pd.DataFrame, pairs: pd.DataFrame, modality: str) -> pd.DataFrame:
    records = []
    for covariate in MATCH_COVARIATES:
        before_t = units.loc[units["treated"].eq(1), covariate]
        before_c = units.loc[units["treated"].eq(0), covariate]
        after_t = pairs[f"treated_{covariate}"]
        after_c = pairs[f"control_{covariate}"]

        def smd(a: pd.Series, b: pd.Series) -> float:
            denominator = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            return float((a.mean() - b.mean()) / denominator) if denominator > 0 else 0.0

        records.append(
            {
                "modality": modality,
                "covariate": covariate,
                "smd_before": smd(before_t, before_c),
                "smd_after": smd(after_t, after_c),
                # Arm-level summaries so the paper can carry a baseline
                # characteristics table, not just standardized differences.
                "n_before_treated": int(before_t.notna().sum()),
                "n_before_control": int(before_c.notna().sum()),
                "before_treated_mean": float(before_t.mean()),
                "before_treated_sd": float(before_t.std(ddof=1)),
                "before_control_mean": float(before_c.mean()),
                "before_control_sd": float(before_c.std(ddof=1)),
                "n_after_treated": int(after_t.notna().sum()),
                "n_after_control": int(after_c.notna().sum()),
                "after_treated_mean": float(after_t.mean()),
                "after_treated_sd": float(after_t.std(ddof=1)),
                "after_control_mean": float(after_c.mean()),
                "after_control_sd": float(after_c.std(ddof=1)),
            }
        )
    return pd.DataFrame(records)


def overlap_balance_table(units: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Participant-collapsed weighted balance for the overlap target population."""
    frame = units[
        ["participant_id", "treated", "propensity", *MATCH_COVARIATES]
    ].copy()
    frame["weight"] = np.where(
        frame["treated"].eq(1),
        1 - frame["propensity"],
        frame["propensity"],
    )
    records = []
    for covariate in MATCH_COVARIATES:
        work = frame[
            ["participant_id", "treated", "weight", covariate]
        ].dropna().copy()
        work["weighted_value"] = work["weight"] * work[covariate]
        collapsed = work.groupby(
            ["participant_id", "treated"],
            as_index=False,
        ).agg(
            weight=("weight", "sum"),
            weighted_value=("weighted_value", "sum"),
        )
        collapsed["value"] = collapsed["weighted_value"] / collapsed["weight"]

        def moments(arm: int) -> tuple[float, float]:
            group = collapsed[collapsed["treated"].eq(arm)]
            values = group["value"].to_numpy(float)
            weights = group["weight"].to_numpy(float)
            mean = float(np.average(values, weights=weights))
            variance = float(np.average((values - mean) ** 2, weights=weights))
            return mean, variance

        treated_mean, treated_variance = moments(1)
        control_mean, control_variance = moments(0)
        denominator = np.sqrt((treated_variance + control_variance) / 2)
        records.append(
            {
                "modality": modality,
                "covariate": covariate,
                "smd_before": np.nan,
                "smd_after": (
                    (treated_mean - control_mean) / denominator
                    if denominator > 0
                    else 0.0
                ),
                "design": "overlap_weighted",
            }
        )
    return pd.DataFrame(records)


def duration_models(
    pair_rows: pd.DataFrame,
    exposure: pd.DataFrame,
    modality: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    duration = exposure[
        ["RegistrationCode", "months_under_usage_at_confirmation"]
    ].rename(
        columns={
            "RegistrationCode": "treated_id",
            "months_under_usage_at_confirmation": "months",
        }
    )
    frame = pair_rows.merge(duration, on="treated_id", how="left")
    model_rows, band_rows = [], []
    for measurement, group in frame.groupby("measurement"):
        group = group.dropna(subset=["did", "months"])
        if len(group) >= 12 and group["months"].std(ddof=1) > 0:
            design_matrix = np.column_stack([np.ones(len(group)), group["months"]])
            beta, standard_error, p_value = fit_hc3(group["did"], design_matrix)
            slope = float(beta[1])
            se = float(standard_error[1])
            model_rows.append(
                {
                    "modality": modality,
                    "measurement": measurement,
                    "n": len(group),
                    "effect_per_month": slope,
                    "ci_low": slope - 1.96 * se,
                    "ci_high": slope + 1.96 * se,
                    "p": float(p_value[1]),
                }
            )
        interpretable = group[group["months"].ge(3)].copy()
        interpretable["duration_band"] = pd.cut(
            interpretable["months"],
            [3, 6, 12, np.inf],
            labels=["3-6 months", "6-12 months", ">12 months"],
            include_lowest=True,
        )
        for band, subgroup in interpretable.groupby("duration_band", observed=True):
            if len(subgroup) < 4:
                continue
            effect = subgroup["did"].mean()
            se = subgroup["did"].std(ddof=1) / np.sqrt(len(subgroup))
            tcrit = stats.t.ppf(0.975, len(subgroup) - 1)
            band_rows.append(
                {
                    "modality": modality,
                    "measurement": measurement,
                    "duration_band": str(band),
                    "n": len(subgroup),
                    "effect": effect,
                    "ci_low": effect - tcrit * se,
                    "ci_high": effect + tcrit * se,
                }
            )
    model_result = pd.DataFrame(model_rows)
    if not model_result.empty:
        model_result["fdr_within_modality"] = bh_fdr(
            model_result["p"].to_numpy()
        )
    return model_result, pd.DataFrame(band_rows)


def contrast_model(
    pair_rows: pd.DataFrame,
    measurement_a: str,
    measurement_b: str,
    label: str,
) -> dict:
    a = pair_rows[pair_rows["measurement"].eq(measurement_a)][["treated_id", "did"]].rename(
        columns={"did": "a"}
    )
    b = pair_rows[pair_rows["measurement"].eq(measurement_b)][["treated_id", "did"]].rename(
        columns={"did": "b"}
    )
    merged = a.merge(b, on="treated_id", how="inner").dropna()
    difference = merged["a"] - merged["b"]
    if len(difference) < MIN_N:
        return {"contrast": label, "n": len(difference)}
    se = difference.std(ddof=1) / np.sqrt(len(difference))
    tcrit = stats.t.ppf(0.975, len(difference) - 1)
    return {
        "contrast": label,
        "n": len(difference),
        "effect_a": merged["a"].mean(),
        "effect_b": merged["b"].mean(),
        "difference": difference.mean(),
        "ci_low": difference.mean() - tcrit * se,
        "ci_high": difference.mean() + tcrit * se,
        "p": stats.ttest_1samp(difference, 0).pvalue,
    }


def attenuation_model(
    outcome_pairs: pd.DataFrame,
    covariate_pairs: pd.DataFrame,
    outcome: str,
    covariate: str,
    label: str,
) -> dict:
    y = outcome_pairs[outcome_pairs["measurement"].eq(outcome)][
        ["treated_id", "did"]
    ].rename(columns={"did": "outcome_did"})
    x = covariate_pairs[covariate_pairs["measurement"].eq(covariate)][
        ["treated_id", "did"]
    ].rename(columns={"did": "covariate_did"})
    merged = y.merge(x, on="treated_id", how="inner").dropna()
    if len(merged) < 12 or merged["covariate_did"].std(ddof=1) == 0:
        return {"model": label, "n": len(merged)}
    design_matrix = np.column_stack([np.ones(len(merged)), merged["covariate_did"]])
    beta, standard_error, p_value = fit_hc3(merged["outcome_did"], design_matrix)
    intercept = float(beta[0])
    unadjusted = float(merged["outcome_did"].mean())
    return {
        "model": label,
        "n": len(merged),
        "unadjusted_pair_did": unadjusted,
        "covariate_adjusted_intercept": intercept,
        "intercept_ci_low": intercept - 1.96 * float(standard_error[0]),
        "intercept_ci_high": intercept + 1.96 * float(standard_error[0]),
        "intercept_p": float(p_value[0]),
        "covariate_slope": float(beta[1]),
        "covariate_slope_p": float(p_value[1]),
        "descriptive_change_after_adjustment": intercept - unadjusted,
    }


def analyse_modality(
    modality: str,
    values: pd.DataFrame,
    exposure: pd.DataFrame,
    covariates: pd.DataFrame,
    ever_glp: set[str],
) -> dict:
    logging.info("%s: building outcome-eligible cohort", modality)
    units = prepare_units(values, exposure, covariates, ever_glp, audit_label=modality)
    if units.empty or units["treated"].sum() == 0:
        logging.warning("%s: no eligible treated units", modality)
        return {}
    primary_pairs = optimal_pair_match(
        units,
        covariate_cols=MATCH_COVARIATES,
        treatment_col="treated",
        id_col="participant_id",
        exact_cols=exact_columns(modality),
        caliper_sd=CALIPER_SD,
        seed=SEED,
    )
    capped_pairs = capacity_caliper_match(
        units,
        covariate_cols=MATCH_COVARIATES,
        treatment_col="treated",
        id_col="participant_id",
        exact_cols=exact_columns(modality),
        max_ratio=3,
        control_capacity=2,
        caliper_sd=CALIPER_SD,
        seed=SEED,
    )
    deltas = unit_deltas(values, units)
    primary_rows = pair_effects(primary_pairs, deltas)
    primary_result = summarize_primary(primary_rows, modality)
    capped_result = weighted_effect(
        capped_pairs,
        deltas,
        design="capped_1_to_3_max_reuse_2",
        modality=modality,
    )
    overlap_result = overlap_effect(units, deltas, modality)
    balance = balance_table(units, primary_pairs, modality)
    balance["design"] = "primary_optimal_1to1"
    overlap_balance = overlap_balance_table(units, modality)
    max_abs_smd_primary = float(balance["smd_after"].abs().max())
    max_abs_smd_overlap = float(overlap_balance["smd_after"].abs().max())
    max_primary_reuse = (
        int(primary_pairs.groupby("control_id").size().max()) if not primary_pairs.empty else 0
    )
    max_capped_reuse = (
        int(capped_pairs.groupby("control_id").size().max()) if not capped_pairs.empty else 0
    )
    diagnostics = {
        "modality": modality,
        "treated_outcome_eligible": int(units["treated"].sum()),
        "control_units_eligible": int(units["treated"].eq(0).sum()),
        "primary_matched_treated": int(primary_pairs["treated_id"].nunique()),
        "primary_retention": float(
            primary_pairs["treated_id"].nunique() / max(units["treated"].sum(), 1)
        ),
        "primary_unique_controls": int(primary_pairs["control_id"].nunique()),
        "primary_max_control_reuse": max_primary_reuse,
        "primary_out_of_caliper": int((~primary_pairs["within_caliper"]).sum()),
        "capped_matched_treated": int(capped_pairs["treated_id"].nunique()),
        "capped_unique_controls": int(capped_pairs["control_id"].nunique()),
        "capped_max_control_reuse": max_capped_reuse,
        "max_abs_smd_primary": max_abs_smd_primary,
        "max_abs_smd_overlap": max_abs_smd_overlap,
        "exact_blocking": "+".join(exact_columns(modality)),
    }
    diagnostics["design_gate_pass"] = bool(
        diagnostics["primary_retention"] >= MIN_MATCH_RETENTION
        and diagnostics["max_abs_smd_primary"] < MAX_ABS_SMD
        and diagnostics["primary_out_of_caliper"] == 0
    )
    if not primary_result.empty:
        primary_result["design_gate_pass"] = diagnostics["design_gate_pass"]
    diagnostics["overlap_design_gate_pass"] = bool(
        diagnostics["max_abs_smd_overlap"] < MAX_ABS_SMD
    )
    if not overlap_result.empty:
        overlap_result["design_gate_pass"] = diagnostics[
            "overlap_design_gate_pass"
        ]
    logging.info("%s diagnostics: %s", modality, diagnostics)
    return {
        "units": units,
        "deltas": deltas,
        "primary_pairs": primary_pairs,
        "primary_rows": primary_rows,
        "primary_result": primary_result,
        # Skipped for microbiome_function: 272 pathways x a five-figure control pool is
        # minutes of bootstrap for a reference nothing quotes. Those pathways are
        # table-only and fail the design gate anyway.
        "untreated_drift": (
            pd.DataFrame()
            if modality == "microbiome_function"
            else untreated_drift(deltas, units, primary_pairs, modality)
        ),
        "capped_result": capped_result,
        "overlap_result": overlap_result,
        "balance": pd.concat([balance, overlap_balance], ignore_index=True),
        "diagnostics": diagnostics,
    }


def active_comparator_cohort(raw_exposure: pd.DataFrame) -> pd.DataFrame:
    transitions = pd.read_csv(TRANSITIONS)
    transitions = transitions[transitions["approach"].eq("strict_visit_reported")].copy()
    family = transitions["drug_family_or_purpose"].fillna("").astype(str)
    comparator = transitions[
        family.str.contains("non-glp diabetes|diabetes", case=False)
        & ~family.str.contains("glp", case=False)
    ].copy()
    comparator["RegistrationCode"] = normalize_reg(comparator["RegistrationCode"]).to_numpy()
    comparator["pre_stage"] = comparator["prior_absent_visit_stage"].map(STAGE_TO_RESEARCH_STAGE)
    comparator["post_stage"] = comparator["confirmation_visit_stage"].map(STAGE_TO_RESEARCH_STAGE)
    comparator = comparator[
        comparator["pre_stage"].notna()
        & comparator["post_stage"].notna()
        & _bool(comparator["prior_visit_date_valid"])
        & pd.to_numeric(
            comparator["months_under_usage_at_confirmation"], errors="coerce"
        ).ge(0)
        & ~comparator["RegistrationCode"].isin(raw_exposure["RegistrationCode"])
    ]
    return (
        comparator.sort_values(
            ["RegistrationCode", "prior_absent_visit_stage_order", "confirmation_visit_stage_order"]
        )
        .drop_duplicates("RegistrationCode")
        .reset_index(drop=True)
    )


def active_comparator_analysis(
    values: pd.DataFrame,
    primary: pd.DataFrame,
    comparator: pd.DataFrame,
    covariates: pd.DataFrame,
    modality: str,
) -> pd.DataFrame:
    availability = outcome_availability(values)
    rows = []
    for arm, cohort in [(1, primary), (0, comparator)]:
        for row in cohort.itertuples(index=False):
            stages = availability.get(row.RegistrationCode, set())
            if row.pre_stage in stages and row.post_stage in stages:
                rows.append(
                    {
                        "participant_id": row.RegistrationCode,
                        "pre_stage": row.pre_stage,
                        "post_stage": row.post_stage,
                        "treated": arm,
                    }
                )
    units = pd.DataFrame(rows)
    if units.empty or units["treated"].nunique() < 2:
        return pd.DataFrame()
    units = units.merge(
        covariates.rename(columns={"research_stage": "pre_stage"}),
        on=["participant_id", "pre_stage"],
        how="left",
    )
    units["vat_missing"] = units["vat_area"].isna().astype(float)
    units["vat_area"] = units.groupby(IMPUTE_GROUPS, dropna=False)["vat_area"].transform(
        lambda x: x.fillna(x.median())
    )
    units["vat_area"] = units["vat_area"].fillna(units["vat_area"].median())
    units = units.dropna(subset=MATCH_COVARIATES)
    if units["treated"].nunique() < 2:
        return pd.DataFrame()
    units = propensity_score(
        units,
        treatment_col="treated",
        covariate_cols=MATCH_COVARIATES,
    )
    pairs = optimal_pair_match(
        units,
        covariate_cols=MATCH_COVARIATES,
        treatment_col="treated",
        id_col="participant_id",
        exact_cols=exact_columns(modality),
        caliper_sd=CALIPER_SD,
        seed=SEED,
    )
    if pairs.empty:
        return pd.DataFrame()
    deltas = unit_deltas(values, units)
    result = summarize_primary(pair_effects(pairs, deltas), modality)
    if not result.empty:
        balance = balance_table(units, pairs, modality)
        retention = pairs["treated_id"].nunique() / max(units["treated"].sum(), 1)
        max_abs_smd = float(balance["smd_after"].abs().max())
        result["design"] = "active_comparator_non_glp_diabetes_new_user"
        result["matched_comparator_pairs"] = pairs["treated_id"].nunique()
        result["outcome_eligible_n"] = int(units["treated"].sum())
        result["retention"] = retention
        result["max_abs_smd"] = max_abs_smd
        result["design_gate_pass"] = bool(
            retention >= MIN_MATCH_RETENTION and max_abs_smd < MAX_ABS_SMD
        )
    return result


def exposure_sensitivity_analysis(
    values: pd.DataFrame,
    exposure: pd.DataFrame,
    covariates: pd.DataFrame,
    ever_glp: set[str],
    modality: str,
    cohort_label: str,
) -> pd.DataFrame:
    """Primary 1:1 design under an alternate exposure adjudication."""
    units = prepare_units(
        values, exposure, covariates, ever_glp,
        audit_label=f"{modality}::{cohort_label}",
    )
    if units.empty or units["treated"].sum() == 0:
        return pd.DataFrame()
    pairs = optimal_pair_match(
        units,
        covariate_cols=MATCH_COVARIATES,
        treatment_col="treated",
        id_col="participant_id",
        exact_cols=exact_columns(modality),
        caliper_sd=CALIPER_SD,
        seed=SEED,
    )
    if pairs.empty:
        return pd.DataFrame()
    deltas = unit_deltas(values, units)
    result = summarize_primary(pair_effects(pairs, deltas), modality)
    if result.empty:
        return result
    balance = balance_table(units, pairs, modality)
    result["exposure_cohort"] = cohort_label
    result["exposure_n"] = exposure["RegistrationCode"].nunique()
    result["outcome_eligible_n"] = int(units["treated"].sum())
    result["matched_n"] = pairs["treated_id"].nunique()
    result["retention"] = pairs["treated_id"].nunique() / max(units["treated"].sum(), 1)
    result["max_abs_smd"] = balance["smd_after"].abs().max()
    result["design_gate_pass"] = (
        result["retention"].ge(MIN_MATCH_RETENTION)
        & result["max_abs_smd"].lt(MAX_ABS_SMD)
    )
    return result


def write_readme(diagnostics: pd.DataFrame, unavailable: list[dict]) -> None:
    text = f"""# GLP-1 paper reanalysis

## Locked design

- Primary exposure: timing-valid, visit-confirmed semaglutide initiation.
- Primary controls: never observed on any A10BJ GLP-1 therapy in the cached
  `Medications10KLoader` dictionary/events, exact same pre/post visit window, sex,
  diabetes, and baseline lipid-lowering therapy. Smoking is also exact-blocked
  for sleep, HMO blood tests, and diet, where the prespecified balance gate
  required it.
- Primary match: globally optimal 1:1 without replacement; propensity-logit caliper
  `{CALIPER_SD}` SD; Mahalanobis distance within the caliper; no fallback.
- Sensitivities: variable 1:1--1:3 with maximum participant reuse 2; overlap weights
  with HC3 inference after collapsing to independent participants; non-GLP diabetes
  new-user active comparator.
- Matching covariates: `{", ".join(MATCH_COVARIATES)}`. VAT is symmetrically
  imputed within sex x visit window with a missingness indicator.
- Primary estimates: pair DiD with {N_BOOT:,}-resample fixed-match bootstrap CIs
  (the match itself is not re-estimated in each resample). Equivalence uses a
  prespecified ±0.2 baseline-SD margin. Endpoint FDR is controlled within each
  prespecified modality family, with a Simes modality gate and BH across modality
  gates for hierarchical discovery control.
- A design is claim-eligible only with at least {MIN_MATCH_RETENTION:.0%} treated
  retention and post-match maximum absolute SMD below {MAX_ABS_SMD:.2f}.
- Duration: continuous effect modification plus 3--6, 6--12, and >12 month bands.
  The <3-month band is not interpreted.
- Weight/diet/CGM "mediation" language is prohibited. `attenuation_models.csv`
  contains descriptive covariate-adjusted association models only.

## Data routing

- Snapshot (authoritative when present): `{SNAPSHOT}`.
- Cached LabData fallbacks: medication history, DXA, and diet logging.
- Previously used mapped 10K HUMAnN aggregate: `{HUMANN_PATHWAYS}`.
- Sleep-state heart rate is analyzed for REM, pooled NREM, and in-study wake
  after requiring at least {MIN_HEART_RATE_STAGE_SECONDS // 60} minutes in the
  relevant state and heart-rate quality score >=50. The snapshot has no
  light/deep heart-rate summaries; its raw channel paths are access-restricted,
  and no LabData summary loader supplies them. N1/N2 versus N3 therefore cannot
  be estimated in this run.
- Stable source gaps and their consequences are recorded in
  `unavailable_sources.csv`.

## Matching diagnostics

{diagnostics.to_markdown(index=False)}

## Interpretation

Cross-sectional prevalent-user analyses from the historical run are triangulation,
not independent replication. The paper-facing claims should be based on the primary
longitudinal results and require directional agreement with both bounded-reuse and
overlap-weighted sensitivity estimates.

The dedicated adjudicated cross-sectional CGM analysis is under
`cgm_cross_sectional/`. It is a supplementary negative robustness analysis,
not a replication of the longitudinal treatment-initiation result.
"""
    (OUT / "README.md").write_text(text)
    pd.DataFrame(
        unavailable,
        columns=["dataset", "status", "consequence"],
    ).to_csv(OUT / "unavailable_sources.csv", index=False)


def main() -> None:
    setup_logging()
    logging.info("Snapshot root: %s", SNAPSHOT)
    primary, semaglutide_sensitivity, tirzepatide = load_exposure_cohorts()
    ever_glp = ever_glp_participants(primary)
    covariates = load_covariates()
    comparator = active_comparator_cohort(primary)
    logging.info(
        "Exposure cohorts: primary semaglutide=%d, visit-confirmed sensitivity=%d, "
        "tirzepatide/Mounjaro=%d, active comparators=%d",
        primary["RegistrationCode"].nunique(),
        semaglutide_sensitivity["RegistrationCode"].nunique(),
        tirzepatide["RegistrationCode"].nunique(),
        comparator["RegistrationCode"].nunique(),
    )

    values_by_modality = {
        name: load_snapshot_panel(name) for name in PANELS
    }
    values_by_modality["dxa"] = load_dxa_panel()
    values_by_modality["microbiome"] = load_microbiome_panel()
    values_by_modality["microbiome_function"] = load_humann_panel()
    values_by_modality["diet"] = load_diet_panel(primary)

    analyses: dict[str, dict] = {}
    primary_results, capped_results, overlap_results = [], [], []
    balances, diagnostics, duration, duration_bands, active_results = [], [], [], [], []
    drift_references = []
    for modality, values in values_by_modality.items():
        result = analyse_modality(modality, values, primary, covariates, ever_glp)
        if not result:
            continue
        analyses[modality] = result
        drift_references.append(result["untreated_drift"])
        primary_results.append(result["primary_result"])
        capped_results.append(result["capped_result"])
        overlap_results.append(result["overlap_result"])
        balances.append(result["balance"])
        diagnostics.append(result["diagnostics"])
        duration_model, bands = duration_models(result["primary_rows"], primary, modality)
        duration.append(duration_model)
        duration_bands.append(bands)
        if modality in {"anthropometrics", "blood_pressure", "sleep", "cgm", "hmo"}:
            active = active_comparator_analysis(
                values, primary, comparator, covariates, modality
            )
            if not active.empty:
                active_results.append(active)

    def concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        frames = [frame for frame in frames if frame is not None and not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    primary_frame = add_hierarchical_multiplicity(
        concat(primary_results),
        p_column="paired_t_p",
    )
    primary_frame.to_csv(OUT / "primary_1to1_results.csv", index=False)
    concat(capped_results).to_csv(OUT / "capped_reuse_sensitivity.csv", index=False)
    concat(overlap_results).to_csv(OUT / "overlap_weighted_sensitivity.csv", index=False)
    concat(active_results).to_csv(OUT / "active_comparator_sensitivity.csv", index=False)
    concat(balances).to_csv(OUT / "covariate_balance.csv", index=False)
    concat(drift_references).to_csv(OUT / "untreated_drift_reference.csv", index=False)
    pd.DataFrame(COVARIATE_ELIGIBILITY_AUDIT).to_csv(
        OUT / "covariate_eligibility_audit.csv", index=False
    )
    concat(duration).to_csv(OUT / "duration_models.csv", index=False)
    concat(duration_bands).to_csv(OUT / "duration_bands_ge3months.csv", index=False)
    diagnostics_frame = pd.DataFrame(diagnostics)
    diagnostics_frame.to_csv(OUT / "matching_diagnostics.csv", index=False)

    exposure_sensitivities = []
    for cohort_label, exposure in [
        ("semaglutide_visit_confirmed_broader", semaglutide_sensitivity),
        ("tirzepatide_or_mounjaro_separate", tirzepatide),
    ]:
        for modality in [
            "anthropometrics",
            "blood_pressure",
            "sleep",
            "dxa",
            "microbiome",
            "microbiome_function",
            "diet",
        ]:
            sensitivity = exposure_sensitivity_analysis(
                values_by_modality[modality],
                exposure,
                covariates,
                ever_glp,
                modality,
                cohort_label,
            )
            if not sensitivity.empty:
                exposure_sensitivities.append(sensitivity)
    concat(exposure_sensitivities).to_csv(
        OUT / "exposure_definition_sensitivities.csv", index=False
    )

    contrasts = []
    if "sleep" in analyses:
        rows = analyses["sleep"]["primary_rows"]
        contrasts.append(
            {
                **contrast_model(
                    rows,
                    "heart_rate_mean_during_rem",
                    "heart_rate_mean_during_nrem",
                    "REM minus NREM heart-rate DiD",
                ),
                "multiplicity_family": "heart_rate_state_specificity",
            }
        )
        contrasts.append(
            {
                **contrast_model(
                    rows,
                    "heart_rate_mean_during_rem",
                    "heart_rate_mean_during_wake",
                    "REM minus wake heart-rate DiD",
                ),
                "multiplicity_family": "heart_rate_state_specificity",
            }
        )
        contrasts.append(
            {
                **contrast_model(
                    rows,
                    "heart_rate_mean_during_nrem",
                    "heart_rate_mean_during_wake",
                    "NREM minus wake heart-rate DiD",
                ),
                "multiplicity_family": "heart_rate_state_specificity",
            }
        )
        contrasts.append(
            {
                **contrast_model(
                    rows,
                    "neurokit_hrv_time_rmssd_during_rem",
                    "neurokit_hrv_time_rmssd_during_nrem",
                    "REM minus NREM RMSSD DiD after segment QC",
                ),
                "multiplicity_family": "hrv_segment_specificity",
            }
        )
    contrast_frame = pd.DataFrame(contrasts)
    if not contrast_frame.empty:
        contrast_frame["fdr_within_contrast_family"] = np.nan
        for _family, index in contrast_frame.groupby(
            "multiplicity_family"
        ).groups.items():
            p_values = contrast_frame.loc[index, "p"]
            finite = p_values.notna()
            if finite.any():
                contrast_frame.loc[
                    p_values.index[finite], "fdr_within_contrast_family"
                ] = bh_fdr(p_values[finite].to_numpy(float))
    contrast_frame.to_csv(OUT / "formal_specificity_contrasts.csv", index=False)

    attenuation = []
    if "anthropometrics" in analyses:
        weight_rows = analyses["anthropometrics"]["primary_rows"]
        for modality, endpoints in {
            "sleep": ["heart_rate_mean_during_rem", "ahi_during_rem"],
            "hmo": ["bt_glucose", "bt_triglycerides", "bt_hdl_cholesterol"],
            "dxa": ["total_scan_vat_area"],
        }.items():
            if modality not in analyses:
                continue
            for endpoint in endpoints:
                attenuation.append(
                    attenuation_model(
                        analyses[modality]["primary_rows"],
                        weight_rows,
                        endpoint,
                        "weight",
                        f"{modality}:{endpoint} adjusted for pairwise weight change",
                    )
                )
    if "cgm" in analyses:
        cgm_rows = analyses["cgm"]["primary_rows"]
        for endpoint in ["iglu_cv", "iglu_sd", "iglu_mage"]:
            attenuation.append(
                attenuation_model(
                    cgm_rows,
                    cgm_rows,
                    endpoint,
                    "iglu_mean",
                    f"cgm:{endpoint} adjusted for pairwise mean-glucose change",
                )
            )
    if "sleep" in analyses:
        sleep_rows = analyses["sleep"]["primary_rows"]
        for segment in ["rem", "nrem"]:
            for family in ["pnn20", "rmssd", "sdnn"]:
                attenuation.append(
                    attenuation_model(
                        sleep_rows,
                        sleep_rows,
                        f"neurokit_hrv_time_{family}_during_{segment}",
                        f"heart_rate_mean_during_{segment}",
                        (
                            f"sleep:{family}_{segment} adjusted for pairwise "
                            f"heart-rate change"
                        ),
                    )
                )
    pd.DataFrame(attenuation).to_csv(OUT / "attenuation_models.csv", index=False)

    unavailable: list[dict] = [
        {
            "dataset": "sleep raw heart-rate x sleep-stage channels",
            "status": "raw S3 channel paths present in snapshot metadata but access denied; "
            "no LabData summary fields for light/deep heart rate",
            "consequence": "state analysis is limited to REM, pooled NREM, and in-study wake; "
            "N1/N2 versus N3 cannot be estimated",
        }
    ]
    write_readme(diagnostics_frame, unavailable)
    logging.info("Paper reanalysis complete: %s", OUT)


if __name__ == "__main__":
    main()
