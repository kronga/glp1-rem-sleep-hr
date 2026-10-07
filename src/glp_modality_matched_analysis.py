#!/usr/bin/env python
"""GLP-1 before/after modality change vs a propensity-matched control cohort.

Reusable across modalities. Outcome measurements are read from the phenotype
data snapshot (parquet) rather than the LabData loaders, so the same matching
design can be re-run for retina, blood pressure, ECG, sleep, etc. by changing
`--outcome-modality`.

Design
------
- Treated arm: participants who started a GLP-1 receptor agonist during cohort
  participation (from the visit-transition analysis), with a pre visit where the
  drug was not reported and a post visit where it was confirmed.
- Each treated participant is matched 1:k (default k=5), with replacement, to
  non-GLP controls who have the chosen modality measured at the SAME pre and post
  visits (exact match on the visit window) and the closest propensity scores,
  where propensity = P(treated) from a logistic regression on age, gender, BMI,
  waist/hip circumference, baseline DXA VAT area (imputed for treated
  participants missing a scan, from the rest of the GLP-1 cohort), and pre-stage
  self-reported diabetes, all measured at the pre visit.
- Outcome: per-measurement delta = post_value - pre_value (repeated rows per
  participant x visit, e.g. both eyes, are averaged first). Treated deltas are
  compared with the mean of that participant's k matched-control deltas
  (difference-in-differences) per measurement.

Data layout
-----------
The snapshot directory holds one folder per dataset, each with one or more
tables stored as `<dataset>/<table>/df.parquet` (+ `df_metadata.parquet`). The
data frame and metadata frame share an identical (RegistrationCode, date) index,
so metadata columns are aligned positionally. `research_stage` lives in the
metadata; covariates (age, gender, BMI, waist/hip) are taken from the
anthropometrics table; VAT and diabetes come from `glp_confounders.py`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

from aggregate_medication_stats import resolve_output_path

DEFAULT_SNAPSHOT_DIR = Path(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/"
    "snapshots/pheno_data_snapshots/out_dir"
)

# Covariates for propensity matching always come from this table.
COVARIATE_TABLE = ("anthropometrics", "anthropometrics")

# Modality key -> (dataset_folder, table_name). Add rows here to support more
# outcome modalities; nothing else needs to change.
MODALITY_TABLES = {
    "retina": ("fundus", "fundus"),
    "anthropometrics": ("anthropometrics", "anthropometrics"),
    "blood_pressure": ("blood_pressure", "blood_pressure"),
    "abi": ("ABI", "ABI"),
    "ecg": ("ECG", "ECG"),
    "sleep": ("sleep", "sleep"),
    "sleep_hrv": ("sleep", "sleep_hrv"),
    "cgm": ("CGM", "CGM"),
    "olink": ("olink", "olink_npx"),
    "hmo_blood_tests": ("HMO_blood_tests", "HMO_blood_tests"),
}

# Substrings dropped from every modality before treating remaining numeric
# columns as outcomes (QC / version / id fields that are not measurements).
GLOBAL_EXCLUDE_SUBSTRINGS = ["pi_version", "uuid", "_id", "version"]

# Per-modality substrings whose columns are dropped before treating remaining
# numeric columns as outcomes (QC / preprocessing / non-measurement fields).
MODALITY_EXCLUDE_SUBSTRINGS = {
    "retina": ["preprocessing", "image_quality", "pixel_spacing", "raws", "columns"],
}

TRANSITION_FILES = {
    "strict": Path(
        "visit_transition_analysis/outputs/"
        "strict_visit_reported_participant_medication_transitions.csv"
    ),
    "carry_forward": Path(
        "visit_transition_analysis/outputs/"
        "carry_forward_participant_medication_transitions.csv"
    ),
}

# Transition visit labels -> snapshot research_stage codes (baseline is 00_00_visit).
STAGE_TO_RESEARCH_STAGE = {
    "baseline": "00_00_visit",
    "02 visit": "02_00_visit",
    "04 visit": "04_00_visit",
    "06 visit": "06_00_visit",
}
SUPPORTED_STAGES = set(STAGE_TO_RESEARCH_STAGE.values())

MATCH_COVARIATES = [
    "age", "gender", "bmi",
    "waist_circumference", "hip_circumference",
    "vat_area", "diabetic", "smoking",
]
# Covariates matched EXACTLY (a treated is only compared with controls that share
# the same value), in addition to the exact visit-window match. Sex is exact-matched
# so every taker's controls are all the same sex; it remains in MATCH_COVARIATES so
# its (now ~0) balance is still reported.
EXACT_MATCH_COVARIATES = ["gender"]
DAYS_PER_YEAR = 365.25

# Significance policy. One test is PRE-SPECIFIED as primary per modality; the other is
# reported only as a robustness check, never as a second path to significance (which
# would inflate the false-positive rate). Roughly-symmetric continuous modalities use
# the paired t-test; skewed ones (HRV, diet, microbiome, proteomics, imaging summaries)
# use the Wilcoxon signed-rank test, which is robust to skew/outliers.
SIG_THRESHOLD = 0.10
T_TEST_MODALITIES = {"anthropometrics", "blood_pressure", "ecg", "abi", "dxa"}


def primary_test(modality: str) -> str:
    """'t' (paired t-test) for symmetric continuous modalities, else 'wilcoxon'."""
    return "t" if modality in T_TEST_MODALITIES else "wilcoxon"


def primary_fdr_column(modality: str) -> str:
    return "paired_t_fdr" if primary_test(modality) == "t" else "wilcoxon_fdr"


def annotate_significance(did: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Add primary_test / primary_fdr / significant columns using the pre-specified
    primary test for this modality. The non-primary test's FDR stays in the table as
    a robustness check."""
    if did.empty:
        return did
    did = did.copy()
    did["primary_test"] = primary_test(modality)
    did["primary_fdr"] = did[primary_fdr_column(modality)]
    did["significant"] = did["primary_fdr"] < SIG_THRESHOLD
    return did


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLP-1 before/after modality change vs propensity-matched controls."
    )
    parser.add_argument(
        "--outcome-modality",
        default="retina",
        choices=sorted(MODALITY_TABLES),
        help="Modality whose measurements are compared before/after.",
    )
    parser.add_argument(
        "--approach",
        choices=sorted(TRANSITION_FILES),
        default="strict",
        help="Medication-start definition used to identify GLP-1 starters.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help="Phenotype data snapshot out_dir holding <dataset>/<table>/df.parquet.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to glp_<modality>_matched_analysis.",
    )
    parser.add_argument(
        "--caliper-sd",
        type=float,
        default=0.2,
        help="Caliper width as a multiple of the SD of the propensity-score logit.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=5,
        help="Minimum matched pairs with a measurement to report a per-measurement DiD.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260630,
        help="Seed for deterministic tie-breaking in nearest-neighbor matching.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="Number of matched controls per treated participant (with replacement).",
    )
    parser.add_argument(
        "--diabetes-timing",
        choices=["pre_stage", "ever"],
        default="pre_stage",
        help="Diabetes matching-covariate timing: pre_stage (default) or ever-reported.",
    )
    return parser.parse_args()


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "glp_modality_matched_analysis.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def read_snapshot_table(snapshot_dir: Path, dataset: str, table: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_dir = snapshot_dir / dataset / table
    df = pd.read_parquet(table_dir / "df.parquet")
    meta = pd.read_parquet(table_dir / "df_metadata.parquet")
    if len(df) != len(meta):
        raise ValueError(
            f"{dataset}/{table}: df ({len(df)}) and metadata ({len(meta)}) row counts differ"
        )
    return df, meta


def normalize_reg(values) -> pd.Series:
    """Snapshot uses '10k_...'; transitions use '10K_...'. Normalize to upper."""
    return pd.Series(values, dtype="string").str.upper()


def attach_research_stage(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Add research_stage positionally (df and meta share an identical index)."""
    if "research_stage" not in meta.columns:
        raise ValueError("metadata has no research_stage column")
    out = df.copy()
    out["research_stage"] = meta["research_stage"].to_numpy()
    return out


def encode_gender(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.lower()
    mapping = {"female": 1.0, "f": 1.0, "male": 0.0, "m": 0.0}
    encoded = text.map(mapping)
    if encoded.isna().all():
        encoded = pd.to_numeric(values, errors="coerce")
    return encoded


def load_outcome_values(snapshot_dir: Path, modality: str) -> pd.DataFrame:
    """Long table: one numeric value per RegistrationCode x research_stage x measurement."""
    dataset, table = MODALITY_TABLES[modality]
    df, meta = read_snapshot_table(snapshot_dir, dataset, table)
    df = attach_research_stage(df, meta)
    # Group A tables carry RegistrationCode in the (RegistrationCode, date) index;
    # Group B (sample_id-indexed, e.g. olink) carry it in the aligned metadata.
    if "RegistrationCode" in df.index.names:
        df = df.reset_index()
    else:
        df["RegistrationCode"] = meta["RegistrationCode"].to_numpy()
        df = df.reset_index(drop=True)
    df["RegistrationCode"] = normalize_reg(df["RegistrationCode"]).to_numpy()
    df = df[df["research_stage"].isin(SUPPORTED_STAGES)].copy()

    exclude_substrings = GLOBAL_EXCLUDE_SUBSTRINGS + MODALITY_EXCLUDE_SUBSTRINGS.get(modality, [])
    value_cols = []
    for col in df.columns:
        if col in {"RegistrationCode", "date", "research_stage", "SampleID", "sample_id"}:
            continue
        if any(sub in col.lower() for sub in exclude_substrings):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            value_cols.append(col)
    if not value_cols:
        raise ValueError(f"{modality}: no numeric outcome columns after exclusions")

    per_stage = (
        df.groupby(["RegistrationCode", "research_stage"], as_index=False)[value_cols]
        .mean(numeric_only=True)
    )
    long_values = per_stage.melt(
        id_vars=["RegistrationCode", "research_stage"],
        value_vars=value_cols,
        var_name="measurement",
        value_name="value",
    ).dropna(subset=["value"])
    return long_values


ANTHROPOMETRIC_MATCH_FIELDS = ["waist_circumference", "hip_circumference"]


def load_covariates(snapshot_dir: Path) -> pd.DataFrame:
    """age (years), gender (female=1), bmi, waist/hip circumference per
    RegistrationCode x research_stage."""
    dataset, table = COVARIATE_TABLE
    df, meta = read_snapshot_table(snapshot_dir, dataset, table)
    work = pd.DataFrame(
        {
            "RegistrationCode": normalize_reg(
                df.index.get_level_values("RegistrationCode")
            ).to_numpy(),
            "research_stage": meta["research_stage"].to_numpy(),
            "age": pd.to_numeric(meta["age"].to_numpy(), errors="coerce") / DAYS_PER_YEAR,
            "gender": encode_gender(meta["gender"]).to_numpy(),
            "bmi": pd.to_numeric(df["bmi"].to_numpy(), errors="coerce"),
        }
    )
    for field in ANTHROPOMETRIC_MATCH_FIELDS:
        work[field] = pd.to_numeric(df[field].to_numpy(), errors="coerce")
    work = work[work["research_stage"].isin(SUPPORTED_STAGES)].copy()
    agg = {"age": ("age", "mean"), "gender": ("gender", "first"), "bmi": ("bmi", "mean")}
    agg.update({field: (field, "mean") for field in ANTHROPOMETRIC_MATCH_FIELDS})
    return work.groupby(["RegistrationCode", "research_stage"], as_index=False).agg(**agg)


def load_glp_treated(path: Path, approach: str) -> pd.DataFrame:
    """One treated row per GLP-1 starter: earliest observed pre/post visit window."""
    transitions = pd.read_csv(path)
    approach_label = {
        "strict": "strict_visit_reported",
        "carry_forward": "carry_forward_from_call",
    }[approach]
    transitions = transitions[transitions["approach"].eq(approach_label)].copy()
    glp = transitions[
        transitions["drug_family_or_purpose"].str.contains("GLP", case=False, na=False)
    ].copy()
    glp["pre_stage"] = glp["prior_absent_visit_stage"].map(STAGE_TO_RESEARCH_STAGE)
    glp["post_stage"] = glp["confirmation_visit_stage"].map(STAGE_TO_RESEARCH_STAGE)
    glp = glp.dropna(subset=["pre_stage", "post_stage"]).copy()
    glp = glp.sort_values(
        ["RegistrationCode", "prior_absent_visit_stage_order", "confirmation_visit_stage_order"]
    )
    glp = glp.drop_duplicates("RegistrationCode", keep="first")
    return glp[
        ["RegistrationCode", "medication", "Generic", "ATC", "pre_stage", "post_stage"]
    ].reset_index(drop=True)


def build_units(
    treated_ids: set[str],
    windows: pd.DataFrame,
    outcome_avail: pd.Series,
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (participant, window) eligible for matching.

    Eligible = has the outcome modality at both the window's pre and post stages
    and has age, gender, and BMI at the pre stage. Treated units use their own
    window; control units are every non-GLP participant evaluated against each
    treated window.
    """
    pre_covariates = covariates.rename(columns={"research_stage": "pre_stage"})
    unique_windows = windows[["pre_stage", "post_stage"]].drop_duplicates()

    rows = []
    for _, win in unique_windows.iterrows():
        pre, post = win["pre_stage"], win["post_stage"]
        eligible = {
            reg for reg, stages in outcome_avail.items() if pre in stages and post in stages
        }
        for reg in eligible:
            rows.append(
                {
                    "RegistrationCode": reg,
                    "pre_stage": pre,
                    "post_stage": post,
                    "treated": int(reg in treated_ids),
                }
            )
    units = pd.DataFrame(rows)

    treated_windows = set(
        map(tuple, windows[["RegistrationCode", "pre_stage", "post_stage"]].values)
    )
    is_treated_row = units["treated"].eq(1)
    keep_treated = units[is_treated_row].apply(
        lambda r: (r["RegistrationCode"], r["pre_stage"], r["post_stage"]) in treated_windows,
        axis=1,
    )
    units = pd.concat(
        [units[is_treated_row][keep_treated.values], units[~is_treated_row]],
        ignore_index=True,
    )

    units = units.merge(pre_covariates, on=["RegistrationCode", "pre_stage"], how="left")
    units = units.dropna(subset=MATCH_COVARIATES).reset_index(drop=True)
    return units


def estimate_propensity(units: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    features = units[MATCH_COVARIATES].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(features)
    model = LogisticRegression(max_iter=1000)
    model.fit(scaled, units["treated"].to_numpy())
    ps = np.clip(model.predict_proba(scaled)[:, 1], 1e-6, 1 - 1e-6)
    out = units.copy()
    out["propensity"] = ps
    out["ps_logit"] = np.log(ps / (1 - ps))
    return out


def match_within_windows(
    units: pd.DataFrame, caliper_sd: float, seed: int, k_neighbors: int = 5,
    caliper_fallback: bool = True,
) -> pd.DataFrame:
    """1:k_neighbors nearest-neighbor on PS logit, exact on window, with replacement.

    Controls may be reused across different treated participants (needed to reliably
    reach k_neighbors matches per treated when the eligible control pool is small).
    Ranking prefers controls within the caliper, ordered by propensity-score logit
    distance, with Mahalanobis distance on the standardized covariates and a random
    jitter as tie-breaks so age/gender/etc. also balance.

    Caliper fallback (default): to retain every treated participant that has any
    control in its visit window, if fewer than k_neighbors controls fall inside the
    caliper the remaining slots are filled with the nearest out-of-caliper controls
    in the same window. Each emitted pair carries `within_caliper` so downstream code
    can see how many matches used the fallback.

    With `caliper_fallback=False` (tighter matching), only in-caliper controls are
    used: a treated participant gets up to k_neighbors in-caliper controls (fewer if
    the pool is smaller) and is dropped entirely if none fall inside the caliper. This
    trades sample size for better covariate balance.
    """
    logit_sd = units["ps_logit"].std(ddof=1)
    caliper = caliper_sd * logit_sd
    cov = np.cov(units[MATCH_COVARIATES].to_numpy(dtype=float), rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    rng = np.random.default_rng(seed)
    pairs = []
    treated = units[units["treated"].eq(1)].sort_values("ps_logit", ascending=False)
    for _, t in treated.iterrows():
        eligible = (
            units["treated"].eq(0)
            & units["pre_stage"].eq(t["pre_stage"])
            & units["post_stage"].eq(t["post_stage"])
        )
        for cov_name in EXACT_MATCH_COVARIATES:
            eligible &= units[cov_name].eq(t[cov_name])
        candidates = units[eligible].copy()
        if candidates.empty:
            continue
        candidates["distance"] = (candidates["ps_logit"] - t["ps_logit"]).abs()
        candidates["within_caliper"] = candidates["distance"] <= caliper
        if not caliper_fallback:
            candidates = candidates[candidates["within_caliper"]].copy()
            if candidates.empty:
                continue
        diff = (
            candidates[MATCH_COVARIATES].to_numpy(dtype=float)
            - t[MATCH_COVARIATES].to_numpy(dtype=float)
        )
        candidates["mahalanobis"] = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))
        candidates["jitter"] = rng.random(len(candidates))
        # In-caliper controls first (nearest by ps distance), then (only if fallback
        # is enabled) top up with the nearest out-of-caliper controls to reach k.
        ranked = candidates.sort_values(
            ["within_caliper", "distance", "mahalanobis", "jitter"],
            ascending=[False, True, True, True],
        )
        matches = ranked.head(k_neighbors)
        for rank, (_, m) in enumerate(matches.iterrows(), start=1):
            row = {
                "pre_stage": t["pre_stage"],
                "post_stage": t["post_stage"],
                "treated_reg": t["RegistrationCode"],
                "control_reg": m["RegistrationCode"],
                "match_rank": rank,
                "within_caliper": bool(m["within_caliper"]),
                "treated_ps": t["propensity"],
                "control_ps": m["propensity"],
                "ps_logit_distance": m["distance"],
            }
            for cov_name in MATCH_COVARIATES:
                row[f"treated_{cov_name}"] = t[cov_name]
                row[f"control_{cov_name}"] = m[cov_name]
            pairs.append(row)
    return pd.DataFrame(pairs)


def standardized_mean_diff(treated: pd.Series, control: pd.Series) -> float:
    t = pd.to_numeric(treated, errors="coerce").dropna()
    c = pd.to_numeric(control, errors="coerce").dropna()
    if len(t) < 2 or len(c) < 2:
        return np.nan
    pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((t.mean() - c.mean()) / pooled_sd)


def covariate_balance(units: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    treated_all = units[units["treated"].eq(1)]
    control_all = units[units["treated"].eq(0)]
    for cov in MATCH_COVARIATES:
        rows.append(
            {
                "covariate": cov,
                "smd_before": standardized_mean_diff(treated_all[cov], control_all[cov]),
                "smd_after": standardized_mean_diff(
                    pairs[f"treated_{cov}"], pairs[f"control_{cov}"]
                ) if not pairs.empty else np.nan,
                "treated_mean_matched": pd.to_numeric(
                    pairs[f"treated_{cov}"], errors="coerce"
                ).mean() if not pairs.empty else np.nan,
                "control_mean_matched": pd.to_numeric(
                    pairs[f"control_{cov}"], errors="coerce"
                ).mean() if not pairs.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def participant_deltas(values: pd.DataFrame, window_lookup: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Per-participant per-measurement post-pre delta using each person's window."""
    records = []
    pivot = values.pivot_table(
        index=["RegistrationCode", "measurement"],
        columns="research_stage",
        values="value",
        aggfunc="mean",
    ).reset_index()
    for _, row in pivot.iterrows():
        reg = row["RegistrationCode"]
        if reg not in window_lookup:
            continue
        pre_stage, post_stage = window_lookup[reg]
        pre_val = row.get(pre_stage)
        post_val = row.get(post_stage)
        if pd.isna(pre_val) or pd.isna(post_val):
            continue
        records.append(
            {
                "RegistrationCode": reg,
                "measurement": row["measurement"],
                "pre_value": pre_val,
                "post_value": post_val,
                "delta": post_val - pre_val,
            }
        )
    return pd.DataFrame(records)


def p_adjust_bh(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    valid = p.notna()
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)
    if not valid.any():
        return adjusted
    valid_p = p[valid].to_numpy()
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    adjusted.loc[valid] = out
    return adjusted


def difference_in_differences(pairs: pd.DataFrame, deltas: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    """Per-measurement paired diff = treated delta - mean(delta over that treated's
    k matched controls), i.e. the standard k:1 matching-estimator collapse. One row
    per treated unit with valid data; `matched_pairs` counts treated units, not
    treated-control link rows.
    """
    from scipy import stats

    if pairs.empty or deltas.empty:
        return pd.DataFrame()
    delta_lookup = deltas.set_index(["RegistrationCode", "measurement"])["delta"]
    treated_controls = pairs.groupby("treated_reg")["control_reg"].apply(list)
    records = []
    for measurement in sorted(deltas["measurement"].unique()):
        treated_deltas, control_deltas, paired_diffs = [], [], []
        for treated_reg, control_regs in treated_controls.items():
            t = delta_lookup.get((treated_reg, measurement))
            if t is None or pd.isna(t):
                continue
            controls = [delta_lookup.get((c, measurement)) for c in control_regs]
            controls = [c for c in controls if c is not None and not pd.isna(c)]
            if not controls:
                continue
            control_mean = float(np.mean(controls))
            treated_deltas.append(t)
            control_deltas.append(control_mean)
            paired_diffs.append(t - control_mean)
        n = len(paired_diffs)
        if n < min_pairs:
            continue
        diffs = np.array(paired_diffs, dtype=float)
        treated_arr = np.array(treated_deltas, dtype=float)
        control_arr = np.array(control_deltas, dtype=float)
        if n >= 3 and diffs.std(ddof=1) > 0:
            t_p = float(stats.ttest_1samp(diffs, 0.0).pvalue)
            try:
                w_p = float(stats.wilcoxon(treated_arr, control_arr).pvalue)
            except ValueError:
                w_p = np.nan
        else:
            t_p = w_p = np.nan
        records.append(
            {
                "measurement": measurement,
                "matched_pairs": n,
                "treated_delta_mean": treated_arr.mean(),
                "control_delta_mean": control_arr.mean(),
                "did_mean": diffs.mean(),
                "did_median": float(np.median(diffs)),
                "did_sd": diffs.std(ddof=1) if n >= 2 else np.nan,
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
            }
        )
    summary = pd.DataFrame.from_records(records)
    if summary.empty:
        return summary
    summary["paired_t_fdr"] = p_adjust_bh(summary["paired_t_p"])
    summary["wilcoxon_fdr"] = p_adjust_bh(summary["wilcoxon_p"])
    return summary.sort_values("paired_t_p", na_position="last").reset_index(drop=True)


def write_readme(
    analysis_dir: Path,
    modality: str,
    approach: str,
    snapshot_dir: Path,
    caliper_sd: float,
    treated: pd.DataFrame,
    pairs: pd.DataFrame,
    balance: pd.DataFrame,
    did: pd.DataFrame,
    k_neighbors: int = 5,
) -> None:
    window_counts = (
        pairs.groupby(["pre_stage", "post_stage"]).size().reset_index(name="matched_pairs")
        if not pairs.empty
        else pd.DataFrame(columns=["pre_stage", "post_stage", "matched_pairs"])
    )
    matched_treated = int(pairs["treated_reg"].nunique()) if not pairs.empty else 0
    matched_controls = int(pairs["control_reg"].nunique()) if not pairs.empty else 0
    covariate_list = ", ".join(MATCH_COVARIATES)
    primary = "paired t-test" if primary_test(modality) == "t" else "Wilcoxon signed-rank test"
    sig_threshold = SIG_THRESHOLD
    n_sig = int(did["significant"].sum()) if (not did.empty and "significant" in did.columns) else 0
    text = f"""# GLP-1 {modality.title()} Before/After vs Propensity-Matched Controls

## Design

- Outcome modality: `{modality}` (snapshot table `{"/".join(MODALITY_TABLES.get(modality, ("", modality))).strip("/")}`).
- Treated arm: GLP-1 receptor agonist starters (`{approach}` transition
  definition), one row per participant at their earliest pre/post visit window.
- Each treated participant is matched 1:{k_neighbors} to non-GLP controls, **with
  replacement** (the same control may match more than one treated participant),
  with the modality measured at the **same** pre and post visits and the nearest
  propensity scores.
- Propensity = P(treated) from logistic regression on **{covariate_list}**
  measured at the pre visit (age/gender/bmi/waist/hip from `anthropometrics`;
  `vat_area` from baseline DXA, imputed for GLP-1 treated participants missing a
  scan using the median of the rest of the GLP-1 cohort; `diabetic` = self-reported
  diabetes at or before the pre visit; `smoking` = current-smoker indicator from
  LifeStyle `smoke_tobacco_now`; continuous covariates standardized).
- Nearest-neighbor matching on the propensity-score logit, **exact on visit window
  and on sex** (a taker is only matched to same-sex controls), caliper =
  {caliper_sd} x SD(logit); within the caliper, candidates are ranked by logit
  distance first, Mahalanobis distance on the covariates as a tie-break, and the
  {k_neighbors} nearest become that treated participant's matched controls.
- Caliper fallback: to retain the maximum number of GLP-1 takers, a treated
  participant with fewer than {k_neighbors} in-caliper controls has the remaining
  slots filled by the nearest out-of-caliper controls in the same window; a treated
  is dropped only if its window contains no non-GLP control at all. The
  `within_caliper` column in `matched_pairs.csv` flags which matches used the
  fallback (see the fallback count below).
- Outcome: per-measurement delta = post_value - pre_value, repeated rows per
  participant x visit averaged first. Difference-in-differences per treated
  participant = treated delta - mean(delta across that participant's {k_neighbors}
  matched controls).
- Significance: each measurement gets a paired t-test and a Wilcoxon signed-rank
  test, each BH-FDR corrected within this modality. **One test is pre-specified as
  primary** ({primary} for `{modality}`); `significant` = primary FDR < {sig_threshold:g}.
  The other test's FDR is reported only as a robustness check, never as a second path
  to significance (which would inflate the false-positive rate).

## Data source

Phenotype data snapshot: `{snapshot_dir}`

## Cohort

- GLP-1 starters identified: **{treated['RegistrationCode'].nunique():,}**
- Treated participants with at least one matched control: **{matched_treated:,}**
- Distinct controls used (reuse across treated participants allowed): **{matched_controls:,}**
- Treated-control match rows (up to {k_neighbors} per treated): **{len(pairs):,}**
- Match rows using the out-of-caliper fallback: **{int((~pairs['within_caliper']).sum()) if not pairs.empty else 0:,}**

### Matched pairs by visit window

{window_counts.to_markdown(index=False) if not window_counts.empty else "(none)"}

## Covariate balance (standardized mean differences)

Rule of thumb: |SMD| < 0.1 indicates good balance. Age is in years.

{balance.to_markdown(index=False)}

## Outputs

- `outputs/glp_treated_cohort.csv`: GLP-1 starters with their pre/post visit windows.
- `outputs/matched_pairs.csv`: treated/control pairs with propensity scores and covariates.
- `outputs/covariate_balance.csv`: SMD before and after matching.
- `outputs/participant_{modality}_deltas.csv`: per-participant per-measurement pre/post/delta.
- `outputs/{modality}_did_summary.csv`: per-measurement difference-in-differences with tests.
- `outputs/glp_{modality}_matched_analysis.xlsx`: workbook of the above.
- `figures/{modality}_did.png`: difference-in-differences per measurement.

## Per-measurement difference-in-differences

Primary test for `{modality}`: **{primary}** (FDR < {sig_threshold:g}). Significant by the
primary test: **{n_sig}** measurement(s). The `significant` column reflects the primary
test only; `paired_t_fdr` and `wilcoxon_fdr` are both shown so the non-primary test
serves as a robustness check.

{did.to_markdown(index=False) if not did.empty else "(no measurements met the minimum matched-pair threshold)"}

## Caveats

Coverage at later visits limits the usable matched sample; treat per-measurement
tests as exploratory. The difference-in-differences removes the secular (control)
change at the same visit window from the treated change. Matching is with
replacement, so a shared control induces some correlation between the treated
participants it's matched to; the paired tests here do not model that correlation,
consistent with this pipeline's existing level of rigor (see the corrected
ANCOVA-IPTW pipeline for a fully weighted/bootstrapped alternative).
"""
    (analysis_dir / "README.md").write_text(text)


def plot_did(did: pd.DataFrame, figure_dir: Path, modality: str) -> None:
    if did.empty:
        return
    import matplotlib.pyplot as plt

    plot_df = did.sort_values("did_mean")
    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * n + 1)))
    se = plot_df["did_sd"] / np.sqrt(plot_df["matched_pairs"].clip(lower=1))
    y = np.arange(n)
    ax.errorbar(
        plot_df["did_mean"], y, xerr=1.96 * se, fmt="o",
        color="#2b6cb0", ecolor="#90cdf4", capsize=3,
    )
    ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["measurement"], fontsize=8)
    ax.set_xlabel("Difference-in-differences (treated delta - matched control delta), mean +- 95% CI")
    ax.set_title(f"GLP-1 vs matched control: {modality} change", loc="left", fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{modality}_did.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    modality: str,
    approach: str,
    snapshot_dir: Path,
    script_dir: Path,
    analysis_dir: Path | None = None,
    caliper_sd: float = 0.2,
    min_pairs: int = 5,
    seed: int = 20260630,
    covariates: pd.DataFrame | None = None,
    treated: pd.DataFrame | None = None,
    outcome_values: pd.DataFrame | None = None,
    k_neighbors: int = 5,
    diabetes_timing: str = "pre_stage",
    caliper_fallback: bool = True,
    dir_suffix: str = "",
) -> dict:
    """Run one (modality, approach) analysis and write its outputs.

    Logging must be configured by the caller. `covariates` (anthropometrics) and
    `treated` (GLP cohort for this approach) may be passed in to avoid reloading
    across a sweep. `outcome_values` may be precomputed long values (RegistrationCode,
    research_stage, measurement, value) — used by the microbiome CLR pipeline to
    supply transformed outcomes instead of `load_outcome_values`.
    """
    if analysis_dir is None:
        analysis_dir = Path(f"glp_{modality}_{approach}{dir_suffix}_matched_analysis")
    analysis_dir = resolve_output_path(analysis_dir, script_dir)
    output_dir = analysis_dir / "outputs"
    figure_dir = analysis_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if treated is None:
        transition_path = resolve_output_path(TRANSITION_FILES[approach], script_dir)
        treated = load_glp_treated(transition_path, approach)
    treated_ids = set(treated["RegistrationCode"])

    if outcome_values is None:
        logging.info("[%s/%s] loading outcomes", modality, approach)
        outcome_values = load_outcome_values(snapshot_dir, modality)
    if covariates is None:
        covariates = load_covariates(snapshot_dir)

    from glp_confounders import augment_for_matching
    covariates = augment_for_matching(covariates, treated_ids, diabetes_timing=diabetes_timing)

    outcome_avail = (
        outcome_values[["RegistrationCode", "research_stage"]]
        .drop_duplicates()
        .groupby("RegistrationCode")["research_stage"]
        .apply(set)
    )

    units = build_units(treated_ids, treated, outcome_avail, covariates)
    n_treated_units = int(units["treated"].sum())
    n_control_units = len(units) - n_treated_units
    logging.info(
        "[%s/%s] eligible units: %d (treated=%d, control=%d)",
        modality, approach, len(units), n_treated_units, n_control_units,
    )

    if n_treated_units == 0 or n_control_units == 0:
        logging.warning("[%s/%s] no eligible treated/control units; skipping match", modality, approach)
        pairs = pd.DataFrame()
        balance = covariate_balance(units, pairs)
        did = pd.DataFrame()
        deltas = pd.DataFrame()
    else:
        units = estimate_propensity(units)
        pairs = match_within_windows(
            units, caliper_sd, seed, k_neighbors=k_neighbors, caliper_fallback=caliper_fallback,
        )
        n_fallback = int((~pairs["within_caliper"]).sum()) if not pairs.empty else 0
        logging.info(
            "[%s/%s] matched pairs: %d (%d treated matched; %d/%d pairs used out-of-caliper fallback)",
            modality, approach, len(pairs),
            pairs["treated_reg"].nunique() if not pairs.empty else 0,
            n_fallback, len(pairs),
        )
        balance = covariate_balance(units, pairs)

        matched_ids = set(pairs["treated_reg"]) | set(pairs["control_reg"]) if not pairs.empty else set()
        window_lookup = {}
        for r in pairs.itertuples():
            window_lookup[r.treated_reg] = (r.pre_stage, r.post_stage)
            window_lookup[r.control_reg] = (r.pre_stage, r.post_stage)
        deltas = participant_deltas(
            outcome_values[outcome_values["RegistrationCode"].isin(matched_ids)], window_lookup
        )
        if not deltas.empty:
            treated_regs = set(pairs["treated_reg"])
            deltas["arm"] = deltas["RegistrationCode"].map(
                lambda r: "treated" if r in treated_regs else "control"
            )
        did = difference_in_differences(pairs, deltas, min_pairs)

    did = annotate_significance(did, modality)

    treated.to_csv(output_dir / "glp_treated_cohort.csv", index=False)
    pairs.to_csv(output_dir / "matched_pairs.csv", index=False)
    balance.to_csv(output_dir / "covariate_balance.csv", index=False)
    deltas.to_csv(output_dir / f"participant_{modality}_deltas.csv", index=False)
    did.to_csv(output_dir / f"{modality}_did_summary.csv", index=False)

    with pd.ExcelWriter(output_dir / f"glp_{modality}_matched_analysis.xlsx") as writer:
        (did if not did.empty else pd.DataFrame({"note": ["no measurements met min pairs"]})).to_excel(
            writer, sheet_name="did_summary", index=False
        )
        balance.to_excel(writer, sheet_name="covariate_balance", index=False)
        (pairs if not pairs.empty else pd.DataFrame({"note": ["no matched pairs"]})).to_excel(
            writer, sheet_name="matched_pairs", index=False
        )
        if not deltas.empty:
            deltas.to_excel(writer, sheet_name="participant_deltas", index=False)

    plot_did(did, figure_dir, modality)
    write_readme(
        analysis_dir, modality, approach, snapshot_dir, caliper_sd, treated, pairs, balance, did,
        k_neighbors=k_neighbors,
    )
    logging.info("[%s/%s] wrote analysis to %s", modality, approach, analysis_dir)

    return {
        "modality": modality,
        "approach": approach,
        "analysis_dir": analysis_dir,
        "did": did,
        "balance": balance,
        "glp_starters": int(treated["RegistrationCode"].nunique()),
        "treated_eligible": n_treated_units,
        "control_eligible": n_control_units,
        "matched_pairs": int(len(pairs)),
        "matched_treated": int(pairs["treated_reg"].nunique()) if not pairs.empty else 0,
        "matched_controls": int(pairs["control_reg"].nunique()) if not pairs.empty else 0,
        "fallback_matches": int((~pairs["within_caliper"]).sum()) if not pairs.empty else 0,
    }


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    analysis_dir = args.analysis_dir or Path(f"glp_{args.outcome_modality}_{args.approach}_matched_analysis")
    log_dir = resolve_output_path(analysis_dir, script_dir) / "logs"
    setup_logging(log_dir)
    run_analysis(
        modality=args.outcome_modality,
        approach=args.approach,
        snapshot_dir=args.snapshot_dir,
        script_dir=script_dir,
        analysis_dir=analysis_dir,
        caliper_sd=args.caliper_sd,
        min_pairs=args.min_pairs,
        seed=args.seed,
        k_neighbors=args.k_neighbors,
        diabetes_timing=args.diabetes_timing,
    )


if __name__ == "__main__":
    main()
