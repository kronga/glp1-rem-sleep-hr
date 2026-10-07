"""Bootstrap the paper-facing sleep estimates with propensity refitting/rematching.

The main reanalysis reports a conventional fixed-match participant bootstrap.
This sensitivity repeats the design step inside every replicate:

1. resample treated participants with replacement;
2. retain the full eligible never-GLP control pool;
3. refit the propensity model;
4. repeat the globally optimal 1:1, no-reuse, hard-caliper match; and
5. re-estimate the sleep-state heart-rate effects and formal contrasts.

Resampling the treated clusters is appropriate for the ATT target while keeping
the empirical control reservoir fixed. Duplicate treated draws receive bootstrap
identifiers, so each draw can be matched independently without permitting
control reuse. No participant identifiers are written to the outputs.

Run with the LabData Python environment, for example:

    PYTHONPATH=<repo>/src:<run>/src \
      /net/mraid20/export/jasmine/david/anaconda3/bin/python3 \
      steps/29_sleep_bootstrap_rematching.py --n-bootstrap 500
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util


RUN = Path(__file__).resolve().parents[1]
PIPELINE_PATH = RUN / "steps" / "26_paper_reanalysis.py"
OUT = RUN / "outputs" / "paper_reanalysis"
DEFAULT_N_BOOTSTRAP = 500

HEART_RATE_ENDPOINTS = [
    "heart_rate_mean_during_sleep",
    "heart_rate_mean_during_rem",
    "heart_rate_mean_during_nrem",
    "heart_rate_mean_during_wake",
]
CONTRASTS = {
    "REM minus NREM heart-rate DiD": (
        "heart_rate_mean_during_rem",
        "heart_rate_mean_during_nrem",
    ),
    "REM minus wake heart-rate DiD": (
        "heart_rate_mean_during_rem",
        "heart_rate_mean_during_wake",
    ),
    "NREM minus wake heart-rate DiD": (
        "heart_rate_mean_during_nrem",
        "heart_rate_mean_during_wake",
    ),
}


def load_pipeline():
    spec = importlib.util.spec_from_file_location("paper_reanalysis", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resample_treated_units(
    units: pd.DataFrame,
    deltas: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample treated clusters and attach private-in-memory bootstrap ids."""
    treated = units[units["treated"].eq(1)].drop_duplicates("participant_id").copy()
    controls = units[units["treated"].eq(0)].copy()
    draw_indices = rng.integers(0, len(treated), size=len(treated))
    draws = treated.iloc[draw_indices].copy().reset_index(drop=True)
    draws["source_participant_id"] = draws["participant_id"].astype(str)
    draws["participant_id"] = [f"bootstrap_treated_{index}" for index in range(len(draws))]

    treated_deltas = deltas.merge(
        draws[["participant_id", "source_participant_id", "pre_stage", "post_stage"]],
        left_on=["participant_id", "pre_stage", "post_stage"],
        right_on=["source_participant_id", "pre_stage", "post_stage"],
        how="inner",
        suffixes=("_source", ""),
    )
    treated_deltas = treated_deltas.drop(
        columns=["participant_id_source", "source_participant_id"]
    )
    treated_deltas = treated_deltas[
        [
            "participant_id",
            "pre_stage",
            "post_stage",
            "measurement",
            "pre_value",
            "post_value",
            "delta",
        ]
    ]
    control_deltas = deltas[
        deltas["participant_id"].isin(controls["participant_id"])
    ].copy()
    bootstrap_units = pd.concat(
        [draws.drop(columns="source_participant_id"), controls],
        ignore_index=True,
    )
    bootstrap_deltas = pd.concat(
        [treated_deltas, control_deltas],
        ignore_index=True,
    )
    return bootstrap_units, bootstrap_deltas


def endpoint_records(pair_rows: pd.DataFrame, replicate: int) -> list[dict]:
    records: list[dict] = []
    values: dict[str, pd.DataFrame] = {}
    for endpoint in HEART_RATE_ENDPOINTS:
        group = pair_rows[pair_rows["measurement"].eq(endpoint)][
            ["treated_id", "did"]
        ].dropna()
        values[endpoint] = group
        records.append(
            {
                "replicate": replicate,
                "estimand_type": "endpoint",
                "estimand": endpoint,
                "n_pairs": group["treated_id"].nunique(),
                "estimate": group["did"].mean() if len(group) else np.nan,
            }
        )
    for label, (endpoint_a, endpoint_b) in CONTRASTS.items():
        paired = values[endpoint_a].rename(columns={"did": "a"}).merge(
            values[endpoint_b].rename(columns={"did": "b"}),
            on="treated_id",
            how="inner",
        )
        records.append(
            {
                "replicate": replicate,
                "estimand_type": "contrast",
                "estimand": label,
                "n_pairs": paired["treated_id"].nunique(),
                "estimate": (paired["a"] - paired["b"]).mean() if len(paired) else np.nan,
            }
        )
    return records


def observed_records(pipeline, analysis: dict) -> pd.DataFrame:
    rows = analysis["primary_rows"]
    records: list[dict] = []
    for endpoint in HEART_RATE_ENDPOINTS:
        group = rows[rows["measurement"].eq(endpoint)]
        records.append(
            {
                "estimand_type": "endpoint",
                "estimand": endpoint,
                "observed_estimate": group["did"].mean(),
                "observed_n_pairs": group["treated_id"].nunique(),
            }
        )
    for label, (endpoint_a, endpoint_b) in CONTRASTS.items():
        result = pipeline.contrast_model(rows, endpoint_a, endpoint_b, label)
        records.append(
            {
                "estimand_type": "contrast",
                "estimand": label,
                "observed_estimate": result.get("difference", np.nan),
                "observed_n_pairs": result.get("n", 0),
            }
        )
    return pd.DataFrame(records)


def summarize(
    replicates: pd.DataFrame,
    observed: pd.DataFrame,
    n_requested: int,
) -> pd.DataFrame:
    records: list[dict] = []
    observed_lookup = observed.set_index(["estimand_type", "estimand"])
    for (estimand_type, estimand), group in replicates.groupby(
        ["estimand_type", "estimand"]
    ):
        estimates = group["estimate"].dropna().to_numpy(float)
        observed_row = observed_lookup.loc[(estimand_type, estimand)]
        if len(estimates):
            lower = float(np.quantile(estimates, 0.025))
            upper = float(np.quantile(estimates, 0.975))
            p_lower = (np.sum(estimates <= 0) + 1) / (len(estimates) + 1)
            p_upper = (np.sum(estimates >= 0) + 1) / (len(estimates) + 1)
            bootstrap_p = min(1.0, 2 * min(p_lower, p_upper))
        else:
            lower = upper = bootstrap_p = np.nan
        records.append(
            {
                "estimand_type": estimand_type,
                "estimand": estimand,
                "observed_estimate": observed_row["observed_estimate"],
                "observed_n_pairs": int(observed_row["observed_n_pairs"]),
                "bootstrap_replicates_requested": n_requested,
                "bootstrap_replicates_estimable": len(estimates),
                "bootstrap_mean": np.mean(estimates) if len(estimates) else np.nan,
                "bootstrap_ci_low": lower,
                "bootstrap_ci_high": upper,
                "bootstrap_two_sided_p": bootstrap_p,
                "median_matched_pairs": group["n_pairs"].median(),
                "minimum_matched_pairs": group["n_pairs"].min(),
            }
        )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUT / "bootstrap_rematching.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    pipeline = load_pipeline()
    primary, _, _ = pipeline.load_exposure_cohorts()
    ever_glp = pipeline.ever_glp_participants(primary)
    covariates = pipeline.load_covariates()
    sleep_values = pipeline.load_snapshot_panel("sleep")
    analysis = pipeline.analyse_modality(
        "sleep",
        sleep_values,
        primary,
        covariates,
        ever_glp,
    )
    if not analysis:
        raise RuntimeError("The primary sleep analysis could not be reconstructed")

    base_units = analysis["units"].drop(columns=["propensity", "ps_logit"])
    base_deltas = analysis["deltas"]
    observed = observed_records(pipeline, analysis)
    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    successful_matches = 0
    for replicate in range(args.n_bootstrap):
        units, deltas = resample_treated_units(base_units, base_deltas, rng)
        units = pipeline.propensity_score(
            units,
            treatment_col="treated",
            covariate_cols=pipeline.MATCH_COVARIATES,
        )
        pairs = pipeline.optimal_pair_match(
            units,
            covariate_cols=pipeline.MATCH_COVARIATES,
            treatment_col="treated",
            id_col="participant_id",
            exact_cols=pipeline.exact_columns("sleep"),
            caliper_sd=pipeline.CALIPER_SD,
            seed=args.seed + replicate,
        )
        if pairs.empty:
            continue
        successful_matches += 1
        rows = pipeline.pair_effects(pairs, deltas)
        records.extend(endpoint_records(rows, replicate))
        if (replicate + 1) % 25 == 0:
            logging.info(
                "Completed %d/%d replicates (%d with a match)",
                replicate + 1,
                args.n_bootstrap,
                successful_matches,
            )

    replicate_frame = pd.DataFrame(records)
    if replicate_frame.empty:
        raise RuntimeError("No bootstrap replicate produced estimable sleep results")
    summary = summarize(replicate_frame, observed, args.n_bootstrap)
    replicate_frame.to_csv(OUT / "bootstrap_rematched_sleep_replicates.csv", index=False)
    summary.to_csv(OUT / "bootstrap_rematched_sleep_summary.csv", index=False)
    logging.info("Bootstrap rematching summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
