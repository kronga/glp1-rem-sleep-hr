#!/usr/bin/env python
"""Prescription medication visit-transition analysis for 10K.

This script is focused on one question:
Which prescription medication families are newly reported after a prior visit
where that medication was not reported?

It writes clean participant-level transition tables for future joins with other
datasets, plus compact family/drug-type summaries and stacked bar plots.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from aggregate_medication_stats import (
    attach_start_date_and_usage_months,
    filter_prescription_only,
    load_medications,
    load_registration_codes,
    prepare_medication_events,
    resolve_output_path,
    stage_sort_key,
    true_start_events,
    validate_and_fix_prior_visit_dates,
)


APPROACH_STRICT = "strict_visit_reported"
APPROACH_CARRY_FORWARD = "carry_forward_from_call"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze prescription medication reports with prior/later visits."
    )
    parser.add_argument(
        "--analysis-dir",
        default="visit_transition_analysis",
        type=Path,
        help="Clean analysis directory for outputs, figures, and logs.",
    )
    parser.add_argument(
        "--reg-ids-csv",
        type=Path,
        help="Optional CSV containing RegistrationCode values to restrict the analysis.",
    )
    parser.add_argument(
        "--gen-cache",
        action="store_true",
        help="Regenerate the LabData medication cache before aggregating.",
    )
    parser.add_argument(
        "--top-families",
        default=10,
        type=int,
        help="Number of top family/purpose groups to plot.",
    )
    parser.add_argument(
        "--top-drug-types",
        default=8,
        type=int,
        help="Number of drug types shown separately within each family; the rest are collapsed to Other.",
    )
    parser.add_argument(
        "--include-unclear-family",
        action="store_true",
        help="Include Other/unclear purpose in family plots.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear previous files inside the analysis directory before writing.",
    )
    return parser.parse_args()


def setup_clean_analysis_dir(path: Path, keep_existing: bool) -> tuple[Path, Path, Path]:
    if path.exists() and not keep_existing:
        shutil.rmtree(path)
    output_dir = path / "outputs"
    figure_dir = path / "figures"
    log_dir = path / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, figure_dir, log_dir


def setup_logging(log_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "visit_transition_analysis.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def is_visit_stage(stage: object) -> bool:
    if pd.isna(stage):
        return False
    stage = str(stage)
    return stage == "baseline" or stage.endswith(" visit")


def latest_prior_absent_visit(
    observed_visit_stages: list[str],
    present_stage_set: set[str],
    reported_order: int,
    stage_rank: dict[str, int],
) -> str | None:
    prior_absent_visits = [
        stage
        for stage in observed_visit_stages
        if stage_rank[stage] < reported_order and stage not in present_stage_set
    ]
    return prior_absent_visits[-1] if prior_absent_visits else None


def first_confirmation_visit(
    observed_visit_stages: list[str],
    reported_order: int,
    stage_rank: dict[str, int],
) -> str | None:
    for stage in observed_visit_stages:
        if stage_rank[stage] >= reported_order:
            return stage
    return None


def has_explicit_stop_between(
    stop_stages: set[str],
    report_order: int,
    confirmation_order: int,
    stage_rank: dict[str, int],
) -> bool:
    return any(
        report_order < stage_rank[stage] <= confirmation_order
        for stage in stop_stages
        if stage in stage_rank
    )


def build_transition_tables(events: pd.DataFrame, medication_metadata: pd.DataFrame) -> pd.DataFrame:
    starts = true_start_events(events)
    stops = events[events["Start"].fillna(False) == False].copy()

    stages = sorted(events["specific_stage"].dropna().unique(), key=stage_sort_key)
    stage_rank = {stage: i for i, stage in enumerate(stages)}
    visit_stages = {stage for stage in stages if is_visit_stage(stage)}

    observed_by_participant = {
        reg: sorted(group["specific_stage"].dropna().unique(), key=stage_sort_key)
        for reg, group in events[["RegistrationCode", "specific_stage"]]
        .drop_duplicates()
        .groupby("RegistrationCode")
    }
    present_by_pair = {
        pair: sorted(group["specific_stage"].dropna().unique(), key=stage_sort_key)
        for pair, group in starts[["RegistrationCode", "medication", "specific_stage"]]
        .drop_duplicates()
        .groupby(["RegistrationCode", "medication"])
    }
    stop_by_pair = {
        pair: set(group["specific_stage"].dropna().unique())
        for pair, group in stops[["RegistrationCode", "medication", "specific_stage"]]
        .drop_duplicates()
        .groupby(["RegistrationCode", "medication"])
    }

    records = []
    for (reg, medication), present_stages in present_by_pair.items():
        observed_stages = observed_by_participant.get(reg, [])
        observed_visit_stages = [stage for stage in observed_stages if stage in visit_stages]
        if not observed_visit_stages:
            continue

        present_stage_set = set(present_stages)
        stop_stage_set = stop_by_pair.get((reg, medication), set())

        strict_record = choose_strict_transition(
            reg,
            medication,
            present_stages,
            present_stage_set,
            observed_visit_stages,
            stage_rank,
        )
        if strict_record is not None:
            records.append(strict_record)

        carry_record = choose_carry_forward_transition(
            reg,
            medication,
            present_stages,
            present_stage_set,
            observed_visit_stages,
            stop_stage_set,
            stage_rank,
        )
        if carry_record is not None:
            records.append(carry_record)

    transitions = pd.DataFrame.from_records(records)
    if transitions.empty:
        return transitions

    metadata_cols = [
        "medication",
        "category",
        "primary_condition",
        "drug_family_or_purpose",
        "Generic",
        "ATC",
    ]
    metadata_cols = [col for col in metadata_cols if col in medication_metadata.columns]
    transitions = transitions.merge(
        medication_metadata[metadata_cols].drop_duplicates("medication"),
        on="medication",
        how="left",
    )
    transitions["drug_family_or_purpose"] = transitions["drug_family_or_purpose"].fillna("Other/unclear purpose")
    transitions["drug_type"] = transitions["Generic"].fillna("").astype(str).str.strip()
    missing_drug_type = transitions["drug_type"].isin(["", "nan", "UNKNOWN"])
    transitions.loc[missing_drug_type, "drug_type"] = transitions.loc[missing_drug_type, "medication"]
    return transitions


def choose_strict_transition(
    reg: str,
    medication: str,
    present_stages: list[str],
    present_stage_set: set[str],
    observed_visit_stages: list[str],
    stage_rank: dict[str, int],
) -> dict[str, object] | None:
    for reported_stage in present_stages:
        if reported_stage not in stage_rank or not is_visit_stage(reported_stage):
            continue
        prior_stage = latest_prior_absent_visit(
            observed_visit_stages,
            present_stage_set,
            stage_rank[reported_stage],
            stage_rank,
        )
        if prior_stage is None:
            continue
        return {
            "approach": APPROACH_STRICT,
            "RegistrationCode": reg,
            "medication": medication,
            "prior_absent_visit_stage": prior_stage,
            "reported_stage": reported_stage,
            "confirmation_visit_stage": reported_stage,
            "prior_absent_visit_stage_order": stage_rank[prior_stage],
            "reported_stage_order": stage_rank[reported_stage],
            "confirmation_visit_stage_order": stage_rank[reported_stage],
            "reported_in_visit": True,
            "carried_forward_from_call": False,
            "explicit_stop_before_confirmation": False,
        }
    return None


def choose_carry_forward_transition(
    reg: str,
    medication: str,
    present_stages: list[str],
    present_stage_set: set[str],
    observed_visit_stages: list[str],
    stop_stage_set: set[str],
    stage_rank: dict[str, int],
) -> dict[str, object] | None:
    for reported_stage in present_stages:
        if reported_stage not in stage_rank:
            continue
        report_order = stage_rank[reported_stage]
        prior_stage = latest_prior_absent_visit(
            observed_visit_stages,
            present_stage_set,
            report_order,
            stage_rank,
        )
        if prior_stage is None:
            continue
        confirmation_stage = first_confirmation_visit(
            observed_visit_stages,
            report_order,
            stage_rank,
        )
        if confirmation_stage is None:
            continue
        explicit_stop = has_explicit_stop_between(
            stop_stage_set,
            report_order,
            stage_rank[confirmation_stage],
            stage_rank,
        )
        if explicit_stop:
            continue
        return {
            "approach": APPROACH_CARRY_FORWARD,
            "RegistrationCode": reg,
            "medication": medication,
            "prior_absent_visit_stage": prior_stage,
            "reported_stage": reported_stage,
            "confirmation_visit_stage": confirmation_stage,
            "prior_absent_visit_stage_order": stage_rank[prior_stage],
            "reported_stage_order": report_order,
            "confirmation_visit_stage_order": stage_rank[confirmation_stage],
            "reported_in_visit": is_visit_stage(reported_stage),
            "carried_forward_from_call": not is_visit_stage(reported_stage),
            "explicit_stop_before_confirmation": False,
        }
    return None


def summarize_family_drug_type(
    transitions: pd.DataFrame,
    top_families: int,
    top_drug_types: int,
    include_unclear_family: bool,
) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame(
            columns=["approach", "drug_family_or_purpose", "drug_type", "participant_count"]
        )

    plot_source = transitions.copy()
    if not include_unclear_family:
        plot_source = plot_source[plot_source["drug_family_or_purpose"] != "Other/unclear purpose"]

    rows = []
    for approach, approach_df in plot_source.groupby("approach"):
        pairs = approach_df[
            ["RegistrationCode", "drug_family_or_purpose", "drug_type"]
        ].drop_duplicates()
        counts = (
            pairs.groupby(["drug_family_or_purpose", "drug_type"])["RegistrationCode"]
            .nunique()
            .reset_index(name="participant_count")
        )
        family_order = (
            counts.groupby("drug_family_or_purpose")["participant_count"]
            .sum()
            .sort_values(ascending=False)
            .head(top_families)
            .index
        )
        for family in family_order:
            family_counts = counts[counts["drug_family_or_purpose"] == family].sort_values(
                "participant_count", ascending=False
            )
            for _, row in family_counts.head(top_drug_types).iterrows():
                rows.append(
                    {
                        "approach": approach,
                        "drug_family_or_purpose": family,
                        "drug_type": row["drug_type"],
                        "participant_count": int(row["participant_count"]),
                    }
                )
            other = family_counts.iloc[top_drug_types:]["participant_count"].sum()
            if other > 0:
                rows.append(
                    {
                        "approach": approach,
                        "drug_family_or_purpose": family,
                        "drug_type": "Other",
                        "participant_count": int(other),
                    }
                )
    return pd.DataFrame(rows)


def summarize_outputs(transitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if transitions.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    family_summary = (
        transitions.groupby(["approach", "drug_family_or_purpose"])
        .agg(
            participant_count=("RegistrationCode", "nunique"),
            participant_medication_pairs=("RegistrationCode", "size"),
            medication_count=("medication", "nunique"),
        )
        .reset_index()
        .sort_values(["approach", "participant_medication_pairs"], ascending=[True, False])
    )

    medication_summary = (
        transitions.groupby(
            [
                "approach",
                "medication",
                "drug_family_or_purpose",
                "drug_type",
                "category",
                "Generic",
                "ATC",
            ],
            dropna=False,
        )
        .agg(
            participant_count=("RegistrationCode", "nunique"),
            participant_medication_pairs=("RegistrationCode", "size"),
        )
        .reset_index()
        .sort_values(["approach", "participant_count"], ascending=[True, False])
    )

    stage_summary = (
        transitions.groupby(
            ["approach", "prior_absent_visit_stage", "reported_stage", "confirmation_visit_stage"]
        )
        .agg(
            participant_count=("RegistrationCode", "nunique"),
            participant_medication_pairs=("RegistrationCode", "size"),
            medication_count=("medication", "nunique"),
        )
        .reset_index()
        .sort_values(["approach", "participant_medication_pairs"], ascending=[True, False])
    )
    return family_summary, medication_summary, stage_summary


def summarize_drug_name_counts(transitions: pd.DataFrame, approach: str) -> pd.DataFrame:
    columns = [
        "drug-family",
        "generic drug-name",
        "commercial drug name",
        "number of participants that have a visit prior reporting taking drug and with a visit after reporting",
    ]
    if transitions.empty:
        return pd.DataFrame(columns=columns)

    approach_df = transitions[transitions["approach"] == approach].copy()
    if approach_df.empty:
        return pd.DataFrame(columns=columns)

    approach_df["generic drug-name"] = (
        approach_df["Generic"].fillna("").astype(str).str.strip()
    )
    missing_generic = approach_df["generic drug-name"].isin(["", "nan", "UNKNOWN"])
    approach_df.loc[missing_generic, "generic drug-name"] = approach_df.loc[
        missing_generic, "drug_type"
    ]

    summary = (
        approach_df.groupby(
            ["drug_family_or_purpose", "generic drug-name", "medication"],
            dropna=False,
        )["RegistrationCode"]
        .nunique()
        .reset_index()
        .rename(
            columns={
                "drug_family_or_purpose": "drug-family",
                "medication": "commercial drug name",
                "RegistrationCode": columns[-1],
            }
        )
        .sort_values(
            [columns[-1], "drug-family", "generic drug-name", "commercial drug name"],
            ascending=[False, True, True, True],
        )
    )
    return summary[columns]


def plot_stacked_family_bars(
    family_drug_type: pd.DataFrame,
    figure_dir: Path,
) -> None:
    if family_drug_type.empty:
        return
    for approach, approach_df in family_drug_type.groupby("approach"):
        wide = approach_df.pivot_table(
            index="drug_family_or_purpose",
            columns="drug_type",
            values="participant_count",
            aggfunc="sum",
            fill_value=0,
        )
        wide = wide.loc[wide.sum(axis=1).sort_values().index]
        wide = wide[wide.sum(axis=0).sort_values(ascending=False).index]

        title = {
            APPROACH_STRICT: "Strict: medication reported at later visit",
            APPROACH_CARRY_FORWARD: "Less stringent: call report carried to later visit",
        }[approach]
        xlabel = {
            APPROACH_STRICT: "Participants with prior visit non-report and later visit report",
            APPROACH_CARRY_FORWARD: "Participants with prior visit non-report and later/same visit after report",
        }[approach]

        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 14,
                "axes.labelsize": 11,
                "figure.titlesize": 18,
                "font.family": "DejaVu Sans",
            }
        )
        fig, ax = plt.subplots(figsize=(18, 10))
        colors = (
            list(plt.get_cmap("tab20").colors)
            + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors)
        )
        wide.plot(kind="barh", stacked=True, ax=ax, color=colors[: len(wide.columns)], width=0.78)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(
            0,
            1.035,
            "Prescription-only cohort. Bars are top family/purpose groups; segments are generic drug types.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            color="#444444",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        max_total = max(wide.sum(axis=1))
        for y, total in enumerate(wide.sum(axis=1)):
            ax.text(total + max_total * 0.01, y, f"{int(total):,}", va="center", fontsize=9)
        ax.set_xlim(0, max_total * 1.22)
        ax.legend(
            title="Generic drug type",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )
        fig.tight_layout()
        filename = {
            APPROACH_STRICT: "strict_visit_reported_family_drug_type.png",
            APPROACH_CARRY_FORWARD: "carry_forward_family_drug_type.png",
        }[approach]
        fig.savefig(figure_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)


def validate_transitions(transitions: pd.DataFrame) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    validation = transitions.copy()
    validation["prior_is_visit"] = validation["prior_absent_visit_stage"].map(is_visit_stage)
    validation["confirmation_is_visit"] = validation["confirmation_visit_stage"].map(is_visit_stage)
    validation["order_ok"] = (
        validation["prior_absent_visit_stage_order"] < validation["reported_stage_order"]
    ) & (
        validation["reported_stage_order"] <= validation["confirmation_visit_stage_order"]
    )
    validation["strict_reported_in_visit_ok"] = (
        validation["approach"].ne(APPROACH_STRICT) | validation["reported_in_visit"]
    )
    return (
        validation.groupby("approach")
        .agg(
            rows=("RegistrationCode", "size"),
            prior_visit_ok=("prior_is_visit", "sum"),
            confirmation_visit_ok=("confirmation_is_visit", "sum"),
            order_ok=("order_ok", "sum"),
            strict_reported_in_visit_ok=("strict_reported_in_visit_ok", "sum"),
            unique_participants=("RegistrationCode", "nunique"),
            unique_medications=("medication", "nunique"),
        )
        .reset_index()
    )


def write_readme(
    analysis_dir: Path,
    transitions: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    strict_rows = int((transitions["approach"] == APPROACH_STRICT).sum()) if not transitions.empty else 0
    carry_rows = int((transitions["approach"] == APPROACH_CARRY_FORWARD).sum()) if not transitions.empty else 0
    text = f"""# Visit-Transition Prescription Medication Analysis

This directory contains the clean, future-analysis-ready outputs for prescription medications only.
Supplement-like products are excluded before any transition analysis.

## Definitions

- **Strict (`{APPROACH_STRICT}`)**: participant has an earlier baseline/visit where the medication was not reported, and the medication is later reported at a visit.
- **Less stringent (`{APPROACH_CARRY_FORWARD}`)**: participant has an earlier baseline/visit where the medication was not reported, and the medication is later reported at a call or visit; if reported at a call, use is carried forward to the next observed visit unless an explicit stop is observed before that visit.

Both approaches require:

- a prior non-report at `baseline` or a numbered `visit`,
- a report stage after that prior visit,
- a confirmation visit at the same stage or after the report stage,
- numeric stage ordering: `baseline`, `01 call`, `02 visit`, `03 call`, `04 visit`, `05 call`, `06 visit`, ...

## Row Counts

- Strict participant-medication rows: **{strict_rows:,}**
- Less-stringent participant-medication rows: **{carry_rows:,}**

## Key Files

- `outputs/participant_medication_transitions.csv`: all participant-medication rows for both approaches.
- `outputs/strict_visit_reported_participant_medication_transitions.csv`: strict-only participant rows.
- `outputs/carry_forward_participant_medication_transitions.csv`: less-stringent participant rows.
- `outputs/family_summary.csv`: family/purpose counts by approach.
- `outputs/medication_summary.csv`: medication-level counts by approach.
- `outputs/carry_forward_drug_family_generic_commercial_counts.csv`: less-stringent counts by drug family, generic drug name, and commercial drug name.
- `outputs/strict_drug_family_generic_commercial_counts.csv`: strict counts by drug family, generic drug name, and commercial drug name.
- `outputs/stage_transition_summary.csv`: prior visit, report stage, and confirmation visit transition counts.
- `outputs/family_drug_type_counts_for_plots.csv`: exact counts used in the stacked figures.
- `outputs/excluded_supplement_like_medications.csv`: supplement filter audit table.
- `figures/strict_visit_reported_family_drug_type.png`: strict family/drug-type figure.
- `figures/carry_forward_family_drug_type.png`: less-stringent family/drug-type figure.
- `outputs/visit_transition_analysis.xlsx`: workbook version of the main tables.

## Validation

Every output row is validated to ensure prior stage is a visit/baseline, confirmation stage is a visit/baseline, and the stage order is prior visit < report <= confirmation visit.

"""
    if not validation.empty:
        text += validation.to_markdown(index=False)
        text += "\n"
    (analysis_dir / "README.md").write_text(text)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    analysis_dir = resolve_output_path(args.analysis_dir, script_dir)
    output_dir, figure_dir, log_dir = setup_clean_analysis_dir(analysis_dir, args.keep_existing)
    setup_logging(log_dir)

    reg_ids = load_registration_codes(args.reg_ids_csv)
    data = load_medications(gen_cache=args.gen_cache, reg_ids=reg_ids)
    events, medication_metadata = prepare_medication_events(data)
    events, medication_metadata, excluded = filter_prescription_only(events, medication_metadata)
    logging.info("Prescription-only events: %d", len(events))
    logging.info("Excluded supplement-like medication names: %d", len(excluded))

    medication_metadata = medication_metadata.copy()
    from aggregate_medication_stats import add_medication_categories

    medication_metadata = add_medication_categories(medication_metadata)
    transitions = build_transition_tables(events, medication_metadata)
    transitions = attach_start_date_and_usage_months(transitions, events)
    transitions = validate_and_fix_prior_visit_dates(transitions, events)
    n_fixed = int(transitions["prior_visit_date_fixed"].sum())
    n_invalid = int((~transitions["prior_visit_date_valid"]).sum())
    logging.info(
        "Prior visit date check: %d rows fixed (prior pushed back), %d rows invalid "
        "(start predates all prior absent visits — drug started before study baseline)",
        n_fixed,
        n_invalid,
    )
    validation = validate_transitions(transitions)
    family_summary, medication_summary, stage_summary = summarize_outputs(transitions)
    carry_forward_drug_name_counts = summarize_drug_name_counts(
        transitions, APPROACH_CARRY_FORWARD
    )
    strict_drug_name_counts = summarize_drug_name_counts(transitions, APPROACH_STRICT)
    family_drug_type = summarize_family_drug_type(
        transitions,
        top_families=args.top_families,
        top_drug_types=args.top_drug_types,
        include_unclear_family=args.include_unclear_family,
    )

    transitions.to_csv(output_dir / "participant_medication_transitions.csv", index=False)
    transitions[transitions["approach"] == APPROACH_STRICT].to_csv(
        output_dir / "strict_visit_reported_participant_medication_transitions.csv",
        index=False,
    )
    transitions[transitions["approach"] == APPROACH_CARRY_FORWARD].to_csv(
        output_dir / "carry_forward_participant_medication_transitions.csv",
        index=False,
    )
    family_summary.to_csv(output_dir / "family_summary.csv", index=False)
    medication_summary.to_csv(output_dir / "medication_summary.csv", index=False)
    carry_forward_drug_name_counts.to_csv(
        output_dir / "carry_forward_drug_family_generic_commercial_counts.csv",
        index=False,
    )
    strict_drug_name_counts.to_csv(
        output_dir / "strict_drug_family_generic_commercial_counts.csv",
        index=False,
    )
    stage_summary.to_csv(output_dir / "stage_transition_summary.csv", index=False)
    validation.to_csv(output_dir / "validation_summary.csv", index=False)
    family_drug_type.to_csv(output_dir / "family_drug_type_counts_for_plots.csv", index=False)
    excluded.to_csv(output_dir / "excluded_supplement_like_medications.csv", index=False)

    with pd.ExcelWriter(output_dir / "visit_transition_analysis.xlsx") as writer:
        transitions.to_excel(writer, sheet_name="participant_transitions", index=False)
        family_summary.to_excel(writer, sheet_name="family_summary", index=False)
        medication_summary.to_excel(writer, sheet_name="medication_summary", index=False)
        carry_forward_drug_name_counts.to_excel(
            writer, sheet_name="carry_name_counts", index=False
        )
        strict_drug_name_counts.to_excel(writer, sheet_name="strict_name_counts", index=False)
        stage_summary.to_excel(writer, sheet_name="stage_summary", index=False)
        validation.to_excel(writer, sheet_name="validation", index=False)
        family_drug_type.to_excel(writer, sheet_name="plot_counts", index=False)
        excluded.to_excel(writer, sheet_name="excluded_supplements", index=False)

    plot_stacked_family_bars(family_drug_type, figure_dir)
    write_readme(analysis_dir, transitions, validation)

    logging.info("Wrote clean analysis to %s", analysis_dir)
    if not validation.empty:
        logging.info("\n%s", validation.to_string(index=False))


if __name__ == "__main__":
    main()
