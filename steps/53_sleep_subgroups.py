"""Subgroup estimates for the primary sleep heart-rate outcomes on the primary matched pairs.

Reuses the sleep match of ``steps/48_unified_cohort.py`` through the ``Pipeline`` of
``steps/51_unified_sleep_robustness.py`` (the rebuilt match is checked against the versioned
step-48 run), then splits the matched pairs by the initiator's value at the baseline visit:

* sex (matched exactly, so both members of a pair share it);
* age below or at least the matched initiators' median;
* body-mass index below or at least 30 kg/m2;
* antihypertensive therapy (which includes rate-lowering drugs) or not.

For REM sleep heart rate and the REM minus NREM contrast, each subgroup's DiD has the
primary pair-bootstrap interval (step 26 ``_bootstrap_mean_ci``), and the difference between
the two subgroups is the coefficient of an HC3 regression of the pair DiD on the subgroup
indicator (step 26 ``fit_hc3``). Diabetes is not split: too few matched initiators had it.
The subgroups were chosen after the primary analysis and are descriptive.

Outputs (aggregate only, no identifiers): ``outputs/sleep_subgroups/``

    python steps/53_sleep_subgroups.py
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
OUT = RUN / "outputs" / "sleep_subgroups"

MEASUREMENTS = {
    "heart_rate_mean_during_rem": "REM sleep heart rate",
    "heart_rate_rem_minus_nrem": "REM minus NREM heart rate",
}
BMI_CUT = 30.0
SEED = 20261018


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pair_rows_with_initiator_baseline(sleep: dict) -> pd.DataFrame:
    """Per-pair DiD rows with the initiator's baseline covariates attached."""
    keys = ["participant_id", "pre_stage", "post_stage"]
    initiators = sleep["units"].loc[sleep["units"].treated.eq(1),
                                    keys + ["age", "gender", "bmi", "diabetic",
                                            "on_antihypertensive"]]
    initiators = initiators.rename(columns={"participant_id": "treated_id"})
    rows = sleep["rows"].loc[sleep["rows"].measurement.isin(MEASUREMENTS),
                             ["treated_id", "pre_stage", "post_stage", "measurement", "did"]]
    return rows.merge(initiators, on=["treated_id", "pre_stage", "post_stage"],
                      how="left", validate="many_to_one")


def subgroup_definitions(matched_initiators: pd.DataFrame) -> dict[str, tuple]:
    """Variable -> (reference label, comparison label, rule marking the comparison level)."""
    age_cut = float(matched_initiators.age.median())
    return {
        "sex": ("Male", "Female", lambda f: f.gender.eq(1.0)),
        "age": (f"Age < {age_cut:.1f} years", f"Age >= {age_cut:.1f} years",
                lambda f: f.age.ge(age_cut)),
        "bmi": (f"BMI < {BMI_CUT:.0f} kg/m2", f"BMI >= {BMI_CUT:.0f} kg/m2",
                lambda f: f.bmi.ge(BMI_CUT)),
        "antihypertensive": ("No antihypertensive therapy", "Antihypertensive therapy",
                             lambda f: f.on_antihypertensive.eq(1.0)),
    }


def subgroup_estimates(step26, rows: pd.DataFrame, definitions: dict,
                       rng: np.random.Generator) -> pd.DataFrame:
    records = []
    for measurement, label in MEASUREMENTS.items():
        group = rows[rows.measurement.eq(measurement)].dropna(subset=["did"])
        for variable, (reference, comparison, rule) in definitions.items():
            indicator = rule(group).astype(float).to_numpy()
            did = group.did.to_numpy(float)
            design = np.column_stack([np.ones(len(did)), indicator])
            beta, se, p = step26.fit_hc3(did, design)
            for level, mask in [(reference, indicator == 0), (comparison, indicator == 1)]:
                low, high = step26._bootstrap_mean_ci(did[mask], rng)
                records.append({
                    "measurement": measurement, "outcome": label,
                    "subgroup_variable": variable, "level": level,
                    "is_reference": level == reference,
                    "n_pairs": int(mask.sum()), "estimate": float(did[mask].mean()),
                    "ci_low": low, "ci_high": high,
                    "difference_vs_reference": float(beta[1]),
                    "difference_ci_low": float(beta[1] - 1.96 * se[1]),
                    "difference_ci_high": float(beta[1] + 1.96 * se[1]),
                    "difference_p": float(p[1]),
                })
    return pd.DataFrame(records)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    step51 = load_module("subgroup_step51", RUN / "steps/51_unified_sleep_robustness.py")
    pipe = step51.Pipeline()
    sleep = pipe.rebuild("sleep", "semaglutide", step51.FINAL_RUN)
    rows = pair_rows_with_initiator_baseline(sleep)
    if rows[["age", "gender", "bmi", "on_antihypertensive"]].isna().any().any():
        raise AssertionError("an initiator in the matched pairs lacks a baseline covariate")

    # The whole-sample means must reproduce the paper's primary estimates on the same pairs.
    primary = sleep["result"].set_index("measurement")
    for measurement in MEASUREMENTS:
        group = rows[rows.measurement.eq(measurement)].dropna(subset=["did"])
        expected = primary.loc[measurement]
        if (len(group) != int(expected.n_pairs)
                or not np.isclose(group.did.mean(), expected.effect, rtol=0, atol=1e-9)):
            raise AssertionError(f"{measurement}: pairs do not reproduce the primary estimate")
        logging.info("%s: %d pairs, DiD %.2f (paper %.2f)", measurement, len(group),
                     group.did.mean(), expected.effect)

    matched_initiators = sleep["units"].loc[
        sleep["units"].treated.eq(1)
        & sleep["units"].participant_id.isin(set(sleep["pairs"].treated_id))]
    definitions = subgroup_definitions(matched_initiators)
    estimates = subgroup_estimates(pipe.step26, rows, definitions, np.random.default_rng(SEED))
    estimates.to_csv(OUT / "subgroup_estimates.csv", index=False)

    pd.DataFrame([{
        "matched_initiators": len(matched_initiators),
        "matched_initiators_with_diabetes": int(matched_initiators.diabetic.eq(1.0).sum()),
        "age_cut_years": float(matched_initiators.age.median()),
        "bmi_cut": BMI_CUT,
        "n_bootstrap": pipe.step26.N_BOOT, "seed": SEED,
    }]).to_csv(OUT / "subgroup_summary.csv", index=False)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
