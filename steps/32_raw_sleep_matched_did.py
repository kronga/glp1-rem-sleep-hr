"""Matched difference-in-differences for stage-resolved nocturnal heart rate.

Why this exists
---------------
`steps/31` reported stage-resolved pre/post changes with only a never-GLP
secular-drift reference. That is defensible for the REM-minus-NREM *contrast* —
a within-recording difference whose never-GLP value is ~0 (-0.14 bpm, n=148), so
between-person confounders have little room to bias it — but it is too weak for
the per-state absolute changes (REM +5.08, NREM +4.05, wake +3.08). Those are
exactly what confounding by indication and regression to the mean attack, and a
drift reference controls only for secular/device trend.

This step adds the missing design: propensity matching against never-GLP
controls and a proper DiD.

Design
------
* Treated: the three exposure arms from `steps/31` (paper_strict, sema_dated,
  anyglp_dated), each with a raw pre and post sleep visit.
* Controls: never-GLP participants (all 788 ever-users excluded) with two raw
  dated sleep visits >= MIN_GAP_DAYS apart. Pool = 4,942.
* Propensity covariates at the pre visit: age, sex, BMI, weight, waist, hip,
  baseline whole-sleep heart rate, and the pre->post interval in days. Baseline
  heart rate matters for regression to the mean; the interval matters because
  never-GLP visit spacing (median 776 d) differs from treated spacing.
* Match: exact on sex, 1:1 nearest-neighbour on the PS logit **without
  replacement**, hard caliper 0.2 x SD(logit), **no fallback** — the same rules
  the paper's reanalysis adopted.
* Estimand: DiD = treated (post-pre) - matched control (post-pre), per state and
  per contrast, with a paired bootstrap CI over matched pairs.
* Gates, as in the reanalysis: retention >= 85% of eligible treated AND max
  post-match |SMD| < 0.10. Reported per arm; a failing arm is labelled, not
  silently dropped.

Efficiency note: the propensity model uses **summary** heart rate from the gold
link table, so no raw channels are read for the 4,942-participant pool. Raw
channels are read only for treated participants and their matched controls, and
cached per uuid across arms.

Usage
-----
    PY=/net/mraid20/export/jasmine/david/anaconda3/bin/python3
    $PY steps/32_raw_sleep_matched_did.py --arm all --boot 2000
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = (  # type: ignore[attr-defined]
        lambda self, func, na_action=None, **kw: self.map(func, na_action=na_action, **kw)
    )

GOLD = Path("/net/mraid20/export/genie/LabData/Data/Pheno/gold/sleep")
TIMESERIES = GOLD / "timeseries"
LINK_TABLE = GOLD / "sleep_all_hrv_with_participant.parquet"
SNAPSHOT = Path(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/snapshots/pheno_data_snapshots/out_dir"
)
RUN = Path(__file__).resolve().parent.parent
AUDIT = RUN / "outputs/paper_reanalysis/cache/exposure_record_audit_private.csv"
OUT = RUN / "outputs/raw_sleep_matched_did"

GLP_PATTERN = re.compile(
    r"ozempic|rybelsus|wegovy|saxenda|victoza|trulicity|mounjaro"
    r"|אוזמפיק|ריבלסוס|וויגובי|סקסנדה|ויקטוזה|טרוליסיטי|מאונג",
    re.I,
)
SEMAGLUTIDE_PATTERN = re.compile(r"ozempic|rybelsus|wegovy|אוזמפיק|ריבלסוס|וויגובי", re.I)

STAGE_CODES = {11: "wake", 12: "rem", 13: "light", 15: "deep"}
NREM_CODES = (13, 15)
STATES = ["rem", "nrem", "light", "deep", "wake"]
CONTRASTS = [("rem", "nrem"), ("rem", "light"), ("rem", "deep"), ("deep", "light")]

MIN_STATE_SECONDS = 300
HR_BOUNDS = (30.0, 140.0)
MIN_GAP_DAYS = 30
CALIPER_SD = 0.2
ANTHRO_WINDOW_DAYS = 180
RETENTION_GATE = 0.85
SMD_GATE = 0.10
SEED = 20260728

PS_COVARIATES = ["age", "bmi", "weight", "waist_circumference", "hip_circumference",
                 "baseline_sleep_hr", "interval_days",
                 "diabetic", "smoking", "vat_area", "vat_missing"]
# Exact-blocked rather than modelled. The reanalysis blocked on sex + diabetes; the
# binary confounders added here (smoking, DXA-missingness) are also cheap to block
# exactly given a ~36:1 control:treated ratio, and blocking them is what finally
# balanced them (see the matcher docstring).
EXACT_BLOCK = ["gender", "diabetic", "smoking", "vat_missing"]
# Hard per-covariate calipers, applied on top of the propensity caliper. These are
# the covariates that a PS-only match left imbalanced.
HARD_CALIPERS = {"age": 5.0, "vat_area": 40.0, "interval_days": 200.0}

_RAW_CACHE: dict[str, dict | None] = {}


# ----------------------------------------------------------------- raw channels


def _values(path: Path) -> np.ndarray:
    return np.asarray(pq.read_table(path, columns=["values"]).column("values"))


def night_stage_means(night_dir: Path) -> dict | None:
    stage_path, hr_path = night_dir / "sleep_stage.parquet", night_dir / "heart_rate.parquet"
    if not stage_path.exists() or not hr_path.exists():
        return None
    try:
        stage, heart_rate = _values(stage_path), _values(hr_path)
    except Exception:
        return None
    if len(stage) != len(heart_rate) or not len(stage):
        return None
    usable = np.isfinite(heart_rate) & (heart_rate >= HR_BOUNDS[0]) & (heart_rate <= HR_BOUNDS[1])
    out: dict = {}
    for code, label in STAGE_CODES.items():
        mask = (stage == code) & usable
        out[label] = float(heart_rate[mask].mean()) if mask.sum() >= MIN_STATE_SECONDS else np.nan
    pooled = np.isin(stage, NREM_CODES) & usable
    out["nrem"] = float(heart_rate[pooled].mean()) if pooled.sum() >= MIN_STATE_SECONDS else np.nan
    return out


def visit_stage_means(uuid: str) -> dict | None:
    if uuid in _RAW_CACHE:
        return _RAW_CACHE[uuid]
    visit = TIMESERIES / uuid
    result = None
    if visit.is_dir():
        nights = [night_stage_means(p) for p in sorted(visit.glob("night_*"))]
        nights = [n for n in nights if n is not None]
        if nights:
            frame = pd.DataFrame(nights)
            result = {c: frame[c].mean(skipna=True) for c in frame.columns}
    _RAW_CACHE[uuid] = result
    return result


# ------------------------------------------------------------------- input data


def load_sleep_visits() -> pd.DataFrame:
    table = pq.read_table(
        LINK_TABLE,
        columns=["participant_id", "research_stage", "uuid", "study_date",
                 "heart_rate_mean_during_sleep", "quality_score_heart_rate"],
    ).to_pandas(ignore_metadata=True)
    table = table[table.uuid.isin(set(os.listdir(TIMESERIES)))].copy()
    table["participant"] = table.participant_id.astype("int64").astype(str)
    table["study_date"] = pd.to_datetime(table.study_date, errors="coerce", utc=True)
    table["summary_sleep_hr"] = pd.to_numeric(
        table.heart_rate_mean_during_sleep, errors="coerce"
    )
    quality = pd.to_numeric(table.quality_score_heart_rate, errors="coerce")
    table = table[(quality >= 50) | quality.isna()]
    return table.dropna(subset=["study_date"]).drop_duplicates("uuid")


def load_glp_reports() -> pd.DataFrame:
    module = importlib.import_module("LabData.DataLoaders.Medications10KLoader")
    data = getattr(module, "Medications10KLoader")().get_data()
    frame = (data.df if hasattr(data, "df") else data).reset_index()
    frame = frame[frame.medication.astype(str).str.contains(GLP_PATTERN)].copy()
    frame["is_semaglutide"] = frame.medication.astype(str).str.contains(SEMAGLUTIDE_PATTERN)
    frame["participant"] = frame.RegistrationCode.astype(str).str.replace(
        "10K_", "", case=False, regex=False
    )
    frame["report_date"] = pd.to_datetime(frame.Date, errors="coerce", utc=True)
    return frame.dropna(subset=["report_date"])


def ever_glp_participants() -> set[str]:
    """Everyone with ANY GLP-1 row, dated or not -- the control-exclusion set.

    `load_glp_reports` drops undated rows because exposure timing needs a date. Using
    that same frame to build the never-GLP control pool was a bug: 113 of 1,123 GLP-1
    rows carry no parseable date, and 47 participants have no dated GLP-1 row at all, so
    they were eligible as "never-GLP-1" controls. 15 of them had enough raw sleep visits
    to enter the pool and 7 reached the retained control arm, carrying ~1% of control
    weight -- and 6 of those 7 showed rising pre->post REM heart rate, the treated-like
    pattern, biasing the DiD toward zero.

    Absence of a date is not evidence of absence of use, and for the control arm the
    conservative choice is to exclude. Audit and magnitude: `MEDICATION_DATE_AUDIT.md`
    (the fix shifts the per-state DiD by +0.02 to +0.04 bpm).
    """
    module = importlib.import_module("LabData.DataLoaders.Medications10KLoader")
    data = getattr(module, "Medications10KLoader")().get_data()
    frame = (data.df if hasattr(data, "df") else data).reset_index()
    frame = frame[frame.medication.astype(str).str.contains(GLP_PATTERN)]
    return set(
        frame.RegistrationCode.astype(str)
        .str.replace("10K_", "", case=False, regex=False)
    )


def load_covariates() -> tuple[pd.DataFrame, pd.DataFrame]:
    anthro = pd.read_parquet(
        SNAPSHOT / "anthropometrics/anthropometrics/df.parquet",
        columns=["bmi", "weight", "waist_circumference", "hip_circumference"],
    ).reset_index()
    anthro["participant"] = anthro.RegistrationCode.astype(str).str.replace(
        "10k_", "", case=False, regex=False
    )
    anthro["date"] = pd.to_datetime(anthro.date, errors="coerce", utc=True)
    anthro = anthro.dropna(subset=["date"])

    people = pd.read_parquet(
        SNAPSHOT / "participants/participants/df.parquet", columns=["gender", "year_of_birth"]
    ).reset_index()
    people["participant"] = people.RegistrationCode.astype(str).str.replace(
        "10k_", "", case=False, regex=False
    )
    return anthro, people


def attach_covariates(
    pairs: pd.DataFrame, anthro: pd.DataFrame, people: pd.DataFrame
) -> pd.DataFrame:
    """Nearest anthropometry within +/-ANTHRO_WINDOW_DAYS of the pre visit."""
    # merge_asof requires identical datetime resolution; gold is us, snapshot is ns.
    left = pairs.sort_values("pre_date")[["participant", "pre_date"]].copy()
    left["pre_date"] = left.pre_date.dt.as_unit("ns")
    right = anthro.sort_values("date").copy()
    right["date"] = right.date.dt.as_unit("ns")
    merged = pd.merge_asof(
        left, right, left_on="pre_date", right_on="date", by="participant",
        direction="nearest", tolerance=pd.Timedelta(days=ANTHRO_WINDOW_DAYS),
    )
    out = pairs.merge(
        merged[["participant", "bmi", "weight", "waist_circumference", "hip_circumference"]],
        on="participant", how="left",
    )
    out = out.merge(people[["participant", "gender", "year_of_birth"]], on="participant", how="left")
    out["age"] = out.pre_date.dt.year - pd.to_numeric(out.year_of_birth, errors="coerce")
    out["interval_days"] = (out.post_date - out.pre_date).dt.days
    return attach_confounders(out)


def attach_confounders(pairs: pd.DataFrame) -> pd.DataFrame:
    """Diabetes (as of the pre visit), current smoking, and baseline DXA visceral fat.

    Reuses `src/glp_confounders`, so definitions match `steps/26` exactly:
    diabetes = self-reported `Consolidated name == 'Diabetes'` in the 10K
    `for_review` condition tables at or before the pre visit's research stage
    (prediabetes excluded); smoking = `smoke_tobacco_now > 0` at any visit;
    vat_area = baseline DXA `total_scan_vat_area`.
    """
    sys.path.insert(0, str(RUN / "src"))
    import glp_confounders as confounders

    out = pairs.copy()
    out["RegistrationCode"] = "10K_" + out.participant.astype(str)

    # Diabetes is stage-timed, so use the pre visit's own research stage.
    keys = out[["RegistrationCode", "pre_research_stage"]].rename(
        columns={"pre_research_stage": "research_stage"}
    )
    diabetes = confounders.diabetes_flags(keys, timing="pre_stage")
    diabetes = diabetes.drop_duplicates(["RegistrationCode", "research_stage"])
    out = out.merge(
        diabetes.rename(columns={"research_stage": "pre_research_stage"}),
        on=["RegistrationCode", "pre_research_stage"],
        how="left",
    )

    out = out.merge(confounders.smoking_status(), on="RegistrationCode", how="left")
    out = out.merge(confounders.baseline_vat_area(), on="RegistrationCode", how="left")

    for column in ["diabetic", "smoking"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    # VAT is missing for participants without a baseline DXA. Impute within sex
    # (symmetric across arms) and carry an explicit missingness indicator, so
    # missing-DXA participants are matched to each other rather than dropped.
    out["vat_missing"] = out.vat_area.isna().astype(float)
    out["vat_area"] = out.groupby("gender").vat_area.transform(
        lambda s: s.fillna(s.median())
    )
    out["vat_area"] = out.vat_area.fillna(out.vat_area.median())
    return out


# ----------------------------------------------------------------------- design


def build_treated(visits: pd.DataFrame, first_report: pd.Series, arm: str) -> pd.DataFrame:
    rows = []
    for participant, start in first_report.items():
        own = visits[visits.participant == participant]
        if own.empty:
            continue
        pre = own[own.study_date <= start]
        post = own[own.study_date > start + pd.Timedelta(days=MIN_GAP_DAYS)]
        if pre.empty or post.empty:
            continue
        pre_row, post_row = pre.loc[pre.study_date.idxmax()], post.loc[post.study_date.idxmin()]
        rows.append({
            "participant": participant, "arm": arm, "treated": 1,
            "pre_uuid": pre_row.uuid, "post_uuid": post_row.uuid,
            "pre_date": pre_row.study_date, "post_date": post_row.study_date,
            "baseline_sleep_hr": pre_row.summary_sleep_hr,
            "pre_research_stage": pre_row.research_stage,
        })
    return pd.DataFrame(rows)


def build_controls(visits: pd.DataFrame, ever_glp: set[str]) -> pd.DataFrame:
    never = visits[~visits.participant.isin(ever_glp)].sort_values("study_date")
    rows = []
    for participant, own in never.groupby("participant"):
        if len(own) < 2:
            continue
        first, last = own.iloc[0], own.iloc[-1]
        if (last.study_date - first.study_date).days < MIN_GAP_DAYS:
            continue
        rows.append({
            "participant": participant, "arm": "never_glp", "treated": 0,
            "pre_uuid": first.uuid, "post_uuid": last.uuid,
            "pre_date": first.study_date, "post_date": last.study_date,
            "baseline_sleep_hr": first.summary_sleep_hr,
            "pre_research_stage": first.research_stage,
        })
    return pd.DataFrame(rows)


def standardised_differences(frame: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    treated, control = frame[frame.treated == 1], frame[frame.treated == 0]
    rows = []
    for covariate in covariates:
        a, b = treated[covariate].astype(float), control[covariate].astype(float)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = 0.0 if not pooled or np.isnan(pooled) else (a.mean() - b.mean()) / pooled
        rows.append({"covariate": covariate, "treated_mean": a.mean(),
                     "control_mean": b.mean(), "smd": smd})
    return pd.DataFrame(rows)


def match_one_to_one(frame: pd.DataFrame, covariates: list[str], rng) -> pd.DataFrame:
    """Exact blocks + hard per-covariate calipers + nearest neighbour on the PS logit.

    Two earlier designs failed, and both failures are recorded here because they
    shaped this one:

      1. **PS-only nearest neighbour** left age off by 0.41 SD (strict arm),
         smoking by 0.21 and VAT by 0.20. A scalar propensity score balances
         covariates only in expectation; with 11 covariates it maps distinct
         profiles onto the same logit.
      2. **Mahalanobis on the standardised covariates** was worse still
         (max |SMD| 0.85 / 0.45 / 0.58). Weight, BMI, waist and hip are strongly
         collinear, so the covariance is ill-conditioned and its inverse inflates
         the near-null directions — the distance then chases noise in collinear
         combinations while ignoring age.

    What works, given a ~36:1 control:treated ratio: block the binaries exactly,
    impose hard calipers on the continuous covariates that actually drifted, and
    use the PS only to pick among the survivors.

      * exact on sex, diabetes, current smoking, and DXA-missingness;
      * hard calipers: age +/-5 y, VAT +/-40 cm2, interval +/-200 d;
      * PS caliper 0.2 x SD(logit), no fallback;
      * nearest PS neighbour among controls satisfying all of the above;
      * treated processed hardest-first (fewest eligible controls) so scarce
        profiles claim partners before common ones exhaust the pool.
    """
    from sklearn.linear_model import LogisticRegression

    work = frame.dropna(subset=covariates + EXACT_BLOCK).copy()
    design = work[covariates].astype(float)
    standardised = (design - design.mean()) / design.std(ddof=0).replace(0, 1)
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(standardised.values, work.treated.values)
    probability = model.predict_proba(standardised.values)[:, 1].clip(1e-6, 1 - 1e-6)
    work["ps_logit"] = np.log(probability / (1 - probability))
    caliper = CALIPER_SD * work.ps_logit.std(ddof=1)

    matched_rows = []
    for _, block in work.groupby(EXACT_BLOCK):
        treated = block[block.treated == 1]
        pool = block[block.treated == 0]
        if treated.empty or pool.empty:
            continue
        pool_logits = pool.ps_logit.to_numpy()
        pool_hard = {name: pool[name].to_numpy(dtype=float) for name in HARD_CALIPERS}

        def eligible_mask(row, available: np.ndarray) -> np.ndarray:
            keep = np.abs(pool_logits[available] - row.ps_logit) <= caliper
            for name, width in HARD_CALIPERS.items():
                keep &= np.abs(pool_hard[name][available] - float(getattr(row, name))) <= width
            return available[keep]

        counts = {
            row.participant: len(eligible_mask(row, np.arange(len(pool))))
            for row in treated.itertuples(index=False)
        }
        order = sorted(treated.itertuples(index=False), key=lambda r: counts[r.participant])

        used: set[int] = set()
        for row in order:
            available = np.array([i for i in range(len(pool)) if i not in used])
            if not len(available):
                continue
            within = eligible_mask(row, available)
            if not len(within):
                continue  # no fallback, by design
            distances = np.abs(pool_logits[within] - row.ps_logit)
            choice = int(within[int(np.argmin(distances))])
            used.add(choice)
            partner = pool.iloc[choice]
            matched_rows.append({**row._asdict(), "pair_id": len(matched_rows)})
            matched_rows.append({**partner.to_dict(), "pair_id": len(matched_rows) - 1})
    return pd.DataFrame(matched_rows)


def weighted_moments(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total = weights.sum()
    mean = float((weights * values).sum() / total)
    variance = float((weights * (values - mean) ** 2).sum() / total)
    return mean, variance


def overlap_weights(frame: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Propensity-score overlap weights: w = 1-e for treated, e for controls.

    Overlap weights exactly balance every covariate entered linearly in the
    propensity model — a mathematical property, not an empirical hope. That is why
    the paper's own reanalysis fell back to overlap weighting for the arms whose
    1:1 matches failed their balance gate (HUMAnN, cross-sectional CGM), reaching
    max |SMD| 0.012-0.029 where 1:1 sat at 0.17-0.20. The same reasoning applies
    here: 11 covariates cannot be balanced by 1:1 no-replacement matching at
    n<=139 treated (three matcher variants failed; see `match_one_to_one`), so
    overlap weighting is the claim-eligible estimator and 1:1 is the sensitivity.
    """
    from sklearn.linear_model import LogisticRegression

    work = frame.dropna(subset=covariates).copy()
    design = work[covariates].astype(float)
    standardised = (design - design.mean()) / design.std(ddof=0).replace(0, 1)
    model = LogisticRegression(max_iter=5000, C=1.0)
    model.fit(standardised.values, work.treated.values)
    propensity = model.predict_proba(standardised.values)[:, 1].clip(1e-6, 1 - 1e-6)
    work["propensity"] = propensity
    work["overlap_weight"] = np.where(work.treated == 1, 1 - propensity, propensity)
    return work


def weighted_balance(frame: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    rows = []
    treated = frame[frame.treated == 1]
    control = frame[frame.treated == 0]
    for covariate in covariates:
        m1, v1 = weighted_moments(
            treated[covariate].to_numpy(dtype=float), treated.overlap_weight.to_numpy()
        )
        m0, v0 = weighted_moments(
            control[covariate].to_numpy(dtype=float), control.overlap_weight.to_numpy()
        )
        pooled = np.sqrt((v1 + v0) / 2)
        rows.append({
            "covariate": covariate, "treated_mean": m1, "control_mean": m0,
            "smd": 0.0 if not pooled or np.isnan(pooled) else (m1 - m0) / pooled,
        })
    return pd.DataFrame(rows)


def overlap_did(frame: pd.DataFrame, metric: str, boot: int, rng) -> dict:
    """Overlap-weighted DiD with a fixed-propensity participant bootstrap."""
    usable = frame.dropna(subset=[metric])
    treated = usable[usable.treated == 1]
    control = usable[usable.treated == 0]
    if len(treated) < 3 or len(control) < 3:
        return {"n_treated": len(treated), "n_control": len(control),
                "did": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    def estimate(t: pd.DataFrame, c: pd.DataFrame) -> float:
        a, _ = weighted_moments(t[metric].to_numpy(dtype=float), t.overlap_weight.to_numpy())
        b, _ = weighted_moments(c[metric].to_numpy(dtype=float), c.overlap_weight.to_numpy())
        return a - b

    point = estimate(treated, control)
    draws = np.empty(boot)
    t_index = np.arange(len(treated))
    c_index = np.arange(len(control))
    for i in range(boot):
        draws[i] = estimate(
            treated.iloc[rng.choice(t_index, len(t_index), replace=True)],
            control.iloc[rng.choice(c_index, len(c_index), replace=True)],
        )
    return {
        "n_treated": len(treated), "n_control": len(control), "did": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def measure_states(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in frame.itertuples(index=False):
        pre, post = visit_stage_means(row.pre_uuid), visit_stage_means(row.post_uuid)
        if pre is None or post is None:
            continue
        record = {"participant": row.participant, "treated": row.treated,
                  "pair_id": getattr(row, "pair_id", np.nan)}
        for state in STATES:
            record[f"d_{state}"] = post.get(state, np.nan) - pre.get(state, np.nan)
        for a, b in CONTRASTS:
            record[f"d_{a}_minus_{b}"] = record[f"d_{a}"] - record[f"d_{b}"]
        records.append(record)
    return pd.DataFrame(records)


def did_with_bootstrap(panel: pd.DataFrame, metric: str, boot: int, rng) -> dict:
    wide = panel.pivot_table(index="pair_id", columns="treated", values=metric)
    wide = wide.dropna()
    if wide.empty or wide.shape[1] < 2:
        return {"n_pairs": 0, "did": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    differences = (wide[1] - wide[0]).to_numpy(dtype=float)
    n = len(differences)
    draws = differences[rng.integers(0, n, size=(boot, n))].mean(axis=1)
    return {"n_pairs": n, "did": float(differences.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="all",
                        choices=["all", "paper_strict", "sema_dated", "anyglp_dated"])
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--control-cap", type=int, default=700,
                        help="max weight-ranked controls to read raw channels for")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    visits = load_sleep_visits()
    reports = load_glp_reports()
    # Exposure timing needs dates; the control-exclusion set must not. See
    # `ever_glp_participants`.
    ever_glp = ever_glp_participants()
    anthro, people = load_covariates()
    print(f"raw dated sleep visits: {len(visits)} | ever-GLP excluded from controls: "
          f"{len(ever_glp)} (of which {len(ever_glp - set(reports.participant))} have no "
          f"dated report)", flush=True)

    controls = attach_covariates(build_controls(visits, ever_glp), anthro, people)
    print(f"never-GLP control pool: {len(controls)}", flush=True)

    first_any = reports.groupby("participant").report_date.min()
    first_sema = reports[reports.is_semaglutide].groupby("participant").report_date.min()
    audit = pd.read_csv(AUDIT)
    adjudicated = set(
        audit[audit.agent_class.astype(str).str.contains("semaglutide", case=False)
              & audit.timing_valid.astype(bool)]
        .RegistrationCode.astype(str).str.replace("10k_", "", case=False, regex=False)
    )
    arms = {
        "paper_strict": first_sema[first_sema.index.isin(adjudicated)],
        "sema_dated": first_sema,
        "anyglp_dated": first_any,
    }
    if args.arm != "all":
        arms = {args.arm: arms[args.arm]}

    all_estimates, all_balance, all_diag = [], [], []
    for arm, first_report in arms.items():
        treated = attach_covariates(build_treated(visits, first_report, arm), anthro, people)
        eligible = len(treated)
        combined = pd.concat([treated, controls], ignore_index=True)

        # ---- primary estimator: overlap weighting -------------------------------
        # Controls whose propensity is ~0 receive ~0 weight and contribute nothing,
        # so reading raw channels for all 4,957 would be wasted I/O. Fit the PS on
        # the full pool, then retain the highest-weight controls (capped) and
        # re-fit on the retained set so the weights are internally consistent.
        weighted_full = overlap_weights(combined, PS_COVARIATES)
        control_rank = (
            weighted_full[weighted_full.treated == 0]
            .sort_values("overlap_weight", ascending=False)
            .head(args.control_cap)
        )
        retained = pd.concat(
            [weighted_full[weighted_full.treated == 1], control_rank], ignore_index=True
        )
        retained = overlap_weights(retained, PS_COVARIATES)
        balance_w = weighted_balance(retained, PS_COVARIATES)
        balance_w.insert(0, "arm", arm)
        balance_w.insert(1, "estimator", "overlap")
        all_balance.append(balance_w)
        max_smd_w = balance_w.smd.abs().max()

        panel_w = measure_states(retained).merge(
            retained[["participant", "overlap_weight"]], on="participant", how="left"
        )
        n_treated_w = int((panel_w.treated == 1).sum())
        print(f"[{arm}] overlap: treated {n_treated_w}/{eligible} "
              f"controls {int((panel_w.treated == 0).sum())} | max|SMD| {max_smd_w:.3f}",
              flush=True)
        metrics = [f"d_{s}" for s in STATES] + [f"d_{a}_minus_{b}" for a, b in CONTRASTS]
        for metric in metrics:
            all_estimates.append({
                "arm": arm, "estimator": "overlap", "metric": metric,
                **overlap_did(panel_w, metric, args.boot, rng),
            })
        all_diag.append({
            "arm": arm, "estimator": "overlap", "eligible_treated": eligible,
            "analysed_treated": n_treated_w,
            "retention": n_treated_w / eligible if eligible else np.nan,
            "max_abs_smd": max_smd_w,
            "smd_gate_pass": bool(max_smd_w < SMD_GATE),
        })

        # ---- sensitivity: 1:1 matching -----------------------------------------
        matched = match_one_to_one(combined, PS_COVARIATES, rng)
        if matched.empty:
            print(f"[{arm}] no matches", flush=True)
            continue
        n_treated_matched = int((matched.treated == 1).sum())
        retention = n_treated_matched / eligible if eligible else np.nan

        balance = standardised_differences(matched, PS_COVARIATES)
        balance.insert(0, "arm", arm)
        balance.insert(1, "estimator", "matched_1to1")
        all_balance.append(balance)
        max_smd = balance.smd.abs().max()

        panel = measure_states(matched)
        print(f"[{arm}] 1:1     : matched {n_treated_matched}/{eligible} "
              f"({100*retention:.1f}%) | max|SMD| {max_smd:.3f}", flush=True)

        for metric in metrics:
            estimate = did_with_bootstrap(panel, metric, args.boot, rng)
            all_estimates.append({"arm": arm, "estimator": "matched_1to1",
                                  "metric": metric, **estimate})

        all_diag.append({
            "arm": arm, "estimator": "matched_1to1", "eligible_treated": eligible,
            "analysed_treated": n_treated_matched,
            "retention": retention, "max_abs_smd": max_smd,
            "retention_gate_pass": bool(retention >= RETENTION_GATE),
            "smd_gate_pass": bool(max_smd < SMD_GATE),
            "design_gate_pass": bool(retention >= RETENTION_GATE and max_smd < SMD_GATE),
            "control_reuse_max": 1,
        })

    estimates = pd.DataFrame(all_estimates)
    estimates.to_csv(OUT / "matched_did_estimates.csv", index=False)
    pd.concat(all_balance, ignore_index=True).to_csv(OUT / "covariate_balance.csv", index=False)
    diagnostics = pd.DataFrame(all_diag)
    diagnostics.to_csv(OUT / "matching_diagnostics.csv", index=False)

    print("\n=== design gates ===")
    print(diagnostics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== DiD estimates (bootstrap 95% CI) ===")
    for estimator in ["overlap", "matched_1to1"]:
        for arm in estimates.arm.unique():
            sub = estimates[(estimates.arm == arm) & (estimates.estimator == estimator)]
            if sub.empty:
                continue
            print(f"\n--- {arm} / {estimator} ---")
            cols = [c for c in ["metric", "n_treated", "n_control", "n_pairs",
                                "did", "ci_low", "ci_high"] if c in sub.columns]
            print(sub[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
