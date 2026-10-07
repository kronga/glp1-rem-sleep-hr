"""Maximise the GLP-1 cohort with raw stage-resolved sleep, then test REM accentuation.

Motivation
----------
`steps/30` used the paper's adjudicated exposure (timing-valid, visit-confirmed,
brand-consistent semaglutide) and matched raw sleep visits to the *exact*
pre_stage/post_stage labels. That yields 28 participants, and widening to a
stage-ORDER window yields the same 28 -- the join rule was not the constraint.
The constraint is the strict exposure adjudication itself.

This step relaxes exposure instead, using medication report dates:

    pre  visit : latest raw sleep visit with study_date <= first GLP report date
    post visit : earliest raw sleep visit with study_date >  first GLP report
                 date + MIN_GAP_DAYS

Yield (raw sleep at both pre and post):

    paper-adjudicated semaglutide, exact stage labels ....  28
    ever-semaglutide, date rule ......................... 107
    ever-any-GLP-1, date rule ........................... 148

Trade-off, stated plainly
-------------------------
The date rule buys 4-5x the sample at the cost of weaker exposure evidence. The
paper's `timing_valid` adjudication requires the drug to be *confirmed absent*
at the pre visit; the date rule only requires the pre sleep visit to precede the
first medication report. A participant who started between visits is misclassified
pre. So the strict arm is the internally valid estimate and the broad arms test
whether the effect survives a larger, noisier cohort. All three are reported.

GLP-1 agents matched (7 strings, 788 ever-users): Ozempic, Wegovy, Rybelsus
(semaglutide); Saxenda, Victoza (liraglutide); Trulicity (dulaglutide); Mounjaro
(tirzepatide, GIP/GLP-1 dual). The reanalysis independently reports 789 ever-GLP
controls excluded, which agrees to one participant.

The drift reference excludes **all 788 ever-GLP participants**, dated or not, which
removes the ~6-7% contamination that `steps/30` had to carry.

Corrected 2026-08: this previously read the exclusion set off `load_glp_reports()`,
which drops undated rows, so it actually excluded 741 rather than 788 and left 47 real
GLP-1 users eligible as never-GLP controls. The docstring's claim that 788 cross-
validated against the reanalysis's 789 was therefore describing a number the code did
not use. See `ever_glp_participants` and `MEDICATION_DATE_AUDIT.md`.

Raw channels, stage codes, and the reading pattern: `assets/datasets/sleep/loader.md` §15b.

Usage
-----
    PY=/net/mraid20/export/jasmine/david/anaconda3/bin/python3
    $PY steps/31_raw_sleep_glp_cohort.py --arm sema_dated --drift 150
    $PY steps/31_raw_sleep_glp_cohort.py --arm all        # every arm, sequentially
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

# LabData's Medications10KLoader calls DataFrame.applymap, removed in this pandas.
if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = (  # type: ignore[attr-defined]
        lambda self, func, na_action=None, **kw: self.map(func, na_action=na_action, **kw)
    )

GOLD = Path("/net/mraid20/export/genie/LabData/Data/Pheno/gold/sleep")
TIMESERIES = GOLD / "timeseries"
LINK_TABLE = GOLD / "sleep_all_hrv_with_participant.parquet"
RUN = Path(__file__).resolve().parent.parent
AUDIT = RUN / "outputs/paper_reanalysis/cache/exposure_record_audit_private.csv"
OUT = RUN / "outputs/raw_sleep_glp_cohort"

GLP_PATTERN = re.compile(
    r"ozempic|rybelsus|wegovy|saxenda|victoza|trulicity|mounjaro"
    r"|אוזמפיק|ריבלסוס|וויגובי|סקסנדה|ויקטוזה|טרוליסיטי|מאונג",
    re.I,
)
SEMAGLUTIDE_PATTERN = re.compile(r"ozempic|rybelsus|wegovy|אוזמפיק|ריבלסוס|וויגובי", re.I)

STAGE_CODES = {11: "wake", 12: "rem", 13: "light", 15: "deep"}
NREM_CODES = (13, 15)
MIN_STATE_SECONDS = 300
HR_BOUNDS = (30.0, 140.0)
MIN_GAP_DAYS = 30
SEED = 20260728
STATES = ["rem", "nrem", "light", "deep", "wake"]
CONTRASTS = [("rem", "nrem"), ("rem", "light"), ("rem", "deep"), ("deep", "light")]


# ----------------------------------------------------------------- raw channels


def _values(path: Path) -> np.ndarray:
    return np.asarray(pq.read_table(path, columns=["values"]).column("values"))


def night_stage_means(night_dir: Path) -> dict | None:
    stage_path = night_dir / "sleep_stage.parquet"
    hr_path = night_dir / "heart_rate.parquet"
    if not stage_path.exists() or not hr_path.exists():
        return None
    try:
        stage = _values(stage_path)
        heart_rate = _values(hr_path)
    except Exception:
        return None
    if len(stage) != len(heart_rate) or not len(stage):
        return None

    usable = np.isfinite(heart_rate)
    usable &= (heart_rate >= HR_BOUNDS[0]) & (heart_rate <= HR_BOUNDS[1])

    out: dict = {}
    for code, label in STAGE_CODES.items():
        mask = (stage == code) & usable
        out[label] = float(heart_rate[mask].mean()) if mask.sum() >= MIN_STATE_SECONDS else np.nan
        out[f"{label}_seconds"] = int(mask.sum())
    pooled = np.isin(stage, NREM_CODES) & usable
    out["nrem"] = float(heart_rate[pooled].mean()) if pooled.sum() >= MIN_STATE_SECONDS else np.nan
    out["nrem_seconds"] = int(pooled.sum())
    return out


def visit_stage_means(uuid: str) -> dict | None:
    visit = TIMESERIES / uuid
    if not visit.is_dir():
        return None
    nights = [night_stage_means(p) for p in sorted(visit.glob("night_*"))]
    nights = [n for n in nights if n is not None]
    if not nights:
        return None
    frame = pd.DataFrame(nights)
    means = {c: frame[c].mean(skipna=True) for c in frame.columns if not c.endswith("_seconds")}
    means["n_nights"] = len(frame)
    for label in list(STAGE_CODES.values()) + ["nrem"]:
        means[f"{label}_seconds"] = int(frame[f"{label}_seconds"].sum())
    return means


# ------------------------------------------------------------------- input data


def load_sleep_visits() -> pd.DataFrame:
    """Raw-available sleep visits with a usable study date."""
    table = pq.read_table(
        LINK_TABLE, columns=["participant_id", "research_stage", "uuid", "study_date"]
    ).to_pandas(ignore_metadata=True)
    available = set(os.listdir(TIMESERIES))
    table = table[table.uuid.isin(available)].copy()
    table["participant"] = table.participant_id.astype("int64").astype(str)
    table["study_date"] = pd.to_datetime(table.study_date, errors="coerce", utc=True)
    return table.dropna(subset=["study_date"]).drop_duplicates("uuid")


def load_glp_reports() -> pd.DataFrame:
    module = importlib.import_module("LabData.DataLoaders.Medications10KLoader")
    data = getattr(module, "Medications10KLoader")().get_data()
    frame = (data.df if hasattr(data, "df") else data).reset_index()
    name = frame.medication.astype(str)
    frame = frame[name.str.contains(GLP_PATTERN)].copy()
    frame["is_semaglutide"] = frame.medication.astype(str).str.contains(SEMAGLUTIDE_PATTERN)
    frame["participant"] = frame.RegistrationCode.astype(str).str.replace(
        "10K_", "", case=False, regex=False
    )
    frame["report_date"] = pd.to_datetime(frame.Date, errors="coerce", utc=True)
    return frame.dropna(subset=["report_date"])


def ever_glp_participants() -> set[str]:
    """Everyone with ANY GLP-1 row, dated or not -- the drift-reference exclusion set.

    `load_glp_reports` drops undated rows because exposure timing needs a date. Reusing
    that frame to exclude ever-GLP participants from the never-GLP drift reference was a
    bug: 113 of 1,123 GLP-1 rows carry no parseable date and 47 participants have no dated
    GLP-1 row at all, so they were eligible as never-GLP controls. Absence of a date is
    not evidence of absence of use. Audit: `MEDICATION_DATE_AUDIT.md`.
    """
    module = importlib.import_module("LabData.DataLoaders.Medications10KLoader")
    data = getattr(module, "Medications10KLoader")().get_data()
    frame = (data.df if hasattr(data, "df") else data).reset_index()
    frame = frame[frame.medication.astype(str).str.contains(GLP_PATTERN)]
    return set(
        frame.RegistrationCode.astype(str)
        .str.replace("10K_", "", case=False, regex=False)
    )


def paper_adjudicated_participants() -> set[str]:
    audit = pd.read_csv(AUDIT)
    strict = audit[
        audit.agent_class.astype(str).str.contains("semaglutide", case=False)
        & audit.timing_valid.astype(bool)
    ]
    return set(
        strict.RegistrationCode.astype(str).str.replace("10k_", "", case=False, regex=False)
    )


# ----------------------------------------------------------------------- design


def build_pairs(visits: pd.DataFrame, first_report: pd.Series, label: str) -> pd.DataFrame:
    """Pick one pre and one post raw sleep visit per participant, around first report."""
    rows = []
    for participant, start in first_report.items():
        own = visits[visits.participant == participant]
        if own.empty:
            continue
        pre = own[own.study_date <= start]
        post = own[own.study_date > start + pd.Timedelta(days=MIN_GAP_DAYS)]
        if pre.empty or post.empty:
            continue
        pre_row = pre.loc[pre.study_date.idxmax()]     # closest before start
        post_row = post.loc[post.study_date.idxmin()]  # first confirmed on-drug
        rows.append(
            {
                "participant": participant,
                "arm": label,
                "pre_uuid": pre_row.uuid,
                "post_uuid": post_row.uuid,
                "pre_date": pre_row.study_date,
                "post_date": post_row.study_date,
                "first_report": start,
                "months_pre_to_start": (start - pre_row.study_date).days / 30.44,
                "months_on_drug_at_post": (post_row.study_date - start).days / 30.44,
            }
        )
    return pd.DataFrame(rows)


SHARE_STATES = ("rem", "light", "deep")


def stage_shares(pre: dict, post: dict) -> dict:
    """Each stage's share of total sleep, pre and post, plus the change.

    Distinct from `pct_deep`, which is deep/(deep+light) -- a *within-NREM* mix that
    says nothing about how much REM the night held. These use total sleep
    (rem+light+deep) as the denominator so the three shares sum to 100, which is what
    "did the night's composition change?" actually asks. Wake is excluded: it is not
    sleep, and including it would make every share move whenever sleep efficiency did.
    """
    out: dict = {}
    for name, visit in (("pre", pre), ("post", post)):
        seconds = {state: visit.get(f"{state}_seconds", 0) for state in SHARE_STATES}
        total = sum(seconds.values())
        nights = max(int(visit.get("n_nights", 1) or 1), 1)
        if total < MIN_STATE_SECONDS:
            # No usable staged sleep at this visit -- every epoch was wake, or the
            # heart-rate channel was out of bounds throughout. Emit NaN.
            #
            # The first version of this helper divided by `max(total, 1)`, which turned
            # "nothing was scorable" into "0% of every stage" and "0.02 min of sleep".
            # Those are real values, so they propagated into the composition means: 12
            # of 421 visits carried them, and the three share changes stopped summing to
            # zero within a participant (they must, since the shares sum to 100). The HR
            # columns were never affected -- they are NaN-guarded by MIN_STATE_SECONDS
            # in night_stage_means -- so this only ever touched the columns added on
            # 2026-09-06.
            for state in SHARE_STATES:
                out[f"share_{state}_{name}"] = np.nan
                out[f"minutes_{state}_{name}"] = np.nan
            out[f"minutes_sleep_{name}"] = np.nan
            continue
        for state in SHARE_STATES:
            out[f"share_{state}_{name}"] = 100 * seconds[state] / total
            # Per-night minutes, not summed across the visit: a visit contributes a
            # variable number of scorable nights, so a raw sum would make "duration"
            # track how many nights were recorded rather than how the night was spent.
            out[f"minutes_{state}_{name}"] = seconds[state] / 60.0 / nights
        out[f"minutes_sleep_{name}"] = total / 60.0 / nights
    for state in (*SHARE_STATES, "sleep"):
        if state in SHARE_STATES:
            out[f"d_share_{state}"] = out[f"share_{state}_post"] - out[f"share_{state}_pre"]
        out[f"d_minutes_{state}"] = (
            out[f"minutes_{state}_post"] - out[f"minutes_{state}_pre"]
        )
    return out


def measure(pairs: pd.DataFrame) -> pd.DataFrame:
    """Stage-resolved pre/post means and contrasts for each pair."""
    records = []
    for row in pairs.itertuples(index=False):
        pre = visit_stage_means(row.pre_uuid)
        post = visit_stage_means(row.post_uuid)
        if pre is None or post is None:
            continue
        rec = {
            "participant": row.participant,
            "arm": row.arm,
            "months_on_drug_at_post": getattr(row, "months_on_drug_at_post", np.nan),
        }
        for state in STATES:
            rec[f"{state}_pre"] = pre.get(state, np.nan)
            rec[f"{state}_post"] = post.get(state, np.nan)
            rec[f"d_{state}"] = rec[f"{state}_post"] - rec[f"{state}_pre"]
        for a, b in CONTRASTS:
            rec[f"d_{a}_minus_{b}"] = rec[f"d_{a}"] - rec[f"d_{b}"]
        deep_pre, light_pre = pre.get("deep_seconds", 0), pre.get("light_seconds", 0)
        deep_post, light_post = post.get("deep_seconds", 0), post.get("light_seconds", 0)
        rec["pct_deep_pre"] = 100 * deep_pre / max(deep_pre + light_pre, 1)
        rec["pct_deep_post"] = 100 * deep_post / max(deep_post + light_post, 1)
        rec["d_pct_deep"] = rec["pct_deep_post"] - rec["pct_deep_pre"]
        rec.update(stage_shares(pre, post))
        records.append(rec)
    return pd.DataFrame(records)


def paired_stats(values: np.ndarray) -> dict:
    from scipy import stats

    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return {"n": n, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    half = stats.t.ppf(0.975, n - 1) * se
    return {
        "n": n,
        "mean": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "p": float(stats.ttest_1samp(values, 0).pvalue),
    }


def summarise(panel: pd.DataFrame, drift: pd.DataFrame, arm: str) -> pd.DataFrame:
    metrics = [f"d_{s}" for s in STATES]
    metrics += [f"d_{a}_minus_{b}" for a, b in CONTRASTS] + ["d_pct_deep"]
    rows = []
    for metric in metrics:
        treated = paired_stats(panel[metric].to_numpy(dtype=float))
        reference = (
            paired_stats(drift[metric].to_numpy(dtype=float))
            if metric in drift.columns
            else {"n": 0, "mean": np.nan}
        )
        rows.append(
            {
                "arm": arm,
                "metric": metric,
                "n": treated["n"],
                "change": treated["mean"],
                "ci_low": treated["ci_low"],
                "ci_high": treated["ci_high"],
                "p": treated["p"],
                "drift_n": reference["n"],
                "drift_change": reference["mean"],
                "drift_adjusted": (treated["mean"] - reference["mean"])
                if np.isfinite(reference.get("mean", np.nan))
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        default="all",
        choices=["all", "paper_strict", "sema_dated", "anyglp_dated"],
    )
    parser.add_argument("--drift", type=int, default=150, help="never-GLP reference participants")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    visits = load_sleep_visits()
    reports = load_glp_reports()
    ever_glp = ever_glp_participants()
    print(f"ever-GLP excluded from the drift reference: {len(ever_glp)} "
          f"({len(ever_glp - set(reports.participant))} with no dated report)", flush=True)
    print(f"raw-available dated sleep visits: {len(visits)}", flush=True)

    first_any = reports.groupby("participant").report_date.min()
    first_sema = reports[reports.is_semaglutide].groupby("participant").report_date.min()
    adjudicated = paper_adjudicated_participants()

    arms = {
        "paper_strict": first_sema[first_sema.index.isin(adjudicated)],
        "sema_dated": first_sema,
        "anyglp_dated": first_any,
    }
    if args.arm != "all":
        arms = {args.arm: arms[args.arm]}

    # Never-GLP secular-drift reference: two raw visits >= MIN_GAP apart.
    rng = np.random.default_rng(SEED)
    never = visits[~visits.participant.isin(ever_glp)]
    counts = never.groupby("participant").size()
    eligible = list(counts[counts >= 2].index)
    if len(eligible) > args.drift:
        eligible = list(rng.choice(eligible, args.drift, replace=False))
    drift_rows = []
    for participant in eligible:
        own = never[never.participant == participant].sort_values("study_date")
        first, last = own.iloc[0], own.iloc[-1]
        if (last.study_date - first.study_date).days < MIN_GAP_DAYS:
            continue
        drift_rows.append(
            {
                "participant": participant,
                "arm": "never_glp",
                "pre_uuid": first.uuid,
                "post_uuid": last.uuid,
            }
        )
    print(f"never-GLP drift candidates: {len(drift_rows)}", flush=True)
    drift = measure(pd.DataFrame(drift_rows))
    print(f"never-GLP drift usable: {len(drift)}", flush=True)

    all_panels, all_summaries = [drift], []
    for arm, first_report in arms.items():
        pairs = build_pairs(visits, first_report, arm)
        print(f"\n[{arm}] participants with raw pre+post: {len(pairs)}", flush=True)
        panel = measure(pairs)
        print(f"[{arm}] usable after QC: {len(panel)}", flush=True)
        if panel.empty:
            continue
        all_panels.append(panel)
        all_summaries.append(summarise(panel, drift, arm))

    panel_out = pd.concat(all_panels, ignore_index=True)
    panel_out.to_csv(OUT / "panel.csv", index=False)
    summary = pd.concat(all_summaries, ignore_index=True)
    summary.to_csv(OUT / "summary.csv", index=False)

    print("\n=== stage-resolved pre->post change, by exposure arm ===")
    for arm in summary.arm.unique():
        sub = summary[summary.arm == arm]
        print(f"\n--- {arm} ---")
        print(
            sub[["metric", "n", "change", "ci_low", "ci_high", "p", "drift_change", "drift_adjusted"]]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )

    # Composition check per arm.
    from scipy import stats as st

    # d_share_* are the three sleep-stage shares of the night; d_pct_deep and
    # pct_deep_pre are the older within-NREM mix, kept so nothing downstream that
    # still reads them breaks.
    predictors = [f"d_share_{state}" for state in SHARE_STATES] + ["d_pct_deep", "pct_deep_pre"]
    checks = []
    for arm in panel_out.arm.unique():
        if arm == "never_glp":
            continue
        sub = panel_out[panel_out.arm == arm][["d_rem_minus_nrem", *predictors]].dropna()
        if len(sub) < 5:
            continue
        for predictor in predictors:
            r, p = st.pearsonr(sub[predictor], sub.d_rem_minus_nrem)
            checks.append({"arm": arm, "predictor": predictor, "r": r, "p": p, "n": len(sub)})
    if checks:
        frame = pd.DataFrame(checks)
        frame.to_csv(OUT / "composition_check.csv", index=False)
        print("\n=== NREM-composition check (does the contrast track light/deep mix?) ===")
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nwrote {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
