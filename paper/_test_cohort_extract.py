"""TEST ONLY -- cache cohort-level stage means and REM-locked curves for panels b/c.

Rebuilds the overlap-weighted matched cohort of `steps/32_raw_sleep_matched_did.py`
by importing that step's own design functions, so the cohort here is the same
cohort the paper's raw-channel DiD is estimated on -- not a re-implementation.

Why overlap weighting and not 1:1
---------------------------------
On the raw-channel cohort, 1:1 no-replacement matching fails both design gates
(`outputs/raw_sleep_matched_did/matching_diagnostics.csv`): retention 48.6% against
a >=85% gate, max |SMD| 0.42 against a <0.10 gate. Overlap weighting passes at
97.1% retention and max |SMD| 0.020, and step 32 already designates it the
claim-eligible estimator with 1:1 as the sensitivity. So "matched" means
overlap-weighted here, and every mean this script writes carries its weight.

Two things step 32 does not persist, both needed for the panels
--------------------------------------------------------------
* `measure_states` keeps only pre->post *deltas*; a dumbbell needs the pre and post
  *levels*, so `state_levels` below records both.
* REM-locked per-second curves are not computed there at all.

Multiple nights per visit
-------------------------
Every WatchPAT visit carries exactly 3 nights (verified on the gold link table: 18,542
visits x 3 = 55,626 rows, no exceptions), and each night is a separate raw
`night_*/` directory. Both outputs collapse them, but at different points, and the
order matters:

* `state_levels` -> per-night mean within each state, then average the nights.
  Mirrors step 32's `visit_stage_means` exactly, including the 300 s floor per state
  per night: a night contributes to a state only if that state holds >= 300 usable
  seconds *in that night*. Pooling all three nights' seconds first would let one long
  night dominate a visit and would not reproduce step 32's published DiD.
* `rem_locked_curves` -> every REM bout >= 3 min from all three nights goes into one
  pool, averaged to ONE curve per visit, so each participant contributes exactly one
  pre and one post curve regardless of how many nights or bouts they have. The figure
  then weights participants, not bouts or nights. Without this, a participant with 12
  scored bouts would count 4x one with 3.

A participant is dropped from the REM-locked output unless BOTH visits yield at least
one qualifying bout -- otherwise the pre and post means would rest on different people.

Outputs, per arm, into `outputs/_test_figures/`:
    cohort_states_<arm>.csv    participant, treated, overlap_weight,
                               <state>_pre / <state>_post for rem/nrem/light/deep/wake
    cohort_remlock_<arm>.csv   participant, overlap_weight, visit, offset_s, hr
                               (treated only, one curve per participant per visit,
                                bout-averaged then binned to BIN_SECONDS)

Verification
------------
Prints the weighted DiD per state recomputed from `cohort_states_<arm>.csv` beside
the published value in `outputs/raw_sleep_matched_did/matched_did_estimates.csv`.
Those must agree to rounding; if they do not, this script has drifted from step 32
and the panels are not showing the matched cohort.

Usage
-----
    PY=/net/mraid20/export/jasmine/david/anaconda3/bin/python3
    $PY paper/_test_cohort_extract.py                    # paper_strict + sema_dated
    $PY paper/_test_cohort_extract.py --arm paper_strict
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "outputs" / "_test_figures"
PUBLISHED = RUN / "outputs" / "raw_sleep_matched_did" / "matched_did_estimates.csv"

REM_CODE = 12
NREM_CODES = (13, 15)
STATE_CODES = {"rem": (12,), "nrem": NREM_CODES, "light": (13,), "deep": (15,), "wake": (11,)}
MIN_STATE_SECONDS = 300

# REM-locked window and bout floor, matching _test_panela_largest_riser.py.
LOCK_PRE, LOCK_POST = -300, 600
MIN_BOUT_SECONDS = 180
BIN_SECONDS = 5

# Step 32's own default. It is an I/O budget, not a statistical choice: the never-GLP
# pool is 4,957, overlap weighting uses every control with a weight, and controls with
# propensity ~0 get weight ~0, so raw channels are read only for the top-weighted ones.
#
# Measured, not assumed (`--control-cap 2000 --suffix _cap2000`): raising the cap to
# 2000 keeps 90.4% of total control weight and moves the per-state DiD by +0.14 to
# +0.34 bpm (d_rem +4.218 -> +4.361), shrinking the REM-minus-NREM contrast from +1.10
# to +0.93. So the cap is NOT free for the per-state DiD -- small against a ~3.8 bpm CI
# width, but real. It is free for the REM-locked curves, which are weighted over
# treated participants only: max shift 0.054 bpm.
CONTROL_CAP = 700
# Seed for the stage-composition control subsample; step 32's own seed.
SEED = 20260728


def load_step32():
    spec = importlib.util.spec_from_file_location(
        "step32", RUN / "steps" / "32_raw_sleep_matched_did.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- raw channels


def night_channels(step32, night_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Stage codes and range-filtered heart rate per second for one night."""
    stage_path, hr_path = night_dir / "sleep_stage.parquet", night_dir / "heart_rate.parquet"
    if not stage_path.exists() or not hr_path.exists():
        return None
    try:
        stage, heart_rate = step32._values(stage_path), step32._values(hr_path)
    except Exception:
        return None
    if len(stage) != len(heart_rate) or not len(stage):
        return None
    heart_rate = heart_rate.astype(float)
    bounds = step32.HR_BOUNDS
    heart_rate[~((heart_rate >= bounds[0]) & (heart_rate <= bounds[1]))] = np.nan
    return stage, heart_rate


def visit_nights(step32, uuid: str) -> list[tuple[np.ndarray, np.ndarray]]:
    visit = step32.TIMESERIES / uuid
    if not visit.is_dir():
        return []
    nights = [night_channels(step32, p) for p in sorted(visit.glob("night_*"))]
    return [n for n in nights if n is not None]


# ------------------------------------------------------------- per-state levels


def visit_state_levels(step32, uuid: str) -> dict | None:
    """Mean heart rate per state for one visit, averaged over its nights.

    Mirrors step 32's `visit_stage_means` -- same MIN_STATE_SECONDS floor, same
    per-night-then-average order -- but keeps the levels instead of differencing.
    """
    nights = visit_nights(step32, uuid)
    if not nights:
        return None
    rows = []
    for stage, heart_rate in nights:
        usable = np.isfinite(heart_rate)
        row = {}
        for state, codes in STATE_CODES.items():
            mask = np.isin(stage, codes) & usable
            row[state] = (
                float(heart_rate[mask].mean()) if mask.sum() >= MIN_STATE_SECONDS else np.nan
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    return {state: frame[state].mean(skipna=True) for state in STATE_CODES}


def state_levels(step32, cohort: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in cohort.itertuples(index=False):
        pre = visit_state_levels(step32, row.pre_uuid)
        post = visit_state_levels(step32, row.post_uuid)
        if pre is None or post is None:
            continue
        record = {"participant": row.participant, "treated": row.treated,
                  "overlap_weight": row.overlap_weight}
        for state in STATE_CODES:
            record[f"{state}_pre"] = pre[state]
            record[f"{state}_post"] = post[state]
        records.append(record)
    return pd.DataFrame(records)


# ------------------------------------------------------------------ REM-locked


def bout_bounds(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(mask.astype(int))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask.size and mask[0]:
        starts.insert(0, 0)
    if mask.size and mask[-1]:
        ends.append(mask.size)
    return list(zip(starts, ends))


# Stage codes, from `build_opener_figure.STAGE_CODES`. Duplicated as a literal rather
# than imported so this extractor stays independent of the figure modules.
STAGE_NAMES = {11: "wake", 12: "rem", 13: "light", 15: "deep"}


def visit_rem_locked(step32, uuid: str,
                     with_stages: bool = False):
    """One bout-averaged, BIN_SECONDS-binned REM-locked curve for a visit.

    With `with_stages`, also returns the stage composition at each offset: the
    fraction of contributing bout-seconds scored as each stage. That is what says
    whether the pre-onset window is representative non-REM or REM-adjacent
    transitional sleep -- the question the bout-locked DiD turns on.
    """
    offsets = np.arange(LOCK_PRE, LOCK_POST + 1)
    rows, stage_rows = [], []
    for stage, heart_rate in visit_nights(step32, uuid):
        for start, end in bout_bounds(stage == REM_CODE):
            if end - start < MIN_BOUT_SECONDS:
                continue
            index = start + offsets
            keep = (index >= 0) & (index < len(heart_rate))
            row = np.full(len(offsets), np.nan)
            row[keep] = heart_rate[index[keep]]
            rows.append(row)
            if with_stages:
                codes = np.full(len(offsets), -1, dtype=int)
                codes[keep] = stage[index[keep]]
                stage_rows.append(codes)
    if not rows:
        return None
    with np.errstate(invalid="ignore"):
        curve = np.nanmean(np.vstack(rows), axis=0)
    # Bin after bout-averaging: a participant's curve is the unit of analysis, and
    # binning first would let a long bout dominate a short one within the bin.
    usable = (len(curve) // BIN_SECONDS) * BIN_SECONDS
    binned = np.nanmean(curve[:usable].reshape(-1, BIN_SECONDS), axis=1)
    if not with_stages:
        return binned, len(rows)

    codes = np.vstack(stage_rows)
    scored = codes >= 0
    fractions = {}
    for code, name in STAGE_NAMES.items():
        # Fraction of scored bout-seconds in this stage, at each offset, then binned
        # the same way the heart rate is.
        with np.errstate(invalid="ignore"):
            share = np.where(scored, codes == code, np.nan).astype(float)
            per_offset = np.nanmean(share, axis=0)
        fractions[name] = np.nanmean(per_offset[:usable].reshape(-1, BIN_SECONDS), axis=1)
    return binned, len(rows), fractions


def rem_locked_curves(step32, units: pd.DataFrame, progress_every: int = 0) -> pd.DataFrame:
    """REM-locked curves for any set of units, treated or control.

    Takes an arbitrary frame rather than the treated arm specifically, so the
    control side can be extracted with exactly the same bout selection, bout
    averaging and binning. A `treated` column is carried through, which is what
    lets the caller keep the two arms in separate files.
    """
    offsets = np.arange(LOCK_PRE, LOCK_POST + 1)
    usable = (len(offsets) // BIN_SECONDS) * BIN_SECONDS
    bin_offsets = offsets[:usable].reshape(-1, BIN_SECONDS).mean(axis=1)

    records = []
    for i, row in enumerate(units.itertuples(index=False), start=1):
        if progress_every and i % progress_every == 0:
            print(f"    ... {i}/{len(units)} units read", flush=True)
        curves = {}
        for visit, uuid in (("pre", row.pre_uuid), ("post", row.post_uuid)):
            result = visit_rem_locked(step32, uuid)
            if result is not None:
                curves[visit] = result
        # Both visits or neither: a participant contributing only one side would
        # shift the pre and post means by different case mixes.
        if len(curves) != 2:
            continue
        for visit, (binned, n_bouts) in curves.items():
            records.append(pd.DataFrame({
                "participant": row.participant,
                "treated": int(row.treated),
                "overlap_weight": row.overlap_weight,
                "visit": visit,
                "n_bouts": n_bouts,
                "offset_s": bin_offsets,
                "hr": binned,
            }))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def rem_locked_stages(step32, units: pd.DataFrame, progress_every: int = 0) -> pd.DataFrame:
    """Stage composition at each offset of the REM-locked window, per unit and visit.

    Same bout selection and binning as `rem_locked_curves`, so the composition
    describes exactly the seconds those curves average over.
    """
    offsets = np.arange(LOCK_PRE, LOCK_POST + 1)
    usable = (len(offsets) // BIN_SECONDS) * BIN_SECONDS
    bin_offsets = offsets[:usable].reshape(-1, BIN_SECONDS).mean(axis=1)

    records = []
    for i, row in enumerate(units.itertuples(index=False), start=1):
        if progress_every and i % progress_every == 0:
            print(f"    ... {i}/{len(units)} units read", flush=True)
        per_visit = {}
        for visit, uuid in (("pre", row.pre_uuid), ("post", row.post_uuid)):
            result = visit_rem_locked(step32, uuid, with_stages=True)
            if result is not None:
                per_visit[visit] = result
        if len(per_visit) != 2:
            continue
        for visit, (_, _, fractions) in per_visit.items():
            frame = pd.DataFrame({
                "participant": row.participant,
                "treated": int(row.treated),
                "overlap_weight": row.overlap_weight,
                "visit": visit,
                "offset_s": bin_offsets,
            })
            for name, values in fractions.items():
                frame[f"frac_{name}"] = values
            records.append(frame)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# ------------------------------------------------------------------- assembly


def build_cohort(step32, arm: str, first_report: pd.Series, visits: pd.DataFrame,
                 controls: pd.DataFrame, anthro: pd.DataFrame, people: pd.DataFrame,
                 control_cap: int = CONTROL_CAP) -> pd.DataFrame:
    """Step 32's overlap-weighted retained set for one arm, weights included."""
    treated = step32.attach_covariates(
        step32.build_treated(visits, first_report, arm), anthro, people
    )
    combined = pd.concat([treated, controls], ignore_index=True)
    weighted_full = step32.overlap_weights(combined, step32.PS_COVARIATES)
    ranked = (
        weighted_full[weighted_full.treated == 0]
        .sort_values("overlap_weight", ascending=False)
    )
    # How much weight the cap discards, on the full-pool fit that the ranking is
    # taken from. This is the number that decides whether the cap is a free I/O
    # saving or a truncation that moves the estimate.
    kept = ranked.overlap_weight.to_numpy()
    share = kept[:control_cap].sum() / kept.sum() * 100
    print(f"[{arm}] control cap {control_cap} of {len(ranked)} keeps {share:.1f}% of "
          f"total control weight; weight at cap {kept[min(control_cap, len(kept)) - 1]:.5f}, "
          f"max {kept[0]:.5f}", flush=True)
    control_rank = ranked.head(control_cap)
    retained = pd.concat(
        [weighted_full[weighted_full.treated == 1], control_rank], ignore_index=True
    )
    retained = step32.overlap_weights(retained, step32.PS_COVARIATES)
    balance = step32.weighted_balance(retained, step32.PS_COVARIATES)
    print(f"[{arm}] eligible treated {len(treated)} | retained treated "
          f"{int((retained.treated == 1).sum())} | controls "
          f"{int((retained.treated == 0).sum())} | max|SMD| "
          f"{balance.smd.abs().max():.3f}", flush=True)
    return retained


def exposure_durations(step32, arm: str, first_report: pd.Series, visits: pd.DataFrame,
                       anthro: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """Months on drug at the post visit, per treated participant.

    Step 32 keeps `post_date` but never stores the interval, and panel b needs it:
    labelling a cohort curve with one exemplar's duration would be wrong. Cheap --
    no raw channels are read here, so this runs in ~2 min rather than ~25.
    """
    treated = step32.attach_covariates(
        step32.build_treated(visits, first_report, arm), anthro, people
    )
    start = treated.participant.map(first_report)
    treated["months_on_drug_at_post"] = (treated.post_date - start).dt.days / 30.44
    return treated[["participant", "months_on_drug_at_post"]]


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    keep = np.isfinite(values) & np.isfinite(weights)
    return float((values[keep] * weights[keep]).sum() / weights[keep].sum())


def verify_against_published(states: pd.DataFrame, arm: str, control_cap: int) -> None:
    """Recompute the weighted DiD per state and compare to step 32's output.

    Only meaningful at the default cap: step 32 published at 700, so a sensitivity run
    at another cap is *expected* to differ and the flag is suppressed rather than
    reported as a mismatch.
    """
    published = pd.read_csv(PUBLISHED)
    published = published[(published.arm == arm) & (published.estimator == "overlap")]
    comparable = control_cap == CONTROL_CAP
    label = "vs published" if comparable else f"vs published (cap {CONTROL_CAP}, sensitivity)"
    print(f"\n[{arm}] weighted DiD recomputed from cached levels {label}:")
    for state in STATE_CODES:
        arms_did = {}
        for treated in (1, 0):
            side = states[states.treated == treated]
            weights = side.overlap_weight.to_numpy(dtype=float)
            delta = (side[f"{state}_post"] - side[f"{state}_pre"]).to_numpy(dtype=float)
            arms_did[treated] = weighted_mean(delta, weights)
        recomputed = arms_did[1] - arms_did[0]
        row = published[published.metric == f"d_{state}"]
        reference = float(row.did.iloc[0]) if not row.empty else np.nan
        if comparable:
            flag = "ok" if abs(recomputed - reference) < 0.02 else "MISMATCH"
        else:
            flag = f"delta {recomputed - reference:+.3f}"
        print(f"  d_{state:<6} recomputed {recomputed:+.3f}   published {reference:+.3f}   {flag}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="both",
                        choices=["both", "paper_strict", "sema_dated"])
    parser.add_argument("--control-cap", type=int, default=CONTROL_CAP,
                        help="weight-ranked controls to read raw channels for")
    parser.add_argument("--suffix", default="",
                        help="appended to output filenames; use for cap sensitivity runs")
    parser.add_argument("--exposure-only", action="store_true",
                        help="write only cohort_exposure_<arm>.csv; skips all raw reads")
    parser.add_argument("--controls-remlock", action="store_true",
                        help="also extract control REM-locked curves into "
                             "cohort_remlock_controls_<arm>.csv (a separate file, so the "
                             "treated-only curves the canonical Figure 1b reads are "
                             "untouched). Roughly 700 extra units of raw reads per arm.")
    parser.add_argument("--remlock-stages", action="store_true",
                        help="also write cohort_remlock_stages_<arm>.csv: the stage "
                             "composition at each offset of the REM-locked window, for "
                             "both arms. Answers whether the pre-onset window is "
                             "representative non-REM or REM-adjacent transitional sleep.")
    parser.add_argument("--stage-control-sample", type=int, default=0,
                        help="with --remlock-stages, read this many randomly sampled "
                             "controls instead of all of them. The composition curve is "
                             "sleep architecture and converges quickly; every treated "
                             "unit is always read. 0 means all controls.")
    parser.add_argument("--skip-states", action="store_true",
                        help="reuse the cached cohort_states_<arm>.csv instead of "
                             "re-reading raw channels for state levels")
    args = parser.parse_args()

    step32 = load_step32()
    OUT.mkdir(parents=True, exist_ok=True)

    visits = step32.load_sleep_visits()
    reports = step32.load_glp_reports()
    anthro, people = step32.load_covariates()
    # Must use `ever_glp_participants()`, NOT `set(reports.participant)`.
    # `load_glp_reports` drops undated rows, so the latter excludes 741 rather than 788 and
    # leaves 47 real GLP-1 users eligible as never-GLP-1 controls -- the bug fixed in
    # steps/31 and steps/32 (`MEDICATION_DATE_AUDIT.md` §3). This file had its own copy of
    # the construction and was missed in that fix, which showed up as every
    # verify_against_published row reporting MISMATCH once step 32 was re-run.
    controls = step32.attach_covariates(
        step32.build_controls(visits, step32.ever_glp_participants()), anthro, people
    )
    print(f"raw dated sleep visits {len(visits)} | never-GLP control pool {len(controls)}",
          flush=True)

    first_sema = reports[reports.is_semaglutide].groupby("participant").report_date.min()
    audit = pd.read_csv(step32.AUDIT)
    adjudicated = set(
        audit[audit.agent_class.astype(str).str.contains("semaglutide", case=False)
              & audit.timing_valid.astype(bool)]
        .RegistrationCode.astype(str).str.replace("10k_", "", case=False, regex=False)
    )
    all_arms = {
        "paper_strict": first_sema[first_sema.index.isin(adjudicated)],
        "sema_dated": first_sema,
    }
    arms = all_arms if args.arm == "both" else {args.arm: all_arms[args.arm]}

    for arm, first_report in arms.items():
        exposure = exposure_durations(step32, arm, first_report, visits, anthro, people)
        exposure.to_csv(OUT / f"cohort_exposure_{arm}.csv", index=False)
        months = exposure.months_on_drug_at_post
        print(f"[{arm}] months on drug at post: median {months.median():.1f}, "
              f"IQR {months.quantile(.25):.1f}-{months.quantile(.75):.1f}, "
              f"range {months.min():.1f}-{months.max():.1f} (n = {len(months)})", flush=True)
        if args.exposure_only:
            continue

        retained = build_cohort(step32, arm, first_report, visits, controls, anthro, people,
                                control_cap=args.control_cap)

        if args.skip_states:
            print(f"[{arm}] state levels: reusing cached "
                  f"cohort_states_{arm}{args.suffix}.csv", flush=True)
        else:
            states = state_levels(step32, retained)
            states.to_csv(OUT / f"cohort_states_{arm}{args.suffix}.csv", index=False)
            print(f"[{arm}] state levels: {int((states.treated == 1).sum())} treated, "
                  f"{int((states.treated == 0).sum())} controls", flush=True)
            verify_against_published(states, arm, args.control_cap)

        if not args.skip_states:
            curves = rem_locked_curves(step32, retained[retained.treated == 1])
            curves.to_csv(OUT / f"cohort_remlock_{arm}{args.suffix}.csv", index=False)
            n_participants = curves.participant.nunique() if not curves.empty else 0
            print(f"[{arm}] REM-locked curves: {n_participants} initiators with both visits",
                  flush=True)

        if args.controls_remlock:
            # Separate file on purpose: appending controls to cohort_remlock_<arm>.csv
            # would silently pull them into the canonical Figure 1b, whose reader does
            # not filter on `treated`.
            control_units = retained[retained.treated == 0]
            print(f"[{arm}] control REM-locked curves: reading {len(control_units)} "
                  f"control units", flush=True)
            control_curves = rem_locked_curves(step32, control_units, progress_every=100)
            control_curves.to_csv(
                OUT / f"cohort_remlock_controls_{arm}{args.suffix}.csv", index=False)
            n_controls = (control_curves.participant.nunique()
                          if not control_curves.empty else 0)
            print(f"[{arm}] control REM-locked curves: {n_controls} controls with "
                  f"both visits", flush=True)

        if args.remlock_stages:
            print(f"[{arm}] REM-locked stage composition: reading {len(retained)} units",
                  flush=True)
            units = retained
            if args.stage_control_sample:
                # Random rather than weight-ranked: taking the top-weighted controls
                # would sample the ones most like the treated arm, which is exactly the
                # wrong subset for a claim about sleep architecture in general.
                sampled = (retained[retained.treated == 0]
                           .sample(n=min(args.stage_control_sample,
                                         int((retained.treated == 0).sum())),
                                   random_state=SEED))
                units = pd.concat([retained[retained.treated == 1], sampled],
                                  ignore_index=True)
                print(f"[{arm}] stage composition: sampling {len(sampled)} of "
                      f"{int((retained.treated == 0).sum())} controls (seed {SEED})",
                      flush=True)
            stages = rem_locked_stages(step32, units, progress_every=50)
            stages.to_csv(OUT / f"cohort_remlock_stages_{arm}{args.suffix}.csv",
                          index=False)
            print(f"[{arm}] stage composition: "
                  f"{stages.participant.nunique() if not stages.empty else 0} units",
                  flush=True)

    print(f"\nwrote {OUT}/cohort_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
