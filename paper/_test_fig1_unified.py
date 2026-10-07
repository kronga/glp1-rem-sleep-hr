"""Build canonical Figure 1 from the reviewed clean unified cohort (steps/48).

By default this replaces
`paper/figures/final/fig1_sleep_phenotyping_and_finding.{pdf,png}`. Pass `--dry-run` to write
`outputs/_test_figures/fig1_unified.{pdf,png}` instead.

What actually changes, panel by panel
-------------------------------------
a  The design raster (`build_fig1_v2.SCHEMATIC`) carries the reviewed clean unified sleep
   count (94 matched pairs); its initiator colours are matched to panel b on load.
b  REBUILT. The exemplar is re-selected from the clean unified matched sleep cohort and its
   nights are resolved at that cohort's exact date-anchored pre/post window.
c  REBUILT. Every REM bout >= 3 min from the unified raw arm's 1:1 matched pairs, treated and
   control, aligned on bout onset. Weights are all 1.0 because the design is 1:1 -- the
   overlap weighting this panel used to carry is gone with the estimator it belonged to.
   Initiator visit is encoded consistently with panel b: teal before and vermillion on drug,
   both solid; the grey controls retain dashed visit 1 and solid visit 2.
d  REBUILT. The unified tabular DiD, read only from the reviewed, versioned 494-person run.

The REM-lock curves are extracted here rather than reused because `steps/48` computes per-
visit STATE MEANS, which is what its DiD needs and is not the same object as a bout-aligned
per-second curve. Only the 1:1 matched participants are read -- roughly 150 people rather
than the 5,667 the state-mean extraction touched -- so this costs minutes, and it is cached.

Panel c is now a subset of panel d's clean unified cohort by construction. The raw arm remains
descriptive because it fails the balance gate; the caption must say so.

Usage
-----
    PY=/net/mraid20/export/jasmine/david/anaconda3/bin/python3
    $PY paper/_test_fig1_unified.py
    $PY paper/_test_fig1_unified.py --dry-run
    $PY paper/_test_fig1_unified.py --refresh-curves   # rebuild the cohort-keyed curve cache
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
SCRATCH = RUN / "outputs" / "_test_figures"
FINAL = RUN / "paper" / "figures" / "final"
FINAL_STEM = "fig1_sleep_phenotyping_and_finding"
UNIFIED = RUN / "outputs" / "unified_cohort"
CANONICAL = (
    UNIFIED / "runs"
    / "semaglutide__floor_3m__stages_main__blocks_plus_lipids__modalities_paper__tabular"
)
CANONICAL_SLEEP = CANONICAL / "did_sleep.csv"
CURVES = UNIFIED / "cache" / "remlock_unified.csv"
CURVES_META = UNIFIED / "cache" / "remlock_unified.meta.json"

# Tested in `_test_before_colour.py`: a muted teal that remains well separated from the
# on-drug vermillion in colour and greyscale, without the harsher blue/orange pairing.
BEFORE = "#17808D"

BIN_SECONDS = 5
LOCK_PRE, LOCK_POST = -300, 600
MIN_BOUT_SECONDS = 180
REM_CODE = 12


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_canonical_run(step48, tabular: dict, exposure: pd.DataFrame) -> pd.DataFrame:
    """Prove the in-memory pair reconstruction matches the reviewed versioned run."""
    manifest_path = CANONICAL / "run_manifest.json"
    if not manifest_path.exists() or not CANONICAL_SLEEP.exists():
        raise FileNotFoundError(f"missing canonical unified run under {CANONICAL}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_exposure = int(manifest["dated_clean_initiators"])
    if expected_exposure != 494 or len(exposure) != expected_exposure:
        raise AssertionError(
            f"Figure 1 requires the reviewed 494-person clean exposure; "
            f"manifest={expected_exposure}, reconstructed={len(exposure)}"
        )
    step_path = Path(step48.__file__)
    if _sha256(step_path) != manifest["script_sha256"]:
        raise AssertionError(
            "steps/48_unified_cohort.py no longer matches the reviewed run manifest; "
            "rerun and review the canonical cohort before rebuilding Figure 1"
        )
    sleep_status = next(row for row in manifest["modality_status"]
                        if row["modality"] == "sleep")
    expected_pairs = int(sleep_status["matched"])
    if len(tabular["matched"]) != expected_pairs:
        raise AssertionError(
            f"clean unified sleep match drifted: expected {expected_pairs}, "
            f"reconstructed {len(tabular['matched'])}"
        )

    canonical = pd.read_csv(CANONICAL_SLEEP).sort_values("measurement").reset_index(drop=True)
    reconstructed = tabular["result"].sort_values("measurement").reset_index(drop=True)
    if canonical.measurement.tolist() != reconstructed.measurement.tolist():
        raise AssertionError("reconstructed sleep endpoints differ from the reviewed run")
    if not np.array_equal(canonical.n_pairs.to_numpy(), reconstructed.n_pairs.to_numpy()):
        raise AssertionError("reconstructed sleep pair counts differ from the reviewed run")
    for column in ("effect", "ci_low", "ci_high", "treated_delta", "control_delta"):
        if not np.allclose(canonical[column], reconstructed[column], rtol=0, atol=1e-12,
                           equal_nan=True):
            raise AssertionError(f"reconstructed sleep {column} differs from reviewed run")
    return canonical


def pair_signature(pairs: pd.DataFrame) -> str:
    """Stable identity for the raw matched pairs and their inherited visit windows."""
    columns = ["treated_id", "control_id", "pre_stage", "post_stage"]
    payload = pairs[columns].astype(str).sort_values(columns).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unified_cohort(step48, step26, step31, step47, want_exposure=False):
    """Re-run steps/48's sleep arms to recover the matched pairs in memory.

    Re-run rather than read from disk: `steps/48` persists estimates, not pair lists, and a
    figure that invented its own cohort from saved summaries would be exactly the drift this
    redesign removed. Same seed, same code path, so the pairs are the ones behind
    `did_sleep.csv`.
    """
    step26.SUPPORTED_STAGES = tuple(step48.MAIN_STAGES)
    exposure = step48.build_exposure(step31)
    exposure["participant_id"] = step48.as_registration_code(
        exposure.participant, step26).to_numpy()
    covariates = step26.load_covariates()
    excluded = set(step48.as_registration_code(
        pd.Series(sorted(step48.load_all_glp_reports().participant.unique())), step26))
    values = step26.load_snapshot_panel("sleep")
    dates = step48.visit_dates_for(step26, "sleep")

    tabular, windows, _ = step48.run_instrument(
        step26, step31, step47, "sleep", values, exposure, dates, covariates,
        excluded, step48.FLOOR_MONTHS, step48.DEFAULT_BLOCKING,
        stages=step48.MAIN_STAGES)
    inherited = windows[windows.RegistrationCode.isin(tabular["matched"])]

    link = step31.load_sleep_visits()
    link["participant_id"] = step48.as_registration_code(link.participant, step26).to_numpy()
    link["research_stage"] = link.research_stage.map(step48.canonical_stage)
    wanted = set(tabular["units"].participant_id)
    needed = step48.visits_to_extract(link, wanted, inherited)
    raw_values = step48.raw_values_panel(step31, needed, wanted)
    raw, _, _ = step48.run_instrument(
        step26, step31, step47, "sleep_raw", raw_values, exposure, dates, covariates,
        excluded, step48.FLOOR_MONTHS, step48.DEFAULT_BLOCKING,
        windows=inherited, stages=step48.MAIN_STAGES)
    if want_exposure:
        return tabular, raw, link, inherited, exposure, dates
    return tabular, raw, link, inherited


def remlock_curves(step32, link: pd.DataFrame, pairs: pd.DataFrame,
                   windows: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Bout-aligned REM-locked curves for every matched participant, both arms.

    Schema is the one `variant_b_both_arms` already consumes -- participant,
    overlap_weight, visit, n_bouts, offset_s, hr -- so the panel builder needs no change.
    `overlap_weight` is 1.0 throughout: the design is 1:1, and carrying a weight column that
    is always one is clearer than forking the panel to drop it.
    """
    signature = pair_signature(pairs)
    if CURVES.exists() and CURVES_META.exists() and not refresh:
        metadata = json.loads(CURVES_META.read_text(encoding="utf-8"))
        if metadata.get("pair_signature") == signature:
            return pd.read_csv(CURVES, dtype={"participant": str})
        logging.info("REM-lock cache belongs to a different cohort; rebuilding")

    extractor = _load("cohort_extract", RUN / "paper" / "_test_cohort_extract.py")
    window = windows.set_index("RegistrationCode")
    uuid_of = link.set_index(["participant_id", "research_stage"]).uuid.to_dict()

    assignments = []
    for row in pairs.itertuples(index=False):
        for participant, arm in ((row.treated_id, "treated"), (row.control_id, "control")):
            if participant not in window.index:
                # Controls inherit the treated unit's window, which is what the exact block
                # on (pre_stage, post_stage) guarantees they share.
                pre, post = row.pre_stage, row.post_stage
            else:
                pre, post = window.loc[participant, ["pre_stage", "post_stage"]]
            assignments.append((participant, arm, pre, post))

    records = []
    for index, (participant, arm, pre, post) in enumerate(assignments, start=1):
        for visit, stage in (("pre", pre), ("post", post)):
            uuid = uuid_of.get((participant, stage))
            if uuid is None:
                continue
            result = extractor.visit_rem_locked(step32, uuid)
            if result is None:
                continue
            curve, n_bouts = result
            offsets = np.arange(LOCK_PRE, LOCK_POST + 1)
            binned_offsets = offsets[: (len(offsets) // BIN_SECONDS) * BIN_SECONDS
                                     ].reshape(-1, BIN_SECONDS).mean(axis=1)
            for offset, value in zip(binned_offsets, curve):
                records.append({"participant": participant, "arm": arm,
                                "overlap_weight": 1.0, "visit": visit,
                                "n_bouts": n_bouts, "offset_s": offset, "hr": value})
        if index % 25 == 0:
            logging.info("  REM-lock %d/%d participants", index, len(assignments))

    frame = pd.DataFrame(records)
    CURVES.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CURVES, index=False)
    CURVES_META.write_text(json.dumps({
        "pair_signature": signature,
        "n_pairs": int(len(pairs)),
        "n_curve_participants": int(frame.participant.nunique()),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frame


def select_exemplar(step48, tabular: dict, windows: pd.DataFrame, link: pd.DataFrame,
                    exposure: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Rank unified-cohort initiators against the exemplar criteria from `_test_fig1_alt`.

    Same four constraints, re-applied to the new cohort rather than inherited: REM-selective
    (d_REM - d_nonREM > 0, so the panel illustrates the contrast the paper claims rather
    than a global rise), therapeutically exposed (>= 6 months -- kept at the STRICTER of the
    two floors even though the cohort's own floor is 3, because an exemplar is an
    illustration and there is no reason to draw a titrating participant), phenotypically
    credible (inside the arm's 10th-90th percentile on visceral fat, waist and hip, which is
    what rejected the previous default), and finally legible, which only the traces can
    settle and is checked after extraction.

    Returns candidates ordered by closeness to the arm's mean d_REM, so the drawn participant
    is typical of the effect rather than its largest riser -- the failure mode recorded in
    `_test_fig1_alt`'s header, where ranking on total rise selected against the contrast.
    """
    uuid_of = link.drop_duplicates(["participant_id", "research_stage"]).set_index(
        ["participant_id", "research_stage"]).uuid.to_dict()
    means = pd.read_parquet(UNIFIED / "cache" / "raw_visit_means.parquet").set_index("uuid")
    start = exposure.set_index("participant_id").start_date

    rows = []
    for window in windows.itertuples(index=False):
        pre = uuid_of.get((window.RegistrationCode, window.pre_stage))
        post = uuid_of.get((window.RegistrationCode, window.post_stage))
        if pre is None or post is None or pre not in means.index or post not in means.index:
            continue
        before, after = means.loc[pre], means.loc[post]
        if not np.isfinite([before.rem, after.rem, before.nrem, after.nrem]).all():
            continue
        post_date = dates.get((window.RegistrationCode, window.post_stage))
        rows.append({
            "participant": window.RegistrationCode,
            "pre_stage": window.pre_stage, "post_stage": window.post_stage,
            "d_rem": after.rem - before.rem,
            "contrast": (after.rem - before.rem) - (after.nrem - before.nrem),
            "months": (post_date - start.get(window.RegistrationCode)).days / 30.44,
        })
    frame = pd.DataFrame(rows)
    arm_mean = frame.d_rem.mean()

    units = tabular["units"]
    units = units[units.treated.eq(1)].set_index("participant_id")
    frame = frame.join(units[["vat_area", "waist_circumference", "hip_circumference"]],
                       on="participant")
    keep = frame[(frame.contrast > 0) & (frame.months >= 6)]
    for column in ("vat_area", "waist_circumference", "hip_circumference"):
        low, high = frame[column].quantile([0.10, 0.90])
        keep = keep[keep[column].between(low, high)]
    logging.info("exemplar: %d in arm, %d REM-selective and >=6 months and phenotypically "
                 "credible (arm mean d_REM %+.2f)", len(frame), len(keep), arm_mean)
    return keep.assign(typicality=(keep.d_rem - arm_mean).abs()).sort_values("typicality")


def cache_unified_nights(step31, link: pd.DataFrame, participant: str, pre_stage: str,
                         post_stage: str, months_on_drug: float) -> str:
    """Write one initiator's per-second nights at the UNIFIED window, for panel b.

    `_test_panela_extract.py` cannot be reused: it resolves pre/post through `steps/31`'s own
    date pairing against the `paper_strict` arm, so it would draw the exemplar at visits the
    unified cohort does not use. Same output schema, so `_test_fig1_alt.load_nights` reads it
    unchanged -- only the visit resolution differs, which is the entire point.

    No uuid and no study date is written, matching the existing convention that identifying
    detail stays out of `outputs/`.
    """
    import pyarrow.parquet as pq

    uuid_of = link.drop_duplicates(["participant_id", "research_stage"]).set_index(
        ["participant_id", "research_stage"]).uuid.to_dict()
    # Stable, non-reversible display label. Python's built-in hash is process-randomized and
    # made the same exemplar print under a different name on every rebuild.
    digest = hashlib.sha256(participant.encode("utf-8")).hexdigest()
    label = f"U{int(digest[:8], 16) % 10000:04d}"
    frames = []
    for visit, stage in (("pre", pre_stage), ("post", post_stage)):
        uuid = uuid_of[(participant, stage)]
        for night_dir in sorted((step31.TIMESERIES / uuid).glob("night_*")):
            stage_codes = np.asarray(pq.read_table(
                night_dir / "sleep_stage.parquet", columns=["values"]).column("values"))
            heart_rate = np.asarray(pq.read_table(
                night_dir / "heart_rate.parquet", columns=["values"]).column("values"))
            usable = min(len(stage_codes), len(heart_rate))
            frames.append(pd.DataFrame({
                "visit": visit, "night": night_dir.name, "second": np.arange(usable),
                "stage_code": stage_codes[:usable],
                "heart_rate": pd.to_numeric(pd.Series(heart_rate[:usable]), errors="coerce"),
            }))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(SCRATCH / f"{label}_nights.csv", index=False)
    pd.DataFrame([{"participant": label, "months_pre_to_start": np.nan,
                   "months_on_drug_at_post": months_on_drug,
                   "gap_months": np.nan}]).to_csv(
                       SCRATCH / f"{label}_meta.csv", index=False)
    return label


def trace_separation(label: str, alt) -> float:
    """Standardised separation between the two nights' heart-rate traces.

    The last exemplar criterion, and the one that cannot be read off summary statistics:
    a participant whose visit-level d_REM is healthy can still have traces that cross and
    read as no effect. Cohen's d over the per-second values of the two drawn nights.
    """
    frame = alt.load_nights(label)
    before = frame.loc[frame.visit == "pre", "hr"].dropna()
    after = frame.loc[frame.visit == "post", "hr"].dropna()
    pooled = np.sqrt((before.var() + after.var()) / 2)
    return float((after.mean() - before.mean()) / pooled) if pooled else float("nan")


def onset_contrast(curves: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Panel c as a number: the change in (after-onset minus before-onset) heart rate.

    Per participant and visit, mean heart rate in the 10 minutes after REM onset minus the
    5 minutes before it; per pair, the initiator's before-to-on-drug change in that step
    minus the control's. Aggregate only; written beside the other secondary sleep outputs.
    """
    from scipy import stats
    window = curves.assign(after=curves.offset_s > 0)
    step = (window.groupby(["participant", "visit", "after"]).hr.mean()
            .unstack("after").pipe(lambda f: f[True] - f[False]).unstack("visit"))
    change = step["post"] - step["pre"]
    did = (pairs.treated_id.map(change) - pairs.control_id.map(change)).dropna().to_numpy(float)
    half = stats.t.ppf(0.975, len(did) - 1) * did.std(ddof=1) / np.sqrt(len(did))
    return pd.DataFrame([{
        "n_pairs": len(did), "onset_step_did": did.mean(),
        "ci_low": did.mean() - half, "ci_high": did.mean() + half,
        "p": float(stats.ttest_1samp(did, 0.0).pvalue),
    }])


def complete_pairs_only(curves: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Keep only matched pairs where all four curves exist: both arms, both visits.

    Without this the panel draws a participant who has a curve at one visit and not the
    other, so its "before" and "on drug" lines are averages over DIFFERENT people -- 75
    against 74 in the first render. For a within-person before/after panel that is simply
    wrong, and it is invisible on the page: the curves look fine, the legend reports the
    larger n, and nothing flags that the two lines have different denominators.

    A visit loses its curve only when the night has no scored REM at all (`rem_seconds` 0,
    so no bout clears MIN_BOUT_SECONDS). Dropping the pair rather than the visit keeps the
    drawn cohort identical to the one the DiD is estimated on.
    """
    complete = {
        participant for participant, group in curves.groupby("participant")
        if group.visit.nunique() == 2
    }
    keep = pairs[pairs.treated_id.isin(complete) & pairs.control_id.isin(complete)]
    members = set(keep.treated_id) | set(keep.control_id)
    dropped = curves.participant.nunique() - len(members)
    if dropped:
        logging.info("panel c: dropped %d participants to keep %d complete pairs",
                     dropped, len(keep))
    return curves[curves.participant.isin(members)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-curves", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="write fig1_unified to outputs/_test_figures instead of final")
    parser.add_argument("--exemplar", default="P02",
                        help="opaque label from the local participant crosswalk")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    SCRATCH.mkdir(parents=True, exist_ok=True)

    step48 = _load("step48", RUN / "steps" / "48_unified_cohort.py")
    step26 = _load("step26", RUN / "steps" / "26_paper_reanalysis.py")
    step31 = _load("step31", RUN / "steps" / "31_raw_sleep_glp_cohort.py")
    step47 = _load("step47", RUN / "steps" / "47_date_anchored_pairing.py")
    step32 = _load("step32", RUN / "steps" / "32_raw_sleep_matched_did.py")
    v1 = _load("opener", RUN / "paper" / "build_opener_figure.py")
    v2 = _load("fig1_v2", RUN / "paper" / "build_fig1_v2.py")
    alt = v2.alt

    tabular, raw, link, windows, exposure, dates = unified_cohort(
        step48, step26, step31, step47, want_exposure=True)
    did = validate_canonical_run(step48, tabular, exposure)
    logging.info("unified: %d tabular pairs, %d raw pairs",
                 len(tabular["matched"]), len(raw["matched"]))

    curves = remlock_curves(step32, link, raw["pairs"], windows, args.refresh_curves)
    curves = complete_pairs_only(curves, raw["pairs"])
    onset = onset_contrast(curves, raw["pairs"])
    onset.to_csv(UNIFIED / "secondary_sleep" / "remlock_onset_contrast.csv", index=False)
    logging.info("REM-onset step DiD:\n%s", onset.to_string(index=False))
    treated = curves[curves.arm == "treated"].drop(columns="arm")
    controls = curves[curves.arm == "control"].drop(columns="arm")
    logging.info("REM-lock curves: %d treated, %d control participants",
                 treated.participant.nunique(), controls.participant.nunique())

    contrast_q = float(
        did.set_index("measurement").loc["heart_rate_rem_minus_nrem"].fdr_within_modality)

    # Keep the paper's established P02 exemplar, but resolve its visits from the clean unified
    # cohort rather than reusing the legacy steps/31 cache. P02 passes every unified boundary:
    # clean dated exposure, date-anchored window, tabular eligibility/match, raw eligibility/
    # match, and the phenotype/REM-selectivity exemplar filters below. The earlier claim that
    # P02 was outside this cohort was false; automatic maximisation of trace separation was
    # what substituted another participant.
    candidates = select_exemplar(step48, tabular, windows, link, exposure, dates)
    exemplar_id = step48.as_registration_code(
        pd.Series([alt.resolve_participant(args.exemplar)]), step26).iloc[0]
    selected = candidates[candidates.participant.eq(exemplar_id)]
    if len(selected) != 1:
        raise AssertionError(
            f"Panel b exemplar {args.exemplar} is not a unique clean-unified candidate"
        )
    if exemplar_id not in raw["matched"]:
        raise AssertionError(
            f"Panel b exemplar {args.exemplar} is not in the clean unified raw match"
        )
    exemplar = next(selected.itertuples(index=False))
    chosen = cache_unified_nights(
        step31, link, exemplar.participant, exemplar.pre_stage,
        exemplar.post_stage, exemplar.months,
    )
    separation = trace_separation(chosen, alt)
    percentile = (candidates.d_rem < exemplar.d_rem).mean() * 100
    print(f"\npanel b exemplar: {args.exemplar} ({chosen})  separation d "
          f"{separation:+.2f}  d_REM {exemplar.d_rem:+.1f} = "
          f"{percentile:.0f}th percentile of the eligible arm")

    v1.configure_style()
    palette = {
        ("initiator", "pre"): BEFORE,
        ("initiator", "post"): alt.ONDRUG,
        ("control", "pre"): alt.CONTROL_COLOUR,
        ("control", "post"): alt.CONTROL_COLOUR,
    }
    fig = v2.build_figure(
        chosen,
        palette=palette,
        cohort_curves=(treated, controls),
        did_source=CANONICAL_SLEEP,
        contrast_q=contrast_q,
        treated_linestyles={"pre": "solid", "post": "solid"},
    )

    stem = SCRATCH / "fig1_unified" if args.dry_run else FINAL / FINAL_STEM
    v2.save_figure(fig, stem)
    plt.close(fig)
    print(f"\npanel c: {treated.participant.nunique()} initiators, "
          f"{controls.participant.nunique()} matched controls (1:1, weights all 1.0)")
    print(f"panel d: unified tabular DiD, {len(tabular['matched'])} matched pairs")
    print(f"panel a: clean unified sleep design, {len(tabular['matched'])} matched pairs")
    print(f"panel b: exemplar {args.exemplar} ({chosen}), clean unified cohort")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
