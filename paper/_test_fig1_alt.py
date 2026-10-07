"""TEST ONLY -- candidate rebuild of Figure 1's top half. Overrides nothing.

Panels
------
a  One initiator's before/on-drug night pair: two hypnograms and both 1 Hz heart-rate
   traces on a common axis anchored at sleep onset, each night's REM bouts on its own
   ribbon. The exemplar -- what a single person's shift looks like.
b  Cohort mean of every REM bout >= 3 min, aligned on bout onset, before vs on drug,
   paper-adjudicated arm with a SEM ribbon. The alignment that actually puts REM on top
   of REM. Exposure-definition robustness is deliberately NOT shown here -- see
   `DRAW_SECONDARY_ARM` and figS6.
c  The paper's primary result: 1:1 no-reuse matched pairs, each arm's own pre -> post
   change, so the DiD is the gap between the two markers.

Panel a is one participant by construction; panels b and c are cohort estimates.

Panels b and c are estimated on DIFFERENT cohorts, and a caption must say so
-------------------------------------------------------------------------------
Panel b needs per-second channels, which exist only for the raw-channel cohort of
`steps/31`/`steps/32`. On that cohort 1:1 matching fails both design gates (17/35
treated retained = 48.6% against a >=85% gate, max |SMD| 0.42 against <0.10), so
step 32 designates overlap weighting the claim-eligible estimator there and 1:1 the
sensitivity. Panel b is therefore overlap-weighted, n = 33 adjudicated initiators.

Panel c is the paper's primary analysis on the WatchPAT summary fields
(`steps/26` -> `outputs/paper_reanalysis/primary_1to1_results.csv`), where 1:1
no-reuse matching *passes* both gates: 44/48 treated retained (91.7%),
max |SMD| 0.089. Those are the numbers the published Figure 1c reports
(REM +6.97 [4.36, 9.72], n = 41 pairs) and this panel reproduces them exactly.

So the same effect appears at +6.97 (panel c, summary fields, 1:1) and +4.22 (panel
b's cohort, raw channels, overlap-weighted). Both are defensible on their own
cohort; neither number may be quoted as the other's.

Panel a's exemplar: how P02 was chosen
---------------------------------------------
Four constraints, applied in this order. Each one killed an earlier candidate, so
they are all load-bearing:

1. REM-SELECTIVE. d_REM-minus-NREM > 0. Rejects P03, the largest riser on raw
   REM heart rate (d_REM +17.8) whose contrast is -0.1 bpm -- it rises globally and
   illustrates the opposite of the claim. Ranking on total rise selects *against*
   the contrast being illustrated.
2. THERAPEUTIC EXPOSURE: >= 6 months on drug. Semaglutide titrates 0.25 -> 0.5 -> 1.0 mg
   over ~3 months (Wegovy to 2.4 mg over 4-5), so a participant scanned earlier is
   still sub-maintenance. Rejects P04 (2.1 months) despite it having the best
   night data of any candidate (161 / 150 min REM). This cut the arm from 35 to 11.
3. PHENOTYPICALLY CREDIBLE. Rejects P05, the previous default: waist 113 /
   hip 93 is inverted against the arm (99.6 / 110.4) and visceral fat 275 is 42% above
   the arm mean, so it is a central-obesity outlier inside an already-obese arm. Also
   rejects P06, whose effect is the most representative of all (d_REM +4.84,
   contrast +1.35, both ~ arm means) but whose VAT is 331.
4. LEGIBLE. The two traces must actually separate. Rejects P07, the most
   covariate-typical of the eligible 11 (typicality 0.41, 15.4 months), because its
   before trace sits ABOVE its on-drug trace through hours 2-4 (separation d = 1.10)
   and reads as no effect.

P02: female 46, 6.6 months on drug, VAT 131 (BELOW the arm mean 194.5),
waist 104 < hip 119. Separation d = 2.77, on-drug above before 99% of the night, and
both drawn nights land within 0.4 bpm of their own visit means -- so the drawn pair
reproduces the participant's visit-level d_REM (+12.9) exactly.

Its cost, stated plainly: d_REM +12.8 is the 83rd percentile of the arm, ~2.8x the
arm mean of +4.57. The title says so. Within the exposure-credible subset, covariate
typicality and visual legibility are anti-correlated -- the typical effect (~+5 bpm)
is small against ~20 bpm of within-night variability, so no participant is both
average and visually clean. That is a property of the data, not of the plotting.

The two sleep studies are 23.0 months apart, so the single-subject contrast is not
drift-controlled. That is what panels b and c are for: the exemplar illustrates, the
cohort panels evidence.

Alternative kept warm: P08 with `--post-night night_1`, at the 60th percentile
for REM rise (d_REM +5.53 vs arm mean +4.57) -- nearer average, separation d 1.61.

Provenance
----------
    paper/_test_panela_extract.py   -> <participant>_nights.csv       (panel a)
    paper/_test_cohort_extract.py   -> cohort_remlock_<arm>.csv       (panel b)
    outputs/paper_reanalysis/primary_1to1_results.csv, already committed  (panel c)
The first two live under `outputs/_test_figures/`, with no UUIDs or study dates
written. The cohort extractor verifies its cached levels reproduce step 32's published
DiD before this script reads them; `summarise_cohort` re-prints panel c's numbers
beside the published file. Every number drawn is printed on each run.

Usage
-----
    PY=/net/mraid20/export/jasmine/david/anaconda3/bin/python3
    $PY paper/_test_panela_extract.py P02   # once per exemplar, ~2 min
    $PY paper/_test_cohort_extract.py              # once, ~25 min (panel b)
    $PY paper/_test_cohort_extract.py --exposure-only   # ~2 min (panel b's labels)
    $PY paper/_test_fig1_alt.py                    # default exemplar
    $PY paper/_test_fig1_alt.py P08 --post-night night_1   # the alternative
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "outputs" / "_test_figures"

# Participant identifiers are deliberately absent from this repository. Exemplars are
# referred to by opaque labels (P01, P02, ...) and resolved through a crosswalk that is
# gitignored and never leaves the analysis host. `build_opener_figure.py` follows the same
# convention with its opaque REPRESENTATIVE_UUID.
DEFAULT_PARTICIPANT = "P02"
CROSSWALK = RUN / "paper" / ".participant_crosswalk.json"


def resolve_participant(label: str) -> str:
    """Map an opaque label to the real participant key via the LOCAL crosswalk.

    A real numeric id passed directly still works, for interactive use. Without the
    crosswalk a label cannot be resolved, which is intentional -- the repository alone is
    not enough to re-identify anyone.
    """
    if label.isdigit():
        return label
    if not CROSSWALK.exists():
        raise SystemExit(
            f"cannot resolve {label!r}: {CROSSWALK.name} is local-only and gitignored. "
            f"Pass the numeric participant key directly instead."
        )
    inverse = {v: k for k, v in json.loads(CROSSWALK.read_text()).items()}
    if label not in inverse:
        raise SystemExit(f"{label!r} not in the crosswalk; have {sorted(inverse)}")
    return inverse[label]

_spec = importlib.util.spec_from_file_location("opener", RUN / "paper" / "build_opener_figure.py")
opener = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(opener)

STAGE_CODES = opener.STAGE_CODES
STAGE_ROW = opener.STAGE_ROW
BLACK, GREY, LIGHT_GREY = opener.BLACK, opener.GREY, opener.LIGHT_GREY
BEFORE = opener.BLUE
ONDRUG = opener.VERMILLION

REM_CODE = 12
SLEEP_CODES = (12, 13, 15)
HR_BOUNDS = (30.0, 140.0)
SMOOTH_SECONDS = 61

# REM-locked window, seconds relative to bout onset.
LOCK_PRE, LOCK_POST = -300, 600
MIN_BOUT_SECONDS = 180

# The hypnogram rows are ~35 pt tall, so their rotated side label has to be short
# enough not to overflow into the neighbouring row.
VISIT_SHORT = {"pre": "Before", "post": "On drug"}
VISIT_COLOUR = {"pre": BEFORE, "post": ONDRUG}

PRIMARY_ARM = "paper_strict"
SECONDARY_ARM = "sema_dated"

# Panel b draws the primary arm only. The dated arm was drawn dashed for a while and
# was removed, for three measured reasons:
#
#  1. Redundant to the eye. Its curve lies inside the primary arm's SEM ribbon at 96%
#     of offsets pre and 100% post; max separation post is 0.55 bpm against a ribbon
#     half-width of 1.15. It added two curves and almost no resolvable signal.
#  2. Redundant to the paper. `submission/figures/figS6_exposure_permissiveness` already
#     answers "does this survive a looser exposure definition", and answers it better:
#     three arms (n = 33 / 96 / 130, the same cohorts), on the REM-minus-NREM estimand,
#     with CIs, quantifying the attenuation (+1.10 -> +1.04 -> +0.94) and the ~1.8-fold
#     precision gain. A dashed line here restates that unquantified.
#  3. Actively misleading. The arms are NESTED -- the adjudicated 33 are a subset of the
#     dated 96 -- so two curves lying on top of each other reads as independent
#     replication when a third of the weight is literally the same people.
#
# The extraction still writes both arms; flip this to draw the comparison while
# iterating. Keep it False for anything that leaves this repo.
DRAW_SECONDARY_ARM = False

# Panel b weights participants, not bouts, so a participant whose visit yielded a
# single qualifying bout counts as much as one with 24 -- and a one-bout "bout-averaged
# curve" is not an average at all. Require both visits to clear this floor.
#
# Measured on the cached curves before adopting it: the floor costs 0 of 33 in the
# primary arm (its thinnest participant already has 3 bouts per visit) and 2 of 96 in
# the secondary arm, shifting the secondary curves by at most 0.24 bpm against a SEM
# ribbon half-width of 0.82-0.85 bpm. Raising it to 5 would drop 8 of 96 and move the
# pre->post gap 4.85 -> 4.56 bpm, which is discarding signal, not noise -- so 2 is the
# floor that removes the indefensible case and nothing else.
MIN_BOUTS_PER_VISIT = 2


def load_nights(participant: str) -> pd.DataFrame:
    frame = pd.read_csv(OUT / f"{participant}_nights.csv")
    heart_rate = pd.to_numeric(frame.heart_rate, errors="coerce")
    frame["hr"] = heart_rate.where(heart_rate.between(*HR_BOUNDS))
    return frame


def visit_labels(participant: str) -> dict[str, str]:
    """Panel a's labels: this exemplar's own time on drug."""
    months = pd.read_csv(OUT / f"{participant}_meta.csv").months_on_drug_at_post.iloc[0]
    return {"pre": "Before initiation", "post": f"On semaglutide ({months:.1f} mo)"}


def exemplar_descriptor(participant: str) -> str:
    """How panel a's title should describe this exemplar, from its rank in the arm.

    Hardcoding "the largest REM-selective rise" was correct only for P05 and
    silently false for anyone else, so the wording is derived from the participant's
    actual percentile on d_rem_minus_nrem within the paper-adjudicated arm.
    """
    panel = pd.read_csv(RUN / "outputs" / "raw_sleep_glp_cohort" / "panel.csv",
                        dtype={"participant": str})
    arm = panel[panel.arm == PRIMARY_ARM]
    own = arm[arm.participant == participant]
    if own.empty:
        return "a GLP-1 initiator"
    contrast = float(own.d_rem_minus_nrem.iloc[0])
    if contrast >= arm.d_rem_minus_nrem.max():
        return "the largest REM-selective rise in the cohort"
    pc_contrast = (arm.d_rem_minus_nrem < contrast).mean() * 100
    pc_rise = (arm.d_rem < float(own.d_rem.iloc[0])).mean() * 100
    # "Typical" has to hold on BOTH metrics. P02 sits at the 66th percentile on
    # the REM-minus-NREM contrast but the 83rd on absolute REM rise, and absolute rise
    # is what the eye reads off panel a -- calling it typical would understate the panel.
    if 25 <= pc_contrast <= 75 and 25 <= pc_rise <= 75:
        return "a typical REM-selective responder"
    # Kept terse on purpose: the assembled title must stay under ~88 characters or it
    # overflows 179 mm, and `fit_to_width` cannot shrink text -- it widens the canvas
    # instead, silently breaking the column fit.
    return f"a REM-selective responder, {ordinal(round(pc_rise))} percentile"


def ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 11th, 91st -- "91th" is not a word."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def cohort_visit_labels(curves: pd.DataFrame, arm: str) -> dict[str, str]:
    """Panel b's labels: the plotted cohort's exposure duration, not the exemplar's.

    Panel a's exemplar was 6.7 months on drug; carrying that number onto a cohort mean
    would misreport every other participant, and the exemplar is in fact below the
    cohort median. Restricted to the participants actually drawn, so the bout floor and
    any dropped visits are reflected.
    """
    exposure = pd.read_csv(OUT / f"cohort_exposure_{arm}.csv", dtype={"participant": str})
    plotted = exposure[exposure.participant.isin(set(curves.participant))]
    months = plotted.months_on_drug_at_post
    return {
        "pre": "Before initiation",
        "post": f"On semaglutide (median {months.median():.1f} mo, "
                f"IQR {months.quantile(0.25):.1f}–{months.quantile(0.75):.1f})",
    }


def representative_nights(frame: pd.DataFrame,
                          override: dict[str, str] | None = None) -> dict[str, str]:
    """Per visit, the night with the most scored REM -- the one worth drawing.

    Most-scored-REM is chosen deliberately over most-dramatic: the participant is already
    selected partly on effect, so also selecting their largest night would compound the
    selection. REM *coverage* is orthogonal to effect size; REM *magnitude* is not.

    It happens to be near-optimal on other criteria too. For P02 it picks pre
    night_2 and post night_2, which are the two nights closest to their own visit means
    (-0.3 and -0.4 bpm), so the drawn pair reproduces the visit-level d_REM exactly.

    `override` exists to render comparisons and to fix duration mismatches -- e.g.
    P08's post night_2 runs 10.2 h against a 7.5 h pre night, leaving 2.7 h of
    single-trace panel, where `--post-night night_1` (8.4 h) matches better AND lands
    nearer that participant's visit mean. Not for reaching for a bigger gap.

    Beware: a visit can have fewer nights in the raw store than the link table's three
    (P05's pre visit has one), so for some participants the rule cannot
    discriminate on that side at all.
    """
    rem_seconds = (
        frame[frame.stage_code == REM_CODE]
        .groupby(["visit", "night"]).hr.count()
    )
    nights = {visit: rem_seconds[visit].idxmax() for visit in ("pre", "post")}
    for visit, night in (override or {}).items():
        available = set(frame[frame.visit == visit].night)
        if night not in available:
            raise SystemExit(f"{visit} {night} not cached; have {sorted(available)}")
        nights[visit] = night
    return nights


def smooth(values: np.ndarray) -> np.ndarray:
    """Display-only box filter; the analysis uses per-second means within stage."""
    filled = np.nan_to_num(values, nan=np.nanmean(values))
    padded = np.pad(filled, SMOOTH_SECONDS // 2, mode="edge")
    kernel = np.ones(SMOOTH_SECONDS) / SMOOTH_SECONDS
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def bout_bounds(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index pairs for each True run."""
    edges = np.diff(mask.astype(int))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask.size and mask[0]:
        starts.insert(0, 0)
    if mask.size and mask[-1]:
        ends.append(mask.size)
    return list(zip(starts, ends))


def sleep_onset_second(stage: np.ndarray) -> int:
    asleep = np.flatnonzero(np.isin(stage, SLEEP_CODES))
    return int(asleep[0]) if asleep.size else 0


def hypnogram_rows(stage: np.ndarray) -> np.ndarray:
    rows = np.full(len(stage), np.nan)
    for code, name in STAGE_CODES.items():
        rows[stage == code] = STAGE_ROW[name]
    return rows


def draw_hypnogram(ax: plt.Axes, hours: np.ndarray, stage: np.ndarray, colour: str,
                   label: str, shade_alpha: float = 0.16) -> None:
    for start, end in bout_bounds(stage == REM_CODE):
        ax.axvspan(hours[start], hours[end - 1], color=colour, alpha=shade_alpha,
                   linewidth=0, zorder=0)
    ax.step(hours, hypnogram_rows(stage), where="post", color=BLACK, linewidth=0.7, zorder=3)
    pretty = {"deep": "Deep", "light": "Light", "rem": "REM", "wake": "Wake"}
    row_to_name = {row: name for name, row in STAGE_ROW.items()}
    ordered = sorted(row_to_name)
    ax.set_yticks(ordered)
    ax.set_yticklabels([pretty[row_to_name[r]] for r in ordered], fontsize=6.2)
    for tick, row in zip(ax.get_yticklabels(), ordered):
        if row_to_name[row] == "rem":
            tick.set_color(colour)
            tick.set_fontweight("bold")
    ax.set_ylim(-0.6, 3.6)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    ax.tick_params(axis="y", length=2)
    # Left of the stage names, rotated. Above the frame it reads as belonging to the
    # panel above it; inside the frame it sits on the step trace.
    ax.text(-0.105, 0.5, label, transform=ax.transAxes, rotation=90, ha="center",
            va="center", fontsize=6.4, color=colour, fontweight="bold")


def variant_a(ax_pre: plt.Axes, ax_post: plt.Axes, ax_hr: plt.Axes, frame: pd.DataFrame,
              labels: dict[str, str], nights: dict[str, str], descriptor: str,
              panel_letter: str = "a", show_title: bool = True,
              colours: dict[str, str] | None = None,
              linestyles: dict[str, str | tuple] | None = None,
              shade_alpha: dict[str, float] | None = None,
              mean_linestyle: str | tuple = (0, (4, 2))) -> None:
    """Both nights on one axis, anchored at sleep onset.

    Every style argument defaults to today's published look -- blue before, vermillion on
    drug, both solid -- so the canonical builder is untouched. They exist for
    `_test_fig1_v3.py`, which is trying schemes that make panels b, c and d encode the same
    thing the same way.

    `colours` recolours EVERY mark belonging to a visit (hypnogram shading, its REM y-tick,
    the rotated side label, the REM-bout ribbon, the trace, its night-mean rule and that
    rule's value label). A panel where only some of them moved would assert two colours for
    one night. `linestyles` applies to the HEART-RATE TRACE only -- the hypnogram is a step
    function whose dashes would read as missing epochs.

    `shade_alpha` is per visit and is NOT decoration. The hypnogram REM bands are a tint of
    the visit's own colour, so a scheme that gives one visit a pale hue renders that visit's
    bands nearly white at the alpha that suits a saturated one. Raising the pale visit's
    alpha restores comparable band weight; it cannot make the two bands *differ* in the way
    two hues did, because two tints of one hue at low alpha converge on the same near-white.
    The bands mark where REM is inside a row, the row labels say which night it is.

    `mean_linestyle` guards a collision. The night-mean rules and the single legend entry
    explaining them are dashed (4, 2), and that swatch is grey as a stand-in for "each rule
    takes its own night's colour". If a scheme also draws a grey dashed TRACE, the swatch
    reads as that trace and one mark carries two meanings -- such a scheme must lengthen
    this dash. Inside the axes the rules are never ambiguous: they are flat and carry their
    value. The default is today's published value.
    """
    visit_colour = {**VISIT_COLOUR, **(colours or {})}
    trace_style = {"pre": "solid", "post": "solid", **(linestyles or {})}
    band_alpha = {"pre": 0.16, "post": 0.16, **(shade_alpha or {})}

    traces = {}
    for visit, night in nights.items():
        night_frame = frame[(frame.visit == visit) & (frame.night == night)]
        stage = night_frame.stage_code.to_numpy()
        onset = sleep_onset_second(stage)
        hours = (night_frame.second.to_numpy() - onset) / 3600.0
        traces[visit] = (hours, stage, smooth(night_frame.hr.to_numpy()))

    draw_hypnogram(ax_pre, *traces["pre"][:2], visit_colour["pre"], VISIT_SHORT["pre"],
                   shade_alpha=band_alpha["pre"])
    draw_hypnogram(ax_post, *traces["post"][:2], visit_colour["post"], VISIT_SHORT["post"],
                   shade_alpha=band_alpha["post"])

    # Two nights of overlapping axvspans read as mud, so each night's REM bouts get
    # their own ribbon above the traces; the hypnograms carry the in-frame shading.
    RIBBON = {"pre": (0.965, 0.030), "post": (0.925, 0.030)}
    for visit, (hours, stage, hr) in traces.items():
        colour = visit_colour[visit]
        base, height = RIBBON[visit]
        for start, end in bout_bounds(stage == REM_CODE):
            ax_hr.add_patch(Rectangle(
                (hours[start], base), hours[end - 1] - hours[start], height,
                transform=ax_hr.get_xaxis_transform(), color=colour, alpha=0.55,
                linewidth=0, zorder=4, clip_on=True))
        ax_hr.plot(hours, hr, color=colour, linewidth=0.9,
                   linestyle=trace_style[visit], zorder=3)
        rem_mean = np.nanmean(np.where(stage == REM_CODE, hr, np.nan))
        ax_hr.axhline(rem_mean, color=colour, linewidth=0.7, linestyle=mean_linestyle,
                      zorder=2)
        # 5 pt of clearance, not 0.004 axes fractions: the fractional offset scaled with
        # the panel, and once the panel spanned the full figure width these two labels
        # printed hard against the right spine.
        ax_hr.annotate(f"{rem_mean:.0f}", (1.0, rem_mean),
                       xycoords=ax_hr.get_yaxis_transform(),
                       xytext=(5, 0), textcoords="offset points",
                       ha="left", va="center", fontsize=6.2, color=colour,
                       fontweight="bold")
    # Inside the frame at ribbon height. Outside on the left it collides with the topmost
    # y-tick label when the ceiling is a 3-digit value; outside on the right it extends
    # the tight bbox and stops `fit_to_width` converging on 179 mm.
    ax_hr.text(0.004, 0.96, "REM episodes", transform=ax_hr.transAxes, ha="left", va="center",
               fontsize=6.0, color=GREY)

    # Robust limits. Two things drag a naive min/max: a settling artifact in the first
    # ~20 s of a recording, and brief real tachycardias during scored awakenings.
    #
    # The awakenings are NOT removed or imputed. Checked on P05's night_2, whose
    # largest excursion is a 110 s plateau at 114-116 bpm: the hypnogram scores it `wake`
    # (mean 99.9 bpm across it), neighbouring samples are smooth rather than erratic, so
    # it is a real arousal and not device noise. "Wake in study" is one of the paper's
    # four reported states, so imputing it would delete signal the analysis reports on.
    # Instead the axis is set from a robust range so the before/on-drug separation stays
    # legible. Brief awakenings may extend beyond that display range; an isolated peak
    # annotation adds clutter without changing the comparison shown by the full traces.
    stacked = np.concatenate([hr for _, _, hr in traces.values()])
    low, high = np.nanpercentile(stacked, [0.5, 99.0])
    ax_hr.set_ylim(low - 4, high + 0.42 * (high - low))

    ax_hr.set_ylabel("Heart rate\n(bpm)")
    ax_hr.set_xlabel("Hours from sleep onset")
    for side in ("top", "right"):
        ax_hr.spines[side].set_visible(False)
    ax_hr.grid(axis="y", color="#ECECEC", linewidth=0.6)
    ax_hr.set_axisbelow(True)
    # A three-item legend spans the panel and lands on whichever trace is lowest.
    # The rotated side labels already name the two nights, so only the dashed line
    # needs explaining, and one line of text fits where no legend box would.
    ax_hr.legend(
        handles=[Line2D([], [], color=GREY, linewidth=0.7, linestyle=mean_linestyle,
                        label="night mean heart rate in REM")],
        # Below the REM ribbons (which occupy the top 8% of the axes), above both traces.
        loc="upper left", frameon=False, fontsize=6.2, handlelength=2.2,
        bbox_to_anchor=(-0.008, 0.90),
    )
    # The title states the exemplar's responder percentile, which is a required
    # disclosure -- it is a SELECTED participant. When the title is suppressed the
    # caller must carry that disclosure in the caption instead; Figure 1's does.
    if show_title:
        ax_pre.set_title(
            f"One participant, two nights: {descriptor} "
            f"({labels['post'].split('(')[-1].rstrip(')')} on drug)",
            loc="left", pad=8.0,
        )
    if panel_letter:
        opener.panel_label(ax_pre, panel_letter, x=-0.115, y=1.90 if show_title else 1.0)


def weighted_mean_and_sem(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Overlap-weighted mean, with the SEM taken over the effective sample size.

    n_eff = (sum w)^2 / sum w^2, the usual Kish effective n. Using the raw count
    would understate the interval, because a handful of high-weight participants
    carry most of the estimate.
    """
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if values.size < 2:
        return (float(values.mean()) if values.size else np.nan), np.nan
    total = weights.sum()
    mean = float((values * weights).sum() / total)
    variance = float((weights * (values - mean) ** 2).sum() / total)
    n_eff = total**2 / float((weights**2).sum())
    return mean, float(np.sqrt(variance / n_eff))


def _thin_participants_dropped(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop participants without MIN_BOUTS_PER_VISIT qualifying bouts at BOTH visits.

    Applied identically to the treated and control arms: if the gate ran on only one
    arm, the two cohort means would rest on different bout-count distributions.
    """
    per_visit = frame.groupby(["participant", "visit"]).n_bouts.first().unstack()
    keep = per_visit[(per_visit >= MIN_BOUTS_PER_VISIT).all(axis=1)].index
    dropped = len(per_visit) - len(keep)
    if dropped:
        print(f"  {label}: dropped {dropped} of {len(per_visit)} participants with "
              f"< {MIN_BOUTS_PER_VISIT} qualifying REM episodes at a visit")
    return frame[frame.participant.isin(set(keep))]


def cohort_remlock(arm: str) -> pd.DataFrame:
    """Cached REM-locked curves for the treated arm."""
    frame = pd.read_csv(OUT / f"cohort_remlock_{arm}.csv", dtype={"participant": str})
    return _thin_participants_dropped(frame, arm)


def cohort_remlock_controls(arm: str) -> pd.DataFrame:
    """Cached REM-locked curves for the matched control arm.

    Written by `_test_cohort_extract.py --controls-remlock` into a file separate from the
    treated curves, so that adding controls cannot silently pull them into any figure
    whose reader does not filter on `treated`.
    """
    path = OUT / f"cohort_remlock_controls_{arm}.csv"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nrun: paper/_test_cohort_extract.py --arm {arm} "
            "--controls-remlock --skip-states")
    frame = pd.read_csv(path, dtype={"participant": str})
    return _thin_participants_dropped(frame, f"{arm} controls")


def cohort_states(arm: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"cohort_states_{arm}.csv", dtype={"participant": str})


def control_n_eff(states: pd.DataFrame) -> float:
    """Kish effective sample size of the weighted control arm."""
    weights = states[states.treated == 0].overlap_weight.to_numpy(dtype=float)
    return float(weights.sum() ** 2 / (weights**2).sum())


def state_arms(states: pd.DataFrame, state: str) -> dict[int, tuple[float, float]]:
    """Weighted (pre level, pre->post delta) per arm, on complete cases only.

    Restricting to participants with both levels present is what makes the arrow
    length equal step 32's published estimand: that estimator differences within
    participant before weighting, so a participant scored at only one visit must not
    contribute to the pre mean either. Without this the REM DiD reads +4.31 against
    a published +4.22.
    """
    out = {}
    for treated in (1, 0):
        side = states[states.treated == treated]
        pre = side[f"{state}_pre"].to_numpy(dtype=float)
        post = side[f"{state}_post"].to_numpy(dtype=float)
        weights = side.overlap_weight.to_numpy(dtype=float)
        complete = np.isfinite(pre) & np.isfinite(post)
        level, _ = weighted_mean_and_sem(pre[complete], weights[complete])
        delta, _ = weighted_mean_and_sem((post - pre)[complete], weights[complete])
        out[treated] = (level, delta)
    return out


def remlock_profile(curves: pd.DataFrame, visit: str) -> pd.DataFrame:
    """Overlap-weighted cohort mean +/- SEM at each offset, taken over participants."""
    rows = []
    for offset, group in curves[curves.visit == visit].groupby("offset_s"):
        mean, sem = weighted_mean_and_sem(
            group.hr.to_numpy(dtype=float), group.overlap_weight.to_numpy(dtype=float)
        )
        rows.append({"offset_s": offset, "mean": mean, "sem": sem})
    return pd.DataFrame(rows).sort_values("offset_s")


def variant_b(ax: plt.Axes, curves_by_arm: dict[str, pd.DataFrame],
              panel_letter: str = "b") -> None:
    """Cohort-mean REM-locked heart rate in initiators, before vs on drug.

    The participant is the unit of analysis: each contributes one bout-averaged
    curve per visit (`_test_cohort_extract.py`), and the cohort mean is
    overlap-weighted across participants. Averaging bouts directly would let the
    participants with the most REM bouts dominate the mean.
    """
    extremes: list[float] = []
    counts: dict[str, int] = {}
    # Labelled from the solid (primary) arm. The dashed arm's plotted median is 11.6 mo
    # against this arm's 10.8, so the legend's duration describes the solid curve; both
    # arms' distributions are printed by `summarise_cohort` on every run.
    labels = cohort_visit_labels(curves_by_arm[PRIMARY_ARM], PRIMARY_ARM)
    for arm, curves in curves_by_arm.items():
        counts[arm] = curves.participant.nunique()
        is_primary = arm == PRIMARY_ARM
        for visit in ("pre", "post"):
            profile = remlock_profile(curves, visit)
            minutes = profile.offset_s.to_numpy() / 60.0
            centre, spread = profile["mean"].to_numpy(), profile["sem"].to_numpy()
            colour = VISIT_COLOUR[visit]
            if is_primary:
                ax.fill_between(minutes, centre - spread, centre + spread, color=colour,
                                alpha=0.20, linewidth=0, zorder=2)
                ax.plot(minutes, centre, color=colour, linewidth=1.2, zorder=4)
                extremes += [np.nanmin(centre - spread), np.nanmax(centre + spread)]
            else:
                ax.plot(minutes, centre, color=colour, linewidth=0.8,
                        linestyle=(0, (2.5, 1.6)), zorder=3)
                extremes += [np.nanmin(centre), np.nanmax(centre)]

    ax.axvspan(0, LOCK_POST / 60.0, color=GREY, alpha=0.07, linewidth=0, zorder=0)
    ax.axvline(0, color=BLACK, linewidth=0.7, linestyle="--", zorder=5)
    ax.annotate("REM onset", xy=(0, 0.02), xycoords=("data", "axes fraction"),
                xytext=(-3, 0), textcoords="offset points", fontsize=6.4, ha="right",
                va="bottom")
    ax.set_xlabel("Minutes from REM-episode onset")
    ax.set_ylabel("Heart rate (bpm)")
    ax.set_xlim(LOCK_PRE / 60.0, LOCK_POST / 60.0)
    low, high = min(extremes), max(extremes)
    ax.set_ylim(low - 0.14 * (high - low), high + 0.34 * (high - low))
    # Whole-bpm ticks: the default locator lands on 62.5/65.0/67.5 here, and a half
    # beat per minute is below anything the data resolves.
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#ECECEC", linewidth=0.6)
    ax.set_axisbelow(True)

    # Legend below the axes, not inside it: with two visits x two arms there is no
    # interior quadrant that all four curves leave empty.
    n_primary = counts[PRIMARY_ARM]
    handles = [
        Line2D([], [], color=VISIT_COLOUR["pre"], linewidth=1.2,
               label=f"{labels['pre']} — {n_primary} initiators"),
        Line2D([], [], color=VISIT_COLOUR["post"], linewidth=1.2, label=labels["post"]),
    ]
    if DRAW_SECONDARY_ARM:
        handles += [
            Line2D([], [], color=GREY, linewidth=1.2,
                   label=f"adjudicated, n = {counts[PRIMARY_ARM]}"),
            Line2D([], [], color=GREY, linewidth=0.8, linestyle=(0, (2.5, 1.6)),
                   label=f"any dated, n = {counts[SECONDARY_ARM]}"),
        ]
    ax.legend(
        # Single column: the on-drug label carries a median and IQR, too wide to sit
        # beside another entry at 179 mm.
        handles=handles, loc="upper left", frameon=False, fontsize=6.0, handlelength=2.0,
        ncol=1, handletextpad=0.5, borderpad=0.0,
        labelspacing=0.30, bbox_to_anchor=(-0.012, -0.28),
    )
    # n lives in the legend's on-drug entry, not the title: appending it here pushes the
    # title into panel c's label at 179 mm.
    ax.set_title("Cohort mean of every REM episode ≥ 3 min", loc="left", pad=6.0)
    if panel_letter:
        opener.panel_label(ax, panel_letter, x=-0.11, y=1.20)


# Colour carries the ARM and linestyle carries the VISIT in the both-arms panel. Controls
# are grey rather than blue because panel a already uses blue for "before"; a blue control
# curve would make blue mean two different things inside one figure.
CONTROL_COLOUR = opener.GREY
ARM_VISIT_STYLE = {"pre": (0, (2.6, 1.6)), "post": "solid"}


def variant_b_both_arms(ax: plt.Axes, treated: pd.DataFrame, controls: pd.DataFrame,
                        panel_letter: str = "b", show_title: bool = True,
                        treated_colours: dict[str, str] | None = None,
                        control_colours: dict[str, str] | None = None,
                        linestyles: dict[str, str | tuple] | None = None,
                        treated_linestyles: dict[str, str | tuple] | None = None,
                        control_linestyles: dict[str, str | tuple] | None = None,
                        ribbon_alpha: dict[str, float] | None = None) -> None:
    """Cohort-mean REM-locked heart rate in BOTH arms, before and after.

    `variant_b` plots the initiators alone, which cannot distinguish a drug effect from
    both arms drifting between visits -- the reader has to leave the panel for panel c to
    find that out. Drawing the matched controls answers it in place: their two visits sit
    on top of each other (-0.12 bpm over the post-onset window) while the initiators rise
    3.84, and the difference reproduces the published REM DiD to within the difference
    between this window and the whole night.

    The participant is the unit of analysis in both arms: one bout-averaged curve per
    participant per visit, then an overlap-weighted mean across participants.

    The style arguments default to today's published look -- one colour per arm, dash per
    visit -- and exist for the Figure 1 variants. Arm-specific linestyle overrides allow
    visit colour to carry the initiator's before/on-drug distinction while retaining the
    dashed/solid visit distinction for the two grey control curves. `ribbon_alpha` is per
    visit for the same
    reason `variant_a`'s `shade_alpha` is: the SEM ribbon is a tint of its curve's colour,
    so a scheme that gives "before" a pale hue renders that ribbon invisible at the alpha
    that suits a saturated one. Matching the ribbons' rendered weight is what keeps the SEM
    readable in all four curves, and it is the only thing these two numbers do.
    """
    treated_colours = {"pre": ONDRUG, "post": ONDRUG, **(treated_colours or {})}
    control_colours = {"pre": CONTROL_COLOUR, "post": CONTROL_COLOUR,
                       **(control_colours or {})}
    linestyles = {**ARM_VISIT_STYLE, **(linestyles or {})}
    treated_linestyles = {**linestyles, **(treated_linestyles or {})}
    control_linestyles = {**linestyles, **(control_linestyles or {})}
    ribbon_alpha = {"pre": 0.13, "post": 0.13, **(ribbon_alpha or {})}

    extremes: list[float] = []
    for frame, arm_colours, arm_linestyles in (
        (treated, treated_colours, treated_linestyles),
        (controls, control_colours, control_linestyles),
    ):
        for visit in ("pre", "post"):
            colour = arm_colours[visit]
            profile = remlock_profile(frame, visit)
            minutes = profile.offset_s.to_numpy() / 60.0
            centre, spread = profile["mean"].to_numpy(), profile["sem"].to_numpy()
            ax.fill_between(minutes, centre - spread, centre + spread, color=colour,
                            alpha=ribbon_alpha[visit], linewidth=0, zorder=2)
            ax.plot(minutes, centre, color=colour, linewidth=1.2,
                    linestyle=arm_linestyles[visit],
                    zorder=4 if visit == "post" else 3)
            extremes += [np.nanmin(centre - spread), np.nanmax(centre + spread)]

    ax.axvspan(0, LOCK_POST / 60.0, color=GREY, alpha=0.07, linewidth=0, zorder=0)
    ax.axvline(0, color=BLACK, linewidth=0.7, linestyle="--", zorder=5)
    ax.annotate("REM onset", xy=(0, 0.02), xycoords=("data", "axes fraction"),
                xytext=(-3, 0), textcoords="offset points", fontsize=6.4, ha="right",
                va="bottom")
    ax.set_xlabel("Minutes from REM-episode onset")
    ax.set_ylabel("Heart rate (bpm)")
    ax.set_xlim(LOCK_PRE / 60.0, LOCK_POST / 60.0)
    low, high = min(extremes), max(extremes)
    ax.set_ylim(low - 0.10 * (high - low), high + 0.16 * (high - low))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#ECECEC", linewidth=0.6)
    ax.set_axisbelow(True)
    if show_title:
        ax.set_title("Every REM episode ≥ 3 min, both arms", loc="left", pad=6.0)

    n_treated = treated.participant.nunique()
    n_control = controls.participant.nunique()
    ax.legend(handles=[
        Line2D([], [], color=treated_colours["pre"], linewidth=1.2,
               linestyle=treated_linestyles["pre"],
               label=f"initiators, before (n={n_treated})"),
        Line2D([], [], color=treated_colours["post"], linewidth=1.2,
               linestyle=treated_linestyles["post"], label="initiators, on drug"),
        Line2D([], [], color=control_colours["pre"], linewidth=1.2,
               linestyle=control_linestyles["pre"],
               label=f"controls, visit 1 (n={n_control})"),
        Line2D([], [], color=control_colours["post"], linewidth=1.2,
               linestyle=control_linestyles["post"], label="controls, visit 2"),
    ], loc="upper left", frameon=False, fontsize=6.0, handlelength=2.2, ncol=2,
        handletextpad=0.5, borderpad=0.0, columnspacing=1.2, labelspacing=0.30,
        bbox_to_anchor=(-0.012, -0.30))
    if panel_letter:
        opener.panel_label(ax, panel_letter, x=-0.11, y=1.20 if show_title else 1.0)


PAIR_LEVEL = RUN / "outputs" / "relative_scale_sleep" / "pair_level_log_did.csv"
CONTRASTS = RUN / "outputs" / "paper_reanalysis" / "formal_specificity_contrasts.csv"
PRIMARY_CONTRAST = "REM minus NREM heart-rate DiD"


def load_paired_contrast() -> tuple[np.ndarray, pd.Series]:
    """Per-pair REM-minus-non-REM DiD in bpm, and the published contrast row.

    The per-pair mean equals the published `difference` exactly -- verified below rather
    than assumed, because the paper also carries a separate endpoint literally named
    `heart_rate_rem_minus_nrem` whose DiD is +1.334, computed from a per-visit difference
    variable rather than as the contrast of the two endpoint DiDs. Annotating one over a
    cloud whose mean is the other would be wrong by 0.02 bpm and impossible to explain.
    """
    pairs = pd.read_csv(PAIR_LEVEL)
    wide = pairs.pivot_table(index="treated_id", columns="measurement", values="did")
    need = ["heart_rate_mean_during_rem", "heart_rate_mean_during_nrem"]
    common = wide.dropna(subset=need)
    gap = (common[need[0]] - common[need[1]]).to_numpy(dtype=float)

    row = pd.read_csv(CONTRASTS).set_index("contrast").loc[PRIMARY_CONTRAST]
    if len(gap) != int(row.n):
        raise ValueError(f"pair count {len(gap)} does not match published n={int(row.n)}")
    if not np.isclose(gap.mean(), float(row.difference), atol=5e-4):
        raise ValueError(f"per-pair mean {gap.mean():.4f} does not reproduce the "
                         f"published contrast {float(row.difference):.4f}")
    return gap, row


def variant_d_paired_contrast(ax: plt.Axes, panel_letter: str = "d",
                              letter_y: float = 1.16,
                              show_title: bool = True) -> None:
    """The primary state contrast, per matched pair.

    Replaces the text banner this figure used to carry. Same visual grammar as the
    relative-scale panel in the supplement, so the main text reads the absolute scale and
    the supplement reads the ratio, and the two are recognisably the same estimand.
    """
    gap, row = load_paired_contrast()
    rng = np.random.default_rng(20260807)
    jitter = rng.uniform(-0.055, 0.055, size=len(gap))

    ax.axvline(0.0, color=BLACK, linestyle="--", linewidth=0.9, zorder=1)
    ax.scatter(gap, jitter, s=16, facecolor=LIGHT_GREY, edgecolor=GREY, linewidth=0.35,
               alpha=0.85, zorder=2)
    summary_y = 0.24
    ax.plot([float(row.ci_low), float(row.ci_high)], [summary_y, summary_y], color=ONDRUG,
            linewidth=2.4, solid_capstyle="round", zorder=3)
    ax.plot(float(row.difference), summary_y, marker="D", markersize=7.0, color=ONDRUG,
            markeredgecolor="white", markeredgewidth=0.7, zorder=4)
    # One annotation only. The narrow test version of this panel could also carry
    # "null = no state selectivity" and a pair-count note, but at full figure width and
    # this height both land on the dot row; the dashed line at zero and the x label
    # already say what the null is, and n now rides on the estimate.
    ax.annotate(
        f"{float(row.difference):+.2f} bpm "
        f"[{float(row.ci_low):.2f}, {float(row.ci_high):.2f}], "
        f"q = {float(row.fdr_within_contrast_family):.5f}  "
        f"(n = {int(row.n)} pairs)",
        xy=(float(row.difference), summary_y), xytext=(0, 11), textcoords="offset points",
        ha="center", va="bottom", color=ONDRUG, fontsize=6.7, fontweight="bold")

    ax.set_ylim(-0.16, 0.42)
    ax.set_yticks([])
    ax.set_xlabel("REM − non-REM difference-in-differences (bpm)")
    if show_title:
        ax.set_title("Paired state contrast, per matched pair", loc="left", pad=5.0)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if panel_letter:
        opener.panel_label(ax, panel_letter, x=-0.055, y=letter_y)


# Why the pair counts differ by row (41 for REM/non-REM, 44 for whole sleep/wake).
#
# Not a different cohort -- one cohort, per-measurement complete cases. The WatchPAT
# link table is one row per night, exactly 3 nights per visit for every visit
# (18,542 visits x 3 = 55,626 rows, no exceptions). `steps/26` NaNs a night's
# per-state heart rate when that state holds < 300 s (MIN_HEART_RATE_STAGE_SECONDS),
# when heart-rate quality < 50, or when the mean falls outside 30-140 bpm, then
# averages the surviving nights per (participant, research_stage) in `_to_long`.
#
# REM is the state most often too short to clear the 300 s floor: measured over all
# nights, REM heart rate is present on 88.0% and wake on 91.0%, and 1,668 nights have
# a usable wake mean with REM missing while ZERO have the reverse. Every REM-missing
# night is also non-REM-missing (non-REM is derived as total_sleep - total_rem), which
# is why REM and non-REM share n = 41 exactly. So a handful of pairs lose REM but keep
# wake, and 44 -> 41 is that loss. Reporting the larger n on every row would be the
# error; these are honest per-row complete cases.
PRIMARY_1TO1 = {
    "REM": "heart_rate_mean_during_rem",
    "Non-REM": "heart_rate_mean_during_nrem",
    "Whole sleep": "heart_rate_mean_during_sleep",
    "Nocturnal wake": "heart_rate_mean_during_wake",
}


def load_primary_1to1(source: Path | None = None) -> pd.DataFrame:
    """The 1:1 no-reuse matched sleep DiD. Defaults to the PUBLISHED file.

    `source` lets a caller point the panel at a different estimate carrying the same
    columns -- `steps/48`'s unified cohort does, via `paper/_test_fig1_unified.py`. The
    default is unchanged so the canonical figure is untouched.
    """
    path = source or RUN / "outputs" / "paper_reanalysis" / "primary_1to1_results.csv"
    frame = pd.read_csv(path)
    if "modality" in frame.columns:
        frame = frame[frame.modality.isin(("sleep", "sleep_raw"))]
    return frame.drop_duplicates("measurement").set_index("measurement")


def load_state_contrast() -> dict:
    """The formal REM-minus-non-REM contrast, absolute and relative scale.

    This is the paper's actual novel claim -- the audit's Finding 4 reframes the
    contribution from "GLP-1 raises nocturnal heart rate" (established) to "the
    chronotropic effect concentrates in REM" (novel) -- and an earlier version of this
    layout dropped it entirely. Restored as a callout.

    Two numbers, deliberately. The absolute contrast is the manuscript's headline. The
    relative-scale ratio is the proportional-scaling null test: REM has a higher baseline,
    so one multiplicative rise applied to it would widen the absolute gap with no
    REM-specific mechanism. 1.0 is that null. Carrying both closes open item 1 of
    REM_HR_SCALE_ROBUSTNESS.md for the figure -- panel c was previously descriptive only.

    Read from source rather than hardcoded: the absolute contrast is the pairwise
    difference of separately-estimated DiDs (+1.3583), which is NOT the same as the
    per-visit derived `heart_rate_rem_minus_nrem` endpoint (+1.3343) drawn from
    primary_1to1_results.csv. Mixing them is an easy and invisible error.
    """
    absolute = pd.read_csv(
        RUN / "outputs" / "paper_reanalysis" / "formal_specificity_contrasts.csv"
    )
    absolute = absolute[absolute.contrast == "REM minus NREM heart-rate DiD"].iloc[0]
    out = {"n": int(absolute.n), "abs_did": float(absolute.difference),
           "abs_lo": float(absolute.ci_low), "abs_hi": float(absolute.ci_high),
           "abs_q": float(absolute.fdr_within_contrast_family)}
    relative_path = RUN / "outputs" / "relative_scale_sleep" / "relative_scale_contrasts.csv"
    if relative_path.exists():
        rel = pd.read_csv(relative_path)
        rel = rel[rel.contrast.str.startswith("REM minus pooled non-REM")]
        if not rel.empty:
            rel = rel.iloc[0]
            out.update({"ratio": float(rel.excess_rise_ratio),
                        "ratio_lo": float(rel.excess_rise_ratio_ci_low),
                        "ratio_hi": float(rel.excess_rise_ratio_ci_high),
                        "ratio_q": float(rel.fdr_within_contrast_family)})
    return out


def significance_stars(q: float) -> str:
    """Conventional star coding, on the FDR-adjusted q rather than the raw p.

    The paper controls the state-specificity family, so q is the quantity a reader
    should be shown; using p here would overstate what survives multiplicity.
    """
    for threshold, stars in ((0.001, "***"), (0.01, "**"), (0.05, "*")):
        if q < threshold:
            return stars
    return "n.s."


def _state_contrast_bracket(ax: plt.Axes, positions, rows, x: float,
                            span: float, q: float | None = None) -> None:
    """Bracket the REM and non-REM rows with the paired contrast's significance.

    The two rows carry marginal intervals that overlap heavily, and a reader trained on
    forest plots will read overlapping intervals as "not different". They are not the
    test: the pairs are matched, so REM-minus-non-REM is estimated on the paired
    differences and is far tighter than either margin. The bracket puts that test on the
    panel instead of leaving it to be inferred from two intervals that cannot support it.
    """
    if q is None:
        q = float(pd.read_csv(CONTRASTS).set_index("contrast")
                  .loc[PRIMARY_CONTRAST].fdr_within_contrast_family)
    y_top = positions[rows.index("REM")]
    y_bottom = positions[rows.index("Non-REM")]
    tick = 0.035 * span
    ax.plot([x - tick, x, x, x - tick], [y_top, y_top, y_bottom, y_bottom],
            color=BLACK, linewidth=0.7, solid_capstyle="butt", zorder=6,
            clip_on=False)
    ax.annotate(significance_stars(q),
                (x + 0.4 * tick, (y_top + y_bottom) / 2), ha="left", va="center",
                fontsize=7.0, fontweight="bold", color=BLACK, annotation_clip=False)


def variant_c(ax: plt.Axes, panel_letter: str = "c", letter_x: float = -0.34,
              show_title: bool = True,
              treated_colour: str = ONDRUG, control_colour: str = GREY,
              source: Path | None = None, contrast_q: float | None = None) -> None:
    """The paper's primary estimator: 1:1 no-reuse matched pairs, both arms shown.

    Source is `outputs/paper_reanalysis/primary_1to1_results.csv` -- the same numbers
    the published Figure 1c reports (REM +6.97, n = 41 pairs), on the WatchPAT summary
    fields. Unlike the raw-channel cohort in panel b, 1:1 matching passes both design
    gates here: 44/48 treated retained (91.7%), max |SMD| 0.089.

    Plotted as each arm's own pre->post *change* rather than as absolute levels: the
    published file carries deltas, not baseline levels, and the DiD is then literally
    the horizontal gap between the two markers -- nothing has to be asserted.

    Each mark here is a CHANGE, so it belongs to an arm and to no single visit. In a scheme
    where lightness carries the visit, these two take their arm's post-visit (darker) tone:
    the marker sits at the end of the arrow the panel is drawing, and a light marker would
    invite reading it as the before level, which is exactly what this panel does not plot.
    """
    published = load_primary_1to1(source)
    rows = list(PRIMARY_1TO1)
    positions = np.arange(len(rows))[::-1]

    # The x extent is set by the INTERVAL bounds, not the point estimates. Each arm
    # gained a bootstrap interval on 2026-09-06; sizing the axis off the points left
    # the REM initiator whisker (to +8.21) running under the DiD annotation column.
    bounds = [
        float(published.loc[column, field])
        for column in PRIMARY_1TO1.values()
        for field in ("treated_delta_ci_low", "treated_delta_ci_high",
                      "control_delta_ci_low", "control_delta_ci_high")
    ]
    low, high = min(bounds), max(bounds)
    span = high - low
    bracket_x = high + 0.06 * span
    label_x = high + 0.20 * span
    ax.set_xlim(low - 0.06 * span, high + 0.98 * span)
    # Ticks stop at the data. The right third of this panel is an annotation column, not
    # plotting area, and letting the locator run across it drew ticks out to 20 bpm in a
    # panel whose largest interval bound is 8.2 -- an axis implying data that is not there.
    ax.set_xticks(np.arange(-2.5, high + 0.1, 2.5))
    # Top margin holds the DiD column header; bottom is tight since the legend is
    # outside the frame now.
    ax.set_ylim(-0.55, len(rows) - 0.28)
    ax.axvline(0, color=BLACK, linewidth=0.7, linestyle="--", zorder=1)

    for y, label in zip(positions, rows):
        row = published.loc[PRIMARY_1TO1[label]]
        treated, control = float(row.treated_delta), float(row.control_delta)
        # The connector *is* the DiD: control change at one end, initiator change at
        # the other. Reading its length off the axis reproduces the annotated number.
        ax.plot([control, treated], [y, y], color=LIGHT_GREY, linewidth=2.4,
                solid_capstyle="round", zorder=2)
        # Each arm's own 95% bootstrap interval, drawn over the connector rather than
        # beside it: these are two intervals on one row, not one interval on the DiD.
        # The DiD's own interval is the annotation at the right, and it is NOT the
        # union or the overlap of these two -- the pairs are matched, so the contrast
        # is estimated on the paired differences.
        for value, field_lo, field_hi, colour in (
            (control, "control_delta_ci_low", "control_delta_ci_high", control_colour),
            (treated, "treated_delta_ci_low", "treated_delta_ci_high", treated_colour),
        ):
            ax.errorbar(value, y,
                        xerr=[[value - float(row[field_lo])],
                              [float(row[field_hi]) - value]],
                        fmt="none", ecolor=colour, elinewidth=0.9, capsize=1.9,
                        capthick=0.9, zorder=3)
        ax.plot([control], [y], marker="o", color=control_colour, markeredgecolor="white",
                markeredgewidth=0.6, markersize=5.0, zorder=4)
        highlight = label == "REM"
        ax.plot([treated], [y], marker="o", color=treated_colour, markeredgecolor="white",
                markeredgewidth=0.6, markersize=6.4 if highlight else 5.4, zorder=5)
        ax.annotate(f"{float(row.effect):+.2f}  [{float(row.ci_low):.2f}, "
                    f"{float(row.ci_high):.2f}]", (label_x, y), ha="left", va="center",
                    fontsize=6.0, fontweight="bold" if highlight else "normal",
                    color=treated_colour if highlight else GREY)

    _state_contrast_bracket(ax, positions, rows, bracket_x, span, contrast_q)

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{label}\n({int(published.loc[PRIMARY_1TO1[label], 'n_pairs'])} pairs)"
         for label in rows],
        fontsize=6.2, linespacing=1.15,
    )
    ax.get_yticklabels()[0].set_color(treated_colour)
    ax.get_yticklabels()[0].set_fontweight("bold")
    # Column header for the annotation stack. Inside the frame at the top -- above it
    # the panel title occupies the same band -- and the y-limit reserves the row.
    ax.annotate("DiD (bpm) [95% CI]", (label_x, len(rows) - 0.52), ha="left", va="center",
                fontsize=6.0, color=GREY)
    ax.set_xlabel("Change in heart rate, after − before (bpm)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.6)
    ax.set_axisbelow(True)
    # Below the axes, matching panel b, so neither legend sits on the data.
    ax.legend(
        handles=[Line2D([], [], color="none", marker="o", markersize=4.6,
                        markerfacecolor=colour, markeredgecolor="white", label=text)
                 for colour, text in ((treated_colour, "semaglutide initiators"),
                                      (control_colour, "matched controls"))],
        loc="upper left", frameon=False, fontsize=6.0, handlelength=1.0,
        borderpad=0.0, labelspacing=0.32, handletextpad=0.4,
        bbox_to_anchor=(-0.012, -0.30),
    )
    if show_title:
        ax.set_title("Matched 1:1 pairs, WatchPAT summary", loc="left", pad=6.0)
    # Two shadowing traps here, both hit during development. The row loop rebinds `label`
    # (a parameter named `label` rendered "Wake in study"), and `label_x` is already a local
    # holding the DiD annotation column in DATA coords (a parameter named `label_x` put the
    # panel letter at x~18 in axes fractions and wrecked the tight bbox). Hence `letter_x`.
    if panel_letter:
        opener.panel_label(ax, panel_letter, x=letter_x, y=1.20 if show_title else 1.0)


def summarise(frame: pd.DataFrame, nights: dict[str, str], participant: str) -> None:
    """Print what the figure asserts, so the numbers can be checked by hand."""
    print("\nvisit-level stage means over all nights (bpm):")
    for visit in ("pre", "post"):
        visit_frame = frame[frame.visit == visit]
        rem = visit_frame.hr[visit_frame.stage_code == REM_CODE]
        nrem = visit_frame.hr[visit_frame.stage_code.isin([13, 15])]
        print(f"  {visit:>4}  REM {rem.mean():5.1f} ({rem.notna().sum() / 60:4.0f} min)"
              f"   NREM {nrem.mean():5.1f} ({nrem.notna().sum() / 60:4.0f} min)")
    deltas = {}
    for state, codes in (("rem", [REM_CODE]), ("nrem", [13, 15])):
        means = [frame.hr[(frame.visit == v) & frame.stage_code.isin(codes)].mean()
                 for v in ("pre", "post")]
        deltas[state] = means[1] - means[0]
    print(f"  d_rem {deltas['rem']:+.1f}   d_nrem {deltas['nrem']:+.1f}   "
          f"d_rem_minus_nrem {deltas['rem'] - deltas['nrem']:+.1f}")
    panel = pd.read_csv(RUN / "outputs" / "raw_sleep_glp_cohort" / "panel.csv",
                        dtype={"participant": str})
    arm = panel[panel.arm == PRIMARY_ARM]
    own = arm[arm.participant == participant]
    if not own.empty:
        pc_r = (arm.d_rem < float(own.d_rem.iloc[0])).mean() * 100
        pc_c = (arm.d_rem_minus_nrem < float(own.d_rem_minus_nrem.iloc[0])).mean() * 100
        print(f"  rank in arm: d_rem {float(own.d_rem.iloc[0]):+.2f} "
              f"({ordinal(round(pc_r))} pct)   contrast "
              f"{float(own.d_rem_minus_nrem.iloc[0]):+.2f} ({ordinal(round(pc_c))} pct)")
    print("representative night pair drawn in panel a:")
    for visit, night in nights.items():
        night_frame = frame[(frame.visit == visit) & (frame.night == night)]
        rem = night_frame.hr[night_frame.stage_code == REM_CODE]
        print(f"  {visit:>4} {night}  REM {rem.mean():5.1f} ({rem.notna().sum() / 60:.0f} min)")


def summarise_cohort(states: pd.DataFrame, curves_by_arm: dict[str, pd.DataFrame]) -> None:
    """Print the cohort numbers panels b and c draw, for checking by hand."""
    published = pd.read_csv(
        RUN / "outputs" / "raw_sleep_matched_did" / "matched_did_estimates.csv"
    )
    published = published[(published.arm == PRIMARY_ARM) & (published.estimator == "overlap")]
    print(f"\npanel b cohort ({PRIMARY_ARM}), raw channels, overlap-weighted:")
    for state in ("rem", "nrem", "light", "deep", "wake"):
        arms = state_arms(states, state)
        did = arms[1][1] - arms[0][1]
        row = published[published.metric == f"d_{state}"]
        reference = float(row.did.iloc[0]) if not row.empty else np.nan
        flag = "ok" if abs(did - reference) < 0.02 else "MISMATCH"
        print(f"  {state:<6} initiators {arms[1][0]:5.2f} -> {arms[1][0] + arms[1][1]:5.2f} "
              f"({arms[1][1]:+.2f})   controls {arms[0][0]:5.2f} -> "
              f"{arms[0][0] + arms[0][1]:5.2f} ({arms[0][1]:+.2f})   "
              f"DiD {did:+.2f} vs published {reference:+.2f}  {flag}")

    print("\npanel c, the paper's primary 1:1 matched DiD as published:")
    published = load_primary_1to1()
    for label, column in PRIMARY_1TO1.items():
        row = published.loc[column]
        print(f"  {label:<14} {int(row.n_pairs):>3} pairs  initiators {row.treated_delta:+.2f}"
              f"   controls {row.control_delta:+.2f}   DiD {row.effect:+.2f} "
              f"[{row.ci_low:.2f}, {row.ci_high:.2f}]   gate "
              f"{'pass' if row.design_gate_pass else 'FAIL'}")

    print("\npanel b exposure duration of the plotted cohort (months on drug at post):")
    for arm, curves in curves_by_arm.items():
        exposure = pd.read_csv(OUT / f"cohort_exposure_{arm}.csv", dtype={"participant": str})
        months = exposure[exposure.participant.isin(set(curves.participant))]\
            .months_on_drug_at_post
        print(f"  {arm:<13} median {months.median():5.1f}  IQR {months.quantile(.25):.1f}-"
              f"{months.quantile(.75):.1f}  range {months.min():.1f}-{months.max():.1f}")

    print("\npanel b REM-locked cohort means (bpm at -5 min / at REM onset / at +10 min):")
    for arm, curves in curves_by_arm.items():
        bouts = curves.groupby(["participant", "visit"]).n_bouts.first().unstack()
        for visit in ("pre", "post"):
            profile = remlock_profile(curves, visit).set_index("offset_s")
            picks = [profile["mean"].iloc[0], profile["mean"].iloc[len(profile) // 3],
                     profile["mean"].iloc[-1]]
            print(f"  {arm:<13} {visit:>4} (n = {curves.participant.nunique():>3}, "
                  f"episodes {int(bouts[visit].min())}-{int(bouts[visit].max())})  "
                  + "  ".join(f"{value:5.2f}" for value in picks))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("participant", nargs="?", default=DEFAULT_PARTICIPANT,
                        help="must already be cached by _test_panela_extract.py")
    parser.add_argument("--pre-night", help="override panel a's before night, e.g. night_0")
    parser.add_argument("--post-night", help="override panel a's on-drug night")
    args = parser.parse_args()
    participant = resolve_participant(args.participant)

    override = {visit: night for visit, night in
                (("pre", args.pre_night), ("post", args.post_night)) if night}

    opener.configure_style()
    frame = load_nights(participant)
    labels = visit_labels(participant)
    nights = representative_nights(frame, override)
    summarise(frame, nights, participant)

    arms = (PRIMARY_ARM, SECONDARY_ARM) if DRAW_SECONDARY_ARM else (PRIMARY_ARM,)
    curves_by_arm = {arm: cohort_remlock(arm) for arm in arms}
    summarise_cohort(cohort_states(PRIMARY_ARM), curves_by_arm)

    # Taller than the 5.2 used while both legends sat inside their axes: moving them
    # below the frames needs real space under the x-labels.
    fig = plt.figure(figsize=(7.2, 6.45))
    # bottom= reserves a band under the axes for the two below-axes legends AND the
    # full-width contrast banner beneath them; without it the banner lands on the legends.
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.62], hspace=0.62,
                             bottom=0.205)
    top = outer[0].subgridspec(3, 1, height_ratios=[0.42, 0.42, 1.0], hspace=0.12)
    ax_pre = fig.add_subplot(top[0])
    ax_post = fig.add_subplot(top[1], sharex=ax_pre)
    ax_hr = fig.add_subplot(top[2], sharex=ax_pre)
    variant_a(ax_pre, ax_post, ax_hr, frame, labels, nights,
              exemplar_descriptor(participant))

    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 0.86], wspace=0.42)
    variant_b(fig.add_subplot(bottom[0]), curves_by_arm)
    variant_c(fig.add_subplot(bottom[1]))

    # The novel claim, spanning the full width. It belongs to the figure rather than to
    # panel c: at 179 mm panel c is only ~2.6 in wide and the text overruns it, and the
    # audit's Finding 4 makes this the contribution rather than a panel-c footnote.
    # Absolute contrast plus the proportional-scaling null (1.0 = pure scaling), which is
    # what gives the panel quantitative backing instead of being descriptive.
    contrast = load_state_contrast()
    # Two lines: one 130-character line pushes the tight bbox past 179 mm and
    # fit_to_width cannot shrink text, so it silently widens the canvas instead.
    banner = (f"REM − non-REM = {contrast['abs_did']:+.2f} bpm "
              f"[{contrast['abs_lo']:.2f}, {contrast['abs_hi']:.2f}], "
              f"q = {contrast['abs_q']:.5f}   (n = {contrast['n']} pairs)")
    if "ratio" in contrast:
        banner += (f"\nnot proportional scaling: {contrast['ratio']:.3f}× "
                   f"[{contrast['ratio_lo']:.3f}, {contrast['ratio_hi']:.3f}], "
                   f"q = {contrast['ratio_q']:.4f}")
    fig.text(0.5, 0.014, banner, ha="center", va="bottom", fontsize=6.4,
             fontweight="bold", color=ONDRUG, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.38", facecolor="#FDF1E7",
                       edgecolor=ONDRUG, linewidth=0.7))

    OUT.mkdir(parents=True, exist_ok=True)
    width = opener.fit_to_width(fig)
    if abs(width * 25.4 - opener.TARGET_WIDTH_MM) > 1.0:
        print(f"  WARNING: {width * 25.4:.0f} mm, not {opener.TARGET_WIDTH_MM:.0f} -- "
              f"a title or an out-of-axes label is wider than the target and "
              f"fit_to_width cannot shrink text; shorten it.")
    # Overrides get their own filename so a comparison never clobbers the default.
    suffix = "".join(f"_{visit}{night.split('_')[-1]}" for visit, night in override.items())
    stem = OUT / f"TEST_fig1a_{args.participant}{suffix}"  # label, not the key
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=opener.PAD_IN)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=opener.PAD_IN)
    plt.close(fig)
    print(f"\nwrote {stem}.png / .pdf  ({width * 25.4:.0f} mm wide)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
