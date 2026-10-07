"""Historical Figure 1 composer retained for layout helpers and provenance.

The canonical clean-cohort entry point is now `build_fig1_unified.py`. This module still owns
the shared four-panel layout and can reproduce the superseded visit-transition figure, but it
refuses to overwrite the final asset unless `--force-legacy-final` is supplied.
The v1 output is backed up under `paper/figures/_pre_v2_backup/`; v1's builder
`build_opener_figure.py` is left in place, both for rollback and because this file and the
test scripts import its style constants and helpers.

Panels, as of 2026-09-04
------------------------

    a  DESIGN SCHEMATIC. The illustration under `paper/figures/`. It opens the figure so
       the matched before/after design and the covariate set are established before any
       estimate is read. Replaced 2026-09-07; see SCHEMATIC for what changed and why it
       is drawn at SCHEMATIC_WIDTH_FRACTION rather than full width.
    b  ONE PARTICIPANT, TWO NIGHTS. The same participant's before and on-drug nights
       overlaid, so the panel carries the phenomenon rather than only the method.
    c  COHORT REM-LOCK. The exemplar in panel b is one person; panel c is every REM bout
       >= 3 min from every night of the adjudicated cohort, aligned on bout onset. This is
       the alignment that actually puts REM on top of REM.
    d  PRIMARY DiD. Each ARM's own pre->post change, so the difference-in-differences is
       the visible gap between the control and initiator markers rather than a single
       number the reader must trust. Per-row pair counts are on the labels because they
       genuinely differ.

The per-matched-pair contrast strip that was panel d until 2026-09-04 is now panel d of
`figS8_scale_robustness`, drawn there by `build_figS8_scale_robustness.py` from this
file's `variant_d_paired_contrast`. It plots the estimand panel d already carries as its
top row, so as a fourth panel here it restated the figure's headline directly underneath
it; in figS2 it sits beside the same 41 pairs on the relative scale, which is where the
comparison is worth making.

Three honest costs, all of which the captions must state
--------------------------------------------------------
1. RESOLVED 2026-09-07 by the new artwork, recorded so it is not reintroduced. The old
   schematic put a lock icon on all six binary covariates, i.e. claimed all six were
   exact-blocked. For the sleep cohort the exact blocking is pre_stage + post_stage +
   gender + diabetic + on_lipid_lowering + smoking; antihypertensive therapy and
   "visceral fat not measured" are propensity/Mahalanobis-balanced only. The new art
   carries no lock icons and asserts only "matched", which is true of everything it
   lists. Two things about its list are deliberate and should not be "corrected":
   its ten covariate icons are exactly Figure 2b's ten rows (BALANCE_BLOCKS), and the
   eleventh propensity covariate `vat_missing` is absent from both -- it is a
   missingness indicator, not a substantive confounder. "Same pre and post study visits"
   is the pre_stage + post_stage exact block, not a propensity term, which is why it
   sits in its own card and not in Figure 2b. The distinction between exact blocking
   and propensity balancing is stated in figS1's covariate key and Methods, not here.
2. In the clean-unified Figure 1 caller, panel b's P02 exemplar is an 87th-percentile
   responder in the eligible-candidate arm. It is REM-selective, therapeutically exposed,
   phenotypically credible and legible (see `_test_fig1_unified.py`).
   It IS a selection and a reviewer may probe it. The panel title used to state the
   percentile; titles were dropped from b/c/d on 2026-09-04, so **the caption is now the
   only disclosure of that selection and must keep saying so** -- see the show_title
   argument on `variant_a`. Panels c and d are cohort estimates and carry the evidence.
3. Panels c and d are estimated on DIFFERENT cohorts. Panel c needs per-second channels,
   available only for the raw-channel cohort where 1:1 matching fails its design gates
   (48.6% retention, max |SMD| 0.42), so panel c is overlap-weighted, n = 33. Panel d is
   the summary-field cohort where 1:1 passes (91.7%, max |SMD| 0.089). The same effect
   reads +6.97 in panel d and +4.27 in panel c's cohort. **Neither number may be quoted as
   the other's, and the caption must say so.**

Usage
-----
    PY=/net/mraid20/export/jasmine/david/anaconda3/bin/python3
    $PY paper/build_fig1_v2.py --dry-run
    $PY paper/build_fig1_v2.py --force-legacy-final   # deliberate rollback only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

RUN = Path(__file__).resolve().parents[1]
FINAL = RUN / "paper" / "figures" / "final"
SCRATCH = RUN / "outputs" / "_test_figures"
STEM = "fig1_sleep_phenotyping_and_finding"

# The hand-drawn design schematic that opens the figure. It is a raster, so it is
# placed at the width it was authored for and never upscaled.
SCHEMATIC = RUN / "paper" / "figures" / "Matched Sleep Study Infographic.png"
# 2026-10-04: replaced the 2026-09-07 art with one regenerated from
# `FIG1A_SCHEMATIC_PROMPT.md`, so its labels agree with Methods: semaglutide (not GLP-1)
# initiators, matched covariates (not confounders), matching on the same pre and post
# study visits (not calendar time), the WatchPAT's three sensors, and "apnea". It carries
# the reviewed 94-pair count, so the white patch that painted over the old art's "n = 44"
# is gone. The rows keep ~21 px of white above and below the card borders (content rows
# 65-863 of 887).
SCHEMATIC_ROWS = slice(44, 885)
# The image model cannot reproduce an exact hex, so the art's two initiator colours,
# measured here, are pulled onto panel b's before and on-drug colours when the raster is
# loaded; the source file stays as generated. Only the left card (columns 0-1149) holds
# arm colours: the right card's pastel fills are left alone.
SCHEMATIC_ARM_COLOURS = {"pre": "#018EA1", "post": "#E46302"}
SCHEMATIC_LEFT_CARD = slice(0, 1150)
# The knob to turn if a later crop is taller: it scales width, and the axes box takes the
# raster's aspect, so height follows and nothing stretches.
SCHEMATIC_WIDTH_FRACTION = 1.0

# The data panels' gridspecs are given these bounds explicitly. Matplotlib's defaults
# are left=0.125, right=0.9, and the schematic axes spans the full canvas 0.0-1.0, so
# on the default margins panels b, c and d printed inset by 12 mm on the left and 10 mm
# on the right of panel a -- they read as squashed inside a wider figure. LEFT is the
# room the exemplar's y label, its rotated visit labels and the panel letters need.
LEFT, RIGHT = 0.085, 0.995

# Figure x shared by the a, b and c panel letters. Slightly negative on purpose: the
# exemplar's rotated visit labels sit at ax_pre x=-0.105, which is figure x=-0.011, and
# a panel letter has to be outside everything it labels. bbox_inches="tight" absorbs it.
LETTER_X = -0.018
# Lift above the band top. The letters use va="top", so 0 would hang them from the axes
# top edge; a small lift separates them from the topmost tick label and title-less axes.
LETTER_LIFT = 0.008


def recolour(image, source: str, target: str, columns: slice, tolerance: float = 0.08):
    """Move `source`, and its anti-aliased blends with white, onto `target`, in place.

    A pixel on the white-to-source line is white - w * (white - source) for a weight w in
    (0, 1]; it becomes white - w * (white - target), so edges keep their antialiasing.
    Black outlines, grey marks and text lie off that line and are left alone.
    """
    region = image[:, columns, :3]
    source_depth = 1 - np.array(to_rgb(source))
    target_depth = 1 - np.array(to_rgb(target))
    pixel_depth = 1 - region
    weight = np.clip(pixel_depth @ source_depth / (source_depth @ source_depth), 0, 1)
    off_line = np.linalg.norm(pixel_depth - weight[..., None] * source_depth, axis=-1)
    on_line = (off_line < tolerance) & (weight > 0.05)
    region[on_line] = 1 - weight[on_line][:, None] * target_depth


def load_schematic(initiator_colours: dict[str, str]):
    """The schematic, trimmed to SCHEMATIC_ROWS, its initiator arm in panel b's colours.

    Its panel letter is drawn by this script.
    """
    image = mpimg.imread(SCHEMATIC)[SCHEMATIC_ROWS].copy()
    for visit, source in SCHEMATIC_ARM_COLOURS.items():
        recolour(image, source, initiator_colours[visit], SCHEMATIC_LEFT_CARD)
    return image


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RUN / "paper" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = _load("opener", "build_opener_figure.py")          # style, helpers, design schematic
alt = _load("fig1_alt", "_test_fig1_alt.py")            # the new panels


# The cohort row's gutter and the c:d width split. `wspace` is matplotlib's own unit --
# a fraction of the AVERAGE axes width -- so at 0.42 the gutter is 1.07 in on a 6.75 in
# canvas, of which panel d's two-line y tick labels ("Wake in study / (44 pairs)") need
# only 0.70 in. The remaining 0.36 in is empty, and because `fit_to_width` scales the
# canvas until the tight bbox hits 179 mm, every inch of it is an inch the two panels
# do not get. `_test_fig1_v3.py` closes it; the defaults here are the published values.
COHORT_WSPACE = 0.42
COHORT_WIDTH_RATIOS = (1.0, 0.86)


def build_figure(participant: str,
                 *,
                 before_colour: str | None = None,
                 palette: dict[tuple[str, str], str] | None = None,
                 visit_linestyle: dict[str, str | tuple] | None = None,
                 treated_linestyles: dict[str, str | tuple] | None = None,
                 control_linestyles: dict[str, str | tuple] | None = None,
                 shade_alpha: dict[str, float] | None = None,
                 ribbon_alpha: dict[str, float] | None = None,
                 mean_linestyle: str | tuple = (0, (4, 2)),
                 cohort_curves: tuple | None = None,
                 did_source: Path | None = None,
                 contrast_q: float | None = None,
                 cohort_wspace: float = COHORT_WSPACE,
                 cohort_width_ratios: tuple[float, float] = COHORT_WIDTH_RATIOS):
    """Assemble the four panels and return the figure, unsaved.

    Every keyword defaults to the published look, so calling this with only a participant
    reproduces `paper/figures/final/fig1_sleep_phenotyping_and_finding` byte for byte.
    They exist for the test scripts in this folder.

    `before_colour` is the narrow one and the only one on the adopted path: it replaces the
    blue that panel b gives the before night, and nothing else. Through
    `_test_fig1_alt.VISIT_COLOUR["pre"]` that single value colours six marks -- the before
    hypnogram's REM bands and REM y-tick, the rotated "Before" label, the REM-bout ribbon,
    the heart-rate trace, and the night-mean rule with its value label -- so the swap is one
    argument rather than six edits. Panels c and d contain no blue and do not move.
    `_test_before_colour.py` is the chooser.

    `cohort_curves` and `did_source` inject the reviewed unified-cohort data into panels c
    and d. When omitted, the historical builder remains reproducible from its old inputs.

    `palette` is the wider one, and is keyed by (arm, visit) over arms "initiator" and "control" and visits
    "pre" and "post", and is the one place a scheme is written down. Passing it makes all
    three data panels encode the same thing the same way, which today's figure does not:
    panel b separates the two nights by HUE (blue before, vermillion on drug) while panel c
    separates them by DASH within one hue per arm, so "before" has two different marks
    depending on which panel the reader is in. Leaving it None keeps that split, because
    changing it is a decision about the published figure and not a default.

    Panel d takes only the two post-visit entries: every mark in it is a pre->post CHANGE,
    which belongs to an arm and to no single visit.
    """
    if before_colour and palette:
        raise ValueError("before_colour and palette both set the before night; pick one")
    if palette is not None:
        missing = {(arm, visit) for arm in ("initiator", "control")
                   for visit in ("pre", "post")} - set(palette)
        if missing:
            raise ValueError(f"palette is missing {sorted(missing)}")
    # Panel b's two nights, and -- only under a palette -- panels c and d's arms. A bare
    # `before_colour` must NOT reach the cohort panels: they carry no blue to replace, and
    # feeding it to panel c would repaint the "initiators, before" curve, which is exactly
    # the cross-panel repaint this argument exists to avoid.
    # Named for the PANEL, not the arm: `exemplar` alone already belongs to the
    # gridspec below, and shadowing it here handed variant_a a GridSpec to unpack.
    exemplar_colours = ({v: palette[("initiator", v)] for v in ("pre", "post")} if palette
                        else {"pre": before_colour} if before_colour
                        else None)
    initiator = ({v: palette[("initiator", v)] for v in ("pre", "post")}
                 if palette else None)
    control = ({v: palette[("control", v)] for v in ("pre", "post")}
               if palette else None)
    v1.configure_style()
    frame = alt.load_nights(participant)
    labels = alt.visit_labels(participant)
    nights = alt.representative_nights(frame)
    alt.summarise(frame, nights, participant)

    if cohort_curves is None:
        treated = alt.cohort_remlock(alt.PRIMARY_ARM)
        controls = alt.cohort_remlock_controls(alt.PRIMARY_ARM)
        alt.summarise_cohort(alt.cohort_states(alt.PRIMARY_ARM), {alt.PRIMARY_ARM: treated})
    else:
        treated, controls = cohort_curves

    # The layout is authored in INCHES measured down from the top, then converted to
    # figure fractions, because the schematic's height is not free: it is a raster with
    # a fixed aspect ratio, drawn at full figure width, so its band must be exactly
    # width/aspect tall or imshow letterboxes it inside a taller axes.
    schematic = load_schematic({"pre": alt.BEFORE, "post": alt.ONDRUG} | (exemplar_colours or {}))
    figure_width = 7.2
    schematic_width = figure_width * SCHEMATIC_WIDTH_FRACTION
    schematic_height = schematic_width * schematic.shape[0] / schematic.shape[1]
    bands = {
        # The panel letters hang from their axes tops (show_title=False sets y=1.0 with
        # va="top"), so no band has to reserve height for a letter -- which is what the
        # 0.44 and 0.85 here were doing. What is left is separation and the exemplar's
        # own x label.
        "gap_below_schematic": 0.16,
        "exemplar": 2.60,   # three stacked axes: two hypnograms plus the heart-rate trace
        "gap_below_exemplar": 0.52,  # the exemplar's x label plus a hair of separation
        "cohort_row": 1.80,
        "bottom_reserve": 0.60,  # the two below-axes legends of the cohort row
    }
    figure_height = schematic_height + sum(bands.values())
    fig = plt.figure(figsize=(figure_width, figure_height))

    def band(top_inches: float, height_inches: float) -> tuple[float, float]:
        """(top, bottom) in figure fractions for a band placed `top_inches` from the top."""
        return (
            1 - top_inches / figure_height,
            1 - (top_inches + height_inches) / figure_height,
        )

    # Panel letters are drawn in FIGURE coordinates, not axes fractions, and the panel
    # builders are called with panel_letter="" so they draw none of their own. Axes-
    # relative letters cannot line up here: the three left-column panels sit in axes of
    # three different widths (the schematic spans the whole canvas; the other two are
    # inset by LEFT), so one shared axes-fraction x lands at three different figure x.
    # a, b and c are therefore flush at one figure x; d is in the right column and takes
    # its own, read off its axes box.
    def figure_letter(text: str, x: float, y: float) -> None:
        fig.text(x, y, text, fontsize=v1.PANEL_LABEL_PT, fontweight="bold",
                 va="top", ha="left")

    # ---- a: design schematic ------------------------------------------------
    # Centred at SCHEMATIC_WIDTH_FRACTION of the canvas, no axes decoration, so the
    # raster's own margins are the panel's margins. The axes box is given the raster's
    # exact aspect so imshow neither letterboxes it nor stretches it.
    ax_schematic = fig.add_axes([(1 - SCHEMATIC_WIDTH_FRACTION) / 2,
                                 1 - schematic_height / figure_height,
                                 SCHEMATIC_WIDTH_FRACTION,
                                 schematic_height / figure_height])
    ax_schematic.imshow(schematic, interpolation="antialiased")
    ax_schematic.set_axis_off()
    figure_letter("a", LETTER_X, 1.0)

    # ---- b: exemplar night pair --------------------------------------------
    cursor = schematic_height + bands["gap_below_schematic"]
    top, bottom = band(cursor, bands["exemplar"])
    exemplar = fig.add_gridspec(3, 1, height_ratios=[0.42, 0.42, 1.0], hspace=0.12,
                                top=top, bottom=bottom, left=LEFT, right=RIGHT)
    ax_pre = fig.add_subplot(exemplar[0])
    ax_post = fig.add_subplot(exemplar[1], sharex=ax_pre)
    ax_hr = fig.add_subplot(exemplar[2], sharex=ax_pre)
    alt.variant_a(ax_pre, ax_post, ax_hr, frame, labels, nights,
                  alt.exemplar_descriptor(participant), panel_letter="",
                  show_title=False, colours=exemplar_colours, linestyles=visit_linestyle,
                  shade_alpha=shade_alpha, mean_linestyle=mean_linestyle)
    figure_letter("b", LETTER_X, top + LETTER_LIFT)

    # ---- c, d: cohort REM-lock and the primary DiD -------------------------
    cursor += bands["exemplar"] + bands["gap_below_exemplar"]
    top, bottom = band(cursor, bands["cohort_row"])
    cohort = fig.add_gridspec(1, 2, width_ratios=list(cohort_width_ratios),
                              wspace=cohort_wspace,
                              top=top, bottom=bottom, left=LEFT, right=RIGHT)
    alt.variant_b_both_arms(fig.add_subplot(cohort[0]), treated, controls,
                            panel_letter="", show_title=False,
                            treated_colours=initiator, control_colours=control,
                            linestyles=visit_linestyle,
                            treated_linestyles=treated_linestyles,
                            control_linestyles=control_linestyles,
                            ribbon_alpha=ribbon_alpha)
    ax_did = fig.add_subplot(cohort[1])
    alt.variant_c(ax_did, panel_letter="", show_title=False,
                  source=did_source, contrast_q=contrast_q,
                  **({"treated_colour": initiator["post"],
                      "control_colour": control["post"]} if palette else {}))
    figure_letter("c", LETTER_X, top + LETTER_LIFT)
    # Panel d's letter clears its own y tick labels, which are the widest in the figure
    # ("Wake in study / (44 pairs)"), so it is offset from the axes box rather than set
    # at a guessed figure x.
    figure_letter("d", ax_did.get_position().x0 - 0.105, top + LETTER_LIFT)
    return fig


def save_figure(fig, stem: Path) -> None:
    """Fit to 179 mm, write both formats, and guard the text-block height."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    width = v1.fit_to_width(fig)
    if abs(width * 25.4 - v1.TARGET_WIDTH_MM) > 1.0:
        print(f"  WARNING: {width * 25.4:.0f} mm, not {v1.TARGET_WIDTH_MM:.0f} -- "
              f"a title or out-of-axes label exceeds the target; shorten it.")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=v1.PAD_IN)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=v1.PAD_IN)
    # The tight bbox plus padding, which is what savefig(bbox_inches="tight") actually
    # wrote -- not fig.get_size_inches(), which this used until 2026-09-07. The canvas is
    # smaller than the saved page, because the panel letters sit at negative figure x and
    # the exemplar's rotated visit labels hang outside their axes, so the guard was
    # reading ~4.5 mm short and would have passed a figure that overran the text block.
    # fit_to_width measures width the same way, which is why width was always right.
    # Left OPEN for the caller. `_test_fig1_v3.py` measures the fitted canvas afterwards,
    # and fit_to_width runs in here -- measuring before this call reports the unscaled
    # canvas. Callers close their own figures.
    height_mm = (fig.get_tightbbox(fig.canvas.get_renderer()).height
                 + 2 * v1.PAD_IN) * 25.4
    print(f"\nwrote {stem}.pdf / .png  ({width * 25.4:.0f} x {height_mm:.0f} mm)")
    # 243.4 mm, not the 245 this used to allow: letterpaper is 279.4 mm tall and both tex
    # files set margin=18mm, so \textheight is 279.4 - 36. A figure taller than that
    # cannot be placed at all, caption or no caption.
    if height_mm > 243.4:
        print(f"  WARNING: {height_mm:.1f} mm tall exceeds the 243.4 mm text block; "
              f"shrink a band in `bands` or SCHEMATIC_WIDTH_FRACTION.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("participant", nargs="?", default=alt.DEFAULT_PARTICIPANT)
    parser.add_argument("--dry-run", action="store_true",
                        help="write to outputs/_test_figures/ instead of paper/figures/final/")
    parser.add_argument("--force-legacy-final", action="store_true",
                        help="deliberately overwrite final Figure 1 with the old cohort")
    args = parser.parse_args()

    if not args.dry_run and not args.force_legacy_final:
        print("SUPERSEDED by paper/build_fig1_unified.py -- refusing to overwrite final "
              "Figure 1\n  pass --dry-run to inspect this historical build or "
              "--force-legacy-final to roll back deliberately")
        return 1

    fig = build_figure(alt.resolve_participant(args.participant))
    target = SCRATCH if args.dry_run else FINAL
    save_figure(fig, target / (f"{STEM}_v2_dryrun" if args.dry_run else STEM))
    plt.close(fig)

    if not args.dry_run:
        print("  v1 backup: paper/figures/_pre_v2_backup/")
        print("  submission/figures/ NOT updated -- copy across deliberately when ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
