"""Build the paper's opening figure: sleep architecture, design, and the finding.

Purpose
-------
The figure set previously opened with exposure adjudication and matching
diagnostics, which buries the one novel result three figures deep. This is the
replacement Figure 1: it shows *what a night looks like*, *what we did*, and
*what we found*, in one page.

Panels
------
a  A real de-identified WatchPAT recording: hypnogram (wake / REM / light / deep)
   with the simultaneous 1 Hz heart-rate trace beneath it, REM epochs shaded.
   This is what makes a stage-resolved heart-rate analysis possible at all.
b  Design schematic: a sleep study before initiation, another on treatment, and
   the same contrast in propensity-matched never-GLP-1 controls.
c  The primary finding: heart rate rises in every measured state and most in REM,
   with the formal REM-minus-NREM contrast called out.

Data provenance
---------------
Panels b and c use only aggregate numbers already reported in
`PAPER_REANALYSIS.md` / `paper/NUMBERS.md` (the paper's primary matched-DiD
estimates on the vendor summary fields, n=41 for the state contrast).

Panel a is the one exception in this repo's figure code: it plots a **single
representative de-identified night** of raw channels, which is conventional for
sleep papers but does touch participant-level data. The plotted arrays are cached
to `outputs/opener_hypnogram.csv` so the figure rebuilds without re-reading the
raw store, and no identifier is written. Swap `REPRESENTATIVE_UUID` to choose a
different night.

Usage
-----
    PY=/net/mraid20/export/jasmine/david/anaconda3/bin/python3
    $PY paper/build_opener_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RUN = Path(__file__).resolve().parents[1]
FIGURES = RUN / "paper" / "figures" / "final"
CACHE = RUN / "outputs" / "opener_hypnogram.csv"
TIMESERIES = Path("/net/mraid20/export/genie/LabData/Data/Pheno/gold/sleep/timeseries")

# Representative night: all four stages >20 min, >97% of heart-rate samples in
# range, 10 distinct REM bouts. Chosen for legibility, not for its effect size.
REPRESENTATIVE_UUID = "5e6a94d4-3c96-4e24-9416-dd8ca23aa972"
REPRESENTATIVE_NIGHT = "night_0"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
BLACK = "#222222"
GREY = "#777777"
LIGHT_GREY = "#D9D9D9"

# 180 mm width cap; both tex files set margin=18mm so \textwidth matches and
# LaTeX applies no rescaling -- printed pt == authored pt.
TARGET_WIDTH_MM = 179.0
TARGET_WIDTH_IN = TARGET_WIDTH_MM / 25.4
PAD_IN = 0.01
PANEL_LABEL_PT = 8


def fit_to_width(fig, pad: float = PAD_IN,
                 target_mm: float = TARGET_WIDTH_MM) -> float:
    """Resize the canvas until the tight-cropped PDF is `target_mm` wide.

    `target_mm` matches the signature the other two builders already use. A figure is
    placed in the tex at exactly the width it was saved at, so a sparse single panel
    authored narrower than the 179 mm text block still prints at source point size --
    it just does not have to stretch its marks across the full block to get there.
    """
    target = target_mm / 25.4 - 2 * pad
    for _ in range(8):
        fig.canvas.draw()
        width = fig.get_tightbbox(fig.canvas.get_renderer()).width
        if abs(width - target) <= 0.001:
            break
        current_w, current_h = fig.get_size_inches()
        fig.set_size_inches(current_w * target / width, current_h)
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad

PALE = "#F2F2F2"

STAGE_CODES = {11: "wake", 12: "rem", 13: "light", 15: "deep"}
# Plot order, top to bottom, matching clinical hypnogram convention.
STAGE_ROW = {"wake": 3, "rem": 2, "light": 1, "deep": 0}
STAGE_COLOUR = {"wake": SKY, "rem": VERMILLION, "light": BLUE, "deep": "#31456B"}

# The paper's primary matched-DiD sleep estimates (PAPER_REANALYSIS.md).
PRIMARY = [
    ("REM", 6.97, 4.36, 9.72, 41),
    ("Non-REM (pooled)", 5.61, 2.77, 8.37, 41),
    ("Whole sleep", 5.54, 2.82, 8.17, 44),
    ("Wake in study", 4.20, 1.32, 6.76, 44),
]
CONTRAST = {"label": "REM − non-REM", "did": 1.36, "low": 0.67, "high": 2.05, "q": 0.00088}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.titlelocation": "left",
            "axes.titlepad": 4.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def panel_label(ax: plt.Axes, text: str, x: float = -0.08, y: float = 1.10) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=PANEL_LABEL_PT,
            fontweight="bold", va="top")


def load_representative_night() -> pd.DataFrame:
    """Stage code + heart rate per second for one de-identified night."""
    if CACHE.exists():
        return pd.read_csv(CACHE)
    night = TIMESERIES / REPRESENTATIVE_UUID / REPRESENTATIVE_NIGHT
    stage = np.asarray(
        pq.read_table(night / "sleep_stage.parquet", columns=["values"]).column("values")
    )
    heart_rate = np.asarray(
        pq.read_table(night / "heart_rate.parquet", columns=["values"]).column("values")
    )
    frame = pd.DataFrame({"second": np.arange(len(stage)), "stage_code": stage,
                          "heart_rate": heart_rate})
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE, index=False)  # no identifier written
    return frame


def panel_a(ax_stage: plt.Axes, ax_hr: plt.Axes) -> None:
    frame = load_representative_night()
    hours = frame.second.to_numpy() / 3600.0
    stage = frame.stage_code.to_numpy()
    heart_rate = pd.to_numeric(frame.heart_rate, errors="coerce").to_numpy()
    heart_rate = np.where((heart_rate >= 30) & (heart_rate <= 140), heart_rate, np.nan)

    rows = np.full(len(stage), np.nan)
    for code, name in STAGE_CODES.items():
        rows[stage == code] = STAGE_ROW[name]

    # Shade REM bouts in both sub-panels so the eye links stage to rate.
    in_rem = stage == 12
    edges = np.diff(in_rem.astype(int))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if in_rem[0]:
        starts = np.r_[0, starts]
    if in_rem[-1]:
        ends = np.r_[ends, len(in_rem)]
    for axis in (ax_stage, ax_hr):
        for s, e in zip(starts, ends):
            axis.axvspan(hours[s], hours[e - 1], color=VERMILLION, alpha=0.13,
                         linewidth=0, zorder=0)

    ax_stage.step(hours, rows, where="post", color=BLACK, linewidth=0.8, zorder=3)
    # Ticks must be given in ascending order and labelled by the row they encode,
    # otherwise the stage names end up inverted against the trace.
    row_to_name = {row: name for name, row in STAGE_ROW.items()}
    pretty = {"deep": "Deep (N3)", "light": "Light (N1+N2)", "rem": "REM", "wake": "Wake"}
    ordered_rows = sorted(row_to_name)
    ax_stage.set_yticks(ordered_rows)
    ax_stage.set_yticklabels([pretty[row_to_name[r]] for r in ordered_rows])
    ax_stage.set_ylim(-0.6, 3.6)
    for tick, row in zip(ax_stage.get_yticklabels(), ordered_rows):
        tick.set_color(STAGE_COLOUR[row_to_name[row]])
        if row_to_name[row] == "rem":
            tick.set_fontweight("bold")
    # labelbottom=False, not set_xticklabels([]) -- the latter installs a fixed
    # formatter on the *shared* x-axis and would blank the heart-rate panel too.
    ax_stage.spines["top"].set_visible(False)
    ax_stage.spines["right"].set_visible(False)
    ax_stage.spines["bottom"].set_visible(False)
    ax_stage.tick_params(axis="x", length=0, labelbottom=False)
    ax_stage.set_title(
        "One night of home sleep phenotyping: heart rate is resolvable within each stage",
        loc="left",
    )
    panel_label(ax_stage, "a", x=-0.075, y=1.30)

    # Smooth only for display; the analysis uses per-second means within stage.
    window = 61
    kernel = np.ones(window) / window
    padded = np.pad(np.nan_to_num(heart_rate, nan=np.nanmean(heart_rate)),
                    window // 2, mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")[: len(hours)]
    ax_hr.plot(hours, smooth, color=BLACK, linewidth=0.8, zorder=3)
    for code, name in STAGE_CODES.items():
        mask = stage == code
        mean_hr = np.nanmean(heart_rate[mask])
        ax_hr.plot([hours[mask].min(), hours[mask].max()], [mean_hr, mean_hr],
                   color=STAGE_COLOUR[name], linewidth=0, zorder=2)
    ax_hr.set_ylabel("Heart rate\n(bpm)")
    ax_hr.set_xlabel("Hours into recording")
    ax_hr.spines["top"].set_visible(False)
    ax_hr.spines["right"].set_visible(False)
    ax_hr.grid(axis="y", color="#ECECEC", linewidth=0.6)
    ax_hr.set_axisbelow(True)
    ax_hr.text(0.995, 0.06, "shading = REM", transform=ax_hr.transAxes, ha="right",
               fontsize=6.6, color=VERMILLION, fontweight="bold")


def panel_b(ax: plt.Axes, panel_letter: str = "b", n_pairs: int = 44) -> None:
    """The matched before/after schematic. Reused as panel a of `build_fig0_design.py`.

    n_pairs is the number of PAIRS THE DESIGN FORMED (44). This used to read
    `PRIMARY[0][4]` = 41, which is the REM row's OUTCOME count -- how many of those
    pairs have REM heart rate scored at both visits -- so the design was being
    labelled with a downstream figure. Per-outcome counts belong on the DiD panel.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.10)
    ax.axis("off")
    # n lives in the title: every in-panel position collides with an arm or a box.
    ax.set_title(f"Matched before/after design (n = {n_pairs} pairs)", loc="left")
    panel_label(ax, panel_letter, x=-0.07, y=1.16)

    # Box geometry: 2.30 wide is what "sleep study" needs at 6.4 pt without the
    # glyphs touching the rounded rim. Text is centred on the box interior, so
    # changing BOX_W alone keeps everything aligned.
    BOX_W, BOX_H = 2.30, 0.66
    LEFT_X, RIGHT_X = 0.12, 6.45
    arrow_from, arrow_to = LEFT_X + BOX_W + 0.14, RIGHT_X - 0.14
    arrow_mid = (arrow_from + arrow_to) / 2

    def visit(x, y, colour, text):
        ax.add_patch(FancyBboxPatch((x, y), BOX_W, BOX_H, boxstyle="round,pad=0.04",
                                    linewidth=0.9, edgecolor=colour,
                                    facecolor="white", zorder=3))
        ax.text(x + BOX_W / 2, y + BOX_H / 2, text, ha="center", va="center",
                fontsize=6.4, zorder=4, linespacing=1.30)

    # Treated arm
    ax.text(0, 3.68, "GLP-1 initiators", fontsize=7.2, fontweight="bold", color=VERMILLION)
    visit(LEFT_X, 2.70, VERMILLION, "sleep study\nbefore")
    visit(RIGHT_X, 2.70, VERMILLION, "sleep study\non drug")
    ax.add_patch(FancyArrowPatch((arrow_from, 3.03), (arrow_to, 3.03), arrowstyle="-|>",
                                mutation_scale=8, linewidth=0.9, color=GREY, zorder=2))
    ax.plot([arrow_mid], [3.03], marker="v", color=BLACK, markersize=6, zorder=4)
    ax.text(arrow_mid, 3.32, "semaglutide started", ha="center", fontsize=6.4,
            fontweight="bold")

    # Control arm
    ax.text(0, 2.12, "Matched never-GLP-1 controls", fontsize=7.2, fontweight="bold",
            color=BLUE)
    visit(LEFT_X, 1.14, BLUE, "sleep study\nvisit 1")
    visit(RIGHT_X, 1.14, BLUE, "sleep study\nvisit 2")
    ax.add_patch(FancyArrowPatch((arrow_from, 1.47), (arrow_to, 1.47), arrowstyle="-|>",
                                mutation_scale=8, linewidth=0.9, color=GREY, zorder=2))
    # Short enough not to reach the right-hand box rim; "no drug" is already
    # carried by the arm heading.
    ax.text(arrow_mid, 1.60, "same interval", ha="center", fontsize=6.4, color=GREY)

    # The estimand. Wrapped onto two lines so it cannot overrun the panel width.
    ax.add_patch(FancyBboxPatch((0.15, 0.02), 9.7, 0.82, boxstyle="round,pad=0.06",
                                linewidth=0, facecolor=PALE, zorder=1))
    ax.text(5.0, 0.43,
            "difference-in-differences =\n"
            "(initiator after − before) − (control after − before)",
            ha="center", va="center", fontsize=6.6, zorder=3, linespacing=1.35)


def panel_c(ax: plt.Axes) -> None:
    labels = [row[0] for row in PRIMARY]
    values = [row[1] for row in PRIMARY]
    lows = [row[2] for row in PRIMARY]
    highs = [row[3] for row in PRIMARY]
    positions = np.arange(len(labels))[::-1]

    for y, label, did, low, high in zip(positions, labels, values, lows, highs):
        highlight = label == "REM"
        colour = VERMILLION if highlight else GREY
        ax.plot([low, high], [y, y], color=colour, linewidth=1.9 if highlight else 1.2,
                solid_capstyle="round", zorder=2)
        ax.plot([did], [y], marker="o", color=colour,
                markersize=7 if highlight else 5, markeredgecolor="white",
                markeredgewidth=0.6, zorder=3)
        ax.annotate(f"+{did:.2f}", (did, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7,
                    fontweight="bold" if highlight else "normal", color=colour)

    ax.axvline(0, color=BLACK, linewidth=0.7, linestyle="--", zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.get_yticklabels()[0].set_color(VERMILLION)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(-0.4, 11.6)
    ax.set_ylim(-1.35, len(labels) - 0.30)
    ax.set_xlabel("Difference-in-differences in heart rate (bpm)")
    ax.set_title("Heart rate rises in every state — most in REM", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.6)
    ax.set_axisbelow(True)

    ax.annotate(
        f"{CONTRAST['label']} = +{CONTRAST['did']:.2f} bpm "
        f"[{CONTRAST['low']:.2f}, {CONTRAST['high']:.2f}], q = {CONTRAST['q']:.5f}",
        xy=(0.5, -1.05), xycoords=("axes fraction", "data"), ha="center", va="center",
        fontsize=6.9, fontweight="bold", color=VERMILLION,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#FDF1E7",
                  edgecolor=VERMILLION, linewidth=0.7),
    )
    panel_label(ax, "c", x=-0.30, y=1.16)


def main() -> int:
    """SUPERSEDED. Kept for rollback; refuses to overwrite Figure 1 without --force.

    `build_fig1_v2.py` now owns `fig1_sleep_phenotyping_and_finding.{pdf,png}` and this
    function writes the same two filenames, so running it out of habit would silently
    revert the opener with no error and no visible diff in the manuscript. The design
    schematic this file still contains lives on as panel a of `build_fig0_design.py`.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--force", action="store_true",
                        help="really overwrite the v2 opener with this v1 layout")
    if not parser.parse_args().force:
        print("SUPERSEDED by paper/build_fig1_v2.py -- refusing to overwrite\n"
              "  fig1_sleep_phenotyping_and_finding.{pdf,png}\n"
              "  pass --force to roll back to this v1 layout deliberately.")
        return 1

    configure_style()
    fig = plt.figure(figsize=(7.2, 6.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.34)
    # Nested grid so the hypnogram and its heart-rate trace sit flush together;
    # a single outer hspace would push them apart as far as panels b/c.
    top = outer[0].subgridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.10)
    ax_stage = fig.add_subplot(top[0])
    ax_hr = fig.add_subplot(top[1], sharex=ax_stage)
    panel_a(ax_stage, ax_hr)

    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.34)
    panel_b(fig.add_subplot(bottom[0]))
    panel_c(fig.add_subplot(bottom[1]))

    FIGURES.mkdir(parents=True, exist_ok=True)
    width = fit_to_width(fig)
    print(f"  fig1_sleep_phenotyping_and_finding: {width:.2f} in "
          f"({width * 25.4:.0f} mm) wide, scale 1.00")
    fig.savefig(FIGURES / "fig1_sleep_phenotyping_and_finding.pdf",
                bbox_inches="tight", pad_inches=PAD_IN)
    fig.savefig(FIGURES / "fig1_sleep_phenotyping_and_finding.png",
                dpi=300, bbox_inches="tight", pad_inches=PAD_IN)
    plt.close(fig)
    print("wrote fig1_sleep_phenotyping_and_finding.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
