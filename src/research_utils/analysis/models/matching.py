"""Caliper-constrained propensity matching.

Promoted from the medication before/after difference-in-differences engine
(``research_runs/original-research/medication_effects_10k``). This extends the
simpler :func:`research_utils.analysis.models.propensity.propensity_match` with the
pieces that tight before/after matching needs and that ``propensity_match`` lacks:

* a **caliper** on the propensity-score logit (with an optional fallback that tops
  up under-filled treated units with the nearest out-of-caliper controls),
* matching **with replacement** (a control may serve several treated units),
* **multi-covariate exact blocking** (e.g. same sex *and* same pre/post visit
  window), and
* a **Mahalanobis** tie-break on the standardized covariates so the non-propensity
  covariates also balance.

All functions are frame-in / frame-out and free of I/O.

The historical :func:`caliper_match` is retained for reproducibility. New
analyses should normally use :func:`optimal_pair_match` (globally optimal 1:1,
without replacement) or :func:`capacity_caliper_match` (variable ratio with a
hard participant-level reuse cap). Neither newer function permits
out-of-caliper fallback.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "propensity_score",
    "caliper_match",
    "optimal_pair_match",
    "capacity_caliper_match",
    "standardized_mean_diff",
    "covariate_balance",
]


def propensity_score(
    units: pd.DataFrame,
    *,
    treatment_col: str,
    covariate_cols: Sequence[str],
    propensity_col: str = "propensity",
    logit_col: str = "ps_logit",
) -> pd.DataFrame:
    """Add ``propensity`` = P(treated|X) and its logit from a standardized logit fit.

    Standardizes ``covariate_cols``, fits ``LogisticRegression(max_iter=1000)`` on the
    treatment indicator, and clips the score to ``[1e-6, 1 - 1e-6]`` before taking the
    logit. Returns a copy of ``units`` with two added columns. ``units`` must already
    be free of NaN in ``covariate_cols``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    features = units[list(covariate_cols)].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(features)
    model = LogisticRegression(max_iter=1000)
    model.fit(scaled, units[treatment_col].to_numpy())
    ps = np.clip(model.predict_proba(scaled)[:, 1], 1e-6, 1 - 1e-6)
    out = units.copy()
    out[propensity_col] = ps
    out[logit_col] = np.log(ps / (1 - ps))
    return out


def caliper_match(
    units: pd.DataFrame,
    *,
    covariate_cols: Sequence[str],
    treatment_col: str = "treated",
    id_col: str = "participant_id",
    exact_cols: Sequence[str] = (),
    k: int = 5,
    caliper_sd: float = 0.2,
    caliper_fallback: bool = True,
    propensity_col: str = "propensity",
    logit_col: str = "ps_logit",
    seed: int = 0,
) -> pd.DataFrame:
    """Match each treated unit to ``k`` controls, nearest on the PS logit, with replacement.

    Controls are eligible only if they share every value in ``exact_cols`` with the
    treated unit (e.g. sex and the pre/post visit window). Within that block, candidates
    are ranked ``within_caliper`` first, then ascending PS-logit ``distance``, then
    ascending Mahalanobis distance on ``covariate_cols``, then a seeded jitter; the top
    ``k`` are kept. The caliper is ``caliper_sd * SD(logit)`` over all units.

    With ``caliper_fallback=True`` (default), a treated unit with fewer than ``k``
    in-caliper controls has its remaining slots filled by the nearest out-of-caliper
    controls in the same block, and is dropped only when its block has no controls at
    all. With ``caliper_fallback=False`` only in-caliper controls are used and a treated
    unit with none is dropped entirely (tighter balance, smaller sample).

    Controls are matched **with replacement**: the same control may match several
    treated units. Returns a long frame with one row per treated-control link:
    ``treated_id``, ``control_id``, ``match_rank``, ``within_caliper``, ``distance``,
    ``treated_ps`` / ``control_ps``, every ``exact_cols`` value, and
    ``treated_<cov>`` / ``control_<cov>`` for each covariate. Empty frame if nothing
    matched.
    """
    covariate_cols = list(covariate_cols)
    exact_cols = list(exact_cols)

    logit_sd = units[logit_col].std(ddof=1)
    caliper = caliper_sd * logit_sd
    cov = np.cov(units[covariate_cols].to_numpy(dtype=float), rowvar=False)
    cov_inv = np.linalg.pinv(np.atleast_2d(cov))
    rng = np.random.default_rng(seed)

    pairs: list[dict] = []
    treated = units[units[treatment_col].eq(1)].sort_values(logit_col, ascending=False)
    controls = units[units[treatment_col].eq(0)]
    for _, t in treated.iterrows():
        eligible = pd.Series(True, index=controls.index)
        for col in exact_cols:
            eligible &= controls[col].eq(t[col])
        candidates = controls[eligible].copy()
        if candidates.empty:
            continue
        candidates["distance"] = (candidates[logit_col] - t[logit_col]).abs()
        candidates["within_caliper"] = candidates["distance"] <= caliper
        if not caliper_fallback:
            candidates = candidates[candidates["within_caliper"]].copy()
            if candidates.empty:
                continue
        diff = (
            candidates[covariate_cols].to_numpy(dtype=float)
            - t[covariate_cols].to_numpy(dtype=float)
        )
        candidates["mahalanobis"] = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))
        candidates["jitter"] = rng.random(len(candidates))
        ranked = candidates.sort_values(
            ["within_caliper", "distance", "mahalanobis", "jitter"],
            ascending=[False, True, True, True],
        )
        matches = ranked.head(k)
        for rank, (_, m) in enumerate(matches.iterrows(), start=1):
            row = {
                "treated_id": t[id_col],
                "control_id": m[id_col],
                "match_rank": rank,
                "within_caliper": bool(m["within_caliper"]),
                "treated_ps": t[propensity_col],
                "control_ps": m[propensity_col],
                "distance": m["distance"],
            }
            for col in exact_cols:
                row[col] = t[col]
            for cov_name in covariate_cols:
                row[f"treated_{cov_name}"] = t[cov_name]
                row[f"control_{cov_name}"] = m[cov_name]
            pairs.append(row)
    return pd.DataFrame(pairs)


def _candidate_edges(
    units: pd.DataFrame,
    *,
    covariate_cols: Sequence[str],
    treatment_col: str,
    id_col: str,
    exact_cols: Sequence[str],
    caliper_sd: float,
    propensity_col: str,
    logit_col: str,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    """Build eligible treated-control edges for the capacity-aware matchers."""
    from scipy.spatial.distance import cdist

    covariate_cols = list(covariate_cols)
    exact_cols = list(exact_cols)
    treated = units[units[treatment_col].eq(1)].copy()
    controls = units[units[treatment_col].eq(0)].copy()
    if treated.empty or controls.empty:
        return pd.DataFrame(), np.nan

    logit_sd = float(pd.to_numeric(units[logit_col], errors="coerce").std(ddof=1))
    caliper = caliper_sd * logit_sd
    cov = np.cov(units[covariate_cols].to_numpy(dtype=float), rowvar=False)
    cov_inv = np.linalg.pinv(np.atleast_2d(cov))
    rng = np.random.default_rng(seed)
    edges: list[dict] = []

    for treated_index, t in treated.iterrows():
        eligible = pd.Series(True, index=controls.index)
        for col in exact_cols:
            eligible &= controls[col].eq(t[col])
        candidates = controls[eligible].copy()
        if candidates.empty:
            continue
        candidates["distance"] = (
            pd.to_numeric(candidates[logit_col], errors="coerce") - float(t[logit_col])
        ).abs()
        candidates = candidates[candidates["distance"].le(caliper)].copy()
        if candidates.empty:
            continue
        difference = (
            candidates[covariate_cols].to_numpy(dtype=float)
            - t[covariate_cols].to_numpy(dtype=float)
        )
        candidates["mahalanobis"] = cdist(
            difference,
            np.zeros((1, len(covariate_cols))),
            metric="mahalanobis",
            VI=cov_inv,
        )[:, 0]
        candidates["jitter"] = rng.uniform(0.0, 1e-9, size=len(candidates))
        for control_index, c in candidates.iterrows():
            row = {
                "_treated_index": treated_index,
                "_control_index": control_index,
                "treated_id": t[id_col],
                "control_id": c[id_col],
                "distance": float(c["distance"]),
                "mahalanobis": float(c["mahalanobis"]),
                "_jitter": float(c["jitter"]),
                "within_caliper": True,
                "caliper": caliper,
                "treated_ps": float(t[propensity_col]),
                "control_ps": float(c[propensity_col]),
            }
            for col in exact_cols:
                row[col] = t[col]
            for cov_name in covariate_cols:
                row[f"treated_{cov_name}"] = t[cov_name]
                row[f"control_{cov_name}"] = c[cov_name]
            edges.append(row)
    frame = pd.DataFrame(edges)
    if frame.empty:
        return frame, caliper

    # A participant may appear as an eligible control unit in several windows.
    # Retain the best row for each treated-control participant edge; capacity is
    # applied to the participant id, not the unit-row index.
    frame = (
        frame.sort_values(["mahalanobis", "distance", "_jitter"])
        .drop_duplicates(["treated_id", "control_id"], keep="first")
        .reset_index(drop=True)
    )
    return frame, caliper


def _edge_costs(edges: pd.DataFrame) -> pd.Series:
    """Lexicographic Mahalanobis, PS-distance, jitter cost scaled below one."""
    if edges.empty:
        return pd.Series(dtype=float)

    def scaled(values: pd.Series) -> pd.Series:
        values = pd.to_numeric(values, errors="coerce")
        lo, hi = values.min(), values.max()
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(0.0, index=values.index)
        return (values - lo) / (hi - lo)

    return (
        0.998 * scaled(edges["mahalanobis"])
        + 0.001999 * scaled(edges["distance"])
        + edges["_jitter"]
    )


def _assignment_round(
    edges: pd.DataFrame,
    treated_ids: Sequence,
    control_slots: Sequence[tuple],
) -> pd.DataFrame:
    """Maximum-cardinality, minimum-cost assignment for one matching round."""
    from scipy.optimize import linear_sum_assignment

    treated_ids = list(treated_ids)
    control_slots = list(control_slots)
    if not treated_ids or not control_slots or edges.empty:
        return pd.DataFrame()

    n_treated = len(treated_ids)
    n_slots = len(control_slots)
    treated_position = {participant: i for i, participant in enumerate(treated_ids)}
    slot_positions: dict[object, list[int]] = {}
    for j, (participant, _slot_number) in enumerate(control_slots):
        slot_positions.setdefault(participant, []).append(j)

    # Valid costs are <1, so a full assignment's edges sum to < n_treated. A dummy
    # (unmatched) assignment costs more than that, so no rearrangement of edges can
    # make dropping a treated unit cheaper: the solver first maximizes match
    # cardinality and only then minimizes edge distance.
    unmatched_cost = float(n_treated + 1)
    cost = np.full((n_treated, n_slots + n_treated), 1e6 * unmatched_cost, dtype=float)
    cost[:, n_slots:] = unmatched_cost
    edge_cost = _edge_costs(edges)
    edge_lookup: dict[tuple, int] = {}
    for edge_index, edge in edges.iterrows():
        i = treated_position.get(edge["treated_id"])
        if i is None:
            continue
        for j in slot_positions.get(edge["control_id"], []):
            candidate_cost = float(edge_cost.loc[edge_index])
            if candidate_cost < cost[i, j]:
                cost[i, j] = candidate_cost
                edge_lookup[(i, j)] = edge_index

    row_ind, col_ind = linear_sum_assignment(cost)
    selected = [
        edge_lookup[(i, j)]
        for i, j in zip(row_ind, col_ind, strict=True)
        if j < n_slots and (i, j) in edge_lookup and cost[i, j] < 1.0
    ]
    return edges.loc[selected].copy().reset_index(drop=True)


def optimal_pair_match(
    units: pd.DataFrame,
    *,
    covariate_cols: Sequence[str],
    treatment_col: str = "treated",
    id_col: str = "participant_id",
    exact_cols: Sequence[str] = (),
    caliper_sd: float = 0.2,
    propensity_col: str = "propensity",
    logit_col: str = "ps_logit",
    seed: int = 0,
) -> pd.DataFrame:
    """Globally optimal 1:1 matching without replacement and without fallback.

    Candidate links must share every ``exact_cols`` value and fall within
    ``caliper_sd * SD(logit)``. Among eligible links, a linear assignment first
    maximizes the number of matched treated participants and then minimizes
    Mahalanobis distance (PS-logit distance is the secondary tie-break). A
    control participant can appear only once across the entire returned match,
    even when that participant is eligible in several visit windows.
    """
    edges, _ = _candidate_edges(
        units,
        covariate_cols=covariate_cols,
        treatment_col=treatment_col,
        id_col=id_col,
        exact_cols=exact_cols,
        caliper_sd=caliper_sd,
        propensity_col=propensity_col,
        logit_col=logit_col,
        seed=seed,
    )
    if edges.empty:
        return edges
    treated_ids = (
        units.loc[units[treatment_col].eq(1), id_col].drop_duplicates().tolist()
    )
    control_ids = (
        units.loc[units[treatment_col].eq(0), id_col].drop_duplicates().tolist()
    )
    selected = _assignment_round(
        edges,
        treated_ids,
        [(participant, 1) for participant in control_ids],
    )
    if selected.empty:
        return selected
    selected["match_rank"] = 1
    selected["match_weight"] = 1.0
    return selected.drop(columns=["_jitter"], errors="ignore").reset_index(drop=True)


def capacity_caliper_match(
    units: pd.DataFrame,
    *,
    covariate_cols: Sequence[str],
    treatment_col: str = "treated",
    id_col: str = "participant_id",
    exact_cols: Sequence[str] = (),
    max_ratio: int = 3,
    control_capacity: int = 2,
    caliper_sd: float = 0.2,
    propensity_col: str = "propensity",
    logit_col: str = "ps_logit",
    seed: int = 0,
) -> pd.DataFrame:
    """Variable-ratio caliper matching with a hard control-participant reuse cap.

    The first optimal assignment gives as many treated participants as possible
    one control. Subsequent optimal rounds add up to ``max_ratio`` distinct
    controls per already-matched treated participant while respecting a global
    ``control_capacity``. This round-wise design prioritizes treated retention,
    then additional precision. No out-of-caliper match is ever emitted.
    """
    if max_ratio < 1:
        raise ValueError("max_ratio must be at least 1")
    if control_capacity < 1:
        raise ValueError("control_capacity must be at least 1")

    edges, _ = _candidate_edges(
        units,
        covariate_cols=covariate_cols,
        treatment_col=treatment_col,
        id_col=id_col,
        exact_cols=exact_cols,
        caliper_sd=caliper_sd,
        propensity_col=propensity_col,
        logit_col=logit_col,
        seed=seed,
    )
    if edges.empty:
        return edges

    treated_ids = (
        units.loc[units[treatment_col].eq(1), id_col].drop_duplicates().tolist()
    )
    control_ids = (
        units.loc[units[treatment_col].eq(0), id_col].drop_duplicates().tolist()
    )
    use_count = {participant: 0 for participant in control_ids}
    selected_frames: list[pd.DataFrame] = []
    selected_by_treated: dict[object, set] = {participant: set() for participant in treated_ids}

    for match_rank in range(1, max_ratio + 1):
        if match_rank > 1:
            treated_ids = [
                participant for participant in treated_ids if selected_by_treated[participant]
            ]
        slots = [
            (participant, slot)
            for participant in control_ids
            for slot in range(use_count[participant] + 1, control_capacity + 1)
        ]
        if not treated_ids or not slots:
            break
        available = edges[
            edges.apply(
                lambda row: row["control_id"] not in selected_by_treated[row["treated_id"]],
                axis=1,
            )
        ]
        chosen = _assignment_round(available, treated_ids, slots)
        if chosen.empty:
            break
        chosen["match_rank"] = match_rank
        selected_frames.append(chosen)
        for row in chosen.itertuples(index=False):
            use_count[row.control_id] += 1
            selected_by_treated[row.treated_id].add(row.control_id)

    if not selected_frames:
        return pd.DataFrame()
    selected = pd.concat(selected_frames, ignore_index=True)
    match_sizes = selected.groupby("treated_id")["control_id"].transform("size")
    selected["match_weight"] = 1.0 / match_sizes
    return selected.drop(columns=["_jitter"], errors="ignore").reset_index(drop=True)


def standardized_mean_diff(treated: pd.Series, control: pd.Series) -> float:
    """Standardized mean difference between two samples (pooled-SD denominator).

    Returns ``nan`` when either arm has fewer than two finite values, and ``0.0`` when
    the pooled SD is exactly zero.
    """
    t = pd.to_numeric(treated, errors="coerce").dropna()
    c = pd.to_numeric(control, errors="coerce").dropna()
    if len(t) < 2 or len(c) < 2:
        return np.nan
    pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((t.mean() - c.mean()) / pooled_sd)


def covariate_balance(
    units: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    covariate_cols: Sequence[str],
    treatment_col: str = "treated",
) -> pd.DataFrame:
    """Per-covariate SMD before matching (all units) and after (matched pairs).

    ``smd_before`` compares every treated unit to every control; ``smd_after`` compares
    the ``treated_<cov>`` and ``control_<cov>`` columns of the matched ``pairs`` frame
    from :func:`caliper_match`. Also reports the matched treated/control means.
    """
    treated_all = units[units[treatment_col].eq(1)]
    control_all = units[units[treatment_col].eq(0)]
    rows = []
    for cov in covariate_cols:
        after = (
            standardized_mean_diff(pairs[f"treated_{cov}"], pairs[f"control_{cov}"])
            if not pairs.empty
            else np.nan
        )
        rows.append(
            {
                "covariate": cov,
                "smd_before": standardized_mean_diff(treated_all[cov], control_all[cov]),
                "smd_after": after,
                "treated_mean_matched": (
                    pd.to_numeric(pairs[f"treated_{cov}"], errors="coerce").mean()
                    if not pairs.empty
                    else np.nan
                ),
                "control_mean_matched": (
                    pd.to_numeric(pairs[f"control_{cov}"], errors="coerce").mean()
                    if not pairs.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
