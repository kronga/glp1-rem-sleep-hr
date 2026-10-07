"""Universal statistics helpers used across HPP analyses.

These are general numerical recipes -- not tied to a single paper. The
default ``bh_fdr`` form matches Reicher 2025's ``utils.fdr_pval``;
opt-in ``monotonic=True`` for the canonical BH procedure.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "bh_fdr",
    "bh_fdr_within_group",
    "bootstrap_ci",
    "bootstrapped_partial_corr",
    "compare_groups",
    "correlation_scan",
    "logistic_or_p",
    "partial_correlation",
    "partial_spearman",
    "pearson_correlation_matrix",
    "quartile_split_mann_whitney",
    "intra_vs_inter_pair_distance",
    "residualize",
    "spearman_matrix_scan",
    "standardized_ols",
    "zscore_series",
]


def bh_fdr(
    pvalues: np.ndarray | Sequence[float],
    *,
    monotonic: bool = False,
) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment, capped at 1.

    With ``monotonic=False`` (default) returns the simple ``p * n / rank``
    form (no monotonicity enforcement), matching the formula used in
    Reicher 2025's ``utils.fdr_pval`` and ties handled via
    :func:`scipy.stats.rankdata`. With ``monotonic=True`` enforces
    ``q[i] = min(q[i:])`` after sorting, matching the canonical BH
    procedure used by ``statsmodels.stats.multitest.multipletests``
    with ``method="fdr_bh"``.
    """
    from scipy.stats import rankdata

    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if not monotonic:
        fdr = p * n / rankdata(p)
        fdr[fdr > 1] = 1
        return fdr
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = q
    return np.clip(out, 0.0, 1.0)


def bh_fdr_within_group(
    frame: pd.DataFrame,
    *,
    p_col: str = "p",
    group_cols: str | Sequence[str],
    q_col: str = "q",
    monotonic: bool = True,
) -> pd.DataFrame:
    """Add BH-FDR q-values within each group of a result table.

    Repeated system/modality scans should not pool unrelated hypothesis
    families by accident. This helper keeps the FDR family explicit in
    the output table.
    """

    groups = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    missing = [c for c in [p_col, *groups] if c not in frame.columns]
    if missing:
        raise KeyError(f"frame missing required columns: {missing!r}")
    out = frame.copy()
    out[q_col] = np.nan
    for _, idx in out.groupby(groups, dropna=False).groups.items():
        p = pd.to_numeric(out.loc[idx, p_col], errors="coerce").fillna(1.0)
        out.loc[idx, q_col] = bh_fdr(p.to_numpy(), monotonic=monotonic)
    return out


def bootstrap_ci(
    values: Sequence[float] | np.ndarray | pd.Series,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 1000,
    ci: tuple[float, float] = (0.025, 0.975),
    random_seed: int = 42,
) -> dict[str, float | int]:
    """Percentile bootstrap interval for a one-sample statistic.

    Promoted as the pure bootstrap core behind repeated scorecard,
    model-summary, and IPW runs. Non-finite values are dropped before
    resampling.
    """

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    lo, hi = ci
    if not 0 <= lo < hi <= 1:
        raise ValueError("ci must be increasing quantiles in [0, 1]")
    if arr.size == 0:
        return {
            "n": 0,
            "statistic": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    rng = np.random.default_rng(random_seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        draw = rng.choice(arr, size=arr.size, replace=True)
        samples[i] = float(statistic(draw))
    return {
        "n": int(arr.size),
        "statistic": float(statistic(arr)),
        "ci_low": float(np.quantile(samples, lo)),
        "ci_high": float(np.quantile(samples, hi)),
    }


def compare_groups(
    df: pd.DataFrame,
    ids_a: Sequence,
    ids_b: Sequence,
    *,
    name_a: str = "a",
    name_b: str = "b",
    sort_by: Literal["fdr_mw", "mw_pvalue", "ttest_pvalue"] = "fdr_mw",
) -> pd.DataFrame:
    """Per-column t-test + Mann-Whitney comparison with BH-FDR on MW p-values.

    One row per column of ``df``: sizes, means, stds, raw p-values, and
    ``fdr_mw``. Rows are sorted by ``sort_by`` ascending.
    """
    from scipy.stats import mannwhitneyu, ttest_ind

    a = [i for i in ids_a if i in df.index]
    b = [i for i in ids_b if i in df.index]
    rows: list[dict] = []
    for col in df.columns:
        x = df[col].loc[a].dropna()
        y = df[col].loc[b].dropna()
        if len(x) < 2 or len(y) < 2:
            tt = mw = float("nan")
        else:
            tt = ttest_ind(x.values, y.values).pvalue
            try:
                mw = mannwhitneyu(x.values, y.values).pvalue
            except ValueError:
                mw = float("nan")
        rows.append(
            {
                "feature": col,
                f"{name_a} size": len(x),
                f"{name_b} size": len(y),
                f"{name_a} mean": x.mean() if len(x) else float("nan"),
                f"{name_a} std": x.std() if len(x) > 1 else float("nan"),
                f"{name_b} mean": y.mean() if len(y) else float("nan"),
                f"{name_b} std": y.std() if len(y) > 1 else float("nan"),
                "ttest_pvalue": tt,
                "mw_pvalue": mw,
            }
        )
    out = pd.DataFrame(rows).set_index("feature")
    out["mw_pvalue"] = out["mw_pvalue"].fillna(1.0)
    out["fdr_mw"] = bh_fdr(out["mw_pvalue"].values)
    return out.sort_values(sort_by)


def logistic_or_p(
    outcome: pd.Series,
    predictor: pd.Series,
    *,
    covariates: pd.DataFrame | None = None,
    name: str | None = None,
) -> dict:
    """Single logistic regression: ``outcome ~ predictor + covariates``.

    Returns a dict with the predictor coefficient summary: ``odds_ratio``,
    ``ci_lower``, ``ci_upper``, ``p_value``, ``coef``, ``se``, ``n``.
    Returns NaN-filled record if the outcome has fewer than 2 unique values
    or fitting fails.
    """
    import statsmodels.api as sm

    df = pd.concat(
        [
            outcome.rename("__y"),
            predictor.rename("__x"),
        ],
        axis=1,
    )
    if covariates is not None and len(covariates.columns) > 0:
        df = df.join(covariates)
    df = df.dropna()
    record: dict = {
        "name": name,
        "n": int(len(df)),
        "n_pos": int(df["__y"].sum()) if len(df) else 0,
        "odds_ratio": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "p_value": float("nan"),
        "coef": float("nan"),
        "se": float("nan"),
    }
    if len(df) < 10 or df["__y"].nunique() < 2:
        return record
    X = sm.add_constant(df.drop(columns="__y"))
    y = df["__y"].astype(float)
    try:
        fit = sm.Logit(y, X).fit(disp=0, method="lbfgs")
    except Exception:  # noqa: BLE001
        return record
    if "__x" not in fit.params.index:
        return record
    coef = float(fit.params["__x"])
    se = float(fit.bse["__x"])
    record.update(
        {
            "coef": coef,
            "se": se,
            "p_value": float(fit.pvalues["__x"]),
            "odds_ratio": float(np.exp(coef)),
            "ci_lower": float(np.exp(coef - 1.96 * se)),
            "ci_upper": float(np.exp(coef + 1.96 * se)),
        }
    )
    return record


def pearson_correlation_matrix(
    df: pd.DataFrame,
    *,
    pairwise: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Pearson correlations + p-values across all numeric columns.

    Returns ``(r_matrix, p_matrix)`` as square DataFrames indexed/columned by
    the input columns. ``pairwise=True`` (default) uses pairwise-complete
    rows; pairs with fewer than 3 complete observations get NaN.
    """
    from scipy.stats import pearsonr

    cols = list(df.columns)
    r = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    p = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i >= j:
                continue
            sub = df[[ci, cj]].dropna() if pairwise else df[[ci, cj]]
            if len(sub) < 3:
                r.iloc[i, j] = r.iloc[j, i] = np.nan
                p.iloc[i, j] = p.iloc[j, i] = np.nan
                continue
            stat = pearsonr(sub[ci], sub[cj])
            r.iloc[i, j] = r.iloc[j, i] = float(stat.statistic)
            p.iloc[i, j] = p.iloc[j, i] = float(stat.pvalue)
    return r, p


def zscore_series(values: pd.Series, *, ddof: int = 0) -> pd.Series | None:
    """Return a z-scored numeric Series, or ``None`` for constant input.

    Promoted from the multimodal-clock and SleepChart translation runs.
    Uses ``ddof=0`` by default because these helpers treat the sample as
    the observed analysis cohort when producing standardized model
    coefficients.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    sd = numeric.std(ddof=ddof)
    if not np.isfinite(sd) or sd == 0:
        return None
    return (numeric - numeric.mean()) / sd


def residualize(
    values: pd.Series,
    covariates: pd.DataFrame | None = None,
) -> pd.Series:
    """OLS residuals from ``values ~ covariates`` with an intercept.

    If no covariates are supplied, returns centered values. Rows with
    missing values in either the outcome or covariates are dropped.
    """
    y = pd.to_numeric(values, errors="coerce").rename("__y")
    if covariates is None or covariates.empty:
        return (y - y.mean()).dropna()

    df = pd.concat([y, covariates], axis=1).replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    if df.empty:
        return pd.Series(dtype=float, name=values.name)
    X = df.drop(columns="__y").astype(float)
    design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(design, df["__y"].to_numpy(dtype=float), rcond=None)
    resid = df["__y"].to_numpy(dtype=float) - design @ beta
    return pd.Series(resid, index=df.index, name=values.name)


def partial_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    covariates: Sequence[str] | None = None,
    *,
    method: Literal["pearson", "spearman"] = "spearman",
    min_n: int = 20,
) -> dict[str, object]:
    """Plain or residualized Pearson/Spearman correlation.

    For ``method="spearman"``, all variables are first rank-transformed
    and Pearson correlation is computed on the covariate-residualized
    ranks. This consolidates the Kohn/Godneva-style age/sex/BMI
    residualized association helpers and adds constant-pair guards for
    sparse omics scans.
    """
    from scipy.stats import pearsonr

    covariates = [c for c in (covariates or []) if c in df.columns and c not in {x, y}]
    cols = [x, y, *covariates]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(sub))
    record: dict[str, object] = {
        "x": x,
        "y": y,
        "covariates": ",".join(covariates),
        "method": method,
        "n": n,
        "r": float("nan"),
        "p": float("nan"),
    }
    if n < min_n:
        return record
    if method == "spearman":
        sub = sub.rank()
    elif method != "pearson":
        raise ValueError(f"unknown method: {method!r}")

    if sub[x].nunique(dropna=True) < 2 or sub[y].nunique(dropna=True) < 2:
        return record
    if covariates:
        for cov in covariates:
            if sub[cov].nunique(dropna=True) < 2:
                return record
        rx = residualize(sub[x], sub[covariates])
        ry = residualize(sub[y], sub[covariates])
        common = rx.index.intersection(ry.index)
        rx = rx.loc[common]
        ry = ry.loc[common]
    else:
        rx = sub[x]
        ry = sub[y]
    if len(rx) < min_n or rx.nunique(dropna=True) < 2 or ry.nunique(dropna=True) < 2:
        record["n"] = int(len(rx))
        return record
    stat = pearsonr(rx, ry)
    record.update({"n": int(len(rx)), "r": float(stat.statistic), "p": float(stat.pvalue)})
    return record


def partial_spearman(
    df: pd.DataFrame,
    x: str,
    y: str,
    covariates: Sequence[str] | None = None,
    *,
    min_n: int = 20,
) -> tuple[float, float, int]:
    """Return ``(rho, p, n)`` for Spearman correlation adjusted for covariates.

    Thin public wrapper over :func:`partial_correlation`, matching the
    tuple-returning Kohn/Pellow run-local helpers while using the shared
    rank-residualization implementation.
    """

    rec = partial_correlation(
        df,
        x=x,
        y=y,
        covariates=covariates,
        method="spearman",
        min_n=min_n,
    )
    return float(rec["r"]), float(rec["p"]), int(rec["n"])


def correlation_scan(
    df: pd.DataFrame,
    *,
    target: str,
    features: Sequence[str] | None = None,
    covariates: Sequence[str] | None = None,
    method: Literal["pearson", "spearman"] = "spearman",
    min_n: int = 20,
    q_col: str = "q",
) -> pd.DataFrame:
    """Scan one target against many features with optional covariates.

    Returns one row per feature with ``n``, ``r``, ``p``, and BH-FDR
    ``q``. This is the generic version of repeated sparse omics and
    sleep/body-system correlation loops in paper replications.
    """
    if features is None:
        skip = {target, *(covariates or [])}
        features = [c for c in df.select_dtypes(include=["number", "bool"]).columns if c not in skip]
    rows = [
        partial_correlation(
            df,
            x=feature,
            y=target,
            covariates=covariates,
            method=method,
            min_n=min_n,
        )
        for feature in features
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.rename(columns={"x": "feature", "y": "target"})
    out[q_col] = bh_fdr(out["p"].fillna(1.0).to_numpy(), monotonic=True)
    return out.sort_values([q_col, "p", "feature"]).reset_index(drop=True)


def spearman_matrix_scan(
    left: pd.DataFrame,
    right: pd.DataFrame | pd.Series,
    *,
    left_features: Sequence[str] | None = None,
    right_features: Sequence[str] | None = None,
    min_n: int = 20,
    q_col: str = "q",
) -> pd.DataFrame:
    """Pairwise Spearman scan between two numeric feature blocks.

    Consolidates repeated sparse omics and lipid/clinical scans. The
    two inputs are inner-joined on their index; constant pairs and pairs
    below ``min_n`` produce NaN statistics rather than raising.
    """

    from scipy.stats import spearmanr

    ldf = left.copy()
    rdf = right.to_frame() if isinstance(right, pd.Series) else right.copy()
    if left_features is None:
        left_features = list(ldf.select_dtypes(include=["number", "bool"]).columns)
    if right_features is None:
        right_features = list(rdf.select_dtypes(include=["number", "bool"]).columns)
    joined = ldf[list(left_features)].join(rdf[list(right_features)], how="inner", rsuffix="__right")
    rows: list[dict[str, object]] = []
    for lcol in left_features:
        for rcol in right_features:
            lkey = lcol
            rkey = rcol if rcol not in ldf.columns else f"{rcol}__right"
            pair = joined[[lkey, rkey]].replace([np.inf, -np.inf], np.nan).dropna()
            rec = {
                "left": lcol,
                "right": rcol,
                "n": int(len(pair)),
                "rho": float("nan"),
                "p": float("nan"),
            }
            if (
                len(pair) >= min_n
                and pair[lkey].nunique(dropna=True) >= 2
                and pair[rkey].nunique(dropna=True) >= 2
            ):
                stat = spearmanr(pair[lkey], pair[rkey])
                rec["rho"] = float(stat.statistic)
                rec["p"] = float(stat.pvalue)
            rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out[q_col] = bh_fdr(out["p"].fillna(1.0).to_numpy(), monotonic=True)
        out = out.sort_values([q_col, "p", "left", "right"]).reset_index(drop=True)
    return out


def standardized_ols(
    data: pd.DataFrame,
    *,
    y: str,
    x: str,
    covariates: Sequence[str] | None = None,
    categorical_covariates: Sequence[str] | None = None,
    min_n: int = 30,
) -> dict[str, object] | None:
    """OLS coefficient for z-scored ``y`` and ``x`` plus covariates.

    Numeric covariates enter on their native scale; named categorical
    covariates are one-hot encoded with the first level dropped. Returns
    ``None`` when the frame is too small or either ``x``/``y`` is
    constant.
    """
    import statsmodels.api as sm

    covariates = [c for c in (covariates or []) if c in data.columns and c not in {x, y}]
    categorical_covariates = set(categorical_covariates or [])
    cols = [y, x, *covariates]
    df = data[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < min_n:
        return None
    yz = zscore_series(df[y])
    xz = zscore_series(df[x])
    if yz is None or xz is None:
        return None
    design = pd.DataFrame({"__x": xz}, index=df.index)
    for cov in covariates:
        if cov in categorical_covariates or not pd.api.types.is_numeric_dtype(df[cov]):
            dummies = pd.get_dummies(df[cov], prefix=cov, drop_first=True)
            design = design.join(dummies.astype(float))
        else:
            design[cov] = pd.to_numeric(df[cov], errors="coerce")
    design = design.replace([np.inf, -np.inf], np.nan).dropna()
    yz = yz.loc[design.index]
    if len(design) < min_n:
        return None
    fit = sm.OLS(yz, sm.add_constant(design, has_constant="add")).fit()
    return {
        "y": y,
        "x": x,
        "covariates": ",".join(covariates),
        "n": int(len(design)),
        "beta": float(fit.params.get("__x", np.nan)),
        "se": float(fit.bse.get("__x", np.nan)),
        "p": float(fit.pvalues.get("__x", np.nan)),
    }


def bootstrapped_partial_corr(
    df: pd.DataFrame,
    x: str,
    y: str,
    covar: Sequence[str],
    *,
    method: str = "spearman",
    n_bootstrap: int = 10_000,
    random_seed: int = 42,
    n_jobs: int = 1,
) -> dict:
    """Bootstrapped partial correlation with empirical p-values.

    Promoted from Carletti 2025 ``correlations.py`` (paper's adapted
    bootstrapped partial-Spearman routine). Returns the observed
    partial-r, percentile CI from the bootstrap distribution, and an
    empirical two-sided p-value computed from the null distribution
    (``samples - r_obs``).

    Parameters
    ----------
    df:
        Wide frame holding ``x``, ``y``, and every ``covar`` column.
    x, y:
        Column names for the partial correlation.
    covar:
        Covariate columns to partial out.
    method:
        Passed through to :func:`pingouin.partial_corr`; default
        ``"spearman"`` matches Carletti 2025.
    n_bootstrap:
        Number of bootstrap resamples (paper default: 10 000).
    random_seed:
        Seed for reproducibility.
    n_jobs:
        Parallel workers for the bootstrap loop (1 = serial).

    Returns
    -------
    dict
        ``{"x", "y", "n", "r_observed", "r_ci_low", "r_ci_high",
        "r_median", "r_mean", "pvalue_twosided", "pvalue_less",
        "pvalue_greater", "n_invalid_iterations"}``.
    """
    import sklearn.utils  # local to avoid hard dep at import time
    from joblib import Parallel, delayed
    from pingouin import partial_corr

    def _iteration(seed: int) -> float:
        resampled = sklearn.utils.resample(df, random_state=seed)
        try:
            res = partial_corr(
                resampled, x=x, y=y, covar=list(covar), method=method,
                alternative="two-sided",
            )
            return float(res["r"].iloc[0])
        except Exception:
            return np.nan

    seeds = random_seed + np.arange(n_bootstrap)
    samples = np.asarray(
        Parallel(n_jobs=n_jobs)(delayed(_iteration)(int(s)) for s in seeds)
    )
    mask = np.isnan(samples)
    valid = samples[~mask]

    observed = partial_corr(
        df, x=x, y=y, covar=list(covar), method=method,
        alternative="two-sided",
    )
    r_obs = float(observed["r"].iloc[0])
    n_obs = int(observed["n"].iloc[0])

    # Empirical p-value from the centered bootstrap (samples - r_obs).
    null = valid - r_obs
    eps = max(1e-14, abs(1e-14 * r_obs))
    adj = 1
    n_null = len(null)
    less = ((null <= r_obs + eps).sum() + adj) / (n_null + adj)
    greater = ((null >= r_obs - eps).sum() + adj) / (n_null + adj)
    two_sided = min(less, greater) * 2

    return {
        "x": x,
        "y": y,
        "n": n_obs,
        "r_observed": r_obs,
        "r_ci_low": float(np.quantile(valid, 0.025)) if valid.size else float("nan"),
        "r_ci_high": float(np.quantile(valid, 0.975)) if valid.size else float("nan"),
        "r_median": float(np.median(valid)) if valid.size else float("nan"),
        "r_mean": float(np.mean(valid)) if valid.size else float("nan"),
        "pvalue_twosided": float(two_sided),
        "pvalue_less": float(less),
        "pvalue_greater": float(greater),
        "n_invalid_iterations": int(mask.sum()),
    }


def quartile_split_mann_whitney(
    df: pd.DataFrame,
    *,
    value_col: str,
    target_col: str,
    sex_col: str | None = None,
    rank_method: str = "average",
) -> dict:
    """Rank-based baseline-quartile split + Mann-Whitney top-vs-bottom test.

    For each participant, ranks `value_col` (e.g. baseline HbA1c) into
    quartiles Q1 (lowest) through Q4 (highest) via percentile rank, then
    tests whether `target_col` (e.g. 2-yr Delta HbA1c) differs between
    Q4 and Q1 with a two-sided Mann-Whitney U.

    Rank-based binning avoids the value-cutpoint degeneracies that hit
    lab-rounded inputs at small n (FIXES #7 in the Lutsker 2026
    GluFormer replication caught this — at n~200 with HbA1c resolution
    0.1%, many pts tied on quartile cutpoints, distorting the contrast).

    Promoted from Lutsker 2026 GluFormer replication
    (`research_runs/paper-replicate/lutsker_2026_a_*/src/
    prediabetes_quartile.py`). Validated by paper Fig 3a baseline arm
    MATCH at Mann-Whitney p=2.76e-5 (paper *** threshold le 1e-4).

    Parameters
    ----------
    df:
        DataFrame with at least `value_col` and `target_col`. Rows with
        NaN in either column are dropped.
    value_col:
        Column to rank into quartiles (the "baseline" stratifier).
    target_col:
        Column to test for differences across quartiles (the "outcome").
    sex_col:
        Optional column name (boolean / categorical). When provided,
        also computes sex-stratified results.

    Returns
    -------
    dict with keys:
        n, q1_n, q4_n, q1_median, q4_median, q1_mean, q4_mean,
        mann_whitney_u, mann_whitney_p, target_median,
        per_quartile (DataFrame), per_sex (DataFrame, if sex_col).
    """
    from scipy.stats import mannwhitneyu

    sub = df[[value_col, target_col] + ([sex_col] if sex_col else [])].dropna()
    if len(sub) < 8:
        return {
            "n": int(len(sub)),
            "q1_n": 0,
            "q4_n": 0,
            "q1_median": float("nan"),
            "q4_median": float("nan"),
            "q1_mean": float("nan"),
            "q4_mean": float("nan"),
            "mann_whitney_u": float("nan"),
            "mann_whitney_p": float("nan"),
            "target_median": float("nan"),
            "per_quartile": pd.DataFrame(),
            "per_sex": pd.DataFrame(),
        }
    ranks = sub[value_col].rank(method=rank_method, pct=True)
    quartiles = pd.cut(
        ranks,
        [0, 0.25, 0.5, 0.75, 1.0],
        labels=["Q1", "Q2", "Q3", "Q4"],
        include_lowest=True,
    )
    sub = sub.assign(__q=quartiles)
    q1 = sub.loc[sub["__q"] == "Q1", target_col]
    q4 = sub.loc[sub["__q"] == "Q4", target_col]
    if len(q1) >= 5 and len(q4) >= 5:
        u, p = mannwhitneyu(q4, q1, alternative="two-sided")
    else:
        u = float("nan")
        p = float("nan")
    per_q = (sub.groupby("__q", observed=True)
                 .agg(n=(target_col, "size"),
                      median=(target_col, "median"),
                      mean=(target_col, "mean"),
                      sd=(target_col, "std"),
                      baseline_median=(value_col, "median")))
    per_sex_rows: list[dict] = []
    if sex_col is not None:
        for sex_val, group in sub.groupby(sex_col, observed=True):
            g_ranks = group[value_col].rank(method=rank_method, pct=True)
            g_q = pd.cut(g_ranks, [0, 0.25, 0.5, 0.75, 1.0],
                          labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)
            g_sub = group.assign(__q=g_q)
            gq1 = g_sub.loc[g_sub["__q"] == "Q1", target_col]
            gq4 = g_sub.loc[g_sub["__q"] == "Q4", target_col]
            if len(gq1) >= 5 and len(gq4) >= 5:
                _, gp = mannwhitneyu(gq4, gq1, alternative="two-sided")
            else:
                gp = float("nan")
            per_sex_rows.append({
                "sex": sex_val,
                "n": int(len(g_sub)),
                "q1_n": int(len(gq1)),
                "q4_n": int(len(gq4)),
                "q1_median": float(gq1.median()) if len(gq1) else float("nan"),
                "q4_median": float(gq4.median()) if len(gq4) else float("nan"),
                "mann_whitney_p": float(gp),
            })
    return {
        "n": int(len(sub)),
        "q1_n": int(len(q1)),
        "q4_n": int(len(q4)),
        "q1_median": float(q1.median()) if len(q1) else float("nan"),
        "q4_median": float(q4.median()) if len(q4) else float("nan"),
        "q1_mean": float(q1.mean()) if len(q1) else float("nan"),
        "q4_mean": float(q4.mean()) if len(q4) else float("nan"),
        "mann_whitney_u": float(u),
        "mann_whitney_p": float(p),
        "target_median": float(sub[target_col].median()),
        "per_quartile": per_q,
        "per_sex": pd.DataFrame(per_sex_rows),
    }


def intra_vs_inter_pair_distance(
    vectors: pd.DataFrame,
    *,
    pid_col: str,
    metric: str = "cosine",
    n_inter_per_pt: int = 100,
    min_rows_per_pt: int = 2,
    seed: int = 42,
    alternative: str = "less",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Intra- vs inter-participant pairwise-distance Mann-Whitney test.

    For each participant with `>= min_rows_per_pt` row vectors in
    `vectors`, computes all pairwise intra-participant distances and a
    stratified sample of `n_inter_per_pt` inter-participant distances
    (one randomly-drawn row from this pt × one randomly-drawn row from
    a randomly-drawn different pt). Tests whether intra distances are
    smaller than inter distances via one-sided Mann-Whitney U.

    Promoted from Lutsker 2026 GluFormer replication
    (`research_runs/paper-replicate/lutsker_2026_a_*/src/
    cosine_distance.py`). Validated by paper Methods claim "intra <
    inter, Mann-Whitney p<0.001" — baseline iglu-vector arm reproduces
    at intra median d=0.23 vs inter d=0.94, p approx 0 (computational
    floor).

    Reusable for any HPP foundation-model evaluation where intra-pt
    embedding tightness is the claim (CGM, ECG, retinal, sleep).

    Parameters
    ----------
    vectors:
        Frame with one row per (pt, time) embedding. Numeric columns
        are the embedding dimensions; `pid_col` carries the participant
        ID. Inputs are typically standardized first (mean 0, var 1) so
        cosine distances are comparable across cohorts.
    pid_col:
        Name of the participant-id column.
    metric:
        Any `scipy.spatial.distance` metric name. Default `"cosine"`.
    n_inter_per_pt:
        Number of inter-pair distances to sample per participant.
        Paper-stated default is 100.
    min_rows_per_pt:
        Minimum row vectors per pt to include in the intra computation.
        Pts below this threshold contribute only to the inter pool.
    seed:
        Random seed for the stratified inter-sample.
    alternative:
        Mann-Whitney alternative; default `"less"` (intra < inter).

    Returns
    -------
    (intra_df, inter_df, mann_whitney_p):
        intra_df: per-row (pid, d) frame of all intra-pt pairwise distances.
        inter_df: per-row (pid, other_pid, d) frame of sampled inter pairs.
        mann_whitney_p: one-sided Mann-Whitney U p-value (intra < inter
        by default).
    """
    from scipy.spatial.distance import cdist
    from scipy.stats import mannwhitneyu

    feat_cols = [c for c in vectors.columns if c != pid_col]
    rng = np.random.default_rng(seed)
    pid_to_V: dict = {}
    for pid, sub in vectors.groupby(pid_col):
        V = sub[feat_cols].to_numpy(dtype=float)
        pid_to_V[pid] = V

    # Intra pairs
    intra_rows: list[dict] = []
    for pid, V in pid_to_V.items():
        if V.shape[0] < min_rows_per_pt:
            continue
        D = cdist(V, V, metric=metric)
        n = V.shape[0]
        iu = np.triu_indices(n, k=1)
        for d in D[iu]:
            intra_rows.append({"pid": pid, "d": float(d)})
    intra_df = pd.DataFrame(intra_rows)

    # Inter pairs — stratified sample
    inter_rows: list[dict] = []
    pid_arr = np.array(list(pid_to_V.keys()))
    for pid, V in pid_to_V.items():
        if V.shape[0] == 0 or len(pid_arr) < 2:
            continue
        others = pid_arr[pid_arr != pid]
        sampled_others = rng.choice(others, size=n_inter_per_pt, replace=True)
        my_picks = V[rng.integers(0, V.shape[0], size=n_inter_per_pt)]
        for op, my_row in zip(sampled_others, my_picks):
            other_V = pid_to_V[op]
            other_row = other_V[rng.integers(0, other_V.shape[0])]
            d = float(cdist(my_row.reshape(1, -1),
                             other_row.reshape(1, -1),
                             metric=metric)[0, 0])
            inter_rows.append({"pid": pid, "other_pid": op, "d": d})
    inter_df = pd.DataFrame(inter_rows)

    if len(intra_df) >= 30 and len(inter_df) >= 30:
        _, p = mannwhitneyu(intra_df["d"], inter_df["d"], alternative=alternative)
        return intra_df, inter_df, float(p)
    return intra_df, inter_df, float("nan")
