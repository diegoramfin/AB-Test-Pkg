"""Runner: orchestrates the full testing battery.

The runner loads data, runs all assumption diagnostics, parametric
tests, non-parametric tests, Bayesian tests, and effect sizes, then
assembles a ``RunReport``. It NEVER makes an accept/reject decision.

Academic rationale
------------------
Running the full battery of tests — rather than selecting one based
on assumption checks — avoids the "garden of forking paths" problem
(Gelman & Loken, 2014). All results are reported transparently, and
the analyst interprets which to weight.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np

from twosample_means.assumptions import (
    anderson_darling,
    bartlett,
    brown_forsythe,
    dagostino_k2,
    flag_outliers,
    levene,
    shapiro_wilk,
)
from twosample_means.bayesian import bayes_factor_jzs, best
from twosample_means.citations import Citation, get_citation
from twosample_means.config import RunConfig
from twosample_means.data_io import LoadedData, validate_samples
from twosample_means.effect_size import (
    ResourceLimitError as PairwiseLimitError,
)
from twosample_means.effect_size import (
    cliff_delta,
    cohens_d,
    hedges_g,
    hodges_lehmann,
    rank_biserial,
)
from twosample_means.frequentist_nonparametric import (
    ResourceLimitError as ResamplingLimitError,
)
from twosample_means.frequentist_nonparametric import (
    bootstrap_ci,
    brunner_munzel,
    mann_whitney_u,
    permutation_test,
)
from twosample_means.frequentist_parametric import (
    MissingVarianceError,
    students_t,
    welch_t,
    z_test,
)
from twosample_means.reporting import RunReport, TestResult


def run(
    data: LoadedData | tuple[np.ndarray, np.ndarray],
    config: RunConfig,
) -> RunReport:
    """Run the full testing battery and return a RunReport.

    Parameters
    ----------
    data:
        Either a ``LoadedData`` object (from ``data_io.load``) or
        a tuple of (sample_a, sample_b) arrays.
    config:
        Run configuration.

    Returns
    -------
    RunReport
        Complete report with all test results.
    """
    if isinstance(data, LoadedData):
        a, b = validate_samples(
            data.sample_a,
            data.sample_b,
            config.missing_values,
        )
        data_hash = data.data_hash
        source_desc = data.source_description
    else:
        a, b = validate_samples(*data, missing_values=config.missing_values)
        a_arr = a
        b_arr = b
        hasher = hashlib.sha256()
        hasher.update(a_arr.tobytes())
        hasher.update(b"|shape=")
        hasher.update(str(a_arr.shape).encode())
        hasher.update(b"|dtype=float64")
        hasher.update(b_arr.tobytes())
        hasher.update(b"|shape=")
        hasher.update(str(b_arr.shape).encode())
        hasher.update(b"|dtype=float64")
        data_hash = hasher.hexdigest()
        source_desc = "in-memory arrays"
    results: list[TestResult] = []
    results.extend(_run_diagnostics(a, b, config))
    results.extend(_run_parametric(a, b, config))
    results.extend(_run_nonparametric(a, b, config))
    results.extend(_run_bayesian(a, b, config))
    results.extend(_run_effect_sizes(a, b, config))
    results = [_normalize_result(result) for result in results]
    return RunReport(
        data_hash=data_hash,
        source_description=source_desc,
        config=_config_to_dict(config),
        results=results,
    )


def _run_diagnostics(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> list[TestResult]:
    """Run assumption diagnostics with method-specific preconditions."""
    results: list[TestResult] = []
    normality_tests: list[
        tuple[Callable[..., Any], str, str, int, int | None]
    ] = [
        (
            shapiro_wilk,
            "Shapiro-Wilk",
            "shapiro",
            3,
            min(5_000, config.max_diagnostic_observations),
        ),
        (
            anderson_darling,
            "Anderson-Darling",
            "anderson_darling",
            3,
            config.max_diagnostic_observations,
        ),
        (dagostino_k2, "D'Agostino K²", "dagostino_k2", 8, None),
    ]
    for label, arr in [("A", a), ("B", b)]:
        for func, method, citation_key, min_n, max_n in normality_tests:
            results.append(
                _run_one_diagnostic(
                    func,
                    f"{method} ({label})",
                    citation_key,
                    arr,
                    config,
                    min_n=min_n,
                    max_n=max_n,
                    requires_variation=True,
                )
            )
    variance_tests: list[tuple[Callable[..., Any], str, str]] = [
        (levene, "Levene", "levene"),
        (bartlett, "Bartlett", "bartlett"),
        (brown_forsythe, "Brown-Forsythe", "brown_forsythe"),
    ]
    for func, method, citation_key in variance_tests:
        if len(a) < 3 or len(b) < 3:
            results.append(
                _skipped_result(
                    method,
                    "diagnostic",
                    _format_citation(get_citation(citation_key)),
                    "Requires at least 3 observations per sample.",
                )
            )
            continue
        if float(np.std(a)) == 0.0 and float(np.std(b)) == 0.0:
            results.append(
                _skipped_result(
                    method,
                    "diagnostic",
                    _format_citation(get_citation(citation_key)),
                    "Variance comparison is not estimable for two constant "
                    "samples.",
                )
            )
            continue
        try:
            diag = func(a, b, config)
            results.append(
                TestResult(
                    method_name=method,
                    category="diagnostic",
                    citation=diag.citation,
                    statistic=diag.statistic,
                    p_value=diag.p_value,
                    ci_lower=None,
                    ci_upper=None,
                    ci_level=None,
                    extra={"assumption_outcome": diag.assumption_outcome},
                    assumption_notes=diag.details,
                )
            )
        except (ValueError, RuntimeError) as error:
            results.append(
                _skipped_result(
                    method,
                    "diagnostic",
                    _format_citation(get_citation(citation_key)),
                    f"Diagnostic failed safely: {error}",
                    status="failed",
                )
            )
    for label, arr in [("Outliers (A)", a), ("Outliers (B)", b)]:
        outlier = flag_outliers(arr, config)
        results.append(
            TestResult(
                method_name=label,
                category="diagnostic",
                citation=outlier.citation,
                statistic=None,
                p_value=None,
                ci_lower=None,
                ci_upper=None,
                ci_level=None,
                extra={
                    "outlier_count": outlier.count,
                    "outlier_indices": list(outlier.indices),
                    "threshold": outlier.threshold_used,
                },
                assumption_notes="Outliers flagged but NOT removed.",
            )
        )
    return results


def _run_parametric(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> list[TestResult]:
    """Run all parametric tests.

    Parameters
    ----------
    a, b:
        Sample arrays.
    config:
        Run configuration.

    Returns
    -------
    list[TestResult]
        Parametric test results.
    """
    results: list[TestResult] = []
    if float(np.std(a)) == 0.0 and float(np.std(b)) == 0.0:
        for method, citation_key in (
            ("Student's t-test", "students_t"),
            ("Welch's t-test", "welch_t"),
        ):
            results.append(
                _skipped_result(
                    method,
                    "parametric",
                    _format_citation(get_citation(citation_key)),
                    "Mean comparison is not estimable for two constant "
                    "samples.",
                )
            )
    else:
        for func in [students_t, welch_t]:
            r = func(a, b, config)
            results.append(
                TestResult(
                    method_name=r.method_name,
                    category="parametric",
                    citation=r.citation,
                    statistic=r.statistic,
                    p_value=r.p_value,
                    ci_lower=r.ci_lower,
                    ci_upper=r.ci_upper,
                    ci_level=config.ci_level,
                    extra={
                        "degrees_of_freedom": r.degrees_of_freedom,
                        "mean_difference": r.mean_difference,
                    },
                    assumption_notes=r.assumption_notes,
                )
            )
    try:
        r = z_test(a, b, config)
        results.append(
            TestResult(
                method_name=r.method_name,
                category="parametric",
                citation=r.citation,
                statistic=r.statistic,
                p_value=r.p_value,
                ci_lower=r.ci_lower,
                ci_upper=r.ci_upper,
                ci_level=config.ci_level,
                extra={
                    "degrees_of_freedom": r.degrees_of_freedom,
                    "mean_difference": r.mean_difference,
                },
                assumption_notes=r.assumption_notes,
            )
        )
    except MissingVarianceError:
        z_cite = get_citation("z_test")
        results.append(
            TestResult(
                method_name="z-test",
                category="parametric",
                citation=_format_citation(z_cite),
                statistic=None,
                p_value=None,
                ci_lower=None,
                ci_upper=None,
                ci_level=None,
                extra={"skipped": True},
                assumption_notes=(
                    "Skipped: population variance not provided."
                ),
            )
        )
    return results


def _run_nonparametric(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> list[TestResult]:
    """Run all non-parametric tests.

    Parameters
    ----------
    a, b:
        Sample arrays.
    config:
        Run configuration.

    Returns
    -------
    list[TestResult]
        Non-parametric test results.
    """
    results: list[TestResult] = []
    for func in [mann_whitney_u, brunner_munzel]:
        if func is brunner_munzel and (
            float(np.std(a)) == 0.0 and float(np.std(b)) == 0.0
        ):
            results.append(
                _skipped_result(
                    "Brunner-Munzel",
                    "nonparametric",
                    _format_citation(get_citation("brunner_munzel")),
                    "Relative-effect test is not estimable for two constant "
                    "samples.",
                )
            )
            continue
        try:
            r = func(a, b, config)
            results.append(
                TestResult(
                    method_name=r.method_name,
                    category="nonparametric",
                    citation=r.citation,
                    statistic=r.statistic,
                    p_value=r.p_value,
                    ci_lower=None,
                    ci_upper=None,
                    ci_level=None,
                    extra={},
                    assumption_notes=r.assumption_notes,
                )
            )
        except (ValueError, RuntimeError) as error:
            method = (
                "Mann-Whitney U"
                if func is mann_whitney_u
                else "Brunner-Munzel"
            )
            citation_key = (
                "mann_whitney" if func is mann_whitney_u else "brunner_munzel"
            )
            results.append(
                _skipped_result(
                    method,
                    "nonparametric",
                    _format_citation(get_citation(citation_key)),
                    f"Non-parametric test failed safely: {error}",
                    status="failed",
                )
            )
    if not config.include_resampling:
        results.extend(
            [
                _skipped_result(
                    "Permutation test",
                    "nonparametric",
                    _format_citation(get_citation("permutation")),
                    "Resampling methods disabled by configuration.",
                ),
                _skipped_result(
                    "Bootstrap CI",
                    "nonparametric",
                    _format_citation(get_citation("bootstrap_ci")),
                    "Resampling methods disabled by configuration.",
                ),
            ]
        )
        return results
    try:
        perm = permutation_test(a, b, config)
        results.append(
            TestResult(
                method_name=perm.method_name,
                category="nonparametric",
                citation=perm.citation,
                statistic=perm.statistic,
                p_value=perm.p_value,
                ci_lower=None,
                ci_upper=None,
                ci_level=None,
                extra={
                    "mode": perm.mode,
                    "iterations": perm.iterations,
                    "seed": perm.seed,
                },
                assumption_notes=perm.assumption_notes,
            )
        )
    except ResamplingLimitError as error:
        results.append(
            _skipped_result(
                "Permutation test",
                "nonparametric",
                _format_citation(get_citation("permutation")),
                str(error),
            )
        )
    try:
        boot = bootstrap_ci(a, b, config)
        results.append(
            TestResult(
                method_name=boot.method_name,
                category="nonparametric",
                citation=boot.citation,
                statistic=boot.point_estimate,
                p_value=None,
                ci_lower=boot.ci_lower,
                ci_upper=boot.ci_upper,
                ci_level=boot.ci_level,
                extra={
                    "iterations": boot.iterations,
                    "seed": boot.seed,
                },
                assumption_notes="Bootstrap percentile CI.",
            )
        )
    except ResamplingLimitError as error:
        results.append(
            _skipped_result(
                "Bootstrap CI",
                "nonparametric",
                _format_citation(get_citation("bootstrap_ci")),
                str(error),
            )
        )
    return results


def _run_bayesian(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> list[TestResult]:
    """Run all Bayesian tests.

    Parameters
    ----------
    a, b:
        Sample arrays.
    config:
        Run configuration.

    Returns
    -------
    list[TestResult]
        Bayesian test results.
    """
    results: list[TestResult] = []
    if not config.include_bayesian:
        return [
            _skipped_result(
                "BEST (Kruschke)",
                "bayesian",
                _format_citation(get_citation("best")),
                "Bayesian methods disabled by configuration.",
            ),
            _skipped_result(
                "JZS Bayes factor",
                "bayesian",
                _format_citation(get_citation("bayes_factor_jzs")),
                "Bayesian methods disabled by configuration.",
            ),
        ]
    pooled = np.concatenate([a, b])
    if len(pooled) > config.max_bayesian_observations:
        reason = (
            "Bayesian methods skipped because the combined sample size "
            f"{len(pooled)} exceeds the configured limit "
            f"{config.max_bayesian_observations}."
        )
        return [
            _skipped_result(
                "BEST (Kruschke)",
                "bayesian",
                _format_citation(get_citation("best")),
                reason,
            ),
            _skipped_result(
                "JZS Bayes factor",
                "bayesian",
                _format_citation(get_citation("bayes_factor_jzs")),
                reason,
            ),
        ]
    if float(np.std(pooled, ddof=0)) == 0.0:
        reason = "Bayesian methods are not estimable for constant pooled data."
        return [
            _skipped_result(
                "BEST (Kruschke)",
                "bayesian",
                _format_citation(get_citation("best")),
                reason,
            ),
            _skipped_result(
                "JZS Bayes factor",
                "bayesian",
                _format_citation(get_citation("bayes_factor_jzs")),
                reason,
            ),
        ]
    try:
        best_result = best(a, b, config)
        results.append(
            TestResult(
                method_name=best_result.method_name,
                category="bayesian",
                citation=best_result.citation,
                statistic=best_result.posterior_mean_diff,
                p_value=None,
                ci_lower=best_result.hdi_lower,
                ci_upper=best_result.hdi_upper,
                ci_level=best_result.hdi_mass,
                extra={
                    "rope_width": best_result.rope_width,
                    "rope_scale": config.rope_scale,
                    "rope_proportion": best_result.rope_proportion,
                    "r_hat": best_result.r_hat,
                    "ess": best_result.ess,
                    "draws": best_result.draws,
                    "chains": best_result.chains,
                    "seed": best_result.seed,
                },
                assumption_notes=best_result.assumption_notes,
            )
        )
    except (ValueError, RuntimeError) as error:
        results.append(
            _skipped_result(
                "BEST (Kruschke)",
                "bayesian",
                _format_citation(get_citation("best")),
                f"BEST failed safely: {error}",
                status="failed",
            )
        )
    try:
        bf = bayes_factor_jzs(a, b, config)
        results.append(
            TestResult(
                method_name=bf.method_name,
                category="bayesian",
                citation=bf.citation,
                statistic=bf.bf10,
                p_value=None,
                ci_lower=None,
                ci_upper=None,
                ci_level=None,
                extra={
                    "bf10": bf.bf10,
                    "bf01": bf.bf01,
                    "prior_width": bf.prior_width,
                },
                assumption_notes=bf.assumption_notes,
            )
        )
    except (ValueError, RuntimeError) as error:
        results.append(
            _skipped_result(
                "JZS Bayes factor",
                "bayesian",
                _format_citation(get_citation("bayes_factor_jzs")),
                f"JZS Bayes factor failed safely: {error}",
                status="failed",
            )
        )
    return results


def _run_effect_sizes(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> list[TestResult]:
    """Run all effect-size computations.

    Parameters
    ----------
    a, b:
        Sample arrays.
    config:
        Run configuration.

    Returns
    -------
    list[TestResult]
        Effect-size results.
    """
    results: list[TestResult] = []
    for func in [
        cohens_d,
        hedges_g,
        cliff_delta,
        rank_biserial,
        hodges_lehmann,
    ]:
        try:
            r = func(a, b, config)
            results.append(
                TestResult(
                    method_name=r.method_name,
                    category="effect_size",
                    citation=r.citation,
                    statistic=r.point_estimate,
                    p_value=None,
                    ci_lower=r.ci_lower,
                    ci_upper=r.ci_upper,
                    ci_level=r.ci_level,
                    extra={},
                    assumption_notes="",
                )
            )
        except (PairwiseLimitError, ValueError, RuntimeError) as error:
            effect_metadata = {
                cohens_d: ("Cohen's d", "cohen_d"),
                hedges_g: ("Hedges' g", "hedges_g"),
                cliff_delta: ("Cliff's delta", "cliff_delta"),
                rank_biserial: ("Rank-biserial", "rank_biserial"),
                hodges_lehmann: ("Hodges-Lehmann", "hodges_lehmann"),
            }
            method_name, citation_key = effect_metadata[func]
            results.append(
                _skipped_result(
                    method_name,
                    "effect_size",
                    _format_citation(get_citation(citation_key)),
                    str(error),
                )
            )
    return results


def _run_one_diagnostic(
    func: Callable[..., Any],
    method_name: str,
    citation_key: str,
    values: np.ndarray,
    config: RunConfig,
    *,
    min_n: int,
    max_n: int | None,
    requires_variation: bool,
) -> TestResult:
    """Run one diagnostic or return a structured precondition skip."""
    citation = _format_citation(get_citation(citation_key))
    if len(values) < min_n:
        return _skipped_result(
            method_name,
            "diagnostic",
            citation,
            f"Requires at least {min_n} observations; got {len(values)}.",
        )
    if max_n is not None and len(values) > max_n:
        return _skipped_result(
            method_name,
            "diagnostic",
            citation,
            f"Supports at most {max_n} observations; got {len(values)}.",
        )
    if requires_variation and float(np.std(values)) == 0.0:
        return _skipped_result(
            method_name,
            "diagnostic",
            citation,
            "Normality is not estimable for a constant sample.",
        )
    try:
        diag = func(values, config)
    except (ValueError, RuntimeError) as error:
        return _skipped_result(
            method_name,
            "diagnostic",
            citation,
            f"Diagnostic failed safely: {error}",
            status="failed",
        )
    return TestResult(
        method_name=method_name,
        category="diagnostic",
        citation=diag.citation,
        statistic=diag.statistic,
        p_value=diag.p_value,
        ci_lower=None,
        ci_upper=None,
        ci_level=None,
        extra={"assumption_outcome": diag.assumption_outcome},
        assumption_notes=diag.details,
    )


def _skipped_result(
    method_name: str,
    category: str,
    citation: str,
    reason: str,
    *,
    status: str = "skipped",
) -> TestResult:
    """Create a reportable result for an unavailable method."""
    return TestResult(
        method_name=method_name,
        category=category,
        citation=citation,
        statistic=None,
        p_value=None,
        ci_lower=None,
        ci_upper=None,
        ci_level=None,
        extra={},
        assumption_notes=reason,
        status=status,
    )


def _contains_nonfinite(value: Any) -> bool:
    """Return whether a scalar or nested result value is non-finite."""
    if isinstance(value, float | np.floating):
        return not bool(np.isfinite(value))
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, np.ndarray):
        return _contains_nonfinite(value.tolist())
    if isinstance(value, list | tuple):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _normalize_result(result: TestResult) -> TestResult:
    """Convert numerical failures into explicit non-estimable results."""
    values = (
        result.statistic,
        result.p_value,
        result.ci_lower,
        result.ci_upper,
        result.ci_level,
        result.extra,
    )
    if result.status != "ok" or not any(
        _contains_nonfinite(value) for value in values
    ):
        return result
    return TestResult(
        method_name=result.method_name,
        category=result.category,
        citation=result.citation,
        statistic=None,
        p_value=None,
        ci_lower=None,
        ci_upper=None,
        ci_level=None,
        extra=result.extra,
        assumption_notes=(
            f"{result.assumption_notes} " if result.assumption_notes else ""
        )
        + (
            "The method returned a non-finite numerical result and was "
            "marked not_estimable."
        ),
        status="not_estimable",
    )


def _config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Convert a RunConfig to a dictionary for reporting.

    Parameters
    ----------
    config:
        Run configuration.

    Returns
    -------
    dict[str, Any]
        Configuration as a dictionary.
    """
    from dataclasses import asdict

    return asdict(config)


def _format_citation(cite: Citation) -> str:
    """Format a citation dict as a readable string.

    Parameters
    ----------
    cite:
        Citation dictionary with authors, year, title, source.

    Returns
    -------
    str
        Formatted citation.
    """
    return (
        f"{cite['authors']} ({cite['year']}). "
        + f"{cite['title']}. {cite['source']}."
    )
