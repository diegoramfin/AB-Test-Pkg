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
from twosample_means.data_io import LoadedData
from twosample_means.effect_size import (
    cliff_delta,
    cohens_d,
    hedges_g,
    hodges_lehmann,
    rank_biserial,
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
        a = data.sample_a
        b = data.sample_b
        data_hash = data.data_hash
        source_desc = data.source_description
    else:
        a, b = data
        data_hash = "in-memory (no hash)"
        source_desc = "in-memory arrays"
    results: list[TestResult] = []
    results.extend(_run_diagnostics(a, b, config))
    results.extend(_run_parametric(a, b, config))
    results.extend(_run_nonparametric(a, b, config))
    results.extend(_run_bayesian(a, b, config))
    results.extend(_run_effect_sizes(a, b, config))
    return RunReport(
        data_hash=data_hash,
        source_description=source_desc,
        config=_config_to_dict(config),
        results=results,
    )


def _run_diagnostics(
    a: np.ndarray, b: np.ndarray, config: RunConfig
) -> list[TestResult]:
    """Run all assumption diagnostics.

    Parameters
    ----------
    a, b:
        Sample arrays.
    config:
        Run configuration.

    Returns
    -------
    list[TestResult]
        Diagnostic results.
    """
    results: list[TestResult] = []
    for func, label in [
        (shapiro_wilk, "Shapiro-Wilk (A)"),
        (anderson_darling, "Anderson-Darling (A)"),
        (dagostino_k2, "D'Agostino K² (A)"),
    ]:
        diag = func(a, config)
        results.append(
            TestResult(
                method_name=label,
                category="diagnostic",
                citation=diag.citation,
                statistic=diag.statistic,
                p_value=diag.p_value,
                ci_lower=None,
                ci_upper=None,
                ci_level=None,
                extra={
                    "assumption_outcome": diag.assumption_outcome,
                },
                assumption_notes=diag.details,
            )
        )
    for func, label in [
        (shapiro_wilk, "Shapiro-Wilk (B)"),
        (anderson_darling, "Anderson-Darling (B)"),
        (dagostino_k2, "D'Agostino K² (B)"),
    ]:
        diag = func(b, config)
        results.append(
            TestResult(
                method_name=label,
                category="diagnostic",
                citation=diag.citation,
                statistic=diag.statistic,
                p_value=diag.p_value,
                ci_lower=None,
                ci_upper=None,
                ci_level=None,
                extra={
                    "assumption_outcome": diag.assumption_outcome,
                },
                assumption_notes=diag.details,
            )
        )
    variance_tests = [
        (levene(a, b, config), "Levene"),
        (bartlett(a, b, config), "Bartlett"),
        (brown_forsythe(a, b, config), "Brown-Forsythe"),
    ]
    for diag, label in variance_tests:
        results.append(
            TestResult(
                method_name=label,
                category="diagnostic",
                citation=diag.citation,
                statistic=diag.statistic,
                p_value=diag.p_value,
                ci_lower=None,
                ci_upper=None,
                ci_level=None,
                extra={
                    "assumption_outcome": (diag.assumption_outcome),
                },
                assumption_notes="",
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
                assumption_notes=("Outliers flagged but NOT removed."),
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
    return results


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
