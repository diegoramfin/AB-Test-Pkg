"""Public experiment-level analysis orchestration."""

from __future__ import annotations

from dataclasses import asdict, replace

import pandas as pd

from .binary import estimate_binary_metric
from .config import ContrastSpec, ExperimentConfig, MetricSpec
from .continuous import estimate_continuous_metric
from .count import estimate_count_metric
from .cuped import estimate_cuped_metric
from .data import NormalizedExperimentData, normalize_experiment_data
from .diagnostics import diagnose_assignment
from .multiplicity import apply_multiplicity
from .ratio import estimate_ratio_metric
from .results import ExperimentResult, MetricResult


def analyze_experiment(
    data: pd.DataFrame,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Analyze configured metrics and predeclared arm contrasts.

    Assignment diagnostics run on the raw frame before strict normalization so
    the result records SRM and unit-integrity information. Normalization still
    raises for invalid unit-level data because metric estimates must not run on
    duplicate or ambiguous randomization units.

    A single-treatment configuration gets one implicit treatment-vs-control
    comparison. Multi-arm analyses must declare ``ExperimentConfig.contrasts``
    so the estimands and multiplicity family are explicit.
    """
    contrasts = _resolve_contrasts(config)
    assignment_diagnostics = diagnose_assignment(data, config)
    normalized = normalize_experiment_data(data, config)
    metric_results: list[MetricResult] = []
    for metric in config.metrics:
        for contrast in contrasts:
            result = _estimate_metric(
                normalized,
                config,
                metric,
                treatment=contrast.treatment,
                control=contrast.control or config.control,
            )
            if len(contrasts) > 1 or config.contrasts is not None:
                result = replace(
                    result,
                    metric_name=f"{metric.name}:{contrast.name}",
                    family=contrast.family or metric.family,
                    contrast_name=contrast.name,
                )
            metric_results.append(result)
    corrected_results = apply_multiplicity(
        tuple(metric_results),
        config.multiplicity,
        alpha=config.alpha,
        scope=config.multiplicity_scope,
    )
    return ExperimentResult(
        experiment_id=config.experiment_id,
        data_hash=normalized.data_hash,
        config=asdict(config),
        source_rows=normalized.source_rows,
        analysis_rows=normalized.analysis_rows,
        excluded_rows=normalized.excluded_rows,
        assignment_diagnostics=assignment_diagnostics,
        metrics=corrected_results,
    )


def _resolve_contrasts(config: ExperimentConfig) -> tuple[ContrastSpec, ...]:
    """Resolve explicit contrasts or the single implicit comparison."""
    if config.contrasts is not None:
        return config.contrasts
    if len(config.treatments) != 1:
        raise ValueError(
            "exactly one treatment arm is supported implicitly; multi-arm "
            "analysis requires explicit ExperimentConfig.contrasts"
        )
    return (
        ContrastSpec(
            name=f"{config.treatments[0]}_vs_{config.control}",
            treatment=config.treatments[0],
            control=config.control,
        ),
    )


def _estimate_metric(
    normalized: NormalizedExperimentData,
    config: ExperimentConfig,
    metric: MetricSpec,
    *,
    treatment: str,
    control: str,
) -> MetricResult:
    """Dispatch one declared metric to its type-specific estimator."""
    comparison_config = config
    if control != config.control:
        # Estimators orient their effect around config.control. A planned
        # arbitrary contrast uses a temporary immutable config with the
        # requested control while retaining every other analysis setting.
        all_arms = (config.control, *config.treatments)
        comparison_config = replace(
            config,
            control=control,
            treatments=tuple(arm for arm in all_arms if arm != control),
        )
    if metric.covariate is not None:
        if metric.kind != "binary" and metric.kind != "ratio":
            return estimate_cuped_metric(
                normalized, comparison_config, metric, treatment=treatment
            )
        raise ValueError(f"covariate is not valid for kind={metric.kind!r}")
    if metric.kind == "binary":
        return estimate_binary_metric(
            normalized, comparison_config, metric, treatment=treatment
        )
    if metric.kind == "continuous":
        return estimate_continuous_metric(
            normalized, comparison_config, metric, treatment=treatment
        )
    if metric.kind == "count":
        return estimate_count_metric(
            normalized, comparison_config, metric, treatment=treatment
        )
    if metric.kind == "ratio":
        return estimate_ratio_metric(
            normalized, comparison_config, metric, treatment=treatment
        )
    raise ValueError(f"Unsupported metric kind: {metric.kind!r}")
