"""Shared validation helpers for experiment metric estimators."""

from __future__ import annotations

from twosample_means.config import RunConfig

from .config import ExperimentConfig, MetricSpec
from .data import NormalizedExperimentData


def validate_metric_inputs(
    data: NormalizedExperimentData,
    config: ExperimentConfig,
    metric: MetricSpec,
    expected_kind: str,
) -> None:
    """Validate an estimator's normalized data and metric declaration."""
    if metric.kind != expected_kind:
        raise ValueError(
            f"metric '{metric.name}' must have kind='{expected_kind}', "
            f"got {metric.kind!r}"
        )
    configured = next(
        (
            candidate
            for candidate in config.metrics
            if candidate.name == metric.name
        ),
        None,
    )
    if configured != metric:
        raise ValueError(f"metric '{metric.name}' is not declared by config")
    if data.unit_id != config.unit_id or data.assignment != config.assignment:
        raise ValueError(
            "normalized data does not match the experiment config"
        )
    if metric.name not in data.metric_names:
        raise ValueError(
            f"metric '{metric.name}' is absent from normalized data"
        )
    required_columns = (
        (metric.numerator, metric.denominator)
        if metric.kind == "ratio"
        else (metric.column,)
    )
    missing_columns = [
        column
        for column in required_columns
        if column is not None and column not in data.frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"metric input column(s) {missing_columns} are absent from data"
        )


def resolve_treatment(
    config: ExperimentConfig,
    treatment: str | None,
) -> str:
    """Resolve an explicit or unambiguous treatment arm."""
    if treatment is None:
        if len(config.treatments) != 1:
            raise ValueError(
                "treatment must be specified when multiple treatment arms "
                "exist"
            )
        return config.treatments[0]
    if treatment not in config.treatments:
        raise ValueError(
            f"unknown treatment {treatment!r}; expected {config.treatments}"
        )
    return treatment


def practical_significance(
    ci_lower: float,
    ci_upper: float,
    practical_effect: float | None,
) -> bool | None:
    """Report whether an entire interval clears an absolute threshold."""
    if practical_effect is None:
        return None
    return ci_lower > practical_effect or ci_upper < -practical_effect


def welch_run_config(config: ExperimentConfig) -> RunConfig:
    """Translate experiment alpha into the low-level Welch configuration."""
    return RunConfig(alpha=config.alpha, ci_level=1.0 - config.alpha)
