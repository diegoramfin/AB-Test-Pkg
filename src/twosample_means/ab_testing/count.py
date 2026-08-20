"""Count metric estimation using unit-level means and Welch inference."""

from __future__ import annotations

from .config import ExperimentConfig, MetricSpec
from .continuous import ContinuousMetricResult, _estimate_continuous_metric
from .data import NormalizedExperimentData


def estimate_count_metric(
    data: NormalizedExperimentData,
    config: ExperimentConfig,
    metric: MetricSpec,
    treatment: str | None = None,
) -> ContinuousMetricResult:
    """Estimate a unit-level count mean difference with Welch inference.

    Counts are analyzed as numeric per-unit outcomes. This avoids silently
    treating repeated event rows as independent observations, while making
    the modeling assumption explicit: the estimand is the mean count per
    randomization unit, not a Poisson event rate.
    """
    return _estimate_continuous_metric(
        data,
        config,
        metric,
        treatment,
        expected_kind="count",
        method_name="welch_t_count",
    )
