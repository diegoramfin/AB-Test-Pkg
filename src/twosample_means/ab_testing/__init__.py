"""Experiment-level configuration, estimators, diagnostics, and reports.

The existing ``twosample_means`` functions remain available for low-level
continuous two-sample analysis. This namespace adds a typed experiment
boundary for two-arm binary and continuous metric analyses.
"""

from .api import analyze_experiment
from .binary import BinaryMetricResult, BinarySummary, estimate_binary_metric
from .config import (
    ContrastSpec,
    ExperimentConfig,
    MetricSpec,
    MultiplicityScope,
)
from .continuous import (
    ContinuousMetricResult,
    ContinuousSummary,
    estimate_continuous_metric,
)
from .count import estimate_count_metric
from .data import NormalizedExperimentData, normalize_experiment_data
from .diagnostics import AssignmentDiagnostics, diagnose_assignment
from .multiplicity import (
    adjust_p_values,
    apply_multiplicity,
    simultaneous_ci_levels,
)
from .power import PowerResult, PowerSpec, estimate_mde, simulate_power
from .ratio import RatioMetricResult, RatioSummary, estimate_ratio_metric
from .results import ExperimentResult
from .sequential import (
    SequentialBoundary,
    SequentialPlan,
    SequentialResult,
    alpha_spending_boundaries,
    evaluate_sequential,
)

__all__ = [
    "AssignmentDiagnostics",
    "ExperimentResult",
    "ContrastSpec",
    "RatioMetricResult",
    "RatioSummary",
    "PowerResult",
    "PowerSpec",
    "SequentialBoundary",
    "SequentialPlan",
    "SequentialResult",
    "BinaryMetricResult",
    "BinarySummary",
    "ContinuousMetricResult",
    "ContinuousSummary",
    "ExperimentConfig",
    "MetricSpec",
    "MultiplicityScope",
    "estimate_binary_metric",
    "estimate_count_metric",
    "estimate_ratio_metric",
    "estimate_mde",
    "simulate_power",
    "alpha_spending_boundaries",
    "evaluate_sequential",
    "adjust_p_values",
    "simultaneous_ci_levels",
    "analyze_experiment",
    "apply_multiplicity",
    "diagnose_assignment",
    "estimate_continuous_metric",
    "NormalizedExperimentData",
    "normalize_experiment_data",
]
