"""Experiment-level result objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .binary import BinaryMetricResult
from .continuous import ContinuousMetricResult
from .cuped import CupedMetricResult
from .diagnostics import AssignmentDiagnostics
from .ratio import RatioMetricResult

MetricResult = (
    BinaryMetricResult
    | ContinuousMetricResult
    | RatioMetricResult
    | CupedMetricResult
)


@dataclass(frozen=True)
class ExperimentResult:
    """Complete result for one normalized experiment analysis."""

    experiment_id: str
    data_hash: str
    config: dict[str, Any]
    source_rows: int
    analysis_rows: int
    excluded_rows: int
    assignment_diagnostics: AssignmentDiagnostics
    metrics: tuple[MetricResult, ...]

    def __post_init__(self) -> None:
        """Normalize metric collections for the frozen result container."""
        object.__setattr__(self, "metrics", tuple(self.metrics))

    @property
    def status(self) -> str:
        """Return warning when assignment or metric diagnostics need review."""
        if self.assignment_diagnostics.status != "ok":
            return "warning"
        if any(metric.status != "ok" for metric in self.metrics):
            return "warning"
        return "ok"
