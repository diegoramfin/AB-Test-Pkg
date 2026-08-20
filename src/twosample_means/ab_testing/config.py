"""Typed specifications for randomized A/B experiment analyses."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Literal

MetricKind = Literal["binary", "continuous", "count", "ratio"]
MetricRole = Literal["primary", "secondary", "guardrail"]
MissingPolicy = Literal["exclude", "error"]
Multiplicity = Literal["none", "holm", "fdr_bh"]
MultiplicityScope = Literal["family", "global"]


@dataclass(frozen=True)
class MetricSpec:
    """Declaration of one experiment outcome metric.

    ``practical_effect`` is expressed in the metric's native absolute units:
    proportion points for binary metrics, outcome units for continuous/count
    metrics, and ratio units for ratio metrics. It is a reporting threshold,
    not a statistical decision rule.
    """

    name: str
    column: str
    kind: MetricKind
    role: MetricRole = "secondary"
    family: str = "default"
    practical_effect: float | None = None
    missing: MissingPolicy = "exclude"
    success_value: int | bool = 1
    numerator: str | None = None
    denominator: str | None = None

    def __post_init__(self) -> None:
        """Validate the metric declaration."""
        for field_name, value in (
            ("name", self.name),
            ("column", self.column),
            ("family", self.family),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.kind not in ("binary", "continuous", "count", "ratio"):
            raise ValueError(
                "kind must be 'binary', 'continuous', 'count', or 'ratio', "
                f"got {self.kind!r}"
            )
        if self.kind == "ratio":
            for field_name, ratio_field_value in (
                ("numerator", self.numerator),
                ("denominator", self.denominator),
            ):
                if not isinstance(ratio_field_value, str) or not (
                    ratio_field_value.strip()
                ):
                    raise ValueError(
                        f"ratio {field_name} must be a non-empty string"
                    )
            if self.numerator == self.denominator:
                raise ValueError("ratio numerator and denominator must differ")
        elif self.numerator is not None or self.denominator is not None:
            raise ValueError(
                "numerator and denominator are only valid for ratio metrics"
            )
        if self.role not in ("primary", "secondary", "guardrail"):
            raise ValueError(
                "role must be 'primary', 'secondary', or 'guardrail', "
                f"got {self.role!r}"
            )
        if self.missing not in ("exclude", "error"):
            raise ValueError(
                f"missing must be 'exclude' or 'error', got {self.missing!r}"
            )
        if self.practical_effect is not None and (
            not isfinite(self.practical_effect) or self.practical_effect < 0.0
        ):
            raise ValueError(
                "practical_effect must be non-negative and finite, "
                f"got {self.practical_effect}"
            )
        if self.kind == "binary":
            if isinstance(self.success_value, bool):
                return
            if not isinstance(
                self.success_value, Integral
            ) or self.success_value not in (0, 1):
                raise ValueError(
                    "binary success_value must be bool, 0, or 1, "
                    f"got {self.success_value!r}"
                )
        elif isinstance(self.success_value, bool) or self.success_value != 1:
            raise ValueError(
                "success_value is only configurable for binary metrics"
            )


@dataclass(frozen=True)
class ContrastSpec:
    """A predeclared comparison between two experiment arms."""

    name: str
    treatment: str
    control: str | None = None
    family: str | None = None

    def __post_init__(self) -> None:
        """Validate contrast labels and optional correction family."""
        for field_name, value in (
            ("name", self.name),
            ("treatment", self.treatment),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.control is not None and (
            not isinstance(self.control, str) or not self.control.strip()
        ):
            raise ValueError("control must be a non-empty string or None")
        if self.family is not None and (
            not isinstance(self.family, str) or not self.family.strip()
        ):
            raise ValueError("family must be a non-empty string or None")


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable analysis plan for a randomized two-arm experiment."""

    experiment_id: str
    unit_id: str
    assignment: str
    control: str
    treatments: tuple[str, ...]
    metrics: tuple[MetricSpec, ...]
    contrasts: tuple[ContrastSpec, ...] | None = None
    alpha: float = 0.05
    multiplicity: Multiplicity = "holm"
    multiplicity_scope: MultiplicityScope = "family"
    expected_allocation: dict[str, float] | None = None
    time_column: str | None = None
    analysis_start: str | None = None
    analysis_end: str | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        """Normalize collection inputs and validate the analysis plan."""
        object.__setattr__(self, "treatments", tuple(self.treatments))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if self.contrasts is not None:
            object.__setattr__(self, "contrasts", tuple(self.contrasts))
        if self.expected_allocation is not None:
            object.__setattr__(
                self,
                "expected_allocation",
                dict(self.expected_allocation),
            )

        for field_name, value in (
            ("experiment_id", self.experiment_id),
            ("unit_id", self.unit_id),
            ("assignment", self.assignment),
            ("control", self.control),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not self.treatments:
            raise ValueError("treatments must contain at least one arm")
        if any(
            not isinstance(arm, str) or not arm.strip()
            for arm in self.treatments
        ):
            raise ValueError("treatment labels must be non-empty strings")
        if len(set(self.treatments)) != len(self.treatments):
            raise ValueError("treatments must not contain duplicate labels")
        if self.control in self.treatments:
            raise ValueError("control must not also be a treatment")
        if not self.metrics:
            raise ValueError("metrics must contain at least one MetricSpec")
        if any(not isinstance(metric, MetricSpec) for metric in self.metrics):
            raise ValueError("metrics must contain only MetricSpec objects")
        if self.contrasts is not None:
            if not self.contrasts:
                raise ValueError(
                    "contrasts must contain at least one ContrastSpec"
                )
            if any(
                not isinstance(contrast, ContrastSpec)
                for contrast in self.contrasts
            ):
                raise ValueError(
                    "contrasts must contain only ContrastSpec objects"
                )
            contrast_names = [contrast.name for contrast in self.contrasts]
            if len(set(contrast_names)) != len(contrast_names):
                raise ValueError("contrast names must be unique")
            arms = set((self.control, *self.treatments))
            for contrast in self.contrasts:
                contrast_control = contrast.control or self.control
                if contrast.treatment not in arms:
                    raise ValueError(
                        f"contrast '{contrast.name}' has unknown treatment "
                        f"{contrast.treatment!r}"
                    )
                if contrast_control not in arms:
                    raise ValueError(
                        f"contrast '{contrast.name}' has unknown control "
                        f"{contrast_control!r}"
                    )
                if contrast_control == contrast.treatment:
                    raise ValueError(
                        f"contrast '{contrast.name}' must compare "
                        "distinct arms"
                    )
        metric_names = [metric.name for metric in self.metrics]
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("metric names must be unique")
        if sum(metric.role == "primary" for metric in self.metrics) != 1:
            raise ValueError("exactly one primary metric is required")
        reserved_columns = {self.unit_id, self.assignment}
        metric_columns = {
            column
            for metric in self.metrics
            for column in (
                metric.column,
                metric.numerator,
                metric.denominator,
            )
            if column is not None
        }
        if metric_columns.intersection(reserved_columns):
            raise ValueError(
                "metric columns must differ from unit and assignment columns"
            )
        if not isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.multiplicity not in ("none", "holm", "fdr_bh"):
            raise ValueError(
                "multiplicity must be 'none', 'holm', or 'fdr_bh', "
                f"got {self.multiplicity!r}"
            )
        if self.multiplicity_scope not in ("family", "global"):
            raise ValueError(
                "multiplicity_scope must be 'family' or 'global', "
                f"got {self.multiplicity_scope!r}"
            )
        if self.time_column is None and (
            self.analysis_start is not None or self.analysis_end is not None
        ):
            raise ValueError(
                "time_column is required when an analysis window is configured"
            )
        if self.time_column is not None and (
            not isinstance(self.time_column, str)
            or not self.time_column.strip()
        ):
            raise ValueError("time_column must be a non-empty string")
        for field_name, timestamp_value in (
            ("analysis_start", self.analysis_start),
            ("analysis_end", self.analysis_end),
        ):
            if timestamp_value is not None and not isinstance(
                timestamp_value, str
            ):
                raise ValueError(f"{field_name} must be a string or None")
        if (
            self.analysis_start is not None
            and self.analysis_end is not None
            and self.analysis_start > self.analysis_end
        ):
            raise ValueError("analysis_start must not be after analysis_end")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.expected_allocation is not None:
            labels = (self.control, *self.treatments)
            if set(self.expected_allocation) != set(labels):
                raise ValueError(
                    "expected_allocation must contain exactly control and "
                    "treatment labels"
                )
            values = tuple(self.expected_allocation[label] for label in labels)
            if any(
                not isfinite(allocation_value) or allocation_value <= 0.0
                for allocation_value in values
            ):
                raise ValueError(
                    "expected allocation values must be positive and finite"
                )
            if abs(sum(values) - 1.0) > 1e-9:
                raise ValueError("expected allocation values must sum to 1")

    @property
    def arms(self) -> tuple[str, ...]:
        """Return control followed by treatment labels."""
        return (self.control, *self.treatments)
