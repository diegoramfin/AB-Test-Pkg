"""Simulation-based power and minimum-detectable-effect planning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal

import numpy as np
from scipy import stats

PowerMetricKind = Literal["binary", "continuous", "count", "ratio"]


@dataclass(frozen=True)
class PowerSpec:
    """Predeclared simulation design for one two-arm metric."""

    kind: PowerMetricKind
    control: float
    effect: float
    sample_size_control: int
    sample_size_treatment: int
    replications: int = 2_000
    alpha: float = 0.05
    standard_deviation: float = 1.0
    treatment_standard_deviation: float | None = None
    denominator_mean: float = 10.0
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate simulation design parameters."""
        if self.kind not in ("binary", "continuous", "count", "ratio"):
            raise ValueError(f"unsupported power metric kind: {self.kind!r}")
        for name, value in (
            ("control", self.control),
            ("effect", self.effect),
            ("standard_deviation", self.standard_deviation),
            ("denominator_mean", self.denominator_mean),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.alpha < 1.0 or not isfinite(self.alpha):
            raise ValueError("alpha must be finite and in (0, 1)")
        for name, value in (
            ("sample_size_control", self.sample_size_control),
            ("sample_size_treatment", self.sample_size_treatment),
            ("replications", self.replications),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 2
            ):
                raise ValueError(f"{name} must be an integer >= 2")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if self.standard_deviation <= 0.0:
            raise ValueError("standard_deviation must be positive")
        if self.treatment_standard_deviation is not None and (
            not isfinite(self.treatment_standard_deviation)
            or self.treatment_standard_deviation <= 0.0
        ):
            raise ValueError(
                "treatment_standard_deviation must be positive and finite"
            )
        if self.denominator_mean <= 0.0:
            raise ValueError("denominator_mean must be positive")
        if self.kind == "binary" and not 0.0 <= self.control <= 1.0:
            raise ValueError("binary control rate must be in [0, 1]")
        if self.kind == "binary" and not (
            0.0 <= self.control + self.effect <= 1.0
        ):
            raise ValueError("binary treatment rate must be in [0, 1]")
        if self.kind == "count" and self.control < 0.0:
            raise ValueError("count control mean must be non-negative")
        if self.kind == "count" and self.control + self.effect < 0.0:
            raise ValueError("count treatment mean must be non-negative")
        if self.kind == "ratio" and self.control + self.effect < 0.0:
            raise ValueError("ratio treatment value must be non-negative")


@dataclass(frozen=True)
class PowerResult:
    """Empirical power estimate and simulation metadata."""

    kind: PowerMetricKind
    control: float
    effect: float
    treatment: float
    power: float
    rejections: int
    replications: int
    alpha: float
    seed: int


def simulate_power(spec: PowerSpec) -> PowerResult:
    """Estimate two-sided rejection power using a deterministic simulation."""
    rng = np.random.default_rng(spec.seed)
    treatment = spec.control + spec.effect
    rejections = 0
    for _ in range(spec.replications):
        control_sample, treatment_sample = _simulate_samples(rng, spec)
        p_value = _p_value(spec.kind, control_sample, treatment_sample)
        rejections += int(p_value <= spec.alpha)
    return PowerResult(
        kind=spec.kind,
        control=spec.control,
        effect=spec.effect,
        treatment=treatment,
        power=rejections / spec.replications,
        rejections=rejections,
        replications=spec.replications,
        alpha=spec.alpha,
        seed=spec.seed,
    )


def estimate_mde(
    spec: PowerSpec,
    target_power: float = 0.80,
    max_effect: float | None = None,
    iterations: int = 30,
) -> float:
    """Estimate the smallest positive effect reaching target simulated power.

    The search assumes power is non-decreasing for positive treatment effects
    and reuses the same seed at each candidate for reproducible comparisons.
    """
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in (0, 1)")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    upper = max_effect
    if upper is None:
        upper = (
            (1.0 - spec.control)
            if spec.kind == "binary"
            else max(1.0, abs(spec.control) * 2.0)
        )
    if upper <= 0.0 or not isfinite(upper):
        raise ValueError("max_effect must be positive and finite")
    upper_spec = replace(spec, effect=upper)
    if simulate_power(upper_spec).power < target_power:
        raise ValueError("max_effect does not reach target_power")
    lower = 0.0
    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        candidate = replace(spec, effect=midpoint)
        if simulate_power(candidate).power >= target_power:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def _simulate_samples(
    rng: np.random.Generator,
    spec: PowerSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one control/treatment sample pair."""
    treatment = spec.control + spec.effect
    if spec.kind == "binary":
        return (
            rng.binomial(1, spec.control, spec.sample_size_control).astype(
                float
            ),
            rng.binomial(1, treatment, spec.sample_size_treatment).astype(
                float
            ),
        )
    if spec.kind == "count":
        return (
            rng.poisson(spec.control, spec.sample_size_control).astype(float),
            rng.poisson(treatment, spec.sample_size_treatment).astype(float),
        )
    if spec.kind == "continuous":
        treatment_sd = (
            spec.treatment_standard_deviation or spec.standard_deviation
        )
        return (
            rng.normal(
                spec.control,
                spec.standard_deviation,
                spec.sample_size_control,
            ),
            rng.normal(
                treatment,
                treatment_sd,
                spec.sample_size_treatment,
            ),
        )
    control_denominator = (
        rng.poisson(
            max(spec.denominator_mean - 1.0, 0.0), spec.sample_size_control
        ).astype(float)
        + 1.0
    )
    treatment_denominator = (
        rng.poisson(
            max(spec.denominator_mean - 1.0, 0.0), spec.sample_size_treatment
        ).astype(float)
        + 1.0
    )
    control_numerator = rng.poisson(
        max(spec.control, 0.0) * control_denominator
    ).astype(float)
    treatment_numerator = rng.poisson(
        max(treatment, 0.0) * treatment_denominator
    ).astype(float)
    return (
        np.column_stack([control_numerator, control_denominator]),
        np.column_stack([treatment_numerator, treatment_denominator]),
    )


def _p_value(
    kind: PowerMetricKind,
    control: np.ndarray,
    treatment: np.ndarray,
) -> float:
    """Calculate the simulation's two-sided p-value."""
    if kind in ("continuous", "count"):
        return float(
            stats.ttest_ind(treatment, control, equal_var=False).pvalue
        )
    if kind == "binary":
        control_successes = int(np.sum(control))
        treatment_successes = int(np.sum(treatment))
        pooled = (control_successes + treatment_successes) / (
            len(control) + len(treatment)
        )
        standard_error = np.sqrt(
            pooled
            * (1.0 - pooled)
            * (1.0 / len(control) + 1.0 / len(treatment))
        )
        if standard_error == 0.0:
            return 1.0
        z_statistic = (np.mean(treatment) - np.mean(control)) / standard_error
        return float(2.0 * stats.norm.sf(abs(z_statistic)))
    control_numerator, control_denominator = control.T
    treatment_numerator, treatment_denominator = treatment.T
    control_ratio = float(
        np.mean(control_numerator) / np.mean(control_denominator)
    )
    treatment_ratio = float(
        np.mean(treatment_numerator) / np.mean(treatment_denominator)
    )
    control_influence = (
        control_numerator - control_ratio * control_denominator
    ) / np.mean(control_denominator)
    treatment_influence = (
        treatment_numerator - treatment_ratio * treatment_denominator
    ) / np.mean(treatment_denominator)
    standard_error = np.sqrt(
        np.var(control_influence, ddof=1) / len(control)
        + np.var(treatment_influence, ddof=1) / len(treatment)
    )
    if standard_error == 0.0:
        return 1.0
    return float(
        2.0
        * stats.norm.sf(
            abs((treatment_ratio - control_ratio) / standard_error)
        )
    )
