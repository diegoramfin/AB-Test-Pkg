"""Tests for the difference-in-differences quasi-experimental namespace."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from twosample_means.quasi_experimental import (
    DidResult,
    DifferenceInDifferences,
    render_did_markdown,
)


def make_panel(
    *,
    units: int = 120,
    periods: int = 4,
    effect: float = 3.0,
    cluster_time_sd: float = 1.5,
    unit_sd: float = 2.0,
    seed: int = 42,
    onset: int = 2,
    treat_clusters: bool = False,
) -> pd.DataFrame:
    """Synthetic balanced panel with unit FE, time FE, and cluster shocks.

    Stores are grouped into clusters of 10. By default treatment is
    assigned at the unit level (even stores), so cluster-by-period shocks
    cancel in the treated-minus-control contrast; pass ``treat_clusters``
    to assign whole clusters so the cluster shocks inflate the estimate's
    sampling variance and the robust standard error exceeds the naive one.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    cluster_count = units // 10
    cluster_time = rng.normal(0.0, cluster_time_sd, (cluster_count, periods))
    time_effects = np.asarray([0.5 * period for period in range(periods)])
    for unit in range(units):
        unit_fe = rng.normal(0.0, unit_sd)
        cluster = unit // 10
        if treat_clusters:
            treated = cluster % 2 == 0
        else:
            treated = unit % 2 == 0
        for period in range(periods):
            post = 1 if period >= onset else 0
            pre_score = rng.normal(0.0, 1.0)
            outcome = (
                unit_fe
                + time_effects[period]
                + cluster_time[cluster, period]
                + 0.7 * pre_score
                + (effect if treated and post else 0.0)
                + rng.normal(0.0, 0.4)
            )
            rows.append(
                {
                    "store_id": unit,
                    "region": f"r{cluster}",
                    "period": period,
                    "treated": int(treated),
                    "post": post,
                    "revenue": outcome,
                    "pre_score": pre_score,
                }
            )
    return pd.DataFrame(rows)


def base_spec(**overrides: Any) -> DifferenceInDifferences:
    """A canonical DiD specification with region clustering."""
    kwargs: dict[str, Any] = {
        "outcome": "revenue",
        "unit": "store_id",
        "time": "period",
        "treated": "treated",
        "post": "post",
        "cluster": "region",
    }
    kwargs.update(overrides)
    return DifferenceInDifferences(**kwargs)


class TestCanonicalEstimation:
    def test_did_recovers_known_effect(self) -> None:
        """The interaction effect matches the injected treatment effect."""
        result = base_spec().fit(make_panel())
        assert isinstance(result, DidResult)
        assert result.method == "difference_in_differences"
        assert result.effect == pytest.approx(3.0, abs=0.3)
        assert result.ci_lower <= 3.0 <= result.ci_upper
        assert result.p_value < 0.001
        assert result.units == 120
        assert result.pre_periods == 2
        assert result.post_periods == 2

    def test_did_null_effect_ci_covers_zero(self) -> None:
        """A no-effect panel produces a null result honestly."""
        result = base_spec().fit(make_panel(effect=0.0))
        assert result.ci_lower < 0.0 < result.ci_upper
        assert result.p_value > 0.01
        assert result.status == "ok"

    def test_did_cluster_robust_se_exceeds_naive(self) -> None:
        """Cluster treatment plus cluster shocks inflate the robust SE."""
        result = base_spec().fit(
            make_panel(cluster_time_sd=2.0, treat_clusters=True)
        )
        assert result.standard_error > result.naive_standard_error
        assert result.effect == pytest.approx(3.0, abs=0.8)

    def test_did_reports_raw_group_time_means(self) -> None:
        """The treated-by-period cell means are reported."""
        result = base_spec().fit(make_panel())
        cells = {
            (mean.treated, mean.post): (mean.mean, mean.n)
            for mean in result.group_time_means
        }
        assert (True, True) in cells and (False, False) in cells
        assert all(n == 120 for _, n in cells.values())

    def test_did_covariate_adjustment_includes_time_varying_covariate(
        self,
    ) -> None:
        """A declared covariate is included without changing the estimand."""
        data = make_panel()
        result = base_spec(covariates=("pre_score",)).fit(data)
        assert result.covariates == ("pre_score",)
        assert result.effect == pytest.approx(3.0, abs=0.4)


class TestPanelValidation:
    def test_did_rejects_staggered_timing(self) -> None:
        """post varying within a period is rejected as staggered adoption."""
        data = make_panel()
        # Treated units do not become post until period 3, so period 2 mixes
        # post=1 (control) and post=0 (treated) rows within the period.
        data.loc[(data["treated"] == 1) & (data["period"] == 2), "post"] = 0
        with pytest.raises(ValueError, match="within a period|staggered"):
            base_spec().fit(data)

    def test_did_requires_both_periods_per_unit(self) -> None:
        """Units missing pre or post observations are rejected."""
        data = make_panel()
        data = data.drop(
            data[(data["store_id"] == 1) & (data["post"] == 1)].index
        )
        with pytest.raises(ValueError, match="missing pre or post"):
            base_spec().fit(data)

    def test_did_rejects_treatment_switching(self) -> None:
        """Treatment status must be constant within a unit."""
        data = make_panel()
        # Store 1 is a control store; mark it treated in its last period.
        data.loc[
            (data["store_id"] == 1) & (data["period"] == 3), "treated"
        ] = 1
        with pytest.raises(ValueError, match="changes within a unit"):
            base_spec().fit(data)

    def test_did_requires_binary_indicators(self) -> None:
        """Non-binary treated columns are rejected with a clear message."""
        data = make_panel()
        data["treated"] = data["treated"].replace({0: "no", 1: "yes"})
        with pytest.raises(ValueError, match="binary"):
            base_spec().fit(data)

    def test_did_requires_declared_columns(self) -> None:
        """Missing columns are reported by name."""
        with pytest.raises(ValueError, match="revenue"):
            base_spec().fit(make_panel().drop(columns=["revenue"]))

    def test_did_requires_enough_clusters(self) -> None:
        """Under three clusters cannot support cluster-robust inference."""
        data = make_panel(units=20)
        with pytest.raises(ValueError, match="at least 3 clusters"):
            base_spec().fit(data)


class TestEventStudy:
    def test_event_study_recovers_pre_and_post_coefficients(self) -> None:
        """Pre coefficients are null; post coefficients track the effect."""
        result = base_spec().fit(make_panel())
        assert result.event_study is not None
        coefficients = {
            coefficient.relative_time: coefficient.coefficient
            for coefficient in result.event_study.coefficients
            if not coefficient.reference
        }
        assert coefficients[-1] == pytest.approx(0.0, abs=0.35)
        assert coefficients[1] == pytest.approx(3.0, abs=0.4)
        assert coefficients[2] == pytest.approx(3.0, abs=0.4)
        assert result.event_study.pre_trend_p_value is not None
        assert result.event_study.pre_trend_p_value > 0.05

    def test_event_study_detects_pre_trend_violation(self) -> None:
        """Differential pre-period trends are caught by the placebo test."""
        data = make_panel()
        treated_rows = data["treated"] == 1
        data.loc[treated_rows & (data["period"] == 0), "revenue"] += 1.0
        data.loc[treated_rows & (data["period"] == 1), "revenue"] += 2.0
        result = base_spec().fit(data)
        assert result.event_study is not None
        assert result.event_study.pre_trend_p_value is not None
        assert result.event_study.pre_trend_p_value < 0.05

    def test_event_study_omits_reference_period(self) -> None:
        """The last pre period is the zero reference."""
        result = base_spec().fit(make_panel())
        assert result.event_study is not None
        reference = next(
            coefficient
            for coefficient in result.event_study.coefficients
            if coefficient.reference
        )
        assert reference.relative_time == 0
        assert reference.coefficient == 0.0

    def test_event_study_requires_two_pre_periods(self) -> None:
        """A two-period panel has no parallel-trends placebo."""
        data = make_panel(periods=2, onset=1, seed=5)
        result = base_spec().fit(data)
        assert result.event_study is None
        assert any(
            "single pre period" in warning for warning in result.warnings
        )


class TestRendering:
    def test_render_did_markdown_contains_assumptions(self) -> None:
        """The rendered report lists the identifying assumptions."""
        result = base_spec().fit(make_panel())
        markdown = render_did_markdown(result)
        assert "Difference-in-Differences" in markdown
        assert "Parallel trends" in markdown
        assert "No anticipation" in markdown
        assert "Raw group-time means" in markdown
