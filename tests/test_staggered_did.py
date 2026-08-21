"""Tests for the staggered-adoption DiD (Callaway & Sant'Anna) namespace."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from twosample_means.quasi_experimental import (
    CallawaySantAnna,
    StaggeredDidResult,
    render_staggered_did_markdown,
)


def make_staggered_panel(
    *,
    units: int = 180,
    periods: int = 5,
    effect: float = 2.0,
    anticipation: int = 0,
    seed: int = 7,
    cluster_time_sd: float = 0.0,
    violate_trends: bool = False,
) -> pd.DataFrame:
    """Synthetic staggered panel with never-treated comparison units.

    Cohorts are ordered by unit index: units [0, 80) are never treated,
    [80, 130) are first treated in period 2, and [130, 180) in period 3.
    ``anticipation`` shifts the treated indicator so outcomes respond one
    period early, and ``violate_trends`` adds a cohort-3-specific linear
    drift before treatment to break parallel trends.
    """
    rng = np.random.default_rng(seed)
    cohort_of = ["never"] * 80 + [2] * 50 + [3] * 50
    time_effects = np.asarray([0.5 * period for period in range(periods)])
    cluster_count = units // 10
    cluster_time = rng.normal(0.0, cluster_time_sd, (cluster_count, periods))
    rows: list[dict[str, object]] = []
    for unit in range(units):
        unit_fe = rng.normal(0.0, 2.0)
        group = cohort_of[unit]
        cluster = unit // 10
        drift = 0.35 if (violate_trends and group == 3) else 0.0
        for period in range(periods):
            treated = int(
                group != "never" and period >= int(group) - anticipation
            )
            outcome = (
                unit_fe
                + time_effects[period]
                + cluster_time[cluster, period]
                + drift * period
                + effect * treated
                + rng.normal(0.0, 0.4)
            )
            rows.append(
                {
                    "store_id": unit,
                    "cluster": cluster,
                    "period": period,
                    "cohort": group,
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)


def base_spec(**overrides: Any) -> CallawaySantAnna:
    """A staggered DiD specification with unit-level clustering."""
    kwargs: dict[str, Any] = {
        "outcome": "outcome",
        "unit": "store_id",
        "time": "period",
        "group": "cohort",
    }
    kwargs.update(overrides)
    return CallawaySantAnna(**kwargs)


class TestStaggeredEstimation:
    def test_recovers_staggered_effects(self) -> None:
        """Group-time, group, calendar, and overall ATTs recover 2.0."""
        result = base_spec().fit(make_staggered_panel())

        assert isinstance(result, StaggeredDidResult)
        assert result.method == "callaway_santanna"
        overall = result.overall_att
        assert 1.8 < overall.att < 2.2
        assert overall.ci_lower < 2.0 < overall.ci_upper
        assert overall.p_value < 0.05
        assert len(result.group_labels) == 2
        assert result.group_labels == ("2", "3")
        for group in result.group_atts:
            assert 1.6 < group.att < 2.4
        for calendar in result.calendar_atts:
            assert 1.5 < calendar.att < 2.5
        post_cells = [
            cell for cell in result.group_time_atts if cell.relative_time >= 0
        ]
        assert post_cells
        assert all(1.5 < cell.att < 2.5 for cell in post_cells)

    def test_clean_pre_cells_pass_placebo(self) -> None:
        """Pre-treatment cells are near zero and the placebo test passes."""
        result = base_spec().fit(make_staggered_panel(seed=7))

        pre_cells = [
            cell for cell in result.group_time_atts if cell.relative_time < 0
        ]
        assert pre_cells
        assert all(abs(cell.att) < 0.5 for cell in pre_cells)
        assert result.placebo is not None
        assert result.placebo.p_value > 0.05
        assert result.status == "ok"

    def test_placebo_rejects_violated_trends(self) -> None:
        """A cohort-specific pre-trend fails the parallel-trends placebo."""
        result = base_spec().fit(make_staggered_panel(violate_trends=True))

        assert result.placebo is not None
        assert result.placebo.p_value < 0.05
        assert result.status == "warning"
        assert any("placebo" in warning for warning in result.warnings)

    def test_anticipation_window_is_estimated_and_clean(self) -> None:
        """Anticipated effects appear at relative time -1, not the placebo."""
        result = base_spec(anticipation=1).fit(
            make_staggered_panel(anticipation=1, seed=3)
        )

        overall = result.overall_att
        assert 1.7 < overall.att < 2.3
        event_by_time = {
            event.relative_time: event for event in result.event_time_atts
        }
        assert event_by_time[-1].att == pytest.approx(2.0, abs=0.4)
        # Clean pre cells are more than one period before onset.
        for relative in (
            relative for relative in event_by_time if relative < -1
        ):
            assert abs(event_by_time[relative].att) < 0.5
        assert result.placebo is not None
        assert result.placebo.p_value > 0.05

    def test_cluster_robust_se_exceeds_naive(self) -> None:
        """Cohort-aligned cluster shocks inflate the robust SE over naive."""
        result = base_spec(cluster="cluster").fit(
            make_staggered_panel(
                cluster_time_sd=1.5,
                seed=21,
            )
        )

        overall = result.overall_att
        assert overall.standard_error > overall.naive_standard_error

    def test_first_period_cohort_is_dropped_with_warning(self) -> None:
        """Cohorts without a clean base period are dropped, not used."""
        frame = make_staggered_panel()
        frame.loc[frame["store_id"] < 40, "cohort"] = 0

        result = base_spec().fit(frame)

        assert result.group_labels == ("2", "3")
        assert any("no clean" in warning for warning in result.warnings)
        assert result.status == "warning"


class TestStaggeredValidation:
    def test_requires_balanced_panel(self) -> None:
        """Every unit must appear exactly once in every period."""
        frame = make_staggered_panel()
        frame = frame.drop(index=5)

        with pytest.raises(ValueError, match="balanced panel"):
            base_spec().fit(frame)

    def test_rejects_unknown_group_values(self) -> None:
        """Cohort values must be period labels or the 'never' sentinel."""
        frame = make_staggered_panel()
        frame.loc[frame["store_id"] == 0, "cohort"] = 9

        with pytest.raises(ValueError, match="not a period label"):
            base_spec().fit(frame)

    def test_rejects_panels_without_treated_cohorts(self) -> None:
        """All-never-treated panels have nothing to estimate."""
        frame = make_staggered_panel()
        frame["cohort"] = "never"

        with pytest.raises(ValueError, match="no treated cohorts"):
            base_spec().fit(frame)

    def test_rejects_all_cohorts_without_base_period(self) -> None:
        """First-period-only cohorts leave no identification."""
        frame = make_staggered_panel(periods=4)
        frame["cohort"] = 0

        with pytest.raises(ValueError, match="clean pre-treatment base"):
            base_spec().fit(frame)

    def test_rejects_negative_anticipation(self) -> None:
        """Anticipation must be a non-negative integer."""
        with pytest.raises(ValueError, match="non-negative"):
            base_spec(anticipation=-1)
        with pytest.raises(ValueError, match="non-negative integer"):
            base_spec(anticipation="one")

    def test_rejects_missing_columns(self) -> None:
        """Declared columns must exist in the panel."""
        with pytest.raises(ValueError, match="missing panel columns"):
            base_spec().fit(make_staggered_panel().drop(columns=["outcome"]))


class TestStaggeredRendering:
    def test_render_contains_all_sections(self) -> None:
        """The Markdown report lists cells, aggregations, and assumptions."""
        result = base_spec().fit(make_staggered_panel())
        markdown = render_staggered_did_markdown(result)

        assert "Callaway" in markdown
        assert "Group-time ATT" in markdown
        assert "ATT by adoption group" in markdown
        assert "Calendar-time ATT" in markdown
        assert "Event-time ATT" in markdown
        assert "Overall ATT" in markdown
        assert "Parallel-trends placebo" in markdown
        assert "Identifying assumptions" in markdown
        assert "Staggered adoption" in markdown
