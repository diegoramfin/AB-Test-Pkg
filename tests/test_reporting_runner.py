"""Tests for reporting and runner."""

import json
from pathlib import Path

import numpy as np
import pytest

from twosample_means.config import RunConfig
from twosample_means.data_io import DataValidationError
from twosample_means.reporting import (
    RunReport,
    TestResult,
    render_json,
    render_markdown,
    write_report,
)


@pytest.fixture
def sample_report() -> RunReport:
    """A sample RunReport for testing."""
    results = [
        TestResult(
            method_name="Welch's t-test",
            category="parametric",
            citation="Welch (1947). ...",
            statistic=-2.5,
            p_value=0.015,
            ci_lower=-1.0,
            ci_upper=-0.1,
            ci_level=0.95,
            extra={"degrees_of_freedom": 95.0},
            assumption_notes="Assumes normality.",
        ),
        TestResult(
            method_name="Mann-Whitney U",
            category="nonparametric",
            citation="Mann & Whitney (1947). ...",
            statistic=1200.0,
            p_value=0.03,
            ci_lower=None,
            ci_upper=None,
            ci_level=None,
            extra={},
            assumption_notes="No normality assumption.",
        ),
    ]
    return RunReport(
        data_hash="abc123",
        source_description="test data",
        config={"alpha": 0.05, "ci_level": 0.95},
        results=results,
    )


class TestReporting:
    """Tests for the reporting module."""

    def test_markdown_contains_all_sections(
        self, sample_report: RunReport
    ) -> None:
        """Markdown contains all expected sections."""
        md = render_markdown(sample_report)
        assert "# Two-Sample Mean Difference" in md
        assert "## Data Provenance" in md
        assert "## Configuration" in md
        assert "## Parametric Tests" in md
        assert "## Non-Parametric Tests" in md
        assert "Welch" in md
        assert "Mann-Whitney" in md

    def test_markdown_contains_citations(
        self, sample_report: RunReport
    ) -> None:
        """Markdown contains citations."""
        md = render_markdown(sample_report)
        assert "Welch (1947)" in md
        assert "Mann & Whitney (1947)" in md

    def test_markdown_contains_no_decision(
        self, sample_report: RunReport
    ) -> None:
        """Markdown states no decision is made."""
        md = render_markdown(sample_report)
        assert "accept/reject" in md.lower()

    def test_json_is_valid(self, sample_report: RunReport) -> None:
        """JSON output is valid JSON."""
        j = render_json(sample_report)
        parsed = json.loads(j)
        assert parsed["data_hash"] == "abc123"
        assert len(parsed["results"]) == 2
        assert parsed["results"][0]["method_name"] == ("Welch's t-test")

    def test_numpy_json_values_are_native_and_nonfinite_values_are_null(
        self, sample_report: RunReport
    ) -> None:
        """JSON handles NumPy scalars and arrays without stringifying them."""
        report = RunReport(
            data_hash=sample_report.data_hash,
            source_description=sample_report.source_description,
            config={"array": np.array([1.0, np.nan])},
            results=[
                TestResult(
                    method_name="numpy",
                    category="diagnostic",
                    citation="citation",
                    statistic=np.float64(1.5),
                    p_value=None,
                    ci_lower=None,
                    ci_upper=None,
                    ci_level=None,
                    extra={"values": np.array([2, np.inf])},
                )
            ],
        )

        parsed = json.loads(render_json(report))

        assert parsed["config"]["array"] == [1.0, None]
        assert parsed["results"][0]["statistic"] == 1.5
        assert parsed["results"][0]["extra"]["values"] == [2, None]

    def test_nonfinite_json_values_become_null(
        self, sample_report: RunReport
    ) -> None:
        """JSON remains strict when a manually built result is non-finite."""
        report = RunReport(
            data_hash=sample_report.data_hash,
            source_description=sample_report.source_description,
            config=sample_report.config,
            results=[
                TestResult(
                    method_name="edge",
                    category="diagnostic",
                    citation="citation",
                    statistic=float("nan"),
                    p_value=None,
                    ci_lower=None,
                    ci_upper=None,
                    ci_level=None,
                )
            ],
        )
        parsed = json.loads(render_json(report))
        assert parsed["results"][0]["statistic"] is None

    def test_write_report(
        self, sample_report: RunReport, tmp_path: Path
    ) -> None:
        """write_report creates both files."""
        md_path, json_path = write_report(sample_report, tmp_path)
        assert md_path.exists()
        assert json_path.exists()
        assert md_path.suffix == ".md"
        assert json_path.suffix == ".json"
        json.loads(json_path.read_text())


class TestRunner:
    """Tests for the runner orchestrator."""

    def test_run_returns_report(self) -> None:
        """Runner returns a RunReport with all categories."""
        from twosample_means.runner import run

        rng = np.random.default_rng(42)
        a = rng.normal(5.0, 1.0, size=30)
        b = rng.normal(5.5, 1.0, size=30)
        config = RunConfig(
            mcmc_draws=100,
            mcmc_chains=2,
            permutation_iterations=100,
            bootstrap_iterations=100,
        )
        report = run((a, b), config)
        assert isinstance(report, RunReport)
        categories = {r.category for r in report.results}
        assert "diagnostic" in categories
        assert "parametric" in categories
        assert "nonparametric" in categories
        assert "bayesian" in categories
        assert "effect_size" in categories

    def test_run_includes_citations(self) -> None:
        """All results include citations."""
        from twosample_means.runner import run

        rng = np.random.default_rng(42)
        a = rng.normal(5.0, 1.0, size=30)
        b = rng.normal(5.5, 1.0, size=30)
        config = RunConfig(
            mcmc_draws=100,
            mcmc_chains=2,
            permutation_iterations=100,
            bootstrap_iterations=100,
        )
        report = run((a, b), config)
        for result in report.results:
            assert len(result.citation) > 0

    def test_constant_samples_return_structured_statuses(self) -> None:
        """Degenerate samples do not crash the full runner."""
        from twosample_means.runner import run

        config = RunConfig(
            include_bayesian=False,
            include_resampling=False,
        )
        report = run(
            (
                np.ones(4),
                np.ones(4),
            ),
            config,
        )
        assert any(
            result.status in {"skipped", "not_estimable"}
            for result in report.results
        )
        assert all(
            value not in {"nan", "inf", "-inf"}
            for result in report.results
            for value in (
                str(result.statistic).lower(),
                str(result.p_value).lower(),
            )
        )

    def test_small_samples_skip_incompatible_diagnostics(self) -> None:
        """Diagnostics with documented minimum sizes are explicit skips."""
        from twosample_means.runner import run

        report = run(
            (np.array([1.0, 2.0]), np.array([2.0, 3.0])),
            RunConfig(include_bayesian=False, include_resampling=False),
        )
        dagostino = [
            result
            for result in report.results
            if result.method_name == "D'Agostino K² (A)"
        ]
        assert len(dagostino) == 1
        assert dagostino[0].status == "skipped"
        assert dagostino[0].statistic is None

    def test_tuple_input_is_validated(self) -> None:
        """Direct runner input uses the same validation boundary as load."""
        from twosample_means.runner import run

        with pytest.raises(DataValidationError, match="non-finite"):
            run((np.array([1.0, np.nan]), np.array([1.0, 2.0])), RunConfig())

    def test_run_tuple_input_has_real_hash(self) -> None:
        """Tuple input produces a real SHA-256 hash, not a placeholder."""
        from twosample_means.runner import run

        rng = np.random.default_rng(42)
        a = rng.normal(5.0, 1.0, size=30)
        b = rng.normal(5.5, 1.0, size=30)
        config = RunConfig(
            mcmc_draws=100,
            mcmc_chains=2,
            permutation_iterations=100,
            bootstrap_iterations=100,
        )
        report = run((a, b), config)
        assert report.data_hash != "in-memory (no hash)"
        assert len(report.data_hash) == 64
        int(report.data_hash, 16)  # valid hex
