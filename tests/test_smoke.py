"""Smoke test: the package imports and all modules are present."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "twosample_means",
        "twosample_means.config",
        "twosample_means.citations",
        "twosample_means.data_io",
        "twosample_means.assumptions",
        "twosample_means.frequentist_parametric",
        "twosample_means.frequentist_nonparametric",
        "twosample_means.bayesian",
        "twosample_means.effect_size",
        "twosample_means.report",
        "twosample_means.runner",
        "twosample_means.ab_testing",
        "twosample_means.ab_testing.binary",
    ],
)
def test_module_importable(module_name: str) -> None:
    """Every declared module must be importable."""
    importlib.import_module(module_name)


def test_legacy_report_module_exports_reporting_api() -> None:
    """The historical report module remains a working compatibility alias."""
    from twosample_means import report, reporting

    assert report.write_report is reporting.write_report
    assert report.render_markdown is reporting.render_markdown
