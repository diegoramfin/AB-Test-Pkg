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
    ],
)
def test_module_importable(module_name: str) -> None:
    """Every declared module must be importable."""
    importlib.import_module(module_name)
