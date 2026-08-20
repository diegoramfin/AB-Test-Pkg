"""Tests for cluster-robust standard errors in experiment analyses."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twosample_means.__main__ import main
from twosample_means.ab_testing import (
    ContinuousMetricResult,
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
)


def _clustered_data(
    cluster_effect: float = 3.0,
    clusters_per_arm: int = 12,
    seed: int = 5,
) -> pd.DataFrame:
    """Generate arm outcomes with shared within-cluster shifts."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for cluster in range(clusters_per_arm * 2):
        shift = rng.normal(0.0, cluster_effect)
        arm = "control" if cluster % 2 == 0 else "treatment"
        for unit in range(10):
            rows.append(
                {
                    "user_id": f"{cluster}-{unit}",
                    "variant": arm,
                    "cluster_id": f"g{cluster}",
                    "outcome": shift + rng.normal(0.0, 1.0),
                }
            )
    return pd.DataFrame(rows)


def test_cluster_robust_se_handles_within_cluster_correlation() -> None:
    """Cluster-robust standard errors exceed naive ones under correlation."""
    data = _clustered_data(cluster_effect=3.0)
    config = ExperimentConfig(
        experiment_id="clusters",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        cluster="cluster_id",
        metrics=(
            MetricSpec("outcome", "outcome", "continuous", role="primary"),
        ),
    )

    metric = analyze_experiment(data, config).metrics[0]
    assert isinstance(metric, ContinuousMetricResult)

    assert metric.method == "cluster_robust"
    assert metric.cluster_robust is True
    assert metric.clusters == 24
    assert metric.degrees_of_freedom == 22
    assert metric.naive_standard_error is not None
    assert metric.standard_error is not None
    assert metric.standard_error > metric.naive_standard_error * 1.5
    assert metric.p_value is not None
    assert 0.0 < metric.p_value < 1.0


def test_cluster_robust_se_matches_naive_when_clusters_fine_grained() -> None:
    """Without within-cluster correlation the sandwich matches Welch."""
    data = _clustered_data(cluster_effect=0.0, clusters_per_arm=40)
    config = ExperimentConfig(
        experiment_id="clusters-iid",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        cluster="cluster_id",
        metrics=(MetricSpec("outcome", "outcome", "count", role="primary"),),
    )

    metric = analyze_experiment(data, config).metrics[0]
    assert isinstance(metric, ContinuousMetricResult)
    naive = metric.naive_standard_error
    assert naive is not None

    assert metric.method == "cluster_robust"
    assert metric.clusters == 80
    assert metric.standard_error == pytest.approx(naive, rel=0.10)


def test_few_clusters_returns_not_estimable() -> None:
    """Fewer than three clusters cannot support sandwich inference."""
    data = _clustered_data().iloc[:20].copy()
    data["cluster_id"] = np.resize(["a", "b"], len(data))
    config = ExperimentConfig(
        experiment_id="too-few",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        cluster="cluster_id",
        metrics=(
            MetricSpec("outcome", "outcome", "continuous", role="primary"),
        ),
    )

    metric = analyze_experiment(data, config).metrics[0]

    assert metric.status == "not_estimable"
    assert any("at least 3 clusters" in warning for warning in metric.warnings)


def test_clusters_spanning_arms_are_warned() -> None:
    """Clusters observed in both arms violate nested-cluster assumptions."""
    data = _clustered_data()
    data.loc[0, "cluster_id"] = data.loc[10, "cluster_id"]

    config = ExperimentConfig(
        experiment_id="spanning",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        cluster="cluster_id",
        metrics=(
            MetricSpec("outcome", "outcome", "continuous", role="primary"),
        ),
    )

    metric = analyze_experiment(data, config).metrics[0]

    assert any("span both arms" in warning for warning in metric.warnings)


def test_cluster_robust_rejects_binary_metrics() -> None:
    """The estimator contract covers continuous and count metrics only."""
    data = _clustered_data().assign(
        binary=np.where(_clustered_data()["outcome"] > 0, 1, 0)
    )
    config = ExperimentConfig(
        experiment_id="binary-cluster",
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        cluster="cluster_id",
        metrics=(MetricSpec("binary", "binary", "binary", role="primary"),),
    )

    with pytest.raises(ValueError, match="continuous and count"):
        analyze_experiment(data, config)


def test_cluster_robust_cli_flag_end_to_end(tmp_path: Path) -> None:
    """The --cluster flag produces a valid cluster-robust report."""
    data_path = tmp_path / "cluster.csv"
    _clustered_data().to_csv(data_path, index=False)
    output = tmp_path / "cluster-report"

    exit_code = main(
        [
            "experiment",
            str(data_path),
            "--unit-col",
            "user_id",
            "--assignment-col",
            "variant",
            "--control",
            "control",
            "--treatment",
            "treatment",
            "--cluster",
            "cluster_id",
            "--metric",
            "outcome=outcome:continuous:primary",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["metrics"][0]["method"] == "cluster_robust"
    assert report["metrics"][0]["clusters"] == 24
