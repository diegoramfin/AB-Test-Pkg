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
    RatioMetricResult,
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

    with pytest.raises(ValueError, match="continuous, count, and ratio"):
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


def _clustered_ratio_data(
    cluster_effect: float = 1.5,
    slope: float = 7.0,
    clusters_per_arm: int = 12,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate numerator/denominator with within-cluster influence shifts.

    ``num = 40 + slope * shift + eps`` and ``den = 10 + shift + eps`` share
    the cluster shift. Denominators stay strictly positive and the arm ratio
    is ``R = 4``; the influence values ``(num - 4 * den) / 10`` carry
    within-cluster correlation whenever ``slope != R``.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for cluster in range(clusters_per_arm * 2):
        shift = rng.normal(0.0, cluster_effect)
        arm = "control" if cluster % 2 == 0 else "treatment"
        for unit in range(10):
            denominator = 10.0 + shift + rng.normal(0.0, 1.0)
            numerator = 40.0 + slope * shift + rng.normal(0.0, 1.0)
            rows.append(
                {
                    "user_id": f"{cluster}-{unit}",
                    "variant": arm,
                    "cluster_id": f"g{cluster}",
                    "revenue": numerator,
                    "orders": denominator,
                }
            )
    return pd.DataFrame(rows)


def _ratio_cluster_config(experiment_id: str) -> ExperimentConfig:
    """Build a cluster-robust ratio analysis plan for the test data."""
    return ExperimentConfig(
        experiment_id=experiment_id,
        unit_id="user_id",
        assignment="variant",
        control="control",
        treatments=("treatment",),
        cluster="cluster_id",
        metrics=(
            MetricSpec(
                "arpu",
                "revenue",
                "ratio",
                role="primary",
                numerator="revenue",
                denominator="orders",
            ),
        ),
    )


def test_cluster_robust_ratio_exceeds_naive_under_correlation() -> None:
    """Cluster-robust ratio SE exceeds the naive one under correlation."""
    data = _clustered_ratio_data(slope=7.0)
    metric = analyze_experiment(
        data, _ratio_cluster_config("ratio-clusters")
    ).metrics[0]

    assert isinstance(metric, RatioMetricResult)
    assert metric.method == "cluster_robust_ratio"
    assert metric.cluster_robust is True
    assert metric.clusters == 24
    assert metric.degrees_of_freedom == 22
    assert metric.standard_error is not None
    assert metric.naive_standard_error is not None
    assert metric.standard_error > metric.naive_standard_error * 2.0
    assert metric.absolute_effect is not None
    assert metric.p_value is not None
    assert 0.0 < metric.p_value < 1.0


def test_cluster_robust_ratio_matches_naive_without_cluster_correlation() -> (
    None
):
    """With slope equal to the ratio, the sandwich matches the naive SE."""
    data = _clustered_ratio_data(
        slope=4.0,
        clusters_per_arm=200,
    )
    metric = analyze_experiment(
        data, _ratio_cluster_config("ratio-iid")
    ).metrics[0]

    assert isinstance(metric, RatioMetricResult)
    assert metric.clusters == 400
    assert metric.standard_error is not None
    assert metric.naive_standard_error is not None
    # The cluster-sum sandwich has chi-square noise of ~sqrt(2/G); with
    # 400 clusters the CR1 correction dominates and the two SEs agree.
    assert metric.standard_error == pytest.approx(
        metric.naive_standard_error, rel=0.15
    )


def test_cluster_robust_ratio_needs_two_clusters_per_arm() -> None:
    """One cluster per arm cannot support sandwich inference."""
    data = _clustered_ratio_data().iloc[:20].copy()
    data["cluster_id"] = np.where(
        data["variant"] == "control", "only-control", "only-treatment"
    )

    metric = analyze_experiment(
        data, _ratio_cluster_config("ratio-few")
    ).metrics[0]

    assert metric.status == "not_estimable"
    assert any("at least 2 clusters" in warning for warning in metric.warnings)


def test_cluster_robust_ratio_warns_on_spanning_clusters() -> None:
    """Clusters observed in both arms violate nested-cluster assumptions."""
    data = _clustered_ratio_data()
    data.loc[0, "cluster_id"] = data.loc[10, "cluster_id"]

    metric = analyze_experiment(
        data, _ratio_cluster_config("ratio-span")
    ).metrics[0]

    assert isinstance(metric, RatioMetricResult)
    assert any("span both arms" in warning for warning in metric.warnings)


def test_cluster_robust_ratio_cli_end_to_end(tmp_path: Path) -> None:
    """The --cluster flag supports ratio metrics in the CLI report."""
    data_path = tmp_path / "ratio-cluster.csv"
    _clustered_ratio_data().to_csv(data_path, index=False)
    output = tmp_path / "ratio-cluster-report"

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
            "arpu=revenue/orders:ratio:primary",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    metric = report["metrics"][0]
    assert metric["method"] == "cluster_robust_ratio"
    assert metric["cluster_robust"] is True
    assert metric["clusters"] == 24
    assert metric["naive_standard_error"] is not None
    assert (output / "report.html").exists()
    assert "cluster-robust" in (output / "report.md").read_text(
        encoding="utf-8"
    )
