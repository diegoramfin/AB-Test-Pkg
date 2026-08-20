"""CLI entry point for the two-sample testing procedure.

Usage:
    uv run twosample-means analyze CSV_PATH [OPTIONS]
    uv run twosample-means experiment CSV_PATH [OPTIONS]
    uv run twosample-means analyze --csv-a GROUP_A.csv \\
        --csv-b GROUP_B.csv [OPTIONS]
    uv run twosample-means fetch DATASET_NAME --output CACHE_DIR

The legacy analyze command defaults to scalable analytical methods. Add
--full-battery to enable Bayesian and resampling methods, or enable them
individually. The experiment command exposes metric-family multiplicity
controls for binary, continuous, count, and ratio outcomes, including
separate-arm CSV
ingestion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from twosample_means.ab_testing import (
    ExperimentConfig,
    MetricSpec,
    analyze_experiment,
    load_separate_experiment_csvs,
)
from twosample_means.ab_testing.config import (
    MetricKind,
    MetricRole,
    MultiplicityScope,
    UnitType,
)
from twosample_means.config import (
    InputSpec,
    MissingValuePolicy,
    RunConfig,
)
from twosample_means.data_io import DataValidationError, load
from twosample_means.kaggle import (
    DATASETS,
    KaggleFetchError,
    fetch_dataset,
    get_dataset_manifest,
)
from twosample_means.reporting import (
    render_experiment_markdown,
    render_markdown,
    write_experiment_report,
    write_report,
)
from twosample_means.runner import run


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Parameters
    ----------
    argv:
        Command-line arguments (defaults to sys.argv[1:]).

    Returns
    -------
    int
        Exit code (0 = success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch":
        return _run_fetch(args)
    if args.command in ("experiment", "analyze-experiment"):
        return _run_experiment(args)

    try:
        return _run_legacy_analysis(args)
    except (
        DataValidationError,
        ValueError,
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


def _run_legacy_analysis(args: argparse.Namespace) -> int:
    """Run the legacy continuous two-sample analysis workflow."""
    config = RunConfig(
        alpha=args.alpha,
        ci_level=args.ci_level,
        hdi_mass=args.hdi_mass,
        rope_width=args.rope_width,
        rope_scale=args.rope_scale,
        mcmc_draws=args.mcmc_draws,
        mcmc_chains=args.mcmc_chains,
        permutation_iterations=args.permutation_iterations,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        include_bayesian=args.full_battery or args.include_bayesian,
        include_resampling=args.full_battery or args.include_resampling,
        missing_values=cast(MissingValuePolicy, args.missing_values),
    )

    spec = _build_spec(args)
    data = load(spec, missing_values=config.missing_values)

    print(
        f"Sample A: n={len(data.sample_a)}, mean={np.mean(data.sample_a):.4f}"
    )
    print(
        f"Sample B: n={len(data.sample_b)}, mean={np.mean(data.sample_b):.4f}"
    )
    print(f"Data hash: {data.data_hash[:16]}...")
    print()

    if config.include_bayesian and config.include_resampling:
        run_description = "full battery"
    else:
        run_description = "scalable analytical battery"
    print(
        f"Running {run_description} "
        "(diagnostics + parametric + non-parametric + effect sizes)..."
    )
    report = run(data, config)

    print(
        f"\nDone. {len(report.results)} results across "
        f"{len(set(r.category for r in report.results))} "
        "categories."
    )

    md_path, json_path = write_report(report, args.output)
    print(f"\nMarkdown report: {md_path}")
    print(f"JSON report:     {json_path}")

    if args.print_summary:
        print()
        print(render_markdown(report))

    return 0


def _run_fetch(args: argparse.Namespace) -> int:
    """Download a registered Kaggle dataset to an explicit cache directory."""
    try:
        files = fetch_dataset(args.dataset, args.output)
    except (KaggleFetchError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    manifest = get_dataset_manifest(args.dataset)
    if manifest.quality != "verified":
        print(
            f"Note: {args.dataset} is flagged as '{manifest.quality}'. "
            f"{manifest.quality_notes}",
            file=sys.stderr,
        )
    print(f"Dataset cache: {args.output}")
    for path in files:
        print(f"Downloaded: {path}")
    return 0


def _run_experiment(args: argparse.Namespace) -> int:
    """Run the experiment-level API from a user-level CSV file."""
    try:
        config = _build_experiment_config(args)
        data = _load_experiment_input(args, config)
        result = analyze_experiment(data, config)
    except (
        DataValidationError,
        ValueError,
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(
        f"Experiment {result.experiment_id}: status={result.status}; "
        f"rows={result.analysis_rows}"
    )
    try:
        md_path, json_path = write_experiment_report(result, args.output)
    except OSError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"\nMarkdown report: {md_path}")
    print(f"JSON report:     {json_path}")
    if args.print_summary:
        print()
        print(render_experiment_markdown(result))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    root_parser = argparse.ArgumentParser(
        prog="twosample-means",
        description="Terminal workflows for two-sample mean analysis.",
    )
    commands = root_parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser(
        "fetch",
        help="Download a supported Kaggle dataset into a local cache.",
    )
    fetch_parser.add_argument(
        "dataset",
        choices=sorted(DATASETS),
        help="Registered Kaggle dataset to download.",
    )
    fetch_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Required cache directory for downloaded data.",
    )
    experiment_parser = commands.add_parser(
        "experiment",
        aliases=["analyze-experiment"],
        help="Analyze declared metrics from a user-level or aggregate CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Metric syntax: NAME=COLUMN:KIND[:ROLE]. "
            "Use --metric-family NAME=FAMILY to override the family."
        ),
    )
    experiment_parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        help="CSV containing one row per randomization unit.",
    )
    experiment_parser.add_argument(
        "--csv-a",
        type=Path,
        help="Separate control-arm CSV (alternative to positional CSV).",
    )
    experiment_parser.add_argument(
        "--csv-b",
        type=Path,
        help="Separate treatment-arm CSV (requires --csv-a).",
    )
    experiment_parser.add_argument(
        "--unit-col",
        required=True,
        help="Randomization-unit ID column.",
    )
    experiment_parser.add_argument(
        "--assignment-col",
        required=True,
        help="Control/treatment assignment column.",
    )
    experiment_parser.add_argument(
        "--control",
        required=True,
        help="Control-arm label.",
    )
    experiment_parser.add_argument(
        "--treatment",
        required=True,
        help="Treatment-arm label (one treatment comparison per run).",
    )
    experiment_parser.add_argument(
        "--metric",
        action="append",
        required=True,
        metavar="NAME=COLUMN:KIND[:ROLE]",
        help=(
            "Metric declaration; repeat for each metric. KIND is binary, "
            "continuous, count, or ratio; ROLE defaults to secondary. Ratio "
            "syntax is NAME=NUMERATOR/DENOMINATOR:ratio[:ROLE]."
        ),
    )
    experiment_parser.add_argument(
        "--covariate",
        action="append",
        default=[],
        metavar="NAME=COLUMN",
        help=(
            "Enable CUPED variance reduction for a continuous or count "
            "metric using a pre-experiment column; repeat as needed."
        ),
    )
    experiment_parser.add_argument(
        "--metric-family",
        action="append",
        default=[],
        metavar="NAME=FAMILY",
        help=(
            "Assign a metric to a multiplicity family; repeat as needed. "
            "Unspecified metrics use family 'default'."
        ),
    )
    experiment_parser.add_argument(
        "--multiplicity",
        choices=["none", "holm", "fdr_bh"],
        default="holm",
        help="P-value and interval correction method (default: holm).",
    )
    experiment_parser.add_argument(
        "--multiplicity-scope",
        choices=["family", "global"],
        default="family",
        help=(
            "Correct within metric families or globally across all metrics "
            "(default: family)."
        ),
    )
    experiment_parser.add_argument(
        "--cluster",
        default=None,
        help=(
            "Cluster column for cluster-robust standard errors on "
            "continuous and count metrics."
        ),
    )
    experiment_parser.add_argument(
        "--expected-allocation",
        default=None,
        metavar="ARM=SHARE,...",
        help=(
            "Intended assignment shares, e.g. control=0.5,treatment=0.5; "
            "enables the sample-ratio mismatch test."
        ),
    )
    experiment_parser.add_argument(
        "--unit-type",
        choices=["user", "aggregate", "unknown"],
        default="user",
        help=(
            "Unit semantics for warnings and audit metadata: user, "
            "aggregate, or unknown (default: user)."
        ),
    )
    experiment_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise significance level (default: 0.05).",
    )
    experiment_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Analysis seed (default: 42).",
    )
    experiment_parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: comma).",
    )
    experiment_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Required directory for experiment reports.",
    )
    experiment_parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Also print the experiment Markdown report to stdout.",
    )

    parser = commands.add_parser(
        "analyze",
        help="Analyze one continuous metric from local CSV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input: single CSV with a group column ---
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        help="CSV file with both groups in one file.",
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Column name for the group label.",
    )
    parser.add_argument(
        "--value-col",
        default=None,
        help="Column name for the numeric values.",
    )
    parser.add_argument(
        "--group-a",
        default=None,
        help="Label for sample A in the group column.",
    )
    parser.add_argument(
        "--group-b",
        default=None,
        help="Label for sample B in the group column.",
    )

    # --- Input: two separate CSV files ---
    parser.add_argument(
        "--csv-a",
        type=Path,
        default=None,
        help="CSV file for sample A.",
    )
    parser.add_argument(
        "--col-a",
        default=None,
        help="Column name in --csv-a.",
    )
    parser.add_argument(
        "--csv-b",
        type=Path,
        default=None,
        help="CSV file for sample B.",
    )
    parser.add_argument(
        "--col-b",
        default=None,
        help="Column name in --csv-b.",
    )

    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter used for all input files (default: comma).",
    )

    # --- Subsampling ---
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Random subsample size per group (for large files).",
    )

    # --- Output ---
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Required directory for this analysis run's reports.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Also print the Markdown report to stdout.",
    )
    parser.add_argument(
        "--full-battery",
        action="store_true",
        help=(
            "Enable Bayesian and resampling methods; the default CLI mode "
            "runs scalable analytical methods only."
        ),
    )
    parser.add_argument(
        "--include-bayesian",
        action="store_true",
        help="Include Bayesian methods without enabling resampling.",
    )
    parser.add_argument(
        "--include-resampling",
        action="store_true",
        help="Include permutation and bootstrap methods.",
    )

    # --- Configuration ---
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05).",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--hdi-mass",
        type=float,
        default=0.95,
        help="HDI mass for Bayesian (default: 0.95).",
    )
    parser.add_argument(
        "--rope-width",
        type=float,
        default=0.1,
        help="ROPE half-width (default: 0.1). When "
        "--rope-scale is 'auto', this is "
        "multiplied by the pooled SD.",
    )
    parser.add_argument(
        "--rope-scale",
        choices=["auto", "fixed"],
        default="auto",
        help="ROPE scaling: 'auto' multiplies "
        "--rope-width by pooled SD (default); "
        "'fixed' uses --rope-width as raw units.",
    )
    parser.add_argument(
        "--mcmc-draws",
        type=int,
        default=2000,
        help="MCMC draws per chain (default: 2000).",
    )
    parser.add_argument(
        "--mcmc-chains",
        type=int,
        default=4,
        help="MCMC chains (default: 4).",
    )
    parser.add_argument(
        "--permutation-iterations",
        type=int,
        default=10000,
        help="Monte Carlo permutation iterations (default: 10000).",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10000,
        help="Bootstrap iterations (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--missing-values",
        choices=["error", "exclude"],
        default="error",
        help=(
            "Legacy NaN handling: fail on missing values or exclude them "
            "before analysis (default: error)."
        ),
    )

    return root_parser


def _build_experiment_config(
    args: argparse.Namespace,
) -> ExperimentConfig:
    """Build an ExperimentConfig from experiment CLI arguments."""
    if args.csv is None and (args.csv_a is None or args.csv_b is None):
        raise ValueError(
            "provide a positional experiment CSV or both --csv-a and --csv-b"
        )
    if args.csv is not None and (
        args.csv_a is not None or args.csv_b is not None
    ):
        raise ValueError(
            "choose either a positional experiment CSV or --csv-a/--csv-b"
        )
    family_overrides = _parse_metric_families(args.metric_family)
    covariate_overrides = _parse_covariates(args.covariate)
    expected_allocation = _parse_expected_allocation(args.expected_allocation)
    metrics = tuple(
        _parse_metric_definition(
            definition,
            family_overrides,
            covariate_overrides,
        )
        for definition in args.metric
    )

    metric_names = {metric.name for metric in metrics}
    unknown_families = set(family_overrides).difference(metric_names)
    if unknown_families:
        labels = ", ".join(sorted(unknown_families))
        raise ValueError(
            f"metric family configured for undeclared metric(s): {labels}"
        )
    return ExperimentConfig(
        experiment_id=(
            args.csv.stem
            if args.csv is not None
            else f"{args.csv_a.stem}_vs_{args.csv_b.stem}"
        ),
        unit_id=args.unit_col,
        assignment=args.assignment_col,
        control=args.control,
        treatments=(args.treatment,),
        metrics=metrics,
        alpha=args.alpha,
        multiplicity=args.multiplicity,
        multiplicity_scope=cast(MultiplicityScope, args.multiplicity_scope),
        unit_type=cast(UnitType, args.unit_type),
        cluster=args.cluster,
        expected_allocation=expected_allocation,
        seed=args.seed,
    )


def _parse_metric_definition(
    definition: str,
    family_overrides: dict[str, str],
    covariate_overrides: dict[str, str],
) -> MetricSpec:
    """Parse ``NAME=COLUMN:KIND[:ROLE]`` into a MetricSpec."""
    if "=" not in definition:
        raise ValueError(
            "metric declarations must use NAME=COLUMN:KIND[:ROLE]"
        )
    name, descriptor = definition.split("=", maxsplit=1)
    parts = descriptor.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"invalid metric declaration {definition!r}; expected "
            "NAME=COLUMN:KIND[:ROLE]"
        )
    name = name.strip()
    column = parts[0].strip()
    if not name or not column:
        raise ValueError("metric name and column must be non-empty")
    kind = cast(MetricKind, parts[1].strip())
    role = cast(
        MetricRole,
        parts[2].strip() if len(parts) == 3 else "secondary",
    )
    if kind == "ratio":
        ratio_columns = column.split("/")
        if len(ratio_columns) != 2 or not all(
            value.strip() for value in ratio_columns
        ):
            raise ValueError(
                "ratio metrics must use "
                "NAME=NUMERATOR/DENOMINATOR:ratio[:ROLE]"
            )
        numerator, denominator = (value.strip() for value in ratio_columns)
        return MetricSpec(
            name=name,
            column=name,
            kind=kind,
            role=role,
            family=family_overrides.get(name, "default"),
            numerator=numerator,
            denominator=denominator,
            covariate=covariate_overrides.get(name),
        )
    return MetricSpec(
        name=name,
        column=column,
        kind=kind,
        role=role,
        family=family_overrides.get(name, "default"),
        covariate=covariate_overrides.get(name),
    )


def _parse_covariates(values: list[str]) -> dict[str, str]:
    """Parse repeated ``NAME=COLUMN`` CUPED covariate overrides."""
    covariates: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("covariates must use NAME=COLUMN")
        name, column = (part.strip() for part in value.split("=", maxsplit=1))
        if not name or not column:
            raise ValueError(
                "covariate metric names and columns must be non-empty"
            )
        if name in covariates:
            raise ValueError(f"covariate specified more than once: {name!r}")
        covariates[name] = column
    return covariates


def _parse_expected_allocation(
    value: str | None,
) -> dict[str, float] | None:
    """Parse ``ARM=SHARE,...`` into an allocation dictionary."""
    if value is None:
        return None
    shares: dict[str, float] = {}
    pairs = value.split(",")
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"expected allocation {pair!r} must use ARM=SHARE"
            )
        arm, share = (part.strip() for part in pair.split("=", maxsplit=1))
        if not arm:
            raise ValueError(
                "expected allocation arm labels must be non-empty"
            )
        try:
            parsed_share = float(share)
        except ValueError as error:
            raise ValueError(
                f"expected allocation share for {arm!r} must be a number"
            ) from error
        if not np.isfinite(parsed_share) or parsed_share <= 0.0:
            raise ValueError(
                f"expected allocation share for {arm!r} must be positive"
            )
        if arm in shares:
            raise ValueError(
                f"expected allocation arm specified more than once: {arm!r}"
            )
        shares[arm] = parsed_share
    return shares


def _load_experiment_input(
    args: argparse.Namespace,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Load one combined CSV or separate control/treatment CSVs."""
    if args.csv is not None and (
        args.csv_a is not None or args.csv_b is not None
    ):
        raise ValueError(
            "choose either a positional experiment CSV or --csv-a/--csv-b"
        )
    if args.csv is None and (args.csv_a is None or args.csv_b is None):
        raise ValueError(
            "provide a positional experiment CSV or both --csv-a and --csv-b"
        )
    if args.csv is not None:
        return pd.read_csv(args.csv, sep=args.delimiter)
    assert args.csv_a is not None and args.csv_b is not None
    return load_separate_experiment_csvs(
        args.csv_a,
        args.csv_b,
        config,
        delimiter=args.delimiter,
    )


def _parse_metric_families(values: list[str]) -> dict[str, str]:
    """Parse repeated ``NAME=FAMILY`` overrides."""
    families: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("metric families must use NAME=FAMILY")
        name, family = (part.strip() for part in value.split("=", maxsplit=1))
        if not name or not family:
            raise ValueError(
                "metric family names and values must be non-empty"
            )
        if name in families:
            raise ValueError(
                f"metric family specified more than once: {name!r}"
            )
        families[name] = family
    return families


def _build_spec(args: argparse.Namespace) -> InputSpec:
    """Build an InputSpec from CLI arguments.

    Parameters
    ----------
    args:
        Parsed CLI arguments.

    Returns
    -------
    InputSpec
        The input specification for data_io.load.

    Raises
    ------
    SystemExit
        If the arguments are insufficient or invalid.
    """
    if args.csv is not None:
        if args.csv_a is not None or args.csv_b is not None:
            print(
                "Error: choose either a positional CSV or --csv-a + --csv-b, "
                "not both.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return _spec_from_single_csv(args)
    if args.csv_a is not None or args.csv_b is not None:
        if args.csv_a is None or args.csv_b is None:
            print(
                "Error: --csv-a and --csv-b must be provided together.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return _spec_from_two_csvs(args)
    print(
        "Error: provide a single CSV positional argument "
        "or --csv-a + --csv-b.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _spec_from_single_csv(
    args: argparse.Namespace,
) -> InputSpec:
    """Build InputSpec from a single CSV with a group column.

    Parameters
    ----------
    args:
        Parsed CLI arguments.

    Returns
    -------
    InputSpec
        The input specification.
    """
    if not args.group_col or not args.value_col:
        print(
            "Error: --group-col and --value-col are "
            "required when using a single CSV.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not args.group_a or not args.group_b:
        print(
            "Error: --group-a and --group-b are "
            "required when using a single CSV.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    df = pd.read_csv(args.csv, sep=args.delimiter)
    if args.group_col not in df.columns:
        print(
            f"Error: column '{args.group_col}' not found. "
            f"Available: {list(df.columns)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.value_col not in df.columns:
        print(
            f"Error: column '{args.value_col}' not found. "
            f"Available: {list(df.columns)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    group_a = df[df[args.group_col] == args.group_a][args.value_col].to_numpy()
    group_b = df[df[args.group_col] == args.group_b][args.value_col].to_numpy()
    if len(group_a) == 0:
        print(
            f"Error: no rows where {args.group_col} == '{args.group_a}'.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if len(group_b) == 0:
        print(
            f"Error: no rows where {args.group_col} == '{args.group_b}'.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.subsample is not None:
        if args.subsample <= 0:
            print("Error: --subsample must be positive.", file=sys.stderr)
            raise SystemExit(2)
        rng = np.random.default_rng(args.seed)
        group_a = rng.choice(
            group_a,
            size=min(args.subsample, len(group_a)),
            replace=False,
        )
        group_b = rng.choice(
            group_b,
            size=min(args.subsample, len(group_b)),
            replace=False,
        )
    return InputSpec(
        sample_a=group_a,
        sample_b=group_b,
    )


def _spec_from_two_csvs(
    args: argparse.Namespace,
) -> InputSpec:
    """Build InputSpec from two separate CSV files.

    Parameters
    ----------
    args:
        Parsed CLI arguments.

    Returns
    -------
    InputSpec
        The input specification.
    """
    df_a = pd.read_csv(args.csv_a, sep=args.delimiter)
    df_b = pd.read_csv(args.csv_b, sep=args.delimiter)
    col_a = args.col_a
    col_b = args.col_b
    if col_a is None:
        col_a = _first_numeric_col(df_a, args.csv_a)
    elif col_a not in df_a.columns:
        print(
            f"Error: column '{col_a}' not found in {args.csv_a}. "
            f"Available: {list(df_a.columns)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if col_b is None:
        col_b = _first_numeric_col(df_b, args.csv_b)
    elif col_b not in df_b.columns:
        print(
            f"Error: column '{col_b}' not found in {args.csv_b}. "
            f"Available: {list(df_b.columns)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    group_a = df_a[col_a].to_numpy()
    group_b = df_b[col_b].to_numpy()
    if args.subsample is not None:
        if args.subsample <= 0:
            print("Error: --subsample must be positive.", file=sys.stderr)
            raise SystemExit(2)
        rng = np.random.default_rng(args.seed)
        group_a = rng.choice(
            group_a,
            size=min(args.subsample, len(group_a)),
            replace=False,
        )
        group_b = rng.choice(
            group_b,
            size=min(args.subsample, len(group_b)),
            replace=False,
        )
    return InputSpec(
        sample_a=group_a,
        sample_b=group_b,
    )


def _first_numeric_col(df: pd.DataFrame, path: Path) -> str:
    """Find the first numeric column in a DataFrame.

    Parameters
    ----------
    df:
        The DataFrame to search.
    path:
        File path (for error messages).

    Returns
    -------
    str
        The first numeric column name.

    Raises
    ------
    SystemExit
        If no numeric column is found.
    """
    for col in df.columns:
        if df[col].dtype.kind in "iuf":
            return str(col)
    print(
        f"Error: no numeric column in {path}. "
        f"Use --col-a/--col-b to specify one.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    sys.exit(main())
