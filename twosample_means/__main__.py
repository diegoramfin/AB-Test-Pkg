"""CLI entry point for the two-sample testing procedure.

Usage:
    uv run python -m twosample_means [OPTIONS] CSV_PATH

    uv run python -m twosample_means data.csv \\
        --group-col "test group" \\
        --value-col "total ads" \\
        --group-a ad \\
        --group-b psa \\
        --output output/

    # Or with two separate CSV files:
    uv run python -m twosample_means \\
        --csv-a group_a.csv --col-a value \\
        --csv-b group_b.csv --col-b value \\
        --output output/

    # Or with a subsample of a large file:
    uv run python -m twosample_means data.csv \\
        --group-col "test group" \\
        --value-col "total ads" \\
        --group-a ad --group-b psa \\
        --subsample 500 \\
        --seed 42 \\
        --output output/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from twosample_means.config import InputSpec, RunConfig
from twosample_means.data_io import load
from twosample_means.reporting import render_markdown, write_report
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
    )

    spec = _build_spec(args)
    data = load(spec)

    print(
        f"Sample A: n={len(data.sample_a)}, mean={np.mean(data.sample_a):.4f}"
    )
    print(
        f"Sample B: n={len(data.sample_b)}, mean={np.mean(data.sample_b):.4f}"
    )
    print(f"Data hash: {data.data_hash[:16]}...")
    print()

    print(
        "Running full battery "
        "(diagnostics + parametric + non-parametric "
        "+ Bayesian + effect sizes)..."
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


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="twosample_means",
        description=(
            "Run the full two-sample mean-difference "
            "testing battery and generate a report."
        ),
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
        default=Path("output"),
        help="Output directory for reports (default: output/).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Also print the Markdown report to stdout.",
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

    return parser


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
        return _spec_from_single_csv(args)
    if args.csv_a is not None and args.csv_b is not None:
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
    df = pd.read_csv(args.csv)
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
    df_a = pd.read_csv(args.csv_a)
    df_b = pd.read_csv(args.csv_b)
    col_a = args.col_a
    col_b = args.col_b
    if col_a is None:
        col_a = _first_numeric_col(df_a, args.csv_a)
    if col_b is None:
        col_b = _first_numeric_col(df_b, args.csv_b)
    group_a = df_a[col_a].to_numpy()
    group_b = df_b[col_b].to_numpy()
    if args.subsample is not None:
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
