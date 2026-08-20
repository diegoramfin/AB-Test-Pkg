"""Exhaustive auditable two-sample mean-difference hypothesis testing.

This package implements a procedure for testing the difference in means
between two independent samples using an exhaustive battery of
hypothesis tests spanning three paradigms: frequentist parametric,
frequentist non-parametric, and Bayesian. Every method is a thin wrapper
over a proven library implementation (scipy, PyMC, pingouin) and is
backed by an academic citation in the citations registry. The
``twosample_means.ab_testing`` namespace adds experiment-level binary,
continuous, count, and ratio metric analysis with assignment diagnostics,
multiplicity correction, and CUPED variance reduction.

The procedure is report-only: it computes and returns evidence (test
statistics, p-values, Bayes factors, posteriors, effect sizes,
confidence intervals, assumption-check outcomes) but NEVER applies an
accept/reject decision. This is the strongest anti-p-hacking stance.
"""

__version__ = "0.2.0"
