"""Academic citations registry for every statistical method.

This module maintains a mapping from method name to its academic
reference. Every public function in the battery looks up its citation
here and includes it in the returned ``TestResult``. This ensures no
method ships without an explicit, traceable academic justification — a
core auditability and anti-p-hacking requirement.

The citations are drawn from peer-reviewed statistics literature and
established reference texts. Where a method is accessed via a library
(scipy, PyMC, pingouin), the citation is the original
academic source for the method, not the library documentation.
"""

from __future__ import annotations

from typing import TypedDict


class Citation(TypedDict):
    """A single academic citation.

    Attributes
    ----------
    authors:
        Author names in the format used by the original publication.
    year:
        Publication year.
    title:
        Title of the paper or book.
    source:
        Journal name, publisher, or other publication venue.
    """

    authors: str
    year: int
    title: str
    source: str


CITATIONS: dict[str, Citation] = {
    "shapiro": {
        "authors": "Shapiro, S. S. and Wilk, M. B.",
        "year": 1965,
        "title": (
            "An analysis of variance test for "
            + "normality (complete samples)"
        ),
        "source": "Biometrika, 52(3-4), 591-611",
    },
    "anderson_darling": {
        "authors": "Anderson, T. W. and Darling, D. A.",
        "year": 1952,
        "title": (
            "Asymptotic theory of certain goodness of "
            "fit criteria based on stochastic processes"
        ),
        "source": ("Annals of Mathematical Statistics, 23(2), 193-212"),
    },
    "dagostino_k2": {
        "authors": "D'Agostino, R. B. and Pearson, E. S.",
        "year": 1973,
        "title": (
            "Tests for departure from normality. "
            "Empirical results for the distributions "
            "of b2 and sqrt(b1)"
        ),
        "source": "Biometrika, 60(3), 613-622",
    },
    "levene": {
        "authors": "Levene, H.",
        "year": 1960,
        "title": "Robust tests for equality of variances",
        "source": (
            "In Olkin et al. (Eds.), Contributions "
            "to Probability and Statistics. "
            "Stanford University Press"
        ),
    },
    "bartlett": {
        "authors": "Bartlett, M. S.",
        "year": 1937,
        "title": ("Properties of sufficiency and " + "statistical tests"),
        "source": (
            "Proceedings of the Royal Society of "
            + "London A, 160(901), 268-282"
        ),
    },
    "brown_forsythe": {
        "authors": "Brown, M. B. and Forsythe, A. B.",
        "year": 1974,
        "title": "Robust tests for the equality of variances",
        "source": (
            "Journal of the American Statistical Association, 69(346), 364-367"
        ),
    },
    "students_t": {
        "authors": "Student (Gosset, W. S.)",
        "year": 1908,
        "title": "The probable error of a mean",
        "source": "Biometrika, 6(1), 1-25",
    },
    "welch_t": {
        "authors": "Welch, B. L.",
        "year": 1947,
        "title": (
            "The generalization of 'Student's' problem "
            "when several different population variances "
            "are involved"
        ),
        "source": "Biometrika, 34(1-2), 28-35",
    },
    "z_test": {
        "authors": "Neyman, J. and Pearson, E. S.",
        "year": 1933,
        "title": (
            "On the problem of the most efficient tests "
            "of statistical hypotheses"
        ),
        "source": (
            "Philosophical Transactions of the Royal "
            "Society of London A, 231, 289-337"
        ),
    },
    "mann_whitney": {
        "authors": "Mann, H. B. and Whitney, D. R.",
        "year": 1947,
        "title": (
            "On a test of whether one of two random "
            "variables is stochastically larger than "
            "the other"
        ),
        "source": ("Annals of Mathematical Statistics, 18(1), 50-60"),
    },
    "brunner_munzel": {
        "authors": "Brunner, E. and Munzel, U.",
        "year": 2000,
        "title": (
            "The nonparametric Behrens-Fisher problem: "
            "asymptotic theory and a small-sample "
            "approximation"
        ),
        "source": "Biometrical Journal, 42(1), 17-25",
    },
    "permutation": {
        "authors": "Fisher, R. A.",
        "year": 1935,
        "title": "The Design of Experiments",
        "source": "Oliver and Boyd, Edinburgh",
    },
    "bootstrap_ci": {
        "authors": "Efron, B. and Tibshirani, R. J.",
        "year": 1993,
        "title": "An Introduction to the Bootstrap",
        "source": "Chapman & Hall/CRC, New York",
    },
    "best": {
        "authors": "Kruschke, J. K.",
        "year": 2013,
        "title": "Bayesian estimation supersedes the t test",
        "source": (
            "Journal of Experimental Psychology: General, 142(2), 573-603"
        ),
    },
    "bayes_factor_jzs": {
        "authors": (
            "Rouder, J. N., Speckman, P. L., Sun, D., "
            "Morey, R. D., and Iverson, G."
        ),
        "year": 2009,
        "title": (
            "Bayesian t tests for accepting and rejecting the null hypothesis"
        ),
        "source": ("Psychonomic Bulletin & Review, 16(2), 225-237"),
    },
    "cuped": {
        "authors": "Deng, A., Xu, Y., Kohavi, R., and Walker, T.",
        "year": 2013,
        "title": (
            "Improving the sensitivity of online controlled "
            "experiments by utilizing pre-experiment data"
        ),
        "source": "Proceedings of the Sixth ACM International "
        "Conference on Web Search and Data Mining (WSDM), 123-132",
    },
    "cohen_d": {
        "authors": "Cohen, J.",
        "year": 1988,
        "title": (
            "Statistical Power Analysis for the Behavioral Sciences (2nd ed.)"
        ),
        "source": "Lawrence Erlbaum, Hillsdale, NJ",
    },
    "hedges_g": {
        "authors": "Hedges, L. V.",
        "year": 1981,
        "title": (
            "Distribution theory for Glass's estimator "
            "of effect size and related estimators"
        ),
        "source": ("Journal of Educational Statistics, 6(2), 107-128"),
    },
    "cliff_delta": {
        "authors": "Cliff, N.",
        "year": 1993,
        "title": (
            "Dominance statistics: Ordinal analyses "
            "to answer ordinal questions"
        ),
        "source": ("Psychological Bulletin, 114(3), 494-509"),
    },
    "rank_biserial": {
        "authors": "Kerby, D. S.",
        "year": 2014,
        "title": (
            "The simple difference formula: An approach "
            "to teaching and testing the difference "
            "between two means using the rank-biserial "
            "correlation"
        ),
        "source": ("Practical Assessment, Research & Evaluation, 19(11), 1-3"),
    },
    "hodges_lehmann": {
        "authors": "Hodges, J. L. and Lehmann, E. L.",
        "year": 1963,
        "title": "Estimation of location based on rank tests",
        "source": ("Annals of Mathematical Statistics, 34(2), 598-611"),
    },
}


def get_citation(method_name: str) -> Citation:
    """Return the citation for a method, raising if not found.

    Parameters
    ----------
    method_name:
        The key into the ``CITATIONS`` registry.

    Returns
    -------
    Citation
        The academic reference for the method.

    Raises
    ------
    KeyError
        If ``method_name`` is not in the registry. This is a
        deliberate safeguard: no method should ship without
        a citation.
    """
    if method_name not in CITATIONS:
        raise KeyError(
            f"No citation registered for method "
            f"'{method_name}'. Every statistical method "
            "must have an academic reference in the "
            "citations registry."
        )
    return CITATIONS[method_name]


def format_citation(method_name: str) -> str:
    """Return a human-readable citation string for a method.

    Parameters
    ----------
    method_name:
        The key into the ``CITATIONS`` registry.

    Returns
    -------
    str
        Formatted as "Authors (Year). Title. Source."
    """
    cite = get_citation(method_name)
    return (
        f"{cite['authors']} ({cite['year']}). "
        + f"{cite['title']}. {cite['source']}."
    )
