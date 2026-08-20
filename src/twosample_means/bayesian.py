"""Bayesian tests: BEST (Kruschke) via PyMC, JZS Bayes factor.

Each function returns posterior summaries, Bayes factors, HDI/ROPE
statistics, and MCMC diagnostics — but NEVER applies an accept/reject
decision. The ROPE and HDI are reported as descriptive evidence.

Academic rationale
------------------
- BEST (Kruschke, 2013): Bayesian Estimation Supersedes the t-test.
  Models each group with a Student-t likelihood (robust to outliers)
  and estimates the posterior of the mean difference. The HDI and
  ROPE provide a principled framework for interpreting the posterior
  without dichotomous decisions.
- JZS Bayes factor (Rouder et al., 2009): the Jeffreys-Zellner-Siow
  prior on the effect size yields a Bayes factor that quantifies the
  relative evidence for H₀ vs H₁. Implemented via pingouin. The
  standard JZS model assumes equal population variances.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import numpy.typing as npt
import pingouin as pg
import pymc as pm
from scipy import stats

from twosample_means.citations import Citation, get_citation
from twosample_means.config import RunConfig


@dataclass(frozen=True)
class BESTResult:
    """Result of the BEST Bayesian t-test.

    Attributes
    ----------
    method_name:
        Name of the test.
    citation:
        Academic reference string.
    posterior_mean_diff:
        Posterior mean of the mean difference (A - B).
    hdi_lower:
        Lower bound of the highest-density interval.
    hdi_upper:
        Upper bound of the HDI.
    hdi_mass:
        HDI mass (e.g., 0.95).
    rope_width:
        Half-width of the ROPE used.
    rope_proportion:
        Proportion of posterior samples inside the ROPE.
    r_hat:
        Gelman-Rubin convergence diagnostic (max R-hat).
    ess:
        Effective sample size (min ESS).
    draws:
        Number of posterior draws per chain.
    chains:
        Number of MCMC chains.
    seed:
        Random seed used.
    assumption_notes:
        Human-readable notes on assumptions.

    """

    method_name: str
    citation: str
    posterior_mean_diff: float
    hdi_lower: float
    hdi_upper: float
    hdi_mass: float
    rope_width: float
    rope_proportion: float
    r_hat: float
    ess: float
    draws: int
    chains: int
    seed: int
    assumption_notes: str


@dataclass(frozen=True)
class BayesFactorResult:
    """Result of the JZS Bayes factor test.

    Attributes
    ----------
    method_name:
        Name of the test.
    citation:
        Academic reference string.
    bf10:
        Bayes factor favoring H₁ over H₀.
    bf01:
        Bayes factor favoring H₀ over H₁ (1/BF10).
    prior_width:
        The prior width (r scale) used.
    assumption_notes:
        Human-readable notes on assumptions.

    """

    method_name: str
    citation: str
    bf10: float
    bf01: float
    prior_width: float
    assumption_notes: str


def best(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    config: RunConfig,
) -> BESTResult:
    """Perform the BEST Bayesian t-test (Kruschke, 2013).

    Models each group with a Student-t likelihood and estimates the
    posterior of the mean difference via PyMC MCMC. Reports the HDI
    and ROPE proportion as descriptive evidence — NO decision.

    Citation: Kruschke (2013).

    Assumptions: The Student-t likelihood is robust to outliers.
    Priors are weakly informative (normal on means, uniform on
    scales, exponential on degrees of freedom), following Kruschke's
    recommendations.

    Parameters
    ----------
    a:
        1-D array of sample A observations.
    b:
        1-D array of sample B observations.
    config:
        Run configuration (uses ``mcmc_draws``, ``mcmc_chains``,
        ``seed``, ``hdi_mass``, ``rope_width``).

    Returns
    -------
    BESTResult
        Posterior summaries, HDI, ROPE, MCMC diagnostics.

    """
    cite = get_citation("best")
    pooled = np.concatenate([a, b])
    mean_prior = float(np.mean(pooled))
    sd_prior = float(np.std(pooled)) * 5.0
    if not np.isfinite(sd_prior) or sd_prior <= 0.0:
        raise ValueError(
            "BEST is not estimable when pooled data have zero variance"
        )

    with pm.Model():
        mu_a = pm.Normal("mu_a", mu=mean_prior, sigma=sd_prior)
        mu_b = pm.Normal("mu_b", mu=mean_prior, sigma=sd_prior)
        sigma_a = pm.Uniform("sigma_a", lower=0.0001, upper=sd_prior * 10)
        sigma_b = pm.Uniform("sigma_b", lower=0.0001, upper=sd_prior * 10)
        nu = pm.Exponential("nu_minus_one", 1.0 / 29.0) + 1
        pm.StudentT("obs_a", nu=nu, mu=mu_a, sigma=sigma_a, observed=a)
        pm.StudentT("obs_b", nu=nu, mu=mu_b, sigma=sigma_b, observed=b)
        pm.Deterministic("mean_diff", mu_a - mu_b)

        idata = pm.sample(
            draws=config.mcmc_draws,
            chains=config.mcmc_chains,
            random_seed=config.seed,
            progressbar=False,
        )

    mean_diff_samples = idata.posterior["mean_diff"].values
    flat_samples = mean_diff_samples.flatten()
    posterior_mean_diff = float(np.mean(flat_samples))

    hdi = az.hdi(  # type: ignore[no-untyped-call]
        flat_samples, hdi_prob=config.hdi_mass
    )
    hdi_arr = np.asarray(hdi).flatten()
    hdi_lo = float(hdi_arr[0])
    hdi_hi = float(hdi_arr[1])

    n_a = len(a)
    n_b = len(b)
    sd_a = float(np.std(a, ddof=1))
    sd_b = float(np.std(b, ddof=1))
    pooled_sd = float(
        np.sqrt(((n_a - 1) * sd_a**2 + (n_b - 1) * sd_b**2) / (n_a + n_b - 2))
    )
    if config.rope_scale == "auto":
        effective_rope = config.rope_width * pooled_sd
    else:
        effective_rope = config.rope_width
    rope_lo = -effective_rope
    rope_hi = effective_rope
    rope_proportion = float(
        np.mean((flat_samples >= rope_lo) & (flat_samples <= rope_hi))
    )

    summary = az.summary(idata)
    r_hat = float(summary["r_hat"].max())
    ess = float(summary["ess_bulk"].min())

    r_hat_ok = r_hat < 1.01
    ess_ok = ess > 400
    convergence_note = ""
    if not (r_hat_ok and ess_ok):
        failures = []
        if not r_hat_ok:
            failures.append(f"r_hat={r_hat:.4f} >= 1.01")
        if not ess_ok:
            failures.append(f"ess={ess:.0f} <= 400")
        convergence_note = (
            " WARNING: MCMC convergence diagnostics below "
            f"threshold ({', '.join(failures)}). "
            "HDI may be unreliable; increase draws/chains "
            "or inspect the trace."
        )

    return BESTResult(
        method_name="BEST (Kruschke)",
        citation=_fmt(cite),
        posterior_mean_diff=posterior_mean_diff,
        hdi_lower=hdi_lo,
        hdi_upper=hdi_hi,
        hdi_mass=config.hdi_mass,
        rope_width=effective_rope,
        rope_proportion=rope_proportion,
        r_hat=r_hat,
        ess=ess,
        draws=config.mcmc_draws,
        chains=config.mcmc_chains,
        seed=config.seed,
        assumption_notes=(
            "Student-t likelihood (robust to outliers). "
            "Weakly informative priors per Kruschke (2013)." + convergence_note
        ),
    )


def bayes_factor_jzs(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    config: RunConfig,
) -> BayesFactorResult:
    """Compute the JZS Bayes factor for a two-sample t-test.

    Uses the Jeffreys-Zellner-Siow prior on the effect size.
    Thin wrapper over ``pingouin.bayesfactor_ttest``.

    Citation: Rouder et al. (2009).

    Assumptions: The JZS prior is a default, objective prior for the
    effect size. The Bayes factor quantifies the relative evidence
    for H₁ vs H₀ — it does NOT make a decision. The standard two-sample
    JZS model assumes equal population variances.

    Parameters
    ----------
    a:
        1-D array of sample A observations.
    b:
        1-D array of sample B observations.
    config:
        Run configuration (uses ``bayes_factor_prior_width``).

    Returns
    -------
    BayesFactorResult
        BF10, BF01, and prior width.

    """
    cite = get_citation("bayes_factor_jzs")
    t_stat, _p = _students_t_stat(a, b)
    n_a = len(a)
    n_b = len(b)
    bf10 = float(
        pg.bayesfactor_ttest(
            t_stat,
            nx=n_a,
            ny=n_b,
            r=config.bayes_factor_prior_width,
            alternative="two-sided",
        )
    )
    bf01 = float("inf") if bf10 == 0 else 1.0 / bf10
    return BayesFactorResult(
        method_name="JZS Bayes factor",
        citation=_fmt(cite),
        bf10=bf10,
        bf01=bf01,
        prior_width=config.bayes_factor_prior_width,
        assumption_notes=(
            "JZS prior on effect size. "
            "Bayes factor quantifies relative evidence. "
            "Assumes equal population variances."
        ),
    )


def _students_t_stat(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """Compute Student's t-statistic for the JZS Bayes factor input.

    The standard JZS two-sample Bayes factor is derived under the
    equal-variance Student t model, so the pooled-variance t-statistic
    is used here.

    Parameters
    ----------
    a, b:
        Sample arrays.

    Returns
    -------
    tuple[float, float]
        t-statistic and p-value.

    """
    pooled_variance = (
        (len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)
    ) / (len(a) + len(b) - 2)
    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        raise ValueError("JZS Bayes factor is not estimable for zero variance")
    result = stats.ttest_ind(a, b, equal_var=True)
    return float(result.statistic), float(result.pvalue)


def _fmt(cite: Citation) -> str:
    """Format a citation as a readable string."""
    return (
        f"{cite['authors']} ({cite['year']}). "
        f"{cite['title']}. {cite['source']}."
    )
