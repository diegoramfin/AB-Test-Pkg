# Two-Sample Mean Difference Analysis Report

## Data Provenance

- **Source**: A: in-memory array (500 values) | B: in-memory array (500 values)
- **SHA-256 hash**: `ee93baa41b3e8a4de969ed2a6650f7c9e441d294ff09b50fcc2a83608d20b931`

## Configuration

- **alpha**: 0.05
- **ci_level**: 0.95
- **hdi_mass**: 0.95
- **rope_width**: 0.1
- **rope_scale**: auto
- **mcmc_draws**: 500
- **mcmc_chains**: 2
- **permutation_iterations**: 1000
- **bootstrap_iterations**: 1000
- **seed**: 42
- **population_variance_a**: None
- **population_variance_b**: None
- **bayes_factor_prior_width**: 0.707
- **outlier_method**: iqr
- **outlier_threshold**: 1.5

## Assumption Diagnostics

### Shapiro-Wilk (A)

> Shapiro, S. S. and Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). Biometrika, 52(3-4), 591-611.

- **Statistic**: 0.541798
- **p-value**: 0.000000
- **assumption_outcome**: not_met

### Anderson-Darling (A)

> Anderson, T. W. and Darling, D. A. (1952). Asymptotic theory of certain goodness of fit criteria based on stochastic processes. Annals of Mathematical Statistics, 23(2), 193-212.

- **Statistic**: 63.024135
- **p-value**: 0.000100
- **assumption_outcome**: not_met

### D'Agostino K² (A)

> D'Agostino, R. B. and Pearson, E. S. (1973). Tests for departure from normality. Empirical results for the distributions of b2 and sqrt(b1). Biometrika, 60(3), 613-622.

- **Statistic**: 481.940626
- **p-value**: 0.000000
- **assumption_outcome**: not_met

### Shapiro-Wilk (B)

> Shapiro, S. S. and Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). Biometrika, 52(3-4), 591-611.

- **Statistic**: 0.506052
- **p-value**: 0.000000
- **assumption_outcome**: not_met

### Anderson-Darling (B)

> Anderson, T. W. and Darling, D. A. (1952). Asymptotic theory of certain goodness of fit criteria based on stochastic processes. Annals of Mathematical Statistics, 23(2), 193-212.

- **Statistic**: 65.062652
- **p-value**: 0.000100
- **assumption_outcome**: not_met

### D'Agostino K² (B)

> D'Agostino, R. B. and Pearson, E. S. (1973). Tests for departure from normality. Empirical results for the distributions of b2 and sqrt(b1). Biometrika, 60(3), 613-622.

- **Statistic**: 591.140103
- **p-value**: 0.000000
- **assumption_outcome**: not_met

### Levene

> Levene, H. (1960). Robust tests for equality of variances. In Olkin et al. (Eds.), Contributions to Probability and Statistics. Stanford University Press.

- **Statistic**: 0.562207
- **p-value**: 0.453549
- **assumption_outcome**: met

### Bartlett

> Bartlett, M. S. (1937). Properties of sufficiency and statistical tests. Proceedings of the Royal Society of London A, 160(901), 268-282.

- **Statistic**: 0.034804
- **p-value**: 0.852007
- **assumption_outcome**: met

### Brown-Forsythe

> Brown, M. B. and Forsythe, A. B. (1974). Robust tests for the equality of variances. Journal of the American Statistical Association, 69(346), 364-367.

- **Statistic**: 0.096168
- **p-value**: 0.756543
- **assumption_outcome**: met

### Outliers (A)

> Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.

- **outlier_count**: 48
- **outlier_indices**: [4, 13, 101, 109, 118, 130, 141, 164, 165, 178, 182, 200, 202, 203, 206, 208, 219, 229, 234, 237, 238, 249, 264, 267, 270, 274, 275, 295, 302, 304, 311, 317, 325, 334, 349, 352, 372, 374, 379, 386, 395, 405, 423, 439, 446, 460, 461, 470]
- **threshold**: 1.500000
- **Assumptions**: Outliers flagged but NOT removed.

### Outliers (B)

> Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.

- **outlier_count**: 45
- **outlier_indices**: [5, 33, 46, 48, 55, 63, 64, 67, 88, 90, 92, 95, 96, 104, 132, 135, 140, 141, 142, 144, 147, 156, 171, 199, 242, 275, 286, 306, 309, 323, 342, 357, 363, 389, 393, 400, 430, 433, 439, 449, 450, 464, 467, 485, 488]
- **threshold**: 1.500000
- **Assumptions**: Outliers flagged but NOT removed.

## Parametric Tests

### Student's t-test

> Student (Gosset, W. S.) (1908). The probable error of a mean. Biometrika, 6(1), 1-25.

- **Statistic**: 0.472259
- **p-value**: 0.636845
- **95% CI**: [-3.880934, 6.340934]
- **degrees_of_freedom**: 998.000000
- **mean_difference**: 1.230000
- **Assumptions**: Assumes independence, normality, and equal variances.

### Welch's t-test

> Welch, B. L. (1947). The generalization of 'Student's' problem when several different population variances are involved. Biometrika, 34(1-2), 28-35.

- **Statistic**: 0.472259
- **p-value**: 0.636845
- **95% CI**: [-3.880934, 6.340934]
- **degrees_of_freedom**: 997.930329
- **mean_difference**: 1.230000
- **Assumptions**: Assumes independence and normality. Does not assume equal variances.

### z-test

> Neyman, J. and Pearson, E. S. (1933). On the problem of the most efficient tests of statistical hypotheses. Philosophical Transactions of the Royal Society of London A, 231, 289-337.

- **skipped**: True
- **Assumptions**: Skipped: population variance not provided.

## Non-Parametric Tests

### Mann-Whitney U

> Mann, H. B. and Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. Annals of Mathematical Statistics, 18(1), 50-60.

- **Statistic**: 126930.000000
- **p-value**: 0.672301
- **Assumptions**: Assumes independence. Does not assume normality.

### Brunner-Munzel

> Brunner, E. and Munzel, U. (2000). The nonparametric Behrens-Fisher problem: asymptotic theory and a small-sample approximation. Biometrical Journal, 42(1), 17-25.

- **Statistic**: -0.422502
- **p-value**: 0.672750
- **Assumptions**: Assumes independence. Does not assume normality or equal variances.

### Permutation test (Monte Carlo)

> Fisher, R. A. (1935). The Design of Experiments. Oliver and Boyd, Edinburgh.

- **Statistic**: 1.230000
- **p-value**: 0.650350
- **mode**: monte_carlo
- **iterations**: 1000
- **seed**: 42
- **Assumptions**: Assumes independence and exchangeability under the null.

### Bootstrap CI

> Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC, New York.

- **Statistic**: 1.230000
- **95% CI**: [-3.847850, 6.022850]
- **iterations**: 1000
- **seed**: 42
- **Assumptions**: Bootstrap percentile CI.

## Bayesian Tests

### BEST (Kruschke)

> Kruschke, J. K. (2013). Bayesian estimation supersedes the t test. Journal of Experimental Psychology: General, 142(2), 573-603.

- **Statistic**: 0.440062
- **95% CI**: [-1.228453, 2.136250]
- **rope_width**: 4.114423
- **rope_scale**: auto
- **rope_proportion**: 1.000000
- **r_hat**: 1.000000
- **ess**: 898.000000
- **draws**: 500
- **chains**: 2
- **seed**: 42
- **Assumptions**: Student-t likelihood (robust to outliers). Weakly informative priors per Kruschke (2013).

### JZS Bayes factor

> Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D., and Iverson, G. (2009). Bayesian t tests for accepting and rejecting the null hypothesis. Psychonomic Bulletin & Review, 16(2), 225-237.

- **Statistic**: 0.079044
- **bf10**: 0.079044
- **bf01**: 12.651170
- **prior_width**: 0.707000
- **Assumptions**: JZS prior on effect size. Bayes factor quantifies relative evidence.

## Effect Sizes

### Cohen's d

> Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum, Hillsdale, NJ.

- **Statistic**: 0.029868
- **95% CI**: [-0.094248, 0.153985]

### Hedges' g

> Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect size and related estimators. Journal of Educational Statistics, 6(2), 107-128.

- **Statistic**: 0.029846
- **95% CI**: [-0.094271, 0.153962]

### Cliff's delta

> Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. Psychological Bulletin, 114(3), 494-509.

- **Statistic**: 0.015440
- **95% CI**: [-0.108442, 0.139322]

### Rank-biserial

> Kerby, D. S. (2014). The simple difference formula: An approach to teaching and testing the difference between two means using the rank-biserial correlation. Practical Assessment, Research & Evaluation, 19(11), 1-3.

- **Statistic**: 0.015440
- **95% CI**: [-0.020362, 0.051242]

### Hodges-Lehmann

> Hodges, J. L. and Lehmann, E. L. (1963). Estimation of location based on rank tests. Annals of Mathematical Statistics, 34(2), 598-611.

- **Statistic**: 0.000000
- **95% CI**: [-1.000000, 2.000000]

---

*This report presents all results without making any accept/reject decision. The analyst interprets the output.*