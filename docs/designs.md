# Designs

## Power and MDE planning

`PowerSpec`, `simulate_power()`, and `estimate_mde()` provide seeded
simulation-based planning for binary, continuous, count, and ratio
metrics.

```python
from twosample_means.ab_testing import PowerSpec, simulate_power, estimate_mde

spec = PowerSpec(
    kind="binary",
    control=0.03,
    effect=0.004,
    sample_size_control=20_000,
    sample_size_treatment=20_000,
)
simulate_power(spec).power   # empirical rejection rate
estimate_mde(spec, target_power=0.8)
```

## Sequential analysis

`SequentialPlan` predeclares a look schedule with O'Brien-Fleming or
Pocock alpha spending. Boundaries are calibrated by recursive numerical
quadrature over the canonical group-sequential joint distribution, so the
family-wise error rate across all looks equals the declared alpha.

```python
from twosample_means.ab_testing import (
    SequentialPlan,
    alpha_spending_boundaries,
    evaluate_sequential,
    sequential_power,
)

plan = SequentialPlan((0.5, 1.0), method="obrien_fleming")
boundaries = alpha_spending_boundaries(plan)
evaluate_sequential(plan, [z_look_1, z_look_2])

# Power and average sample information under the calibrated boundaries.
sequential_power(plan, drift=2.5)
```

### Always-valid confidence sequences

`always_valid_confidence_sequence()` builds a time-uniform confidence
sequence for a running mean: every interval is valid simultaneously, so
repeated peeking at intermediate times does not inflate the error rate
(normal-mixture bound of Howard et al., 2021). The widths are wider than
the fixed-sample interval, especially early.

```python
from twosample_means.ab_testing import always_valid_confidence_sequence

cs = always_valid_confidence_sequence(
    observations,
    alpha=0.05,
    variance_proxy=0.25,  # known proxy; e.g. 0.25 for a proportion
)
```

`difference_confidence_sequence()` combines two arm sequences (each at
`alpha/2`) into an always-valid interval for the treatment-minus-control
difference.

## Difference in differences

The `twosample_means.quasi_experimental` namespace is deliberately
separate from randomized A/B inference: DiD rests on the parallel-trends
assumption, which no amount of data can prove.

```python
from twosample_means.quasi_experimental import DifferenceInDifferences

model = DifferenceInDifferences(
    outcome="revenue",
    unit="store_id",
    time="period",
    treated="treated_store",
    post="post_launch",
    cluster="region",
)
result = model.fit(dataframe)
```

The canonical two-group design uses unit and period fixed effects with
cluster-robust standard errors. Fit validation rejects staggered adoption
and units missing pre or post observations. When at least two pre periods
exist, the result includes event-study coefficients (last pre period
omitted as reference) and a joint parallel-trends placebo p-value on the
pre-period coefficients. `render_did_markdown(result)` writes a report
that lists the identifying assumptions explicitly.

!!! note

    Staggered-adoption designs and heterogeneous treatment effects
    require dedicated estimators (for example Callaway & Sant'Anna);
    the canonical estimator here intentionally rejects those designs.
