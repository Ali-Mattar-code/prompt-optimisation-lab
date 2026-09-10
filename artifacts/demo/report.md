# Prompt optimisation research report

> Evidence level: deterministic synthetic response-surface experiment. This validates the
> optimisation machinery; it is not a performance claim about any commercial model.

## Experiment scale

- Trials: **4,032**
- Tasks: **24**
- Prompt variants: **8**
- Simulated model profiles: **3**
- Repetitions per cell: **7**

## Best development prompt by model

| Model profile | Variant | Mean quality | Pass rate | Prompt tokens | Selection stability |
|---|---|---:|---:|---:|---:|
| atlas-sim | schema_first | 0.625 | 56.2% | 47.2 | 30.3% |
| nova-sim | verify | 0.652 | 60.7% | 64.9 | 57.2% |
| ember-sim | role_only | 0.644 | 61.6% | 42.9 | 49.3% |

## Interpretation

The synthetic profiles select different prompt architectures and exhibit non-zero cross-model transfer regret. This demonstrates why prompts should be optimised against task-specific holdouts instead of treated as universal text recipes.

Selection stability is the share of paired task-cluster bootstrap resamples in which
the reported development winner remains selected. It measures sensitivity to the task
suite, not the probability that a prompt is universally best.

The machine-readable result, component effects, Pareto frontier, transfer matrix,
selection-stability table, and successive-halving trace are stored beside this report.
