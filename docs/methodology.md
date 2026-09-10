# Methodology

## Research objective

The lab measures interactions among tasks, prompt architectures and model profiles. The committed run validates the research system using a deterministic synthetic response surface. It must not be presented as a commercial-model leaderboard.

## Experimental unit

One trial is uniquely identified by model profile, task, prompt variant and repetition. Each trial records the rendered prompt hash, prompt-token estimate, task-specific quality, pass/fail decision, latency and cost estimate.

## Task suite

The 24 synthetic tasks cover classification, structured extraction, grounded reasoning and constrained writing. Sixteen development tasks are available for selection. Eight holdout tasks are never used to select the model-specific development winner.

## Prompt treatments

The Prompt Genome contains seven controlled components. Eight named architectures provide interpretable combinations rather than random strings. Component effects are descriptive difference-in-means estimates across this non-balanced library; they are not causal claims about every possible prompt.

## Repetition and uncertainty

The deterministic provider samples from a stable SHA-256-derived response surface. Changing the seed changes the sample while preserving reproducibility. Pairing by model, task and repetition makes architecture comparisons less sensitive to task composition. Bootstrap intervals use 1,000 seeded resamples.

Development-winner stability uses a separate paired task-cluster bootstrap. A draw samples complete development tasks with replacement while retaining every variant and repetition for each sampled task, then reruns the quality-first selection rule. The resulting selection frequency measures sensitivity to the composition of this development suite; it is not a posterior probability that a prompt is universally optimal.

## Optimisation without leakage

Successive halving operates only on development tasks. Cross-model transfer freezes the source profile's development winner before evaluating target-profile holdouts. The target oracle is reported only as an analytical upper bound; it is not available during source selection.

## Multi-objective selection

Quality is maximised while prompt tokens and latency are minimised. A point is excluded when another configuration on the same model is no worse on every objective and strictly better on at least one.

## Synthetic response surface

The synthetic profiles contain declared task-category skill, prompt-component effects, selected interactions, model-specific over-specification penalties and deterministic sampling. These choices create a test landscape with multiple local winners. They are designed to exercise the methods, not imitate a named provider.

## Appropriate live-study extensions

- increase task coverage using authorised domain examples;
- maintain a hidden final test set;
- repeat stochastic calls and retain raw outputs;
- calibrate deterministic and model-based graders against humans;
- correct for multiple comparisons when screening many variants;
- preregister primary outcomes before expensive runs;
- analyse failure clusters, not only aggregate scores.
