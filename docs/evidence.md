# Evidence boundary

## Committed run

- Evidence type: deterministic synthetic response-surface experiment
- Core trials: 4,032
- Robustness trials: 1,728
- External model calls: none
- Real customer data: none
- Paid services: none
- Seed: 20260906

## Interpretation

The committed evidence proves that the repository can render prompts, execute repeated factorial experiments, grade outputs, maintain development/holdout separation, estimate uncertainty, analyse prompt components, find Pareto-efficient configurations and measure transfer regret.

It does not establish the best prompt for an external model. The profile names `atlas-sim`, `nova-sim` and `ember-sim` are fictional and deliberately avoid resembling provider product names.

## Reproduction

```bash
python -m pip install -e ".[dev]"
python scripts/run_research_demo.py
pytest
```

Every generated artifact states or links to this boundary. Live evidence belongs in `artifacts/live/` until it has been checked for secrets, licences, customer information, pricing accuracy and methodological comparability.

