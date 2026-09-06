# Contributing

Contributions should strengthen experimental validity, reproducibility or clarity.

```bash
python -m pip install -e ".[dev]"
ruff check .
mypy src/promptlab
pytest --cov=promptlab
python scripts/run_research_demo.py --verify
```

New prompt techniques must include a falsifiable rationale, an implementation in the Prompt Genome or renderer, and tests. Do not remove failed trials to improve aggregate results. Never contribute proprietary prompts, private customer content, working credentials or unlicensed evaluation data.

