.PHONY: install test lint typecheck demo dashboard

install:
	python -m pip install -e ".[dev]"

test:
	pytest --cov=promptlab --cov-report=term-missing

lint:
	ruff check .

typecheck:
	mypy src/promptlab

demo:
	python scripts/run_research_demo.py

dashboard:
	streamlit run dashboard/app.py

