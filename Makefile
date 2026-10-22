.PHONY: setup check test data notebooks

setup:
	uv sync --frozen --python 3.11
	uv run pre-commit install

check:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

data:
	uv run python scripts/download_data.py

notebooks: data
	uv run python scripts/run_notebooks.py
