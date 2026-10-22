.PHONY: setup check test

setup:
	uv sync --frozen --python 3.11
	uv run pre-commit install

check:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q
