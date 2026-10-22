.PHONY: setup check test data notebooks train flow prefect serve api up down smoke

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

train: data
	uv run python -m src.modelling.main

flow: data
	PREFECT_HOME="$(CURDIR)/.prefect" PREFECT_SERVER_ANALYTICS_ENABLED=false uv run python -m src.modelling.workflows run

prefect:
	PREFECT_HOME="$(CURDIR)/.prefect" PREFECT_SERVER_ANALYTICS_ENABLED=false uv run prefect server start --host 127.0.0.1 --port 4251

serve:
	PREFECT_HOME="$(CURDIR)/.prefect" PREFECT_API_URL=http://127.0.0.1:4251/api uv run python -m src.modelling.workflows serve

api:
	uv run uvicorn src.web_service.main:app --host 127.0.0.1 --port 8526

up: data
	docker compose up --build -d

down:
	docker compose down

smoke:
	uv run python -m scripts.smoke_test
