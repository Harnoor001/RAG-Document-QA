PY ?= .venv/bin/python
# extra eval flags, e.g. make eval EVAL_ARGS="--real-pdf mydoc.pdf"
EVAL_ARGS ?=
FIXTURE = tests/fixtures/handbook.pdf

.PHONY: install test test-int lint format fixture ingest run eval docker-up clean

install:
	python3 -m venv .venv
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/unit -m "not integration" --cov --cov-report=term-missing --cov-fail-under=85

test-int:
	$(PY) -m pytest tests/integration -m integration

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

fixture:
	$(PY) scripts/make_fixture_pdf.py

ingest:
	$(PY) -m app.ingest $(FIXTURE)

run:
	$(PY) -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000

eval:
	$(PY) -m eval.run_eval --sweep $(EVAL_ARGS)

docker-up:
	docker compose up --build

clean:
	rm -rf data/ .pytest_cache .coverage htmlcov
