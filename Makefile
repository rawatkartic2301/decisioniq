.PHONY: help install lint format test eval ingest run clean

PYTHON ?= .venv/Scripts/python
ifeq ($(OS),Windows_NT)
    VENV_BIN = .venv/Scripts
else
    VENV_BIN = .venv/bin
endif

help:
	@echo "DecisionIQ Developer Commands:"
	@echo "  make install   - Install dependencies in editable mode"
	@echo "  make lint      - Run ruff linter and format check"
	@echo "  make format    - Automatically format code with ruff"
	@echo "  make test      - Run test suite with pytest"
	@echo "  make ingest    - Run ingestion pipeline on synthetic documents"
	@echo "  make eval      - Run full evaluation harness on golden set"
	@echo "  make run-api   - Start FastAPI server on port 8000"
	@echo "  make run-ui    - Start Streamlit UI on port 8501"
	@echo "  make run       - Start API and UI concurrently"

install:
	uv pip install -e ".[dev]"

lint:
	$(VENV_BIN)/ruff check src tests eval
	$(VENV_BIN)/ruff format --check src tests eval

format:
	$(VENV_BIN)/ruff format src tests eval
	$(VENV_BIN)/ruff check --fix src tests eval

test:
	$(VENV_BIN)/pytest tests -v --cov=src/decisioniq --cov-report=term-missing

ingest:
	$(VENV_BIN)/python -m decisioniq.ingestion.pipeline --store faiss
	$(VENV_BIN)/python -m decisioniq.ingestion.pipeline --store chroma

eval:
	$(VENV_BIN)/python -m decisioniq.eval.runner --output docs/EVALUATION.md

run-api:
	$(VENV_BIN)/uvicorn decisioniq.api.app:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	$(VENV_BIN)/streamlit run src/decisioniq/ui/app.py --server.port 8501

run:
	@echo "Starting API and UI..."
	$(VENV_BIN)/python -m decisioniq.run

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
