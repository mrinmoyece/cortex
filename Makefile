.PHONY: install test lint fmt perf perf-report load gate

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests perf
	ruff format --check src tests perf

fmt:
	ruff format src tests perf
	ruff check src tests perf --fix

perf:
	python -m perf.benchmark --concurrency 6 --iterations 25

perf-report:
	python -m perf.benchmark --concurrency 6 --iterations 25 --write

load:
	@echo "Against a DEPLOYED instance (not CI):"
	@echo "  locust -f perf/locustfile.py --host http://localhost:8000"

gate: lint test perf
