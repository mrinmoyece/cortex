.PHONY: install test lint fmt types security audit perf perf-report load gate

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests perf scripts
	ruff format --check src tests perf scripts

fmt:
	ruff format src tests perf scripts
	ruff check src tests perf scripts --fix

types:
	mypy src

security:
	bandit -r src -ll

# Scoped to Cortex's own dependency closure; `--fresh` resolves from the
# index instead of auditing whatever else is installed on this machine.
audit:
	python scripts/audit.py --extra dev

audit-fresh:
	python scripts/audit.py --extra dev --fresh

perf:
	python -m perf.benchmark --concurrency 6 --iterations 25

perf-report:
	python -m perf.benchmark --concurrency 6 --iterations 25 --write

load:
	@echo "Against a DEPLOYED instance (not CI):"
	@echo "  locust -f perf/locustfile.py --host http://localhost:8000"

# Everything CI runs, in the same order.
gate: lint types test security audit perf
