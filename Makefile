PYTHON ?= python3

.PHONY: install docs test lint fmt types security audit audit-fresh perf perf-report load gate

install:
	$(PYTHON) -m pip install -e ".[dev]"

docs:
	$(PYTHON) scripts/check_docs.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests perf scripts
	$(PYTHON) -m ruff format --check src tests perf scripts

fmt:
	$(PYTHON) -m ruff format src tests perf scripts
	$(PYTHON) -m ruff check src tests perf scripts --fix

types:
	$(PYTHON) -m mypy src

security:
	$(PYTHON) -m bandit -r src -ll

# Scoped to Cortex's own dependency closure; `--fresh` resolves from the
# index instead of auditing whatever else is installed on this machine.
audit:
	$(PYTHON) scripts/audit.py --extra dev

audit-fresh:
	$(PYTHON) scripts/audit.py --extra dev --fresh

perf:
	$(PYTHON) -m perf.benchmark --concurrency 6 --iterations 25

perf-report:
	$(PYTHON) -m perf.benchmark --concurrency 6 --iterations 25 --write

load:
	@echo "Against a DEPLOYED instance (not CI):"
	@echo "  locust -f perf/locustfile.py --host http://localhost:8000"

# Everything CI runs.
gate: docs lint types test security audit-fresh perf
