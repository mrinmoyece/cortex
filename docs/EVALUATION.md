# Evaluation

This document defines the evaluation methodology that the repository actually
implements. Cortex has evaluation harnesses and regression inputs; it does
**not** contain a validated production benchmark, a checked-in score report,
or a CI gate on live-model quality.

## Questions the harnesses answer

| Harness | Question | Implementation | Evidence |
|---|---|---|---|
| RAG evaluation | Is an answer grounded in and relevant to supplied contexts? | [`ragas_runner.py`](../src/cortex/eval/ragas_runner.py) | [`test_eval.py`](../tests/test_eval/test_eval.py) |
| Agent evaluation | Did a run complete tasks, use tools correctly, plan efficiently, and terminate correctly? | [`agent_eval.py`](../src/cortex/eval/agent_eval.py) | [`test_agent_eval.py`](../tests/test_eval/test_agent_eval.py) |

RAG evaluation uses Ragas when the optional dependency imports successfully.
Otherwise it uses an LLM judge for supported dimensions. Agent evaluation uses
deterministic run-structure metrics where possible and reserves model judging
for task completion. The critic inside the agent graph is not independent
evaluation evidence.

## Regression dataset

[`tests/eval/regression_cases.json`](../tests/eval/regression_cases.json)
contains the checked-in questions, answers, contexts, and optional ground
truth. These are fixed inputs for exercising the evaluator; they are not
outputs generated from a deployed Cortex instance.

Add a case only after:

1. recording the model, prompt/config revision, and source corpus used;
2. manually checking the answer, contexts, and ground truth;
3. removing sensitive or proprietary content;
4. running the suite with the same environment recorded for the case; and
5. reviewing whether the case covers a new failure mode rather than duplicating
   an existing one.

Dataset changes should be reviewed like test changes: a case that cannot fail
does not protect quality.

## Metrics and decision rules

### RAG axes

- **Faithfulness:** support for answer claims in supplied context.
- **Answer relevancy:** alignment between answer and question.
- **Context precision:** proportion of retrieved context useful to the answer.
- **Context recall:** ground-truth coverage when labels are available.

The implementation's threshold values are target policy, not observed
baselines. They are defined with the result models in
[`ragas_runner.py`](../src/cortex/eval/ragas_runner.py).

### Agent axes

- **Task completion:** required tasks produced usable results.
- **Tool correctness:** tool calls and outcomes match the run record.
- **Plan efficiency:** completed work relative to plan size; reported but not
  used alone as a correctness gate.
- **Termination:** the run reached an expected terminal state instead of
  exhausting limits or hanging.

The agent evaluator gates axes individually rather than hiding a collapsed
dimension in an average. If the judge is unavailable, a neutral score marked
as non-evidence is returned; it must not be quoted as a measurement.

## Reproducing evaluation

Install the optional evaluation dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval]"
```

First validate the harness without network calls:

```bash
pytest tests/test_eval --no-cov
```

Run the checked-in RAG regression inputs:

```bash
python3 - <<'PY'
import asyncio
from cortex.eval.ragas_runner import run_regression_suite

result = asyncio.run(run_regression_suite())
print(result.to_dict())
print("passed targets:", result.passes_threshold())
PY
```

Both the Ragas path and the fallback are model-backed and may make network
calls, require provider credentials, and incur provider charges. A
reproducible report must record:

- commit SHA and dirty/clean state;
- Python and installed package versions;
- evaluator path used (Ragas, DeepEval, deterministic, or LLM fallback);
- provider and model identifiers;
- prompt/config revision and dataset hash;
- sample count, per-axis values, errors, and missing dimensions; and
- execution timestamp and operator.

No such report is committed today, so this repository makes no current quality
score claim.

## Scheduled execution

[`workers/celery_app.py`](../src/cortex/workers/celery_app.py) registers an
evaluation task and a six-hour Celery beat schedule. It runs only where both
worker and beat processes are operating with required provider configuration.
The Kubernetes beat workload is fixed at one replica to avoid duplicate
schedules. Scheduling a harness is not a deployment gate and does not by
itself retain an auditable report.

Operators can invoke the registered task:

```bash
celery -A cortex.workers.celery_app call cortex.workers.celery_app.run_eval_regression
```

## Interpreting a regression

1. Confirm the same evaluator path, model, prompt/config, and dataset were used.
2. Separate evaluator/provider outages from answer-quality changes.
3. Inspect per-sample results; do not diagnose from a composite alone.
4. For faithfulness or precision changes, compare retrieved chunks and active
   retrieval filters before changing the generation prompt.
5. For termination or tool-correctness changes, inspect graph state and tool
   traces rather than RAG metrics.
6. Record accepted baseline changes with rationale; do not silently lower a
   threshold.

Prometheus quality series and alert thresholds are described in
[Operations](OPERATIONS.md#service-indicators-and-objectives). They are
operational signals, not substitutes for a versioned evaluation report.
