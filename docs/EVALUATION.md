# Evaluation Framework

Cortex treats quality as a first-class engineering concern. Evaluation runs automatically — not as a one-off experiment.

## Philosophy

Most AI systems are evaluated manually and infrequently. Cortex runs automated evaluation:
- **On every deployment** (CI gate)
- **Every 6 hours** (Celery beat schedule)
- **On demand** (any developer can run locally)

Results are stored as Prometheus time-series metrics, so quality regressions are visible in Grafana before users notice them.

## Ragas Metrics

Cortex uses [Ragas](https://docs.ragas.io) as its primary evaluation library.

| Metric | Definition | Target |
|--------|-----------|--------|
| **Faithfulness** | What fraction of the answer's claims are supported by retrieved context? Measures hallucination. | ≥ 0.85 |
| **Answer Relevancy** | How well does the answer address the original question? | ≥ 0.80 |
| **Context Precision** | Of the retrieved chunks, what fraction were actually relevant? | ≥ 0.75 |
| **Context Recall** | Was all ground-truth information present in the retrieved context? (requires labelled data) | ≥ 0.70 |
| **Composite** | Average of faithfulness, relevancy, precision | ≥ 0.78 |

## Regression Test Cases

Test cases live in `tests/eval/regression_cases.json`. Format:

```json
[
  {
    "question": "What are the main benefits of using MCP for tool exposure?",
    "answer": "MCP provides native LLM client compatibility...",
    "contexts": [
      "Model Context Protocol (MCP) is the protocol adopted by...",
      "Cortex exposes all its capabilities as MCP tools..."
    ],
    "ground_truth": "MCP enables native LLM client compatibility..."
  }
]
```

- `question` — the input
- `answer` — the model's output to evaluate
- `contexts` — the RAG chunks used to generate the answer
- `ground_truth` — optional; required for context recall

**Adding new test cases:**
1. Run the system on a representative query
2. Manually verify the answer is correct
3. Add the question, answer, contexts, and ground truth to `regression_cases.json`
4. Run the eval suite to confirm the new case passes

## Running Evaluations

### Local

```bash
# Full regression suite
python -c "
import asyncio
from cortex.eval.ragas_runner import run_regression_suite
result = asyncio.run(run_regression_suite())
print(result.to_dict())
print('PASSED' if result.passes_threshold() else 'FAILED')
"

# Single sample
from cortex.eval.ragas_runner import RagasEvaluator, EvalSample
evaluator = RagasEvaluator()
sample = EvalSample(
    question="What is RRF?",
    answer="Reciprocal Rank Fusion combines rankings from multiple retrievers...",
    contexts=["RRF is a rank aggregation method..."]
)
result = asyncio.run(evaluator.evaluate([sample]))
```

### In CI

Add to your CI pipeline (GitHub Actions example):

```yaml
- name: Run eval regression
  run: |
    python -c "
    import asyncio, sys
    from cortex.eval.ragas_runner import run_regression_suite
    result = asyncio.run(run_regression_suite())
    print(result.to_dict())
    sys.exit(0 if result.passes_threshold() else 1)
    "
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    SECRET_KEY: ${{ secrets.CORTEX_SECRET_KEY }}
```

### Scheduled (Celery beat)

The eval regression runs automatically every 6 hours. Check the results:

```bash
# Check Celery task history
celery -A cortex.workers.celery_app inspect registered

# Force-run now
celery -A cortex.workers.celery_app call cortex.workers.celery_app.run_eval_regression

# View results in Grafana: Quality & Evaluation panel
```

## Baseline Scores

Current baseline on `tests/eval/regression_cases.json` (5 cases):

| Metric | Baseline | Target |
|--------|---------|--------|
| Faithfulness | 0.87 | ≥ 0.85 |
| Answer Relevancy | 0.84 | ≥ 0.80 |
| Context Precision | 0.79 | ≥ 0.75 |
| Composite | 0.83 | ≥ 0.78 |

**These are aspirational baselines.** Your actual scores depend on your LLM, document corpus, and prompt versions. Establish your own baseline on first run, then protect it.

## LLM-as-Judge Fallback

When Ragas isn't available, Cortex falls back to LLM-as-judge scoring. The judge prompt asks GPT-4o to score faithfulness and relevancy for each sample.

This is less reliable than Ragas (which uses dedicated models and multi-sample calibration) but better than nothing. The fallback is transparent — logs will show `eval.ragas_not_available`.

## Interpreting Results

**Faithfulness drops below 0.75:**
- RAG quality issue — check that retrieved chunks contain the relevant information
- Model is ignoring retrieved context — strengthen the system prompt's RAG instructions
- Prompt drift — a recent prompt change caused hallucination; revert and re-test

**Answer Relevancy drops below 0.70:**
- System prompt is too restrictive and causing refusals
- Planner is creating tasks that don't address the user's actual goal
- Memory context is polluting the prompt with irrelevant past information

**Context Precision drops below 0.60:**
- Retrieval quality degraded — check embedding model and Qdrant index health
- Chunk size too large — chunks contain too much irrelevant content
- Consider tuning `RAG_BM25_WEIGHT` and `RAG_DENSE_WEIGHT`


## Two harnesses, because they answer different questions

| harness | scores | backend |
|---|---|---|
| `eval/ragas_runner.py` | retrieval and generation — faithfulness, answer relevancy, context precision | Ragas, falling back to an LLM judge |
| `eval/agent_eval.py` | the agent loop — task completion, tool correctness, plan efficiency, termination | DeepEval, falling back to deterministic metrics |

Ragas cannot see the part of Cortex that makes it an agent. It has no notion
of a goal, tool calls are invisible to it, and it cannot tell a run that
converged from one that was stopped by a budget ceiling.

Before this, the only thing scoring the agent loop was the critic — which is
*part of* the loop. A system grading its own homework produces a number
that is stable, plausible and worthless.

### The fallback is not a stub

Three of the four agent axes need no model at all: tool correctness, plan
efficiency and termination are computed from the run's own structure. Only
task completion degrades to an LLM judge.

That split is deliberate. DeepEval and Ragas are both heavy optional
dependencies, and the environment least likely to have them installed is
CI — which is exactly where the harness needs to still measure something.

### Thresholds are per-axis, not on the average

`passes()` checks completion, tool correctness and termination separately.
An average lets one collapsed dimension hide behind three healthy ones, and
"terminated correctly 40% of the time" is not something an overall 0.8
should be able to conceal.

`termination` carries the strictest threshold (0.90) because it is the
failure users actually notice: a loop that exhausts its budget produces a
bill and no answer.

Plan efficiency is **reported but not gated**. An inefficient plan that
reaches the right answer is a cost problem, not a correctness one — it
belongs on a dashboard, not in a gate that blocks a release.

### An unreachable judge scores 0.5

Not 0.0 and not 1.0. Scoring an outage as a failure makes it look like a
quality regression; scoring it as a pass lets an outage hide one. The score
carries `"not evidence"` in its detail string so a report cannot quote it
as a measurement.
