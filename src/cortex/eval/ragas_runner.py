"""
Cortex Ragas evaluation suite.

Runs automated quality evaluation on:
  1. RAG retrieval — context precision, context recall, faithfulness
  2. Agent output — answer relevancy, factual consistency
  3. End-to-end runs — composite quality score

Results are stored as time-series metrics in Prometheus so regressions
are visible in Grafana dashboards and trigger alerts.

Run on-demand or schedule via Celery beat for continuous eval.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from cortex.logging_config import get_logger
from cortex.obs.metrics import hallucination_score

logger = get_logger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample."""

    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    ground_truth: str | None = None
    run_id: str | None = None


@dataclass
class EvalResult:
    """Aggregated evaluation results for a batch of samples."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float | None
    composite: float
    sample_count: int
    failed_samples: int
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4) if self.context_recall else None,
            "composite": round(self.composite, 4),
            "sample_count": self.sample_count,
            "failed_samples": self.failed_samples,
            "evaluated_at": self.evaluated_at,
        }

    def passes_threshold(
        self,
        faithfulness_min: float = 0.80,
        relevancy_min: float = 0.75,
        composite_min: float = 0.78,
    ) -> bool:
        return (
            self.faithfulness >= faithfulness_min
            and self.answer_relevancy >= relevancy_min
            and self.composite >= composite_min
        )


class RagasEvaluator:
    """
    Wrapper around the Ragas evaluation library.
    Falls back to LLM-as-judge scoring when Ragas isn't available.
    """

    def __init__(self) -> None:
        self._ragas_available = False
        try:
            import ragas  # noqa: F401

            self._ragas_available = True
            logger.info("eval.ragas_loaded")
        except ImportError:
            logger.warning("eval.ragas_not_available — using LLM-as-judge fallback")

    async def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        """Evaluate a batch of samples. Returns aggregated metrics."""
        if not samples:
            raise ValueError("No samples to evaluate")

        if self._ragas_available:
            return await self._ragas_eval(samples)
        return await self._llm_judge_eval(samples)

    async def _ragas_eval(self, samples: list[EvalSample]) -> EvalResult:
        """Full Ragas evaluation pipeline."""
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            data = {
                "question": [s.question for s in samples],
                "answer": [s.answer for s in samples],
                "contexts": [s.contexts for s in samples],
                "ground_truth": [s.ground_truth or "" for s in samples],
            }
            dataset = Dataset.from_dict(data)

            metrics = [faithfulness, answer_relevancy, context_precision]
            has_ground_truth = any(s.ground_truth for s in samples)
            if has_ground_truth:
                metrics.append(context_recall)

            result = evaluate(dataset, metrics=metrics)
            df = result.to_pandas()

            faithfulness_score = float(df["faithfulness"].mean())
            relevancy_score = float(df["answer_relevancy"].mean())
            precision_score = float(df["context_precision"].mean())
            recall_score = float(df["context_recall"].mean()) if has_ground_truth else None

            composite = (faithfulness_score + relevancy_score + precision_score) / 3.0

            # Emit to Prometheus
            hallucination_score.observe(faithfulness_score)

            return EvalResult(
                faithfulness=faithfulness_score,
                answer_relevancy=relevancy_score,
                context_precision=precision_score,
                context_recall=recall_score,
                composite=composite,
                sample_count=len(samples),
                failed_samples=0,
            )

        except Exception as exc:
            logger.error("eval.ragas_failed", error=str(exc))
            raise

    async def _llm_judge_eval(self, samples: list[EvalSample]) -> EvalResult:
        """
        LLM-as-judge fallback. Cheaper but less reliable than Ragas.
        Uses GPT-4o to score faithfulness and relevancy per sample.
        """
        from cortex.llm.router import get_router

        router = get_router()

        faithfulness_scores = []
        relevancy_scores = []
        failed = 0

        for sample in samples:
            try:
                prompt = f"""Score this answer on two dimensions (0.0 to 1.0 each).

Question: {sample.question}
Answer: {sample.answer}
Context: {" ".join(sample.contexts[:3])[:2000]}

Return ONLY JSON: {{"faithfulness": float, "relevancy": float, "reasoning": "one sentence"}}"""

                response = await router.complete(
                    messages=[{"role": "user", "content": prompt}],
                    run_id="eval-judge",
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                data = json.loads(response.choices[0].message.content)
                faithfulness_scores.append(float(data.get("faithfulness", 0.5)))
                relevancy_scores.append(float(data.get("relevancy", 0.5)))

            except Exception as exc:
                logger.warning("eval.judge_sample_failed", error=str(exc))
                failed += 1
                faithfulness_scores.append(0.5)
                relevancy_scores.append(0.5)

        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
        composite = (avg_faithfulness + avg_relevancy) / 2.0

        hallucination_score.observe(avg_faithfulness)

        return EvalResult(
            faithfulness=avg_faithfulness,
            answer_relevancy=avg_relevancy,
            context_precision=0.0,  # Not available from LLM judge
            context_recall=None,
            composite=composite,
            sample_count=len(samples),
            failed_samples=failed,
        )


async def run_regression_suite(
    test_cases_path: str = "tests/eval/regression_cases.json",
) -> EvalResult:
    """
    Load test cases from a JSON file and run evaluation.
    Called by CI and by the Celery beat scheduler.

    Test case format:
    [
      {
        "question": "...",
        "answer": "...",      // optional — will run the agent to get this
        "contexts": [...],
        "ground_truth": "..."
      }
    ]
    """
    import os

    if not os.path.exists(test_cases_path):
        logger.warning("eval.no_test_cases", path=test_cases_path)
        return EvalResult(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=None,
            composite=0.0,
            sample_count=0,
            failed_samples=0,
        )

    with open(test_cases_path) as f:
        cases = json.load(f)

    samples = [EvalSample(**c) for c in cases]
    evaluator = RagasEvaluator()
    result = await evaluator.evaluate(samples)

    logger.info(
        "eval.regression_complete",
        composite=result.composite,
        faithfulness=result.faithfulness,
        passes=result.passes_threshold(),
    )

    if not result.passes_threshold():
        logger.error(
            "eval.regression_FAILED",
            composite=result.composite,
            faithfulness=result.faithfulness,
        )

    return result
