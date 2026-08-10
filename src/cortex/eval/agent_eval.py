"""Agent-level evaluation, using DeepEval where available.

Ragas scores *retrieval and generation*: was the answer faithful to the
context, was the context relevant. It says nothing about the part of Cortex
that makes it an agent — whether the plan was sensible, whether the right
tools were called, whether the loop terminated for the right reason.

That gap mattered here specifically, because the only thing scoring the
agent loop was the critic, and the critic is part of the loop. A system
grading its own homework produces a number that is stable, plausible, and
worthless.

## What is measured

| metric | question | why Ragas cannot answer it |
|---|---|---|
| task completion | did the run achieve the stated goal | Ragas has no notion of a goal, only Q&A |
| tool correctness | were the expected tools called | tool calls are invisible to a RAG metric |
| plan efficiency | how much redundant work was done | needs the task graph |
| loop termination | did it stop for a good reason | needs run status, not output |

## DeepEval is optional, and the fallback is not a stub

DeepEval is a heavy dependency that pulls its own model stack. Where it is
absent, the deterministic metrics below still run — tool correctness, plan
efficiency and termination are computed from the run's own structure and
need no model at all. Only task completion degrades to an LLM judge.

That split is deliberate. A harness whose fallback measures nothing is a
harness that measures nothing in CI, where the heavy dependency is exactly
the one nobody installs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cortex.graph.state import CortexState, RunStatus, TaskStatus
from cortex.logging_config import get_logger
from cortex.obs.metrics import agent_eval_score

logger = get_logger(__name__)


@dataclass
class AgentScore:
    """One run, scored on four axes."""

    task_completion: float
    tool_correctness: float
    plan_efficiency: float
    termination: float
    details: dict[str, str] = field(default_factory=dict)
    backend: str = "deterministic"

    @property
    def overall(self) -> float:
        return round(
            (self.task_completion + self.tool_correctness + self.plan_efficiency + self.termination)
            / 4,
            4,
        )

    def passes(
        self,
        *,
        min_completion: float = 0.70,
        min_tool_correctness: float = 0.80,
        min_termination: float = 0.90,
    ) -> bool:
        """Per-axis thresholds, not an average.

        An average lets a collapsed dimension hide behind three healthy
        ones — and "the agent terminated correctly 40% of the time" is not
        something an overall score of 0.8 should be able to conceal.

        `termination` has the strictest threshold because a run that ends
        for the wrong reason is the failure users actually notice: a loop
        that exhausts its budget produces a bill and no answer.
        """
        return (
            self.task_completion >= min_completion
            and self.tool_correctness >= min_tool_correctness
            and self.termination >= min_termination
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_completion": round(self.task_completion, 4),
            "tool_correctness": round(self.tool_correctness, 4),
            "plan_efficiency": round(self.plan_efficiency, 4),
            "termination": round(self.termination, 4),
            "overall": self.overall,
            "backend": self.backend,
            "details": self.details,
        }


def score_tool_correctness(state: CortexState, expected: set[str] | None) -> tuple[float, str]:
    """Fraction of expected tools that were actually called.

    Precision is deliberately NOT penalised. An agent that calls an extra
    tool to check its work is behaving well; scoring that as an error would
    train the plan towards under-verifying, which is the opposite of what
    an evaluation should reward.
    """
    if not expected:
        return 1.0, "no expected tools declared"
    called = {t.tool for t in state.tasks if t.tool}
    hit = expected & called
    return len(hit) / len(expected), f"called {sorted(called)}, expected {sorted(expected)}"


def score_plan_efficiency(state: CortexState) -> tuple[float, str]:
    """Useful work over attempted work.

    Replans are counted against the score because each one repeats the
    whole plan-execute cycle: a run that needed three attempts cost roughly
    three times what it should have, whatever the final answer looked like.
    """
    if not state.tasks:
        return 0.0, "no tasks planned"
    completed = sum(1 for t in state.tasks if t.status == TaskStatus.COMPLETED)
    attempts = 1 + state.critique_iteration
    efficiency = (completed / len(state.tasks)) / attempts
    return (
        min(efficiency, 1.0),
        f"{completed}/{len(state.tasks)} tasks over {attempts} plan attempt(s)",
    )


def score_termination(state: CortexState) -> tuple[float, str]:
    """Did the run stop for a good reason?

    A completed run scores 1.0. A run that failed on a *validated* condition
    - an unsatisfiable dependency, a refused input - scores 0.5: it stopped
    correctly, but it did not deliver. A run that ran out of budget or
    iterations scores 0.0, because that is the loop failing to terminate on
    its own and being stopped from outside.
    """
    if state.status == RunStatus.COMPLETED:
        return 1.0, "completed"
    if state.iteration_count >= state.max_iterations:
        return 0.0, "hit the iteration ceiling - the loop did not converge"
    if state.total_cost_usd > 0 and state.status == RunStatus.FAILED and not state.error:
        return 0.0, "halted by the cost ceiling"
    if state.status == RunStatus.FAILED and state.error:
        return 0.5, f"stopped on a validated condition: {state.error[:80]}"
    return 0.0, f"ended in an unexpected state: {state.status.value}"


class AgentEvaluator:
    """DeepEval when present; deterministic metrics always."""

    def __init__(self) -> None:
        self._deepeval_available = False
        try:
            import deepeval  # noqa: F401

            self._deepeval_available = True
            logger.info("eval.deepeval_loaded")
        except ImportError:
            logger.info("eval.deepeval_absent_using_deterministic_metrics")

    @property
    def backend(self) -> str:
        return "deepeval" if self._deepeval_available else "deterministic"

    async def score(self, state: CortexState, expected_tools: set[str] | None = None) -> AgentScore:
        tool_correctness, tool_detail = score_tool_correctness(state, expected_tools)
        plan_efficiency, plan_detail = score_plan_efficiency(state)
        termination, term_detail = score_termination(state)
        completion, completion_detail = await self._score_completion(state)

        score = AgentScore(
            task_completion=completion,
            tool_correctness=tool_correctness,
            plan_efficiency=plan_efficiency,
            termination=termination,
            backend=self.backend,
            details={
                "task_completion": completion_detail,
                "tool_correctness": tool_detail,
                "plan_efficiency": plan_detail,
                "termination": term_detail,
            },
        )
        for axis, value in (
            ("task_completion", score.task_completion),
            ("tool_correctness", score.tool_correctness),
            ("plan_efficiency", score.plan_efficiency),
            ("termination", score.termination),
        ):
            agent_eval_score.labels(metric=axis).observe(value)
        return score

    async def _score_completion(self, state: CortexState) -> tuple[float, str]:
        """Did the output actually address the goal?

        The only axis that needs a model, because it is the only one that is
        a judgement rather than a measurement.
        """
        if not state.final_output:
            return 0.0, "no output produced"

        if self._deepeval_available:
            try:
                from deepeval.metrics import TaskCompletionMetric  # type: ignore[attr-defined]
                from deepeval.test_case import LLMTestCase  # type: ignore[attr-defined]

                metric = TaskCompletionMetric(threshold=0.7)
                case = LLMTestCase(input=state.user_goal, actual_output=state.final_output)
                metric.measure(case)
                # `score` is None when deepeval could not reach its judge model.
                if metric.score is None:
                    raise RuntimeError("deepeval returned no score")
                return float(metric.score), f"deepeval: {metric.reason or 'scored'}"
            except Exception as exc:
                logger.warning("eval.deepeval_failed_falling_back", error=str(exc))

        return await self._judge_completion(state)

    async def _judge_completion(self, state: CortexState) -> tuple[float, str]:
        import json

        from cortex.llm.router import get_router

        try:
            response = await get_router().complete(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Score 0.0-1.0 how completely the answer achieves the stated goal. "
                            'Return ONLY JSON: {"score": float, "reason": "one sentence"}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Goal: {state.user_goal}\n\nAnswer: {(state.final_output or '')[:4000]}",
                    },
                ],
                run_id="agent-eval",
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return float(data.get("score", 0.5)), f"llm-judge: {data.get('reason', '')}"
        except Exception as exc:
            # 0.5, not 0.0 and not 1.0. A judge that could not be reached is
            # an absence of evidence; scoring it as a failure would make an
            # outage look like a quality regression, and scoring it as a
            # pass would let an outage hide one.
            logger.warning("eval.judge_unavailable", error=str(exc))
            return 0.5, f"judge unavailable ({type(exc).__name__}) - score is not evidence"
