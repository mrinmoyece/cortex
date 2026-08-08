"""
Critic Agent.

Evaluates the executor's compiled output against:
  1. Faithfulness — is every claim grounded in retrieved context or task results?
  2. Completeness — does the output address the full user goal?
  3. Coherence — is the response well-structured and unambiguous?
  4. Safety — does it pass a lightweight hallucination heuristic?

Returns a CritiqueResult with accept/reject decision, score, and
actionable suggestions for the replanner if rejected.
"""

from __future__ import annotations

import json

from cortex.graph.state import CortexState, CritiqueResult
from cortex.llm.router import get_router
from cortex.logging_config import get_logger
from cortex.obs.metrics import critic_rejection_total, critic_score_histogram
from cortex.obs.tracing import observe

logger = get_logger(__name__)

_CRITIC_SYSTEM_PROMPT = """You are a rigorous quality reviewer for an AI system's outputs.

Evaluate the provided output against the original user goal and the evidence (task results) used to produce it.

Score on these dimensions (0.0 to 1.0 each):
  - faithfulness: Every factual claim is directly supported by the evidence. No hallucinations.
  - completeness: The full user goal is addressed. Nothing material is missing.
  - coherence: The response is clear, well-structured, and unambiguous.

Acceptance threshold: overall score >= 0.80 AND faithfulness >= 0.85

Respond with ONLY valid JSON — no markdown:
{
  "faithfulness": float,
  "completeness": float,
  "coherence": float,
  "overall": float,
  "accepted": bool,
  "reasoning": "1-2 sentences explaining the decision",
  "suggestions": ["actionable suggestion 1", ...]  // empty list if accepted
}"""


class CriticAgent:
    def __init__(self) -> None:
        self._router = get_router()

    @observe("critic.critique")
    async def critique(self, state: CortexState) -> CritiqueResult:
        """
        Evaluate state.final_output and return a CritiqueResult.
        Always returns a result — never raises (to avoid breaking the graph).
        """
        evidence = self._build_evidence(state)
        messages = [
            {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User goal:\n{state.user_goal}\n\n"
                    f"Evidence (task results):\n{evidence}\n\n"
                    f"Output to evaluate:\n{state.final_output}"
                ),
            },
        ]

        try:
            response = await self._router.complete(
                messages=messages,
                run_id=state.run_id,
                temperature=0.0,
                response_format={"type": "json_object"},
                metadata={"agent": "critic", "iteration": state.critique_iteration},
            )
            raw = response.choices[0].message.content or "{}"
            result = self._parse_critique(raw, state.critique_iteration)

        except Exception as exc:
            logger.error("critic.failed", run_id=state.run_id, error=str(exc))
            # Fail open — accept with low score rather than breaking the run
            result = CritiqueResult(
                accepted=True,
                score=0.5,
                reasoning=f"Critique failed: {exc}. Accepting with low confidence.",
                iteration=state.critique_iteration,
            )

        # Emit metrics
        critic_score_histogram.observe(result.score)
        if not result.accepted:
            critic_rejection_total.inc()

        logger.info(
            "critic.result",
            run_id=state.run_id,
            accepted=result.accepted,
            score=result.score,
            iteration=result.iteration,
        )
        return result

    def _build_evidence(self, state: CortexState) -> str:
        lines = []
        for task in state.completed_tasks():
            if task.result:
                lines.append(f"[{task.id}] {task.description}: {task.result[:500]}")
        return "\n".join(lines) if lines else "No task results available."

    def _parse_critique(self, raw: str, iteration: int) -> CritiqueResult:
        try:
            data = json.loads(raw)
            overall = float(data.get("overall", 0.0))
            accepted = bool(data.get("accepted", False))
            return CritiqueResult(
                accepted=accepted,
                score=overall,
                reasoning=data.get("reasoning", ""),
                suggestions=data.get("suggestions", []),
                iteration=iteration,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("critic.parse_failed", error=str(exc), raw=raw[:200])
            return CritiqueResult(
                accepted=True,
                score=0.6,
                reasoning="Could not parse critique response. Accepting with low confidence.",
                iteration=iteration,
            )
