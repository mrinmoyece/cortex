"""
Planner Agent.

Receives the user goal and memory context, returns a topologically-ordered
list of Tasks. Each task has an optional MCP tool hint and explicit
dependency edges so the executor can parallelise safely.

Prompt engineering principles applied:
  - Chain-of-thought via structured JSON output (no freeform prose)
  - Memory context injected to avoid re-doing known facts
  - Explicit constraint: no more than 8 tasks (prevents runaway plans)
  - Failure mode: if plan parsing fails, raises AgentPlanningError
"""

from __future__ import annotations

import json
import uuid

from cortex.config import settings
from cortex.exceptions import AgentPlanningError
from cortex.graph.state import CortexState, Task
from cortex.llm.router import get_router
from cortex.logging_config import get_logger
from cortex.obs.tracing import observe

logger = get_logger(__name__)

_PLANNER_SYSTEM_PROMPT = """You are a task planner for an agentic AI system.

Given a user goal and relevant memory context, produce a minimal, executable task plan.

Rules:
1. Maximum 8 tasks. Every task must be essential.
2. Each task must have a unique `id` (short slug, e.g. "fetch_data").
3. List tasks in execution order. Use `depends_on` to declare dependencies — executor uses this for parallelism.
4. For each task, suggest the best `tool` from: [search_knowledge, query_memory, execute_code, query_data, web_search, synthesise]. Use null if no specific tool is needed.
5. Tasks should be atomic — one clear action each.
6. Do NOT include meta-tasks like "plan the work" or "summarise". Only concrete actions.

Respond with ONLY valid JSON in this exact schema — no markdown, no explanation:
{
  "reasoning": "<1-2 sentences explaining your approach>",
  "tasks": [
    {
      "id": "string",
      "description": "string",
      "tool": "string|null",
      "depends_on": ["task_id", ...]
    }
  ]
}"""


class PlannerAgent:
    def __init__(self) -> None:
        self._router = get_router()

    @observe("planner.plan", capture=("run_id",))
    async def plan(self, state: CortexState) -> list[Task]:
        """
        Decompose `state.user_goal` into an ordered task list.

        Raises:
            AgentPlanningError: If the LLM returns malformed JSON or
                                an invalid task structure.
        """
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(state)},
        ]

        response = await self._router.complete(
            messages=messages,
            model=settings.default_model,
            run_id=state.run_id,
            temperature=0.0,  # Deterministic planning
            response_format={"type": "json_object"},
            metadata={"agent": "planner", "session_id": state.session_id},
        )

        raw = response.choices[0].message.content
        return self._parse_plan(raw, state.run_id)

    def _build_user_prompt(self, state: CortexState) -> str:
        parts = [f"User goal: {state.user_goal}"]

        if state.context:
            parts.append(f"\nAdditional context:\n{json.dumps(state.context, indent=2)}")

        if state.memory_context.semantic:
            facts = [f["content"] for f in state.memory_context.semantic[:5]]
            parts.append(
                "\nRelevant memory (already known — don't re-fetch):\n"
                + "\n".join(f"- {f}" for f in facts)
            )

        if state.critique_results:
            last = state.critique_results[-1]
            parts.append(
                f"\nPrevious plan was rejected (score: {last.score:.2f}).\n"
                f"Critique: {last.reasoning}\n"
                f"Suggestions:\n" + "\n".join(f"- {s}" for s in last.suggestions)
            )

        return "\n".join(parts)

    def _parse_plan(self, raw: str, run_id: str) -> list[Task]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AgentPlanningError(
                f"Planner returned invalid JSON: {exc}",
                details={"raw_response": raw[:500]},
            ) from exc

        tasks_data = data.get("tasks")
        if not tasks_data or not isinstance(tasks_data, list):
            raise AgentPlanningError(
                "Planner response missing 'tasks' list",
                details={"parsed": data},
            )

        tasks = []
        seen_ids: set[str] = set()

        for item in tasks_data:
            task_id = item.get("id") or str(uuid.uuid4())[:8]
            if task_id in seen_ids:
                task_id = f"{task_id}_{len(seen_ids)}"
            seen_ids.add(task_id)

            tasks.append(
                Task(
                    id=task_id,
                    description=item.get("description", ""),
                    tool=item.get("tool"),
                    depends_on=item.get("depends_on", []),
                )
            )

        logger.info(
            "planner.plan_created",
            run_id=run_id,
            task_count=len(tasks),
            reasoning=data.get("reasoning", ""),
        )
        return tasks
